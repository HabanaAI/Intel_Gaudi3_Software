#include "gc_singleton.h"

#include "data_serializer/data_serializer.h"
#include "filesystem.h"
#include "gc_scheme.h"
#include "graph_serializer/graph_serializer.h"
#include "habana_global_conf.h"
#include "log_manager.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "syn_singleton.hpp"
#include "synapse_common_types.h"
#include "types_exception.h"
#include "utils.h"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_set>
#include <vector>

enum ValidationType
{
    VALIDATION_NONE = 0,
    VALIDATION_NAN  = 1,
    VALIDATION_INF  = 2
};

enum SplitType
{
    SPLIT_NONE  = 0,
    SPLIT_GRAPH = 1,
    SPLIT_RANK  = 2
};

static std::string validationTypeToString(ValidationType validationType)
{
    switch (validationType)
    {
        case VALIDATION_NONE:
            return "none";
        case VALIDATION_NAN:
            return "nan";
        case VALIDATION_INF:
            return "inf";
    }
    return "invalid";
}

static ValidationType stringToValidationType(const std::string_view value)
{
    if (value == "nan") return VALIDATION_NAN;
    if (value == "inf") return VALIDATION_INF;
    return VALIDATION_NONE;
}

static SplitType stringToSplitType(const std::string_view value)
{
    if (value == "none") return SPLIT_NONE;
    if (value == "graph") return SPLIT_GRAPH;
    if (value == "rank") return SPLIT_RANK;
    return SPLIT_NONE;
}

static uint32_t getValidationTypes(const std::vector<std::string>& values)
{
    uint32_t ret = VALIDATION_NONE;
    for (auto& val : values)
    {
        ret |= stringToValidationType(val);
    }
    return ret;
}

static void throwIfNot(bool cond, const std::string& msg, synStatus sts = synFail)
{
    if (false == cond)
    {
        throw SynapseStatusException(msg, sts);
    }
}

template<class T>
static std::optional<T> get(const json& data, const std::vector<std::string>& keys)
{
    if (keys.empty()) return std::nullopt;
    json ret = data;
    for (auto it = keys.begin(); it != keys.end(); ++it)
    {
        auto next = ret.find(*it);
        if (next == ret.end()) return std::nullopt;
        ret = *next;
    }
    return ret;
}

// w/a for deb-10 compilation [SW-117906]
template<class T>
static std::optional<std::vector<T>> getVector(const json& data, const std::vector<std::string>& keys)
{
    if (keys.empty()) return std::optional<std::vector<T>>();
    json ret = data;
    for (auto it = keys.begin(); it != keys.end(); ++it)
    {
        auto next = ret.find(*it);
        if (next == ret.end()) return std::optional<std::vector<T>>();
        ret = *next;
    }
    throwIfNot(
        ret.type() == json::value_t::array,
        fmt::format("failed to parse json. expected an array at: [{}]", fmt::join(keys.begin(), keys.end(), "->")),
        synInvalidArgument);
    return ret.get<std::vector<T>>();
}

static uint64_t getRankId()
{
    const char* rankId = getenv("OMPI_COMM_WORLD_RANK");
    if (rankId == nullptr)
    {
        rankId = getenv("ID");
    }
    return rankId ? std::stoull(rankId) : 0;
}

static std::string getFilePathWithRank(const fs::path& path, const std::string& ext)
{
    return path / fmt::format("{}.{}.{}", path.filename().string(), getRankId(), ext);
}

struct Config
{
    void parse(const std::string& values)
    {
        json conf = json::parse(values);
        if (conf.empty()) return;
        ignoreErrors = get<bool>(conf, {"ignore_errors"}).value_or(false);
        ranks        = getVector<uint64_t>(conf, {"ranks"}).value_or(std::vector<uint64_t> {});
        graphsFilter = getVector<std::string>(conf, {"graphs_filter", "value"}).value_or(std::vector<std::string> {});
        graphPath    = get<std::string>(conf, {"graph", "path", "value"});
        splitType    = stringToSplitType(get<std::string>(conf, {"split_type", "value"}).value_or("none"));
        if (graphPath.has_value())
        {
            if (splitType == SPLIT_RANK)
            {
                graphPath = getFilePathWithRank(fs::path(graphPath.value()), "json");
            }
            preGraph     = get<bool>(conf, {"graph", "type", "pre", "value"}).value_or(false);
            postGraph    = get<bool>(conf, {"graph", "type", "post", "value"}).value_or(false);
            passesGraph  = get<bool>(conf, {"graph", "type", "passes", "enable", "value"}).value_or(false);
            passesFilter = get<std::string>(conf, {"graph", "type", "passes", "filter", "value"});
        }

        tensorsPath = get<std::string>(conf, {"tensors", "path", "value"});
        if (tensorsPath.has_value())
        {
            if (splitType == SPLIT_RANK)
            {
                tensorsPath = getFilePathWithRank(fs::path(tensorsPath.value()), "db");
            }
            tensorsMinIter = get<uint64_t>(conf, {"tensors", "min_iter", "value"}).value_or(0);
            tensorsMaxIter =
                get<uint64_t>(conf, {"tensors", "max_iter", "value"}).value_or(std::numeric_limits<uint64_t>::max());
            tensorsLastIters        = get<uint64_t>(conf, {"tensors", "last_iters", "value"});
            tensorsFilterByElements = get<uint64_t>(conf, {"tensors", "filter_by_elements", "value"});
            compression             = get<data_serialize::Compression>(conf, {"tensors", "compression", "value"})
                              .value_or(data_serialize::Compression::LZ4);
            validationTypes = getValidationTypes(
                getVector<std::string>(conf, {"tensors", "validation", "value"}).value_or(std::vector<std::string> {}));
            stopOnFailure       = get<bool>(conf, {"tensors", "stop_on_failure", "value"}).value_or(false);
            recFailures         = get<bool>(conf, {"tensors", "rec_failures", "value"}).value_or(false);
            recOnlyConstTensors = get<bool>(conf, {"tensors", "const", "value"}).value_or(false);
        }
    }

    bool      stopOnFailure       = false;
    bool      recFailures         = false;
    bool      recOnlyConstTensors = false;
    uint32_t  validationTypes     = VALIDATION_NONE;
    SplitType splitType           = SPLIT_NONE;

    std::optional<std::string>                 graphPath;
    std::optional<std::string>                 tensorsPath;
    std::optional<uint64_t>                    tensorsMinIter;
    std::optional<uint64_t>                    tensorsMaxIter;
    std::optional<uint64_t>                    tensorsLastIters;
    std::optional<uint64_t>                    tensorsFilterByElements;
    std::optional<bool>                        preGraph;
    std::optional<bool>                        postGraph;
    std::optional<bool>                        passesGraph;
    std::optional<bool>                        ignoreErrors;
    std::optional<std::string>                 passesFilter;
    std::vector<std::string>                   graphsFilter;
    std::vector<uint64_t>                      ranks;
    std::optional<data_serialize::Compression> compression;
};

constexpr const char* PLUGIN_NAME = "GraphCompiler";
static Config         sConfig;

uint32_t GraphCompiler_Load()
{
    return API_BIT(SHIM_API_SYNAPSE);
}

void GraphCompiler_Unload() {}

static gc_shim::GraphCompilerSingleton* sGraphCompilerSingleton = nullptr;

ShimFunctions SYNAPSE_GraphCompiler_Init(ShimFunctions pDefaultFunctions)
{
    synSingletonInterface* defaultInterface = reinterpret_cast<synSingletonInterface*>(pDefaultFunctions);
    sGraphCompilerSingleton                 = new gc_shim::GraphCompilerSingleton(defaultInterface);

    return reinterpret_cast<ShimFunctions>(sGraphCompilerSingleton);
}

const char* SYNAPSE_GraphCompiler_GetVersion()
{
    return SYNAPSE_SINGLETON_INTERFACE_VERSION;
}

bool GraphCompiler_GetScheme(const char* schemeName, json* scheme)
{
    if (!scheme) return false;

    if (!std::strncmp(schemeName, PLUGIN_NAME, std::strlen(PLUGIN_NAME)))
    {
        *scheme = json::parse(s_graphCompilerScheme);
        return !scheme->is_null();
    }
    return false;
}

const char** GraphCompiler_GetSchemeNames()
{
    static const char* schemeNames[] = {PLUGIN_NAME, nullptr};
    return schemeNames;
}

void GraphCompiler_SetSchemeValues(const std::string& values)
{
    // use ShimSdk::CompileScheme when shim_sdk is moved to a different repo
    if (values.empty()) return;
    sConfig.parse(values);
    if (sConfig.splitType == SPLIT_NONE)
    {
        std::ofstream lockFile(fmt::format("{}.lock", sConfig.graphPath.value()), std::ios_base::app);
    }
}

const char** GetPluginsNames()
{
    static const char* gPluginsNames[] = {PLUGIN_NAME, nullptr};
    return gPluginsNames;
}

void SYNAPSE_GraphCompiler_Fini()
{
    if (sGraphCompilerSingleton)
    {
        delete sGraphCompilerSingleton;
        sGraphCompilerSingleton = nullptr;
    }
}

namespace gc_shim
{

bool isSynrecRecording()
{
    bool useSynrec = true;
    if (const char* env_p = std::getenv("SYNREC"))
    {
        useSynrec = (atoi(env_p) == 1);
    }
    return useSynrec;
}

synStatus GraphCompilerSingleton::initialize()
{
    auto type         = sConfig.splitType == SPLIT_GRAPH ? graph_serialize::SerializeType::SPLIT
                                                         : graph_serialize::SerializeType::UNIFIED;
    m_graphSerializer = graph_serialize::GraphSerializer::createGraphSerializer(
        {type, m_rankId, sConfig.graphPath.value(), sConfig.tensorsPath.value_or(""), sConfig.tensorsFilterByElements});
    return m_originalImpl->initialize();
}

synStatus GraphCompilerSingleton::destroy()
{
    return m_originalImpl->destroy();
}

static std::string
getFilePath(const fs::path& recipeName, const fs::path& path, uint64_t rankId, const std::string& postFix)
{
    const fs::path nonAbsRecipeName = recipeName.is_absolute() ? recipeName.string().substr(1) : recipeName.string();

    const fs::path recipeDir = nonAbsRecipeName.parent_path();
    const fs::path fileName  = fmt::format("{}.{}.{}", sanitizeFileName(nonAbsRecipeName.filename()), rankId, postFix);
    const fs::path fullPath  = recipeDir.empty() ? path / fileName : path / recipeDir / fileName;
    const fs::path fullPathDir = fullPath.parent_path();

    if (!fs::is_directory(fullPathDir) || !fs::exists(fullPathDir))
    {
        fs::create_directories(fullPathDir);
    }

    return fullPath;
}

static std::string tensorInfoToString(const synLaunchTensorInfoExt& info)
{
    return fmt::format("name: {}, ID: {}, type: {}",
                       info.tensorName,
                       info.tensorId,
                       synTensorType2Txt(info.tensorType));
}

static std::string tensorInfoToString(const synLaunchTensorInfoExt&          info,
                                      const synRetrievedLaunchTensorInfoExt& infoExt,
                                      bool                                   isDynamicRecipe)
{
    auto& actualShape = isDynamicRecipe ? info.tensorSize : infoExt.tensorMaxSize;
    return fmt::format("name: {}, ID: {}, type: {}, shape: [{}]",
                       info.tensorName,
                       info.tensorId,
                       synTensorType2Txt(info.tensorType),
                       toString(actualShape, actualShape + infoExt.tensorDims, ','));
}

static bool matchesPattern(const std::string& item, const std::vector<std::string>& patterns)
{
    return patterns.empty() || std::any_of(patterns.begin(), patterns.end(), [&](const std::string& i) {
               return item.find(i) != std::string::npos;
           });
}

GraphCompilerSingleton::GraphCompilerSingleton(synSingletonInterface* interface)
: synSingletonInterface(interface), m_rankId(getRankId())
{
}

synStatus GraphCompilerSingleton::enqueue(const synStreamHandle      streamHandle,
                                          const synLaunchTensorInfo* enqueueInputTensorsInfo,
                                          const uint32_t             inputInfoSize,
                                          const synLaunchTensorInfo* enqueueOutputTensorsInfo,
                                          const uint32_t             outputInfoSize,
                                          uint64_t                   workspaceAddress,
                                          const synRecipeHandle      pRecipeHandle,
                                          uint32_t                   flags)
{
    synLaunchTensorInfoExt enqueueInputTensorsInfoExt[inputInfoSize];
    synSingleton::elevateSynLaunchTensorInfo(enqueueInputTensorsInfoExt, enqueueInputTensorsInfo, inputInfoSize);
    synLaunchTensorInfoExt enqueueOutputTensorsInfoExt[outputInfoSize];
    synSingleton::elevateSynLaunchTensorInfo(enqueueOutputTensorsInfoExt, enqueueOutputTensorsInfo, outputInfoSize);
    return enqueue(streamHandle,
                   enqueueInputTensorsInfoExt,
                   inputInfoSize,
                   enqueueOutputTensorsInfoExt,
                   outputInfoSize,
                   workspaceAddress,
                   pRecipeHandle,
                   flags);
}

synStatus GraphCompilerSingleton::enqueue(const synStreamHandle      streamHandle,
                                          const synLaunchTensorInfo* enqueueTensorsInfo,
                                          const uint32_t             enqueueTensorsAmount,
                                          uint64_t                   pWorkspace,
                                          const synRecipeHandle      pRecipeHandle,
                                          uint32_t                   flags)
{
    synLaunchTensorInfoExt enqueueInputTensorsInfoExt[enqueueTensorsAmount];
    synSingleton::elevateSynLaunchTensorInfo(enqueueInputTensorsInfoExt, enqueueTensorsInfo, enqueueTensorsAmount);
    return enqueue(streamHandle, enqueueInputTensorsInfoExt, enqueueTensorsAmount, pWorkspace, pRecipeHandle, flags);
}

synStatus GraphCompilerSingleton::enqueueWithExternalEvents(const synStreamHandle      streamHandle,
                                                            const synLaunchTensorInfo* enqueueTensorsInfo,
                                                            const uint32_t             enqueueTensorsAmount,
                                                            uint64_t                   pWorkspace,
                                                            const synRecipeHandle      pRecipeHandle,
                                                            synEventHandle*            eventHandleList,
                                                            uint32_t                   numberOfEvents,
                                                            uint32_t                   flags)
{
    synLaunchTensorInfoExt enqueueInputTensorsInfoExt[enqueueTensorsAmount];
    synSingleton::elevateSynLaunchTensorInfo(enqueueInputTensorsInfoExt, enqueueTensorsInfo, enqueueTensorsAmount);
    return enqueueWithExternalEventsExt(streamHandle,
                                        enqueueInputTensorsInfoExt,
                                        enqueueTensorsAmount,
                                        pWorkspace,
                                        pRecipeHandle,
                                        eventHandleList,
                                        numberOfEvents,
                                        flags);
}

synStatus GraphCompilerSingleton::enqueue(const synStreamHandle         streamHandle,
                                          const synLaunchTensorInfoExt* enqueueInputTensorsInfo,
                                          const uint32_t                inputSize,
                                          const synLaunchTensorInfoExt* enqueueOutputTensorsInfo,
                                          const uint32_t                outputSize,
                                          uint64_t                      pWorkspace,
                                          const synRecipeHandle         pRecipeHandle,
                                          uint32_t                      flags)
{
    return enqueueAndCapture(streamHandle, enqueueInputTensorsInfo, inputSize, pRecipeHandle, [&]() {
        return m_originalImpl->enqueue(streamHandle,
                                       enqueueInputTensorsInfo,
                                       inputSize,
                                       enqueueOutputTensorsInfo,
                                       outputSize,
                                       pWorkspace,
                                       pRecipeHandle,
                                       flags);
    });
}

synStatus GraphCompilerSingleton::enqueue(const synStreamHandle         streamHandle,
                                          const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                          const uint32_t                enqueueTensorsAmount,
                                          uint64_t                      pWorkspace,
                                          const synRecipeHandle         pRecipeHandle,
                                          synEventHandle*               eventHandleList,
                                          uint32_t                      numberOfEvents,
                                          uint32_t                      flags)
{
    return enqueueWithExternalEventsExt(streamHandle,
                                        enqueueTensorsInfo,
                                        enqueueTensorsAmount,
                                        pWorkspace,
                                        pRecipeHandle,
                                        eventHandleList,
                                        numberOfEvents,
                                        flags);
}

synStatus GraphCompilerSingleton::enqueue(const synStreamHandle         streamHandle,
                                          const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                          const uint32_t                enqueueTensorsAmount,
                                          uint64_t                      pWorkspace,
                                          const synRecipeHandle         pRecipeHandle,
                                          uint32_t                      flags)
{
    return enqueueAndCapture(streamHandle, enqueueTensorsInfo, enqueueTensorsAmount, pRecipeHandle, [&]() {
        return m_originalImpl
            ->enqueue(streamHandle, enqueueTensorsInfo, enqueueTensorsAmount, pWorkspace, pRecipeHandle, flags);
    });
}

synStatus GraphCompilerSingleton::enqueueWithExternalEventsExt(const synStreamHandle         streamHandle,
                                                               const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                               const uint32_t                enqueueTensorsAmount,
                                                               uint64_t                      pWorkspace,
                                                               const synRecipeHandle         pRecipeHandle,
                                                               synEventHandle*               eventHandleList,
                                                               uint32_t                      numberOfEvents,
                                                               uint32_t                      flags)
{
    return enqueueAndCapture(streamHandle, enqueueTensorsInfo, enqueueTensorsAmount, pRecipeHandle, [&]() {
        return m_originalImpl->enqueueWithExternalEventsExt(streamHandle,
                                                            enqueueTensorsInfo,
                                                            enqueueTensorsAmount,
                                                            pWorkspace,
                                                            pRecipeHandle,
                                                            eventHandleList,
                                                            numberOfEvents,
                                                            flags);
    });
}

static bool skipRecording(const std::string recipeName, uint64_t rankId)
{
    if (!sConfig.ranks.empty() && std::find(sConfig.ranks.begin(), sConfig.ranks.end(), rankId) == sConfig.ranks.end())
    {
        LOG_INFO(SYNREC, "skip serializing tensors of graph: {}, rank ID: {} was not selected", recipeName, rankId);
        return true;
    }

    if (!matchesPattern(recipeName, sConfig.graphsFilter))
    {
        LOG_INFO(SYNREC,
                 "skip serializing tensors of graph: {}, rank ID: {}, graph was not selected",
                 recipeName,
                 rankId);
        return true;
    }

    return false;
}

static std::shared_ptr<data_serialize::DataSerializer> createDataSerializer(const std::string&    dumpPath,
                                                                            const synRecipeHandle pRecipeHandle,
                                                                            uint16_t              recipeId,
                                                                            uint64_t              rankId)
{
    const char*       name       = pRecipeHandle->basicRecipeHandle.recipe->name;
    const std::string recipeName = name ? name : "";

    return data_serialize::DataSerializer::create(dumpPath, data_serialize::GraphInfo {recipeName, recipeId, rankId});
}

static bool isDeviceTensor(synTensorType type)
{
    switch (type)
    {
        case DATA_TENSOR:
        case DATA_TENSOR_DYNAMIC:
        case DEVICE_SHAPE_TENSOR:
            return true;
        case OUTPUT_DESCRIBING_SHAPE_TENSOR:
        case INPUT_DESCRIBING_SHAPE_TENSOR:
        case HOST_SHAPE_TENSOR:
        case HOST_TO_DEVICE_TENSOR:
        case TENSOR_TYPE_MAX:
            return false;
    }
    throw SynapseException(fmt::format("unsupported tensor type: {}", int(type)));
}

bool GraphCompilerSingleton::skipGraphRecording(const std::string& recipeName)
{
    if (!sConfig.graphPath.has_value()) return true;
    return skipRecording(recipeName, m_rankId);
}

bool GraphCompilerSingleton::skipTensorRecording(const std::string& recipeName)
{
    if (!sConfig.tensorsPath.has_value()) return true;
    if (!isSynrecRecording()) return true;
    return skipRecording(recipeName, m_rankId);
}

template<class T>
synStatus GraphCompilerSingleton::dumpToFile(const std::string&                           recipeName,
                                             const synStreamHandle                        streamHandle,
                                             std::vector<data_serialize::TensorMetadata>& tensorsMetadata,
                                             const DataSerializerPtr                      dataSerializer,
                                             T                                            callback)
{
    for (auto& tmd : tensorsMetadata)
    {
        dataSerializer->serialize(tmd);
    }

    // serialize inputs data
    serializeLaunchTensors(streamHandle, tensorsMetadata, dataSerializer, true);

    synStatus sts = callback();

    if (synSuccess != sts)
    {
        LOG_WARN(SYNREC,
                 "Failed to enqueue recipe: {}, the output tensors will be serialized without data",
                 recipeName);
    }

    // serialize outputs data
    serializeLaunchTensors(streamHandle, tensorsMetadata, dataSerializer, false);

    if (sConfig.tensorsLastIters.has_value())
    {
        dataSerializer->removePrevIterations(sConfig.tensorsLastIters.value());
    }

    return sts;
}

template<class T>
synStatus GraphCompilerSingleton::enqueueAndCapture(const synStreamHandle         streamHandle,
                                                    const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                    const uint32_t                enqueueTensorsAmount,
                                                    const synRecipeHandle         pRecipeHandle,
                                                    T                             callback)
{
    const auto        launchCounter = m_launchCounter++;
    const char*       name          = pRecipeHandle->basicRecipeHandle.recipe->name;
    const std::string recipeName    = name ? name : "";

    if (skipTensorRecording(recipeName) || sConfig.recOnlyConstTensors)
    {
        return callback();
    }

    const debug_info_t* dbgInfo;
    getRecipeDebugInfo(pRecipeHandle, &dbgInfo);

    // Checks if graph recording is needed / has not been done yet.
    m_graphSerializer->recordGraph(pRecipeHandle, dbgInfo->recipe_id);

    std::string dumpPath = sConfig.splitType == SPLIT_GRAPH
                               ? getFilePath(recipeName, sConfig.tensorsPath.value(), m_rankId, "db")
                               : sConfig.tensorsPath.value();

    std::vector<data_serialize::TensorMetadata> tensorsMetadata =
        getTensorsMetadata(enqueueTensorsInfo, enqueueTensorsAmount, pRecipeHandle, launchCounter);

    if (sConfig.recFailures == false)
    {
        DataSerializerPtr ds = createDataSerializer(dumpPath, pRecipeHandle, dbgInfo->recipe_id, m_rankId);
        LOG_INFO(SYNREC, "serializing graph: {} tensors data to: {}, rank ID: {}", recipeName, dumpPath, m_rankId);
        return dumpToFile(recipeName, streamHandle, tensorsMetadata, ds, callback);
    }

    synStatus sts = callback();

    if (validateTensors(streamHandle, tensorsMetadata, false) == false)
    {
        DataSerializerPtr ds = createDataSerializer(dumpPath, pRecipeHandle, dbgInfo->recipe_id, m_rankId);
        LOG_INFO(SYNREC,
                 "serializing failed graph: {} tensors data to: {}, rank ID: {}",
                 recipeName,
                 dumpPath,
                 m_rankId);
        dumpToFile(recipeName, streamHandle, tensorsMetadata, ds, callback);
        LOG_ERR(SYNREC, "Validation failed, graph: {}, rank ID: {}", recipeName, m_rankId);
        if (sConfig.stopOnFailure) return synFail;
    }
    return sts;
}

std::shared_ptr<uint64_t> GraphCompilerSingleton::getTensorData(const uint32_t        deviceId,
                                                                const synTensorType&  tensorType,
                                                                const uint64_t        tensorAddress,
                                                                const uint64_t        tensorDataSize,
                                                                const synStreamHandle stream)
{
    std::shared_ptr<uint64_t> ret = nullptr;

    switch (tensorType)
    {
        case HOST_SHAPE_TENSOR:
        case HOST_TO_DEVICE_TENSOR:
        {
            ret = std::shared_ptr<uint64_t>(reinterpret_cast<uint64_t*>(tensorAddress), [](uint64_t* p) {});
            break;
        }
        case DATA_TENSOR:
        case DATA_TENSOR_DYNAMIC:
        case DEVICE_SHAPE_TENSOR:  // device shape tensor has data and need to be allocated
        {
            uint64_t* bufferPtr = nullptr;
            synStatus sts       = allocateDeviceMemory(deviceId,
                                                 tensorDataSize,
                                                 synMemFlags::synMemHost,
                                                 reinterpret_cast<void**>(&bufferPtr));
            throwIfNot(synSuccess == sts, "Failed to allocate buffer");

            ret = std::shared_ptr<uint64_t>(bufferPtr, [deviceId, bufferPtr, this](uint64_t* p) {
                synStatus sts = deallocateDeviceMemory(deviceId, bufferPtr, synMemFlags::synMemHost);
                throwIfNot(synSuccess == sts, "Failed to de-allocate buffer");
            });

            uint64_t address = reinterpret_cast<uint64_t>(bufferPtr);
            sts              = memcpyAsync(stream, &tensorAddress, &tensorDataSize, &address, synDmaDir::DRAM_TO_HOST);
            throwIfNot(synSuccess == sts, "Failed to copy tensors");

            throwIfNot(synSuccess == synchronizeStream(stream), "Failed to synchronize upload stream");
            break;
        }
        case OUTPUT_DESCRIBING_SHAPE_TENSOR:
        case INPUT_DESCRIBING_SHAPE_TENSOR:
        case TENSOR_TYPE_MAX:
            break;
    }
    return ret;
}

std::vector<data_serialize::TensorMetadata>
GraphCompilerSingleton::getTensorsMetadata(const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                           const uint32_t                enqueueTensorsAmount,
                                           const synRecipeHandle         pRecipeHandle,
                                           uint64_t                      launchIndex)
{
    std::vector<data_serialize::TensorMetadata> ret;
    ret.reserve(enqueueTensorsAmount);

    bool isDynamicRecipe = pRecipeHandle->basicRecipeHandle.shape_plan_recipe;

    std::set<uint16_t> constSectionsIdx;
    for (size_t i = 0; i < pRecipeHandle->basicRecipeHandle.recipe->const_sections_nr; ++i)
    {
        constSectionsIdx.insert(pRecipeHandle->basicRecipeHandle.recipe->const_sections[i].section_idx);
    }

    std::unordered_set<std::string_view> recordedTensors;
    for (size_t i = 0; i < enqueueTensorsAmount; ++i)
    {
        synLaunchTensorInfoExt          info    = enqueueTensorsInfo[i];
        synRetrievedLaunchTensorInfoExt infoExt = {};

        auto validation = data_serialize::TensorValidation::VALID;
        // Tensors with invalid ID are serialized with it for bug reproduction.
        if (info.tensorId == TENSOR_INVALID_ID)
        {
            validation = data_serialize::TensorValidation::INVALID_ID;
        }

        // We want to keep tensors that are launched twice for bug reproduction.
        else if (recordedTensors.find(info.tensorName) != recordedTensors.end())
        {
            LOG_WARN(SYNREC,
                     "Tensor was already serialized, serializing it as duplicate, recipe: {}, index: {}, {}",
                     pRecipeHandle->basicRecipeHandle.recipe->name,
                     i,
                     tensorInfoToString(info));
            validation = data_serialize::TensorValidation::DUPLICATE;
        }

        infoExt.tensorId = info.tensorId;
        // w/a to issue: [SW-72058] Bert tiny SBS call synLaunch with non persistent tensors
        // this condition should be asserted
        if (synSuccess != tensorRetrieveLaunchInfoByIdExt(pRecipeHandle, 1, &infoExt))
        {
            LOG_WARN(SYNREC,
                     "Skipping request to serialize tensor with invalid ID, recipe: {}, index: {}, {}",
                     pRecipeHandle->basicRecipeHandle.recipe->name,
                     i,
                     tensorInfoToString(info));
            continue;
        }

        if (constSectionsIdx.find(infoExt.tensorSectionId) != constSectionsIdx.end())
        {
            LOG_DEBUG(SYNREC,
                      "Skipping const tensor, recipe: {}, index: {}, {}",
                      pRecipeHandle->basicRecipeHandle.recipe->name,
                      i,
                      tensorInfoToString(info));
            continue;
        }

        // for dynamic shapes the runtime time shape is required
        auto& actualShape = isDynamicRecipe ? info.tensorSize : infoExt.tensorMaxSize;

        uint64_t size            = getActualTensorSize(infoExt.tensorDims, actualShape, infoExt.tensorDataType);
        uint64_t permutationSize = isDeviceTensor(infoExt.tensorType) ? infoExt.tensorDims : 0;

        LOG_DEBUG(SYNREC,
                  "Serializing {} tensor, recipe: {}, index: {}, {}, size: {}, address: {}, rank ID: {}",
                  infoExt.isInput ? "input" : "output",
                  pRecipeHandle->basicRecipeHandle.recipe->name,
                  i,
                  tensorInfoToString(info, infoExt, isDynamicRecipe),
                  size,
                  info.pTensorAddress,
                  m_rankId);

        data_serialize::TensorMetadata md;
        md.address     = info.pTensorAddress;
        md.id          = info.tensorId;
        md.name        = info.tensorName;
        md.type        = infoExt.tensorType;
        md.dataType    = infoExt.tensorDataType;
        md.compression = sConfig.compression.value();
        md.shape       = std::vector<TSize>(actualShape, actualShape + infoExt.tensorDims);
        md.permutation = permutationSize > 0 ? std::vector<uint8_t>(infoExt.tensorPermutation,
                                                                    infoExt.tensorPermutation + permutationSize)
                                             : std::vector<uint8_t> {};
        md.dataSize    = size;
        md.data        = nullptr;
        md.validation  = validation;
        md.launchIndex = launchIndex;
        md.constTensor = false;
        md.input       = infoExt.isInput;

        recordedTensors.insert(info.tensorName);
        ret.push_back(md);
    }
    return ret;
}

template<typename T>
static ValidationType validateValue(T value)
{
    float v = static_cast<float>(value);
    if ((sConfig.validationTypes & VALIDATION_NAN) && std::isnan(v)) return VALIDATION_NAN;
    if ((sConfig.validationTypes & VALIDATION_INF) && std::isinf(v)) return VALIDATION_INF;
    return VALIDATION_NONE;
}

template<typename T>
static ValidationType validateData(T* data, TSize dataSize)
{
    TSize elements = dataSize / sizeof(T);
    for (TSize i = 0; i < elements; ++i)
    {
        auto res = validateValue(data[i]);
        if (res != VALIDATION_NONE) return res;
    }
    return VALIDATION_NONE;
}

static ValidationType validateTensor(const data_serialize::TensorMetadata& tmd)
{
    switch (tmd.dataType)
    {
        case syn_type_float:
            return validateData(reinterpret_cast<float*>(tmd.data.get()), tmd.dataSize);
        case syn_type_bf16:
            return validateData(reinterpret_cast<bfloat16*>(tmd.data.get()), tmd.dataSize);
        case syn_type_fp16:
            return validateData(reinterpret_cast<fp16_t*>(tmd.data.get()), tmd.dataSize);
        case syn_type_tf32:
            return validateData(reinterpret_cast<Tfloat32*>(tmd.data.get()), tmd.dataSize);
        case syn_type_fp8_143:
            return validateData(reinterpret_cast<fp8_143_t*>(tmd.data.get()), tmd.dataSize);
        case syn_type_fp8_152:
            return validateData(reinterpret_cast<fp8_152_t*>(tmd.data.get()), tmd.dataSize);
        case syn_type_ufp16:
        case syn_type_hb_float:
        case syn_type_na:
        case syn_type_fixed:
        case syn_type_int16:
        case syn_type_int32:
        case syn_type_uint8:
        case syn_type_int4:
        case syn_type_uint4:
        case syn_type_uint16:
        case syn_type_uint32:
        case syn_type_int64:
        case syn_type_uint64:
        case syn_type_max:
            break;
    }
    return VALIDATION_NONE;
}

void GraphCompilerSingleton::serializeLaunchTensors(const synStreamHandle                        streamHandle,
                                                    std::vector<data_serialize::TensorMetadata>& tensorsMetadata,
                                                    const DataSerializerPtr                      dataSerializer,
                                                    bool                                         serializeInputs)
{
    synStatus   sts;
    synDeviceId deviceId = {};

    throwIfNot(synSuccess == getDeviceId(streamHandle, deviceId), "Failed to get device ID");

    uint64_t iteration = dataSerializer->getIterationCount();

    bool shouldCaptureData = true;
    if (sConfig.tensorsMinIter.has_value() && iteration < sConfig.tensorsMinIter.value()) shouldCaptureData = false;
    if (sConfig.tensorsMaxIter.has_value() && iteration > sConfig.tensorsMaxIter.value()) shouldCaptureData = false;

    if (shouldCaptureData)
    {
        throwIfNot(synSuccess == synchronizeStream(streamHandle), "Failed to synchronize stream");

        ValidationType validationStatus = VALIDATION_NONE;

        for (const auto& tmd : tensorsMetadata)
        {
            if (tmd.input != serializeInputs) continue;

            auto dataReleaserTmd = tmd;
            if (dataReleaserTmd.validation == data_serialize::TensorValidation::VALID)
            {
                dataReleaserTmd.data = getTensorData(deviceId, tmd.type, tmd.address, tmd.dataSize, streamHandle);
            }
            dataSerializer->updateData(dataReleaserTmd);

            if (sConfig.stopOnFailure && validationStatus == VALIDATION_NONE && serializeInputs == false)
            {
                validationStatus = validateTensor(dataReleaserTmd);
                if (validationStatus != VALIDATION_NONE)
                {
                    LOG_ERR(SYNREC,
                            "Tensor validation failed, tensor name: {}, validation type: {}",
                            dataReleaserTmd.name,
                            validationTypeToString(validationStatus));
                }
            }
        }
        throwIfNot(validationStatus == VALIDATION_NONE, "synrec data validation failure");
    }
}

bool GraphCompilerSingleton::validateTensors(const synStreamHandle                        streamHandle,
                                             std::vector<data_serialize::TensorMetadata>& tensorsMetadata,
                                             bool                                         serializeInputs)
{
    synStatus   sts;
    synDeviceId deviceId = {};

    throwIfNot(synSuccess == getDeviceId(streamHandle, deviceId), "Failed to get device ID");
    throwIfNot(synSuccess == synchronizeStream(streamHandle), "Failed to synchronize stream");

    for (const auto& tmd : tensorsMetadata)
    {
        if (tmd.input != serializeInputs) continue;

        auto dataReleaserTmd = tmd;
        if (dataReleaserTmd.validation != data_serialize::TensorValidation::VALID) continue;
        dataReleaserTmd.data = getTensorData(deviceId, tmd.type, tmd.address, tmd.dataSize, streamHandle);
        auto sts             = validateTensor(dataReleaserTmd);
        if (sts != VALIDATION_NONE)
        {
            LOG_ERR(SYNREC,
                    "Tensor validation failed, tensor name: {}, validation type: {}",
                    tmd.name,
                    validationTypeToString(sts));
            return false;
        }
    }
    return true;
}

synStatus GraphCompilerSingleton::createGraph(synGraphHandle* pGraphHandle,
                                              synDeviceType   deviceType,
                                              CompilationMode compilationMode)
{
    synStatus status = m_originalImpl->createGraph(pGraphHandle, deviceType, compilationMode);
    if (synSuccess == status)
    {
        m_graphSerializer->addGraph(*pGraphHandle, compilationMode);
    }
    return status;
}

synStatus GraphCompilerSingleton::destroyGraph(const synGraphHandle graphHandle)
{
    synStatus status = m_originalImpl->destroyGraph(graphHandle);
    if (synSuccess == status)
    {
        m_graphSerializer->removeGraph(graphHandle);
    }
    return status;
}

synStatus GraphCompilerSingleton::duplicateGraph(synGraphHandle      graphHandle,
                                                 synGraphHandle*     newGraphHandle,
                                                 synTensorHandleMap* tensorsMap,
                                                 uint32_t*           numTensors,
                                                 synNodeHandleMap*   nodesMap,
                                                 uint32_t*           numNodes)
{
    // Perform the actual duplication
    synStatus status =
        m_originalImpl->duplicateGraph(graphHandle, newGraphHandle, tensorsMap, numTensors, nodesMap, numNodes);

    // if any of tensorsMap or nodesMap is nullptr this is just a query
    // so no need to add a new graph handle.
    if (tensorsMap && nodesMap && synSuccess == status)
    {
        const graph_serialize::Graph& graph           = m_graphSerializer->getGraph(graphHandle);
        CompilationMode               compilationMode = graph.graphCompilationMode;

        // First add the new graph
        m_graphSerializer->addGraph(*newGraphHandle, compilationMode);

        for (uint32_t i = 0; i < *numNodes; i++)
        {
            // Find original node
            const synNodeHandleMap&      nodeMap = nodesMap[i];
            const graph_serialize::Node* orgNode = nullptr;
            for (const graph_serialize::Node& node : graph.nodes)
            {
                if (node.nodeUniqueId == nodeMap.origHandle)
                {
                    orgNode = &node;
                    break;
                }
            }
            if (orgNode == nullptr)
            {
                LOG_ERR(SYNREC, "Original node '{}' was not found", nodeMap.origHandle);
                return synStatus::synFail;
            }

            // Create and add the new node
            graph_serialize::Node newNode(*newGraphHandle, *orgNode, nodeMap.newHandle, tensorsMap, *numTensors);
            m_graphSerializer->addNode(*newGraphHandle, newNode);
        }
    }

    return status;
}

synStatus GraphCompilerSingleton::createGenericNode(const synGraphHandle graphHandle,
                                                    const synTensor*     inputs,
                                                    const synTensor*     outputs,
                                                    const uint32_t       sizeInputs,
                                                    const uint32_t       sizeOutputs,
                                                    const void*          userParams,
                                                    const unsigned       paramsSize,
                                                    const char*          guid,
                                                    const char**         inputLayouts,
                                                    const char**         outputLayouts,
                                                    const std::string&   name)
{
    return createGenericNodeWithId(graphHandle,
                                   inputs,
                                   outputs,
                                   sizeInputs,
                                   sizeOutputs,
                                   userParams,
                                   paramsSize,
                                   guid,
                                   inputLayouts,
                                   outputLayouts,
                                   name,
                                   nullptr);
}

synStatus GraphCompilerSingleton::createGenericNodeWithId(const synGraphHandle graphHandle,
                                                          const synTensor*     inputs,
                                                          const synTensor*     outputs,
                                                          const uint32_t       sizeInputs,
                                                          const uint32_t       sizeOutputs,
                                                          const void*          userParams,
                                                          const unsigned       paramsSize,
                                                          const char*          guid,
                                                          const char**         inputLayouts,
                                                          const char**         outputLayouts,
                                                          const std::string&   name,
                                                          synNodeId*           nodeUniqueId)
{
    synNodeId nodeId = 0;
    synStatus status = m_originalImpl->createGenericNodeWithId(graphHandle,
                                                               inputs,
                                                               outputs,
                                                               sizeInputs,
                                                               sizeOutputs,
                                                               userParams,
                                                               paramsSize,
                                                               guid,
                                                               inputLayouts,
                                                               outputLayouts,
                                                               name,
                                                               &nodeId);

    if (nodeUniqueId)
    {
        *nodeUniqueId = nodeId;
    }

    graph_serialize::Node node(graphHandle,
                               inputs,
                               outputs,
                               sizeInputs,
                               sizeOutputs,
                               userParams,
                               paramsSize,
                               guid,
                               inputLayouts,
                               outputLayouts,
                               name,
                               nodeId);

    // [CID: 45552] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
    // link:
    // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
    m_graphSerializer->addNode(graphHandle, node);

    return status;
}

synStatus GraphCompilerSingleton::createControlDependency(const synGraphHandle graphHandle,
                                                          const synNodeId*     pBlockingNodesIdList,
                                                          const synNodeId*     pBlockedNodesIdList,
                                                          const uint32_t       numberBlocking,
                                                          const uint32_t       numberBlocked)
{
    for (size_t i = 0; i < numberBlocked; ++i)
    {
        std::vector<synNodeId> blocking(pBlockingNodesIdList, pBlockingNodesIdList + numberBlocking);
        m_graphSerializer->setBlockingNodes(graphHandle, pBlockedNodesIdList[i], blocking);
    }
    return m_originalImpl->createControlDependency(graphHandle,
                                                   pBlockingNodesIdList,
                                                   pBlockedNodesIdList,
                                                   numberBlocking,
                                                   numberBlocked);
}

synStatus GraphCompilerSingleton::nodeSetDeterministic(const synGraphHandle graphHandle,
                                                       const synNodeId      nodeId,
                                                       const bool           useDeterministic)
{
    m_graphSerializer->setDeterministic(graphHandle, nodeId, useDeterministic);
    return m_originalImpl->nodeSetDeterministic(graphHandle, nodeId, useDeterministic);
}

synStatus GraphCompilerSingleton::nodeGetDeterministic(const synGraphHandle graphHandle,
                                                       const synNodeId      nodeId,
                                                       bool*                pUseDeterministic)
{
    return m_originalImpl->nodeGetDeterministic(graphHandle, nodeId, pUseDeterministic);
}

synStatus GraphCompilerSingleton::nodeSetRoundingMode(const synGraphHandle  graphHandle,
                                                      const synNodeId       nodeId,
                                                      const synRoundingMode roundingMode)
{
    m_graphSerializer->setRoundingMode(graphHandle, nodeId, roundingMode);  // to record the roundingMode
    return m_originalImpl->nodeSetRoundingMode(graphHandle, nodeId, roundingMode);
}

synStatus GraphCompilerSingleton::nodeGetRoundingMode(const synGraphHandle graphHandle,
                                                      const synNodeId      nodeId,
                                                      synRoundingMode*     pRoundingMode)
{
    return m_originalImpl->nodeGetRoundingMode(graphHandle, nodeId, pRoundingMode);
}

synStatus GraphCompilerSingleton::nodeSetParams(const synGraphHandle graphHandle,
                                                const synNodeId      nodeId,
                                                const void*          userParams,
                                                const unsigned       paramsSize)
{
    m_graphSerializer->setParams(graphHandle, nodeId, userParams, paramsSize);
    return m_originalImpl->nodeSetParams(graphHandle, nodeId, userParams, paramsSize);
}

/*DEPRECATED*/
synStatus GraphCompilerSingleton::graphSetAttribute(const synGraphHandle     graphHandle,
                                                    const synGraphAttribute* attributes,
                                                    const uint64_t*          values,
                                                    const uint32_t           size)
{
    m_graphSerializer->setGraphAttributes(graphHandle, attributes, values, size);
    return m_originalImpl->graphSetAttribute(graphHandle, attributes, values, size);
}

synStatus GraphCompilerSingleton::graphSetAttributes(const synGraphHandle        graphHandle,
                                                     const synGraphAttribute*    attributes,
                                                     const synGraphAttributeVal* values,
                                                     const uint32_t              size)
{
    m_graphSerializer->setGraphAttributesV2(graphHandle, attributes, values, size);
    return m_originalImpl->graphSetAttributes(graphHandle, attributes, values, size);
}

std::optional<uint16_t> GraphCompilerSingleton::getRecipeId(const synRecipeHandle recipeHandle)
{
    const debug_info_t* dbgInfo = nullptr;
    return getRecipeDebugInfo(recipeHandle, &dbgInfo) == synSuccess ? std::optional<uint16_t>(dbgInfo->recipe_id)
                                                                    : std::nullopt;
}

synStatus GraphCompilerSingleton::compileGraph(synRecipeHandle*     pRecipeHandle,
                                               const synGraphHandle graphHandle,
                                               const char*          recipeName,
                                               const char*          buildLog)
{
    GCFG_ENABLE_PROFILER.setValue(true);  // enable debug info for recipe ID, doesn't actually enable profiling.
    const std::string notEmptyRecipeName = recipeName != nullptr && *recipeName != '\0'
                                               ? recipeName
                                               : fmt::format("synrec-auto-gen-{}", (void*)graphHandle);

    bool        isSynrecRecordingOn = isSynrecRecording();
    bool        serializeCalled     = false;
    uint32_t    graphIndex          = -1;
    std::string uniqueId            = fmt::format("{}-{}-{}", (void*)graphHandle, recipeName, m_rankId);
    if (!skipGraphRecording(notEmptyRecipeName))
    {
        if (sConfig.preGraph.value())
        {
            graphIndex = m_graphSerializer->serialize(graphHandle, notEmptyRecipeName, isSynrecRecordingOn, uniqueId);
            serializeCalled = true;
        }
        if (sConfig.postGraph.value())
        {
            m_originalImpl->setCfg(GCFG_DUMP_POST_GRAPHS.primaryName().c_str(), sConfig.graphPath.value().c_str());
        }
        if (sConfig.passesGraph.value())
        {
            m_originalImpl->setCfg(hl_gcfg::getEnableExperimentalFlagsPrimaryName().c_str(), "True");
            m_originalImpl->setCfg(GCFG_DUMP_PASSES_GRAPHS.primaryName().c_str(), sConfig.graphPath.value().c_str());
            if (sConfig.passesFilter.has_value())
            {
                m_originalImpl->setCfg(GCFG_DUMP_PASSES_FILTER.primaryName().c_str(), sConfig.passesFilter->c_str());
            }
        }
    }

    synStatus status = m_originalImpl->compileGraph(pRecipeHandle, graphHandle, notEmptyRecipeName.c_str(), buildLog);
    if (graphIndex != -1)
    {
        m_graphSerializer->postCompilationUpdate(graphIndex, getRecipeId(*pRecipeHandle));
    }
    if (!isSynrecRecordingOn && serializeCalled)
    {
        m_graphSerializer->mapRecipeToGraphJson(pRecipeHandle, uniqueId);
    }
    return status;
}

}  // namespace gc_shim

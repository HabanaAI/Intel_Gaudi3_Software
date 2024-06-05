#include "performance_base_test.h"

#include "hpp/syn_tensor.hpp"
#include "performance_playback_tests.h"

#include "base_test.h"
#include "hpp/syn_graph.hpp"
#include "hpp/syn_infra.hpp"

#include "graph_loader.h"
#include "habana_global_conf.h"
#include "json_utils.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "tpc_elf_api.hpp"
#include "utils/data_provider.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#define TIME_SYNAPSE_API(function, ...)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        synStatus      status_ = synSuccess;                                                                           \
        constexpr auto type    = getStatisticsCollectionVal(#function);                                                \
        if (type != STATISTICS_COLLECTION_LAST && m_enabled_statistics[type])                                          \
        {                                                                                                              \
            const auto start = std::chrono::high_resolution_clock::now();                                              \
            status_          = function(__VA_ARGS__);                                                                  \
            const auto end   = std::chrono::high_resolution_clock::now();                                              \
            m_statistics[type] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();           \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            status_ = function(__VA_ARGS__);                                                                           \
        }                                                                                                              \
        if (SYN_UNLIKELY(status_ != synSuccess))                                                                       \
        {                                                                                                              \
            throw syn::Exception(status_,                                                                              \
                                 fmt::format("{}, in file: {} ({}), function: {}, after: {}",                          \
                                             syn::toString(status_),                                                   \
                                             SYN_FILENAME,                                                             \
                                             __LINE__,                                                                 \
                                             __func__,                                                                 \
                                             #function "(" #__VA_ARGS__ ")"));                                         \
        }                                                                                                              \
    } while (false)

namespace json_tests
{
void SimpleAllocator::setBuffer(void* buffer, size_t capacity, size_t alignment)
{
    if (m_buffer != nullptr)
    {
        throw std::runtime_error(fmt::format("allocator {} already has assigned memory area", m_name));
    }
    m_buffer    = buffer;
    m_capacity  = capacity;
    m_alignment = alignment;
}

void SimpleAllocator::resetBuffer()
{
    m_buffer      = nullptr;
    m_capacity    = 0;
    m_nextAddress = 0;
    m_alignment   = 128;
}

void* SimpleAllocator::allocate(uint64_t sizeInBytes)
{
    uint64_t origOffset = m_nextAddress;
    m_nextAddress += sizeInBytes;
    // align to cache line
    m_nextAddress += m_alignment - (m_nextAddress % m_alignment);
    if (m_nextAddress > m_capacity)
    {
        throw std::runtime_error(
            fmt::format("allocator {} has insufficient memory for allocation of {} bytes", m_name, sizeInBytes));
    }
    return static_cast<std::byte*>(m_buffer) + origOffset;
}

PerformanceBaseTest::PerformanceBaseTest(const ArgParser& args, bool run, bool enableAPIStats, bool eventPerLaunch)
: JsonTest(args),
  m_run(run),
  m_quietMode(args.getValueOrDefault(an_quiet, false)),
  m_keepGoing(args.getValueOrDefault(an_keep_going, false)),
  m_eagerMode(args.getValueOrDefault(an_eager, false)),
  m_eventPerLaunch(eventPerLaunch),
  m_testIterations(args.getValueOrDefault(an_test_iter, 1)),
  m_deviceType(deviceTypeFromString(args.getValue<std::string>(an_device_type))),
  m_statsFilePath(args.getValueOrDefault(an_stats_file, std::string())),
  m_deviceBufferAllocator("DeviceBufferAllocator")
{
    setup();
    if (enableAPIStats)
    {
        auto synAPIFunctionNames = args.getValues<std::string>(an_synapse_api_funcs);
        if (synAPIFunctionNames.empty())
        {
            m_enabled_statistics.set();
        }
        for (const auto& funcName : synAPIFunctionNames)
        {
            auto statsIndex                  = getStatisticsCollectionVal(funcName);
            m_enabled_statistics[statsIndex] = true;
        }
    }
}

PerformanceBaseTest::~PerformanceBaseTest()
{
    try
    {
        releaseDeviceResources();
        cleanupSynapseInstance();
    }
    catch (...)
    {
    }
}

void PerformanceBaseTest::setup()
{
    if (m_testIterations < 1)
    {
        throw std::runtime_error("NUMBER_OF_TEST_ITERATIONS must be >= 1");
    }
    initializeSynapseInstance();
    setConfiguration();
    acquireDeviceResources();
}

void PerformanceBaseTest::setConfiguration()
{
    nlohmann_hcl::json globalConfig =
        json_utils::get(m_jsonFileLoader->getJsonData(), "global_config", nlohmann_hcl::json());
    std::string_view                                 blackList[] = {"POD_SIZE"};
    std::vector<std::pair<std::string, std::string>> configValues;
    configValues.reserve(globalConfig.size());
    // ENABLE_EXPERIMENTAL_FLAGS should come first to handle properly the
    // experimental configurations.
    if (auto it = globalConfig.find("ENABLE_EXPERIMENTAL_FLAGS"); it != globalConfig.end())
    {
        configValues.emplace_back(it.key(), it.value());
    }
    for (auto it = globalConfig.begin(); it != globalConfig.end(); ++it)
    {
        if (find(std::begin(blackList), std::end(blackList), it.key()) != std::end(blackList)) continue;
        configValues.emplace_back(it.key(), it.value());
    }
    for (const auto& [configName, configVal] : configValues)
    {
        if (std::getenv(configName.c_str()) == nullptr)
        {
            synStatus status = synConfigurationSet(configName.c_str(), configVal.c_str());
            if (status != synSuccess)
            {
                JT_LOG_ERR(fmt::format("failed to set configuration {} to value {}", configName, configVal));
            }
        }
    }
}

void PerformanceBaseTest::resetStats()
{
    m_statistics.fill(0);
}

void PerformanceBaseTest::recordJsonStats(const size_t graphIndex)
{
    if (!m_statsFilePath.empty())
    {
        for (int statIndex = 0; statIndex < m_statistics.size(); statIndex++)
        {
            auto type = static_cast<enum StatisticsCollection>(statIndex);
            if (m_enabled_statistics[type])
            {
                std::string graphIndexStr(std::to_string(graphIndex));
                std::string functionName(getSynAPIFunctionName(type, m_eagerMode));
                m_stats["graphs"][graphIndexStr][functionName].push_back(m_statistics[type]);
            }
        }
        m_stats["iters"].push_back(m_totalHostTime);
    }
}

void PerformanceBaseTest::dumpStats() const
{
    if (!m_statsFilePath.empty())
    {
        json_utils::jsonToFile(m_stats, m_statsFilePath);
    }
}

void PerformanceBaseTest::fillTensorAndSectionCreationParams(const nlohmann_hcl::json& jsonGraph,
                                                             GraphParams&              graphParams)
{
    const auto& tensors = json_utils::get(jsonGraph, "tensors");

    graphParams.sections.reserve(tensors.size());
    graphParams.tensors.reserve(tensors.size());
    graphParams.tensorHandleToInternalIndex.reserve(tensors.size());
    graphParams.tensorNameToInternalIndex.reserve(tensors.size());
    graphParams.inputTensors.reserve(tensors.size());
    graphParams.sectionsParams.resize(tensors.size());
    graphParams.tensorsParams.resize(tensors.size());
    graphParams.tensorHandleMappingVec.resize(tensors.size());
    graphParams.maxSectionIdx = 0;
    std::unordered_map<uint32_t, uint32_t> persistentSectionIndexToInternalIndex;
    std::unordered_map<uint32_t, uint32_t> nonPersistentSectionIndexToInternalIndex;

    uint32_t internalTensorIndex  = 0;
    uint32_t internalSectionIndex = 0;
    for (const auto& t : tensors)
    {
        auto& tensorParams = graphParams.tensorsParams[internalTensorIndex++];

        tensorParams.name                 = json_utils::get(t, "name");
        tensorParams.isConst              = json_utils::get(t, "is_const");
        tensorParams.isExternal           = json_utils::get(t, "external", false);
        tensorParams.isPersistent         = json_utils::get(t, "persistent");
        tensorParams.userMemOffset        = json_utils::get<uint64_t>(t, "user_mem_offset", 0);
        tensorParams.tensorType           = JsonGraphLoader::tensorTypeFromString(json_utils::get(t, "type"));
        tensorParams.allowPermutation     = json_utils::get(t, "allow_permutation", false);
        tensorParams.internalSectionIndex = INVALID_INTERNAL_SECTION_INDEX;

        if (tensorParams.userMemOffset > 0)
        {
            throw std::runtime_error("section offset is unsupported");
        }

        if (Launcher::getTensorMemType(tensorParams.tensorType) == Launcher::TensorMemType::HOST)
        {
            throw std::runtime_error("dynamic shapes are not supported");
        }

        auto dataType    = JsonGraphLoader::dataTypeFromString(json_utils::get(t, "dtype"));
        auto maxShape    = json_utils::get(t, "max_shape", std::vector<TSize>());
        auto minShape    = json_utils::get(t, "min_shape", std::vector<TSize>());
        auto strides     = json_utils::get(t, "strides", std::vector<TStride>());
        auto permutation = json_utils::get(t, "permutation", std::vector<uint8_t>());
        bool isIdentityPermutation = true;
        for (unsigned i = 0; i < permutation.size(); i++)
        {
            if (permutation[i] != i)
            {
                isIdentityPermutation = false;
                break;
            }
        }
        if (isIdentityPermutation)
        {
            permutation.clear();
        }
        else
        {
            tensorParams.permutation.dims = permutation.size();
            std::copy(permutation.begin(), permutation.end(), std::begin(tensorParams.permutation.permutation));
        }

        tensorParams.layout.deviceDataType = dataType;
        if (permutation.empty())
        {
            std::copy(strides.begin(), strides.end(), std::begin(tensorParams.layout.strides));
        }
        else
        {
            std::fill(std::begin(tensorParams.layout.strides), std::end(tensorParams.layout.strides), 0);
        }

        tensorParams.geometry[synGeometryMinSizes].dims = minShape.size();
        std::copy(minShape.begin(), minShape.end(), std::begin(tensorParams.geometry[synGeometryMinSizes].sizes));

        tensorParams.geometry[synGeometryMaxSizes].dims = maxShape.size();
        std::copy(maxShape.begin(), maxShape.end(), std::begin(tensorParams.geometry[synGeometryMaxSizes].sizes));

        graphParams.tensorNameToInternalIndex[tensorParams.name] = internalTensorIndex - 1;

        if (tensorParams.isConst)
        {
            std::vector<uint8_t> constData  = json_utils::get(t, "data", std::vector<uint8_t>());
            uint64_t             tensorSize = syn::Tensor::getSizeInBytes(tensorParams.geometry[synGeometryMaxSizes],
                                                              tensorParams.layout,
                                                              tensorParams.layout.deviceDataType);
            uint64_t             dataSize   = constData.size();
            if (dataSize > 0 && dataSize < tensorSize)
            {
                constData = JsonGraphLoader::decompress(constData, tensorSize);
            }
            if (constData.empty())
            {
                tensorParams.data.resize(tensorSize, 0);
            }
            else
            {
                tensorParams.data = std::move(constData);
            }
            continue;
        }

        if (tensorParams.tensorType == HOST_SHAPE_TENSOR || tensorParams.tensorType == HOST_TO_DEVICE_TENSOR)
        {
            tensorParams.data = json_utils::get(t, "data", std::vector<uint8_t>());
            if (tensorParams.data.empty())
            {
                throw std::runtime_error(fmt::format("host tensor: {} is missing required data", tensorParams.name));
            }
        }

        bool isPersistent = tensorParams.isPersistent;
        bool rmwSection   = json_utils::get(t, "rmw_section", false);

        if (!isPersistent && !rmwSection) continue;

        auto sectionsIndices = json_utils::get(t, "user_mem_section_index", std::vector<unsigned>());
        if (sectionsIndices.empty())
        {
            tensorParams.internalSectionIndex                  = internalSectionIndex;
            graphParams.sectionsParams[internalSectionIndex++] = {isPersistent, rmwSection};
        }
        else
        {
            auto& sectionMap =
                isPersistent ? persistentSectionIndexToInternalIndex : nonPersistentSectionIndexToInternalIndex;
            uint32_t sectionIndex = sectionsIndices.front();
            auto     it           = sectionMap.find(sectionIndex);
            if (it == sectionMap.end())
            {
                tensorParams.internalSectionIndex                   = internalSectionIndex;
                persistentSectionIndexToInternalIndex[sectionIndex] = internalSectionIndex;
                graphParams.sectionsParams[internalSectionIndex++]  = {isPersistent, rmwSection};
                graphParams.maxSectionIdx = std::max<size_t>(graphParams.maxSectionIdx, sectionIndex);
            }
            else
            {
                tensorParams.internalSectionIndex = it->second;
            }
        }
    }
}

void PerformanceBaseTest::fillNodeCreationParams(const nlohmann_hcl::json& jsonGraph, GraphParams& graphParams)
{
    const auto& jsonNodes = json_utils::get(jsonGraph, "nodes");

    graphParams.nodes.reserve(jsonNodes.size());
    graphParams.nodesParams.resize(jsonNodes.size());
    graphParams.nodeHandleMappingVec.resize(jsonNodes.size());

    uint32_t internalNodeIndex = 0;
    for (const auto& n : jsonNodes)
    {
        auto& nodeParams = graphParams.nodesParams[internalNodeIndex++];

        nodeParams.name   = json_utils::get(n, "name");
        nodeParams.guid   = json_utils::get(n, "guid");
        nodeParams.params = json_utils::get(n, "params", std::vector<uint8_t>());

        graphParams.nodeNameToInternalIndex[nodeParams.name] = internalNodeIndex - 1;

        std::optional<synRoundingMode> roundingMode = json_utils::get_opt<synRoundingMode>(n, "rounding_mode");
        if (GCFG_ENABLE_ROUNDING_MODE_PLAYBACK.value())
        {
            nodeParams.roundingMode = roundingMode;
        }

        auto inputTensorNames = json_utils::get(n, "input_tensors", std::vector<std::string>());
        nodeParams.inputTensorInternalIndices.reserve(inputTensorNames.size());
        nodeParams.inputTensors.reserve(inputTensorNames.size());
        for (const std::string& tensorName : inputTensorNames)
        {
            nodeParams.inputTensorInternalIndices.push_back(
                tensorName.empty() ? INVALID_INTERNAL_TENSOR_INDEX : graphParams.tensorNameToInternalIndex[tensorName]);
            graphParams.inputTensors.insert(tensorName);
        }

        auto outputTensorNames = json_utils::get(n, "output_tensors", std::vector<std::string>());
        nodeParams.outputTensorInternalIndices.reserve(outputTensorNames.size());
        nodeParams.outputTensors.reserve(outputTensorNames.size());
        for (const std::string& tensorName : outputTensorNames)
        {
            nodeParams.outputTensorInternalIndices.push_back(
                tensorName.empty() ? INVALID_INTERNAL_TENSOR_INDEX : graphParams.tensorNameToInternalIndex[tensorName]);
        }

        nodeParams.inputLayouts = json_utils::get(n, "input_layouts", std::vector<std::string>());
        nodeParams.cstringInputLayouts.reserve(nodeParams.inputLayouts.size());
        for (const auto& layout : nodeParams.inputLayouts)
        {
            nodeParams.cstringInputLayouts.emplace_back(layout.c_str());
        }

        nodeParams.outputLayouts = json_utils::get(n, "output_layouts", std::vector<std::string>());
        nodeParams.cstringOutputLayouts.reserve(nodeParams.outputLayouts.size());
        for (const auto& layout : nodeParams.outputLayouts)
        {
            nodeParams.cstringOutputLayouts.emplace_back(layout.c_str());
        }

        nodeParams.blockingNodeNames = json_utils::get(n, "blocking_nodes", std::vector<std::string>());
    }

    internalNodeIndex = 0;
    for (auto& nodeParams : graphParams.nodesParams)
    {
        nodeParams.blockingNodeInternalIndices.reserve(nodeParams.blockingNodeNames.size());
        nodeParams.blockingNodes.reserve(nodeParams.blockingNodeNames.size());
        for (const auto& nodeName : nodeParams.blockingNodeNames)
        {
            nodeParams.blockingNodeInternalIndices.push_back(graphParams.nodeNameToInternalIndex[nodeName]);
        }
    }
}

void PerformanceBaseTest::fillRecipeParams(const GraphParams& graphParams, RecipeParams& recipeParams)
{
    // we follow the Pytorch bridge behavior and construct the tensor launch info
    // based on the known knowledge we have about persistent tensor creation ordering
    // instead of following the API suggestion to use synTensorRetrieveLaunchAmount
    // and synTensorRetrieveLaunchIds before calling synTensorRetrieveLaunchInfoByIdExt API.
    recipeParams.sectionAddresses.resize(graphParams.maxSectionIdx + 1);
    recipeParams.retrievedLaunchTensorInfo.reserve(graphParams.tensorsParams.size());
    for (uint64_t i = 0; i < graphParams.tensorsParams.size(); i++)
    {
        if (graphParams.tensorsParams[i].isPersistent)
        {
            recipeParams.retrievedLaunchTensorInfo.push_back({});
            synRetrievedLaunchTensorInfoExt& tensorLaunchParams = recipeParams.retrievedLaunchTensorInfo.back();
            tensorLaunchParams.tensorId                         = recipeParams.retrievedLaunchTensorInfo.size() - 1;
        }
    }
    recipeParams.retrievedLaunchTensorInfo.shrink_to_fit();
    recipeParams.launchTensorInfo.resize(recipeParams.retrievedLaunchTensorInfo.size());
}

void PerformanceBaseTest::createSections(synGraphHandle graph, GraphParams& graphParams)
{
    for (const auto& sectionParams : graphParams.sectionsParams)
    {
        synSectionHandle currentSection;
        TIME_SYNAPSE_API(synSectionCreate, &currentSection, 0, graph);
        TIME_SYNAPSE_API(synSectionSetPersistent, currentSection, sectionParams.isPersistent);
        TIME_SYNAPSE_API(synSectionSetRMW, currentSection, sectionParams.rmwSection);
        graphParams.sections.emplace_back(currentSection);
    }
}

void PerformanceBaseTest::createTensors(synGraphHandle graph, GraphParams& graphParams)
{
    for (auto& tensorParams : graphParams.tensorsParams)
    {
        synTensor currentTensor;
        TIME_SYNAPSE_API(synTensorHandleCreate,
                         &currentTensor,
                         graph,
                         tensorParams.tensorType,
                         tensorParams.name.c_str());
        TIME_SYNAPSE_API(synTensorSetExternal, currentTensor, tensorParams.isExternal);
        TIME_SYNAPSE_API(synTensorSetAllowPermutation, currentTensor, tensorParams.allowPermutation);

        if (tensorParams.permutation.dims > 0)
        {
            TIME_SYNAPSE_API(synTensorSetPermutation, currentTensor, &tensorParams.permutation);
        }

        if (tensorParams.geometry[synGeometryMinSizes].dims > 0)
        {
            TIME_SYNAPSE_API(synTensorSetGeometry,
                             currentTensor,
                             &tensorParams.geometry[synGeometryMinSizes],
                             synGeometryMinSizes);
        }

        if (tensorParams.geometry[synGeometryMaxSizes].dims > 0)
        {
            TIME_SYNAPSE_API(synTensorSetGeometry,
                             currentTensor,
                             &tensorParams.geometry[synGeometryMaxSizes],
                             synGeometryMaxSizes);
            TIME_SYNAPSE_API(synTensorSetDeviceFullLayout, currentTensor, &tensorParams.layout);
        }

        if (!tensorParams.data.empty())
        {
            TIME_SYNAPSE_API(synTensorSetHostPtr,
                             currentTensor,
                             reinterpret_cast<void*>(tensorParams.data.data()),
                             tensorParams.data.size(),
                             tensorParams.layout.deviceDataType,
                             true);
        }

        if (tensorParams.internalSectionIndex != INVALID_INTERNAL_SECTION_INDEX)
        {
            TIME_SYNAPSE_API(synTensorAssignToSection,
                             currentTensor,
                             graphParams.sections[tensorParams.internalSectionIndex],
                             tensorParams.userMemOffset);
        }
        graphParams.tensorHandleToInternalIndex[currentTensor] = graphParams.tensors.size();
        graphParams.tensors.emplace_back(currentTensor);
    }
}

void PerformanceBaseTest::createNodes(synGraphHandle graph, GraphParams& graphParams)
{
    for (auto& nodeParams : graphParams.nodesParams)
    {
        for (const auto& internalTensorIndex : nodeParams.inputTensorInternalIndices)
        {
            nodeParams.inputTensors.emplace_back(graphParams.tensors[internalTensorIndex]);
        }

        for (const auto& internalTensorIndex : nodeParams.outputTensorInternalIndices)
        {
            nodeParams.outputTensors.emplace_back(graphParams.tensors[internalTensorIndex]);
        }

        synNodeId currentNode;
        TIME_SYNAPSE_API(synNodeCreateWithId,
                         graph,
                         nodeParams.inputTensors.data(),
                         nodeParams.outputTensors.data(),
                         nodeParams.inputTensors.size(),
                         nodeParams.outputTensors.size(),
                         static_cast<const void*>(nodeParams.params.data()),
                         nodeParams.params.size(),
                         nodeParams.guid.c_str(),
                         nodeParams.name.c_str(),
                         &currentNode,
                         nodeParams.cstringInputLayouts.data(),
                         nodeParams.cstringOutputLayouts.data());

        if (nodeParams.roundingMode)
        {
            TIME_SYNAPSE_API(synNodeSetRoundingMode, graph, currentNode, *nodeParams.roundingMode);
        }
        graphParams.nodes.emplace_back(currentNode);
    }
}

void PerformanceBaseTest::setBlockingNodes(synGraphHandle graph, GraphParams& graphParams)
{
    for (int nodeInternalIndex = 0; nodeInternalIndex < graphParams.nodesParams.size(); nodeInternalIndex++)
    {
        auto& nodeParams = graphParams.nodesParams[nodeInternalIndex];

        if (nodeParams.blockingNodeInternalIndices.empty()) continue;

        for (const auto& blockingNodeTensorIndex : nodeParams.blockingNodeInternalIndices)
        {
            nodeParams.blockingNodes.emplace_back(graphParams.nodes[blockingNodeTensorIndex]);
        }

        TIME_SYNAPSE_API(synNodeDependencySet,
                         graph,
                         nodeParams.blockingNodes.data(),
                         &graphParams.nodes[nodeInternalIndex],
                         nodeParams.blockingNodes.size(),
                         1);
    }
}

synGraphHandle PerformanceBaseTest::createGraph()
{
    synGraphHandle graphHandle;
    if (m_eagerMode)
    {
        TIME_SYNAPSE_API(synGraphCreateEager, &graphHandle, m_deviceType);
    }
    else
    {
        TIME_SYNAPSE_API(synGraphCreate, &graphHandle, m_deviceType);
    }
    return graphHandle;
}

synGraphHandle PerformanceBaseTest::duplicateGraph(synGraphHandle graph, GraphParams& graphParams)
{
    synGraphHandle duplicateGraph;
    uint32_t       numTensors = 0;
    uint32_t       numNodes   = 0;

    TIME_SYNAPSE_API(synGraphDuplicate, graph, &duplicateGraph, nullptr, &numTensors, nullptr, &numNodes);

    uint32_t origNumTensors = graphParams.tensorHandleMappingVec.size();
    if (numTensors > origNumTensors)
    {
        graphParams.tensorHandleMappingVec.resize(numTensors);
    }

    uint32_t origNumNodes = graphParams.nodeHandleMappingVec.size();
    if (numNodes > origNumNodes)
    {
        graphParams.nodeHandleMappingVec.resize(numNodes);
    }

    TIME_SYNAPSE_API(synGraphDuplicate,
                     graph,
                     &duplicateGraph,
                     graphParams.tensorHandleMappingVec.data(),
                     &numTensors,
                     graphParams.nodeHandleMappingVec.data(),
                     &numNodes);

    for (const auto& [srcTensorHandle, dstTensorHandle] : graphParams.tensorHandleMappingVec)
    {
        auto        origTensorIndex  = graphParams.tensorHandleToInternalIndex[srcTensorHandle];
        const auto& origTensorParams = graphParams.tensorsParams[origTensorIndex];
        if (origTensorParams.isPersistent)
        {
            TIME_SYNAPSE_API(synTensorSetGeometry,
                             dstTensorHandle,
                             &origTensorParams.geometry[synGeometryMaxSizes],
                             synGeometryMaxSizes);
            TIME_SYNAPSE_API(synTensorSetDeviceFullLayout, dstTensorHandle, &origTensorParams.layout);
        }
    }

    try
    {
        TIME_SYNAPSE_API(synGraphInferShapes, duplicateGraph);
    }
    catch (const std::runtime_error& re)
    {
        // shape inference can fail for instance if we have a node requiring a shape tensor such
        // as strided view or reshape. In those cases we set all the shapes.
        for (const auto& [srcTensorHandle, dstTensorHandle] : graphParams.tensorHandleMappingVec)
        {
            auto        origTensorIndex  = graphParams.tensorHandleToInternalIndex[srcTensorHandle];
            const auto& origTensorParams = graphParams.tensorsParams[origTensorIndex];
            TIME_SYNAPSE_API(synTensorSetGeometry,
                             dstTensorHandle,
                             &origTensorParams.geometry[synGeometryMaxSizes],
                             synGeometryMaxSizes);
            TIME_SYNAPSE_API(synTensorSetDeviceFullLayout, dstTensorHandle, &origTensorParams.layout);
        }
    }
    return duplicateGraph;
}

synRecipeHandle PerformanceBaseTest::compileGraph(synGraphHandle graph, const std::string& recipeName)
{
    synRecipeHandle recipe;
    TIME_SYNAPSE_API(synGraphCompile, &recipe, graph, recipeName.c_str(), nullptr);
    return recipe;
}

void PerformanceBaseTest::executeRecipe(synRecipeHandle recipe, RecipeParams& recipeParams, bool waitForCompletion)
{
    // query worksapce size
    uint64_t workspaceSize = 0;
    TIME_SYNAPSE_API(synWorkspaceGetSize, &workspaceSize, recipe);

    // query tensor launch information
    TIME_SYNAPSE_API(synTensorRetrieveLaunchInfoByIdExt,
                     recipe,
                     recipeParams.retrievedLaunchTensorInfo.size(),
                     recipeParams.retrievedLaunchTensorInfo.data());

    // fill launch tensor info
    for (uint64_t i = 0; i < recipeParams.retrievedLaunchTensorInfo.size(); i++)
    {
        const synRetrievedLaunchTensorInfoExt& tensorRetrievedParams = recipeParams.retrievedLaunchTensorInfo[i];
        synLaunchTensorInfoExt&                tensorLaunchParams    = recipeParams.launchTensorInfo[i];
        std::optional<uint64_t>& sectionAddress = recipeParams.sectionAddresses[tensorRetrievedParams.tensorSectionId];
        if (sectionAddress.has_value())
        {
            tensorLaunchParams.pTensorAddress = *sectionAddress;
        }
        else
        {
            tensorLaunchParams.pTensorAddress = 0;
            uint64_t tensorSizeInBytes        = syn::Tensor::getMaxSizeInBytes(tensorRetrievedParams);
            if (tensorSizeInBytes > 0)
            {
                if (Launcher::getTensorMemType(tensorRetrievedParams.tensorType) == Launcher::TensorMemType::DEVICE)
                {
                    tensorLaunchParams.pTensorAddress =
                        reinterpret_cast<uint64_t>(m_deviceBufferAllocator.allocate(tensorSizeInBytes));
                }
            }
            sectionAddress = tensorLaunchParams.pTensorAddress;
        }
        tensorLaunchParams.tensorId   = tensorRetrievedParams.tensorId;
        tensorLaunchParams.tensorName = tensorRetrievedParams.tensorName;
        memcpy(tensorLaunchParams.tensorSize,
               tensorRetrievedParams.tensorMaxSize,
               sizeof(tensorLaunchParams.tensorSize));
        tensorLaunchParams.tensorType = tensorRetrievedParams.tensorType;
    }

    uint64_t workspaceBuffer =
        (workspaceSize == 0) ? 0 : reinterpret_cast<uint64_t>(m_deviceBufferAllocator.allocate(workspaceSize));

    // launch recipe
    TIME_SYNAPSE_API(synLaunchWithExternalEventsExt,
                     m_computeStreamHandle,
                     recipeParams.launchTensorInfo.data(),
                     recipeParams.launchTensorInfo.size(),
                     workspaceBuffer,
                     recipe,
                     nullptr,
                     0,
                     0);

    // record and potentially wait for completion
    TIME_SYNAPSE_API(synEventRecord, m_eventHandles[m_nextLaunchEventId], m_computeStreamHandle);
    if (waitForCompletion)
    {
        TIME_SYNAPSE_API(synEventSynchronize, m_eventHandles[m_nextLaunchEventId]);
    }
    m_nextLaunchEventId = (m_nextLaunchEventId + 1) % m_eventHandles.size();
}

void PerformanceBaseTest::waitForLaunchCompletion()
{
    TIME_SYNAPSE_API(synEventSynchronize, m_eventHandles[m_nextsynchronizeEventId]);
    m_nextsynchronizeEventId = (m_nextsynchronizeEventId + 1) % m_eventHandles.size();
}

void PerformanceBaseTest::cleanup(synGraphHandle*  graph,
                                  synGraphHandle*  duplicateGraph,
                                  synRecipeHandle* recipe,
                                  GraphParams&     graphParams,
                                  RecipeParams&    recipeParams)
{
    for (auto& section : graphParams.sections)
    {
        TIME_SYNAPSE_API(synSectionDestroy, section);
    }
    graphParams.sections.clear();

    for (auto& tensor : graphParams.tensors)
    {
        TIME_SYNAPSE_API(synTensorDestroy, tensor);
    }
    graphParams.tensors.clear();

    graphParams.nodes.clear();
    graphParams.tensorHandleToInternalIndex.clear();
    graphParams.tensorHandleMappingVec.clear();
    graphParams.nodeHandleMappingVec.clear();

    for (auto& nodeParams : graphParams.nodesParams)
    {
        nodeParams.inputTensors.clear();
        nodeParams.outputTensors.clear();
        nodeParams.blockingNodes.clear();
    }

    for (auto& sectionAddress : recipeParams.sectionAddresses)
    {
        sectionAddress.reset();
    }
    graphParams.maxSectionIdx = 0;

    if (graph)
    {
        TIME_SYNAPSE_API(synGraphDestroy, *graph);
        *graph = nullptr;
    }

    if (duplicateGraph)
    {
        TIME_SYNAPSE_API(synGraphDestroy, *duplicateGraph);
        *duplicateGraph = nullptr;
    }

    if (recipe)
    {
        TIME_SYNAPSE_API(synRecipeDestroy, *recipe);
        *recipe = nullptr;
    }
}

void PerformanceBaseTest::initializeSynapseInstance()
{
    TIME_SYNAPSE_API(synInitialize);
}

void PerformanceBaseTest::cleanupSynapseInstance()
{
    TIME_SYNAPSE_API(synDestroy);
}

void PerformanceBaseTest::pickAllocationSize()
{
    m_passingGraphsIndices.reserve(m_graphsIndices.size());
    // disable statistics for the duration of the calculation
    auto enabled_statistics = m_enabled_statistics;
    m_enabled_statistics.reset();
    for (uint64_t graphIndex : m_graphsIndices)
    {
        const nlohmann_hcl::json& currentJsonGraph = m_jsonFileLoader->getGraph(graphIndex);
        auto               graphName        = json_utils::get(currentJsonGraph, "name");
        GraphParams        graphParams;
        RecipeParams       recipeParams;
        graphParams.idx  = graphIndex;
        graphParams.name = graphName;
        fillTensorAndSectionCreationParams(currentJsonGraph, graphParams);
        fillNodeCreationParams(currentJsonGraph, graphParams);
        synGraphHandle currentGraph = createGraph();
        createSections(currentGraph, graphParams);
        createTensors(currentGraph, graphParams);
        createNodes(currentGraph, graphParams);
        setBlockingNodes(currentGraph, graphParams);
        synRecipeHandle currentRecipe = nullptr;
        try
        {
            currentRecipe = compileGraph(currentGraph, graphName);
        }
        catch (const std::runtime_error& re)
        {
            cleanup(&currentGraph, nullptr, &currentRecipe, graphParams, recipeParams);
            if (!m_keepGoing) throw;
            continue;
        }
        m_passingGraphsIndices.push_back(graphIndex);
        size_t                       currentPersistentTensorsSize = 0;
        std::unordered_set<unsigned> visitedSections;
        for (const auto& tensor : graphParams.tensorsParams)
        {
            if (!tensor.isPersistent) continue;
            if (Launcher::getTensorMemType(tensor.tensorType) != Launcher::TensorMemType::DEVICE) continue;
            if (visitedSections.insert(tensor.internalSectionIndex).second == false) continue;
            const auto& currentGeometry = tensor.geometry[synGeometryMaxSizes];
            currentPersistentTensorsSize += syn::Tensor::getElementSizeInBytes(tensor.layout.deviceDataType) *
                                            multiplyElements(std::begin(currentGeometry.sizes),
                                                             std::begin(currentGeometry.sizes) + currentGeometry.dims);
            // align allocation size
            currentPersistentTensorsSize += m_deviceAlignment - (currentPersistentTensorsSize % m_deviceAlignment);
        }
        // query worksapce size
        size_t currentWorkspaceSize = 0;
        TIME_SYNAPSE_API(synWorkspaceGetSize, &currentWorkspaceSize, currentRecipe);
        size_t currentDeviceSize = currentWorkspaceSize + currentPersistentTensorsSize;
        // align allocation size
        currentDeviceSize += m_deviceAlignment - (currentDeviceSize % m_deviceAlignment);
        m_deviceBufferCapacity = std::max(currentDeviceSize, m_deviceBufferCapacity);

        cleanup(&currentGraph, nullptr, &currentRecipe, graphParams, recipeParams);
    }
    // re-enable statistics
    m_enabled_statistics = enabled_statistics;
}

void PerformanceBaseTest::acquireDeviceResources()
{
    if (m_run)
    {
        TIME_SYNAPSE_API(synDeviceAcquireByDeviceType, &m_deviceId, m_deviceType);
        TIME_SYNAPSE_API(synStreamCreateGeneric, &m_uploadStreamHandle, m_deviceId, 0);
        TIME_SYNAPSE_API(synStreamCreateGeneric, &m_downloadStreamHandle, m_deviceId, 0);
        TIME_SYNAPSE_API(synStreamCreateGeneric, &m_computeStreamHandle, m_deviceId, 0);

        synDeviceAttribute attribute       = DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE;
        TIME_SYNAPSE_API(synDeviceTypeGetAttribute, &m_deviceAlignment, &attribute, 1, m_deviceType);

        pickAllocationSize();

        m_eventHandles.resize(m_eventPerLaunch ? m_passingGraphsIndices.size() : 1);
        for (synEventHandle& currentEventHandle : m_eventHandles)
        {
            TIME_SYNAPSE_API(synEventCreate, &currentEventHandle, m_deviceId, 0);
        }

        uint64_t deviceBuffer         = 0;
        TIME_SYNAPSE_API(synDeviceMalloc, m_deviceId, m_deviceBufferCapacity, 0, 0, &deviceBuffer);
        m_deviceBufferAllocator.setBuffer(reinterpret_cast<void*>(deviceBuffer),
                                          m_deviceBufferCapacity,
                                          m_deviceAlignment);
    }
}

void PerformanceBaseTest::releaseDeviceResources()
{
    if (m_run)
    {
        TIME_SYNAPSE_API(synDeviceFree, m_deviceId, reinterpret_cast<uint64_t>(m_deviceBufferAllocator.getBuffer()), 0);
        m_deviceBufferAllocator.resetBuffer();

        // wait for all pending jobs on the device to finish
        TIME_SYNAPSE_API(synDeviceSynchronize, m_deviceId);

        TIME_SYNAPSE_API(synStreamDestroy, m_uploadStreamHandle);
        TIME_SYNAPSE_API(synStreamDestroy, m_downloadStreamHandle);
        TIME_SYNAPSE_API(synStreamDestroy, m_computeStreamHandle);

        for (synEventHandle& currentEventHandle : m_eventHandles)
        {
            TIME_SYNAPSE_API(synEventDestroy, currentEventHandle);
        }
        m_eventHandles.clear();

        TIME_SYNAPSE_API(synDeviceRelease, m_deviceId);
    }
}

}  // namespace json_tests

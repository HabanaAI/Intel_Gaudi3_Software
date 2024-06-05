#include "graph_serializer/graph_serializer.h"

#include "common_type_utils.h"
#include "file_lock.h"
#include "graph_entries_container.hpp"
#include "graph_serializer_impl.h"
#include "graph_serializers/serialize_utils.h"
#include "habana_global_conf_runtime.h"
#include "include/data_serializer/data_serializer.h"
#include "include/data_serializer/ds_types.h"
#include "json_utils.h"
#include "types_exception.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace graph_serialize
{
using Json = nlohmann_hcl::json;

class LazyDataSerializer
{
public:
    LazyDataSerializer(const std::string& filePath, const data_serialize::GraphInfo& graphInfo, const Config& config)
    : m_filePath(filePath), m_graphInfo(graphInfo), m_config(config)
    {
    }

    std::shared_ptr<data_serialize::DataSerializer> get() const
    {
        if (m_filePath.empty()) return nullptr;
        if (!m_dataSerializer)
        {
            m_dataSerializer = data_serialize::DataSerializer::create(m_filePath, m_graphInfo);
        }
        return m_dataSerializer;
    }

    const Config& getConfig() const { return m_config; }

private:
    const std::string                                       m_filePath;
    const data_serialize::GraphInfo                         m_graphInfo;
    const Config                                            m_config;
    mutable std::shared_ptr<data_serialize::DataSerializer> m_dataSerializer;
};

std::unique_ptr<GraphSerializer> GraphSerializer::createGraphSerializer(const Config& config)
{
    switch (config.type)
    {
        case SPLIT:
            return std::make_unique<SplitGraphSerializer>(config);
        case UNIFIED:
            return std::make_unique<UnifiedGraphSerializer>(config);
    }
    return nullptr;
}

static const uint32_t FILE_VERSION = 4;

static Json serializeNode(const Node& n)
{
    Json node;
    node["name"]           = n.name;
    node["guid"]           = n.guid;
    node["graph_index"]    = n.graphId;
    node["params"]         = n.userParams;
    node["input_tensors"]  = graph_serializer::getTensorsNames(n.inputTensors, false);
    node["output_tensors"] = graph_serializer::getTensorsNames(n.outputTensors, false);
    node["input_layouts"]  = n.inputLayouts;
    node["output_layouts"] = n.outputLayouts;
    node["blocking_nodes"] = n.blockingNodes;
    node["id"]             = n.nodeUniqueId;
    if (n.roundingMode.has_value()) node["rounding_mode"] = (int)n.roundingMode.value();
    if (n.deterministic.has_value()) node["deterministic"] = n.deterministic.value();

    return node;
}

static Json serializeGlobalConfigs()
{
    static const std::set<std::string> SKIP = {GCFG_DUMP_PASSES_FILTER.primaryName(),
                                               GCFG_DUMP_PASSES_GRAPHS.primaryName(),
                                               GCFG_DUMP_POST_GRAPHS.primaryName(),
                                               GCFG_DUMP_PRE_GRAPHS.primaryName(),
                                               GCFG_SCAL_CONFIG_FILE_PATH.primaryName()};

    Json ret;
    hl_gcfg::forEachRegisteredGcfgItem([&](auto& key, auto& value) {
        if (SKIP.count(key) > 0) return;
        if (value.isSetFromDefault()) return;
        std::string str = value.getValueStr();

        auto pos = str.find(' ');  // w/a - some global configs add space and units to the value
        ret[key] = str.substr(0, pos);
    });
    return ret;
}

const std::string_view getGraphAttributeEnumName(synGraphAttribute ret)
{
#if MAGIC_ENUM_SUPPORTED
    return magic_enum::enum_name(ret);
#else
    switch (ret)
    {
        TRANSLATE_ENUM_TO_STRING(GRAPH_ATTRIBUTE_INFERENCE)
        TRANSLATE_ENUM_TO_STRING(GRAPH_ATTRIBUTE_QUANTIZATION)
        TRANSLATE_ENUM_TO_STRING(GRAPH_ATTRIBUTE_BACKOFF_FACTOR)
        TRANSLATE_ENUM_TO_STRING(GRAPH_ATTRIBUTE_MAX)
        default:
            HB_ASSERT(false, "Unexpected graph attribute enum");
            return "UNKNOWN RETURN VALUE";
    }
#endif
}

/*DEPRECATED*/
static inline void serializeGraphAttribute(Json&                                                    attributes,
                                           const std::pair<const synGraphAttribute, unsigned long>& attribute)
{
    const std::string attrName(getGraphAttributeEnumName(attribute.first));
    attributes[attrName] = attribute.second;
    return;
}

static inline void serializeGraphAttribute(Json&                                                           attributes,
                                           const std::pair<const synGraphAttribute, synGraphAttributeVal>& attribute)
{
    const std::string attrName(getGraphAttributeEnumName(attribute.first));
    switch (attribute.first)
    {
        case GRAPH_ATTRIBUTE_INFERENCE:
        case GRAPH_ATTRIBUTE_QUANTIZATION:
            attributes[attrName] = attribute.second.iAttrVal;
            break;
        case GRAPH_ATTRIBUTE_BACKOFF_FACTOR:
            attributes[attrName] = attribute.second.dAttrVal;
            break;
        default:
            LOG_CRITICAL(SYNREC, "Unrecognized Graph Attribute during serialize: {}", attribute.first);
    }
    return;
}

static std::string getFilePath(const fs::path& recipeName, const fs::path& path, const std::string& postFix)
{
    const fs::path nonAbsRecipeName = recipeName.is_absolute() ? recipeName.string().substr(1) : recipeName.string();

    const fs::path recipeDir   = nonAbsRecipeName.parent_path();
    const fs::path fileName    = sanitizeFileName(nonAbsRecipeName.filename()) + postFix;
    const fs::path fullPath    = recipeDir.empty() ? path / fileName : path / recipeDir / fileName;
    const fs::path fullPathDir = fullPath.parent_path();

    if (!fs::is_directory(fullPathDir) || !fs::exists(fullPathDir))
    {
        fs::create_directories(fullPathDir);
    }

    return fullPath;
}

static Json generateModelInfo(const std::string& graphFilePath)
{
    Json model;
    model["global_config"] = serializeGlobalConfigs();
    model["name"]          = fs::path(graphFilePath).replace_extension().filename();
    model["version"]       = FILE_VERSION;
    return model;
}

static void writeGraphToFile(const std::string& graphFilePath, const Json& graphData)
{
    Json model = generateModelInfo(graphFilePath);

    model["graphs"].push_back(graphData);

    json_utils::jsonToFile(model, graphFilePath);
}

static std::optional<size_t> writeTensorsToFile(const Tensor* t, const LazyDataSerializer& dataSerializer)
{
    if (!dataSerializer.get()) return std::nullopt;
    const auto& filterByElements = dataSerializer.getConfig().filterByElements;

    data_serialize::TensorMetadata md;
    md.id          = t->getId();
    md.name        = t->getName();
    md.type        = t->getTensorType();
    md.dataType    = t->getElementType();
    md.compression = data_serialize::Compression::NO_COMP;
    md.shape       = std::vector<TSize>(t->getShape().getNSizes().begin(), t->getShape().getNSizes().end());
    md.permutation = std::vector<uint8_t> {};
    md.dataSize    = t->getBufferSizeInBytes();
    md.data        = (!filterByElements.has_value() || t->getTotalElements() == filterByElements.value())
                         ? std::shared_ptr<uint64_t>(reinterpret_cast<uint64_t*>(t->getData()), [](uint64_t* p) {})
                         : nullptr;
    md.validation  = md.data ? data_serialize::TensorValidation::VALID : data_serialize::TensorValidation::INVALID_DATA;
    md.constTensor = true;
    md.launchIndex = -1;

    return dataSerializer.get()->serialize(md);
}

static void updateConstTensors(const SerializeGraphInfo& serializeGraphInfo,
                               const uint16_t            recipeId,
                               const std::string&        tensorsFilePath,
                               const Config&             config)
{
    data_serialize::GraphInfo graphInfo {serializeGraphInfo.recipeName, recipeId, config.rankId};
    auto                      dataSerializer = LazyDataSerializer(tensorsFilePath, graphInfo, config);
    for (const auto& index : serializeGraphInfo.tensorIndices)
    {
        dataSerializer.get()->updateRecipeId(recipeId, index);
    }
}

static std::pair<nlohmann_hcl::json, std::optional<uint16_t>> serializeTensor(const Tensor*             t,
                                                                              const LazyDataSerializer& dataSerializer)
{
    nlohmann_hcl::json tensor;
    synTensorType      tensorType = t->getTensorType();

    tensor["graph_index"] = t->getGraphID();
    tensor["name"]        = t->getName();
    tensor["type"]        = tensorTypeToString(tensorType);
    tensor["dtype"]       = std::string(getStringFromSynDataType(t->getElementType()));
    tensor["is_const"]    = t->isStaticParam();
    tensor["persistent"]  = t->isPersistent();
    tensor["rmw_section"] = t->isPartOfRMWSection();
    tensor["external"]    = t->getTensorIsExternal();

    NSizeArray maxShape = t->getAllNSizesInElements();
    NSizeArray minShape = t->getNMinimalSizesInElements();
    tensor["max_shape"] = std::vector<TSize>(maxShape.begin(), maxShape.begin() + t->getDim());
    tensor["min_shape"] = std::vector<TSize>(minShape.begin(), minShape.begin() + t->getDim());

    tensor["strides"]           = graph_serializer::getStrides(t);
    tensor["permutation"]       = graph_serializer::getPermutation(t);
    tensor["allow_permutation"] = t->getTensorAnnotation().memory.allowPermutation;

    if (t->isPersistent())
    {
        tensor["user_mem_offset"]        = t->getMemorySectionOffset();
        tensor["user_mem_section_index"] = {t->getMemorySectionID()};
    }
    else if (t->isPartOfRMWSection())
    {
        const auto& sectionInfo          = t->getTensorAnnotation().nonPersistentSectionInfo;
        tensor["user_mem_offset"]        = sectionInfo.offsetFromBase.value();
        tensor["user_mem_section_index"] = {sectionInfo.sectionId.value()};
    }

    if (t->inConstSection())
    {
        tensor["is_const_section"] = true;
    }

    if (tensorType == HOST_SHAPE_TENSOR || tensorType == HOST_TO_DEVICE_TENSOR)
    {
        const char*    data     = t->getData();
        const uint64_t dataSize = t->getBufferSizeInBytes();
        tensor["data"]          = std::vector<char>(data, data + dataSize);
    }

    graph_serializer::serializeQuantParams(tensor, t);

    std::optional<size_t> tensorIndex;
    if (t->isStaticParam() || t->inConstSection())
    {
        tensorIndex = writeTensorsToFile(t, dataSerializer);
    }

    return {tensor, tensorIndex};
}

GraphSerializer::GraphSerializer() {}

GraphSerializerImpl::GraphSerializerImpl(const Config& config) : GraphSerializer(), m_config(config) {}

void GraphSerializerImpl::addGraph(const synGraphHandle graph, CompilationMode compileMode)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    m_graphs.emplace(graph, Graph {graph->graphId, compileMode});
}

void GraphSerializerImpl::removeGraph(const synGraphHandle graph)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    m_graphs.erase(graph);
}

const Graph& GraphSerializerImpl::getGraph(const synGraphHandle graph) const
{
    return m_graphs.at(graph);
}

void GraphSerializerImpl::addNode(const synGraphHandle graph, const Node& node)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph& gm = m_graphs.at(graph);
    gm.nodes.push_back(node);

    std::string& nodeName = gm.nodes.back().name;
    if (nodeName.empty())
    {
        nodeName = node.guid + "-" + std::to_string(node.nodeUniqueId);
    }

    auto it = gm.nodesNames.insert(nodeName);
    if (!it.second)
    {
        nodeName += "-" + std::to_string(node.nodeUniqueId);
        gm.nodesNames.insert(nodeName);
    }

    gm.nodeIdToIndex.emplace(node.nodeUniqueId, gm.nodes.size() - 1);
}

void GraphSerializerImpl::setBlockingNodes(const synGraphHandle          graph,
                                           const synNodeId               nodeId,
                                           const std::vector<synNodeId>& blocking)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph&   gm               = m_graphs.at(graph);
    uint64_t blockedNodeIndex = gm.nodeIdToIndex.at(nodeId);
    Node&    blockedNode      = gm.nodes[blockedNodeIndex];
    for (const auto& id : blocking)
    {
        uint64_t nodeIndex = gm.nodeIdToIndex.at(id);
        blockedNode.blockingNodes.push_back(gm.nodes[nodeIndex].name);
    }
}

void GraphSerializerImpl::setDeterministic(const synGraphHandle graph, const synNodeId nodeId, const bool deterministic)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph&   gm        = m_graphs.at(graph);
    uint64_t index     = gm.nodeIdToIndex.at(nodeId);
    Node&    node      = gm.nodes[index];
    node.deterministic = deterministic;
}

void GraphSerializerImpl::setRoundingMode(const synGraphHandle  graph,
                                          const synNodeId       nodeId,
                                          const synRoundingMode roundingMode)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph&   gm       = m_graphs.at(graph);
    uint64_t index    = gm.nodeIdToIndex.at(nodeId);
    Node&    node     = gm.nodes[index];
    node.roundingMode = roundingMode;
}

void GraphSerializerImpl::setParams(const synGraphHandle graphHandle,
                                    const synNodeId      nodeId,
                                    const void*          userParams,
                                    const unsigned int   paramsSize)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph&   gm        = m_graphs.at(graphHandle);
    uint64_t nodeIndex = gm.nodeIdToIndex.at(nodeId);
    Node&    node      = gm.nodes[nodeIndex];

    const char* params = static_cast<const char*>(userParams);
    node.userParams    = params == nullptr ? std::vector<char>() : std::vector<char>(params, params + paramsSize);
}

/*DEPRECATED*/
void GraphSerializerImpl::setGraphAttributes(synGraphHandle           graphHandle,
                                             const synGraphAttribute* attributes,
                                             const uint64_t*          values,
                                             const uint32_t           size)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph& gm = m_graphs.at(graphHandle);
    for (unsigned i = 0; i < size; ++i)
    {
        gm.graphAttributes.emplace(attributes[i], values[i]);
    }
}

void GraphSerializerImpl::setGraphAttributesV2(synGraphHandle              graphHandle,
                                               const synGraphAttribute*    attributes,
                                               const synGraphAttributeVal* values,
                                               const uint32_t              size)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    Graph& gm = m_graphs.at(graphHandle);
    for (unsigned i = 0; i < size; ++i)
    {
        gm.graphAttributes.emplace(attributes[i], values[i].iAttrVal);
    }
}

void GraphSerializerImpl::generateGraph(const synGraphHandle graphHandle,
                                        const std::string&   recipeName,
                                        const std::string&   tensorsFilePath)
{
    auto graphIter = m_graphs.find(graphHandle);
    if (graphIter == m_graphs.end())
    {
        throw SynapseStatusException("failed to serialize graph, graphHandle not found", synFail);
    }
    Graph& graph = graphIter->second;

    std::unordered_set<Tensor*> nodesTensors;
    std::vector<Json>           nodes;
    nodes.reserve(graph.nodes.size());
    for (const auto& n : graph.nodes)
    {
        nodesTensors.insert(n.inputTensors.begin(), n.inputTensors.end());
        nodesTensors.insert(n.outputTensors.begin(), n.outputTensors.end());
        nodes.emplace_back(serializeNode(n));
    }

    nodesTensors.erase(nullptr);

    // if tensors recording is required, recording must be done before the tensors are released and can't be postponed
    data_serialize::GraphInfo graphInfo {recipeName, std::numeric_limits<uint16_t>::max(), m_config.rankId};
    LazyDataSerializer        dataSerializer(tensorsFilePath, graphInfo, m_config);

    std::vector<Json>   tensors;
    std::vector<size_t> tensorIndices;
    tensors.reserve(nodesTensors.size());
    for (const auto& t : nodesTensors)
    {
        auto tensor = serializeTensor(t, dataSerializer);
        tensors.emplace_back(tensor.first);
        if (tensor.second)
        {
            tensorIndices.push_back(tensor.second.value());
        }
    }

    // sorting the tensors by name can make file comparison simpler
    std::sort(tensors.begin(), tensors.end(), [](const Json& a, const Json& b) { return a.at("name") < b.at("name"); });

    Json attributes;
    for (const auto& attribute : graph.graphAttributes)
    {
        serializeGraphAttribute(attributes, attribute);
    }

    Json graphData;
    graphData["compilation_mode"] = graph.graphCompilationMode;
    graphData["nodes"]            = nodes;
    graphData["tensors"]          = tensors;
    graphData["name"]             = recipeName;
    graphData["group"]            = m_config.rankId;

    if (!attributes.empty()) graphData["attributes"] = attributes;

    m_serializeGraphsInfo.emplace_back(SerializeGraphInfo(recipeName, std::move(graphData), std::move(tensorIndices)));
    m_graphs.erase(graphIter);
}

Node::Node(const synGraphHandle graphHandle,
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
           synNodeId            nodeUniqueId)
{
    if (graphHandle == nullptr)
    {
        throw SynapseStatusException(fmt::format("null graph handle provided (func: {})", __func__),
                                     synInvalidArgument);
    }
    const char* params = static_cast<const char*>(userParams);

    graphId = graphHandle->graphId;

    if (sizeInputs > 0)
    {
        inputTensors.resize(sizeInputs);
        std::memcpy(inputTensors.data(), inputs, sizeInputs * sizeof(synTensor));
    }
    if (sizeOutputs > 0)
    {
        outputTensors.resize(sizeOutputs);
        std::memcpy(outputTensors.data(), outputs, sizeOutputs * sizeof(synTensor));
    }

    Node::userParams   = params == nullptr ? std::vector<char>() : std::vector<char>(params, params + paramsSize);
    Node::guid         = guid;
    Node::name         = name;
    Node::nodeUniqueId = nodeUniqueId;

    if (inputLayouts)
    {
        Node::inputLayouts.reserve(sizeInputs);
        for (size_t i = 0; i < sizeInputs; ++i)
        {
            const char* layout = inputLayouts[i];
            Node::inputLayouts.emplace_back(layout == nullptr ? "" : layout);
        }
    }

    if (outputLayouts)
    {
        Node::outputLayouts.reserve(sizeOutputs);
        for (size_t i = 0; i < sizeOutputs; ++i)
        {
            const char* layout = outputLayouts[i];
            Node::outputLayouts.emplace_back(layout == nullptr ? "" : layout);
        }
    }
}

Node::Node(synGraphHandle      newGraphHandle,
           const Node&         orgNode,
           synNodeId           newNodeUniqueId,
           synTensorHandleMap* tensorsMap,
           uint32_t            numTensors)
: graphId(newGraphHandle->graphId),
  inputLayouts(orgNode.inputLayouts),
  outputLayouts(orgNode.outputLayouts),
  userParams(orgNode.userParams),
  guid(orgNode.guid),
  name(orgNode.name),
  nodeUniqueId(newNodeUniqueId),
  deterministic(orgNode.deterministic),
  roundingMode(orgNode.roundingMode)
{
    if (newGraphHandle == nullptr)
    {
        throw SynapseStatusException(fmt::format("null graph handle provided (func: {})", __func__),
                                     synInvalidArgument);
    }

    // Categorize new tensors as inputs or outputs
    auto categorizeTensors = [&](const std::vector<Tensor*>& src, std::vector<Tensor*>& dst) -> bool {
        dst.reserve(src.size());
        for (Tensor* orgTensor : src)
        {
            if (orgTensor == nullptr)
            {
                dst.push_back(nullptr);
                continue;
            }
            for (uint32_t j = 0; j < numTensors; j++)
            {
                if (tensorsMap[j].origHandle == reinterpret_cast<synTensor>(orgTensor))
                {
                    dst.push_back(reinterpret_cast<Tensor*>(tensorsMap[j].newHandle));
                }
            }
        }
        if (dst.size() != src.size())
        {
            LOG_ERR(SYNREC, "Tensors of original node {} don't not match duplicated", name);
            return false;
        }
        return true;
    };

    if (!categorizeTensors(orgNode.inputTensors, Node::inputTensors) ||
        !categorizeTensors(orgNode.outputTensors, Node::outputTensors))
    {
        throw SynapseStatusException(
            fmt::format("Inconsistent number of tensors between original and duplicated graphs (func: {})", __func__),
            synInvalidArgument);
    }
}

SplitGraphSerializer::SplitGraphSerializer(const Config& config) : GraphSerializerImpl(config) {}

std::string SplitGraphSerializer::getJsonFilePath(const std::string& recipeName)
{
    return getFilePath(recipeName, m_config.filePath, fmt::format(".{}.json", m_config.rankId));
}

uint32_t SplitGraphSerializer::serialize(const synGraphHandle graph,
                                         const std::string&   recipeName,
                                         bool                 isRecording,
                                         const std::string&   uniqueId)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    const std::string tensorsFilePath =
        m_config.tensorsFilePath.empty()
            ? ""
            : getFilePath(recipeName, m_config.tensorsFilePath, fmt::format(".{}.db", m_config.rankId));
    generateGraph(graph, recipeName, tensorsFilePath);
    const std::string filePath = getJsonFilePath(recipeName);

    if (isRecording)
    {
        LOG_INFO(SYNREC, "serializing pre graph {} to: {}, rank ID: {}", recipeName, filePath, m_config.rankId);
        writeGraphToFile(filePath, m_serializeGraphsInfo.back().jsonData);
        return m_serializeGraphsInfo.size() - 1;
    }
    else
    {
        LOG_INFO(SYNREC,
                 "mapping pre graph {} and file path {} to unqiueID: {}, rank ID: {}",
                 recipeName,
                 filePath,
                 uniqueId,
                 m_config.rankId);
        m_uniqueIdToRecAttrsMap.emplace(
            uniqueId,
            std::make_shared<RecordAttributes>(recipeName, m_serializeGraphsInfo.back().jsonData, filePath));
        return -1;
    }
}

void SplitGraphSerializer::mapRecipeToGraphJson(synRecipeHandle* pRecipeHandle, const std::string& uniqueId)
{
    if (m_uniqueIdToRecAttrsMap.find(uniqueId) != m_uniqueIdToRecAttrsMap.end())
    {
        m_recipeGraphMap.emplace(*pRecipeHandle, uniqueId);
    }
    else
    {
        LOG_ERR(SYNREC,
                "cannot find graph data mapped to unique ID {} - "
                "graph will cannot be recorded.",
                uniqueId);
        throw SynapseStatusException("failed to map recipe to graph, "
                                     "graph unique ID not found",
                                     synFail);
    }
}

void SplitGraphSerializer::recordGraph(synRecipeHandle recipeHandle, uint16_t recipeId)
{
    if (m_recipeGraphMap.find(recipeHandle) != m_recipeGraphMap.end())
    {
        const std::string&                uniqueId = m_recipeGraphMap[recipeHandle];
        std::shared_ptr<RecordAttributes> recAttrs = m_uniqueIdToRecAttrsMap[uniqueId];
        recAttrs->graphJson["recipe_id"]           = recipeId;
        LOG_INFO(SYNREC,
                 "serializing pre graph {} to: {}, rank ID: {}",
                 recAttrs->name,
                 recAttrs->filePath,
                 m_config.rankId);
        writeGraphToFile(recAttrs->filePath, recAttrs->graphJson);
        m_uniqueIdToRecAttrsMap.erase(uniqueId);
        m_recipeGraphMap.erase(recipeHandle);
    }
}

void SplitGraphSerializer::postCompilationUpdate(uint32_t graphIndex, std::optional<uint16_t> recipeId)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    SerializeGraphInfo& serializeGraphInfo = m_serializeGraphsInfo.at(graphIndex);

    if (recipeId.has_value())
    {
        if (!m_config.tensorsFilePath.empty())
        {
            updateConstTensors(serializeGraphInfo,
                               recipeId.value(),
                               getFilePath(serializeGraphInfo.recipeName,
                                           m_config.tensorsFilePath,
                                           fmt::format(".{}.db", m_config.rankId)),
                               m_config);
        }
        serializeGraphInfo.jsonData["recipe_id"] = recipeId.value();
    }
    writeGraphToFile(getJsonFilePath(serializeGraphInfo.recipeName), serializeGraphInfo.jsonData);
}

UnifiedGraphSerializer::UnifiedGraphSerializer(const Config& config)
: GraphSerializerImpl(config),
  m_lockFilePath(fmt::format("{}.lock", m_config.filePath)),
  m_writeThread(&UnifiedGraphSerializer::writeThread, this)
{
    std::ofstream lockFile(m_lockFilePath, std::ios_base::app);
    generateModel();
}

UnifiedGraphSerializer::~UnifiedGraphSerializer()
{
    m_writing = false;
    m_writeCv.notify_one();

    if (m_writeThread.joinable())
    {
        m_writeThread.join();
    }
}

void UnifiedGraphSerializer::generateModel()
{
    auto fileLock = FileLock::lock(m_lockFilePath);

    Json model = json_utils::jsonFromFile(m_config.filePath);

    if (!model.empty()) return;

    model = generateModelInfo(m_config.filePath);

    json_utils::jsonToFile(model, m_config.filePath);
}

uint32_t UnifiedGraphSerializer::serialize(const synGraphHandle graph,
                                           const std::string&   recipeName,
                                           bool                 isRecording,
                                           const std::string&   uniqueId)
{
    const std::lock_guard<std::mutex> lock(m_mutex);

    generateGraph(graph, recipeName, m_config.tensorsFilePath);
    if (isRecording)
    {
        return m_serializeGraphsInfo.size() - 1;
    }
    else
    {
        LOG_INFO(SYNREC, "mapping pre graph {} to unqiueID: {}, rank ID: {}", recipeName, uniqueId, m_config.rankId);
        m_uniqueIdToGraphJsonMap.emplace(uniqueId, m_serializeGraphsInfo.back().jsonData);
    }
    return -1;
}

void UnifiedGraphSerializer::postCompilationUpdate(uint32_t graphIndex, std::optional<uint16_t> recipeId)
{
    const std::lock_guard<std::mutex> lock(m_writeMutex);

    const SerializeGraphInfo& serializeGraphInfo = m_serializeGraphsInfo.at(graphIndex);

    if (recipeId.has_value())
    {
        if (!m_config.tensorsFilePath.empty())
        {
            updateConstTensors(serializeGraphInfo, recipeId.value(), m_config.tensorsFilePath, m_config);
        }
        m_serializeGraphsInfo[graphIndex].jsonData["recipe_id"] = recipeId.value();
    }

    m_graphsDataWithRecipeId.push_back(m_serializeGraphsInfo[graphIndex].jsonData);
    m_writeCv.notify_one();
}

void UnifiedGraphSerializer::writeThread()
{
    while (m_writing)
    {
        {
            std::unique_lock lk(m_writeMutex);
            m_writeCv.wait(lk, [&] { return !m_graphsDataWithRecipeId.empty() || !m_writing; });
            if (m_graphsDataWithRecipeId.empty()) continue;
        }

        std::vector<nlohmann_hcl::json> graphs;

        {
            std::lock_guard<std::mutex> lk(m_writeMutex);
            graphs.swap(m_graphsDataWithRecipeId);
        }

        auto fileLock = FileLock::lock(m_lockFilePath);

        Json model = json_utils::jsonFromFile(m_config.filePath);

        for (auto& graphData : graphs)
        {
            model["graphs"].push_back(graphData);

            LOG_INFO(SYNREC,
                     "serializing pre graph {} to: {}, rank ID: {}",
                     json_utils::get(graphData, "name", std::string()),
                     m_config.filePath,
                     m_config.rankId);
        }

        json_utils::jsonToFile(model, m_config.filePath);
    }
}

void UnifiedGraphSerializer::mapRecipeToGraphJson(synRecipeHandle* pRecipeHandle, const std::string& uniqueId)
{
    if (m_uniqueIdToGraphJsonMap.find(uniqueId) != m_uniqueIdToGraphJsonMap.end())
    {
        m_recipeGraphMap.emplace(*pRecipeHandle, uniqueId);
    }
    else
    {
        LOG_ERR(SYNREC,
                "cannot find graph data mapped to unique ID {} - "
                "graph cannot be recorded.",
                uniqueId);
        throw SynapseStatusException("failed to map recipe to graph, "
                                     "graph unique ID not found",
                                     synFail);
    }
}

void UnifiedGraphSerializer::recordGraph(synRecipeHandle recipeHandle, uint16_t recipeId)
{
    if (m_recipeGraphMap.find(recipeHandle) != m_recipeGraphMap.end())
    {
        const std::string& uniqueId                     = m_recipeGraphMap[recipeHandle];
        m_uniqueIdToGraphJsonMap[uniqueId]["recipe_id"] = recipeId;
        {
            std::lock_guard<std::mutex> lk(m_writeMutex);
            m_graphsDataWithRecipeId.push_back(m_uniqueIdToGraphJsonMap[uniqueId]);
        }
        m_writeCv.notify_one();
        m_uniqueIdToGraphJsonMap.erase(uniqueId);
        m_recipeGraphMap.erase(recipeHandle);
    }
}
}  // namespace graph_serialize
#include "post_graph_serializer.h"
#include "common_type_utils.h"
#include "graph_serializers/serialize_utils.h"
#include "data_type_utils.h"
#include "habana_graph.h"
#include "include/mme_common/mme_brain.h"
#include "mme_brain_ifc.h"

using namespace graph_serializer;
using Json = nlohmann_hcl::json;

uint32_t PostSerializer::version() const
{
    return 2;
}

std::string PostSerializer::name() const
{
    return "";
}

bool PostSerializer::supports(GraphState state) const
{
    switch (state)
    {
        case GraphState::POST_COMPILE:
        case GraphState::POST_PASS:
            return true;
        case GraphState::PRE_COMPILE:
        case GraphState::GRAPH_STATE_MAX:
            return false;
    }
    return false;
}

NodeVector PostSerializer::getNodes(const HabanaGraph& graph) const
{
    return graph.getExeSortedNodes();
}

Json PostSerializer::serializeTensorBaseInfo(const TensorPtr& t) const
{
    Json tensor;

    tensor["name"]       = t->getName();
    tensor["type"]       = tensorTypeToString(t->getTensorType());
    tensor["dtype"]      = std::string(getStringFromSynDataType(t->getElementType()));
    tensor["persistent"] = t->isPersistent();

    NSizeArray maxShape = t->getAllNSizesInElements();
    NSizeArray minShape = t->getNMinimalSizesInElements();
    tensor["max_shape"] = std::vector<unsigned>(maxShape.begin(), maxShape.begin() + t->getDim());
    tensor["min_shape"] = std::vector<unsigned>(minShape.begin(), minShape.begin() + t->getDim());

    return tensor;
}

Json PostSerializer::serializeTensor(const TensorPtr& t) const
{
    Json     tensor    = serializeTensorBaseInfo(t);
    uint16_t memId     = 0;
    uint64_t memOffset = 0;

    getSectionInfoFromVirtualAddress(t->getTensorOffset(), memId, memOffset);
    bool in_persist_storage = memId >= MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;

    tensor["dtype_bit_size"]     = t->isControlEdge() ? 0 : dataTypeToSizeInBits(t->getElementType());
    tensor["allocation"]         = t->tensorAllocatedInSram() ? "SRAM" : "DRAM";
    tensor["graph_index"]        = t->getGraphID();
    tensor["in_persist_storage"] = in_persist_storage;
    tensor["is_shape"]           = t->isShapeTensor();  // remove when dependent components are aligned
    tensor["is_const"]           = t->isStaticParam();
    tensor["sp_index"]           = t->getShapePlaneIndex();

    tensor["alias"] = t->isAliasedTensor();

    if (t->isAliasedTensor())
    {
        tensor["alias_of"] = Tensor::getRealTensor(t)->getName();
    }

    tensor["is_dense"]          = t->isDenseLayout();
    tensor["is_reduction"]      = t->isReductionEnabled();
    tensor["strides"]           = getStrides(t.get());
    tensor["permutation"]       = getPermutation(t.get());
    tensor["allow_permutation"] = t->getTensorAnnotation().memory.allowPermutation;
    tensor["offset"]            = t->getTensorOffset();
    tensor["rmw_section"]       = t->isPartOfRMWSection();

    if (in_persist_storage)
    {
        tensor["user_mem_offset"]        = memOffset;
        tensor["user_mem_section_index"] = {memId};
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

    return tensor;
}

Json PostSerializer::serializeNodeBaseInfo(const NodePtr& node) const
{
    Json ret;
    ret["name"]           = node->getNodeName();
    ret["guid"]           = node->getGUID();
    ret["type"]           = node->getNodeTypeStr();
    ret["input_tensors"]  = getTensorsNames(node->getInputs(), true);
    ret["output_tensors"] = getTensorsNames(node->getOutputs(), true);

    return ret;
}

Json PostSerializer::serializeNode(const HabanaGraph& graph, const NodePtr& node, uint32_t graphIndex)
{
    Json ret = serializeNodeBaseInfo(node);

    auto& nodeAnnotations = node->getNodeAnnotation();
    ret["id"]             = node->getId();
    ret["context_id"]     = node->getFullContextId();
    ret["graph_index"]    = graphIndex;
    ret["blocking_nodes"] = getNodeNames(graph.getBlockingNodes(node));
    ret["params"]         = node->getParamsRawData();

    ret["input_ctrl_tensors"]  = getTensorsNames(node->getControlInputs(), true);
    ret["output_ctrl_tensors"] = getTensorsNames(node->getControlOutputs(), true);

    ret["exec_order_idx"] = node->getExecutionOrderedIndex();
    ret["is_logical"]     = node->isLogicalOperation();

    if (nodeAnnotations.bundleInfo.is_set())
    {
        ret["bundle_index"]    = nodeAnnotations.bundleInfo.value().bundleIndex;
        ret["operation_index"] = nodeAnnotations.bundleInfo.value().operationIndex;
    }

    ret["dcore_config"] = getPerforationDebugInfo(node);

    if (!nodeAnnotations.inputsCacheMetaData.empty())
    {
        ret["input_cache_allocation"] = Json::array();
        for (size_t inputIdx = 0; inputIdx < node->getNumInputs(); inputIdx++)
        {
            if (!node->getInput(inputIdx)) continue;
            CacheMetaData& cacheMeta = nodeAnnotations.inputsCacheMetaData.at(inputIdx);

            Json cache_alloc;
            cache_alloc["directive"] = cacheMeta.print_directive();
            cache_alloc["class"]     = cacheMeta.print_class();
            cache_alloc["action"]    = cacheMeta.print_action();
            ret["input_cache_allocation"].push_back(cache_alloc);
        }
    }

    if (!nodeAnnotations.outputsCacheMetaData.empty())
    {
        ret["output_cache_allocation"] = Json::array();
        for (size_t outputIdx = 0; outputIdx < node->getNumOutputs(); outputIdx++)
        {
            if (!node->getOutput(outputIdx)) continue;
            CacheMetaData& cacheMeta = nodeAnnotations.outputsCacheMetaData.at(outputIdx);

            Json cache_alloc;
            cache_alloc["directive"] = cacheMeta.print_directive();
            cache_alloc["class"]     = cacheMeta.print_class();
            cache_alloc["action"]    = cacheMeta.print_action();
            ret["output_cache_allocation"].push_back(cache_alloc);
        }
    }

    ret["origin_nodes"] = node->getOriginNodes();

    auto [nodeBreakpointNr, roiNr] = graph.getBreakpointsAndNodeROINr(node);
    ret["breakpoint_before_node"] = m_numBPs;
    ret["num_of_ROIs"]             = roiNr;
    m_numBPs += nodeBreakpointNr;

    ret["engine"] = std::string {node->getEngineTypeStr()};
    if (graph.runsOnMME(node))
    {
        bool hasPerfAttr = (nodeAnnotations.mmeMetaData.mmePerfAttr != nullptr);
        ret["rollups"] =
            hasPerfAttr ? nodeAnnotations.mmeMetaData.mmePerfAttr->rollUpArray : graph.getRollupsArray(node);
        ret["mme_expected_compute_cycles"] =
            hasPerfAttr ? nodeAnnotations.mmeMetaData.mmePerfAttr->expectedRuntimeCycles : 0;
        ret["mme_compute_utilization"] = hasPerfAttr ? nodeAnnotations.mmeMetaData.mmePerfAttr->mmeUtilization : 0;
        ret["mme_node_strategy"]       = nodeAnnotations.mmeMetaData.mmeStrategyDebugString;
        ret["mme_node_recipe"]         = getMmeRecipeDebugInfo(graph, node);
        if (hasPerfAttr && nodeAnnotations.origBigNode)
        {
            const auto& origBigNode   = nodeAnnotations.origBigNode;
            ret["unsliced_node_name"] = origBigNode->getNodeName();
            bool unslicedHasPerfAttr  = (origBigNode->getNodeAnnotation().mmeMetaData.mmePerfAttr != nullptr);
            if (!unslicedHasPerfAttr)
            {
                auto mmeNode = std::dynamic_pointer_cast<MmeNode>(origBigNode);
                mmeNode->getNodeAnnotation().mmeMetaData.mmePerfAttr =
                    std::make_shared<MmeCommon::PerfAttr>(mmeNode->getMmeBrainIfc()->getRecommendedConfigMmePerf());
            }
            ret["unsliced_expected_compute_cycles"] =
                origBigNode->getNodeAnnotation().mmeMetaData.mmePerfAttr->expectedRuntimeCycles;
            ret["unsliced_expected_compute_utilization"] =
                origBigNode->getNodeAnnotation().mmeMetaData.mmePerfAttr->mmeUtilization;
        }
    }

    double dmaDurationUnderDescsSplit = node->getNodeAnnotation().dmaCost.durationUnderDescsSplitInUsec;
    ret["dma_expected_duration"] =
        dmaDurationUnderDescsSplit ? dmaDurationUnderDescsSplit : node->getNodeAnnotation().dmaCost.durationInUsec;

    if (graph.runsOnTPC(node))
    {
        // Add TPC cost model estimations
        if (GCFG_DUMP_TPC_COST_MODEL_DATA.value())
        {
            TPCNode* tpcNode = dynamic_cast<TPCNode*>(node.get());
            HB_ASSERT(tpcNode != nullptr, "invalid node type");

            std::optional<TPCNode::CostModelResult> cost = tpcNode->getCostModelResult();
            if (cost.has_value())
            {
                ret["tpc_tpc_cycles"]  = cost->tpcCyclesFinalDecision;
                ret["tpc_asic_cycles"] = cost->asicCycles;
                ret["tpc_asic_time"]   = cost->asicTimeInUsec;
            }
        }
        ret["tpc_working_engines"] = std::vector<unsigned> {};
        for (unsigned roiIdx = 0; roiIdx < nodeAnnotations.tpcMetaData.utilizationPerLogicalRoi.size(); roiIdx++)
        {
            auto numOfPhysicalEngines = 0;
            for (auto& dcore : nodeAnnotations.tpcMetaData.utilizationPerLogicalRoi[roiIdx])
            {
                numOfPhysicalEngines += dcore.totalNumWorkingEngines;
            }
            ret["tpc_working_engines"].push_back(numOfPhysicalEngines);
        }
    }

    if (!node->getNodeAnnotation().fusedNodes.empty())
    {
        ret["fused_node_graph"] = serializeFusedGraph(graph, node->getNodeName(), node->getNodeAnnotation().fusedNodes);
    }

    return ret;
}

Json PostSerializer::serializeFusedGraph(const HabanaGraph& graph, const std::string& fusedNode, NodeList nodes)
{
    TensorSet nodesTensors;
    nodes.sort([](const NodePtr& a, const NodePtr& b) {
        return a->getExecutionOrderedIndex() < b->getExecutionOrderedIndex();
    });

    std::vector<Json> serializedNodes;
    serializedNodes.reserve(nodes.size());
    for (const auto& n : nodes)
    {
        serializedNodes.emplace_back(serializeNodeBaseInfo(n));
        for (TensorPtr t : n->getOperands())
        {
            nodesTensors.insert(t);
        }
    }

    std::vector<Json> serializedTensors;
    serializedTensors.reserve(nodesTensors.size());
    for (const auto& t : nodesTensors)
    {
        serializedTensors.emplace_back(serializeTensorBaseInfo(t));
    }

    // sorting the tensors by name can make file comparison simpler
    std::sort(serializedTensors.begin(), serializedTensors.end(), [](const Json& a, const Json& b) {
        return a.at("name") < b.at("name");
    });

    Json graphData;
    graphData["nodes"]   = serializedNodes;
    graphData["tensors"] = serializedTensors;
    graphData["name"]    = fusedNode;

    return graphData;
}

Json PostSerializer::serializeGraph(const HabanaGraph& graph, const std::string& graphName, uint32_t index)
{
    m_numBPs                            = 0; // reset the number of breakpoints for multiple serializations support
    Json graphData                      = SerializerBase::serializeGraph(graph, graphName, index);
    graphData["breakpoint_after_graph"] = m_numBPs;
    graphData["recipe_debug_id"]        = graph.getRecipeDebugId();
    graphData["device"]                 = graph.getDeviceType();
    return graphData;
}

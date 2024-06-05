#include "defs.h"
#include "graph_editor.h"
#include "node.h"
#include "passes.h"
#include "habana_graph.h"
#include "synapse_common_types.h"
#include "tpc_node.h"
#include "types.h"
#include "cast_nodes_handler.h"
#include "scoped_configuration_change.h"


TensorPtr insertCastOnMmeInput(HabanaGraph& g, TensorPtr& tensor, synDataType castTo, tpc_lib_api::DeviceId deviceId)
{
    TensorPtr castedInput = nullptr;
    if (tensor->getElementType() != castTo)
    {
        castedInput = createCastTensor(tensor, castTo, tensor->getName() + "_casted");
        NodePtr   castNode  = CastNodeHandler::createCastNode(tensor,
                                                              castedInput,
                                                              tensor->getName() + "_cast",
                                                              deviceId);
        GraphEditor::addNode(g, castNode);
    }

    return castedInput;
}

bool updateMMENodePrecision(HabanaGraph& g)
{
    if (!GCFG_UPDATE_MME_PRECISION.value())
    {
        LOG_DEBUG(DATA_TYPES,
                  "GCFG_UPDATE_MME_PRECISION=false. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(DATA_TYPES,
                  "Update MME input/output data type is enabled in synapse only for Inference and quantization enabled. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    synDataType profilePrecision = getSynDataTypeFromString(GCFG_PROFILE_PRECISION.value());
    HB_ASSERT(profilePrecision != syn_type_na, "{} is an invalid profile precision", GCFG_PROFILE_PRECISION.value());

    bool useDefaultQuantParams = GCFG_ALLOW_DEFAULT_QUANT_PARAMS.value();

    bool updateOutputNodes = GCFG_UPDATE_GRAPH_OUTPUT_MME.value();
    LOG_TRACE(DATA_TYPES,
              "Update {} MME inputs data type to {}",
              updateOutputNodes ? "all" : "non graph output",
              getStringFromSynDataType(profilePrecision));

    const NodeSet& nodes    = g.getNodes();
    NodeVector     mmeNodes;
    for (const NodePtr& node : nodes)
    {
        if (!g.runsOnMME(node)) continue;
        if (!updateOutputNodes && g.isOutputNode(node))
        {
            LOG_DEBUG(DATA_TYPES, "Graph Output mme node {}, will not update its inputs type", node->getNodeName());
            continue;
        }

        bool insertCast = true;
        if (!useDefaultQuantParams && profilePrecision == syn_type_fp8_143)
        {
            for (const TensorPtr& input : {node->getInput(TENSOR_IFM), node->getInput(TENSOR_WEIGHT)})
            {
                if (!input->inConstSection() &&
                    !input->getDynamicRange().isSet &&
                    !input->getQuantizationParams(profilePrecision).m_isUserQuantInfo &&
                    !input->getQuantizationParams(profilePrecision).m_isUserPCQuantInfo)
                {
                    insertCast = false;
                    break;
                }
            }
        }

        if (insertCast)
        {
            mmeNodes.push_back(node);
        }
    }

    // TODO: Remove once [SW-151898] is done
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    tpc_lib_api::DeviceId     deviceId = g.getDeviceId();
    for (const NodePtr& node : mmeNodes)
    {
        TensorPtr ifmTensor = node->getInput(TENSOR_IFM);
        TensorPtr ifmCasted = insertCastOnMmeInput(g, ifmTensor, profilePrecision, deviceId);

        TensorPtr weightTensor = node->getInput(TENSOR_WEIGHT);
        TensorPtr weightCasted = insertCastOnMmeInput(g, weightTensor, profilePrecision, deviceId);

        if (ifmTensor->getElementType() != weightTensor->getElementType())
        {
            LOG_WARN(DATA_TYPES, "MME node {} inputs has different data types", node->getNodeName());
        }

        if (ifmCasted || weightCasted)
        {
            LOG_DEBUG(DATA_TYPES, "Casts were added on inputs of MME node {}", node->getNodeName());
            GraphEditor::editNode(g, node, [&](){
              node->replaceInput(TENSOR_IFM, ifmCasted);
              node->replaceInput(TENSOR_WEIGHT, weightCasted);
            });
        }
    }
    std::string nodeFilter = GCFG_UPDATE_MME_OUTPUT_PRECISION_FILTER.value();
    if (nodeFilter != "")
    {
        std::vector<std::string> splitNodeFilter = splitString(nodeFilter, ',');
        for (const NodePtr& node : mmeNodes)
        {
            const std::string& nodeName = node->getNodeName();
            if (node->getOutput(0)->getElementType() == profilePrecision || std::all_of(splitNodeFilter.begin(), splitNodeFilter.end(),
                 [&](const std::string& nodeFilter) { return nodeName.find(nodeFilter) == std::string::npos; }))
            {
                continue;
            }
            LOG_DEBUG(DATA_TYPES,
                      "Setting MME node {} output precision to  {}",
                      node->getNodeName(),
                      profilePrecision);
            TensorPtr oldMMEOutput = node->getOutput(0);
            TensorPtr newMMEOutput = node->getOutput(0)->clone(false, false);
            newMMEOutput->changeDefaultElementType(profilePrecision);
            GraphEditor::replaceOutput(g, node, 0, newMMEOutput);
            NodePtr castNode = CastNodeHandler::createCastNode(newMMEOutput,
                                                               oldMMEOutput,
                                                               newMMEOutput->getName() + "_cast",
                                                               deviceId);
            GraphEditor::addNode(g, castNode);
        }
    }
    return true;
}
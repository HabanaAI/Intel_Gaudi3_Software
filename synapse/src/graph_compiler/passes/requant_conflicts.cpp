#include <habana_nodes/node_factory.h>
#include "graph_editor.h"
#include "habana_pass.h"
#include "quantizer_factory.h"
#include "data_type_utils.h"

bool requantConflicts(HabanaGraph& g)
{
    if (!GCFG_ENABLE_SYNAPSE_QUANTIZATION.value())
    {
        LOG_DEBUG(QUANT, "Quantization is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    std::set<std::tuple<NodePtr , TensorPtr, TensorPtr>> nodesToUpdate;
    std::set<NodePtr> nodesToAdd;
    std::map<std::string, NodePtr> requantNodesMap;

    for (auto inputTensor : g.getTensors())
    {
        if (inputTensor == nullptr) continue;
        QuantizationConflictMap conflictsMap = inputTensor->getConflictingQuants();
        if (!inputTensor->isRequantLocked()) continue;

        // if the data type is 32 bit, we should not requant
        synDataType inputType = inputTensor->getElementType();
        std::string requantGuid;

        switch (inputType)
        {
            case syn_type_int4:
            case syn_type_uint4:
                requantGuid = "requant_i4";
                break;
            case syn_type_fixed:
            case syn_type_uint8:
                requantGuid = "requant_i8";
                break;
            case syn_type_int16:
            case syn_type_uint16:
                requantGuid = "requant_i16";
                break;
            case syn_type_int32:
            case syn_type_uint32:
            case syn_type_single:
                continue;
            case syn_type_na:
            default:
                continue;
        }

        int nodesPerInputTensor = 0;

        for (auto const& conflict : conflictsMap)
        {
            if (conflict.first == 0) continue;  // Ignore a conflict with a dummy node ID

            std::string additionToNodeName;
            if (nodesPerInputTensor != 0)
            {
                additionToNodeName = "_" + std::to_string(nodesPerInputTensor);
            }

            NodePtr conflictedNode = g.getNodeByID(conflict.first);
            HB_ASSERT(conflictedNode != nullptr, "conflicted node does not exist");
            // Apply pass on node inputs only; if tensor is output of conflictedNode then continue
            if (conflictedNode == g.getTensorProducer(inputTensor)) continue;

            QuantizationMap quantizationParams = conflict.second;
            if (quantizationParams.empty()) continue;

            std::string requantNodeName = fmt::format("requant_{}{}", inputTensor->getName(), additionToNodeName);

            if (requantNodesMap.find(requantNodeName) == requantNodesMap.end())
            {
                nodesPerInputTensor++;
                TensorPtr newOutputTensor = inputTensor->clone(false, false);
                newOutputTensor->setName(fmt::format("{}_requant", inputTensor->getName()), true);
                newOutputTensor->setAllQuantizationParams(quantizationParams);

                NodePtr requantNode = NodeFactory::createGenericTPCNode({inputTensor},
                                                                        {newOutputTensor},
                                                                        nullptr,
                                                                        requantGuid,
                                                                        requantNodeName);
                nodesToAdd.insert(requantNode);
                LOG_DEBUG(QUANT, "Inserted requant node \"{}\" before node \"{}\" on tensor \"{}\"",
                          requantNodeName, conflictedNode->getNodeName(), inputTensor->getName());
                nodesToUpdate.insert(std::make_tuple(conflictedNode, inputTensor, newOutputTensor));
                requantNodesMap[requantNodeName] = requantNode;
            }
            else
            {
                NodePtr requantNode = requantNodesMap[requantNodeName];
                TensorPtr newOutputTensor = requantNode->getOutput(0);
                nodesToUpdate.insert(std::make_tuple(conflictedNode, inputTensor, newOutputTensor));
            }
        }
    }

    for (auto& t : nodesToUpdate)
    {
        NodePtr node;
        TensorPtr oldInputTensor, newInputTensor;
        std::tie(node, oldInputTensor, newInputTensor) = t;
        GraphEditor::replaceTensor(g, node, oldInputTensor, newInputTensor);
    }

    for (auto& n : nodesToAdd)
    {
        GraphEditor::addNode(g, n);
    }

    return true;
}

#include <map>
#include <perf_lib_layer_params.h>
#include "passes.h"
#include "habana_graph.h"

#include "utils.h"
#include "node_factory.h"
#include "graph_editor.h"

float tensorDataAsFloat(pTensor tensor)
{
    char* data = tensor->getData();
    synDataType type = tensor->getElementType();
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
            return (float)(*data);
        case syn_type_int16:
            return (float)(*((int16_t*)data));
        case syn_type_int32:
            return (float)(*((int32_t*)data));
        case syn_type_single:
            return (*((float*)data));

        default:
            HB_ASSERT(false, "Invalid data type {}", type);
            return 0;
    }
}

static void findEluMultNodes(HabanaGraph& g, std::vector<std::tuple<pNode, pNode, synDataType, float>>& foundPatterns)
{
    std::shared_ptr<TPCNode> tpcNode = nullptr;
    NodeSet nodes = g.getNodes();

    for (auto node : nodes)
    {
        tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        // find elu nodes
        if (tpcNode == nullptr || !tpcNode->isGuidPrefix("elu") ||
            !GraphEditor::canEliminateTensor(g, tpcNode->getOutput(0)))
        {
            continue;
        }

        HB_ASSERT(node->getNumOutputs() == 1, "the number of outputs is {} but should be 1", node->getNumOutputs());

        auto nodeOutputs = g.getTensorConsumers(node->getOutput(0));
        for (auto nodeOutput : nodeOutputs)
        {
            tpcNode = std::dynamic_pointer_cast<TPCNode>(nodeOutput);
            if (tpcNode == nullptr || !tpcNode->isGuidPrefix("mult") ||
                !GraphEditor::canEliminateTensor(g, tpcNode->getOutput(0)))
            {
                continue;
            }

            auto multInputTensors = nodeOutput->getInputs();
            for (auto multInputTensor : multInputTensors)
            {
                if (multInputTensor->getTotalElements() != 1 || !multInputTensor->isStaticParam() ||
                    !GraphEditor::canEliminateTensor(g, multInputTensor))
                {
                    // not a scalar tensor or scalar that is used by someone other than the mult node
                    continue;
                }

                foundPatterns.push_back(std::make_tuple(node,
                                                        nodeOutput,
                                                        node->getOutput(0)->getElementType(),
                                                        tensorDataAsFloat(multInputTensor)));
            }
        }
    }
}

pNode createSeluNode(HabanaGraph& g, pNode eluNode, pNode multNode, synDataType type, ns_SeluKernel::Params* params)
{
    std::string_view guid;
    if (GCFG_SYNAPSE_DATA_TYPE_SELECTION.value() && type == syn_type_na)
    {
        guid = "selu";
    }
    else
    {
        switch (type)
        {
            case syn_type_int32:
                guid = "selu_i32";
                break;
            case syn_type_single:
                guid = "selu_f32";
                break;
            case syn_type_int16:
                guid = "selu_i16";
                break;
            case syn_type_uint8:
                guid = "selu_u8";
                break;
            case syn_type_fixed:
                guid = "selu_i8";
                break;
            case syn_type_bf16:
            default:
                HB_ASSERT(false, "unsupported data type for selu, type: {}", type);
                guid = "selu_na";
        }
    }

    pNode seluNode = NodeFactory::createNode(eluNode->getInputs(), multNode->getOutputs(), params, guid, "");

    return seluNode;
}

bool eluMulScalarFusion(HabanaGraph& g)
{

    pNode eluNode = nullptr;
    pNode multNode = nullptr;
    pNode seluNode = nullptr;
    TPCNode* tpcEluNode = nullptr;
    float scalar;
    synDataType type;
    std::vector<std::tuple<pNode, pNode, synDataType, float>> foundPatterns;

    findEluMultNodes(g, foundPatterns);

    for (auto pattern : foundPatterns)
    {
        std::tie(eluNode, multNode, type, scalar) = pattern;
        tpcEluNode = dynamic_cast<TPCNode*>(eluNode.get());
        ns_SeluKernel::Params params;
        params.alpha = ((ns_EluKernel::Params*)tpcEluNode->getParams())->alpha;
        params.gamma = scalar;
        seluNode                         = createSeluNode(g, eluNode, multNode, type, nullptr);
        std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(seluNode);
        HB_ASSERT(tpcNode != nullptr, "created a non TPC node");
        tpcNode->storeParamsInBuffer(&params, sizeof(ns_SeluKernel::Params));
        auto status = GraphEditor::replaceNodes(g, {eluNode, multNode}, {seluNode});
        HB_ASSERT(status == REPLACE_NODE_SUCCESS, "{}: failed to fuse elu and mul nodes", __FUNCTION__);
    }
    return true;
}

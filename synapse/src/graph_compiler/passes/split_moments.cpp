#include "habana_graph.h"
#include "node_factory.h"
#include "node_utils.h"
#include "graph_editor.h"
#include "perf_lib_layer_params.h"
#include "moments_node.h"
#include "bn_utils.h"
#include <unordered_set>
#include "graph_editor.h"

/*
 * Introduction:
 *      Pass to find standalone tf-moments nodes,
 *      and replace them with subgraph that will perform the required moments operartion
 * Precondition:
 *      it is assumed that the pass for moments->bn has already been executed
 */

bool splitMoments(HabanaGraph& g)
{
    NodeVector nodes = g.getExeSortedNodes();

    for (auto node : nodes)
    {
        if (node->getNodeType() != Node::TYPE_MOMENTS) continue;

        HB_ASSERT(node->getNumOutputs() == 2,
        "Impropper number of outputs, must be 2 while {} are defined", node->getNumOutputs());

        TensorPtr outputMean = node->getOutput(0);
        // will be created inside getMoments
        TensorPtr outputSigma;
        TensorPtr outputSigmaSq;

        NodeList nodesList = BNUtils::getMoments(node->getInput(0), outputMean, outputSigma, outputSigmaSq, node->getNodeName());

        // need to add 3rd mult_node to mult in 1/bhw to generate var
        unsigned channels = node->getInput(0)->getSizeInElements(0);
        float_t bhwInvTensorData = (float)channels / (float)node->getInput(0)->getTotalElements();
        char* tData = new char[sizeof(float_t)];
        memcpy(tData, &bhwInvTensorData, sizeof(float_t));

        SizeArray bhwInvSizes  = {1};
        TensorPtr bhwInvTensor = std::make_shared<Tensor>(1, bhwInvSizes.data(), syn_type_float, tData);
        bhwInvTensor->setShouldFreeBuffer(true);
        bhwInvTensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
        bhwInvTensor->setName(node->getNodeName() + "_bhw_inv");

        NodePtr mulNode = NodeFactory::createNode({outputSigmaSq, bhwInvTensor},
                                                {node->getOutput(1)},
                                                nullptr,
                                                "mult_fwd_f32",
                                                node->getNodeName() + "_mult_bhw_inv");
        nodesList.push_back(mulNode);

        LOG_DEBUG(GC, "splitMoments: splitting node: {}", node->getNodeName());

        NodeList oldNodes = {node};
        if (GraphEditor::replaceNodes(g, oldNodes, nodesList) != REPLACE_NODE_SUCCESS)
        {
            LOG_ERR(GC, "{}: failed splitting moments node {}", HLLOG_FUNC, node->getNodeName());
            return false;
        }
    }

    return true;
}

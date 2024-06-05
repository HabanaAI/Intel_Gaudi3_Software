#include "habana_graph.h"
#include "node_factory.h"
#include "node_utils.h"
#include "graph_editor.h"
#include "perf_lib_layer_params.h"
#include "bn_utils.h"
#include "moments_node.h"
#include "tf_batch_norm_node.h"
#include "tpc_kernel_names.h"
#include "graph_editor.h"

bool splitTfBatchNorm(HabanaGraph& g)
{
    auto fn    = [](const NodePtr& node) { return (std::dynamic_pointer_cast<TfBatchNormNode>(node) != nullptr); };
    auto nodes = g.getNodesCond(fn);

    for (auto node : nodes)
    {
        std::shared_ptr<TfBatchNormNode> bnNode = std::dynamic_pointer_cast<TfBatchNormNode>(node);
        if (bnNode == nullptr)
        {
            continue;
        }

        TensorPtr inputFM = node->getInput(0);
        TensorPtr inputMean = node->getInput(1);
        TensorPtr inputVar = node->getInput(2);
        TensorPtr inputBeta = node->getInput(3);
        TensorPtr inputGamma = node->getInput(4);
        TensorPtr outputFM = node->getOutput(0);

        auto dtype = type2Str(inputFM->getElementType());

        NodePtr meanProducer = g.getTensorProducer(inputMean);
        NodePtr varProducer  = g.getTensorProducer(inputVar);

        // if mean and var producer are the same Moments node
        if (meanProducer && varProducer && meanProducer->getId() == varProducer->getId() &&
            meanProducer->getNodeType() == Node::TYPE_MOMENTS)
        {
            LOG_DEBUG(GC, "splitTfBatchNorm: splitting node: {}, with moments", node->getNodeName());
            unsigned channels = inputFM->getSizeInElements(0);
            float    bhw      = (float)inputFM->getTotalElements() / (float)channels;

            // replace moments->tf-bn subgraph with moments1->moments2->fused-bn-var
            NodePtr momentsNode = meanProducer;
            TensorPtr outputSigma;
            TensorPtr outputSigmaSq;
            NodeList momentsList = BNUtils::getMoments(node->getInput(0), inputMean, outputSigma, outputSigmaSq, node->getNodeName());
            ns_BatchNormVarienceKernel::Params bnParams;
            memset(&bnParams, 0, sizeof(bnParams));
            bnParams.epsilon = bnNode->getParams().variance_epsilon;
            bnParams.N = (1 / bhw);
            NodePtr normalizeNode = NodeFactory::createNode({inputFM, inputBeta, inputGamma, inputMean, outputSigmaSq},
                                                            {outputFM, inputVar},
                                                            &bnParams,
                                                            fmt::format("batch_norm_variance_inf{}", dtype),
                                                            node->getNodeName());
            std::dynamic_pointer_cast<TPCNode>(normalizeNode)->storeParamsInBuffer(&bnParams, sizeof(bnParams));

            NodeList nodesToFuse = {momentsNode, bnNode};
            NodeList fusedNodes(momentsList.begin(), momentsList.end());
            fusedNodes.push_back(normalizeNode);

            if (GraphEditor::replaceNodes(g, nodesToFuse, fusedNodes) != REPLACE_NODE_SUCCESS)
            {
                return false;
            }

        }
        else /* tf-bn not connected to moments */
        {
            LOG_DEBUG(GC, "splitTfBatchNorm: replacing node {} with normalize-inf node", node->getNodeName());
            ns_BatchNormKernel::Params bnParams;
            memset(&bnParams, 0, sizeof(bnParams));
            bnParams.epsilon = bnNode->getParams().variance_epsilon;
            NodePtr normalizeNode = NodeFactory::createNode({inputFM, inputBeta, inputGamma, inputMean, inputVar},
                                                            {outputFM},
                                                            &bnParams,
                                                            fmt::format("batch_norm_inf{}", dtype),
                                                            node->getNodeName());
            std::dynamic_pointer_cast<TPCNode>(normalizeNode)->storeParamsInBuffer(&bnParams, sizeof(bnParams));

            NodeList newNodes = {normalizeNode};
            NodeList oldNodes = {node};
            if (GraphEditor::replaceNodes(g, oldNodes, newNodes) != REPLACE_NODE_SUCCESS)
            {
                LOG_ERR(GC,
                        "{}: failed replacing TfBatchNorm node {} with normalize-inf node",
                        HLLOG_FUNC,
                        node->getNodeName());
                return false;
            }
        }
    }

    return true;
}


#include "node_factory.h"
#include "graph_editor.h"
#include "graph_visualization.h"
#include "perf_lib_layer_params.h"
#include "layer_norm_fuser.h"
#include "data_type_utils.h"

bool LayerNormFuser::constructLayerNormPattern(Graph* pattern)
{
    bool patternStatus = true;

    /* Tensors: */
    pTensor input            = std::make_shared<Tensor>();
    pTensor reduceMean1Out   = std::make_shared<Tensor>();
    pTensor squareDiffSubOut = std::make_shared<Tensor>();
    pTensor squareDiffMulOut = std::make_shared<Tensor>();
    pTensor reduceMean2Out   = std::make_shared<Tensor>();
    pTensor batchNormAdd1Out = std::make_shared<Tensor>();
    pTensor sqrtOut          = std::make_shared<Tensor>();
    pTensor reciprocalOut    = std::make_shared<Tensor>();
    pTensor batchNormMul1Out = std::make_shared<Tensor>();
    pTensor batchNormMul2Out = std::make_shared<Tensor>();
    pTensor batchNormMul3Out = std::make_shared<Tensor>();
    pTensor batchNormSubOut  = std::make_shared<Tensor>();
    pTensor batchNormAdd2Out = std::make_shared<Tensor>();

    /* Static tensors: */
    pTensor batchNormAdd1Scalar = std::make_shared<Tensor>();  // epsilon
    pTensor batchNormMul1Scalar = std::make_shared<Tensor>();  // gamma
    pTensor batchNormSubScalar  = std::make_shared<Tensor>();  // betta

    /* Nodes: */
    pNode reduceMean1 = NodeFactory::createGenericTPCNode({input}, {reduceMean1Out}, nullptr,
                                                        "reduce_mean", "reduce_mean1");
    patternStatus = patternStatus && pattern->addNode(reduceMean1);

    pNode squareDiffSub = NodeFactory::createGenericTPCNode({input, reduceMean1Out}, {squareDiffSubOut},
                                                            nullptr, "sub", "squareDiffSub");
    patternStatus = patternStatus && pattern->addNode(squareDiffSub);

    pNode squareDiffMul = NodeFactory::createGenericTPCNode({squareDiffSubOut, squareDiffSubOut}, {squareDiffMulOut},
                                                            nullptr, "mult", "squareDiffMul");
    patternStatus = patternStatus && pattern->addNode(squareDiffMul);

    pNode reduceMean2 = NodeFactory::createGenericTPCNode({squareDiffMulOut}, {reduceMean2Out},
                                                        nullptr, "reduce_mean", "reduce_mean2");
    patternStatus = patternStatus && pattern->addNode(reduceMean2);

    pNode batchNormAdd1 = NodeFactory::createGenericTPCNode({reduceMean2Out, batchNormAdd1Scalar}, {batchNormAdd1Out},
                                                            nullptr, "add", "batchNormAdd1");
    patternStatus = patternStatus && pattern->addNode(batchNormAdd1);

    pNode sqrt = NodeFactory::createGenericTPCNode({batchNormAdd1Out}, {sqrtOut},
                                                nullptr, "sqrt", "sqrt");
    patternStatus = patternStatus && pattern->addNode(sqrt);

    pNode reciprocal = NodeFactory::createGenericTPCNode({sqrtOut}, {reciprocalOut},
                                                        nullptr, "reciprocal", "reciprocal");
    patternStatus = patternStatus && pattern->addNode(reciprocal);

    pNode batchNormMul1 = NodeFactory::createGenericTPCNode({reciprocalOut, batchNormMul1Scalar}, {batchNormMul1Out},
                                                            nullptr, "mult", "batchNormMul1");
    patternStatus = patternStatus && pattern->addNode(batchNormMul1);

    pNode batchNormMul2 = NodeFactory::createGenericTPCNode({input, batchNormMul1Out}, {batchNormMul2Out},
                                                            nullptr, "mult", "batchNormMul2");
    patternStatus = patternStatus && pattern->addNode(batchNormMul2);

    pNode batchNormMul3 = NodeFactory::createGenericTPCNode({reduceMean1Out, batchNormMul1Out}, {batchNormMul3Out},
                                                            nullptr, "mult", "batchNormMul3");
    patternStatus = patternStatus && pattern->addNode(batchNormMul3);

    pNode batchNormSub = NodeFactory::createGenericTPCNode({batchNormSubScalar, batchNormMul3Out}, {batchNormSubOut},
                                                        nullptr, "sub", "batchNormSub");
    patternStatus = patternStatus && pattern->addNode(batchNormSub);

    pNode batchNormAdd2 = NodeFactory::createGenericTPCNode({batchNormMul2Out, batchNormSubOut}, {batchNormAdd2Out},
                                                            nullptr, "add", "batchNormAdd2");
    patternStatus = patternStatus && pattern->addNode(batchNormAdd2);

    return patternStatus;
}

bool LayerNormFuser::fuseLayerNorm(Graph* pattern)
{
    /* Trying to find the layer-norm pattern as it exists BERT */
    /* TODO: SW-13494 generalize the pattern, check latest work done as part of SW-10794 */

    if (constructLayerNormPattern(pattern) == false)
    {
        LOG_DEBUG(GC, "failed to create layer-norm pattern.");
        return false;
    }

    // find all matches for above patterns
    NodeSet matchingNodes = m_graph.matchPatternWithSingleOutputNode(pattern, NodeTypeMatchingFunc);
    unsigned fusions = 0;
    for (pNode lastNode : matchingNodes)
    {
        /* extracting: input, beta, gamma and epsilon tensors. */

        NodeList nodesToRemove;

        pNode batchNormAdd2 = lastNode;
        if (batchNormAdd2 == nullptr) continue;
        nodesToRemove.push_back(batchNormAdd2);

        pTensor output = batchNormAdd2->getOutput(0);

        pNode batchNormMul2 = m_graph.getTensorProducer(batchNormAdd2->getInput(0));
        if (batchNormMul2 == nullptr) continue;
        nodesToRemove.push_back(batchNormMul2);

        pNode batchNormSub = m_graph.getTensorProducer(batchNormAdd2->getInput(1));
        if (batchNormSub == nullptr) continue;
        nodesToRemove.push_back(batchNormSub);

        pTensor betaTensor = batchNormSub->getInput(0);
        if (betaTensor == nullptr) continue;

        pNode batchNormMul3 = m_graph.getTensorProducer(batchNormSub->getInput(1));
        if (batchNormMul3 == nullptr) continue;
        nodesToRemove.push_back(batchNormMul3);

        pNode batchNormMul1 = m_graph.getTensorProducer(batchNormMul3->getInput(1));
        if (batchNormMul1 == nullptr) continue;
        nodesToRemove.push_back(batchNormMul1);

        pTensor gammaTensor = batchNormMul1->getInput(1);
        if (gammaTensor == nullptr) continue;

        pNode reciprocal =  m_graph.getTensorProducer(batchNormMul1->getInput(0));
        if (reciprocal == nullptr) continue;
        nodesToRemove.push_back(reciprocal);

        pNode sqrt = m_graph.getTensorProducer(reciprocal->getInput(0));
        if (sqrt == nullptr) continue;
        nodesToRemove.push_back(sqrt);

        pNode batchnormAdd1 = m_graph.getTensorProducer(sqrt->getInput(0));
        if (batchnormAdd1 == nullptr) continue;
        nodesToRemove.push_back(batchnormAdd1);

        pTensor epsilonTensor = batchnormAdd1->getInput(1);
        if (epsilonTensor == nullptr) continue;
        if (!epsilonTensor->isStaticParam() || epsilonTensor->isEnforcedOutput()) continue;

        pNode reduceMean2 = m_graph.getTensorProducer(batchnormAdd1->getInput(0));
        if (reduceMean2 == nullptr) continue;
        std::shared_ptr<TPCNode> pTpcNode = std::dynamic_pointer_cast<TPCNode>(reduceMean2);
        ns_Reduction::Params *reduceParams = (ns_Reduction::Params *)pTpcNode->getParams();
        if (reduceParams == nullptr)
        {
            HB_ASSERT(false, "mean_reduction node {} has NULL params", reduceMean2->getNodeName());
            continue;
        }
        if (reduceParams->reductionDimension != 0) continue;

        nodesToRemove.push_back(reduceMean2);

        pNode squareDiffMul = m_graph.getTensorProducer(reduceMean2->getInput(0));
        if (squareDiffMul == nullptr) continue;
        nodesToRemove.push_back(squareDiffMul);

        pNode squareDiffSub = m_graph.getTensorProducer(squareDiffMul->getInput(0));
        if (squareDiffSub == nullptr) continue;
        nodesToRemove.push_back(squareDiffSub);

        pNode reduceMean1 = m_graph.getTensorProducer(batchNormMul3->getInput(0));
        if (reduceMean1 == nullptr) continue;
        pTpcNode = std::dynamic_pointer_cast<TPCNode>(reduceMean1);
        reduceParams = (ns_Reduction::Params *)pTpcNode->getParams();
        if (reduceParams == nullptr)
        {
            HB_ASSERT(false, "mean_reduction node {} has NULL params", reduceMean1->getNodeName());
            continue;
        }
        if (reduceParams->reductionDimension != 0) continue;
        nodesToRemove.push_back(reduceMean1);

        pTensor input = reduceMean1->getInput(0);
        if (input == nullptr) continue;

        // check that all nodes can be removed
        bool status = true;
        for (pNode nodeToRemove : nodesToRemove)
        {
            if (nodeToRemove == batchNormAdd2) continue; //this is the last node, its output tensor is kept

            unsigned maxOutConsumersLim = 1;
            if (nodeToRemove == batchNormMul1 || nodeToRemove == reduceMean1 || nodeToRemove == squareDiffSub)
            {
                maxOutConsumersLim = 2;
            }

            if (!GraphEditor::canEliminateTensor(m_graph, nodeToRemove->getOutput(0), maxOutConsumersLim))
            {
                LOG_TRACE(GC,
                        "node '{}' is not eligible for removal, can't fuse layer norm",
                        nodeToRemove->getNodeName());
                status = false;
                break;
            }
        }
        if (!status)
        {
            continue;
        }

        // epsilon is provided as param to layer_norm kernel - need to extract the value and make sure is [1] sized.
        float epsilon;
        if (!extractScalarFromStaticTensor(epsilonTensor, epsilon)) continue;

        // creating the fused layer-norm node
        ns_LayerNormKernel::Params params;
        params.epsValid = true;
        params.eps = epsilon;
        std::string kernelName = "layer_norm";
        if (input->getElementType() != syn_type_na)
        {
            kernelName = fmt::format("{}_{}", kernelName, getDtypeSuffixFromSynDataType(input->getElementType()));
        }
        std::string nodeName      = "fused_layer_norm_" + std::to_string(fusions++);
        pNode       layerNormNode = NodeFactory::createGenericTPCNode({input, betaTensor, gammaTensor},
                                                                {output},
                                                                &params,
                                                                kernelName,
                                                                nodeName);

        // removing all pattern nodes and adding the new node
        if (GraphEditor::replaceNodes(m_graph, nodesToRemove, {layerNormNode}) != REPLACE_NODE_SUCCESS)
        {
            LOG_ERR(GC, "{}: failed to fuse layer norm pattern", HLLOG_FUNC);
            return false;
        }
        LOG_INFO(GC, "layer norm pattern was found and replaced by a single node '{}'", layerNormNode->getNodeName());
    }


    return true;
}

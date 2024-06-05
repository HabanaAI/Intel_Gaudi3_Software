#include "pair_grads.h"

#include "node_factory.h"
#include "infer_shape_node.h"
#include "tpc_kernel_loader.h"

bool GradAReshapedGradBPairTransformer::optimizeGradPairs()
{
    if (!GCFG_ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING.value())
    {
        return true;
    }

    auto origTensors = m_graph.getTensors();
    for (const TensorPtr& tensor : origTensors)
    {
        if (!filterTensor(tensor)) continue;

        std::optional<PairingPattern> pairingPattern = patternMatch(tensor);
        if (pairingPattern)
        {
            NodeVector replacementPattern = createPairedGradA(*pairingPattern);

            LOG_DEBUG(GRAD_PAIR,
                      "replacing {}[{}] with <{}[{}], {}[{}]>",
                      pairingPattern->bgemm->getNodeName(),
                      pairingPattern->bgemm->getGUID(),
                      replacementPattern.front()->getNodeName(),
                      replacementPattern.front()->getGUID(),
                      replacementPattern.back()->getNodeName(),
                      replacementPattern.back()->getGUID());

            ReplaceNodeReturnStatus res =
                GraphEditor::replaceNodes(m_graph,
                                          {pairingPattern->bgemm},
                                          replacementPattern);

            auto tpcNode = std::dynamic_pointer_cast<TPCNode>(replacementPattern.back());
            if(tpcNode)
            {
                TpcKernelLoader kernelLoader(&m_graph);
                auto            isInstantiate = kernelLoader.load(replacementPattern.back());
                HB_ASSERT(isInstantiate,
                            "Could not instantiate node {} <GUID: {}>",
                            tpcNode->getNodeName(),
                            tpcNode->getGUID());
            }

            HB_ASSERT(res == REPLACE_NODE_SUCCESS,
                      "failed to replace {} with {}",
                      pairingPattern->bgemm->getNodeName(),
                      replacementPattern.front()->getNodeName());
        }
    }
    return true;
}

bool GradAReshapedGradBPairTransformer::filterTensor(const TensorPtr& tensor) const
{
    return (tensor && !tensor->isShapeTensor() && !tensor->isZeroSizedDataTensor());
}

// Find the handled pattern:
// [t]->bgemm_gradA->[dA]
//  +-->reshape->[flat_t]->gemm_gradB->[dB]
// t is referred to as the seed of the pattern
std::optional<GradAReshapedGradBPairTransformer::PairingPattern>
GradAReshapedGradBPairTransformer::patternMatch(const TensorPtr& tensor) const
{
    if (m_graph.getNumberOfTensorConsumers(tensor) < 2) return {};

    PairingPattern pattern {};
    for (const auto& consumer : m_graph.getTensorConsumers(tensor))
    {
        switch (consumer->getNodeType())
        {
            case Node::TYPE_BATCH_GEMM:
                if (matchBGemmSubPattern(tensor, consumer))
                {
                    pattern.bgemm = consumer;
                }
                break;
            case Node::TYPE_INTERNAL_RESHAPE:
            {
                if (matchReshapeSubPattern(consumer))
                {
                    pattern.reshape = consumer;
                }
                break;
            }
            default:
                break;
        }
    }
    if (pattern.bgemm && pattern.reshape)
    {
        LOG_DEBUG(GRAD_PAIR,
                  "Pattern match for tensor {}: bgemm {} and reshape {}",
                  tensor->getName(),
                  pattern.bgemm->getNodeName(),
                  pattern.reshape->getNodeName());
        return pattern;
    }
    return std::nullopt;
}

bool GradAReshapedGradBPairTransformer::matchReshapeSubPattern(const NodePtr& reshape) const
{
    // reshape matches the pattern if its output is 2D (X,Y,1,1,1..) and is consumed by a gemm node.

    auto rOut = reshape->getOutput(0);
    if (rOut->getNon1SizeDimsCount() != 2 || rOut->getSizeInElements(0) <= 1 || rOut->getSizeInElements(1) <= 1)
    {
        return false;
    }

    for (const NodePtr& reshapeConsumer : m_graph.getNodeConsumers(reshape))
    {
        if (reshapeConsumer->getNodeType() == Node::TYPE_GEMM)
        {
            return true;
        }
    }
    return false;
}

bool GradAReshapedGradBPairTransformer::matchBGemmSubPattern(const TensorPtr& seed, const NodePtr& bgemm) const
{
    // bgemm match the pattern if the seed is it's A operand and it is implementing a gemm operation.

    if (bgemm->getInputIndexOfTensor(seed) != 0) return false;
    if (bgemm->getInput(1)->getNon1SizeDimsCount() != 2) return false;

    HB_ASSERT(bgemm->getNodeType() == Node::TYPE_BATCH_GEMM,
              "Unexpected node type: {} for bgemm sub-pattern",
              bgemm->getNodeTypeStr());
    auto bgemmNode = std::static_pointer_cast<BatchGemmNode>(bgemm);
    return bgemmNode->canBeConvertedToGEMM();
}

// Creates the paired gemm_gradA and reshape pattern to replace the original bgemm, in order to get:
// [t]->reshape->[flat_t]->gemm_gradA->[flat_da]->reshape->[dA]
//                   +---->gemm_gradB->[dB]
NodeVector GradAReshapedGradBPairTransformer::createPairedGradA(const PairingPattern& matchPattern) const
{
    NodeVector res {};
    res.push_back(createGemmFromBGemm(matchPattern.bgemm, matchPattern.reshape->getOutput(0)));
    // use inferShape on matchPattern.bgemm to produce shape tensor for unflatten
    TensorPtr shape = nullptr;
    if (matchPattern.bgemm->isDynamicShape())
    {
        shape = matchPattern.bgemm->getOutput(0)->cloneGeometry();
        shape->setTensorType(SHAPE_TENSOR);
        shape->setName(matchPattern.bgemm->getNodeName() + "_original_shape");
        res.push_back(createInferShape(matchPattern.bgemm, shape));
    }
    res.push_back(createUnflatten(res.front()->getOutput(0), shape, matchPattern.bgemm->getOutput(0)));

    return res;
}

NodePtr GradAReshapedGradBPairTransformer::createGemmFromBGemm(const NodePtr& bgemm, const TensorPtr& flatInputA) const
{
    TensorPtr     flatOutput = createFlatOperand(bgemm->getOutput(0));
    synGEMMParams params     = static_cast<BatchGemmNode*>(bgemm.get())->getGEMMParams();
    NodePtr       gemm       = NodeFactory::createNode({flatInputA, bgemm->getInput(1)},
                                           {flatOutput},
                                           &params,
                                           NodeFactory::gemmNodeTypeName,
                                           bgemm->getNodeName() + "_as_gemm");
    return gemm;
}

TensorPtr GradAReshapedGradBPairTransformer::createFlatOperand(const TensorPtr& orig) const
{
    TensorPtr   flat      = orig->clone(false, false, false);
    const auto& origShape = flat->getAllNSizesInElements();
    TSize       flatShape[2];
    flatShape[0] = origShape[0];
    flatShape[1] = multiplyElements(origShape.begin() + 1, origShape.end());
    flat->reshape(2, flatShape, nullptr);
    return flat;
}

NodePtr GradAReshapedGradBPairTransformer::createInferShape(const NodePtr& bgemm, const TensorPtr& shape) const
{
    InferShapeParams params;
    params.isTpc           = false;
    params.sifId           = SIF_BATCH_GEMM;
    params.sifMetadata     = bgemm->getShapeInferenceFunctionUserParams();
    params.sifMetadataSize = bgemm->getShapeInferenceFunctionUserParamsSize();

    NodePtr inferShape = NodeFactory::createNode({bgemm->getInput(0), bgemm->getInput(1)},
                                                 {shape},
                                                 &params,
                                                 NodeFactory::inferShapeNodeTypeName,
                                                 bgemm->getNodeName() + "_infer_shape");

    return inferShape;
}

NodePtr GradAReshapedGradBPairTransformer::createUnflatten(const TensorPtr& flat,
                                                           const TensorPtr& shape,
                                                           const TensorPtr& orig) const
{
    NodePtr unFlatten = NodeFactory::createNode(shape ? TensorVector{flat, shape} : TensorVector{flat},
                                                {orig},
                                                nullptr,
                                                NodeFactory::reshapeNodeTypeName,
                                                "unflatten_" + orig->getName());
    return unFlatten;
}

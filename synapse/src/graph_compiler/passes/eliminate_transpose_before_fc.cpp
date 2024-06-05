#include "graph_editor.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "transpose_node.h"
#include "types.h"

static bool isTransposeToEliminate(pNode node)
{
    std::shared_ptr<TransposeNode> pTranspose = std::dynamic_pointer_cast<TransposeNode>(node);
    if (pTranspose == nullptr)
    {
        return false;
    }
    unsigned transposeIFMDim = node->getInput(TENSOR_IFM)->getDim();
    const TransposePermutationArray& permutation = pTranspose->permutation();
    // Only looking for 4D transposes that keep the batch position at the slowest-changing dimesion
    return (transposeIFMDim == 4 && permutation.size() == 4 && permutation[TPD_4Dim_Batch] == TPD_4Dim_Batch);
}

static bool isReshapeOrFlattenBeforeFC(HabanaGraph &g, pNode node)
{
    if (node == nullptr)
    {
        return false;
    }
    Node::eNodeType type = node->getNodeType();
    if (type != Node::TYPE_INTERNAL_FLATTEN &&
        type != Node::TYPE_INTERNAL_RESHAPE)
    {
        return false;
    }
    TensorVector outputs = node->getOutputs();
    HB_ASSERT(outputs.size() == 1, "reshape/flatten must have 1 output");
    if (g.getNumberOfTensorConsumers(outputs.front()) != 1)
    {
        //The node has more than 1 consumer so can't optimize
        return false;
    }
    return true;
}

struct TransposeBeforeFC
{
    pNode fc;
    pNode nodeAfterTranspose;
    pNode transpose;
};

static std::vector<TransposeBeforeFC> findTransposeFCSubgraphs(HabanaGraph &g)
{
    std::vector<TransposeBeforeFC> matchingSubgraphs;

    const NodeVector& nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        TransposeBeforeFC matchedPattern;
        if (node->getNodeType() != Node::TYPE_FC && node->getNodeType() != Node::TYPE_GEMM)
        {
            continue;
        }

        if (node->getNodeType() == Node::TYPE_GEMM && !node->getInput(TENSOR_WEIGHT)->isStaticParam())
        {
            continue;
        }

        matchedPattern.fc = node;

        pTensor fcIFM = node->getInput(TENSOR_IFM);
        if (fcIFM->getDim() != 2)
        {
            continue;
        }
        unsigned fcBatch = fcIFM->getSizeInElements(1);
        pNode prevNode = g.getTensorProducer(fcIFM);
        unsigned prevNodeBatch = 0;
        while (isReshapeOrFlattenBeforeFC(g, prevNode))
        {
            matchedPattern.nodeAfterTranspose = prevNode;
            TensorVector inputs               = prevNode->getInputs();
            HB_ASSERT(inputs.size() == 1, "reshape/flatten must have 1 input");
            pTensor input = inputs.front();
            prevNodeBatch = input->getSizeInElements(input->getDim() - 1);
            prevNode = g.getTensorProducer(input);
        }
        if (prevNode == nullptr)
        {
            continue;
        }
        if (isTransposeToEliminate(prevNode) && prevNodeBatch == fcBatch)
        {
            matchedPattern.transpose = prevNode;
            matchingSubgraphs.push_back(matchedPattern);
        }
    }

    return matchingSubgraphs;
}


static std::vector<unsigned> calcSpatialInverseTranspose(const TransposePermutationArray& permutation)
{
    std::vector<unsigned> inverse(3);
    for (unsigned i = 0; i < 3; ++i)
    {
        inverse[permutation[i]] = static_cast<TransposePermutationDim >(i);
    }
    return inverse;
}


static void reArrangeGraph(HabanaGraph& g, pNode oldTranspose, pNode newWeights, pNode nodeAfterTranspose, pNode fc)
{
    // remove the transpose from the graph only if it has 1 consumer
    TensorVector outputs = oldTranspose->getOutputs();
    if (g.getNumberOfTensorConsumers(outputs.front()) == 1)
    {
        GraphEditor::removeNode(g, oldTranspose);
    }
    else
    {
        // The node has more than 1 consumers - must keep it. May be removed later on once it reaches last consumer
        LOG_DEBUG(DATA_LAYOUT,
                  "{}: NodeName: {} has more than 1 consumers - keeping the transposed node",
                  HLLOG_FUNC,
                  oldTranspose->getNodeName());
    }

    // attach the transposed weights to the fully connected
    g.attachNodes(newWeights, fc, TENSOR_OFM, TENSOR_WEIGHT);

    // Set the transpose IFM as input for the flatten/reshape node that was after it
    pTensor transposeIFM = oldTranspose->getInput(TENSOR_IFM);
    pTensor transposeOFM = oldTranspose->getOutput(TENSOR_OFM);

    GraphEditor::replaceTensor(g, nodeAfterTranspose, transposeOFM, transposeIFM);
}

static NodePtr createReshape2to4(const TensorPtr&                 fcWeights,
                                 const TensorPtr&                 fcIFM,
                                 std::string_view                 fcNodeName,
                                 const TransposePermutationArray& permutation)
{
    TSize IFMShape[Tensor::c_tensorMaxDim];
    fcIFM->getAllSizesInElements(IFMShape, Tensor::c_tensorMaxDim);
    TSize wShape[Tensor::c_tensorMaxDim];
    fcWeights->getAllSizesInElements(wShape, Tensor::c_tensorMaxDim);

    HB_ASSERT(wShape[1] == IFMShape[0] * IFMShape[1] * IFMShape[2],
             "Weights depth dimensions don't match data spatial size");

    // assume {c, w, h, b}  is the shape of the IFM transposed by the given permutation,
    // and weights shape is {s, k} where s == c * w * h,
    // new shape will be {c, w, h, k}
    TSize wNewShape[Tensor::c_tensorMaxDim];
    wNewShape[0] = wShape[0];
    wNewShape[1] = IFMShape[permutation[0]];
    wNewShape[2] = IFMShape[permutation[1]];
    wNewShape[3] = IFMShape[permutation[2]];

    TensorPtr   reshapedWeights = std::make_shared<Tensor>(4, wNewShape, fcWeights->getElementType());
    reshapedWeights->setName(fmt::format("{}_reshaped_to_4d", fcWeights->getName()));
    reshapedWeights->setAllQuantizationParams(fcWeights->getAllQuantizationParams());
    reshapedWeights->setDynamicRange(fcWeights->getDynamicRange());
    reshapedWeights->setPerChannelQuant(fcWeights->isPerChannelQuant(), true);
    if (fcWeights->isDataTypeMatchData())
    {
        reshapedWeights->setAsDataTypeMatchData();
    }

    NodePtr reshapeWeights = NodeFactory::createNode({fcWeights},
                                                     {reshapedWeights},
                                                     nullptr,
                                                     "reshape",
                                                     fmt::format("{}_reshape_weights_from_2d_to_4d", fcNodeName));

    return reshapeWeights;
}

static NodePtr createWeightsTranspose(const TensorPtr&                 weights4d,
                                      const TransposePermutationArray& permutation,
                                      std::string_view                 fcNodeName,
                                      std::string_view                 wName)
{
    synTransposeParams transposeParams;
    transposeParams.tensorDim = 4;
    std::vector<unsigned> inv = calcSpatialInverseTranspose(permutation);
    transposeParams.permutation[0] = TransposePermutationDim::TPD_Weights_K;
    for (unsigned i = 0; i < 3; ++i)
    {
        transposeParams.permutation[i + 1] = static_cast<TransposePermutationDim >(inv[i] + 1);
    }
    TSize wShape[Tensor::c_tensorMaxDim];
    weights4d->getAllSizesInElements(wShape, Tensor::c_tensorMaxDim);
    TSize transposeShape[Tensor::c_tensorMaxDim];
    for (unsigned dim = 0; dim < 4; dim++)
    {
        transposeShape[dim] = wShape[transposeParams.permutation[dim]];
    }
    TensorPtr transposedWeights = std::make_shared<Tensor>(4, transposeShape, weights4d->getElementType());
    transposedWeights->setName(fmt::format("{}_transposed", wName));
    transposedWeights->setAllQuantizationParams(weights4d->getAllQuantizationParams());
    transposedWeights->setDynamicRange(weights4d->getDynamicRange());
    transposedWeights->setPerChannelQuant(weights4d->isPerChannelQuant(), true);
    if (weights4d->isDataTypeMatchData())
    {
        transposedWeights->setAsDataTypeMatchData();
    }
    NodePtr transposeWeights = NodeFactory::createNode(TensorVector({weights4d}),
                                                       TensorVector({transposedWeights}),
                                                       &transposeParams,
                                                       "transpose",
                                                       fmt::format("{}_transpose_Weights", fcNodeName));
    return transposeWeights;
}

static NodePtr createReshape4to2(const TensorPtr& weights4d, std::string_view fcNodeName, std::string_view wName)
{
    TSize wShape[Tensor::c_tensorMaxDim];
    weights4d->getAllSizesInElements(wShape, Tensor::c_tensorMaxDim);
    TSize spatialElements = wShape[1] * wShape[2] * wShape[3];
    TSize wNewShape[Tensor::c_tensorMaxDim];
    wNewShape[1] = spatialElements;
    wNewShape[0] = wShape[0];

    TensorPtr reshapedWeights = std::make_shared<Tensor>(2, wNewShape, weights4d->getElementType());
    reshapedWeights->setName(fmt::format("{}_reshaped_4d_to_2d", wName));
    reshapedWeights->setAllQuantizationParams(weights4d->getAllQuantizationParams());
    reshapedWeights->setDynamicRange(weights4d->getDynamicRange());
    reshapedWeights->setPerChannelQuant(weights4d->isPerChannelQuant(), true);
    if (weights4d->isDataTypeMatchData())
    {
        reshapedWeights->setAsDataTypeMatchData();
    }
    NodePtr reshapeWeights = NodeFactory::createNode(TensorVector({weights4d}),
                                                     TensorVector({reshapedWeights}),
                                                     nullptr,
                                                     "reshape",
                                                     fmt::format("{}_reshape_weights_4d_to_2d", fcNodeName));
    return reshapeWeights;
}

static void
eliminateTranspose(HabanaGraph& g, const NodePtr& transpose, const NodePtr& nodeAfterTranspose, const NodePtr& fc)
{
    const std::string& fcName = fc->getNodeName();

    LOG_DEBUG(DATA_LAYOUT,
              "Eliminating Transpose node {} before node {}, which is leading into FC {}",
              transpose->getNodeName(),
              nodeAfterTranspose->getNodeName(),
              fcName);

    // reshape weights to 4 dim tensor
    pTensor IFM = transpose->getInput(TENSOR_IFM);
    pTensor fcWeights = fc->getInput(TENSOR_WEIGHT);
    HB_ASSERT(fcWeights->getDim() == 2, "fcWeights dimension is not 2");
    std::shared_ptr<TransposeNode> pTranspose = std::dynamic_pointer_cast<TransposeNode>(transpose);

    pNode reshapeW2to4 = createReshape2to4(fcWeights, IFM, fcName, pTranspose->permutation());
    GraphEditor::addNode(g, reshapeW2to4);

    // transpose weights
    pNode transposeWeights = createWeightsTranspose(reshapeW2to4->getOutput(TENSOR_OFM), pTranspose->permutation(), fcName, fcWeights->getName());
    GraphEditor::addNode(g, transposeWeights);

    // reshape weights back to 2 dim
    pNode reshapeFinalWeights = createReshape4to2(transposeWeights->getOutput(TENSOR_OFM), fcName, fcWeights->getName());
    GraphEditor::addNode(g, reshapeFinalWeights);

    // remove the transpose and attach the new weights and IFM
    reArrangeGraph(g, transpose, reshapeFinalWeights, nodeAfterTranspose, fc);
}


bool eliminateTransposeBeforeFC(HabanaGraph& g)
{
    std::vector<TransposeBeforeFC> matchingSubgraphs = findTransposeFCSubgraphs(g);
    for (auto pattern : matchingSubgraphs)
    {
        eliminateTranspose(g, pattern.transpose, pattern.nodeAfterTranspose, pattern.fc);
    }
    return true;
}

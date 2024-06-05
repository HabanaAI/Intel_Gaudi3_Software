#include "handle_tensor_permutations.h"
#include "habana_graph.h"
#include "transpose_node.h"
#include "node_factory.h"
#include "transpose_utils.h"
#include "types.h"
#include <tuple>

bool TensorPermutationHandler::isNodeModifiable(bool isInternalTranspose) const
{
    // TODO [SW-89250] remove limitation for user nodes
    // Disable allow permutation for user nodes
    const bool canPermuteUserTranspose = GCFG_ALLOW_PERMUTATION_ON_USER_TRANSPOSE.value();
    return isInternalTranspose || canPermuteUserTranspose;
}

bool TensorPermutationHandler::isSamePermutation(const gc::Permutation& tensorPermutation,
                                                 const gc::Permutation& transposePermutation,
                                                 bool                   inversePermutation) const
{
    return tensorPermutation ==
           (inversePermutation ? transposePermutation.getInversePermutation() : transposePermutation);
}

bool TensorPermutationHandler::isTensorCandidate(const TensorPtr& t) const
{
    return t->getPermutation() || t->getTensorAnnotation().memory.allowPermutation;
}

void TensorPermutationHandler::permuteTensor(const TensorPtr& t, gc::Permutation& permutation) const
{
    HB_ASSERT(!t->getPermutation().has_value(), "Tensor {} has a permutation although expected not to", t->getName());
    HB_ASSERT(t->isTrivialStrided() && t->getTensorAnnotation().memory.allowPermutation,
              "tensor {} doesn't have a permutation but isn't trivially strided",
              t->getName());
    t->setPermutation(permutation);
    t->reshape(t->getDim(),
               t->getNSizesInElements().data(),
               nullptr,
               t->getNMinimalSizesInElements().data());  // set new default strides
}

NodePtr TensorPermutationHandler::getAsLogicalTranspose(const TensorPtr&                 input,
                                                        const TensorPtr&                 output,
                                                        const TransposePermutationArray& permutation,
                                                        std::string_view                 name,
                                                        bool                             stridedOutput) const
{
    NodePtr logicalTranspose =
        NodeFactory::createInternalNode({input}, {output}, &permutation, NodeFactory::transposeLogicNodeTypeName, name);

    auto logicalNode = dynamic_cast<LogicalTransposeNode*>(logicalTranspose.get());
    HB_ASSERT_PTR(logicalNode);
    // force direction of logical node, so that the permutated tensor will be te Real Tensor.
    if (stridedOutput)
    {
        logicalNode->swapAliasDirection();
    }
    logicalNode->setAsUserPermutationTranspose();
    return logicalTranspose;
}

std::tuple<NodePtr, NodePtr, TensorPtr> TensorPermutationHandler::getTransposeSequence(const TensorPtr& permutedTensor,
                                                                                       const NodePtr&   adjacentNode,
                                                                                       bool isAdjacentProducer) const
{
    HB_ASSERT_PTR(permutedTensor);
    HB_ASSERT_PTR(adjacentNode);
    HB_ASSERT(permutedTensor->getPermutation().has_value(),
              "Expecting input tensor {} to be permuted",
              permutedTensor->getName());
    const gc::Permutation& permutedTensorPerm = permutedTensor->getPermutation().value();

    LOG_DEBUG(DATA_LAYOUT,
              "Inserting transpose sequence. Permuted tensor: {}, Adjacent node<{}>: {}",
              permutedTensor->getName(),
              isAdjacentProducer ? "Producer" : "Consumer",
              adjacentNode->getNodeName());
    const std::string uniqueBaseName =
        fmt::format("nodeId{}_{}_perm_seq_", adjacentNode->getId(), isAdjacentProducer ? "produced" : "consumed");

    const auto perm    = getTransposePermutationArray(permutedTensorPerm.getValues().begin(), permutedTensor->getDim());
    const auto invPerm = getTransposePermutationArray(permutedTensorPerm.getInversePermutation().getValues().begin(),
                                                      permutedTensor->getDim());

    const TensorPtr& seqIntermTensor = getTensorAfterTranspose(*permutedTensor, perm, uniqueBaseName + "interm");
    const TensorPtr& seqOut =
        isAdjacentProducer
            ? permutedTensor
            : getTensorAfterTranspose(*seqIntermTensor, isAdjacentProducer ? perm : invPerm, uniqueBaseName + "seqOut");
    const TensorPtr& seqIn =
        isAdjacentProducer
            ? getTensorAfterTranspose(*seqIntermTensor, isAdjacentProducer ? invPerm : perm, uniqueBaseName + "seqIn")
            : permutedTensor;

    // Logic transpose permutation is set as P^-1 for a produced permuted tensor t because t is the output
    // of a logical op as well as the real tensor in that logic op and thus the op
    // is performed backwards and effectively performs permutation P on the permuted tensor which by definition
    // results in a dense alias of the permuted tensor.
    // For a consumed permuted tensor the permutation of the logic transpose is simply the inverse of the
    // above which is P.
    const TransposePermutationArray& transposePermArray   = isAdjacentProducer ? perm : invPerm;
    const TransposePermutationArray& logicTransposeParams = isAdjacentProducer ? invPerm : perm;
    const synTransposeParamsNDims    transposeParams      = permutationToParams(transposePermArray);

    const TensorVector logicTransposeIn {isAdjacentProducer ? seqIntermTensor : seqIn};
    const TensorVector logicTransposeOut {isAdjacentProducer ? seqOut : seqIntermTensor};
    const TensorVector transposeIn {isAdjacentProducer ? seqIn : seqIntermTensor};
    const TensorVector transposeOut {isAdjacentProducer ? seqIntermTensor : seqOut};

    const unsigned logicTransposeIdx = isAdjacentProducer ? 1 : 0;
    const unsigned transposeIdx      = 1 ^ logicTransposeIdx;

    LOG_DEBUG(DATA_LAYOUT,
              "{}: create logical transpose: idx={}, permutation={}, <inputShape={}, outputShape={}>, "
              "<inputStrides={}, outputStrides={}>",
              HLLOG_FUNC,
              logicTransposeIdx,
              TransposeNode::getPermutationString(logicTransposeParams),
              logicTransposeIn[0]->getDimSizesStr(),
              logicTransposeOut[0]->getDimSizesStr(),
              logicTransposeIn[0]->getStridesStr(),
              logicTransposeOut[0]->getStridesStr());
    LOG_DEBUG(DATA_LAYOUT,
              "{}: create transpose: idx={}, permutation={}, <inputShape={}, outputShape={}>, "
              "<inputStrides={}, outputStrides={}>",
              HLLOG_FUNC,
              transposeIdx,
              TransposeNode::getPermutationString(transposePermArray),
              transposeIn[0]->getDimSizesStr(),
              transposeOut[0]->getDimSizesStr(),
              transposeIn[0]->getStridesStr(),
              transposeOut[0]->getStridesStr());

    const NodePtr transpose =
        NodeFactory::createInternalNode(transposeIn,
                                        transposeOut,
                                        &transposeParams,
                                        NodeFactory::transposeNodeTypeName,
                                        fmt::format("{}transpose/{}", uniqueBaseName, transposeIdx));
    NodePtr logicTranspose =
        NodeFactory::createInternalNode(logicTransposeIn,
                                        logicTransposeOut,
                                        &logicTransposeParams,
                                        NodeFactory::transposeLogicNodeTypeName,
                                        fmt::format("{}logic_transpose/{}", uniqueBaseName, logicTransposeIdx));

    // force direction of logical node, such that the permuted tensor is the real tensor in the logical operation.
    LogicalTransposeNode* pLogicTranspose = static_cast<LogicalTransposeNode*>(logicTranspose.get());
    if (isAdjacentProducer)
    {
        pLogicTranspose->swapAliasDirection();
    }
    pLogicTranspose->setAsUserPermutationTranspose();

    // Maintain tracking of origin nodes for debug purposes
    const auto& originNodes = adjacentNode->getOriginNodes();
    transpose->setOriginNodes(originNodes);
    logicTranspose->setOriginNodes(originNodes);

    return std::make_tuple(transpose, logicTranspose, isAdjacentProducer ? seqIn : seqOut);
}

bool GraphModeTensorPermutationHandler::canConvertAdjacentNodeToLogicTranspose(
    const NodePtr&                  adjacentNode,
    std::optional<gc::Permutation>& optionalPerm,
    bool                            inversePerm) const
{
    // Adjacent node is transpose
    auto* transpose = dynamic_cast<TransposeNode*>(adjacentNode.get());
    if (transpose == nullptr) return false;

    if (!isNodeModifiable(transpose->getNodeAnnotation().insertedNode)) return false;

    // Candidate tensor perm exists and is same as that of adjacent transpose
    gc::Permutation transposePerm = gc::Permutation(transpose->permutation());
    if (optionalPerm.has_value() && !isSamePermutation(optionalPerm.value(), transposePerm, inversePerm)) return false;

    // allowed_permutation tensor, setting current transpose permutation as potential tensor permutation
    if (!optionalPerm.has_value())
    {
        optionalPerm = inversePerm ? transposePerm.getInversePermutation() : transposePerm;
    }
    return true;
}

void GraphModeTensorPermutationHandler::setAsLogicalTranspose(const NodePtr& node, bool stridedOutput)
{
    auto* transpose = dynamic_cast<TransposeNode*>(node.get());
    HB_ASSERT_PTR(transpose);
    auto logicalTranspose = getAsLogicalTranspose(transpose->getInput(0),
                                                  transpose->getOutput(0),
                                                  transpose->permutation(),
                                                  transpose->getNodeName(),
                                                  stridedOutput);
    LOG_DEBUG(DATA_LAYOUT, "Set transpose node: {} as logical transpose", node->getNodeName());
    GraphEditor::replaceNodes(m_graph, {node}, {logicalTranspose});
}

void GraphModeTensorPermutationHandler::insertTransposeSequence(const TensorPtr& permutedTensor,
                                                                const NodePtr&   adjacentNode,
                                                                bool             isAdjacentProducer)
{
    auto [transpose, logicTranspose, newTensor] =
        getTransposeSequence(permutedTensor, adjacentNode, isAdjacentProducer);

    GraphEditor::replaceTensor(m_graph, adjacentNode, permutedTensor, newTensor);

    const bool addTransposeSuccess = GraphEditor::addNode(m_graph, transpose);
    HB_ASSERT(addTransposeSuccess, "Failed adding transpose node {} to graph", transpose->getNodeName());

    const bool addLogicTransposeSuccess = GraphEditor::addNode(m_graph, logicTranspose);
    HB_ASSERT(addLogicTransposeSuccess,
              "Failed adding logic transpose node {} to graph",
              logicTranspose->getNodeName());
}

void GraphModeTensorPermutationHandler::handleTensorPermutation(const TensorPtr& t)
{
    const NodePtr& producer  = m_graph.getTensorProducer(t);
    auto           consumers = m_graph.getTensorConsumers(t);

    const auto&                    candidateTensorPerm = t->getPermutation();
    std::optional<gc::Permutation> optionalPerm        = candidateTensorPerm;

    bool canConvertAdjacentTransposesToLogic = true;
    if (producer)
    {
        canConvertAdjacentTransposesToLogic &=
            canConvertAdjacentNodeToLogicTranspose(producer, optionalPerm /* INOUT */, true /* inverse */);
    }
    for (const NodePtr& consumer : consumers)
    {
        if (!canConvertAdjacentTransposesToLogic) break;
        canConvertAdjacentTransposesToLogic &=
            canConvertAdjacentNodeToLogicTranspose(consumer, optionalPerm /* INOUT */, false /* inverse */);
    }

    if (canConvertAdjacentTransposesToLogic && optionalPerm.has_value())
    {
        // Permuted tensor's adjacent nodes are transpose nodes that can be converted to logical transposes
        // after which the permuted tensor is read from/written to densly.
        if (producer)
        {
            setAsLogicalTranspose(producer, true /* strided Output or Input */);
        }

        for (const NodePtr& consumer : consumers)
        {
            setAsLogicalTranspose(consumer, false /* strided Output or Input */);
        }

        if (!candidateTensorPerm.has_value())
        {
            permuteTensor(t, optionalPerm.value());
        }
    }
    else if (candidateTensorPerm.has_value())
    {
        // Candidate tensor is permuted but connot be handled by converting the adjacent nodes to logical transposes.
        // In this case a transpose and logical transpose sequence is planted between the permuted tensor and its
        // consumer/producer.
        if (producer)
        {
            insertTransposeSequence(t, producer, true /* adjacent producer */);
        }

        for (const NodePtr& consumer : consumers)
        {
            insertTransposeSequence(t, consumer, false /* adjacent consumer */);
        }
    }
    // [CID: 45811] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
    // link:
    // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
}

void GraphModeTensorPermutationHandler::shortcutTransposeSequence(const TensorPtr& t)
{
    if (!t) return;
    if (!t->getPermutation().has_value()) return;
    if (m_graph.getNumberOfTensorConsumers(t) == 0 || m_graph.getNumberOfTensorProducers(t) == 0) return;

    NodePtr producer = m_graph.getTensorProducer(t);
    HB_ASSERT(producer && producer->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE,
              "{}: expected a logical transpose producer for {}",
              __func__,
              t->getName());

    for (NodePtr consumer : m_graph.getTensorConsumers(t))
    {
        HB_ASSERT(consumer && consumer->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE,
                  "{}: expected a logical transpose consumer for {}",
                  __func__,
                  t->getName());
        LOG_DEBUG(DATA_LAYOUT,
                  "{}: creating shortcut for tensor: {}, and removing: {}",
                  HLLOG_FUNC,
                  t->getName(),
                  consumer->getNodeName());

        TensorPtr consumerOutput = consumer->getOutput(0);
        TensorPtr producerInput  = producer->getInput(0);
        for (const NodePtr& consumerOfConsumer : m_graph.getTensorConsumers(consumerOutput))
        {
            GraphEditor::replaceTensor(m_graph, consumerOfConsumer, consumerOutput, producerInput);
        }
        GraphEditor::removeNode(m_graph, consumer);
    }
}

void GraphModeTensorPermutationHandler::handlePermutedTensors()
{
    TensorVector permutedTensors;
    for (const TensorPtr& t : m_graph.getTensors())
    {
        if (t && isTensorCandidate(t))
        {
            permutedTensors.push_back(t);
        }
    }
    // needs to be in a separate loop since g.getTensors uses a cached container, and we are going to modify the graph.
    for (const TensorPtr& t : permutedTensors)
    {
        // apply optimization of thus tensors
        handleTensorPermutation(t);
        shortcutTransposeSequence(t);
    }
}

bool EagerModeTransposeTensorPermutationHandler::canExtract() const
{
    if (!isNodeModifiable(true)) return false;
    auto transposePerm = gc::Permutation(m_nodeParams.permutation);

    if (isTensorCandidate(m_nodeParams.input) &&
        (!m_nodeParams.input->getPermutation().has_value() ||
         isSamePermutation(m_nodeParams.input->getPermutation().value(), transposePerm, false)))
    {
        m_permuteInputTensor = true;
        return true;
    }
    if (isTensorCandidate(m_nodeParams.output) &&
        (!m_nodeParams.output->getPermutation().has_value() ||
         isSamePermutation(m_nodeParams.output->getPermutation().value(), transposePerm, true)))
    {
        m_permuteInputTensor = false;
        return true;
    }
    return false;
}

NodePtr EagerModeTransposeTensorPermutationHandler::extract() const
{
    const auto& tensor           = m_permuteInputTensor ? m_nodeParams.input : m_nodeParams.output;
    auto        logicalTranspose = getAsLogicalTranspose(m_nodeParams.input,
                                                         m_nodeParams.output,
                                                         m_nodeParams.permutation,
                                                         *m_nodeParams.nodeName,
                                                         !m_permuteInputTensor);
    if (!tensor->getPermutation().has_value())
    {
        auto transposePermutation = gc::Permutation(m_nodeParams.permutation);
        transposePermutation =
            m_permuteInputTensor ? transposePermutation : transposePermutation.getInversePermutation();
        permuteTensor(tensor, transposePermutation);
    }
    return logicalTranspose;
}

bool EagerModeTensorPermutationHandler::canExtract() const
{
    auto fillPermutedTensors = [](const TensorVector& tensors, TensorVector& permutedTensors) {
        for (const auto& t : tensors)
        {
            if (t == nullptr) continue;
            const std::optional<gc::Permutation>& p = t->getPermutation();
            if (p.has_value() && !p.value().isIdentity())
            {
                permutedTensors.push_back(t);
            }
        }
    };

    fillPermutedTensors(m_node->getInputs(), m_premutedInputTensors);
    fillPermutedTensors(m_node->getOutputs(), m_premutedOutputTensors);

    if (!m_premutedInputTensors.empty())
    {
        // need to cope with tensors appearing more than once as an input
        auto newEnd = std::unique(m_premutedInputTensors.begin(), m_premutedInputTensors.end());
        m_premutedInputTensors.erase(newEnd, m_premutedInputTensors.end());
        return true;
    }
    return !m_premutedOutputTensors.empty();
}

NodeVector EagerModeTensorPermutationHandler::extract() const
{
    NodeVector result;

    auto handlePermutedTensor = [this, &result](const TensorVector& permutedTensors, bool isOutputTensor) {
        for (const auto& permutedTensor : permutedTensors)
        {
            auto [transpose, logicTranspose, newTensor] = getTransposeSequence(permutedTensor, m_node, isOutputTensor);
            m_node->replaceTensor(permutedTensor, newTensor);
            result.push_back(transpose);
            result.push_back(logicTranspose);
        }
    };

    handlePermutedTensor(m_premutedInputTensors, false);
    handlePermutedTensor(m_premutedOutputTensors, true);
    return result;
}

/*
    Definitions:
    ------------
    (1) Permuted tensor - A tensor set from the API as one which's strides need to be permuted
        for the tensor to be read/written densly from/to memory.

    (2) Tensor permutation - The permutation that need be performed on a tensors strides for the tensor
        to be read/written densly from/to memory.


    Notation for explanation (and examples) below:
    ----------------------------------------------
    (*) P    - permuted tensor's permutation.
    (*) P^-1 - permuted tensor's inverse permutation.


    Logic of this pass:
    -------------------
    (1) If all consumers of a candidate tensor t are transpose nodes with permutation P and the producer of t
        is a transpose node with permutation P^-1, replace all adjacent transpose nodes with logic transposes.
        If t is not permuted but is an allow_permutation tensor (set by API) then in addition to the above,
        set it's permutation to P.

    (2) If t is a permuted tensor (**not allow permutation**) that cannot be handled via (1) (for example due to
        adjacent nodes not not all being transpose nodes), a transpose and logic transpose sequence is planted
        such that the permuted tensor will always be the real tensor in the logical transpose operation.

        If permuted tensor has an adjacent producer (Producer)->[t], the below sequence will be inserted:
            (Producer)->[t_clone]->(Transpose <P>)->[t_interm]->(LogicTranspose <P^-1>)->[t]

        If permuted tensor has an adjacent consumer [t]->(Consumer), the below sequence will be inserted:
            [t]->(LogicTranspose<P>)->[t_interm]->(Transpose)->[t_clone]->(Consumer)

*/
bool handlePermutedTensors(HabanaGraph& g)
{
    GraphModeTensorPermutationHandler handler(g);
    handler.handlePermutedTensors();
    return true;
}
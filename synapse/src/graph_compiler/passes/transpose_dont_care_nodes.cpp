#include "transpose_dont_care_nodes.h"
#include "transpose_node.h"
#include "adjust_data_layout.h"
#include "node_layouts_handler.h"
#include "aggregate_fcd_node.h"
#include "node_factory.h"
#include "eager/lib/utils/general_defs.h"  // for EAGER_ASSERT
#include "types.h"

// general functions
static bool isDenseAfterPermute(const TensorPtr& input, const Permutation& perm)
{
    // get the sizes
    TSize sizes[Tensor::c_tensorMaxNDim];
    input->getAllNSizesInElements(sizes);
    perm.permuteShape(sizes, input->getDim());

    // get the strides
    TStride strides[Tensor::c_numOfNStrides];
    input->getNStridesInBytes(strides);
    perm.permuteShape(strides, input->getDim());

    // create the potential tensor for the assertion
    TensorPtr potentialTensor =
        std::make_shared<Tensor>(input->getDim(), sizes, input->getElementType(), nullptr, strides);
    return potentialTensor->isDenseLayout();
}

static void fixupAllowPermutation(HabanaGraph& g)
{
    // set 'allow permutation' only for tensors with a single reader/write (transpose node)
    // otherwise, we might get to a point that we are permuting the FCD for a tensor being used by other nodes.
    if (!GCFG_ENABLE_INTERMEDIATE_TENSOR_PERMUTATION.value())
    {
        for (const TensorPtr& t : g.getTensors())
        {
            if (t->getTensorAnnotation().memory.allowPermutation)
            {
                if (g.getNumberOfTensorConsumers(t) + g.getNumberOfTensorProducers(t) > 1)
                {
                    t->getTensorAnnotation().memory.allowPermutation = false;
                }
            }
        }
    }
}

// todo: [SW-103625] This code can be removed when aggregation nodes are refactored.
// currently aggregation nodes such as concat can be more than one class, according to their dim, and this is
// decided during the original node creation - but this dim can change, like in this pass. So until we refactor,
// we must recreate the node so it will be created with the correct class
static NodePtr fixupAggregationNode(const NodePtr& node, const PermutationVector& inputPermutations)
{
    auto aggregationNode  = node->isLogicalOperation() ? dynamic_cast<AggregationNode*>(node.get()) : nullptr;
    auto aggregateFcdNode = node->isMultiNode() ? dynamic_cast<AggregateFcdNode*>(node.get()) : nullptr;
    if (aggregationNode || aggregateFcdNode)
    {
        unsigned aggregationDim = 0;
        if (aggregationNode)
        {
            aggregationDim = aggregationNode->getAggregationDim();
        }
        else if (aggregateFcdNode)
        {
            aggregationDim = 0;
        }
        else
        {
            HB_ASSERT(false, "Expected aggregationNode or aggregateFcdNode");
        }

        // update the dim according to the permutations
        HB_ASSERT(inputPermutations.size() > 0, "Given empty permutations vector for a wrapped aggregation node");
        for (const auto& p : inputPermutations)
        {
            HB_ASSERT(p == inputPermutations[0], "Cannot convert params. All input permutations should be identical");
        }
        unsigned permutedAggregationDim = inputPermutations[0].permuteDim(aggregationDim);

        if (permutedAggregationDim != aggregationDim)
        {
            LOG_TRACE(DATA_LAYOUT, "The axis in aggregation node {} has been changed from {} to {} due to transpose permutations"
                                   " - recreating the node", node->getNodeName(), aggregationDim, permutedAggregationDim);
            auto newNode = NodeFactory::createNode(node->getInputs(),
                                                   node->getOutputs(),
                                                   &permutedAggregationDim,
                                                   node->getGUID(),
                                                   fmt::format("{}_fixed", node->getNodeName()));

            newNode->getNodeAnnotation().inputPermutations = node->getNodeAnnotation().inputPermutations;
            return newNode;
        }
    }
    return NodePtr();
}

// Wraps a node with transposes (if possible) according to the permutation given and isInput
static transposeWrapStatus wrapWithTransposes(HabanaGraph& g, NodePtr& node, Permutation& permutation, bool isInput)
{
    PermutationVector  inputsPermutations;
    PermutationVector  outputsPermutations;
    NodeLayoutsHandler nodeLayoutsHandler(g, node, permutation, isInput, inputsPermutations, outputsPermutations);
    // This function checks if the node may be wrapped and updates the permutation vectors accordingly.
    // It also might expand a tensor dim, by inserting reshape, to allow the permutation wrap (in broadcasted input case).
    node->accept(&nodeLayoutsHandler);

    if (const auto status = nodeLayoutsHandler.shouldWrap(); status != transposeWrapSuccess)
    {
        return status;
    }

    // the wrapping itself
    TransposeInserter transposeHandler(node, inputsPermutations, outputsPermutations);
    if (!transposeHandler.InsertTransposesForNodeIO(g))
    {
        LOG_DEBUG(DATA_LAYOUT, "failed to insert transpose for node {}", node->getNodeName());
        return transposeWrapFail;
    }

    LOG_TRACE(DATA_LAYOUT,
              "Wrapping node {} with permutation {} on its {} tensors",
              node->getNodeName(),
              permutation.toString(),
              (isInput ? "input" : "output"));

    // todo: remove the condition below once aggregation nodes are refactored [SW-103625]
    NodePtr newNode = fixupAggregationNode(node, inputsPermutations);
    if (newNode == nullptr)
    {
        node->permuteParams(inputsPermutations);
    }
    else
    {
        GraphEditor::replaceNodes(g, {node}, {newNode});
        // Update the replaced node for the graph traversal
        node = newNode;
    }

    return transposeWrapSuccess;
}

static bool isGCTranspose(const NodePtr& node)
{
    return node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE && node->getNodeAnnotation().insertedNode;
}

static bool bfsTraversalAndWrap(HabanaGraph& g, NodeList& queue)
{
    LOG_TRACE(DATA_LAYOUT, "BFS traversal for wrapping don't care nodes begins");
    std::unordered_set<NodePtr> visited(queue.begin(), queue.end());

    while (!queue.empty())
    {
        // Get next node to handle in BFS order
        NodePtr node = queue.front();
        LOG_TRACE(DATA_LAYOUT, "Visiting node {}", node->getNodeName());
        queue.pop_front();

        // Add all neighboring nodes to BFS next phase set
        NodeSet neighbors;
        NodeSet producers = g.getNodeProducers(node);
        NodeSet consumers = g.getNodeConsumers(node);
        neighbors.insert(producers.begin(), producers.end());
        neighbors.insert(consumers.begin(), consumers.end());

        // prepare transpose related variables
        const bool      nodeIsTranspose   = isGCTranspose(node);
        auto            transpose         = std::dynamic_pointer_cast<TransposeNode>(node);
        Permutation     permutation;
        NodePtr         transposeProducer;
        if (transpose != nullptr)
        {
            permutation       = ((Permutation)transpose->permutation()).getInversePermutation();
            transposeProducer = g.getTensorProducer(transpose->getInput(0));
        }

        // Go over the node's neighbors and:
        // 1) If the node is a (gc-inserted) transpose - try to wrap them (if they aren't transposes).
        // 2) If the node is a non-transpose - simply add its transpose neighbors to the queue.
        for (NodePtr neighbor : neighbors)
        {
            LOG_TRACE(DATA_LAYOUT, "Visiting neighbor node {}", neighbor->getNodeName());
            if (visited.count(neighbor))
            {
                LOG_TRACE(DATA_LAYOUT, "Neighbor is already visited");
                continue;
            }

            const bool neighborIsTranspose = isGCTranspose(neighbor);
            // gc-inserted transpose case
            if (nodeIsTranspose && !neighborIsTranspose)
            {
                LOG_TRACE(DATA_LAYOUT, "Attempting to propagate transpose node to its neighbor");
                bool     wrapBackward = (transposeProducer == neighbor);

                const auto status = wrapWithTransposes(g, neighbor, permutation, !wrapBackward);
                if (status == transposeWrapSuccess)
                {
                    queue.push_back(neighbor);
                }
                // The node was not wrapped but might be in a future attempt
                if (status != transposeWrapPostpone)
                {
                    visited.insert(neighbor);
                }
            }
            // non-transpose node case
            else if (!nodeIsTranspose && neighborIsTranspose)
            {
                LOG_TRACE(DATA_LAYOUT, "Neighbor is a transpose - adding it to the queue");
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
    }

    return true;
}

static bool areInputsDenseAfterPermutation(const NodePtr& node, TensorIndicesVector& indices)
{
    for (unsigned index : indices)
    {
        const TensorPtr&       input       = node->getInput(index);
        const Permutation&     permutation = input->getPermutation().value();
        if (!isDenseAfterPermute(input, permutation)) return false;
    }
    return true;
}

static bool getPermutationFromInputs(NodePtr node, Permutation& permutation, TensorIndicesVector& indices)
{
    auto isPermutedTensor = [](const TensorPtr& t) {
        if (t == nullptr) return false;
        const std::optional<Permutation>& permutedInputPerm = t->getPermutation();
        return permutedInputPerm && !permutedInputPerm->isIdentity();
    };

    // permuted outputs are handled as part of tensor permutation logic
    const auto& outputs = node->getOutputs();
    if (std::any_of(outputs.begin(), outputs.end(), isPermutedTensor)) return false;

    // go over all the inputs and look for a permuted tensor
    for (int i = 0; i < node->getNumInputs(); i++)
    {
        const TensorPtr& input = node->getInput(i);
        if (isPermutedTensor(input))
        {
            if (permutation.isEmpty())
            {
                permutation = input->getPermutation().value();
            }
            else if (permutation != input->getPermutation().value())
            {
                LOG_DEBUG(DATA_LAYOUT, "Node {} got different permutations from its inputs", node->getNodeName());
                return false;
            }
            indices.push_back(i);
        }
    }
    return true;
}

static bool wrapPermutedInputs(HabanaGraph& g, NodePtr& node)
{
    TensorIndicesVector   indices;
    Permutation           permutation;

    if (!getPermutationFromInputs(node, permutation, indices)) return false;

    HB_ASSERT(areInputsDenseAfterPermutation(node, indices), "Permutation should make the strides dense");

    if (!permutation.isEmpty())
    {
        return wrapWithTransposes(g, node, permutation, true) == transposeWrapSuccess;
    }
    return false;
}

Permutation GraphModeTransposeDontCareNodesHandler::extractPermutationsForNodeForward(const NodePtr& node) const
{
    Permutation perm;
    const auto& inputs = node->getInputs();
    for (const auto& input : inputs)
    {
        if (input == nullptr) continue;
        const std::optional<gc::Permutation>& permutedInputPerm = input->getPermutation();
        if (permutedInputPerm && !permutedInputPerm.value().isIdentity())
        {
            if (perm.size() == 0)
            {
                perm = permutedInputPerm.value();
                LOG_DEBUG(DATA_LAYOUT,
                          "Node {} is taking permutation {} from permuted tensor",
                          node->getNodeName(),
                          perm.toString());
                HB_ASSERT(isDenseAfterPermute(input, permutedInputPerm.value()),
                          "Permutation should make the strides dense");
            }
            else if (perm != permutedInputPerm)
            {
                // if more than one input tensor is permuted, they must have the same permutation
                LOG_DEBUG(DATA_LAYOUT, "Node {} got different permutations from its inputs", node->getNodeName());
                return Permutation();
            }
        }

        auto producer = m_graph.getTensorProducer(input);
        if (producer == nullptr) continue;

        if (TransposeNode* transposeProducer = dynamic_cast<TransposeNode*>(producer.get()))
        {
            if (!producer->getNodeAnnotation().insertedNode) continue;
            if (perm.size() == 0)
            {
                perm = Permutation(transposeProducer->permutation()).getInversePermutation();
                LOG_DEBUG(DATA_LAYOUT,
                          "Node {} is taking permutation {} from transpose producer",
                          node->getNodeName(),
                          perm.toString());
            }
            else if (perm != Permutation(transposeProducer->permutation()).getInversePermutation())
            {
                // if more than one transpose is a producer of the node's tensors, they must have the same
                // permutation
                LOG_DEBUG(DATA_LAYOUT, "Node {} got different permutations from its producers", node->getNodeName());
                return Permutation();
            }
        }
    }
    return perm;
}

Permutation GraphModeTransposeDontCareNodesHandler::extractPermutationsForNodeBackward(const NodePtr& node) const
{
    Permutation         perm;
    const TensorVector& outputs = node->getOutputs();
    for (const auto& output : outputs)
    {
        if (output == nullptr) continue;
        NodeList consumers = m_graph.getTensorConsumers(output);
        for (const auto& consumer : consumers)
        {
            if (consumer == nullptr) continue;
            if (TransposeNode* transposeConsumer = dynamic_cast<TransposeNode*>(consumer.get()))
            {
                if (!consumer->getNodeAnnotation().insertedNode) continue;
                if (perm.size() == 0)
                {
                    perm = Permutation(transposeConsumer->permutation()).getInversePermutation();
                    LOG_DEBUG(DATA_LAYOUT,
                              "Node {} is taking permutation {} from transpose consumer",
                              node->getNodeName(),
                              perm.toString());
                }
                else if (perm != Permutation(transposeConsumer->permutation()).getInversePermutation())
                {
                    // if more than one transpose is a consumer of the node's tensors, they must have the same
                    // permutation
                    LOG_DEBUG(DATA_LAYOUT, "Node {} got different permutations from its consumers", node->getNodeName());
                    return Permutation();
                }
            }
        }
    }
    return perm;
}

bool GraphModeTransposeDontCareNodesHandler::extractPermutationsForNode(const NodePtr& node, bool forward, Permutation& perm) const
{
    if (forward)
    {
        // perm is the permutation on the input
        perm = extractPermutationsForNodeForward(node);
    }
    else if (m_wrappedNodes.find(node) == m_wrappedNodes.end())  // when traversing backwards, don't re-wrap nodes
    {
        // perm is the permutation on the output
        perm = extractPermutationsForNodeBackward(node);
    }

    /* only wrap nodes that have a transpose around them, or that a permutation was given for their input tensor */
    return perm.size() != 0;
}

bool GraphModeTransposeDontCareNodesHandler::singleIteration(bool forward)
{
    NodeVector nodes = m_graph.getExeSortedNodes();
    if (!forward)
    {
        std::reverse(nodes.begin(), nodes.end());
    }

    for (auto& node : nodes)
    {
        Permutation perm;
        if (!extractPermutationsForNode(node, forward, perm)) continue;
        if (wrapWithTransposes(m_graph, node, perm, forward) == transposeWrapSuccess)
        {
            m_wrappedNodes.insert(node);
        }
    }

    return true;
}

// Only if TRANSPOSE_DONT_CARE_USE_BFS is true (default is false) then:
// Traverses the graph by doing a BFS traversal over the non-directed version of the graph and wrap nodes with
// transposes if possible. This is done in order for the removeContiguousTransposes pass to remove them later on
// and as a result, transposes will be propagated throughout the graph (optimally - to its edges).
bool transposeDontCareNodes(HabanaGraph& g)
{
    if (!GCFG_TRANSPOSE_DONT_CARE_USE_BFS.value())
    {
        LOG_DEBUG(DATA_LAYOUT, "Wrapping non-layout-restricted nodes with transposes");
        GraphModeTransposeDontCareNodesHandler handler(g);
        LOG_DEBUG(DATA_LAYOUT, "Begin to iterate forward (wrapping nodes that have transpose before them)");
        handler.singleIteration();
        LOG_DEBUG(DATA_LAYOUT, "Begin to iterate backward (wrapping nodes that have transpose after them)");
        handler.singleIteration(false);
    }
    else
    {
        LOG_DEBUG(DATA_LAYOUT, "Starting bfsTransposeDontCare pass");

        NodeList queue;

        // Obtain the initial set of nodes to begin the bfsTraversalAndWrap from
        for (NodePtr node : g.getTopoSortedNodes())
        {
            if (isGCTranspose(node)) continue;

            // a) Adjusted layout nodes - nodes that was handled in adjustDataLayout pass
            const auto& ioManager = node->getNodeIOManager();
            if (ioManager.isAdjusted())
            {
                queue.push_back(node);
            }

            // b) Nodes with a persistent input/output which is permuted by the user
            const bool nodeWasWrapped = wrapPermutedInputs(g, node);
            if (nodeWasWrapped)
            {
                queue.push_back(node);
            }
        }
        bfsTraversalAndWrap(g, queue);
    }

    fixupAllowPermutation(g);

    // remove redundant transpose nodes that were added during this pass
    g.turnOnPredicate(PREDICATE_ID_ELIMINATE_REDUNDANT_NODES);

    return true;
}

// eager functionality
bool EagerModeTransposeDontCareNodesHandler::canExtract()
{
    TensorIndicesVector   indices;
    Permutation           permutation;

    if (!getPermutationFromInputs(m_node, permutation, indices)) return false;

    EAGER_ASSERT(areInputsDenseAfterPermutation(m_node, indices), "Permutation should make the strides dense");

    if (!permutation.isEmpty())
    {
        auto& ioManager = m_node->getNodeIOManager();
        ioManager.setSupportedIOLayouts(m_graph.getDeviceType());

        NodeLayoutsHandler nodeLayoutsHandler(m_graph,
                                              m_node,
                                              permutation,
                                              true,
                                              m_inputTensorPermutations,
                                              m_outputTensorPermutations,
                                              true);

        // This function checks if the node may be wrapped and updates the permutation vectors accordingly.
        // It also might expand a tensor dim, by inserting reshape, to allow the permutation wrap (in broadcasted input case).
        m_node->accept(&nodeLayoutsHandler);

        if (const auto status = nodeLayoutsHandler.shouldWrap(); status != transposeWrapPostpone)
        {
            return status == transposeWrapSuccess;
        }
    }
    return false;
}

const TransposeNodeParamsVector& EagerModeTransposeDontCareNodesHandler::extract()
{
    m_transposeInserter.emplace(m_node, m_inputTensorPermutations, m_outputTensorPermutations);
    return m_transposeInserter->extract(m_graph);
}

NodePtr EagerModeTransposeDontCareNodesHandler::fixupNode()
{
    // todo: remove the condition below once aggregation nodes are refactored [SW-103625]
    NodePtr newNode = fixupAggregationNode(m_node, m_inputTensorPermutations);
    if (newNode == nullptr)
    {
        m_node->permuteParams(m_inputTensorPermutations);
    }
    return newNode;
}
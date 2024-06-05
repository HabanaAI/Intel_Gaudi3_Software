#include "handle_huge_tensors.h"
#include "huge_tensor_slicer.h"
#include "aggregation_node.h"
#include "habana_graph.h"
#include "node.h"
#include <stack>

static inline bool isConcat(const NodePtr& n)
{
    return n && n->getNodeType() == Node::TYPE_INTERNAL_CONCAT;
}

static inline unsigned getAggregationDim(const NodePtr& node)
{
    const auto& aggNode = std::dynamic_pointer_cast<AggregationNode>(node);
    return aggNode ? aggNode->getAggregationDim() : 0 /* FcdAggregationNode */;
}

bool HugeTensorHandler::shouldHandleHugeTensor(const NodePtr& n)
{
    return m_hugeTensorSlicer.doesRequireSlicing(n);
}

NodeVector HugeTensorHandler::extractNodeWithHugeTensors(const NodePtr& n)
{
    // [CID: 74303] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
    // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
    return m_hugeTensorSlicer.sliceNode(n);
}

std::optional<NSizeArray> HugeTensorHandler::findChunkSizeOfAggregatedTensor(const HabanaGraph& g,
                                                                             const TensorPtr&   tensor,
                                                                             bool               checkForSplit) const
{
    if (!tensor->isDataTensor()) return std::nullopt;
    const NodePtr& node = checkForSplit ? g.getTensorSingleConsumer(tensor) : g.getTensorProducer(tensor);
    if (checkForSplit)
    {
        if (!node || !node->isSplit()) return std::nullopt;
    }
    else
    {
        if (!node || !isConcat(node)) return std::nullopt;
    }

    unsigned         aggregationDim = getAggregationDim(node);
    const TensorPtr& t0             = checkForSplit ? node->getOutput(0) : node->getInput(0);
    unsigned         numOperands = checkForSplit ? node->getNumOutputsDataTensors() : node->getNumInputsDataTensors();
    HB_ASSERT_PTR(t0);
    // check that all operands of split/concat have the same size (except for the remainder)
    for (const TensorPtr& t : checkForSplit ? node->getOutputs() : node->getInputs())
    {
        if (!t || !t->isDataTensor()) continue;
        if (!t0->compareGeometry(*t)) return std::nullopt;
    }
    // the that the remainder is smaller than the other tensor chunks
    const TensorPtr& t = checkForSplit ? node->getOutput(numOperands - 1) : node->getInput(numOperands - 1);
    if (t->getSizeInElements(aggregationDim) > t0->getSizeInElements(aggregationDim)) return std::nullopt;

    return t0->getNSizesInElements();
}

OptionalTensorSplitSuggestion HugeTensorHandler::generateSlicingHint(const HabanaGraph& g, const NodePtr& n) const
{
    std::vector<TensorSplitSuggestion> potentialHints;
    for (const TensorPtr& in : n->getInputs())
    {
        auto chunkSize = findChunkSizeOfAggregatedTensor(g, in, false);
        if (chunkSize.has_value())
        {
            potentialHints.push_back({in, chunkSize.value()});
        }
    }
    for (const TensorPtr& out : n->getOutputs())
    {
        auto chunkSize = findChunkSizeOfAggregatedTensor(g, out, true);
        if (chunkSize.has_value())
        {
            potentialHints.push_back({out, chunkSize.value()});
        }
    }
    if (potentialHints.empty()) return std::nullopt;
    // for now return the first one without any heuristic. may be extended in the future
    OptionalTensorSplitSuggestion ret = potentialHints.front();
    LOG_DEBUG(HUGE_TENSOR_SLICE,
              "generated slicing suggestion: tensor: {}. chunkSize: {}",
              ret->tensor->getName(),
              toString(ret->chunkSize, ','));
    return ret;
}

void HugeTensorHandler::handleHugeTensors(HabanaGraph& g)
{
    const auto topoSortedNodes = g.getTopoSortedNodes();
    for (const NodePtr& node : topoSortedNodes)
    {
        if (shouldHandleHugeTensor(node))
        {
            LOG_DEBUG(HUGE_TENSOR_SLICE, "Handling node {} that has huge tensors", node->getNodeName());
            // [CID: 74304] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
            // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
            const auto extracted = m_hugeTensorSlicer.sliceNode(node, generateSlicingHint(g, node));
            HB_ASSERT(!extracted.empty(), "cannot slice huge tensors of node {}", node->getNodeName());

            ReplaceNodeReturnStatus status    = GraphEditor::replaceNodes(g, {node}, extracted);
            HB_ASSERT(status == REPLACE_NODE_SUCCESS,
                      "failed handling node {} with huge tensors!",
                      node->getNodeName());
        }
    }
}

bool handleHugeTensors(HabanaGraph& g)
{
    if (GCFG_ENABLE_HUGE_TENSOR_SLICING.value())
    {
        HugeTensorHandler(g).handleHugeTensors(g);
    }
    return true;
}
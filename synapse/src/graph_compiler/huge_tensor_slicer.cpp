#include "huge_tensor_slicer.h"
#include "huge_tensor_broadcast_slicer.h"
#include "huge_tensor_transpose_slicer.h"
#include "node_visitor.h"
#include "node_factory.h"
#include "node.h"
#include "habana_graph.h"
#include "huge_tensor_splitter.h"

template<class T>
class ReturnTypeNodeVisitor : public NodeVisitor
{
public:
    T get()
    {
        HB_ASSERT(m_result.has_value(), "trying to get result without setting it!");
        std::optional<T> tmp = std::nullopt;
        m_result.swap(tmp);
        // [CID: 74273] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
        // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
        return std::move(tmp.value());
    }

protected:
    std::optional<T> m_result = std::nullopt;
};

class HugeNodeSlicerVisitor : public ReturnTypeNodeVisitor<NodeVector>
{
public:
    HugeNodeSlicerVisitor(const OptionalTensorSplitSuggestion& splitSuggestion) : m_suggestion(splitSuggestion) {}

    void visit(Node* node) override { m_result = NodeVector(); }
    void visit(BatchGemmNode* node) override;
    void visit(GEMMNode* node) override;
    void visit(TransposeNode* node) override;
    void visit(BroadcastNode* node) override;

private:
    const OptionalTensorSplitSuggestion& m_suggestion;
};

class DoesRequireSlicingVisitor : public ReturnTypeNodeVisitor<bool>
{
public:
    DoesRequireSlicingVisitor() {}

    void visit(Node* node) override { m_result = false; }
    void visit(BatchGemmNode* node) override;
    void visit(GEMMNode* node) override;
    void visit(TransposeNode* node) override;
    void visit(BroadcastNode* node) override;
};

bool HugeTensorSlicer::doesRequireSlicing(const NodePtr& node)
{
    DoesRequireSlicingVisitor result;
    node->accept(&result);
    return result.get();
}

NodeVector HugeTensorSlicer::sliceNode(const NodePtr& node, const OptionalTensorSplitSuggestion& splitSuggestion)
{
    LOG_DEBUG(HUGE_TENSOR_SLICE, "slicing huge tensors for node {}", node->getNodeName());
    HugeNodeSlicerVisitor result(splitSuggestion);
    node->accept(&result);
    return result.get();
}

void DoesRequireSlicingVisitor::visit(BatchGemmNode* node)
{
    NodePtr                 nodeToCheck = node->shared_from_this();
    BgemmHugeTensorSplitter splitter(nodeToCheck);
    if (splitter.doesMmeNodeRequireSlicing())
    {
        m_result            = true;
        if (!splitter.doesNodeSupportSlicing())
        {
            LOG_ERR(HUGE_TENSOR_SLICE,
                    "node {} contain huge tensors but currently does not support slicing it",
                    node->getNodeName());
            m_result = false;
        }
        return;
    }
    m_result = false;
}

void HugeNodeSlicerVisitor::visit(BatchGemmNode* node)
{
    NodePtr                 nodeToSplit = node->shared_from_this();
    BgemmHugeTensorSplitter splitter(nodeToSplit);
    NodeList                ret = splitter.splitNodeWithHugeTensor();
    m_result                    = NodeVector(ret.begin(), ret.end());
}

void DoesRequireSlicingVisitor::visit(GEMMNode* node)
{
    NodePtr                nodeToCheck = node->shared_from_this();
    GemmHugeTensorSplitter splitter(nodeToCheck);
    if (splitter.doesMmeNodeRequireSlicing())
    {
        m_result = true;
        if (!splitter.doesNodeSupportSlicing())
        {
            LOG_ERR(HUGE_TENSOR_SLICE,
                    "node {} contain huge tensors but currently does not support slicing it",
                    node->getNodeName());
            m_result = false;
        }
        return;
    }
    m_result = false;
}

void HugeNodeSlicerVisitor::visit(GEMMNode* node)
{
    NodePtr                nodeToSplit = node->shared_from_this();
    GemmHugeTensorSplitter splitter(nodeToSplit);
    NodeList               ret = splitter.splitNodeWithHugeTensor();
    m_result                   = NodeVector(ret.begin(), ret.end());
}

void DoesRequireSlicingVisitor::visit(TransposeNode* node)
{
    m_result = HugeTensorTransposeSlicer::doesRequireSlicing(node);
}

void HugeNodeSlicerVisitor::visit(TransposeNode* node)
{
    HugeTensorTransposeSlicer slicer(node, m_suggestion);
    m_result = slicer.slice();
    // [CID: 75434] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
    // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
}

void DoesRequireSlicingVisitor::visit(BroadcastNode* node)
{
    m_result = HugeTensorBroadcastSlicer::doesRequireSlicing(node);
}

void HugeNodeSlicerVisitor::visit(BroadcastNode* node)
{
    HugeTensorBroadcastSlicer slicer(node, m_suggestion);
    m_result = slicer.slice();
    // [CID: 77479] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
    // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
}

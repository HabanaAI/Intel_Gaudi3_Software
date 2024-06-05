#include "habana_graph.h"
#include "habana_pass.h"
#include "logical_op_node.h"
#include "remove_sequences.h"
#include "sequence.h"
#include "strided_op_node_utils.h"
#include "strided_view_node.h"
#include "transpose_utils.h"
#include "node_factory.h"
#include "contiguous_reshape_remover.h"

class ReshapeSeqStrategy : public SequenceStrategy
{
public:
    static bool isReshape(const NodePtr& node)
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE) return true;
        if (node->getNodeType() == Node::TYPE_SQUEEZE_NODE) return true;
        if (node->getNodeType() == Node::TYPE_INTERNAL_EXPAND_DIMS) return true;
        if (node->getNodeType() == Node::TYPE_INTERNAL_FLATTEN) return true;
        if (node->getNodeType() == Node::TYPE_STATIC_RESHAPE) return true;
        if (node->getNodeType() == Node::TYPE_STRIDED_VIEW)
        {
            const auto* svNode = dynamic_cast<const StridedViewNode*>(node.get());
            if (!svNode) return false;  // can be the case after multiNode extraction.
            // sv node with the same number of elements and dense strides is actually a reshape
            if (svNode->getInput(0)->getDenseSizeInElements() == svNode->getOutput(0)->getDenseSizeInElements() &&
                StridedOpUtils::isDenseStridedOpParams(svNode->getParams(), svNode->getOutput(0)))
            {
                svNode->getInput(0)->getDenseSizeInElements();
            }
        }
        if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE)
        {
            auto transposeNode = dynamic_cast<TransposeNode*>(node.get());
            if (transposeNode && isSameDataMemOrder(*node->getInput(0), transposeNode->permutation())) return true;
        }
        return false;
    }

    static bool isValidReshapeForSequence(const NodePtr& n, bool sequenceStart)
    {
        if (n->isDynamicShape()) return false;  // Don't fuse dynamic reshapes
        bool isReshapeNode = isReshape(n) || (!sequenceStart && n->getNodeType() == Node::TYPE_IDENTITY);
        if (!isReshapeNode) return false;

        const TensorPtr& in  = n->getInput(0);
        const TensorPtr& out = n->getOutput(0);
        HB_ASSERT_PTR(in);
        HB_ASSERT_PTR(out);
        if (in->getElementType() != out->getElementType()) return false;

        if (n->isLogicalOperation())
        {
            auto* logicalNode = dynamic_cast<LogicalOpNode*>(n.get());
            HB_ASSERT_PTR(logicalNode);
            if (logicalNode->getRunLogicalOperationDone()) return false;
        }
        return !n->getNodeAnnotation().bundleInfo.is_set();
    }

    bool isSeqStart(const NodePtr& n) const override { return isValidReshapeForSequence(n, true); }

    bool canContinueSeq(const NodePtr& n) const override { return isValidReshapeForSequence(n, false); }

    bool isIdentity(const NodeVector& seq) const override
    {
        if (seq.empty()) return true;
        const TensorPtr& in  = seq.front()->getInput(0);
        const TensorPtr& out = seq.back()->getOutput(0);
        HB_ASSERT_PTR(in);
        HB_ASSERT_PTR(out);
        if (in->getDim() != out->getDim()) return false;
        return in->compareGeometry(*out);
    }

    bool isFusibleSequence(const NodeVector& seq) const override { return true; }

    std::optional<NodePtr> fuseSequence(const NodeVector& seq) const override
    {
        NodePtr reshape = NodeFactory::createNode({seq.front()->getInput(0)},
                                                  {seq.back()->getOutput(0)},
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  seq.back()->getNodeName());
        HB_ASSERT_PTR(reshape);
        return reshape;
    }
};

bool removeContiguousReshapeNodes(HabanaGraph& g)
{
    const ReshapeSeqStrategy strategy;
    SequenceRemover          sequenceRemover(g, strategy);
    bool                     status = sequenceRemover.removeSequences();
    if (!status) return status;

    ContiguousReshapeRemover remover(g);
    remover.removeContiguousReshapesForGraph();
    return true;
}

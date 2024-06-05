#include "habana_graph.h"
#include "habana_pass.h"
#include "layout.h"
#include "remove_sequences.h"
#include "sequence.h"
#include "transpose_node.h"
#include "transpose_utils.h"
#include "types.h"
#include "node_factory.h"

class TransposeSeqStrategy : public SequenceStrategy
{
public:
    bool isSeqStart(const NodePtr& n) const override
    {
        if (!n->isTranspose()) return false;
        return std::dynamic_pointer_cast<TransposeNode>(n) != nullptr && !n->getNodeAnnotation().bundleInfo.is_set();
    }
    bool canContinueSeq(const NodePtr& n) const override { return isSeqStart(n); }
    bool isIdentity(const NodeVector& seq) const override
    {
        if (seq.empty()) return true;
        const auto& first = std::dynamic_pointer_cast<TransposeNode>(seq.front());
        HB_ASSERT_PTR(first);
        gc::Permutation accumulatedPermutation = gc::Permutation(first->permutation().size());
        for (const NodePtr& node : seq)
        {
            const auto& transposeNode = std::dynamic_pointer_cast<TransposeNode>(node);
            HB_ASSERT_PTR(node);
            accumulatedPermutation.permute(gc::Permutation(transposeNode->permutation()));
        }
        return accumulatedPermutation.isIdentity();
    }

    bool isFusibleSequence(const NodeVector& seq) const override
    {
        return GCFG_ENABLE_FUSING_CONTIGUOUS_TRANSPOSE_NODES.value();
    }

    std::optional<NodePtr> fuseSequence(const NodeVector& seq) const override
    {
        HB_ASSERT(!seq.empty(), "{}: cannot fuse empty sequence!", __FUNCTION__);
        const auto& first = std::dynamic_pointer_cast<TransposeNode>(seq.front());
        HB_ASSERT_PTR(first);

        gc::Permutation perm = gc::Permutation(first->permutation().size());
        for (const NodePtr& node : seq)
        {
            const auto& transpose = std::dynamic_pointer_cast<TransposeNode>(node);
            HB_ASSERT_PTR(node);
            perm.permute(gc::Permutation(transpose->permutation()));
        }
        synTransposeParamsNDims params = permutationToParams(perm);

        const TensorPtr& input  = (seq.front())->getInput(0);
        const TensorPtr& output = (seq.back())->getOutput(0);

        NodePtr transposeNode =
            NodeFactory::createNode({input},
                                    {output},
                                    &params,
                                    NodeFactory::transposeNodeTypeName,
                                    "/fused_transpose/in_tensor_id_" + std::to_string(input->getId()) +
                                        "out_tensor_id_" + std::to_string(output->getId()));
        HB_ASSERT_PTR(transposeNode);
        return transposeNode;
    }
};

bool removeContiguousTransposes(HabanaGraph& g)
{
    if (!GCFG_ENABLE_CONTIGUOUS_TRANSPOSE_REMOVAL.value())
    {
        return true;
    }
    const TransposeSeqStrategy strategy;
    SequenceRemover            sequenceRemover(g, strategy);
    bool                       res = sequenceRemover.removeSequences();
    return res;
}

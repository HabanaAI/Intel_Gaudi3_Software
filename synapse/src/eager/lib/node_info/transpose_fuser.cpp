#include "transpose_fuser.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/passes/fuse_transpose_mme.h"
#include "graph_compiler/habana_nodes/node.h"
#include "include/tensor.h"

namespace eager_mode
{
// we need visibility of all the nodes in the graph but we do not need the full set
// of producer\consumer relations, and we do not attempt to optimize beyond the adjacent
// nodes in case of logical operations separating the transpose and gemm \ batch gemm.
// in addition, in case the gemm's transposed input has more than a single consumer
// or the gemm's output has additional consumers apart from the transpose we'll avoid
// the fusion and would not make further checks for fusion potential.
// This is not truly a limiting factor and should cover for most cases.
// only exception is identity nodes which are quite frequent, so for those it is beneficial
// to drop them, to avoid the need to look further ahead\backward.
// We do not try to optimize for cases where a Gemm\Batch Gemm is a consumer of another's
// input operand \ output operand. This is a case where we have more than a single consumer
// in case transpose fusion is possible but might still be able to drop the transpose, but it is not
// frequent enough to invest the extra computation to optimize for it as this is a less likely scenario.
void EagerTransposeFuser::fuseTransposes()
{
    buildCandidateListFromGemmOperands();
    // Early exit in case all Gemm operands are persistent
    if (candidateListEmpty()) return;
    fillCandidateListInfo();
    processCandidateList();
}

void EagerTransposeFuser::buildCandidateListFromGemmOperands()
{
    auto addFusionCandidate = [&](Tensor* candidate, unsigned gemmIndex, GemmOperandType operandType) {
        if (candidate->isPersistent()) return;
        if (std::find(m_fusionCandidates.begin(), m_fusionCandidates.end(), candidate) == m_fusionCandidates.end())
        {
            m_fusionCandidates.push_back(candidate);
            m_fusionCandidatesInfo.emplace_back(gemmIndex, operandType);
        }
    };

    // build gemm operand candidate list
    for (unsigned nodeIndex = 0; nodeIndex < m_nodes.size(); ++nodeIndex)
    {
        const EagerNode& node = m_nodes[nodeIndex];
        switch (node->getNodeType())
        {
            case Node::TYPE_GEMM:
            case Node::TYPE_BATCH_GEMM:
            {
                // collect gemm operands
                // we can have cases where a tensor acts as both inputs of the gemm or
                // is an output of one gemm and input of a following one.
                // The later is not relevant for fusion and we do not attempt to optimize
                // the first one as it is an infrequent use case.
                const TensorVector& inputs = node->getInputs();
                addFusionCandidate(inputs[0].get(), nodeIndex, GemmOperandType::INPUT_A);
                addFusionCandidate(inputs[1].get(), nodeIndex, GemmOperandType::INPUT_B);
                const TensorVector& outputs = node->getOutputs();
                addFusionCandidate(outputs[0].get(), nodeIndex, GemmOperandType::OUTPUT);
                break;
            }
            default:
                break;
        }
    }
}

void EagerTransposeFuser::fillCandidateListInfo()
{
    for (unsigned nodeIndex = 0; nodeIndex < m_nodes.size(); ++nodeIndex)
    {
        const EagerNode& node        = m_nodes[nodeIndex];
        bool             isTranspose = node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE;
        // update cunsumers count
        for (const auto& TensorPtr : node->getInputs())
        {
            auto candidateIter = std::find(m_fusionCandidates.begin(), m_fusionCandidates.end(), TensorPtr.get());
            if (candidateIter != m_fusionCandidates.end())
            {
                unsigned candidateIndex = std::distance(m_fusionCandidates.begin(), candidateIter);
                // we can drop the candidate at this point if we have more than a single consumer
                // which would improve the search performance but even with remove-erase idiom
                // which only requires a swap and size update, we'll pay some cost so we avoid the optimization.
                ++m_fusionCandidatesInfo[candidateIndex].consumerCount;

                // update transpose consumer corresponding to gemm output
                if (isTranspose && m_fusionCandidatesInfo[candidateIndex].operandType == GemmOperandType::OUTPUT)
                {
                    m_fusionCandidatesInfo[candidateIndex].transposeIndex = nodeIndex;
                }
            }
        }
        // update transpose producer corresponding to gemm input
        if (isTranspose)
        {
            const auto& outputs = node->getOutputs();
            auto candidateIter  = std::find(m_fusionCandidates.begin(), m_fusionCandidates.end(), outputs[0].get());
            if (candidateIter != m_fusionCandidates.end())
            {
                unsigned candidateIndex = std::distance(m_fusionCandidates.begin(), candidateIter);
                EAGER_ASSERT(m_fusionCandidatesInfo[candidateIndex].transposeIndex ==
                                 GemmTransposedTensorCandidate::INVALID_NODE_INDEX,
                             "gemm input fusion candidate can't have two producers");
                m_fusionCandidatesInfo[candidateIndex].transposeIndex = nodeIndex;
            }
        }
    }
}

void EagerTransposeFuser::processCandidateList()
{
    for (unsigned candidateIndex = 0; candidateIndex < m_fusionCandidatesInfo.size(); ++candidateIndex)
    {
        const GemmTransposedTensorCandidate& currentCandidate = m_fusionCandidatesInfo[candidateIndex];
        if (currentCandidate.consumerCount > 1 ||
            currentCandidate.transposeIndex == GemmTransposedTensorCandidate::INVALID_NODE_INDEX)
        {
            continue;
        }
        auto transposeNode = m_nodes[currentCandidate.transposeIndex].get<TransposeNode>();
        bool canFuse =
            TransposeFuserBase::isValidPermutation(transposeNode->permutation(), m_nodes[currentCandidate.gemmIndex]);
        if (canFuse)
        {
            auto          gemmNode   = m_nodes[currentCandidate.gemmIndex].get<GEMMNode>();
            synGEMMParams gemmParams = gemmNode->getGEMMParams();
            switch (currentCandidate.operandType)
            {
                case GemmOperandType::INPUT_A:
                {
                    gemmParams.transpose_a     = !gemmParams.transpose_a;
                    const TensorVector& inputs = transposeNode->getInputs();
                    gemmNode->replaceInput(0, inputs[0]);
                    break;
                }
                case GemmOperandType::INPUT_B:
                {
                    gemmParams.transpose_b     = !gemmParams.transpose_b;
                    const TensorVector& inputs = transposeNode->getInputs();
                    gemmNode->replaceInput(1, inputs[0]);
                    break;
                }
                case GemmOperandType::OUTPUT:
                {
                    // given A*B=C -> (B^T)*(A^T)=(C^T)
                    const synGEMMParams& origGemmParams = gemmNode->getGEMMParams();
                    gemmParams.transpose_a              = !origGemmParams.transpose_b;
                    gemmParams.transpose_b              = !origGemmParams.transpose_a;
                    // swap the input operands
                    TensorPtr operandA = gemmNode->getInput(0);
                    TensorPtr operandB = gemmNode->getInput(1);
                    gemmNode->replaceInput(0, operandB);
                    gemmNode->replaceInput(1, operandA);
                    // replace the output operand
                    const TensorVector& outputs = transposeNode->getOutputs();
                    gemmNode->replaceOutput(0, outputs[0]);
                    break;
                }
            }
            gemmNode->setGEMMParams(gemmParams);
            // free the transpose without actually freeing it during compile time
            // to postpone the deletion overhead to cleanup phase.
            m_nodes[currentCandidate.transposeIndex].invalidate();
        }
    }
}

}  // namespace eager_mode
#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/eager_node.h"

namespace eager_mode
{
class EagerTransposeFuser
{
public:
    EagerTransposeFuser(EagerNodesVec& nodes) : m_nodes(nodes) {}
    void fuseTransposes();

private:
    void buildCandidateListFromGemmOperands();
    void fillCandidateListInfo();
    void processCandidateList();
    bool candidateListEmpty() { return m_fusionCandidates.empty(); }

    enum class GemmOperandType : uint8_t
    {
        INPUT_A,
        INPUT_B,
        OUTPUT
    };

    struct GemmTransposedTensorCandidate
    {
        static constexpr unsigned INVALID_NODE_INDEX = -1;

        GemmTransposedTensorCandidate(unsigned _gemmIndex, GemmOperandType _operandType)
        : gemmIndex(_gemmIndex), operandType(_operandType)
        {
        }

        unsigned        gemmIndex      = INVALID_NODE_INDEX;
        unsigned        transposeIndex = INVALID_NODE_INDEX;
        unsigned        consumerCount  = 0;
        GemmOperandType operandType;
    };

    EagerNodesVec& m_nodes;
    // we pick a threshold which is allowing us to almost always avoid allcoations
    // but on the other hand allows for a better cpu data cache utilization.
    // this is also why we choose to use a separate vector for tensor candidates
    // information and actual tensor pointers to improve search performance.
    static constexpr unsigned GEMM_LOCAL_BUFFER_ELEMENTS            = 4;
    static constexpr unsigned GEMM_CANDIDATES_LOCAL_BUFFER_ELEMENTS = 3 * GEMM_LOCAL_BUFFER_ELEMENTS;
    SmallVector<Tensor*, GEMM_CANDIDATES_LOCAL_BUFFER_ELEMENTS>                       m_fusionCandidates;
    SmallVector<GemmTransposedTensorCandidate, GEMM_CANDIDATES_LOCAL_BUFFER_ELEMENTS> m_fusionCandidatesInfo;
};

}  // namespace eager_mode

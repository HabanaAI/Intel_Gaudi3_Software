#pragma once
#include "habana_graph.h"

class TpcRecompileDb
{
public:
    struct RecompileCacheKey
    {
        std::string guid;
        unsigned    tensorIdToDuplicate;
    };
    struct RecompileCacheValue
    {
        bool     isSuccessulFusion;
        void*    resultElf;
        unsigned resultElfSize;
    };
    std::optional<RecompileCacheValue> getFusionFromDb(const RecompileCacheKey& key);
    void registerTpcFusion(const RecompileCacheKey& key, const RecompileCacheValue& value);

private:
    struct RecompileCacheComparator
    {
        bool operator()(const RecompileCacheKey& pair1, const RecompileCacheKey& pair2) const
        {
            if (pair1.guid < pair2.guid)
            {
                return true;
            }
            if (pair1.guid > pair2.guid)
            {
                return false;
            }
            return pair1.tensorIdToDuplicate < pair2.tensorIdToDuplicate;
        }
    };
    std::map<RecompileCacheKey, RecompileCacheValue, RecompileCacheComparator> m_recompileCache;
};

class CandidateInfo
{
public:
    CandidateInfo(const NodePtr& node, const TensorPtr& connectingTensor, bool isInput);
    const NodePtr&   getNode() const { return m_candidate; }
    const TensorPtr& getConnectingTensor() const { return m_connectingTensor; }
    bool             isInput() const { return m_isInput; }
    unsigned         getOrigTensorIdx() const { return m_gcTensorIdx; }
    unsigned         getTensorIdToDuplicate() const { return m_llvmTensorIdx; }
    bool             operator==(const CandidateInfo& rhs) const
    {
        return m_candidate == rhs.getNode() && m_connectingTensor == rhs.getConnectingTensor() &&
               m_isInput == rhs.isInput() && m_gcTensorIdx == rhs.getOrigTensorIdx() &&
               m_llvmTensorIdx == rhs.getTensorIdToDuplicate();
    }

private:
    NodePtr   m_candidate;         // The node that the directive can be fused to
    TensorPtr m_connectingTensor;  // The shared tensor between the candidate and the directive (for spill, it would be
                                   // the spill input)
    bool     m_isInput;            // Marks if the tensor to be duplicated is input or output of the candidate
    unsigned m_gcTensorIdx;        // Index of the original tensor
    unsigned m_llvmTensorIdx;      // Id of the original tensor (id in llvm-order)
};

struct FusionInfo
{
    CandidateInfo nodeToFuse;
    NodePtr       sfd;
    FusionInfo(CandidateInfo info, NodePtr directive) : nodeToFuse(info), sfd(directive) {};
};
struct CandidateInfoCmp
{
    bool operator()(const CandidateInfo& lhs, const CandidateInfo& rhs) const { return lhs.getNode() < rhs.getNode(); }
};
using CandidatesInfoVector = std::vector<CandidateInfo>;
using CandidatesInfoSet    = std::set<CandidateInfo, CandidateInfoCmp>;

class FusionCandidatesCollector
{
public:
    CandidatesInfoVector getSpillFusionCandidates(const HabanaGraph& g, const NodePtr& spill);

private:
    std::optional<CandidateInfo> getProducerCandidate(const HabanaGraph& g, const TensorPtr& tensor);
    CandidatesInfoVector getConsumerCandidates(const HabanaGraph& g, const TensorPtr& tensor, const NodePtr& spill);
};
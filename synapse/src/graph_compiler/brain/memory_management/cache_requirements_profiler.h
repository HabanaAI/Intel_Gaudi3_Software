#pragma once

#include "cache_management_apis.h"
#include "memory_usage_db.h"
#include "strategy.h"

namespace gc::layered_brain
{
class CacheRequirementProfiler : public CacheRequirementProfilerIfc
{
public:
    CacheRequirementProfiler(const HabanaGraph& graph, const MemoryUsageDB& db, const StrategyPtr& strategy)
    : m_graph(graph), m_db(db), m_strategy(strategy)
    {
    }

    InputCacheUsageProfile  inputProfile(size_t opIdx, size_t inputIdx) override;
    OutputCacheUsageProfile outputProfile(size_t opIdx, size_t outputIdx) override;

    virtual ~CacheRequirementProfiler() = default;

private:
    const HabanaGraph&   m_graph;
    const MemoryUsageDB& m_db;
    const StrategyPtr    m_strategy;

    size_t    m_nodeIdx;  // Using nodeIdx instead of opIdx to distinguish it from operandIdx
    size_t    m_operandIdx;
    NodePtr   m_node;
    TensorPtr m_slice;
    TensorPtr m_realSlice;

    void initInputAnalysis(size_t nodeIdx, size_t inputIdx);
    void initOutputAnalysis(size_t nodeIdx, size_t outputIdx);
    void initAnalysis(size_t nodeIdx, size_t operandIdx);
    void initSlices(const TensorVector& operands);

    //
    // input profiling
    //
    void setProducerProperties(InputCacheUsageProfile& profile) const;
    bool samePerforation(const NodePtr& producer, const NodePtr& consumer) const;
    bool samePerforationLogicalProducer(const NodePtr& producer, const NodePtr& consumer) const;
    bool samePerforationPhysicalNodes(const NodePtr& producer, const NodePtr& consumer) const;
    void setConsumersProfile(InputCacheUsageProfile& profile) const;
    void setBPTAccess(InputCacheUsageProfile& profile) const;
    void setAllRequired(InputCacheUsageProfile& profile) const;
    void setNumReads(InputCacheUsageProfile& profile) const;

    std::vector<size_t> physicalConsumerSteps() const;

    //
    // output profiling
    //
    void setRealSliceConsumersProperties(OutputCacheUsageProfile& profile) const;
    void setRMWAccess(OutputCacheUsageProfile& profile) const;
    bool isRMWReduction(const ReductionNode* reductionNode) const;
    void setPerforation(OutputCacheUsageProfile& profile) const;

    std::optional<size_t> findReductionConsumerIdx() const;
    size_t                findLastReductionProducerIdx(const NodePtr& reductionNode) const;
    std::optional<size_t> findFirstPhysicalConsumerIdx() const;
    bool                  isTensorPerforated() const;

    //
    // inline utilities
    //
    const MemoryUsageDB::SliceEntry::Properties& properties(const TensorPtr& t) const
    {
        return m_db.slices.at(t).properties;
    }

    const NodePtr& sliceNode(size_t nodeIdx) const { return m_db.steps.at(nodeIdx).sliceNode; }

    template<class ProfileType>
    void setSize(ProfileType& profile) const
    {
        // TODO [147167] - align to cache lines maybe?
        profile.size = m_slice->getDenseSizeInBytes();
    }

    static const NodePtr& bigNode(const NodePtr& anyNode)
    {
        const auto& annotationBigNode = anyNode->getNodeAnnotation().origBigNode;
        return annotationBigNode ? annotationBigNode : anyNode;
    }
};
}  // namespace gc::layered_brain
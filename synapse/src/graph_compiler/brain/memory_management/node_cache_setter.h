#pragma once

#include "cache_management_apis.h"
#include "layered_brain.h"
#include "memory_management/bundle_cache_allocator.h"
#include "memory_management/bundle_cache_state.h"
#include "memory_management/node_access_cache_setter.h"
#include "memory_management/cache_yield_queue.h"
#include "memory_usage_db.h"

namespace gc::layered_brain
{
class NodeCacheSetter : public NodeCacheSetterIfc
{
public:
    NodeCacheSetter(HabanaGraph&         graph,
                    const BundleNodes&   nodes,
                    const MemoryUsageDB& db,
                    BundleCacheState&    cacheStateTracker,
                    unsigned             pipelineDepth);

    // Set a node's cache directive and return whether all accesses succeeded as required.
    bool setDirectives(size_t nodeIdx, CacheRequirementsAnalyzerIfc* requirementAnalyzer) override;

    virtual ~NodeCacheSetter() = default;

private:
    HabanaGraph&         m_graph;
    const BundleNodes&   m_nodes;
    const MemoryUsageDB& m_db;
    BundleCacheState&    m_cacheState;
    BundleCacheAllocator m_allocator;
    CacheYieldQueue      m_yieldQueue;
    unsigned             m_pipelineDepth;  // Number of concurrent threads scheduled by the scheduler

    // Convenient holders of the node being handled by the setDirectives method
    NodePtr m_currentNode {};
    size_t  m_currentNodeIdx {};

    using Requirement = CacheRequirementsAnalyzerIfc::RequirementDetails;
    using AllocResult = BundleCacheAllocator::Result;

    struct CachingResult
    {
        bool successful = true;

        // Operands that needs releasing after the current node has had its operands cached
        TensorSet release {};

        // Node indices of accesses to reclaimed budget. Need to ensure strong order s.t these nodes end before the
        // current node begins.
        std::set<size_t> dependencies {};

        CachingResult& operator+=(const CachingResult& other)
        {
            successful &= other.successful;
            release.insert(other.release.begin(), other.release.end());
            dependencies.insert(other.dependencies.begin(), other.dependencies.end());
            return *this;
        }
    };

    static bool shouldSkipNode(const NodePtr& node);
    static bool shouldSkipTensor(const TensorPtr& t);

    CachingResult setInputDirectives(CacheRequirementsAnalyzerIfc* requirementAnalyzer);
    CachingResult setOutputDirectives(CacheRequirementsAnalyzerIfc* requirementAnalyzer);

    CachingResult cacheAccess(const TensorPtr& t, const Requirement& reqs, NodeAccessCacheSetter& setter);
    void          setCacheAccessRelease(const Requirement& reqs, NodeAccessCacheSetter& setter);
    AllocResult   allocateCache(const TensorPtr& tensor, const Requirement& req);
    TensorPtr     cacheKey(const TensorPtr& tensor) const;
    LogicalMcid   allocateMCID(const Requirement& requirements);

    void releaseAll(const TensorSet& tensors);
    void releaseByCacheKey(const TensorPtr& tensor);

    void ensureDependencies(const std::set<size_t>& depIndices);
    void ensureDependency(const NodePtr& blocker);
    bool skipAddingSync(const NodePtr& blocker) const;
    bool sameEngine(const NodePtr& a, const NodePtr& b) const;

    static void          initAnnotations(NodePtr& node);
    static void          initCMDContainer(std::vector<CacheMetaData>& operandsCMDVec, size_t numOperands);
    static CacheMetaData defaultCMD();

    void                             registerYieldingOption(size_t& inputIdx, Requirement& req);
    bool                             tryYielding(uint64_t capacity);
    CacheYieldQueue::YieldingOptions yieldingOptions(size_t maxYieldingThread) const;
    void                             yield(uint64_t capacity, const CacheYieldQueue::YieldingOptions& options);
    void                             executeYieldRelease(const CacheYieldQueue::Candidate& candidate);

    void logNodeSummary(const CachingResult& result) const;
};

}  // namespace gc::layered_brain
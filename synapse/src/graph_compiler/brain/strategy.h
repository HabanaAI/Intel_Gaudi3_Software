#pragma once

#include "brain_conf.h"
#include "bundle_view.h"
#include "mme_brain_ifc.h"
#include "types.h"

namespace gc::layered_brain
{
class BVDMultiplier
{
public:
    BVDMultiplier() : m_isSliced(false), m_multiplier(1) {}
    explicit BVDMultiplier(uint64_t multiplier) : m_isSliced(true), m_multiplier(multiplier) {}

    bool     isSliced() const { return m_isSliced; }
    uint64_t getMultiplier() const
    {
        HB_ASSERT(isSliced(), "Expected sliced BVD");
        return m_multiplier * m_inflationFactor;
    }
    uint64_t getInflationFactor() const { return m_inflationFactor; }
    void     inflateOneStep() { m_inflationFactor++; }
    void     unslice() { m_isSliced = false; }

private:
    bool     m_isSliced;
    uint64_t m_multiplier;
    uint64_t m_inflationFactor = 1;
};

using MultiplierPerBVD   = std::unordered_map<BundleViewId, BVDMultiplier>;
using PerforationPerNode = std::map<NodePtr, std::optional<BundleViewId>>;

// A MME strategy is granularity of execution in various dimensions, accompanied by its quality metrics: utilization, BW
// requirements, number of reads of input tiles, etc.
class Strategy
{
public:
    Strategy();
    explicit Strategy(const MmeSolutionPtr& mmeSolution);
    ~Strategy() = default;
    Strategy(const Strategy& other);
    Strategy(Strategy&&) = delete;
    Strategy& operator=(const Strategy&) = delete;
    Strategy& operator=(Strategy&&) = delete;

    std::shared_ptr<Strategy> clone() const;

    uint64_t index() const;

    BVDMultiplier getBVDMultiplier(BundleViewId bvd) const;
    void          setBVDMultiplier(BundleViewId bvd, const BVDMultiplier& multiplier);
    void          inflateBVD(BundleViewId bvd, uint64_t resolution);

    void     fillMissingMultipliers(const BundleViewContainerPtr& bundleViews, uint64_t value);
    uint64_t getNumOfSlicesForBVD(BundleViewId bvd, const BundleViewContainerPtr& bundleViews) const;

    MmeSolutionPtr getMmeSolution() const;
    void           updateMmeSolution(const MmeSolutionPtr& mmeSolution);

    std::vector<BundleViewId> getMMEInflateForUtilizationBVDs(const NodePtr& node) const;
    std::vector<BundleViewId> getMMEInflateForBwBVDs(const NodePtr& node) const;
    std::vector<BundleViewId> getMMEPreferredPerforationBVDs(const NodePtr& node) const;
    std::vector<BundleViewId> getCommonDimsOfAllMMEs() const;
    std::vector<BundleViewId> getMMECommonDims(const NodePtr& node) const;
    bool                      isSlicedOnCommonDim() const;

    NodeToItemOrderedMap<std::vector<BundleViewId>> getWalkPatternPerMmeNode() const;

    void                        setPerforationData(const PerforationPerNode& perforationPerNode);
    const PerforationPerNode&   getPerforationData() const;
    std::optional<BundleViewId> getPerforationBVDForNode(const NodePtr& node) const;
    unsigned                    getNumPerforatedNodes() const;
    SolutionParamsPtr           getNodeQORs(const NodePtr& node) const;

    unsigned getPipelineDepth() const;
    void     setPipelineDepth(unsigned pipelineDepth);

    void log() const;

private:
    void validateMmeSolutionBundleViewDims(const std::vector<unsigned int>& dims, const NodePtr& node) const;

    uint64_t         m_idx;
    MultiplierPerBVD m_granularityMultiplier;  // Tile sizes as granularity multipliers, per bundle-view idx
    MmeSolutionPtr   m_mmeSolution;  // Solution from MME-brain: includes quality and expansion information per MME node
    PerforationPerNode m_perforation;  // BVD for DCORE partition per bundle node

    // Number of concurrent threads scheduled by the layered brain scheduler
    unsigned m_pipelineDepth = GCFG_LAYERED_BRAIN_SCHEDULER_MIN_PIPELINE_DEPTH.value();

    static std::atomic<uint64_t> m_nextIdx;
};

using StrategyPtr    = std::shared_ptr<Strategy>;
using StrategyVector = std::vector<StrategyPtr>;

}  // namespace gc::layered_brain
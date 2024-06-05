#pragma once

#include "node.h"
#include "types.h"
#include <cstdint>
#include <unordered_map>
#include "bundle_view.h"
#include "strategy.h"
#include "brain_conf.h"

namespace gc::layered_brain
{
using BundleIdx              = uint32_t;
using BVDCoord               = llvm_vecsmall::SmallVector<uint64_t, HABANA_DIM_MAX>;
using InputsCoordsPerNode    = NodeToItemOrderedMap<std::vector<BVDCoord>>;
using NumSlicesPerBVD        = llvm_vecsmall::SmallVector<uint64_t, HABANA_DIM_MAX>;
using BPTClonePersistenceMap = std::unordered_map<TensorPtr, bool>;

enum class SlicingPolicy
{
    ENOUGH_REUSE
};

class BundleData
{
public:
    BundleData() {}
    void                   setBundleViews(const BundleViewContainerPtr& bundleViews) { m_bundleViews = bundleViews; }
    BundleViewContainerPtr getBundleViews() const { return m_bundleViews; }

    void        setFinalStrategy(const StrategyPtr& strategy) { m_finalStrategy = strategy; }
    StrategyPtr getFinalStrategy() const { return m_finalStrategy; }
    unsigned    getPipelineDepth() const
    {
        const auto& strategy = getFinalStrategy();
        HB_ASSERT_PTR(strategy);
        return strategy->getPipelineDepth();
    }

    void     setNumOfSlicesPerBVD(const NumSlicesPerBVD& numOfSlicesPerBVD) { m_numOfSlicesPerBVD = numOfSlicesPerBVD; }
    uint64_t getNumOfSlicesPerBVD(BundleViewId bvdId) const
    {
        HB_ASSERT(bvdId < m_numOfSlicesPerBVD.size(), "invalid bvd ID {}", bvdId);
        return m_numOfSlicesPerBVD.at(bvdId);
    }

    uint64_t getTotalNofBundleSlices() const
    {
        uint64_t nofSlices = 1;
        for (auto bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); ++bvdId)
        {
            uint64_t nofBvdSlices = getNumOfSlicesPerBVD(bvdId);
            HB_ASSERT(nofBvdSlices >= 1, "Expecting nofBvdSlices {} >= 1", std::to_string(nofBvdSlices));
            nofSlices *= nofBvdSlices;
        }
        return nofSlices;
    }

    const InputsCoordsPerNode& getRouteEndInputsCoords() const { return m_routeEndInputsCoords; }

    void addRouteEndInputsCoords(const NodePtr& routeEnd, const std::vector<BVDCoord>& inputsCoords)
    {
        m_routeEndInputsCoords.emplace(routeEnd, inputsCoords);
    }

    // Returns the routeEnd input index with the corresponding BVD coordinate, or no value if there's no slice for the
    // given coord. Each bundle tensor contains a subset of the BVDs, and the caller may iterate all BVDs, so no value
    // is a valid output.
    static std::optional<uint64_t> getRouteEndInputIndexByCoord(const InputsCoordsPerNode& routeEndInputsCoords,
                                                                const NodePtr&             routeEnd,
                                                                const BVDCoord&            coord)
    {
        const auto& routeEndIt = routeEndInputsCoords.find(routeEnd);
        HB_ASSERT(routeEndIt != routeEndInputsCoords.end(),
                  "all route end nodes must have inputs coordinates data available");
        const std::vector<BVDCoord>& inputsCoords = routeEndIt->second;
        size_t                       inputIdx     = index_of(inputsCoords, coord);
        if (inputIdx == -1) return std::nullopt;
        return inputIdx;
    }

    bool isSlicerReduction(const NodePtr& n) const
    {
        return (m_routeEndInputsCoords.find(n) != m_routeEndInputsCoords.end()) &&
               (n->getNodeType() == Node::TYPE_INTERNAL_REDUCTION);
    }

    void     setMaxCacheUsage(uint64_t capacityBytes) { m_maxCacheUsageBytes = capacityBytes; }
    uint64_t maxCacheUsageBytes() const { return m_maxCacheUsageBytes; }

    void addBPTClonePersistence(const TensorPtr& bptClone, bool persistent)
    {
        HB_ASSERT_PTR(bptClone);
        m_bptClonePersistence.emplace(bptClone, persistent);
    }

    const BPTClonePersistenceMap& getBPTClonePersistenceMap() const { return m_bptClonePersistence; }

private:
    BundleViewContainerPtr        m_bundleViews;
    StrategyPtr                   m_finalStrategy;
    NumSlicesPerBVD               m_numOfSlicesPerBVD;
    uint64_t                      m_maxCacheUsageBytes = uint64_t(0);
    BPTClonePersistenceMap        m_bptClonePersistence;  // {bpt clone (dry run) -> is cloned tensor persistent}
    InputsCoordsPerNode           m_routeEndInputsCoords;
};

class LayeredBrainData
{
public:
    std::unordered_map<BundleIdx, BundleData> m_bundleData;

    bool isLayeredBrainBundle(BundleIdx bundleIdx) const { return m_bundleData.find(bundleIdx) != m_bundleData.end(); }
};

}  // namespace gc::layered_brain

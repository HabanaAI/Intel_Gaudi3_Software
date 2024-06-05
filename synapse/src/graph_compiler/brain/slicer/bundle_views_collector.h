#pragma once

#include "common_tile_size_calculator.h"
#include "types.h"
#include "brain_data.h"

namespace gc::layered_brain
{
// The bundle-views collector returns a list of all separate dimensions identified in the bundle subgraph.
class BundleViewsCollector
{
public:
    using TensorDimToBVDSet = std::map<std::pair<TensorPtr, Dim>, std::set<BundleViewId>>;
    using NodeDimToBVD      = std::map<std::pair<NodePtr, Dim>, BundleViewId>;

    explicit BundleViewsCollector(const NodeVector& bundleNodes) : m_bundleNodes(bundleNodes) {}
    BundleViewContainerPtr getAllBundleViews(const TileSizePerTensor& granularityPerTensor,
                                             const TileSizePerNode&   granularityPerNode);

private:
    // Allocates a BVD id to each node dim and add it to all its mapped tensor dims.
    // At this stage tensor dim may belong to multiple BVDs that will be unified later to a single BVD.
    void createInitialBundleViews();

    // Merges BVDs so each tensor dim will be mapped to exactly one BVD.
    void mergeInitialBundleViews();

    // Checks if 2 sets intersect
    bool isIntersect(const std::set<BundleViewId>& s1, const std::set<BundleViewId>& s2) const;

    // Looks for 2 intersecting sets in merged BVDs. Returns empty pair if not found.
    std::optional<std::pair<size_t, size_t>> findIntersectingPair() const;

    // Allocates final BVD ids to the merged BVDs.
    // Creates a mapping from initial BVD id to final BVD id.
    void reallocateBundleViewIds();

    // Creates final BVDs - updates granularity per node dim and per tensor dim.
    BundleViewContainerPtr createFinalBundleViews(const TileSizePerTensor& granularityPerTensor,
                                                  const TileSizePerNode&   granularityPerNode);

    const NodeVector m_bundleNodes;

    TensorDimToBVDSet                   m_tensorDimToBVDSet;
    NodeDimToBVD                        m_nodeDimToBVD;
    uint32_t                            m_numOfInitialBVDs = 0;
    std::vector<std::set<BundleViewId>> m_mergedBVDs;
    uint32_t                            m_numOfFinalBVDs = 0;
    std::vector<BundleViewId>           m_initialBVDToFinalBVD;
};

}  // namespace gc::layered_brain
#pragma once

#include "types.h"
#include "bundle_view.h"
#include "strategy.h"

namespace gc::layered_brain::slicer
{
class ConflictingPerforationDetector final
{
public:
    void addNodeBVDs(const std::optional<BundleViewId>& perforationBVD, const BVDSet& nodeDimsBVDs);

    // return true if there is an internal conflict in the perforation BVD assignment of the added nodes.
    bool hasConflict() const;

private:
    using Alternatives = std::unordered_map<BundleViewId, BVDSet>;

    struct NodeBVDs
    {
        BundleViewId perforationBVD;
        BVDSet       nodeDimsBVDs;
    };
    std::vector<NodeBVDs> m_nodeBVDs;

    bool         validPerforations() const;
    bool         disjointPerforations() const;
    Alternatives calcAlternatives() const;

    static BVDSet intersect(const BVDSet& s1, const BVDSet& s2);
    static bool   contains(const BVDSet& bvds, const BundleViewId& bvdId);
};

// Filtering strategies before an expansive evaluation procedure is performed on them
class StrategyFilter final
{
public:
    StrategyFilter(const BundleViewContainerPtr& bundleViewContainer, const NodeVector& keyNodes)
    : m_bundleViewContainer(bundleViewContainer), m_keyNodes(keyNodes)
    {
    }

    // return true if the strategy is valid (should not be filtered out). The caller is expected to filter out any
    // strategy for which this method returns false (the object is suitable for std::copy_if(..., strategyFilter)).
    bool operator()(const StrategyPtr& strategy) const;

private:
    const BundleViewContainerPtr& m_bundleViewContainer;
    const NodeVector&             m_keyNodes;

    ConflictingPerforationDetector cpdFromStrategy(const StrategyPtr& strategy) const;
};

}  // namespace gc::layered_brain::slicer
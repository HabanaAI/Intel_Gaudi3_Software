#pragma once

#include "bundle_view.h"
#include "strategy.h"
#include "types.h"
#include <optional>

namespace gc::layered_brain
{
struct PerforationCandidates
{
    std::optional<BundleViewId> mmeCandidate;
    std::vector<BundleViewId>   preferredCandidates;
    std::vector<BundleViewId>   validCandidates;
};

using ReducedBVDsPerNode = std::map<NodePtr, BVDSet>;

// The perforation BVD candidates finder is responsible to find perforation candidates for each node in the bundle:
// 1) MME candidate - taken from MME brain solution for MME nodes.
// 2) Preferred candidates:
//    - Have a max multiplier of at least 4 (according to numDcores).
//    - Orthogonal to reduction dimensions.
//    Sorted by number of occurrences in bundle nodes - BVDs that are common in all (or most) operations will be first.
// 3) Valid candidates:
//    - Have a max multiplier of at least 4 (according to numDcores).
//    Sorted by distance from FCD - BVDs that are mapped to external tensor dims will be first.
class PerforationBVDCandidatesFinder
{
public:
    PerforationBVDCandidatesFinder(const NodeVector&             bundleNodes,
                                   const BundleViewContainerPtr& bundleViews,
                                   unsigned                      numDcores)
    : m_bundleNodes(bundleNodes), m_bundleViews(bundleViews), m_numDcores(numDcores)
    {
    }

    std::map<NodePtr, PerforationCandidates>
    findPerforationCandidates(const StrategyPtr& strategy, const ReducedBVDsPerNode& reducedBVDsPerNode) const;

private:
    std::vector<BundleViewId>        getBundlePreferredCandidates(const StrategyPtr&        strategy,
                                                                  const ReducedBVDsPerNode& reducedBVDsPerNode) const;
    std::vector<BundleViewId> getBundleValidCandidates(const StrategyPtr& strategy) const;
    PerforationCandidates            getNodeCandidates(const NodePtr&                   node,
                                                       const StrategyPtr&               strategy,
                                                       const std::vector<BundleViewId>& bundlePreferredCandidates,
                                                       const std::vector<BundleViewId>& bundleValidCandidates,
                                                       const BVDSet&                    reducedBVDs) const;
    bool isCDPerforationAllowed(const NodePtr& mmeNode, const StrategyPtr& strategy, BundleViewId cdBVD) const;
    bool
    isValidNodeBVD(const NodePtr& node, BundleViewId bvd, const StrategyPtr& strategy, const BVDSet& reducedBVDs) const;
    std::pair<Dim, unsigned>  getNodeExternalTensorDimInBVD(const NodePtr& node, BundleViewId bvd) const;
    BVDSet getNodeReducedBVDs(const NodePtr& node, const ReducedBVDsPerNode& reducedBVDsPerNode) const;

    const NodeVector             m_bundleNodes;
    const BundleViewContainerPtr m_bundleViews;
    const unsigned               m_numDcores;
};

}  // namespace gc::layered_brain
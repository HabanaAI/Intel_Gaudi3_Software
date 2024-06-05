#pragma once

#include "bundle_scheduler_interfaces.h"
#include "bundle_view.h"

namespace gc::layered_brain
{
class SlicedBVDsTraversalSorter : public BVDsTraversalSorter
{
public:
    // BVDs traversal can be done efficiently by setting the order of BVDs which are traversed first.
    // Efficiency is set by multiple considerations.
    // Each efficiency consideration selects the higher priority BVDs, which should be traversed before the lower
    // priority BVDs. For multiple higher priority BVDs - the order between them is set according to the order in
    // highPrioBvdsByOrder. Lower priority BVDs should change priority with respect to BVDs in highPrioBvdsByOrder only.
    // Lower priority are placed after the higher priority BVDs, but without modifying relations to other bundle BVDs.
    struct BvdsOrderPreference
    {
        BvdTraversalPattern    highPrioBvdsByOrder;
        std::set<BundleViewId> lowPrioBvds;
    };

    SlicedBVDsTraversalSorter() {}
    BvdTraversalPattern getBundleViewsByTraversalOrder(const BundleData& bundleData) override;

protected:
    BvdTraversalPattern orderBundleViews(const BundleData& bundleData);

    std::set<BundleViewId> getSlicedBundleViews(const BundleData& bundleData);

    void applyOrderPreference(std::list<BundleViewId>& bundleViews, const BvdsOrderPreference& orderPreference);

    std::vector<BvdsOrderPreference> getWalkingPatterns(const BundleData& bundleData);

    std::optional<BundleViewId> getMinIndexSlicedBvd(std::list<BundleViewId>& slicedBVDs,
                                                     std::set<BundleViewId>   searchedBVDs);
    void
    ensureBvdsPairOrder(BundleViewId shouldBeFirst, BundleViewId shouldBeAfter, std::list<BundleViewId>& slicedBVDs);

    std::vector<BundleViewId> getFilteredWalkDirection(const std::vector<BundleViewId> unfilteredWalkDir,
                                                       const BundleData&               bundleData,
                                                       const StrategyPtr&              strategy) const;
};

}  // namespace gc::layered_brain
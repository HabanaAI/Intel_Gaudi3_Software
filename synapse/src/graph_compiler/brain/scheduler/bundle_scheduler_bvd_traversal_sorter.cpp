#include "bundle_scheduler_bvd_traversal_sorter.h"
#include "brain_data.h"
#include "bundle_view.h"
#include "habana_graph.h"
#include <limits>

using namespace gc::layered_brain;

template<class CONTAINER>
BvdTraversalPattern intersectTraversalPattern(const BvdTraversalPattern& order, const CONTAINER& bvds)
{
    // Keep just the orderd BVDs which are in "bvds"
    BvdTraversalPattern slicedOrder;
    for (unsigned i = 0; i < order.size(); i++)
    {
        if (std::find(bvds.begin(), bvds.end(), order[i]) != bvds.end())
        {
            slicedOrder.push_back(order[i]);
        }
    }
    return slicedOrder;
}

BvdTraversalPattern SlicedBVDsTraversalSorter::getBundleViewsByTraversalOrder(const BundleData& bundleData)
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);

    auto slicedBVDs = getSlicedBundleViews(bundleData);
    if (slicedBVDs.empty())
    {
        return {};
    }
    std::list<BundleViewId> orderedBVDIds(slicedBVDs.begin(), slicedBVDs.end());

    // 1. relaxed dims - not supported yet
    // 2. maximize reuse - TODO SW-137966
    // 3. chosen MME strategy walk - per MME
    auto walkingPatterns = getWalkingPatterns(bundleData);
    for (const auto& preference : walkingPatterns)
    {
        applyOrderPreference(orderedBVDIds, preference);
    }
    // 4. soluion anchors - not supported yet
    BvdTraversalPattern ret {orderedBVDIds.begin(), orderedBVDIds.end()};
    LOG_DEBUG(LB_SCHEDULER,
              "bundle views : {} -> ordered bundle views: {}",
              toString(slicedBVDs, ','),
              toString(orderedBVDIds, ','));
    return ret;
}

// Get sliced BVDs according to actual multiplier - if smaller than max multiplier
std::set<BundleViewId> SlicedBVDsTraversalSorter::getSlicedBundleViews(const BundleData& bundleData)
{
    std::set<BundleViewId> slicedBVDs;
    BundleViewContainerPtr bundleViews = bundleData.getBundleViews();
    HB_ASSERT_PTR(bundleViews);
    for (BundleViewId bvdIdx = 0; bvdIdx < bundleViews->getNumOfBundleViews(); bvdIdx++)
    {
        auto numSlices = bundleData.getNumOfSlicesPerBVD(bvdIdx);
        if (numSlices > 1)  // BVD is sliced
        {
            slicedBVDs.insert(bvdIdx);
        }
    }
    return slicedBVDs;
}

// Order slicedBVDs such that the sliced dims from orderPreference have the same order as in this vector, while all
// lowerPrioBVDs are after all dims in orderPreference, without changing their order.
void SlicedBVDsTraversalSorter::applyOrderPreference(std::list<BundleViewId>&   slicedBVDs,
                                                     const BvdsOrderPreference& orderPreference)
{
    LOG_DEBUG(LB_SCHEDULER, "Given bvds order: {}", toString(slicedBVDs, ','));
    // Keep just the sliced BVDs in the order preference
    BvdTraversalPattern slicedOrderPreference =
        intersectTraversalPattern(orderPreference.highPrioBvdsByOrder, slicedBVDs);

    LOG_DEBUG(LB_SCHEDULER,
              "Apply order preference: {} (sliced: {}) before group {}",
              toString(orderPreference.highPrioBvdsByOrder, ','),
              toString(slicedOrderPreference, ','),
              toString(orderPreference.lowPrioBvds, ','));

    // The lower priority BVDs shouldn't change their "location", only the high priority ordered BVDs should be "moved".
    // Place the last element of the order preference before all the BVDs of the lower priority to apply this order.
    // Get the lowest index BVD among the lower priority BVDs, so all higher priority BVDs should be placed before it.
    std::optional<BundleViewId> minIndexBvd = getMinIndexSlicedBvd(slicedBVDs, orderPreference.lowPrioBvds);
    if (minIndexBvd.has_value())
    {
        // Move the last BVD is slicedOrderPreference before minIndexBvd
        ensureBvdsPairOrder(slicedOrderPreference.back(), minIndexBvd.value(), slicedBVDs);
    }
    // else - none of the lower priority BVDs is sliced

    // Iterate slicedOrderPreference backwards, so that the sequence of insert before will retain the original order
    // If slicedOrderPreference includes just 1 sliced dim - the loop will be skipped and indeed no ordering is
    // required, as the order is between the preference BVDs
    for (unsigned i = slicedOrderPreference.size() - 1; i > 0; i--)
    {
        ensureBvdsPairOrder(slicedOrderPreference.at(i - 1), slicedOrderPreference.at(i), slicedBVDs);
    }
    LOG_DEBUG(LB_SCHEDULER, "New bvds order: {}", toString(slicedBVDs, ','));
}

// Per MME node - set the walk direction BVDs to be traversed first by their order, before the group of MME output BVDs
std::vector<SlicedBVDsTraversalSorter::BvdsOrderPreference>
SlicedBVDsTraversalSorter::getWalkingPatterns(const BundleData& bundleData)
{
    std::vector<BvdsOrderPreference> walkPatterns;

    const auto& slicedBVDs = getSlicedBundleViews(bundleData);
    const auto& strategy   = bundleData.getFinalStrategy();
    HB_ASSERT(strategy, "final strategy is expected to be defined");

    for (auto [mmeNode, walkDirection] : strategy->getWalkPatternPerMmeNode())
    {
        if (walkDirection.empty()) continue;
        const auto filteredWalkDir = getFilteredWalkDirection(walkDirection, bundleData, strategy);
        if (!filteredWalkDir.empty())
        {
            BvdsOrderPreference preference;
            preference.highPrioBvdsByOrder = filteredWalkDir;
            // group the output BVDs, which are not part of the walk pattern
            const auto& mmeOutput = mmeNode->getOutput(0);
            for (Dim dim = 0; dim < mmeOutput->getDim(); dim++)
            {
                BundleViewId bvd = bundleData.getBundleViews()->getBVDForTensorDim(mmeOutput, dim);
                if (std::find(preference.highPrioBvdsByOrder.begin(), preference.highPrioBvdsByOrder.end(), bvd) ==
                    preference.highPrioBvdsByOrder.end())
                {
                    preference.lowPrioBvds.insert(bvd);
                }
            }
            LOG_DEBUG(LB_SCHEDULER,
                      "Add mme walking pattern {}, output BVDs {}, for {}",
                      toString(preference.highPrioBvdsByOrder, ','),
                      toString(preference.lowPrioBvds, ','),
                      mmeNode->getNodeName());
            walkPatterns.push_back(preference);
        }
    }
    return walkPatterns;
};

// Returns the min index in slicedBVDs of the any bvd from searchedBVDs, or nullopt if all searchedBVDs don't appear in
// slicedBVDs
std::optional<BundleViewId> SlicedBVDsTraversalSorter::getMinIndexSlicedBvd(std::list<BundleViewId>& slicedBVDs,
                                                                            std::set<BundleViewId>   searchedBVDs)
{
    std::optional<BundleViewId> minIndexBvd;
    size_t                      minIndex = std::numeric_limits<size_t>::max();
    for (BundleViewId bvd : searchedBVDs)
    {
        size_t index = index_of(slicedBVDs, bvd);
        if (index != -1)  // the BVD is found
        {
            if (index < minIndex)
            {
                minIndex    = index;
                minIndexBvd = bvd;
            }
        }
    }
    return minIndexBvd;
}

void SlicedBVDsTraversalSorter::ensureBvdsPairOrder(BundleViewId             shouldBeFirst,
                                                    BundleViewId             shouldBeAfter,
                                                    std::list<BundleViewId>& slicedBVDs)
{
    // check the pair order
    size_t firstIndex = index_of(slicedBVDs, shouldBeFirst);
    HB_ASSERT(firstIndex != -1, "sliced bvd wasn't found in the sliced BVDs list");
    size_t secondIndex = index_of(slicedBVDs, shouldBeAfter);
    HB_ASSERT(secondIndex != -1, "sliced bvd wasn't found in the sliced BVDs list");
    if (firstIndex > secondIndex)
    {
        // move the first BVD to be just before the second BVD
        slicedBVDs.remove(shouldBeFirst);
        auto secondIt = std::find(slicedBVDs.begin(), slicedBVDs.end(), shouldBeAfter);
        slicedBVDs.insert(secondIt, shouldBeFirst);
    }
    // else - no need to change the placement
}

std::vector<BundleViewId>
SlicedBVDsTraversalSorter::getFilteredWalkDirection(const std::vector<BundleViewId> unfilteredWalkDir,
                                                    const BundleData&               bundleData,
                                                    const StrategyPtr&              strategy) const
{
    // TODO [SW-160489] - remove this method when the walk direction is fixed for all MME nodes
    // Filter out unsliced bvds from walk direction.
    // They originate in multi mme strategies in which a single mme node alone can be sliced on CD but
    // when extending solution to multiple MMEs, CD is eventually unsliced but the walk direction
    // of the first MME isn't fixed.
    std::vector<BundleViewId> filteredWalkDir;
    std::copy_if(unfilteredWalkDir.begin(),
                 unfilteredWalkDir.end(),
                 std::back_inserter(filteredWalkDir),
                 [&strategy](BundleViewId bvd) {
                     const auto& bvdMultiplier = strategy->getBVDMultiplier(bvd);
                     return bvdMultiplier.isSliced();
                 });
    return filteredWalkDir;
}
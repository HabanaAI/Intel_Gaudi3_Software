#include "bundle_scheduler_slices_sequencer.h"
#include "brain_data.h"
#include "bundle_view.h"
#include "habana_graph.h"
#include "types.h"

using namespace gc::layered_brain;

std::vector<TensorVector> SimpleSlicesSequencer::getSliceSetsSequence(const BvdTraversalPattern&  traversalPattern,
                                                                      const BundleData&           bundleData,
                                                                      const SliceToReductionInfo& sliceToReducedBvds,
                                                                      const std::vector<NodePtr>& routeEndNodes)
{
    LOG_TRACE(LB_SCHEDULER, "{} for traversal pattern {}", HLLOG_FUNC, toString(traversalPattern, ','));
    validateTraversalPattern(traversalPattern);
    std::vector<TensorVector> sliceSetsSequence;
    const auto                routeEndInputsCoords =
        getVirtualBVDCoords(bundleData.getRouteEndInputsCoords(), bundleData, sliceToReducedBvds);
    CoordinateTraversalPattern bvdsTraversal = getBvdCoordTraversalPattern(traversalPattern, bundleData);
    // Coord may be virtual in sense of having a value > nSlices-1 for a certain bvdId.
    for (const Coordinate& coord : bvdsTraversal)
    {
        TensorVector sliceSet;
        // Find all slices with this coordinate, and add to this set
        for (const auto& routeEndNode : routeEndNodes)
        {
            const auto inputIdx = BundleData::getRouteEndInputIndexByCoord(routeEndInputsCoords, routeEndNode, coord);
            if (inputIdx.has_value())
            {
                const TensorPtr& slice = routeEndNode->getInput(inputIdx.value());
                HB_ASSERT_PTR(slice);
                sliceSet.push_back(slice);
            }
        }
        // Sets might be empty, with slices from part of the big BPTs, or with slices from all BPTs
        if (!sliceSet.empty())
        {
            sliceSetsSequence.push_back(sliceSet);
            LOG_DEBUG(LB_SCHEDULER,
                      "Add slices set for coord ({}) with {} slices: {}",
                      toString(coord, ','),
                      sliceSet.size(),
                      toString(sliceSet, ',', [](const TensorPtr& t) { return t ? t->getName() : "N/A"; }));
        }
    }
    HB_ASSERT(routeEndNodes.empty() || !sliceSetsSequence.empty(),
              "If there is output BPT, there must be at least 1 slices set");
    return sliceSetsSequence;
}

CoordinateTraversalPattern
SimpleSlicesSequencer::getBvdCoordTraversalPattern(const BvdTraversalPattern& traversalPattern,
                                                   const BundleData&          bundleData) const
{
    // Insert the sliced BVDs first, then extend to all BVDs, as the BVD coordinate includes all BVDs
    DimVector dimOrder(traversalPattern.begin(), traversalPattern.end());

    // Define a coordinate for all bundle views, sliced and unsliced, to match the slices mapping in bundle data.
    // The limit for non sliced BVDs is 1, and for the sliced BVDs the number of slices
    uint64_t   numBVDs = bundleData.getBundleViews()->getNumOfBundleViews();
    Coordinate limits(numBVDs, 1);
    for (BundleViewId bvdId : traversalPattern)
    {
        limits[bvdId] = bundleData.getNumOfSlicesPerBVD(bvdId);
    }
    // TODO SW-120887 - snake on all mme output bvds (or fcd bvd).
    DimVector                  snakeDims;
    CoordinateTraversalPattern bvdsTraversal(limits, dimOrder, snakeDims);
    return bvdsTraversal;
}

void SimpleSlicesSequencer::validateTraversalPattern(const BvdTraversalPattern& traversalPattern) const
{
    HB_ASSERT(areAllElementsUnique(traversalPattern), "traversal pattern BVDs must be unique");
}

InputsCoordsPerNode SimpleSlicesSequencer::getVirtualBVDCoords(const InputsCoordsPerNode&  routeEndInputsCoords,
                                                               const BundleData&           bundleData,
                                                               const SliceToReductionInfo& sliceToReducedBvds) const
{
    // For each slice, if there are reduced BVDs in its route, set bvd coordinate to the last valid coord of the BVD,
    // such that it will be scheduled with the last reduction input. If it's after it - it will be a new thread and
    // produces unoptimal scheduling.
    InputsCoordsPerNode virtualCoords(routeEndInputsCoords);
    for (auto& [slice, reducedSliceInfo] : sliceToReducedBvds)
    {
        if (!reducedSliceInfo.reducedBvdIds.empty())
        {
            auto& bvdCoords = virtualCoords.at(reducedSliceInfo.sliceConsumer)
                                  .at(reducedSliceInfo.sliceConsumer->getInputIndexOfTensor(slice));
            for (const auto& bvd : reducedSliceInfo.reducedBvdIds)
            {
                bvdCoords.at(bvd) = bundleData.getNumOfSlicesPerBVD(bvd) - 1;
            }
        }
    }
    return virtualCoords;
}
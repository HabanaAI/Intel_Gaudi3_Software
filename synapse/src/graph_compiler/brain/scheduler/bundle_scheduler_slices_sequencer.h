#pragma once

#include "bundle_scheduler_interfaces.h"
#include "bundle_view.h"
#include "coordinate_traversal.h"
#include "layered_brain.h"

namespace gc::layered_brain
{
class SimpleSlicesSequencer : public OutputSlicesSequencer
{
public:
    std::vector<TensorVector> getSliceSetsSequence(const BvdTraversalPattern&  traversalPattern,
                                                   const BundleData&           bundleData,
                                                   const SliceToReductionInfo& sliceToReducedBvds,
                                                   const std::vector<NodePtr>& routeEndNodes) override;

protected:
    CoordinateTraversalPattern getBvdCoordTraversalPattern(const BvdTraversalPattern& orderedBundleViews,
                                                           const BundleData&          bundleData) const;
    void                       validateTraversalPattern(const BvdTraversalPattern& traversalPattern) const;
    InputsCoordsPerNode        getVirtualBVDCoords(const InputsCoordsPerNode&  routeEndInputsCoords,
                                                   const BundleData&           bundleData,
                                                   const SliceToReductionInfo& sliceToReducedBvds) const;
};

}  // namespace gc::layered_brain
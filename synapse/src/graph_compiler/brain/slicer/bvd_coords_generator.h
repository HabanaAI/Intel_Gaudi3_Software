#pragma once

#include "brain_data.h"
#include "strategy.h"
#include "types.h"

namespace gc::layered_brain
{
// The BVD coords generator is responsible to calculate the number of slices for each BVD.
// It generates a set of BVD coords per node and project these coords on the node operands.
// For example: GEMM node (3 BVDs) sliced to 4 slices on common dim (BVD 0).
//-------------------------------------------------------------------------------------------------
// BVD coord for node  |  BVD coord for input A  |  BVD coord for input B  |   BVD coord for output
//-------------------------------------------------------------------------------------------------
//      [0, 0, 0]      |        [0, 0, 0]        |        [0, 0, 0]        |        [0, 0, 0]
//      [1, 0, 0]      |        [1, 0, 0]        |        [1, 0, 0]        |        [0, 0, 0]
//      [2, 0, 0]      |        [2, 0, 0]        |        [2, 0, 0]        |        [0, 0, 0]
//      [3, 0, 0]      |        [3, 0, 0]        |        [3, 0, 0]        |        [0, 0, 0]

class BVDCoordsGenerator
{
public:
    BVDCoordsGenerator(const BundleViewContainerPtr& bundleViews,
                       const StrategyPtr&            slicingStrategy,
                       bool                          dryRun = false);

    std::set<BVDCoord> getBVDCoordsForNode(const NodePtr& node) const;
    BVDCoord           projectBVDCoordOnTensor(const TensorPtr& tensor, const BVDCoord& bvdCoord) const;
    NumSlicesPerBVD    getNumOfSlicesPerBVD() const;

private:
    const BundleViewContainerPtr m_bundleViews;
    NumSlicesPerBVD              m_numOfSlicesPerBVD;
};

}  // namespace gc::layered_brain
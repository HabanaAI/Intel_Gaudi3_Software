#include "bvd_coords_generator.h"
#include "coordinate_traversal.h"
#include "node.h"

using namespace gc::layered_brain;

BVDCoordsGenerator::BVDCoordsGenerator(const BundleViewContainerPtr& bundleViews,
                                       const StrategyPtr&            slicingStrategy,
                                       bool                          dryRun)
: m_bundleViews(bundleViews)
{
    m_numOfSlicesPerBVD.resize(bundleViews->getNumOfBundleViews(), 1);  // Init with a single slice per BVD

    bool isDryRun = dryRun && GCFG_ENABLE_LB_SAMPLE_MODE.value();
    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
    {
        const auto& multiplier = slicingStrategy->getBVDMultiplier(bvdId);
        if (multiplier.isSliced())  // BVD is sliced
        {
            const uint64_t totalNumSlices =
                div_round_up(bundleViews->getBundleView(bvdId).resolution, multiplier.getMultiplier());
            m_numOfSlicesPerBVD[bvdId] =
                isDryRun ? std::min(totalNumSlices, (uint64_t)slicingStrategy->getPipelineDepth()) : totalNumSlices;
            LOG_DEBUG(LB_SLICER,
                      "Number of slices to generate for BVD {} : {} out of {} slices (dry run = {})",
                      bvdId,
                      m_numOfSlicesPerBVD.at(bvdId),
                      totalNumSlices,
                      isDryRun);
        }
    }
}

std::set<BVDCoord> BVDCoordsGenerator::getBVDCoordsForNode(const NodePtr& node) const
{
    std::set<BVDCoord> bvdCoords;
    const auto&        nodeAP = node->getNodeAccessPattern();
    HB_ASSERT_PTR(nodeAP);

    HB_ASSERT(m_numOfSlicesPerBVD.size() == m_bundleViews->getNumOfBundleViews(), "Expected num of slices per BVD");
    Coordinate limits(m_bundleViews->getNumOfBundleViews(), 1);
    for (Dim nodeDim = 0; nodeDim < nodeAP->getNodeResolution().size(); nodeDim++)
    {
        if (m_bundleViews->isNodeDimMappedToBVD(node, nodeDim))
        {
            BundleViewId bvdId = m_bundleViews->getBVDForNodeDim(node, nodeDim);
            limits[bvdId]      = m_numOfSlicesPerBVD.at(bvdId);
        }
    }
    CoordinateTraversalPattern bvdsTraversal(limits, {}, {});  // Traversal order is not important
    for (const Coordinate& coord : bvdsTraversal)
    {
        bvdCoords.insert(coord);
    }

    return bvdCoords;
}

BVDCoord BVDCoordsGenerator::projectBVDCoordOnTensor(const TensorPtr& tensor, const BVDCoord& bvdCoord) const
{
    BVDCoord tensorBVDcoord(m_bundleViews->getNumOfBundleViews(), 0);
    for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
    {
        BundleViewId bvdId    = m_bundleViews->getBVDForTensorDim(tensor, tensorDim);
        tensorBVDcoord[bvdId] = bvdCoord[bvdId];
    }
    return tensorBVDcoord;
}

NumSlicesPerBVD BVDCoordsGenerator::getNumOfSlicesPerBVD() const
{
    return m_numOfSlicesPerBVD;
}
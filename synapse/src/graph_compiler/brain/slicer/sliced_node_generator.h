
#pragma once

#include "brain_data.h"
#include "mme_services.h"
#include "types.h"

namespace gc::layered_brain
{
// The sliced node generator is responsible to create a sliced node from original node and BVD coordinate.
class SlicedNodeGenerator
{
public:
    SlicedNodeGenerator(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy)
    : m_bundleIdx(std::nullopt), m_bundleViews(bundleViews), m_strategy(strategy) {};

    SlicedNodeGenerator(const BundleIdx               bundleIdx,
                        const BundleViewContainerPtr& bundleViews,
                        const StrategyPtr&            strategy)
    : m_bundleIdx(bundleIdx), m_bundleViews(bundleViews), m_strategy(strategy) {};

    NodePtr     getSlicedNode(const NodePtr& origNode, const BVDCoord& bvdCoord) const;
    static void updateTensorSliceOffset(const NodePtr&     slicedNode,
                                        const TensorPtr&   origTensor,
                                        const TensorPtr&   slicedTensor,
                                        const OffsetArray& slicedTensorOffset);
    void        addAuxTensors(const NodePtr& origNode, const NodePtr& slicedNode);

private:
    void generateSlicedNodeROI(const NodePtr& origNode, const BVDCoord& bvdCoord, const NodePtr& slicedNode) const;
    void splitToDcoreROIs(const NodePtr& origNode, const NodePtr& slicedNode) const;

    const std::optional<BundleIdx> m_bundleIdx;
    const BundleViewContainerPtr   m_bundleViews;
    const StrategyPtr              m_strategy;
    MmeCommon::MmeServices         m_mmeServices;
};

}  // namespace gc::layered_brain
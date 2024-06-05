#pragma once

#include "sliced_graph_generator.h"

namespace gc::layered_brain
{
// The sliced node graph generator is responsible to create a sliced representation of a single node according to the
// bundle-views slicing, by generating all node slices (slice per ISR)
// Node inputs need to be connected to their slices via fork (read TensorView) and Node outputs
// need to be connected to their slices via join (write TensorView).
// Multiple ISRs with the same output ISMR should go through reduction.
class SlicedNodeGraphGenerator : public SlicedGraphGenerator
{
public:
    using BaseClass = SlicedGraphGenerator;

    explicit SlicedNodeGraphGenerator(const NodePtr&                node,
                                      const BundleViewContainerPtr& bundleViews,
                                      const StrategyPtr&            slicingStrategy,
                                      const NodeSet&                requireF32Reduction)
    : BaseClass({node},
                bundleViews,
                BVDCoordsGenerator(bundleViews, slicingStrategy),
                SlicedTensorGenerator(),
                SlicedNodeGenerator(bundleViews, slicingStrategy),
                BPTHandler(node),
                ReductionHandler(requireF32Reduction, {}))
    {
    }

    NodeSet createSlicedNode();

protected:
};

}  // namespace gc::layered_brain
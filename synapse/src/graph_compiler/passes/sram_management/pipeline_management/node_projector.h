#pragma once

#include "node_solver.h"
#include "bundle_solver.h"
#include "sram_management/bundle.h"
#include "sliced_dims_projector.h"

using namespace gc::access_pattern;

class NodeSolutionProjector
{
public:
    explicit NodeSolutionProjector(const NodePtr& node) : m_node(node) {}
    virtual ~NodeSolutionProjector() = default;

    // Given a bundle strategy with 1 or more sliced operands of the projected node, objects implementing this interface
    // need to generate a node strategy based on the slicing of those operands.
    virtual NodeStrategyPtr getNodeStrategy(const BundleStrategyPtr& bundleStrategy,
                                            const TensorPtr&         sliceDefiningTensor) const = 0;

protected:
    using SlicedOperands = std::vector<pSlicedOperand>;

    NodePtr m_node;

    SlicedOperands findSlicedOperandsInBundleStrategy(const BundleStrategyPtr& bundleStrategy) const;
};

// This class implements the following algorithm:
//
// Inputs
//   A node with access pattern support
//   Bundle strategy containing slicing for some of the node's operands.
//
// Output
//   Node strategy containing the slicing for all the node operands.
//
// Procedure
//   1. Project the sliced tensors on the index space geometry, producing the node tile of each slice.
//      (This assumes that the size of all the slices maps to a fixed index space geometry slice size, except maybe
//      the last slice in each dimension)
//   2. Project the index space geometry slice back on each operand to get it's slice size
class AccessPatternNodeSolutionProjector : public NodeSolutionProjector
{
public:
    explicit AccessPatternNodeSolutionProjector(const NodePtr& node);
    virtual ~AccessPatternNodeSolutionProjector() = default;

    NodeStrategyPtr getNodeStrategy(const SlicedOperands& slicedOperands, const TensorPtr& sliceDefiningTensor) const;
    NodeStrategyPtr getNodeStrategy(const BundleStrategyPtr& bundleStrategy,
                                    const TensorPtr&         sliceDefiningTensor) const override;

private:
    using SlicedOperands = std::vector<pSlicedOperand>;

    NodeTile        getNodeTileFromSlicedOperands(const SlicedOperands& slicedOperands,
                                                  const TensorPtr&      sliceDefiningTensor) const;
    NodeStrategyPtr createStrategyFromNodeTile(const NodeTile& nodeTile, const SlicedOperands& slicedOperands) const;
    void            projectSlicingOnTensor(pSlicedOperand              tensorSlicedOperand,
                                           const NodeAccessPatternPtr& nodeAP,
                                           const NodeTile&             nodeTile,
                                           const SlicedDimsProjector&  slicedDimsProjector) const;
};

class SharedMMENodeSolutionProjector : public NodeSolutionProjector
{
public:
    explicit SharedMMENodeSolutionProjector(const NodePtr& node) : NodeSolutionProjector(node) {}
    virtual ~SharedMMENodeSolutionProjector() = default;

    NodeStrategyPtr getNodeStrategy(const BundleStrategyPtr& bundleStrategy,
                                    const TensorPtr&         sliceDefiningTensor) const override;
};

#pragma once

#include "access_pattern.h"
#include "bundle_view.h"
#include "habana_nodes.h"
#include "passes/sram_management/pipeline_management/node_solver.h"
#include "strategy.h"
#include "tensor.h"
#include "node_solver.h"
#include "bundle.h"
#include "types.h"
#include "node_factory.h"

class MmeHugeTensorSplitter
{
public:
    explicit MmeHugeTensorSplitter(const NodePtr& m_nodeToSplit);
    virtual ~MmeHugeTensorSplitter() = default;

    bool         doesMmeNodeRequireSlicing() const;
    virtual bool doesNodeSupportSlicing() const;
    NodeList     splitNodeWithHugeTensor() const;

protected:
    virtual bool                   sliceNodeForFunctionality(SlicingStrategyPtr& strategy) const;
    SlicingStrategyPtr             createInitialStrategy() const;
    NodeList                       sliceHugeNode(const NodeTile& nodeTile) const;
    bool                           isSliceSizeFitsHw(const pSlicedOperand& operand) const;
    NodeTile                       getNodeTileFromStrategy(const SlicingStrategyPtr& strategy) const;
    gc::layered_brain::StrategyPtr getBvdNodeStrategy(const BundleViewContainerPtr& bundleViews,
                                                      const NodeTile&               nodeTile) const;
    BundleViewContainerPtr         createBundleViews() const;
    void                           handleBiasedNodesWithReduction(NodeSet& slicedNodes) const;

    const NodePtr    m_nodeToSplit;
    MmeDimController m_dimController;
};

class GemmHugeTensorSplitter : public MmeHugeTensorSplitter
{
public:
    GemmHugeTensorSplitter(const NodePtr& m_nodeToSplit);
    virtual bool sliceNodeForFunctionality(SlicingStrategyPtr& strategy) const override;
    bool         doesNodeSupportSlicing() const override;
    virtual void projectSingleDimSlice(SlicingStrategyPtr& strategy, const pSlicedOperand& slicedOperand) const;
};

class BgemmHugeTensorSplitter : public GemmHugeTensorSplitter
{
public:
    BgemmHugeTensorSplitter(const NodePtr& m_nodeToSplit);
    bool doesNodeSupportSlicing() const override;
};
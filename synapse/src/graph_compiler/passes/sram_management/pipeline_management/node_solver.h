#pragma once

#include "conv_base_node.h"
#include "sram_management/slicing_strategy.h"
#include "sram_management/pipeline_management/common_tile_size_calculator.h"

using NodeStrategyPtr      = SlicingStrategyPtr;
using GranularityPerTensor = TileSizePerTensor;
using TensorGranularity    = TileSizePerTensor::mapped_type;

// Single node solver constraints, which are derived from the other nodes in the bundle it belongs to
struct BundleSolutionConstraints
{
    GranularityPerTensor slicingGranularityPerTensor;
    unsigned             availableSramBytes = 0;      // The SRAM capacity allocated for this node
    bool      canSliceMultipleDims          = false;  // Can assume no granularity constraint and slice on filtered dim
    bool      canAlignInputToCL             = false;  // The input tensor in SRAM can be aligned to CL
    TensorSet tensorsInSram;
    std::map<Dim, TSize> sharedOperandSlicingDimsAlignment;  // Required alignment for bundled MME nodes utilization
    bool                 sliceNonMasterOperand = false;      // Slice the non-master operand on different BVD
    std::vector<Dim>     sharedOperandSlicingDims;           // Bundlizer selected shared operand slicing dims
    bool                 isSingleNodeBundle;
};

class NodeSolver
{
public:
    explicit NodeSolver(const NodePtr& node) : m_node(node) {}
    virtual ~NodeSolver() = default;

    virtual NodeStrategyPtr solveNode(const BundleSolutionConstraints& constraints) = 0;
    // Gets the matching slicing dim based on the given tensor and slicing dim, for each given dim.
    // If there's no 1:1 mapping on all given dims - returns an empty vector
    static std::vector<Dim> getTensorMatchingSlicedDims(const NodePtr&          node,
                                                        const TensorPtr&        givenTensor,
                                                        const std::vector<Dim>& givenSlicingDims,
                                                        const TensorPtr&        queriedTensor);
    static bool             doesNodeSupportSlicing(const NodePtr& node);
    static bool             isInputSliceable(const NodePtr& node, unsigned inputIdx);
    static std::vector<Dim> getInputSlicingDims(const NodePtr& node, unsigned inputIdx);
    static NodeStrategyPtr  solveNode(const NodePtr& mmeNode, const BundleSolutionConstraints& constraints);
    static bool             canSliceNodeOnMultipleDims(const NodePtr&          mmeNode,
                                                       const std::vector<Dim>& sharedOperandSlicingDims,
                                                       const bool              isSingleNodeBundle);

protected:
    const NodePtr&     m_node;
};

class MmeNodeSolver : public NodeSolver
{
public:
    explicit MmeNodeSolver(const NodePtr& node)
    : NodeSolver(node), m_dimController(node)
    {
    }

    NodeStrategyPtr          solveNode(const BundleSolutionConstraints& constraints) override;
    virtual bool             isNodeSupported();
    virtual std::vector<Dim> getInputSlicingDims(unsigned inputIdx)                 = 0;
    virtual bool             canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                        const bool              isSingleNodeBundle);
    bool                     isDuplicateInputMme();

    TSize calcMinSlicedDimSizeInElements(const TensorPtr& slicedInput, unsigned slicingDim, const TSize dimGranularity);

    static std::unique_ptr<MmeNodeSolver> getSpecificMmeSolver(const NodePtr& node);
    static TSize                          getMinSlicedDimSizeInElements(const NodePtr&   node,
                                                                        const TensorPtr& slicedInput,
                                                                        unsigned         slicingDim,
                                                                        const TSize      dimGranularity);

    static unsigned getInputIndexToSlice(const TensorSet& tensorsInSram, const NodePtr& node);
    static TSize    alignDimSizeToGranularity(TSize dimSize, TSize granularity);

protected:
    virtual NodeStrategyPtr createInitialStrategy();
    void                    finalizeStrategy(NodeStrategyPtr& strategy);

    void validateStrategy(const SlicingStrategyPtr& strategy);
    void validateSlicesSize(const SlicingStrategyPtr& strategy);

    static std::vector<unsigned> getInputsIdicesInSram(const TensorSet& tensorsInSram, const NodePtr& node);
    bool                         isOutputInSram(const TensorSet& tensorsInSram);

    TensorGranularity getSlicingGranularity(unsigned slicedInputIdx, const GranularityPerTensor& slicingGranularityMap);
    // Calculates the sliced dim granularity, taking MME util and the given granularity into account.
    // If outSliceDim is valid, calculates for this output dim, otherwise calc for the input common dim
    TSize getDimAlignmentInElements(unsigned inputIdx, std::optional<Dim> outSliceDim, TSize dimGranularity);
    // Returns the output dim size granularity for best MME utilization
    virtual TSize getOutDimAlignmentForMmeUtil(unsigned slicedInputIdx, unsigned outputSliceDim);
    // Returns the MME geometry axis size in elements, which corresponds to the given input index.
    // The geometry is selected based on the MME node recommended configuration
    virtual unsigned getMmeGeometrySizeInElementsToAlign(unsigned slicedInputIdx, Dim slicedDim);
    // Checks if slicing the oprand on the given slicing dimension to any size will surely reduce utilization
    virtual bool
         slicingReducesUtilization(SlicingStrategyPtr& strategy, unsigned outputSliceDim, unsigned slicedInputIdx);
    bool         sliceNodeOnSingleNonCommonDim(SlicingStrategyPtr&              strategy,
                                               Dim                              inputSliceDim,
                                               const BundleSolutionConstraints& constraints);
    bool sliceNodeOnSingleCommonDim(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints);
    virtual bool sliceNodeOnSingleDim(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints);
    virtual bool sliceNodeOnMultipleDims(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints);
    virtual void sliceNonMasterOperand(SlicingStrategyPtr&      strategy,
                                       unsigned                 nonMasterInputIdx,
                                       const TensorGranularity& nonMasterGranularity) {};

    bool reducePerformanceToFitSram(NodeStrategyPtr& strategy, unsigned availableSram);
    bool retryWithoutCLAlign(NodeStrategyPtr& strategy, unsigned availableSram);
    bool retryWithoutOutputInSram(NodeStrategyPtr& strategy, unsigned availableSram);
    bool retryWithoutDoubleBuffer(NodeStrategyPtr& strategy, unsigned availableSram);
    bool sramPlacementIsHighPrio(const NodeStrategyPtr& strategy, const BundleSolutionConstraints& constraints) const;

    void resetStrategySlicingData(SlicingStrategyPtr& strategy);

    virtual TSize getOperandAxisElements(const pSlicedOperand& operand, unsigned slicedInputIdx, unsigned slicedDim);
    virtual TSize
    getCommonDimMinSliceSize(const pSlicedOperand& slicedOperandA, Dim slicingDim, TSize slicingDimGranularity) const;

    virtual void setTraversalPatternAndMapping(StrategySlicingData& data);
    virtual void projectSlicingOnNodeOperands(NodeStrategyPtr& strategy, unsigned slicedInputIdx) {}
    virtual bool singleBufferStrategyAllowed() { return true; }

    virtual TSize getMmeMinCDInElementsForPartials() const;

    MmeDimController m_dimController;
};

class ConvNodeSolver : public MmeNodeSolver
{
public:
    explicit ConvNodeSolver(const NodePtr& node)
    : MmeNodeSolver(node), m_conv(std::dynamic_pointer_cast<ConvBaseNode>(m_node))
    {
    }
    std::vector<Dim> getInputSlicingDims(unsigned inputIdx) override;
    bool             canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                const bool              isSingleNodeBundle) override;
    TSize            getOutDimAlignmentForMmeUtil(unsigned slicedInputIdx, unsigned outputSliceDim) override;
    TSize getOperandAxisElements(const pSlicedOperand& operand, unsigned slicedInputIdx, unsigned slicedDim) override;

    bool sliceNodeOnMultipleDims(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints) override;

protected:
    void setTraversalPatternAndMapping(StrategySlicingData& data) override;

    std::tuple<bool, int, int> sliceNonCommonSpatialDim(unsigned        slicingDim,
                                                        TSize           outputSliceSize,
                                                        pSlicedOperand& inputOperand,
                                                        pSlicedOperand& outputOperand);
    TSize         getInputSliceSize(const SizeArray& outputSliceSizes, unsigned rank, unsigned slicedDim) const;
    virtual TSize getSpatialDimGranularity(unsigned slicingDim) const;
    // Returns the middle slices padding values - before and after, to be updated for the sliced nodes
    virtual std::tuple<int, int> setSlicedOperandsPadding(unsigned        slicingDim,
                                                          int             overlap,
                                                          TSize           outputSliceSize,
                                                          pSlicedOperand& inputOperand,
                                                          pSlicedOperand& outputOperand) const;

    std::shared_ptr<ConvBaseNode> m_conv;
};

class DedxNodeSolver : public ConvNodeSolver
{
public:
    explicit DedxNodeSolver(const NodePtr& node) : ConvNodeSolver(node) {}

protected:
    TSize                getSpatialDimGranularity(unsigned slicingDim) const override;
    std::tuple<int, int> setSlicedOperandsPadding(unsigned        slicingDim,
                                                  int             overlap,
                                                  TSize           outputSliceSize,
                                                  pSlicedOperand& inputOperand,
                                                  pSlicedOperand& outputOperand) const override;
    void                 projectSlicingOnNodeOperands(NodeStrategyPtr& strategy, unsigned slicedInputIdx) override;
};

class DedwNodeSolver : public MmeNodeSolver
{
public:
    explicit DedwNodeSolver(const NodePtr& node)
    : MmeNodeSolver(node), m_conv(std::dynamic_pointer_cast<ConvBaseNode>(m_node))
    {
    }
    std::vector<Dim> getInputSlicingDims(unsigned inputIdx) override;
    bool             canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                const bool              isSingleNodeBundle) override;

protected:
    void  setTraversalPatternAndMapping(StrategySlicingData& data) override;
    bool  sliceNodeOnSingleDim(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints) override;
    bool  sliceNodeOnMultipleDims(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints) override;
    TSize getCommonDimMinSliceSize(const pSlicedOperand& slicedOperandA,
                                   Dim                   slicingDim,
                                   TSize                 slicingDimGranularity) const override;
    bool  canSliceSpatially(const pSlicedOperand& inputOperand);
    TSize getMinSpatialDimSize(const pSlicedOperand& dyOperand, Dim spatialSliceDim);
    void
    sliceCommonSpatialDim(unsigned slicingDim, TSize dySliceSize, pSlicedOperand& dyOperand, pSlicedOperand& xOperand);

    bool singleBufferStrategyAllowed() override { return false; }

    std::shared_ptr<ConvBaseNode> m_conv;
};

class GemmNodeSolver : public MmeNodeSolver
{
public:
    explicit GemmNodeSolver(const NodePtr& node) : MmeNodeSolver(node) {}

    bool             isNodeSupported() override;
    std::vector<Dim> getInputSlicingDims(unsigned inputIdx) override;

protected:
    void setTraversalPatternAndMapping(StrategySlicingData& data) override;
    void sliceNonMasterOperand(SlicingStrategyPtr&      strategy,
                               unsigned                 nonMasterInputIdx,
                               const TensorGranularity& nonMasterGranularity) override;
};

class BatchGemmNodeSolver : public MmeNodeSolver
{
public:
    explicit BatchGemmNodeSolver(const NodePtr& node) : MmeNodeSolver(node) {}

    bool             isNodeSupported() override;
    std::vector<Dim> getInputSlicingDims(unsigned inputIdx) override;
    bool             canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                const bool              isSingleNodeBundle) override;

protected:
    bool         isOperandsLayoutSupported();
    void         setTraversalPatternAndMapping(StrategySlicingData& data) override;

    unsigned getMmeGeometrySizeInElementsToAlign(unsigned slicedInputIdx, Dim slicedDim) override;
    bool     sliceNodeOnMultipleDims(NodeStrategyPtr& strategy, const BundleSolutionConstraints& constraints) override;
    void     projectSlicingOnNodeOperands(NodeStrategyPtr& strategy, unsigned slicedInputIdx) override;

    std::vector<Dim> clearLeadingOnesAndReverseBatchDims(const SizeArray& sizes, const DimVector& batchDim) const;
};

class MaskedBatchGemmNodeSolver : public BatchGemmNodeSolver
{
public:
    explicit MaskedBatchGemmNodeSolver(const NodePtr& node) : BatchGemmNodeSolver(node) {}

    bool isNodeSupported() override;

};

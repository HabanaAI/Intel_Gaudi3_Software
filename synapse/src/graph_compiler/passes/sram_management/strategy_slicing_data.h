#pragma once

#include "bundle.h"
#include "node.h"
#include "slice_mapping.h"
#include "types.h"

class StrategySlicingData;
using StrategySlicingDataPtr = std::shared_ptr<StrategySlicingData>;

class StrategySlicingData
{
public:
    using slicedOperandAndDim  = std::pair<pSlicedOperand, unsigned>;
    using slicedOperandAndDimList = std::list<slicedOperandAndDim>;
    enum class WalkingDir {LeftToRight, TopToBottom};

    StrategySlicingData(const TensorVector& inputTensors,
                        const pTensor& outputTensor);
    StrategySlicingData(const StrategySlicingData& other);

    virtual ~StrategySlicingData();

    virtual StrategySlicingDataPtr clone() const;

    virtual bool compareInitialSlicing(const StrategySlicingData& other, bool exactMatch = true) const;

    std::vector<pSlicedOperand> getSlicedOperands() const;
    pSlicedOperand            getSlicedOperand(const TensorPtr& tensor) const;
    // return list of an actually sliced (non trivially) operand+dim
    slicedOperandAndDimList getSlicedOperandsAndDims() const;

    SliceReferenceList getInputsForSlice(const SliceRefCommonDimIdxPair& slice) const;
    SliceReferenceList getInputsForSlice(const pSliceReference& slice) const;
    SliceReferenceList getOutputsForSlice(const pSliceReference& outputSlice) const;

    virtual SlicedOperandTraversalPattern getOutputSlices() const;

    std::list<std::pair<SliceReferenceList, SliceReferenceList>> getFwdMappedSlices(const pSliceReference& slice) const;

    std::pair<std::list<pSlicedOperand>, std::list<pSlicedOperand>>
    getFwdMappedSlicedOperands(const pSlicedOperand& slicedOperand) const;
    std::pair<std::list<pSlicedOperand>, std::list<pSlicedOperand>>
    getBwdMappedSlicedOperands(const pSlicedOperand& slicedOperand) const;

    virtual StrategySlicingData::WalkingDir getWalkingDir() const;

    // Gather all the bundle and valid candidates nodes
    virtual NodeSet getStrategyNodes(const pBundle& bundle) const;

    std::vector<pSlicedOperand> addNodeOperandsToStrategy(const TensorVector&        nodeOperands,
                                                          const StrategySlicingData& nodeSlicingData,
                                                          const pSlicedOperand&      stitchedOperand);

    ///////////////////////////////////////////////////////////////////////////////////
    // A Note about mapping operands:
    //
    // The master output is mapped to the input that generate it,
    // Which in turn are mapped to inputs of bundle producers which generate them.
    // This is called "Backward mapping".
    //
    // The master inputs and master output may be mapped to *other* consumers (not
    // the master node), that should be scheduled once those inputs or output is
    // generated.
    // This is called "Forward mapping"
    //
    // Note that the master node (for example, the MME node in a single MME with TPC
    // consumer/producer/both bundle) has *only* "backward mapping" from the output to
    // the inputs. Those inputs may be forward mapped only to *other* nodes (for example
    // DeDw in a DeDx+DeDw bundle where the DeDx is the master node).
    /////////////////////////////////////////////////////////////////////////////////////

    // Map output slice to inputs
    void setOutputSliceBackwardMapping(pBackwardSliceMapping mapping);
    // Map input slice to other inputs
    void setOperandSliceBackwardMapping(pSlicedOperand operand, pBackwardSliceMapping mapping);
    // Map an operand to inputs and outputs of a dependant operation
    void setOperandSliceForwardMapping(pSlicedOperand operand, pForwardSliceMapping mapping);
    // Add mapping for an operand to inputs and outputs of a dependant operation
    void addOperandSliceForwardMapping(pSlicedOperand operand, pForwardSliceMapping mapping);

    void updateNumOfOperandBuffers(bool doubleBuffer);

    std::list<SlicedOperandTraversalPattern>& getSlavesPatterns();

    void addSlaveTraversalPattern(const pSlicedOperand& operand);
    bool isSnakeWalkingPatternEnabled() const;
    void setSnakeWalkingPattern(bool enable);

    std::vector<pSlicedOperand>            bundleTensors;
    pSlicedOperand                         masterOperand;
    bool                                   enableGraphSizeOptimization = true; // disable operand chunk sizes modifications for dependent operands (like convolution with filter)
    DimVector                              traversalPattern = SlicedOperandTraversalPattern::LEFT_TO_RIGHT_2D;
    unsigned                               numCommonDimSlices = 1;

protected:
    bool m_snakeWalkingPatternEnabled = true;
    std::list<SlicedOperandTraversalPattern>   m_slaveTraversalPatterns;

private:
    std::unordered_map<Bundle::Solution::pSlicedOperand, pBackwardSliceMapping> m_operandBackwardMappings;
    std::unordered_map<Bundle::Solution::pSlicedOperand, std::vector<pForwardSliceMapping>> m_operandForwardMappings;
};

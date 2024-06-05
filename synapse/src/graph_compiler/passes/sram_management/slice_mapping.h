#pragma once

#include "types.h"
#include "bundle.h"
#include "mme_dim_controller.h"
#include "sliced_operand_traversal.h"

class BackwardSliceMapping;
using pBackwardSliceMapping = std::shared_ptr<BackwardSliceMapping>;
using SliceReferenceList = std::list<pSliceReference>;

class ForwardSliceMapping;
using pForwardSliceMapping = std::shared_ptr<ForwardSliceMapping>;

// Interface for output slice to input slices mapping
class BackwardSliceMapping
{
public:
    BackwardSliceMapping(const std::vector<pSlicedOperand>& inOperands, const pSlicedOperand& outOperand)
    : m_inOperands(inOperands)
    , m_outOperand(outOperand)
    {}

    virtual ~BackwardSliceMapping() {}

    // Returns a list of SlicesReferences {referenceA, referenceB}
    virtual SliceReferenceList getInputs(const SliceRefCommonDimIdxPair& slice) const = 0;

    // Return a list of all output slices that match the given output slice.
    virtual SliceReferenceList getOutputs(const pSliceReference& outputSlice) const
    {
        // TODO SW-67410 - temporary implementation. This should be moved to the concrete implementations and be as
        // accurate as possible
        return {outputSlice};
    }

    virtual pBackwardSliceMapping clone(std::vector<pSlicedOperand>& inputOperands, pSlicedOperand& output) = 0;

    const std::vector<pSlicedOperand>& getInOperands() const { return m_inOperands; }
    const pSlicedOperand& getOutOperand() const { return m_outOperand; }

protected:
    std::vector<pSlicedOperand> m_inOperands;
    pSlicedOperand m_outOperand;
};

class ForwardSliceMapping
{
public:
    virtual ~ForwardSliceMapping() {}

    // Given any input or output slice, this method returns a complete set of inputs and outputs
    // that comprise an activation of the mapped outputs producer. It is assumed there are no
    // partials or inputs reuse for this producer node.
    virtual std::list<std::pair<SliceReferenceList, SliceReferenceList>> getInputsAndOutputs(const pSliceReference& input) = 0;

    virtual pForwardSliceMapping clone(const std::list<pSlicedOperand>& inputs,
                                       const std::list<pSlicedOperand>& outputs) = 0;

    virtual const std::list<pSlicedOperand>& getInputs() const = 0;
    virtual const std::list<pSlicedOperand>& getOutputs() const = 0;
};

// Static logic to create slice mapping for MME nodes
class MMESliceMapper
{
public:
    static pBackwardSliceMapping mapOutputToInputs(pNode mmeNode,
                                                   pSlicedOperand inputA,
                                                   pSlicedOperand inputB,
                                                   pSlicedOperand output,
                                                   pSlicedOperand inputShapeTensor);
    static pBackwardSliceMapping clone(const pBackwardSliceMapping& other, std::vector<pSlicedOperand> inputOperands, pSlicedOperand outputOperand)
    {
        return other->clone(inputOperands, outputOperand);
    }
};

// Static logic to create slice mapping for MME nodes solving non-batched part of a BatchGemm
class MMETriviallyBatchedSliceMapper
{
public:
    static pBackwardSliceMapping mapOutputToInputs(pNode mmeNode,
                                                   pSlicedOperand inputA,
                                                   pSlicedOperand inputB,
                                                   pSlicedOperand output);
    static pBackwardSliceMapping clone(const pBackwardSliceMapping& other, std::vector<pSlicedOperand> inputOperands, pSlicedOperand outputOperand)
    {
        return other->clone(inputOperands, outputOperand);
    }
};

// Static logic to create slice mapping for TPC nodes
class TPCSliceMapper
{
public:
    static pBackwardSliceMapping mapOutputToInputs(pNode tpcNode,
                                                   std::list<pSlicedOperand> inputs,
                                                   pSlicedOperand output);
};


// Static logic to create slice mapping for trivially sliced node
class TrivialSliceMapper
{
public:
    static pBackwardSliceMapping mapOutputToInputs(pNode node,
                                                   std::list<pSlicedOperand> inputs,
                                                   pSlicedOperand output);
    static pForwardSliceMapping mapSlicedOperandForward(const std::list<pSlicedOperand>& allInputs,
                                                        const std::list<pSlicedOperand>& allOutputs);
};

// Static logic to create slice mapping for reshape nodes
class ReshapeSliceMapper
{
public:
    static pBackwardSliceMapping mapOutputToInput(pSlicedOperand input,
                                                  pSlicedOperand output);
};

class AccessPatternSliceMapper
{
public:
    static pBackwardSliceMapping createBwdMapping(NodePtr                            node,
                                                  const std::vector<pSlicedOperand>& allInputs,
                                                  const std::vector<pSlicedOperand>& allOutputs);

    static pForwardSliceMapping createFwdMapping(NodePtr                          node,
                                                 const std::list<pSlicedOperand>& allInputs,
                                                 const std::list<pSlicedOperand>& allOutputs);
};
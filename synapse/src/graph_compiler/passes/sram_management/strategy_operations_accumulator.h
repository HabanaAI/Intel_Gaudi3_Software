#pragma once

#include "slicing_strategy.h"
#include "habana_graph.h"
#include "sliced_operand_traversal.h"
#include <unordered_set>
#include "slicing_utils.h"

class OperationHandler
{
public:
    virtual void handleOperation(const pNode& node,
                                 const SliceReferenceList& inputs,
                                 const SliceReferenceList& outputs) = 0;

};

// Invokes a handler on each operation that will be used to execute the bundle computation, in the order of execution,
// according to the slicing strategy.
class HandleEachStrategyOperation
{
public:
    HandleEachStrategyOperation(const HabanaGraph& graph,
                                unsigned           bundleIdx,
                                const NodeVector&  bundleNodes,
                                SlicingStrategyPtr strategy,
                                bool               traceLog = false);

    // Handles operations in execution order.
    bool operator()(OperationHandler& operationHandler);

protected:
    const HabanaGraph&    m_graph;
    const SlicingStrategyPtr m_strategy;
    const uint32_t           m_bundleIndex;  // For logging
    OperationHandler*     m_operationHandler = nullptr;

    MultiOperandSliceIterator       m_outputGenerationIterator;
    const MultiOperandSliceIterator m_iterationsEnd;

    // a set that holds all the sliced references that we already created the TPC operations in order to generate them.
    std::unordered_set<pSliceReference, SliceReference::Hasher, SliceReference::IsEqual> m_intermediateSliceRefSet;
    // this enum is relevant to support stitching of wide operand when both input tensors are sliced on multiple
    // dimensions (in  all other cases it doesn't affect the flow)
    enum class generationState
    {
        normal, // behaviour as usual algorithm
        stitchingOnMultiDimsEncountered, // identify multiple dims slicing - precondition before createMultipleInputOps
        createMultipleInputOps // create multiple input (TPC) operations to support double buffer - same as what
        // happens at the beginning of the normal algorithm
    };
    std::unordered_map<pSlicedOperand, generationState>                                  m_operandGenerationState;
    using InputSwapPointList = std::list<OperandSliceIterator>;

    // Swap point: is an output slice. When a swapping point is reached, we should schedule an input generation before
    // the generation of the swapping point itself.
    // Mapping input operand to it's swapping points in the output
    std::unordered_map<pSlicedOperand, InputSwapPointList> m_operandInputsSwapPoints;

    // Mapping input operand to an operand slice queue which holds input slices
    // according to generation order.
    std::unordered_map<pSlicedOperand, SliceReferenceList> m_operandSlicesGenerationQueue;

    // control trace logging since this objects may be used many times for strategy compare and it may blow up the log.
    const bool m_traceLog;

    // Tensor information that is used multiple times in all the operations that refer to a the tensor,
    // is pre-mapped to prevent the same calculations from happening every time.
    class TensorInfo
    {
    public:
        explicit TensorInfo(pSlicedOperand op) : operand(std::move(op))
        {
            for (uint32_t dim = 1 /*skip fcd*/; dim < operand->originalTensor->getDim(); dim++)
            {
                if (SlicedOperandUtils::isSlicedOnDimension(operand, dim))
                {
                    slicedOnFcdOnly = false;
                    break;
                }
            }
        }

        TensorInfo(const TensorInfo&) = default;

        const pSlicedOperand operand;

        // Operands that are not sliced on any dimension except the first
        bool slicedOnFcdOnly = true;

        // Is the tensor produced by a bundle node
        bool intermediate = false;
    };
    // Tensor info per tensor
    std::unordered_map<pTensor, TensorInfo>  m_slicedOperandByTensor;
    void                                     mapStrategyOperands(const NodeVector& bundleNodes);
    TensorInfo getTensorInfo(const pSlicedOperand& slicedOp, const NodeVector& bundleNodes) const;

    // Create the master node's input generation queue and swapping points in order to know when to
    // schedule an input generation operation (if the input is generated in the bundle).
    void initInputGenerationPoints();

    // Schedule pre-creation of inputs in order to hide input generation using double (or more) buffer (if available)
    void generateInputLookahead();
    void generateInputLookahead(const pSlicedOperand& operand);

    // Is the solution ready
    bool done() const;

    // Create operations to generate the next set of inputs for a master output slice
    // and their forward mapped slices if any.
    void generateNextInputs();

    // Create operations to generate the next master output slice and its forward mapped slices if any.
    virtual void generateNextOutputAndFwdMapped();

    // When stitching the wide tensor in case there is double buffer and the input tensors are sliced on multiple
    // dimensions, we want the operations for each row to look like the following pattern:
    // tpc, [multiple times]*{tpc, mme}, [multiple times]*{mme}
    // this function helps doing that.
    // Returns whether the calling function should generate the next input
    bool handleMultipleDimensionWideStitching(const pSlicedOperand& operand);

    // Create inputs for the last node in the bundle, taking care to re-use previously generated
    // inputs and generate new inputs using bundle intermediate nodes. returns true upon success.
    void generateFinalInputs(const SliceReferenceList& inputList);

    // If a bundle intermediate node needs generation, adds the operations to generate it (recursively if necessary).
    void generateIntermediateInput(const pSliceReference& slice);

    // Is the specified tensor produced by a bundle node.
    bool isIntermediateTensor(const pTensor& tensor) const;

    // print log about creation of an operation from the given inputs to the given output
    void logOperation(const SliceReferenceList& inputs,
                      const pSliceReference&    output,
                      const std::string&        operationName) const;

    // Adds an operation to the execution order which computes the output from the inputs.
    void handleOperation(const SliceReferenceList& inputs, const SliceReferenceList& outputs);

    // Adds operations for slices of operands which are forward mapped in the strategy
    virtual void addFwdMappedOperation(const SliceReferenceList& generatedSlices);

    // In case the original node has more than 1 output, this generate a list with the given
    // output slice and all the other outputs as slice references with coordinate <0, 0, 0, 0, 0>.
    SliceReferenceList generateAllOutputReferences(const pSliceReference& outputSlice) const;

    // checks if this is the last slice in the partials dim
    bool isLastSliceOnPartialDim(MultiOperandSliceIterator& outputIterator) const;
};

class HandleEachStrategyOperationMantaRay : public HandleEachStrategyOperation
{
public:
    HandleEachStrategyOperationMantaRay(const HabanaGraph& graph,
                                        unsigned           bundleIdx,
                                        const NodeVector&  bundleNodes,
                                        SlicingStrategyPtr strategy,
                                        bool               traceLog = false)
    : HandleEachStrategyOperation(graph, bundleIdx, bundleNodes, strategy, traceLog)
    {
    }

protected:
    void generateNextOutputAndFwdMapped() override;
    void addFwdMappedOperation(const SliceReferenceList& generatedSlices) override;
};

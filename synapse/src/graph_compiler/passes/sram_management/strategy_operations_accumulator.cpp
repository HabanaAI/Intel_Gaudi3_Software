#include "strategy_operations_accumulator.h"

#include <utility>
#include "habana_graph.h"
#include "slicing_utils.h"

#define SOL_GEN_TRACE(...) LOG_TRACE(SRAM_SOL_GEN, __VA_ARGS__)
#define SOL_GEN_DEBUG(...) LOG_DEBUG(SRAM_SOL_GEN, __VA_ARGS__)

HandleEachStrategyOperation::HandleEachStrategyOperation(const HabanaGraph& graph,
                                                         unsigned           bundleIdx,
                                                         const NodeVector&  bundleNodes,
                                                         SlicingStrategyPtr strategy,
                                                         bool               traceLog)
: m_graph(graph),
  m_strategy(std::move(strategy)),
  m_bundleIndex(bundleIdx),
  m_outputGenerationIterator(m_strategy->getSlicingData().getOutputSlices().begin()),
  m_iterationsEnd(m_strategy->getSlicingData().getOutputSlices().end()),
  m_traceLog(traceLog)
{
    mapStrategyOperands(bundleNodes);
}

void HandleEachStrategyOperation::mapStrategyOperands(const NodeVector& bundleNodes)
{
    for (const pSlicedOperand& slicedOp : m_strategy->getSlicingData().getSlicedOperands())
    {
        m_slicedOperandByTensor.emplace(slicedOp->originalTensor, getTensorInfo(slicedOp, bundleNodes));
    }
}

HandleEachStrategyOperation::TensorInfo HandleEachStrategyOperation::getTensorInfo(const pSlicedOperand& slicedOp,
                                                                                   const NodeVector& bundleNodes) const
{
    TensorInfo info(slicedOp);
    const pNode& producer = m_graph.getTensorProducer(slicedOp->originalTensor);
    info.intermediate = (std::find(bundleNodes.begin(), bundleNodes.end(), producer) != bundleNodes.end());
    return info;
}


bool HandleEachStrategyOperation::operator()(OperationHandler& operationHandler)
{
    SOL_GEN_DEBUG("Handling all operations for bundle {} according to strategy - ", m_bundleIndex);
    m_strategy->printLog(1, synapse::LogManager::LogType::SRAM_SOL_GEN);

    m_operationHandler = &operationHandler;

    /* Pre-traversing of the outputs in order to predict input generation */
    initInputGenerationPoints();

    /* Schedule the pre-creation of the first input slice. Only in case of double buffer.
     * In more than double, More than one slice will be pre-created.
     * In single buffer, there is nothing to do */
    generateInputLookahead();

    while (!done())
    {
        generateNextInputs();
        generateNextOutputAndFwdMapped();
    }

    return true;
}

void HandleEachStrategyOperation::initInputGenerationPoints()
{
    const StrategySlicingData& slicingData            = m_strategy->getSlicingData();

    /* Iterate over outputs */
    for (auto iter = m_outputGenerationIterator; iter != m_iterationsEnd; ++iter)
    {
        /* Get the input operands that required to generate current output */
        const SliceReferenceList& inputTuple = slicingData.getInputsForSlice(*iter);

        for (const pSliceReference& inputSlice : inputTuple)
        {
            /* We wish to filter inputs that are produced by a node from the bundle */
            if (!isIntermediateTensor(inputSlice->operand->originalTensor)) continue;

            /* If the following condition fulfilled, it is not a swapping point
             * current input is already pushed to the slices generation q */
            if (!m_operandSlicesGenerationQueue[inputSlice->operand].empty() &&
                inputSlice->coordinates == m_operandSlicesGenerationQueue[inputSlice->operand].back()->coordinates)
            {
                continue;
            }

            /* Each operand is being mapped to a generation queue.*/
            m_operandSlicesGenerationQueue[inputSlice->operand].emplace_back(inputSlice);

            /* Each operand is being mapped to a queue that holds its swapping points.
             * When swapping point is reached, the input that should be generated is the input at the head of
             * the generation queue*/
            m_operandInputsSwapPoints[inputSlice->operand].emplace_back(iter.getCurrentIterator());
            m_operandGenerationState[inputSlice->operand] = generationState::normal;
        }
    }
}

void HandleEachStrategyOperation::generateInputLookahead()
{
    for (const auto& operandAndGenerationQueue : m_operandSlicesGenerationQueue)
    {
        generateInputLookahead(operandAndGenerationQueue.first);
    }
}

void HandleEachStrategyOperation::generateInputLookahead(const pSlicedOperand& operand)
{
    uint32_t numLookAheads = SlicedOperandUtils::isTriviallySliced(operand)? 1 : operand->numOfBuffers - 1; //catch corner case of single-buffer-single-slice
    for (uint32_t i = 0; i < numLookAheads; i++)
    {
        SliceReferenceList& generationQueue = m_operandSlicesGenerationQueue[operand];
        if (generationQueue.empty()) break; // nothing more to do for this operand
        generateFinalInputs({generationQueue.front()});
        generationQueue.pop_front();
    }
}


bool HandleEachStrategyOperation::done() const
{
    return m_outputGenerationIterator == m_iterationsEnd;
}

bool HandleEachStrategyOperation::handleMultipleDimensionWideStitching(const pSlicedOperand& operand)
{
    bool shouldGenerateNextInput = true;
    const SliceReferenceList& generationQueue         = m_operandSlicesGenerationQueue[operand];
    if (operand->numOfBuffers > 1 && !generationQueue.empty())
    {
        switch (m_operandGenerationState[operand])
        {
            case generationState::normal:
                if (m_intermediateSliceRefSet.find(generationQueue.front()) != m_intermediateSliceRefSet.end())
                {
                    // we reached the point where we try to create a (tpc) operation for a slice reference that we
                    // already created the (tpc) operation in order to generate it before - this happens when stitching
                    // a tensor that was sliced on multiple dimensions (it can also happen when stitching to a narrow
                    // tensor that was/was not sliced on multiple dimension - but we don't care about this scenario
                    // because we will never reach all the conditions to move to createMultipleInputOps state later)
                    m_operandGenerationState[operand] = generationState::stitchingOnMultiDimsEncountered;
                }
                break;
            case generationState::stitchingOnMultiDimsEncountered:
                if (m_intermediateSliceRefSet.find(generationQueue.front()) == m_intermediateSliceRefSet.end())
                {
                    //      1. we have double buffer (or more) for that operand,
                    // and: 2. we just arrived to a new row that we haven't created the (tpc) operations for it yet
                    // and: 3. we are in WIDE stitching of multiple dimension partials
                    m_operandGenerationState[operand] = generationState::createMultipleInputOps;
                    // don't generate the next slice reference yet- wait one "turn" and then create 2 (tpc) operations
                    // (as done in the normal algorithm at the very beginning for double buffer)
                    shouldGenerateNextInput = false;
                }
                break;
            case generationState::createMultipleInputOps:
                generateInputLookahead(operand);
                m_operandGenerationState[operand] = generationState::normal;
                break;
            default:
                HB_ASSERT(false, "Unhandled generationState enum!");
        }
    }
    return shouldGenerateNextInput;
}

void HandleEachStrategyOperation::generateNextInputs()
{
    SliceReferenceList nextInputs;
    for (auto& operandAndSwapPoints : m_operandInputsSwapPoints)
    {
        const pSlicedOperand& operand = operandAndSwapPoints.first;
        auto& swapPoints = operandAndSwapPoints.second;
        if (!swapPoints.empty() && swapPoints.front() == m_outputGenerationIterator.getCurrentIterator())
        {
            swapPoints.pop_front();
            bool generateNextInput = handleMultipleDimensionWideStitching(operand);
            SliceReferenceList& generationQueue = m_operandSlicesGenerationQueue[operand];
            if (generationQueue.empty()) continue;

            if (generateNextInput)
            {
                nextInputs.push_back(generationQueue.front());
                generationQueue.pop_front();
            }
            else
            {
                if (m_traceLog)
                {
                    SOL_GEN_TRACE("{}: Postpone generation of {}[{}]",
                                  HLLOG_FUNC,
                                  operand->originalTensor->getName(),
                                  toString(generationQueue.front()->coordinates, ','));
                }
            }
        }
    }
    generateFinalInputs(nextInputs);
}

void HandleEachStrategyOperation::generateNextOutputAndFwdMapped()
{
    const SliceRefCommonDimIdxPair& outputSlicePair = *m_outputGenerationIterator;
    const pSliceReference&          outputSlice     = outputSlicePair.first;
    const SliceReferenceList&       nextInputs      = m_strategy->getSlicingData().getInputsForSlice(outputSlicePair);
    const SliceReferenceList&       nextOutputs     = m_strategy->getSlicingData().getOutputsForSlice(outputSlice);
    HB_ASSERT(!nextInputs.empty(),
              "output generation list should not be empty - something went wrong.");

    logOperation(nextInputs, outputSlice, "master output");
    handleOperation(nextInputs, nextOutputs);

    // In case of partial (common dim slicing) multiple operations may be needed to fully
    // process the outputSlice. When reaching the last slice of the partial dim, all necessary calls were made.
    if (isLastSliceOnPartialDim(m_outputGenerationIterator))
    {
        addFwdMappedOperation({outputSlice});
    }

    // operation processing is done , can go on to the next output slice.
    ++m_outputGenerationIterator;
}

void HandleEachStrategyOperation::generateFinalInputs(const SliceReferenceList& inputList)
{
    for (const pSliceReference& input : inputList)
    {
        // in order to support TPC stitching of tensors which are sliced on multiple dimensions (can happen with
        // common solver which slices additionally on the spatial dim), we hold a set with all the previous sliced
        // references that we already created the TPC operations in order to generate them. This allows us not to
        // create the same operations again (the slicer will take care of creating dma nodes from sram to hbm and
        // from hbm to sram in order to support the next times the same slice reference is needed again)
        const auto& ret = m_intermediateSliceRefSet.insert(input);
        if (ret.second)
        {
            generateIntermediateInput(input);
        }
    }
}

void HandleEachStrategyOperation::generateIntermediateInput(const pSliceReference& slice)
{
    if (isIntermediateTensor(slice->operand->originalTensor))
    {
        const SliceReferenceList& inputSet = m_strategy->getSlicingData().getInputsForSlice(slice);
        const SliceReferenceList& outputSet = m_strategy->getSlicingData().getOutputsForSlice(slice);
        for (const pSliceReference& input : inputSet)
        {
            generateIntermediateInput(input);
        }
        logOperation(inputSet, slice, "intermediate slice");
        handleOperation(inputSet, outputSet);
    }
}

bool HandleEachStrategyOperation::isIntermediateTensor(const pTensor& tensor) const
{
    auto iter = m_slicedOperandByTensor.find(tensor);
    if (iter != m_slicedOperandByTensor.end())
    {
        return iter->second.intermediate;
    }
    return false;
}

void HandleEachStrategyOperation::logOperation(const SliceReferenceList& inputs,
                                               const pSliceReference&    output,
                                               const std::string&        operationName) const
{
    if (LOG_LEVEL_AT_LEAST_TRACE(SRAM_SOL_GEN) && m_traceLog)
    {
        std::stringstream ss;
        for (const auto& inputSlice : inputs)
        {
            if (inputSlice != inputs.front())
            {
                ss << ", ";
            }
            ss << inputSlice->operand->originalTensor->getName() << "[" << toString(inputSlice->coordinates, ',')
               << "]";
        }
        std::string inputsForLog = ss.str();

        SOL_GEN_TRACE("add {} operation to generate {}[{}] (inputs={})", operationName,
                      output->operand->originalTensor->getName(), toString(output->coordinates, ','), inputsForLog);
    }
}

void HandleEachStrategyOperation::handleOperation(const SliceReferenceList& inputs, const SliceReferenceList& outputs)
{
    const pNode& node = m_graph.getTensorProducer(outputs.front()->operand->originalTensor);
    SliceReferenceList allOutputs;
    if (outputs.size() < node->getNumOutputs())
    {
        // TODO SW-67410 - generate all outputs references is depracated and should be replaced by as accurate as
        // possible replacements in the mappers themselves.
        allOutputs = generateAllOutputReferences(outputs.front());
    }
    else
    {
        allOutputs = outputs;
    }
    m_operationHandler->handleOperation(node, inputs, allOutputs);
}

// TODO SW-67410 - delete this method.
SliceReferenceList HandleEachStrategyOperation::generateAllOutputReferences(const pSliceReference& outputSlice) const
{
    SliceReferenceList outputs;
    const pNode& producer = m_graph.getTensorProducer(outputSlice->operand->originalTensor);
    for (const pTensor& nextOutput : producer->getOutputs())
    {
        if (nextOutput == outputSlice->operand->originalTensor)
        {
            outputs.push_back(outputSlice);
            continue;
        }
        auto slicedOutputIt = m_slicedOperandByTensor.find(nextOutput);
        if (slicedOutputIt != m_slicedOperandByTensor.end())
        {
            const pSlicedOperand& slicedOutput = slicedOutputIt->second.operand;
            pSliceReference slice = std::make_shared<SliceReference>(slicedOutput);
            if (slicedOutputIt->second.slicedOnFcdOnly)
            {
                slice->coordinates[0] = outputSlice->coordinates[0];
            }
            else
            {
                slice->coordinates = outputSlice->coordinates;
            }
            outputs.push_back(slice);
        }
        else
        {
            // should never get here.
            HB_ASSERT(false, "Unexpected producer output");
        }
    }
    return outputs;
}

void HandleEachStrategyOperation::addFwdMappedOperation(const SliceReferenceList& generatedSlices)
{
    SliceReferenceList fwdGenerationQueue = generatedSlices;
    while (!fwdGenerationQueue.empty())
    {
        pSliceReference& slice = fwdGenerationQueue.front();
        pSliceReference newRef = std::make_shared<SliceReference>(*slice);

        const auto& inputsOutputsList = m_strategy->getSlicingData().getFwdMappedSlices(newRef);
        for (const auto& inputsOutputs : inputsOutputsList)
        {
            const SliceReferenceList& operationInputs  = inputsOutputs.first;
            const SliceReferenceList& operationOutputs = inputsOutputs.second;
            if (!operationOutputs.empty())
            {
                const pSliceReference& outputSlice = operationOutputs.front();
                logOperation(operationInputs, outputSlice, "fwd mapped");
                handleOperation(operationInputs, operationOutputs);
                // The generated outputs may also be mapped forward. Add them to the queue
                fwdGenerationQueue.insert(fwdGenerationQueue.end(), operationOutputs.begin(), operationOutputs.end());
            }
        }
        // Remove the processed slice
        fwdGenerationQueue.pop_front();
    }
}

bool HandleEachStrategyOperation::isLastSliceOnPartialDim(MultiOperandSliceIterator& outputIterator) const
{
    unsigned currentCDSlice = (*outputIterator).second;
    return (currentCDSlice == outputIterator.getCurrentIterator().getTotalOfCommonDimSlices()-1);
}

void HandleEachStrategyOperationMantaRay::generateNextOutputAndFwdMapped()
{
    const SliceRefCommonDimIdxPair& outputSlicePair = *m_outputGenerationIterator;
    const pSliceReference&          outputSlice     = outputSlicePair.first;
    const SliceReferenceList&       nextInputs      = m_strategy->getSlicingData().getInputsForSlice(outputSlicePair);
    const SliceReferenceList&       nextOutputs     = m_strategy->getSlicingData().getOutputsForSlice(outputSlice);
    HB_ASSERT(!nextInputs.empty(), "output generation list should not be empty - something went wrong.");

    logOperation(nextInputs, outputSlice, "master output");
    handleOperation(nextInputs, nextOutputs);

    // In case of partial (common dim slicing) multiple operations may be needed to fully
    // process the outputSlice. When reaching the last slice of the partial dim, all necessary calls were made.
    if (isLastSliceOnPartialDim(m_outputGenerationIterator))
    {
        addFwdMappedOperation({outputSlice});
    }

    addFwdMappedOperation(nextInputs);

    // operation processing is done , can go on to the next output slice.
    ++m_outputGenerationIterator;
}

void HandleEachStrategyOperationMantaRay::addFwdMappedOperation(const SliceReferenceList& generatedSlices)
{
    SliceReferenceList fwdGenerationQueue = generatedSlices;
    // Remove duplicate slices from the slices list to prevent generating the same output slice multiple times
    fwdGenerationQueue.sort([](const pSliceReference& slice1, const pSliceReference& slice2) {
        return slice1->operand->originalTensor < slice2->operand->originalTensor;
    });
    auto last = std::unique(fwdGenerationQueue.begin(),
                            fwdGenerationQueue.end(),
                            [](const pSliceReference& slice1, const pSliceReference& slice2) {
                                return slice1->operand->originalTensor == slice2->operand->originalTensor;
                            });
    fwdGenerationQueue.erase(last, fwdGenerationQueue.end());

    while (!fwdGenerationQueue.empty())
    {
        pSliceReference& slice  = fwdGenerationQueue.front();
        pSliceReference  newRef = std::make_shared<SliceReference>(*slice);

        const auto& inputsOutputsList = m_strategy->getSlicingData().getFwdMappedSlices(newRef);
        for (const auto& inputsOutputs : inputsOutputsList)
        {
            const SliceReferenceList& operationInputs = inputsOutputs.first;

            // The fwd mapped inputs may have producers themselves in the bundle. Schedule their generation.
            generateFinalInputs(operationInputs);

            const SliceReferenceList& operationOutputs = inputsOutputs.second;
            if (!operationOutputs.empty())
            {
                const pSliceReference& outputSlice = operationOutputs.front();
                logOperation(operationInputs, outputSlice, "fwd mapped");
                handleOperation(operationInputs, operationOutputs);
                // The generated outputs may also be mapped forward. Add them to the queue
                fwdGenerationQueue.insert(fwdGenerationQueue.end(), operationOutputs.begin(), operationOutputs.end());
            }
        }
        // Remove the processed slice
        fwdGenerationQueue.pop_front();
    }
}

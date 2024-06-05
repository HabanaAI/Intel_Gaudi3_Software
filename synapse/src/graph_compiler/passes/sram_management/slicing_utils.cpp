#include "slicing_utils.h"
#include "synapse_common_types.h"
#include "types.h"
#include "utils.h"
#include "habana_global_conf.h"
#include "graph_compiler/compilation_hal_reader.h"

const int HW_FIRST_DIM_STRIDE = 1;

// return the tensor coordinate offset for the slice coordinate, for a single dimension
unsigned SlicedOperandUtils::getSliceCoordOffset(unsigned coordIdx, unsigned chunkSize, int overlap, int tensoroffsetBefore)
{
    unsigned offsetBefore = 0;
    HB_ASSERT(tensoroffsetBefore >= 0, " negative offset is still not supported"); // TODO - SW-25558
    if (tensoroffsetBefore >= 0)
    {
        // the tensor may have offset due to padding. It affects all slices offset.
        // for the first slice it causes the dimension offset to be negative. Zero the offset as the first slice clipping handles that.
        offsetBefore = (coordIdx == 0) ? 0 : tensoroffsetBefore;
    }
    // dimension offset in the tensor is reduced by the overlap of the slices, and the offset before the first slice (padding)
    int dimensionOffset = coordIdx * (chunkSize - overlap) - offsetBefore;
    HB_ASSERT(dimensionOffset >= 0, "Slice coordinate is negative");
    return static_cast<unsigned>(dimensionOffset);
}

OffsetArray SlicedOperandUtils::getSliceOffsets(const pSliceReference& sliceRef)
{
    OffsetArray ret {};
    ret.fill(0);
    for (unsigned dim = 0; dim < sliceRef->operand->originalTensor->getDim(); dim++)
    {
        ret[dim] = getSliceCoordOffset(sliceRef->coordinates[dim],
                                       sliceRef->operand->chunkDimensions[dim],
                                       sliceRef->operand->overlapElementsCount[dim],
                                       sliceRef->operand->offsetBefore[dim]);
    }
    return ret;
}

// given original sizes of the tensor, chunkDimension with metadata about slice sizes,
// and requested coordinate inside it, return the matching slice size .
// i.e. originalTensorSizes={4,4}, chunkDimensions={4,3} and coord={0,0} should return {4,3}
//                                                       and coord={0,1} should return {4,1}
SizeArray SlicedOperandUtils::calcSliceSizesFromCoordinate(const SizeArray&      originalTensorSizes,
                                                           const pSlicedOperand& operand,
                                                           const CoordArray&     coord)
{
    SizeArray sliceSizes = operand->chunkDimensions;
    for (uint32_t dim = 0; dim < operand->chunkDimensions.size(); ++dim)
    {
        // get the coordinate offset in the dimension elements
        uint32_t dimensionOffset = getSliceCoordOffset(coord[dim],
                                                       operand->chunkDimensions[dim],
                                                       operand->overlapElementsCount[dim],
                                                       operand->offsetBefore[dim]);

        // Offset must fall inside the dimension, without padding since there is no padding only slice, unless that
        // dimension size is 0 (which may happen in input-describing shape tensors)
        HB_ASSERT((originalTensorSizes[dim] > dimensionOffset) ||
                      (originalTensorSizes[dim] == 0 && dimensionOffset == 0),
                  "Slice coordinate is out of bound for operand {}, dim={}, coord[dim]={}, chunkDimensions[dim]={}, "
                  "overlapElementsCount[dim]={}, offsetBefore[dim]={} (chunkDimensions=[{}], originalTensorSizes=[{}], "
                  "dimensionOffset={})",
                  operand->originalTensor->getName(),
                  dim,
                  coord[dim],
                  operand->chunkDimensions[dim],
                  operand->overlapElementsCount[dim],
                  operand->offsetBefore[dim],
                  toString(operand->chunkDimensions, ','),
                  toString(originalTensorSizes, ','),
                  dimensionOffset);

        // clip the size of the first slice, if there's an offset before the tensor (padding)
        if ((coord[dim] == 0) && (operand->offsetBefore[dim] > 0))
        {
            HB_ASSERT(sliceSizes[dim] > operand->offsetBefore[dim], "Slice is too small to clip padding.");
            sliceSizes[dim] -= operand->offsetBefore[dim];
            // TODO - SW-25558 - handle negative offset:
            // If offsetBefore < 0, this increases the slice[0] size. Need to make sure there is SRAM capacity to support this
        }

        // If operand has leftover elements on the current dim, extend the size of its last slice to cover the extra
        // leftover tensor elements
        if (operand->extraLeftoverAfter[dim] > 0)
        {
            unsigned dimNumSlices = SlicedOperandUtils::nofSlices(operand, dim);
            if (coord[dim] == dimNumSlices - 1)
            {
                sliceSizes[dim] += operand->extraLeftoverAfter[dim];
            }
        }

        // clip the size of the last slice according to tensor bounds
        sliceSizes[dim] = std::min(originalTensorSizes[dim] - dimensionOffset, sliceSizes[dim]);
    }
    return sliceSizes;
}
SizeArray SlicedOperandUtils::calcSliceSizesFromSliceRef(const pSliceReference& sliceRef)
{
    return calcSliceSizesFromCoordinate(sliceRef->operand->finalShape, sliceRef->operand, sliceRef->coordinates);
}

TSize SlicedOperandUtils::getAggregatedDimSizeInElements(const pSliceReference& sliceRef, const DimVector& dims)
{
    TSize sizeInElements = 1;
    // get actual slice sizes
    SizeArray sliceSizesA = calcSliceSizesFromSliceRef(sliceRef);
    for (const auto& dim : dims)
    {
        sizeInElements *= sliceSizesA[dim];
    }
    return sizeInElements;
}

TSize SlicedOperandUtils::getSliceSizeInElements(const pSliceReference& sliceRef)
{
    SizeArray sliceSize = calcSliceSizesFromSliceRef(sliceRef);
    return multiplyElements(sliceSize.begin(), sliceSize.begin() + sliceRef->operand->originalTensor->getDim());
}

TStride SlicedOperandUtils::getSliceSizeInBytes(const pSliceReference& sliceRef,
                                                bool                   useOriginalTensorElementType /*= false*/)
{
    if (sliceRef->operand->alignWithCacheLine)
    {
        auto alignedStrides = getCacheLineAlignedStrides(calcSliceSizesFromSliceRef(sliceRef), sliceRef->operand);
        if (alignedStrides)  // strides should be aligned
        {
            return alignedStrides->at(sliceRef->operand->originalTensor->getDim());
        }
    }
    if (useOriginalTensorElementType)
    {
        return getSliceSizeInElements(sliceRef) *
               dataTypeSizeInBytes(sliceRef->operand->originalTensor->getElementType());
    }
    return getSliceSizeInElements(sliceRef) * dataTypeSizeInBytes(sliceRef->operand->finalElementType);
}

TSize SlicedOperandUtils::getSliceSizeInElements(const pSlicedOperand& operand)
{
    return multiplyElements(operand->chunkDimensions.begin(),
                            operand->chunkDimensions.begin() + operand->originalTensor->getDim());
}

TStride SlicedOperandUtils::getSliceSizeInBytes(const pSlicedOperand& operand)
{
    if (operand->alignWithCacheLine)
    {
        auto alignedStrides = getCacheLineAlignedStrides(operand->chunkDimensions, operand);
        if (alignedStrides)  // strides should be aligned
        {
            return alignedStrides->at(operand->originalTensor->getDim());
        }
    }
    return getSliceSizeInElements(operand) * dataTypeSizeInBytes(operand->finalElementType);
}

std::optional<StrideArray> SlicedOperandUtils::getCacheLineAlignedStrides(const SizeArray&      finalSizes,
                                                                          const pSlicedOperand& operand)
{
    if (!GCFG_SRAM_SLICER_ALIGN_TO_CACHE_LINE.value()) return std::optional<StrideArray>{};
    const TensorPtr& tensor = operand->originalTensor;
    // calculate regular (unaligned) strides given the input chunk dimensions (finalSizes)
    StrideArray initialStrides = {};
    unsigned    firstStrideVal = finalSizes[0] * dataTypeSizeInBytes(operand->finalElementType);
    HB_ASSERT(!tensor->isStridedOnFCD(), "operand {} is strided on fcd", tensor->getName());
    fillStrides(initialStrides, finalSizes, 0 /* from dim */, firstStrideVal, tensor->getDim());
    // try to align them
    return tensor->getCacheLineAlignedStrides(finalSizes, initialStrides, tensor->getDim());
}

std::optional<StrideArray> SlicedOperandUtils::getCacheLineAlignedStrides(const SizeArray&   sizes,
                                                                          const StrideArray& initialStrides,
                                                                          unsigned           numOfDims,
                                                                          unsigned           dimToAlign /* = 0 */)
{
    // check if alignment optimization is on:
    if (!GCFG_SRAM_SLICER_ALIGN_TO_CACHE_LINE.value()) return std::optional<StrideArray>{};
    return Tensor::getCacheLineAlignedStrides(sizes, initialStrides, numOfDims, dimToAlign);
}

void SlicedOperandUtils::fillStrides(StrideArray& strides,
                                     const SizeArray& sizes,
                                     unsigned fromDim,
                                     unsigned fromDimValue,
                                     unsigned ToDim)
{
    strides[fromDim + 1] = fromDimValue;
    for (unsigned dim = fromDim + 1; dim < ToDim; ++dim)
    {
        strides[dim + 1] = strides[dim] * sizes[dim];
    }
}

bool SlicedOperandUtils::isTriviallySliced(const pSlicedOperand &operand)
{
    return operand->chunkDimensions == operand->finalShape &&
           !operand->hasOverlap();  // The operand might be sliced to the original size,
                                    // but with the overlap there's another slice
}

bool SlicedOperandUtils::isTriviallySliced(const SlicingStrategy& strategy)
{
    for (const auto& op : strategy.getSlicingData().getSlicedOperands())
    {
        if (!SlicedOperandUtils::isTriviallySliced(op)) return false;
    }
    return true;
}

bool SlicedOperandUtils::isSlicedOnDimension(const pSlicedOperand &operand, unsigned dimIdx)
{
    if (operand->chunkDimensions[dimIdx] != operand->finalShape[dimIdx])
    {
        return true;
    }
    if (operand->overlapElementsCount[dimIdx] != 0)
    {
        // The operand might be sliced to the original size, but with the overlap there's another slice
        return true;
    }
    if ((operand->offsetBefore[dimIdx] != 0) || (operand->offsetAfter[dimIdx] != 0))
    {
        // The operand might be sliced to the original size without overlap, but with the padding there's another slice.
        // nofSlices checks all the corner cases (if this slice is valid or not) and is the most accurate check
        // but limit this call to reduce computation time.
        return nofSlices(operand, dimIdx) > 1;
    }
    return false;
}

TSize SlicedOperandUtils::getNarrowFullAxisSize(const MmeSlicingStrategy& strategy)
{
    const SizeArray& outputSizes = strategy.getMmeSlicingData().masterOperand->finalShape;
    const DimVector& narrowOutputSlicingDims = strategy.getMmeSlicingData().getNarrowOutputSlicingDims();
    return multiplyElements(outputSizes.data() + narrowOutputSlicingDims.front(),
                            outputSizes.data() + narrowOutputSlicingDims.back() + 1);
}

TSize SlicedOperandUtils::getWideFullAxisSize(const MmeSlicingStrategy& strategy)
{
    const SizeArray& outputSizes = strategy.getMmeSlicingData().masterOperand->finalShape;
    const DimVector& wideOutputSlicingDims = strategy.getMmeSlicingData().getWideOutputSlicingDims();
    return multiplyElements(outputSizes.data() + wideOutputSlicingDims.front(),
                            outputSizes.data() + wideOutputSlicingDims.back() + 1);
}

bool SlicedOperandUtils::isTensor2D(const pTensor& tensor)
{
    return isTensorShape2D(tensor, tensor->getAllSizesInElements());
}

bool SlicedOperandUtils::isFinalShape2D(const pSlicedOperand& op)
{
    return isTensorShape2D(op->originalTensor, op->finalShape);
}

bool SlicedOperandUtils::shouldOperandBeFlattened(const pSlicedOperand& op)
{
    return (op->finalShape != op->originalTensor->getAllSizesInElements()) && isFinalShape2D(op);
}

bool SlicedOperandUtils::shouldAnyOperandBeFlattened(const std::vector<pSlicedOperand>& operands)
{
    for (const pSlicedOperand& operand : operands)
    {
        if (shouldOperandBeFlattened(operand))
        {
            return true;
        }
    }
    return false;
}
unsigned SlicedOperandUtils::nofSlices(const pSlicedOperand& op, const DimVector& dims)
{
    unsigned totalNumSlices = 1;
    for (auto& dim : dims)
    {
        totalNumSlices *= SlicedOperandUtils::nofSlices(op, dim);
    }
    return totalNumSlices;
}

unsigned SlicedOperandUtils::nofSlices(const pSlicedOperand& op, unsigned dim)
{
    if (op->chunkDimensions[dim] == 0) return 1;
    // total_size = tensor_size + offset_before + offset_after
    // total_size[t]  = (num_of_slices[n] - 1) * (slice_size[s] - overlap[o]) + slice_size[s]
    // => t = ns - no -s + o + s
    // => t - o = n * (s - o)
    // => n = (t - o) / (s - o)

    // According to the formula above -
    // First calculate number of full slices - total dim size (tensor_size + offset) minus overlap, divided by (slice_size - overlap).
    // Then need to check the remainder - if remainder + overlap >= minimal valid slice size to generate output -> increase number of slices by 1.
    unsigned totalSize = op->finalShape[dim] + op->offsetBefore[dim];
    unsigned totalSizeMinusOverlap = totalSize - op->overlapElementsCount[dim];
    unsigned chunkMinusOverlap = op->chunkDimensions[dim] - op->overlapElementsCount[dim];
    unsigned numOfValidSlices = totalSizeMinusOverlap / chunkMinusOverlap;

    // remainder = total_size - first_slice_size - (numOfValidSlices - 1) * (slice_size - overlap)
    // remainder = totalSize - op->chunkDimensions[dim] - (numOfValidSlices - 1) * chunkMinusOverlap;
    unsigned remainder = totalSizeMinusOverlap % chunkMinusOverlap;

    int lastSliceActualSize = remainder + op->overlapElementsCount[dim] + op->offsetAfter[dim];
    // Check if the remainder with the overlap and padding is enough to create another slice
    if (lastSliceActualSize >= static_cast<int>(op->minValidSliceSize[dim]))
    {
        bool lastSliceIsPadding = (lastSliceActualSize <= op->offsetAfter[dim]) && (remainder == 0);
        if (!lastSliceIsPadding || op->countPaddingOnlySlice)
        {
            numOfValidSlices++;
        }
    }
    // In case of large remainder and enough overlap, there might enough data to create another slice from the overlap.
    // numerical example: tensor size = 256, chunk size = 7, overlap = 2 => remainder = 4
    // last slice size = remainder + overlap = 6 => extra slice (computed above) is not a full slice.
    // This slice also includes 1 line of the overlap of the next slice, but this is where the tensor ends, so there is 1 more
    // slice, which is just this 1 overlap line. In dedx this line of overlap may be required to produce another dx slice.
    lastSliceActualSize = lastSliceActualSize - chunkMinusOverlap;
    if (lastSliceActualSize >= static_cast<int>(op->minValidSliceSize[dim]))
    {
        bool lastSliceIsPadding = (lastSliceActualSize <= op->offsetAfter[dim]);
        if (!lastSliceIsPadding || op->countPaddingOnlySlice)
        {
            numOfValidSlices++;
        }
    }

    return numOfValidSlices;
}

unsigned SlicedOperandUtils::nofSlices(pSlicedOperand op)
{
    unsigned numofSlices = 1;
    for (unsigned i = 0; i < op->originalTensor->getDim(); ++i)
    {
        numofSlices *= nofSlices(op, i);
    }
    return numofSlices;
}

unsigned SlicedOperandUtils::getNumOfSlicedDims(const pSlicedOperand& operand)
{
    unsigned slicedDims = 0;
    for (unsigned i = 0; i < MAX_DIMENSIONS_NUM; ++i)
    {
        if (SlicedOperandUtils::isSlicedOnDimension(operand, i))
        {
            slicedDims++;
        }
    }
    return slicedDims;
}

const pSlicedOperand& SlicedOperandUtils::getNonStitchedOperand(const SlicingData& slicingData,
                                                                const pSlicedOperand& stitchedOperand)
{
    HB_ASSERT(stitchedOperand == slicingData.bundleTensors[0] || stitchedOperand == slicingData.bundleTensors[1],
              "failed to find stitched operand");
    return (stitchedOperand == slicingData.bundleTensors[0] ? slicingData.bundleTensors[1] :
                                                              slicingData.bundleTensors[0]);
}

pSlicedOperand SlicedOperandUtils::createNonSharedOperandOfNode(const pSlicedOperand& sharedOperand,
                                                                const pNode& node)
{
    // get slave non shared input
    pTensor nonSharedInput;
    if (node->getInput(0) == sharedOperand->originalTensor)
    {
        nonSharedInput = node->getInput(1);
    }
    else
    {
        nonSharedInput = node->getInput(0);
    }
    return std::make_shared<SlicedOperand>(nonSharedInput);
}

bool SlicedOperandUtils::isSlicedOnCommonDim(const pSlicedOperand& operand, const pNode& node)
{
    HB_ASSERT(operand->originalTensor != node->getOutput(0), "operand is node output");
    MmeDimController controller(node);
    unsigned inputIndex = node->getInputIndexOfTensor(operand->originalTensor);
    const DimVector& commonDims = (inputIndex == 0) ? controller.commonDimOperandA() : controller.commonDimOperandB();
    for (const auto& dim : commonDims)
    {
        if (SlicedOperandUtils::isSlicedOnDimension(operand, dim))
        {
            return true;
        }
    }
    return false;

}

Settable<unsigned> SlicedOperandUtils::getFirstNonDegeneratedDim(const DimVector& dims, const pSlicedOperand& operand)
{
    auto dimIt = std::find_if(dims.rbegin(), dims.rend(),
                              [&](const uint32_t d)
                              {
                                return (operand->finalShape[d] != 1);
                              });
    if (dimIt != dims.rend())
    {
        return {*dimIt};
    }
    else
    {
        return {};
    }
}

bool SlicedOperandUtils::areDimsDegenerated(const pTensor& tensor, unsigned fromDim, unsigned toDim)
{
    // we only care about input dims that are inside the tensor dimensions:
    if (fromDim >= tensor->getDim()) return true;
    toDim = (toDim >= tensor->getDim() ? tensor->getDim() - 1 : toDim);
    const SizeArray& tensorSizes = tensor->getAllSizesInElements();
    return (multiplyElements(tensorSizes.begin() + fromDim, tensorSizes.begin() + toDim + 1) == 1);
}

bool SlicedOperandUtils::areDegeneratedDimsReshaped(const pNode& reshapeNode)
{
    const pTensor& reshapeInput   = reshapeNode->getInput(0);
    const pTensor& reshapeOutput  = reshapeNode->getOutput(0);
    if (reshapeInput->getDim() == reshapeOutput->getDim()) return true; // not interesting
    const pTensor& minDimTensor = reshapeInput->getDim() < reshapeOutput->getDim() ? reshapeInput : reshapeOutput;
    const pTensor& maxDimTensor = reshapeInput->getDim() < reshapeOutput->getDim() ? reshapeOutput : reshapeInput;
    // if all the dims that consist the difference are degenerated - return true
    return areDimsDegenerated(maxDimTensor, minDimTensor->getDim(), maxDimTensor->getDim() - 1);
}

DimVector SlicedOperandUtils::getSlicedDims(const pSlicedOperand& operand)
{
    DimVector slicedDims;
    for (uint32_t dim = 0; dim < operand->originalTensor->getDim(); ++dim)
    {
        if (SlicedOperandUtils::isSlicedOnDimension(operand, dim))
        {
            slicedDims.push_back(dim);
        }
    }
    return slicedDims;
}

bool SlicedOperandUtils::isTensorShape2D(const pTensor& tensor, const SizeArray& shape)
{
    const unsigned tensorDim = tensor->getDim();
    uint64_t tensorSize = tensor->getDenseSizeInElements();
    // 2D size is calculated as the size of the first two dimensions (DIM - 0, DIM - 1)
    uint64_t twoDimSize = (uint64_t)shape[DIM_C] * (uint64_t)shape[DIM_W];
    // Tensor is 2D by definition or total size of it equals the size of the first two dimension.
    return tensorDim < 3 || tensorSize == twoDimSize;
}

std::array<bool, SYN_MAX_TENSOR_DIM> SlicedOperandUtils::isFirstSlice(const pSliceReference& sliceRef)
{
    std::array<bool, SYN_MAX_TENSOR_DIM> isFirst = {false};
    for (unsigned dim = 0; dim < sliceRef->operand->originalTensor->getDim(); ++dim)
    {
        isFirst[dim] = (sliceRef->coordinates[dim] == 0);
    }
    return isFirst;
}

std::array<bool, SYN_MAX_TENSOR_DIM> SlicedOperandUtils::isLastSlice(const pSliceReference& sliceRef)
{
    std::array<bool, SYN_MAX_TENSOR_DIM> isLast = {false};
    for (unsigned dim = 0; dim < sliceRef->operand->originalTensor->getDim(); ++dim)
    {
        isLast[dim] = (sliceRef->coordinates[dim] == nofSlices(sliceRef->operand, dim) - 1);
    }
    return isLast;
}

SlicedOperandUtils::ReshapeOutputToInputMapping
SlicedOperandUtils::getReshapeOutputToInputMapping(const TensorShape& input, const TensorShape& output)
{
    /*
        calculated the reshape mapping from output to input - which dims from input are required in output.
        for example, in the reshape of [32, 16, 8] -> [32, 128] the reshapeMap is [(0), (1, 2)].
        for example, in the reshape of [32, 16, 8] -> [64, 64] the reshapeMap is [(0, 1), (1, 2)].
        for example, in the reshape of [32, 128] -> [32, 16, 8] the reshapeMap is [(0), (1), (1)].
    */
    const NSizeArray& inputSize        = input.getNSizes();
    const NSizeArray& outputSize       = output.getNSizes();
    uint64_t          numElementsTotal = multiplyElements(std::begin(outputSize), std::begin(outputSize) + output.getDim());
    HB_ASSERT(numElementsTotal == multiplyElements(std::begin(inputSize), std::begin(inputSize) + input.getDim()), "number of dense elements don't match");

    ReshapeOutputToInputMapping mapping(output.getDim());

    unsigned dimIn          = 0;
    unsigned dimOut         = 0;
    uint64_t numElementsIn  = inputSize[0];   // elements accounted for in input so far
    uint64_t numElementsOut = outputSize[0];  // elements accounted for in output so far
    mapping[0].push_back(0);    // first dim of input is always reshaped into first dim of output

    while ((numElementsIn < numElementsTotal) || (numElementsOut < numElementsTotal))
    {
        if (numElementsIn == numElementsOut) // move on to the next dim of both output and input
        {
            numElementsIn *= inputSize[++dimIn];
            numElementsOut *= outputSize[++dimOut];
        }
        else if (numElementsIn < numElementsOut) // collect the next dim in input
        {
            numElementsIn *= inputSize[++dimIn];
        }
        else //(numElementsIn > numElementsOut) // collect the next dim in output
        {
            numElementsOut *= outputSize[++dimOut];
        }
        mapping[dimOut].push_back(dimIn);
    }

    unsigned inputLastDim  = input.getDim() - 1;
    unsigned outputLastDim = output.getDim() - 1;
    // if there are outer dimensions of size 1 we can map them in arbitrary order
    while (dimIn < inputLastDim || dimOut < outputLastDim)
    {
        dimIn += (dimIn < inputLastDim);
        dimOut += (dimOut < outputLastDim);
        mapping[dimOut].push_back(dimIn);
    }
    return mapping;
}

bool SlicedOperandUtils::isBroadcast(const NodePtr& tpcNode)
{
    if (!tpcNode->isBroadcastableOperation()) return false;
    if (tpcNode->getNumInputs() == 1) return false;
    const SizeArray& outputSizes = tpcNode->getOutput(0)->getAllSizesInElements();
    for (uint32_t i=0; i < tpcNode->getNumInputs(); i++)
    { // broadcasted operation with input dimensions that differ from output
        if (outputSizes != tpcNode->getInput(i)->getAllSizesInElements()) return true;
    }
    return false;
}

bool SlicedOperandUtils::canReshapeBroadcast(const TensorPtr& input, const TensorPtr& output, const TensorPtr& reshapeOutput)
{
    /*
        checks if a broadcast input can be reshaped.
        for each dimension in the reshape output, look at all needed dimensions from reshape output.
        we require that all of the needed dimension values in the broadcasted input
        either match the dimensions of the broadcast output, or are all ones.
    */
    // get reshape mapping from output -> input dims
    const auto& reshapeMap = SlicedOperandUtils::getReshapeOutputToInputMapping(
        output->getShape(), reshapeOutput->getShape());
    const SizeArray& inputSizes = input->getAllSizesInElements();
    const SizeArray& outputSizes = output->getAllSizesInElements();
    for (const auto& dimMap : reshapeMap)
    {
        bool allOnes = true;
        bool allSame = true;
        for (uint32_t dim : dimMap)
        {
            allOnes &= (inputSizes[dim] == 1);
            allSame &= (inputSizes[dim] == outputSizes[dim]);
        }
        if (!(allOnes || allSame)) return false; // mapped dims in broadcast input must be all '1' or all the same as output
    }
    return true;
}

bool SlicedOperandUtils::canAlignProducerReshape(const pBundleExpansion& candidate)
{
    TensorPtr             producerOutput  = nullptr;
    TensorPtr             reshapeOutput   = nullptr;
    const pSlicedOperand& producedOperand = candidate->stitchedOperand;
    const NodePtr&        producer        = candidate->nodeToStitch;
    if (candidate->reshapeNode != nullptr)
    {  // explicity reshape
        producerOutput = candidate->reshapeNode->getInput(0);
        reshapeOutput  = candidate->reshapeNode->getOutput(0);
    }
    else if (producedOperand->finalShape != producedOperand->originalTensor->getAllSizesInElements())
    {  // this means we insert a flattening to 2D later.
        producerOutput = producedOperand->originalTensor;
        reshapeOutput  = producedOperand->originalTensor->clone();
        reshapeOutput->reshape(2, producedOperand->finalShape.data(), nullptr);
    }
    else
    {  // no reshape
        return true;
    }

    if (!SlicedOperandUtils::isBroadcast(producer))
        return true;  // no problem to align reshapes for nodes that aren't using broadcast

    for (const TensorPtr& input : producer->getInputs())
    {
        if (!canReshapeBroadcast(input, producerOutput, reshapeOutput))
            return false;  // broadcast that cannot be reshaped
    }
    return true;
}

bool SlicedOperandUtils::isOperandFlattened(const pSlicedOperand& operand)
{
    return (operand->finalShape != operand->originalTensor->getAllSizesInElements());
}

synDataType SlicedOperandUtils::getTypeForPartials(synDataType currentMmeOutputDataType)
{
    // The data type used for the summation of partials should be high precision.
    if (isHighPrecisionFloat(currentMmeOutputDataType))
    {
        return currentMmeOutputDataType;
    }
    else
    {
        return CompilationHalReader::getHalReader()->getMmeHighPrecisionTypeForPartials();
    }
}

TStride SlicedOperandUtils::getTensorSliceSizeInBytes(const TensorPtr&                 t,
                                                      const std::map<unsigned, TSize>& sizePerSlicedDim)
{
    NSizeArray sliceSizes = t->getAllNSizesInElements();
    // Update the sliced dims size to the given sizes
    for (auto [slicedDim, slicedDimSize] : sizePerSlicedDim)
    {
        sliceSizes[slicedDim] = slicedDimSize;
    }
    TStride sliceSizeInBytes = multiplyElements(sliceSizes.begin(), sliceSizes.begin() + t->getDim());
    sliceSizeInBytes *= t->getElementSizeInBytes();
    return sliceSizeInBytes;
}

bool SlicedOperandUtils::isTensorAlignedToMmeCL(const NodePtr& n, unsigned inputIdx)
{
    const auto& t = n->getInput(inputIdx);
    auto alignedStrides = Tensor::getCacheLineAlignedStrides(t->getAllSizesInElements(), t->getAllStridesInBytes(), t->getDim());
    return !alignedStrides.has_value();
}

TStride SlicedOperandUtils::getSliceAlignmentSize(const TensorPtr& t, const std::map<unsigned, TSize>& sizePerSlicedDim)
{
    unsigned  rank  = t->getDim();
    SizeArray sizes = t->getAllSizesInElements();
    for (auto [slicedDim, slicedDimSize] : sizePerSlicedDim)
    {
        sizes[slicedDim] = slicedDimSize;
    }
    StrideArray strides;
    TStride     stride = t->getElementSizeInBytes();
    for (unsigned d = 0; d <= rank; ++d)
    {
        strides[d] = stride;
        stride *= (TStride)sizes[d];
    }
    auto alignedStrides     = Tensor::getCacheLineAlignedStrides(sizes, strides, rank);
    auto unalignedSliceSize = getTensorSliceSizeInBytes(t, sizePerSlicedDim);
    auto alignmentSize      = alignedStrides.has_value() ? (alignedStrides->at(rank) - unalignedSliceSize) : 0;

    return alignmentSize;
}

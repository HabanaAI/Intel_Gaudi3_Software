#include "tpc_slice_roi_generator.h"
#include "utils.h"

TPCSliceROIGenerator::TPCSliceROIGenerator(const TPCSlice& sliceNode)
: m_sliceNode(sliceNode),
  m_instance(sliceNode.getInstance()),
  m_roiSize(m_instance.indexSpaceRank, 0),
  m_roiOffset(m_instance.indexSpaceRank, 0)
{
    init();
}

void TPCSliceROIGenerator::init()
{
    initROISize();
    initROIOffset();
}

void TPCSliceROIGenerator::initROISize()
{
    // By default the ROI size is the full index space geometry
    auto* pIdxSpaceGeometryBegin = m_instance.indexSpaceGeometry;
    auto* pIdxSpaceGeometryEnd   = pIdxSpaceGeometryBegin + m_instance.indexSpaceRank;
    std::copy(pIdxSpaceGeometryBegin, pIdxSpaceGeometryEnd, m_roiSize.begin());
}

void TPCSliceROIGenerator::initROIOffset()
{
    // Nothing to do for now - Offsets are initialized to 0 in the offset vector constructor.
}

NodeROI TPCSliceROIGenerator::generateROI()
{
    // First need to set the size and offset according to the inputs and outputs.
    analyzeInputs();
    analyzeOutputs();

    // Now a correct node ROI can be generated.
    return convertSizeOffsetToNodeROI();
}

void TPCSliceROIGenerator::analyzeInputs()
{
    analyzeTensors(m_sliceNode.getInputs(), m_instance.inputTensorAccessPattern);
}

void TPCSliceROIGenerator::analyzeOutputs()
{
    analyzeTensors(m_sliceNode.getOutputs(), m_instance.outputTensorAccessPattern);
}

template<typename Container>
void TPCSliceROIGenerator::analyzeTensors(const Container& sliceTensors, const TensorAccessPattern* tensorAPs)
{
    unsigned tensorAPBaseIdx = 0;
    for (const TensorPtr& sliceTensor : sliceTensors)
    {
        analyzeTensor(sliceTensor, tensorAPs + tensorAPBaseIdx);
        tensorAPBaseIdx += TPCNode::numTensorGlueCodeAccessPatternEntries(sliceTensor);
    }
}

// Update the ROI sizes and offsets according to the tensorSlice.
void TPCSliceROIGenerator::analyzeTensor(const TensorPtr& sliceTensor, const TensorAccessPattern* tensorAPs)
{
    const auto* tensorAP = findTensorAccessPattern(sliceTensor, tensorAPs);
    if (!tensorAP)
    {
        // No access pattern found for the tensor with allRequired == false ==> The tensor is allRequired.
        // AllRequired tensors do not have valid access patterns and they are degenerate cases anyway, so skipping them.
        return;
    }
    for (unsigned dim = 0; dim < sliceTensor->getDim(); dim++)
    {
        if (shouldSkipTensorForDimIdxSpaceAnalysis(sliceTensor, dim)) continue;

        const auto& tensorDimAP = tensorAP->mapping[dim];
        analyzeDimSize(sliceTensor, dim, tensorDimAP);
        analyzeDimOffset(sliceTensor, dim, tensorDimAP);
    }
}

// given the array of access patterns for the tensor, return the "real" one (not allRequired) or nullptr of no such
// entry exists
const TPCSliceROIGenerator::TensorAccessPattern*
TPCSliceROIGenerator::findTensorAccessPattern(const TensorPtr& sliceTensor, const TensorAccessPattern* tensorAPs) const
{
    auto numAPs = TPCNode::numTensorGlueCodeAccessPatternEntries(sliceTensor);
    for (const auto* currAP = tensorAPs; currAP < tensorAPs + numAPs; currAP++)
    {
        if (!currAP->allRequired) return currAP;
    }
    return nullptr;
}

// Update the ROI size for the specific dimension
void TPCSliceROIGenerator::analyzeDimSize(const TensorPtr&              sliceTensor,
                                          unsigned                      tensorDim,
                                          const TensorDimAccessPattern& tensorDimAP)
{
    unsigned idxSpaceDim         = tensorDimAP.indexSpaceDim;
    auto     sliceSizeInElements = sliceTensor->getSizeInElements(tensorDim);
    auto     indexSpaceSize      = getOperandIdxSpaceRegionSize(sliceSizeInElements, tensorDimAP);

    // Some slice tensors may cover the whole dimension (when the dimension is
    // all-required), so take the minimal to get the concrete roi size based on
    // tensors that are actually sliced on this dimension (if there are any)
    m_roiSize[idxSpaceDim] = std::min(m_roiSize[idxSpaceDim], indexSpaceSize);
}

// Update the ROI offset for the specific dimension
void TPCSliceROIGenerator::analyzeDimOffset(const TensorPtr&              sliceTensor,
                                            unsigned                      tensorDim,
                                            const TensorDimAccessPattern& tensorDimAP)
{
    unsigned idxSpaceDim      = tensorDimAP.indexSpaceDim;
    auto     offsetInElements = m_sliceNode.getTensorSliceOffsetInDim(sliceTensor, tensorDim);
    auto     indexSpaceOffset = getOperandIdxSpaceOffset(offsetInElements, tensorDimAP);

    // Some slice tensors may cover the whole dimension (when the dimension is
    // all-required), so take the maximal offset to get the concrete roi offset based on
    // tensors that are actually sliced on this dimension (if there are any)
    m_roiOffset[idxSpaceDim] = std::max(m_roiOffset[idxSpaceDim], indexSpaceOffset);
}

// Get the number of index space elements that take part in reading/writing tensorDimSize elements from a tensor with
// the specified access pattern, or MAX_UINT in case of fixed access pattern (a==0).
TPCSliceROIGenerator::Size TPCSliceROIGenerator::getOperandIdxSpaceRegionSize(
    TPCSliceROIGenerator::Size                          tensorDimSize,
    const TPCSliceROIGenerator::TensorDimAccessPattern& tensorDimAccessPattern)
{
    if (tensorDimAccessPattern.a == 0)
    {
        // Assuming the caller clips whatever this function returns to the size of the indexSpace in the relevant
        // dimension.
        return std::numeric_limits<unsigned>::max();
    }
    // If the tensor size is a multiple of index space windows, then we can extract the number of index elements from
    // the formula to the end element in the tensor: a * i + end_b = lastElementIndex, solve for i.
    // The tensor can be a little smaller, though, since at the tail, an index space element window may exceed the
    // tensor boundary.
    unsigned lastIdx =
        std::ceil((tensorDimSize - 1 - (double)tensorDimAccessPattern.end_b) / (double)tensorDimAccessPattern.a);

    // Size = Last-Element-index + 1. If the last element is 0, it means the size is 1 element, etc...
    return lastIdx + 1;
}

// Get the index space element of the first element of the tensor slice
TPCSliceROIGenerator::Offset TPCSliceROIGenerator::getOperandIdxSpaceOffset(
    TPCSliceROIGenerator::Offset                        offsetInElements,
    const TPCSliceROIGenerator::TensorDimAccessPattern& tensorDimAccessPattern)
{
    if (tensorDimAccessPattern.a == 0)
    {
        // Access pattern with a==0 always start in a constant offset.
        HB_ASSERT(offsetInElements == tensorDimAccessPattern.start_b,
                  "Slice offset is incompatible with access pattern. Offset: {}, a: {}, start_b: {}",
                  offsetInElements,
                  tensorDimAccessPattern.a,
                  tensorDimAccessPattern.start_b);
        // Assume the first index is the offset of the slice in this dimension.
        return 0;
    }

    // The offset is assumed to be a multiple of index element window size. So it can be extracted from the formula:
    // Offset = a * i + start_b, solve for i.
    float firstIdxElement =
        (offsetInElements - (double)tensorDimAccessPattern.start_b) / (double)tensorDimAccessPattern.a;
    HB_ASSERT(firstIdxElement == std::floor(firstIdxElement),
              "Expected offset in full index space elements. Offset: {}, a: {}, start_b: {}",
              offsetInElements,
              tensorDimAccessPattern.a,
              tensorDimAccessPattern.start_b);
    return static_cast<unsigned>(firstIdxElement);
}

bool TPCSliceROIGenerator::shouldSkipTensorForDimIdxSpaceAnalysis(const TensorPtr& sliceTensor, unsigned dim) const
{
    return sliceTensor->isAuxTensor() || sliceTensor->isShapeTensor() ||
           m_sliceNode.getOriginalTensor(sliceTensor)->getSizeInElements(dim) == sliceTensor->getSizeInElements(dim);
}

NodeROI TPCSliceROIGenerator::convertSizeOffsetToNodeROI() const
{
    NodeROI roi;
    std::copy(m_roiSize.begin(), m_roiSize.end(), roi.size);
    std::copy(m_roiOffset.begin(), m_roiOffset.end(), roi.baseOffset);

    // ROI size may have more dimensions than the index space geometry. The un-used dimension sizes should be set to 1.
    std::fill(roi.size + m_instance.indexSpaceRank, roi.size + ARRAY_SIZE(roi.size), 1);

    LOG_TRACE(TPC_SLICE,
              "ROI generated for node '{}': Size [{}], Offset [{}]",
              m_sliceNode.getNodeName(),
              toString(roi.size, roi.size + m_instance.indexSpaceRank, ','),
              toString(roi.baseOffset, roi.baseOffset + m_instance.indexSpaceRank, ','));

    return roi;
}

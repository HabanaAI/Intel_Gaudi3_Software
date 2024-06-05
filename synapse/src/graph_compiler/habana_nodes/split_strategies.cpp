#include "split_strategies.h"

#include "compilation_hal_reader.h"
#include "dma_transpose_helper.h"
#include "dma_transpose_node.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "include/sync/data_range.h"
#include "log_manager.h"
#include "math_utils.h"
#include "node.h"
#include "node_roi.h"
#include "roi_splitter.h"
#include "transpose_utils.h"
#include "utils.h"

#include <algorithm>
#include <iterator>
#include <tuple>

void SplitStrategy::updateNumSignals(std::list<NodeROI>::iterator start,
                                     std::list<NodeROI>::iterator last,
                                     uint32_t                     physicalEngineCount)
{
    for (std::list<NodeROI>::reverse_iterator it(last), rEnd(start); it != rEnd; it++)
    {
        // In round robin, the last physicalEngineCount items are the last in each of the respective queue.
        if (physicalEngineCount > 0)
        {
            it->numSignals = 1;
            physicalEngineCount--;
        }
        else
        {
            it->numSignals = 0;
        }
    }
}

static DmaTransposeHelper getHelperForNode(const Node& node)
{
    auto&              transposeNode = static_cast<const DMATransposeNode&>(node);
    auto elementType = node.getInput(0)->getElementType();
    auto params = transposeNode.getGraphTraits()->getHalReader()->getDmaTransposeEngineParams();
    DmaTransposeHelper helper(elementType, params);
    return helper;
}

std::list<NodeROI> SplitStrategy::splitToPhysical(const Node& node, const std::list<NodeROI>& roisToSplit, uint32_t physicalEngineCount)
{
    std::list<NodeROI> ret;
    for (auto& roi : roisToSplit)
    {
        std::list<NodeROI> temp;
        splitRoiToEngines(node, roi, physicalEngineCount, temp);
        updateNumSignals(temp.begin(), temp.end(), physicalEngineCount);
        ret.splice(ret.end(), std::move(temp));
    }
    const Tensor*          inputTensor        = node.getInput(0).get();
    const uint32_t         elementSizeBits    = inputTensor->getElementSizeInBits();
    const deviceAddrOffset inputTensorOffset  = inputTensor->getTensorOffset();
    const deviceAddrOffset outputTensorOffset = node.getOutput(0)->getTensorOffset();

    for (auto& roi : ret)
    {
        auto& iLayout = roi.inputRois[0].getLayout();
        auto& oLayout = roi.outputRois[0].getLayout();
        iLayout.baseAddress = inputTensorOffset + safeBitsToByte(iLayout.m_baseOffset[0] * elementSizeBits);
        oLayout.baseAddress = outputTensorOffset + safeBitsToByte(oLayout.m_baseOffset[0] * elementSizeBits);
        for (size_t i = 1; i < Tensor::c_tensorMaxDim; i++)
        {
            iLayout.baseAddress += iLayout.m_baseOffset[i] * iLayout.spatialStrides[i - 1];
            oLayout.baseAddress += oLayout.m_baseOffset[i] * oLayout.spatialStrides[i - 1];
        }
        finalize(node, roi);
    }
    return ret;
}

/*
 * For when transpose BIG (c,w) ROI of (C,W)  such that w is a power of 2 (or multiple of 128 / ELEMENT_SIZE), and c is constructed such as: c=c'*h*k, such that k <= 64, h=max(128/w/ ELEMENT_SIZE, 1)

// When c= max(128/w/ ELEMENT_SIZE, 1)*k, and c < 128 w doesn't have to be a power of 2.
One must set:

Input:
(k*ELEMENT_SIZE, w, h*c', 1)
Strides:
(C*ELEMENT_SIZE, k*ELEMENT_SIZE, C*W*ELEMENT_SIZE)



Output:
(min(w *ELEMENT_SIZE, 128), h, k, max(w*ELEMENT_SIZE/128, 1), c')
Strides:
(k*W*ELEMENT_SIZE, W*ELEMENT_SIZE, 128, k*h*W*ELEMENT_SIZE)



The splitting strategy for a complete CxW would be as follows:
Split W to multiplication of 128 and reminder. The reminder split to powers of 2.
Split C to the multiplication of 64*h, set c' as the coefficient, k=64 and h = max(128/w, 1) for each chunk
Split the reminder to multiplication of 64, such that h is the coefficient, c'=1 and k=64.
The reminder is of size k, and h=1, c'=1
Each ROI receives a unique descriptor.
This algorithm has a worst case scenario is 16449x255 which produces 8*3 = 24 descriptors with low utilization (writes of 1 byte, or 2 bytes each time)
In conclusion, this algorithm produces maximal of 24 descriptors overall.
 */

static uint32_t highestPowerOf2WithMaximalFreecoefficientLowerThan(uint32_t n, uint32_t maximalcoefficient)
{
    auto proposal = highestOneBit(n);
    while (proposal != 0 && n / proposal <= maximalcoefficient)
    {
        proposal /= 2;
    }
    if (proposal == 0)
    {
        proposal = 1;
    }
    else
    {
        proposal *= 2;
    }
    HB_ASSERT(n / proposal <= maximalcoefficient, "Logical error");
    return proposal;
}

using PhysicalEngineSplitFactors = std::pair<uint32_t, uint32_t>;

struct SimpleRoi
{
    SimpleRoi() = default;
    SimpleRoi(const NodeROI& r)
    {
        std::copy(r.size, r.size + Tensor::c_tensorMaxDim, size);
        std::copy(r.baseOffset, r.baseOffset + Tensor::c_tensorMaxDim, baseOffset);
    }

    TOffset baseOffset[Tensor::c_tensorMaxDim] = {0};
    TSize size[Tensor::c_tensorMaxDim]       = {0};
};

// Small vectors sizes are chosen based on RESNET50 and BERT graphs for Eager to avoid
// any dynamic memory allocations.
constexpr unsigned FACTORIZATIONS_LOCAL_ELEMENTS = 10;
using FactorizationVector                        = llvm_vecsmall::SmallVector<uint32_t, SYN_MAX_TENSOR_DIM>;
using FactorizationsVector        = llvm_vecsmall::SmallVector<FactorizationVector, FACTORIZATIONS_LOCAL_ELEMENTS>;
using SimpleRoiVector             = llvm_vecsmall::SmallVector<SimpleRoi, 5>;
using PhysicalSplitAdditionalData = SimpleRoiVector;
using PhysicalSplitAndAdditionalData = llvm_vecsmall::SmallVector<std::pair<SimpleRoi, SimpleRoiVector>, 5>;
using SplitDimVector              = llvm_vecsmall::SmallVector<DataRange<TSize>, 8>;
using SplitVector                 = llvm_vecsmall::SmallVector<SplitDimVector, SYN_MAX_TENSOR_DIM>;
constexpr unsigned PHYSICAL_ENGINE_SPLIT_FACTORS_LOCAL_ELEMENTS = 5;
using PhysicalEngineSplitFactorsVector =
    llvm_vecsmall::SmallVector<PhysicalEngineSplitFactors, PHYSICAL_ENGINE_SPLIT_FACTORS_LOCAL_ELEMENTS>;

[[maybe_unused]] static PhysicalSplitAdditionalData
splitEvenly(uint32_t cCount, uint32_t wCount, const NodeROI& nodeRoi)
{
    PhysicalSplitAdditionalData data;
    data.reserve(cCount * wCount);
    for (size_t i = 0; i < cCount; i++)
    {
        for (size_t j = 0; j < wCount; j++)
        {
            SimpleRoi sroi(nodeRoi);
            sroi.size[0] /= cCount;
            sroi.baseOffset[0] += sroi.size[0] * i;
            sroi.size[1] /= wCount;
            sroi.baseOffset[1] += sroi.size[1] * j;
            data.push_back(sroi);
        }
    }
    return data;
}

static std::pair<SplitDimVector, TSize> splitW(TSize w, uint32_t splitCount, const DmaTransposeHelper& helper)
{
    std::pair<SplitDimVector, TSize> splitsPair = {};
    auto& [splits, count]                       = splitsPair;
    // First, split W dimension, to multiples of chunks size.
    if (w >= helper.chunkSizeDim1())
    {
        count  = w - w % helper.chunkSizeDim1();
        DataRange<TSize> fullRange {0, count};
        fullRange.splitEvenlyAsBestAsPossible(splits, splitCount, helper.chunkSizeDim1());
    }
    else
    {
        count = highestOneBit(w);
        DataRange<TSize> fullRange {0, count};
        splits.push_back(fullRange);
    }
    return splitsPair;
}

static std::pair<SplitDimVector, TSize> splitC(TSize c, TSize w, uint32_t splitCount, const DmaTransposeHelper& helper)
{
    HB_ASSERT(w % helper.chunkSizeDim1() == 0 || isPowerOf2(w),
              "Expecting {} to be multiple of chunk size or a power of 2",
              w);
    HB_ASSERT(c != 0, "Expecting {} != 0", c);

    auto                             h     = std::max<uint32_t>(helper.maximalDestElementsDim0() / w, 1u);
    std::pair<SplitDimVector, TSize> splitsPair = {};
    auto& [splits, count]                       = splitsPair;
    // First, split W dimension, to multiples of chunks size.

    auto maxSrcDim0 = helper.maxSrcDim0() * splitCount;
    // First we check if we need more than 1 TE invocation (it's a specific structure)
    if (c >= maxSrcDim0 * h)
    {
        count  = c - c % (maxSrcDim0 * h);
        DataRange<TSize> fullRange {0, count};
        fullRange.splitEvenlyAsBestAsPossible(splits, splitCount, helper.maxSrcDim0() * h);
    }
    else if (c >= maxSrcDim0)
    {
        h      = c / maxSrcDim0;
        count  = h * maxSrcDim0;
        DataRange<TSize> fullRange {0, count};
        fullRange.splitEvenlyAsBestAsPossible(splits, splitCount, helper.maxSrcDim0() * h);
    }
    else
    {
        // h = 1
        count  = c;
        DataRange<TSize> fullRange {0, count};
        fullRange.splitEvenlyAsBestAsPossible(splits, splitCount, helper.maxSrcDim0());
    }
    return splitsPair;
}

static void splitTransposeLowDescriptorCount(const Node&                     node,
                                             const NodeROI&                  inputRoi,
                                             PhysicalEngineSplitFactors      coefficient,
                                             DmaTransposeEngineParams        teParams,
                                             PhysicalSplitAndAdditionalData& output)
{
    // Note: This could be improved (for more info, talk with Chen Koren)
    using SplitNodeROIVector = llvm_vecsmall::SmallVector<std::pair<SimpleRoi, SplitDimVector>, 3>;
    SplitNodeROIVector splittedW;  // Contains splitted w ROIs and physical split data
    SimpleRoi          leftOver(inputRoi);

    DmaTransposeHelper helper(node.getInput(0)->getElementType(), teParams);
    while (leftOver.size[1] != 0)
    {
        auto    splitSizesAndTotal = splitW(leftOver.size[1], coefficient.second, helper);
        splittedW.emplace_back(leftOver, splitSizesAndTotal.first);
        SimpleRoi& powerOf2        = splittedW.back().first;
        powerOf2.size[1]           = splitSizesAndTotal.second;
        leftOver.baseOffset[1] += powerOf2.size[1];
        leftOver.size[1] -= powerOf2.size[1];
    }

    // For each split splitted W we then split on C and output it.
    for (const auto& roiAndSplit : splittedW)
    {
        const auto& roi = roiAndSplit.first;
        leftOver  = roi;
        while (leftOver.size[0] != 0)
        {
            auto    splittedC = splitC(leftOver.size[0], leftOver.size[1], coefficient.first, helper);
            output.emplace_back(leftOver, PhysicalSplitAdditionalData());
            auto& [chunk, additionalData] = output.back();
            chunk.size[0] = splittedC.second;
            leftOver.baseOffset[0] += chunk.size[0];
            leftOver.size[0] -= chunk.size[0];
            additionalData.reserve(splittedC.first.size() * roiAndSplit.second.size());
            for (auto& cSegment : splittedC.first)
            {
                for (auto& wSegment : roiAndSplit.second)
                {
                    additionalData.push_back(chunk);
                    SimpleRoi& sroi = additionalData.back();
                    sroi.baseOffset[0] += cSegment.start();
                    sroi.size[0] = cSegment.size();
                    sroi.baseOffset[1] += wSegment.start();
                    sroi.size[1] = wSegment.size();
                }
            }
        }
    }
}

// We can prefer to split on dim 0 or dim 1. This is all the ways we can have (x, y) such that x*y <= count
static PhysicalEngineSplitFactorsVector getAllFactorizationToTwoFactors(uint32_t count)
{
    PhysicalEngineSplitFactorsVector ret;
    uint32_t                         bound            = sqrt(count);
    bool                             isRealSquareRoot = (bound * bound == count);
    ret.reserve(2 * bound);
    if (isRealSquareRoot) bound--;  // we'll handle the last element separatly
    for (uint32_t i = 1; i <= bound; i++)
    {
        uint32_t complement = count / i;
        ret.emplace_back(i, complement);
        ret.emplace_back(complement, i);
    }
    // now add the last split in case it was a real square root
    if (isRealSquareRoot)
    {
        bound++;
        ret.emplace_back(bound, bound);
    }
    return ret;
}

// Lower is better.
static uint64_t teTotalWork(const Node&                           node,
                            const NodeROI&                        inputRoi,
                            const PhysicalSplitAndAdditionalData& physicalSplitAndAdditionalData,
                            PhysicalEngineSplitFactors            strategy,
                            DmaTransposeEngineParams              teParams)
{
    uint64_t           count = 0;
    DmaTransposeHelper helper(node.getInput(0)->getElementType(), teParams);

    TSize maxSrcDim0              = helper.maxSrcDim0();
    TSize maximalDestElementsDim0 = helper.maximalDestElementsDim0();
    for (const auto& splitPair : physicalSplitAndAdditionalData)
    {
        const auto& roi = splitPair.first;
        if (inputRoi.size[0] < maxSrcDim0)
        {
            count += roi.size[1] / strategy.second;
        }
        else
        {
            auto w = highestPowerOf2WithMaximalFreecoefficientLowerThan(roi.size[1], strategy.second);  // w writes
            count += maximalDestElementsDim0 * (roi.size[1] / w) * roi.size[0];
        }
    }
    return count;
}

/**
 * @brief Splits the inputROI to multiple ROIs based on a simpler version of an ROI.
 *
 * @param inputRoi The ROI to split
 * @param simpleRois The list of simple ROIs that are just for the input.
 * @param transposePerm The permutation that needs to be held.
 * @param outputContainer The container to which all the ROIs will be written
 */
static void splitNodeRoiBasedOnSimpleRoi(const NodeROI&                   inputRoi,
                                         const SimpleRoiVector&           simpleRois,
                                         const TransposePermutationArray& transposePerm,
                                         NodeROIContainer&                outputContainer)
{
    for (auto& roi : simpleRois)
    {
        outputContainer.push_back(inputRoi);
        NodeROI& added = outputContainer.back();

        std::copy(roi.size, roi.size + Tensor::c_tensorMaxDim, added.size);
        std::copy(roi.baseOffset, roi.baseOffset + Tensor::c_tensorMaxDim, added.baseOffset);

        HB_ASSERT(!added.inputRois.empty(), "Invalid input ROIs");
        std::copy(roi.size, roi.size + Tensor::c_tensorMaxDim, added.inputRois[0].getLayout().m_size.data());
        std::copy(roi.baseOffset, roi.baseOffset + Tensor::c_tensorMaxDim, added.inputRois[0].getLayout().m_baseOffset);

        HB_ASSERT(!added.outputRois.empty(), "Invalid output ROIs");
        applyPermutation(roi.size, transposePerm, added.outputRois[0].getLayout().m_size.data());
        applyPermutation(roi.baseOffset, transposePerm, added.outputRois[0].getLayout().m_baseOffset);
    }
}

void SplitTransposeToLowDescriptorCount::splitLogical(const Node&       node,
                                                      const NodeROI&    inputRoi,
                                                      uint32_t          logicalEngineCount,
                                                      uint32_t          futurePhysicalEngineCount,
                                                      NodeROIContainer& outputContainer)
{
    auto copy = inputRoi;
    unsigned multiply = 1;
    if (node.getInput(0)->getElementSizeInBits() == 4)
    {
        multiply = 2;
    }
    copy.size[0] /= multiply;
    copy.size[1] /= multiply;
    copy.baseOffset[0] /= multiply;
    copy.baseOffset[1] /= multiply;
    DmaTransposeHelper helper = getHelperForNode(node);
    // Remove unapplicable dimensions
    uint32_t removedDims = 0;
    if (copy.size[1] < helper.chunkSizeDim1() * futurePhysicalEngineCount)
    {
        removedDims++;
        m_prefferedDimensionsOrder.erase(
            std::remove(m_prefferedDimensionsOrder.begin(), m_prefferedDimensionsOrder.end(), 1),
            m_prefferedDimensionsOrder.end());
    }
    if (copy.size[0] < helper.maxSrcDim0() * futurePhysicalEngineCount)
    {
        removedDims++;
        m_prefferedDimensionsOrder.erase(
            std::remove(m_prefferedDimensionsOrder.begin(), m_prefferedDimensionsOrder.end(), 0),
            m_prefferedDimensionsOrder.end());
    }
    if (m_prefferedDimensionsOrder.empty() || removedDims == 2)
    {
        // No split anyway (or split on a dim with size 1)
        outputContainer.push_back(copy);
        return;
    }

    uint32_t optimizedNumOfChunks =
        SplitTransposeToLowDescriptorCount::optimizeNumOfChunksNum(node, inputRoi, logicalEngineCount);

    auto ret = ROISplitter::splitFullRoiToLogicalRoisAlongExternalAxis(copy,
                                                                       m_prefferedDimensionsOrder,
                                                                       optimizedNumOfChunks,
                                                                       node.getNodeName());
    for (auto& element : ret)
    {
        element.size[0] *= multiply;
        element.baseOffset[0] *= multiply;
        element.size[1] *= multiply;
        element.baseOffset[1] *= multiply;
    }

    outputContainer.splice(outputContainer.end(),
                           std::move(ret));
}

// In case we can reduce logical pipeline chunks and to get chunks with 100% HW CL utilization we will prefer it.
uint32_t SplitTransposeToLowDescriptorCount::optimizeNumOfChunksNum(const Node&    node,
                                                                    const NodeROI& inputRoi,
                                                                    uint32_t       logicalEngineCount)
{
    // The dim we are take into account is the first in m_prefferedDimensionsOrder
    TSize firstPrefferedDimSizeInElements = inputRoi.size[m_prefferedDimensionsOrder[0]];

    DmaTransposeHelper helper = getHelperForNode(node);
    uint64_t srcLineSize = helper.chunkSizeDim1();
    if  (m_prefferedDimensionsOrder[0] == 0)
    {
        srcLineSize = helper.maxSrcDim0();
    }

    if (firstPrefferedDimSizeInElements / logicalEngineCount % srcLineSize == 0) return logicalEngineCount;

    for (int numOfChunks = logicalEngineCount - 1; numOfChunks >= GCFG_DEFAULT_PIPELINE_DEPTH.value();
         numOfChunks--)
    {
        if (numOfChunks != 0 && firstPrefferedDimSizeInElements / numOfChunks % srcLineSize == 0)
        {
            return numOfChunks;
        }
    }
    return logicalEngineCount;
}

void SplitTransposeToLowDescriptorCount::splitRoiToEngines(const Node&       node,
                                                           const NodeROI&    inputRoi,
                                                           uint32_t          physicalEngineCount,
                                                           NodeROIContainer& outputContainer)
{
    // The algorithm `splitTransposeLowDescriptorCount` receives 2 factor:
    // dim 0 factor and dim 1 factor.
    // For the dim 0 factor, splitTransposeLowDescriptorCount will return ROIs that must be splitted across dim 0 by a number lower or equal to the factor,
    // and the same goes to dim 1 factor.
    // Across all possible ways to have dim 1 and dim 0 factors, we choose the one with the highest utilization.
    // This doesn't provide the fastest solution. We are bound here (due to gc design) to provide a single ROI after the logical split.
    // If we fix this requirement, and could set buckets of descriptors per engine per pipeline-level (an not do round robbin),
    // we could be closer to the optimal solution.
    PhysicalEngineSplitFactorsVector strategies = getAllFactorizationToTwoFactors(physicalEngineCount);
    using NodeROISplitsVector =
        llvm_vecsmall::SmallVector<PhysicalSplitAndAdditionalData, PHYSICAL_ENGINE_SPLIT_FACTORS_LOCAL_ELEMENTS>;
    NodeROISplitsVector splits;
    splits.reserve(strategies.size());
    using TotalWorkVector = llvm_vecsmall::SmallVector<uint64_t, PHYSICAL_ENGINE_SPLIT_FACTORS_LOCAL_ELEMENTS>;
    TotalWorkVector totalWork;
    totalWork.reserve(strategies.size());
    for (auto& strategy : strategies)
    {
        splits.emplace_back();
        splitTransposeLowDescriptorCount(node, inputRoi, strategy, m_teParams, splits.back());
        totalWork.push_back(teTotalWork(node, inputRoi, splits.back(), strategy, m_teParams));
    }
    auto offset = std::distance(totalWork.begin(), std::min_element(totalWork.begin(), totalWork.end()));

    for (const auto& [simpleRoiSplit, roiSplitList] : splits[offset])
    {
        splitNodeRoiBasedOnSimpleRoi(inputRoi,
                                     roiSplitList.empty() ? SimpleRoiVector {simpleRoiSplit} : roiSplitList,
                                     m_permutation,
                                     outputContainer);
    }
}

void SplitTransposeToLowDescriptorCount::finalize(const Node& node, NodeROI& roi)
{
    // w should be either a multiplication of 128 or a power of 2.
    DmaTransposeHelper helper = getHelperForNode(node);
    TSize              maxSrcDim0              = helper.maxSrcDim0();
    TSize              maximalDestElementsDim0 = helper.maximalDestElementsDim0();
    auto               maxH   = std::max(helper.maximalDestElementsDim0() / roi.size[1], 1ul);
    TSize              w      = roi.size[1];
    TSize              k                       = std::min(roi.size[0], static_cast<TSize>(maxSrcDim0));
    TSize              h                       = std::clamp<TSize>(roi.size[0] / maxSrcDim0, 1, maxH);
    TSize              cTag                    = std::max<TSize>(1, roi.size[0] / (maxSrcDim0 * maxH));
    HB_ASSERT(cTag * h * k == roi.size[0],
              "Expecting {} to be divisable to dimensions ({}, {}, {})",
              roi.size[0],
              k,
              h,
              cTag);

    // fixate maxH?
    TensorROILayout& inputLayout = roi.inputRois[0].m_layout;
    inputLayout.m_size[0]        = k;
    inputLayout.m_size[1]        = w;
    inputLayout.m_size[2]        = h;
    inputLayout.m_size[3]        = cTag;
    inputLayout.m_size[4]        = 1;
    // Spatial[0] stays the same
    auto elementSize = node.getInput(0)->getElementSizeInBits();
    inputLayout.spatialStrides[1] = safeBitsToByte(maxSrcDim0 * elementSize);
    inputLayout.spatialStrides[2] = inputLayout.spatialStrides[1] * maxH;
    inputLayout.spatialStrides[3] = inputLayout.spatialStrides[0] * w;

    TensorROILayout& outputLayout  = roi.outputRois[0].m_layout;
    outputLayout.m_size[0]         = std::min<TSize>(maximalDestElementsDim0, w);
    outputLayout.m_size[1]         = h;
    outputLayout.m_size[2]         = k;
    outputLayout.m_size[3]         = std::max(1ul, w / maximalDestElementsDim0);
    outputLayout.m_size[4]         = cTag;
    outputLayout.spatialStrides[1] = outputLayout.spatialStrides[0];
    outputLayout.spatialStrides[0] = maxSrcDim0 * outputLayout.spatialStrides[0];
    outputLayout.spatialStrides[2] = helper.maximalDestDim0Bytes();
    outputLayout.spatialStrides[3] = outputLayout.spatialStrides[0] * h;
    helper.checkInOut(roi.inputRois[0], roi.outputRois[0]);
}

/**
 * @brief all the ways we can have (x, y..) such that x*y*... <= count
 *
 * @param count The maximal value of the multiplication of all factors
 * @param factorCount The number of factors
 * @return FactorizationsVector List of all possible factors that their multiplication is less than count.
 */
static FactorizationsVector getAllFactorizationToFactors(uint32_t count, uint32_t factorCount)
{
    FactorizationsVector factors;
    if (factorCount == 1)
    {
        factors.emplace_back(std::initializer_list<uint32_t> {count});
        return factors;
    }
    if (count == 1)
    {
        factors.emplace_back(factorCount, 1);
        return factors;
    }
    for (uint32_t i = 1; i <= count; i++)
    {
        auto oneLess = getAllFactorizationToFactors(count / i, factorCount - 1);
        for (auto& factorsWithOneLess : oneLess)
        {
            factorsWithOneLess.push_back(i);
        }
        std::move(oneLess.begin(), oneLess.end(), std::back_inserter(factors));
    }
    return factors;
}

/// less is better
static std::pair<TSize, TSize> cost(const FactorizationVector& factorization, const TSize* sizes, uint32_t engines)
{
    // Small vectors sizes are chosen based on RESNET50 and BERT graphs for Eager to avoid
    // any dynamic memory allocations.
    using MaxSizesVector  = llvm_vecsmall::SmallVector<TSize, 3>;
    using SizeArrayVector = llvm_vecsmall::SmallVector<TSize, 5>;

    SplitVector split;
    split.resize(factorization.size());
    for (size_t i = 0; i < factorization.size(); i++)
    {
        DataRange<TSize> splitedDim {0, sizes[i]};
        splitedDim.splitEvenlyAsBestAsPossible(split[i], factorization[i], 1);
    }
    MaxSizesVector maxSizes;
    maxSizes.reserve(factorization.size());
    std::transform(split.begin(),
                   split.end(),
                   std::back_inserter(maxSizes),
                   [](const SplitDimVector& elements) {
                       return std::max_element(elements.begin(),
                                               elements.end(),
                                               [](const DataRange<TSize>& a, const DataRange<TSize>& b) {
                                                   return a.size() < b.size();
                                               })
                           ->size();
                   });
    TSize longestTime            = multiplyElements(maxSizes);
    TSize optimalAverageTime     = std::max<TSize>(multiplyElements(sizes, sizes + factorization.size()) / engines, 1);
    auto  sizeArrayReservation =
        std::accumulate(split.begin(), split.end(), 1ULL, [](uint64_t acc, const auto& splitDim) {
            return acc * splitDim.size();
        });
    SizeArrayVector sizeArray;
    sizeArray.reserve(sizeArrayReservation);
    sizeArray.push_back(1);
    for (const auto& splitDim : split)
    {
        // we rely on sizeArray previous values so can't mutate it yet.
        size_t sizeArrayEnd = sizeArray.size();
        for (size_t splitDimIdx = 1; splitDimIdx < splitDim.size(); splitDimIdx++)
        {
            auto rangeSize = splitDim[splitDimIdx].size();
            for (size_t sizeArrayIdx = 0; sizeArrayIdx < sizeArrayEnd; sizeArrayIdx++)
            {
                sizeArray.push_back(sizeArray[sizeArrayIdx] * rangeSize);
            }
        }
        // mutate sizeArray previous values.
        auto rangeSize = splitDim[0].size();
        for (size_t sizeArrayIdx = 0; sizeArrayIdx < sizeArrayEnd; sizeArrayIdx++)
        {
            sizeArray[sizeArrayIdx] *= rangeSize;
        }
    }
    TSize distanceFromOptimal = 0;
    for (const auto& size : sizeArray)
    {
        if (optimalAverageTime >= size)
        {
            distanceFromOptimal += optimalAverageTime - size;
        }
        else
        {
            distanceFromOptimal += size - optimalAverageTime;
        }
    }
    return {longestTime, distanceFromOptimal};
}

void SplitFullyUtilizedTranspose::splitLogical(const Node&       node,
                                               const NodeROI&    inputRoi,
                                               uint32_t          logicalEngineCount,
                                               uint32_t          /*futurePhysicalEngineCount*/,
                                               NodeROIContainer& outputContainer)
{
    DmaTransposeHelper helper(node.getInput(0)->getElementType(), m_teParams);
    uint32_t minIndex = 0;
    TSize              minIndexChunkSize, minIndexRemainder;
    std::tie(minIndex, minIndexChunkSize, minIndexRemainder) = getSizeAfterUtilizationProtection(inputRoi.size, Tensor::c_tensorMaxDim, helper);
    auto copy                                                = inputRoi;
    bool is4bit = node.getInput(0)->getElementSizeInBits() == 4;
    // For fully utilized scenario we must write 128 bytes each time
    // therefore, we are setting an NodeROI such that each element represents an indivisable amount of work.
    for (auto i = 0; i < Tensor::c_tensorMaxDim; i++)
    {
        if (i == 0)
        {
            if (is4bit)
            {
                copy.size[i] = inputRoi.size[i] / 2;
            }
            else
            {
                copy.size[i] = inputRoi.size[i];
            }
        }
        else if (i < minIndex)
        {
            copy.size[i] = 1;
        }
        else if (i == minIndex)
        {
            copy.size[i] = minIndexRemainder;
        }
        else
        {
            copy.size[i] = inputRoi.size[i];
        }
        copy.baseOffset[i] = 0;
    }
    auto partialLogicalRois = ROISplitter::splitFullRoiToLogicalRoisAlongExternalAxis(copy,
                                                                                      m_prefferedDimensionsOrder,
                                                                                      logicalEngineCount,
                                                                                      node.getNodeName());
    for (auto& partial : partialLogicalRois)
    {
        for (auto i = 0; i < Tensor::c_tensorMaxDim; i++)
        {
            if (i == 0)
            {
                if (is4bit)
                {
                    partial.size[i] *= 2;
                    partial.baseOffset[i] = partial.baseOffset[i] * 2 + inputRoi.baseOffset[i];
                }
                else
                {
                    partial.baseOffset[i] += inputRoi.baseOffset[i];
                }
            }
            else if (i < minIndex)
            {
                partial.size[i] *= inputRoi.size[i];
                partial.baseOffset[i] = partial.baseOffset[i] * inputRoi.size[i] + inputRoi.baseOffset[i];
            }
            else if (i == minIndex)
            {
                partial.baseOffset[i] = partial.baseOffset[i] * minIndexChunkSize + inputRoi.baseOffset[i];
                partial.size[i] *= minIndexChunkSize;
            }
            else
            {
                partial.baseOffset[i] += inputRoi.baseOffset[i];
            }
        }
    }
    outputContainer.splice(outputContainer.end(), std::move(partialLogicalRois));
}

void SplitFullyUtilizedTranspose::splitRoiToEngines(const Node&       node,
                                                    const NodeROI&    inputRoi,
                                                    uint32_t          physicalEngineCount,
                                                    NodeROIContainer& outputContainer)
{
    // Input assumptions:
    // We have a DMA transpose node that wishes to change the input dimension one of 1,2,3 dimensions (zero-based), we leave the other dimensions
    // in the same order and we do not change them.
    // We assume the multiplications of all the dimensions that change faster than the old FCD (in the new transposed tensor)
    // is a multiplication of 128 bytes (or 64 bytes when we deal with 4bit)
    // Out of the constraints, we must provide an ROI that have these restrictions:
    // There is a contiguous chunks of 128 B (or 64B for 4bit)
    DmaTransposeHelper helper(node.getInput(0)->getElementType(), m_teParams);
    HB_ASSERT(helper.numLines(inputRoi.size) % m_teParams.numLinesDivisor == 0,
              "num lines ({}) must divide divisor ({})",
              helper.numLines(inputRoi.size),
              m_teParams.numLinesDivisor);
    DataRange<TSize> firstDimRange(inputRoi.baseOffset[0], inputRoi.baseOffset[0] + inputRoi.size[0]);
    SplitDimVector   firstDimChunks;
    firstDimRange.splitToChunks(firstDimChunks, helper.maxSrcDim0());
    physicalEngineCount = std::max<uint32_t>(physicalEngineCount / firstDimChunks.size(), 1);
    auto inputRoiCopy = inputRoi;
    for(auto& firstDimChunk : firstDimChunks)
    {
        inputRoiCopy.size[0]       = firstDimChunk.size();
        inputRoiCopy.baseOffset[0] = firstDimChunk.start();
        auto maxIndex = Tensor::c_tensorMaxDim - 1;
        while (inputRoiCopy.size[maxIndex] == 1)
        {
            --maxIndex;
        }
        uint32_t minIndex = 0;
        TSize    minIndexDivisor, minIndexRemainder;
        std::tie(minIndex, minIndexDivisor, minIndexRemainder) = getSizeAfterUtilizationProtection(inputRoiCopy.size, maxIndex + 1, helper);
        if (minIndex == maxIndex && minIndexRemainder == 1)
        {
            splitNodeRoiBasedOnSimpleRoi(inputRoiCopy,
                                         SimpleRoiVector {SimpleRoi(inputRoiCopy)},
                                         m_permutation,
                                         outputContainer);
            continue;
        }
        SizeVector splitableSizes;
        splitableSizes.reserve(maxIndex - minIndex + 1);
        splitableSizes.push_back(minIndexRemainder);
        for (size_t i = minIndex + 1; i <= maxIndex; i++)
        {
            splitableSizes.push_back(inputRoiCopy.size[i]);
        }
        FactorizationsVector factorizations = getAllFactorizationToFactors(physicalEngineCount, splitableSizes.size());
        using CostsVector = llvm_vecsmall::SmallVector<std::pair<TSize, TSize>, FACTORIZATIONS_LOCAL_ELEMENTS>;
        CostsVector costs;
        costs.reserve(factorizations.size());
        std::transform(factorizations.begin(),
                       factorizations.end(),
                       std::back_inserter(costs),
                       [&](const FactorizationVector& factorization) {
                           return cost(factorization, splitableSizes.data(), physicalEngineCount);
                       });
        auto bestFactorizationIndex = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
        const auto& factorization          = factorizations[bestFactorizationIndex];
        SplitVector split;
        split.resize(factorization.size());
        for (size_t i = 0; i < factorization.size(); i++)
        {
            DataRange<TSize> splittedDim {0, splitableSizes[i]};
            splittedDim.splitEvenlyAsBestAsPossible(split[i], factorization[i], 1);
        }
        for (auto& range : split[0]) // this is minIndex
        {
            range.end(range.end() * minIndexDivisor);
            range.start(range.start() * minIndexDivisor);
        }
        SimpleRoiVector simpleRois;
        auto            simpleRoisReservation =
            std::accumulate(split.begin(),
                            split.begin() + maxIndex - minIndex + 1,
                            1ULL,
                            [](uint64_t acc, const auto& splitDim) { return acc * splitDim.size(); });
        simpleRois.reserve(simpleRoisReservation);
        simpleRois.emplace_back(inputRoiCopy);
        for (uint32_t i = minIndex, splitIdx = 0; i <= maxIndex; i++, splitIdx++)
        {
            const auto& currentSplit = split[splitIdx];
            // we rely on simpleRois previous values so can't mutate it yet.
            size_t simpleRoisEnd = simpleRois.size();
            size_t rangeEnd      = currentSplit.size();
            for (size_t rangeIdx = 1; rangeIdx < rangeEnd; rangeIdx++)
            {
                const auto& range = currentSplit[rangeIdx];
                for (size_t simpleRoisIdx = 0; simpleRoisIdx < simpleRoisEnd; simpleRoisIdx++)
                {
                    simpleRois.push_back(simpleRois[simpleRoisIdx]);
                    auto& copy   = simpleRois.back();
                    copy.size[i] = range.size();
                    copy.baseOffset[i] += range.start();
                }
            }
            // mutate simpleRois previous values.
            const auto& range = currentSplit[0];
            for (size_t simpleRoisIdx = 0; simpleRoisIdx < simpleRoisEnd; simpleRoisIdx++)
            {
                simpleRois[simpleRoisIdx].size[i] = range.size();
                simpleRois[simpleRoisIdx].baseOffset[i] += range.start();
            }
        }
        splitNodeRoiBasedOnSimpleRoi(inputRoiCopy, simpleRois, m_permutation, outputContainer);
    }
}

void SplitFullyUtilizedTranspose::finalize(const Node& node, NodeROI& ret)
{
    auto& transposeNode = static_cast<const DMATransposeNode&>(node);
    auto cIndex = index_of(transposeNode.permutation(), TPD_Channel);

    auto helper = getHelperForNode(node);
    // (we only have one transpose for multidimensional tensors, which is the fully utilized)
    // Assuming:
    // The new FCD is the old's 2nd FCD.
    // The order of the dimensions except the old FCD stays the same.
    // There might've been a logical transpose prior to this transpose
    uint32_t minIndex;
    std::tie(minIndex, std::ignore, std::ignore) =
        getSizeAfterUtilizationProtection(ret.size, Tensor::c_tensorMaxDim, helper);
    auto outDimensionsToFlatten = minIndex;
    auto deletedDimsCount       = minIndex - 1;
    HB_ASSERT(cIndex >= outDimensionsToFlatten, "Expecting {} >= {}", cIndex, outDimensionsToFlatten);

    if (deletedDimsCount == 0)
    {
        const auto&            inputROI             = ret.inputRois[0];
        const TensorROILayout& inputTensorROILayout = inputROI.getLayout();
        const auto&            inTensor             = inputROI.m_parentTensor;
        const NSizeArray&      inSizes              = inTensor->getAllNSizesInElements();
        helper.transposeByCIndex(cIndex, inTensor->getElementType(), inputTensorROILayout, inSizes, ret.outputRois[0]);
    }
    else
    {
        // We need to "flatten" the output. Since we are best at writing 128B each time, we set
        // the DMA to actually write contiguous 128B each time. For this to happen,
        // we need to flatten the output. The value outDimensionsToFlatten is how many dimensions we need to
        // take into considiration for the flattening to occure.
        // The logic below just calculates the sizes of the input tensor as if it was flattened,
        // then creates a temporary input tensor, and transpose that "fake" input tensor.
        // In reality, we just use the input since the output is created exlusively by the sizes and strides
        // of the input tensor.
        const auto&            inputROI                 = ret.inputRois[0];
        const TensorROILayout& origInputTensorROILayout = inputROI.getLayout();
        const auto&            inTensor                 = inputROI.m_parentTensor;
        const NSizeArray&      origInSizes              = inTensor->getAllNSizesInElements();
        unsigned               dims                     = inTensor->getDim();
        NSizeArray             inSizes                  = {};
        inSizes[0]                                      = inTensor->getSizeInElements(0);
        inSizes[1] = multiplyElements(origInSizes.begin() + 1, origInSizes.begin() + outDimensionsToFlatten + 1);
        TensorROILayout inputTensorROILayout = origInputTensorROILayout;
        inputTensorROILayout.m_size[1] =
            multiplyElements(inputTensorROILayout.m_size.data() + 1,
                             inputTensorROILayout.m_size.data() + outDimensionsToFlatten + 1);
        size_t unflattanedDims = dims - deletedDimsCount;
        for (size_t i = 2; i < unflattanedDims; i++)
        {
            size_t origDim                 = i + deletedDimsCount;
            inSizes[i]                     = origInSizes[origDim];
            inputTensorROILayout.m_size[i] = origInputTensorROILayout.m_size[origDim];
        }
        std::fill(inSizes.begin() + unflattanedDims, inSizes.begin() + dims, 1);
        std::fill(std::begin(inputTensorROILayout.m_size) + unflattanedDims,
                  std::begin(inputTensorROILayout.m_size) + dims,
                  1);
        // Since we flattened outputs that are before cIndex, we update the value of cIndex.
        cIndex -= deletedDimsCount;
        helper.transposeByCIndex(cIndex, inTensor->getElementType(), inputTensorROILayout, inSizes, ret.outputRois[0]);
    }
}

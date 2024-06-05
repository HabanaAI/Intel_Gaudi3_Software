#include "calculate_tensor_roi_linear_ranges.h"

#include "habana_graph.h"
#include "habana_nodes.h"
#include "log_manager.h"
#include "node_roi.h"
#include "physical_memory_ops_nodes.h"
#include "include/sync/data_range.h"
#include "tensor_roi.h"
#include "include/mme_common/mme_descriptor_generator_base.h"

template<class C, class V>
static bool contains(const C& c, const V& v)
{
    return std::find(std::begin(c), std::end(c), v) != std::end(c);
}

static uint64_t getRealSizeInBytes(const pTensor& tensor)
{
    return Tensor::getRealTensor(tensor)->getTotalSizeInBytes();
}

class MemoryRangesCalculator
{
public:
    MemoryRangesCalculator(TensorROI& tRoi, const NodePtr& n, bool isInput);
    void calculateMemoryRanges();

    static constexpr int     MIN_RANGES_FOR_CYCLIC = 8;  // minimum number of linear ranges per cyclic range
    static constexpr int32_t MAX_STRIDE_SIZE       = 1024 * 1024 * 8;  // maximal stride for cyclic range

private:
    void     calculateSortedStridesWithMatchingSizes();
    void     calculateLinearRanges();
    void     calculateCyclicRanges();
    void     calculateNumberOfLinearRangesPerCycle();
    void     calculateDenseSizeAndDim();
    void     calculateCyclicDim();
    bool     isParentZeroSizedTensor();
    uint64_t getExpectedNumberOfLinearRanges();
    uint64_t getExpectedNumberOfCyclicRanges();

    TensorROI&         m_tRoi;
    const unsigned int m_tensorDim;
    const uint64_t     m_parentStart;

    TensorSizesVector   m_dimSizesSortedByStrides;
    TensorStridesVector m_dimStridesSorted;
    uint64_t            m_denseSize;          // size of a continuous memory block in the Tensor ROI
    unsigned            m_lowestNonDenseDim;  // lowest dimension that consists of more than one continious memory block
    unsigned            m_lowestNonCyclicDim;  // lowest dimension that consists of more than one cyclic memory block
    unsigned            m_baseCyclicDim;       // the dimension that defines the stride of the cyclic ranges
    uint64_t            m_numLinearRangesPerCycle;  // number of continuous memory blocks each cyclic range consists of
    uint32_t            m_cyclicStrideSize;         // stride size of the cyclic range
};

void CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(TensorROI& tRoi)
{
    /*
        conversion from linear ranges to cyclic ranges.
        for each subRoi:
        take the linear ranges and try to "absorb" them into cyclic ranges.
        we assume the linear ranges are sorted by start address.
        either we convert all of the linear ranges to cyclic ranges, or none of them.
        the hyper parameters are pretty arbitrary - how many ranges should we convert to a single cyclic range for it to
       be "worth it".
    */

    static const int     MIN_RANGES_FOR_CYCLIC = 8;                // minimum number of linear ranges per cyclic range
    static const int32_t MAX_STRIDE_SIZE       = 1024 * 1024 * 8;  // maximal stride for cyclic range
    for (OverlapSubRoi& subRoi : *tRoi.m_overlapRoi.subRois)
    {
        if (!subRoi.cyclicRanges.empty()) continue;  // already calculated cyclic ranges
        uint32_t numRanges = subRoi.ranges.size();
        // if we have many range, try unite them into a cyclic range
        if (numRanges < MIN_RANGES_FOR_CYCLIC) continue;

        bool isSorted = std::is_sorted(subRoi.ranges.begin(), subRoi.ranges.end());
        if (!isSorted)
        {
            std::sort(subRoi.ranges.begin(), subRoi.ranges.end());
        }

        // new cyclic ranges are built from range bounds, and cyclic ranges
        std::vector<DataRange<uint64_t>> cyclicRangeBounds;
        std::vector<CyclicDataRange>     cyclicRanges;

        bool success = false;  // if we manages to "absorb" all the ranges
        int  i       = 0;
        while (i < numRanges)
        {
            const auto& range = subRoi.ranges[i];
            // find stride to next range.
            uint64_t stride = i + 1 < numRanges ? subRoi.ranges[i + 1].start() - range.start() : 0;
            int      j      = 1;
            if (stride <= MAX_STRIDE_SIZE && stride > 0)
            {
                // continue to "absorb" linear ranges that apply the cyclic parameters rules:
                while ((i + j < numRanges) &&                            // they exist
                       (range.size() == subRoi.ranges[i + j].size()) &&  // have the same size
                       (subRoi.ranges[i + j].start() - range.start() ==
                        j * stride))  // start at the same point (mod stride)
                {
                    j++;
                }
            }

            if (j < MIN_RANGES_FOR_CYCLIC)  // if we couldn't absorb enough linear ranges
            {
                LOG_DEBUG(ROI_RANGE, "could not find cyclic ranges for subroi.. using regular ranges");
                success = false;
                break;
            }
            // save the new linear ranges as cyclic ranges
            success = true;
            cyclicRangeBounds.emplace_back(subRoi.ranges[i].start(), subRoi.ranges[i + j - 1].end());
            uint64_t cyclicRangeSize = subRoi.ranges[i].size();
            if (cyclicRangeSize > stride)
            {
                LOG_WARN(ROI_RANGE, "overlapping ranges in tensor ROI for tensor {}", tRoi.m_parentTensor->getName());
                cyclicRangeSize = stride;
            }
            cyclicRanges.emplace_back(subRoi.ranges[i].start(), subRoi.ranges[i].start() + cyclicRangeSize, stride);
            i += j;
        }

        if (success)
        {
            subRoi.ranges       = cyclicRangeBounds;
            subRoi.cyclicRanges = cyclicRanges;

            if (LOG_LEVEL_AT_LEAST_TRACE(ROI_RANGE))
            {
                for (uint32_t i = 0; i < subRoi.cyclicRanges.size(); i++)
                {
                    LOG_TRACE(ROI_RANGE,
                              "tensor: {}, range {}-{}, moduluBase: {}, start: {}, end: {}",
                              tRoi.m_parentTensor->getName(),
                              cyclicRangeBounds[i].start(),
                              cyclicRangeBounds[i].end(),
                              subRoi.cyclicRanges[i].stride(),
                              subRoi.cyclicRanges[i].start(),
                              subRoi.cyclicRanges[i].end());
                }
            }
        }
    }
}

static void createCyclicRanges(HabanaGraph& g, const NodePtr& n)
{
    if (!GCFG_ENABLE_CYCLIC_RANGES.value()) return;

    std::list<NodeROI>& nodeRois = *g.GetNodeROIs(n);
    for (NodeROI& roi : nodeRois)
    {
        for (TensorROI& tRoi : roi.inputRois)
        {
            CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi);
        }
        for (TensorROI& tRoi : roi.outputRois)
        {
            CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi);
        }
    }
}

// merge linear ranges that are continuous
static void squeezeRanges(std::vector<DataRange<uint64_t>>& ranges)
{
    std::sort(ranges.begin(), ranges.end());
    std::vector<DataRange<uint64_t>> newRanges;
    for (const auto& r : ranges)
    {
        if (newRanges.size() > 0 && r.start() == newRanges.back().end())
        {
            newRanges.back().addEnd(r.size());
        }
        else
        {
            newRanges.push_back(r);
        }
    }
    ranges = newRanges;
}

static uint64_t getParentStart(TensorROI& tRoi, const NodePtr& n, bool isInput)
{
    TOffset coord[Tensor::c_tensorMaxNDim];
    tRoi.getStartIndex(coord);
    return tRoi.getLayout().getByteOffset(coord, tRoi.m_parentTensor->getElementSizeInBits());
}

void CalculateTensorROIsLinearRanges::calculateMemoryRanges(TensorROI& tRoi, const NodePtr& n, bool isInput)
{
    if (!GCFG_ENABLE_DIRECT_CYCLIC_RANGE_CALC.value())
    {
        CalculateTensorROIsLinearRanges::calculateLinearRangesLegacy(tRoi, n, isInput);
        return;
    }
    MemoryRangesCalculator memoryRangesCalculator(tRoi, n, isInput);
    memoryRangesCalculator.calculateMemoryRanges();
}

void CalculateTensorROIsLinearRanges::calculateLinearRangesLegacy(TensorROI& tRoi, const NodePtr& n, bool isInput)
{
    //Todo: only works for DMA/TPC
    //Find highest dimension that's still dense
    unsigned denseDim = 1;
    const TensorROILayout& tLayout = tRoi.getLayout();
    unsigned int tensorDim = tRoi.m_layout.tensorDim;
    uint64_t               denseSize = safeBitsToByte(tLayout.m_size[0] * tRoi.m_parentTensor->getElementSizeInBits());
    if (contains(tLayout.spatialStrides, 0))
    {
        // Broadcast node - take the real tensor size in bytes as denseSize
        denseDim = tensorDim;
        denseSize = getRealSizeInBytes(tRoi.m_parentTensor);
    }
    else
    {
        for (; denseDim < tensorDim; ++denseDim)
        {
            if (denseSize != tLayout.spatialStrides[denseDim - 1])
            {
                break;
            }
            denseSize *= tLayout.m_size[denseDim];
        }
    }

    DataRange<uint64_t> r;
    uint64_t            parentStart = getParentStart(tRoi, n, isInput);
    r.start(parentStart);
    r.size(denseSize);

    if (denseSize == 0) return;

    HB_ASSERT(tRoi.m_overlapRoi.subRois->empty(), "subRois isn't empty");
    tRoi.m_overlapRoi.subRois->resize(1);
    tRoi.m_overlapRoi.subRois->back().relSoIdx = 0;
    auto& ranges                               = tRoi.m_overlapRoi.subRois->back().ranges;
    ranges.push_back(r);
    for (unsigned dim = denseDim; dim < tRoi.m_parentTensor->getDim(); ++dim)
    {
        std::vector<DataRange<uint64_t>> newRanges;
        uint64_t                         curStride = tLayout.spatialStrides[dim - 1];
        for (unsigned i = 1; i < tLayout.m_size[dim]; ++i)
        {
            for (auto dr : ranges)
            {
                dr.shiftRange(curStride * i);
                newRanges.push_back(dr);
            }
        }
        ranges.insert(ranges.end(), newRanges.begin(), newRanges.end());
    }

    // when strides are mixed up (like in transpose) we can potentially benefit from squeezing linear ranges
    if (!std::is_sorted(tLayout.spatialStrides, tLayout.spatialStrides + tRoi.m_layout.tensorDim))
    {
        HB_ASSERT(!tRoi.m_overlapRoi.subRois->empty(), "subRois is empty");
        squeezeRanges(tRoi.m_overlapRoi.subRois->back().ranges);
    }
}

static void calculateNonMMELinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput)
{
    if (n->isDma())
    {
        auto* physicalDma = dynamic_cast<DMAPhysicalMemoryOpNode*>(n.get());
        if (physicalDma)
        {
            physicalDma->calculateLinearRanges(tRoi, n, isInput);
            return;
        }
    }
    CalculateTensorROIsLinearRanges::calculateMemoryRanges(tRoi, n, isInput);
}

bool CalculateTensorROIsLinearRanges::apply(HabanaGraph& g) const
{
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        if (n->isLogicalOperation()) continue;
        SET_TEMP_LOG_CONTEXT(n->getNodeName());
        if (!HabanaGraph::runsOnMME(n))
        {
            std::list<NodeROI>& nodeRois = *g.GetNodeROIs(n);
            for (NodeROI& roi : nodeRois)
            {
                for (TensorROI& tRoi : roi.inputRois)
                {
                    calculateNonMMELinearRanges(tRoi, n, true);
                    printLinearRanges(tRoi, true);
                }
                for (TensorROI& tRoi : roi.outputRois)
                {
                    calculateNonMMELinearRanges(tRoi, n, false);
                    printLinearRanges(tRoi, false);
                }
                // If not an MME node, each ROI is one activation with one increment
                roi.numSignals = 1;
            }
        }
        else
        {
            calculateMmeLinearRanges(g, n);
        }
        createCyclicRanges(g, n);
    }

    return true;
}

static int64_t mod(int64_t x, uint32_t N)
{
    int64_t res = x % N;
    return res >= 0 ? res : N + res;
}

/*
    This function converts cyclic ranges to linear ranges.
    It can be used for debugging purposes, to get explictly the linear ranges from whom cyclic ranges consist.
*/
std::vector<DataRange<uint64_t>>
CalculateTensorROIsLinearRanges::toLinearRanges(const CyclicDataRange& cr, uint64_t start, uint64_t end)
{
    std::vector<DataRange<uint64_t>> ret;
    if (cr.size() == 0) return ret;

    int64_t alignedStart = ((start - cr.start() + cr.stride() - 1) / cr.stride()) * cr.stride() + cr.start();
    int64_t alignedEnd   = ((end - cr.end()) / cr.stride()) * cr.stride() + cr.end();

    if ((mod(start, cr.stride()) != cr.start()) && cr.pointInRange(start))
    {
        ret.emplace_back(DataRange<uint64_t>(start, start + mod(cr.end() - start, cr.stride())));
    }

    for (int64_t addr = alignedStart; addr < alignedEnd; addr += cr.stride())
    {
        ret.emplace_back(DataRange<uint64_t>(addr, addr + cr.size()));
    }

    if ((mod(end, cr.stride()) != cr.start()) && cr.pointInRange(end))
    {
        ret.emplace_back(DataRange<uint64_t>(end - mod(end - cr.start(), cr.stride()), end));
    }

    return ret;
}

void CalculateTensorROIsLinearRanges::printLinearRanges(const TensorROI& tRoi, bool input)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(ROI_RANGE)) return;

    LOG_TRACE(ROI_RANGE, "{} memory space", input? "Read" : "Write");

    for (const auto& subRoi : *tRoi.m_overlapRoi.subRois)
    {
        LOG_TRACE(ROI_RANGE, "    Signal {} ranges:", subRoi.relSoIdx);
        if (subRoi.cyclicRanges.empty())
        {
            for (const auto& range : subRoi.ranges)
            {
                LOG_TRACE(ROI_RANGE,
                          "        Start 0x{:x}, End 0x{:x}",
                          range.start() + tRoi.m_overlapRoi.offset,
                          range.end() + tRoi.m_overlapRoi.offset);
            }
        }
        else
        {
            for (int i = 0; i < subRoi.cyclicRanges.size(); ++i)
            {
                LOG_TRACE(ROI_RANGE,
                          "        Printing Cyclic Range {}. Range Bound Start 0x{:x}, Range Bound End 0x{:x}. Cyclic "
                          "Params: Start: {}, End: {}, Stride: {}",
                          i,
                          subRoi.ranges[i].start() + tRoi.m_overlapRoi.offset,
                          subRoi.ranges[i].end() + tRoi.m_overlapRoi.offset,
                          subRoi.cyclicRanges[i].start(),
                          subRoi.cyclicRanges[i].end(),
                          subRoi.cyclicRanges[i].stride());

                LOG_TRACE(ROI_RANGE, "        Explicit linear ranges of the cyclic range:");
                std::vector<DataRange<uint64_t>> convertedRanges =
                    toLinearRanges(subRoi.cyclicRanges[i], subRoi.ranges[i].start(), subRoi.ranges[i].end());
                for (const auto& range : convertedRanges)
                {
                    LOG_TRACE(ROI_RANGE,
                              "        Start 0x{:x}, End 0x{:x}",
                              range.start() + tRoi.m_overlapRoi.offset,
                              range.end() + tRoi.m_overlapRoi.offset);
                }
            }
        }
    }
}

static void addTensorRoiLinearRanges(const OverlapRoi& overlapRoi, const TensorPtr& tensor, TensorROI& tensorRoi)
{
    if (!tensor)
    {
        return;
    }
    tensorRoi.m_layout.inSram     = tensor->tensorAllocatedInSram();
    tensorRoi.m_overlapRoi        = overlapRoi;
    tensorRoi.m_overlapRoi.offset = tensor->getTensorOffset();
}

void CalculateTensorROIsLinearRanges::calculateForActivation(const ActivationOverlapRoi& activation,
                                                             const MmeNode&              mmeNode,
                                                             NodeROI&                    roi) const
{
    const OverlapRoi* firstInputRoi      = nullptr;
    const OverlapRoi* secondInputRoi     = nullptr;
    const OverlapRoi* outputRoi          = nullptr;
    const OverlapRoi* secondaryOutputRoi = &activation.roiO;
    unsigned          opAIndex           = TENSOR_IFM;
    unsigned          opBIndex           = TENSOR_WEIGHT;
    unsigned          opAInputTensor     = opAIndex;
    unsigned          opBInputTensor     = opBIndex;

    bool isCDParallelReductionAdd =
        activation.operandRoles[MmeCommon::INPUT_TENSOR_A] == MmeCommon::AUX_TENSOR_REDUCTION &&
        activation.operandRoles[MmeCommon::INPUT_TENSOR_B] == MmeCommon::AUX_TENSOR_SCRATCHPAD;
    MmeCommon::EMmeOpType opType = isCDParallelReductionAdd ? MmeCommon::e_mme_reductionAdd : getMmeNodeOpType(mmeNode);

    switch (opType)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        {
            const GEMMNode& gemmNode = dynamic_cast<const GEMMNode&>(mmeNode);
            opAIndex                 = gemmNode.getMMEOperandAIndex();
            opBIndex                 = gemmNode.getMMEOperandBIndex();

            if (mmeNode.getNodeType() == Node::TYPE_MASKED_BATCH_GEMM && roi.isAux)
            {
                opAInputTensor = TENSOR_AUX_BGEMM_MASK_A;
                opBInputTensor = TENSOR_AUX_BGEMM_MASK_B;
            }
            else
            {
                opAInputTensor = opAIndex;
                opBInputTensor = opBIndex;
            }
            // fallthrough
        }
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_gemm_transpose:
        case MmeCommon::e_mme_reductionAdd:
            firstInputRoi  = &activation.roiX;
            secondInputRoi = &activation.roiW;
            outputRoi      = &activation.roiY;
            break;
        case MmeCommon::e_mme_dedw:
            firstInputRoi  = &activation.roiY;
            secondInputRoi = &activation.roiX;
            outputRoi      = &activation.roiW;
            break;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            firstInputRoi  = &activation.roiY;
            secondInputRoi = &activation.roiW;
            outputRoi      = &activation.roiX;
            break;
        case MmeCommon::e_mme_trans:
        case MmeCommon::e_mme_memcpy:
            firstInputRoi  = &activation.roiX;
            secondInputRoi = nullptr;
            outputRoi      = &activation.roiY;
            break;
        default:
            HB_ASSERT(0, "Unsupport MME operation type");
    }

    const TensorPtr& inputA = activation.operandRoles[MmeCommon::INPUT_TENSOR_A] == MmeCommon::AUX_TENSOR_REDUCTION
                                  ? mmeNode.getInput(TENSOR_AUX_CD_REDUCTION)
                                  : mmeNode.getInput(opAInputTensor);
    addTensorRoiLinearRanges(*firstInputRoi, inputA, roi.inputRois[opAIndex]);
    if (secondInputRoi)
    {
        const TensorPtr& inputB = activation.operandRoles[MmeCommon::INPUT_TENSOR_B] == MmeCommon::AUX_TENSOR_SCRATCHPAD
                                      ? mmeNode.getInput(TENSOR_AUX_CD_SCRATCHPAD)
                                      : mmeNode.getInput(opBInputTensor);
        addTensorRoiLinearRanges(*secondInputRoi, inputB, roi.inputRois[opBIndex]);
    }
    else
    {
        HB_ASSERT(opType == MmeCommon::e_mme_memcpy || opType == MmeCommon::e_mme_trans,
                  "not expecting secondInputRoi to be null pointer");
    }

    const TensorPtr& output = activation.operandRoles[MmeCommon::OUTPUT_TENSOR_C] == MmeCommon::AUX_TENSOR_SCRATCHPAD
                                  ? mmeNode.getInput(TENSOR_AUX_CD_SCRATCHPAD)
                                  : mmeNode.getOutput(TENSOR_OFM);
    addTensorRoiLinearRanges(*outputRoi, output, roi.outputRois[TENSOR_OFM]);
    addTensorRoiLinearRanges(*secondaryOutputRoi,
                             mmeNode.getOutput(TENSOR_SECONDARY_OFM),
                             roi.outputRois[TENSOR_SECONDARY_OFM]);
}

void CalculateTensorROIsLinearRanges::resizeOrigRoi(NodeROI& origROI) const
{
    static const uint32_t CONV_NUM_OF_INPUTS  = 2;  // X, W
    static const uint32_t CONV_NUM_OF_OUTPUTS = 2;  // Y, O

    origROI.inputRois.resize(CONV_NUM_OF_INPUTS);
    origROI.outputRois.resize(CONV_NUM_OF_OUTPUTS);
}

void CalculateTensorROIsLinearRanges::addRoi(NodeROI                     newRoi,
                                             const ActivationOverlapRoi& activation,
                                             std::list<NodeROI>&         newRois,
                                             unsigned&                   pipeLevel,
                                             bool                        isAux,
                                             const MmeNode&              mmeNode) const
{
    bool hasCDParallelAuxTensor = false;

    newRoi.numSignals    = activation.numSignals;
    newRoi.pipelineLevel = pipeLevel++;
    newRoi.isAux         = isAux;

    calculateForActivation(activation, mmeNode, newRoi);

    // In case of cd parallel - update roi cache metadata with aux tensors metadata
    if (activation.operandRoles[MmeCommon::OUTPUT_TENSOR_C] == MmeCommon::AUX_TENSOR_SCRATCHPAD)
    {
        newRoi.outputsCacheMetaData[0] = newRoi.inputsCacheMetaData[TENSOR_AUX_CD_SCRATCHPAD];
        hasCDParallelAuxTensor         = true;
    }
    else if (activation.operandRoles[MmeCommon::INPUT_TENSOR_A] == MmeCommon::AUX_TENSOR_REDUCTION &&
             activation.operandRoles[MmeCommon::INPUT_TENSOR_B] == MmeCommon::AUX_TENSOR_SCRATCHPAD)
    {
        newRoi.inputsCacheMetaData[TENSOR_IFM]    = newRoi.inputsCacheMetaData[TENSOR_AUX_CD_REDUCTION];
        newRoi.inputsCacheMetaData[TENSOR_WEIGHT] = newRoi.inputsCacheMetaData[TENSOR_AUX_CD_SCRATCHPAD];
        hasCDParallelAuxTensor                    = true;
    }

    if (hasCDParallelAuxTensor)
    {
        // When CD Parallel - once we updated ROIs with the relevant inputs, remove all auxTensors from inputs
        // cacheMetaData (auxTensors indices are 2-5, so need to pop all last 4 entries)
        for (int i = 0; i < 4; i++)
        {
            newRoi.inputsCacheMetaData.pop_back();
        }
    }

    newRois.push_back(newRoi);
    printLinearRanges(newRoi.inputRois[0], true);
    printLinearRanges(newRoi.inputRois[1], true);
    printLinearRanges(newRoi.outputRois[0], false);
    printLinearRanges(newRoi.outputRois[1], false);
}

MmeCommon::EMmeOpType CalculateTensorROIsLinearRanges::getMmeNodeOpType(const MmeNode&(mmeNode)) const
{
    HB_ASSERT(0, "Shouldn't be called- overriden by derived classes");
    return MmeCommon::e_mme_fwd;
}

MemoryRangesCalculator::MemoryRangesCalculator(TensorROI& tRoi, const NodePtr& n, bool isInput)
: m_tRoi(tRoi), m_tensorDim(m_tRoi.m_layout.tensorDim), m_parentStart(getParentStart(tRoi, n, isInput))
{
}

void MemoryRangesCalculator::calculateSortedStridesWithMatchingSizes()
{
    m_dimSizesSortedByStrides = m_tRoi.getDimSizesInElements();
    m_dimStridesSorted        = m_tRoi.getStridesWithFcdDim();
    m_dimStridesSorted.pop_back();  // degenerate stride is not needed for memory ranges calculation

    if (!std::is_sorted(m_dimStridesSorted.begin(), m_dimStridesSorted.end()))
    {
        // Zip dim sizes and dim strides together
        std::vector<std::pair<uint64_t, uint64_t>> zipped;
        std::transform(m_dimSizesSortedByStrides.begin(),
                       m_dimSizesSortedByStrides.end(),
                       m_dimStridesSorted.begin(),
                       std::back_inserter(zipped),
                       [](const auto& size, const auto& stride) { return std::make_pair(size, stride); });

        // Sort dim sizes and dim strides together according to dim strides
        std::sort(std::begin(zipped), std::end(zipped), [&](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        // Unzip dim sizes and dim strides sorted together
        m_dimSizesSortedByStrides.clear();
        m_dimStridesSorted.clear();
        for (const auto& [size, stride] : zipped)
        {
            m_dimSizesSortedByStrides.push_back(size);
            m_dimStridesSorted.push_back(stride);
        }
    }
}

void MemoryRangesCalculator::calculateLinearRanges()
{
    auto& ranges = m_tRoi.m_overlapRoi.subRois->back().ranges;

    // Create initial linear range
    ranges.emplace_back(m_parentStart, m_parentStart + m_denseSize);

    // Expand linear ranges over the non dense dims
    for (unsigned dim = m_lowestNonDenseDim; dim < m_tensorDim; ++dim)
    {
        std::vector<DataRange<uint64_t>> newRanges;
        uint64_t                         curStride = m_dimStridesSorted[dim];
        for (uint64_t i = 1; i < m_dimSizesSortedByStrides[dim]; ++i)
        {
            for (auto dr : ranges)
            {
                dr.shiftRange(curStride * i);
                newRanges.push_back(dr);
            }
        }
        ranges.insert(ranges.end(), newRanges.begin(), newRanges.end());
    }
}

void MemoryRangesCalculator::calculateCyclicRanges()
{
    /*
        After highest cyclic dimension is determined, we create the first cyclic range over it.
        Then, for every non cyclic dimension, we exapnd the previously calculated cyclic ranges according to dimension's
       size and stride.
    */

    HB_ASSERT(m_denseSize <= m_cyclicStrideSize,
              "Dense size: {}, should be smaller than stride size: {}",
              m_denseSize,
              m_cyclicStrideSize);

    auto& cyclicRangeBounds = m_tRoi.m_overlapRoi.subRois->back().ranges;
    auto& cyclicRanges      = m_tRoi.m_overlapRoi.subRois->back().cyclicRanges;

    // Create initial cyclic range
    cyclicRanges.emplace_back(m_parentStart, m_parentStart + m_denseSize, m_cyclicStrideSize);

    // Create initial cyclic range bound
    HB_ASSERT(m_numLinearRangesPerCycle >= MIN_RANGES_FOR_CYCLIC, "Number of continious ranges per cycle is too small");
    uint64_t rangeSize = m_denseSize + ((m_numLinearRangesPerCycle - 1) * m_cyclicStrideSize);
    cyclicRangeBounds.emplace_back(m_parentStart, m_parentStart + rangeSize);

    // Expand cyclic ranges and bounds over the non cyclic dimensions
    for (unsigned dim = m_lowestNonCyclicDim; dim < m_tensorDim; ++dim)
    {
        std::vector<DataRange<uint64_t>> newCyclicRangeBounds;
        std::vector<CyclicDataRange>     newCyclicRanges;
        uint64_t                         curStride = m_dimStridesSorted[dim];

        for (uint64_t i = 1; i < m_dimSizesSortedByStrides[dim]; ++i)
        {
            for (unsigned j = 0; j < cyclicRangeBounds.size(); j++)
            {
                auto dr = cyclicRangeBounds[j];
                auto cr = cyclicRanges[j];

                dr.shiftRange(curStride * i);
                cr.shift(curStride * i);

                newCyclicRangeBounds.push_back(dr);
                newCyclicRanges.push_back(cr);
            }
        }
        cyclicRangeBounds.insert(cyclicRangeBounds.end(), newCyclicRangeBounds.begin(), newCyclicRangeBounds.end());
        cyclicRanges.insert(cyclicRanges.end(), newCyclicRanges.begin(), newCyclicRanges.end());

        HB_ASSERT(cyclicRangeBounds.size() == cyclicRanges.size(),
                  "Number of cyclic ranges, and of range bounds, should be equal");
    }
}

void MemoryRangesCalculator::calculateNumberOfLinearRangesPerCycle()
{
    m_numLinearRangesPerCycle = 1;
    for (unsigned dim = m_lowestNonDenseDim; dim < m_lowestNonCyclicDim; ++dim)
    {
        m_numLinearRangesPerCycle *= m_dimSizesSortedByStrides[dim];
    }
}

bool MemoryRangesCalculator::isParentZeroSizedTensor()
{
    if (m_tRoi.m_parentTensor->isZeroSizedDataTensor() || m_tensorDim == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void MemoryRangesCalculator::calculateDenseSizeAndDim()
{
    // Find lowest dimension that is not dense, and the size in bytes of the dense range
    m_lowestNonDenseDim = 1;  // FCD (dim 0) is always dense
    m_denseSize =
        safeBitsToByte((uint64_t)m_dimSizesSortedByStrides[0] * m_tRoi.m_parentTensor->getElementSizeInBits());
    if (contains(m_dimStridesSorted, 0))
    {
        // Broadcast node - take the real tensor size in bytes as denseSize
        m_lowestNonDenseDim = m_tensorDim;
        m_denseSize         = getRealSizeInBytes(m_tRoi.m_parentTensor);
    }
    else
    {
        for (; m_lowestNonDenseDim < m_tensorDim; ++m_lowestNonDenseDim)
        {
            if (m_denseSize < m_dimStridesSorted[m_lowestNonDenseDim])
            {
                break;
            }
            uint64_t leftOver = m_denseSize - m_dimStridesSorted[m_lowestNonDenseDim];
            m_denseSize *= m_dimSizesSortedByStrides[m_lowestNonDenseDim];

            /*
                It means dense ranges from different dimensions share memory, which is counted twice in the naive
               calculation. We would like to substract the parts that were counted twice.
            */
            if (leftOver > 0)
            {
                m_denseSize -= leftOver * (m_dimSizesSortedByStrides[m_lowestNonDenseDim] - 1);
            }
        }
    }
}

void MemoryRangesCalculator::calculateCyclicDim()
{
    // Calculate the basic cyclic dim
    m_baseCyclicDim = m_lowestNonDenseDim;
    for (; m_baseCyclicDim < m_tensorDim; ++m_baseCyclicDim)
    {
        if (m_dimSizesSortedByStrides[m_baseCyclicDim] != 1)
        {
            break;
        }
    }

    // Find lowest dimension that is not cyclic
    if (m_baseCyclicDim == m_tensorDim)
    {
        m_lowestNonCyclicDim = m_baseCyclicDim;  // All dims are cyclic
    }
    else
    {
        uint64_t expectedCyclicBoundSize =
            m_dimStridesSorted[m_baseCyclicDim] * m_dimSizesSortedByStrides[m_baseCyclicDim];
        m_lowestNonCyclicDim = m_baseCyclicDim + 1;
        for (; m_lowestNonCyclicDim < m_tensorDim; ++m_lowestNonCyclicDim)
        {
            if (expectedCyclicBoundSize != m_dimStridesSorted[m_lowestNonCyclicDim])
            {
                break;
            }
            expectedCyclicBoundSize *= m_dimSizesSortedByStrides[m_lowestNonCyclicDim];
        }
    }
}

uint64_t MemoryRangesCalculator::getExpectedNumberOfLinearRanges()
{
    uint64_t expectedNumLinearRanges = 1;
    for (unsigned dim = m_lowestNonDenseDim; dim < m_tensorDim; ++dim)
    {
        expectedNumLinearRanges *= m_dimSizesSortedByStrides[dim];
    }
    return expectedNumLinearRanges;
}

uint64_t MemoryRangesCalculator::getExpectedNumberOfCyclicRanges()
{
    uint64_t expectedNumCyclicRanges = 1;
    for (unsigned dim = m_lowestNonCyclicDim; dim < m_tensorDim; ++dim)
    {
        expectedNumCyclicRanges *= m_dimSizesSortedByStrides[dim];
    }
    return expectedNumCyclicRanges;
}

void MemoryRangesCalculator::calculateMemoryRanges()
{
    if (isParentZeroSizedTensor())
    {
        return;
    }

    calculateSortedStridesWithMatchingSizes();
    calculateDenseSizeAndDim();

    if (m_denseSize == 0)  // tRoi is zero sized
    {
        return;
    }

    calculateCyclicDim();
    calculateNumberOfLinearRangesPerCycle();

    m_cyclicStrideSize = 0;
    if (m_baseCyclicDim != m_tensorDim)
    {
        CHECK_MAX_VAL(m_dimStridesSorted[m_baseCyclicDim], uint32_t);
        m_cyclicStrideSize = m_dimStridesSorted[m_baseCyclicDim];
    }

    HB_ASSERT(m_tRoi.m_overlapRoi.subRois->empty(), "subRois isn't empty");
    m_tRoi.m_overlapRoi.subRois->resize(1);
    m_tRoi.m_overlapRoi.subRois->back().relSoIdx = 0;

    if ((m_numLinearRangesPerCycle >= MIN_RANGES_FOR_CYCLIC) && m_cyclicStrideSize <= MAX_STRIDE_SIZE)
    {
        HB_ASSERT(m_cyclicStrideSize > 0, "Stride should be positive for cyclic ranges calculation");
        LOG_DEBUG(ROI_RANGE,
                  "Calculating cyclic ranges for tensor sub ROI, expected number of memory ranges: {}",
                  getExpectedNumberOfCyclicRanges());
        calculateCyclicRanges();
    }
    else
    {
        LOG_DEBUG(ROI_RANGE,
                  "Calculating linear ranges for tensor sub ROI, expected number of memory ranges: {}",
                  getExpectedNumberOfLinearRanges());
        calculateLinearRanges();
    }
}

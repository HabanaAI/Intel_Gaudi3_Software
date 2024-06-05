#include "include/mme_common/recurring_misalignment_opt.h"
#include "include/mme_common/conv_sub_problems.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe_generator.h"
#include "mme_hal_factory.h"

// Currently, only a single sub-problem is supported
#define SUPPORT_MULTI_SUB_PROBLEMS 0

namespace MmeCommon
{
bool RecurringMisalignmentOptimization::canApplyOptimization(const MmeLayerParams& params,
                                                             const CommonGeoAttr& geoAttr,
                                                             const MmeHalReader& mmeHalReader)
{
    if (!params.strategy.recurringMisalignmentOptEn || !params.strategy.sbReuse)
    {
        return false;
    }
    if (!params.strategy.loweringEn || !params.canLower())
    {
        return false;
    }
    // Only fwd is currently supported
    if (params.opType != MmeCommon::e_mme_fwd && params.opType != MmeCommon::e_mme_transposed_dedx)
    {
        return false;
    }
    // Number of elements brought in every step must be more than CL, otherwise we add redundant CL access
    unsigned ClElements = mmeHalReader.getMemoryClSize() / getElementSize(params.getOperand(e_mme_op_a).elementType);
    if (params.getSingleGemmCD() <= ClElements)
    {
        return false;
    }
    unsigned numSubProblems = calcNumSubProblems(params, mmeHalReader, e_mme_op_a);
    if (!isNumSubProblemsValid(numSubProblems))
    {
        return false;
    }
    // Every sub-problem is responsible to produce part of the output. If number of sub-problems is larger
    // than the output width, some sub-problems have 0 work and many assumptions break
    if (numSubProblems > params.getOperand(e_mme_op_c).sizes[1])
    {
        return false;
    }

    return true;
}

static unsigned getStepSize(const MmeLayerParams& params, MmeCommon::EMmeInternalOperand operand)
{
    unsigned commonDim = params.getOperand(operand).sizes[0];
    unsigned kernelStrideW = params.conv.stride[0];
    return commonDim * kernelStrideW;
}

// Calculate the actual num of sub problems.
unsigned RecurringMisalignmentOptimization::calcNumSubProblems(const MmeLayerParams& params,
                                                               const MmeHalReader& mmeHalReader,
                                                               MmeCommon::EMmeInternalOperand operand)
{
    unsigned elementSize = getElementSize(params.getOperand(operand).elementType);
    unsigned cacheLineSizeInElements =
        mmeHalReader.getMemoryClSize() / getElementSize(params.getOperand(operand).elementType);
    unsigned stepSize = getStepSize(params, operand);
    unsigned numSubProblems = std::lcm(cacheLineSizeInElements, stepSize) / stepSize;
    return numSubProblems;
}

// Returns the valid number of sub problems for external use.
// It verifies that the optimization is valid, and that the number of sub-problems meet the constaints
unsigned RecurringMisalignmentOptimization::getNumSubProblems(const MmeLayerParams& params,
                                                              const CommonGeoAttr& geoAttr,
                                                              const MmeHalReader& mmeHalReader)
{
    // First verify that optimization can be applied
    if (!canApplyOptimization(params, geoAttr, mmeHalReader))
    {
        return 1;
    }
    return calcNumSubProblems(params, mmeHalReader, e_mme_op_a);
}

// returns the size of the i'th sub-problem
static unsigned getSubProblemSize(unsigned size, unsigned numSubProblems, unsigned subProblemIdx)
{
    unsigned mod = size % numSubProblems;
    unsigned subProblemSize = size / numSubProblems;
    // If size is divisible by numSubProblems, return subProblemSize.
    // Otherwise, split the residue among the first sub-problems.
    return ((mod == 0) || (subProblemIdx >= mod)) ? subProblemSize : subProblemSize + 1;
}

void RecurringMisalignmentOptimization::makeParamsForSubProblem(const MmeLayerParams& originalParams,
                                                                unsigned numSubProblems,
                                                                unsigned subProblemIdx,
                                                                MmeCommon::MmeLayerParams& subProblemParams,
                                                                OffsetArray& descAddrOffset)
{
    // Params for sub-ptoblem i are the same as the original params except that:
    // - A tensor: dim 1 base = conv.stride[0] * i    (or i??)
    // - B tensor: No change
    // - C tensor: dim 0 base = i
    //             stride[0] *= numSubProblems
    // - conv.stride[0] *= numSubProblems

    auto& aOperand = subProblemParams.getOperand(e_mme_op_a);
    auto& cOperand = subProblemParams.getOperand(e_mme_op_c);
    // Set the base offsets in the params.
    aOperand.bases[1] += subProblemIdx * originalParams.conv.stride[0];
    cOperand.bases[1] += subProblemIdx;
    // Update the conv stride
    subProblemParams.conv.stride[0] *= numSubProblems;
    if (subProblemParams.isDedxOperation())
    {
        // adjust padding to the packing factor
        subProblemParams.conv.padding[0] += originalParams.strategy.packingFactor - 1;
    }
    // As a result the output is smaller
    cOperand.sizes[1] = getSubProblemSize(cOperand.sizes[1], numSubProblems, subProblemIdx);
    cOperand.strides[1] *= numSubProblems;

    // Update the address offsets for the related descriptor
    // Note that the tensor strides need to be taken from the original
    unsigned aOffset = aOperand.bases[1] * originalParams.getOperand(e_mme_op_a).strides[1];
    unsigned cOffset = cOperand.bases[1] * originalParams.getOperand(e_mme_op_c).strides[1];
    switch (originalParams.opType)
    {
        case e_mme_fwd:
            descAddrOffset.xOffset[1] = aOffset;
            descAddrOffset.yOffset[1] = cOffset;
            break;
        case e_mme_transposed_dedx:
            descAddrOffset.xOffset[1] = cOffset;
            descAddrOffset.yOffset[1] = aOffset;
            break;
        default:
            MME_ASSERT(0,"operation not supported");
    }
}

unsigned RecurringMisalignmentOptimization::getCutPointPerSubProblem(const MmeLayerParams& params,
                                                                     const CommonGeoAttr& geoAttr,
                                                                     ChipType chipType)
{
    return getCutPointPerSubProblem(params, geoAttr, getMmeHal(chipType));
}

unsigned RecurringMisalignmentOptimization::getCutPointPerSubProblem(const MmeLayerParams& params,
                                                                     const CommonGeoAttr& geoAttr,
                                                                     const MmeHalReader& mmeHalReader)
{
    static unsigned cutPointCounter = 0;

    if (!canApplyOptimization(params, geoAttr, mmeHalReader))
    {
        return 0;  // No cut point
    }

    // params that can yield more than one sub-problem does not fit the optimization
    if (calcNumSubProblems(params, mmeHalReader, e_mme_op_a) > 1)
    {
        return 0;
    }

    // Initial offset is cdBase - padding * commonDim

    unsigned cacheLineSizeInElements =
        mmeHalReader.getMemoryClSize() / getElementSize(params.getOperand(e_mme_op_a).elementType);
    unsigned cdDimInA = isTransposed(params.opType, e_mme_in_a) ? 0 : 1;
    const MmeTensorView& aOperand = params.getOperand(e_mme_op_a);
    // The offset of the sub-problem is already set in base.
    // It is the mul of:
    // - The SP idx as number of pixels in the W direction
    // - The commonDim
    // - The conv stride, because each SP starts convStride pixels ahead
    unsigned baseWidth = aOperand.bases[1] * aOperand.strides[1];
    // The actual read offset is 'padding' pixels back
    int actualPadding = params.conv.padding[0];
    if (params.isDedxOperation())
    {
        actualPadding -= params.strategy.packingFactor - 1;
    }
    int paddingOffset = actualPadding * aOperand.sizes[0];
    int readOffset = baseWidth - paddingOffset;
    // And the alignment cut point includes all elements up to the alignment point
    unsigned spAlignmentCutPoint = readOffset % cacheLineSizeInElements;

    if (spAlignmentCutPoint == 0)  // cut point is aligned to CL size
    {
        return 0;  // No cut
    }

    // The alignment point is where we want to cut between the prefix and the remainder
    return (cacheLineSizeInElements - spAlignmentCutPoint) % cacheLineSizeInElements;
}

// return true if an AGU is expected to access CL more then once if they are not interleaved
bool RecurringMisalignmentOptimization::isMultipleAccessToSameCL(const MmeLayerParams& params,
                                                                 const MmeHalReader& mmeHalReader,
                                                                 EMmeInternalOperand operand)
{
    // when the step size is less than a CL, then frequently part of the fetched data will be in the cache
    // in these cases it is better to interleave on the second spatial dimension instead of the first one
    unsigned elementSize = getElementSize(params.getOperand(operand).elementType);
    unsigned cacheLineSizeInElements =
        mmeHalReader.getMemoryClSize() / getElementSize(params.getOperand(operand).elementType);
    unsigned stepSize = params.conv.stride[0] * params.getOperand(operand).strides[1];
    // Case 2: when the step size is less than a CL, then frequently part of the fetched data will be in the cache
    if (stepSize < cacheLineSizeInElements)
    {
        return true;
    }

    return false;
}

bool RecurringMisalignmentOptimization::isRecurringMisalignment(const MmeLayerParams& params,
                                                                EMmeInternalOperand operand,
                                                                ChipType chipType)
{
    return isRecurringMisalignment(params, getMmeHal(chipType), operand);

}

// return true if the AGU is expected to have the recurring misalignment access pattern.
// in case the strides are exactly 1 CL than this misalignment can be alleviated by the SB caches.
// this function is used for DEDW under CD concurrency to check if we should avoid interleaving on the first spatial
// dimensions.
bool RecurringMisalignmentOptimization::isRecurringMisalignment(const MmeLayerParams& params,
                                                                const MmeHalReader& mmeHalReader,
                                                                EMmeInternalOperand operand)
{
    MME_ASSERT(operand == e_mme_op_a, "The recurring misalignment currently supports op a only");

    // The first access is unaligned, and the effective stride between accesses
    // is a full number of CLs. Meaning every MME CL require two memory access.
    // if the effective stride is only a single CL then we can use the sbCaches to use the data
    // from the previous unaligned access.
    unsigned elementSize = getElementSize(params.getOperand(operand).elementType);
    unsigned cacheLineSizeInElements =
        mmeHalReader.getMemoryClSize() / getElementSize(params.getOperand(operand).elementType);
    unsigned startOffset = params.conv.padding[0] * params.getOperand(operand).strides[1];
    unsigned stepSize = params.conv.stride[0] * params.getOperand(operand).strides[1];
    if ((startOffset % cacheLineSizeInElements !=
         0) &&  // start is CL unaligned
                // TODO this modulu operation should be changed to a check if the stepSize is exactly 1 CL.
        (stepSize % cacheLineSizeInElements == 0))  // effective stride is CL aligned
    {
        return true;
    }

    return false;
}

std::string RecurringMisalignmentOptimization::getDebugInfo(const MmeCommon::ConvSubProblemContainer& convSubProblems,
                                                            const CommonGeoAttr& geoAttr,
                                                            const MmeHalReader& mmeHal,
                                                            const MmeLayerParams& originalParams)
{
    if (!canApplyOptimization(originalParams, geoAttr, mmeHal))
    {
        return "";  // No debug info
    }

    std::string debugInfo;

    unsigned numSubProblems = convSubProblems.size();
    unsigned commonDim = originalParams.getOperand(e_mme_op_a).sizes[0];
    unsigned effectiveStride = commonDim * originalParams.conv.stride[0];
    unsigned clSize = mmeHal.getMemoryClSize() / getElementSize(originalParams.getOperand(e_mme_op_a).elementType);

    debugInfo = "Recurring misalignment opt:";
    debugInfo += " CD=" + std::to_string(commonDim);
    debugInfo += ", EffStride=" + std::to_string(effectiveStride);
    debugInfo += ", clSize=" + std::to_string(clSize);
    debugInfo += ", #SPs=" + std::to_string(numSubProblems);
    debugInfo += ", CD cut points=";
    for (unsigned subProblemIdx = 0; subProblemIdx < numSubProblems; subProblemIdx++)
    {
        const auto subProblemParams = convSubProblems[subProblemIdx].params;
        unsigned cdCutPoint =
            RecurringMisalignmentOptimization::getCutPointPerSubProblem(subProblemParams, geoAttr, mmeHal);
        debugInfo += std::to_string(cdCutPoint);
        unsigned numPartials = convSubProblems[subProblemIdx].recipe.getNonSpatialSubviews().size();
        if (cdCutPoint != 0 && numPartials > 2)
        {
            debugInfo += "p";  // This sub-problem includes real partials (beyond the cd cut)
        }
        if (subProblemIdx < (numSubProblems - 1))
        {
            debugInfo += ", ";
        }
    }

    return debugInfo;
}

}  // namespace MmeCommon

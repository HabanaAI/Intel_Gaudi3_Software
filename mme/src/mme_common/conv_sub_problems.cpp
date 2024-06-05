#include "include/mme_common/conv_sub_problems.h"
#include "include/mme_common/mme_common_enum.h"
#include "mme_geo_factory.h"
#include "mme_hal_factory.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include <algorithm>
#include <array>
namespace MmeCommon
{
void ConvSubProblemContainer::createConvSubProblem(const MmeLayerParams& params)
{
    unsigned numOfSubProblems = calcTotalNumOfSubProblems(params);
    reserve(numOfSubProblems);
    for (unsigned currSubProblemIdx = 0; currSubProblemIdx < numOfSubProblems; currSubProblemIdx++)
    {
        push(params);
        switch (params.opType)
        {
            default:
                break;
            case MmeCommon::e_mme_dedx:
                makeParamsForDedxSubProblem(params, numOfSubProblems, currSubProblemIdx);
                current->setMemsetDesc(isMemsetDesc(current->params));
                if (current->isMemsetDesc())
                {
                    handleMemsetDescRecipe();
                }
                if (skipRecipeGeneration(current->params))
                {
                    pop();
                    continue;
                }
                break;
            case MmeCommon::e_mme_transposed_dedx:
            case MmeCommon::e_mme_fwd:
                makeParamsForRecurringMisalignmentSubProblem(params, numOfSubProblems, currSubProblemIdx);
                break;
        }
    }
}

bool ConvSubProblemContainer::extractGcdFromConvParams(std::array<unsigned, MME_MAX_CONV_DIMS - 1>* stride,
                                                       std::array<unsigned, MME_MAX_CONV_DIMS - 1>* dilation,
                                                       std::array<unsigned, MME_MAX_CONV_DIMS - 1>* commonDivs) const
{
    bool hasGcd = false;
    // divide both stride and dilation by their gcd, and keep that gcd in a separate list to be used later
    for (unsigned i = 0; i < MME_MAX_CONV_DIMS - 1; ++i)
    {
        unsigned gcd = std::__gcd((*stride)[i], (*dilation)[i]);
        if (gcd != 0 && gcd != 1)
        {
            hasGcd = true;
            (*dilation)[i] /= gcd;
            (*stride)[i] /= gcd;
            (*commonDivs)[i] = gcd;
        }
    }
    return hasGcd;
}

void ConvSubProblemContainer::makeParamsForRecurringMisalignmentSubProblem(const MmeLayerParams& originalParams,
                                                                           unsigned numOfSubProblems,
                                                                           unsigned subProblemIdx)
{
    MME_ASSERT(originalParams.opType == MmeCommon::e_mme_fwd ||
                   originalParams.opType == MmeCommon::e_mme_transposed_dedx,
               "Recurring Misalignment optimization is currently supported for fwd and transposed dedx only");

    OffsetArray& descAddrOffset = current->addressOffset;
    RecurringMisalignmentOptimization::makeParamsForSubProblem(originalParams,
                                                               numOfSubProblems,
                                                               subProblemIdx,
                                                               current->params,
                                                               descAddrOffset);
}

void ConvSubProblemContainer::makeParamsForDedxSubProblem(const MmeLayerParams& originalParams,
                                                          unsigned numOfSubProblems,
                                                          unsigned subProblemIdx)
{
    MmeLayerParams& convParamsForSubProblem = current->params;
    OffsetArray& descAddrOffset = current->addressOffset;

    if (originalParams.strategy.packingFactor > 1)
    {
        // adjust padding to the packing factor
        convParamsForSubProblem.conv.padding[0] += originalParams.strategy.packingFactor - 1;
        return;
    }
    // Modify pipeline level
    convParamsForSubProblem.strategy.pipelineLevel =
        div_round_up(convParamsForSubProblem.strategy.pipelineLevel, numOfSubProblems);

    // group offset
    unsigned K0 = subProblemIdx;

    // for each group the formula to get the input location from output (Dy location from DX)
    // M = N + KD - (K0D-P)/S
    // S' = 1 , D' = D , P' = (K0D-P)/S
    unsigned remX = 0, remW = 0;
    std::fill(convParamsForSubProblem.conv.stride.begin(), convParamsForSubProblem.conv.stride.end(), 1);

    // if gcd(dilation[i], strides[i]) > 1 for some 0 <= i <= 2, transfer it to the tensors stride
    auto dilation = originalParams.conv.dilation;
    auto strides = originalParams.conv.stride;
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> commonDivs = {1, 1, 1};
    bool hasGcd = extractGcdFromConvParams(&strides, &dilation, &commonDivs);
    if (hasGcd)
    {
        for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; ++convDim)
        {
            if (commonDivs[convDim] != 1)
            {
                convParamsForSubProblem.conv.dilation[convDim] = dilation[convDim];

                unsigned tensorDim = convDim + 1;
                convParamsForSubProblem.x.strides[tensorDim] *= commonDivs[convDim];
                convParamsForSubProblem.x.sizes[tensorDim] /= commonDivs[convDim];
            }
        }
    }

    // each descriptor takes the relevant weights, and is written strided to the output dx.
    unsigned rem = K0;
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> offsetWithinGcd;
    for (int convDim = MME_MAX_CONV_DIMS - 2; convDim >= 0; --convDim)
    {
        unsigned weightDim = convDim + 2;
        offsetWithinGcd[convDim] = rem % commonDivs[convDim];
        convParamsForSubProblem.w.bases[weightDim] = (rem / commonDivs[convDim]) % strides[convDim];
        descAddrOffset.wOffset[weightDim] =
            convParamsForSubProblem.w.bases[weightDim] * convParamsForSubProblem.w.strides[weightDim];
        rem /= originalParams.conv.stride[convDim];
    }

    for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; convDim++)
    {
        unsigned tensorDim = convDim + 1;
        unsigned weightDim = convDim + 2;

        int padding = div_round_down(originalParams.conv.padding[convDim], commonDivs[convDim]);
        int paddingRem = 0;
        if (hasGcd)
        {
            // fix padding and offset in case of stride & dilation gcd
            paddingRem = originalParams.conv.padding[convDim] % commonDivs[convDim];

            if (offsetWithinGcd[convDim] < paddingRem)
            {
                if (paddingRem != 0)
                {
                    padding++;
                }
                paddingRem = (commonDivs[convDim] - originalParams.conv.padding[convDim]) % commonDivs[convDim];
            }
            else
            {
                paddingRem = -paddingRem;
            }
        }

        int K0DMinusP = (convParamsForSubProblem.w.bases[weightDim] * dilation[convDim]) - padding;
        int newPadding = -div_round_down(K0DMinusP, strides[convDim]);
        convParamsForSubProblem.conv.padding[convDim] = newPadding;
        convParamsForSubProblem.x.bases[tensorDim] = mod_neg(K0DMinusP, strides[convDim]);
        descAddrOffset.xOffset[tensorDim] =
            convParamsForSubProblem.x.bases[tensorDim] * convParamsForSubProblem.x.strides[tensorDim];
        // fix xOffset to match the original padding, and the gcd part of the offset which is not present in x.bases
        descAddrOffset.xOffset[tensorDim] +=
            (paddingRem + offsetWithinGcd[convDim]) * originalParams.x.strides[tensorDim];

        // write each desc to output (X) strided
        remX =
            (convParamsForSubProblem.x.bases[tensorDim] * commonDivs[convDim]) + paddingRem + offsetWithinGcd[convDim];
        convParamsForSubProblem.x.strides[tensorDim] *= strides[convDim];
        convParamsForSubProblem.x.sizes[tensorDim] /= strides[convDim];
        convParamsForSubProblem.x.bases[tensorDim] /= strides[convDim];
        // read each desc from w strided
        remW = convParamsForSubProblem.w.bases[weightDim] % strides[convDim];
        convParamsForSubProblem.w.strides[weightDim] *= strides[convDim];
        convParamsForSubProblem.w.sizes[weightDim] /= strides[convDim];
        convParamsForSubProblem.w.bases[weightDim] /= strides[convDim];

        // add the remainder
        if (remX < (originalParams.x.sizes[tensorDim] % originalParams.conv.stride[convDim]))
        {
            convParamsForSubProblem.x.sizes[tensorDim]++;
        }
        if (remW < (originalParams.w.sizes[weightDim] % strides[convDim]))
        {
            convParamsForSubProblem.w.sizes[weightDim]++;
        }
    }

    if (hasGcd)
    {
        // if gcd(dilation[i], strides[i]) > 1 for some 0 <= i <= 2, some subproblems are empty and should be
        // replaced with memset
        for (int convDim = MME_MAX_CONV_DIMS - 2; convDim >= 0; --convDim)
        {
            if (offsetWithinGcd[convDim] != 0)
            {
                unsigned weightDim = convDim + 2;
                convParamsForSubProblem.w.sizes[weightDim] = 0;  // this will turn into memset desc later
            }
        }
    }
}

bool ConvSubProblemContainer::isOutOfBounds(const MmeLayerParams& newParams)
{
    // after we generated new params for a specific dedx desc,
    // if x.sizes[someDim] = 0 this means that the descriptor will write out-of-bound,
    const bool isOutOfBounds = hasZeros(newParams.x.sizes);
    return isOutOfBounds;
}

bool ConvSubProblemContainer::shouldAddMemsetDesc(const MmeLayerParams& newParams)
{
    return isMemsetDesc(newParams) && newParams.strategy.memsetDedxVoidPixels;
}

bool ConvSubProblemContainer::isMemsetDesc(const MmeLayerParams& newParams)
{
    const bool isMemset = hasZeros(newParams.w.sizes);
    return !isOutOfBounds(newParams) && isMemset;
}

bool ConvSubProblemContainer::isComputeDesc(const MmeLayerParams& newParams)
{
    return !isOutOfBounds(newParams) && !isMemsetDesc(newParams);
}

void ConvSubProblemContainer::handleMemsetDescRecipe()
{
    current->params.strategy.pipelineLevel = 1;
    current->params.strategy.sbReuse = false;
}

bool ConvSubProblemContainer::skipRecipeGeneration(const MmeLayerParams& params) const
{
    if (current->isMemsetDesc())
    {
        // in case we should not add memset -> we need to skip this descriptor.
        return !shouldAddMemsetDesc(params);
    }
    else
    {
        // skip if this descriptor is out of bound - but not zero CD .
        const unsigned spSize = multiplyElements(params.x.sizes.begin() + 1, params.x.sizes.end());
        return spSize == 0;
    }
}

unsigned ConvSubProblemContainer::calcTotalNumOfSubProblems(const MmeLayerParams& params) const
{
    switch (params.opType)
    {
        case e_mme_dedx:
            return getTotalDedxNumOfDesc(params);
        case e_mme_transposed_dedx:
            MME_ASSERT(getTotalDedxNumOfDesc(params) == 1, "transpose dedx doesnt support multiple sub problems");
        case e_mme_fwd:
        {
            auto geoAttr = getGeoAttr(m_chipType, params);
            return RecurringMisalignmentOptimization::getNumSubProblems(params, *geoAttr, getMmeHal(m_chipType));
        }
        default:
            return 1;
    }
}

unsigned ConvSubProblemContainer::getTotalDedxNumOfDesc(const MmeLayerParams& params) const
{
    unsigned numOfSubProblems = multiplyElements(params.conv.stride.begin(), params.conv.stride.end());
    if (params.strategy.packingFactor > 1)
    {
        //  currently packing is supported only for convolutions without strides.
        //  packing increases the conv stride by packing factor on the first dimension.
        //  after packing we expect the conv strides to be [packingFactor, 1, 1]
        //  thus we expect numOfSubProblems to be equal to the packingFactor.
        MME_ASSERT(numOfSubProblems == params.strategy.packingFactor,
                   "conv stride is not yet supported with dedx packing");
        return 1;
    }
    return numOfSubProblems;
}

void ConvSubProblemContainer::reset(const MmeLayerParams& newParams)
{
    SubProblemVec::clear();
    current = nullptr;
    createConvSubProblem(newParams);
}

}  // namespace MmeCommon

#include "include/mme_common/mme_common_enum.h"
#include <cstring>
#include <sstream>
#define FMT_HEADER_ONLY
#include "spdlog/fmt/bundled/format.h"

namespace
{
template<typename T>
inline bool areEqual(const T& lhs, const T& rhs)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return !memcmp(&lhs, &rhs, sizeof(T));
}
}

namespace MmeCommon
{
bool isInputAligned(const MmeLayerParams& params, unsigned alignVal, std::optional<EMmeInputOperand> inputOp)
{
    if (!params.strategy.alignedAddresses)
    {
        return false;
    }
    if (!inputOp.has_value())
    {
        return params.getOperand(e_mme_op_a).areStridesFullyAligned(alignVal) ||
               params.getOperand(e_mme_op_b).areStridesFullyAligned(alignVal);
    }
    if (inputOp == EMmeInputOperand::e_mme_in_a)
    {
        return params.getOperand(e_mme_op_a).areStridesFullyAligned(alignVal);
    }
    MME_ASSERT(inputOp == EMmeInputOperand::e_mme_in_b, "Unsupported input");
    return params.getOperand(e_mme_op_b).areStridesFullyAligned(alignVal);
}

bool MmeLayerParams::operator==(const MmeLayerParams& rhs) const
{
    // This function requires that all fields of MmeLayerParams are included. So it needs
    // to be updated when new fields are added
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep MmeLayerParams compilation determinstic, the following assert intends to prevent
     * making changes in the MmeLayerParams struct without respectively updating the needed padding. If you've added new
     * fields to the MmeLayerParams struct or change their order or change inner structs of the MmeLayerParams, please
     * update the changed struct with 'spare' field that will hold the padded bits done by the compiler, with value of
     * zero. After that - please update structs size in this assert, so build can pass successfully. Thanks for paying
     * attention, by doing so we make the hashing done by the Descriptor Cache to be deterministic.*/

    size_t constexpr STRUCT_SIZE = 752;
    static_assert(sizeof(MmeLayerParams) == STRUCT_SIZE,
                  "MmeLayerParams struct is updated, need to check it's padded with zeros and not with garbage by the compiler,"
                  " read the comment above this assert");

    return opType == rhs.opType &&
    areEqual(x, rhs.x) &&
    areEqual(y, rhs.y) &&
    areEqual(w, rhs.w) &&
    areEqual(xAux, rhs.xAux) &&
    areEqual(yAux, rhs.yAux) &&
    areEqual(wAux, rhs.wAux) &&
    spBase == rhs.spBase &&
    spSize == rhs.spSize &&
    areEqual(conv, rhs.conv) &&
    areEqual(controls, rhs.controls) &&
    areEqual(strategy, rhs.strategy) &&
    areEqual(tracing, rhs.tracing) &&
    areEqual(memoryCfg, rhs.memoryCfg);
}

bool MmeLayerParams::operator<(const MmeLayerParams& rhs) const
{
    if (rhs.opType < opType) return true;

    bool currConditionEq = (rhs.opType == opType);
    if ((currConditionEq) && (rhs.spBase < spBase)) return true;

    currConditionEq &= (rhs.spBase == spBase);
    if ((currConditionEq) && (rhs.spSize < spSize)) return true;

    // controls

    currConditionEq &= (rhs.spSize == spSize);
    if ((currConditionEq) && (rhs.controls.atomicAdd < controls.atomicAdd)) return true;

    currConditionEq &= (rhs.controls.atomicAdd == controls.atomicAdd);
    if ((currConditionEq) && (rhs.controls.reluEn < controls.reluEn)) return true;

    currConditionEq &= (rhs.controls.reluEn == controls.reluEn);
    if ((currConditionEq) && (rhs.controls.roundingMode < controls.roundingMode)) return true;

    currConditionEq &= (rhs.controls.roundingMode == controls.roundingMode);
    if ((currConditionEq) && (rhs.controls.conversionRoundingMode < controls.conversionRoundingMode)) return true;

    currConditionEq &= (rhs.controls.conversionRoundingMode == controls.conversionRoundingMode);
    if ((currConditionEq) && (rhs.controls.signalingMode < controls.signalingMode)) return true;

    currConditionEq &= (rhs.controls.signalingMode == controls.signalingMode);
    if ((currConditionEq) && (rhs.controls.squashIORois < controls.squashIORois)) return true;

    currConditionEq &= (rhs.controls.squashIORois == controls.squashIORois);
    if ((currConditionEq) && (rhs.controls.fp8BiasIn < controls.fp8BiasIn)) return true;

    currConditionEq &= (rhs.controls.fp8BiasIn == controls.fp8BiasIn);
    if ((currConditionEq) && (rhs.controls.fp8BiasIn2 < controls.fp8BiasIn2)) return true;

    currConditionEq &= (rhs.controls.fp8BiasIn2 == controls.fp8BiasIn2);
    if ((currConditionEq) && (rhs.controls.fp8BiasOut < controls.fp8BiasOut)) return true;

    // strategy

    currConditionEq &= (rhs.controls.fp8BiasIn2 == controls.fp8BiasIn2);
    if ((currConditionEq) && (rhs.strategy.geometry < strategy.geometry)) return true;

    currConditionEq &= (rhs.strategy.geometry == strategy.geometry);
    if ((currConditionEq) && (rhs.strategy.loweringEn < strategy.loweringEn)) return true;

    currConditionEq &= (rhs.strategy.loweringEn == strategy.loweringEn);
    if ((currConditionEq) && (rhs.strategy.memsetDedxVoidPixels < strategy.memsetDedxVoidPixels)) return true;

    currConditionEq &= (rhs.strategy.memsetDedxVoidPixels == strategy.memsetDedxVoidPixels);
    if ((currConditionEq) && (rhs.strategy.pattern < strategy.pattern)) return true;

    currConditionEq &= (rhs.strategy.pattern == strategy.pattern);
    if ((currConditionEq) && (rhs.strategy.sbReuse < strategy.sbReuse)) return true;

    currConditionEq &= (rhs.strategy.sbReuse == strategy.sbReuse);
    if ((currConditionEq) && (rhs.strategy.unrollEn < strategy.unrollEn)) return true;

    currConditionEq &= (rhs.strategy.unrollEn == strategy.unrollEn);

    // tracing

    if ((currConditionEq) && (rhs.tracing.traceModeX < tracing.traceModeX)) return true;

    currConditionEq &= (rhs.tracing.traceModeX == tracing.traceModeX);
    if ((currConditionEq) && (rhs.tracing.traceModeY < tracing.traceModeY)) return true;

    currConditionEq &= (rhs.tracing.traceModeY == tracing.traceModeY);
    if ((currConditionEq) && (rhs.tracing.traceModeW < tracing.traceModeW)) return true;

    currConditionEq &= (rhs.tracing.traceModeW == tracing.traceModeW);

    // memory config

    if ((currConditionEq) && (rhs.memoryCfg.reductionOp < memoryCfg.reductionOp)) return true;

    currConditionEq &= (rhs.memoryCfg.reductionOp == memoryCfg.reductionOp);
    if ((currConditionEq) && (rhs.memoryCfg.reductionRm < memoryCfg.reductionRm)) return true;

    currConditionEq &= (rhs.memoryCfg.reductionRm == memoryCfg.reductionRm);

    // x,y,z + conv

    if (currConditionEq)
    {
        // sizes

        if ((rhs.x.sizes[0] < x.sizes[0]) || (rhs.x.sizes[0] == x.sizes[0] && rhs.x.sizes[1] < x.sizes[1]) ||
            (rhs.x.sizes[0] == x.sizes[0] && rhs.x.sizes[1] == x.sizes[1] && rhs.x.sizes[2] < x.sizes[2]) ||
            (rhs.x.sizes[0] == x.sizes[0] && rhs.x.sizes[1] == x.sizes[1] && rhs.x.sizes[2] == x.sizes[2] &&
             rhs.x.sizes[3] < x.sizes[3]))
        {
            return true;
        }

        currConditionEq &= (rhs.x.sizes[0] == x.sizes[0] && rhs.x.sizes[1] == x.sizes[1] &&
                            rhs.x.sizes[2] == x.sizes[2] && rhs.x.sizes[3] == x.sizes[3]);

        if ((currConditionEq) &&
            ((rhs.y.sizes[0] < y.sizes[0]) || (rhs.y.sizes[0] == y.sizes[0] && rhs.y.sizes[1] < y.sizes[1]) ||
             (rhs.y.sizes[0] == y.sizes[0] && rhs.y.sizes[1] == y.sizes[1] && rhs.y.sizes[2] < y.sizes[2]) ||
             (rhs.y.sizes[0] == y.sizes[0] && rhs.y.sizes[1] == y.sizes[1] && rhs.y.sizes[2] == y.sizes[2] &&
              rhs.y.sizes[3] < y.sizes[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.y.sizes[0] == y.sizes[0] && rhs.y.sizes[1] == y.sizes[1] &&
                            rhs.y.sizes[2] == y.sizes[2] && rhs.y.sizes[3] == y.sizes[3]);

        if ((currConditionEq) &&
            ((rhs.w.sizes[0] < w.sizes[0]) || (rhs.w.sizes[0] == w.sizes[0] && rhs.w.sizes[1] < w.sizes[1]) ||
             (rhs.w.sizes[0] == w.sizes[0] && rhs.w.sizes[1] == w.sizes[1] && rhs.w.sizes[2] < w.sizes[2]) ||
             (rhs.w.sizes[0] == w.sizes[0] && rhs.w.sizes[1] == w.sizes[1] && rhs.w.sizes[2] == w.sizes[2] &&
              rhs.w.sizes[3] < w.sizes[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.w.sizes[0] == w.sizes[0] && rhs.w.sizes[1] == w.sizes[1] &&
                            rhs.w.sizes[2] == w.sizes[2] && rhs.w.sizes[3] == w.sizes[3]);

        // strides

        if ((currConditionEq) && ((rhs.x.strides[0] < x.strides[0]) ||
                                  (rhs.x.strides[0] == x.strides[0] && rhs.x.strides[1] < x.strides[1]) ||
                                  (rhs.x.strides[0] == x.strides[0] && rhs.x.strides[1] == x.strides[1] &&
                                   rhs.x.strides[2] < x.strides[2]) ||
                                  (rhs.x.strides[0] == x.strides[0] && rhs.x.strides[1] == x.strides[1] &&
                                   rhs.x.strides[2] == x.strides[2] && rhs.x.strides[3] < x.strides[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.x.strides[0] == x.strides[0] && rhs.x.strides[1] == x.strides[1] &&
                            rhs.x.strides[2] == x.strides[2] && rhs.x.strides[3] == x.strides[3]);

        if ((currConditionEq) && ((rhs.y.strides[0] < y.strides[0]) ||
                                  (rhs.y.strides[0] == y.strides[0] && rhs.y.strides[1] < y.strides[1]) ||
                                  (rhs.y.strides[0] == y.strides[0] && rhs.y.strides[1] == y.strides[1] &&
                                   rhs.y.strides[2] < y.strides[2]) ||
                                  (rhs.y.strides[0] == y.strides[0] && rhs.y.strides[1] == y.strides[1] &&
                                   rhs.y.strides[2] == y.strides[2] && rhs.y.strides[3] < y.strides[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.y.strides[0] == y.strides[0] && rhs.y.strides[1] == y.strides[1] &&
                            rhs.y.strides[2] == y.strides[2] && rhs.y.strides[3] == y.strides[3]);

        if ((currConditionEq) && ((rhs.w.strides[0] < w.strides[0]) ||
                                  (rhs.w.strides[0] == w.strides[0] && rhs.w.strides[1] < w.strides[1]) ||
                                  (rhs.w.strides[0] == w.strides[0] && rhs.w.strides[1] == w.strides[1] &&
                                   rhs.w.strides[2] < w.strides[2]) ||
                                  (rhs.w.strides[0] == w.strides[0] && rhs.w.strides[1] == w.strides[1] &&
                                   rhs.w.strides[2] == w.strides[2] && rhs.w.strides[3] < w.strides[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.w.strides[0] == w.strides[0] && rhs.w.strides[1] == w.strides[1] &&
                            rhs.w.strides[2] == w.strides[2] && rhs.w.strides[3] == w.strides[3]);

        // bases

        if ((currConditionEq) &&
            ((rhs.x.bases[0] < x.bases[0]) || (rhs.x.bases[0] == x.bases[0] && rhs.x.bases[1] < x.bases[1]) ||
             (rhs.x.bases[0] == x.bases[0] && rhs.x.bases[1] == x.bases[1] && rhs.x.bases[2] < x.bases[2]) ||
             (rhs.x.bases[0] == x.bases[0] && rhs.x.bases[1] == x.bases[1] && rhs.x.bases[2] == x.bases[2] &&
              rhs.x.bases[3] < x.bases[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.x.bases[0] == x.bases[0] && rhs.x.bases[1] == x.bases[1] &&
                            rhs.x.bases[2] == x.bases[2] && rhs.x.bases[3] == x.bases[3]);

        if ((currConditionEq) &&
            ((rhs.y.bases[0] < y.bases[0]) || (rhs.y.bases[0] == y.bases[0] && rhs.y.bases[1] < y.bases[1]) ||
             (rhs.y.bases[0] == y.bases[0] && rhs.y.bases[1] == y.bases[1] && rhs.y.bases[2] < y.bases[2]) ||
             (rhs.y.bases[0] == y.bases[0] && rhs.y.bases[1] == y.bases[1] && rhs.y.bases[2] == y.bases[2] &&
              rhs.y.bases[3] < y.bases[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.y.bases[0] == y.bases[0] && rhs.y.bases[1] == y.bases[1] &&
                            rhs.y.bases[2] == y.bases[2] && rhs.y.bases[3] == y.bases[3]);

        if ((currConditionEq) &&
            ((rhs.w.bases[0] < w.bases[0]) || (rhs.w.bases[0] == w.bases[0] && rhs.w.bases[1] < w.bases[1]) ||
             (rhs.w.bases[0] == w.bases[0] && rhs.w.bases[1] == w.bases[1] && rhs.w.bases[2] < w.bases[2]) ||
             (rhs.w.bases[0] == w.bases[0] && rhs.w.bases[1] == w.bases[1] && rhs.w.bases[2] == w.bases[2] &&
              rhs.w.bases[3] < w.bases[3])))
        {
            return true;
        }

        currConditionEq &= (rhs.w.bases[0] == w.bases[0] && rhs.w.bases[1] == w.bases[1] &&
                            rhs.w.bases[2] == w.bases[2] && rhs.w.bases[3] == w.bases[3]);

        // elementType
        if ((currConditionEq) && (rhs.x.elementType < x.elementType))
        {
            return true;
        }

        currConditionEq &= (rhs.x.elementType == x.elementType);

        if ((currConditionEq) && (rhs.y.elementType < y.elementType))
        {
            return true;
        }

        currConditionEq &= (rhs.y.elementType == y.elementType);

        if ((currConditionEq) && (rhs.w.elementType < w.elementType))
        {
            return true;
        }

        currConditionEq &= (rhs.w.elementType == w.elementType);

        // conv

        if ((currConditionEq) && (rhs.conv.paddingValue < conv.paddingValue))
        {
            return true;
        }

        currConditionEq &= (rhs.conv.paddingValue == conv.paddingValue);

        if ((currConditionEq) && ((rhs.conv.stride[0] < conv.stride[0]) ||
                                  (rhs.conv.stride[0] == conv.stride[0] && rhs.conv.stride[1] < conv.stride[1]) ||
                                  (rhs.conv.stride[0] == conv.stride[0] && rhs.conv.stride[1] == conv.stride[1] &&
                                   rhs.conv.stride[2] < conv.stride[2])))
        {
            return true;
        }

        currConditionEq &= (rhs.conv.stride[0] == conv.stride[0] && rhs.conv.stride[1] == conv.stride[1] &&
                            rhs.conv.stride[2] == conv.stride[2]);

        if ((currConditionEq) &&
            ((rhs.conv.dilation[0] < conv.dilation[0]) ||
             (rhs.conv.dilation[0] == conv.dilation[0] && rhs.conv.dilation[1] < conv.dilation[1]) ||
             (rhs.conv.dilation[0] == conv.dilation[0] && rhs.conv.dilation[1] == conv.dilation[1] &&
              rhs.conv.dilation[2] < conv.dilation[2])))
        {
            return true;
        }

        currConditionEq &= (rhs.conv.dilation[0] == conv.dilation[0] && rhs.conv.dilation[1] == conv.dilation[1] &&
                            rhs.conv.dilation[2] == conv.dilation[2]);

        if ((currConditionEq) && ((rhs.conv.padding[0] < conv.padding[0]) ||
                                  (rhs.conv.padding[0] == conv.padding[0] && rhs.conv.padding[1] < conv.padding[1]) ||
                                  (rhs.conv.padding[0] == conv.padding[0] && rhs.conv.padding[1] == conv.padding[1] &&
                                   rhs.conv.padding[2] < conv.padding[2])))
        {
            return true;
        }
    }
    return false;
}

std::string MmeStrategy::print() const
{
    std::stringstream ss;
    ss << "MmeStrategy:" << std::endl;
    ss << "geometry: " << fmt::format("{}",geometry) << " pattern: " << fmt::format("{}",pattern) << " mmeLimit: " << mmeLimit << std::endl;
    ss << "pipelineLevel: " << pipelineLevel << " packingFactor: " << packingFactor << " reductionLevel: " << reductionLevel << std::endl;
    ss << "lowering: " << loweringEn << " sbResue: " << sbReuse << " alignedAddresses: " << alignedAddresses << std::endl;
    ss << "batchConcurrency: " << fmt::format("{}", batchConcurrencyEn) << " CD Concurrency: " << fmt::format("{}", cdConcurrencyEn) << " isDeterministic: " << isDeterministic << std::endl;
    ss << "flattenEn: " << flattenEn << " maskedBGemm: " << maskedBgemm << " recurringMisAlignment: "<< recurringMisalignmentOptEn << std::endl;
    ss << "teAcceleration: " << teAccelerationEn << " partialsToMemoryEn: " << partialsToMemoryEn << std::endl;
    ss << "dualGemm: " << dualGemm << " partial: " << partial << " signalPartial: "<< signalPartial << std::endl;
    ss << "memsetDedxVoidPixels: " << memsetDedxVoidPixels << " dedxDynamicPadding: " << dedxDynamicPadding << std::endl;
    return ss.str();
}
}  // namespace MmeCommon
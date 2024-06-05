#include "const_tensor_optimizer.h"

// eager includes (relative to src/eager/lib/)
#include "synapse_common_types.h"
#include "utils/memory_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_global_conf.h"

// synapse api (relative to include/)
#include "internal/define_synapse_common.hpp"
#include "tensor.h"

// relative to <mme>/
#include "mme_reference/data_types/bfloat16.h"
#include "mme_reference/data_types/fp16.h"

// std includes
#include <cfenv>
#include <cmath>
#include <optional>
#include <string_view>

namespace eager_mode
{
// unnamed namespace to avoid exposing the type
namespace
{
class RoundingModeSetter
{
public:
    RoundingModeSetter(int roundingMode) : m_requestedRoundingMode(roundingMode), m_originalRoundingMode(fegetround())
    {
        if (m_requestedRoundingMode != m_originalRoundingMode)
        {
            fesetround(m_requestedRoundingMode);
        }
    }
    ~RoundingModeSetter()
    {
        if (m_requestedRoundingMode != m_originalRoundingMode)
        {
            fesetround(m_originalRoundingMode);
        }
    }

private:
    const int m_requestedRoundingMode;
    const int m_originalRoundingMode;
};
}  // namespace

static constexpr std::optional<int> translateToLibcRoundingMode(_CastF32RoundMode_t from)
{
    switch (from)
    {
        case CAST_ROUND_DEFAULT:
        case CAST_ROUND_HALF_NE:
            return FE_TONEAREST;
        case CAST_ROUND_DOWN:
            return FE_DOWNWARD;
        case CAST_ROUND_UP:
            return FE_UPWARD;
        case CAST_ROUND_ZERO:
            return FE_TOWARDZERO;
        default:
            return std::nullopt;
    }
}

static constexpr MmeCommon::RoundingMode translateToMMERoundingMode(_CastF32RoundMode_t from)
{
    switch (from)
    {
        case CAST_ROUND_DEFAULT:
        case CAST_ROUND_HALF_NE:
            return MmeCommon::RoundingMode::RoundToNearest;
        case CAST_ROUND_DOWN:
            return MmeCommon::RoundingMode::RoundDown;
        case CAST_ROUND_UP:
            return MmeCommon::RoundingMode::RoundUp;
        case CAST_ROUND_SR:
            return MmeCommon::RoundingMode::StochasticRounding;
        case CAST_ROUND_ZERO:
            return MmeCommon::RoundingMode::RoundToZero;
        case CAST_ROUND_HALF_AZ:
            return MmeCommon::RoundingMode::RoundAwayFromZero;
        default:
            EAGER_ASSERT(false, "unsupported rounding mode for eager {}", from);
            return MmeCommon::RoundingMode::RoundToNearest;
    }
}

template<typename TO, typename FROM>
static inline TO clamp(FROM val)
{
    static_assert(std::is_integral_v<FROM>, "roundAndClamp flavor is only intended for integers");
    constexpr auto range_min = std::numeric_limits<TO>::min();
    constexpr auto range_max = std::numeric_limits<TO>::max();
    if (std::is_unsigned_v<FROM>)
    {
        return static_cast<TO>(std::clamp<uint64_t>(val, 0, range_max));
    }
    return static_cast<TO>(std::clamp<int64_t>(val, range_min, range_max));
}

template<typename T>
static inline T roundAndClamp(float val, CastF32RoundMode_t roundingMode = CastF32RoundMode_t::CAST_ROUND_HALF_AZ)
{
    static_assert(std::is_integral_v<T>, "roundAndClamp flavor is only intended for integers");
    if (roundingMode == CastF32RoundMode_t::CAST_ROUND_HALF_AZ)
    {
        val = std::round(val);
    }
    else
    {
        EAGER_ASSERT(translateToLibcRoundingMode(roundingMode).has_value(),
                     "unsupported rounding mode used {}",
                     roundingMode);
        EAGER_ASSERT(fegetround() == translateToLibcRoundingMode(roundingMode),
                     "rounding mode was not updated {}",
                     roundingMode);
        val = std::nearbyint(val);
    }
    constexpr auto range_min  = static_cast<float>(std::numeric_limits<T>::min());
    constexpr auto range_max  = static_cast<float>(std::numeric_limits<T>::max());
    float          clampedVal = std::clamp<float>(val, range_min, range_max);
    // widening to biggest integer resolution to workaround edge cases as INT32_MAX
    // is not representable by a float and as such clamp would clamp to INT32_MAX + 1
    // causing the cast to turn a positive large integer into a negative one.
    if (std::is_unsigned_v<T>)
    {
        return clamp<T, uint64_t>(clampedVal);
    }
    return clamp<T, int64_t>(clampedVal);
}

template<typename T>
static inline bool setAsConstTensor(const TensorPtr& t, T val)
{
    size_t tensorSizeInBytes = t->getDenseSizeInBytes();
    size_t numElements       = tensorSizeInBytes / sizeof(T);
    auto   data              = new T[numElements];
    std::fill_n(data, numElements, val);
    t->bind(data, /*shouldFreeBuffer*/ true);
    t->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
    return true;  // allows cleaner code in the caller switch case
}

static inline bool isUnsignedType(synDataType dataType)
{
    switch (dataType)
    {
        case syn_type_uint4:
        case syn_type_uint8:
        case syn_type_uint16:
        case syn_type_uint32:
        case syn_type_uint64:
            return true;
        default:
            return false;
    }
}

static inline bool isSignedToUnsignedSameWidthInt(synDataType from, synDataType to)
{
    switch (from)
    {
        case syn_type_int4:
            return to == syn_type_uint4;
        case syn_type_int8:
            return to == syn_type_uint8;
        case syn_type_int16:
            return to == syn_type_uint16;
        case syn_type_int32:
            return to == syn_type_uint32;
        case syn_type_int64:
            return to == syn_type_uint64;
        default:
            return false;
    }
}

static inline bool isUnsignedToSignedSameWidthInt(synDataType from, synDataType to)
{
    return isSignedToUnsignedSameWidthInt(to, from);
}

static inline bool isConstOutputTensorIncreasingRecipeSize(const TensorPtr& output, size_t nodeRecipeBytes)
{
    // we skip the optimization in case including the const tensor in the recipe
    // will increase the recipe size, hence adding overhead to H2D copy.
    // this is not a completly accurate check as we assume the tensor will also
    // be omitted from the recipe which is not guaranteed in case there are remaining
    // nodes with the same kernel (we include each kernel once).
    return (output->getDenseSizeInBytes() > nodeRecipeBytes);
}

static inline bool isTensorUsingDefaultQuantization(const Tensor& t)
{
    return t.getQuantizationParams().getChannelParams() ==
           QuantizationData::getDefaultQuantizationChannelParams(t.getElementType());
}

bool ConstantTensorOptimizer::canReplaceConstantByConstTensor(const TensorVector& inputs,
                                                              const TensorVector& outputs,
                                                              std::string_view    guid) const
{
    const TensorPtr& output = outputs[0];
    // We only optimize for complex guid and user node context so we do not expect
    // to have to deal with reduction or RMW sections and as such do not check for
    // those.

    // we avoid the optimization for persistent outputs since those can not be const and have to written to the device.
    if (output->isPersistent()) return false;

    if (isConstOutputTensorIncreasingRecipeSize(output, m_recipeHal.getConstantNodeRecipeMemoryRequirement()))
    {
        return false;
    }

    // we do not support constant flavor receiving an input tensor
    if (!inputs.empty()) return false;

    // for the moment we do not support quantization
    if (!isTensorUsingDefaultQuantization(*output))
    {
        EAGER_ASSERT(false,
                     "constant tensor has non default qunatization {}, skipping const tensor optimization",
                     output->getName());
        return false;
    }

    // for the moment we do not support strided tensors
    if (!output->isTrivialStrided())
    {
        EAGER_ASSERT(false,
                     "constant tensor has non default strides {}, skipping const tensor optimization",
                     output->getName());
        return false;
    }

    return true;
}

bool ConstantTensorOptimizer::canReplaceCastByConstTensor(const TensorVector& inputs,
                                                          const TensorVector& outputs,
                                                          std::string_view    guid,
                                                          CastF32RoundMode_t  roundingMode) const
{
    const TensorPtr& input  = inputs[0];
    const TensorPtr& output = outputs[0];

    // only possible if input tensor is const and output is intermidate.
    // as if output is persistent we would still need to insert a memcpy.
    if (output->isPersistent() || input->getMemorySectionID() != MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
    {
        return false;
    }

    // for the moment we only optimize for cast of single element const tensor.
    // It is possible to optimize for a repeated value const tensor, but this
    // incurs some overhead for detecting this pattern.
    // It is also possible to support a general case of const tensor but this
    // adds more overhead as we need to make rounding calculations per element.
    if (output->getTotalElements() != 1)
    {
        EAGER_ASSERT(false,
                     "cast output tensor has more than a single element {}, skipping const tensor optimization",
                     output->getName());
        return false;
    }
    // To support the logic for repeated single values one needs to
    // replace the check above and enable the following:

#if 0
    if (isConstOutputTensorIncreasingRecipeSize(output, m_recipeHal.getCastNodeRecipeMemoryRequirement(input->getElementType(), output->getElementType())))
    {
        return false;
    }

    // for the moment we do not support strided tensors
    for(const Tensor* t : {input.get(), output.get()})
    {
        if (!t->isTrivialStrided())
        {
            EAGER_ASSERT(false,
                    "cast tensor has non default strides {}, skipping const tensor optimization",
                    t->getName());
            return false;
        }
    }
#endif

    // for the moment we do not support quantization
    for (const Tensor* t : {input.get(), output.get()})
    {
        if (!isTensorUsingDefaultQuantization(*t))
        {
            EAGER_ASSERT(false,
                         "cast tensor has non default qunatization {}, skipping const tensor optimization",
                         t->getName());
            return false;
        }
    }

    // cheaper to compare based on length instead of constructing a string for comparison
    auto   fromDataTypeStr = getDtypeSuffixFromSynDataType(input->getElementType());
    auto   toDataTypeStr   = getDtypeSuffixFromSynDataType(output->getElementType());
    size_t castStrLength   = std::string_view("cast__to_").size() + fromDataTypeStr.size() + toDataTypeStr.size();

    if (castStrLength != guid.size())
    {
        // we currently do not support specialized flavors such as cast_tf_f32_to_u32.
        // lfsr and lut are only applicable to Gaudi device so are not applicable to Eager
        // anyhow.
        EAGER_ASSERT(false,
                     "cast kernel has unsupported specialized guid {}, skipping const tensor optimization",
                     guid);
        return false;
    }

    // for now we only support rounding modes available through libc (everything up to stochastic).
    // CAST_ROUND_HALF_AZ is available through round API and rest through rint\nearbyint.
    std::optional<int> libcRoundingMode = translateToLibcRoundingMode(roundingMode);
    if (!libcRoundingMode.has_value() && roundingMode != CAST_ROUND_HALF_AZ)
    {
        EAGER_ASSERT(false,
                     "cast kernel with unsupported rounding mode {}, skipping const tensor optimization",
                     roundingMode);
        return false;
    }
    return true;
}

bool ConstantTensorOptimizer::tryReplaceConstantByConstTensor(const TensorPtr& output, ns_ConstantKernel::Params val)
{
    switch (output->getElementType())
    {
        case syn_type_int8:
            return setAsConstTensor(output, roundAndClamp<int8_t>(val.constant.f));
        case syn_type_uint8:
            return setAsConstTensor(output, roundAndClamp<uint8_t>(val.constant.f));
        case syn_type_int16:
            return setAsConstTensor(output, roundAndClamp<int16_t>(val.constant.f));
        case syn_type_uint16:
            return setAsConstTensor(output, roundAndClamp<uint16_t>(val.constant.f));
        case syn_type_int32:
            return setAsConstTensor(output, static_cast<int32_t>(val.constant.i));
        case syn_type_uint32:
            return setAsConstTensor(output, static_cast<uint32_t>(val.constant.i));
        case syn_type_float:
            return setAsConstTensor(output, static_cast<float>(val.constant.f));
        case syn_type_bf16:
            return setAsConstTensor(output, bfloat16(val.constant.f).value());
        case syn_type_fp16:
            return setAsConstTensor(output, HalfFloat<true>(val.constant.f).value());
        default:
            return false;
    }
}

bool ConstantTensorOptimizer::tryReplaceConstantByConstTensor(const TensorVector& inputs,
                                                              const TensorVector& outputs,
                                                              UserParams          userParams,
                                                              unsigned            userParamsSize,
                                                              std::string_view    guid) const
{
    if (sizeof(ns_ConstantKernel::Params) != userParamsSize)
    {
        EAGER_ASSERT(false, "constant kernel has unexpected parameters size {}", userParamsSize);
        return false;
    }

    if (!canReplaceConstantByConstTensor(inputs, outputs, guid)) return false;

    // we can still fail the conversion if this is an unsupported type
    auto constantVal = readAs<ns_ConstantKernel::Params>(userParams);
    return tryReplaceConstantByConstTensor(outputs[0], constantVal);
}

bool ConstantTensorOptimizer::tryReplaceCastFromFloatToIntByConstTensor(const TensorPtr&   output,
                                                                        float              val,
                                                                        CastF32RoundMode_t roundingMode)
{
    // for now we only support rounding modes available through libc (everything up to stochastic).
    // CAST_ROUND_HALF_AZ is available through round API and rest through rint\nearbyint.
    std::optional<int>                libcRoundingMode = translateToLibcRoundingMode(roundingMode);
    std::optional<RoundingModeSetter> roundingModeSetter;

    if (libcRoundingMode.has_value() && roundingMode != CAST_ROUND_HALF_AZ)
    {
        roundingModeSetter.emplace(*libcRoundingMode);
    }

    switch (output->getElementType())
    {
        case syn_type_int8:
            return setAsConstTensor(output, roundAndClamp<int8_t>(val, roundingMode));
        case syn_type_uint8:
            return setAsConstTensor(output, roundAndClamp<uint8_t>(val, roundingMode));
        case syn_type_int16:
            return setAsConstTensor(output, roundAndClamp<int16_t>(val, roundingMode));
        case syn_type_uint16:
            return setAsConstTensor(output, roundAndClamp<uint16_t>(val, roundingMode));
        case syn_type_int32:
            return setAsConstTensor(output, roundAndClamp<int32_t>(val, roundingMode));
        case syn_type_uint32:
            return setAsConstTensor(output, roundAndClamp<uint32_t>(val, roundingMode));
        default:
            return false;
    }
}

bool ConstantTensorOptimizer::tryReplaceCastFromFloatToFloatByConstTensor(const TensorPtr&   output,
                                                                          float              val,
                                                                          CastF32RoundMode_t roundingMode)
{
    switch (output->getElementType())
    {
        case syn_type_float:
            return setAsConstTensor(output, val);
        case syn_type_bf16:
            return setAsConstTensor(output, bfloat16(val, translateToMMERoundingMode(roundingMode)).value());
        case syn_type_fp16:
            return setAsConstTensor(output, HalfFloat<true>(val, translateToMMERoundingMode(roundingMode)).value());
        default:
            return false;
    }
}

template<typename T>
bool ConstantTensorOptimizer::tryReplaceCastFromIntByConstTensor(const TensorPtr& output, T val)
{
    switch (output->getElementType())
    {
        case syn_type_int8:
            return setAsConstTensor(output, static_cast<int8_t>(val));
        case syn_type_uint8:
            return setAsConstTensor(output, static_cast<uint8_t>(val));
        case syn_type_int16:
            return setAsConstTensor(output, static_cast<int16_t>(val));
        case syn_type_uint16:
            return setAsConstTensor(output, static_cast<uint16_t>(val));
        case syn_type_int32:
            return setAsConstTensor(output, static_cast<int32_t>(val));
        case syn_type_uint32:
            return setAsConstTensor(output, static_cast<uint32_t>(val));
        case syn_type_float:
            return setAsConstTensor(output, static_cast<float>(val));
        case syn_type_bf16:
            return setAsConstTensor(output, bfloat16(static_cast<float>(val)).value());
        case syn_type_fp16:
            return setAsConstTensor(output, HalfFloat<true>(static_cast<float>(val)).value());
        default:
            return false;
    }
}

bool ConstantTensorOptimizer::tryReplaceCastByConstTensor(const TensorPtr&   input,
                                                          const TensorPtr&   output,
                                                          CastF32RoundMode_t roundingMode)
{
    auto tryReplacement = [&](auto val) {
        synDataType toDataType = output->getElementType();
        // handle cases of integral from value
        if constexpr (std::is_integral_v<decltype(val)>)
        {
            synDataType fromDataType = input->getElementType();
            if (isSignedToUnsignedSameWidthInt(fromDataType, toDataType) && val < 0)
            {
                val = 0;
            }
            if (isUnsignedToSignedSameWidthInt(fromDataType, toDataType))
            {
                val = clamp<std::make_signed_t<decltype(val)>>(val);
            }
            return tryReplaceCastFromIntByConstTensor(output, val);
        }
        // handle cases of floating point from value
        if (isTypeFloat(toDataType))
        {
            return tryReplaceCastFromFloatToFloatByConstTensor(output, val, roundingMode);
        }
        if (isUnsignedType(toDataType) && val < 0)
        {
            // default cast behavior rounds float negative values to zero (unlike cast_tf_*)
            val = 0;
        }
        return tryReplaceCastFromFloatToIntByConstTensor(output, val, roundingMode);
    };

    switch (input->getElementType())
    {
        case syn_type_float:
        {
            return tryReplacement(readAs<float>(input->getData()));
        }
        case syn_type_bf16:
        {
            return tryReplacement(Bfloat16(readAs<uint16_t>(input->getData())).toFloat());
        }
        case syn_type_fp16:
        {
            return tryReplacement(HalfFloat<true>(readAs<uint16_t>(input->getData())).toFloat());
        }
        case syn_type_int32:
        {
            return tryReplacement(readAs<int32_t>(input->getData()));
        }
        case syn_type_uint32:
        {
            return tryReplacement(readAs<uint32_t>(input->getData()));
        }
        case syn_type_int16:
        {
            return tryReplacement(readAs<int16_t>(input->getData()));
        }
        case syn_type_uint16:
        {
            return tryReplacement(readAs<uint16_t>(input->getData()));
        }
        case syn_type_int8:
        {
            return tryReplacement(readAs<int8_t>(input->getData()));
        }
        case syn_type_uint8:
        {
            return tryReplacement(readAs<uint8_t>(input->getData()));
        }
        default:
            return false;
    }
}

static inline std::optional<CastF32RoundMode_t> getRoundingMode(UserParams userParams, unsigned userParamsSize)
{
    switch (userParamsSize)
    {
        // empty user params is a special case where we use the default values.
        // consistent with tpc kernels handling.
        case 0:
            return CAST_ROUND_DEFAULT;
        case sizeof(ns_CastKernel::Params):
            return readAs<ns_CastKernel::Params>(userParams).round_mode;
        case sizeof(ns_CastKernel::ParamsV2):
            return readAs<ns_CastKernel::ParamsV2>(userParams).round_mode;
        case sizeof(ns_CastKernel::ParamsV3):
            return readAs<ns_CastKernel::ParamsV3>(userParams).round_mode;
        case sizeof(ns_CastKernel::ParamsV4):
            return readAs<ns_CastKernel::ParamsV4>(userParams).round_mode;
        default:
            return std::nullopt;
    }
}

// we rely on nodes to be handled based on topological order to handle constant
// nodes before cast nodes, otherwise we might no optimize out the cast even
// when we can. For the moment seems the constant nodes acting as cast inputs
// always come prior to the cast nodes.
bool ConstantTensorOptimizer::tryReplaceCastByConstTensor(const TensorVector& inputs,
                                                          const TensorVector& outputs,
                                                          UserParams          userParams,
                                                          unsigned            userParamsSize,
                                                          std::string_view    guid) const
{
    std::optional<CastF32RoundMode_t> roundingMode = getRoundingMode(userParams, userParamsSize);
    if (!roundingMode.has_value())
    {
        EAGER_ASSERT(false,
                     "cast kernel has unsupported parameters size {}, skipping const tensor optimization",
                     userParamsSize);
        return false;
    }
    if (!canReplaceCastByConstTensor(inputs, outputs, guid, *roundingMode)) return false;
    return tryReplaceCastByConstTensor(inputs[0], outputs[0], *roundingMode);
}

bool ConstantTensorOptimizer::tryReplaceNodeByConstTensor(const TensorVector& inputs,
                                                          const TensorVector& outputs,
                                                          UserParams          userParams,
                                                          unsigned            userParamsSize,
                                                          std::string_view    guid) const
{
    if (!GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value()) return false;
    if (GCFG_ENABLE_CAST_OPTIMIZATION_IN_EAGER.value() && isCastGUID(guid))
    {
        return tryReplaceCastByConstTensor(inputs, outputs, userParams, userParamsSize, guid);
    }
    if (GCFG_ENABLE_CONSTANT_OPTIMIZATION_IN_EAGER.value() && isConstantGUID(guid))
    {
        return tryReplaceConstantByConstTensor(inputs, outputs, userParams, userParamsSize, guid);
    }
    return false;
}

}  // namespace eager_mode

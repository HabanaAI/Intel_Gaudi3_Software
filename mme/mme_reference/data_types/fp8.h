#pragma once

#include <limits>
#include <mme_reference/data_types/fp32.h>
#include "fs_fma_gaudi3.h"
#include "include/general_utils.h"
#include "include/mme_common/mme_common_enum.h"


#define EXPONENT_BIAS_FP8_152_15        15

#define EXPONENT_BIAS_FP8_152_MIN_VALUE 1

#define EXPONENT_BIAS_FP8_152_MAX_VALUE 30

#define EXPONENT_BIAS_FP8_143_TYPES     4

#define EXPONENT_BIAS_FP8_143_3         3

#define EXPONENT_BIAS_FP8_143_7         7

#define EXPONENT_BIAS_FP8_143_11        11

#define EXPONENT_BIAS_FP8_143_15        15

#define FP8_MODE143_EXP 4
#define FP8_MODE143_MAN 3
#define FP8_MODE152_EXP 5
#define FP8_MODE152_MAN 2
#define FP8_MODE152_BIAS 15

// Wrapper to FuncSim's FP8 operations
template <uint8_t EXPONENT_SIZE>
class FP8Wrapper
{
protected:
    // Compact alias
    using FType = FP8Wrapper<EXPONENT_SIZE>;

    enum : uint8_t
    {
        SIZE = 8 * sizeof(uint8_t),  // Size in bits of the float object
        // Define mantissa size
        MANTISSA_SIZE = SIZE - 1 /*sign size*/ - EXPONENT_SIZE,
        // Define default bias value
        DEFAULT_BIAS = EXPONENT_SIZE == FP8_MODE143_EXP ? EXPONENT_BIAS_FP8_143_15 : EXPONENT_BIAS_FP8_152_15,
    };

public:
    // Construction
    explicit FP8Wrapper(uint8_t value = 0) : val(value) {}
    FP8Wrapper(float f,
               MmeCommon::RoundingMode rm,
               unsigned expBias = DEFAULT_BIAS,
               int32_t lfsrVal = 0,
               bool flushDenorms = false,
               bool clipFp = false,
               bool clipInfIn = false,
               bool stochasticFTZfp8 = false,
               MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        val =
            (uint8_t) fp32ToFp8(f, rm, expBias, lfsrVal, flushDenorms, clipFp, clipInfIn, stochasticFTZfp8, infNanMode);
    }

    static FType fma(const FType& a,
                     const FType& b,
                     const FType& c,
                     MmeCommon::RoundingMode rm,
                     uint8_t expBiasIn,
                     uint8_t expBiasOut)
    {
        float cAsFp32 = c.toFloat(expBiasIn);
        uint32_t ui32c = reinterpret_ptr<uint32_t>(&cAsFp32);
        auto res = gaudi3::fma_fp8_fp32((uint8_t) a,
                                        (uint8_t) b,
                                        ui32c,
                                        (uint8_t) rm,
                                        EXPONENT_SIZE,
                                        FType::MANTISSA_SIZE,
                                        expBiasIn);
        return FType(reinterpret_ptr<float>(&res), rm, expBiasOut);
    }

    static float
    fma(const FType& a, const FType& b, float c, MmeCommon::RoundingMode rm, uint8_t expBias = DEFAULT_BIAS)
    {
        uint32_t ui32c = reinterpret_ptr<uint32_t>(&c);
        auto res = gaudi3::fma_fp8_fp32((uint8_t) a,
                                        (uint8_t) b,
                                        ui32c,
                                        (uint8_t) rm,
                                        EXPONENT_SIZE,
                                        FType::MANTISSA_SIZE,
                                        expBias);
        float val = reinterpret_ptr<float>(&res);
        MME_ASSERT(std::isnan(val) != true, "got NaN value after fma");
        return val;
    }

    static FType
    add(const FType& a, const FType& b, MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest)
    {
        return fma(a, FType(1.0f), b, rm);
    }

    static FType max(unsigned expBias, MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        return fp32ToFp8(std::numeric_limits<float>::max(),
                         MmeCommon::RoundingMode::RoundToNearest,
                         expBias,
                         0,
                         false,
                         true,
                         false,
                         false,
                         infNanMode);
    }
    static FType min(unsigned expBias, MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        return fp32ToFp8(std::numeric_limits<float>::min(),
                         MmeCommon::RoundingMode::RoundToNearest,
                         expBias,
                         0,
                         false,
                         true,
                         false,
                         false,
                         infNanMode);
    }
    static FType lowest(unsigned expBias, MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        return fp32ToFp8(std::numeric_limits<float>::lowest(),
                         MmeCommon::RoundingMode::RoundToNearest,
                         expBias,
                         0,
                         false,
                         true,
                         false,
                         false,
                         infNanMode);
    }

    // Other operators, converters and queries..
    bool isZero() const { return gaudi3::is_zero_bfp16(val); }
    bool isNan(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        float32 fp32(fp8ToFp32(val, DEFAULT_BIAS, false, false, infNanMode));
        return fp32.isNan();
    }
    bool isInf(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        float32 fp32(fp8ToFp32(val, DEFAULT_BIAS, false, false, infNanMode));
        return fp32.isInf();
    }
    uint8_t& value() {return val;}
    const uint8_t& value() const {return val;}
    explicit operator uint8_t() const { return val; }
    explicit operator uint32_t() const { return val; }
    explicit operator bool() const { return !gaudi3::fp8_is_zero(val); }
    explicit operator double() const { return toFloat(); }
    explicit operator float() const { return toFloat(); }
    bool operator>(const FType& f) const { return (float)*this > (float)f; }
    bool operator<(const FType& f) const { return (float)*this < (float)f; }
    float operator*(const FType& f) const { return fma(*this, f, 0, MmeCommon::RoundingMode::RoundToNearest); }
    bool operator==(const FType& f) const { return val == f.val; }
    bool operator!=(const FType& rhs) const { return val != rhs.val; }

    float toFloat(unsigned expBias = DEFAULT_BIAS,
                  MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        return fp8ToFp32(val, expBias, false, false, infNanMode);
    }

private:
    uint8_t val = 0;

    static float fp8ToFp32(const uint8_t& input,
                           uint8_t expBias = DEFAULT_BIAS,
                           bool clip_fp = false,
                           bool is_fp19 = false,
                           MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        uint32_t res =
            gaudi3::fp8_to_fp32(input, EXPONENT_SIZE, FType::MANTISSA_SIZE, expBias, clip_fp, is_fp19, infNanMode);
        return reinterpret_ptr<float>(&res);
    }

    static FType fp32ToFp8(float input,
                           MmeCommon::RoundingMode rm,
                           uint8_t expBias = DEFAULT_BIAS,
                           int32_t lfsrVal = 0,
                           bool flushDenorms = false,
                           bool clipFp = false,
                           bool clipInfIn = false,
                           bool stochasticFTZfp8 = false,
                           MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        uint8_t res = gaudi3::fp32_to_fp8(input,
                                          EXPONENT_SIZE,
                                          FType::MANTISSA_SIZE,
                                          expBias,
                                          (int) rm,
                                          lfsrVal,
                                          flushDenorms,
                                          clipFp,
                                          clipInfIn,
                                          stochasticFTZfp8,
                                          infNanMode);
        return FType(res);
    }
};


// Define FP8 formats
using fp8_143_t = FP8Wrapper<FP8_MODE143_EXP>;  // 1-sign, 4-exponent, 3-mantissa
static_assert(sizeof(fp8_143_t) == 1, "size of fp8-143 must be equal to one byte.");
using fp8_152_t = FP8Wrapper<FP8_MODE152_EXP>;  // 1-sign, 5-exponent, 2-mantissa
static_assert(sizeof(fp8_152_t) == 1, "size of fp8-152 must be equal to one byte.");
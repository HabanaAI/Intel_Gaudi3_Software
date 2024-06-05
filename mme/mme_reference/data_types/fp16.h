#pragma once

#include <cmath>
#include <cstdint>
#include "fs_fma_gaudi3.h"
#include "include/general_utils.h"
#include "include/mme_assert.h"

#define FLOAT_F16_MIN_VAL     (0x0400)
#define FLOAT_F16_MAX_VAL     (0x7BFF)
#define FLOAT_F16_MIN_NEG_VAL (0xFBFF)

#define FP16_SIGN_MODE_EXP   5
#define FP16_SIGN_MODE_MAN   10
#define FP16_UNSIGN_MODE_EXP 6
#define FP16_UNSIGN_MODE_MAN 10

/*
 * This file will contain the implementation of both signed and unsigned fp16 or half-float data type (fp16)
 * fp16 has the following bit order -
 * signed fp16 -
 * 1 sign bit
 * 5 bit exponent
 * 10 bit mantissa
 * unsigned fp16 -
 * 6 bit exponent
 * 10 bit mantissa
 *
 */
template<bool SIGNED>
class HalfFloat
{
private:
    enum : uint8_t
    {
        // Define mantissa size
        EXPONENT_SIZE = SIGNED ? FP16_SIGN_MODE_EXP : FP16_UNSIGN_MODE_EXP,
        MANTISSA_SIZE = SIGNED ? FP16_SIGN_MODE_MAN : FP16_UNSIGN_MODE_MAN,
        DEFAULT_BIAS = SIGNED ? EXPONENT_BIAS_FP16_15 : EXPONENT_BIAS_UFP16_31,
    };

public:
    static HalfFloat max(unsigned expBias = DEFAULT_BIAS,
                         MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        return HalfFloat(std::numeric_limits<float>::max(),
                         MmeCommon::RoundingMode::RoundToNearest,
                         expBias,
                         0,
                         false,
                         true /*clipFP*/,
                         false,
                         infNanMode);
    }
    static HalfFloat min(unsigned expBias = DEFAULT_BIAS,
                         MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        return HalfFloat(std::numeric_limits<float>::min(),
                         MmeCommon::RoundingMode::RoundToNearest,
                         expBias,
                         0,
                         false,
                         true /*clipFP*/,
                         false,
                         infNanMode);
    }
    static HalfFloat lowest(unsigned expBias = DEFAULT_BIAS,
                            MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        return HalfFloat(std::numeric_limits<float>::lowest(),
                         MmeCommon::RoundingMode::RoundToNearest,
                         expBias,
                         0,
                         false,
                         true /*clipFP*/,
                         false,
                         infNanMode);
    }

    HalfFloat() = default;
    explicit HalfFloat(float v,
                       MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
                       uint8_t expBias = DEFAULT_BIAS,
                       uint32_t lfsrVal = 0,
                       bool flushDenorms = false,
                       bool clipFP = false,
                       bool clipInfIn = false,
                       MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan)
    {
        this->val = gaudi3::fp32_to_cfp16(v,
                                          EXPONENT_SIZE,
                                          MANTISSA_SIZE,
                                          expBias,
                                          (int) rm,
                                          lfsrVal,
                                          flushDenorms,
                                          clipFP,
                                          clipInfIn,
                                          infNanMode);
    }
    explicit HalfFloat(uint16_t bitArray) { this->val = bitArray; }
    explicit HalfFloat(uint32_t bitArray)
    {
        MME_ASSERT(gaudi3::sbs(bitArray, 31, 16) == 0, "bits will be dropped");
        val = gaudi3::sbs(bitArray, 15, 0);
    }
    ~HalfFloat() = default;
    uint16_t& value() {return val;}
    const uint16_t& value() const {return val;}
    HalfFloat& operator=(const HalfFloat& other) = default;
    // relational operators
    bool operator<(float rhs) const { return toFloat() < rhs; }
    bool operator>(float rhs) const { return toFloat() > rhs; }
    bool operator<(HalfFloat rhs) const { return toFloat() < rhs.toFloat(); }
    bool operator>(HalfFloat rhs) const { return toFloat() > rhs.toFloat(); }
    bool operator==(const HalfFloat& rhs) const { return val == rhs.val; }
    bool operator!=(const HalfFloat& rhs) const { return val != rhs.val; }
    bool isZero() const { return gaudi3::cfp16_is_zero(val, !SIGNED); }
    bool isNan(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        switch (infNanMode)
        {
            default:
                MME_ASSERT(0, "invalid infNan Mode");
            case MmeCommon::e_mme_no_inf_nan:
                return false;
            case MmeCommon::e_mme_full_inf_nan:
                return gaudi3::cfp16_is_nan(val, MANTISSA_SIZE, !SIGNED);
        }
    }
    bool isInf(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        switch (infNanMode)
        {
            default:
                MME_ASSERT(0, "invalid infNan Mode");
            case MmeCommon::e_mme_no_inf_nan:
                return false;
            case MmeCommon::e_mme_full_inf_nan:
                return gaudi3::cfp16_is_infinity(val, MANTISSA_SIZE, !SIGNED);
        }
    }
    explicit operator double() const { return toFloat(); }
    explicit operator float() const { return toFloat(); }
    /* bit representation methods */
    explicit operator uint32_t() const
    {
        float fp = toFloat();
        return reinterpret_ptr<uint32_t>(&fp);
    }
    explicit operator uint16_t() const { return val; }

    float toFloat(uint8_t expBias = DEFAULT_BIAS,
                  MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        uint32_t fpBits = gaudi3::cfp16_to_fp32(val, EXPONENT_SIZE, MANTISSA_SIZE, expBias, false, false, infNanMode);
        return reinterpret_ptr<float>(&fpBits);
    }

    static HalfFloat fma(const HalfFloat& a, const HalfFloat& b, const HalfFloat& c, MmeCommon::RoundingMode rm)
    {
        uint16_t resInBits = gaudi3::fma_fp16_fp16((uint16_t) a,
                                                   (uint16_t) b,
                                                   (uint16_t) c,
                                                   (int) rm,
                                                   false /*ftz_in*/,
                                                   true /*ftz_out*/);
        return HalfFloat(resInBits);
    }
    static float fma(const HalfFloat& a, const HalfFloat& b, const float& c, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits = gaudi3::fma_fp16_fp32(*(uint16_t*) &a,
                                                   *(uint16_t*) &b,
                                                   *(uint32_t*) &c,
                                                   (int) rm,
                                                   false /*ftz_in*/,
                                                   true /*ftz_out*/);
        return reinterpret_ptr<float>(&resInBits);
    }

    static HalfFloat
    add(const HalfFloat& a, const HalfFloat& b, MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest)
    {
        return fma(a, HalfFloat(1.0f), b, rm);
    }

private:
    // actual representation in bits of the number.
    uint16_t val = 0;
};

typedef HalfFloat<true> fp16_t;
static_assert(sizeof(fp16_t) == sizeof(uint16_t), "reinterpret casting to HalfFloat won't work");
typedef HalfFloat<false> ufp16_t;
static_assert(sizeof(ufp16_t) == sizeof(uint16_t), "reinterpret casting to HalfFloat won't work");
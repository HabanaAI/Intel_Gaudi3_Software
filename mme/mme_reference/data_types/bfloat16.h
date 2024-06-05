#pragma once

#include <cmath>
#include <cstdint>
#include "fs_fma_gaudi3.h"
#include "include/general_utils.h"
#include "include/mme_common/mme_common_enum.h"

/*
 * This file will contain the implementation of bfloat16 data type
 * bfloat16 has the following bit order -
 * 1 sign bit
 * 8 bit exponent
 * 7 bit mantissa
 *
 * conversion to float32 is trivial
 * conversion from float32 needs rounding.
 * addition and multiplication needs rounding as well as they are done on float.
 */

#define FLOAT_BF16_MIN_VAL     (0x0080)
#define FLOAT_BF16_MAX_VAL     (0x7f7f)
#define FLOAT_BF16_MIN_NEG_VAL (0xff7f)
class Bfloat16
{
public:
    static Bfloat16 max() { return Bfloat16((uint16_t) FLOAT_BF16_MAX_VAL); }
    static Bfloat16 min() { return Bfloat16((uint16_t) FLOAT_BF16_MIN_VAL); }
    static Bfloat16 lowest() { return Bfloat16((uint16_t) FLOAT_BF16_MIN_NEG_VAL); }
    Bfloat16() = default;
    Bfloat16(float v,
             MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
             uint32_t lfsrVal = 0,
             bool clipFP = false,
             bool flushDenorms = false,
             bool clipInfIn = false)
    {
        this->val = gaudi3::fp32_to_bf16(v, (int) rm, lfsrVal, clipFP, flushDenorms, clipInfIn);
    }
    explicit Bfloat16(uint16_t bitArray) { this->val = bitArray; }
    explicit Bfloat16(uint32_t bitArray) { val = gaudi3::sbs(bitArray, 15, 0); }
    ~Bfloat16() = default;
    uint16_t& value() {return val;}
    const uint16_t& value() const {return val;}
    Bfloat16& operator=(const Bfloat16& other) = default;
    // relational operators
    bool operator<(float rhs) const { return toFloat() < rhs; }
    bool operator>(float rhs) const { return toFloat() > rhs; }
    bool operator<(Bfloat16 rhs) const { return toFloat() < rhs.toFloat(); }
    bool operator>(Bfloat16 rhs) const { return toFloat() > rhs.toFloat(); }
    bool operator==(const Bfloat16& rhs) const { return val == rhs.val; }
    bool operator!=(const Bfloat16& rhs) const { return val != rhs.val; }

    bool isZero() const { return gaudi3::is_zero_bfp16(val); }
    bool isNan(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        return gaudi3::is_nan_bfp16(val);
    }
    bool isInf(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        return gaudi3::is_inf_bfp16(val);
    }
    // casting operators
    explicit operator double() const {return toFloat();}
    explicit operator float()  const {return toFloat();}
    /* bit representation methods */
    explicit operator uint32_t() const
    {
        float fp = toFloat();
        return reinterpret_ptr<uint32_t>(&fp);
    }
    operator uint16_t() const {return val;}

    float toFloat() const { return gaudi3::fs_bf16_to_f32(val); }

    static float fma(const Bfloat16& a, const Bfloat16& b, const float& c, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits =
            gaudi3::fma_bfp16_fp32((uint16_t) a, (uint16_t) b, reinterpret_ptr<uint32_t>((float*) &c), (int) rm);
        return reinterpret_ptr<float>(&resInBits);
    }

    static Bfloat16 fma(const Bfloat16& a, const Bfloat16& b, const Bfloat16& c, MmeCommon::RoundingMode rm)
    {
        uint16_t resInBits = gaudi3::fma_bfp16((uint16_t) a, (uint16_t) b, (uint16_t) c, (int) rm);
        return Bfloat16(resInBits);
    }

    static Bfloat16
    add(const Bfloat16& a, const Bfloat16& b, MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest)
    {
        return Bfloat16(gaudi3::add_bf16(a.value(), b.value(), (int) rm));
    }

private:
    // actual representation in bits of the number.
    uint16_t val = 0;
};

static_assert(sizeof(Bfloat16) == sizeof(uint16_t), "reinterpret casting to Bfloat16 won't work");
typedef Bfloat16 bf16_t;
typedef Bfloat16 bfloat16;  //  for synapse compilation - remove once synapse is aligned
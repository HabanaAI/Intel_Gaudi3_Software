#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cassert>
#include "fp32.h"
#include "fs_fma_gaudi3.h"
#include "include/general_utils.h"
#include "include/mme_common/mme_common_enum.h"

/*
 * This file will contain the implementation of tensor-float32 data type (tf32)
 * tf32 is a floating point representation (32 bit) of fp19 (19 bit)
 * fp19 has the following bit order -
 * 1 sign bit
 * 8 bit exponent
 * 10 bit mantissa
 * (19 bits in total)
 *
 * tf32 is 13 shift left of fp19.
 *
 * conversion to float32 is trivial
 * conversion from float32 needs rounding.
 * addition and multiplication needs rounding as well as they are done on float.
 */

class Tfloat32
{
public:
    static Tfloat32 max() { return Tfloat32((uint32_t) 0x7f7fe000); }
    static Tfloat32 min() { return Tfloat32((uint32_t) 0x00002000); }
    static Tfloat32 lowest() { return Tfloat32((uint32_t) 0xff7fe000); }

    Tfloat32() = default;
    explicit Tfloat32(float v,
                      MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
                      uint32_t lfsrVal = 0,
                      bool clipFp = false,
                      bool clipInfIn = false)
    {
        val = gaudi3::fp32_to_tf32(v, (int) rm, lfsrVal, clipFp, clipInfIn);
    }
    explicit Tfloat32(uint32_t bitArray)
    {
        MME_ASSERT(gaudi3::sbs(bitArray, 12, 0) == 0, "non relevant bits should be zero");
        val = bitArray;
    }
    explicit Tfloat32(uint16_t bitArray) = delete;
    ~Tfloat32() = default;
    uint32_t& value() {return val;}
    const uint32_t& value() const {return val;}
    Tfloat32& operator=(const Tfloat32& other) = default;
    // relational operators
    bool operator<(float rhs) const { return toFloat() < rhs; }
    bool operator>(float rhs) const { return toFloat() > rhs; }
    bool operator==(const Tfloat32& rhs) const { return val == rhs.val; }
    bool operator!=(const Tfloat32& rhs) const { return val != rhs.val; }

    bool isZero() const
    {
        // check that all bits are 0; sign bit is dont care.
        return gaudi3::is_zero_fp32(val);
    }
    bool isNan(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        // check that bits 10-17 are 1, and bits 0-9 are not zero
        return gaudi3::is_nan_fp32(val);
    }
    bool isInf(MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan) const
    {
        // check that bits 10-17 are 1, and bits 0-9 are zero
        return gaudi3::is_inf_fp32(val);
    }
    // casting operators
    explicit operator double() const {return toFloat();}
    explicit operator float()  const {return toFloat();}
    explicit operator uint16_t() const {assert(0 && "cannot convert to 16 bit representation"); return 0;}
    /* bit representation method */
    explicit operator uint32_t() const
    {
        return val;
    }

    float toFloat() const
    {
        uint32_t nonConstVal = val;
        return reinterpret_ptr<float>(&nonConstVal);
    }

    static float fma(const Tfloat32& a, const Tfloat32& b, const float& c, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits =
            gaudi3::fma_fp32((uint32_t) a, (uint32_t) b, reinterpret_ptr<uint32_t>((float*) &c), (int) rm);
        return reinterpret_ptr<float>(&resInBits);
    }

    static Tfloat32 fma(const Tfloat32& a, const Tfloat32& b, const Tfloat32& c, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits = gaudi3::fma_fp32((uint32_t) a, (uint32_t) b, (uint32_t) c, (int) rm);
        return Tfloat32(resInBits);
    }

    static Tfloat32
    add(const Tfloat32& a, const Tfloat32& b, MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest)
    {
        return fma(a, Tfloat32(1.0f), b, rm);
    }

private:
    uint32_t val = 0;
};

static_assert(sizeof(Tfloat32) == sizeof(uint32_t), "reinterpret casting to Tfloat32 won't work");
typedef Tfloat32 tf32_t;
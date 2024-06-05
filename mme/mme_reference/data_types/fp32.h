#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "fs_fma_gaudi3.h"
#include "include/general_utils.h"
#include "include/mme_assert.h"
#include "include/mme_common/mme_common_enum.h"

class float32
{
public:
    static float32 max() {return float32(std::numeric_limits<float>::max()); }
    static float32 min() {return float32(std::numeric_limits<float>::min()); }
    static float32 lowest() {return float32(std::numeric_limits<float>::lowest()); }

    float32() = default;
    explicit float32(float v,
                     MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
                     bool clipFP = false,
                     bool flushDenorms = false,
                     bool clipInfIn = false)
    {
        this->val = gaudi3::fp32_to_fp32(v, (int) rm, clipFP, flushDenorms, clipInfIn);
    }
    explicit float32(uint32_t bitArray)
    {
        val = bitArray;
    }
    explicit float32(uint16_t bitArray) {MME_ASSERT(0, "cannot initialize with 16 bit input"); val = 0;}
    ~float32() = default;
    uint32_t& value() {return val;}
    const uint32_t& value() const {return val;}
    float32& operator=(const float32& other) = default;
    // relational operators
    bool operator<(float rhs) const { return toFloat() < rhs; }
    bool operator>(float rhs) const { return toFloat() > rhs; }
    bool operator==(const float32& rhs) const { return val == rhs.val; }
    bool operator!=(const float32& rhs) const { return val != rhs.val; }

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
    explicit operator uint16_t() const {MME_ASSERT(0, "cannot convert to 16 bit representation"); return 0;}
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

    static float fma(const float32& a, const float32& b, const float& c, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits = gaudi3::fma_fp32(a.value(), b.value(), reinterpret_ptr<uint32_t>((float*) &c), (int) rm);
        return float32(resInBits).toFloat();
    }

    static float32 fma(const float32& a, const float32& b, const float32& c, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits = gaudi3::fma_fp32(a.value(), b.value(), reinterpret_ptr<uint32_t>((float*) &c), (int) rm);
        return float32(resInBits);
    }

    static float32 add(const float32& a, const float32& b, MmeCommon::RoundingMode rm)
    {
        uint32_t resInBits = gaudi3::add_fp32(a.value(), b.value(), rm);
        return float32(resInBits);
    }

private:
    uint32_t val = 0;
};

// Define fp32 formats
using fp32_t = float32;
static_assert(sizeof(float32) == sizeof(uint32_t), "reinterpret casting to float32 won't work");
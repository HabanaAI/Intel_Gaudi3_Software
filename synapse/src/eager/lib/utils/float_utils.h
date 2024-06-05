// IEEE 754 conforming single precision floating point utils
#if defined(__STDC_IEC_559__)
#pragma once

// eager includes (relative to src/eager/lib/)
#include "utils/memory_utils.h"   // for readAs
#include "utils/numeric_utils.h"  // for isPowerOf2

// std includes
#include <cassert>
#include <cstdint>
#include <type_traits>

#if __cplusplus >= 202002L
#warning TODO: replace readAs with bit_cast which allows changing all inline functions into constexpr
#endif

namespace eager_mode::single_precision_float
{
namespace detail
{

// Implementation Notes:
// - Constants used directly for comparison are unsigned to avoid warnings about un/signed compare
// - Shifted values are unsigned as well to avoid ambiguity

constexpr auto     MANTISSA_BITS = 23;
constexpr auto     EXP_BITS      = 8;
constexpr uint32_t EXP_BIAS      = 127;

constexpr auto EXP_SHIFT  = MANTISSA_BITS;
constexpr auto SIGN_SHIFT = EXP_BITS + EXP_SHIFT;

constexpr auto     MANTISSA_MASK = (1u << MANTISSA_BITS) - 1;
constexpr uint32_t EXP_MASK      = ((1u << EXP_BITS) - 1) << EXP_SHIFT;
constexpr auto     SIGN_MASK     = 1u << SIGN_SHIFT;

// clang-format off
constexpr uint32_t mantissa(uint32_t u) { return u & MANTISSA_MASK; }
constexpr uint32_t unshifted_exp(uint32_t u) { return u & EXP_MASK; }
constexpr uint32_t unshifted_sign(uint32_t u) { return u & SIGN_MASK; }
constexpr uint32_t exp(uint32_t u) { return u >> EXP_SHIFT; }
constexpr uint32_t sign(uint32_t u) { return u >> SIGN_SHIFT; }
// clang-format on

constexpr uint32_t INF_EXP           = EXP_MASK;
constexpr uint32_t INF_EXP_UNSHIFTED = INF_EXP >> EXP_SHIFT;
}  // namespace detail

// Implementation Notes:
// - The template + assert are used to avoid promotion of types to float (In case there's no -Wconversion)
// - The readAs is used to avoid type punning and compiles into a single asm instr in release
// - The results of intermediates are stored to avoid recalc in debug
// - "&" is used rather than "&&" to avoid branching where possible

// Check if abs(f) is an 2^n where n is an int >=0.
//
// (See "Implementation Notes" above)
template<typename T>
/*constexpr*/ inline bool isFloatUintPowerOf2(T f)
{
    static_assert(std::is_same_v<T, float>);

    const uint32_t u {readAs<uint32_t>(&f)};
    const uint32_t e {detail::unshifted_exp(u)};
    return (detail::mantissa(u) == 0) & (e >= (detail::EXP_BIAS << detail::EXP_SHIFT)) & (e < detail::INF_EXP);
}

// Check if abs(f) is a normal power of 2.
//
// (See "Implementation Notes" above)
template<typename T>
/*constexpr*/ inline bool isFloatNormalPowerOf2(T f)
{
    static_assert(std::is_same_v<T, float>);

    const uint32_t u {readAs<uint32_t>(&f)};
    const uint32_t e {detail::unshifted_exp(u)};
    return (detail::mantissa(u) == 0) & (e > 0) & (e < detail::INF_EXP);
}

// Check if abs(f) is a normal or subnormal power of 2.
//
// (See "Implementation Notes" above)
template<typename T>
/*constexpr*/ inline bool isFloatPowerOf2(T f)
{
    static_assert(std::is_same_v<T, float>);

    const uint32_t u {readAs<uint32_t>(&f)};
    const uint32_t e {detail::exp(u)};
    const uint32_t m {detail::mantissa(u)};
    return e != 0 ? (m == 0) & (e < detail::INF_EXP_UNSHIFTED) : isPowerOf2(m);
}

}  // namespace eager_mode::single_precision_float

#endif  //  defined(__STDC_IEC_559__)
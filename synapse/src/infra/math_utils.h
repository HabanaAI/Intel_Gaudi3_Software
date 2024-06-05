#pragma once

#include "defs.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

// Check whether an integral value is a power of 2
// Safe to use with signed but < 1 values are always false.
//
// Examples:
//  static_assert(isPowerOf2(0u) == false);
//  static_assert(isPowerOf2(1u) == true);
//  static_assert(isPowerOf2(2u) == true);
//  static_assert(isPowerOf2(3u) == false);
//  static_assert(isPowerOf2(4u) == true);
//  static_assert(isPowerOf2(-1) == false);
//  static_assert(isPowerOf2(0) == false);
//  static_assert(isPowerOf2(1) == true);
// TODO[c++20]: replace with `v > 0 && std::has_single_bit` from <bit> which will opt to a using asm popcnt,
//              can achieve the same with bitset<sizeof(v) * CHAR_BIT>(v).count() == 1 but it costs
//              more in debug builds than this - Though maybe still worth it, here until c++20?
template<typename T>
constexpr bool isPowerOf2(T v)
{
    static_assert(std::is_integral_v<T>);
    return v > 0 && (v & (v - 1)) == 0;
}

// Align an unsigned value to the next multiple of N.
// Assuming that the next value must fit in the given type.
//
// Examples:
//  static_assert(alignUpTo(0u, 5) == 0);
//  static_assert(alignUpTo(1u, 5) == 5);
//  static_assert(alignUpTo(5u, 5) == 5);
//  static_assert(alignUpTo(6u, 5) == 10);
//
//  static_assert(alignUpTo(0u, 8) == 0);
//  static_assert(alignUpTo(1u, 8) == 8);
//  static_assert(alignUpTo(8u, 8) == 8);
//  static_assert(alignUpTo(9u, 8) == 16);
template<typename UnsignedIntegral>
constexpr UnsignedIntegral alignUpTo(UnsignedIntegral value, std::size_t N)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    auto m = value % N;
    return m ? value + N - m : value;
}

// Align an unsigned value to the next multiple of N.
// Assuming that the next value must fit in the given type.
// Assuming N is a power of 2.
//
// Examples:
//  static_assert(alignUpToPowerOf2(0u, 4) == 0);
//  static_assert(alignUpToPowerOf2(1u, 4) == 4);
//  static_assert(alignUpToPowerOf2(5u, 4) == 8);
//  static_assert(alignUpToPowerOf2(6u, 4) == 8);
template<typename UnsignedIntegral>
constexpr UnsignedIntegral alignUpToPowerOf2(UnsignedIntegral value, std::size_t N)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    return value + (-value & (N - 1));
}

// Given non negative number, the function returns the largest divisor of the number which is power of 2.
// Examples: biggestDivisorWhichIsPowerOf2(1953) = 1, biggestDivisorWhichIsPowerOf2(9216) = 1024 (9216 = 1024 * 9)
// (From bit representation view, for number that equal to ??????1000...00 it returns 0000001000...00)
inline uint64_t biggestDivisorWhichIsPowerOf2(const uint64_t num)
{
    return num & (~(num - 1));
}

inline uint64_t div_round_up(uint64_t a, uint64_t b)
{
    HB_ASSERT(b, "Divider should be non zero");
    return (a + b - 1) / b;
}

inline uint64_t round_down(uint64_t value, uint64_t alignment)
{
    return (value / alignment) * alignment;
}

inline uint64_t round_to_multiple(uint64_t a, uint64_t mul)
{
    return mul == 0 ? 0 : mul * div_round_up(a, mul);
}

inline uint8_t* round_to_multiple(uint8_t* a, uint64_t mul)
{
    uint64_t rtn = round_to_multiple((uint64_t)a, mul);
    return (uint8_t*)rtn;
}

inline float calc_tanh(float x)
{
    return tanh(x);
}

inline float calc_sigmoid(float x)
{
    return 1.0 / (1.0 + exp(x));
}

// GELU = Gaussian Error Linear Unit activation function.
inline float calc_gelu(float x)
{
    return x * 0.5 * (1 + tanh(0.79788 * (x + 0.044715 * pow(x, 3))));
}

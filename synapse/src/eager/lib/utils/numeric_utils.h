#pragma once

// std includes
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if __cplusplus >= 202002L
#warning TODO: Should re-evaluate if some of the functions in this file can be replaced with newly added c++20 features
#endif

namespace eager_mode
{
// Check whether an unsigned integral value is a power of 2
//
// Examples:
//  static_assert(isPowerOf2(0u) == false);
//  static_assert(isPowerOf2(1u) == true);
//  static_assert(isPowerOf2(2u) == true);
//  static_assert(isPowerOf2(3u) == false);
//  static_assert(isPowerOf2(4u) == true);
// TODO[c++20]: replace with `std::has_single_bit` from <bit> which will opt to a using asm popcnt,
//              can achieve the same with bitset<sizeof(v) * CHAR_BIT>(v).count() == 1 but it costs
//              more in debug builds than this - Though maybe still worth it, here until c++20?
template<typename UnsignedIntegral>
constexpr bool isPowerOf2(UnsignedIntegral v)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    return v && (v & (v - 1)) == 0;
}

// Align an unsigned value to the next multiple of N.
// Assuming that the next value must fit in the given type.
//
// Examples:
//  static_assert(alignUpTo<5>(0u) == 0);
//  static_assert(alignUpTo<5>(1u) == 5);
//  static_assert(alignUpTo<5>(5u) == 5);
//  static_assert(alignUpTo<5>(6u) == 10);
//
//  static_assert(alignUpTo<8>(0u) == 0);
//  static_assert(alignUpTo<8>(1u) == 8);
//  static_assert(alignUpTo<8>(8u) == 8);
//  static_assert(alignUpTo<8>(9u) == 16);
template<std::size_t N, typename UnsignedIntegral>
constexpr UnsignedIntegral alignUpTo(UnsignedIntegral value)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    if constexpr (isPowerOf2(N))
    {
        return value + (-value & (N - 1));
    }
    else
    {
        auto m = value % N;
        return m ? value + N - m : value;
    }
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

// Align a ptr up to match a given types alignment requirement
template<typename T>
constexpr std::byte* alignFor(std::byte* ptr)
{
    return reinterpret_cast<std::byte*>(alignUpTo<alignof(T)>(reinterpret_cast<std::uintptr_t>(ptr)));
}

template<typename UnsignedIntegral>
constexpr auto divRoundUp(UnsignedIntegral value, std::size_t N)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);

    return (value / N) + !!(value % N);
}

template<typename UnsignedIntegral>
constexpr UnsignedIntegral calcPaddingSize(UnsignedIntegral size, UnsignedIntegral alignment)
{
    return (size % alignment) == 0 ? 0 : alignment - (size % alignment);
}

// Extract specific bits from a value.
//
// Examples:
//  static_assert(extractBits<0, 64>((uint64_t)-1) == ((uint64_t)-1));
//  static_assert(extractBits<0, 3>(0x7u) == 0b111);
//  static_assert(extractBits<1, 3>(0x7u) == 0b011);
template<std::size_t start_bit, std::size_t count, typename UnsignedIntegral>
constexpr UnsignedIntegral extractBits(UnsignedIntegral v)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    const auto digits = std::numeric_limits<UnsignedIntegral>::digits;

    static_assert(start_bit >= 0 && start_bit < digits && count > 0 && start_bit + count <= digits);
    if (count < digits)
    {
        return (v >> start_bit) & ((UnsignedIntegral {1} << count) - 1);
    }
    return v;
}

// Extract specific bits from a value.
//
// Examples:
//  assert(extractBits((uint64_t)-1, 0, 64) == ((uint64_t)-1));
//  assert(extractBits(0x7u, 0, 3) == 0b111);
//  assert(extractBits(0x7u, 1, 3) == 0b011);
template<typename UnsignedIntegral>
static inline UnsignedIntegral extractBits(UnsignedIntegral v, std::size_t start_bit, std::size_t count)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    const auto digits = std::numeric_limits<UnsignedIntegral>::digits;

    assert(start_bit >= 0 && start_bit < digits && count > 0 && start_bit + count <= digits);
    if (count < digits)
    {
        return (v >> start_bit) & ((UnsignedIntegral {1} << count) - 1);
    }
    return v;
}

template<typename UnsignedIntegral>
constexpr bool isOffsetInRange(UnsignedIntegral start, UnsignedIntegral size, UnsignedIntegral offset)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    return offset >= start && offset < start + size;
}

}  // namespace eager_mode
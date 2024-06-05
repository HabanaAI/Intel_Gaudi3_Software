#pragma once

// synapse-internal includes (relative to src/)
#include "infra/log_manager.h"

// std includes
#include <array>
#include <cstdint>

#ifndef likely
#define likely(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef NDEBUG
#include "defs.h"
#define EAGER_ASSERT(condition, message, ...) HB_ASSERT(condition, message, ##__VA_ARGS__)
#define EAGER_REPORT_ERROR(message, ...)      EAGER_ASSERT(0, message, ##__VA_ARGS__)
#else
#define EAGER_ASSERT(condition, message, ...) ((void)0)
#define EAGER_REPORT_ERROR(message, ...) LOG_ERR(EAGER, message, ##__VA_ARGS__)
#endif
#define EAGER_LOG_WARN(message, ...) LOG_WARN(EAGER, message, ##__VA_ARGS__)

#define EAGER_ASSERT_PTR(ptr_) EAGER_ASSERT((ptr_) != nullptr, "unexpected nullptr: " #ptr_)
#define EAGER_ASSERT_0 EAGER_ASSERT(0, "Unsupported case.")
#define TODO           EAGER_ASSERT_0;

namespace eager_mode
{
// List of all chips (HBU) that are supported by Eager
enum class ChipType : uint8_t
{
    GAUDI2,
    GAUDI3,
    // More can be added before this line..

    // Must be last:
    INVALID,
    CHIPS_NR = INVALID  // Specifies total number of chips in this enum
};

// List of all engines that are supported by struct recipe_t
// This enum must be identical to recipe_t::EngineType
enum class EngineType : uint8_t
{
    TPC,
    MME,
    DMA,
    ROT,
    CME,
    // More can be added before this line..

    // Must be last:
    INVALID,
    ENGINES_NR = INVALID  // Specifies total number of engines in this enum
};

// Type representing number of engines types for all chips
using EnginesNrType = uint32_t;

// Currently only relevant to MME in Gaudi3, in case it is utilized
// to perform a transpose. can be extended in the future to
// new use cases where an engine operates in a different mode.
enum class SecondaryEngineType : uint8_t
{
    TRANSPOSE,

    NONE  // Means the engine operates in it's primary mode
};

struct StatisticsOfEngineType
{
    unsigned nodeNum;
    unsigned activationNum;
};
using AllStatisticsType = std::array<StatisticsOfEngineType, static_cast<EnginesNrType>(EngineType::ENGINES_NR)>;

}  // namespace eager_mode

#pragma once

#include "log_manager.h"
#include "types_exception.h"
#include "types.h"

#include <climits>

#ifdef NDEBUG
constexpr bool releaseMode = true;
#else
constexpr bool releaseMode = false;
#endif

namespace synapse
{
// this function is only for MACROs not use it without them
template<typename... Args>
void hbAssert(bool        throwException,
              unsigned    logLevel,
              const char* conditionString,
              const char* message,
              const char* file,
              const int   line,
              const char* func,
              Args&&... args)
{
    std::string messageWithArgs = fmt::format(message, std::forward<Args>(args)...);
    std::string logMessage = fmt::format("{}:{} function: {}, failed condition: ({}), message: {}",
                                         file,
                                         line,
                                         func,
                                         conditionString,
                                         messageWithArgs);
    HLLOG_TYPED_FULL(GC, logLevel, false, file, line, "{}", logMessage);

    if (throwException)
    {
        throw SynapseStatusException(logMessage, synStatus::synFail);
    }
}

template<int N>
constexpr bool charExists(const char (&str)[N], char c, int index)
{
    return (index < N && (str[index] == c || charExists(str, c, index + 1)));
}

}  // namespace synapse

// Guideline for using MACROs:
// if it possible use only HB_ASSERT or HB_ASSERT_PTR
// if running validatation function that cost in performance and should not effect on Release use HB_DEBUG_VALIDATE
#define HB_ASSERT(condition, message, ...)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (unlikely(!(condition))) /* likely not taken, hint to optimize runtime */                                   \
        {                                                                                                              \
            synapse::hbAssert(releaseMode,                                                                             \
                              HLLOG_LEVEL_CRITICAL,                                                                    \
                              #condition,                                                                              \
                              message,                                                                                 \
                              HLLOG_FILENAME,                                                                          \
                              __LINE__,                                                                                \
                              __FUNCTION__ HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));      \
            assert(condition);                                                                                         \
            throw;                                                                                                     \
        }                                                                                                              \
    } while (false)

#define HB_ASSERT_DEBUG_ONLY(condition, message, ...)                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (unlikely(!(condition))) /* likely not taken, hint to optimize runtime */                                   \
        {                                                                                                              \
            synapse::hbAssert(false,                                                                                   \
                              HLLOG_LEVEL_ERROR,                                                                       \
                              #condition,                                                                              \
                              message,                                                                                 \
                              HLLOG_FILENAME,                                                                          \
                              __LINE__,                                                                                \
                              __FUNCTION__                                                                             \
                              HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));                   \
            assert(condition);                                                                                         \
        }                                                                                                              \
    } while (false)

#define HB_DEBUG_VALIDATE(function) assert(function)

#define HB_ASSERT_PTR(pointer) HB_ASSERT(pointer != nullptr, "{} is null pointer", #pointer)

#define CHECK_RET_FALSE(condition, message, ...)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (unlikely(!(condition)))                                                                                    \
        {                                                                                                              \
            LOG_ERR(GC, "{}: The condition [ {} ] failed. " message ".", HLLOG_FUNC, #condition, ##__VA_ARGS__);       \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_RET_NULL(condition, message, ...)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (unlikely(!(condition)))                                                                                    \
        {                                                                                                              \
            LOG_ERR(GC, "{}: The condition [ {} ] failed. " message ".", HLLOG_FUNC, #condition, ##__VA_ARGS__);       \
            return nullptr;                                                                                            \
        }                                                                                                              \
    } while (0)

#define NUM_BITS(var) (CHAR_BIT * sizeof(var))

// get variable name as a string.
// given macros and constants won't be expanded.
// use only with an explicit name.
#define STRINGIFY(X) #X

// get code as a string.
// given macros and constants *will* be expanded.
#define TOSTRING(X) STRINGIFY(X)

#define UINT64_HIGH_PART(x) (x & 0xFFFFFFFF00000000)

#pragma once

#include <climits>
#include "logger.h"

#ifdef NDEBUG
constexpr bool releaseMode = true;
#else
constexpr bool releaseMode = false;
#endif

namespace scal
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
    std::string logMessage = fmt::format(FMT_COMPILE("{}::{} function: {}, failed condition: ({}), message: {}"),
                                         file,
                                         line,
                                         func,
                                         conditionString,
                                         message);
     auto logger = hl_logger::getLogger(HLLOG_ENUM_TYPE_NAME::SCAL);
     hl_logger::log(logger,
                    HLLOG_LEVEL_ERROR,
                    true,
                    file,
                    line,
                    logMessage,
                    args...);

    if (throwException)
    {
        throw;
    }
}

template<int N>
constexpr bool charExists(const char (&str)[N], char c)
{
    bool hasChar = false;
    for (int i = 0 ; i < N; ++i)
    {
        hasChar = hasChar || str[i] == c;
    }
    return hasChar;
}

}  // namespace synapse

// Guideline for using MACROs:
// if it possible use only HB_ASSERT or HB_ASSERT_PTR
// if running validatation function that cost in performance and should not effect on Release use HB_DEBUG_VALIDATE
#define HB_ASSERT(condition, message, ...)                                                                             \
    static_assert(!scal::charExists(#condition, '{'), "\'{\' is forbidden char because of fmt::format");               \
    static_assert(!scal::charExists(#condition, '}'), "\'}\' is forbidden char because of fmt::format");               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (HLLOG_UNLIKELY(!(condition))) /* likely not taken, hint to optimize runtime */                             \
        {                                                                                                              \
            scal::hbAssert(releaseMode,                                                                                \
                           HLLOG_LEVEL_CRITICAL,                                                                       \
                           #condition,                                                                                 \
                           message,                                                                                    \
                           HLLOG_FILENAME,                                                                             \
                           __LINE__,                                                                                   \
                           __FUNCTION__                                                                                \
                           HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));                      \
            assert(condition);                                                                                         \
        }                                                                                                              \
    } while (false)

#define HB_ASSERT_PTR(pointer) HB_ASSERT(pointer != nullptr, "{} is null pointer", #pointer)

// get variable name as a string.
// given macros and constants won't be expanded.
// use only with an explicit name.
#define STRINGIFY(X) #X

#pragma once
#include <hl_logger/hllog.hpp>
namespace scal
{
enum class LoggerTypes
{
    SCAL,
    LOG_MAX
};
// define HLLOG_ENUM_TYPE_NAME that provides full name of your enum with logger
#define HLLOG_ENUM_TYPE_NAME scal::LoggerTypes

#define TO64(x)  ((uint64_t)x)

#define LOG_TRACE    HLLOG_TRACE
#define LOG_DEBUG    HLLOG_DEBUG
#define LOG_INFO     HLLOG_INFO
#define LOG_WARN     HLLOG_WARN
#define LOG_ERR      HLLOG_ERR
#define LOG_CRITICAL HLLOG_CRITICAL

#define LOG_TRACE_F    HLLOG_TRACE_F
#define LOG_DEBUG_F    HLLOG_DEBUG_F
#define LOG_INFO_F     HLLOG_INFO_F
#define LOG_WARN_F     HLLOG_WARN_F
#define LOG_ERR_F      HLLOG_ERR_F
#define LOG_CRITICAL_F HLLOG_CRITICAL_F
} // namespace scal

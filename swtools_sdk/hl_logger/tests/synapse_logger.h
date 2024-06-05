#pragma once
//#define HLLOG_DISABLE_FMT_COMPILE
//#define HLLOG_USE_STD_TIMESTAMP
#define HLLOG_ENABLE_LAZY_LOGGING
#include "hl_logger/hllog.hpp"
namespace synapse{
enum class LoggerTypes
{
    SCAL_1,
    SCAL_2,
    SCAL_3,
    SCAL_API,
    SCAL_API_3,
    SYN_API,
    SYN_API2,
    SYN_API_CRASH,
    SYN_STREAM,
    DFA,
    SYN_TEST,
    LOG_MAX // must be the last value
};
}

#define HLLOG_ENUM_TYPE_NAME synapse::LoggerTypes
HLLOG_DECLARE_MODULE_LOGGER()
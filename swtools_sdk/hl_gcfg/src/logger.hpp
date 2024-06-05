#pragma once
#include "hl_logger/hllog.hpp"
namespace hl_gcfg{
    enum class LoggerTypes
    {
        HL_GCFG,
        LOG_MAX // must be the last value
    };
}

#define HLLOG_ENUM_TYPE_NAME hl_gcfg::LoggerTypes
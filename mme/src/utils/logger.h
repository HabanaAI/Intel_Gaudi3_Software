#pragma once
#ifdef SWTOOLS_DEP
#define HLLOG_ENABLE_LAZY_LOGGING
#include <hl_logger/hllog.hpp>

namespace mme_stack
{
namespace log
{
enum class Type : uint32_t
{
    MME_BRAIN,
    MME_CONFIG_PARSER,
    MME_RECIPE,
    MME_DESC_DUMP,
    LOG_MAX
};

#define LOG_TRACE(log_type, msg, ...)    HLLOG_TRACE(log_type, msg, ##__VA_ARGS__)
#define LOG_DEBUG(log_type, msg, ...)    HLLOG_DEBUG(log_type, msg, ##__VA_ARGS__)
#define LOG_INFO(log_type, msg, ...)     HLLOG_INFO(log_type, msg, ##__VA_ARGS__)
#define LOG_WARN(log_type, msg, ...)     HLLOG_WARN(log_type, msg, ##__VA_ARGS__)
#define LOG_ERR(log_type, msg, ...)      HLLOG_ERR(log_type, msg, ##__VA_ARGS__)
#define LOG_CRITICAL(log_type, msg, ...) HLLOG_CRITICAL(log_type, msg, ##__VA_ARGS__)

}  // namespace log
}  // namespace mme_stack

#define HLLOG_ENUM_TYPE_NAME mme_stack::log::Type
HLLOG_DECLARE_MODULE_LOGGER()
#else
#include "defs.h"
#include "spdlog/common.h"
#endif

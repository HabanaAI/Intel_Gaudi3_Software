#pragma once

#include "log_manager.h"
#include "timer.h"

// Should be used only for initialization step
#define LOG_INFO_F(loggerName, msg, ...)                                                                               \
    {                                                                                                                  \
        unsigned log_level = synapse::LogManager::instance().get_log_level(loggerName);                                \
        synapse::LogManager::instance().set_log_level(loggerName, HLLOG_LEVEL_INFO);                                   \
        SYN_LOG(loggerName, HLLOG_LEVEL_INFO, msg, ##__VA_ARGS__);                                                     \
        synapse::LogManager::instance().set_log_level(loggerName, log_level);                                          \
    }

#define LOG_RECIPE_STATS(msg, ...)  LOG_INFO(RECIPE_STATS, msg, ##__VA_ARGS__);

#define RECIPE_STATS_START(x) StatTimeStart x(HLLOG_LEVEL_AT_LEAST_INFO(RECIPE_STATS))
inline uint64_t RECIPE_STATS_END(StatTimeStart x) { return TimeTools::timeFromNs(x.startTime); }

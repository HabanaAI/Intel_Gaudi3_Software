#pragma once
#include <vector>
#define SPDLOG_NO_THREAD_ID
#include <spdlog/spdlog.h>
#include <hl_logger/impl/hllog_internal_api.hpp>

namespace hl_logger{

class Logger : public spdlog::logger
{
public:
    using spdlog::logger::logger;
    bool sepLogPerThread = false;
    enum class PrintOptions
    {
        none                   = 0,
        tid                    = 1,
        pid                    = 2,
        pid_tid                = 3,
        specialContext         = 4,
        tid_specialContext     = 5,
        pid_specialContext     = 6,
        pid_tid_specialContext = 7,
        file                   = 8,
        tid_file               = 9,
        pid_file               = 10,
        pid_tid_file           = 11,
        specialContext_file    = 12,
        tid_specialContext_file= 13,
        pid_specialContext_file= 14,
        pid_tid_specialContext_file = 15,
    };
    PrintOptions printOptions = PrintOptions::tid;
    std::string  spdlogPattern;
    internal::AddToRecentLogsQueueFunc * addToRecentLogsQueueFunc = nullptr;
    void *       recentLogsQueueVoidPtr = nullptr;
    uint8_t      lazy_level             = HLLOG_LEVEL_OFF;
};

class Sinks
{
public:
    std::vector<spdlog::sink_ptr> sinks;
    ~Sinks(){
        if (!sinks.empty())
        {
            sinks.clear();
            hl_logger::refreshInternalSinkCache();
        }
    }
};
}  // namespace hl_logger
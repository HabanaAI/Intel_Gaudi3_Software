/* SPDX-License-Identifier: MIT
 *
 * Copyright 2016-2019 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef LOGGING_H_
#define LOGGING_H_

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <vector>
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <pwd.h>

#ifdef _WIN32
#define COLOR_MACRO spdlog::sinks::wincolor_stdout_sink_mt
#define ERR_COLOR_MACRO spdlog::sinks::wincolor_stderr_sink_mt
#else
#define COLOR_MACRO spdlog::sinks::ansicolor_stdout_sink_mt
#define ERR_COLOR_MACRO spdlog::sinks::ansicolor_stderr_sink_mt
#endif

#define SINK_STDOUT    0
#define SINK_STDERR    1

#ifdef ENFORCE_TRACE_MODE_LOGGING
extern bool g_traceModeLogging;
#define TURN_ON_TRACE_MODE_LOGGING() g_traceModeLogging = true
#define TURN_OFF_TRACE_MODE_LOGGING() g_traceModeLogging = false
#else
const bool g_traceModeLogging = false;
#endif

inline std::shared_ptr<spdlog::logger> validate(const std::string& logname, const std::string& msg)
{
     auto log = spdlog::get(logname);
     if (log == nullptr)
     {
         log = spdlog::get("default");
         if (log == nullptr)
         {
             log = spdlog::stdout_color_mt("default");
         }
         log->critical("logger with name - {}, does not exist [on msg \"{}\"]", logname, msg);
         return log;
     }
     else
     {
         return log;
     }
}

#define GET_LOGGER(logname, msg) ::validate(logname, msg)

#define LOG_TRACE(logname,msg,...) do {     \
        GET_LOGGER(logname, msg)->trace(msg, ## __VA_ARGS__);      \
    } while (0)

#define LOG_DEBUG(logname,msg,...) do {     \
        g_traceModeLogging ? \
            GET_LOGGER(logname, msg)->trace(msg, ## __VA_ARGS__) : \
            GET_LOGGER(logname, msg)->debug(msg, ## __VA_ARGS__);  \
    } while (0)

#define LOG_INFO(logname,msg,...) do {      \
        g_traceModeLogging ? \
            GET_LOGGER(logname, msg)->trace(msg, ## __VA_ARGS__) : \
            GET_LOGGER(logname, msg)->info(msg, ## __VA_ARGS__);   \
    } while (0)

#define LOG_WARN(logname,msg,...) do {      \
        g_traceModeLogging ? \
            GET_LOGGER(logname, msg)->trace(msg, ## __VA_ARGS__) : \
            GET_LOGGER(logname, msg)->warn(msg, ## __VA_ARGS__);   \
    } while (0)

#define LOG_ERR(logname,msg,...) do {       \
        g_traceModeLogging ? \
            GET_LOGGER(logname, msg)->trace(msg, ## __VA_ARGS__) : \
            GET_LOGGER(logname, msg)->error(msg, ## __VA_ARGS__);  \
    } while (0)

#define LOG_CRITICAL(logname,msg,...) do {  \
        g_traceModeLogging ? \
            GET_LOGGER(logname, msg)->trace(msg, ## __VA_ARGS__) : \
            GET_LOGGER(logname, msg)->critical(msg, ## __VA_ARGS__);\
    } while (0)

inline spdlog::level::level_enum getLogLevel(const std::string& logname)
{
    auto log = spdlog::get(logname);
    if (log == nullptr)
    {
         log = spdlog::get("default");
         if (log == nullptr)
         {
             log = spdlog::stdout_color_mt("default");
         }
         log->critical("logger with name - {}, does not exist [on msg \"{}\"]", logname, "getLogLevel");
    }
    return log->level();
}
inline void SET_LOGGER_SINK(const std::string& logname, const std::string& pathname, spdlog::level::level_enum lvl, size_t size, size_t amount)
{
    const char* enable_file_colors = getenv("ENABLE_LOG_FILE_COLORS");
    bool should_do_colors = enable_file_colors && !strcmp(enable_file_colors, "true");
    spdlog::sink_ptr sink = std::make_shared<spdlog::sinks::ansicolor_rotating_file_sink_mt>( pathname, size, amount, should_do_colors);
    std::vector<spdlog::sink_ptr>& sinks = const_cast<std::vector<spdlog::sink_ptr>&>((GET_LOGGER(logname, "set sink level")->sinks()));
    switch (lvl)
    {
    case spdlog::level::trace :
        sink->set_level(spdlog::level::trace);
        break;
    case spdlog::level::debug:
        sink->set_level(spdlog::level::debug);
        break;
    case spdlog::level::info:
        sink->set_level(spdlog::level::info);
        break;
    case spdlog::level::warn:
        sink->set_level(spdlog::level::warn);
        break;
    case spdlog::level::err:
        sink->set_level(spdlog::level::err);
        break;
    case spdlog::level::critical:
        sink->set_level(spdlog::level::critical);
        break;
    case spdlog::level::off:
        return;
    default:
        assert(0 && "No such log level");
    };
    sinks.push_back(sink);
}

#define SET_LOG_LEVEL(logname, loglevel) \
    do {  \
        switch (loglevel) { \
        case 0: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::trace);   \
            break;  \
        case 1: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::debug);   \
            break;  \
        case 2: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::info);    \
            break;  \
        case 3: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::warn);    \
            break;  \
        case 4: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::err); \
            break;  \
        case 5: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::critical);    \
            break;  \
        case 6: \
            GET_LOGGER(logname, "set log level")->set_level(spdlog::level::off); \
            break;  \
        default:    \
            assert(0 && "No such log level");   \
        }; \
    } while (0)

inline void DEFINE_LOGGER(const std::string& logname, const std::string& filename, unsigned size, unsigned amount, unsigned int sink)
{
    char *user_gid_str = getenv("SUDO_GID");
    char *user_uid_str = getenv("SUDO_UID");
    std::string updated_filename = filename;
    uid_t uid = 0;
    gid_t gid = 0;

    if (user_gid_str)
    {
        gid = std::stoul(user_gid_str, NULL, 10);
    }
    if (user_uid_str)
    {
        uid = std::stoul(user_uid_str, NULL, 10);
    }

    if (strncmp((updated_filename).c_str(),"/",1) && getenv ("HOME") != nullptr)
    {
        std::string directory = std::string(getenv ("HOME"));
        if (getenv("HABANA_LOGS") != nullptr)
        {
            directory = getenv("HABANA_LOGS");
        }
        else
        {
            directory += "/.habana_logs";
        }
        if (mkdir(directory.c_str(), 0777) != 0 && errno != EEXIST) assert(0);
        /* if the app run with sudo, we still want the original user to own the directory */
        if (uid && gid)
        {
            if (chown(directory.c_str(), uid, gid) != 0) assert(0);
        }
        updated_filename = directory + "/" + filename;
    }

    std::vector<spdlog::sink_ptr> sinks;
    const char* enable_console = getenv("ENABLE_CONSOLE");
    bool should_enable_console = enable_console && !strcmp(enable_console, "true");
    if (should_enable_console)
    {
        if (sink == SINK_STDOUT)
        {
            sinks.push_back(std::make_shared<COLOR_MACRO>());
        }
    else if (sink == SINK_STDERR)
        {
            sinks.push_back(std::make_shared<ERR_COLOR_MACRO>());
        }
    }
    const char* enable_file_colors = getenv("ENABLE_LOG_FILE_COLORS");
    bool should_do_colors = enable_file_colors && !strcmp(enable_file_colors, "true");
    sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_rotating_file_sink_mt>(updated_filename, size, amount, should_do_colors));
    auto combined_logger = std::make_shared<spdlog::logger>(logname, begin(sinks), end(sinks));
    spdlog::register_logger(combined_logger);
    combined_logger->flush_on(spdlog::level::trace);
    if (uid && gid)
    {
        if (chown(updated_filename.c_str(), uid, gid) != 0) assert(0);
    }
}

inline void CREATE_SINK_LOGGER(const std::string& logName,
                               const std::string& fileName,
                               unsigned defaultLogFileSize,
                               unsigned defaultLogFileAmount,
                               unsigned defaultLogLevel = 5,
                               unsigned int sink = SINK_STDOUT)
{
    auto log = spdlog::get(logName);
    if (log != nullptr)
    {
        log->critical("Logger was redefined {}", logName);
    }
    else
    {
        int logLevel = defaultLogLevel;
        unsigned logFileSize = defaultLogFileSize;
        if (getenv ("LOG_FILE_SIZE") != nullptr)
        {
            logFileSize = std::stol(getenv ("LOG_FILE_SIZE"));
        }

        unsigned logFileAmount = defaultLogFileAmount;
        if (getenv ("LOG_FILE_AMOUNT") != nullptr)
        {
            logFileAmount = std::stol(getenv ("LOG_FILE_AMOUNT"));
        }

        DEFINE_LOGGER(logName, fileName, logFileSize, logFileAmount, sink);
        if (getenv ("LOG_LEVEL_ALL") != nullptr)
        {
            logLevel = std::stoi(getenv ("LOG_LEVEL_ALL"));
        }
        if (getenv ((std::string("LOG_LEVEL_") + std::string(logName)).c_str()) != nullptr)
        {
            std::string logLevelStr = getenv ((std::string("LOG_LEVEL_") + std::string(logName)).c_str());
            logLevel = std::stoi(logLevelStr);
        }
        SET_LOG_LEVEL(logName, logLevel);
    }
}

inline void CREATE_LOGGER(const std::string& logName,
                          const std::string& fileName,
                          unsigned defaultLogFileSize,
                          unsigned defaultLogFileAmount,
                          unsigned defaultLogLevel = 5)
{
    CREATE_SINK_LOGGER(logName, fileName, defaultLogFileSize, defaultLogFileAmount, defaultLogLevel, SINK_STDOUT);
}

inline void CREATE_ERR_LOGGER(const std::string& logName,
                              const std::string& fileName,
                              unsigned defaultLogFileSize,
                              unsigned logFileAmount,
                              unsigned defaultLogLevel = 5)
{
    CREATE_SINK_LOGGER(logName, fileName, defaultLogFileSize, logFileAmount, defaultLogLevel, SINK_STDERR);
}

#define FLUSH_LOGGER(logname,msg) GET_LOGGER(logname, msg)->flush()
#define DROP_ALL_LOGGERS     spdlog::drop_all()
#define DROP_LOGGER(logname) do {                        \
            FLUSH_LOGGER(logname, "drop logger");        \
            spdlog::drop(logname);                       \
    } while (0)

#endif /* LOGGING_H_ */

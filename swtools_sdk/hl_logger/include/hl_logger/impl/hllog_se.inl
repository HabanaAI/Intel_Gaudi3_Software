#pragma once

namespace hl_logger {
    inline namespace HLLOG_INLINE_API_NAMESPACE{
        template<typename... Args>
        inline void log(LoggerSPtr const & logger, int logLevel, bool printFileLine, std::string_view file, int line, const char * fmtMsg, Args const & ... args)
        try
        {
            fmt::memory_buffer buf;
            fmt::format_to(buf, fmtMsg, args...);
            if (printFileLine)
            {
                fmt::memory_buffer buf2;
                fmt::format_to(buf2, "{}::{} {}", file, line, std::string_view(buf.data(), buf.size()));
                hl_logger::log(logger, logLevel, std::string_view(buf2.data(), buf2.size()));
            }
            else
            {
                hl_logger::log(logger, logLevel, std::string_view(buf.data(), buf.size()));
            }
        }
        catch(std::exception const & e)
        {
            hl_logger::log(logger, HLLOG_LEVEL_ERROR, "failed to format message");
            hl_logger::log(logger, HLLOG_LEVEL_ERROR, e.what());
            hl_logger::log(logger, HLLOG_LEVEL_ERROR, fmtMsg);
            hl_logger::logStackTrace(logger, HLLOG_LEVEL_ERROR);
        }
        catch(...)
        {
            hl_logger::log(logger, HLLOG_LEVEL_ERROR, "failed to format message");
            hl_logger::log(logger, HLLOG_LEVEL_ERROR, fmtMsg);
            hl_logger::logStackTrace(logger, HLLOG_LEVEL_ERROR);
        }
    };
}

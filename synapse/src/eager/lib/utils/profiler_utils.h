#undef BEGING_PROFILING
#undef END_PROFILING
#if (PROFILING_NUM_ITERATIONS == 0)
#define BEGING_PROFILING(TAG)
#define END_PROFILING(TAG)
#else
#include <cstdio>
#define BEGING_PROFILING(TAG)                                                                                          \
    eager_mode::MicProfiler::pin(TAG);                                                                                 \
    for (auto i = 0; i < PROFILING_NUM_ITERATIONS; ++i)                                                                \
    {
#define END_PROFILING(TAG)                                                                                             \
    }                                                                                                                  \
    {                                                                                                                  \
        const auto        eopStr = std::to_string(eager_mode::MicProfiler::sample(TAG) / PROFILING_NUM_ITERATIONS);    \
        const std::string msg    = "\n[PROFILING] Tag=" + std::to_string(TAG) + ". Repeating " +                       \
                                std::to_string(PROFILING_NUM_ITERATIONS) + " Iterations. Average time: " + eopStr +    \
                                " microseconds.\n";                                                                    \
        puts(msg.c_str());                                                                                             \
    }
#endif

///////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#define ENABLE_PRINTING 0

#include <chrono>
#include <map>
#include <string>

#if ENABLE_PRINTING
#include <iostream>
#endif

namespace eager_mode
{
enum ProfilerResolution
{
    NANOSECONDS,
    MICROSECONDS,
    MILLISECONDS,
    SECONDS,
    MINUTES,
    HOURS
};

template<class Tag, ProfilerResolution Resolution>
class Profiler
{
public:
    static inline void     pin(Tag tag);
    static inline uint64_t sample(Tag tag);

    static inline void pause(Tag tag);
    static inline void resume(Tag tag);

private:
    static inline Profiler& getInstance();
    inline uint64_t         getDuration(Tag tag);
    static inline uint64_t  getTimestamp();

    std::map<Tag, uint64_t> m_pinPoints;
    std::map<Tag, uint64_t> m_offsets;
};

using NanoProfiler = Profiler<std::string, NANOSECONDS>;
using MicProfiler  = Profiler<std::string, MICROSECONDS>;
using MilProfiler  = Profiler<std::string, MILLISECONDS>;
using SecProfiler  = Profiler<std::string, SECONDS>;
using MinProfiler  = Profiler<std::string, MINUTES>;
using HourProfiler = Profiler<std::string, HOURS>;

template<class Tag, ProfilerResolution Resolution>
Profiler<Tag, Resolution>& Profiler<Tag, Resolution>::getInstance()
{
    static Profiler<Tag, Resolution> instance;
    return instance;
}

template<class Tag, ProfilerResolution Resolution>
void Profiler<Tag, Resolution>::pin(Tag tag)
{
    const uint64_t now             = getTimestamp();
    getInstance().m_pinPoints[tag] = now;
    getInstance().m_offsets[tag]   = 0;
}

template<class Tag, ProfilerResolution Resolution>
uint64_t Profiler<Tag, Resolution>::getDuration(Tag tag)
{
    const uint64_t now = getTimestamp();
    if (m_pinPoints[tag] == 0)
    {
        return m_offsets[tag];
    }
    return m_offsets[tag] + now - m_pinPoints[tag];
}

template<class Tag, ProfilerResolution Resolution>
uint64_t Profiler<Tag, Resolution>::sample(Tag tag)
{
    const uint64_t duration = getInstance().getDuration(tag);
#if ENABLE_PRINTING
    static std::string resolutionStr;
    if (resolutionStr.empty())
    {
        switch (Resolution)
        {
            case NANOSECONDS:
                resolutionStr = "nanoseconds";
                break;
            case MICROSECONDS:
                resolutionStr = "microseconds";
                break;
            case MILLISECONDS:
                resolutionStr = "milliseconds";
                break;
            case SECONDS:
                resolutionStr = "seconds";
                break;
            case MINUTES:
                resolutionStr = "minutes";
                break;
            case HOURS:
                resolutionStr = "hours";
                break;
            default:
                throw "Unknown profiler timestamp resolution";
        };
    }
    std::cout << "[PROFILER] tag=" << tag << " duration=" << duration << " " << resolutionStr << std::endl;
#endif
    return duration;
}

template<class Tag, ProfilerResolution Resolution>
void Profiler<Tag, Resolution>::pause(Tag tag)
{
    auto&          instance = getInstance();
    const uint64_t duration = instance.getDuration(tag);
    instance.m_offsets[tag] += duration;
    instance.m_pinPoints[tag] = 0;
}

template<class Tag, ProfilerResolution Resolution>
void Profiler<Tag, Resolution>::resume(Tag tag)
{
    const uint64_t now             = getTimestamp();
    getInstance().m_pinPoints[tag] = now;
}

template<class Tag, ProfilerResolution Resolution>
uint64_t Profiler<Tag, Resolution>::getTimestamp()
{
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    switch (Resolution)
    {
        case NANOSECONDS:
            return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        case MICROSECONDS:
            return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
        case MILLISECONDS:
            return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        case SECONDS:
            return std::chrono::duration_cast<std::chrono::seconds>(now).count();
        case MINUTES:
            return std::chrono::duration_cast<std::chrono::minutes>(now).count();
        case HOURS:
            return std::chrono::duration_cast<std::chrono::hours>(now).count();
        default:
            throw "Unknown profiler timestamp resolution";
    };

    return -1;
}

}  // namespace eager_mode

#pragma once

#include <string>
#include <map>
#include <memory>
#include "time.h"
#include <chrono>

namespace gc
{

typedef std::pair<timespec, timespec> TimerBeginEnd;

class Timer
{
public:
    Timer();
    ~Timer();

    void        start(std::string_view timerName);
    void        stop(std::string_view timerName);
    double      getTotalSeconds(std::string_view timerName);
    std::string getTotalTimeStr(std::string_view timerName);

private:
    std::map<std::string, TimerBeginEnd, std::less<>> m_timers;
};

};

namespace TimeTools
{
    using StdTime = std::chrono::time_point<std::chrono::steady_clock>;

    inline StdTime timeNow() {return std::chrono::steady_clock::now();}
    inline uint64_t timeFromUs(StdTime t) {return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();}
    inline uint64_t timeFromNs(StdTime t) {return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();}

    std::string timePoint2string(std::chrono::time_point<std::chrono::system_clock> time);
}

struct StatTimeStart
{
    StatTimeStart(bool enable) { if (enable) startTime = TimeTools::timeNow(); } // collect time only if enabled

    TimeTools::StdTime startTime;
};

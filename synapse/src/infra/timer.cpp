#include "timer.h"
#include "math.h"
#include "syn_logging.h"

using namespace gc;


Timer::Timer() {}

Timer::~Timer() {}

void Timer::start(std::string_view timerName)
{
    // prepare the data structure before starting the clock
    struct timespec begin = {};
    auto            insertRes = m_timers.emplace(timerName, TimerBeginEnd(begin, begin));
    TimerBeginEnd&  timer     = insertRes.first->second;
    clock_gettime(CLOCK_MONOTONIC_RAW, &timer.first);
}

void Timer::stop(std::string_view timerName)
{
    // first stop the clock, update the data structure only after
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    auto timerIter = m_timers.find(timerName);
    assert(timerIter != m_timers.end() && "No such timer");
    if (timerIter == m_timers.end()) return;
    TimerBeginEnd& timer = timerIter->second;
    timer.second         = end;
}

double Timer::getTotalSeconds(std::string_view timerName)
{
    auto timerIter = m_timers.find(timerName);
    assert(timerIter != m_timers.end() && "No such timer");
    if (timerIter == m_timers.end()) return 0;
    TimerBeginEnd& timer = timerIter->second;

    timespec& begin = timer.first;
    timespec& end   = timer.second;
    double    total = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + (end.tv_sec - begin.tv_sec);

    return ceilf(total * 100) / 100;
}

std::string Timer::getTotalTimeStr(std::string_view timerName)
{
    double   seconds = getTotalSeconds(timerName);
    unsigned hours   = (unsigned)seconds / 3600;
    seconds -= hours * 3600;
    unsigned minutes = (unsigned)seconds / 60;
    seconds -= minutes * 60;
    return fmt::format("{}{}{:.2f} seconds",
                       (hours != 0) ? fmt::format("{} hours, ", hours) : "",
                       (minutes != 0) ? fmt::format("{} minutes, ", minutes) : "",
                       seconds);
}

namespace TimeTools
{

std::string timePoint2string(std::chrono::time_point<std::chrono::system_clock> time)
{
    const std::time_t               t_c = std::chrono::system_clock::to_time_t(time);
    const std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch());

    char buffer [80];

    struct tm* timeinfo;
    timeinfo = localtime (&t_c);
    strftime (buffer,80,"%F %X.",timeinfo);

    std::string out = std::string(buffer) + fmt::format("{:06}", us.count() % 1000000);

    return out;
}

} // namespace TimeTools

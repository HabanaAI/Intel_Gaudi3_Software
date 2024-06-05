#pragma once

#include "statistics.hpp"
#include "timer.h"
#include "vtune_stat.h"

// this creates the enum
enum class globalStatPointsEnum
{
    #define ENUM_TXT_COL(enum, txt) enum,
    #include "stat_points.hpp"
    #undef ENUM_TXT_COL
    LAST
};

// this creates the array
inline constexpr auto enumNameGlobal = toStatArray<globalStatPointsEnum>(
{
    #define ENUM_TXT_COL(enum, txt) {globalStatPointsEnum::enum, txt},
    #include "stat_points.hpp"
    #undef ENUM_TXT_COL
});

class GlobalStatistics : public Statistics<enumNameGlobal>
{
public:
    GlobalStatistics(std::string_view statName, uint32_t dumpFreq, bool enable) : Statistics(statName, dumpFreq, enable)
    {
    }

    virtual ~GlobalStatistics() override = default;

    void configurePostGcfgAndLoggerInit();
};

extern GlobalStatistics g_globalStat;

#ifndef VTUNE_ENABLED
#define STAT_GLBL_START(x)               StatTimeStart x(g_globalStat.isEnabled());
inline void STAT_GLBL_COLLECT_TIME(StatTimeStart x, globalStatPointsEnum point) { g_globalStat.collectTime(point, x); }
#endif

#define STAT_GLBL_COLLECT(n, point)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        g_globalStat.collect(globalStatPointsEnum::point, n);                                                          \
    } while (0)

#define STAT_COLLECT_COND(n, usePoint1, point1, point2)                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (usePoint1) STAT_GLBL_COLLECT(n, point1);                                                                   \
        else                                                                                                           \
            STAT_GLBL_COLLECT(n, point2);                                                                              \
    } while (0)

#include "global_statistics.hpp"
#include "habana_global_conf.h"
#include "runtime/common/syn_logging.h"
void GlobalStatistics::configurePostGcfgAndLoggerInit()
{
    g_globalStat.setEnableState(true);

    if (GCFG_STATS_PER_SYNLAUNCH.value())
    {
        m_dumpFreq = 1;
    }

    if (GCFG_STATS_FREQ.value() > 0)
    {
        m_dumpFreq = GCFG_STATS_FREQ.value();
    }

    if (m_enabled)
    {
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);
    }
}

GlobalStatistics g_globalStat {"Global", 0, false}; // set to disable until synInit
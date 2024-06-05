#include "statistics.hpp"
#include "synapse_runtime_logging.h"
#include "habana_global_conf.h"
#include "log_manager.h"
#include <sstream>

//NOTE: enable below line to collect globally.
//diffMeasureN<NUM_GLOBAL_COLLECT_POINTS> perfCollect("Global");



/**************************************************************************************************************************/
/******                                     Statistics collection                                                 *********/
/**************************************************************************************************************************/

void StatisticsBase::flush()
{
    if (!m_enabled || m_isFlushed) return;
    // For table format, print only if at least one member is not 0
    if (m_isTbl)
    {
        bool allZero = true;
        for (int i = 0; i < m_maxEnum; i++)
        {
            uint64_t count = m_collectedData[i].count.load();
            if (count != 0)
            {
                allZero = false;
                break;
            }
        }
        if (allZero) return;
    }

    printToLog("Flush during exit", true);

    m_isFlushed = true;
}

StatisticsBase::~StatisticsBase()
{
    flush();
}

void StatisticsBase::printToLog(const std::string& rMsg, bool dumpAll, bool clear)
{
    if (!LOG_LEVEL_AT_LEAST_INFO(PERF)) return;

    if (m_isTbl) // output in table format
    {
        dumpAll = true;
        if (!m_headerPrinted)
        {
            m_headerPrinted = true;
            outputHeader();
        }
    }

    bool first = true;
    std::stringstream out;
    out << m_grep;
    std::string msgOut = "   -----  " + m_statName + " " + rMsg;
    for (int i = 0; i < m_maxEnum; i++)
    {
        uint64_t count = m_collectedData[i].count.load();
        uint64_t sum   = m_collectedData[i].sum.load();
        if((count == 0) && ! dumpAll) continue;

        uint64_t averg = 0;
        if(count != 0)
        {
            averg = sum / count;
        }
        if (m_isTbl)
        {
            out << sum << "," <<  count << "," <<  averg << ",";
        }
        else
        {
            LOG_INFO(PERF, "{:<{}}|sum {:15}|count {:8}|average {:15}{}",
                     m_statEnumMsg[i].pointName, m_maxMsgLength, sum, count, averg, msgOut);
        }
        if(first && !m_isTbl)
        {
            msgOut = "";
            first = false;
        }
    }
    if (m_isTbl)
    {
        LOG_INFO(PERF, "{} {}", out.str(), msgOut);
    }
    if(clear)
    {
        clearAll();
    }
}

void StatisticsBase::outputHeader()
{
    std::stringstream out;
    out << m_grep;
    for (int i = 0; i < m_maxEnum; i++)
    {
        out << m_statEnumMsg[i].pointName << "-sum,";
        out << m_statEnumMsg[i].pointName << "-count,";
        out << m_statEnumMsg[i].pointName << "-average,";
    }
    out << m_statName;
    LOG_INFO(PERF, "{}", out.str());
}


// This is needed for syn_singleton class. It constructs the stats before GCFG are set, so we call this
// function to update the enable state after GCFG is set.
void StatisticsBase::updateEnable()
{
    m_enabled = GCFG_ENABLE_STATS.value() || GCFG_ENABLE_STATS_TBL.value() || GCFG_STATS_PER_SYNLAUNCH.value() ||
        (GCFG_STATS_FREQ.value() > 0);
    m_isTbl   = GCFG_ENABLE_STATS_TBL.value();
}

void StatisticsBase::clearAll()
{
    for (int i = 0; i < m_maxEnum; i++)
    {
        m_collectedData[i].count = 0;
        m_collectedData[i].sum   = 0;
    }
}

void StatisticsBase::setEnableState(bool enable)
{
    if (enable)
    {
        updateEnable();
    }
    else
    {
        printToLog("Disabled", true, true);
        m_isFlushed = true;
        m_enabled   = false;
    }
}

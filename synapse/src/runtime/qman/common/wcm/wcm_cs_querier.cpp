#include "wcm_cs_querier.hpp"
#include "habana_global_conf_runtime.h"
#include "hlthunk.h"
#include "synapse_common_types.h"
#include <asm-generic/errno.h>
#include "synapse_runtime_logging.h"
#include "defs.h"
#include "syn_singleton.hpp"

WcmCsQuerier::WcmCsQuerier(int fd) : m_fd(fd), m_reportAmount(GCFG_WCM_QUERIER_REPORT_AMOUNT.value()), m_stats {}
{
    LOG_DEBUG(SYN_WORK_COMPL, "{} m_fd {}", HLLOG_FUNC, m_fd);
}

synStatus WcmCsQuerier::query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams)
{
    HB_ASSERT_PTR(inParams);
    HB_ASSERT_PTR(outParams);

    m_stats.m_cntTotalQueries++;
    m_stats.m_cntTotalCs += inParams->seq_len;

    for (unsigned csIter = 0; csIter < inParams->seq_len; csIter++)
    {
        LOG_TRACE(SYN_WORK_COMPL, "{} csHandle {}", HLLOG_FUNC, inParams->seq[csIter]);
    }

    synStatus status      = synSuccess;
    int       queryStatus = hlthunk_wait_for_multi_cs(m_fd, inParams, outParams);
    if (queryStatus != 0)
    {
        m_stats.m_cntTotalFailures++;
        const int errNum = errno;
        LOG_ERR(SYN_WORK_COMPL, "{} failed queryStatus {} errNum {}", HLLOG_FUNC, queryStatus, errNum);
        queryStatus = hlthunk_get_device_status_info(m_fd);
        if (errNum == ENODEV || errNum == EIO || queryStatus == HL_DEVICE_STATUS_IN_RESET)
        {
            LOG_CRITICAL(SYN_WORK_COMPL, "{} device reset detected", HLLOG_FUNC);
            status = synDeviceReset;
            _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::waitForMultiCsFailed);
        }
        else
        {
            if (errNum == ETIMEDOUT)
            {
                LOG_ERR(SYN_WORK_COMPL,
                        "{} command submission timeout, the LKD is expected to reset the device",
                        HLLOG_FUNC);
                _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::waitForMultiCsTimedOut);
            }
            status = synFail;
        }
    }
    else
    {
        m_stats.m_cntTotalSuccesses++;
        m_stats.m_cntTotalCompleted += outParams->completed;

        LOG_TRACE(SYN_WORK_COMPL,
                  "{} success seq_set 0b{:b} completed {}",
                  HLLOG_FUNC,
                  outParams->seq_set,
                  outParams->completed);
    }
    return status;
}

void WcmCsQuerier::dump() const
{
    dumpQueryStatistics(m_stats);
}

void WcmCsQuerier::report()
{
    if ((m_reportAmount > 0) && (m_stats.m_cntTotalQueries >= (m_stats.m_cntTotalQueriesReported + m_reportAmount)))
    {
        dumpQueryStatistics(m_stats);
        m_stats.m_cntTotalQueriesReported = m_stats.m_cntTotalQueries;
    }
}

void WcmCsQuerier::dumpQueryStatistics(const WcmCsQuerierStats& rStats)
{
    LOG_DEBUG(SYN_WORK_COMPL, "m_cntTotalQueries   {}", rStats.m_cntTotalQueries);
    LOG_DEBUG(SYN_WORK_COMPL, "m_cntTotalCs        {}", rStats.m_cntTotalCs);
    LOG_DEBUG(SYN_WORK_COMPL, "m_cntTotalFailures  {}", rStats.m_cntTotalFailures);
    LOG_DEBUG(SYN_WORK_COMPL, "m_cntTotalSuccesses {}", rStats.m_cntTotalSuccesses);
    LOG_DEBUG(SYN_WORK_COMPL, "m_cntTotalCompleted {}", rStats.m_cntTotalCompleted);
}
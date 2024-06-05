#include "wcm_cs_querier_mock.hpp"
#include "defs.h"

WcmCsQuerierCheckerMock::WcmCsQuerierCheckerMock(
    const std::deque<std::array<std::set<uint64_t>, 32>>& rQueryPermutations)
: m_queryPermutations(rQueryPermutations)
{
}

synStatus WcmCsQuerierCheckerMock::query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams)
{
    for (unsigned csIndex = 0; csIndex < inParams->seq_len; csIndex++)
    {
        const uint64_t                     csHandleActual = inParams->seq[csIndex];
        std::set<uint64_t>::const_iterator iter           = m_queryPermutations.front()[csIndex].find(csHandleActual);
        HB_ASSERT(iter != m_queryPermutations.front()[csIndex].end(),
                  "{}: Failure csHandleActual {} is not expected",
                  __FUNCTION__,
                  csHandleActual);
    }

    m_queryPermutations.pop_front();

    uint32_t currentPos = 1;
    for (uint64_t handlesAmountHandled = 0; handlesAmountHandled < inParams->seq_len;
         currentPos <<= 1, handlesAmountHandled++)
    {
        outParams->seq_set |= currentPos;
    }
    outParams->completed = inParams->seq_len;

    return synSuccess;
}

WcmCsQuerierIncompleteMock::WcmCsQuerierIncompleteMock(uint64_t csHandle, uint64_t queryAmount)
: m_csHandle(csHandle), m_queryAmount(queryAmount), m_csStatus(PENDING_FIRST_QUERY)
{
}

synStatus WcmCsQuerierIncompleteMock::query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams)
{
    bool     csExist     = false;
    bool     csCompleted = false;
    uint32_t currentPos  = 1;
    for (uint64_t handlesAmountHandled = 0; handlesAmountHandled < inParams->seq_len;
         currentPos <<= 1, handlesAmountHandled++)
    {
        const uint64_t csHandle = inParams->seq[handlesAmountHandled];

        if (m_csHandle == csHandle)
        {
            csExist = true;
            switch (m_csStatus)
            {
                case PENDING_FIRST_QUERY:
                {
                    m_queryAmount--;
                    m_csStatus = RETRYING;
                    break;
                }
                case RETRYING:
                {
                    m_queryAmount--;
                    if (m_queryAmount == 0)
                    {
                        m_csStatus = PENDING_LAST_QUERY;
                    }
                    break;
                }
                case PENDING_LAST_QUERY:
                {
                    outParams->seq_set |= currentPos;
                    m_csStatus  = NA;
                    csCompleted = true;
                    break;
                }
                case NA:
                {
                    HB_ASSERT(false, "{}: Failure m_csHandle {}", __FUNCTION__, m_csHandle);
                    break;
                }
            }
        }
        else
        {
            outParams->seq_set |= currentPos;
        }
    }

    if (csExist)
    {
        outParams->completed = inParams->seq_len - 1 + (uint32_t)csCompleted;
    }
    else
    {
        if ((m_csStatus == RETRYING) || (m_csStatus == PENDING_LAST_QUERY))
        {
            HB_ASSERT(false,
                      "{}: Failure m_csHandle {} was not queried in state {}",
                      __FUNCTION__,
                      m_csHandle,
                      m_csStatus);
        }
        outParams->completed = inParams->seq_len;
    }

    return synSuccess;
}

WcmCsQuerierStatusMock::WcmCsQuerierStatusMock(uint64_t queryAmount, synStatus status)
: m_queryAmount(queryAmount), m_status(status)
{
}

WcmCsQuerierStatusMock::~WcmCsQuerierStatusMock()
{
    if (m_queryAmount != 0)
    {
        LOG_CRITICAL(GC, "{}: m_amount {} is not zero", HLLOG_FUNC, m_queryAmount);
        std::terminate();
    }
}

synStatus WcmCsQuerierStatusMock::query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams)
{
    if (m_queryAmount > 0)
    {
        m_queryAmount--;
        outParams->seq_set   = 0;
        outParams->completed = 0;
        return m_status;
    }

    uint32_t currentPos = 1;
    for (uint64_t handlesAmountHandled = 0; handlesAmountHandled < inParams->seq_len;
         currentPos <<= 1, handlesAmountHandled++)
    {
        outParams->seq_set |= currentPos;
    }

    outParams->completed = inParams->seq_len;
    return synSuccess;
}

WcmCsQuerierRecorderMock::WcmCsQuerierRecorderMock(unsigned queryAmount) : m_queryAmount(queryAmount) {}

synStatus WcmCsQuerierRecorderMock::query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams)
{
    HB_ASSERT(m_queryAmount > 0, "{}: Failure too many queries", __FUNCTION__);
    m_queryAmount--;

    if (m_queryAmount == 0)
    {
        m_inParams.set_value(*inParams);
    }

    uint32_t currentPos = 1;
    for (uint64_t handlesAmountHandled = 0; handlesAmountHandled < inParams->seq_len;
         currentPos <<= 1, handlesAmountHandled++)
    {
        outParams->seq_set |= currentPos;
    }

    outParams->completed = inParams->seq_len;

    return synSuccess;
}

WcmCsQuerierPause2Mock::WcmCsQuerierPause2Mock(std::array<std::future<hlthunk_wait_multi_cs_out>, 2>& rQueryPausers)
: m_rQueryPausers(rQueryPausers), queryIndex(0)
{
}

synStatus WcmCsQuerierPause2Mock::query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams)
{
    m_inParams[queryIndex].set_value(*inParams);
    *outParams = m_rQueryPausers[queryIndex].get();
    queryIndex++;
    return synSuccess;
}
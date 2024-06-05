#include <chrono>
#include "scal_base.h"

using namespace std::chrono;

Scal::CompQTdr::StatusTdr Scal::CompQTdr::tdr(const std::string& cgName, uint64_t ctr, uint64_t timeoutUs, bool timeoutDisabled, void (*logFunc)(int,const char *))
{
    StatusTdr status {};

    auto now          = steady_clock::now();
    uint64_t sinceArm = armed ? duration_cast<microseconds>(now - armTime).count() : 0;

    if (logFunc)
    {
        int logLevel = (armed && (sinceArm > timeoutUs)) ? HLLOG_LEVEL_ERROR : HLLOG_LEVEL_INFO;
        std::string msg = fmt::format(FMT_COMPILE("tdr: {:40}{} longSo: curr/target/prev {:#8x} / {:#8x} / {:#8x} armed:{} since armed {}us"
                  " timeoutUs {}"),
                  cgName,
                  expectedCqCtr != ctr ? "*" : " ",
                  ctr,
                  expectedCqCtr,
                  prevCtr,
                  armed,
                  sinceArm,
                  timeoutUs);

        logFunc(logLevel, msg.c_str());
        return status;
    }

    if (prevCtr != ctr)
    {
        status.hasChanged = true;
    }

    if (ctr >= expectedCqCtr) // can be ">" if user doesn't update the expectedCqCtr
    {
        prevCtr = ctr;
        armed   = false;
        return status;
    }

    if (!armed) // if we see something in flight for the first time
    {
        armed   = true; // start checking
        prevCtr = ctr;  // keep the current value, so it can be checked next time
        armTime = steady_clock::now();
        return status;
    }

    // we get here if something is in flight from previous time
    if (prevCtr < ctr)  // Something was finished, update the time
    {
        prevCtr = ctr;  // keep the current value, so it can be checked next time
        armTime       = steady_clock::now();  // restart the time
        return status;
    }

    // If disabled or "timeout" hasn't passed -> all is good
    if (sinceArm <= timeoutUs)
    {
        return status;
    }

    if (timeoutDisabled)
    {
        LOG_INFO_F(SCAL, "tdr timeout - {} ctr_value {:#x} expected {:#x} armed {} prevCtr {:#x} since armed {}us"
                                     " timeoutUs {}",
                                     cgName,
                                     ctr,
                                     expectedCqCtr,
                                     armed,
                                     prevCtr,
                                     sinceArm,
                                     timeoutUs);
        return status;
    }

    LOG_ERR_F(SCAL, "tdr timeout - {} ctr_value {:#x} expected {:#x} armed {} prevCtr {:#x} since armed {}us"
                    " timeoutUs {}",
                    cgName,
                    ctr,
                    expectedCqCtr,
                    armed,
                    prevCtr,
                    sinceArm,
                    timeoutUs);

    status.timeout = true;
    return status;
}

void Scal::CompQTdr::debugCheckStatus(const std::string& cgName, uint64_t ctr)
{
    if ((debugLastCounter       != ctr) ||
        ((enginesCtr != nullptr) &&
         (debugLastEnginesCtr != *enginesCtr)) ||
        (debugLastExpectedCqCtr != expectedCqCtr))
    {
        LOG_INFO_F(SCAL,
                   "{} - counters values (Completed, Pushed to S-ARC, SW):"
                   " Current ({}, {}, {}), Prev ({}, {}, {})",
                   cgName,
                   ctr,
                   (enginesCtr != nullptr) ? std::to_string(*enginesCtr) : "NA",
                   expectedCqCtr,
                   debugLastCounter,
                   (enginesCtr != nullptr) ? std::to_string(debugLastEnginesCtr) : "NA",
                   debugLastExpectedCqCtr);

        debugLastCounter       = ctr;
        debugLastEnginesCtr    = (enginesCtr != nullptr) ? *enginesCtr : 0;
        debugLastExpectedCqCtr = expectedCqCtr;
    }
}

Scal::BgWork::BgWork(uint64_t timeoutUs, uint64_t timeoutDisabled)
: m_lastChange(steady_clock::now()),
  m_timeoutUsNoProgress(timeoutUs),
  m_timeoutDisabled(timeoutDisabled)
{
}

void Scal::BgWork::setTimeouts(uint64_t timeoutUs, uint64_t timeoutDisabled)
{
    m_timeoutUsNoProgress = timeoutUs;
    m_timeoutDisabled     = timeoutDisabled;
}

void Scal::BgWork::addCompletionGroup(CompletionGroupInterface* pCompletionGroup)
{
    m_cgs.push_back(pCompletionGroup);
}

int Scal::BgWork::tdr(void (*logFunc)(int, const char*), char *errMsg, int errMsgSize)
{
    std::string failedCgNoTdr;
    std::string failedCgTdr;

    bool timeout    = false;
    bool hasChanged = false;

    for (auto pCg : m_cgs)
    {
        CompQTdr::StatusTdr status = pCg->compQTdr.tdr(pCg->name, *(pCg->pCounter), m_timeoutUsNoProgress, m_timeoutDisabled, logFunc);

        timeout    |= status.timeout;
        hasChanged |= status.hasChanged;

        if (status.timeout && !status.hasChanged) // might be an error, log the cg name
        {
            std::string& s = pCg->compQTdr.enabled ? failedCgTdr : failedCgNoTdr;

            if (!s.empty())
            {
                s += ", ";
            }
            s += pCg->name;
        }
    }

    if (hasChanged || timeout)
    {
        uint64_t sinceChange = duration_cast<microseconds>(steady_clock::now() - m_lastChange).count();
        LOG_TRACE_F(SCAL, "hasChanged {} timeout {} sinceChange {}", hasChanged, timeout, sinceChange);
    }
    if (hasChanged)
    {
        m_lastChange = steady_clock::now();
    }
    else if (timeout)
    {
        uint64_t sinceChange = duration_cast<microseconds>(steady_clock::now() - m_lastChange).count();
        if (sinceChange > m_timeoutUsNoProgress)
        {
            LOG_ERR_F(SCAL, "no progress in {} us on cg-noTdr: {} cg-Tdr: {}", m_timeoutUsNoProgress, failedCgNoTdr, failedCgTdr);

            // If we have a no progress timeout than most most most likely a stream without TDR caused it, return only those streams
            // as a user message
            if (errMsg && (errMsgSize > 0))
            {
                std::memset(errMsg, 0, errMsgSize);

                if (!failedCgNoTdr.empty())
                {
                    failedCgNoTdr.copy(errMsg, errMsgSize - 1, 0);
                }
                else if (!failedCgTdr.empty())
                {
                    failedCgTdr.copy(errMsg, errMsgSize - 1, 0);
                }
            }
            return SCAL_TIMED_OUT;
        }
    }

    return SCAL_SUCCESS;
}

int Scal::BgWork::debugCheckStatus()
{
    for (auto pCg : m_cgs)
    {
        pCg->compQTdr.debugCheckStatus(pCg->name, *(pCg->pCounter));
    }

    return SCAL_SUCCESS;
}

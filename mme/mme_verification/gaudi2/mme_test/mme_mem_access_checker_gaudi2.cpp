#include "mme_mem_access_checker_gaudi2.h"
#include "mme_assert.h"

namespace gaudi2
{
static MmeMemAccessCheckerGaudi2::signalsCntArr signalingCnt = {0, 0};
static MmeMemAccessCheckerGaudi2::signalsCntArr expectedSignals = {0, 0};

const MmeMemAccessCheckerGaudi2* MmeMemAccessCheckerGaudi2::m_lastInitInstance = nullptr;

MmeMemAccessCheckerGaudi2::MmeMemAccessCheckerGaudi2(unsigned euNr) : MmeMemAccessChecker(), m_euNr(euNr)
{
    clearStaticVars();
}

// Clear static variable and save pointer to this instance
void MmeMemAccessCheckerGaudi2::clearStaticVars()
{
    signalingCnt = {0, 0};
    expectedSignals = {0, 0};
    m_lastInitInstance = this;
}

template<unsigned colorIdx>
bool shouldStopWaiting()
{
    MME_ASSERT(expectedSignals[colorIdx] != 0, "expected signals is 0, if got here expected some signal.");
    return (signalingCnt[colorIdx] >= expectedSignals[colorIdx]);
}

void MmeMemAccessCheckerGaudi2::signal(const unsigned colorSetIdx)
{
    m_auLock.adminBegin();
    MME_ASSERT(colorSetIdx < COLORS_NR, "unknown color set idx");
    // Init the static vars if some other test consume them after this instance initialization
    std::unique_lock<std::mutex> staticVarsLck(m_staticVarsInitMtx, std::defer_lock);
    staticVarsLck.lock();
    if (m_lastInitInstance != this)
    {
        clearStaticVars();
    }
    expectedSignals[colorSetIdx] = calcSignalsNr(colorSetIdx);  // Refresh every activation
    signalingCnt[colorSetIdx]++;
    MmeCommon::segmentCnt[colorSetIdx]++;
    m_signalsCtr[colorSetIdx] = signalingCnt[colorSetIdx];
    staticVarsLck.unlock();

    const auto stopWaiting = (colorSetIdx == 0) ? shouldStopWaiting<0> : shouldStopWaiting<1>;
    if (stopWaiting() && signalingCnt[colorSetIdx] == expectedSignals[colorSetIdx])
    {
        MmeMemAccessChecker::signal(colorSetIdx);
    }
    m_auLock.adminEnd();
}

}  // namespace gaudi2
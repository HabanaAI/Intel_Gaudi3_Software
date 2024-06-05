#pragma once

#include "mme_verification/common/mme_mem_access_checker.h"
#include <mutex>
#include <condition_variable>

namespace gaudi2
{
class MmeMemAccessCheckerGaudi2 : public MmeCommon::MmeMemAccessChecker
{
public:

    MmeMemAccessCheckerGaudi2(unsigned euNr);
    ~MmeMemAccessCheckerGaudi2() = default;

private:
    void signal(const unsigned colorSetIdx) override;
    void clearStaticVars();

    const unsigned m_euNr;
    std::mutex m_polesSyncMutex;
    std::condition_variable cv;

    // A variable to stor last object that cleared the variables above
    static const MmeMemAccessCheckerGaudi2* m_lastInitInstance;
    std::mutex m_staticVarsInitMtx;
};

}  // namespace gaudi2
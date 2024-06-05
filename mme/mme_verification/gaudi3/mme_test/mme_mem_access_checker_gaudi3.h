#pragma once

#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "mme_verification/common/mme_mem_access_checker.h"
#include "sync/overlap.h"

namespace gaudi3
{
static std::mutex polesSyncMutex;
static std::condition_variable cv;

static const unsigned COLORS_NR = 2;
static bool isFirst[COLORS_NR] = {true, true};

template<unsigned colorIdx>
static bool isFirstOfSignal()
{
    return isFirst[colorIdx];
}

class MmeMemAccessCheckerGaudi3 : public MmeCommon::MmeMemAccessChecker
{
public:
    MmeMemAccessCheckerGaudi3() : MmeMemAccessChecker() {}
    ~MmeMemAccessCheckerGaudi3() = default;

    void signal(const unsigned colorSetIdx) override { MME_ASSERT(0, "not yet implemented for gaudi3"); }
};
}  // namespace gaudi3
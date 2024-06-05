#pragma once

#include "stdint.h"
#include "wcm_types.hpp"

class CommandSubmissionDataChunks;

// Todo rename WcmObserverInterface to WcmListnerInterface
class WcmObserverInterface
{
public:
    virtual ~WcmObserverInterface() = default;

    virtual void notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed) = 0;
};
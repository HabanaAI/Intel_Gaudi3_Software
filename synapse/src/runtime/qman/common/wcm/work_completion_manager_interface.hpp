#pragma once

#include "synapse_common_types.h"
#include "wcm_types.hpp"

class WcmObserverInterface;

class WorkCompletionManagerInterface
{
public:
    virtual ~WorkCompletionManagerInterface() = default;

    virtual void addCs(WcmPhysicalQueuesId phyQueId, WcmObserverInterface* pObserver, WcmCsHandle csHandle) = 0;

    virtual void dump() = 0;
};

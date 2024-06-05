#pragma once

#include <stdint.h>

class DfaObserver
{
public:
    DfaObserver() {};
    virtual ~DfaObserver() = default;

    // failureType       => change to a DFA defined enum
    // suspectedCsHandle => 0 in case no knowledge about a suspected CS
    virtual bool notifyFailureObserved() = 0;
};
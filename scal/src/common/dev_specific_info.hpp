#pragma once

#include "scal.h"

class DevSpecificInfo
{
public:

    DevSpecificInfo() = default;
    virtual ~DevSpecificInfo() = default;

    virtual uint32_t getHeartBeatOffsetInSchedRegs() = 0;
    virtual uint32_t getSizeOfschedRegs() = 0;
    virtual uint32_t getHeartBeatOffsetInEngRegs() = 0;
    virtual uint32_t getSizeOfEngRegs() = 0;
    virtual uint32_t getHeartBeatOffsetInCmeRegs() = 0;
};

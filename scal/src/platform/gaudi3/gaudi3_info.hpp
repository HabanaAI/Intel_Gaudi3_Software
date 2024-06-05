#pragma once

#include "dev_specific_info.hpp"

class Gaudi3Info : public DevSpecificInfo
{
public:
    uint32_t getHeartBeatOffsetInSchedRegs() override;
    uint32_t getSizeOfschedRegs() override;
    uint32_t getHeartBeatOffsetInEngRegs() override;
    uint32_t getSizeOfEngRegs() override;
    uint32_t getHeartBeatOffsetInCmeRegs() override;
};
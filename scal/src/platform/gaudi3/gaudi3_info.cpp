#include "gaudi3_info.hpp"
#include "gaudi3_arc_host_packets.h"
#include "scal_utilities.h"

uint32_t Gaudi3Info::getHeartBeatOffsetInSchedRegs()
{
    return offsetof(sched_registers_t, heartbeat);
}

uint32_t Gaudi3Info::getSizeOfschedRegs()
{
    return sizeof(sched_registers_t);
}

uint32_t Gaudi3Info::getHeartBeatOffsetInEngRegs()
{
    return offsetof(engine_arc_reg_t, heartbeat); // TBD : !! add support for cme
}

uint32_t Gaudi3Info::getSizeOfEngRegs()
{
    return sizeof(engine_arc_reg_t);
}

uint32_t Gaudi3Info::getHeartBeatOffsetInCmeRegs()
{
    return offsetof(cme_registers_t, heartbeat);
}

#include <limits>

#include "gaudi2_info.hpp"
#include "gaudi2_arc_host_packets.h"
#include "scal_utilities.h"

uint32_t Gaudi2Info::getHeartBeatOffsetInSchedRegs()
{
    return offsetof(sched_registers_t, heartbeat);
}

uint32_t Gaudi2Info::getSizeOfschedRegs()
{
    return sizeof(sched_registers_t);
}

uint32_t Gaudi2Info::getHeartBeatOffsetInEngRegs()
{
    return offsetof(engine_arc_reg_t, heartbeat);
}

uint32_t Gaudi2Info::getSizeOfEngRegs()
{
    return sizeof(engine_arc_reg_t);
}

uint32_t Gaudi2Info::getHeartBeatOffsetInCmeRegs()
{
    return std::numeric_limits<uint32_t>::max(); // No cme for gaudi2
}

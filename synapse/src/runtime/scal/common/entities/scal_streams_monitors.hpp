#pragma once

#include "synapse_common_types.h"

class ScalStreamsMonitors
{
public:
    synStatus init(unsigned baseIdx, unsigned size, unsigned jump = 1);
    synStatus getMonitorId(uint64_t& monitorId);

private:
    uint64_t m_baseIdx;
    uint64_t m_size;
    unsigned m_jump;

    uint64_t m_curIdx;
};

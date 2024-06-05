#include "scal_streams_monitors.hpp"

#include "log_manager.h"

synStatus ScalStreamsMonitors::init(unsigned baseIdx, unsigned size, unsigned jump)
{
    m_baseIdx = baseIdx;
    m_size    = size;
    m_curIdx  = 0;
    m_jump    = jump;
    return synSuccess;
}

synStatus ScalStreamsMonitors::getMonitorId(uint64_t& monitorId)
{
    if (m_curIdx >= m_size)
    {
        LOG_ERR(SYN_STREAM, "getMonitorId out of monitors");
        return synFail;
    }

    monitorId = m_baseIdx + m_curIdx;
    m_curIdx += m_jump;
    return synSuccess;
}

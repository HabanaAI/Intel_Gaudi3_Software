#include "scal_streams_fences.hpp"

#include "defs.h"

#include "log_manager.h"

synStatus ScalStreamsFences::init(unsigned baseIdx, unsigned size)
{
    m_baseIdx = baseIdx;
    m_size    = size;
    m_curIdx  = 0;

    return synSuccess;
}

void ScalStreamsFences::getFenceId(FenceIdType& fenceId)
{
    HB_ASSERT(m_curIdx < m_size, "getFenceId out of monitors");
    HB_ASSERT(m_baseIdx + m_curIdx <= 256, "fenceId overflow");

    fenceId = m_baseIdx + m_curIdx;
    m_curIdx++;
}

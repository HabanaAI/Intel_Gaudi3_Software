#include "queue_base.hpp"
#include "log_manager.h"

QueueBase::QueueBase(const BasicQueueInfo& rBasicQueueInfo) : m_basicQueueInfo(rBasicQueueInfo)
{
    LOG_INFO_T(SYN_STREAM, "Stream: {:#x} created {}", TO64(this), m_basicQueueInfo.getDescription());
}

QueueBase::~QueueBase()
{
    LOG_INFO_T(SYN_STREAM, "Stream: {:#x} destroyed {}", TO64(this), m_basicQueueInfo.getDescription());
}

#include "stream_wait_for_event_scal.hpp"
#include "queues/queue_interface.hpp"
#include "defs.h"

WaitForEventJobScal::WaitForEventJobScal(ScalEvent& rEvent, const unsigned int flags)
: StreamJob(JobType::WAIT_FOR_EVENT), m_event(rEvent), m_flags(flags), m_debugEventPointerAddress((uint64_t)&rEvent)
{
}

synStatus WaitForEventJobScal::run(QueueInterface* pStreamInterface)
{
    HB_ASSERT_PTR(pStreamInterface);
    return pStreamInterface->eventWait(m_event, m_flags, nullptr);
}

std::string WaitForEventJobScal::getJobParams() const
{
    char buff[100];
    snprintf(buff, sizeof(buff), "event orig handle 0x%lx, %s", m_debugEventPointerAddress, m_event.toString().c_str());
    return buff;
}
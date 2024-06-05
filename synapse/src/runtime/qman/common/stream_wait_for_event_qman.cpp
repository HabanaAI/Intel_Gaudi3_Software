#include "stream_wait_for_event_qman.hpp"
#include "queues/queue_interface.hpp"
#include "defs.h"

WaitForEventJobQman::WaitForEventJobQman(const QmanEvent& rEvent, const unsigned int flags, synStreamHandle streamHandle)
: StreamJob(JobType::WAIT_FOR_EVENT), m_event(rEvent), m_flags(flags), m_streamHandleHcl(streamHandle)
{
}

synStatus WaitForEventJobQman::run(QueueInterface* pStreamInterface)
{
    HB_ASSERT_PTR(pStreamInterface);
    return pStreamInterface->eventWait(m_event, m_flags, m_streamHandleHcl);
}

std::string WaitForEventJobQman::getJobParams() const
{
    char buff[100];
    snprintf(buff, sizeof(buff), "event=%p", &m_event);
    return buff;
}

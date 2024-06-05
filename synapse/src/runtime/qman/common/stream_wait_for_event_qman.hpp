#pragma once

#include "streams/stream_job.hpp"
#include "qman_event.hpp"

class WaitForEventJobQman : public StreamJob
{
public:
    WaitForEventJobQman(const QmanEvent& rEvent, const unsigned int flags, synStreamHandle streamHandle);

    virtual synStatus run(QueueInterface* pStreamInterface) override;

    virtual std::string getJobParams() const override;

protected:
    const QmanEvent    m_event;
    const unsigned int m_flags;
    synStreamHandle    m_streamHandleHcl;
};
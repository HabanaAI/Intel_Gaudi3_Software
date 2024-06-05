#pragma once

#include "streams/stream_job.hpp"
#include "runtime/scal/common/scal_event.hpp"

class WaitForEventJobScal : public StreamJob
{
public:
    WaitForEventJobScal(ScalEvent& rEvent, const unsigned int flags);

    virtual synStatus run(QueueInterface* pStreamInterface) override;

    virtual std::string getJobParams() const override;

protected:
    ScalEvent          m_event;
    const unsigned int m_flags;
    uint64_t           m_debugEventPointerAddress;
};
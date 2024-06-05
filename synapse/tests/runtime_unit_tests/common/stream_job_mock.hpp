#pragma once

#include "runtime/common/streams/stream_job.hpp"

class StreamJobMock : public StreamJob
{
public:
    StreamJobMock(JobType jobType, QueueInterface*& rpStreamInterface);
    virtual ~StreamJobMock() = default;
    virtual synStatus run(QueueInterface* pStreamInterface) override;

private:
    QueueInterface*& m_rpStreamInterface;
};

class MemcopyD2HJobMock : public StreamJobMock
{
public:
    MemcopyD2HJobMock(QueueInterface*& rpStreamInterface);
    virtual ~MemcopyD2HJobMock() = default;
};

class MemcopyH2DJobMock : public StreamJobMock
{
public:
    MemcopyH2DJobMock(QueueInterface*& rpStreamInterface);
    virtual ~MemcopyH2DJobMock() = default;
};

class NetworkJobMock : public StreamJobMock
{
public:
    NetworkJobMock(QueueInterface*& rpStreamInterface);
    virtual ~NetworkJobMock() = default;
};

class EventRecordJobMock : public StreamJobMock
{
public:
    EventRecordJobMock(QueueInterface*& rpStreamInterface);
    virtual ~EventRecordJobMock() = default;
};

class WaitForEventJobMock : public StreamJobMock
{
public:
    WaitForEventJobMock(QueueInterface*& rpStreamInterface);
    virtual ~WaitForEventJobMock() = default;
};

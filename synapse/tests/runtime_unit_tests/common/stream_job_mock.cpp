#include "stream_job_mock.hpp"

StreamJobMock::StreamJobMock(JobType jobType, QueueInterface*& rpStreamInterface)
: StreamJob(jobType), m_rpStreamInterface(rpStreamInterface)
{
}

synStatus StreamJobMock::run(QueueInterface* pStreamInterface)
{
    m_rpStreamInterface = pStreamInterface;
    return synSuccess;
}

MemcopyD2HJobMock::MemcopyD2HJobMock(QueueInterface*& rpStreamInterface)
: StreamJobMock(JobType::MEMCOPY_D2H, rpStreamInterface)
{
}

MemcopyH2DJobMock::MemcopyH2DJobMock(QueueInterface*& rpStreamInterface)
: StreamJobMock(JobType::MEMCOPY_H2D, rpStreamInterface)
{
}

NetworkJobMock::NetworkJobMock(QueueInterface*& rpStreamInterface) : StreamJobMock(JobType::NETWORK, rpStreamInterface)
{
}

EventRecordJobMock::EventRecordJobMock(QueueInterface*& rpStreamInterface)
: StreamJobMock(JobType::RECORD_EVENT, rpStreamInterface)
{
}

WaitForEventJobMock::WaitForEventJobMock(QueueInterface*& rpStreamInterface)
: StreamJobMock(JobType::WAIT_FOR_EVENT, rpStreamInterface)
{
}

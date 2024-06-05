#include <gtest/gtest.h>
#include "runtime/common/streams/stream.hpp"
#include "runtime/scal/common/stream_wait_for_event_scal.hpp"
#include "runtime/scal/common/scal_event.hpp"
#include "stream_copy_mock.hpp"
#include "stream_job_mock.hpp"
#include <deque>

class UTStreamTest : public ::testing::Test
{
public:
    UTStreamTest() = default;

    virtual ~UTStreamTest() = default;
};

TEST_F(UTStreamTest, check_failure_on_second_set_affinity)
{
    QueueMock            streamD2H0;
    QueueMock            streamD2H1;

    synEventHandle eventHandle = nullptr;
    ScalEvent      event(0, 0, nullptr);

    const unsigned     streamAffinityFirst = 0;
    QueueInterfacesArray internalStreamHandles0 {&streamD2H0, nullptr, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, streamAffinityFirst, internalStreamHandles0);
    EXPECT_EQ(streamAffinityFirst, Stream.getAffinity());

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    const unsigned     streamAffinitySecond = 1;
    QueueInterfacesArray internalStreamHandles1 {&streamD2H1, nullptr, nullptr, nullptr, nullptr};
    synStatus status = Stream.setAffinity(streamAffinitySecond, internalStreamHandles1, JobType::MEMCOPY_D2H);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(streamAffinitySecond, Stream.getAffinity());

    // The second setAffinity has to fail
    const unsigned     streamAffinityThird = 2;
    QueueInterfacesArray internalStreamHandles2 {nullptr, nullptr, nullptr, nullptr, nullptr};
    status = Stream.setAffinity(streamAffinityThird, internalStreamHandles2);
    EXPECT_EQ(status, synFail);
    EXPECT_EQ(streamAffinitySecond, Stream.getAffinity());
}

TEST_F(UTStreamTest, check_set_affinity_same_type_record)
{
    QueueMock            streamD2H0;
    QueueMock            streamD2H1;

    synEventHandle eventHandle = nullptr;
    ScalEvent      event(0, 0, nullptr);

    const unsigned     streamAffinityFirst = 0;
    QueueInterfacesArray internalStreamHandles0 {&streamD2H0, nullptr, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, streamAffinityFirst, internalStreamHandles0);
    EXPECT_EQ(streamAffinityFirst, Stream.getAffinity());

    // Add  job and make sure it is running on default stream
    QueueInterface*            pStreamInterface1 = nullptr;
    std::unique_ptr<StreamJob> job1              = std::make_unique<EventRecordJobMock>(pStreamInterface1);
    synStatus                  status            = Stream.addJob(job1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterface1, &streamD2H0);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    const unsigned     streamAffinitySecond = 1;
    QueueInterfacesArray internalStreamHandles1 {&streamD2H1, nullptr, nullptr, nullptr, nullptr};
    status = Stream.setAffinity(streamAffinitySecond, internalStreamHandles1, JobType::MEMCOPY_D2H);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(streamAffinitySecond, Stream.getAffinity());
    EXPECT_EQ(1, streamD2H0.m_recordCounter);
    EXPECT_EQ(1, streamD2H1.m_waitCounter);

    // Add D2H job and ensure there is no redundant synchronization
    QueueInterface*            pStreamInterfaceD2H = nullptr;
    std::unique_ptr<StreamJob> jobD2H              = std::make_unique<MemcopyD2HJobMock>(pStreamInterfaceD2H);
    status                                         = Stream.addJob(jobD2H);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceD2H, &streamD2H1);
    EXPECT_EQ(0, streamD2H1.m_recordCounter);
    EXPECT_EQ(1, streamD2H1.m_waitCounter);
}

TEST_F(UTStreamTest, check_set_affinity_same_type_wait)
{
    QueueMock            streamD2H0;
    QueueMock            streamD2H1;

    synEventHandle eventHandle = nullptr;
    ScalEvent      event(0, 0, nullptr);

    const unsigned     streamAffinityFirst = 0;
    QueueInterfacesArray internalStreamHandles0 {&streamD2H0, nullptr, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, streamAffinityFirst, internalStreamHandles0);
    EXPECT_EQ(streamAffinityFirst, Stream.getAffinity());

    // Add  job and make sure it is running on default stream
    QueueInterface*            pStreamInterface1 = nullptr;
    std::unique_ptr<StreamJob> job1              = std::make_unique<WaitForEventJobMock>(pStreamInterface1);
    synStatus                  status            = Stream.addJob(job1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterface1, nullptr);


    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    const unsigned     streamAffinitySecond = 1;
    QueueInterfacesArray internalStreamHandles1 {&streamD2H1, nullptr, nullptr, nullptr, nullptr};
    status = Stream.setAffinity(streamAffinitySecond, internalStreamHandles1, JobType::MEMCOPY_D2H);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(streamAffinitySecond, Stream.getAffinity());
    EXPECT_EQ(0, streamD2H0.m_recordCounter);
    EXPECT_EQ(0, streamD2H1.m_waitCounter);

    // Add D2H job and ensure there is no redundant synchronization
    QueueInterface*            pStreamInterfaceD2H = nullptr;
    std::unique_ptr<StreamJob> jobD2H              = std::make_unique<MemcopyD2HJobMock>(pStreamInterfaceD2H);
    status                                         = Stream.addJob(jobD2H);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterface1, &streamD2H1);
    EXPECT_EQ(pStreamInterfaceD2H, &streamD2H1);
    EXPECT_EQ(0, streamD2H1.m_recordCounter);
    EXPECT_EQ(0, streamD2H1.m_waitCounter);
}

TEST_F(UTStreamTest, check_set_affinity_different_type)
{
    QueueMock            streamD2H;
    QueueMock            streamH2D;

    synEventHandle eventHandle = nullptr;
    ScalEvent      event(0, 0, nullptr);

    const unsigned     streamAffinityFirst = 0;
    QueueInterfacesArray internalStreamHandles0 {&streamD2H, nullptr, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, streamAffinityFirst, internalStreamHandles0);
    EXPECT_EQ(streamAffinityFirst, Stream.getAffinity());

    // Add record job and make sure it is running on default stream
    QueueInterface*            pStreamInterfaceRecord1 = nullptr;
    std::unique_ptr<StreamJob> jobRecord1              = std::make_unique<EventRecordJobMock>(pStreamInterfaceRecord1);
    synStatus                  status                  = Stream.addJob(jobRecord1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceRecord1, &streamD2H);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    const unsigned     streamAffinitySecond = 1;
    QueueInterfacesArray internalStreamHandles1 {nullptr, &streamH2D, nullptr, nullptr, nullptr};
    status = Stream.setAffinity(streamAffinitySecond, internalStreamHandles1, JobType::MEMCOPY_H2D);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(streamAffinitySecond, Stream.getAffinity());
    EXPECT_EQ(1, streamD2H.m_recordCounter);
    EXPECT_EQ(1, streamH2D.m_waitCounter);

    // Add H2D job and ensure there is no redundant synchronization
    QueueInterface*            pStreamInterfaceH2D = nullptr;
    std::unique_ptr<StreamJob> jobH2D              = std::make_unique<MemcopyH2DJobMock>(pStreamInterfaceH2D);
    status                                         = Stream.addJob(jobH2D);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceH2D, &streamH2D);
    EXPECT_EQ(0, streamH2D.m_recordCounter);
    EXPECT_EQ(1, streamH2D.m_waitCounter);
}

TEST_F(UTStreamTest, check_record)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            streamD2H;
    QueueMock            streamNet;

    // Record job is done on the default stream QUEUE_TYPE_COPY_DEVICE_TO_HOST (FIRST_JOB_QUEUE_TYPE) and later on the
    // QUEUE_TYPE_NETWORK_COLLECTIVE
    QueueInterfacesArray internalStreamHandles {&streamD2H, nullptr, nullptr, nullptr, &streamNet};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Add record job and make sure it is running on default stream
    QueueInterface*            pStreamInterfaceRecord1 = nullptr;
    std::unique_ptr<StreamJob> jobRecord1              = std::make_unique<EventRecordJobMock>(pStreamInterfaceRecord1);
    status                                             = Stream.addJob(jobRecord1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceRecord1, &streamD2H);

    // Add Network job
    QueueInterface*            pStreamInterfaceNetwork = nullptr;
    std::unique_ptr<StreamJob> jobNetwork              = std::make_unique<NetworkJobMock>(pStreamInterfaceNetwork);
    status                                             = Stream.addJob(jobNetwork);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceNetwork, &streamNet);

    // Add record job and make sure it is running on network stream
    QueueInterface*            pStreamInterfaceRecord2 = nullptr;
    std::unique_ptr<StreamJob> jobRecord2              = std::make_unique<EventRecordJobMock>(pStreamInterfaceRecord2);
    status                                             = Stream.addJob(jobRecord2);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceRecord2, &streamNet);
}

TEST_F(UTStreamTest, check_wait)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            streamNet;

    // Wait job is done on the QUEUE_TYPE_NETWORK_COLLECTIVE
    QueueInterfacesArray internalStreamHandles {nullptr, nullptr, nullptr, nullptr, &streamNet};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Add wait job and make sure that it is not run on this call
    QueueInterface*            pStreamInterfaceWait = nullptr;
    std::unique_ptr<StreamJob> jobWait              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait);
    status                                          = Stream.addJob(jobWait);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, nullptr);
    EXPECT_EQ(0, streamNet.m_recordCounter);
    EXPECT_EQ(0, streamNet.m_waitCounter);

    // Add Network job and make sure that pending wait and network jobs are running on net stream
    QueueInterface*            pStreamInterfaceNetwork = nullptr;
    std::unique_ptr<StreamJob> jobNetwork              = std::make_unique<NetworkJobMock>(pStreamInterfaceNetwork);
    status                                             = Stream.addJob(jobNetwork);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(0, streamNet.m_recordCounter);
    EXPECT_EQ(0, streamNet.m_waitCounter);
    EXPECT_EQ(pStreamInterfaceWait, &streamNet);
    EXPECT_EQ(pStreamInterfaceNetwork, &streamNet)
    ;
}

TEST_F(UTStreamTest, check_record_wait_record_sync)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            streamD2H;

    // Record job is done on the default stream QUEUE_TYPE_COPY_DEVICE_TO_HOST (FIRST_JOB_QUEUE_TYPE)
    QueueInterfacesArray internalStreamHandles {&streamD2H, nullptr, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Add record job and make sure it is running on default stream
    QueueInterface*            pStreamInterfaceRecord1 = nullptr;
    std::unique_ptr<StreamJob> jobRecord1              = std::make_unique<EventRecordJobMock>(pStreamInterfaceRecord1);
    status                                             = Stream.addJob(jobRecord1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceRecord1, &streamD2H);

    // Add wait job and make sure that it is not run on this call
    QueueInterface*            pStreamInterfaceWait = nullptr;
    std::unique_ptr<StreamJob> jobWait              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait);
    status                                          = Stream.addJob(jobWait);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, nullptr);

    // Add record job and make sure it is running on default stream
    QueueInterface*            pStreamInterfaceRecord2 = nullptr;
    std::unique_ptr<StreamJob> jobRecord2              = std::make_unique<EventRecordJobMock>(pStreamInterfaceRecord2);
    status                                             = Stream.addJob(jobRecord2);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, &streamD2H);
    EXPECT_EQ(pStreamInterfaceRecord2, &streamD2H);
}

TEST_F(UTStreamTest, check_synchronize)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            streamD2H;
    QueueMock            streamH2D;

    QueueInterfacesArray internalStreamHandles {&streamD2H, &streamH2D, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    QueueInterface*            pStreamInterfaceD2H = nullptr;
    std::unique_ptr<StreamJob> jobD2H              = std::make_unique<MemcopyD2HJobMock>(pStreamInterfaceD2H);
    status                                         = Stream.addJob(jobD2H);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceD2H, &streamD2H);
    EXPECT_EQ(0, streamD2H.m_recordCounter);
    EXPECT_EQ(0, streamD2H.m_waitCounter);

    status = Stream.synchronize();
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(0, streamD2H.m_recordCounter);
    EXPECT_EQ(0, streamH2D.m_waitCounter);
    EXPECT_EQ(1, streamD2H.m_syncCounter);

    // we can't assume this call is from the same thread that called stream sync
    // so sync queues is must
    QueueInterface*            pStreamInterfaceH2D = nullptr;
    std::unique_ptr<StreamJob> jobH2D              = std::make_unique<MemcopyH2DJobMock>(pStreamInterfaceH2D);
    status                                         = Stream.addJob(jobH2D);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceH2D, &streamH2D);
    EXPECT_EQ(1, streamD2H.m_recordCounter);
    EXPECT_EQ(1, streamH2D.m_waitCounter);
}

TEST_F(UTStreamTest, check_synchronize_empty)
{
    synEventHandle                                               eventHandle = nullptr;
    ScalEvent                                                    event(0, 0, nullptr);
    std::array<QueueMock, QUEUE_TYPE_MAX_USER_TYPES>             streams;
    std::array<QueueInterface*, QUEUE_TYPE_MAX_USER_TYPES>       streamHandles;

    for (unsigned index = QUEUE_TYPE_COPY_DEVICE_TO_HOST; index < QUEUE_TYPE_MAX_USER_TYPES; index++)
    {
        streamHandles[index] = &streams[index];
    }

    // Synchronize is executed on all streams
    QueueInterfacesArray internalStreamHandles {streamHandles[QUEUE_TYPE_COPY_DEVICE_TO_HOST],
                                                streamHandles[QUEUE_TYPE_COPY_HOST_TO_DEVICE],
                                                streamHandles[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE],
                                                streamHandles[QUEUE_TYPE_COMPUTE],
                                                streamHandles[QUEUE_TYPE_NETWORK_COLLECTIVE]};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Synchronize
    status = Stream.synchronize();
    EXPECT_EQ(status, synSuccess);

    EXPECT_EQ(1, streams[QUEUE_TYPE_COPY_DEVICE_TO_HOST].m_syncCounter);
    EXPECT_EQ(0, streams[QUEUE_TYPE_COPY_HOST_TO_DEVICE].m_syncCounter);
    EXPECT_EQ(0, streams[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE].m_syncCounter);
    EXPECT_EQ(0, streams[QUEUE_TYPE_COMPUTE].m_syncCounter);
    EXPECT_EQ(0, streams[QUEUE_TYPE_NETWORK_COLLECTIVE].m_syncCounter);
}

TEST_F(UTStreamTest, check_synchronize_with_wait)
{
    synEventHandle                                               eventHandle = nullptr;
    ScalEvent                                                    event(0, 0, nullptr);
    std::array<QueueMock, QUEUE_TYPE_MAX_USER_TYPES>             streams;
    std::array<QueueInterface*, QUEUE_TYPE_MAX_USER_TYPES>       streamHandles;

    for (unsigned index = QUEUE_TYPE_COPY_DEVICE_TO_HOST; index < QUEUE_TYPE_MAX_USER_TYPES; index++)
    {
        streamHandles[index] = {&streams[index]};
    }

    // Synchronize is executed on all streams (note we mark the last one as nullptr in order to ensure we do not crash)
    QueueInterfacesArray internalStreamHandles {streamHandles[QUEUE_TYPE_COPY_DEVICE_TO_HOST],
                                                streamHandles[QUEUE_TYPE_COPY_HOST_TO_DEVICE],
                                                streamHandles[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE],
                                                streamHandles[QUEUE_TYPE_COMPUTE],
                                                nullptr};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Add wait job and make sure that it is not run on this call
    QueueInterface*            pStreamInterfaceWait = nullptr;
    std::unique_ptr<StreamJob> jobWait              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait);
    status                                          = Stream.addJob(jobWait);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, nullptr);

    // Synchronize
    status = Stream.synchronize();
    EXPECT_EQ(status, synSuccess);

    unsigned syncCounter = 0;
    for (unsigned index = QUEUE_TYPE_COPY_DEVICE_TO_HOST; index < QUEUE_TYPE_MAX_USER_TYPES; index++)
    {
        syncCounter += streams[index].m_syncCounter;
    }

    // Ensure that synchronize executed the pending wait command on the default stream
    EXPECT_EQ(pStreamInterfaceWait, &streams[QUEUE_TYPE_COPY_DEVICE_TO_HOST]);
    EXPECT_EQ(syncCounter, 1);
}

TEST_F(UTStreamTest, check_query)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            streamD2H;

    // Query job is done on the default stream QUEUE_TYPE_COPY_DEVICE_TO_HOST (FIRST_JOB_QUEUE_TYPE)
    QueueInterfacesArray internalStreamHandles {&streamD2H, nullptr, nullptr, nullptr, nullptr};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Add wait job and make sure that it is not run on this call
    QueueInterface*            pStreamInterfaceWait = nullptr;
    std::unique_ptr<StreamJob> jobWait              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait);
    status                                          = Stream.addJob(jobWait);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, nullptr);

    // Query
    status = Stream.query();
    EXPECT_EQ(status, synSuccess);

    // Ensure that synchronize executed the pending wait command on the network stream
    EXPECT_EQ(pStreamInterfaceWait, &streamD2H);
    EXPECT_EQ(streamD2H.m_queryCounter, 1);
}

TEST_F(UTStreamTest, check_flush_waits)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            streamNetwork;

    // Record job is done on the network stream QUEUE_TYPE_NETWORK_COLLECTIVE
    QueueInterfacesArray internalStreamHandles {nullptr, nullptr, nullptr, nullptr, &streamNetwork};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    // Add wait job and make sure that it is not run on this call
    QueueInterface*            pStreamInterfaceWait = nullptr;
    std::unique_ptr<StreamJob> jobWait              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait);
    status                                          = Stream.addJob(jobWait);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, nullptr);

    // Flush
    status = Stream.flushWaitsOnCollectiveQueue();
    EXPECT_EQ(status, synSuccess);

    // Ensure that synchronize executed the pending wait command on the network stream
    EXPECT_EQ(pStreamInterfaceWait, &streamNetwork);
}

class CounterStreamJobMock : public StreamJob
{
    FRIEND_TEST(UTStreamTest, check_different_stream_jobs_sync);

public:
    CounterStreamJobMock(JobType jobType);
    virtual ~CounterStreamJobMock() = default;
    virtual synStatus run(QueueInterface* pStreamInterface) override;

private:
    // Note: m_sharedCounter is not atomic since Stream is taking care of the synchronization
    static uint64_t m_sharedCounter;
    // Note: Used to lock only same type. Different type should be locked by Stream
    static std::mutex m_jobMutex[QUEUE_TYPE_MAX_USER_TYPES];
};

uint64_t   CounterStreamJobMock::m_sharedCounter = 0;
std::mutex CounterStreamJobMock::m_jobMutex[QUEUE_TYPE_MAX_USER_TYPES] {};

CounterStreamJobMock::CounterStreamJobMock(JobType jobType) : StreamJob(jobType) {}

synStatus CounterStreamJobMock::run(QueueInterface* pStreamInterface)
{
    std::lock_guard<std::mutex> lockJobType(m_jobMutex[(uint64_t)Stream::_getQueueType(m_jobType)]);
    m_sharedCounter++;
    return synSuccess;
}

// It is preferable to execute this test using the thread sanitizer which intercepts
// CounterStreamJobMock::m_sharedCounter data race
TEST_F(UTStreamTest, check_different_stream_jobs_sync)
{
    synEventHandle       eventHandle = nullptr;
    ScalEvent            event(0, 0, nullptr);
    QueueMock            stream;
    QueueInterfacesArray internalStreamHandles {&stream, &stream, &stream, &stream, &stream};
    Stream               Stream(eventHandle, event, 0, internalStreamHandles);

    // Simulate user flow in which the stream container is responsible to ensure that affinity is locked
    synStatus status = Stream.setAffinity(0, internalStreamHandles);
    EXPECT_EQ(status, synSuccess);

    const std::array<JobType, 18> JobTypes = {JobType::MEMCOPY_H2D,
                                              JobType::MEMCOPY_D2H,
                                              JobType::MEMCOPY_D2D,
                                              JobType::MEMSET,
                                              JobType::COMPUTE,
                                              JobType::NETWORK,
                                              JobType::MEMCOPY_H2D,
                                              JobType::MEMCOPY_D2H,
                                              JobType::MEMCOPY_D2D,
                                              JobType::MEMSET,
                                              JobType::COMPUTE,
                                              JobType::NETWORK,
                                              JobType::MEMCOPY_H2D,
                                              JobType::MEMCOPY_D2H,
                                              JobType::MEMCOPY_D2D,
                                              JobType::MEMSET,
                                              JobType::COMPUTE,
                                              JobType::NETWORK};

    const unsigned iterMax = 100;
    std::deque<std::thread> threads;
    for (auto jobType : JobTypes)
    {
        threads.emplace_back([jobType, &Stream]() {
            std::unique_ptr<StreamJob> job = std::make_unique<CounterStreamJobMock>(jobType);

            for (unsigned iter = 0; iter < iterMax; ++iter)
            {
                synStatus status = Stream.addJob(job);
                EXPECT_EQ(status, synSuccess);
            }
        });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(CounterStreamJobMock::m_sharedCounter, JobTypes.size() * iterMax);
}
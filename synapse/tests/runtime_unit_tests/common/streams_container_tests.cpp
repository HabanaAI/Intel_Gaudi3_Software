#include <gtest/gtest.h>
#include "runtime/common/streams/streams_container.hpp"
#include "runtime/scal/common/scal_event.hpp"
#include "runtime/common/streams/stream.hpp"
#include "stream_copy_mock.hpp"
#include "stream_job_mock.hpp"
#include "runtime/common/streams/stream.hpp"

class UTStreamsContainerTest : public ::testing::Test
{
public:
    UTStreamsContainerTest() = default;

    virtual ~UTStreamsContainerTest() = default;
};

TEST_F(UTStreamsContainerTest, check_implicit_set_affinity)
{
    QueueMock            streamD2H;
    QueueMock            streamNet;

    const bool             setStreamAffinityByJobType = true;
    StreamsContainer       container(setStreamAffinityByJobType);

    QueueInterfacesArrayVector queueHandles {{&streamD2H, nullptr, nullptr, nullptr, &streamNet}};
    synStatus                status = container.addStreamAffinities(queueHandles);

    synStreamHandle streamHandle;
    synEventHandle  eventHandle = nullptr;
    ScalEvent       event(0, 0, nullptr);

    status = container.createStream(eventHandle, event, &streamHandle);
    EXPECT_EQ(status, synSuccess);

    // Add wait job and make sure that it is not run on this call and that the add affinity is not set
    QueueInterface*            pStreamInterfaceWait = nullptr;
    std::unique_ptr<StreamJob> jobWait              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait);
    Stream* pStream                                 = container.getStreamSptr(streamHandle).get();
    status                                          = container.addJob(pStream, jobWait);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, nullptr);
    EXPECT_EQ(pStream->isAffinityLocked(), false);

    // Add Network job which implicitly trigger set affinity
    QueueInterface*            pStreamInterfaceNetwork = nullptr;
    std::unique_ptr<StreamJob> jobNetwork              = std::make_unique<NetworkJobMock>(pStreamInterfaceNetwork);
    status                                             = container.addJob(pStream, jobNetwork);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait, &streamNet);
    EXPECT_EQ(pStreamInterfaceNetwork, &streamNet);
    EXPECT_EQ(pStream->isAffinityLocked(), true);

    const unsigned affinity = pStream->getAffinity();
    EXPECT_EQ(affinity, 0);

    status = container.destroyStream(streamHandle);
    EXPECT_EQ(status, synSuccess);

    status = container.removeAffinities(queueHandles);
    EXPECT_EQ(status, synSuccess);
}

TEST_F(UTStreamsContainerTest, check_implicit_set_affinity_round_robin)
{
    std::array<QueueMock, 2>            streamD2H;
    std::array<QueueMock, 2>            streamNet;

    const bool             setStreamAffinityByJobType = true;
    StreamsContainer       container(setStreamAffinityByJobType);

    QueueInterfacesArrayVector queueHandles {{&streamD2H[0], nullptr, nullptr, nullptr, &streamNet[0]},
                                             {&streamD2H[1], nullptr, nullptr, nullptr, &streamNet[1]}};
    synStatus                status = container.addStreamAffinities(queueHandles);

    std::array<synStreamHandle, 2> streamHandles;

    synEventHandle                 eventHandle = nullptr;
    ScalEvent                      event(0, 0, nullptr);

    status = container.createStream(eventHandle, event, &streamHandles[0]);
    EXPECT_EQ(status, synSuccess);

    status = container.createStream(eventHandle, event, &streamHandles[1]);
    EXPECT_EQ(status, synSuccess);

    std::array<Stream*, 2>  streams = {container.getStreamSptr(streamHandles[0]).get(), container.getStreamSptr(streamHandles[1]).get()};
    // Add wait job and make sure that it is not run on this call and that the add affinity is not set
    QueueInterface*            pStreamInterfaceWait0 = nullptr;
    std::unique_ptr<StreamJob> jobWait0              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait0);
    status                                           = container.addJob(streams[0], jobWait0);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait0, nullptr);
    EXPECT_EQ(streams[0]->isAffinityLocked(), false);

    // Add wait job and make sure that it is not run on this call and that the add affinity is not set
    QueueInterface*            pStreamInterfaceWait1 = nullptr;
    std::unique_ptr<StreamJob> jobWait1              = std::make_unique<WaitForEventJobMock>(pStreamInterfaceWait1);
    status                                           = container.addJob(streams[1], jobWait1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait1, nullptr);
    EXPECT_EQ(streams[1]->isAffinityLocked(), false);

    // Add Network job which implicitly trigger set affinity
    QueueInterface*            pStreamInterfaceNetwork0 = nullptr;
    std::unique_ptr<StreamJob> jobNetwork0              = std::make_unique<NetworkJobMock>(pStreamInterfaceNetwork0);
    status                                              = container.addJob(streams[0], jobNetwork0);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait0, &streamNet[0]);
    EXPECT_EQ(pStreamInterfaceNetwork0, &streamNet[0]);
    EXPECT_EQ(streams[0]->isAffinityLocked(), true);

    // Add Network job which implicitly trigger set affinity
    QueueInterface*            pStreamInterfaceNetwork1 = nullptr;
    std::unique_ptr<StreamJob> jobNetwork1              = std::make_unique<NetworkJobMock>(pStreamInterfaceNetwork1);
    status                                              = container.addJob(streams[1], jobNetwork1);
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(pStreamInterfaceWait1, &streamNet[1]);
    EXPECT_EQ(pStreamInterfaceNetwork1, &streamNet[1]);
    EXPECT_EQ(streams[1]->isAffinityLocked(), true);

    const unsigned affinity0 = streams[0]->getAffinity();
    EXPECT_EQ(affinity0, 0);

    const unsigned affinity1 = streams[1]->getAffinity();
    EXPECT_EQ(affinity1, 1);

    status = container.destroyStream(streamHandles[0]);
    EXPECT_EQ(status, synSuccess);

    status = container.destroyStream(streamHandles[1]);
    EXPECT_EQ(status, synSuccess);

    status = container.removeAffinities(queueHandles);
    EXPECT_EQ(status, synSuccess);
}
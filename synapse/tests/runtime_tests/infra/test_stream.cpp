#include "test_stream.hpp"
#include "test_host_buffer_malloc.hpp"
#include "test_device_buffer_alloc.hpp"
#include "test_event.hpp"
#include "synapse_api.h"

TestStream::TestStream(synStreamHandle streamHandle) : m_streamHandle(streamHandle) {}

TestStream::TestStream(TestStream&& other) : m_streamHandle(other.m_streamHandle)
{
    other.m_streamHandle = nullptr;
}

TestStream::~TestStream()
{
    try
    {
        destroy();
    }
    catch (...)
    {
    }
}

void TestStream::memcopyAsync(const TestHostBufferMalloc& rSource, const TestDeviceBufferAlloc& rDestination) const
{
    memcopyAsync((uint64_t)rSource.getBuffer(), rSource.getSize(), (uint64_t)rDestination.getBuffer(), HOST_TO_DRAM);
}

void TestStream::memcopyAsync(const TestDeviceBufferAlloc& rSource, const TestHostBufferMalloc& rDestination) const
{
    memcopyAsync((uint64_t)rSource.getBuffer(), rSource.getSize(), (uint64_t)rDestination.getBuffer(), DRAM_TO_HOST);
}

void TestStream::memcopyAsync(const uint64_t  src,
                              const uint64_t  size,
                              const uint64_t  dst,
                              const synDmaDir direction) const
{
    const synStatus status = synMemCopyAsync(m_streamHandle, src, size, dst, direction);
    ASSERT_EQ(status, synSuccess) << "synMemCopyAsync failed";
}

void TestStream::memcopyAsyncMultiple(const uint64_t* src,
                                      const uint64_t* size,
                                      const uint64_t* dst,
                                      const synDmaDir direction,
                                      const uint64_t  numCopies) const
{
    const synStatus status = synMemCopyAsyncMultiple(m_streamHandle, src, size, dst, direction, numCopies);
    ASSERT_EQ(status, synSuccess) << "synMemCopyAsyncMultiple failed";
}

void TestStream::memsetD8Async(uint64_t deviceMem, const unsigned char value, const size_t numOfElements) const
{
    const synStatus status = synMemsetD8Async(deviceMem, value, numOfElements, m_streamHandle);
    ASSERT_EQ(status, synSuccess) << "synMemsetD8Async failed";
}

void TestStream::launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                        const uint32_t                numberTensors,
                        uint64_t                      pWorkspace,
                        const synRecipeHandle         pRecipeHandle,
                        uint32_t                      flags) const
{
    const synStatus status =
        synLaunchExt(m_streamHandle, launchTensorsInfo, numberTensors, pWorkspace, pRecipeHandle, flags);
    ASSERT_EQ(status, synSuccess) << "synLaunch failed";
}

void TestStream::launchWithExternalEvents(const synLaunchTensorInfoExt* launchTensorsInfo,
                                          const uint32_t                numberTensors,
                                          uint64_t                      pWorkspace,
                                          const synRecipeHandle         pRecipeHandle,
                                          uint32_t                      flags,
                                          std::vector<TestEvent>&       sfgEvents) const
{
    const synStatus status = synLaunchWithExternalEventsExt(m_streamHandle,
                                                            launchTensorsInfo,
                                                            numberTensors,
                                                            pWorkspace,
                                                            pRecipeHandle,
                                                            (synEventHandle*)sfgEvents.data(),
                                                            sfgEvents.size(),
                                                            flags);
    ASSERT_EQ(status, synSuccess) << "synLaunch failed";
}

void TestStream::eventRecord(const TestEvent& rEvent) const
{
    ASSERT_EQ(synEventRecord(rEvent, m_streamHandle), synSuccess);
}

void TestStream::eventWait(const TestEvent& rEvent, uint32_t flags) const
{
    ASSERT_EQ(synStreamWaitEvent(m_streamHandle, rEvent, flags), synSuccess);
}

void TestStream::synchronize() const
{
    ASSERT_NE(m_streamHandle, nullptr) << "Invalid stream ID";
    const synStatus status = synStreamSynchronize(m_streamHandle);
    ASSERT_EQ(status, synSuccess) << "synStreamSynchronize failed";
}

void TestStream::query(synStatus& status) const
{
    ASSERT_NE(m_streamHandle, nullptr) << "Invalid stream ID";
    status = synStreamQuery(m_streamHandle);
}

void TestStream::destroy()
{
    if (m_streamHandle != nullptr)
    {
        const synStatus status = synStreamDestroy(m_streamHandle);
        ASSERT_EQ(status, synSuccess) << "synStreamDestroy failed";
        m_streamHandle = nullptr;
    }
}
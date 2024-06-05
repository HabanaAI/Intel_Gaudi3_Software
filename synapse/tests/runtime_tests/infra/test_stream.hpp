#pragma once

#include "../infra/test_types.hpp"

class TestEvent;
class TestHostBufferMalloc;
class TestDeviceBufferAlloc;

class TestStream final
{
public:
    TestStream(synStreamHandle streamHandle);
    TestStream(const TestStream&) = delete;
    TestStream(TestStream&& other);
    ~TestStream();
                operator synStreamHandle() const { return m_streamHandle; }
    TestStream& operator=(TestStream const&) = delete;
    TestStream& operator                     =(TestStream&& other)
    {
        m_streamHandle       = other.m_streamHandle;
        other.m_streamHandle = 0;
        return *this;
    }

    void memcopyAsync(const TestHostBufferMalloc& rSource, const TestDeviceBufferAlloc& rDestination) const;

    void memcopyAsync(const TestDeviceBufferAlloc& rSource, const TestHostBufferMalloc& rDestination) const;

    void memcopyAsync(const uint64_t src, const uint64_t size, const uint64_t dst, const synDmaDir direction) const;

    void memcopyAsyncMultiple(const uint64_t* src,
                              const uint64_t* size,
                              const uint64_t* dst,
                              const synDmaDir direction,
                              const uint64_t  numCopies) const;

    void memsetD8Async(uint64_t deviceMem, const unsigned char value, const size_t numOfElements) const;

    void launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                const uint32_t                numberTensors,
                uint64_t                      pWorkspace,
                const synRecipeHandle         pRecipeHandle,
                uint32_t                      flags) const;

    void launchWithExternalEvents(const synLaunchTensorInfoExt* launchTensorsInfo,
                                  const uint32_t                numberTensors,
                                  uint64_t                      pWorkspace,
                                  const synRecipeHandle         pRecipeHandle,
                                  uint32_t                      flags,
                                  std::vector<TestEvent>&       sfgEvents) const;

    void eventRecord(const TestEvent& rEvent) const;

    void eventWait(const TestEvent& rEvent, uint32_t flags = 0) const;

    void synchronize() const;

    void query(synStatus& status) const;

private:
    void            destroy();
    synStreamHandle m_streamHandle;
};

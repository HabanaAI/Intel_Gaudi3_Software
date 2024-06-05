#pragma once

#include "synapse_api_types.h"

#include "test_stream.hpp"
#include "test_event.hpp"
#include "test_device_buffer_alloc.hpp"
#include "test_host_buffer_malloc.hpp"
#include "test_host_buffer_map.hpp"

class TestDevice final
{
public:
    TestDevice() = delete;
    TestDevice(synDeviceType deviceType);
    TestDevice(synDeviceType deviceType, unsigned retries, std::chrono::milliseconds sleepMs);
    TestDevice(TestDevice const&) = delete;
    TestDevice(TestDevice&& other);
    ~TestDevice();

    synDeviceId getDeviceId() const { return m_deviceId; };
    void        getDeviceInfo(synDeviceInfo& deviceInfo);
    void        getDeviceMemoryInfo(uint64_t& freeSize, uint64_t& totalsize);

    void synchronize() const;

    TestStream createStream();

    TestEvent createEvent(uint32_t flags = 0);

    TestHostBufferMalloc  allocateHostBuffer(const uint64_t size, const uint32_t flags) const;
    TestHostBufferMap     mapHostBuffer(const void* hostBuffer, const uint64_t size) const;
    TestDeviceBufferAlloc allocateDeviceBuffer(const uint64_t size, const uint32_t flags) const;
    // Explicit definition for a case where the user wants to request for some specifi address (reqAddress)
    TestDeviceBufferAlloc allocateDeviceBuffer(const uint64_t size, const uint32_t flags, uint64_t reqAddress) const;

private:
    void acquire(synDeviceType             deviceType,
                 unsigned                  retries = 1,
                 std::chrono::milliseconds sleepMs = (std::chrono::milliseconds)3000);
    void release();

    // Todo [SW-152220] Come up with a better assert solution that will allow us to use ASSERT_EQ in non void methods
    void createStream(synStreamHandle& rStreamHandle);
    // Todo [SW-152220] Come up with a better assert solution that will allow us to use ASSERT_EQ in non void methods
    void createEvent(synEventHandle& rEventHandle, uint32_t flags);

    static const synDeviceId INVALID_DEVICE_ID;

    synDeviceId m_deviceId;
};

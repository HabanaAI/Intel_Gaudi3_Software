#include "test_device.hpp"

#include "synapse_api.h"

#include "test_tensor_init.hpp"

TestDevice::TestDevice(synDeviceType deviceType) : m_deviceId(SYN_INVALID_DEVICE_ID)
{
    acquire(deviceType);
}

TestDevice::TestDevice(synDeviceType deviceType, unsigned retries, std::chrono::milliseconds sleepMs)
: m_deviceId(SYN_INVALID_DEVICE_ID)
{
    acquire(deviceType, retries, sleepMs);
}

TestDevice::TestDevice(TestDevice&& other) : m_deviceId(other.m_deviceId)
{
    other.m_deviceId = SYN_INVALID_DEVICE_ID;
}

TestDevice::~TestDevice()
{
    try
    {
        release();
    }
    catch (...)
    {
    }
}

void TestDevice::getDeviceInfo(synDeviceInfo& deviceInfo)
{
    synStatus status = synDeviceGetInfo(m_deviceId, &deviceInfo);
    ASSERT_EQ(status, synSuccess) << "Failed to Get device info (" << status << ")";
}

void TestDevice::getDeviceMemoryInfo(uint64_t& freeSize, uint64_t& totalsize)
{
    synStatus status = synDeviceGetMemoryInfo(m_deviceId, &freeSize, &totalsize);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";
}

void TestDevice::acquire(synDeviceType deviceType, unsigned retries, std::chrono::milliseconds sleepTime)
{
    ASSERT_EQ(m_deviceId, SYN_INVALID_DEVICE_ID) << "Valid device ID";
    synStatus status     = synDeviceAcquireByDeviceType(&m_deviceId, deviceType);
    unsigned  attemptIdx = 0;
    while (status != synSuccess && attemptIdx++ < retries)
    {
        printf("re trying to acquire, attempt %d/%d\n", attemptIdx, retries);
        std::this_thread::sleep_for(sleepTime);
        status = synDeviceAcquireByDeviceType(&m_deviceId, deviceType);
    }
    ASSERT_EQ(status, synSuccess) << "synDeviceAcquireByDeviceType failed";
}

void TestDevice::synchronize() const
{
    ASSERT_NE(m_deviceId, SYN_INVALID_DEVICE_ID) << "Invalid device ID";
    const synStatus status = synDeviceSynchronize(m_deviceId);
    ASSERT_EQ(status, synSuccess) << "synDeviceSynchronize failed";
}

TestStream TestDevice::createStream()
{
    synStreamHandle streamHandle = 0;
    createStream(streamHandle);
    return TestStream(streamHandle);
}

TestEvent TestDevice::createEvent(uint32_t flags)
{
    synEventHandle eventHandle = 0;
    createEvent(eventHandle, flags);
    return TestEvent(eventHandle);
}

void TestDevice::release()
{
    if (SYN_INVALID_DEVICE_ID != m_deviceId)
    {
        synStatus status = synDeviceRelease(m_deviceId);
        ASSERT_EQ(status, synSuccess);
        m_deviceId = SYN_INVALID_DEVICE_ID;
    }
}

void TestDevice::createStream(synStreamHandle& rStreamHandle)
{
    const synStatus status = synStreamCreateGeneric(&rStreamHandle, m_deviceId, 0);
    ASSERT_EQ(status, synSuccess);
}

void TestDevice::createEvent(synEventHandle& rEventHandle, uint32_t flags)
{
    const synStatus status = synEventCreate(&rEventHandle, m_deviceId, flags);
    ASSERT_EQ(status, synSuccess);
}

TestHostBufferMalloc TestDevice::allocateHostBuffer(const uint64_t size, const uint32_t flags) const
{
    return TestHostBufferMalloc(m_deviceId, size, flags, nullptr);
}

TestHostBufferMap TestDevice::mapHostBuffer(const void* hostBuffer, const uint64_t size) const
{
    return TestHostBufferMap(hostBuffer, m_deviceId, size);
}

TestDeviceBufferAlloc TestDevice::allocateDeviceBuffer(const uint64_t size, const uint32_t flags) const
{
    return TestDeviceBufferAlloc(m_deviceId, size, flags, 0 /* requestedAddress */);
}

TestDeviceBufferAlloc
TestDevice::allocateDeviceBuffer(const uint64_t size, const uint32_t flags, uint64_t reqAddress) const
{
    return TestDeviceBufferAlloc(m_deviceId, size, flags, reqAddress);
}
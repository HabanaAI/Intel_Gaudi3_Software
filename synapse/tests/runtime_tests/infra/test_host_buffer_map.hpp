#pragma once

#include "test_host_buffer.hpp"

#include "synapse_api_types.h"

#include <cstdint>
#include <vector>

// Buffer will only be mapped (but not allocated - buffer is given to CTR)
class TestHostBufferMap : public TestHostBuffer
{
public:
    TestHostBufferMap(const void*& buffer, synDeviceId deviceId, uint64_t size);

    ~TestHostBufferMap();

    TestHostBufferMap(const TestHostBufferMap&) = delete;

    TestHostBufferMap(TestHostBufferMap&& other) noexcept
    : TestHostBuffer(std::move(other)), m_deviceId(other.m_deviceId), m_mappedBuffer(other.m_mappedBuffer)
    {
        other.m_deviceId     = SYN_INVALID_DEVICE_ID;
        other.m_mappedBuffer = nullptr;
    }

    void*       getBuffer() override { return nullptr; }
    const void* getBuffer() const override { return m_mappedBuffer; }

    void unmap();

private:
    synDeviceId m_deviceId;
    const void* m_mappedBuffer;
};

typedef std::vector<TestHostBufferMap> MappedHostBuffersVec;
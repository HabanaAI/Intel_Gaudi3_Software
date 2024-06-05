#pragma once

#include "test_host_buffer.hpp"

// Buffer will be allocated and mapped to device
class TestHostBufferMalloc : public TestHostBuffer
{
public:
    TestHostBufferMalloc();

    // Buffer is given and will only be mapped
    TestHostBufferMalloc(synDeviceId deviceId, uint64_t size, uint32_t flags, void* ref_arr);

    virtual ~TestHostBufferMalloc();

    TestHostBufferMalloc(const TestHostBufferMalloc&) = delete;

    TestHostBufferMalloc(const TestHostBufferMalloc&, uint64_t offset);

    TestHostBufferMalloc(TestHostBufferMalloc&& other) noexcept
    : TestHostBuffer(std::move(other)), m_deviceId(other.m_deviceId), m_buffer(other.m_buffer), m_owner(other.m_owner)
    {
        other.m_deviceId = SYN_INVALID_DEVICE_ID;
        other.m_buffer   = nullptr;
    }

    TestHostBufferMalloc& operator=(TestHostBufferMalloc&& other) noexcept
    {
        m_deviceId       = other.m_deviceId;
        m_buffer         = other.m_buffer;
        m_owner          = other.m_owner;
        other.m_deviceId = SYN_INVALID_DEVICE_ID;
        other.m_buffer   = 0;
        return *this;
    }

    void*       getBuffer() override { return m_buffer; }
    const void* getBuffer() const override { return m_buffer; }

    bool read_file(const std::string& rBufferFileName);

    void fill(uint32_t value) const;

private:
    static bool read_file(const std::string& rBufferFileName, uint32_t size, void* pBuffer);

    synDeviceId m_deviceId;
    void*       m_buffer;
    bool        m_owner = true; // only owners free the buffer
};

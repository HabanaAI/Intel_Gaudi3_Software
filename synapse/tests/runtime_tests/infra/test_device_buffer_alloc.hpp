#pragma once

#include "synapse_api_types.h"

#include <cstdint>
#include <vector>

class TestDeviceBufferAlloc
{
public:
    TestDeviceBufferAlloc();

    TestDeviceBufferAlloc(synDeviceId deviceId, uint64_t size, uint32_t flags, uint64_t requestedAddress);

    ~TestDeviceBufferAlloc();

    TestDeviceBufferAlloc(const TestDeviceBufferAlloc& other, uint64_t offset, uint64_t size);

    TestDeviceBufferAlloc(const TestDeviceBufferAlloc&) = delete;

    TestDeviceBufferAlloc(TestDeviceBufferAlloc&& other) noexcept
    : m_deviceId(other.m_deviceId), m_buffer(other.m_buffer), m_size(other.m_size), m_flags(other.m_flags), m_owner(other.m_owner)
    {
        other.m_deviceId = SYN_INVALID_DEVICE_ID;
        other.m_buffer   = 0;
        other.m_size     = 0;
        other.m_flags    = 0;
    }

    TestDeviceBufferAlloc& operator=(TestDeviceBufferAlloc&& other) noexcept
    {
        m_deviceId       = other.m_deviceId;
        m_buffer         = other.m_buffer;
        m_size           = other.m_size;
        m_flags          = other.m_flags;
        m_owner          = other.m_owner;
        other.m_deviceId = SYN_INVALID_DEVICE_ID;
        other.m_buffer   = 0;
        other.m_size     = 0;
        other.m_flags    = 0;
        return *this;
    }
    TestDeviceBufferAlloc& operator=(TestDeviceBufferAlloc& other) noexcept = delete;

    inline uint64_t getSize() const { return m_size; }
    inline uint32_t getFlags() const { return m_flags; }

    inline uint64_t       getBuffer() { return m_buffer; }
    inline const uint64_t getBuffer() const { return m_buffer; }

private:
    synDeviceId m_deviceId;
    uint64_t    m_buffer;
    uint64_t    m_size;
    uint32_t    m_flags;
    bool        m_owner = true;
};

typedef std::vector<TestDeviceBufferAlloc> AllocDeviceBuffersVec;
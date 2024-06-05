#include "test_device_buffer_alloc.hpp"

#include "../infra/test_types.hpp"

#include "synapse_api.h"

TestDeviceBufferAlloc::TestDeviceBufferAlloc() : m_deviceId(SYN_INVALID_DEVICE_ID), m_buffer(0), m_size(0), m_flags(0)
{
}

TestDeviceBufferAlloc::TestDeviceBufferAlloc(synDeviceId deviceId,
                                             uint64_t    size,
                                             uint32_t    flags,
                                             uint64_t    requestedAddress)
: m_deviceId(deviceId), m_size(size), m_flags(flags)
{
    synStatus status = synDeviceMalloc(m_deviceId, size, requestedAddress, flags, &m_buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate device-buffer";
}

TestDeviceBufferAlloc::TestDeviceBufferAlloc(const TestDeviceBufferAlloc& other, uint64_t offset, uint64_t size)
{
    // this creates a copy of the class which holds only part of the original buffer
    // and does not "own" it, e.g.  does not free it
    m_deviceId = other.m_deviceId;
    m_buffer   = other.m_buffer + offset;
    m_size     = size;
    m_flags    = other.m_flags;
    m_owner    = false;
}

TestDeviceBufferAlloc::~TestDeviceBufferAlloc()
{
    if (m_buffer == 0 || !m_owner)
    {
        return;
    }
    synStatus status = synDeviceFree(m_deviceId, m_buffer, m_flags);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_TEST, "Failed to free device-buffer");
    }

    m_buffer = 0;
    m_size   = 0;
    m_flags  = 0;
}
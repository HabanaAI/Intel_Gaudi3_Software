#include "test_host_buffer_map.hpp"

#include "../infra/test_types.hpp"

#include "synapse_api.h"

TestHostBufferMap::TestHostBufferMap(const void*& buffer, synDeviceId deviceId, uint64_t size)
: TestHostBuffer(size, 0), m_deviceId(deviceId), m_mappedBuffer(buffer)
{
    synStatus status = synHostMap(m_deviceId, size, m_mappedBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed to map host-buffer";
}

TestHostBufferMap::~TestHostBufferMap()
{
    if (m_mappedBuffer != nullptr)
    {
        synStatus status = synHostUnmap(m_deviceId, m_mappedBuffer);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_TEST, "Failed to unmap host-buffer");
        }
    }
}

void TestHostBufferMap::unmap()
{
    if (m_mappedBuffer != nullptr)
    {
        synStatus status = synHostUnmap(m_deviceId, m_mappedBuffer);
        ASSERT_EQ(status, synSuccess) << "Failed to map host-buffer";
    }
    m_mappedBuffer = nullptr;
}
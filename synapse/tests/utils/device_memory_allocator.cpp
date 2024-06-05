#include "device_memory_allocator.h"
#include "gtest/gtest.h"
#include "defs.h"

TrivialDeviceMemoryAllocator::TrivialDeviceMemoryAllocator(const synDeviceId deviceId, const uint64_t size)
: m_deviceId(deviceId), m_size(size), m_buffer(0), m_offset(0)
{
    synStatus status = synDeviceMalloc(m_deviceId, m_size, 0, 0, &m_buffer);
    EXPECT_EQ(status, synSuccess) << "Failed to allocate HBM memory";
}

TrivialDeviceMemoryAllocator::~TrivialDeviceMemoryAllocator()
{
    synStatus status = synDeviceFree(m_deviceId, m_buffer, 0);
    EXPECT_EQ(status, synSuccess) << "Failed to free HBM memory";
}

uint64_t TrivialDeviceMemoryAllocator::getDeviceMemory(const uint64_t size, const unsigned alignTo)
{
    if ((m_buffer + m_offset) % alignTo != 0)
    {
        m_offset += alignTo - ((m_buffer + m_offset) % alignTo);
    }
    HB_ASSERT(m_offset + size <= m_size, "try to get {} bytes of memory but only {} left", size, m_size - m_offset);
    m_offset += size;
    return m_buffer + m_offset - size;
}
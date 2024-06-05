#include "test_host_buffer_malloc.hpp"
#include "../infra/test_types.hpp"
#include "synapse_api.h"

TestHostBufferMalloc::TestHostBufferMalloc() : TestHostBuffer(), m_deviceId(SYN_INVALID_DEVICE_ID), m_buffer(nullptr)
{
}

TestHostBufferMalloc::TestHostBufferMalloc(synDeviceId deviceId, uint64_t size, uint32_t flags, void* ref_arr)
: TestHostBuffer(size, flags), m_deviceId(deviceId)
{
    synStatus status = synHostMalloc(m_deviceId, size, flags, &m_buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate host-buffer";
}

TestHostBufferMalloc::TestHostBufferMalloc(const TestHostBufferMalloc& other, uint64_t offset)
{
    // this creates a copy of the class which holds only part of the original buffer
    // and does not "own" it, e.g.  does not free it
    m_buffer = (void*)((uint64_t)other.m_buffer + offset);
    m_deviceId = other.m_deviceId;
    m_owner = false;
}

TestHostBufferMalloc::~TestHostBufferMalloc()
{
    if ((m_buffer != nullptr) && m_owner)
    {
        synStatus status = synHostFree(m_deviceId, m_buffer, m_flags);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_TEST, "Failed to free host-buffer");
        }
    }
}

bool TestHostBufferMalloc::read_file(const std::string& rBufferFileName, uint32_t size, void* pBuffer)
{
    std::ifstream file(rBufferFileName);
    if (file.good())
    {
        file.seekg(0, std::ios::end);
        uint32_t length = file.tellg();
        if (length != size)
        {
            file.close();
            LOG_ERR(SYN_RT_TEST,
                    "File '{}' unexpected length. Expected: {}, actual: {}.",
                    rBufferFileName,
                    size,
                    length);
            return false;
        }
        file.seekg(0, std::ios::beg);
        file.read((char*)pBuffer, length);
        file.close();
        return true;
    }
    else
    {
        LOG_ERR(SYN_RT_TEST, "File '{}' doesn't exist", rBufferFileName);
        return false;
    }
}

bool TestHostBufferMalloc::read_file(const std::string& rBufferFileName)
{
    return read_file(rBufferFileName, m_size, m_buffer);
}

void TestHostBufferMalloc::fill(uint32_t value) const
{
    std::fill((uint32_t*)m_buffer, ((uint32_t*)m_buffer) + (m_size / sizeof(uint32_t)), uint32_t(value));
}
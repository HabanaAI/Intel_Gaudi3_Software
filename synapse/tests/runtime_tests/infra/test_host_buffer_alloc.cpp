#include "test_host_buffer_alloc.hpp"

#include "../infra/test_types.hpp"

TestHostBufferAlloc::TestHostBufferAlloc() : TestHostBuffer(), m_buffer(nullptr) {}

TestHostBufferAlloc::TestHostBufferAlloc(uint64_t size, uint32_t flags) : TestHostBuffer(size, flags)
{
    m_buffer = new uint8_t[size];
}

TestHostBufferAlloc::~TestHostBufferAlloc()
{
    if (m_buffer != nullptr)
    {
        delete[](uint8_t*) m_buffer;
        m_buffer = nullptr;
    }
}
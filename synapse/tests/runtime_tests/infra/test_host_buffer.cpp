#include "test_host_buffer.hpp"

#include "../infra/test_types.hpp"

TestHostBuffer::TestHostBuffer() : m_size(0), m_flags(0) {}

TestHostBuffer::TestHostBuffer(uint64_t size, uint32_t flags) : m_size(size), m_flags(flags)
{
    ASSERT_GT(size, 0) << "Invlid size";
}
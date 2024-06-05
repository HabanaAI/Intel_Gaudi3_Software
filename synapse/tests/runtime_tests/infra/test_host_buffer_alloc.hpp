#pragma once

#include "test_host_buffer.hpp"

#include "synapse_api_types.h"

#include <cstdint>
#include <vector>

// Buffer will only be allocated (but not mapped)
class TestHostBufferAlloc : public TestHostBuffer
{
public:
    TestHostBufferAlloc();

    TestHostBufferAlloc(uint64_t size, uint32_t flags);

    ~TestHostBufferAlloc();

    TestHostBufferAlloc(const TestHostBufferAlloc&) = delete;

    TestHostBufferAlloc(TestHostBufferAlloc&& other) noexcept
    : TestHostBuffer(std::move(other)), m_buffer(other.m_buffer)
    {
        other.m_buffer = nullptr;
    }

    void*       getBuffer() override { return m_buffer; }
    const void* getBuffer() const override { return m_buffer; }

private:
    void* m_buffer;
};

typedef std::vector<TestHostBufferAlloc> AllocHostBuffersVec;
#pragma once

#include "synapse_api_types.h"

#include <cstdint>
#include <vector>

class TestHostBuffer
{
public:
    TestHostBuffer();
    TestHostBuffer(const TestHostBuffer&) = delete;

    virtual ~TestHostBuffer() {};

    uint64_t getSize() const { return m_size; }

    virtual void*       getBuffer()       = 0;
    virtual const void* getBuffer() const = 0;

protected:
    TestHostBuffer(uint64_t size, uint32_t flags);

    TestHostBuffer(TestHostBuffer&& other) noexcept : m_size(other.m_size), m_flags(other.m_flags)
    {
        other.m_size  = 0;
        other.m_flags = 0;
    }

    uint64_t m_size;
    uint32_t m_flags;
};
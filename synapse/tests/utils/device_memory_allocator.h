#pragma once

#include <memory>
#include "synapse_common_types.h"
#include "synapse_api.h"

class TrivialDeviceMemoryAllocator
{
public:
    explicit TrivialDeviceMemoryAllocator(const synDeviceId deviceId, const uint64_t size);
    ~TrivialDeviceMemoryAllocator();
    // align to 1, meaning there is no alignment
    uint64_t getDeviceMemory(const uint64_t size, const unsigned alignTo = 1);

private:
    uint64_t m_deviceId;
    uint64_t m_size;
    uint64_t m_buffer;
    uint64_t m_offset;
};

using TrivialDeviceMemoryAllocatorPtr = std::shared_ptr<TrivialDeviceMemoryAllocator>;

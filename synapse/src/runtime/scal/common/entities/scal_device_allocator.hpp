#pragma once

#include "runtime/common/osal/buffer_allocator.hpp"

class ScalMemoryPool;

class ScalDeviceAllocator : public BufferAllocator
{
public:
    ScalDeviceAllocator(ScalMemoryPool& mpGlobalHbm);

    virtual ~ScalDeviceAllocator() = default;

    virtual synStatus AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest) override;

    virtual synStatus FreeMemory() override;

private:
    ScalMemoryPool& m_mpGlobalHbm;
};

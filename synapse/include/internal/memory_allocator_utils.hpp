#pragma once

#include <sys/mman.h>
#include "synapse_common_types.h"

class MemoryAllocatorUtils
{
public:
    static void* alloc_memory_to_be_mapped_to_device(size_t length,
                                                     void*  addr   = nullptr,
                                                     int    prot   = PROT_READ | PROT_WRITE,
                                                     int    flags  = MAP_PRIVATE | MAP_ANONYMOUS,
                                                     int    fd     = -1,
                                                     off_t  offset = 0);

    static void free_memory(void* hostAddr, const size_t length) noexcept;
};
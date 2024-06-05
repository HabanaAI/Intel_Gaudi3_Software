#include "memory_allocator_utils.hpp"

#include "syn_logging.h"
#include "synapse_common_types.h"
#include "synapse_types.h"
#include "types_exception.h"
#include "types.h"

/* in order to prevent physical pages mapped to device from being swapped, thus causing MMU mapping problems, need to
   add madvise flag to those pages.
   for more details see https://jira.habana-labs.com/browse/SW-49789 */
void* MemoryAllocatorUtils::alloc_memory_to_be_mapped_to_device(size_t length,
                                                                void*  addr,
                                                                int    prot,
                                                                int    flags,
                                                                int    fd,
                                                                off_t  offset)
{
    // need to allocate page-aligned page-sized memory
    void* hostAddr = mmap(addr, length, prot, flags, fd, offset);
    if (hostAddr == MAP_FAILED)
    {
        LOG_ERR(SYN_API, "Failed to allocate {} host memory due to {}", length, errno);
        throw SynapseException("MemoryAllocatorUtils: alloc_memory_to_be_mapped_to_device");
    }

    if (prot == PROT_READ)
    {
        LOG_TRACE(SYN_MEM_MAP, "mapping in protected {:x}/{:x}", TO64(hostAddr), length);
    }

    int ret = madvise(hostAddr, length, MADV_DONTFORK);
    if (ret)
    {
        LOG_ERR(SYN_API,
                "madvise failed with errno={}, hostAddr=0x{:x}, length={}",
                std::strerror(errno),
                (uint64_t)hostAddr,
                length);
        if (munmap(hostAddr, length) == -1)
        {
            LOG_ERR(SYN_API, "Failed to unmap host memory, errno={}, hostAddr={:p}", errno, hostAddr);
        }
        throw SynapseException("MemoryAllocatorUtils: failed to madvise with MADV_DONTFORK");
    }

    return hostAddr;
}

void MemoryAllocatorUtils::free_memory(void* hostAddr, const size_t length) noexcept
{
    int ret = munmap(hostAddr, length);
    if (ret)
    {
        LOG_ERR(SYN_API, "Failed to unmap host memory, errno={}, hostAddr={:p}", errno, hostAddr);
    }
}

#include <sys/mman.h>
#include <cstdint>
#include "habana_global_conf_runtime.h"
#include "defs.h"
#include "log_manager.h"
#include "memory_protection.hpp"

namespace MemProtectUtils
{
int memWrProtectPages(uint8_t* startAddr, uint64_t size)
{
    const uint64_t pageSize = GCFG_DISABLE_SYNAPSE_HUGE_PAGES.value() ? pageSize4K : pageSize2M;
    const uint64_t pageMask = GCFG_DISABLE_SYNAPSE_HUGE_PAGES.value() ? pageMask4K : pageMask2M;

    uint8_t* alignedAddr = (uint8_t*)((uint64_t)(startAddr + pageSize - 1) & ~pageMask);

    uint64_t offset = alignedAddr - startAddr;

    if (offset > size) return 0;

    const uint64_t alignedSize = (size - offset) & ~pageMask;

    int prot = PROT_READ;

    LOG_TRACE(SYN_MEM_MAP,
              "protect user {:x}/{:x} actual {:x}/{:x} pageSize {:x}",
              TO64(startAddr),
              size,
              TO64(alignedAddr),
              alignedSize,
              pageSize);

    int rtn = mprotect(alignedAddr, alignedSize, prot);

    if (rtn != 0)
    {
        LOG_ERR_T(
            SYN_MEM_MAP,
            "failed to protect rtn {} errno {} addr {:x} size {:x} alignedAddr {:x} alignedSize {:x} pageSize {:x}",
            rtn,
            errno,
            TO64(startAddr),
            size,
            TO64(alignedAddr),
            alignedSize,
            pageSize);
        HB_ASSERT(0, "mprotect protect fail");
    }
    return rtn;
}

int memWrUnprotectPages(uint8_t* startAddr, uint64_t size)
{
    const uint64_t pageSize = GCFG_DISABLE_SYNAPSE_HUGE_PAGES.value() ? pageSize4K : pageSize2M;
    const uint64_t pageMask = GCFG_DISABLE_SYNAPSE_HUGE_PAGES.value() ? pageMask4K : pageMask2M;

    uint8_t* alignedAddr = (uint8_t*)((uint64_t)(startAddr) & ~pageMask);
    uint64_t offset      = startAddr - alignedAddr;

    const uint64_t alignedSize = (offset + size + pageSize - 1) & ~pageMask;

    int prot = PROT_READ | PROT_WRITE;

    LOG_TRACE(SYN_MEM_MAP,
              "unprotect user {:x}/{:x} actual {:x}/{:x} pageSize {:x}",
              TO64(startAddr),
              size,
              TO64(alignedAddr),
              alignedSize,
              pageSize);

    int rtn = mprotect(alignedAddr, alignedSize, prot);

    if (rtn != 0)
    {
        LOG_ERR_T(
            SYN_MEM_MAP,
            "failed to unprotect rtn {} errno {} addr {:x} size {:x} alignedAddr {:x} alignedSize {:x} pageSize {:x}",
            rtn,
            errno,
            TO64(startAddr),
            size,
            TO64(alignedAddr),
            alignedSize,
            pageSize);
        HB_ASSERT(0, "mprotect unprotect fail");
    }
    return rtn;
}

} // namespace MemProtectUtils

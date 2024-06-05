#include "arc_hbm_mem_mgr.hpp"

#include "defs.h"
#include "math_utils.h"
#include "mem_mgrs_types.hpp"

#include "log_manager.h"

/*
 ***************************************************************************************************
 *   @brief init() is called to give the module the arc-hbm memory area to use and the max number of recipes
 *                 that can be in arc-hbm waiting to be executed.
 *                 It also does some sanity checks on the address/size (alignment)
 *
 *   @param  addrCore, addrDev, size - start address of the memory to use (in core/dev addr) and size
 *   @param  downloadedRecipes - max number of recipes that can be downloaded waiting to be executed
 *   @return None
 *
 ***************************************************************************************************
 */

void ArcHbmMemMgr::init(uint32_t addrCore, uint64_t addrDev, uint32_t size)
{
    HB_ASSERT((addrCore % ALIGN) == 0, "Invalid dynamic addrCore 0x{:x}", addrCore);
    HB_ASSERT((addrDev  % ALIGN) == 0, "Invalid dynamic addrDev 0x{:x}",  addrDev);
    HB_ASSERT((size     % ALIGN) == 0, "Invalid dynamic size 0x{:x}",     size);

    m_addrCore = addrCore;
    m_addrDev  = addrDev;
    m_size     = size;

    uint64_t chunkSize = getChunkSize(size);
    NBuffAllocator::init(chunkSize);

    LOG_INFO(SYN_PROG_DWNLD,
                  "ArcHbmMemMgr: buffer addrCore {:x} addrDev {:x} size {:x} chunkSize {:x}",
                  m_addrCore,
                  m_addrDev,
                  m_size,
                  chunkSize);
}

uint64_t ArcHbmMemMgr::getChunkSize(uint64_t size)
{
    uint64_t chunkSize = size / NUM_BUFF;
    return round_down(chunkSize, ALIGN);
}

uint64_t ArcHbmMemMgr::getMaxRecipeSize(uint64_t size)
{
    uint64_t chunkSize = getChunkSize(size);
    return chunkSize * NUM_BUFF;
}

uint64_t ArcHbmMemMgr::getAddr(uint64_t size, uint64_t& hbmArcAddr, uint32_t& hbmArcCoreAddr)
{
    AllocRtn rtn       = alloc(size);
    uint64_t offset    = rtn.offset;  // start from a different chunk every time

    hbmArcAddr     = m_addrDev  + offset;
    hbmArcCoreAddr = m_addrCore + offset;

    return rtn.longSo; // return what longSo the user needs to wait for before using the buffer
}

void ArcHbmMemMgr::unuseIdOnError()
{
    // if we had an error we just need to "release" the chunks. We are doing it by setting the longSo to 0,
    // which means that the next allocator can use them immediately
    setLongSo(0);
}

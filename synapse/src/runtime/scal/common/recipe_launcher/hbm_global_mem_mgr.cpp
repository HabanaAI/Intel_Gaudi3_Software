#include "hbm_global_mem_mgr.hpp"

#include "math_utils.h"

#include "log_manager.h"

#include "runtime/scal/common/recipe_static_info_scal.hpp"
#include "runtime/scal/common/recipe_static_processor_scal.hpp"

/*
 ***************************************************************************************************
 *   @brief init() is called to give the module the hbm-global memory area to use and the max number of recipes
 *                 that can be in hbm-global waiting to be executed.
 *                 It also does some sanity checks on the address/size (alignment)
 *
 *   @param  addr, size - start address of the memory to use and size
 *   @param  downloadedRecipes - max number of recipes that can be downloaded and waiting to be executed
 *   @return None
 *
 ***************************************************************************************************
 */
void HbmGlblMemMgr::init(uint64_t addr, uint64_t size)
{
    m_addr = round_to_multiple(addr, DeviceAgnosticRecipeStaticProcessorScal::PRG_DATA_ALIGN);
    m_size = size - (m_addr - addr);

    uint64_t chunkSize = getChunkSize(size);
    NBuffAllocator::init(chunkSize);

    LOG_INFO(SYN_PROG_DWNLD,
                  "HbmGlblMemMgr: buffer addr given/actual {:x}/{:x} size given/actual{:x}/{:x} chunkSize {:x}",
                  addr,
                  addr,
                  m_addr,
                  m_size,
                  chunkSize);
}

uint64_t HbmGlblMemMgr::getChunkSize(uint64_t size)
{
    uint64_t chunkSize = size / NUM_BUFF;
    return round_down(chunkSize, DeviceAgnosticRecipeStaticProcessorScal::PRG_DATA_ALIGN);
}

uint64_t HbmGlblMemMgr::getMaxRecipeSize(uint64_t size)
{
    uint64_t chunkSize = getChunkSize(size);
    return chunkSize * NUM_BUFF;
}

/*
 ***************************************************************************************************
 *   @brief getAddr() - assign hbm-arc address for the hbm-glb sections
 *
 *   @param  needed size
 *   @param  hbmGlbAddr
 *   @param  inGlbHbm - is data already on HBM
 *   @return None
 *
 ***************************************************************************************************
 */
uint64_t HbmGlblMemMgr::getAddr(uint64_t size, uint64_t& hbmGlbAddr)
{
    AllocRtn rtn       = alloc(size);
    uint64_t offset    = rtn.offset;  // start from a different chunk every time

    hbmGlbAddr = m_addr + offset;

    return rtn.longSo; // return what longSo the user needs to wait for before using the buffer
}

void HbmGlblMemMgr::unuseIdOnError()
{
    // if we had an error we just need to "release" the chunks. We are doing it by setting the longSo to 0,
    // which means that the next allocator can use them immediately
    setLongSo(0);

}

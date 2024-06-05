#include <assert.h>
#include <cstring>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <fstream>
#include <algorithm>
#include "scal.h"
#include "scal_allocator.h"
#include "scal_base.h"
#include "scal_utilities.h"
#include "scal_gaudi2.h"
#include "logger.h"

#include "scal_data_gaudi2.h"
#include "gaudi2/gaudi2.h"
#include "gaudi2/gaudi2_packets.h"
#include "scal_qman_program_gaudi2.h"
#include "gaudi2_arc_sched_packets.h"
#include "gaudi2/asic_reg_structs/qman_arc_aux_regs.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"
#include "gaudi2/asic_reg_structs/sob_glbl_regs.h"
#include "common/pci_ids.h"

#include "scal_macros.h"
#include "scal_gaudi2_sfg_configuration_helpers.h"
#include "common/scal_init_sfg_configuration_impl.hpp"
#include "infra/monitor.hpp"
#include "infra/sync_mgr.hpp"
#include "infra/sob.hpp"

//#define HARD_CODED_NOP_KERNEL
#ifndef HARD_CODED_NOP_KERNEL
#include "gaudi2_nop.h"
#endif
// clang-format off

// called by the SCAL constructor
// * parses the config file
// * initializes the class DB
// * allocates the memory and configure the cores

using CqEn      = Monitor::CqEn;
using LongSobEn = Monitor::LongSobEn;
using LbwEn     = Monitor::LbwEn;
using CompType  = Monitor::CompType;

int Scal_Gaudi2::init(const std::string & configFileName)
{
    int ret;

    ret = setup();
    if (ret != SCAL_SUCCESS) return ret;

    // parse the config file and initialize the class DB
    ret = parseConfigFile(configFileName);
    if (ret != SCAL_SUCCESS) return ret;

    ret = initTimeout();
    if (ret != SCAL_SUCCESS) return ret;

    // allocate ARC / device / host memory and initialize the memory pools
    ret = initMemoryPools();
    if (ret != SCAL_SUCCESS) return ret;

    ret = warmupHbmTlb();
    if (ret != SCAL_SUCCESS) return ret;

    // map the ACPs and DCCMs to the device.
    ret = mapLBWBlocks();
    if (ret != SCAL_SUCCESS) return ret;

    // reset the qmans and MME dccm selections
    ret = resetQMANs();
    if (ret != SCAL_SUCCESS) return ret;

    // configure the sync managers and completion queues
    ret = configureSMs();
    if (ret != SCAL_SUCCESS) return ret;

    // load the FW to the active cores
    ret = loadFWImage();
    if (ret != SCAL_SUCCESS) return ret;

    // TPC Nop kernel is hard coded
    ret = loadTPCNopKernel();
    if (ret != SCAL_SUCCESS) return ret;

    // check the canary register of the active cores
    // todo: re-enable once access memory issue is fixed
    ret = checkCanary();
    if (ret != SCAL_SUCCESS) return ret;

    // configure the active schedulers arcs and take them out of halt mode
    ret = configSchedulers();
    if (ret != SCAL_SUCCESS) return ret;

    // configure the active engines arcs and take them out of halt mode
    ret = configEngines();
    if (ret != SCAL_SUCCESS) return ret;

    ret = configureStreams();
    if (ret != SCAL_SUCCESS) return ret;

    // write the canary register and activate the ARCs.
    ret = activateEngines();
    if (ret != SCAL_SUCCESS) return ret;

    m_bgWork = std::make_unique<BgWork>(m_timeoutUsNoProgress, m_timeoutDisabled);
    for (auto& [cgName, cgInstance] : m_completionGroups)
    {
        (void)cgName; // Unuesd
        m_bgWork->addCompletionGroup(&cgInstance);//remove!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    }

    LOG_DEBUG(SCAL,"{}: fd={} Init() Done.", __FUNCTION__, m_fd);
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::allocateHBMPoolMem(uint64_t size, uint64_t* handle, uint64_t* addr, uint64_t hintOffset, bool shared, bool contiguous)
{
    // allocate hbm memory for initMemoryPools
    // map it using the supplied hint
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HBM_START  0x1001900000000000ull
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HBM_END    0x1001AFFFFFFFFFFFull

    static constexpr uint64_t c_48_bits_mask = 0x0000ffffffffffffull;

    uint64_t hintsRangeBase  = RESERVED_VA_RANGE_FOR_ARC_ON_HBM_START;
    LOG_INFO_F(SCAL, "hlthunk_device_memory_alloc size {:#x} hintOffset {:#x} shared {} contiguous {}",
               size, hintOffset, shared, contiguous);
    *handle = hlthunk_device_memory_alloc(m_fd, size, 0, contiguous, shared);
    if (!*handle)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to allocate {:#x} bytes of device memory", __FUNCTION__, m_fd, size);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }
    // hint must be aligned to hbm page size in lower 48 bits
    if ((hintsRangeBase & c_48_bits_mask) % m_hw_ip.device_mem_alloc_default_page_size)
    {
        LOG_INFO_F(SCAL, "hint {:#x} not aligned to page, adding 16384GB", hintsRangeBase);
        hintsRangeBase += 0x100000000000ull; // use the 0x1001a00.. range for different hbm page size
    }

    if ((hintsRangeBase & c_48_bits_mask) % m_hw_ip.device_mem_alloc_default_page_size)
    {
        LOG_ERR(SCAL,"{}: fd={} hint {:#x} is not aligned to dram alloc page size {:#x}", __FUNCTION__, m_fd, hintsRangeBase,  m_hw_ip.device_mem_alloc_default_page_size);
        assert(0);
        return SCAL_FAILURE;
    }
    uint64_t hint = hintsRangeBase + hintOffset;

    LOG_INFO_F(SCAL, "hlthunk_device_memory_map with hint {:#x}", hint);
    *addr  = hlthunk_device_memory_map(m_fd, *handle, hint);
    if (!*addr)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to map device memory", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }
    if (*addr != hint)
    {
        LOG_ERR(SCAL,"{}: fd={} addr {:#x} != hint address {:#x}", __FUNCTION__, m_fd, *addr, hint);
        assert(0);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::allocateHostPoolMem(uint64_t size, void** hostAddr, uint64_t* deviceAddr, uint64_t hintOffset)
{
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HOST_START 0xFFF0780000000000ull
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HOST_END   0xFFF07FFFFFFFFFFFull

    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HOST_HPAGE_START 0xFFF0F80000000000ull
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HOST_HPAGE_END   0xFFF0FFFFFFFFFFFFull

    static const uint64_t c_host_hints_range = RESERVED_VA_RANGE_FOR_ARC_ON_HOST_START;
    static const uint64_t c_host_hints_range_huge = RESERVED_VA_RANGE_FOR_ARC_ON_HOST_HPAGE_START;

    uint64_t hint = hintOffset + c_host_hints_range_huge;
    LOG_INFO(SCAL, "{}: mmap size {:#x} hintOffset {:#x} hint {:#x}", __FUNCTION__, size, hintOffset, hint);
    *hostAddr = mmap(0, size, (PROT_READ | PROT_WRITE), (MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB), m_fd, 0);
    if (*hostAddr == MAP_FAILED)
    {
        LOG_DEBUG(SCAL,"{}: fd={} Failed to allocate {:#x} bytes of huge pages in host memory", __FUNCTION__, m_fd, size);
        // revert to regular mmap
        *hostAddr = mmap(0, size, (PROT_READ | PROT_WRITE), (MAP_SHARED | MAP_ANONYMOUS), m_fd, 0);
        if (*hostAddr == MAP_FAILED)
        {
            LOG_ERR(SCAL,"{}: fd={} Failed to allocate {:#x} bytes of host memory", __FUNCTION__, m_fd, size);
            assert(0);
            return SCAL_FAILURE;
        }
        hint = hintOffset + c_host_hints_range;
    }
    int ret = madvise(*hostAddr, size, MADV_DONTFORK);
    if (ret)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to madvise addr {:p} size {:#x} host memory ret {}", __FUNCTION__, m_fd, *hostAddr, size, ret);
        assert(0);
        return SCAL_FAILURE;
    }
    // map to device
    *deviceAddr =  hlthunk_host_memory_map(m_fd, *hostAddr , hint, size);
    if (*deviceAddr != hint)
    {
        LOG_ERR(SCAL,"{}: fd={} deviceAddr {:#x} != hint address {:#x} hostAddr {:p} size {:#x}", __FUNCTION__, m_fd, *deviceAddr, hint, *hostAddr, size);
        assert(0);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::initMemoryPools()
{
    LOG_INFO_F(SCAL, "===== initMemoryPools =====");

    uint64_t  totalMem = 0;
    uint64_t  totalArc = 0;
    Pool     *zeroPool = nullptr;

    LOG_INFO_F(SCAL, "number memory pools {:#x}", m_pools.size());

    for (auto & [k, pool]  : m_pools)
    {
        (void)k; // unused, compiler error in gcc 7.5
        std::string msg = fmt::format(FMT_COMPILE("Pool {} of type {} size {:#x}"), pool.name, HLLOG_DUPLICATE_PARAM(pool.type), pool.size);
        if ((pool.type == Pool::Type::HBM) || (pool.type == Pool::Type::HBM_EXTERNAL))
        {
            pool.fromFullHbmPool = true;
            totalMem += pool.size;

            if (pool.addressExtensionIdx != 0)
            {
                totalArc += pool.size;
            }

            if (pool.size == 0)
            {
                if (zeroPool)
                {
                    LOG_ERR(SCAL, "{}: fd={} {} Multiple HBM pools with size of 0. A: {}, B: {}",
                            __FUNCTION__, m_fd, msg, zeroPool->name, pool.name);
                    assert(0);
                    return SCAL_FAILURE;
                }

                zeroPool = &pool;
            }
            msg += fmt::format(" totalMem {:#x}", totalMem);
        }
        LOG_INFO_F(SCAL, "{}", msg);
    }

    if (totalArc > c_core_memory_extension_range_size)
    {
        LOG_ERR_F(SCAL, "All memory mapped to arc should fit in one page (256M), we have {:#x}", totalArc);

        assert(0);
    }

    if (totalMem > m_hw_ip.dram_size ||
        ((totalMem == m_hw_ip.dram_size) && zeroPool))
    {
        LOG_ERR(SCAL,"{}: fd={} Out of HBM memory totalMem {} deviceMem {} zeroPool {}", __FUNCTION__, m_fd,
                totalMem, m_hw_ip.dram_size, TO64(zeroPool));
        assert(0);
        return SCAL_FAILURE;
    }

    // zeroPool is the available user memory size and we provide all remaining memory on HBM.
    // we should reduce 4KB from it to avoid HW bug OOB prefetcher of TPC cache line, which can try
    // to read tensor from HBM + 4KB and result with a RAZWI access (SW-112283)
    if (zeroPool)
    {
        // Check if zeroPool size will be bigger than 0 (uint64_t wrap around)
        if (m_hw_ip.dram_size < totalMem + (4 * 1024))
        {
            LOG_ERR(SCAL,"{}: fd={} Out of HBM memory for zeroPool, totalMem {} deviceMem {}", __FUNCTION__, m_fd,
                    totalMem, m_hw_ip.dram_size);
            assert(0);
            return SCAL_FAILURE;
        }
        zeroPool->size = m_hw_ip.dram_size - totalMem - (4 * 1024);
        LOG_INFO_F(SCAL, "zero pool size {:#x}", zeroPool->size);
    }

    uint64_t offset = 0;

    std::vector<Pool*> poolsToAllocate;
    std::vector<Pool*> poolsFromFullHbmPool;
    // We want the binary pool to be the first, then all the ones that are mapped

    for (auto & poolPair : m_pools)
    {
        if (!poolPair.second.fromFullHbmPool)
        {
            poolsToAllocate.push_back(&poolPair.second);
        }
        else
        {
            if (&poolPair.second != m_binaryPool)
            {
                if (poolPair.second.addressExtensionIdx == 0)
                {
                    poolsFromFullHbmPool.push_back(&poolPair.second);
                }
                else
                {
                    poolsFromFullHbmPool.insert(poolsFromFullHbmPool.begin(), &poolPair.second);
                }
            }
        }
    }

    if (totalMem > 0)
    {
        poolsFromFullHbmPool.insert(poolsFromFullHbmPool.begin(), m_binaryPool);
        m_fullHbmPool.size = m_hw_ip.dram_size;
        m_fullHbmPool.name = "FullPool";
        m_fullHbmPool.type = Pool::Type::HBM;
        m_fullHbmPool.addressExtensionIdx = 5;
        poolsToAllocate.push_back(&m_fullHbmPool);
    }

    LOG_INFO_F(SCAL, "--- Allocate memroy for pools ---");
    for (auto & poolP : poolsToAllocate)
    {
        auto & pool = *poolP;

        pool.scal = this;
        pool.allocator = new ScalHeapAllocator(pool.name);
        pool.allocator->setSize(pool.size);
        const uint32_t rangeLSBs = c_core_memory_extension_range_size * pool.addressExtensionIdx; // 0x10000000 * X

        LOG_INFO_F(SCAL, "--- pool {} ---", pool.name);
        if (rangeLSBs < (uint32_t)offset)
        {
            LOG_INFO_F(SCAL, "rangeLSB {:#x} < offset {:#x}, increasing offset by 4G", rangeLSBs, offset);
            offset += 0x100000000ull; //4GB
        }
        offset = (offset & 0xFFFFFFFF00000000ull) | rangeLSBs;
        LOG_INFO_F(SCAL, "addressExtensionIdx {:#x} rangeLSBs {:#x} offset {:#x}",
                   pool.addressExtensionIdx, rangeLSBs, offset);
        if (pool.type == Pool::Type::HBM)
        {
            //We use exhaustive search instead of a formula for simplicity.
            //The number of steps is bounded by the number of memory regions (16)
            LOG_INFO_F(SCAL, "--- pool {} allocated on HBM  ---", pool.name);
            while (offset % m_hw_ip.device_mem_alloc_default_page_size != 0)
            {
                LOG_INFO_F(SCAL, "offset not multiple of page, increasing from {:#x} by 4G", offset);
                offset += 0x100000000ull; //4GB
            }
            LOG_INFO_F(SCAL, "{} request HBM allocation size {:#x} offset {:#x}", pool.name, pool.size, offset);
            int ret = allocateHBMPoolMem(pool.size, &pool.deviceHandle, &pool.deviceBase, offset);
            if (ret != SCAL_SUCCESS)
                return ret;
            pool.deviceBaseAllocatedAddress = pool.deviceBase;
        }
        else if (pool.type == Pool::Type::HOST)
        {
            LOG_INFO_F(SCAL, "--- pool {} allocated on HOST  ---", pool.name);
            int ret = allocateHostPoolMem(pool.size, &pool.hostBase, &pool.deviceBase, offset);
            if (ret != SCAL_SUCCESS)
                return ret;
        }
        offset += pool.size;
        pool.coreBase = pool.addressExtensionIdx ? (uint32_t)pool.deviceBase : 0;
        LOG_DEBUG_F(SCAL, "allocated {} ({:#x}) bytes to pool {} deviceBase {:#x} coreBase {:#x}",
                    pool.size, pool.size, pool.name, pool.deviceBase, pool.coreBase);
    }

    LOG_INFO_F(SCAL, "--- Allocate memroy from fullPool ---");
    for (auto &poolP : poolsFromFullHbmPool)
    {
        auto & pool = *poolP;

        pool.scal = this;

        uint64_t allocAddr = m_fullHbmPool.allocator->alloc(pool.size);
        if (allocAddr == Scal::Allocator::c_bad_alloc)
        {
            LOG_ERR_F(SCAL, "Failed to allocate for pool {}", pool.name);
            assert(0);
            return SCAL_FAILURE;
        }

        pool.deviceBase  = allocAddr + m_fullHbmPool.deviceBase;
        pool.deviceBaseAllocatedAddress = m_fullHbmPool.deviceBase;
        LOG_INFO_F(SCAL, "for pool {}, allocated from fullPool at addr {:#x}", pool.name, pool.deviceBase);
        if (pool.addressExtensionIdx != 0)
        {
            pool.coreBase = (uint32_t)pool.deviceBase;
        }
        pool.allocator = new ScalHeapAllocator(pool.name);
        pool.allocator->setSize(pool.size);
        LOG_INFO_F(SCAL, "for pool {}, coreBase {:#x} deviceBase {:#x}", pool.name, pool.coreBase, pool.deviceBase);
    }

    int ret = fillRegions(c_core_memory_extension_range_size);
    {
        if (ret != SCAL_SUCCESS)
        {
            return ret;
        }
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::warmupHbmTlb()
{
    if (isSimFD(m_fd)) return SCAL_SUCCESS;
    const char * hbmTLBWarmupEnable = getenv("SCAL_ENABLE_HBM_TLB_WARMUP");
    if (!hbmTLBWarmupEnable ||
        (hbmTLBWarmupEnable && (std::string(hbmTLBWarmupEnable) == std::string("1") || std::string(hbmTLBWarmupEnable) == std::string("true"))))
    {
        LOG_INFO(SCAL, "{}: writing to full HBM size {}GB starting from address {:#x}", __FUNCTION__, m_fullHbmPool.size / 1024 / 1024 / 1024.0, m_fullHbmPool.deviceBase);

        Qman::Workload workload;
        Qman::Program program;

        constexpr uint64_t CHUNK_SIZE = 2ul * 1024 * 1024 * 1024;
        uint64_t hbmLeft = m_fullHbmPool.size;
        uint64_t hbmChunkPtr = m_fullHbmPool.deviceBase;
        while (hbmLeft > 0)
        {
            uint64_t chunkSize = std::min(hbmLeft, CHUNK_SIZE);
            program.addCommand(LinDma(hbmChunkPtr, 0x0, chunkSize, 0, 1/*memset*/));
            LOG_TRACE(SCAL, "{}: writing {} to address {:#x}", __FUNCTION__, chunkSize, hbmChunkPtr);
            hbmChunkPtr += chunkSize;
            hbmLeft -= chunkSize;
        }

        if (!workload.addProgram(program, GAUDI2_QUEUE_ID_PDMA_1_0))
        {
            LOG_ERR(SCAL,"{}: fd={} addProgram failed. dmaIdx={}", __FUNCTION__, m_fd, GAUDI2_QUEUE_ID_PDMA_1_0);
            assert(0);
            return SCAL_FAILURE;
        }

        if (!submitQmanWkld(workload))
        {
            LOG_ERR(SCAL,"{}: fd={} workload.submit failed.", __FUNCTION__, m_fd);
            assert(0);
            return SCAL_FAILURE;
        }
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::resetQMANs()
{
    // iterate over the active cores and find the active qmans:
    int rc = SCAL_SUCCESS;
    Qman::Workload workload;
    for (unsigned idx =0; idx <  c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if(core)
        {
            // send QMAN program that resets the active Qmans: (upper CP program)
            Qman::Program prog;
            uint32_t cpSwitchAddr  = offsetof(gaudi2::block_qman,cp_ext_switch ) + c_dccm_to_qm_offset;
            // - forces the CP_SWITCH to 0 (twice because of hw bug)
            prog.addCommand(WReg32(cpSwitchAddr, 0));
            prog.addCommand(WReg32(cpSwitchAddr, 0));
            // - disables the shadow CI/ICI writes
            uint32_t arcCfgReg =  offsetof(gaudi2::block_qman,arc_cq_cfg0 ) + c_dccm_to_qm_offset;
            prog.addCommand(WReg32(arcCfgReg, 1));
            uint32_t cqCfgReg =  offsetof(gaudi2::block_qman, cq_cfg0[4] ) + c_dccm_to_qm_offset;
            prog.addCommand(WReg32(cqCfgReg, 1));

            // configure arbitration for both PDMA0 and PDMA1
            if ((core->qmanID == GAUDI2_QUEUE_ID_PDMA_1_0) || (core->qmanID == GAUDI2_QUEUE_ID_PDMA_0_0))
            {
                uint32_t arbCfgAddr  = offsetof(gaudi2::block_qman, arb_cfg_0) + c_dccm_to_qm_offset;
                //Enable ARB
                const unsigned arbcfg0Value = 0x110; // priority type 0 (priority ARB) , is master 1, EN=1 (enable), mst_msg_nostall = 0
                prog.addCommand(WReg32(arbCfgAddr, arbcfg0Value));

            }

            // - only for QMANs that load MME scheduler: select the bottom HBM DCCM of the scheduler and wait for message barrier.
            if (Scheduler * scheduler = core->getAs<Scheduler>(); scheduler && !scheduler->arcFarm)
            {
                uint64_t auxOffset = getCoreAuxOffset(scheduler);
                uint64_t dccmEnable = auxOffset + offsetof(gaudi2::block_qman_arc_aux, mme_arc_upper_dccm_en) + core->dccmDevAddress;
                // the 0 write here
                prog.addCommand(MsgLong(dccmEnable,0));
                prog.addCommand(Nop(true,0,0));
            }

            workload.addProgram(prog, core->qmanID, true);
        }
    }
    if (!submitQmanWkld(workload))
    {
        LOG_ERR(SCAL,"{} failed submit workload of resetQman", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }
    return rc;
}


int Scal_Gaudi2::mapLBWBlocks()
{

    // for engines - map the DCCM (update address[0] in the stream struct)
    // for ARC farm schedulers map the 2 DCCM blocks + ACP (update the 2 addresses in the stream struct)
    // for MME slave scheduler map the DCCM block + ACP. (update the 2 addresses in the stream struct to the same address)
    // iterate over the streams, update the pi pointer and set the pi local value counter to 0

    // arc farm schedulers: map both parts of the DCCM for arc farm
    for (unsigned idx =0; idx <  c_scheduler_nr; idx++)
    {
        if(Scheduler * scheduler = getCore<Scheduler>(idx))
        {
            if( scheduler->arcFarm)
            {
                scheduler->dccmHostAddress = mapLBWBlock(scheduler->dccmDevAddress, c_arc_farm_dccm_size);
                uint64_t physicalAcpAddress = ((uint64_t)scheduler->dccmDevAddress + mmARC_FARM_ARC0_ACP_ENG_BASE - mmARC_FARM_ARC0_DCCM0_BASE);
                scheduler->acpHostAddress = mapLBWBlock(physicalAcpAddress, c_acp_block_size);
            }
            else // mme slave
            {
                // MME slave has only one block 32 KBs in size that is used as a mirror to the 2X32KBs
                scheduler->dccmHostAddress = mapLBWBlock(scheduler->dccmDevAddress, c_engine_image_dccm_size);
                uint64_t physicalAcpAddress = ((uint64_t)scheduler->dccmDevAddress + mmDCORE0_MME_QM_ARC_ACP_ENG_BASE - mmDCORE0_MME_QM_ARC_DCCM_BASE);
                scheduler->acpHostAddress = mapLBWBlock(physicalAcpAddress, c_acp_block_size);
            }
            if (!scheduler->dccmHostAddress || !scheduler->acpHostAddress)
            {
                LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed scheduler core {} ",__FUNCTION__, m_fd, idx);
                return SCAL_FAILURE;
            }
        }
    }

    // map all the lower part DCCM blocks of the active engine cores.
    for (unsigned idx = c_scheduler_nr; idx < c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if (core)
        {
            core->dccmHostAddress = mapLBWBlock(core->dccmDevAddress, c_engine_image_dccm_size);
            if (!core->dccmHostAddress)
            {
                LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed core {} ",__FUNCTION__, m_fd, idx);
                return SCAL_FAILURE;
            }
        }
    }

    // map SMs to memory
    for (auto & sm : m_syncManagers)
    {
        if (sm.baseAddr && sm.map2userSpace)
        {
            sm.objsHostAddress = (volatile uint32_t*)mapLBWBlock(sm.baseAddr, SyncMgrG2::getSmMappingSize());
            if (!sm.objsHostAddress)
            {
                LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed sm objs {} ",__FUNCTION__, m_fd, sm.smIndex);
                return SCAL_FAILURE;
            }

            sm.glblHostAddress = (volatile uint32_t*)mapLBWBlock(sm.baseAddr + mmDCORE0_SYNC_MNGR_GLBL_BASE - mmDCORE0_SYNC_MNGR_OBJS_BASE, c_acp_block_size);
            if (!sm.glblHostAddress)
            {
                LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed sm glbl {} ",__FUNCTION__, m_fd, sm.smIndex);
                return SCAL_FAILURE;
            }
        }
    }

    for (auto & streamMapPair : m_streams)
    {
        auto & stream = streamMapPair.second;
        stream.localPiValue = 0;

        //points to the location of the pi register in the DCCM
        stream.pi = (uint32_t*)stream.scheduler->acpHostAddress + stream.id;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::checkCanary()
{
    // iterate over the active cores and read the canary registers. Make sure they are all 0, return error otherwise
    int rc = SCAL_SUCCESS;
    for (unsigned idx =0; idx <  c_scheduler_nr; idx++)
    {
        Scheduler * core = getCore<Scheduler>(idx);
        if(core)
        {
            uint32_t canaryValue = 1;
            uint64_t canaryAddress = offsetof(sched_registers_t, canary) + (uint64_t)(core->dccmHostAddress);
            readLbwMem( (void*)&canaryValue, (volatile void*) canaryAddress, sizeof(uint32_t));
            if(canaryValue !=0)
            {
                LOG_ERR(SCAL,"{}: canary {} !=0 core id {} ", __FUNCTION__, canaryValue, core->cpuId);
                assert(0);
                rc = SCAL_FAILURE;
            }
        }
    }
    for (unsigned idx =c_scheduler_nr; idx < c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if(core)
        {
            uint32_t canaryValue = 1;
            uint64_t canaryAddress = offsetof(engine_arc_reg_t, canary) + (uint64_t)(core->dccmHostAddress);
            readLbwMem( (void*)&canaryValue, (volatile void*) canaryAddress, sizeof(canaryValue));
            if (canaryValue !=0)
            {
                LOG_ERR(SCAL,"{}: canary {} !=0 core id {} ", __FUNCTION__, canaryValue, core->cpuId);
                assert(0);
                rc = SCAL_FAILURE;
            }
        }
    }

    return rc;
}

int Scal_Gaudi2::configureSMs()
{
    // configure the QMANs
    int ret;

    // allocates memory for the completion queues
    ret = allocateCompletionQueues();
    if (ret != SCAL_SUCCESS) return ret;

    // configure the active CQs
    ret = configureCQs();
    if (ret != SCAL_SUCCESS) return ret;

    // Configure Special monitors
    ret = configureMonitors();
    if (ret != SCAL_SUCCESS) return ret;

    return SCAL_SUCCESS;

}

int Scal_Gaudi2::configureTdrCq(Qman::Program & prog, CompletionGroup &cg)
{
    CompQTdr& compQTdr = cg.compQTdr;
    if (!compQTdr.enabled) return SCAL_SUCCESS;

    // Clear the SOB
    uint64_t syncObjAddress = SobG2::getAddr(compQTdr.sosPool->smBaseAddr, compQTdr.sos);

    uint32_t reg = SobG2::buildEmpty();

    prog.addCommand(MsgLong(syncObjAddress, reg)); // clear the SOB

    LOG_INFO_F(SCAL, "tdr sm {} sos {} writing {:#x} to addr {:#x}", compQTdr.sosPool->smIndex, compQTdr.sos, reg, syncObjAddress);

    // Configure the CQ
    auto     cqIdx                               = compQTdr.cqIdx;
    uint32_t cq_offset_64                        = (cqIdx + (cg.syncManager->dcoreIndex *c_cq_ctrs_in_dcore)) * sizeof(uint64_t);
    uint64_t cqCompletionQueueCountersDeviceAddr = cq_offset_64 + m_completionQueueCountersDeviceAddr;

    compQTdr.enginesCtr = m_completionQueueCounters + cq_offset_64/ sizeof(uint64_t) ;
    uint64_t smGlobalBase = mmDCORE0_SYNC_MNGR_GLBL_BASE + cg.syncManager->dcoreIndex * (mmDCORE1_SYNC_MNGR_GLBL_BASE - mmDCORE0_SYNC_MNGR_GLBL_BASE);

    // Set the address of the completion group counter (LSB)
    uint64_t cq_base_addr_l = smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_base_addr_l[cqIdx]);
    prog.addCommand(
        MsgLong(cq_base_addr_l, lower_32_bits(cqCompletionQueueCountersDeviceAddr)));

    // Set the address of the completion group counter (MSB)
    uint64_t cq_base_addr_h = smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_base_addr_h[cqIdx]);
    prog.addCommand(
        MsgLong(cq_base_addr_h, upper_32_bits(cqCompletionQueueCountersDeviceAddr)));

    // set the size of the counter (to 8 bytes)
    uint64_t cq_size_log2addr = smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_size_log2[cqIdx]);
    prog.addCommand(
        MsgLong(cq_size_log2addr, c_cq_size_log2));

    // Set it to increment
    prog.addCommand(
        MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_inc_mode[cqIdx]), 0x0));

    prog.addCommand(
        MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_inc_mode[cqIdx]), 0x1));

    // Note, unlike the "regular' cq, for the dtr-cq, we are not setting an interrupt (lbw_addr_l, lbw_addr_h, lbw_data), they remain 0

    LOG_INFO_F(SCAL, "tdr, cqIdx {} config cq dcore {} cq_base_addr h/l {:#x} {:#x} cqCompletionQueueCountersDeviceAddr {:#x} ctr {:#x}",
               cqIdx, cg.syncManager->dcoreIndex, cq_base_addr_h, cq_base_addr_l, cqCompletionQueueCountersDeviceAddr, TO64(compQTdr.enginesCtr));

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::configureCQs()
{
    // configure the active CQs.
    // For each CQ - update the isrIdx and Counter in the Completion Group Struct
    // send QMAN programs to configure the active CQs:
    // - set the PQ mode
    // - set the PQ address
    // - set the PQ ISR interupt service routine
    // - set the PQ size to 1
    std::map<unsigned, Qman::Program> qid2prog;
//    Qman::Program prog;
    for (auto& completionGroupIter : m_completionGroups)
    {
        auto& completionGroup = completionGroupIter.second;
        Qman::Program & prog = qid2prog[completionGroup.qmanID];

        auto cqIdx = completionGroup.cqIdx;
        uint32_t cq_offset_64 = (cqIdx + (completionGroup.syncManager->dcoreIndex *c_cq_ctrs_in_dcore)) * sizeof(uint64_t);
        uint64_t cqCompletionQueueCountersDeviceAddr = cq_offset_64 + m_completionQueueCountersDeviceAddr;

        completionGroup.pCounter = m_completionQueueCounters + cq_offset_64/ sizeof(uint64_t) ;
        uint64_t smGlobalBase = mmDCORE0_SYNC_MNGR_GLBL_BASE + completionGroup.syncManager->dcoreIndex * (mmDCORE1_SYNC_MNGR_GLBL_BASE - mmDCORE0_SYNC_MNGR_GLBL_BASE);

        prog.addCommand(
            MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_base_addr_l[cqIdx]), lower_32_bits(cqCompletionQueueCountersDeviceAddr)));

        prog.addCommand(
            MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_base_addr_h[cqIdx]), upper_32_bits(cqCompletionQueueCountersDeviceAddr)));

        prog.addCommand(
            MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_size_log2[cqIdx]), c_cq_size_log2));

        // Configure CQ LBW Address
        // writing to MSIX_TABLE as LKD is expected to allow "user-initiated interrupts" only from this specific range of 256-511
        // Use WA for issue SW-93019
        uint64_t msix_db_reg = RESERVED_VA_FOR_VIRTUAL_MSIX_DOORBELL_START;

        if (completionGroup.isrIdx != scal_illegal_index)// SW-82256
        {
            prog.addCommand(
                MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, lbw_addr_l[cqIdx]), lower_32_bits(msix_db_reg)));

            prog.addCommand(
                MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, lbw_addr_h[cqIdx]), upper_32_bits(msix_db_reg)));

            prog.addCommand(
                MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, lbw_data[cqIdx]), completionGroup.isrIdx));
        }

        prog.addCommand(
            MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_inc_mode[cqIdx]), 0x0)); // set 32bit mode in order to zero the cq

        prog.addCommand(
            MsgLong(smGlobalBase + varoffsetof(gaudi2::block_sob_glbl, cq_inc_mode[cqIdx]), 0x1)); // set 64bit mode

        if (completionGroup.compQTdr.enabled)
        {
            configureTdrCq(prog, completionGroup);
        }

        LOG_DEBUG(SCAL, "configureCQs() cqIdx={} ctr={:#x} deviceAddress={:#x} msix_db_reg={:#x} isrIdx={:#x}",
                cqIdx, (uint64_t)completionGroup.pCounter, cqCompletionQueueCountersDeviceAddr, msix_db_reg,
                completionGroup.isrIdx);
    }

    Qman::Workload workload;
    for (auto & qidProgPair : qid2prog)
    {
        workload.addProgram(qidProgPair.second, qidProgPair.first);
    }

    if (!submitQmanWkld(workload))
    {
        LOG_ERR(SCAL,"{} failed submit workload of configureCQs", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}
void Scal_Gaudi2::configureMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData)
{
    configureOneMonitor(prog, monIdx, smBase, configValue, payloadAddress, payloadData);
}

void Scal_Gaudi2::configureOneMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData)
{
    // send QMAN command MsgLong to config this monitor
    MonitorG2 monitor(smBase, monIdx);
    Monitor::ConfInfo confInfo{.payloadAddr = payloadAddress, .payloadData = payloadData, .config = configValue};
    monitor.configure(prog, confInfo);

    LOG_DEBUG(SCAL, "configureOneMonitor() smBase={:#x} monIdx={} configValue={:#x} payloadAddress={:#x} payloadData={:#x}",
              smBase, monIdx, configValue, payloadAddress, payloadData);
}
// clang-format off
/*

‘In Order Completion’
      The purpose of this feature is to ensue that the completion group notification issued in order even when the workloads themselves complete out of order.
      (Network workloads can potentially complete out of order whenever the communication group changes).
      To force this order, the groups monitors are programmed to issue an additional write
      that increments the next sync object in the group.
      With this write, the expiration messages are guaranteed to leave the sync manager only after the all the previous workloads are completed.
      SCAL is expected to configure this new write and point it to the first sync objects.
      The FW is then expected to update the address of the writes just before rearming the monitors.

      SCAL Responsibilities:
      - Parse the json file and for each completion group conclude if the in-order completion feature should be enabled.
      o A new mandatory Boolean field will be added to the completion queue records in the json file – “force_order”.
      o SCAL should report an error whenever the “force_order” field is missing or if its type is not a Boolean type.

      o If the monitor depth is set to 1, SCAL should ignore the “force_order” knob and disable the in-order completion feature.
        Otherwise, SCAL should enable the in-order completion when “force_order” is set to true.
      - When the In-Order-Completion feature is enabled.
            o SCAL should configure the master monitors to issue an additional message upon their expiration.
            ? The new message should increment the value of the next SO in the group by 1.
            • The message payload should be set to SO atomic inc (0x80000001).
            • The message address should be set to point the next SO in the group:
                  (e.g. if we have monitor depth=2,  mon0.3 (the new 4th monitor we now add) payload addr = &so1.  mon1.3 payload addr  = &so2 etc)
            o The 2nd SO in the first monitor.
            o The 3rd SO in the second monitor.
            o ….
      Note: If the “monitor_depth” equals the “so_depth”, the address of the new message in the last monitor should point the first SO in the group (in a cyclic manner).
      o In the Completion Group configuration blob, SCAL should specify the offset from the address of the first monitor ARM register to the MON_PAY_ADDRL register of the new increment message.
               e.g  if we set it  mon0.3 payload addr = &so1.
                    and once this job is done and FW needs to changed the addr
                    mon0.3  payload addr = &so3.  ---> so our delta is   &so1-&so3 (because he wants this negative ??)
      ? The FW will use this offset to update the address of the increment message whenever the monitor expires.
      ? The offset is expected to be a negative 32 bits value – @Rakesh Ughreja, please add it to the config blob structure. (Negative DELTA ??)
      ? When the in-order-completion feature is disabled (or if the scheduler is not the completion group’s master – see below) the offset should be set to 0.
      o As part of the Sync Manager configuration, SCAL should set the value of the first SO in the completion group to “1” in all the groups in which the in-order completion feature is enabled.
      - SCAL will add a new field to the completion group info structure to allow its users to tell whether the in-order completion is enabled for a particular completion group.


‘Distributed Completion Group’
          This feature is intended to allow multiple streams from different schedulers to signal to the same completion group.
          The idea behind it is that the monitors that watch the completion group SOs fire multiple expiration messages to the schedulers
          that use the completion group.
          For each completion group, the json file can specify up to 5 slave schedulers (in addition to the master scheduler).
          The monitor expiration messages are fired also to the DCCM queues of the slave schedulers.
          Handling of the expiration message in the slave schedulers is limited only to incrementing the completion queue counter and un-blocking any stream that is waiting for it.
          The slave schedulers do not rearm the monitors, nor they decrement the sync object.
          Note that from the user perspective all the schedulers are equal. The user is not aware of the concept of master and slaves.

          SCAL Responsibilities:
          - Parse the json file and get the list of slave schedulers for each completion group:
          o The “scheduler” knob in the “completion_queues” record will be replaced by an array of schedulers. (knob name: “schedulers”)

          o SCAL should return an error if one of the following errors occur:
                ? The schedulers knob is missing from the record.
                ? The schedulers knob is not an array.
                ? The schedulers knob array is empty.
                ? An entry in the array is not a string or if it’s not a valid scheduler name.
          o In "completion_queues" : "schedulers" config in the json (just above the force_order config)
            The first entry in the (schedulers) array is the master. The rest of the schedulers (if any) are the slaves.
          - SCAL should configure the completion queue in all the schedulers including the slaves.
          o A new Boolean configuration will be added to the config blob to tell the scheduler if the scheduler is the master of each completion queue. @Rakesh Ughreja – please add to the config blob.
          o Note that it’s possible that in each scheduler the completion group will get a different index.
          - SCAL should configure the monitors to fire the expiration messages also to the slave schedulers.
          o SCAL should support up to 5 slaves with and without in-order completion as shown in the table below.


                    no slaves in-order disabled          no slaves in-order enabled                1 slave in-order disabled            1 slave in-order enabled            2 slaves in-order disabled         2 slaves in-order enabled
                    +------------------------------+--------------------------------------+---------------------------------------+------------------------------------+-----------------------------------+-------------------------------+
          Monitor 0 |     Completion Queue HW      |        Completion Queue HW           |           Completion Queue HW         |        Completion Queue HW         |        Completion Queue HW        |       Completion Queue HW     |
          Monitor 1 |     Long Sync Object         |        Long Sync Object              |           Long Sync Object            |        Long Sync Object            |        Long Sync Object           |       Long Sync Object        |
          Monitor 2 |     Master DCCM queue        |        Inc Next Sync Object          |           Slave 0 DCCM queue          |        Slave 0 DCCM queue          |        Slave 0 DCCM queue         |       Slave 0 DCCM queue      |
          Monitor 3 |                              |        Master DCCM queue             |           Master DCCM queue           |        GTE0 PTR to Mon4 ARM        |        GTE0 PTR to Mon4 ARM       |       GTE0 PTR to Mon4 ARM    |
          Monitor 4 |                              |                                      |                                       |        Inc Next Sync Object        |        Slave 1 DCCM queue         |       Slave 1 DCCM queue      |
          Monitor 5 |                              |                                      |                                       |        Master DCCM queue           |        Master DCCM queue          |       Inc Next Sync Object    |
          Monitor 6 |                              |                                      |                                       |                                    |                                   |       Master DCCM queue       |
                    +------------------------------+--------------------------------------+---------------------------------------+------------------------------------+-----------------------------------+-------------------------------+
                    | 3 slaves in-order disabled   |    3 slaves in-order enabled         |       4 slaves in-order disabled      |       4 slaves in-order enabled    |      5 slaves in-order disabled   |    5 slaves in-order enabled  |
                    +------------------------------+--------------------------------------+---------------------------------------+------------------------------------+-----------------------------------+-------------------------------+
          Monitor 0 |     Completion Queue HW      |        Completion Queue HW           |           Completion Queue HW         |        Completion Queue HW         |        Completion Queue HW        |       Completion Queue HW     |
          Monitor 1 |     Long Sync Object         |        Long Sync Object              |           Long Sync Object            |        Long Sync Object            |        Long Sync Object           |       Long Sync Object        |
          Monitor 2 |     Slave 0 DCCM queue       |        Slave 0 DCCM queue            |           Slave 0 DCCM queue          |        GTE0 PTR to Mon4 ARM        |        Slave 0 DCCM queue         |       GTE0 PTR to Mon4 ARM    |
          Monitor 3 |     GTE0 PTR to Mon4 ARM     |        GTE0 PTR to Mon4 ARM          |           GTE0 PTR to Mon4 ARM        |        GTE0 PTR to Mon8 ARM        |        GTE0 PTR to Mon4 ARM       |       GTE0 PTR to Mon8 ARM    |
          Monitor 4 |     Slave 1 DCCM queue       |        Slave 1 DCCM queue            |           Slave 1 DCCM queue          |        Slave 0 DCCM queue          |        Slave 1 DCCM queue         |       Slave 0 DCCM queue      |
          Monitor 5 |     Slave 2 DCCM queue       |        Slave 2 DCCM queue            |           Slave 2 DCCM queue          |        Slave 1 DCCM queue          |        Slave 2 DCCM queue         |       Slave 1 DCCM queue      |
          Monitor 6 |     Master DCCM queue        |        Inc Next Sync Object          |           Slave 3 DCCM queue          |        Slave 2 DCCM queue          |        Slave 3 DCCM queue         |       Slave 2 DCCM queue      |
          Monitor 7 |                              |        Master DCCM queue             |           Master DCCM queue           |        Inc Next Sync Object        |        GTE0 PTR to Mon8 ARM       |       Inc Next Sync Object    |
          Monitor 8 |                              |                                      |                                       |        Slave 3 DCCM queue          |        Slave 4 DCCM queue         |       Slave 3 DCCM queue      |
          Monitor 9 |                              |                                      |                                       |        Master DCCM queue           |        Master DCCM queue          |       Slave 4 DCCM queue      |
          Monitor 10|                              |                                      |                                       |                                    |                                   |       Master DCCM queue       |
                    +------------------------------+--------------------------------------+---------------------------------------+------------------------------------+-----------------------------------+-------------------------------+

          o monitors #0/#4/#8 (red font) should be configured in the SM as master monitors with multiple expiration messages each.
          o To avoid race conditions when rearming the monitors, the order of the messages in each of the above cases should be as specified in the table above.
          o SCAL should allocate the monitors of each group according to the table above. In other words, the number of monitors depends on the number of schedulers and the in-order feature.
          - The Completion Group Info struct should be changed to return all the scheduler and the indices of the group inside each scheduler.

Some clarifications:

    Q:  "In Order Completion"  - how does writing +1 to next so achieves that?
    A:  if we have depth=2 (e.g 2 monitors that look at so.0 and so.1 - how do we know if the 2 events completed and not just 1 of them?)
        the solution:  mon0 will add another inc by 1 msg to the next so (e.g the one the mon1 is looking at)
                       and mon1 will expire only when so1 will reach V (the intended value) + 1 (the extra +1 from mon0)

        Explain: "In the Completion Group configuration blob, SCAL should specify the offset from the address of the first monitor ARM register to the MON_PAY_ADDRL register of the new increment message|
        A:
           since the 1st config is ours, but as the 1st job is done, FW needs to config mon0,mon1 to look at the next sos (so2,so3)
           so it needs to update the payload address of mon0.3  (e.g. the slave monitor that fires the "new" inc_by_1 payload)
           it does that by adding the "offset" e.g.  if initially mon0.3 payload address = &so1
                                                     the next batch should have mon0.3 payload address = &so3  so the offset is &so1 - &so3  (or vice versus)

    Q: "Distributed Completion Group"

    GTE0 ->  need to triger the next "demi master", so it configures it the same as monitor 0, but uses >=,   and it does not ARM it
             so when mon0 expires, GTE0 will arm the demi monitor that will immediately expires and send the extra messages
             (all this since we have a maximum of 4 messages that a single master monitor can send)


  */
// clang-format on
void Scal_Gaudi2::AddIncNextSyncObjectMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx, unsigned numWrites)
{
    // The new message should increment the value of the next SO in the group by 1
    // The message payload should be set to SO atomic inc (0x80000001).
    // The message address should be set to point the next SO in the group

    uint32_t confVal = MonitorG2::buildConfVal(soIdx, numWrites, CqEn::off, LongSobEn::off, LbwEn::off);

    unsigned nextSoIndex    = cq->sosBase + ((soIdx - cq->sosBase + 1) % cq->sosNum);
    uint64_t payloadAddress = SobG2::getAddr(smBase, nextSoIndex); // address of next so
    uint32_t payloadData    = 0x80000001;
    LOG_DEBUG(SCAL, "AddIncNextSyncObjectMonitor() cq idx {} config mon {} to inc sob {}", cq->cqIdx, monIdx, nextSoIndex);
    configureOneMonitor(prog, monIdx, smBase, confVal, payloadAddress, payloadData);
    if (cq->nextSyncObjectMonitorId == -1)
    {
        cq->nextSyncObjectMonitorId = monIdx - cq->monBase;;
    }
}

// The monitor expiration messages are fired also to the DCCM queues of the slave schedulers.
// Handling of the expiration message in the slave schedulers is limited only to incrementing the completion queue counter
// and un-blocking any stream that is waiting for it.
// The slave schedulers do not rearm the monitors, nor they decrement the sync object.
void Scal_Gaudi2::AddSlaveXDccmQueueMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned slaveIndex, unsigned numWrites)
{

    // Address of scheduler DCCM messageQ,
    // value according to struct(Mon info of the master, so scheduler can "rearm" the master monitor)
    uint32_t confVal = MonitorG2::buildConfVal(0, numWrites, CqEn::off, LongSobEn::off, LbwEn::off);
    // the payload address: mmARC_FARM_ARC0_AUX_DCCM_QUEUE_PUSH_REG_0
    // The 64 bit address to write the completion message to in case CQ_EN=0.
    const Scheduler* scheduler   = cq->slaveSchedulers[slaveIndex].scheduler;
    uint64_t    payloadAddress   = scheduler->dccmMessageQueueDevAddress;// scheduler->dccmDevAddress + getCoreAuxOffset(scheduler) + (mmARC_FARM_ARC0_AUX_DCCM_QUEUE_PUSH_REG_0 & 0xFFF);
    int         comp_group_index = cq->slaveSchedulers[slaveIndex].idxInScheduler;

    // payload data
    struct sched_mon_exp_comp_fence_t pd;
    memset(&pd, 0x0, sizeof(uint32_t));
    pd.opcode           = MON_EXP_COMP_FENCE_UPDATE;
    pd.comp_group_index = comp_group_index;
    pd.mon_id           = monIdx + (cq->syncManager->dcoreIndex * c_max_monitors_per_sync_manager);
    pd.mon_sm_id        = cq->syncManager->smIndex;
    pd.mon_index        = 0;
    uint32_t pdRaw      = 0;
    memcpy(&pdRaw, &pd, sizeof(uint32_t));
    configureOneMonitor(prog, monIdx, smBase, confVal, payloadAddress, pdRaw);
}

void Scal_Gaudi2::AddTriggerDemiMasterMonitor(Qman::Program& prog, uint64_t smBase, unsigned monIdx, unsigned soIdx, unsigned monIdxToTrigger)
{
    /*
        GTE0 ->  need to trigger the next "demi master", so it configures it the same as monitor 0, but uses >=,   and it does not ARM it
                 so when mon0 expires, GTE0 will arm the demi monitor that will immediately expires and send the extra messages
                 (all this since we have a maximum of 4 messages that a single master monitor can send)
    */
    uint32_t confVal = MonitorG2::buildConfVal(soIdx, 0, CqEn::off, LongSobEn::off, LbwEn::off);

    // address of monitor to arm
    uint64_t payloadAddress = MonitorG2(smBase, monIdxToTrigger).getRegsAddr().arm;

    // since we want the (monIdxToTrigger) monitor to fire once it is armed
    // no matter what SOB id it is looking at, and no matter what value this SOB has
    // will just use  >= 0 as its trigger. This should always be true
    // so once the (monIdxToTrigger) monitor is armed, it will immediately fire

    // arm with op >= 0  (which should always be true)
    uint32_t ma = MonitorG2::buildArmVal(soIdx, 0);

    configureOneMonitor(prog, monIdx, smBase, confVal, payloadAddress, ma);
    LOG_DEBUG(SCAL, "AddTriggerDemiMasterMonitor() arm mon {} to trigger mon {} config={:#x} payload={:#x}", monIdx, monIdxToTrigger, confVal, ma);
}

unsigned Scal_Gaudi2::AddCompletionGroupSupportForHCL(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx)
{
    //
    // Mon 0,1, and last are always the same (the original 3 ...)
    // and there are configured by the caller
    // so  here we config the monitors in between them according to the table above
    //

    // notes
    //    * each "demi master monitor" should be configured as the master, but using >= , and without arming it
    //      the GTE0 monitor will fire and arm it
    unsigned NumAdded              = (unsigned)cq->force_order + cq->slaveSchedulers.size(); // Num monitors to add (without the extra triggering monitors)
    unsigned NumTriggeringMonitors = 0;
    // when called, monIdx should be firstMonIdx + 2
    switch (NumAdded)
    {
    case 1: // either force_order or 1 slave
        if (cq->force_order)
            AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx, soIdx, 0);
        else
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx, 0, 0); // Slave 0 DCCM queue
        break;
    case 2:                                                                       // either 1 slave + 1 force-order    OR   2 slaves
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx, 0, 0);                // Slave 0 DCCM queue
        AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 1, soIdx, monIdx + 2); // GTE0 PTR to Mon4 ARM
        NumTriggeringMonitors = 1;
        if (cq->force_order)
        {
            AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx + 2, soIdx, 1);
        }
        else
        {
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 2, 1, 1); // Slave 1 DCCM queue
        }
        break;
    case 3:                                                                       // either 2 slaves + 1 force-order    OR   3 slaves
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx, 0, 0);                // Slave 0 DCCM queue
        AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 1, soIdx, monIdx + 2); // GTE0 PTR to Mon4 ARM
        NumTriggeringMonitors = 1;
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 2, 1, 2); // Slave 1 DCCM queue
        if (cq->force_order)
        {
            AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx + 3, soIdx, 0);
        }
        else
        {
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 3, 2, 0); // Slave 2 DCCM queue
        }
        break;
    case 4:                                                                       // either 3 slaves + 1 force-order    OR   4 slaves
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx, 0, 0);                // Slave 0 DCCM queue
        AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 1, soIdx, monIdx + 2); // GTE0 PTR to Mon4 ARM
        NumTriggeringMonitors = 1;
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 2, 1, 3); // Slave 1 DCCM queue
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 3, 2, 0); // Slave 2 DCCM queue
        if (cq->force_order)
        {
            AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx + 4, soIdx, 0);
        }
        else
        {
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 4, 3, 0); // Slave 3 DCCM queue
        }
        break;
    case 5: // either 4 slaves + 1 force-order    OR   5 slaves
        if (cq->force_order)
        {
            AddTriggerDemiMasterMonitor(prog, smBase, monIdx, soIdx, monIdx + 2);     // GTE0 PTR to Mon4 ARM
            AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 1, soIdx, monIdx + 6); // GTE0 PTR to Mon8 ARM
            NumTriggeringMonitors = 2;
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 2, 0, 3); // Slave 0 DCCM queue
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 3, 1, 0); // Slave 1 DCCM queue
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 4, 2, 0); // Slave 2 DCCM queue
            AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx + 5, soIdx, 0);
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 6, 3, 1); // Slave 3 DCCM queue
        }
        else
        {
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx, 0, 0);                // Slave 0 DCCM queue
            AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 1, soIdx, monIdx + 2); // GTE0 PTR to Mon4 ARM
            NumTriggeringMonitors = 2;
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 2, 1, 3);            // Slave 1 DCCM queue
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 3, 2, 0);            // Slave 2 DCCM queue
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 4, 3, 0);            // Slave 3 DCCM queue
            AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 5, soIdx, monIdx + 6); // GTE0 PTR to Mon8 ARM
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 6, 4, 1);            // Slave 4 DCCM queue
        }
        break;
    case 6:                                                                       // 5 slaves + 1 force-order
        AddTriggerDemiMasterMonitor(prog, smBase, monIdx, soIdx, monIdx + 2);     // GTE0 PTR to Mon4 ARM
        AddTriggerDemiMasterMonitor(prog, smBase, monIdx + 1, soIdx, monIdx + 6); // GTE0 PTR to Mon8 ARM
        NumTriggeringMonitors = 2;
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 2, 0, 3); // Slave 0 DCCM queue
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 3, 1, 0); // Slave 1 DCCM queue
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 4, 2, 0); // Slave 2 DCCM queue
        AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx + 5, soIdx, 0);
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 6, 3, 2); // Slave 3 DCCM queue
        AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + 7, 4, 0); // Slave 4 DCCM queue
        break;
    default:
        assert(0);
    };
    //
    return monIdx + NumAdded + NumTriggeringMonitors; // should be the index of the last monitor to add
}

void Scal_Gaudi2::configureTdrMon(Qman::Program & prog, const CompletionGroup *cg)
{
    const CompQTdr& compQTdr = cg->compQTdr;

    uint64_t smBase     = cg->compQTdr.monPool->smBaseAddr;

    // Monitor 1) Dec sob
    uint64_t addr1 = SobG2::getAddr(compQTdr.sosPool->smBaseAddr, compQTdr.sos);

    // If Op=1 & Long=0, an atomic add is performed such that SOB[14:0]+= Signed[15:0]. As the incoming data is S16, one can perform a subtraction of the monitor.
    uint32_t data1 = SobG2::buildVal(-1, SobLongSobEn::off, SobOp::inc);

    // Monitor 2) Rearm monitor
    uint64_t addr2 = MonitorG2(compQTdr.monPool->smBaseAddr, compQTdr.monitor).getRegsAddr().arm;

    uint32_t data2 = MonitorG2::buildArmVal(cg->compQTdr.sos, 1);

    // Monitor 3) Inc Cq
    uint64_t addr3 = compQTdr.cqIdx;
    uint32_t data3 = 1; // Inc Cq

    // Monitor config
    uint32_t confVal = MonitorG2::buildConfVal(cg->compQTdr.sos, 2, CqEn::off, LongSobEn::off, LbwEn::off);

    //    Field Name                 |  Bits |   Comments
    //------------------------------------------------------------------------------------------
    //    LONG_SOB MON_CONFIG        |    [0]|Indicates that the monitor monitors 60bit SOBs
    //    CQ EN MON_CONFIG           |    [4]|Indicates the monitor is associated with a completion queue
    //    NUM_WRITES MON_CONFIG      | [5..6]|“0”: single write, “1”: 2 writes, “2”: 3 writes, “3”: 4 writes.
    //    LBW_EN MON_CONFIG          |    [8]|Indicates that a LBW message should be sent post a write to the CQ. Relevant only if CQ_EN is set
    //    LONG_HIGH_GROUP MON_CONFIG |   [31]|Defined which SOB’s would be used for 60xbit count “0”: Lower 4xSOB’s “1”: Upper 4xSOB’s
    //    SID_MSB MON_CONFIG         |[19:16]|Extended SID to monitor groups 256-1023

    configureOneMonitor(prog, compQTdr.monitor + 0, smBase, confVal, addr1, data1);
    configureOneMonitor(prog, compQTdr.monitor + 1, smBase, confVal, addr2, data2);

    confVal = MonitorG2::buildConfVal(0, 0, CqEn::on, LongSobEn::off, LbwEn::off);
    configureOneMonitor(prog, compQTdr.monitor + 2, smBase, confVal, addr3, data3);

    MonitorG2 monitor(smBase, compQTdr.monitor);
    uint64_t addrArm = monitor.getRegsAddr().arm;
    LOG_INFO_F(SCAL, "tdr arm {} addr {:#x} data {:#x}", compQTdr.monitor, addrArm, data2);
    prog.addCommand(MsgLong(addrArm, data2));

}

void Scal_Gaudi2::configSfgSyncHierarchy(Qman::Program & prog, const CompletionGroup *cg)
{
    Scal::configSfgSyncHierarchy<sfgMonitorsHierarchyMetaDataG2>(prog, cg, c_maximum_messages_per_monitor);

}
void Scal_Gaudi2::configFenceMonitorCounterSMs(Qman::Program & prog, const CompletionGroup *cg)
{
    const unsigned curMonIdx   = cg->monBase + cg->monNum;
    const unsigned smIdx       = cg->syncManager->smIndex;
    const unsigned sobIdx      = cg->sosBase;
    const unsigned cqIdx       = cg->cqIdx;
    const uint64_t smBaseAddr  = cg->syncManager->baseAddr;
    const bool     hasInterrupts = cg->isrIdx != scal_illegal_index;
    const uint32_t mon1 = curMonIdx + 0; // Master monitor for dec sob
    const uint32_t mon2 = curMonIdx + 1;
    const uint32_t mon3 = curMonIdx + 2;

    LOG_INFO(SCAL,"{}: =========================== Fence Counter Configuration =============================", __FUNCTION__);
    LOG_INFO(SCAL,"{}: Config master mon SM{}_MON_{} with 3 messages: dec SOB SM{}_SOB_{}, inc cq CQ{}, ReArm monitor SM{}_MON_{}",
            __FUNCTION__, smIdx, mon1, smIdx, sobIdx, cqIdx, smIdx, mon1);

    /* ====================================================================================== */
    /*                                  Create a monitor group                                */
    /* ====================================================================================== */

    // Need 3 writes (dec sob, inc CQ, Arm master monitor)
    uint32_t confVal = MonitorG2::buildConfVal(sobIdx, 2, CqEn::off, LongSobEn::off, LbwEn::off);

    /* ====================================================================================== */
    /*                              Add massages to monitor groups                            */
    /* ====================================================================================== */

    // msg0:   decrement by 1
    uint32_t decSobPayloadData = SobG2::buildVal(-1, SobLongSobEn::off, SobOp::inc);
    LOG_DEBUG(SCAL,"{}: Configure SM{}_MON_{} to dec sob SM{}_SOB_{} by 1. payload: {:#x}", __FUNCTION__, smIdx, mon1, smIdx, sobIdx, decSobPayloadData);
    configureOneMonitor(prog, mon1, smBaseAddr, confVal, SobG2::getAddr(smBaseAddr, sobIdx), decSobPayloadData);

    // msg1:     ReArm master monitor
    uint32_t armMonPayloadData = MonitorG2::buildArmVal(sobIdx, 1);

    // Rearm self
    uint64_t masterMonAddr = MonitorG2(smBaseAddr, mon1).getRegsAddr().arm;
    LOG_DEBUG(SCAL,"{}: Configure SM{}_MON_{} to rearm self", __FUNCTION__, smIdx, mon3);
    configureOneMonitor(prog, mon3, smBaseAddr, confVal, masterMonAddr, armMonPayloadData);

    // msg2:    Increment CQ by 1
    LbwEn lbwEn = hasInterrupts ? LbwEn::on : LbwEn::off;
    confVal     = MonitorG2::buildConfVal(sobIdx, 2, CqEn::on, LongSobEn::off, lbwEn);
    LOG_DEBUG(SCAL,"{}: Configure SM{}_MON_{} to inc cg's CQ {}", __FUNCTION__, smIdx, mon2, cqIdx);
    configureOneMonitor(prog, mon2, smBaseAddr, confVal, (uint64_t)cqIdx, 0x1);

    LOG_DEBUG(SCAL,"{}: Arm master SM{}_MON_{} with payload data {:#x}, SM{}_SOB_{}",
              __FUNCTION__, smIdx, mon1, armMonPayloadData, smIdx, sobIdx);
    prog.addCommand(MsgLong(masterMonAddr, armMonPayloadData));

}

int Scal_Gaudi2::configureMonitors()
{
    std::map<unsigned, Qman::Program> qid2prog;

    for (unsigned dcoreID=0; dcoreID < c_sync_managers_nr; dcoreID++)
    {
        for (auto cq : m_syncManagers[dcoreID].completionGroups)
        {
            Qman::Program & prog = qid2prog[cq->qmanID];
            unsigned mon_depth = cq->monNum / cq->actualNumberOfMonitors;
            uint64_t longSoSmBase = cq->longSosPool->smBaseAddr;
            uint64_t smBase = cq->monitorsPool->smBaseAddr;
            uint64_t syncObjAddress = SobG2::getAddr(longSoSmBase, cq->longSoIndex);
            prog.addCommand(MsgLong(syncObjAddress, 0x01000000)); // set to 1 in long SO mode (bit 24)
            LOG_DEBUG(SCAL, "reset long sync obj {} at address {:#x}", cq->longSoIndex, syncObjAddress);
            const bool isFenceCounterConnected = !cq->fenceCounterName.empty();
            for (unsigned mIdx = 0;mIdx < mon_depth && isFenceCounterConnected == false; mIdx++)
            {
                unsigned soIdx = cq->sosBase + mIdx;
                unsigned soGroupIdx = soIdx >> 3;
                 // each user monitor is actually a consecutive set of size 3 (c_completion_queue_monitors_set_size)
                //     (when using HCL config, slave schedulers and/or force-order, it can be more than 3, up to 11)
                //


                //
                //  1st monitor - the master -
                //
                //
                // ( num_writes=2) + longso configuration ( address, data) :
                // Master Monitor + inc long SO ----- MSG_NR = 2 (3 messages)

                unsigned monIdx = cq->monBase + mIdx * cq->actualNumberOfMonitors;
                LOG_INFO(SCAL, "configureMonitors() cq {} {} smIndex={} monIdx={} soIdx={} {} actualNumberOfMonitors={} cq->longSoIndex={}",
                         cq->cqIdx, cq->name.c_str(), cq->sosPool->smIndex, monIdx,
                         soIdx, soIdx + (c_max_sos_per_sync_manager * cq->syncManager->dcoreIndex),
                         cq->actualNumberOfMonitors, cq->longSoIndex);
                unsigned firstMonIdx = monIdx;
                if (cq->force_order && mIdx == 0)
                {
                    //  As part of the Sync Manager configuration, SCAL should set the value of the first SO in the completion group to “1”
                    //  in all the groups in which the in-order completion feature is enabled.

                    prog.addCommand(MsgLong(SobG2::getAddr(smBase, soIdx), 1));
                    LOG_INFO(SCAL, "force-order is true so setting SOB {} to 1", soIdx);
                }

                unsigned numWrites = std::min(cq->actualNumberOfMonitors, c_maximum_messages_per_monitor);
                uint32_t confVal = MonitorG2::buildConfVal(soIdx, numWrites -1 , CqEn::off, LongSobEn::off, LbwEn::off);
                // mc.long_high_group = 0 Not relevant for short monitor

                // the payload address: Address of the long SO + ID * 4SOs * 8 bytes
                //  e.g the address inside the sob array (8K of them)
                //   of the 1st so that belongs to this long SO
                // Field Name| Bits |   Comments
                //-----------|------|-------------------------------------------------
                //           |      | If Op=0 & Long=0, the SOB is written with bits [14:0]
                //           |      | If Op=0 & Long=1, 4 consecutive SOBs are written with
                //           |      | ZeroExtendTo60bitsof bits [15:0]
                //  Value    | 15:0 | If Op=1 & Long=0, an atomic add is performed such that
                //           |      | SOB[14:0]+= Signed[15:0]. As the incoming data is S16,
                //           |      | one can perform a subtraction of the monitor.
                //           |      | If Op=1 & Long=1, The 60 bits SOB which aggregates
                //           |      | 4x15bits physical SOB is atomically added with Signed[15:0]
                //-----------|------|-------------------------------------------------
                //   Long    |  24  | See value field description
                //-----------|------|-------------------------------------------------
                //Trace Event|  30  | When set, a trace event is triggered
                //-----------|------|-------------------------------------------------
                //    Op     |  31  | See value field description
                //-----------|------|-------------------------------------------------
                // so in our case op=1 long=1 and bits [15:0] are 1
                // Value = 0x81000001 (long SO++)
                configureOneMonitor(prog, monIdx, smBase, confVal, syncObjAddress, 0x81000001);

                //
                //
                // 2nd monitor- cq_en configuration :
                //
                //
                // CQ Enable ---- CQ_EN = 1
                // LBW_EN = 1
                LbwEn lbwEn = (cq->isrIdx != scal_illegal_index) ? LbwEn::on : LbwEn::off;
                confVal     = MonitorG2::buildConfVal(0, 0, CqEn::on, LongSobEn::off, lbwEn);
                // the payload address:
                // In case CQ_EN=1, the 6 LSB of the field points to the CQ structure
                // Address = CQ_ID, Data = 1 (CQ will treat 1 as CQ_COUNTER++)
                configureOneMonitor(prog, monIdx + 1, smBase, confVal, (uint64_t)cq->cqIdx, 0x1);
                monIdx += 2;
                //
                //  if (force_order  AND/OR  Scheduler Slaves are defined) as part of SCAL completion group support for HCL
                //    we need to config more monitors
                //
                if (cq->actualNumberOfMonitors > c_completion_queue_monitors_set_size)
                {
                    monIdx = AddCompletionGroupSupportForHCL(prog, cq, smBase, monIdx, soIdx);
                }
                //
                // last monitor- notify the scheduler with writing to the dccm queue of the scheduler:
                //
                //
                // Address of scheduler DCCM messageQ,
                // value according to struct(Mon info of the master, so scheduler can "rearm" the master monitor)
                confVal = 0;
                // The 64 bit address to write the completion message to in case CQ_EN=0.
                uint64_t payloadAddress = cq->scheduler->dccmMessageQueueDevAddress;
                // payload data
                struct sched_mon_exp_comp_fence_t pd;
                memset(&pd, 0x0, sizeof(uint32_t));
                pd.opcode = MON_EXP_COMP_FENCE_UPDATE;
                pd.comp_group_index = cq->idxInScheduler;
                pd.mon_id           = firstMonIdx;
                pd.mon_sm_id = cq->monitorsPool->smIndex;
                pd.mon_index = mIdx;
                uint32_t pdRaw = 0;
                memcpy(&pdRaw,&pd,sizeof(uint32_t));
                LOG_DEBUG(SCAL,"{}: config monIdx {} scheduler DCCM Q payload monId={} dcore={} comp_group_index={}",
                    __FUNCTION__, monIdx, firstMonIdx, dcoreID, cq->idxInScheduler);
                configureOneMonitor(prog, monIdx, smBase, confVal, payloadAddress, pdRaw);

                // arm to CMAX  COMP_SYNC_GROUP_CMAX_TARGET
                // -->   keep this last  <--
                uint32_t ma = MonitorG2::buildArmVal(soIdx, COMP_SYNC_GROUP_CMAX_TARGET, CompType::EQUAL);

                uint64_t armRegAddr = MonitorG2(smBase, firstMonIdx).getRegsAddr().arm;
                prog.addCommand(MsgLong(armRegAddr, ma));

                LOG_DEBUG(SCAL, "arming monitor {} on sync object {} of SO group {}", firstMonIdx, soIdx, soGroupIdx);

            } // loop on mon_depth

            if (cq->compQTdr.enabled)
            {
                configureTdrMon(prog, cq);
            }
            if (cq->sfgInfo.sfgEnabled)
            {
                configSfgSyncHierarchy(prog, cq);
            }
            if (isFenceCounterConnected)
            {
                auto it = m_hostFenceCounters.find(cq->fenceCounterName);
                assert(it != m_hostFenceCounters.end());
                if (it != m_hostFenceCounters.end())
                {
                    if (it->second.isrEnable == false)
                    {
                        cq->isrIdx = scal_illegal_index;
                    }
                }
                configFenceMonitorCounterSMs(prog, cq);
            }
        }     // loop on cq
    }         // loop on dcore

    Qman::Workload workload;
    for (auto & qidProgPair : qid2prog)
    {
        workload.addProgram(qidProgPair.second, qidProgPair.first);
    }

    if (!submitQmanWkld(workload))
    {
        LOG_ERR(SCAL,"{} failed submit workload of configureMonitors", __FUNCTION__);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::loadFWImage()
{
    ImageMap images;

    int ret;

    // load the FWimages from files
    ret = loadFWImagesFromFiles(images);
    if (ret != SCAL_SUCCESS) return ret;

    // copy the FW HBM image to the host binary buffer
    ret = LoadFWHbm(images);
    if (ret != SCAL_SUCCESS) return ret;

    // write the DCCM images to cores DCCM
    ret = LoadFWDccm(images);
    if (ret != SCAL_SUCCESS) return ret;

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::loadFWImagesFromFiles(ImageMap &images)
{
    // iterate over the active cores and pick the image names.
    // search for each image name in the bin search path.
    // call loadFWImageFromFile for each image and return the map of images.
    std::string homePath = getHomeFolder();
    struct arc_fw_metadata_t metaData;
    memset(&metaData,0,sizeof(metaData));
    bool schedVerSet = false;
    bool engVerSet = false;
    for (unsigned idx = 0; idx < c_cores_nr; idx++)
    {
        // we assume cores 0..c_scheduler_nr-1 are scheduler cores (arcs)
        ArcCore * core = getCore<ArcCore>(idx);
        if (core)
        {
            if (images.find(core->imageName)  == images.end())
            {
                bool loaded = false;
                std::vector<std::string> searchedPaths;
                for (auto path : m_fwImageSearchPath)
                {
                    if(path[0]=='~')
                    {
                        path = homePath + path.substr(1);
                    }
                    std::string imagePath = path + "/" + core->imageName + ".bin";
                    if (!fileExists(imagePath))
                    {
                        searchedPaths.push_back(imagePath);
                        continue;
                    }
                    // schedulers arcs dccm size is 64K engine arcs dccm size is 32K
                    FWImage fwImage;
                    images.insert(std::make_pair(core->imageName, fwImage));
                    ImageMap::iterator it = images.find(core->imageName);
                    FWImage* pfwImage = &(it->second);
                    pfwImage->dccmChunksNr = (idx < c_scheduler_nr) ? 2 : 1;
                    pfwImage->image_dccm_size = (idx < c_scheduler_nr ? c_scheduler_image_dccm_size : c_engine_image_dccm_size);
                    pfwImage->image_hbm_size = c_image_hbm_size;
                    bool res = loadFWImageFromFile_gaudi2(m_fd, imagePath,
                        pfwImage->image_dccm_size, pfwImage->image_hbm_size,
                        pfwImage->dccm[0], pfwImage->hbm, &metaData);
                    if(!res)
                    {
                        LOG_ERR(SCAL,"{}: fd={} loadFWImageFromFile() Failed for {}", __FUNCTION__, m_fd, imagePath);
                        assert(0);
                        return SCAL_FW_FILE_LOAD_ERR;
                    }
                    if (!schedVerSet && (metaData.uuid[0] == SCHED_FW_UUID_0) && (metaData.uuid[1] == SCHED_FW_UUID_1) &&
                            (metaData.uuid[2] == SCHED_FW_UUID_2) && (metaData.uuid[3] == SCHED_FW_UUID_3))
                    {
                        schedVerSet = true;
                        m_fw_sched_major_version = metaData.major_version;
                        m_fw_sched_minor_version = metaData.minor_version;
                        LOG_DEBUG(SCAL,"{}: fd={} FW SCHED VERSION {}.{}", __FUNCTION__, m_fd, m_fw_sched_major_version, m_fw_sched_minor_version);
                    }
                    else
                    if (!engVerSet && (metaData.uuid[0] == ENG_FW_UUID_0) && (metaData.uuid[1] == ENG_FW_UUID_1) &&
                            (metaData.uuid[2] == ENG_FW_UUID_2) && (metaData.uuid[3] == ENG_FW_UUID_3))
                    {
                        engVerSet = true;
                        m_fw_eng_major_version = metaData.major_version;
                        m_fw_eng_minor_version = metaData.minor_version;
                        LOG_DEBUG(SCAL,"{}: fd={} FW ENG VERSION {}.{}", __FUNCTION__, m_fd, m_fw_eng_major_version, m_fw_eng_minor_version);
                    }
                    LOG_DEBUG(SCAL,"{}: fd={} Loaded FW from {}", __FUNCTION__, m_fd, imagePath);
                    loaded = true;
                    break;
                }
                if(!loaded)
                {
                    LOG_ERR(SCAL,"{}: fd={} could not find fw file for imageName={}", __FUNCTION__, m_fd, core->imageName);
                    for (auto path : searchedPaths)
                    {
                        LOG_ERR(SCAL,"{}: fd={} could not find fw file for imageName={} in {}", __FUNCTION__, m_fd, core->imageName, path);
                    }
                    assert(0);
                    return SCAL_FW_FILE_LOAD_ERR;
                }
            }
        }
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::LoadFWHbm(const ImageMap &images)
{
    uint64_t binaryOffset = m_binaryPool->allocator->alloc(c_hbm_bin_buffer_size, c_core_memory_extension_range_size);
    if (binaryOffset == Allocator::c_bad_alloc)
    {
        LOG_ERR(SCAL,"{}: fd={} could not allocate {}", __FUNCTION__, m_fd, (uint64_t)c_hbm_bin_buffer_size);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    m_coresBinaryDeviceAddress = m_binaryPool->deviceBase + binaryOffset;
    LOG_INFO_F(SCAL, "binaryOffset {:#x} m_coresBinaryDeviceAddress {:#x} size {:#x}", binaryOffset, m_coresBinaryDeviceAddress, c_hbm_bin_buffer_size);

    std::map<std::string, uint64_t> imageSramAddresses;
    DeviceDmaBuffer sramDmaBuff(m_hw_ip.sram_base_address, this, images.size() * c_image_hbm_size);

    uint8_t * sramDmaBuffHostAddr = (uint8_t*)sramDmaBuff.getHostAddress();
    uint64_t sramDmaBuffDeviceAddr = sramDmaBuff.getDeviceAddress();

    if (!sramDmaBuffHostAddr || !sramDmaBuffDeviceAddr)
    {
        LOG_ERR(SCAL,"{}: fd={} could not allocate sramDmaBuffHostAddr {} or sramDmaBuffDeviceAddr {}", __FUNCTION__, m_fd, sramDmaBuffHostAddr, sramDmaBuffDeviceAddr);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    for (const auto & image : images)
    {
        memcpy(sramDmaBuffHostAddr, &image.second.hbm[0], c_image_hbm_size);
        imageSramAddresses[image.first] = sramDmaBuffDeviceAddr;

        sramDmaBuffDeviceAddr += c_image_hbm_size;
        sramDmaBuffHostAddr += c_image_hbm_size;
    }
    sramDmaBuff.commit();

    std::vector<unsigned> activeDMAs;
    // for (const auto & core : m_cores)
    // {
    //     if (core)
    //     {
    //         if ((core->qmanID == GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE0_EDMA_1_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE1_EDMA_1_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE2_EDMA_1_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0) ||
    //             (core->qmanID == GAUDI2_QUEUE_ID_DCORE3_EDMA_1_0))
    //         {
    //             if (std::find(activeDMAs.begin(), activeDMAs.end(), core->qmanID) == activeDMAs.end())
    //             {
    //                 activeDMAs.emplace_back(core->qmanID);
    //                 // break;
    //             }
    //         }
    //     }
    // }
    if (activeDMAs.empty())
    {
        activeDMAs.emplace_back(GAUDI2_QUEUE_ID_PDMA_1_0);
    }

    const unsigned cls             = (c_image_hbm_size + c_cl_size - 1) / c_cl_size;
    const unsigned clsPerDmaCore   = cls / activeDMAs.size();                 // devide the cls between the EDMAs, round down
    const unsigned clsLastDmaCore  = clsPerDmaCore + cls % activeDMAs.size(); // add the leftover cls to the last EDMA
    const unsigned sizePerDmaCore = clsPerDmaCore * c_cl_size;
    const unsigned sizeLastDmaCore = clsLastDmaCore * c_cl_size;

    Qman::Workload workload;

    for (unsigned dmaIdx = 0; dmaIdx < activeDMAs.size(); dmaIdx++)
    {
        const unsigned dmaQueueIdx = activeDMAs[dmaIdx];
        const unsigned dmaOffset = sizePerDmaCore * dmaIdx;
        const unsigned dmaSize = (dmaIdx == (activeDMAs.size()-1)) ? sizeLastDmaCore : sizePerDmaCore;
        Qman::Program program;
        for (unsigned coreIdx = 0; coreIdx < c_cores_nr; coreIdx++)
        {
            const ArcCore * core = getCore<ArcCore>(coreIdx);
            if (core)
            {
                if (imageSramAddresses.find(core->imageName) == imageSramAddresses.end())
                {
                    LOG_ERR(SCAL,"{}: fd={} could not find {} in imageSramAddresses", __FUNCTION__, m_fd, core->imageName);
                    assert(0);
                    return SCAL_FAILURE;
                }
                uint64_t hbmAddress = m_coresBinaryDeviceAddress + (coreIdx * c_image_hbm_size) + dmaOffset;
                uint64_t sramAddress = imageSramAddresses[core->imageName] + dmaOffset;
                program.addCommand(LinDma(hbmAddress, sramAddress, dmaSize));
            }
        }

        if (!workload.addProgram(program, dmaQueueIdx))
        {
            LOG_ERR(SCAL,"{}: fd={} addProgram failed. dmaIdx={}", __FUNCTION__, m_fd, dmaIdx);
            assert(0);
            return SCAL_FAILURE;
        }
    }

    if (!submitQmanWkld(workload))
    {
        LOG_ERR(SCAL,"{}: fd={} workload.submit failed.", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::loadTPCNopKernel()
{
#ifdef HARD_CODED_NOP_KERNEL
    // generated from running tpc_assembler on NOP; NOP; NOP; NOP
    static const unsigned char NOP_SEQUENCE[] =
        { 0x3f, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x0f,
            0x00, 0xf0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

    //generated from running tpc_assembler on HALT
    static const unsigned char HALT_SEQUENCE[] =
        { 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x0f,
            0x00, 0xf0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

    unsigned tpc_nop_kernel_size = 8 * sizeof(NOP_SEQUENCE) + sizeof(HALT_SEQUENCE);
#else
    unsigned tpc_nop_kernel_size = sizeof_tpc_nop_kernel;
#endif

    if (m_tpcBarrierInSram)
    {
        if (m_hw_ip.sram_size < SCAL_RESERVED_SRAM_SIZE_H6)
        {
            LOG_ERR(SCAL,"{}: fd={} not enough SRAM memory for NOP kernel alignment {}", __FUNCTION__, m_fd, SCAL_RESERVED_SRAM_SIZE_H6);
            assert(0);
            return SCAL_FAILURE;
        }

        uint64_t tpcNopKernelSramDeviceAddr = m_hw_ip.sram_base_address + m_hw_ip.sram_size - SCAL_RESERVED_SRAM_SIZE_H6;
        m_dmaBuff4tpcNopKernel = new DeviceDmaBuffer(tpcNopKernelSramDeviceAddr, this, tpc_nop_kernel_size);
    }
    else
    {
        m_dmaBuff4tpcNopKernel = new DeviceDmaBuffer(m_globalPool, tpc_nop_kernel_size,  SCAL_RESERVED_SRAM_SIZE_H6);
    }

    if (!m_dmaBuff4tpcNopKernel)
    {
        LOG_ERR(SCAL,"{}: fd={} m_dmaBuff4tpcNopKernel allocation error", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    int ret = SCAL_SUCCESS;

    m_tpcNopKernelDeviceAddr = m_dmaBuff4tpcNopKernel->getDeviceAddress();
    // address should be aligned by 2K
    if (!m_tpcNopKernelDeviceAddr || (m_tpcNopKernelDeviceAddr % SCAL_RESERVED_SRAM_SIZE_H6))
    {
        LOG_ERR(SCAL,"{}: fd={} m_tpcNopKernelDeviceAddr 0x{:x} allocation or alignment error", __FUNCTION__, m_fd, m_tpcNopKernelDeviceAddr);
        assert(0);
        ret = SCAL_FAILURE;
    }
    else
    {
        // add the binary for the TPC default NOP kernel
        uint8_t* buf = (uint8_t*)m_dmaBuff4tpcNopKernel->getHostAddress();
        if (buf == nullptr)
        {
            assert(0);
            return SCAL_OUT_OF_MEMORY;
        }
#ifdef HARD_CODED_NOP_KERNEL
        //"kernel" binary: 4 nop sequences, halt, 4 nop sequences
        for (unsigned i = 0; i < 4; ++i)
        {
            memcpy(buf + sizeof(NOP_SEQUENCE) * i, NOP_SEQUENCE, sizeof(NOP_SEQUENCE));
        }
        memcpy(buf + sizeof(NOP_SEQUENCE) * 4, HALT_SEQUENCE, sizeof(HALT_SEQUENCE));

        for (unsigned i = 4; i < 8; ++i)
        {
            memcpy(buf +sizeof(HALT_SEQUENCE) + sizeof(NOP_SEQUENCE) * i, NOP_SEQUENCE, sizeof(NOP_SEQUENCE));
        }
#else
        memcpy(buf, tpc_nop_kernel, tpc_nop_kernel_size);
#endif
        if (!m_dmaBuff4tpcNopKernel->commit())
        {
            LOG_ERR(SCAL,"{}: fd={} nop kernel commit  error", __FUNCTION__, m_fd);
            ret = SCAL_FAILURE;
        }
    }

    return ret;
}


// helper function for writeDCCMs, generates 8k msgShort command for copy fw to 32 kb dccm
void Scal_Gaudi2::genLowerCpFWLoad(Qman::Program &prog, const uint8_t * dccmFWBuff)
{

    uint32_t* binPtr = (uint32_t*) dccmFWBuff;

    for(int i = 0; i < (c_arc_farm_half_dccm_size/sizeof(uint32_t)); i++)
    {
        // TODO do we need to document somewhere what bases we use to which purpses?
        // for example base #3 reserved for program copy?
        MsgShort msgShort(c_message_short_base_index,i*(sizeof(uint32_t)),*binPtr);
        prog.addCommand(msgShort);
        binPtr++;
    }
    // increment fence counter of upper cp
    // 5 cps: [0-3] upper, [4] lower
    uint16_t cpFenceOffset = offsetof(gaudi2::block_qman, cp_fence0_rdata[0]) + c_dccm_to_qm_offset;
    // msgShort, msgLong --> go outside to the LBW fabric
    // wreg32 - write rgister : either in the HW that is attached ( for example in tpc from the tpc arc)
    // or in the qman of the same engine
    bool setMb = true;
    unsigned fenceVal = 1;
    WReg32 wreg32(cpFenceOffset,fenceVal,setMb);
    prog.addCommand(wreg32);
}


int Scal_Gaudi2::LoadFWDccm(const ImageMap &images)
{

    int rc = SCAL_SUCCESS;
    // save location of workload per image name for stage 2
     std::map<std::string, uint64_t> imageFWOffsetMap;

     // single program is composed of 8k msgShort + 'wreg32'. for scheduler and mme slave
     // we have 2 programs: for upper dccm and lower dccm
    static const unsigned halfDCCMCopyProgSize = (c_arc_farm_half_dccm_size / sizeof(uint32_t) )*sizeof(packet_msg_short) + sizeof(packet_wreg32);
    static const unsigned fullDCCMCopyProgSize = 2*halfDCCMCopyProgSize;

    // allocate HBM memory for all FW flavors
    unsigned sizOfFWCopyPrograms = 0;
    for (auto it=images.begin(); it!=images.end(); ++it)
    {
        if( it->second.dccmChunksNr==2)
        {
            sizOfFWCopyPrograms += fullDCCMCopyProgSize;
        }
        else
        {
            sizOfFWCopyPrograms += halfDCCMCopyProgSize;
        }
    }


    DeviceDmaBuffer dmaBuffer(m_binaryPool, sizOfFWCopyPrograms);
    uint8_t* HostProgBuffPtr = (uint8_t*)dmaBuffer.getHostAddress();
    if(!HostProgBuffPtr)
    {
        LOG_ERR(SCAL,"{} aligned_alloc  host memory failed {}", __FUNCTION__, sizOfFWCopyPrograms);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }
    // step 1: generate programs to copy Fw from HBM to DCCM
    unsigned offset = 0;
    for (const auto & it : images)
    {
        Qman::Program progLowDCCM;

        genLowerCpFWLoad(progLowDCCM, it.second.dccm[0]);

        progLowDCCM.serialize(HostProgBuffPtr + offset);
        if (progLowDCCM.getSize() != halfDCCMCopyProgSize)
        {
            LOG_ERR(SCAL,"{} SCAL internal error", __FUNCTION__);
            assert(0);
            return SCAL_FAILURE;
        }

        const std::string & fwName = it.first;
        imageFWOffsetMap[fwName] = offset;
        offset += halfDCCMCopyProgSize;

        if(it.second.dccmChunksNr==2)
        {
            Qman::Program progHighDCCM;
            genLowerCpFWLoad(progHighDCCM, it.second.dccm[1]);
            progHighDCCM.serialize(HostProgBuffPtr + offset);
            if (progHighDCCM.getSize() != halfDCCMCopyProgSize)
            {
                LOG_ERR(SCAL,"{} SCAL internal error", __FUNCTION__);
                assert(0);
                return SCAL_FAILURE;
            }
            offset += halfDCCMCopyProgSize;
        }
    }
    // copy the programs to the device
    rc = dmaBuffer.commit();
    if (!rc) return SCAL_FAILURE;

    // step 2: for each core: schedulers and then engine arcs:
    // generate upper cp program that configures base register 3 to point to the current engine DCCM address
    // (MME slave has special handling) from upper cp wreg0 tp cp_fence0_cnt

    Qman::Workload coreWorkload;
    for (unsigned idx = 0; idx < c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if (core)
        {
            Qman::Program prog;

            // a.	WREG 0 to CP_FENCE0_CNT
            //wreg32 is qman command that doesn't go "outside" to the mesh: addresses are always relative to qman
            uint16_t cpFenceOffset = offsetof(gaudi2::block_qman, cp_fence0_cnt[0]) + c_dccm_to_qm_offset;
            prog.addCommand(WReg32(cpFenceOffset, 0 )); // CSMR protocol; no need to wait

            // b.	WREG to config base3 of the lower CP to point the low part of the DCCM.
            uint16_t base3AddressLow = c_dccm_to_qm_offset + offsetof(gaudi2::block_qman,cp_msg_base3_addr_lo[4]);
            uint16_t base3AddressHigh = c_dccm_to_qm_offset + offsetof(gaudi2::block_qman,cp_msg_base3_addr_hi[4]);

            // if local, use c_local_address, otherwise use core->dccmDevAddress
            bool localMode;
            if(isLocal(core, localMode) != SCAL_SUCCESS)
            {
                LOG_ERR(SCAL,"{}, failed to query localMode", __FUNCTION__);
                assert(0);
                return SCAL_FAILURE;
            }
            uint64_t dccmAddress = localMode ? c_local_address : core->dccmDevAddress;
            prog.addCommand(WReg32(base3AddressLow, (dccmAddress & 0xFFFFFFFF) ));
            prog.addCommand(WReg32(base3AddressHigh, (dccmAddress >> 32) ));

            // c.	Wait 32 cycles followed by fence.
            prog.addCommand(Wait(c_wait_cycles,1,0));
            prog.addCommand(Fence(0,1,1));
            auto it = imageFWOffsetMap.find(core->imageName);
            if(it ==imageFWOffsetMap.end() )
            {
                LOG_ERR(SCAL,"{}, imageName {} in core {} not found", __FUNCTION__, core->imageName, core->cpuId);
                assert(0);
                return SCAL_FAILURE;
            }
            // d.	cpDMA lower part program.
            uint64_t progAddress = imageFWOffsetMap[core->imageName] + dmaBuffer.getDeviceAddress();
            prog.addCommand(CpDma(progAddress, halfDCCMCopyProgSize));
            // e.	Fence (wait for 1 on fence 0, decrement by 1)
            //f.	add message barrier ( mb=1) to fence.
            bool mb=true;
            prog.addCommand(Fence(0,1,1, mb));

            const auto & imageIt = images.find(core->imageName);
            if (imageIt->second.dccmChunksNr == 2) // scheduler, 64 kb fw
            {
                Scheduler * scheduler = core->getAs<Scheduler>();
                if(scheduler->arcFarm)  // arc farm
                {
                    //dccm address was already computed before, just add offset to the higher part of that
                    dccmAddress += c_arc_farm_half_dccm_size;
                    prog.addCommand(WReg32(base3AddressLow, (dccmAddress & 0xFFFFFFFF)));
                    prog.addCommand(WReg32(base3AddressHigh , (dccmAddress>> 32) ));
                }
                else /* mme slave  */
                {
                    // get aux here
                    uint64_t auxAddr = getCoreAuxOffset(scheduler);
                    uint64_t dccmEnable  = auxAddr +  offsetof(gaudi2::block_qman_arc_aux, mme_arc_upper_dccm_en) + core->dccmDevAddress;
                    //i.	Message long to switch DCCM – write 1 to MME_ARC_UPPER_DCCM_EN
                    prog.addCommand(MsgLong(dccmEnable, 1));
                    //j.	Nop with message barrier
                    prog.addCommand(Nop(true));
                }
                //p.	Wait 32 cycles followed by fence.
                prog.addCommand(Wait(c_wait_cycles,1,0));
                prog.addCommand(Fence(0,1,1));
                progAddress += halfDCCMCopyProgSize;

                //q.	cpDMA high part program.
                prog.addCommand(CpDma(progAddress, halfDCCMCopyProgSize));
                //r.	Fence (wait for 1 on fence 0) +
                // S.	Nop with message barrier .
                prog.addCommand(Fence(0,1,1));

                if( !scheduler->arcFarm)  // MME slave
                {
                    // get aux here
                    uint64_t auxAddr = getCoreAuxOffset(scheduler);
                    uint64_t dccmEnable  = auxAddr +  offsetof(gaudi2::block_qman_arc_aux, mme_arc_upper_dccm_en) + core->dccmDevAddress;

                    //v.	Message long to switch DCCM – write 0 to MME_ARC_UPPER_DCCM_EN
                    prog.addCommand(MsgLong(dccmEnable, 0));

                    //x.	Nop with message barrier.
                    prog.addCommand(Nop(true,0,0));
                }
            }
            LOG_DEBUG(SCAL, "{}: loading FW dccm image {} to {} from QMAN {}",
                      __FUNCTION__, core->imageName, core->name, core->qman);
            coreWorkload.addProgram(prog, core->qmanID, true ); // program to run from upper cp
        }
    }

    if (!submitQmanWkld(coreWorkload))
    {
        LOG_ERR(SCAL,"{}, coreWorkload submit failed ", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::configSchedulers()
{
    int ret = SCAL_SUCCESS;
    DeviceDmaBuffer buffs[c_scheduler_nr];
    Qman::Workload workload;

    uint32_t coreIds[c_scheduler_nr]     = {};
    uint32_t coreQmanIds[c_scheduler_nr] = {};

    uint32_t counter = 0;

    LOG_INFO_F(SCAL, "=== configSchedulers ===");
    for (unsigned idx = 0; idx < c_scheduler_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if (core)
        {
            coreIds[counter]       = idx;
            coreQmanIds[counter++] = core->qmanID;

            LOG_DEBUG(SCAL, "{}: scheduler {}", __FUNCTION__, core->name);

            ret = allocSchedulerConfigs(idx, buffs[idx]);
            if (ret != SCAL_SUCCESS) break;

            RegToVal regToVal;
            ret = fillSchedulerConfigs(idx, buffs[idx], regToVal);
            if (ret != SCAL_SUCCESS) break;

            ret = (buffs[idx].commit(&workload) ? SCAL_SUCCESS : SCAL_FAILURE); // relevant only if uses HBM pool
            if (ret != SCAL_SUCCESS) break;

            Qman::Program program;
            ret = createCoreConfigQmanProgram(idx, buffs[idx], false /* lowerCP */, program, &regToVal);
            if (ret != SCAL_SUCCESS) break;

            workload.addProgram(program, core->qmanID, false /* lowerCP */);
        }
    }

    if (ret == SCAL_SUCCESS)
    {
        if (!submitQmanWkld(workload))
        {
            LOG_ERR(SCAL,"{}: fd={} workload submit failed", __FUNCTION__, m_fd);
            assert(0);
            return SCAL_FAILURE;
        }
    }
    else
    {
        LOG_ERR(SCAL,"{}: fd={} workload creation failed", __FUNCTION__, m_fd);
        assert(0);
    }

    if (ret == SCAL_SUCCESS)
    {
        ret = runCores(coreIds, coreQmanIds, counter);
        if (ret != SCAL_SUCCESS)
        {
            LOG_ERR(SCAL,"{}: Failed to run schedulers' cores", __FUNCTION__);
            assert(0);
            return SCAL_FAILURE;
        }
    }

    return ret;
}

uint32_t Scal_Gaudi2::getSRAMSize() const
{
    return m_hw_ip.sram_size - (m_tpcBarrierInSram ? SCAL_RESERVED_SRAM_SIZE_H6 : 0);
}

int Scal_Gaudi2::allocSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff)
{
    // allocate an ARC host buffer for the config file
    ArcCore * core = getCore<ArcCore>(coreIdx);
    if(!core)
    {
        LOG_ERR(SCAL,"{}: fd={} called with empty core[{}]", __FUNCTION__, m_fd, coreIdx);
        assert(0);
        return SCAL_FAILURE;
    }
    bool ret = buff.init(core->configPool, sizeof(struct scheduler_config_t));
    if(!ret)
    {
        assert(0);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

/*
 * fillSchedulerConfigs stages:
 * 0. allocate cqs and sobs
 *     0.a. find cqs
 *     0.b. find so sets
 * 1. configure dup engines and calculate bitmask
 *     1.a. calculate bitmask
 *     1.b. configure dup engines
 * 2. configure the bitmask
 * 3. configure eng_grp_cfg
 * 4. validations
 */
int Scal_Gaudi2::fillSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff, RegToVal &regToVal)
{
    // fill the scheduler config file according to the configuration
    //TODO,version header should be added also?
    scheduler_config_t* fwInitCfg = (scheduler_config_t *)(buff.getHostAddress());
    if(!fwInitCfg)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to allocate host memory for scheduler_config_t", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    memset(fwInitCfg, 0x0, sizeof(scheduler_config_t));
    fwInitCfg->version = ARC_FW_INIT_CONFIG_VER;
    fwInitCfg->psoc_arc_intr_addr = m_hw_ip.engine_core_interrupt_reg_addr;
    fwInitCfg->synapse_params = m_arc_fw_synapse_config_t; // binary copy of struct
    Scheduler * scheduler = getCore<Scheduler>(coreIdx);
    if (scheduler == nullptr)
    {
        LOG_ERR(SCAL, "{}: fd={} core index {} is not a scheduler", __FUNCTION__, m_fd, coreIdx);
        assert(0);
        return SCAL_INVALID_CONFIG;
    }

    if (!m_distributedCompletionGroupCreditsSosPool || !m_distributedCompletionGroupCreditsMonitorsPool)
    {
        LOG_ERR(SCAL,"{}: fd={} Distributed-Completion-Group's CreditsSosPool or CreditsMonitorsPool is null",
                __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }

    if (!m_completionGroupCreditsSosPool || !m_completionGroupCreditsMonitorsPool)
    {
        LOG_ERR(SCAL,"{}: fd={} Completion-Group's CreditsSosPool or CreditsMonitorsPool is null", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }

    // 0. allocate cqs and sobs
    // =======================

    // 0.a. find cqs
    // --------------
    if (m_schedulersCqsMap.find(scheduler->name) != m_schedulersCqsMap.end())
    {
        for (CompletionGroup* cq : m_schedulersCqsMap[scheduler->name])
        {
            if (cq == nullptr)
            {
                // HCL credit system depends on all master and slave cqs having
                // the same index number within their respective scheduler cq array.
                // so sometimes we need to create a "hole" in the scheduler cq array
                LOG_INFO(SCAL, "{}: sched {} {} found empty cq, \"hole\", ignoring",
                    __FUNCTION__, scheduler->cpuId, scheduler->name);
                continue;
            }
            // check if the cq is owned by this scheduler (i.e. the scheduler is this cq master scheduler) or else is the slave scheduler of this cq
            bool     isSlave        = (cq->scheduler != scheduler);
            unsigned csg_configIdx  = cq->idxInScheduler;
            unsigned numOfCqsSlaves = cq->slaveSchedulers.size();

            if (isSlave)
            {
                // find the index of our scheduler
                for (auto& schedCQ : cq->slaveSchedulers)
                {
                    if (schedCQ.scheduler == scheduler)
                    {
                        scal_assert_return(csg_configIdx == schedCQ.idxInScheduler, SCAL_FAILURE,
                                            "{}: CQ {} has inconsistent. index in master ({}) = {}. index in slave ({}) = {}",
                                            __FUNCTION__, cq->name, cq->scheduler->name, csg_configIdx, scheduler->name, schedCQ.idxInScheduler);
                        break;
                    }
                }
            }

            assert(csg_configIdx < COMP_SYNC_GROUP_COUNT);
            struct comp_sync_group_config_t* pfwCSG = &fwInitCfg->csg_config[csg_configIdx];
            pfwCSG->sob_start_id                    = cq->sosBase + (c_max_sos_per_sync_manager * cq->syncManager->dcoreIndex);
            pfwCSG->sob_count                       = cq->sosNum;
            pfwCSG->mon_count                       = cq->actualNumberOfMonitors;
            pfwCSG->mon_group_count                 = cq->monNum / cq->actualNumberOfMonitors;
            pfwCSG->slave_comp_group                = (uint32_t)isSlave; //  0 - Master Completion Group  1 - Slave Completion Group
            pfwCSG->in_order_completion             = (uint32_t)cq->force_order;

            if (isSlave)
            {
                pfwCSG->credit_sync_sob_id_base = cq->creditManagementSobIndex;
            }
            else if (numOfCqsSlaves != 0)
            {
                // Set Num of CQ slaves
                pfwCSG->num_slaves = numOfCqsSlaves;
            }

            pfwCSG->in_order_monitor_offset = cq->nextSyncObjectMonitorId;

            if (cq->compQTdr.enabled)
            {
                pfwCSG->watch_dog_sob_addr = SobG2::getAddr(cq->compQTdr.sosPool->smBaseAddr, cq->compQTdr.sos);
                LOG_DEBUG_F(SCAL, "tdr enabled for {} using sm {} sos {} addr {:#x}", cq->name, cq->compQTdr.sosPool->smIndex, cq->compQTdr.sos, pfwCSG->watch_dog_sob_addr);
            }

            LOG_DEBUG(SCAL, "{}: sched {} {} cq {} {}  sob_start_id {} sob_count {} mon_count {} mon_group_count {} mon Range [{}-{}] ForceOrder {} Slave {}", __FUNCTION__, scheduler->cpuId,
                      scheduler->name, cq->name, csg_configIdx, pfwCSG->sob_start_id, pfwCSG->sob_count, pfwCSG->mon_count, pfwCSG->mon_group_count, cq->monBase,
                      cq->monBase + cq->monNum - 1, // not sent to FW, configured elsewhere
                      pfwCSG->in_order_completion, pfwCSG->slave_comp_group);
        }
    }
    else
    {
        // no cqs for this scheduler
    }

    // 0.b. find so sets
    // ------------------
    if (m_schedulersSosMap.find(scheduler->name) != m_schedulersSosMap.end())
    {
        unsigned soSet_configIdx = 0;
        for (SyncObjectsSetGroup* sg : m_schedulersSosMap[scheduler->name])
        {
            fwInitCfg->so_sets_count = sg->numSets;
            for (unsigned idx = 0; idx < sg->numSets; idx++)
            {
                if (soSet_configIdx >= SO_SET_COUNT)
                {
                    LOG_ERR(SCAL,"{}: fd={} so_set_config index > max {} at so set group {}",
                            __FUNCTION__, m_fd, SO_SET_COUNT, sg->name);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                fwInitCfg->so_set_config[soSet_configIdx].sob_start_id = sg->sosPool->nextAvailableIdx + (c_max_sos_per_sync_manager * sg->sosPool->dcoreIndex);
                sg->sosPool->nextAvailableIdx += sg->setSize;
                if(sg->sosPool->nextAvailableIdx > sg->sosPool->baseIdx + sg->sosPool->size)
                {
                    LOG_ERR(SCAL,"{}: fd={} so next available index {} > so pool size at so set group {}",
                            __FUNCTION__, m_fd, sg->sosPool->nextAvailableIdx, sg->name);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                fwInitCfg->so_set_config[soSet_configIdx].mon_id = sg->resetMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * sg->resetMonitorsPool->dcoreIndex);
                sg->resetMonitorsPool->nextAvailableIdx++; // 1 monitor per set
                if(sg->resetMonitorsPool->nextAvailableIdx > sg->resetMonitorsPool->baseIdx + sg->resetMonitorsPool->size)
                {
                    LOG_ERR(SCAL,"{}: fd={} monitor pool next available index {} > monitor pool size at so set group {}",
                            __FUNCTION__, m_fd, sg->resetMonitorsPool->nextAvailableIdx, sg->name);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                soSet_configIdx++;
            }
        }

        for (/* nothing */; soSet_configIdx < SO_SET_COUNT; soSet_configIdx++)
        {
            fwInitCfg->so_set_config[soSet_configIdx].sob_start_id = UINT32_MAX;
        }

        // prepare nextAvailableIdx for engines monitors allocation
        for (SyncObjectsSetGroup* sg : m_schedulersSosMap[scheduler->name])
        {
            sg->resetMonitorsPool->nextAvailableIdx = sg->resetMonitorsPool->baseIdx;
        }
    }

    uint64_t dupAddr = 0;
    if(scheduler->arcFarm)
    {
        dupAddr = mmARC_FARM_ARC0_DUP_ENG_BASE - mmARC_FARM_ARC0_DCCM0_BASE + scheduler->dccmDevAddress;
        // dupAddr = mmARC_FARM_ARC0_DUP_ENG_BASE - mmDCORE0_TPC0_QM_ARC_AUX_BASE;
    }
    else // mme slave
    {
        dupAddr = mmDCORE0_MME_QM_ARC_DUP_ENG_BASE - mmDCORE0_MME_QM_ARC_DCCM_BASE + scheduler->dccmDevAddress;
        // dupAddr = mmDCORE0_MME_QM_ARC_DUP_ENG_BASE - mmDCORE0_TPC0_QM_ARC_AUX_BASE;
    }


    for (auto clusterItr : scheduler->clusters)
    {
        auto cluster = clusterItr.second;
        if (cluster->queues.size() >= QMAN_ENGINE_GROUP_TYPE_COUNT)
        {
            LOG_ERR(SCAL, "scheduler's cluster has to many queues {} > QMAN_ENGINE_GROUP_TYPE_COUNT ({})",
                    cluster->queues.size(), QMAN_ENGINE_GROUP_TYPE_COUNT);
            assert(0);
            return SCAL_INVALID_CONFIG;
        }
        unsigned queueIndex  = 0;
        struct queueIndex2DupTriggerIndex
        {
            unsigned queueIndex;
            unsigned dupTriggerIndex;
        };
        std::map<Scal_Gaudi2::DupTrigger, std::vector<queueIndex2DupTriggerIndex>> dupTrigger2IndexesMap;

        for (auto& queueItr : cluster->queues)
        {
            unsigned group_engine_bitmap = 0;
            auto& queue = queueItr.second;
            if (queue.scheduler->name != scheduler->name)
            {
                continue;
            }
            auto& fwInitCfgGrp = fwInitCfg->eng_grp_cfg[queue.group_index];
            fwInitCfgGrp.engine_count_in_group = cluster->engines.size();

            // 1. configure dup engines and calculate bitmask
            // ===============================================
            unsigned dupAddrOffset     = c_dup_trigger_info[queue.dupTrigger].dup_engines_address_base_entry;
            unsigned engineMaskInQueue = 0;
            LOG_DEBUG(SCAL, "{}: group {} will configure dup trigger {} of scheduler {}", __FUNCTION__,queue.group_index, queue.dupTrigger, scheduler->name);
            for (auto engine_ : cluster->engines)
            {
                ArcCore * engine = engine_->getAs<ArcCore>();
                // 1.a. calculate bitmask
                // -----------------------
                group_engine_bitmap |= (0x1 << (queue.bit_mask_offset + engineMaskInQueue));
                assert(queue.bit_mask_offset + engineMaskInQueue < c_dup_trigger_info[queue.dupTrigger].engines);

                // 1.b. configure dup engines
                // ---------------------------
                // dupEngEntry is the entry in block_arc_dup_eng_defaults which we want to overwrite for the current
                // engine, from here we get the address to write into
                assert(queue.bit_mask_offset + engineMaskInQueue < c_dup_trigger_info[queue.dupTrigger].engines);
                unsigned dupEngineEntry   = dupAddrOffset + queue.bit_mask_offset + engineMaskInQueue;
                uint64_t dupEngAddr       = dupAddr + gaudi2::block_arc_dup_eng_defaults[dupEngineEntry].offset;
                unsigned dupEngineAddress = lbw_block_address(getCoreAuxOffset(engine) + engine->dccmDevAddress);
                if (!addRegisterToMap(regToVal, dupEngAddr, dupEngineAddress))
                {
                    LOG_ERR(SCAL, "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                            __FUNCTION__, cluster->name, dupEngAddr, regToVal[dupEngAddr], dupEngineAddress, engine->name);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                LOG_DEBUG(SCAL, "engine {}: group {} will write dup address {:#x} to {:#x}", engine->name, queue.group_index, dupEngineAddress, dupEngAddr);
                engineMaskInQueue++;
            }
            // 2. configure the bitmask (for primary and secondery dup triggers)
            // ==================================================================
            std::vector<Scal_Gaudi2::DupTrigger> dupTriggersCombined;
            dupTriggersCombined.push_back(static_cast<Scal_Gaudi2::DupTrigger>(queue.dupTrigger));
            for (const auto trigger : queue.secondaryDupTriggers)
            {
                dupTriggersCombined.push_back(static_cast<Scal_Gaudi2::DupTrigger>(trigger));
            }
            for (auto& dupTrigger : dupTriggersCombined)
            {
                uint64_t clusterMaskAddr   = dupAddr + c_dup_trigger_info[dupTrigger].cluster_mask;
                if (!addRegisterToMap(regToVal, clusterMaskAddr, group_engine_bitmap))
                {
                    LOG_ERR(SCAL, "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                            __FUNCTION__, cluster->name, queue.group_index, clusterMaskAddr, regToVal[clusterMaskAddr], group_engine_bitmap);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                LOG_DEBUG(SCAL, "cluster {}: will write bitmask {:#x} to {:#x}", cluster->name, group_engine_bitmap, clusterMaskAddr);

                // configure the dup trigger queue index
                // =====================================
                // required when queue.index != queue.dup_trans_data_q_index, but we configure for all dup trans data queues
                uint64_t dup_trans_data_q_base_address = dupAddr + varoffsetof(gaudi2::block_arc_dup_eng, dup_grp_eng_addr_offset[dupTrigger]);
                uint32_t dup_trans_data_q_base_val = offsetof(gaudi2::block_qman_arc_aux, dccm_queue_push_reg) + sizeof(uint32_t) * (queue.index - queue.dup_trans_data_q_index);
                if (!addRegisterToMap(regToVal, dup_trans_data_q_base_address, dup_trans_data_q_base_val))
                {
                    LOG_ERR(SCAL, "{}: cluster {}: dup_trans_data_q_base register {:#x} was already configured to {:#x} and configured again to {:#x}",
                            __FUNCTION__, cluster->name, dup_trans_data_q_base_address, regToVal[dup_trans_data_q_base_address], dup_trans_data_q_base_val);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                LOG_DEBUG(SCAL, "{}: scheduler {} cluster {} queue {} dup trigger {} dup trans data queue {} => dup_trans_data_q_base_address {:#x} dup_trans_data_q_base_val {:#x}",
                            __FUNCTION__, queue.scheduler->name, cluster->name, queue.index, dupTrigger, queue.dup_trans_data_q_index, dup_trans_data_q_base_address, dup_trans_data_q_base_val);

            }

            // 3. configure eng_grp_cfg
            // =========================
            assert(queue.dup_trans_data_q_index < c_dup_trigger_info[queue.dupTrigger].dup_trans_data_queues);
            unsigned dup_trans_data_q_offset_address = 0;
            switch (queue.dup_trans_data_q_index)
            {
                case 0:
                {
                    dup_trans_data_q_offset_address = varoffsetof(gaudi2::block_arc_dup_eng, dup_trans_data_q_0[queue.dupTrigger]);
                    break;
                }
                case 1:
                {
                    dup_trans_data_q_offset_address = varoffsetof(gaudi2::block_arc_dup_eng, dup_trans_data_q_1[queue.dupTrigger]);
                    break;
                }
                case 2:
                {
                    dup_trans_data_q_offset_address = varoffsetof(gaudi2::block_arc_dup_eng, dup_trans_data_q_2[queue.dupTrigger]);
                    break;
                }
                case 3:
                {
                    dup_trans_data_q_offset_address = varoffsetof(gaudi2::block_arc_dup_eng, dup_trans_data_q_3[queue.dupTrigger]);
                    break;
                }
                default:
                {
                    LOG_ERR(SCAL,"invalid queue index {}, there are only 4 queues for dup_trans_data", queue.dup_trans_data_q_index);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
            }

            // when configuring the dup_trans_data_q_addr for the FW, the base address is from the device perspective.
            //   thats why the base address is mmARC_FARM_ARC0_DUP_ENG_DUP_TPC_ENG_ADDR_0 and not dupAddr
            uint32_t dup_trans_data_q_addr = scheduler->dupEngLocalDevAddress + dup_trans_data_q_offset_address;
            uint32_t dup_base_local_addr = c_arc_acc_engs_virtual_addr + c_local_dup_offset;
            uint32_t dup_trans_data_q_addr_lbu = ((dup_trans_data_q_addr & 0xFFF) + dup_base_local_addr) | c_arc_lbw_access_msb;
            fwInitCfgGrp.dup_trans_data_q_addr = dup_trans_data_q_addr_lbu;

            // Credits-Management
            unsigned cmSobjBaseIndex = 0;
            unsigned cmMonBaseIndex  = 0;
            int status = getCreditManagmentBaseIndices(cmSobjBaseIndex, cmMonBaseIndex, true);
            if (status != SCAL_SUCCESS)
            {
                LOG_ERR(SCAL,
                        "{}: cluster {}: Invalid Engines' Cluster-Queue Credit-Management configuration",
                        __FUNCTION__, cluster->name);
                assert(0);
                return SCAL_INVALID_CONFIG;
            }
            queue.sobjBaseIndex    = cmSobjBaseIndex;
            queue.monitorBaseIndex = cmMonBaseIndex;

            LOG_DEBUG(SCAL,
                      "{}, cluster {}(dup_trans_data_q_index {}) group_index {} sobjs-start-index {} "
                      "monitors-start-index {} dup_trans_data_q_addr = {:#x}",
                      __FUNCTION__, cluster->name, queue.dup_trans_data_q_index, queue.group_index,
                      queue.sobjBaseIndex, queue.monitorBaseIndex,
                      fwInitCfgGrp.dup_trans_data_q_addr);

            dupTrigger2IndexesMap[static_cast<Scal_Gaudi2::DupTrigger>(queue.dupTrigger)].push_back({.queueIndex = queue.index, .dupTriggerIndex = queue.dup_trans_data_q_index});
            queueIndex++;
        }
        // cluster dup trigger validation
        for (const auto& dupTrigger2IndexesEntry : dupTrigger2IndexesMap)
        {
            auto& indexesVec = dupTrigger2IndexesEntry.second;
            for (unsigned i = 1; i < indexesVec.size(); i++)
            {
                if (indexesVec[i].queueIndex - indexesVec[i - 1].queueIndex != indexesVec[i].dupTriggerIndex - indexesVec[i - 1].dupTriggerIndex)
                {
                    LOG_ERR(SCAL, "{}: fd={} error in cluster {} queue indexes offsets ({}) should match dup trigger offsets ({})",
                            __FUNCTION__, m_fd, cluster->name, indexesVec[i].queueIndex - indexesVec[i - 1].queueIndex,
                            indexesVec[i].dupTriggerIndex - indexesVec[i - 1].dupTriggerIndex);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
            }
        }
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::createCoreConfigQmanProgram(const unsigned coreIdx, DeviceDmaBuffer &buff, const bool upperCP, Qman::Program & program, RegToVal *regToVal)
{
    // create a core QMAN configuration program
    // configure base3 and use message shorts - scheduler: absolute address and engine ARCs local address.
    // configure the ARC ID
    // configure the memory extensions
    //   - region 4 according to the coreIDX and m_fwImageHbmAddr. In region 4 also set the offset.
    //   - set the rest of the regions according to the cores pools.
    // configure SP 0 with the LBW address of fence 0 of the current CP - schedulers absolute address and eARCs local address
    // configure SP 1 with (buff->pool->coreBase + buff->offset)
    // configure SP 2 with the size of the config buff in bytes
    // request run mode
    // wait for fence
    // reset the run request
    // nop with message barrier

    LOG_INFO_F(SCAL, "===== createCoreConfigQmanProgram core {} =====", coreIdx);

    ArcCore * core = getCore<ArcCore>(coreIdx);
    if (core == nullptr)
    {
        LOG_ERR(SCAL, "{}: fd={} core index {} is not valid", __FUNCTION__, m_fd, coreIdx);
        assert(0);
        return SCAL_FAILURE;
    }

    bool localMode;
    if (upperCP)
    {
        localMode = false;
    }
    else
    {
        if(isLocal(core, localMode) == SCAL_FAILURE)
        {
            LOG_ERR(SCAL,"{}, failed to query localMode", __FUNCTION__);
            assert(0);
            return SCAL_FAILURE;
        }
    }

    Scheduler * scheduler = core->getAs<Scheduler>();
    if (!scheduler)
    { // configure base2 to be used by FW, specifically required by PDMA-TX ARC
        // set cores's base2
        for (unsigned cp=0; cp<c_cps_nr; cp++)
        {
            uint64_t addr = (cp == (c_cps_nr - 1)) ? c_local_address : core->dccmDevAddress;
            uint64_t base2AddressLo = core->dccmDevAddress + c_dccm_to_qm_offset +
                                      varoffsetof(gaudi2::block_qman, cp_msg_base2_addr_lo[cp]);
            uint64_t base2AddressHi = core->dccmDevAddress + c_dccm_to_qm_offset +
                                      varoffsetof(gaudi2::block_qman, cp_msg_base2_addr_hi[cp]);

            if (localMode)
            {
                program.addCommand(WReg32((uint16_t) base2AddressLo, lower_32_bits(addr)));
                program.addCommand(WReg32((uint16_t) base2AddressHi, upper_32_bits(addr)));
            }
            else
            {
                program.addCommand(MsgLong(base2AddressLo, lower_32_bits(addr)));
                program.addCommand(MsgLong(base2AddressHi, upper_32_bits(addr)));
            }
        }
    }
    else
    {
        if (m_schedulersCqsMap.find(core->name) != m_schedulersCqsMap.end())
        {
            for (CompletionGroup* cq : m_schedulersCqsMap[scheduler->name])
            {
                if (cq == nullptr)
                {
                    continue; // deliberate "hole"
                }
                bool     isSlave        = (cq->scheduler != scheduler);
                unsigned numOfCqsSlaves = cq->slaveSchedulers.size();

                if ((!isSlave) && (numOfCqsSlaves != 0))
                {
                    // Configure Distributed-Completion-Group (CQs) Master-Slaves CM Monitors
                    LOG_DEBUG(SCAL, "{}: Configure credit-management (DCG) sched {} {} cq {}",
                                __FUNCTION__, scheduler->cpuId, scheduler->name, cq->name);
                    //
                    uint64_t counterAddress = scheduler->dccmMessageQueueDevAddress;
                    uint32_t countervalue   = getDistributedCompletionGroupCreditManagmentCounterValue(cq->idxInScheduler);
                    //
                    int rtn = configureCreditManagementMonitors(program, cq->creditManagementSobIndex, cq->creditManagementMonIndex, numOfCqsSlaves, counterAddress, countervalue);
                    if (rtn != SCAL_SUCCESS)
                    {
                        return rtn;
                    }
                }
            }
        }

        for (auto clusterItr : scheduler->clusters)
        {
            auto cluster = clusterItr.second;

            for (auto& queueItr : cluster->queues)
            {
                auto& queue = queueItr.second;
                if (queue.scheduler->name != core->name)
                {
                    continue;
                }

                // Configure Completion-Group CM-Monitors
                LOG_DEBUG(SCAL, "{}: Configure credit-management (CG) sched {} cluster {} queue-group-index {}",
                          __FUNCTION__, scheduler->name, cluster->name, queue.group_index);
                //
                uint64_t counterAddress = scheduler->dccmMessageQueueDevAddress;
                uint32_t countervalue   = getCompletionGroupCreditManagmentCounterValue(queue.group_index);
                //
                int rtn = configureCreditManagementMonitors(program, queue.sobjBaseIndex, queue.monitorBaseIndex, cluster->engines.size(), counterAddress, countervalue);
                if (rtn != SCAL_SUCCESS)
                {
                    return rtn;
                }
            }
        }
    }

    // configure base3 and use message shorts -
    //      scheduler: absolute address and engine ARCs local address.

    const unsigned cpIdx = upperCP ? 0 : 4;

    uint64_t auxAddr =  getCoreAuxOffset(core) + (localMode? c_local_address: core->dccmDevAddress);

    uint64_t fenceAddr = c_local_address;
    if (!localMode)
    {
        if (!qmanId2DccmAddr(core->qmanID, fenceAddr))
        {
            LOG_ERR(SCAL,"{}, qmanId2DccmAddr() failed for core id {}", __FUNCTION__, core->qmanID);
            assert(0);
            return SCAL_FAILURE;
        }
    }
    fenceAddr += c_dccm_to_qm_offset + varoffsetof(gaudi2::block_qman, cp_fence0_rdata[cpIdx]);

    // set base3
    uint16_t base3Address = c_dccm_to_qm_offset + varoffsetof(gaudi2::block_qman,cp_msg_base3_addr_lo[cpIdx]);
    program.addCommand(WReg32(base3Address, lower_32_bits(auxAddr)));
    base3Address = c_dccm_to_qm_offset + varoffsetof(gaudi2::block_qman,cp_msg_base3_addr_hi[cpIdx]);
    program.addCommand(WReg32(base3Address, upper_32_bits(auxAddr)));
    // reset fence0 before the wait. WREG 0 to FENCE_CNT0.
    program.addCommand(WReg32(c_dccm_to_qm_offset + varoffsetof(gaudi2::block_qman, cp_fence0_cnt[cpIdx]), 0));
    program.addCommand(Wait(c_wait_cycles,1,0)); // Wait 32 cycles followed by fence
    program.addCommand(Fence(0,1,1));
    // configure the memory extensions
    //   - region 4 according to the coreIDX and m_fwImageHbmAddr. In region 4 also set the offset.
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, hbm0_msb_addr), upper_32_bits(m_coresBinaryDeviceAddress)));
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, hbm0_lsb_addr), lower_32_bits(m_coresBinaryDeviceAddress) / c_core_memory_extension_range_size));
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, hbm0_offset), (coreIdx * c_image_hbm_size)));

    LOG_INFO_F(SCAL, "core {} {} m_coresBinaryDeviceAddress {:#x}", coreIdx, core->name, m_coresBinaryDeviceAddress);

    uint64_t region[MemoryExtensionRange::RANGES_NR] {};
    for (const auto & pool : core->pools)
    {
        unsigned numRanges = ((uint64_t)pool->size + c_core_memory_extension_range_size - 1) / c_core_memory_extension_range_size;
        for (unsigned range = 0; range < numRanges; range++)
        {
            LOG_INFO_F(SCAL, "pool {} addressExtensionIdx {} range {}", pool->name, pool->addressExtensionIdx, range);
            if (region[pool->addressExtensionIdx + range] != 0)
            {
                LOG_ERR_F(SCAL, "trying to set the same region {} twice", pool->addressExtensionIdx + range);
                assert(0);
                return SCAL_FAILURE;
            }
            region[pool->addressExtensionIdx + range] = (pool->deviceBase + (range * c_core_memory_extension_range_size)) & 0xFFFFFFFFF0000000;
        }
    }

    for (int extEntry = 0; extEntry < MemoryExtensionRange::RANGES_NR; extEntry++)
    {
        uint64_t extAddr = region[extEntry];
        unsigned addrLowRegOffset = 0;
        unsigned addrHighRegOffset = 0;
        unsigned offsetRegOffset = 0;

        switch (extEntry)
        {
            case ICCM:
                // not configurable
                break;
            case SRAM:
                addrLowRegOffset = offsetof(gaudi2::block_qman_arc_aux, sram_lsb_addr);
                addrHighRegOffset = offsetof(gaudi2::block_qman_arc_aux, sram_msb_addr);
                break;
            case CFG:
                // not configurable
                break;
            case GP0:
                addrLowRegOffset = offsetof(gaudi2::block_qman_arc_aux, general_purpose_lsb_addr[0]);
                addrHighRegOffset = offsetof(gaudi2::block_qman_arc_aux, general_purpose_msb_addr[0]);
                break;
            case HBM0:
                // not configurable
                break;
            case HBM1:
                addrLowRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm1_lsb_addr);
                addrHighRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm1_msb_addr);
                offsetRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm1_offset);
                break;
            case HBM2:
                addrLowRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm2_lsb_addr);
                addrHighRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm2_msb_addr);
                offsetRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm2_offset);
                break;
            case HBM3:
                addrLowRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm3_lsb_addr);
                addrHighRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm3_msb_addr);
                offsetRegOffset = offsetof(gaudi2::block_qman_arc_aux, hbm3_offset);
                break;
            case DCCM:
                // not configurable
                break;
            case PCI:
                addrLowRegOffset = offsetof(gaudi2::block_qman_arc_aux, pcie_lsb_addr);
                addrHighRegOffset = offsetof(gaudi2::block_qman_arc_aux, pcie_msb_addr);
                break;
            case GP1:
            case GP2:
            case GP3:
            case GP4:
            case GP5:
                addrLowRegOffset = varoffsetof(gaudi2::block_qman_arc_aux, general_purpose_lsb_addr[1+extEntry-GP1]);
                addrHighRegOffset = varoffsetof(gaudi2::block_qman_arc_aux, general_purpose_msb_addr[1+extEntry-GP1]);
                break;
            case LBU:
                // not configurable
                break;
            }

            if (!addrLowRegOffset || !addrHighRegOffset)
            {
                if (extAddr != 0)
                {
                    LOG_ERR(SCAL,"{}: fd={} illegal memory extension index - qmanId2DccmAddr() failed for core id {}", __FUNCTION__, m_fd, core->qmanID);
                    assert(0);
                    return SCAL_FAILURE;
                }
                else
                {
                    continue;
                }
            }

            if (extAddr == 0)
            {
                extAddr = m_coresBinaryDeviceAddress;
            }

            program.addCommand(MsgShort(c_message_short_base_index, addrHighRegOffset, upper_32_bits(extAddr)));
            program.addCommand(MsgShort(c_message_short_base_index, addrLowRegOffset, lower_32_bits(extAddr) / c_core_memory_extension_range_size));

            if (offsetRegOffset)
            {
                program.addCommand(MsgShort(c_message_short_base_index, offsetRegOffset, 0));
            }
            LOG_INFO_F(SCAL, "core {}\t #{}\t extEntry {}\t extAddr {:#x}", coreIdx, core->name, extEntry, extAddr);

    }

    // configure the ARC ID
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, arc_num), core->cpuId));
    LOG_DEBUG(SCAL, "core {} writing block_qman_arc_aux::arc_num = {} from qman_id = {}", core->name, core->cpuId, core->qmanID);

    // configure SP 0 with the LBW address of fence 0 of the current CP - schedulers absolute address and eARCs local address
    uint32_t fenceCoreAddress = lower_27_bits(fenceAddr) | (MemoryExtensionRange::LBU * c_core_memory_extension_range_size);
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, scratchpad[0]), fenceCoreAddress));

    // configure SP 1 with (buff->pool->coreBase + buff->offset)
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, scratchpad[1]), buff.getDeviceAddress()));
    // configure SP 2 with the size of the config buff in bytes
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi2::block_qman_arc_aux, scratchpad[2]), buff.getSize()));
    LOG_INFO_F(SCAL, "core {} {} scratchpad 0 1 2 {:#x} {:#x} {:#x}", coreIdx, core->name, fenceCoreAddress, buff.getDeviceAddress(), buff.getSize());

    if (regToVal)
    {
        for (auto& regToValPair : *regToVal)
        {
            if (regToValPair.first < auxAddr || (regToValPair.first - auxAddr) != uint16_t(regToValPair.first - auxAddr))
            {
                LOG_DEBUG(SCAL, "writing reg {:#x} = {:#x} using long message - might cause performance degredation", regToValPair.first, regToValPair.second);
                program.addCommand(MsgLong(regToValPair.first, regToValPair.second));
            }
            else
            {
                program.addCommand(MsgShort(c_message_short_base_index, regToValPair.first - auxAddr, regToValPair.second));
                // LOG_DEBUG(SCAL, "writing reg {:#x} = {:#x}", regToValPair.first, regToValPair.second);
            }
        }
    }

    return SCAL_SUCCESS;
}

const std::string cqGroup0Key = "compute_completion_queue0";

int Scal_Gaudi2::configEngines()
{
    LOG_INFO_F(SCAL, "===== configEngines =====");

    int ret = SCAL_SUCCESS;
    DeviceDmaBuffer buffs[c_cores_nr];
    Qman::Workload workload;
    const bool isSfgEnabledInJson = m_completionGroups.count(cqGroup0Key) && m_completionGroups.at(cqGroup0Key).sfgInfo.sfgEnabled;
    int baseSfgSobPerEngineType[unsigned(EngineTypes::items_count)] = {0};
    const int baseSfgSobPerEngineTypeIncrement[unsigned(EngineTypes::items_count)] = {1, 1, 1, 1};
    if (isSfgEnabledInJson)
    {
        for (unsigned i = 0 ; i < unsigned(EngineTypes::items_count); ++i)
        {
            baseSfgSobPerEngineType[i] = m_completionGroups.at(cqGroup0Key).sfgInfo.baseSfgSob[i];
        }
    }
    uint32_t sfgBaseSobId  = 0;

    uint32_t coreIds[c_cores_nr - c_scheduler_nr]     = {};
    uint32_t coreQmanIds[c_cores_nr - c_scheduler_nr] = {};

    uint32_t counter = 0;
    // for all engine Arcs
    for (unsigned idx = c_scheduler_nr; idx < c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if (core)
        {
            coreIds[counter]       = idx;
            coreQmanIds[counter++] = core->qmanID;

            LOG_DEBUG(SCAL, "{}: engine {} qman {}", __FUNCTION__, core->arcName, core->qman);
            ret = allocEngineConfigs(core, buffs[idx]);
            if (ret != SCAL_SUCCESS) break;

            const bool sfgConfEnabled = (core->clusters.begin()->second) && isSfgEnabledInJson &&
                            getSfgSobIdx(core->clusters.begin()->second->type, sfgBaseSobId, baseSfgSobPerEngineType, baseSfgSobPerEngineTypeIncrement);

            ret = fillEngineConfigs(core, buffs[idx], sfgBaseSobId, sfgConfEnabled);
            if (ret != SCAL_SUCCESS) break;

            ret = (buffs[idx].commit(&workload) ? SCAL_SUCCESS : SCAL_FAILURE); // relevant only if uses HBM pool
            if (ret != SCAL_SUCCESS) break;

            Qman::Program program;
            ret = createCoreConfigQmanProgram(idx, buffs[idx], false /* lower CP */, program);
            if (ret != SCAL_SUCCESS) break;

            workload.addProgram(program, core->qmanID, false /* lower CP */);
        }
    }

    if (ret == SCAL_SUCCESS)
    {
        if (!submitQmanWkld(workload))
        {
            LOG_ERR(SCAL,"{}: fd={} workload.submit() failed", __FUNCTION__, m_fd);
            assert(0);
            return SCAL_FAILURE;
        }
    }

    if (ret == SCAL_SUCCESS)
    {
        ret = runCores(coreIds, coreQmanIds, counter);
        if (ret != SCAL_SUCCESS)
        {
            LOG_ERR(SCAL,"{}: Failed to run engines' cores", __FUNCTION__);
            assert(0);
            return SCAL_FAILURE;
        }
    }

    return ret;
}

int Scal_Gaudi2::allocEngineConfigs(const ArcCore * arcCore, DeviceDmaBuffer &buff)
{
    // allocate an ARC host buffer for the config file
    if(!arcCore)
    {
        LOG_ERR(SCAL,"{}: fd={} called with empty core", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }
    bool ret = buff.init(arcCore->configPool, sizeof(engine_config_t));
    if(!ret)
    {
        assert(0);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::fillEngineConfigs(const ArcCore* arcCore, DeviceDmaBuffer &buff, int sfgBaseSobId, bool sfgConfEnabled)
{
    // fill the engine ARC config file according to the configuration
    //TODO,version header should be added also

    engine_config_t* fwInitCfg = (engine_config_t *)(buff.getHostAddress());
    if(!fwInitCfg)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to allocate host memory for arc_fw_init_config", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    memset(fwInitCfg, 0x0, sizeof(engine_config_t));
    fwInitCfg->version = ARC_FW_INIT_CONFIG_VER;
    fwInitCfg->tpc_nop_kernel_addr_lo = lower_32_bits(m_tpcNopKernelDeviceAddr);
    fwInitCfg->tpc_nop_kernel_addr_hi = upper_32_bits(m_tpcNopKernelDeviceAddr);

    bool isCompute = false;
    for (const auto& cluster : arcCore->clusters)
    {
        for (const auto& queue : cluster.second->queues)
        {
            fwInitCfg->eng_resp_config[queue.second.index].sob_start_id = queue.second.sobjBaseIndex;
        }
        fwInitCfg->dccm_queue_count += cluster.second->queues.size();
        isCompute |= cluster.second->isCompute;
    }
    fwInitCfg->synapse_params   = m_arc_fw_synapse_config_t; // binary copy of struct

    if (isCompute)
    {
        const Scheduler * queueScheduler = nullptr;
        for (auto queueItr : arcCore->clusters.begin()->second->queues)
        {
            auto queue = queueItr.second;
            if (queue.scheduler)
            {
                if (queueScheduler && queueScheduler->cpuId != queue.scheduler->cpuId)
                {
                    LOG_ERR(SCAL, "{}: compute engine {} is associated with more than one scheduler: {}, {}",
                        __FUNCTION__, arcCore->name, queueScheduler->name, queue.scheduler->name);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                queueScheduler = queue.scheduler;
            }
        }

        auto* gcMonitorsPool = queueScheduler->m_sosSetGroups[0]->gcMonitorsPool;
        fwInitCfg->mon_start_id = gcMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * gcMonitorsPool->dcoreIndex);
        gcMonitorsPool->nextAvailableIdx += SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT;
        if (gcMonitorsPool->nextAvailableIdx - gcMonitorsPool->baseIdx > gcMonitorsPool->size)
        {
            LOG_ERR(SCAL,"{}, compute engine sync monitor use ({}) exceeds max pool monitors ({}). from pool {} total queues {}",
                __FUNCTION__, gcMonitorsPool->nextAvailableIdx - gcMonitorsPool->baseIdx,
                gcMonitorsPool->size, gcMonitorsPool->name, fwInitCfg->dccm_queue_count);
            return SCAL_FAILURE;
        }
        auto* computeBack2BackMonitorsPool = queueScheduler->m_sosSetGroups[0]->computeBack2BackMonitorsPool;
        fwInitCfg->b2b_mon_id = computeBack2BackMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * computeBack2BackMonitorsPool->smIndex);
        computeBack2BackMonitorsPool->nextAvailableIdx += SCHED_CMPT_ENG_B2B_MON_COUNT;
        if (computeBack2BackMonitorsPool->nextAvailableIdx - computeBack2BackMonitorsPool->baseIdx > computeBack2BackMonitorsPool->size)
        {
            LOG_ERR(SCAL,"{}, compute back2bACK monitor use ({}) exceeds max pool monitors ({}). from pool {} total queues {}",
                __FUNCTION__, computeBack2BackMonitorsPool->nextAvailableIdx - computeBack2BackMonitorsPool->baseIdx,
                computeBack2BackMonitorsPool->size, computeBack2BackMonitorsPool->name, fwInitCfg->dccm_queue_count);
            return SCAL_FAILURE;
        }

        auto* topologyDebuggerMonitorsPool = queueScheduler->m_sosSetGroups[0]->topologyDebuggerMonitorsPool;
        fwInitCfg->soset_dbg_mon_start_id = topologyDebuggerMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * topologyDebuggerMonitorsPool->dcoreIndex);
        topologyDebuggerMonitorsPool->nextAvailableIdx += SCHED_CMPT_ENG_SYNC_SCHEME_DBG_MON_COUNT;
        if (topologyDebuggerMonitorsPool->nextAvailableIdx - topologyDebuggerMonitorsPool->baseIdx > topologyDebuggerMonitorsPool->size)
        {
            LOG_ERR(SCAL,"{}, compute engine sync monitor use ({}) exceeds max pool monitors ({}). from pool {} total queues {}",
                __FUNCTION__, topologyDebuggerMonitorsPool->nextAvailableIdx - topologyDebuggerMonitorsPool->baseIdx,
                topologyDebuggerMonitorsPool->size, topologyDebuggerMonitorsPool->name, fwInitCfg->dccm_queue_count);
            return SCAL_FAILURE;
        }

        // soset sob pool
        auto* gcSosPool = queueScheduler->m_sosSetGroups[0]->sosPool;
        fwInitCfg->soset_pool_start_sob_id = queueScheduler->m_sosSetGroups[0]->sosPool->baseIdx + (c_max_sos_per_sync_manager * gcSosPool->dcoreIndex);
    }
    else
    {
        fwInitCfg->mon_start_id = scal_illegal_index;
    }

    for (const auto& cluster : m_computeClusters)
    {
        unsigned virtualSobIndex = -1;
        if (!coreType2VirtualSobIndex(cluster->type, virtualSobIndex))
        {
            LOG_ERR(SCAL,"{}, compute engine cluster {} type {} doesn't match VIRTUAL_SOB_INDEX", __FUNCTION__, cluster->name, cluster->type);
            return SCAL_FAILURE;
        }
        fwInitCfg->engine_count_in_asic[virtualSobIndex] = cluster->engines.size();
    }
    fwInitCfg->engine_index = arcCore->indexInGroup;
    fwInitCfg->num_engines_in_group = arcCore->numEnginesInGroup;

    fwInitCfg->cmpt_csg_sob_id_base  = m_computeCompletionQueuesSos->baseIdx + (c_max_sos_per_sync_manager * m_computeCompletionQueuesSos->dcoreIndex);

    fwInitCfg->cmpt_csg_sob_id_count = m_computeCompletionQueuesSos->size;

    fwInitCfg->psoc_arc_intr_addr = m_hw_ip.engine_core_interrupt_reg_addr;

    if (sfgConfEnabled)
    {
        fwInitCfg->sfg_sob_base_id = sfgBaseSobId + m_completionGroups.at(cqGroup0Key).sfgInfo.sfgSosPool->dcoreIndex * c_max_sos_per_sync_manager;
        fwInitCfg->sfg_sob_per_stream = m_completionGroups.at(cqGroup0Key).sfgInfo.sobsOffsetToNextStream;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::activateEngines()
{
    // write "1" to the canary registers of all the active cores
    for (unsigned idx =0; idx <  c_scheduler_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if(core)
        {
            uint64_t canaryAddress = offsetof(sched_registers_t, canary) + (uint64_t)(core->dccmHostAddress);
            LOG_TRACE(SCAL, "{}: writing canary for scheduler {} at address {:#x}",
                      __FUNCTION__, core->name, canaryAddress);
            writeLbwReg((volatile uint32_t*)(canaryAddress), SCAL_INIT_COMPLETED);
        }
    }
    for (unsigned idx =c_scheduler_nr; idx < c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if(core)
        {
            uint64_t canaryAddress = offsetof(engine_arc_reg_t, canary) + (uint64_t)(core->dccmHostAddress);
            writeLbwReg((volatile uint32_t*)(canaryAddress), SCAL_INIT_COMPLETED);
        }
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::configureStreams()
{
    for (auto& streamMapPair : m_streams)
    {
        auto& stream = streamMapPair.second;
        auto  ret    = streamSetPriority(&stream, stream.priority);
        if (ret != SCAL_SUCCESS) return ret;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi2::getNumberOfSignalsPerMme() const
{
    return 2; // each MME Engine sends 2 messages upon completion - SRAM/DRAM write. slave signaling is disabled in FW.
}

int Scal_Gaudi2::getCreditManagmentBaseIndices(unsigned& cmSobjBaseIndex,
                                               unsigned& cmMonBaseIndex,
                                               bool      isCompletionGroupCM) const
{
    std::string creditsManagementTypeName[2] = {"schedulers-cluster", "completion-group"};

    SyncObjectsPool* cmCreditsSosPool      = nullptr;
    MonitorsPool*    cmCreditsMonitorsPool = nullptr;

    if (isCompletionGroupCM)
    {
        cmCreditsSosPool      = m_completionGroupCreditsSosPool;
        cmCreditsMonitorsPool = m_completionGroupCreditsMonitorsPool;
    }
    else
    {
        cmCreditsSosPool      = m_distributedCompletionGroupCreditsSosPool;
        cmCreditsMonitorsPool = m_distributedCompletionGroupCreditsMonitorsPool;
    }

    cmSobjBaseIndex = cmCreditsSosPool->nextAvailableIdx + (c_max_sos_per_sync_manager * cmCreditsSosPool->smIndex);
    cmMonBaseIndex  = cmCreditsMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * cmCreditsMonitorsPool->smIndex);;

    LOG_INFO(SCAL,"{}: fd={} isCompletionGroupCM {} dcore {} cmSobjBaseIndex {} ({}) cmMonBaseIndex {} ({})",
             __FUNCTION__, m_fd, isCompletionGroupCM, cmCreditsSosPool->dcoreIndex,
             cmSobjBaseIndex, cmCreditsSosPool->nextAvailableIdx,
             cmMonBaseIndex, cmCreditsMonitorsPool->nextAvailableIdx);

    // For both Elements (SOBJs and MONs):
    // 1 - Get Base value
    // 2 - Validate set's indeices are at the same SM
    // 3 - Increment nextAvailableIdx
    // 4 - Validate that we didn't exceed the amount of elements

    // validate that the sos and the monitors are from the same quarter
    unsigned sosQuarterSize = c_max_sos_per_sync_manager / 4;
    unsigned queueSosQuarter = cmSobjBaseIndex / sosQuarterSize;
    unsigned queueSosQuarterEnd = (cmSobjBaseIndex + c_sos_for_completion_group_credit_management - 1) / sosQuarterSize;
    if (queueSosQuarter != queueSosQuarterEnd)
    {
        LOG_ERR(SCAL,"{}: fd={} {} credits sos {} range [{},{}] is not from the same quarter at dcore {}",
                __FUNCTION__, m_fd, creditsManagementTypeName[isCompletionGroupCM],
                cmCreditsSosPool->name, cmSobjBaseIndex,
                cmSobjBaseIndex + c_sos_for_completion_group_credit_management - 1,
                cmCreditsSosPool->dcoreIndex);
        assert(0);
        return SCAL_INVALID_CONFIG;
    }

    unsigned monitorQuarterSize  = c_max_monitors_per_sync_manager / 4;
    unsigned queueMonitorQuarter = cmMonBaseIndex / monitorQuarterSize;
    unsigned queueMonitorQuarterEnd = (cmMonBaseIndex + c_monitors_for_completion_group_credit_management - 1) / monitorQuarterSize;
    if (queueMonitorQuarter != queueMonitorQuarterEnd)
    {
        LOG_ERR(SCAL,"{}: fd={} {} credits monitors {} range [{},{}] is not from the same quarter at dcore {}",
                __FUNCTION__, m_fd, creditsManagementTypeName[isCompletionGroupCM],
                cmCreditsMonitorsPool->name, cmMonBaseIndex,
                cmMonBaseIndex + c_monitors_for_completion_group_credit_management - 1,
                cmCreditsMonitorsPool->dcoreIndex);
        assert(0);
        return SCAL_INVALID_CONFIG;
    }

    cmCreditsSosPool->nextAvailableIdx += c_sos_for_completion_group_credit_management;
    if (cmCreditsSosPool->nextAvailableIdx >
        cmCreditsSosPool->baseIdx + cmCreditsSosPool->size)
    {
        LOG_ERR(SCAL,"{}, {} credits so index ({}) exceeds max pool index ({}). from pool {}",
                __FUNCTION__, creditsManagementTypeName[isCompletionGroupCM],
                cmCreditsSosPool->nextAvailableIdx - 1,
                cmCreditsSosPool->baseIdx + cmCreditsSosPool->size,
                cmCreditsSosPool->name);
        return SCAL_INVALID_CONFIG;
    }

    cmCreditsMonitorsPool->nextAvailableIdx += c_monitors_for_completion_group_credit_management;
    if (cmCreditsMonitorsPool->nextAvailableIdx >
        cmCreditsMonitorsPool->baseIdx + cmCreditsMonitorsPool->size)
    {
        LOG_ERR(SCAL,"{}, {} credits monitor index (completion group credits {}) exceeds max pool index ({}). from pool {}",
                __FUNCTION__, creditsManagementTypeName[isCompletionGroupCM],
                cmCreditsMonitorsPool->nextAvailableIdx - 1,
                cmCreditsMonitorsPool->baseIdx + cmCreditsMonitorsPool->size,
                cmCreditsMonitorsPool->name);
        return SCAL_INVALID_CONFIG;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::configureCreditManagementMonitors(Qman::Program& prog,
                                                    unsigned       sobjIndex,
                                                    unsigned       monitorBaseIndex,
                                                    unsigned       numOfSlaves,
                                                    uint64_t       counterAddress,
                                                    uint32_t       counterValue)
{
    LOG_INFO(SCAL, "{} sobjIndex = {} monitorBaseIndex = {} numOfSlaves = {} "
             "counterAddress = {:#x} counterValue = {:#x}",
             __FUNCTION__, sobjIndex, monitorBaseIndex, numOfSlaves, counterAddress, counterValue);

    const unsigned doubleBuffer = 2;

    for (unsigned i = 0; i < doubleBuffer; i++)
    {
        unsigned dcoreId          = getMonitorDcoreId(monitorBaseIndex);
        uint64_t smBase           = SyncMgrG2::getSmBase(dcoreId);
        uint64_t monitorIndexInSm = monitorBaseIndex % c_max_monitors_per_sync_manager;

        // we configure 3 monitors, all should be in same core
        if ((monitorIndexInSm + 2) >= c_max_monitors_per_sync_manager)
        {
            LOG_ERR_F(SCAL, "Bad config; monitorBaseIndex {} monitorIndexInSm {}", monitorBaseIndex, monitorIndexInSm);
            assert(0);
            return SCAL_FAILURE;
        }

        uint64_t syncObjIndexInSm = sobjIndex % c_max_sos_per_sync_manager;

        // Arming the monitors to wait for SOBJ to reach >= numOfSlaves :
        // Monitor-ARM value:
        uint32_t monArmData = MonitorG2::buildArmVal(syncObjIndexInSm, numOfSlaves);
        //
        // Monitor-ARM address:
        uint64_t monArmAddress = MonitorG2(smBase, monitorIndexInSm).getRegsAddr().arm;

        // Sync-Object update fields' description :
        //
        // Field Name| Bits |   Comments
        //-----------|------|-------------------------------------------------
        //           |      | If Op=0 & Long=0, the SOB is written with bits [14:0]
        //           |      | If Op=0 & Long=1, 4 consecutive SOBs are written with
        //           |      | ZeroExtendTo60bitsof bits [15:0]
        //  Value    | 15:0 | If Op=1 & Long=0, an atomic add is performed such that
        //           |      | SOB[14:0]+= Signed[15:0]. As the incoming data is S16,
        //           |      | one can perform a subtraction of the monitor.
        //           |      | If Op=1 & Long=1, The 60 bits SOB which aggregates
        //           |      | 4x15bits physical SOB is atomically added with Signed[15:0]
        //-----------|------|-------------------------------------------------
        //   Long    |  24  | See value field description
        //-----------|------|-------------------------------------------------
        //Trace Event|  30  | When set, a trace event is triggered
        //-----------|------|-------------------------------------------------
        //    Op     |  31  | See value field description
        //-----------|------|-------------------------------------------------

        //
        //  1st monitor - the master
        //
        //
        // Monitor Configuration : Increment counter
        // Master Monitor + inc long SO ----- MSG_NR = 2 (3 messages)
        const unsigned numOfMsgs = 3; // MSGs - Update Counter, Reset SOBJ (decrease num_of_slaves), Re-Arm Master-Monitor

        uint32_t confVal = MonitorG2::buildConfVal(syncObjIndexInSm, numOfMsgs - 1, CqEn::off, LongSobEn::off, LbwEn::off);
        // mc.long_high_group = 0 Not relevant for short monitor
        //
        // Payload address : The counterAddress
        //
        // Payload Value : The counterValue
        configureOneMonitor(prog, monitorIndexInSm, smBase, confVal, counterAddress, counterValue);

        //
        //
        // 2nd monitor - SO Decrementation
        //
        // Monitor Configuration :
        confVal = MonitorG2::buildConfVal(0, 0, CqEn::off, LongSobEn::off, LbwEn::off);
        //
        // Payload address : The SOBJ Address
        uint64_t sobjAddress = SobG2::getAddr(smBase, syncObjIndexInSm);
        //
        // Payload Value :
        // We want to deccrement (by num_slaves) a regular (15-bits) SOB
        // So in our case op=1 long=0 and bits [15:0] are (-num_slaves)
        //
        sync_object_update syncObjUpdate;
        syncObjUpdate.raw                  = 0;
        syncObjUpdate.so_update.sync_value = (-numOfSlaves);
        syncObjUpdate.so_update.mode       = 1;
        //
        configureOneMonitor(prog, monitorIndexInSm + 1, smBase, confVal, sobjAddress, syncObjUpdate.raw);

        //
        //
        // 3rd monitor - Re-ARM of master monitor
        //
        // Monitor Configuration :
        confVal = MonitorG2::buildConfVal(0, 0, CqEn::off, LongSobEn::off, LbwEn::off);
        //
        // Payload address : The monArmAddress
        //
        // Payload Value : Re-ARM master monitor command (see above - monArmData setting)
        // We want to set the monArmData value
        configureOneMonitor(prog, monitorIndexInSm + 2, smBase, confVal, monArmAddress, monArmData);

        //
        //
        // Monitor-ARM
        //
        LOG_INFO_F(SCAL, "Monitor-Arm {} ({} {}) addr {:#x} data {:#x}",
                   monitorBaseIndex, dcoreId, monitorIndexInSm, monArmAddress, monArmData);
        prog.addCommand(MsgLong(monArmAddress, monArmData));

        sobjIndex++;
        monitorBaseIndex += numOfMsgs;
    }
    return SCAL_SUCCESS;
}

uint32_t Scal_Gaudi2::getDistributedCompletionGroupCreditManagmentCounterValue(uint32_t completionGroupIndex)
{
    sched_mon_exp_comp_fence_t message;

    message.raw = 0;

    message.opcode              = MON_EXP_COMP_FENCE_UPDATE;
    message.comp_group_index    = completionGroupIndex;
    message.update_slave_credit = true;

    return message.raw;
}

uint32_t Scal_Gaudi2::getCompletionGroupCreditManagmentCounterValue(uint32_t engineGroupType)
{
    sched_mon_exp_update_q_credit_t message;

    message.raw = 0;

    message.opcode            = MON_EXP_UPDATE_Q_CREDIT;
    message.engine_group_type = engineGroupType;

    return message.raw;
}

unsigned Scal_Gaudi2::getSobjDcoreId(unsigned sobjIndex)
{
    return sobjIndex / c_max_sos_per_sync_manager;
}

unsigned Scal_Gaudi2::getMonitorDcoreId(unsigned monitorIndex)
{
    return monitorIndex / c_max_monitors_per_sync_manager;
}

inline bool Scal_Gaudi2::coreType2VirtualSobIndex(const CoreType coreType, unsigned& virtualSobIndex)
{
    switch (coreType)
    {
        case TPC:  virtualSobIndex = VIRTUAL_SOB_INDEX_TPC;  return true;
        case MME:  virtualSobIndex = VIRTUAL_SOB_INDEX_MME;  return true;
        case EDMA: virtualSobIndex = VIRTUAL_SOB_INDEX_EDMA; return true;
        case ROT:  virtualSobIndex = VIRTUAL_SOB_INDEX_ROT;  return true;
        default: return false;
    }
}

void Scal_Gaudi2::addFencePacket(Qman::Program& program, unsigned id, uint8_t targetVal, unsigned decVal)
{
    program.addCommand(Fence(id, targetVal, decVal));
}

void Scal_Gaudi2::enableHostFenceCounterIsr(CompletionGroup * cg, bool enableIsr)
{
    const unsigned curMonIdx = cg->monBase + cg->monNum;
    const unsigned smIdx     = cg->syncManager->smIndex;
    const uint32_t mon2      = curMonIdx + 1;

    LOG_INFO(SCAL,"{}: =========================== Fence Counter ISR =============================", __FUNCTION__);
    LOG_DEBUG(SCAL,"{}: Configure SM{}_MON_{} ISR {} ", __FUNCTION__, smIdx, mon2, enableIsr ? "enable" : "disable");

    uint64_t offset = MonitorG2(0, mon2).getRegsAddr().config;
    offset = offset / sizeof(*cg->syncManager->objsHostAddress);
    uint32_t monConfig = scal_read_mapped_reg(&cg->syncManager->objsHostAddress[offset]);
    monConfig = MonitorG2::setLbwEn(monConfig, enableIsr);
    scal_write_mapped_reg(&cg->syncManager->objsHostAddress[offset],  monConfig);
    [[maybe_unused]] auto v = scal_read_mapped_reg(&cg->syncManager->objsHostAddress[offset]); // read config in order to be sure the value is written into register
    assert(v == monConfig);
}

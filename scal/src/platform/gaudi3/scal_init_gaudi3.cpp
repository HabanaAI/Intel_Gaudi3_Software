#include <assert.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <algorithm>
#include "scal.h"
#include "scal_allocator.h"
#include "scal_base.h"
#include "scal_utilities.h"
#include "logger.h"
#include "gaudi3/gaudi3.h"
#include "scal_qman_program_gaudi3.h"
#include "gaudi3_arc_common_packets.h"
#include "gaudi3_arc_sched_packets.h"
#include "gaudi3/asic_reg/gaudi3_blocks.h"

#include "gaudi3/asic_reg_structs/cache_maintenance_regs.h"
#include "gaudi3/asic_reg_structs/qman_arc_aux_regs.h"
#include "gaudi3/asic_reg_structs/qman_regs.h"
#include "gaudi3/asic_reg_structs/sob_glbl_regs.h"
#include "gaudi3/asic_reg_structs/arc_af_eng_regs.h"
#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"
#include "gaudi3/asic_reg_structs/pdma_ch_b_regs.h"
#include "gaudi3/asic_reg_structs/nic_qpc_regs.h"
#include "gaudi3/asic_reg/mstr_if_dup_crdt_regs.h"
#include "gaudi3/asic_reg_structs/arc_aux_regs.h"
#include "scal_macros.h"
#include "scal_data_gaudi3.h"
#include <bitset>
#include <common/pci_ids.h>
#include "scal_gaudi3_sfg_configuration_helpers.h"
#include "common/scal_init_sfg_configuration_impl.hpp"
#include "infra/monitor.hpp"
#include "infra/sob.hpp"
#include "infra/sync_mgr.hpp"

// clang-format off

using CqEn      = Monitor::CqEn;
using LongSobEn = Monitor::LongSobEn;
using LbwEn     = Monitor::LbwEn;
using CompType  = Monitor::CompType;

const unsigned EDUPCreditValue = 252;

constexpr unsigned cme_cs_edup_trigger = 0;
constexpr unsigned cme_dccmq_edup_trigger = 1;
constexpr unsigned cme_tpc_fence0_edup_trigger = 2; // 4 groups
constexpr unsigned cme_mme_fence0_edup_trigger = cme_tpc_fence0_edup_trigger + 4;
constexpr unsigned cme_rot_fence0_edup_trigger = cme_mme_fence0_edup_trigger + 1;
constexpr unsigned cme_tpc_cid_offset_trigger = cme_rot_fence0_edup_trigger + 1; // 4 groups

static std::string getClusterNameByCoreType(CoreType type)
{
    switch(type)
    {
        case MME: return "mme";
        case TPC: return "compute_tpc";
        case ROT: return "rotator";
        default:  return "invalid";
    }
    return "invalid";
}
const std::map<std::string, unsigned> c_fenceEdupTriggers = {
    {getClusterNameByCoreType(TPC), cme_tpc_fence0_edup_trigger},
    {getClusterNameByCoreType(MME), cme_mme_fence0_edup_trigger},
    {getClusterNameByCoreType(ROT), cme_rot_fence0_edup_trigger}};

inline int8_t getEngineTypeOffsetInSet(CoreType clusterType)
{
    switch(clusterType)
    {
        case TPC:
            return SOB_OFFSET_TPC_IN_SO_SET;
        case MME:
            return SOB_OFFSET_MME_IN_SO_SET;
        default:
            return NUM_OF_CORE_TYPES;
    }

}

int Scal_Gaudi3::init(const std::string & configFileName)
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

    // map the ACPs and DCCMs to the device.
    ret = mapLBWBlocks();
    if (ret != SCAL_SUCCESS) return ret;

    // MR - TODO: in case there is a DirectMode PDMA-channel, choose one has the config-channel
    //            and avoid re- / override configuration

    // open the pdma channel
    ret = openConfigChannel();
    if (ret != SCAL_SUCCESS) return ret;

    // allocates memory for the completion queues
    ret = allocateCompletionQueues();
    if (ret != SCAL_SUCCESS) return ret;

    // init direct-mode PDMA-channels
    ret = initDirectModePdmaChannels();
    if (ret != SCAL_SUCCESS) return ret;

    // configure the sync managers and completion queues
    ret = configureSMs();
    if (ret != SCAL_SUCCESS) return ret;

    // load the FW to the active cores
    ret = loadFWImage();
    if (ret != SCAL_SUCCESS) return ret;

    // check the canary register of the active cores
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

    // close the pdma channel
    ret = closeConfigChannel();
    if (ret != SCAL_SUCCESS) return ret;

    // init ptr-mode pdma
    ret = configurePdmaPtrsMode();
    if (ret != SCAL_SUCCESS) return ret;

    // write the canary register and activate the ARCs.
    ret = activateEngines();
    if (ret != SCAL_SUCCESS) return ret;

    m_bgWork = std::make_unique<BgWork>(m_timeoutUsNoProgress, m_timeoutDisabled);
    for (auto& [cgName, pCgInstance] : m_completionGroups)
    {
        (void)cgName; // Unuesd
        m_bgWork->addCompletionGroup(&pCgInstance); // use m_cgs!!
    }
    for (auto& [cgName, pCgInstance] : m_directModeCompletionGroups)
    {
        (void)cgName; // Unuesd
        m_bgWork->addCompletionGroup(pCgInstance);
    }

    LOG_DEBUG(SCAL,"{}: fd={} Init() Done.", __FUNCTION__, m_fd);
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::allocateHBMPoolMem(uint64_t size, uint64_t* handle, uint64_t* addr, uint64_t hintOffset, bool shared, bool contiguous)
{
    // allocate hbm memory for initMemoryPools
    // map it using the supplied hint
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HBM_START  0x0201E00000000000ull
    //#define RESERVED_VA_RANGE_FOR_ARC_ON_HBM_END    0x0201FFFFFFFFFFFFull

    static constexpr uint64_t c_48_bits_mask = 0x0000ffffffffffffull;

    uint64_t hintsRangeBase  = RESERVED_VA_RANGE_FOR_ARC_ON_HBM_START;
    LOG_INFO_F(SCAL, "hlthunk_device_memory_alloc size {:#x} hintOffset {:#x} shared {} contiguous {}",
               size, hintOffset, shared, contiguous);
    *handle = hlthunk_device_memory_alloc(m_fd, size, m_hw_ip.device_mem_alloc_default_page_size, contiguous, shared);
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

int Scal_Gaudi3::allocateHostPoolMem(uint64_t size, void** hostAddr, uint64_t* deviceAddr, uint64_t hintOffset)
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
        LOG_ERR(SCAL,"{}: fd={} deviceAddr {:#x} != hint address {:#x}", __FUNCTION__, m_fd, *deviceAddr, hint);
        assert(0);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::initMemoryPools()
{
    LOG_INFO_F(SCAL, "===== initMemoryPools =====");

    uint64_t totalPagesNr = 0;
    Pool *zeroPool = nullptr;

    LOG_INFO_F(SCAL, "number memory pools {:#x}", m_pools.size());

    for (auto & [k, pool]  : m_pools)
    {
        (void)k; // unused, compiler error in gcc 7.5
        std::string msg = fmt::format(FMT_COMPILE("Pool {} of type {} size {:#x}"), pool.name, HLLOG_DUPLICATE_PARAM(pool.type), pool.size);
        if ((pool.type == Pool::Type::HBM) || (pool.type == Pool::Type::HBM_EXTERNAL))
        {
            uint64_t pages = (pool.size + m_hw_ip.device_mem_alloc_default_page_size - 1) / m_hw_ip.device_mem_alloc_default_page_size;
            pool.size = pages * m_hw_ip.device_mem_alloc_default_page_size;
            msg += fmt::format(FMT_COMPILE(" {} aligned size {:#x}"), pool.name, pool.size);

            totalPagesNr += pages;

            if (pages == 0)
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
            msg += fmt::format(FMT_COMPILE(" pages {:#x} totalPagesNr {:#x}"), pages, totalPagesNr);
        }
        LOG_INFO_F(SCAL, "{}", msg);
    }
    uint64_t availableHBMPagesNr =  (m_hw_ip.device_mem_alloc_default_page_size == 0) ? 0 : m_hw_ip.dram_size / m_hw_ip.device_mem_alloc_default_page_size;

    if ((totalPagesNr > availableHBMPagesNr) ||
        ((totalPagesNr == availableHBMPagesNr) && zeroPool))
    {
        LOG_ERR(SCAL,"{}: fd={} Out of HBM memory totalPagesNr {} availableHBMPagesNr {} zeroPool {}", __FUNCTION__, m_fd,
        		totalPagesNr, availableHBMPagesNr, (uint64_t)(zeroPool));
        assert(0);
        return SCAL_FAILURE;
    }

    // zeroPool is the available user memory size and we provide all remaining memory on HBM.
    // we should reduce 4KB from it to avoid HW bug OOB prefetcher of TPC cache line, which can try
    // to read tensor from HBM + 4KB and result with a RAZWI access (SW-112283)
    if (zeroPool)
    {
        // Check if zeroPool size will be bigger than 0 (uint64_t wrap around)
        uint64_t totalPagesSize = (availableHBMPagesNr * m_hw_ip.device_mem_alloc_default_page_size);
        if ( totalPagesSize < (totalPagesNr * m_hw_ip.device_mem_alloc_default_page_size) + (4 * 1024) )
        {
            LOG_ERR(SCAL,"{}: fd={} Out of HBM memory for zeroPool, totalPagesNr {} availableHBMPagesNr {}", __FUNCTION__, m_fd,
            totalPagesNr, availableHBMPagesNr);
            assert(0);
            return SCAL_FAILURE;
        }
        zeroPool->size = totalPagesSize - (totalPagesNr * m_hw_ip.device_mem_alloc_default_page_size) - (4 * 1024);
        LOG_INFO_F(SCAL, "zero pool size {:#x}, availableHBMPagesNr {:#x}", zeroPool->size, availableHBMPagesNr);
    }

    uint64_t offset = 0;

    for (auto & poolPair : m_pools)
    {
        auto & pool = poolPair.second;

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
            int ret = allocateHostPoolMem(pool.size, &pool.hostBase, &pool.deviceBase, offset);
            if (ret != SCAL_SUCCESS)
                return ret;
        }
        offset += pool.size;
        pool.coreBase = pool.addressExtensionIdx ? (uint32_t)pool.deviceBase : 0;
        LOG_DEBUG_F(SCAL, "allocated {} ({:#x}) bytes to pool {}", pool.size, pool.size, pool.name);
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

int Scal_Gaudi3::mapLBWBlocks()
{

    // for engines - map the DCCM (update address[0] in the stream struct)
    // for ARC farm schedulers map the 2 DCCM blocks + ACP (update the 2 addresses in the stream struct)
    // iterate over the streams, update the pi pointer and set the pi local value counter to 0

    // arc farm schedulers: map both parts of the DCCM for arc farm
    for (unsigned idx =0; idx <  c_scheduler_nr; idx++)
    {
        if(Scheduler * scheduler = getCore<Scheduler>(idx))
        {
            scheduler->dccmHostAddress = mapLBWBlock(scheduler->dccmDevAddress, c_arc_farm_dccm_size);
            uint64_t physicalAcpAddress = ((uint64_t)scheduler->dccmDevAddress + mmHD0_ARC_FARM_ARC0_ACP_ENG_BASE - mmHD0_ARC_FARM_ARC0_DCCM0_BASE);
            scheduler->acpHostAddress = mapLBWBlock(physicalAcpAddress, c_acp_block_size + c_af_block_size);
            static_assert(mmHD0_ARC_FARM_ARC0_ACP_ENG_BASE + c_acp_block_size == mmHD0_ARC_FARM_ARC0_AF_BASE);
            scheduler->afHostAddress = (void*)((uint8_t*)scheduler->acpHostAddress + c_acp_block_size);

            if (!scheduler->dccmHostAddress || !scheduler->acpHostAddress || !scheduler->afHostAddress)
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
                LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed core {} {}",__FUNCTION__, m_fd, idx, core->arcName);
                return SCAL_FAILURE;
            }
        }
    }

    // map sync managers
    for (auto & sm : m_syncManagers)
    {
        if (sm.baseAddr && sm.map2userSpace)
        {
            sm.objsHostAddress = (volatile uint32_t*)mapLBWBlock(sm.baseAddr, SyncMgrG3::getSmMappingSize());
            if (!sm.objsHostAddress)
            {
                LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed sm objs {} ",__FUNCTION__, m_fd, sm.smIndex);
                return SCAL_FAILURE;
            }

            if ((sm.smIndex & 0x1) == 0) // only even sms have CQs block
            {
                sm.glblHostAddress = (volatile uint32_t*)mapLBWBlock(sm.baseAddr + mmHD0_SYNC_MNGR_GLBL_BASE - mmHD0_SYNC_MNGR_OBJS_BASE, c_acp_block_size);
                if (!sm.glblHostAddress)
                {
                    LOG_ERR(SCAL,"{}: fd={} mapLBWBlock failed sm glbl {} ",__FUNCTION__, m_fd, sm.smIndex);
                    return SCAL_FAILURE;
                }
            }
            else
            {
                sm.glblHostAddress = nullptr;
            }
        }
    }

    for (auto & streamMapPair : m_streams)
    {
        auto & stream = streamMapPair.second;
        stream.localPiValue = 0;

        //points to the location of the pi register in the DCCM
        if (!m_use_auto_fetcher)
        {
            // use the  gaudi2 way (without auto fetcher)
            uint32_t offset = varoffsetof(gaudi3::block_arc_af_eng, af_dccm_stream_pi[stream.id]);
            stream.pi = (uint32_t*)((uint8_t*)stream.scheduler->afHostAddress + offset);
        }
        else
        {
            uint64_t afBaseAddr = (uint64_t)stream.scheduler->afHostAddress;
            uint32_t offset = varoffsetof(gaudi3::block_arc_af_eng, af_dccm_stream_addr[stream.id]);
            writeLbwReg((volatile uint32_t*)(afBaseAddr + offset), lower_32_bits((uint64_t)stream.scheduler->dccmHostAddress));// AF_DCCM_STREAM_ADDR
            // write the PI of the auto fetcher (new to gaudi3)
            offset = varoffsetof(gaudi3::block_arc_af_eng, af_host_stream_pi[stream.id]);
            stream.pi = (uint32_t*)((uint8_t*)stream.scheduler->afHostAddress + offset);
            stream.ccbBufferAlignment = c_ccb_buffer_alignment;
        }
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::openConfigChannel()
{
    if (!m_configChannel.m_stream)
    {
        m_configChannel.m_stream = new PDMAStream_Gaudi3();
        m_configChannel.m_stream->init(m_configChannel.m_qid, m_fd);
        if (!m_configChannel.m_stream)
        {
            LOG_ERR(SCAL,"{}: failed to create a pdma stream object. fd: {}", __FUNCTION__, m_fd);
            assert(0);
            return SCAL_OUT_OF_MEMORY;
        }
    }

    int ret = m_configChannel.init();
    return ret;
}

int Scal_Gaudi3::initDirectModePdmaChannels()
{
    LOG_TRACE(SCAL,"{}", __FUNCTION__);

    int      status = SCAL_SUCCESS;
    Qman::Program program;
    RegToVal regToVal;
    for (auto& directModePdmaChannelEntry : m_directModePdmaChannels)
    {
        status = initDirectModeSinglePdmaChannel(*(directModePdmaChannelEntry.second), // channel
                                                 regToVal, program);
    }

    for (auto& regToValPair : regToVal)
    {
        program.addCommand(MsgLong(regToValPair.first, regToValPair.second));
    }

    // Configure SM base addrs to all SPDMAs for PDMA msgShort
    for (uint64_t& spdmaMsgBaseAddr : m_spdmaMsgBaseAddrDb)
    {
        uint64_t currSpdmaMsgBaseAddr = spdmaMsgBaseAddr;
        for (scal_sm_base_addr_tuple_t& smBaseAddrTuple : m_smBaseAddrDb)
        {
            currSpdmaMsgBaseAddr += sizeof(uint64_t) * smBaseAddrTuple.spdmaMsgBaseIndex;

            uint32_t low  = (uint32_t)smBaseAddrTuple.smBaseAddr;
            uint32_t high = (uint32_t)(smBaseAddrTuple.smBaseAddr >> 32);
            program.addCommand(MsgLong(currSpdmaMsgBaseAddr, low));
            program.addCommand(MsgLong(currSpdmaMsgBaseAddr + sizeof(uint32_t), high));
        }
    }

    if (!m_configChannel.submitPdmaConfiguration(program))
    {
        LOG_ERR(SCAL,"{} Failed to submit PDMA-Channels configuration", __FUNCTION__);
        status = SCAL_FAILURE;
    }

    return status;
}

int Scal_Gaudi3::initDirectModeSinglePdmaChannel(DirectModePdmaChannel& directModePdmaChannel,
                                                 RegToVal&              regToVal,
                                                 Qman::Program&         prog)
{
    int status = SCAL_SUCCESS;

    PdmaChannelInfo  const* pdmaChannelInfo = nullptr;
    bool ret = pdmaName2PdmaChannelInfo(directModePdmaChannel.getPdmaEngineName(), pdmaChannelInfo);
    if (!ret)
    {
        LOG_ERR(SCAL,"{}: can't find pdma channel info for {}",
                __FUNCTION__, directModePdmaChannel.getPdmaEngineName());

        return SCAL_FAILURE;
    }

    CompletionGroupInterface* cg = directModePdmaChannel.getCompletionGroup();
    unsigned cqIndex = cg->globalCqIndex;

    directModePdmaChannel.setCounterHost(m_completionQueueCounters + cqIndex);
    directModePdmaChannel.setCounterMmuAddr(m_completionQueueCountersDeviceAddr + cqIndex * sizeof(uint64_t));

    // init stream
    directModePdmaChannel.getStream()->init(pdmaChannelInfo->engineId, m_fd);

    // init channel's CQ
    status = directModePdmaChannel.initChannelCQ();
    if (status != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL,"{}: QMAN-ID for pdma-channel {} failed to init channels-CQ's ISR",
                __FUNCTION__, pdmaChannelInfo->channelId);
        assert(0);

        return SCAL_FAILURE;
    }
    // init tdr if enabled
    if (cg->compQTdr.enabled)
    {
        configureTdrCq(prog, *cg);
        // configure the monitors associated with this tdr
        // unlike regular cq, the direct tdr monitors are not configured in configureMonitors()
        configureTdrMon(prog, cg);
    }

    // Set priority
    {
        unsigned priority = directModePdmaChannel.getStream()->getPriority();
        const unsigned priorityOffset = offsetof(gaudi3::block_pdma_ch_b, ch_priority);
        scal_assert_return(
                addRegisterToMap(regToVal, pdmaChannelInfo->baseAddrB + priorityOffset, priority),
                SCAL_INVALID_CONFIG,
                "{}: Direct-Mode PDMA-Channel priority register {:#x} was already configured to {:#x}"
                " and configured again to {:#x}",
                __FUNCTION__, pdmaChannelInfo->baseAddrB, regToVal[pdmaChannelInfo->baseAddrB + priorityOffset], priority);
    }

    // Set no-alloc-cache
    {
        const unsigned hbwAxCacheRegOffset = offsetof(gaudi3::block_pdma_ch_b, hbw_axcache);
        const unsigned hbwAxCacheValue     = 0x33; // No-Allocate
        scal_assert_return(
                addRegisterToMap(regToVal, pdmaChannelInfo->baseAddrB + hbwAxCacheRegOffset, hbwAxCacheValue),
                SCAL_INVALID_CONFIG,
                "{}: Direct-Mode PDMA-Channel hbw_axcache register {:#x} was already configured to {:#x}"
                " and configured again to {:#x}",
                __FUNCTION__, pdmaChannelInfo->baseAddrB, regToVal[pdmaChannelInfo->baseAddrB + hbwAxCacheRegOffset],
                hbwAxCacheValue);
    }

    LOG_TRACE(SCAL,"{} channel name {} qmanId {} has been initialized",
              __FUNCTION__, pdmaChannelInfo->name, pdmaChannelInfo->engineId);

    return status;
}

int Scal_Gaudi3::closeConfigChannel()
{
    m_configChannel.deinit();

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::configurePdmaPtrsMode()
{
    int ret = m_configChannel.configurePdmaPtrsMode();
    return ret;
}

int Scal_Gaudi3::checkCanary()
{
    // iterate over the active cores and read the canary registers. Make sure they are all 0, return error otherwise
    int rc = SCAL_SUCCESS;
    for (unsigned idx =0; idx <  c_cores_nr; idx++)
    {
        Scheduler * core = getCore<Scheduler>(idx);
        if(core)
        {
            uint64_t canaryAddress = 0;
            uint32_t canaryValue = 1;
            if (core->isScheduler)
            {
                canaryAddress = offsetof(sched_registers_t, canary) + (uint64_t)(core->dccmHostAddress);
            }
            else
            {
                if (core->name.find("cme") == 0)
                {
                    canaryAddress = offsetof(cme_registers_t, canary) + (uint64_t)(core->dccmHostAddress);
                }
                else
                {
                    canaryAddress = offsetof(engine_arc_reg_t, canary) + (uint64_t)(core->dccmHostAddress);
                }
            }
            readLbwMem( (void*)&canaryValue, (volatile void*) canaryAddress, sizeof(uint32_t));
            if(canaryValue !=0)
            {
                LOG_ERR(SCAL,"{}: canary of {} (core id {}) is {} !=0", __FUNCTION__, core->name, core->cpuId, canaryValue);
                assert(0);
                rc = SCAL_FAILURE;
            }
        }
    }

    return rc;
}

int Scal_Gaudi3::configureSMs()
{
    // configure the QMANs
    int ret;

    // configure the active CQs
    ret = configureCQs();
    if (ret != SCAL_SUCCESS) return ret;

    // Configure Special monitors
    ret = configureMonitors();
    if (ret != SCAL_SUCCESS) return ret;

    if (m_arc_fw_synapse_config.sync_scheme_mode == ARC_FW_GAUDI3_SYNC_SCHEME)
    {
        ret = configureLocalMonChain();
        if (ret != SCAL_SUCCESS) return ret;
    }
    return SCAL_SUCCESS;

}

int Scal_Gaudi3::configureTdrCq(Qman::Program & prog, CompletionGroupInterface &cg)
{
    CompQTdr& compQTdr = cg.compQTdr;
    if (!compQTdr.enabled) return SCAL_SUCCESS;

    // Clear the SOB
    uint64_t syncObjAddress = SobG3::getAddr(compQTdr.sosPool->smBaseAddr, compQTdr.sos);

    uint32_t reg = SobG3::buildEmpty();
    prog.addCommand(MsgLong(syncObjAddress, reg)); // clear the SOB

    LOG_INFO_F(SCAL, "tdr SM-Index {} sos {} writing {:#x} to addr {:#x}",
               compQTdr.sosPool->smIndex, compQTdr.sos, reg, syncObjAddress);

    // Configure the CQ
    auto     cqIdx        = compQTdr.cqIdx;
    uint32_t cq_offset_64 = compQTdr.globalCqIndex * sizeof(uint64_t);
    //
    uint64_t cqCompletionQueueCountersDeviceAddr = cq_offset_64 + m_completionQueueCountersDeviceAddr;

    compQTdr.enginesCtr   = m_completionQueueCounters + cq_offset_64/ sizeof(uint64_t);
    uint64_t dcoreIndex   = compQTdr.monPool->smIndex & (~(1));
    uint64_t smGlobalBase = SyncMgrG3::getSmBase(dcoreIndex) + (mmHD0_SYNC_MNGR_GLBL_BASE - mmHD0_SYNC_MNGR_OBJS_BASE);

    // Set the address of the completion group counter (LSB)
    uint64_t cq_base_addr_l = smGlobalBase + varoffsetof(gaudi3::block_sob_glbl, cq_base_addr_l[cqIdx]);
    prog.addCommand(
        MsgLong(cq_base_addr_l, lower_32_bits(cqCompletionQueueCountersDeviceAddr)));

    // Set the address of the completion group counter (MSB)
    uint64_t cq_base_addr_h = smGlobalBase + varoffsetof(gaudi3::block_sob_glbl, cq_base_addr_h[cqIdx]);
    prog.addCommand(
        MsgLong(cq_base_addr_h, upper_32_bits(cqCompletionQueueCountersDeviceAddr)));

    // set the size of the counter (to 8 bytes)
    uint64_t cq_size_log2addr = smGlobalBase + varoffsetof(gaudi3::block_sob_glbl, cq_size_log2[cqIdx]);
    prog.addCommand(
        MsgLong(cq_size_log2addr, c_cq_size_log2));

    // Set it to increment
    prog.addCommand(
        MsgLong(smGlobalBase + varoffsetof(gaudi3::block_sob_glbl, cq_inc_mode[cqIdx]), 0x0));

    prog.addCommand(
        MsgLong(smGlobalBase + varoffsetof(gaudi3::block_sob_glbl, cq_inc_mode[cqIdx]), 0x1));

    // Note, unlike the "regular' cq, for the dtr-cq, we are not setting an interrupt (lbw_addr_l, lbw_addr_h, lbw_data), they remain 0

    LOG_INFO_F(SCAL, "tdr init, cg {} cqIdx {} globalCqIndex {}  cq_base_addr h/l {:#x} {:#x} cqCompletionQueueCountersDeviceAddr {:#x} ctr {:#x}",
               cg.name, cqIdx, compQTdr.globalCqIndex, cq_base_addr_h, cq_base_addr_l, cqCompletionQueueCountersDeviceAddr, TO64(compQTdr.enginesCtr));

    return SCAL_SUCCESS;
}

void Scal_Gaudi3::configureCQ(Qman::Program &prog, const uint64_t smBase, const unsigned cqIdx, const uint64_t ctrAddr, const unsigned isrIdx)
{
        prog.addCommand(
            MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, cq_base_addr_l[cqIdx]), lower_32_bits(ctrAddr)));

        prog.addCommand(
            MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, cq_base_addr_h[cqIdx]), upper_32_bits(ctrAddr)));

        prog.addCommand(
            MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, cq_size_log2[cqIdx]), c_cq_size_log2));

        // Configure CQ LBW Address
        const uint64_t msix_db_reg = mmD0_PCIE_MSIX_BASE;

        if (isrIdx != scal_illegal_index)// SW-82256
        {
            prog.addCommand(
                MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, lbw_addr_l[cqIdx]), lower_32_bits(msix_db_reg)));

            prog.addCommand(
                MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, lbw_addr_h[cqIdx]), upper_32_bits(msix_db_reg)));

            prog.addCommand(
                MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, lbw_data[cqIdx]), isrIdx));
        }
        prog.addCommand(
            MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, cq_inc_mode[cqIdx]), 0x0));

        prog.addCommand(
            MsgLong(smBase + varoffsetof(gaudi3::block_sob_glbl, cq_inc_mode[cqIdx]), 0x1));
        LOG_DEBUG(SCAL, "configureCQ() cqIdx={} deviceAddress={:#x} msix_db_reg={:#x} isrIdx={:#x}",
                cqIdx, ctrAddr, msix_db_reg, isrIdx);

}

int Scal_Gaudi3::configureCQs()
{
    // configure the active CQs.
    // For each CQ - update the isrIdx and Counter in the Completion Group Struct
    // send QMAN programs to configure the active CQs:
    // - set the PQ mode
    // - set the PQ address
    // - set the PQ ISR interupt service routine
    // - set the PQ size to 1
    std::map<unsigned, Qman::Program> qid2prog;

    for (auto& completionGroupIter : m_completionGroups)
    {
        auto& completionGroup = completionGroupIter.second;
        Qman::Program & prog = qid2prog[completionGroup.qmanID];

        auto cqIdx = completionGroup.cqIdx;
        uint32_t cq_offset_64 = (cqIdx + (completionGroup.syncManager->dcoreIndex * c_cq_ctrs_in_hdcore)) * sizeof(uint64_t);
        uint64_t cqCompletionQueueCountersDeviceAddr = cq_offset_64 + m_completionQueueCountersDeviceAddr;

        completionGroup.pCounter = m_completionQueueCounters + cq_offset_64/ sizeof(uint64_t) ;
        uint64_t smBase = SyncMgrG3::getSmBase(completionGroup.monitorsPool->smIndex);
        if (!smBase)
        {
            assert(0);
            return SCAL_INVALID_PARAM;
        }
        uint64_t smGlobalBase = smBase + mmHD0_SYNC_MNGR_GLBL_BASE - mmHD0_SYNC_MNGR_OBJS_BASE;

        configureCQ(prog, smGlobalBase, cqIdx, cqCompletionQueueCountersDeviceAddr, completionGroup.isrIdx);

        if (completionGroup.compQTdr.enabled)
        {
            configureTdrCq(prog, completionGroup);
        }
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
void Scal_Gaudi3::configureMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData)
{
    configureOneMonitor(prog, monIdx, smBase, configValue, payloadAddress, payloadData);
}
void Scal_Gaudi3::configureOneMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData)
{
    // send QMAN command MsgLong to config this monitor
    MonitorG3 monitor(smBase, monIdx, 0);
    Monitor::ConfInfo confInfo{.payloadAddr = payloadAddress, .payloadData = payloadData, .config = configValue};
    monitor.configure(prog, confInfo);
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









            G A U D I 3

               *  monitor can support up to 16 messages (wr_num has 4 bits)
                  so we don't need the complicated chain reaction table above

  */
// clang-format on
void Scal_Gaudi3::AddIncNextSyncObjectMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx)
{
    // The new message should increment the value of the next SO in the group by 1
    // The message payload should be set to SO atomic inc (0x80000001).
    // The message address should be set to point the next SO in the group
    uint32_t mc             = MonitorG3::buildConfVal(soIdx, 0, CqEn::off, LongSobEn::off, LbwEn::off);
    unsigned nextSoIndex    = cq->sosBase + ((soIdx - cq->sosBase + 1) % cq->sosNum);
    uint64_t payloadAddress = SobG3::getAddr(smBase, nextSoIndex); // address of next so
    uint32_t payloadData    = 0x80000001;
    LOG_DEBUG(SCAL, "AddIncNextSyncObjectMonitor() cq idx {} config mon {} to inc sob {}", cq->cqIdx, monIdx, nextSoIndex);
    configureOneMonitor(prog, monIdx, smBase, mc, payloadAddress, payloadData);
    if (cq->nextSyncObjectMonitorId == -1)
    {
        cq->nextSyncObjectMonitorId = monIdx - cq->monBase;
    }
}

// The monitor expiration messages are fired also to the DCCM queues of the slave schedulers.
// Handling of the expiration message in the slave schedulers is limited only to incrementing the completion queue counter
// and un-blocking any stream that is waiting for it.
// The slave schedulers do not rearm the monitors, nor they decrement the sync object.
void Scal_Gaudi3::AddSlaveXDccmQueueMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned slaveIndex)
{

    // Address of scheduler DCCM messageQ,
    // value according to struct(Mon info of the master, so scheduler can "rearm" the master monitor)
    uint32_t mc = MonitorG3::buildConfVal(0, 0, CqEn::off, LongSobEn::off, LbwEn::off);
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
    pd.mon_id           = monIdx + (cq->syncManager->smIndex * c_max_monitors_per_sync_manager);
    pd.mon_sm_id        = cq->monitorsPool->smIndex;
    pd.mon_index        = 0;
    uint32_t pdRaw      = 0;
    memcpy(&pdRaw, &pd, sizeof(uint32_t));
    configureOneMonitor(prog, monIdx, smBase, mc, payloadAddress, pdRaw);
}

unsigned Scal_Gaudi3::AddCompletionGroupSupportForHCL(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx)
{
    //
    // Mon 0,1, and last are always the same (the original 3 ...)
    // and are configured by the caller
    // so  here we config the monitors in between them according to the table above
    //
    //  Gaudi3 -- no need for "Demi Masters" and such tricks. Monitor now support up to 16 messages

    unsigned NumSlaves   = cq->slaveSchedulers.size(); // Num monitors to add
    // when called, monIdx should be firstMonIdx + 2
    for (unsigned idx = 0; idx < NumSlaves; idx++)
    {
            AddSlaveXDccmQueueMonitor(prog, cq, smBase, monIdx + idx, idx); // Slave (idx) DCCM queue
    }
    if (cq->force_order)
        AddIncNextSyncObjectMonitor(prog, cq, smBase, monIdx+NumSlaves, soIdx);
    //
    return monIdx + NumSlaves + (unsigned)cq->force_order; // should be the index of the last monitor to add
}

void Scal_Gaudi3::configureTdrMon(Qman::Program & prog, const CompletionGroupInterface *cg)
{
    const CompQTdr& compQTdr = cg->compQTdr;

    unsigned monSmIndex = compQTdr.monPool->smIndex;
    uint64_t monSmBase  = compQTdr.monPool->smBaseAddr;

    uint64_t sobjAddr   = SobG3::getAddr(monSmBase, compQTdr.sos);
    uint64_t monArmAddr = MonitorG3(monSmBase, compQTdr.monitor).getRegsAddr().arm;

    // Monitor 1) Dec sob
    sync_object_update decSO;
    //
    decSO.raw = 0;
    //
    decSO.so_update.sync_value = (uint16_t)(-1);
    decSO.so_update.mode       = 1;

    // Monitor 2) Rearm monitor
    unsigned monSod  = 1; // Data to compare SOB against

    uint32_t monArmRawValue = MonitorG3::buildArmVal(cg->compQTdr.sos, monSod);

    // Monitor 3) Inc Cq
    uint64_t cqIndex = compQTdr.cqIdx;
    uint32_t data3   = 1; // Inc Cq

    // Monitor config
    const unsigned numOfMsgs = 3; // #MSGs
    // The master is monitoring a regular sob
    uint32_t monConfigRaw  = MonitorG3::buildConfVal(cg->compQTdr.sos, numOfMsgs - 1, CqEn::off, LongSobEn::off, LbwEn::off, monSmIndex);

    configureOneMonitor(prog, compQTdr.monitor + 0, monSmBase, monConfigRaw, sobjAddr, decSO.raw);
    configureOneMonitor(prog, compQTdr.monitor + 1, monSmBase, monConfigRaw, monArmAddr, monArmRawValue);

    monConfigRaw  = MonitorG3::buildConfVal(0, 0, CqEn::on, LongSobEn::off, LbwEn::off, monSmIndex);
    configureOneMonitor(prog, compQTdr.monitor + 2, monSmBase, monConfigRaw, cqIndex, data3);

    LOG_INFO_F(SCAL, "tdr arm {} addr {:#x} data {:#x}", compQTdr.monitor, monArmAddr, monArmRawValue);
    prog.addCommand(MsgLong(monArmAddr, monArmRawValue));
}

void Scal_Gaudi3::configSfgSyncHierarchy(Qman::Program & prog, const CompletionGroup *cg)
{
    Scal::configSfgSyncHierarchy<sfgMonitorsHierarchyMetaDataG3>(prog, cg, c_maximum_messages_per_monitor);
}

void Scal_Gaudi3::configFenceMonitorCounterSMs(Qman::Program & prog, const CompletionGroup *cg)
{
    const unsigned curMonIdx   = cg->monBase + cg->monNum;
    const unsigned smIdx       = cg->syncManager->smIndex;
    const int      dcoreIdx    = cg->syncManager->dcoreIndex;
    const unsigned sobIdx      = cg->sosBase;
    const unsigned cqIdx       = cg->cqIdx;
    const uint64_t smBaseAddr  = cg->syncManager->baseAddr;
    const bool     hasInterrupts = cg->isrIdx != scal_illegal_index;
    const uint32_t mon1 = curMonIdx + 0; // Master monitor for dec sob
    const uint32_t mon2 = curMonIdx + 1;
    const uint32_t mon3 = curMonIdx + 2;

    LOG_INFO(SCAL,"{}: =========================== Fence Counter Configuration =============================", __FUNCTION__);
    LOG_INFO(SCAL,"{}: Config master mon HD{}_SM{}_MON_{} with 3 messages: dec SOB HD{}_SM{}_SOB_{}, inc cq CQ{}, ReArm monitor SM{}_MON_{}",
             __FUNCTION__, dcoreIdx, smIdx, mon1, dcoreIdx, smIdx, sobIdx, cqIdx, smIdx, mon1);

    /* ====================================================================================== */
    /*                                  Create a monitor group                                */
    /* ====================================================================================== */

    // Need 3 writes (dec sob, inc CQ, Arm master monitor)
    uint32_t monConfig = MonitorG3::buildConfVal(sobIdx, 2, CqEn::off, LongSobEn::off, LbwEn::off);

    /* ====================================================================================== */
    /*                              Add massages to monitor groups                            */
    /* ====================================================================================== */

    // msg0:   decrement by 1
    uint32_t decSobPayloadData = SobG3::buildVal(-1, SobLongSobEn::off, SobOp::inc);
    LOG_DEBUG(SCAL,"{}: Configure HD{}_SM{}_MON_{} to dec sob HD{}_SM{}_SOB_{} by 1. payload: {:#x}",
                   __FUNCTION__, dcoreIdx, smIdx, mon1, dcoreIdx, smIdx, sobIdx, decSobPayloadData);
    configureOneMonitor(prog, mon1, smBaseAddr, monConfig, SobG3::getAddr(smBaseAddr, sobIdx), decSobPayloadData);

    // msg1:     ReArm master monitor
    uint32_t armMonPayloadData = MonitorG3::buildArmVal(sobIdx, 1);

    // Rearm self
    uint64_t masterMonAddr = MonitorG3(smBaseAddr, mon1).getRegsAddr().arm;
    LOG_DEBUG(SCAL,"{}: Configure HD{}_SM{}_MON_{} to rearm self", __FUNCTION__, dcoreIdx, smIdx, mon3);
    configureOneMonitor(prog, mon3, smBaseAddr, monConfig, masterMonAddr, armMonPayloadData);

    // msg2:    Increment CQ by 1
    LbwEn lbwEn = hasInterrupts ? LbwEn::on : LbwEn::off;
    monConfig = MonitorG3::buildConfVal(sobIdx, 2, CqEn::on, LongSobEn::off, lbwEn);
    LOG_DEBUG(SCAL,"{}: Configure HD{}_SM{}_MON_{} to inc cg's CQ {}", __FUNCTION__, dcoreIdx, smIdx, mon2, cqIdx);
    configureOneMonitor(prog, mon2, smBaseAddr, monConfig, (uint64_t)cqIdx, 0x1);

    LOG_DEBUG(SCAL,"{}: Arm master HD{}_SM{}_MON_{} with payload data {:#x}, HD{}_SM{}_SOB_{}",
              __FUNCTION__, dcoreIdx, smIdx, mon1, armMonPayloadData, dcoreIdx, smIdx, sobIdx);
    prog.addCommand(MsgLong(masterMonAddr, armMonPayloadData));

}

int Scal_Gaudi3::configureMonitors()
{
    std::map<unsigned, Qman::Program> qid2prog;

    for (unsigned smIndex=0; smIndex < c_sync_managers_nr; smIndex++)
    {
        // skipping sync manager '1' per hdcore
        for (auto cq : m_syncManagers[smIndex].completionGroups)
        {
            Qman::Program & prog = qid2prog[cq->qmanID];
            unsigned mon_depth = cq->monNum / cq->actualNumberOfMonitors;
            uint64_t smBase = cq->monitorsPool->smBaseAddr;
            uint64_t longSoSmBase = cq->longSosPool->smBaseAddr;
            uint64_t syncObjAddress = SobG3::getAddr(longSoSmBase, cq->longSoIndex);
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
                LOG_INFO(SCAL, "configureMonitors() cq {} {} monIdx={} soIdx={} actualNumberOfMonitors={} cq->longSoIndex={}",
                         cq->cqIdx, cq->name.c_str(), monIdx, soIdx, cq->actualNumberOfMonitors, cq->longSoIndex);
                unsigned firstMonIdx = monIdx;
                if (cq->force_order && mIdx == 0)
                {
                    //  As part of the Sync Manager configuration, SCAL should set the value of the first SO in the completion group to “1”
                    //  in all the groups in which the in-order completion feature is enabled.

                    prog.addCommand(MsgLong(SobG3::getAddr(smBase, soIdx), 1));
                    LOG_INFO(SCAL, "force-order is true so setting SOB {} to 1", soIdx);
                }

                unsigned numWrites = std::min(cq->actualNumberOfMonitors, c_maximum_messages_per_monitor);
                uint32_t mc        = MonitorG3::buildConfVal(soIdx, numWrites - 1, CqEn::off, LongSobEn::off, LbwEn::off);
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
                configureOneMonitor(prog, monIdx, smBase, mc, syncObjAddress, 0x81000001);

                //
                //
                // 2nd monitor- cq_en configuration :
                //
                //
                // CQ Enable ---- CQ_EN = 1
                // LBW_EN = 1
                LbwEn lbwEn = (cq->isrIdx != scal_illegal_index) ? LbwEn::on : LbwEn::off;
                mc          = MonitorG3::buildConfVal(0, 0, CqEn::on, LongSobEn::off, lbwEn);
                // the payload address:
                // In case CQ_EN=1, the 6 LSB of the field points to the CQ structure
                // Address = CQ_ID, Data = 1 (CQ will treat 1 as CQ_COUNTER++)
                configureOneMonitor(prog, monIdx + 1, smBase, mc, (uint64_t)cq->cqIdx, 0x1);
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
                mc = 0;
                // The 64 bit address to write the completion message to in case CQ_EN=0.
                uint64_t payloadAddress = cq->scheduler->dccmMessageQueueDevAddress;
                // payload data
                struct sched_mon_exp_comp_fence_t pd;
                memset(&pd, 0x0, sizeof(uint32_t));
                pd.opcode = MON_EXP_COMP_FENCE_UPDATE;
                pd.comp_group_index = cq->idxInScheduler;
                pd.mon_id           = firstMonIdx;
                pd.mon_sm_id        = cq->monitorsPool->smIndex;
                pd.mon_index = mIdx;
                uint32_t pdRaw = 0;
                memcpy(&pdRaw,&pd,sizeof(uint32_t));
                LOG_DEBUG(SCAL,"{}: config monIdx {} scheduler DCCM Q payload monId={} smIndex={} comp_group_index={}",
                    __FUNCTION__, monIdx, firstMonIdx, smIndex, cq->idxInScheduler);
                configureOneMonitor(prog, monIdx, smBase, mc, payloadAddress, pdRaw);

                // arm to CMAX  COMP_SYNC_GROUP_CMAX_TARGET
                // -->   keep this last  <--
                uint32_t ma = MonitorG3::buildArmVal(soIdx, COMP_SYNC_GROUP_CMAX_TARGET, CompType::EQUAL);


                prog.addCommand(MsgLong(MonitorG3(smBase, firstMonIdx, 0).getRegsAddr().arm, ma));

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

/************************************************************************************************************/
/*
*                                          For each SO SET For each HDcore!
*
* This functionality is supported for clusters with localDup enabled.
*
* For example TPC.
* TPC Local SOBs:
*
*    ------------------ ------------------                                 ------------------
*    | TPC0 local sob | | TPC1 local sob |.................................| TPC7 local sob |
*    ------------------ ------------------                                 ------------------
*
* Monitor chain with 10 payloads:
* - MSG0: Increment central TPC SOB
* - MSG1 to MSG8: Auto decrement (by 1) TPCs local sobs
* - MSG9: Rearm Self
*
*    --------------------------------------
*    | MON group for Local SOBs           |
*    |                                    |
*    | msg0: Increment central TPC SOB    |
*    | msg1: TPC0 local sob Auto dec      |
*    | msg2: TPC1 local sob Auto dec      |
*    | msg3: TPC2 local sob Auto dec      |
*    | msg4: TPC3 local sob Auto dec      |
*    | msg5: TPC4 local sob Auto dec      |
*    | msg6: TPC5 local sob Auto dec      |
*    | msg7: TPC6 local sob Auto dec      |
*    | msg8: TPC7 local sob Auto dec      |
*    | msg9: Rearm Self                   |
*    --------------------------------------
*
* Central TPC SOB (represents 8 TPCs):
*    -------------------
*    | Central TPC SOB |
*    -------------------
*
*************************************************************************************************************/

int Scal_Gaudi3::configureLocalMonChain()
{
    std::map<unsigned, Qman::Program> qid2prog;

    for (const auto& cluster : m_computeClusters)
    {
        if (cluster->localDup == false)
        {
            continue; // localDup is unsupported for this cluster - skip
        }

        unsigned totalEnginesNum = 0;

        /* ====================================================================================== */
        /*        For each HDcore configure a chain of monitors for the local Engine's sobs       */
        /* ====================================================================================== */
        for (unsigned hdIndex : m_hdcores)
        {
            // Get SO SET group
            SyncObjectsSetGroupGaudi3* sg = static_cast<SyncObjectsSetGroupGaudi3*>(m_soSetsGroups["compute_sos_sets"].get());

            unsigned numOfEngsInHd = cluster->enginesPerHDCore[hdIndex];
            if (numOfEngsInHd == 0)
            {
                continue;  // No engines in this hdcore - skip
            }
            // Since we have at leats 1 engines in this HD we expect to get +1 central sob signal
            cluster->numOfCentralSignals++;

            // Get the first engine qman ID in each hdcore
            unsigned qmanId = cluster->engines[totalEnginesNum]->qmanID;
            Qman::Program &prog = qid2prog[qmanId];
            totalEnginesNum += numOfEngsInHd;

            // Init sobs and mons data
            unsigned localMonBaseId    = sg->localSoSetResources[hdIndex].localMonitorsPool->baseIdx;
            unsigned localSobBaseId    = sg->localSoSetResources[hdIndex].localSosPool->baseIdx;

            unsigned offset = getEngineTypeOffsetInSet(cluster->type);
            if (offset == NUM_OF_CORE_TYPES)
            {
                LOG_ERR(SCAL,"{} failed to get offset for cluster {}", __FUNCTION__, cluster->name);
                return SCAL_FAILURE;
            }
            unsigned centralSobId      = sg->sosPool->baseIdx + offset + cluster->numOfCentralSignals - 1;
            unsigned centralSoSetNum   = sg->numSets;
            unsigned centralSoSetSize  = sg->setSize;

            uint64_t smBase            = sg->localSoSetResources[hdIndex].localMonitorsPool->smBaseAddr;
            unsigned smIndex           = sg->localSoSetResources[hdIndex].localMonitorsPool->smIndex;

            // For each SO SET
            for (unsigned soSetIndex = 0; soSetIndex < centralSoSetNum; soSetIndex++)
            {
                unsigned localSobId        = localSobBaseId;
                unsigned localMonId        = localMonBaseId;

                LOG_INFO(SCAL,"{}: Config local sobs monitor chain for cluster {} HD{} SO_SET {}",
                            __FUNCTION__, cluster->name, hdIndex, soSetIndex);
                LOG_INFO(SCAL,"{}: Config master mon HD{}_SM{}_MON_{} with up to {} messages: "
                            "inc by 1 HD{}_SM{}_SOB_{}, dec {} seq SOBs by 1 starting from HD{}_SM{}_SOB_{} and rearm self",
                            __FUNCTION__, sg->localSoSetResources[hdIndex].localSosPool->dcoreIndex,
                            (smIndex % c_sync_managers_per_hdcores == 0) ? 0 : 1, localMonBaseId,
                            numOfEngsInHd + 2, sg->sosPool->smIndex / c_sync_managers_per_hdcores,
                            (sg->sosPool->smIndex % c_sync_managers_per_hdcores == 0) ? 0 : 1,
                            centralSobId, numOfEngsInHd, sg->localSoSetResources[hdIndex].localSosPool->dcoreIndex,
                            (smIndex % c_sync_managers_per_hdcores == 0) ? 0 : 1, localSobBaseId);

                // Create a local monitors chain

                /* ====================================================================================== */
                /*                                  Create a monitor group                                */
                /* ====================================================================================== */
                uint32_t groupMonitor = MonitorG3::buildConfVal(localSobBaseId, 1 + numOfEngsInHd, CqEn::off, LongSobEn::off, LbwEn::off);

                /* ====================================================================================== */
                /*                              Add massages to monitor group                             */
                /* ====================================================================================== */

                // MSG0: Master monitor to increment SO in Central SO set
                uint64_t payloadAddress = SobG3::getAddr(sg->sosPool->smBaseAddr, centralSobId);
                uint32_t payloadData    = 0x80000001;
                configureOneMonitor(prog, localMonId, smBase, groupMonitor, payloadAddress, payloadData);
                localMonId++;

                // MSG1 to MSG8: Up to 8 monitors to auto decrement by 1 for all Engine's local sobs
                sync_object_update syncObjUpdate;
                unsigned decVal                    = 1;
                syncObjUpdate.raw                  = 0;
                syncObjUpdate.so_update.sync_value = (-decVal);
                syncObjUpdate.so_update.mode       = 1;

                int8_t mask = 0xFF;
                for (unsigned engIdInHd = 0 ; engIdInHd < numOfEngsInHd; engIdInHd++)
                {
                    mask &= ~(1 << (localSobId % c_so_group_size));
                    payloadAddress = SobG3::getAddr(smBase, localSobId);
                    configureOneMonitor(prog, localMonId, smBase, groupMonitor, payloadAddress, syncObjUpdate.raw);
                    localMonId++;
                    localSobId++;
                }

                // MSG9: Rearm self
                uint32_t monArm        = MonitorG3::buildArmVal(localSobBaseId, 1, mask);
                uint64_t masterMonAddr = MonitorG3(smBase, localMonBaseId).getRegsAddr().arm;
                configureOneMonitor(prog, localMonId, smBase, groupMonitor, masterMonAddr, monArm);

                /* ====================================================================================== */
                /*                                  ARM the monitor group                                 */
                /* ====================================================================================== */

                prog.addCommand(MsgLong(masterMonAddr, monArm));

                /* ====================================================================================== */
                /*                                  Update for next SO SET                                */
                /* ====================================================================================== */

                localMonBaseId += c_soset_local_monitors_nr;
                localSobBaseId += c_soset_local_sobs_nr;
                centralSobId   += centralSoSetSize;

            } // SO SET loop
        } // HDcore loop
    } // Cluster loop

    Qman::Workload workload;
    for (auto & qidProgPair : qid2prog)
    {
        workload.addProgram(qidProgPair.second, qidProgPair.first);
    }

    if (!submitQmanWkld(workload))
    {
        LOG_ERR(SCAL,"{} failed submit workload of configureLocalMonChain", __FUNCTION__);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::loadFWImage()
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

int Scal_Gaudi3::loadFWImagesFromFiles(ImageMap &images)
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
                    pfwImage->image_dccm_size = (core->cpuId < c_scheduler_nr ? c_scheduler_image_dccm_size : c_engine_image_dccm_size);
                    pfwImage->image_hbm_size = c_image_hbm_size;
                    bool res = loadFWImageFromFile_gaudi3(m_fd, imagePath,
                        pfwImage->image_dccm_size, pfwImage->image_hbm_size,
                        pfwImage->dccm, pfwImage->hbm, &metaData);
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

int Scal_Gaudi3::LoadFWHbm(const ImageMap &images)
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

    bool foundEDMA = false;
    std::set<unsigned> activeDMAs;
    /*  SW-82992  Not able to use binary multiple times in gaudi3 Json
        the LinDma commands below do not work. Needs to be investigated.

    for (const auto & core : m_cores)
    {
        if ((core) && (activeDMAs.find(core->qmanID) == activeDMAs.end()))
        {
            if ((core->qmanID == GAUDI3_HDCORE1_ENGINE_ID_EDMA_0) ||
                (core->qmanID == GAUDI3_HDCORE1_ENGINE_ID_EDMA_1) ||
                (core->qmanID == GAUDI3_HDCORE3_ENGINE_ID_EDMA_0) ||
                (core->qmanID == GAUDI3_HDCORE3_ENGINE_ID_EDMA_1) ||
                (core->qmanID == GAUDI3_HDCORE4_ENGINE_ID_EDMA_0) ||
                (core->qmanID == GAUDI3_HDCORE4_ENGINE_ID_EDMA_1) ||
                (core->qmanID == GAUDI3_HDCORE6_ENGINE_ID_EDMA_0) ||
                (core->qmanID == GAUDI3_HDCORE6_ENGINE_ID_EDMA_1))
            {
                activeDMAs.insert(core->qmanID);
                foundEDMA = true;
            }
        }
    }
    */
    if (activeDMAs.empty())
    {
        activeDMAs.insert(GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_1);
    }

    std::vector<Qman::Program> dmaProgs(activeDMAs.size());
    Qman::Workload workload;
    auto progIt = dmaProgs.begin();

    std::map<std::string, uint64_t> imageAddresses;
    DeviceDmaBuffer dmaBuffs[c_cores_nr];
    for (unsigned coreIdx = 0; coreIdx < c_cores_nr; coreIdx++)
    {
        ArcCore * core = getCore<ArcCore>(coreIdx);
        if (core)
        {
            uint64_t hbmDstAddress = m_coresBinaryDeviceAddress + (coreIdx * c_image_hbm_size);
            if (imageAddresses.find(core->imageName) == imageAddresses.end() || !foundEDMA)
            {
                // first download images from Host to HBM using PDMA
                dmaBuffs[coreIdx].init(hbmDstAddress, this, c_image_hbm_size);
                auto hostAddr = dmaBuffs[coreIdx].getHostAddress();
                if (hostAddr == nullptr)
                {
                    assert(0);
                    return SCAL_OUT_OF_MEMORY;
                }
                memcpy(hostAddr, (void*)images.at(core->imageName).hbm, c_image_hbm_size);
                dmaBuffs[coreIdx].commit(&workload);
                imageAddresses[core->imageName] = hbmDstAddress;
            }
            else
            {
                // second transfer from original image copy on the HBM to the other engines using this image
                // unless there are no EDMAs
                progIt->addCommand(LinDma(hbmDstAddress, imageAddresses[core->imageName], c_image_hbm_size));
                if ((++progIt) == dmaProgs.end()) progIt = dmaProgs.begin();
            }
        }
    }

    progIt = dmaProgs.begin();
    for (const unsigned qid : activeDMAs)
    {
        if (!progIt->getSize())
        {
            break;
        }

        if (!workload.addProgram(*(progIt++), qid))
        {
            LOG_ERR(SCAL,"{}: fd={} addProgram failed. qid={}", __FUNCTION__, m_fd, qid);
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

// helper function for writeDCCMs, generates 8k msgShort command for copy fw to 32 kb dccm
void Scal_Gaudi3::genFwDccmLoadProgram(Qman::Program &prog, const uint8_t * dccmFWBuff, const unsigned dccmSize)
{
    uint32_t* binPtr = (uint32_t*) dccmFWBuff;

    for(unsigned i = 0; i < (dccmSize/sizeof(uint32_t)); i++)
    {
        MsgShort msgShort(c_message_short_base_index,i*(sizeof(uint32_t)),*binPtr);
        prog.addCommand(msgShort);
        binPtr++;
    }
}


int Scal_Gaudi3::LoadFWDccm(const ImageMap &images)
{
    int rc = SCAL_SUCCESS;
    Qman::Workload wkld;

    // save location of workload per image name for stage 2
    std::map<std::string, std::pair<unsigned, unsigned>> imageFWOffsetAndSizeMap;

    // step 1: generate programs to copy Fw from HBM to DCCM
    Qman::Program progDCCM;
    for (const auto & it : images)
    {
        unsigned offset = progDCCM.getSize();
        genFwDccmLoadProgram(progDCCM, it.second.dccm, it.second.image_dccm_size);
        imageFWOffsetAndSizeMap[it.first] = {offset, progDCCM.getSize()-offset};
    }

    // copy the programs to the device
    DeviceDmaBuffer dmaBuffer(m_globalPool, progDCCM.getSize());
    uint8_t* HostProgBuffPtr = (uint8_t*)dmaBuffer.getHostAddress();
    if(!HostProgBuffPtr)
    {
        LOG_ERR(SCAL,"{} aligned_alloc  host memory failed", __FUNCTION__);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }
    progDCCM.serialize(HostProgBuffPtr);
    rc = dmaBuffer.commit(&wkld);
    if (!rc) return SCAL_FAILURE;

    // step 2: for each core: schedulers and then engine arcs:
    // generate upper cp program that configures base register 3 to point to the current engine DCCM address
    for (unsigned idx = 0; idx < c_cores_nr; idx++)
    {
        ArcCore * core = getCore<ArcCore>(idx);
        if (core)
        {
            bool localMode;
            if(isLocal(core, localMode) != SCAL_SUCCESS)
            {
                LOG_ERR(SCAL,"{}, failed to query localMode", __FUNCTION__);
                assert(0);
                return SCAL_FAILURE;
            }

            Qman::Program prog0;

            // a.	WREG to config base3 of the lower CP to point the low part of the DCCM.
            uint64_t dccmAddress = localMode ? c_local_address : core->dccmDevAddress;
            uint16_t baseAddressLo = c_dccm_to_qm_offset + offsetof(gaudi3::block_qman,cp_msg_base_addr[(c_message_short_base_index * 2) + 0]);
            uint16_t baseAddressHi = c_dccm_to_qm_offset + offsetof(gaudi3::block_qman,cp_msg_base_addr[(c_message_short_base_index * 2) + 1]);

            prog0.addCommand(WReg32(baseAddressLo, lower_32_bits(dccmAddress)));
            prog0.addCommand(WReg32(baseAddressHi, upper_32_bits(dccmAddress)));

            // b. find the image to load
            auto it = imageFWOffsetAndSizeMap.find(core->imageName);
            if(it == imageFWOffsetAndSizeMap.end() )
            {
                LOG_ERR(SCAL,"{}, imageName {} in core {} not found", __FUNCTION__, core->imageName, core->cpuId);
                assert(0);
                return SCAL_FAILURE;
            }

            uint64_t progAddress = it->second.first + dmaBuffer.getDeviceAddress();
            unsigned progSize = it->second.second;

            prog0.addCommand(WReg32(c_dccm_to_qm_offset + offsetof(gaudi3::block_qman, cq_tsize), progSize));
            prog0.addCommand(WReg32(c_dccm_to_qm_offset + offsetof(gaudi3::block_qman, cq_ptr_lo), lower_32_bits(progAddress)));
            prog0.addCommand(WReg32(c_dccm_to_qm_offset + offsetof(gaudi3::block_qman, cq_ptr_hi), upper_32_bits(progAddress)));
            prog0.addCommand(WReg32(c_dccm_to_qm_offset + offsetof(gaudi3::block_qman, cq_ctl), 0));

            wkld.addProgram(prog0, core->qmanID);

            Qman::Program prog1;
            prog1.addCommand(Nop(true));
            wkld.addProgram(prog1, core->qmanID);
        }
    }

    if (!submitQmanWkld(wkld))
    {
        LOG_ERR(SCAL,"{}, wkld submit failed ", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::configSchedulers()
{
    int ret = SCAL_SUCCESS;
    DeviceDmaBuffer buffs[c_scheduler_nr];
    Qman::Workload workload;

    uint32_t coreIds[c_scheduler_nr]     = {};
    uint32_t coreQmanIds[c_scheduler_nr] = {};

    uint32_t counter = 0;
    unsigned qmanID = -1;
    for (unsigned idx = 0; idx < c_scheduler_nr; idx++)
    {
        Scheduler * core = getCore<Scheduler>(idx);
        if (core)
        {
            if (!core->isScheduler) continue;
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
            ret = createCoreConfigQmanProgram(idx, buffs[idx], program, &regToVal);
            if (ret != SCAL_SUCCESS) break;

            workload.addProgram(program, core->qmanID);
            qmanID = core->qmanID;
        }
    }

    if (ret == SCAL_SUCCESS && qmanID != -1)
    {
        Qman::Program program;
        // config the credits for the EDUP
        handleSarcCreditsToEdup(program);
        workload.addProgram(program, qmanID);
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

uint32_t Scal_Gaudi3::getSRAMSize() const
{
    return m_hw_ip.sram_size;
}

int Scal_Gaudi3::allocSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff)
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

    https://jira.habana-labs.com/browse/SW-115886   SW-115886 H9 - SCAL to program SARC credits to EDUP

    The default JSON should limit the compute SARC -> EDUPs (0, 2, 4, 6) to 450 messages.
    in comments: RTR_CRDT_SIZE register is 8bit field but due to some bug the maximum allowed value is 252.

    Each master IF maintain credits towards up to 14 EDUPs
    when sending transaction towards a DUP engine-
        1. if there are available credits, the HW will forward the request
        2. when the credits run out, the HW will halt the requests until credits are freed up
    Credit configuration sequence:

    for(i=0;i<8;i++){
        DUP_CRDT_RTR_CRDT_SIZE[i] = Credit_Value
        DUP_CRDT_CRED_TAR_ST_ADDR[i] = HD[i]_SCD_EDUP_P.START_ADDR; // SoC online start address
        DUP_CRDT_CRED_TAR_END_ADDR[i] = HD[i]_SCD_EDUP_P.END_ADDR; // SoC online end address
        DUP_CRDT_CRED_TAR_EN[i] = 0x1; // Enable credits mechanism towards this DUP
    }
    DUP_CRDT_CRED_EN = 0x1; // Enable Dup credit mechanism
*/
int Scal_Gaudi3::handleSarcCreditsToEdup(Qman::Program & program)
{
    unsigned numOfHd = (m_hw_ip.device_id == PCI_IDS_GAUDI3_SINGLE_DIE) ||
                            (m_hw_ip.device_id == PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE) ? 4 : 8;

    for (unsigned mstrIfNum = 0; mstrIfNum < numOfHd ; mstrIfNum++)// for all Master IF (in all HDs)
    {
        uint64_t base_addr   = mmHD0_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_BASE +
                            mstrIfNum * (mmHD1_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_BASE - mmHD0_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_BASE);
        for (unsigned hdnum = 0; hdnum < numOfHd ; hdnum++)// for all HDs
        {

            // mmHD0_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_CRDT_EN is at offset 0 of this block
            uint64_t offset = mmMSTR_IF_DUP_CRDT_RTR_CRDT_SIZE_0 - mmMSTR_IF_DUP_CRDT_CRDT_EN;
            // DUP_CRDT_RTR_CRDT_SIZE[i] = Credit_Value
            offset += 4 * hdnum;
            uint32_t value = EDUPCreditValue;

            LOG_INFO(SCAL,"{}: Master IF {} DUP_CRDT {} SIZE - base addr 0x{:x} offset 0x{:x} value {}", __FUNCTION__, mstrIfNum, hdnum,
                    base_addr, offset, value);
            program.addCommand(MsgLong(base_addr + offset, value));

            // DUP_CRDT_CRED_TAR_ST_ADDR[i] = HD[i]_SCD_EDUP_P.START_ADDR;
            // using offsets 0xE4 + 4 * j ..  from mmHDi_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_BASE  (i = 0..7)
            // SoC online start address. For example, 0x0300007FFE3B0000   (#define mmHD0_SCD_EDUP_P_BASE 0x300007FFE3B0000ull)
            // take bits Credit Target Start Address[28:12]
            offset = mmMSTR_IF_DUP_CRDT_CRED_TAR_ST_ADDR_0 - mmMSTR_IF_DUP_CRDT_CRDT_EN;
            offset += 4 * hdnum;
            uint64_t value64 = mmHD0_SCD_EDUP_P_BASE + hdnum * (mmHD1_SCD_EDUP_P_BASE - mmHD0_SCD_EDUP_P_BASE);
            value64  = (value64 >> 12) & 0x1FFFF;
            LOG_INFO(SCAL,"{}: Master IF {} DUP_CRDT {} ST_ADDR - base addr 0x{:x} offset 0x{:x} value 0x{:x}", __FUNCTION__, mstrIfNum, hdnum,
                    base_addr, offset, value64);
            program.addCommand(MsgLong(base_addr + offset, (uint32_t)value64));

            // DUP_CRDT_CRED_TAR_END_ADDR[i] = HD[i]_SCD_EDUP_P.END_ADDR; ;
            // SoC online end address. For example, 0x0300007FFE3B4000      (#define mmHD0_SCD_EDUP_ENG_BASE 0x300007FFE3B4000ull)
            // using offsets 0x11C + 4 * j ..  from mmHDi_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_BASE  (i = 0..7)
            offset = mmMSTR_IF_DUP_CRDT_CRED_TAR_END_ADDR_0 - mmMSTR_IF_DUP_CRDT_CRDT_EN;
            offset += 4 * hdnum;
            value64 = mmHD0_SCD_EDUP_ENG_BASE + hdnum * (mmHD1_SCD_EDUP_ENG_BASE - mmHD0_SCD_EDUP_ENG_BASE);
            value64  = (value64 >> 12) & 0x1FFFF;
            LOG_INFO(SCAL,"{}: Master IF {} DUP_CRDT {} END_ADDR - base addr 0x{:x} offset 0x{:x} value 0x{:x}", __FUNCTION__, mstrIfNum, hdnum,
                    base_addr, offset, value64);
            program.addCommand(MsgLong(base_addr + offset, (uint32_t)value64));

            // DUP_CRDT_CRED_TAR_EN[i] = 0x1; // Enable credits mechanism towards this DUP
            // using offsets 0xAC + 4 * j ..  from mmHDi_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_BASE  (i = 0..7)

            offset = mmMSTR_IF_DUP_CRDT_CRED_TAR_EN_0 - mmMSTR_IF_DUP_CRDT_CRDT_EN;
            offset += 4 * hdnum;
            value = 0x1;
            LOG_INFO(SCAL,"{}: Master IF {} DUP_CRDT {} TAR_EN - base addr 0x{:x} offset 0x{:x} value {}", __FUNCTION__, mstrIfNum, hdnum,
                    base_addr, offset, value);
            program.addCommand(MsgLong(base_addr + offset, value));
        } // for all hds
        // DUP_CRDT_CRED_EN = 0x1; // Enable Dup credit mechanism
        // offset = 0 --> mmHD0_ARC_FARM_FARM_MSTR_IF_DUP_CRDT_CRDT_EN;
        uint32_t value = 0x1;
        LOG_INFO(SCAL,"{}: Master IF {} DUP_CRDT_EN - base addr 0x{:x} value {}", __FUNCTION__, mstrIfNum,
                base_addr, value);
        program.addCommand(MsgLong(base_addr, value));
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::add32BitMask(RegToVal & regToVal, uint64_t edupBaseEng, uint32_t fenceEdupTrigger, uint64_t mask, AddMaskLogParams const & logParams)
{
    uint64_t clusterMaskAddr   = edupBaseEng + c_dup_trigger_info[fenceEdupTrigger].cluster_mask;
    // lower 32 bits of the mask
    scal_assert_return(Scal::addRegisterToMap(regToVal, clusterMaskAddr, lower_32_bits(mask), true), SCAL_INVALID_CONFIG,
        "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
        __FUNCTION__, logParams.clusterName, logParams.groupIndex, clusterMaskAddr, regToVal[clusterMaskAddr], (uint32_t)mask);
    assert(upper_32_bits(mask) == 0);
    return 0;
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
int Scal_Gaudi3::fillSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff, RegToVal &regToVal)
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
    fwInitCfg->enable_auto_fetcher = (uint32_t)m_use_auto_fetcher;
    fwInitCfg->synapse_params = m_arc_fw_synapse_config; // binary copy of struct

    const unsigned SCAL_MCID_START_ID_GAUDI3 = 1 * 1024;
    fwInitCfg->mcid_set_config.mcid_start_id = SCAL_MCID_START_ID_GAUDI3;
    fwInitCfg->mcid_set_config.discard_mcid_count = SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3;
    fwInitCfg->mcid_set_config.degrade_mcid_count = SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3;
    static_assert(SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3 + SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3 + SCAL_MCID_START_ID_GAUDI3 <= 64 * 1024, "mcid total count must be within 64K");

    G3Scheduler* scheduler = getCore<G3Scheduler>(coreIdx);
    if (!m_cores[coreIdx]->isScheduler)
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
            pfwCSG->sob_start_id                    = cq->sosBase + (c_max_sos_per_sync_manager * cq->syncManager->smIndex);
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
                pfwCSG->watch_dog_sob_addr = SobG3::getAddr(cq->compQTdr.sosPool->smBaseAddr, cq->compQTdr.sos);
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
        LOG_INFO(SCAL,"{}: fd={} no CQs for scheduler {}", __FUNCTION__, m_fd, scheduler->name);
    }

    // 0.b. find so sets
    // ------------------
    if (m_schedulersSosMap.find(scheduler->name) != m_schedulersSosMap.end())
    {
        unsigned soSet_configIdx = 0;
        for (auto& sgBase : m_schedulersSosMap[scheduler->name])
        {
            SyncObjectsSetGroupGaudi3* sg = static_cast<SyncObjectsSetGroupGaudi3*>(sgBase);
            fwInitCfg->so_sets_count = sg->numSets;
            for (unsigned idx = 0; idx < sg->numSets; idx++)
            {
                scal_assert_return(soSet_configIdx < SO_SET_COUNT, SCAL_INVALID_CONFIG,
                    "{}: fd={} so_set_config index > max {} at so set group {}", __FUNCTION__, m_fd, SO_SET_COUNT, sg->name);
                fwInitCfg->so_set_config[soSet_configIdx].sob_start_id = sg->sosPool->nextAvailableIdx + (c_max_sos_per_sync_manager * sg->sosPool->smIndex);
                sg->sosPool->nextAvailableIdx += sg->setSize;
                scal_assert_return(sg->sosPool->nextAvailableIdx <= sg->sosPool->baseIdx + sg->sosPool->size, SCAL_INVALID_CONFIG,
                    "{}: fd={} so next available index {} > so pool size at so set group {}", __FUNCTION__, m_fd, sg->sosPool->nextAvailableIdx, sg->name);

                fwInitCfg->so_set_config[soSet_configIdx].mon_id = sg->resetMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * sg->resetMonitorsPool->smIndex);
                sg->resetMonitorsPool->nextAvailableIdx++; // 1 monitor per set
                scal_assert_return(sg->resetMonitorsPool->nextAvailableIdx <= sg->resetMonitorsPool->baseIdx + sg->resetMonitorsPool->size, SCAL_INVALID_CONFIG,
                    "{}: fd={} monitor pool next available index {} > monitor pool size at so set group {}", __FUNCTION__, m_fd, sg->resetMonitorsPool->nextAvailableIdx, sg->name);
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

    uint64_t dupAddrEngBlock = mmHD0_ARC_FARM_ARC0_DUP_ENG_BASE - mmHD0_ARC_FARM_ARC0_DCCM0_BASE + scheduler->dccmDevAddress;
    unsigned numOfDups = sizeof(c_dup_trigger_info) / sizeof(c_dup_trigger_info[0]);
    for (unsigned dup = 0; dup < numOfDups; dup++)
    {
        for (unsigned maskIndex = 0; maskIndex < (c_dup_trigger_info[dup].max_engines + 31) / 32; maskIndex++)
        {
            uint64_t clusterMaskAddr = dupAddrEngBlock + c_dup_trigger_info[dup].cluster_mask + (maskIndex * sizeof(uint32_t));
            scal_assert_return(addRegisterToMap(regToVal, clusterMaskAddr, 0), SCAL_INVALID_CONFIG,
                    "{}: register {:#x} was already configured to {:#x} and configured again to {:#x}",
                    __FUNCTION__, clusterMaskAddr, regToVal[clusterMaskAddr], 0);
            LOG_DEBUG(SCAL, "write bitmask 0 to {:#x}", clusterMaskAddr);
        }
    }

    scal_assert_return(m_completionGroupCreditsSosPool && m_completionGroupCreditsMonitorsPool, SCAL_INVALID_CONFIG,
                       "{}: fd={} fillSchedulerConfigs completionGroupCreditsSosPool or completionGroupCreditsMonitorsPool is null", __FUNCTION__, m_fd);

    uint32_t dupAddrPush = lower_32_bits(c_local_address); // DUP_P is at the base of the local address range
    uint32_t dupAddrMask = lower_32_bits(c_local_address) + (mmHD0_ARC_FARM_ARC0_DUP_ENG_BASE - mmHD0_ARC_FARM_ARC0_DUP_P_BASE);
    for (auto& [k, cluster] : scheduler->clusters)
    {
        (void)k;
        unsigned commands_edup_trigger = c_commands_edup_trigger_offset + cluster->type;
        for (auto& [k, queue] : cluster->queues)
        {
            (void)k;
            std::array<std::bitset<64>, c_hdcores_nr> groupEngineBitmaps;
            if (queue.scheduler->name != scheduler->name)
            {
                continue;
            }
            auto& fwInitCfgGrp = fwInitCfg->eng_grp_cfg[queue.group_index];
            fwInitCfgGrp.engine_count_in_group = cluster->engines.size();

            // 1. configure dup engines and calculate bitmask
            // ===============================================
            Cluster::DupConfig& masterDupConfig = queue.dupConfigs[0];
            LOG_DEBUG(SCAL, "{}: cluster {} group {} will configure dup trigger {} of scheduler {} group_index {} engine_count_in_group {}", __FUNCTION__, cluster->name, queue.group_index, masterDupConfig.dupTrigger, scheduler->name, queue.group_index, fwInitCfgGrp.engine_count_in_group);
            scal_assert_return(cluster->engines.size() <= c_dup_trigger_info[masterDupConfig.dupTrigger].max_engines, SCAL_INVALID_CONFIG,
                    "{}: cluster {}: too many engines {} in dup {} max allowed engines {}",
                    __FUNCTION__, cluster->name, cluster->engines.size(), masterDupConfig.dupTrigger, c_dup_trigger_info[masterDupConfig.dupTrigger].max_engines);

            AddMaskLogParams logParams{
                .schedulerName = scheduler->name,
                .clusterName   = cluster->name,
                .groupIndex    = queue.group_index,
                .edupIndex     = 0
            };
            scal_assert_return(queue.dupConfigs.size() <= 2, SCAL_INVALID_CONFIG,
                "{}: cluster {}: too many dup triggers {}, up to 2 triggers are allowed", __FUNCTION__, cluster->name, queue.dupConfigs.size());

            for (unsigned dupConfigIndex = 0; dupConfigIndex < queue.dupConfigs.size(); dupConfigIndex++)
            {
                auto& dupConfig = queue.dupConfigs[dupConfigIndex];
                std::bitset<64> schedulerDupBitmap;
                uint64_t dupAddrOffset = c_dup_trigger_info[dupConfig.dupTrigger].dup_engines_address_base;
                for (auto engine_ : cluster->engines)
                {
                    G3ArcCore * engine = engine_->getAs<G3ArcCore>();
                    // If EDUP is enabled
                    //      We connect the scheduler to each of the hdcores edups
                    //      We each of the hdcores edups to the cluster engines in the same hdcore
                    // if EDUP is disabled
                    //      We connect the scheduler to all cluster's engines
                    // bool edup = true;//
                    if (cluster->localDup)
                    {
                        unsigned usedEdupIndex = engine->dCore * 2 + cluster->isCompute;
                        logParams.edupIndex = usedEdupIndex;
                        auto& groupEngineBitmap = groupEngineBitmaps[usedEdupIndex];
                        if (cluster->isCompute)
                        {
                            uint64_t edupBaseEng = mmHD0_SCD_EDUP_ENG_BASE + (usedEdupIndex * (mmHD1_SCD_EDUP_ENG_BASE - mmHD0_SCD_EDUP_ENG_BASE));
                            uint64_t edupBaseP = mmHD0_SCD_EDUP_P_BASE + (usedEdupIndex * (mmHD1_SCD_EDUP_P_BASE - mmHD0_SCD_EDUP_P_BASE));

                            if (groupEngineBitmap.count() == 0) // first engine in edup
                            {
                                // connect the SDUPs to HD#_EDUP_P (one entry per edup)
                                // set the HDCore bit in the dup mask
                                uint64_t dupEngSendAddr = dupAddrEngBlock + dupAddrOffset + (schedulerDupBitmap.count() * sizeof(uint32_t)); // entry point to dup
                                uint32_t dupReceiveOffsetToQueue = commands_edup_trigger * c_max_push_regs_per_dup;
                                uint32_t dupEngReceiveAddress = lower_29_bits(edupBaseP + dupReceiveOffsetToQueue);
                                scal_assert_return(addRegisterToMap(regToVal, dupEngSendAddr, dupEngReceiveAddress), SCAL_INVALID_CONFIG,
                                    "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                    __FUNCTION__, cluster->name, dupEngSendAddr, regToVal[dupEngSendAddr], dupEngReceiveAddress, engine->name);
                                LOG_DEBUG(SCAL, "scheduler {}: HD{}_{}_DUP_ENG DUP_ADDR_GR_{}[{}] ({:#x}) = HD{}_SCD_EDUP_P_BASE GR_{} ({:#x})", scheduler->name, usedEdupIndex, scheduler->arcName, dupConfig.dupTrigger, schedulerDupBitmap.count(), dupEngSendAddr, usedEdupIndex, commands_edup_trigger, dupEngReceiveAddress);
                                schedulerDupBitmap.set(schedulerDupBitmap.count());
                            }

                            // connect HD#_EDUP_ENG_GR# to the local engines DCCM of hdcore
                            uint64_t edupEngSendAddrCommands = edupBaseEng + c_dup_trigger_info[commands_edup_trigger].dup_engines_address_base + (groupEngineBitmap.count() * sizeof(uint32_t));
                            uint32_t edupReceiveOffsetToQueue = offsetof(gaudi3::block_qman_arc_aux, dccm_queue_push_reg) + sizeof(uint32_t) * queue.index;
                            uint32_t edupEngReceiveAddressCommands = lower_29_bits(engine->dccmDevAddress + getCoreAuxOffset(engine) + edupReceiveOffsetToQueue);
                            scal_assert_return(addRegisterToMap(regToVal, edupEngSendAddrCommands, edupEngReceiveAddressCommands), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                __FUNCTION__, cluster->name, edupEngSendAddrCommands, regToVal[edupEngSendAddrCommands], edupEngReceiveAddressCommands, engine->name);
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {}: EDUP_ADDR_HD_{}_GR_{} CMDS EDUP ({:#x}) = HD{}_{}_QM_DCCM queue {} ({:#x})", scheduler->name, cluster->name, usedEdupIndex, commands_edup_trigger, edupEngSendAddrCommands, usedEdupIndex, engine->arcName, queue.index, edupEngReceiveAddressCommands);

                            // connect HD#_EDUP_ENG_GR<c_fence_edup_trigger> to the local engines QMANs FENCE of hdcore
                            uint64_t edupAddrOffset = c_dup_trigger_info[c_fence_edup_trigger].dup_engines_address_base;
                            uint64_t edupEngSendAddrFence = edupBaseEng + edupAddrOffset + (groupEngineBitmap.count() * sizeof(uint32_t));
                            uint32_t edupReceiveOffsetToFence = offsetof(gaudi3::block_qman, cp_fence0_rdata);
                            uint32_t edupEngReceiveAddressFence = lower_29_bits(engine->dccmDevAddress + getCoreQmOffset() + edupReceiveOffsetToFence);
                            scal_assert_return(addRegisterToMap(regToVal, edupEngSendAddrFence, edupEngReceiveAddressFence), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                __FUNCTION__, cluster->name, edupEngSendAddrFence, regToVal[edupEngSendAddrFence], edupEngReceiveAddressFence, engine->name);
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {}: EDUP_ADDR_HD_{}_GR_{} FENCE EDUP ({:#x}) = HD{}_{}_QM CP_FENCE0_RDATA ({:#x})", scheduler->name, cluster->name, usedEdupIndex, c_fence_edup_trigger, edupEngSendAddrFence, usedEdupIndex, engine->arcName, edupEngReceiveAddressFence);

                            if (groupEngineBitmap.count() == 0) // first engine in EDUP
                            {
                                uint64_t edup_sync_scheme_lbw_addr = edupBaseP + c_max_push_regs_per_dup * c_fence_edup_trigger; // offset to the dup group in DUP_P
                                cluster->edup_sync_scheme_lbw_addr[engine->dCore] = lower_32_bits(edup_sync_scheme_lbw_addr);
                                uint64_t edup_b2b_lbw_addr = edup_sync_scheme_lbw_addr + (offsetof(gaudi3::block_qman, cp_fence1_rdata) - offsetof(gaudi3::block_qman, cp_fence0_rdata));
                                cluster->edup_b2b_lbw_addr[engine->dCore] = lower_32_bits(edup_b2b_lbw_addr);
                                LOG_DEBUG(SCAL, "scheduler {}: edup_sync_scheme_lbw_addr = HD{}_SCD_EDUP_P_BASE GR_{} = {:#x} ({:#x})", scheduler->name, usedEdupIndex, c_fence_edup_trigger,
                                    cluster->edup_sync_scheme_lbw_addr[engine->dCore], edup_sync_scheme_lbw_addr);
                                LOG_DEBUG(SCAL, "scheduler {}: edup_b2b_lbw_addr = HD{}_SCD_EDUP_P_BASE GR_{} = {:#x} ({:#x})", scheduler->name, usedEdupIndex, c_fence_edup_trigger,
                                    cluster->edup_b2b_lbw_addr[engine->dCore], edup_b2b_lbw_addr);
                            }

                            // set engine in bitmap
                            groupEngineBitmap.set(groupEngineBitmap.count());
                        }
                        // this is (very) quick and dirty !
                        else // cluster "cme")
                        {
                            usedEdupIndex = scheduler->dCore * 2;
                            uint64_t edupBaseEng = mmHD0_SCD_EDUP_ENG_BASE + (usedEdupIndex * (mmHD1_SCD_EDUP_ENG_BASE - mmHD0_SCD_EDUP_ENG_BASE));
                            uint64_t edupBaseP = mmHD0_SCD_EDUP_P_BASE + (usedEdupIndex * (mmHD1_SCD_EDUP_P_BASE - mmHD0_SCD_EDUP_P_BASE));
                            (void)edupBaseP; // TBD fill it in cme_init

                            // connect the SDUPs to DCCM queue
                            // set the HDCore bit in the dup mask
                            uint64_t dupEngSendAddr = dupAddrEngBlock + dupAddrOffset + (schedulerDupBitmap.count() * sizeof(uint32_t)); // entry point to dup
                            uint32_t edupReceiveOffsetToQueue = offsetof(gaudi3::block_qman_arc_aux, dccm_queue_push_reg) + sizeof(uint32_t) * queue.index;
                            uint32_t dupEngReceiveAddress     = lower_29_bits(engine->dccmDevAddress + getCoreAuxOffset(engine) + edupReceiveOffsetToQueue);
                            scal_assert_return(addRegisterToMap(regToVal, dupEngSendAddr, dupEngReceiveAddress), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                __FUNCTION__, cluster->name, dupEngSendAddr, regToVal[dupEngSendAddr], dupEngReceiveAddress, engine->name);
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {}: HD{}_{}_DUP_ENG DUP_ADDR_GR_{}[{}] ({:#x}) = engine {}  core {} ({:#x})", scheduler->name, cluster->name, usedEdupIndex, scheduler->arcName, dupConfig.dupTrigger, schedulerDupBitmap.count(), dupEngSendAddr, engine->name, engine->arcName, dupEngReceiveAddress);
                            schedulerDupBitmap.set(schedulerDupBitmap.count());

                            std::vector<uint64_t> cacheControllers =
                            {
                                mmHD0_CS0_MAINT_BASE, mmHD0_CS1_MAINT_BASE, mmHD0_CS2_MAINT_BASE, mmHD0_CS3_MAINT_BASE, mmHD0_CS4_MAINT_BASE, mmHD0_CS5_MAINT_BASE, mmHD0_CS6_MAINT_BASE, mmHD0_CS7_MAINT_BASE,
                                mmHD1_CS0_MAINT_BASE, mmHD1_CS1_MAINT_BASE, mmHD1_CS2_MAINT_BASE, mmHD1_CS3_MAINT_BASE, mmHD1_CS4_MAINT_BASE, mmHD1_CS5_MAINT_BASE, mmHD1_CS6_MAINT_BASE, mmHD1_CS7_MAINT_BASE,
                                mmHD2_CS0_MAINT_BASE, mmHD2_CS1_MAINT_BASE, mmHD2_CS2_MAINT_BASE, mmHD2_CS3_MAINT_BASE, mmHD2_CS4_MAINT_BASE, mmHD2_CS5_MAINT_BASE, mmHD2_CS6_MAINT_BASE, mmHD2_CS7_MAINT_BASE,
                                mmHD3_CS0_MAINT_BASE, mmHD3_CS1_MAINT_BASE, mmHD3_CS2_MAINT_BASE, mmHD3_CS3_MAINT_BASE, mmHD3_CS4_MAINT_BASE, mmHD3_CS5_MAINT_BASE, mmHD3_CS6_MAINT_BASE, mmHD3_CS7_MAINT_BASE,
                                mmHD4_CS0_MAINT_BASE, mmHD4_CS1_MAINT_BASE, mmHD4_CS2_MAINT_BASE, mmHD4_CS3_MAINT_BASE, mmHD4_CS4_MAINT_BASE, mmHD4_CS5_MAINT_BASE, mmHD4_CS6_MAINT_BASE, mmHD4_CS7_MAINT_BASE,
                                mmHD5_CS0_MAINT_BASE, mmHD5_CS1_MAINT_BASE, mmHD5_CS2_MAINT_BASE, mmHD5_CS3_MAINT_BASE, mmHD5_CS4_MAINT_BASE, mmHD5_CS5_MAINT_BASE, mmHD5_CS6_MAINT_BASE, mmHD5_CS7_MAINT_BASE,
                                mmHD6_CS0_MAINT_BASE, mmHD6_CS1_MAINT_BASE, mmHD6_CS2_MAINT_BASE, mmHD6_CS3_MAINT_BASE, mmHD6_CS4_MAINT_BASE, mmHD6_CS5_MAINT_BASE, mmHD6_CS6_MAINT_BASE, mmHD6_CS7_MAINT_BASE,
                                mmHD7_CS0_MAINT_BASE, mmHD7_CS1_MAINT_BASE, mmHD7_CS2_MAINT_BASE, mmHD7_CS3_MAINT_BASE, mmHD7_CS4_MAINT_BASE, mmHD7_CS5_MAINT_BASE, mmHD7_CS6_MAINT_BASE, mmHD7_CS7_MAINT_BASE
                            };
                            // by default it's a doble die
                            unsigned startIdx = 0;
                            unsigned endIdx = cacheControllers.size();
                            if (m_hw_ip.device_id == PCI_IDS_GAUDI3_DIE1)
                            {
                                // only upper half is available
                                startIdx = endIdx / 2;
                            }
                            else if (m_hw_ip.device_id == PCI_IDS_GAUDI3_SINGLE_DIE || m_hw_ip.device_id == PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE)
                            {
                                // only lower half is available
                                endIdx = endIdx / 2;
                            }

                            for (unsigned i = startIdx; i < endIdx; ++i)
                            {
                                // connect HD#_EDUP_ENG_GR# to the cache controllers
                                uint64_t cs = cacheControllers[i] + offsetof(gaudi3::block_cache_maintenance, attr_mcid);
                                uint64_t edupEngSendAddrCommands = edupBaseEng + c_dup_trigger_info[cme_cs_edup_trigger].dup_engines_address_base + (i * sizeof(uint32_t));
                                uint32_t edupEngReceiveAddressCommands = lower_29_bits(cs);
                                scal_assert_return(addRegisterToMap(regToVal, edupEngSendAddrCommands, edupEngReceiveAddressCommands), SCAL_INVALID_CONFIG,
                                    "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                    __FUNCTION__, cluster->name, edupEngSendAddrCommands, regToVal[edupEngSendAddrCommands], edupEngReceiveAddressCommands, engine->name);
                                LOG_DEBUG(SCAL, "scheduler {}: cluster {}: CME EDUP_ADDR_HD_{}_GR_{}[{}] EDUP ({:#x}) = {:#x})", scheduler->name, cluster->name, usedEdupIndex, cme_cs_edup_trigger, groupEngineBitmap.count(), edupEngSendAddrCommands, edupEngReceiveAddressCommands);
                                groupEngineBitmap.set(i);
                            }

                            // set EDUP mask
                            uint64_t clusterMaskAddr   = edupBaseEng + c_dup_trigger_info[cme_cs_edup_trigger].cluster_mask;
                            // lower 32 bits of the mask
                            scal_assert_return(addRegisterToMap(regToVal, clusterMaskAddr, lower_32_bits(groupEngineBitmap.to_ulong()), true), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                                __FUNCTION__, cluster->name, queue.group_index, clusterMaskAddr, regToVal[clusterMaskAddr], (uint32_t)schedulerDupBitmap.to_ulong());
                            // higher 32 bits of the mask
                            scal_assert_return(addRegisterToMap(regToVal, clusterMaskAddr + 4, upper_32_bits(groupEngineBitmap.to_ulong()), true), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                                __FUNCTION__, cluster->name, queue.group_index, clusterMaskAddr, regToVal[clusterMaskAddr], schedulerDupBitmap.to_ulong());
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {}: CME HD{}_SCD_EDUP_ENG DUP_MASK_GR_{} ({:#x}) = {:#x}", scheduler->name, cluster->name, usedEdupIndex, cme_cs_edup_trigger, clusterMaskAddr, groupEngineBitmap.to_ulong());

                            // connect HD#_EDUP_ENG_GR# to the DCCM Message Queue
                            {
                                uint64_t edupEngSendAddrCommands = edupBaseEng + c_dup_trigger_info[cme_dccmq_edup_trigger].dup_engines_address_base;
                                uint32_t edupReceiveOffsetToQueue = offsetof(gaudi3::block_qman_arc_aux, dccm_queue_push_reg) + sizeof(uint32_t) * 1/*queue.index*/;
                                uint32_t edupEngReceiveAddressCommands = lower_29_bits(engine->dccmDevAddress + getCoreAuxOffset(engine) + edupReceiveOffsetToQueue);
                                scal_assert_return(addRegisterToMap(regToVal, edupEngSendAddrCommands, edupEngReceiveAddressCommands), SCAL_INVALID_CONFIG,
                                    "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                    __FUNCTION__, cluster->name, edupEngSendAddrCommands, regToVal[edupEngSendAddrCommands], edupEngReceiveAddressCommands, engine->name);
                                LOG_DEBUG(SCAL, "scheduler {}: cluster {}: CME DCCMQ EDUP_ADDR_HD_{}_GR_{}[{}] EDUP ({:#x}) = {:#x})", scheduler->name, cluster->name, usedEdupIndex, cme_dccmq_edup_trigger, groupEngineBitmap.count(), edupEngSendAddrCommands, edupEngReceiveAddressCommands);
                                // set EDUP mask
                                uint64_t clusterMaskAddr   = edupBaseEng + c_dup_trigger_info[cme_dccmq_edup_trigger].cluster_mask;
                                scal_assert_return(addRegisterToMap(regToVal, clusterMaskAddr, 1, true), SCAL_INVALID_CONFIG,
                                    "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                                    __FUNCTION__, cluster->name, queue.group_index, clusterMaskAddr, regToVal[clusterMaskAddr], schedulerDupBitmap.to_ulong());
                                LOG_DEBUG(SCAL, "scheduler {}: cluster {}: CME DCCMQ HD{}_SCD_EDUP_ENG DUP_MASK_GR_{} ({:#x}) = {:#x}", scheduler->name, cluster->name, usedEdupIndex, cme_dccmq_edup_trigger, clusterMaskAddr, groupEngineBitmap.to_ulong());
                            }
                        }
                    }
                    else
                    {
                        // 1.b. configure dup engines
                        // ---------------------------
                        // engine, from here we get the address to write into
                        uint64_t dupReceiveOffsetToQueue = engine->dccmDevAddress;
                        if (cluster->type == CoreType::NIC)
                        {
                            // for nics (and only for nics) we may have different db_fifo per queue
                            uint64_t dupEngSendAddr = dupAddrEngBlock + dupAddrOffset + (schedulerDupBitmap.count() * sizeof(uint32_t));
                            G3NicCore * nicCore = engine->getAs<G3NicCore>();
                            unsigned port = nicCore->portsMask.test(0) ? nicCore->ports[0] : nicCore->ports[1];
                            unsigned dbFifoMaxValue = port % 2 ? ODD_PORT_MAX_DB_FIFO : EVEN_PORT_MAX_DB_FIFO;
                            auto db_fifo_id = dbFifoMaxValue - queue.index;
                            dupReceiveOffsetToQueue += c_nic_dccm_to_qpc_offset + offsetof(gaudi3::block_nic_qpc, dup_db_fifo) + (db_fifo_id * sizeof(uint32_t)); // add the db_fifo_ids offset
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {} DUP_ADDR_GR_{} queue {} engine {} adding offset for db_fifo_id {} address {:#x}", scheduler->name, cluster->name, dupConfig.dupTrigger, queue.index, engine->qman, db_fifo_id, dupReceiveOffsetToQueue);
                            uint32_t dupEngReceiveAddress = lower_29_bits(dupReceiveOffsetToQueue);
                            scal_assert_return(addRegisterToMap(regToVal, dupEngSendAddr, dupEngReceiveAddress), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                __FUNCTION__, cluster->name, dupEngSendAddr, regToVal[dupEngSendAddr], dupEngReceiveAddress, engine->name);
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {} DUP_ADDR_GR_{} ({:#x}) = {:#x} ({:#x})", scheduler->name, cluster->name, dupConfig.dupTrigger,  dupEngSendAddr, dupEngReceiveAddress, dupReceiveOffsetToQueue);
                        }
                        else
                        {
                            uint64_t dupEngSendAddr = dupAddrEngBlock + dupAddrOffset + (schedulerDupBitmap.count() * sizeof(uint32_t));
                            dupReceiveOffsetToQueue += getCoreAuxOffset(engine) + offsetof(gaudi3::block_qman_arc_aux, dccm_queue_push_reg);
                            dupReceiveOffsetToQueue += sizeof(uint32_t) * queue.index;
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {} DUP_ADDR_GR_{} queue {} engine {} adding offset for queue", scheduler->name, cluster->name, dupConfig.dupTrigger, queue.index, engine->qman);
                            uint32_t dupEngReceiveAddress = lower_29_bits(dupReceiveOffsetToQueue);
                            scal_assert_return(addRegisterToMap(regToVal, dupEngSendAddr, dupEngReceiveAddress), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                __FUNCTION__, cluster->name, dupEngSendAddr, regToVal[dupEngSendAddr], dupEngReceiveAddress, engine->name);
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {} DUP_ADDR_GR_{} ({:#x}) = {:#x} ({:#x})", scheduler->name, cluster->name, dupConfig.dupTrigger,  dupEngSendAddr, dupEngReceiveAddress, dupReceiveOffsetToQueue);
                        }
                        // set the engine bit in the dup mask
                        schedulerDupBitmap.set(schedulerDupBitmap.count(), true);
                    }
                }

                // tpc, mme, rot fences for CME
                if (cluster->isCompute)
                {
                    unsigned usedEdupIndex = scheduler->dCore * 2;
                    std::bitset<64> groupEngineBitmap;
                    uint64_t edupBaseEng = mmHD0_SCD_EDUP_ENG_BASE + (usedEdupIndex * (mmHD1_SCD_EDUP_ENG_BASE - mmHD0_SCD_EDUP_ENG_BASE));
                    unsigned fence_edup_trigger = c_fenceEdupTriggers.find(cluster->name)->second;

                    for (auto engine_ : cluster->engines)
                    {
                        ArcCore * engine = engine_->getAs<ArcCore>();
                        // connect HD#_EDUP_ENG_GR<c_fence_edup_trigger> to the local engines QMANs FENCE of hdcore
                        uint64_t edupAddrOffset = c_dup_trigger_info[fence_edup_trigger].dup_engines_address_base;
                        uint64_t edupEngSendAddrFence = edupBaseEng + edupAddrOffset + (groupEngineBitmap.count() * sizeof(uint32_t));
                        uint32_t edupReceiveOffsetToFence = offsetof(gaudi3::block_qman, cp_fence0_rdata);
                        uint32_t edupEngReceiveAddressFence = lower_29_bits(engine->dccmDevAddress + getCoreQmOffset() + edupReceiveOffsetToFence);
                        scal_assert_return(addRegisterToMap(regToVal, edupEngSendAddrFence, edupEngReceiveAddressFence), SCAL_INVALID_CONFIG,
                            "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                            __FUNCTION__, cluster->name, edupEngSendAddrFence, regToVal[edupEngSendAddrFence], edupEngReceiveAddressFence, engine->name);
                        LOG_DEBUG(SCAL, "scheduler {}: cluster {}: EDUP_ADDR_HD_{}_GR_{} FENCE EDUP ({:#x}) = HD{}_{}_QM CP_FENCE0_RDATA ({:#x})", scheduler->name, cluster->name, usedEdupIndex, c_fence_edup_trigger, edupEngSendAddrFence, usedEdupIndex, engine->arcName, edupEngReceiveAddressFence);
                        // set engine in bitmap
                        groupEngineBitmap.set(groupEngineBitmap.count());
                        if (groupEngineBitmap.count() == 16)
                        {
                            int ret = add32BitMask(regToVal, edupBaseEng, fence_edup_trigger, groupEngineBitmap.to_ulong(), logParams);
                            if (ret) return ret;
                            fence_edup_trigger++;
                            groupEngineBitmap.reset();
                        }
                    }
                    // process masks
                    if (groupEngineBitmap.count())
                    {
                        int ret = add32BitMask(regToVal, edupBaseEng, fence_edup_trigger, groupEngineBitmap.to_ulong(), logParams);
                        if (ret) return ret;
                    }
                    // cid_offset
                    if (cluster->name == getClusterNameByCoreType(TPC))
                    {
                        fence_edup_trigger = cme_tpc_cid_offset_trigger;
                        groupEngineBitmap.reset();
                        for (auto engine_ : cluster->engines)
                        {
                            ArcCore * engine = engine_->getAs<ArcCore>();
                            // connect HD#_EDUP_ENG_GR<c_fence_edup_trigger> to the local engines ARC CID of hdcore
                            uint64_t edupAddrOffset = c_dup_trigger_info[fence_edup_trigger].dup_engines_address_base;
                            uint64_t edupEngSendAddrCidOffset = edupBaseEng + edupAddrOffset + (groupEngineBitmap.count() * sizeof(uint32_t));
                            uint32_t edupReceiveOffsetToCidOffset = offsetof(gaudi3::block_qman_arc_aux, cid_offset);
                            uint32_t edupEngReceiveAddressCidOffset = lower_29_bits(engine->dccmDevAddress + getCoreAuxOffset(engine) + edupReceiveOffsetToCidOffset);
                            scal_assert_return(addRegisterToMap(regToVal, edupEngSendAddrCidOffset, edupEngReceiveAddressCidOffset), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: register {:#x} was already configured to {:#x} and configured again to {:#x} by engine {}",
                                __FUNCTION__, cluster->name, edupEngSendAddrCidOffset, regToVal[edupEngSendAddrCidOffset], edupEngReceiveAddressCidOffset, engine->name);
                            LOG_DEBUG(SCAL, "scheduler {}: cluster {}: EDUP_ADDR_HD_{}_GR_{} CID EDUP ({:#x}) = HD{}_{}_QM CID_OFFSET ({:#x})",
                                            scheduler->name, cluster->name, usedEdupIndex, c_fence_edup_trigger,
                                            edupEngSendAddrCidOffset, usedEdupIndex, engine->arcName, edupEngReceiveAddressCidOffset);
                            // set engine in bitmap
                            groupEngineBitmap.set(groupEngineBitmap.count());
                            if (groupEngineBitmap.count() == 16)
                            {
                                int ret = add32BitMask(regToVal, edupBaseEng, fence_edup_trigger, groupEngineBitmap.to_ulong(), logParams);
                                if (ret) return ret;
                                fence_edup_trigger++;
                                groupEngineBitmap.reset();
                            }
                        }
                        if (groupEngineBitmap.count())
                        {
                            int ret = add32BitMask(regToVal, edupBaseEng, fence_edup_trigger, groupEngineBitmap.to_ulong(), logParams);
                            if (ret) return ret;
                        }
                    }
                }
                // 2. configure the bitmask (for primary and secondary dup triggers)
                // ==================================================================
                uint64_t clusterMaskAddr   = dupAddrEngBlock + c_dup_trigger_info[dupConfig.dupTrigger].cluster_mask;
                scal_assert_return(addRegisterToMap(regToVal, clusterMaskAddr, schedulerDupBitmap.to_ulong(), true), SCAL_INVALID_CONFIG,
                    "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                    __FUNCTION__, cluster->name, queue.group_index, clusterMaskAddr, regToVal[clusterMaskAddr], schedulerDupBitmap.to_ulong());
                LOG_DEBUG(SCAL, "scheduler {}: DUP_MASK_GR_{} ({:#x}) = {:#x}", scheduler->name, dupConfig.dupTrigger, clusterMaskAddr, schedulerDupBitmap.to_ulong());

                // configure the dup trigger offset according to queue index
                // =========================================================
                uint64_t dupSendOffset = c_local_address + c_dup_trigger_info[dupConfig.dupTrigger].dup_offset;
                uint32_t dupReceiveOffsetToQueue = 0;
                scal_assert_return(addRegisterToMap(regToVal, dupSendOffset, dupReceiveOffsetToQueue), SCAL_INVALID_CONFIG,
                    "{}: cluster {}: dup_trans_data_q_base register {:#x} was already configured to {:#x} and configured again to {:#x}",
                    __FUNCTION__, cluster->name, dupSendOffset, regToVal[dupSendOffset], dupReceiveOffsetToQueue);
                LOG_DEBUG(SCAL, "scheduler {} DUP_OFFSET_GR_{} ({:#x}) = {:#x}",
                          queue.scheduler->name, dupConfig.dupTrigger, dupSendOffset, dupReceiveOffsetToQueue);

                // 2.1. configure the bitmask (for edup triggers)
                // ===============================================
                if (cluster->localDup && cluster->isCompute)
                {
                    for (unsigned hdCore = 0; hdCore < c_hdcores_nr; hdCore++)
                    {
                        const auto& groupEngineBitmap = groupEngineBitmaps[hdCore];
                        if (groupEngineBitmap.any())
                        {
                            uint64_t edupBaseEng = mmHD0_SCD_EDUP_ENG_BASE + (hdCore * (mmHD1_SCD_EDUP_ENG_BASE - mmHD0_SCD_EDUP_ENG_BASE));
                            uint64_t clusterMaskAddr   = edupBaseEng + c_dup_trigger_info[commands_edup_trigger].cluster_mask;
                            scal_assert_return(addRegisterToMap(regToVal, clusterMaskAddr, groupEngineBitmap.to_ulong(), true), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                                __FUNCTION__, cluster->name, queue.group_index, clusterMaskAddr, regToVal[clusterMaskAddr], schedulerDupBitmap.to_ulong());
                            LOG_DEBUG(SCAL, "scheduler {}: HD{}_SCD_EDUP_ENG DUP_MASK_GR_{} CMDS ({:#x}) = {:#x}", scheduler->name, hdCore, commands_edup_trigger, clusterMaskAddr, groupEngineBitmap.to_ulong());

                            uint64_t clusterFenceMaskAddr   = edupBaseEng + c_dup_trigger_info[c_fence_edup_trigger].cluster_mask;
                            scal_assert_return(addRegisterToMap(regToVal, clusterFenceMaskAddr, groupEngineBitmap.to_ulong(), true), SCAL_INVALID_CONFIG,
                                "{}: cluster {}: group_engine_bitmap of group {} register {:#x} was already configured to {:#x} and configured again to {:#x}",
                                __FUNCTION__, cluster->name, queue.group_index, clusterFenceMaskAddr, regToVal[clusterFenceMaskAddr], schedulerDupBitmap.to_ulong());
                            LOG_DEBUG(SCAL, "scheduler {}: HD{}_SCD_EDUP_ENG DUP_MASK_GR_{} FENCE ({:#x}) = {:#x}", scheduler->name, hdCore, c_fence_edup_trigger, clusterFenceMaskAddr, groupEngineBitmap.to_ulong());
                        }
                    }
                }

                // 3. configure eng_grp_cfg
                // =========================
                // when configuring the dup_trans_data_q_addr for the FW, the base address is from the device perspective.
                //   thats why the base address is mmARC_FARM_ARC0_DUP_ENG_DUP_TPC_ENG_ADDR_0 and not dupAddr
                uint32_t dup_trans_data_q_addr = dupAddrPush +
                                                 c_max_push_regs_per_dup * dupConfig.dupTrigger; // offset to the dup group in DUP_P
                uint32_t dup_mask_addr         = dupAddrMask +
                                                 c_dup_trigger_info[dupConfig.dupTrigger].cluster_mask; // offset to the dup group in DUP_P
                if (dupConfigIndex == 0)
                {
                    fwInitCfgGrp.dup_trans_data_q_addr = dup_trans_data_q_addr;
                    fwInitCfgGrp.dup_mask_addr = dup_mask_addr;
                }
                else
                {
                    fwInitCfgGrp.sec_dup_trans_data_q_addr = dup_trans_data_q_addr;
                    fwInitCfgGrp.sec_dup_mask_addr = dup_mask_addr;
                }
            }

            // Credits-Management
            unsigned cmSobjBaseIndex = 0;
            uint64_t cmSobjBaseAddr = 0;
            unsigned cmMonBaseIndex  = 0;
            int status = getCreditManagmentBaseIndices(cmSobjBaseIndex, cmSobjBaseAddr, cmMonBaseIndex, true);
            if (status != SCAL_SUCCESS)
            {
                LOG_ERR(SCAL,
                        "{}: cluster {}: Invalid Engines' Cluster-Queue Credit-Management configuration",
                        __FUNCTION__, cluster->name);
                assert(0);
                return SCAL_INVALID_CONFIG;
            }
            queue.sobjBaseIndex    = cmSobjBaseIndex;
            queue.sobjBaseAddr     = cmSobjBaseAddr - LBW_BASE;
            queue.monitorBaseIndex = cmMonBaseIndex;

            if (cluster->type == CoreType::NIC)
            {
                fwInitCfg->eng_grp_cfg[queue.group_index].nic_db_fifo_size = c_nic_db_fifo_size;
                fwInitCfg->eng_grp_cfg[queue.group_index].nic_db_fifo_threshold = c_nic_db_fifo_bp_treshold;
                LOG_DEBUG(SCAL, "{}: scheduler {}: cluster {}: nic_db_fifo_size {} nic_db_fifo_threshold {}",
                __FUNCTION__, scheduler->name, cluster->name, fwInitCfg->eng_grp_cfg[queue.group_index].nic_db_fifo_size, fwInitCfg->eng_grp_cfg[queue.group_index].nic_db_fifo_threshold);
            }

            LOG_DEBUG(SCAL,"{}, scheduler {} cluster {}(dup_trans_data_q_index {}) queue.index {} queue.group_index {} sob_start_id {} monitors-start-index {} dup_trans_data_q_addr = {:#x} dup_mask_addr = {:#x} sec_dup_trans_data_q_addr = {:#x} sec_dup_mask_addr = {:#x}",
                __FUNCTION__, scheduler->name, cluster->name, queue.dup_trans_data_q_index, queue.index, queue.group_index, queue.sobjBaseIndex, queue.monitorBaseIndex,
                fwInitCfg->eng_grp_cfg[queue.group_index].dup_trans_data_q_addr, fwInitCfg->eng_grp_cfg[queue.group_index].dup_mask_addr, fwInitCfg->eng_grp_cfg[queue.group_index].sec_dup_trans_data_q_addr, fwInitCfg->eng_grp_cfg[queue.group_index].sec_dup_mask_addr);
        }
    }

    // pdma ptr-mode. memory configuration
    std::bitset<NUM_OF_PDMA_CH> userPdmaChannels(m_hw_ip.pdma_user_owned_ch_mask);
    scal_assert_return(userPdmaChannels.count() >= SCHED_ARC_MAX_PDMA_CHANNEL_COUNT, SCAL_INVALID_CONFIG,
                "{}: Not enough user pdma channels {} required {}",
                __FUNCTION__, userPdmaChannels.count(), SCHED_ARC_MAX_PDMA_CHANNEL_COUNT);
    fwInitCfg->pdma_cfg.valid_pdma_ch_count = scheduler->pdmaChannels.size();
    for (unsigned i = 0 ; i < scheduler->pdmaChannels.size(); ++ i)
    {
        unsigned channelId = scheduler->pdmaChannels[i].pdmaChannelInfo->channelId;
        scal_assert_return(userPdmaChannels.test(channelId), SCAL_INVALID_CONFIG,
                "{}: PDMA channel {} is used by LKD, channels mask: 0b{:b}",
                __FUNCTION__, channelId, m_hw_ip.pdma_user_owned_ch_mask);
        sched_pdma_ch_cfg_t & pdma_ch_cfg = fwInitCfg->pdma_cfg.pdma_ch_cfg[i];
        pdma_ch_cfg.channel_number        = channelId;
        pdma_ch_cfg.pdma_engine_group_id  = scheduler->pdmaChannels[i].engineGroup;
        const unsigned buffer_size = (m_pdmaPool->size / SCHED_ARC_MAX_PDMA_CHANNEL_COUNT / c_scheduler_nr) & ~0xF;// size devided by 18. It might be aligned (to 0x10) otherwise PDMA HW (on PLDM) doesn't work
        pdma_ch_cfg.desc_buffer_size = buffer_size;
        pdma_ch_cfg.desc_buffer_addr = m_pdmaPool->deviceBase + buffer_size * (coreIdx  *   SCHED_ARC_MAX_PDMA_CHANNEL_COUNT + i);
        LOG_INFO(SCAL, "scheduler {} group {} (index {}) using PDMA channel {}",
                 scheduler->name, pdma_ch_cfg.pdma_engine_group_id, i, channelId);
        LOG_INFO(SCAL, "scheduler {} pdma_ch_cfg.desc_buffer_addr {:#x} desc_buffer_size={:#x}",
                 scheduler->name, pdma_ch_cfg.desc_buffer_addr, pdma_ch_cfg.desc_buffer_size);
    }

    // add priority configuration
    const unsigned offset = offsetof(gaudi3::block_pdma_ch_b, ch_priority);
    unsigned idx = 0;
    for (auto & pdmaChannel : scheduler->pdmaChannels)
    {
        const auto * pdmaChannelInfo = pdmaChannel.pdmaChannelInfo;
        scal_assert_return(
                addRegisterToMap(regToVal, pdmaChannelInfo->baseAddrB + offset, pdmaChannel.priority),
                SCAL_INVALID_CONFIG,
                "{}: spdma config register {:#x} idx: {} was already configured to {:#x} and configured again to {:#x}",
                __FUNCTION__, pdmaChannelInfo->baseAddrB, idx, regToVal[pdmaChannelInfo->baseAddrB + offset], pdmaChannel.priority);
        idx++;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::createCoreConfigQmanProgram(const unsigned coreIdx, DeviceDmaBuffer &buff, Qman::Program & program, RegToVal *regToVal)
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
    if(isLocal(core, localMode) != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL,"{}, failed to query localMode", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    if (core->isScheduler)
    {
        if ((m_schedulersCqsMap.find(core->name) != m_schedulersCqsMap.end()))
        {
            for (CompletionGroup* cq : m_schedulersCqsMap[core->name])
            {
                if (cq == nullptr)
                {
                    continue; // deliberate "hole"
                }
                bool     isSlave        = (cq->scheduler != core);
                unsigned numOfCqsSlaves = cq->slaveSchedulers.size();

                if ((!isSlave) && (numOfCqsSlaves != 0))
                {
                    // Configure Distributed-Completion-Group (CQs) Master-Slaves CM Monitors
                    LOG_DEBUG(SCAL, "{}: Configure credit-management (DCG) sched {} {} cq {}",
                                __FUNCTION__, core->cpuId, core->name, cq->name);
                    //
                    uint64_t counterAddress = core->dccmMessageQueueDevAddress;
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

        for (auto clusterItr : core->clusters)
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
                LOG_DEBUG(SCAL, "{}: Configure credit-management (CG) sched {} cluster {} queue.group_index {} queue.index {} sob_start_id {} queue.monitorBaseIndex {} dccmMessageQueueDevAddress {:#x}",
                          __FUNCTION__, core->name, cluster->name, queue.group_index, queue.index, queue.sobjBaseIndex, queue.monitorBaseIndex, core->dccmMessageQueueDevAddress);
                //
                uint64_t counterAddress = core->dccmMessageQueueDevAddress;
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

    uint64_t auxAddr =  getCoreAuxOffset(core) + (localMode? c_local_address: core->dccmDevAddress);

    uint64_t fenceAddr = c_local_address;
    if (!localMode)
    {
        if (!queueId2DccmAddr(core->qmanID, fenceAddr))
        {
            LOG_ERR(SCAL,"{}, queueId2DccmAddr() failed for queue id {}", __FUNCTION__, core->qmanID);
            assert(0);
            return SCAL_FAILURE;
        }
    }
    fenceAddr += c_dccm_to_qm_offset + offsetof(gaudi3::block_qman, cp_fence0_rdata);

    // set base3
    uint16_t baseRegAddressLo = c_dccm_to_qm_offset + offsetof(gaudi3::block_qman,cp_msg_base_addr[(c_message_short_base_index * 2) + 0]);
    uint16_t baseRegAddressHi = c_dccm_to_qm_offset + offsetof(gaudi3::block_qman,cp_msg_base_addr[(c_message_short_base_index * 2) + 1]);
    program.addCommand(WReg32(baseRegAddressLo, lower_32_bits(auxAddr)));
    program.addCommand(WReg32(baseRegAddressHi, upper_32_bits(auxAddr)));
    // reset fence0 before the wait. WREG 0 to FENCE_CNT0.
    program.addCommand(WReg32(c_dccm_to_qm_offset + offsetof(gaudi3::block_qman, cp_fence0_cnt), 0));
    program.addCommand(Wait(c_wait_cycles,1,0)); // Wait 32 cycles followed by fence
    program.addCommand(Fence(0,1,1));
    // configure the memory extensions
    //   - region 4 according to the coreIDX and m_fwImageHbmAddr. In region 4 also set the offset.
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, vir_mem0_msb_addr), upper_32_bits(m_coresBinaryDeviceAddress)));// on gaudi3 it was hbm0_msb_addr
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, vir_mem0_lsb_addr), lower_32_bits(m_coresBinaryDeviceAddress) / c_core_memory_extension_range_size));
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, vir_mem0_offset), (coreIdx * c_image_hbm_size)));

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
            region[pool->addressExtensionIdx + range] = pool->deviceBase + (range * c_core_memory_extension_range_size);
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
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, sram_lsb_addr);
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, sram_msb_addr);
                break;
            case CFG:
                // should not be configured
                break;
            case GP0:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, general_purpose_lsb_addr[0]);
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, general_purpose_msb_addr[0]);
                break;
            case HBM0:
                // not configurable
                break;
            case HBM1:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem1_lsb_addr);// hbm1_lsb_addr
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem1_msb_addr);
                offsetRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem1_offset);
                break;
            case HBM2:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem2_lsb_addr);// hbm2_lsb_addr
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem2_msb_addr);
                offsetRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem2_offset);
                break;
            case HBM3:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem3_lsb_addr);// hbm3_lsb_addr
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem3_msb_addr);
                offsetRegOffset = offsetof(gaudi3::block_qman_arc_aux, vir_mem3_offset);
                break;
            case DCCM:
                // not configurable
                break;
            case PCI:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, pcie_lower_lsb_addr);// pcie_lsb_addr
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, pcie_lower_msb_addr);
                break;
            case GP1:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, pcie_upper_lsb_addr);// pcie_lsb_addr
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, pcie_upper_msb_addr);
                break;
            case GP2:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, general_purpose_lsb_addr[1]);
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, general_purpose_msb_addr[1]);
                break;
            case GP3:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, d2d_hbw_lsb_addr);
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, d2d_hbw_msb_addr);
                break;
            case GP4:
                addrLowRegOffset = offsetof(gaudi3::block_qman_arc_aux, general_purpose_lsb_addr[2]);
                addrHighRegOffset = offsetof(gaudi3::block_qman_arc_aux, general_purpose_msb_addr[2]);
                break;
            case GP5:
                addrLowRegOffset = varoffsetof(gaudi3::block_qman_arc_aux, general_purpose_lsb_addr[3]);
                addrHighRegOffset = varoffsetof(gaudi3::block_qman_arc_aux, general_purpose_msb_addr[3]);
                break;
            case LBU:
                // not configurable
                break;
        }

        if (!addrLowRegOffset || !addrHighRegOffset)
        {
            if (extAddr != 0)
            {
                LOG_ERR(SCAL,"{}: fd={} illegal memory extension index - queueId2DccmAddr() failed for queue id {}", __FUNCTION__, m_fd, core->qmanID);
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
        LOG_INFO_F(SCAL, "core {} {} extEntry {} extAddr {:#x}", coreIdx, core->name, extEntry, extAddr);
    }

    // configure the ARC ID
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, arc_num), core->cpuId));
    LOG_DEBUG(SCAL, "core {} writing block_qman_arc_aux::arc_num = {} from qman_id = {}", core->name, core->cpuId, core->qmanID);

    // configure SP 0 with the LBW address of fence 0 of the current CP - schedulers absolute address and eARCs local address
    uint32_t fenceCoreAddress = lower_29_bits(fenceAddr) | (MemoryExtensionRange::LBU * c_core_memory_extension_range_size);
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, scratchpad[0]), fenceCoreAddress));

    // configure SP 1 with (buff->pool->coreBase + buff->offset)
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, scratchpad[1]), buff.getDeviceAddress()));
    // configure SP 2 with the size of the config buff in bytes
    program.addCommand(MsgShort(c_message_short_base_index, offsetof(gaudi3::block_qman_arc_aux, scratchpad[2]), buff.getSize()));

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
int Scal_Gaudi3::configEngines()
{
    int ret = SCAL_SUCCESS;
    DeviceDmaBuffer buffs[c_cores_nr];
    Qman::Workload workload;

    uint32_t coreIds[c_cores_nr - c_scheduler_nr] = {};
    uint32_t coreQmanIds[c_cores_nr - c_scheduler_nr] = {};
    uint32_t counter = 0;

    const bool isSfgEnabledInJson = m_completionGroups.count(cqGroup0Key) && m_completionGroups.at(cqGroup0Key).sfgInfo.sfgEnabled;
    int baseSfgSobPerEngineType[unsigned(EngineTypes::items_count)] = {};
    const int baseSfgSobPerEngineTypeIncrement[unsigned(EngineTypes::items_count)] = {2, 1, 1, 1};
    if (isSfgEnabledInJson)
    {
        for (unsigned i = 0 ; i < unsigned(EngineTypes::items_count); ++i)
        {
            baseSfgSobPerEngineType[i] = m_completionGroups.at(cqGroup0Key).sfgInfo.baseSfgSob[i];
        }
    }
    uint32_t sfgBaseSobId  = 0;

    // for all engine Arcs
    for (unsigned idx = 0; idx < c_cores_nr; idx++)
    {
        G3ArcCore * core = getCore<G3ArcCore>(idx);
        if (core)
        {
            if (core->isScheduler) continue;
            coreIds[counter]       = idx;
            coreQmanIds[counter++] = core->qmanID;

            LOG_DEBUG(SCAL, "{}: engine {}", __FUNCTION__, core->qman);

            ret = allocEngineConfigs(core, buffs[idx]);
            if (ret != SCAL_SUCCESS) break;

            if (core->getAs<G3CmeCore>())
            {
                ret = fillCmeEngineConfigs(core->getAs<G3CmeCore>(), buffs[idx]);
            }
            else
            {
                const bool sfgConfEnabled = (core->clusters.begin()->second) &&  isSfgEnabledInJson &&
                                            getSfgSobIdx(core->clusters.begin()->second->type, sfgBaseSobId, baseSfgSobPerEngineType, baseSfgSobPerEngineTypeIncrement);

                ret = fillEngineConfigs(core, buffs[idx], sfgBaseSobId, sfgConfEnabled);
            }
            if (ret != SCAL_SUCCESS) break;

            ret = (buffs[idx].commit(&workload) ? SCAL_SUCCESS : SCAL_FAILURE); // relevant only if uses HBM pool
            if (ret != SCAL_SUCCESS) break;

            Qman::Program program;
            ret = createCoreConfigQmanProgram(idx, buffs[idx], program);
            if (ret != SCAL_SUCCESS) break;

            workload.addProgram(program, core->qmanID);
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

int Scal_Gaudi3::allocEngineConfigs(const G3ArcCore * arcCore, DeviceDmaBuffer &buff)
{
    // allocate an ARC host buffer for the config file
    if(!arcCore)
    {
        LOG_ERR(SCAL,"{}: fd={} called with empty core", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }

    bool ret = buff.init(arcCore->configPool, arcCore->getAs<G3CmeCore>() ? sizeof(cme_config_t) : sizeof(engine_config_t));
    if(!ret)
    {
        assert(0);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::fillEngineConfigs(const G3ArcCore* arcCore, DeviceDmaBuffer &buff, int sfgBaseSobId, bool sfgConfEnabled)
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

    bool isCompute = false;
    for (const auto& [k, cluster] : arcCore->clusters)
    {
        (void)k;
        for (const auto& [k, queue] : cluster->queues)
        {
            (void)k;
            fwInitCfg->eng_resp_config[queue.index].sob_start_id = queue.sobjBaseIndex;
            LOG_DEBUG(SCAL, "engine {} idx {} cluster {} scheduler {} queue.index {} queue.group_index {}: sob_start_id = {}",
                arcCore->qman, arcCore->indexInGroup, cluster->name, queue.scheduler->name, queue.index, queue.group_index, fwInitCfg->eng_resp_config[queue.index].sob_start_id);
        }
        fwInitCfg->dccm_queue_count += cluster->queues.size();
        isCompute |= cluster->isCompute;
    }
    fwInitCfg->synapse_params = m_arc_fw_synapse_config; // binary copy of struct

    if (isCompute)
    {
        if (arcCore->clusters.size() != 1)
        {
            LOG_ERR(SCAL, "{}: compute engine {} can belong to a single cluster only", __FUNCTION__, arcCore->name);
            assert(0);
            return SCAL_INVALID_CONFIG;
        }
        Cluster* cluster = arcCore->clusters.begin()->second;
        const Scheduler* scheduler = nullptr;
        for (auto queueItr : cluster->queues)
        {
            auto queue = queueItr.second;
            if (queue.scheduler)
            {
                if (scheduler && scheduler->cpuId != queue.scheduler->cpuId)
                {
                    LOG_ERR(SCAL, "{}: compute engine {} is associated with more than one scheduler: {}, {}",
                        __FUNCTION__, arcCore->name, scheduler->name, queue.scheduler->name);
                    assert(0);
                    return SCAL_INVALID_CONFIG;
                }
                scheduler = queue.scheduler;
            }
        }
        auto* gcMonitorsPool = scheduler->m_sosSetGroups[0]->gcMonitorsPool;
        fwInitCfg->mon_start_id = gcMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * gcMonitorsPool->smIndex);
        gcMonitorsPool->nextAvailableIdx += SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT;
        if (gcMonitorsPool->nextAvailableIdx - gcMonitorsPool->baseIdx > gcMonitorsPool->size)
        {
            LOG_ERR(SCAL,"{}, compute engine sync monitor use ({}) exceeds max pool monitors ({}). from pool {} total queues {}",
                __FUNCTION__, gcMonitorsPool->nextAvailableIdx - gcMonitorsPool->baseIdx,
                gcMonitorsPool->size, gcMonitorsPool->name, fwInitCfg->dccm_queue_count);
            return SCAL_FAILURE;
        }
        auto* computeBack2BackMonitorsPool = scheduler->m_sosSetGroups[0]->computeBack2BackMonitorsPool;
        fwInitCfg->b2b_mon_id = computeBack2BackMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * computeBack2BackMonitorsPool->smIndex);
        computeBack2BackMonitorsPool->nextAvailableIdx += SCHED_CMPT_ENG_B2B_MON_COUNT;
        if (computeBack2BackMonitorsPool->nextAvailableIdx - computeBack2BackMonitorsPool->baseIdx > computeBack2BackMonitorsPool->size)
        {
            LOG_ERR(SCAL,"{}, compute back2bACK monitor use ({}) exceeds max pool monitors ({}). from pool {} total queues {}",
                __FUNCTION__, computeBack2BackMonitorsPool->nextAvailableIdx - computeBack2BackMonitorsPool->baseIdx,
                computeBack2BackMonitorsPool->size, computeBack2BackMonitorsPool->name, fwInitCfg->dccm_queue_count);
            return SCAL_FAILURE;
        }
        auto* topologyDebuggerMonitorsPool = scheduler->m_sosSetGroups[0]->topologyDebuggerMonitorsPool;
        fwInitCfg->soset_dbg_mon_start_id = topologyDebuggerMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * topologyDebuggerMonitorsPool->smIndex);
        topologyDebuggerMonitorsPool->nextAvailableIdx += SCHED_CMPT_ENG_SYNC_SCHEME_DBG_MON_COUNT;
        if (topologyDebuggerMonitorsPool->nextAvailableIdx - topologyDebuggerMonitorsPool->baseIdx > topologyDebuggerMonitorsPool->size)
        {
            LOG_ERR(SCAL,"{}, compute topology Debugger monitor use ({}) exceeds max pool monitors ({}). from pool {} total queues {}",
                __FUNCTION__, topologyDebuggerMonitorsPool->nextAvailableIdx - topologyDebuggerMonitorsPool->baseIdx,
                topologyDebuggerMonitorsPool->size, topologyDebuggerMonitorsPool->name, fwInitCfg->dccm_queue_count);
            return SCAL_FAILURE;
        }
        auto* gcSosPool = scheduler->m_sosSetGroups[0]->sosPool;
        fwInitCfg->soset_pool_start_sob_id = scheduler->m_sosSetGroups[0]->sosPool->baseIdx + (c_max_sos_per_sync_manager * gcSosPool->smIndex);
    }
    else
    {
        fwInitCfg->mon_start_id = scal_illegal_index;
    }

    for (const auto& cluster : m_computeClusters)
    {
        std::vector<int8_t> virtualSobIndexes;

        if (!coreType2VirtualSobIndexes(cluster->type, virtualSobIndexes))
        {
            LOG_ERR(SCAL,"{}, compute engine cluster {} type {} doesn't match VIRTUAL_SOB_INDEX", __FUNCTION__, cluster->name, cluster->type);
            return SCAL_FAILURE;
        }

        for (int8_t virtualSob : virtualSobIndexes)
        {
            fwInitCfg->engine_count_in_asic[virtualSob]   = cluster->engines.size();
            fwInitCfg->engine_count_in_dcore[virtualSob]  = cluster->enginesPerDCore[arcCore->dCore];
            fwInitCfg->engine_count_in_hdcore[virtualSob] = cluster->enginesPerHDCore[arcCore->hdCore];
        }

        if (cluster->localDup == true)
        {
            fwInitCfg->num_tpc_signals = cluster->numOfCentralSignals;
        }
    }
    if (arcCore->clusters.begin()->second->localDup && m_arc_fw_synapse_config.sync_scheme_mode == ARC_FW_GAUDI3_SYNC_SCHEME)
    {
        SyncObjectsSetGroupGaudi3* sg = static_cast<SyncObjectsSetGroupGaudi3*>(m_schedulersSosMap.begin()->second[0]);
        fwInitCfg->barrier_local_sob_id_base = sg->localSoSetResources[arcCore->hdCore].localBarrierSosPool->baseIdx + (c_max_sos_per_sync_manager * sg->localSoSetResources[arcCore->hdCore].localBarrierSosPool->smIndex);
        fwInitCfg->barrier_local_mon_id_base = sg->localSoSetResources[arcCore->hdCore].localBarrierMonitorsPool->baseIdx + (c_max_monitors_per_sync_manager * sg->localSoSetResources[arcCore->hdCore].localBarrierMonitorsPool->smIndex);

        auto& localSosPool = sg->localSoSetResources[arcCore->hdCore].localSosPool;
        fwInitCfg->local_soset_pool_start_sob_id = localSosPool->baseIdx + (c_max_sos_per_sync_manager * localSosPool->smIndex);
        fwInitCfg->edup_sync_scheme_lbw_addr = arcCore->clusters.begin()->second->edup_sync_scheme_lbw_addr[arcCore->dCore];
        fwInitCfg->edup_b2b_lbw_addr = arcCore->clusters.begin()->second->edup_b2b_lbw_addr[arcCore->dCore];
    }
    fwInitCfg->engine_index = arcCore->indexInGroup;
    fwInitCfg->engine_index_in_hdcore = arcCore->indexInGroupInHdCore;
    fwInitCfg->engine_index_in_dcore = arcCore->indexInGroupInDCore;
    fwInitCfg->num_engines_in_group = arcCore->numEnginesInGroup;

    fwInitCfg->cmpt_csg_sob_id_base  = m_computeCompletionQueuesSos->baseIdx + (c_max_sos_per_sync_manager * m_computeCompletionQueuesSos->smIndex);
    fwInitCfg->cmpt_csg_sob_id_count = m_computeCompletionQueuesSos->size;

    if (sfgConfEnabled)
    {
        fwInitCfg->sfg_sob_base_id = sfgBaseSobId + m_completionGroups.at(cqGroup0Key).sfgInfo.sfgSosPool->smIndex * c_max_sos_per_sync_manager;
        fwInitCfg->sfg_sob_per_stream = m_completionGroups.at(cqGroup0Key).sfgInfo.sobsOffsetToNextStream;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::fillCmeEngineConfigs(const G3CmeCore * core, DeviceDmaBuffer &buff)
{
    // fill the engine ARC config file according to the configuration
    //TODO,version header should be added also

    cme_config_t* fwInitCfg = (cme_config_t *)(buff.getHostAddress());
    if(!fwInitCfg)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to allocate host memory for arc_fw_init_config", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    memset(fwInitCfg, 0x0, sizeof(cme_config_t));
    fwInitCfg->version = ARC_FW_INIT_CONFIG_VER;
    bool hasTpc = false;
    bool hasMme = false;
    bool hasRotator = false;
    for (const auto& cluster : m_computeClusters)
    {
        std::vector<int8_t> virtualSobIndexes;
        hasTpc = hasTpc || cluster->type == TPC;
        hasMme = hasMme || cluster->type == MME;
        hasRotator = hasRotator || cluster->type == ROT;
        if (!coreType2VirtualSobIndexes(cluster->type, virtualSobIndexes))
        {
            LOG_ERR(SCAL,"{}, compute engine cluster {} type {} doesn't match VIRTUAL_SOB_INDEX", __FUNCTION__, cluster->name, cluster->type);
            return SCAL_FAILURE;
        }

        for (int8_t virtualSob : virtualSobIndexes)
        {
            fwInitCfg->engine_count_in_asic[virtualSob]   = cluster->engines.size();
            fwInitCfg->engine_count_in_dcore[virtualSob]  = cluster->enginesPerDCore[core->dCore];
            fwInitCfg->engine_count_in_hdcore[virtualSob] = cluster->enginesPerHDCore[core->hdCore];
        }

        if (cluster->localDup == true)
        {
            fwInitCfg->num_tpc_signals = cluster->numOfCentralSignals;
        }
    }
    assert(hasTpc);
    fwInitCfg->sync_scheme_mon_id = m_cmeMonitorsPool->baseIdx + (c_max_monitors_per_sync_manager * m_cmeMonitorsPool->smIndex);
    fwInitCfg->mon_start_id       = m_cmeEnginesMonitorsPool->baseIdx + c_max_monitors_per_sync_manager * m_cmeEnginesMonitorsPool->smIndex;
    fwInitCfg->mon_count          = m_cmeEnginesMonitorsPool->size;
    // add udup offset by edup index
    fwInitCfg->dup_cs_config_lbw_addr = lower_32_bits(mmHD0_SCD_EDUP_P_BASE + c_max_push_regs_per_dup * cme_cs_edup_trigger);
    LOG_DEBUG(SCAL,"fwInitCfg->dup_cs_config_lbw_addr = 0x{:x}", fwInitCfg->dup_cs_config_lbw_addr);
    fwInitCfg->dup_cme_dccmq_lbw_addr = lower_32_bits(mmHD0_SCD_EDUP_P_BASE + c_max_push_regs_per_dup * cme_dccmq_edup_trigger); // EDUP programmed to write to cme->dccmMessageQueueDevAddress
    LOG_DEBUG(SCAL,"fwInitCfg->dup_cme_dccmq_lbw_addr = 0x{:x}", fwInitCfg->dup_cme_dccmq_lbw_addr);

    uint64_t edupBaseP = mmHD0_SCD_EDUP_P_BASE;
    fwInitCfg->dup_tpc_fence0_lbw_addr = lower_32_bits(edupBaseP + c_max_push_regs_per_dup * c_fenceEdupTriggers.find(getClusterNameByCoreType(TPC))->second);
    if (hasMme)
    {
        assert(m_hw_ip.mme_enabled_mask != 0);
        if (m_hw_ip.mme_enabled_mask == 0)
        {
            LOG_CRITICAL(SCAL, "mme are enabled in json but there are no mme in device");
        }
        fwInitCfg->dup_mme_fence0_lbw_addr = lower_32_bits(edupBaseP + c_max_push_regs_per_dup * c_fenceEdupTriggers.find(getClusterNameByCoreType(MME))->second);
    }
    if (hasRotator)
    {
        assert(m_hw_ip.rotator_enabled_mask != 0);
        if (m_hw_ip.rotator_enabled_mask == 0)
        {
            LOG_CRITICAL(SCAL, "rotators are enabled in json but there are no rotators in device");
        }
        fwInitCfg->dup_rot_fence0_lbw_addr = lower_32_bits(edupBaseP + c_max_push_regs_per_dup * c_fenceEdupTriggers.find(getClusterNameByCoreType(ROT))->second);
    }
    fwInitCfg->dup_tpc_aux_cid_offset0_lbw_addr = lower_32_bits(edupBaseP + c_max_push_regs_per_dup * cme_tpc_cid_offset_trigger);

    // set cme_poll_main_status if not all the tpcs available
    const Cluster * tpcsCluster = getClusterByName(getClusterNameByCoreType(TPC));
    bool hasAllTPCs = tpcsCluster && tpcsCluster->engines.size() == 64;
    fwInitCfg->cme_poll_main_status = hasAllTPCs == false;

    const Scheduler * computeScheduler = nullptr;
    for (auto queueItr : tpcsCluster->queues)
    {
        auto queue = queueItr.second;
        if (queue.scheduler)
        {
            if (computeScheduler && computeScheduler->cpuId != queue.scheduler->cpuId)
            {
                LOG_ERR(SCAL, "{}: compute engine {} is associated with more than one scheduler: {}, {}",
                    __FUNCTION__, core->name, computeScheduler->name, queue.scheduler->name);
                assert(0);
                return SCAL_INVALID_CONFIG;
            }
            computeScheduler = queue.scheduler;
        }
    }
    auto* gcSosPool = computeScheduler->m_sosSetGroups[0]->sosPool;
    fwInitCfg->soset_pool_start_sob_id = computeScheduler->m_sosSetGroups[0]->sosPool->baseIdx + (c_max_sos_per_sync_manager * gcSosPool->smIndex);

    for (const auto& [k, cluster] : core->clusters)
    {
        (void)k;
        for (const auto& [_, queue] : cluster->queues)
        {
            (void)_;
            assert(queue.index < DCCM_QUEUE_COUNT);
            fwInitCfg->eng_resp_config[queue.index].sob_start_id = queue.sobjBaseIndex;
            LOG_DEBUG(SCAL, "engine {} idx {} cluster {} scheduler {} queue.index {} queue.group_index {}: sob_start_id = {}",
                core->qman, core->indexInGroup, cluster->name, queue.scheduler->name, queue.index, queue.group_index, fwInitCfg->eng_resp_config[queue.index].sob_start_id);
        }
        fwInitCfg->dccm_queue_count += cluster->queues.size();
    }

    fwInitCfg->synapse_params = m_arc_fw_synapse_config; // binary copy of struct

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::activateEngines()
{
    // write "1" to the canary registers of all the active cores
    for (unsigned idx = 0; idx <  c_cores_nr; idx++)
    {
        G3ArcCore * core = getCore<G3ArcCore>(idx);
        if(core)
        {
            uint64_t canaryAddress = 0;
            if (core->isScheduler)
            {
                canaryAddress = offsetof(sched_registers_t, canary) + (uint64_t)(core->dccmHostAddress);
            }
            else
            {
                if (core->getAs<G3CmeCore>())
                {
                    canaryAddress = offsetof(cme_registers_t, canary) + (uint64_t)(core->dccmHostAddress);
                }
                else
                {
                    canaryAddress = offsetof(engine_arc_reg_t, canary) + (uint64_t)(core->dccmHostAddress);
                }
            }
            LOG_TRACE(SCAL, "{}: writing canary of {} at address {:#x}", __FUNCTION__, core->name, canaryAddress);
            writeLbwReg((volatile uint32_t*)(canaryAddress), SCAL_INIT_COMPLETED);
        }
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::configureStreams()
{
    for (auto& streamMapPair : m_streams)
    {
        auto& stream = streamMapPair.second;
        auto  ret    = streamSetPriority(&stream, stream.priority);
        if (ret != SCAL_SUCCESS) return ret;
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::getNumberOfSignalsPerMme() const
{
    return 4; // each MME Engine sends 4 messages upon completion - SRAM/DRAM write per transpose/convolution. slave signaling is disabled in FW.
}

unsigned Scal_Gaudi3::getUsedSmBaseAddrs(const scal_sm_base_addr_tuple_t ** smBaseAddrDb)
{
    *smBaseAddrDb = m_smBaseAddrDb.data();
    return m_smBaseAddrDb.size();
}

int Scal_Gaudi3::getCreditManagmentBaseIndices(unsigned& cmSobjBaseIndex,
                                               uint64_t& cmSobjBaseAddr,
                                               unsigned& cmMonBaseIndex,
                                               bool      isCompletionGroupCM) const
{
    std::string creditsManagementTypeName[2] = {"distributed-completion-group", "completion-group"};

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
    cmSobjBaseAddr  = SobG3::getAddr(cmCreditsSosPool->smBaseAddr, cmCreditsSosPool->nextAvailableIdx);
    cmMonBaseIndex  = cmCreditsMonitorsPool->nextAvailableIdx + (c_max_monitors_per_sync_manager * cmCreditsMonitorsPool->smIndex);

    LOG_INFO(SCAL,"{}: fd={} isCompletionGroupCM {} dcore {} sm {} cmSobjBaseIndex {} ({}) cmMonBaseIndex {} ({})",
             __FUNCTION__, m_fd, isCompletionGroupCM, cmCreditsSosPool->dcoreIndex, cmCreditsSosPool->smIndex,
             cmSobjBaseIndex, cmCreditsSosPool->nextAvailableIdx,
             cmMonBaseIndex, cmCreditsMonitorsPool->nextAvailableIdx);

    // For both Elements (SOBJs and MONs):
    // 1 - Get Base value
    // 2 - Validate set's indeices are at the same SM
    // 3 - Increment nextAvailableIdx
    // 4 - Validate that we didn't exceed the amount of elements

    // validate that the sos and the monitors are from the same quarter
    unsigned queueSosQuarter = cmSobjBaseIndex / c_max_sos_per_sync_manager;
    unsigned queueSosQuarterEnd = (cmSobjBaseIndex + c_sos_for_completion_group_credit_management - 1) / c_max_sos_per_sync_manager;
    if (queueSosQuarter != queueSosQuarterEnd)
    {
        LOG_ERR(SCAL,"{}: fd={} {} credits sos {} range [{},{}] is not from the same quarter at dcore {} sm {}",
                __FUNCTION__, m_fd, creditsManagementTypeName[isCompletionGroupCM],
                cmCreditsSosPool->name, cmSobjBaseIndex,
                cmSobjBaseIndex + c_sos_for_completion_group_credit_management - 1,
                cmCreditsSosPool->dcoreIndex, cmCreditsMonitorsPool->smIndex);
        assert(0);
        return SCAL_INVALID_CONFIG;
    }

    unsigned queueMonitorQuarter = cmMonBaseIndex / c_max_monitors_per_sync_manager;
    unsigned queueMonitorQuarterEnd = (cmMonBaseIndex + c_monitors_for_completion_group_credit_management - 1) / c_max_monitors_per_sync_manager;
    if (queueMonitorQuarter != queueMonitorQuarterEnd)
    {
        LOG_ERR(SCAL,"{}: fd={} {} credits monitors {} range [{},{}] is not from the same quarter at dcore {} sm {}",
                __FUNCTION__, m_fd, creditsManagementTypeName[isCompletionGroupCM],
                cmCreditsMonitorsPool->name, cmMonBaseIndex,
                cmMonBaseIndex + c_monitors_for_completion_group_credit_management - 1,
                cmCreditsMonitorsPool->dcoreIndex, cmCreditsMonitorsPool->smIndex);
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

int Scal_Gaudi3::configureCreditManagementMonitors(Qman::Program& prog,
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

    for (unsigned j = 0; j < doubleBuffer; j++)
    {
        unsigned smId             = getMonitorSmId(monitorBaseIndex);
        uint64_t smBase           = SyncMgrG3::getSmBase(smId);
        uint64_t monitorIndexInSm = monitorBaseIndex % c_max_monitors_per_sync_manager;

        // we configure 3 monitors, all should be in same sync manager
        if ((monitorIndexInSm + 2) >= c_max_monitors_per_sync_manager)
        {
            LOG_ERR_F(SCAL, "Bad config; monitorBaseIndex {} monitorIndexInSm {}", monitorBaseIndex, monitorIndexInSm);
            assert(0);
            return SCAL_FAILURE;
        }

        uint64_t syncObjIndexInSm = sobjIndex % c_max_sos_per_sync_manager;

        // Arming the monitors to wait for SOBJ to reach >= numOfSlaves :
        // Monitor-ARM value:
        unsigned monSod  = numOfSlaves; // Data to compare SOB against

        uint32_t monArmRawValue = MonitorG3::buildArmVal(syncObjIndexInSm, monSod);
        //
        // Monitor-ARM address:
        uint64_t monArmAddress = MonitorG3(smBase, monitorIndexInSm).getRegsAddr().arm;

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
        // The master is monitoring a regular sob
        uint32_t monConfigRaw  = MonitorG3::buildConfVal(syncObjIndexInSm, numOfMsgs - 1, CqEn::off, LongSobEn::off, LbwEn::off, smId);
        //
        // Payload address : The counterAddress
        //
        // Payload Value : The counterValue
        configureOneMonitor(prog, monitorIndexInSm, smBase, monConfigRaw, counterAddress, counterValue);

        //
        //
        // 2nd monitor - SO Decrementation
        //
        // Monitor Configuration :
        monConfigRaw = 0; // Irrelevant
        //
        // Payload address : The SOBJ Address
        uint64_t sobjAddress = SobG3::getAddr(smBase, syncObjIndexInSm);
        //
        // Payload Value :
        // We want to deccrement (by num_slaves) a regular (15-bits) SOB
        // So in our case op=1 long=0 and bits [15:0] are (-num_slaves)
        sync_object_update syncObjUpdate;
        syncObjUpdate.raw                  = 0;
        syncObjUpdate.so_update.sync_value = (-numOfSlaves);
        syncObjUpdate.so_update.mode       = 1;
        //
        configureOneMonitor(prog, monitorIndexInSm + 1, smBase, monConfigRaw, sobjAddress, syncObjUpdate.raw);

        //
        //
        // 3rd monitor - Re-ARM of master monitor
        //
        // Monitor Configuration :
        monConfigRaw = 0; // Irrelevant
        //
        // Payload address : The monArmAddress
        //
        // Payload Value : Re-ARM master monitor command (see above - monArmData setting)
        // We want to set the monArmData value
        configureOneMonitor(prog, monitorIndexInSm + 2, smBase, monConfigRaw, monArmAddress, monArmRawValue);

        //
        //
        // Monitor-ARM
        //
        LOG_INFO_F(SCAL, "Monitor-Arm {} ({} {}) addr {:#x} data {:#x}",
                   monitorBaseIndex, smId, monitorIndexInSm, monArmAddress, monArmRawValue);
        prog.addCommand(MsgLong(monArmAddress, monArmRawValue));

        sobjIndex++;
        monitorBaseIndex += numOfMsgs;
    }
    return SCAL_SUCCESS;
}

uint32_t Scal_Gaudi3::getDistributedCompletionGroupCreditManagmentCounterValue(uint32_t completionGroupIndex)
{
    sched_mon_exp_comp_fence_t message;

    message.raw = 0;

    message.opcode              = MON_EXP_COMP_FENCE_UPDATE;
    message.comp_group_index    = completionGroupIndex;
    message.update_slave_credit = true;

    return message.raw;
}

uint32_t Scal_Gaudi3::getCompletionGroupCreditManagmentCounterValue(uint32_t engineGroupType)
{
    sched_mon_exp_update_q_credit_t message;

    message.raw = 0;

    message.opcode            = MON_EXP_UPDATE_Q_CREDIT;
    message.engine_group_type = engineGroupType;

    return message.raw;
}

unsigned Scal_Gaudi3::getSobjSmId(unsigned sobjIndex)
{
    return sobjIndex / c_max_sos_per_sync_manager;
}

unsigned Scal_Gaudi3::getMonitorSmId(unsigned monitorIndex)
{
    return monitorIndex / c_max_monitors_per_sync_manager;
}

inline bool Scal_Gaudi3::coreType2VirtualSobIndexes(const CoreType coreType, std::vector<int8_t> &virtualSobIndexes)
{
    switch (coreType)
    {
        case TPC:
        {
            virtualSobIndexes.push_back(VIRTUAL_SOB_INDEX_TPC);
            return true;
        }
        case ROT:
        {
            virtualSobIndexes.push_back(VIRTUAL_SOB_INDEX_ROT);
            return true;
        }
        // In case of MME engine that we use both for GEMM and XPOS (transpose)
        case MME:
        {
            virtualSobIndexes.push_back(VIRTUAL_SOB_INDEX_MME);
            virtualSobIndexes.push_back(VIRTUAL_SOB_INDEX_MME_XPOSE);
            return true;
        }
        default: return false;
    }
}

void Scal_Gaudi3::addFencePacket(Qman::Program& program, unsigned id, uint8_t targetVal, unsigned decVal)
{
    program.addCommand(Fence(id, targetVal, decVal));
}
void Scal_Gaudi3::enableHostFenceCounterIsr(CompletionGroup * cg, bool enableIsr)
{
    const unsigned curMonIdx = cg->monBase + cg->monNum;
    const unsigned smIdx     = cg->syncManager->smIndex;
    const uint32_t mon2      = curMonIdx + 1;

    LOG_INFO(SCAL,"{}: =========================== Fence Counter ISR =============================", __FUNCTION__);

    // Need 3 writes (dec sob, inc CQ, Arm master monitor)
    LOG_DEBUG(SCAL,"{}: Configure SM{}_MON_{} ISR {} ", __FUNCTION__, smIdx, mon2, enableIsr ? "enable" : "disable");

    uint64_t offset = MonitorG3(0, mon2).getRegsAddr().config;
    offset = offset / sizeof(*cg->syncManager->objsHostAddress);
    uint32_t monConfig = scal_read_mapped_reg(&cg->syncManager->objsHostAddress[offset]);
    monConfig = MonitorG3::setLbwEn(monConfig, enableIsr);
    scal_write_mapped_reg(&cg->syncManager->objsHostAddress[offset],  monConfig);
    [[maybe_unused]] auto v = scal_read_mapped_reg(&cg->syncManager->objsHostAddress[offset]); // read config in order to be sure the value is written into register
    assert(v == monConfig);
}

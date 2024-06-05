#include <assert.h>
#include <cstring>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <iterator>
#include "scal.h"
#include "scal_allocator.h"
#include "scal_utilities.h"
#include "scal_base.h"
#include "scal_macros.h"
#include "gaudi2/scal_gaudi2_factory.h"
#include "gaudi3/scal_gaudi3_factory.h"
#include "logger.h"
#include "hlthunk.h"
#include "common/qman_if.h"
#include "common/pci_ids.h"
#include "internal_jsons.h"
#include "infra/json_update_mask.hpp"

bool getSfgSobIdx(CoreType type,
                  uint32_t &sfgBaseSobId,
                  int (&sfgBaseSobIdPerEngine)[unsigned(EngineTypes::items_count)],
                  const int (&sfgBaseSobIdPerEngineIncrement)[unsigned(EngineTypes::items_count)])
{
    unsigned engineIdx = 0;
    switch(type)
    {
        case MME:
            engineIdx = unsigned(EngineTypes::mme);
            break;
         case TPC:
            engineIdx = unsigned(EngineTypes::tpc);
            break;
        case EDMA:
            engineIdx = unsigned(EngineTypes::edma);
            break;
         case ROT:
            engineIdx = unsigned(EngineTypes::rot);
            break;
        default:
            return false;
    }
    // in gaudi3 mme has 2 logical engines - mme and transpose
    // we provide one sob and this sob is used for mme and sob+1 for transpose
    // it means we need to provide a correct increment for sob - it's in sfgBaseSobIdPerEngineIncrement
    sfgBaseSobId = sfgBaseSobIdPerEngine[engineIdx];
    sfgBaseSobIdPerEngine[engineIdx] += sfgBaseSobIdPerEngineIncrement[engineIdx];
    return true;
}

#define MERGED_JSON_DUMP_FILENAME "scal_merged_conf.json"
#define AUTO_JSON_PATCH_DUMP_FILENAME "auto_patch.json"
#define AUTO_JSON_FINAL_DUMP_FILENAME "auto_patch_final.json"

bool Scal::scalStub = false;

static constexpr char c_scal_timeout_msec_value_env_var_name[]         = "SCAL_TIMEOUT_VALUE"; //micro seconds
static constexpr char c_scal_timeout_sec_value_env_var_name[]          = "SCAL_TIMEOUT_VALUE_SECONDS";
static constexpr char c_scal_timeout_no_progress_env_var_name[]        = "SCAL_TIMEOUT_NO_PROGRESS";
static constexpr char c_scal_disable_timeout_env_var_name[]            = "SCAL_DISABLE_TIMEOUT";
static constexpr uint64_t c_scal_default_timeout                       = 10UL * 60 * 1000000; // 10 minutes
static constexpr uint64_t c_scal_default_no_progress_timeout_addition  = 1UL  * 60 * 1000000; // 1 minutes

static void logHwIp(int fd, const hlthunk_hw_ip_info& hw_ip)
{
    LOG_INFO_F(SCAL, "----- hw_ip fd {}-----", fd);
    LOG_INFO_F(SCAL, "sram_base_address {:#x}",                         hw_ip.sram_base_address                     );
    LOG_INFO_F(SCAL, "dram_base_address {:#x}",                         hw_ip.dram_base_address                     );
    LOG_INFO_F(SCAL, "dram_size {:#x}",                                 hw_ip.dram_size                             );
    LOG_INFO_F(SCAL, "sram_size {:#x}",                                 hw_ip.sram_size                             );
    LOG_INFO_F(SCAL, "num_of_events {:#x}",                             hw_ip.num_of_events                         );
    LOG_INFO_F(SCAL, "device_id {:#x}",                                 hw_ip.device_id                             );
    LOG_INFO_F(SCAL, "cpld_version {:#x}",                              hw_ip.cpld_version                          );
    LOG_INFO_F(SCAL, "psoc_pci_pll_nr {:#x}",                           hw_ip.psoc_pci_pll_nr                       );
    LOG_INFO_F(SCAL, "psoc_pci_pll_nf {:#x}",                           hw_ip.psoc_pci_pll_nf                       );
    LOG_INFO_F(SCAL, "psoc_pci_pll_od {:#x}",                           hw_ip.psoc_pci_pll_od                       );
    LOG_INFO_F(SCAL, "psoc_pci_pll_div_factor {:#x}",                   hw_ip.psoc_pci_pll_div_factor               );
    LOG_INFO_F(SCAL, "tpc_enabled_mask {:#x}",                          hw_ip.tpc_enabled_mask                      );
    LOG_INFO_F(SCAL, "dram_enabled {:#x}",                              hw_ip.dram_enabled                          );
    //    LOG_INFO_F(SCAL, "cpucp_version[HL_INFO_VERSION_MAX_LEN] {:#x}",    hw_ip.cpucp_version[HL_INFO_VERSION_MAX_LEN]);
    LOG_INFO_F(SCAL, "module_id {:#x}",                                 hw_ip.module_id                             );
    //    LOG_INFO(SCAL, "card_name[HL_INFO_CARD_NAME_MAX_LEN] {:#x}",      hw_ip.card_name[HL_INFO_CARD_NAME_MAX_LEN]  );
    LOG_INFO_F(SCAL, "decoder_enabled_mask {:#x}",                      hw_ip.decoder_enabled_mask                  );
    LOG_INFO_F(SCAL, "mme_master_slave_mode {:#x}",                     hw_ip.mme_master_slave_mode                 );
    LOG_INFO_F(SCAL, "tpc_enabled_mask_ext {:#x}",                      hw_ip.tpc_enabled_mask_ext                  );
    LOG_INFO_F(SCAL, "dram_default_page_size {:#x}",                    hw_ip.device_mem_alloc_default_page_size    );
    LOG_INFO_F(SCAL, "dram_page_size {:#x}",                            hw_ip.dram_page_size                        );
    LOG_INFO_F(SCAL, "first_available_interrupt_id {:#x}",              hw_ip.first_available_interrupt_id          );
    LOG_INFO_F(SCAL, "edma_enabled_mask {:#x}",                         hw_ip.edma_enabled_mask                     );
    LOG_INFO_F(SCAL, "server_type {:#x}",                               hw_ip.server_type                           );
    LOG_INFO_F(SCAL, "pdma_user_owned_ch_mask {:#x}",                   hw_ip.pdma_user_owned_ch_mask               );
    LOG_INFO_F(SCAL, "number_of_user_interrupts {:#x}",                 hw_ip.number_of_user_interrupts             );
    LOG_INFO_F(SCAL, "nic_ports_mask {:#x}",                            hw_ip.nic_ports_mask                        );
    LOG_INFO_F(SCAL, "nic_ports_external_mask {:#x}",                   hw_ip.nic_ports_external_mask               );
    LOG_INFO_F(SCAL, "security_enabled {:#x}",                          hw_ip.security_enabled                      );

}
int Scal::setup()
{
    struct hl_info_args args = {};
    args.op = HL_INFO_MODULE_PARAMS;
    args.return_pointer = reinterpret_cast<__u64>(&m_hl_info);
    args.return_size = sizeof(m_hl_info);

    int ret = hlthunk_get_info(m_fd, &args);
    if (ret != 0)
    {
        LOG_ERR(SCAL, "{}: Get-Info ioctl failed for fd {} with return value {}", __FUNCTION__, m_fd, ret);
        return SCAL_FAILURE;
    }
    return SCAL_SUCCESS;
}

int Scal::Stream::submit(const unsigned pi, const unsigned submission_alignment)
{
    if ((scheduler == nullptr) || (scheduler->scal == nullptr))
    {
        LOG_ERR(SCAL,"{}, Invalid stream", __FUNCTION__);
        return SCAL_FAILURE;
    }

    if (scheduler->scal->scalStub || isStub)
    {
        return SCAL_SUCCESS;
    }

    int ret = scheduler->scal->streamSubmit(this, pi, submission_alignment);

    return ret;
}

int Scal::Stream::setBuffer(Buffer *buffer)
{
    if ((scheduler == nullptr) || (scheduler->scal == nullptr))
    {
        LOG_ERR(SCAL,"{}, Invalid stream", __FUNCTION__);
        return SCAL_FAILURE;
    }

    return scheduler->scal->setStreamBuffer(this, buffer);
}

int Scal::Stream::setPriority(unsigned priority)
{
    if ((scheduler == nullptr) || (scheduler->scal == nullptr))
    {
        LOG_ERR(SCAL,"{}, Invalid stream", __FUNCTION__);
        return SCAL_FAILURE;
    }

    return scheduler->scal->setStreamPriority(this, priority);
}

int Scal::Stream::getInfo(scal_stream_info_t& info) const
{
    info.name                = name.c_str();
    info.scheduler_handle    = toCoreHandle(scheduler);
    info.index               = id;
    info.type                = 0; // TODO
    info.current_pi          = localPiValue;
    info.control_core_buffer = (scal_buffer_handle_t)coreBuffer;

    if (coreBuffer == nullptr)
    {
        LOG_ERR(SCAL,
                "{}: stream coreBuffer is null."
                " You must call scal_stream_set_commands_buffer() before calling scal_stream_get_info()",
                __FUNCTION__);
        info.submission_alignment = 0;
        return SCAL_FAILURE;
    }

    info.submission_alignment = coreBuffer->size >> 16;
    info.command_alignment    = dccmBufferSize / 2;
    info.priority             = priority;

    info.isDirectMode = false;

    return SCAL_SUCCESS;
}

int Scal::Stream::getCcbBufferAlignment(unsigned& ccbBufferAlignment) const
{
    ccbBufferAlignment = this->ccbBufferAlignment;

    return SCAL_SUCCESS;
}

int Scal::create(const int fd, const std::string & configFileName, scal_arc_fw_config_handle_t fwCfg, Scal **scal)
{
    assert(scal);
    int ret;
    struct hlthunk_hw_ip_info hw_ip;

    char* stubEnv = getenv("ENABLE_SCAL_STUB");
    scalStub = (stubEnv != nullptr) &&
               ((std::strcmp(stubEnv, "true") == 0) || (std::strcmp(stubEnv, "1") == 0));
    if (fd <= 0)
    {
        LOG_ERR(SCAL,"{}: fd={} illegal fd", __FUNCTION__, fd);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    ret = hlthunk_get_hw_ip_info(fd, &hw_ip);
    if(ret)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to call hlthunk_get_hw_ip_info. ret = {} errno = {} {}", __FUNCTION__, fd, ret, errno, std::strerror(errno));
        assert(0);
        return SCAL_FAILURE;
    }

    logHwIp(fd, hw_ip);

    switch (hw_ip.device_id)
    {
    case PCI_IDS_GAUDI2:
    case PCI_IDS_GAUDI2_SIMULATOR:
    case PCI_IDS_GAUDI2_FPGA:
    case PCI_IDS_GAUDI2B_SIMULATOR:
        *scal = create_scal_gaudi2(fd, hw_ip, fwCfg);
        break;

    case PCI_IDS_GAUDI3:
    case PCI_IDS_GAUDI3_DIE1:
    case PCI_IDS_GAUDI3_SINGLE_DIE:
    case PCI_IDS_GAUDI3_SIMULATOR:
    case PCI_IDS_GAUDI3_ARC_SIMULATOR:
    case PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE:
        *scal = create_scal_gaudi3(fd, hw_ip, fwCfg);
        break;

    default:
        assert(0);
        return SCAL_FAILURE;
    }

    LOG_INFO_F(SCAL, "New device fd {} fwCfg handle {:#x}", fd, TO64(fwCfg));

    ret = (*scal)->init(configFileName);
    if (ret != SCAL_SUCCESS)
    {
        delete(*scal);
        *scal = nullptr;
        assert(0);
        return ret;
    }
    return SCAL_SUCCESS;
}

void Scal::destroy(Scal * scal)
{
    assert(scal);
    delete scal;
}

bool Scal::addRegisterToMap(RegToVal &regToVal, uint64_t key, uint32_t value, bool replace)
{
    if (regToVal.find(key) == regToVal.end())
    {
        regToVal[key] = value;
        return true;
    }
    else
    {
        if (regToVal[key] == value)
        {
            return true;
        }
        else
        {
            if (replace)
            {
                regToVal[key] = value;
                return true;
            }
        }
    }
    return false;
}

void * Scal::mapLBWBlock(const uint64_t lbwAddress, const uint32_t size)
{
    uint32_t allocatedSize = 0;
    void * ret = mapLbwMemory(m_fd, lbwAddress, size, allocatedSize);
    if (!ret)
    {
        LOG_ERR(SCAL,"{}: fd={}, failed to map LBW block (addr: {:#x}, size {})",
                __FUNCTION__, m_fd, lbwAddress, size);
        assert(0);
        return 0;
    }
    m_mappedLBWBlocks.push_back({ret, allocatedSize});
    return ret;
}

Scal::Scal(
    const int fd,
    const struct hlthunk_hw_ip_info & hw_ip,
    const unsigned coresNr,
    const unsigned smNr,
    const unsigned enginesTypesNr,
    std::unique_ptr<DevSpecificInfo> devApi)
: m_devSpecificInfo(std::move(devApi))
, m_fd(fd)
, m_hw_ip(hw_ip)
, m_cores(coresNr, nullptr)
, m_syncManagers(smNr)
, m_completionGroupCreditsSosPool(nullptr)
, m_completionGroupCreditsMonitorsPool(nullptr)
, m_distributedCompletionGroupCreditsSosPool(nullptr)
, m_distributedCompletionGroupCreditsMonitorsPool(nullptr)
{
}

Scal::~Scal()
{
    int ret=0;

    ret |= unmapLBWBlocks();
    scal_assert(ret == SCAL_SUCCESS,"in Scal destructor, unmapLBWBlocks failed");

    if (m_coresBinaryDeviceAddress)
    {
        m_binaryPool->allocator->free(m_coresBinaryDeviceAddress - m_binaryPool->deviceBase);
    }

    ret |= releaseMemoryPools();
    scal_assert(ret == SCAL_SUCCESS,"in Scal destructor, releaseMemoryPools failed");

    ret |= releaseCompletionQueues();
    scal_assert(ret == SCAL_SUCCESS,"in Scal destructor, releaseCompletionQueues failed");

    for (auto & pool : m_pools)
    {
        if (pool.second.allocator)
        {
            delete pool.second.allocator;
        }
    }

    if (m_fullHbmPool.size != 0)
    {
        delete m_fullHbmPool.allocator;
    }

    deleteAllAllocatedBuffers();

    for (Core *core : m_cores)
    {
        if (core)
        {
            delete core;
        }
    }

    if (ret != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL,"{}: fd={} errors in scal destructor", __FUNCTION__, m_fd);
    }
}

int Scal::unmapLBWBlocks()
{
    int ret = SCAL_SUCCESS;
    for (const auto & block : m_mappedLBWBlocks)
    {
        int rc = unmapLbwMemory(block.first, block.second);
        if (rc)
        {
            LOG_ERR(SCAL, "{}: fd={} unmapLbwMemory() error. (addr: {:#x}, size: {})", __FUNCTION__, m_fd, (uint64_t)block.first, block.second);
            assert(0);
            ret = SCAL_FAILURE;
        }
    }

    return ret;
}

int Scal::releaseMemoryPools()
{
    LOG_INFO_F(SCAL, "===== releaseMemoryPools =====");

    int ret = SCAL_SUCCESS;
    int rc;

    for (auto addr : m_extraMapping)
    {
        LOG_INFO_F(SCAL, "hlthunk_memory_unmap extra mapping {:#x}", addr);
        rc = hlthunk_memory_unmap(m_fd, addr);
        if (rc)
        {
            LOG_ERR(SCAL,"{}: fd={} addr {:#x} hlthunk_memory_unmap() failed rc={}", __FUNCTION__, m_fd, addr, rc);
            assert(0);
            ret = SCAL_FAILURE;
        }
    }

    // Prepare a list of all the pools. Add the m_fullPool at the *end*, we want to remove it last
    std::vector<Pool*> pools;
    for (auto & poolPair : m_pools)
    {
        pools.push_back(&poolPair.second);
    }
    if (m_fullHbmPool.size != 0)
    {
        pools.push_back(&m_fullHbmPool);
    }

    // Go over all pools, release resource
    for (auto pool : pools)
    {
        if (pool->deviceBase)
        {
            if (pool->fromFullHbmPool) // if this pool was allocated from the fullPool, just release it
            {
                LOG_INFO_F(SCAL, "pool {} was allocated from fulPool. Free {:#x}-{:#x}={:#x}",
                           pool->name, pool->deviceBase, m_fullHbmPool.deviceBase, pool->deviceBase - m_fullHbmPool.deviceBase);
                m_fullHbmPool.allocator->free(pool->deviceBase - m_fullHbmPool.deviceBase);
            }
            else
            {
                LOG_INFO_F(SCAL, "pool {} allocated with lkd, hlthunk_memory_unmap deviceBase {:#x}", pool->name, pool->deviceBase);
                rc = hlthunk_memory_unmap(m_fd, pool->deviceBase);
                if (rc)
                {
                    LOG_ERR(SCAL,"{}: fd={} pool.deviceBase {:#x} hlthunk_memory_unmap() failed rc={}", __FUNCTION__, m_fd, pool->deviceBase, rc);
                    assert(0);
                    ret = SCAL_FAILURE;
                }
            }
        }

        if (pool->type == Pool::Type::HOST)
        {
            if (pool->hostBase)
            {
                rc = munmap(pool->hostBase, pool->size);
                if (rc)
                {
                    LOG_ERR(SCAL,"{}: fd={} pool.hostBase munmap() failed rc={}", __FUNCTION__, m_fd, rc);
                    assert(0);
                    ret = SCAL_FAILURE;
                }
            }
        }
        else
        {
            if (pool->deviceHandle)
            {
                if (!pool->fromFullHbmPool) // free only if not allocated from the fullPool
                {
                    LOG_INFO_F(SCAL, "hlthunk_device_memory_free fd {} deviceHandle {}", m_fd, pool->deviceHandle);
                    rc = hlthunk_device_memory_free(m_fd, pool->deviceHandle);
                    if (rc)
                    {
                        LOG_ERR(SCAL,"{}: fd={} pool  hlthunk_device_memory_free() failed rc={}", __FUNCTION__, m_fd, rc);
                        assert(0);
                        ret = SCAL_FAILURE;
                    }
                }
            }
        }
    }

    return ret;
}

int Scal::fillRegions(unsigned core_memory_extension_range_size)
{
    LOG_INFO_F(SCAL, "===== fillRegions =====");

    for (const auto & [k, pool] : m_pools)
    {
        (void)k; // unused, compiler error in gcc 7.5
        LOG_INFO_F(SCAL, "pool {} addressExtensionIdx {} fillRegion {} size {:#x}",
                   pool.name, pool.addressExtensionIdx, pool.fillRegion, pool.size);

        if (pool.fillRegion)
        {
            uint64_t devBase    = pool.deviceBase;
            uint64_t addrToFill = devBase + pool.size;
            uint64_t filled     = pool.size;

            if((pool.type == Pool::Type::HBM) && pool.fromFullHbmPool) // no padding if allocated from memory
            {
                continue;
            }

            while (filled < core_memory_extension_range_size)
            {
                uint64_t hint = addrToFill;
                LOG_INFO_F(SCAL, "hlthunk_device_memory_map devHandle {} hint {:#x}", pool.deviceHandle, hint);

                uint64_t mappedAddr;
                if (pool.type == Pool::Type::HBM)
                {
                    mappedAddr  = hlthunk_device_memory_map(m_fd, pool.deviceHandle, hint);
                }
                else if (pool.type == Pool::Type::HOST)
                {
                    mappedAddr = hlthunk_host_memory_map(m_fd, pool.hostBase , hint, pool.size);
                }
                else
                {
                    break;
                }
                if (mappedAddr != hint)
                {
                    LOG_ERR_F(SCAL,"fd={} Failed to map memory on pool {} hint {:#x} mappedAddr {:#x}",
                              m_fd, pool.name, hint, mappedAddr);
                    assert(0);
                    return SCAL_FAILURE;
                }
                m_extraMapping.push_back(mappedAddr);

                addrToFill += pool.size;
                filled     += pool.size;
            } // while
        } // if(fillRegion)
    } // for (m_pools)
    return SCAL_SUCCESS;
}

int Scal::allocateCompletionQueues()
{
    // allocate host page aligned memory for all the completion queue counters dcore_nr * 64
    // - update m_completionQueueCounters
    size_t  size = (getSchedulerModeCQsCount() + getDirectModeCqsAmount()) * sizeof(uint64_t);
    size_t  alignment = getpagesize(); // regular page size
    // size should be a multiple of alignment
    size = (((size - 1) / alignment) + 1) * alignment;

    int rc = hlthunk_request_mapped_command_buffer(m_fd, size, &m_completionQueuesHandle);
    if(rc || (m_completionQueuesHandle == 0ULL))
    {
        LOG_ERR(SCAL,"{}: fd={} hlthunk_request_mapped_command_buffer failed. rc = {} errno = {} {}", __FUNCTION__, m_fd, rc, errno, std::strerror(errno));
        assert(0);
        return SCAL_FAILURE;
    }
    // map the completion queue counters buffer to the device
    rc =  hlthunk_get_mapped_cb_device_va_by_handle(m_fd, m_completionQueuesHandle, &m_completionQueueCountersDeviceAddr);
    if(rc || !m_completionQueueCountersDeviceAddr)
    {
        LOG_ERR(SCAL,"{}: fd={} hlthunk_get_mapped_cb_device_va_by_handle (m_completionQueuesHandle {:#x} m_completionQueueCountersDeviceAddr {:#x}) failed. rc = {} errno = {} {}", __FUNCTION__, m_fd, m_completionQueuesHandle, m_completionQueueCountersDeviceAddr, rc, errno, std::strerror(errno));
        assert(0);
        return SCAL_FAILURE;
    }
    m_completionQueueCounters = (volatile uint64_t*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, m_completionQueuesHandle);
    if (m_completionQueueCounters == MAP_FAILED || !m_completionQueueCounters)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to mmap m_completionQueuesHandle {:#x} m_completionQueueCounters {:#x} with {} bytes in host memory", __FUNCTION__, m_fd, m_completionQueuesHandle, (uint64_t)m_completionQueueCounters, size);
        assert(0);
        return SCAL_FAILURE;
    }
    m_completionQueueCountersSize = size;
    memset((void*)m_completionQueueCounters, 0, size);
    return SCAL_SUCCESS;
}

int Scal::releaseCompletionQueues()
{
    int ret = SCAL_SUCCESS;
    int rc;
    if (m_completionQueueCounters)
    {
        munmap((void*)m_completionQueueCounters, m_completionQueueCountersSize);
    }

    if (m_completionQueuesHandle)
    {
        rc = hlthunk_destroy_command_buffer(m_fd, m_completionQueuesHandle);
        if (rc)
        {
            LOG_ERR(SCAL,"{}: fd={} pool  hlthunk_device_memory_free() failed rc={}", __FUNCTION__, m_fd, rc);
            assert(0);
            ret = SCAL_FAILURE;
        }
    }

    return ret;
}

/**
 * initializes timeout members of Scal.
 * First gets timeout from driver.
 * if the driver timeout is 0, then we use an infinite timeout.
 * if SCAL_TIMEOUT_VALUE env var is configured, it's value is used as the timeout instead.
 * if SCAL_DISABLE_TIMEOUT==1, then m_timeoutDisabled is set to 'true'.
 * @return SCAL_SUCCESS if retrieving timeout from the driver succeeded. otherwise, return SCAL_FAILURE.
 */
int Scal::initTimeout()
{
    m_timeoutMicroSec = 0;
    // check if timeout is overridden by user
    const char * timeoutMsecEnvVarValue = getenv(c_scal_timeout_msec_value_env_var_name);
    if (timeoutMsecEnvVarValue)
    {
        uint64_t timeout = 0;
        try
        {
            timeout = std::stoull(timeoutMsecEnvVarValue);
            m_timeoutMicroSec += timeout;
            LOG_INFO(SCAL, "{}: SCAL timeout was set to {} microseconds by env var '{}'", __FUNCTION__, m_timeoutMicroSec, c_scal_timeout_msec_value_env_var_name);
        }
        catch (const std::exception &e)
        {
            LOG_ERR(SCAL, "{}: stoull() failed parsing env var '{}'", __FUNCTION__, c_scal_timeout_msec_value_env_var_name);
        }
    }

    // check if timeout is overridden by user
    const char * timeoutSecEnvVarValue = getenv(c_scal_timeout_sec_value_env_var_name);
    if (timeoutSecEnvVarValue)
    {
        uint64_t timeout = 0;
        try
        {
            timeout = std::stoull(timeoutSecEnvVarValue);
            m_timeoutMicroSec += (timeout * 1000000);
            LOG_INFO(SCAL, "{}: SCAL seconds timeout was set to {} microseconds by env var '{}'", __FUNCTION__, m_timeoutMicroSec, c_scal_timeout_sec_value_env_var_name);
        }
        catch (const std::exception &e)
        {
            LOG_ERR(SCAL, "{}: stoull() failed parsing env var '{}'", __FUNCTION__, c_scal_timeout_sec_value_env_var_name);
        }
    }
    if (!timeoutMsecEnvVarValue && !timeoutSecEnvVarValue)
    {
        // use default timeout - todo once devops are using the new timeout
        // m_timeoutMicroSec = c_scal_default_timeout; // 10 minutes

        // get timeout from driver
        uint64_t driverTimeout = m_hl_info.timeout_locked;

        // when setting "timeout_locked=0", it is represented by (UINT32_MAX / 1000).
        // In this case we should disable the timeout and run forever.
        if (driverTimeout == UINT32_MAX / 1000)
        {
            // set to a very long timeout
            m_timeoutMicroSec = UINT64_MAX;
            LOG_INFO(SCAL, "{}: driver timeout_locked {} was disabled, so SCAL timeout was set to {} microseconds",
                    __FUNCTION__, driverTimeout, m_timeoutMicroSec);
        }
        else
        {
            // convert from seconds to microseconds
            m_timeoutMicroSec = driverTimeout * 1000000;
            LOG_INFO(SCAL, "{}: driver timeout_locked is {} seconds, so SCAL timeout was set to {} microseconds",
                    __FUNCTION__, driverTimeout, m_timeoutMicroSec);
        }
    }

    // check if timeout is disabled by user
    const char * timeoutDisabledEnvVarValue = getenv(c_scal_disable_timeout_env_var_name);
    if (timeoutDisabledEnvVarValue)
    {
        try
        {
            int isDisabled = std::stoi(timeoutDisabledEnvVarValue);
            if (isDisabled == 1)
            {
                m_timeoutDisabled = true;
                LOG_INFO(SCAL, "{}: SCAL timeout was disabled by env var '{}'", __FUNCTION__, c_scal_disable_timeout_env_var_name);
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERR(SCAL, "{}: stoi() failed parsing env var '{}'", __FUNCTION__, c_scal_disable_timeout_env_var_name);
        }
    }

    m_timeoutUsNoProgress = std::max(c_scal_default_timeout, m_timeoutMicroSec + c_scal_default_no_progress_timeout_addition); // default max(10 min, engine timeout + 1 minute)
    const char * timeoutNoProgressEnvVarValue = getenv(c_scal_timeout_no_progress_env_var_name);
    if (timeoutNoProgressEnvVarValue)
    {
        uint64_t timeoutNoProgress = 0;
        try
        {
            timeoutNoProgress = std::stoull(timeoutNoProgressEnvVarValue);
            m_timeoutUsNoProgress = timeoutNoProgress;
            LOG_INFO_F(SCAL, "SCAL timeoutNoProgress was set to {} microseconds by env var '{}'",
                       m_timeoutUsNoProgress, c_scal_timeout_no_progress_env_var_name);
        }
        catch (const std::exception &e)
        {
            LOG_ERR_F(SCAL, "stoull() failed parsing env var '{}'", c_scal_timeout_no_progress_env_var_name);
        }
    }
    return SCAL_SUCCESS;
}

void Scal::allocateCqIndex(unsigned&             cqIndex,
                           unsigned&             globalCqIndex,
                           const scaljson::json& json,
                           const unsigned        smID,
                           unsigned              dcoreIndex,
                           uint32_t              firstAvailableCq)
{
    cqIndex       = firstAvailableCq + m_syncManagers[smID].activeCQsNr;
    globalCqIndex = cqIndex + (c_cq_ctrs_in_dcore * dcoreIndex);

    m_syncManagers[smID].activeCQsNr++;
    if (m_syncManagers[smID].activeCQsNr > c_cq_ctrs_in_dcore - firstAvailableCq)
    {
        THROW_INVALID_CONFIG(json,
                             "too many cqs used in dcore {}. max num of cqs is {}",
                             smID, c_cq_ctrs_in_dcore - firstAvailableCq);
    }
}

int Scal::openConfigFileAndParseJson(const std::string & configFileName, scaljson::json &json)
{
    // open the file
    LOG_INFO(SCAL,"{}: fd={} loading config from {}", __FUNCTION__, m_fd, configFileName.empty() ? Scal::getDefaultJsonName() : configFileName);
    std::string content;
    if  (configFileName.empty() || configFileName.find(Scal::getDefaultJsonName()) == 0)
    {
        // default json
        LOG_INFO(SCAL,"{}: fd={} translating config from {} to {}", __FUNCTION__, m_fd, configFileName, getDefaultJsonName());
        content = getInternalFile(getDefaultJsonName());
        m_isInternalJson = true;
    }
    else if (configFileName.find(internalFileSignature) == 0)
    {
        // non default internal json
        LOG_INFO(SCAL,"{}: fd={} translating config from {} to {} (Non-default)", __FUNCTION__, m_fd, configFileName, configFileName.substr(2));
        content = getInternalFile(configFileName.substr(2));
        m_isInternalJson = true;
    }
    else
    {
        std::ifstream jsonFile(configFileName);
        if (jsonFile)
        {
            content.assign(std::istreambuf_iterator<char>(jsonFile), std::istreambuf_iterator<char>());
        }
    }
    if (content.empty())
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to load config from {}. config not found", __FUNCTION__, m_fd, configFileName);
        return SCAL_FILE_NOT_FOUND;
    }
    printConfigInfo(configFileName, content);
    // parse the json file
    try
    {
        if (m_isInternalJson)
        {
            json = scaljson::json::from_cbor(content);
        }
        else
        {
            json = scaljson::json::parse(content, nullptr, true, true);
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to parse config file {}. err={}", __FUNCTION__, m_fd, configFileName, e.what());
        return SCAL_INVALID_CONFIG;
    }

    const char * envVarName = "SCAL_CFG_OVERRIDE_PATH";
    const char * envVarValue = getenv(envVarName);
    if (envVarValue)
    {
        std::stringstream ss(envVarValue);
        std::vector<std::string> overrides;
        std::string token;
        // split the environment var value (comma separated) to a vectors of strings
        while(std::getline(ss, token, ','))
        {
            overrides.push_back(token);
        }

        for (auto& override : overrides)
        {
            LOG_INFO(SCAL,"{}: fd={} loading override config from  {}", __FUNCTION__, m_fd, override);
            std::ifstream jsonOverrideFile(override);
            if (!jsonOverrideFile)
            {
                LOG_ERR(SCAL,"{}: fd={} Failed to load override config from  {}", __FUNCTION__, m_fd, override);
                return SCAL_FILE_NOT_FOUND;
            }

            // parse the json file
            scaljson::json jsonOverride;
            try
            {
                jsonOverride = scaljson::json::parse(jsonOverrideFile, nullptr, true, true);
            }
            catch (const std::exception &e)
            {
                LOG_ERR(SCAL,"{}: fd={} Failed to parse override config file {}. err={}", __FUNCTION__, m_fd, override, e.what());
                return SCAL_INVALID_CONFIG;
            }
            if (jsonOverride.is_array())
            {
                json = json.patch(jsonOverride);
            }
            else
            {
                json.merge_patch(jsonOverride);
            }
        }
    }
    bool dump_json = (getenv("SCAL_DUMP_JSON") != nullptr);
    if (dump_json)
    {
        std::ofstream file;
        file.open(MERGED_JSON_DUMP_FILENAME);
        file << json.dump(4);
        file.close();
        LOG_INFO(SCAL,"merged json dumped to file {}", MERGED_JSON_DUMP_FILENAME);
    }

    const char * envVarAutoJson = "SCAL_CFG_AUTO_JSON";
    const char * envVarAutoJsonValue = getenv(envVarAutoJson);
    if (envVarAutoJsonValue)
    {
        LOG_INFO(SCAL, "Auto json: adjustment requested");
        JsonUpdateMask u(json, m_fd); // create the object that adjust the maks

        bool rtn = u.initMasks(); // get the masks from lkd

        if (rtn == true)
        {
            u.run(); // create a patch based on the masks (not, for testing we can "play" with the masks here by getting the masks as reference

            if (dump_json)
            {
                std::ofstream filePatch;
                filePatch.open(AUTO_JSON_PATCH_DUMP_FILENAME);
                filePatch << u.getPatch().dump(4);
                filePatch.close();
                LOG_INFO(SCAL,"merged json dumped to file {}", AUTO_JSON_PATCH_DUMP_FILENAME);
            }

            auto           jsonPatch = u.getPatch();
            scaljson::json newJson   = json.patch(jsonPatch); // do the patching

            json = newJson;

            if (dump_json)
            {
                std::ofstream fileFinal;
                fileFinal.open(AUTO_JSON_FINAL_DUMP_FILENAME);
                fileFinal << json.dump(4);
                fileFinal.close();
                LOG_INFO(SCAL,"merged json dumped to file {}", AUTO_JSON_FINAL_DUMP_FILENAME);
            }
        }
    }
    return SCAL_SUCCESS;
}

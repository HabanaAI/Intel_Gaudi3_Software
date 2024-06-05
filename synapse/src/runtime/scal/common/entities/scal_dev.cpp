#include "scal_dev.hpp"

#include "defs.h"

#include "synapse_common_types.h"

#include "runtime/scal/common/entities/scal_completion_group.hpp"
#include "runtime/scal/common/entities/scal_memory_pool.hpp"

#include "log_manager.h"

#include "runtime/scal/gaudi2/entities/streams_container.hpp"
#include "runtime/scal/gaudi2/entities/device_info.hpp"
#include "runtime/scal/gaudi3/entities/streams_container.hpp"
#include "runtime/scal/gaudi3/entities/device_info.hpp"

#define upper_32_bits(n)  ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n)  ((uint32_t)(n))

using namespace common;

static ResourceStreamType getResourceType(internalStreamType queueType)
{
    switch (queueType)
    {
        case INTERNAL_STREAM_TYPE_DMA_UP:
            return ResourceStreamType::USER_DMA_UP;

        case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
            return ResourceStreamType::USER_DEV_TO_DEV;

        case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
            return ResourceStreamType::USER_DMA_DOWN;

        case INTERNAL_STREAM_TYPE_COMPUTE:
            return ResourceStreamType::COMPUTE;

        default:
            break;
    }

    return ResourceStreamType::AMOUNT;
}

ScalDev::ScalDev(synDeviceType deviceType)
: m_devHndl(nullptr),
  m_memPools {nullptr, nullptr, nullptr}
{
    s_debugLastConstuctedDevice = this;

    if (deviceType == synDeviceGaudi2)
    {
        m_apDeviceInfoInterface = std::make_unique<gaudi2::DeviceInfo>();
        m_apStreamsContainer = std::unique_ptr<common::ScalStreamsContainer>(new gaudi2::StreamsContainer(m_apDeviceInfoInterface.get()));
    }
    else if (deviceType == synDeviceGaudi3)
    {
        m_apDeviceInfoInterface = std::make_unique<gaudi3::DeviceInfo>();
        m_apStreamsContainer = std::unique_ptr<common::ScalStreamsContainer>(new gaudi3::StreamsContainer(m_apDeviceInfoInterface.get()));
    }

    m_msixAddrress                 = m_apDeviceInfoInterface.get()->getMsixAddrress();
    m_msixUnexpectedInterruptValue = m_apDeviceInfoInterface.get()->getMsixUnexpectedInterruptValue();

    HB_ASSERT((m_apStreamsContainer != nullptr), "Invalid device-type for ScalDev");
}

synStatus ScalDev::acquire(int hlthunkFd, const std::string& scalCfgFile)
{
    HB_ASSERT((m_devHndl == nullptr), "devHndl 0x{:x} is not nullptr", (uint64_t)m_devHndl);

    LOG_INFO(SYN_DEVICE, "fd {} initializing", hlthunkFd);

    synStatus status = initScalDevice(hlthunkFd, scalCfgFile);
    if (status != synSuccess)
    {
        return status;
    }

    status = allocateMemoryPools();
    if (status != synSuccess)
    {
        return status;
    }

    status = allocateStreamsMonitors();
    if (status != synSuccess)
    {
        return status;
    }

    status = allocateTdrIrqMonitor();
    if (status != synSuccess && status != synUnavailable)
    {
        return status;
    }

    status = allocateStreams();
    if (status != synSuccess)
    {
        return status;
    }

    LOG_INFO(SYN_DEVICE, "devHndl 0x{:x} initialized", (uint64_t)m_devHndl);
    return synSuccess;
}

synStatus ScalDev::release()
{
    LOG_INFO(SYN_DEVICE, "devHndl 0x{:x} destroying", (uint64_t)m_devHndl);

    synStatus status = releaseStreams();
    if (status != synSuccess)
    {
        return status;
    }

    releaseMemoryPools();

    releaseScalDevice();

    LOG_INFO(SYN_DEVICE, "Device destroyed");
    return synSuccess;
}

bool ScalDev::getFreeStream(internalStreamType queueType, StreamAndIndex& streamInfo)
{
    ResourceStreamType resourceType = getResourceType(queueType);
    if ((resourceType == ResourceStreamType::COMPUTE) ||
        (resourceType >= ResourceStreamType::AMOUNT))
    {
        return false;
    }

    return m_apStreamsContainer->getFreeStream(resourceType, streamInfo);
}

ScalStreamBaseInterface* ScalDev::getFreeComputeResources(const ComputeCompoundResources*& pComputeCompoundResource)
{
    return m_apStreamsContainer->getFreeComputeResources(pComputeCompoundResource);
}

ScalStreamBaseInterface* ScalDev::debugGetCreatedStream(internalStreamType queueType)
{
    ResourceStreamType resourceType = getResourceType(queueType);
    if ((resourceType == ResourceStreamType::COMPUTE) ||
        (resourceType == ResourceStreamType::AMOUNT))
    {
        return nullptr;
    }

    return m_apStreamsContainer->debugGetCreatedStream(resourceType);
}

ScalStreamBaseInterface*
ScalDev::debugGetCreatedComputeResources(const ComputeCompoundResources*& pComputeCompoundResource)
{
    return m_apStreamsContainer->debugGetCreatedComputeResources(pComputeCompoundResource);
}

bool ScalDev::isDireceModeStream(internalStreamType queueType) const
{
    StreamModeType streamModeType;

    ResourceStreamType resourceType = getResourceType(queueType);
    bool status = m_apStreamsContainer->getResourcesStreamMode(streamModeType, resourceType);
    if (!status)
    {
        return false;
    }

    return (streamModeType == StreamModeType::DIRECT);
}

synStatus ScalDev::releaseStream(ScalStreamBaseInterface* pScalStream)
{
    return m_apStreamsContainer->releaseStream(pScalStream);
}

synStatus ScalDev::releaseComputeResources(ScalStreamBaseInterface* pComputeStream)
{
    return m_apStreamsContainer->releaseComputeResources(pComputeStream);
}

ScalMemoryPool* ScalDev::getMemoryPool(MemoryPoolType type)
{
    HB_ASSERT(type != MEMORY_POOL_LAST, "Invalid memory type");
    return m_memPools[type];
}

synStatus ScalDev::getMemoryPoolStatus(MemoryPoolType memoryPoolType, PoolMemoryStatus& poolMemoryStatus) const
{
    return m_memPools[memoryPoolType]->getMemoryStatus(poolMemoryStatus);
}

scal_handle_t ScalDev::getScalHandle() const
{
    return m_devHndl;
}

synStatus ScalDev::initScalDevice(int hlthunkFd, const std::string& scalCfgFile)
{
    std::string loggerPath;
    synapse::LogManager::getLogsFolderPath(loggerPath);
    if (scal_set_logs_folder(loggerPath.c_str()) != SCAL_SUCCESS)
    {
        LOG_WARN(SYN_API, "failed to set logger path {} to scal. maybe scal was initialized before.", loggerPath);
    }

    LOG_INFO(SYN_DEVICE, "Device initializing... file {}", scalCfgFile);
    ScalRtn rc = scal_init(hlthunkFd, scalCfgFile.c_str(), &m_devHndl, m_apStreamsContainer->getArcFwConfigHandle());

    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE, "Device scal_init fail with rc {}", rc);
        return synFail;
    }

    LOG_INFO(SYN_DEVICE, "Device SCAL initialized");
    return synSuccess;
}

synStatus ScalDev::releaseScalDevice()
{
    LOG_INFO(SYN_DEVICE, "m_devHndl {:#x}", TO64(m_devHndl));
    if (m_devHndl != nullptr)
    {
        scal_destroy(m_devHndl);
        m_devHndl = nullptr;
    }

    return synSuccess;
}

synStatus ScalDev::allocateMemoryPools()
{
    const std::string MEMORY_POOL_NAMES[MEMORY_POOL_LAST] {"global_hbm", "host_shared", "hbm_shared"};

    for (uint32_t mpType = MEMORY_POOL_GLOBAL; mpType < MEMORY_POOL_LAST; mpType++)
    {
        m_memPools[mpType] = new ScalMemoryPool(m_devHndl, MEMORY_POOL_NAMES[mpType]);
        synStatus status   = m_memPools[mpType]->init();
        if (status != synSuccess)
        {
            return status;
        }
    }

    LOG_INFO(SYN_DEVICE, "Device memory-pools allocated");
    return synSuccess;
}

void ScalDev::releaseMemoryPools()
{
    for (uint32_t mpType = MEMORY_POOL_GLOBAL; mpType < MEMORY_POOL_LAST; mpType++)
    {
        if (m_memPools[mpType] != nullptr)
        {
            delete m_memPools[mpType];
            m_memPools[mpType] = nullptr;
        }
    }
}

synStatus ScalDev::allocateStreams()
{
    synStatus status = m_apStreamsContainer->createStreams(m_devHndl,
                                                           *m_memPools[MEMORY_POOL_HOST_SHARED],
                                                           m_streamsMonitors,
                                                           m_streamsFences);
    if (status != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE, "Create Device-Streams failed with rc {}", status);
        return synFail;
    }

    // make sure we have the correct number of compute streams defined
    assert(m_apStreamsContainer->getResourcesAmount(ResourceStreamType::COMPUTE) == MaxNumOfComputeStreams);

    LOG_INFO(SYN_DEVICE, "Device streams allocated");
    return status;
}

synStatus ScalDev::allocateStreamsMonitors()
{
    scal_monitor_pool_handle_t monPoolHandle;
    scal_monitor_pool_info     monPoolInfo;
    // Monitor ID
    ScalRtn rc = scal_get_so_monitor_handle_by_name(m_devHndl, "compute_gp_monitors", &monPoolHandle);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE,
                     "devHndl 0x{:x} scal_get_pool_handle_by_name failed with rc {}",
                     TO64(m_devHndl),
                     rc);
        return synFail;
    }

    rc = scal_monitor_pool_get_info(monPoolHandle, &monPoolInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE,
                     "devHndl 0x{:x} scal_monitor_pool_get_info failed with rc {}",
                     TO64(m_devHndl),
                     rc);
        return synFail;
    }
    const unsigned longMonitorJump = 4;  // long monitor uses 4 monitors
    LOG_INFO(SYN_DEVICE, "init stream monitors BaseIdx {:x} size {:x}", monPoolInfo.baseIdx, monPoolInfo.size);
    synStatus status = m_streamsMonitors.init(monPoolInfo.baseIdx, monPoolInfo.size, longMonitorJump);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "m_streamsMonitors.init failed with rc {}", rc);
        return synFail;
    }

    status = m_streamsFences.init(0, numberOfFences);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "m_streamsFences.init failed with rc {}", rc);
        return synFail;
    }

    LOG_INFO(SYN_DEVICE, "Device streams monitors allocated");
    return synSuccess;
}

synStatus ScalDev::allocateTdrIrqMonitor()
{
    scal_monitor_pool_handle_t monPoolHandle;
    scal_monitor_pool_info     monPoolInfo;

    ScalRtn rc = scal_get_so_monitor_handle_by_name(m_devHndl, "compute_tdr_irq_mon", &monPoolHandle);
    if (rc == SCAL_NOT_FOUND)
    {
        LOG_TRACE(SYN_DEVICE,
                       "devHndl 0x{:x} scal_get_pool_handle_by_name not found rc {}",
                       TO64(m_devHndl),
                       rc);
        return synUnavailable;
    }
    else if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE,
                     "devHndl 0x{:x} scal_get_pool_handle_by_name failed with rc {}",
                     TO64(m_devHndl),
                     rc);
        return synFail;
    }

    rc = scal_monitor_pool_get_info(monPoolHandle, &monPoolInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE,
                     "devHndl 0x{:x} scal_monitor_pool_get_info failed with rc {}",
                     TO64(m_devHndl),
                     rc);
        return synFail;
    }

    scal_handle_t tdrIrqMonScalHandle = monPoolInfo.scal;
    unsigned      tdrIrqMonSmId       = monPoolInfo.smIndex;
    uint64_t      tdrIrqMonId         = monPoolInfo.baseIdx;

    scal_sm_info_t smInfo;
    rc = scal_get_sm_info(tdrIrqMonScalHandle, tdrIrqMonSmId, &smInfo);

    uint64_t configRegOffset      = m_apDeviceInfoInterface.get()->getMonitorConfigRegisterSmOffset(tdrIrqMonId) / sizeof(uint32_t);
    uint64_t targetAddrLowOffset  = m_apDeviceInfoInterface.get()->getMonitorPayloadAddrLowRegisterSmOffset(tdrIrqMonId) / sizeof(uint32_t);
    uint64_t targetAddrHighOffset = m_apDeviceInfoInterface.get()->getMonitorPayloadAddrHighRegisterSmOffset(tdrIrqMonId) / sizeof(uint32_t);
    uint64_t targetValueOffset    = m_apDeviceInfoInterface.get()->getMonitorPayloadDataRegisterSmOffset(tdrIrqMonId) / sizeof(uint32_t);
    uint64_t monitorArmOffset     = m_apDeviceInfoInterface.get()->getMonitorArmRegisterSmOffset(tdrIrqMonId) / sizeof(uint32_t);

    // config
    scal_write_mapped_reg(&smInfo.objs[configRegOffset], 0);
    LOG_TRACE(SYN_DEVICE,
                   "{}: Write to mapped register address {} value 0",
                   HLLOG_FUNC,
                   TO64(&smInfo.objs[configRegOffset]));

    // Completion write addr lower bits
    scal_write_mapped_reg(&smInfo.objs[targetAddrLowOffset], lower_32_bits(m_msixAddrress));
    LOG_TRACE(SYN_DEVICE,
                   "{}: Write to mapped register address {} value {}",
                   HLLOG_FUNC,
                   TO64(&smInfo.objs[targetAddrLowOffset]),
                   TO64(lower_32_bits(m_msixAddrress)));

    // Completion write addr upper bits
    scal_write_mapped_reg(&smInfo.objs[targetAddrHighOffset], upper_32_bits(m_msixAddrress));
    LOG_TRACE(SYN_DEVICE,
                   "{}: Write to mapped register address {} value {}",
                   HLLOG_FUNC,
                   TO64(&smInfo.objs[targetAddrHighOffset]),
                   TO64(upper_32_bits(m_msixAddrress)));

    // Completion write value
    scal_write_mapped_reg(&smInfo.objs[targetValueOffset], m_msixUnexpectedInterruptValue);
    LOG_TRACE(SYN_DEVICE,
                   "{}: Write to mapped register address {} value {}",
                   HLLOG_FUNC,
                   TO64(&smInfo.objs[targetValueOffset]),
                   TO64(m_msixUnexpectedInterruptValue));

    m_tdrIrqMonitorArmRegAddr = &smInfo.objs[monitorArmOffset];

    LOG_INFO(SYN_DEVICE, "Device TDR IRQ monitor allocated and configured");

    return synSuccess;
}

synStatus ScalDev::releaseStreams()
{
    return m_apStreamsContainer->releaseAllDeviceStreams();
}

ScalDevSpecificInfo ScalDev::getDevSpecificInfo()
{
    PoolMemoryStatus poolMemoryStatus;

    m_memPools[MEMORY_POOL_GLOBAL]->getMemoryStatus(poolMemoryStatus);

    ScalDevSpecificInfo devSpecificInfo;

    devSpecificInfo.dramBaseAddr = poolMemoryStatus.devBaseAddr;
    devSpecificInfo.dramEndAddr  = poolMemoryStatus.devBaseAddr + poolMemoryStatus.totalSize;

    LOG_INFO(SYN_DEVICE, "HBM GLBL ADDR {:x}-{:x}", devSpecificInfo.dramBaseAddr, devSpecificInfo.dramEndAddr);

    return devSpecificInfo;
}

synStatus ScalDev::getClusterInfo(scal_cluster_info_t& clusterInfo, char* clusterName)
{
    scal_cluster_handle_t clusterHandle;

    int scalStatus = scal_get_cluster_handle_by_name(m_devHndl, clusterName, &clusterHandle);
    if (scalStatus != SCAL_SUCCESS)
    {
        if (scalStatus == SCAL_NOT_FOUND)
        {
            LOG_WARN(SYN_DEVICE, "Cluster {} was not found", clusterName);
            return synUnavailable;
        }
        LOG_ERR(SYN_DEVICE, "Failed to get cluster handle for name {}", clusterName);
        return synFail;
    }

    scalStatus = scal_cluster_get_info(clusterHandle, &clusterInfo);
    if (scalStatus != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get cluster info for name {}", clusterName);
        return synFail;
    }
    return synSuccess;
}

ScalDev* ScalDev::s_debugLastConstuctedDevice = nullptr;

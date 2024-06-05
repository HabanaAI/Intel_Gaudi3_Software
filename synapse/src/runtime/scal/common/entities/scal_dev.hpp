#pragma once

#include "define_synapse_common.hpp"
#include "synapse_common_types.h"

#include "scal_streams_fences.hpp"
#include "scal_streams_monitors.hpp"

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/entities/scal_streams_container.hpp"

#include <memory>

struct PoolMemoryStatus;
class ScalMemoryPool;
class ScalCompletionGroup;
class ScalStreamBaseInterface;

class ScalDev
{
public:
    ScalDev(synDeviceType deviceType);
    ~ScalDev() = default;

    synStatus acquire(int hlthunkFd, const std::string& scalCfgFile);

    synStatus release();

    bool                             getFreeStream(internalStreamType queueType, StreamAndIndex& streamInfo);
    ScalStreamBaseInterface*         getFreeComputeResources(const ComputeCompoundResources*& pComputeCompoundResource);

    ScalStreamBaseInterface* debugGetCreatedStream(internalStreamType queueType);
    ScalStreamBaseInterface* debugGetCreatedComputeResources(const ComputeCompoundResources*& pComputeCompoundResource);

    bool isDireceModeStream(internalStreamType queueType) const;

    synStatus releaseStream(ScalStreamBaseInterface* pScalStream);
    synStatus releaseComputeResources(ScalStreamBaseInterface* pComputeStream);

    enum MemoryPoolType
    {
        MEMORY_POOL_GLOBAL,
        MEMORY_POOL_HOST_SHARED,
        MEMORY_POOL_HBM_SHARED,
        MEMORY_POOL_LAST,
    };

    ScalMemoryPool* getMemoryPool(MemoryPoolType type);
    synStatus       getMemoryPoolStatus(MemoryPoolType memoryPoolType, PoolMemoryStatus& poolMemoryStatus) const;

    scal_handle_t   getScalHandle() const;
    static ScalDev* debugGetLastConstuctedDevice() { return s_debugLastConstuctedDevice; }

    ScalDevSpecificInfo getDevSpecificInfo();

    synStatus getClusterInfo(scal_cluster_info_t& clusterInfo, char* clusterName);

    static const unsigned MaxNumOfComputeStreams = 3;

    inline volatile uint32_t* getTdrIrqMonitorArmRegAddr() { return m_tdrIrqMonitorArmRegAddr; }
    void setTimeouts(scal_timeouts_t const & timeouts, bool disableTimeouts)
    {
        scal_set_timeouts(m_devHndl, &timeouts);
        scal_disable_timeouts(m_devHndl, disableTimeouts);
    }

    const common::DeviceInfoInterface* getDeviceInfoInterface() { return m_apDeviceInfoInterface.get(); }

private:
    synStatus initScalDevice(int hlthunkFd, const std::string& scalCfgFile);
    synStatus releaseScalDevice();
    synStatus allocateMemoryPools();
    void      releaseMemoryPools();
    synStatus allocateStreams();
    synStatus releaseStreams();
    synStatus allocateStreamsMonitors();
    synStatus allocateTdrIrqMonitor();

    static ScalDev* s_debugLastConstuctedDevice;

    scal_handle_t m_devHndl;

    std::unique_ptr<common::ScalStreamsContainer> m_apStreamsContainer;

    ScalMemoryPool*     m_memPools[MEMORY_POOL_LAST];
    ScalStreamsMonitors m_streamsMonitors;
    ScalStreamsFences   m_streamsFences;

    std::unique_ptr<const common::DeviceInfoInterface> m_apDeviceInfoInterface;

    volatile uint32_t* m_tdrIrqMonitorArmRegAddr = nullptr;

    const unsigned numberOfFences = 32;

    uint64_t m_msixAddrress;
    uint32_t m_msixUnexpectedInterruptValue;
};

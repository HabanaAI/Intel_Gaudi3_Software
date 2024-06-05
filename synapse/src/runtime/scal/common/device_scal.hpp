#pragma once

#include "hcl_public_streams.h"
#include "runtime/common/hcl/hcl_api_wrapper.h"

#include "runtime/common/device/device_common.hpp"

#include "runtime/scal/common/scal_event.hpp"

#include "runtime/scal/common/entities/scal_dev.hpp"
#include "runtime/scal/common/entities/scal_stream_base_interface.hpp"
#include "runtime/scal/common/entities/scal_streams_container.hpp"

#include "defs.h"

// Infra
#include "event_dispatcher.hpp"

class QueueInterface;
struct LaunchInfo;

namespace common
{
class CoeffTableConf;
}

class ScalStreamComputeInterface;

namespace common
{
class DeviceScal : public DeviceCommon
{
public:
    DeviceScal(synDeviceType deviceType, const DeviceConstructInfo& deviceConstructInfo);

    virtual ~DeviceScal();

    virtual synStatus release(std::atomic<bool>& rDeviceBeingReleased) override;

    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) override;

    virtual synStatus getDramMemInfo(uint64_t& free, uint64_t& total) const override;

    virtual synStatus
    createStreamQueue(QueueType queueType, uint32_t flags, bool isReduced, QueueInterface*& rpQueueInterface) override;

    virtual synStatus destroyStreamQueue(QueueInterface* pQueueInterface) override;

    virtual synStatus synchronizeEvent(const EventInterface* pEventInterface) override;

    virtual synStatus createEvent(synEventHandle* pEventHandle, const unsigned int flags) override;

    virtual synStatus destroyEvent(synEventHandle eventHandle) override;

    virtual EventSptr getEventSptr(synEventHandle eventHandle) override;

    virtual synStatus eventQuery(const EventInterface* streamHandle) override;

    virtual synStatus getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const override;

    virtual synStatus
    kernelsPrintf(const InternalRecipeHandle& rInternalRecipeHandle, uint64_t wsAddr, void* hostBuff) override;

    virtual void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle) override;

    virtual synStatus getDeviceInfo(synDeviceInfo& rDeviceInfo) const override;
    virtual synStatus getDeviceInfo(synDeviceInfoV2& rDeviceInfo) const override;

    virtual void
    checkDevFailure(uint64_t csSeqTimeout, DfaStatus dfaStatus, ChkDevFailOpt option, bool isSimulator) override;

    synStatus getClusterInfo(scal_cluster_info_t& clusterInfo, char* clusterName);

    synStatus releasePreAllocatedMem();

    static uint64_t getHbmGlblSize();
    static uint64_t getHbmGlblMaxRecipeSize();
    static uint64_t getHbmSharedMaxRecipeSize();

    uint64_t getNumOfStreams() { return m_queueInterfaces.size(); };

    void   dfaInfo(DfaReq dfaReq, uint64_t csSeq = 0);
    TdrRtn streamsTdr(TdrType tdrType);

    virtual bool isDirectModeUserDownloadStream() const override;

    virtual synStatus getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                      synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;
    // for testing only
    void setTimeouts(scal_timeouts_t const & timeouts, bool disableTimeouts)
    {
        m_scalDev.setTimeouts(timeouts, disableTimeouts);
    }

    scal_handle_t testOnlyGetScalHandle() const { return m_scalDev.getScalHandle(); }

    virtual void getDeviceHbmVirtualAddresses(uint64_t& hbmBaseAddr, uint64_t& hbmEndAddr) override;

protected:
    virtual synStatus eventRecord(EventInterface* pEventInterface, synStreamHandle streamHandle) override;

    virtual synStatus streamGenericWaitEvent(synStreamHandle       streamHandle,
                                             const EventInterface& rEventInterface,
                                             const unsigned int    flags) override;

    virtual synStatus _acquire(const uint16_t numSyncObj, common::CoeffTableConf& rCoeffTableConf);

private:
    void notifyAllRecipeRemoval();

    virtual synStatus launch(Stream*                       pStream,
                             const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsInfoAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             EventWithMappedTensorDB&      events,
                             uint32_t                      flags) override;

    const static uint64_t hbmSharedSize = 32 * 1024 * 1024;

    synStatus preAllocateMemoryForComputeStreams();

    virtual void bgWork() override;
    virtual void debugCheckWorkStatus() override;

    synStatus createStreamScal(QueueInterface*&                rpQueueInterface,
                               const BasicQueueInfo&           rBasicQueueInfo,
                               internalStreamType              internalType,
                               ScalStreamBaseInterface*        pScalStream,
                               unsigned                        streamIdx,
                               const ComputeCompoundResources* pComputeResources);

    void removeFromStreamHandlesL(QueueInterface* pQueueInterface);

    QueueInterface* getAnyStream(internalStreamType requestedType);

    synStatus clearKernelsPrintfWs(uint64_t workspaceAddress, uint64_t workspaceSize);

    void checkArcsHeartBeat(bool isSimulator);

    synStatus releaseAllStreams();

    synStatus getTdrIrqMonitorArmRegAddr(volatile uint32_t*& tdrIrqMonitorArmRegAddr) override;

    static const uint32_t MAX_AFFINITY_QUEUES_NUM;

    // This mutex is to protect m_queueInterfaces
    // mutable std::shared_mutex m_queueInterfacesMutex;
    // This keeps a list of all streams. Used to destroy
    // all streams if needed. Searching on it is linear, but I
    // don't see a reason to make it more efficient for now
    std::deque<QueueInterface*> m_queueInterfaces;

    ScalDev                     m_scalDev;
    ScalEventsPool*             m_scalEventsPool;
    hclApiWrapper               m_hclApiWrapper;
    ScalDevSpecificInfo         m_devSpecificInfo;
    PreAllocatedStreamMemoryAll m_preAllocatedMem[ScalDev::MaxNumOfComputeStreams] {};
    EventConnection             m_eventConnection;
    uint64_t                    m_collectiveStreamNum;
    bool                        m_hclInit;

    static const AffinityCountersArray s_maxAffinitiesDefault;
    static const AffinityCountersArray s_maxAffinitiesHCLDisable;
};
}  // namespace common

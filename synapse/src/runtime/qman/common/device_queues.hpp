#pragma once

#include <memory>

#include "queue_handle_container.hpp"
#include "runtime/common/common_types.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "define_synapse_common.hpp"
#include "queue_creator.hpp"
#include "runtime/qman/common/qman_event.hpp"
#include "infra/containers/slot_map.hpp"
#include "runtime/common/device/dfa_base.hpp"

class DataChunksAllocatorCommandBuffer;
class PhysicalQueuesManager;
class DataChunksAllocator;
class CommandSubmissionDataChunks;
class QueueInterface;
class WorkCompletionManagerInterface;
class DevMemoryAllocInterface;
class DeviceRecipeAddressesGeneratorInterface;
class DeviceRecipeDownloaderContainer;

class DeviceQueues
{
public:
    DeviceQueues(synDeviceType                            devType,
                 const uint32_t                           amountOfEnginesInComputeArbGroup,
                 DevMemoryAllocInterface&                 rDevMemAlloc,
                 DeviceRecipeAddressesGeneratorInterface& rDevRecipeAddress,
                 WorkCompletionManagerInterface&          rWorkCompletionManager);
    virtual ~DeviceQueues() = default;

    void init(uint16_t numSyncObj);

    void finalize();

    synStatus createEvent(synEventHandle* pEventHandle, const unsigned int flags);

    synStatus destroyEvent(synEventHandle eventHandle);

    SlotMapItemSptr<QmanEvent> getEventSptr(synEventHandle eventHandle);

    TrainingRetCode validateEventHandle(const QmanEvent* pEventHandle);

    synStatus createStream(internalStreamType               internalStreamType,
                           const uint32_t                   flags,
                           bool                             isReduced,
                           DeviceRecipeDownloaderContainer& rDeviceRecipeDownloaderContainer,
                           QueueInterface*&                 rpQueueInterface);

    synStatus destroyStream(QueueInterface* pQueueInterface);

    void finalizeStreams();

    synStatus createDownSynStream();

    synStatus destroyDownSynStream();

    synStatus destroyUserStreams();

    synStatus getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const;

    void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
    {
        m_queueInterfaceContainer.notifyRecipeRemoval(rRecipeHandle);
    }

    void notifyAllRecipeRemoval();

    static std::string getInternalStreamName(internalStreamType queueType);

    QueueInterface* getDmaDownStream();

    std::deque<QueueInterface*>& getQueueInterfaces() { return m_queueInterfaceContainer.getQueueInterfaces(); }

    synStatus synchronizeAllStreams();

    void logStreamsSyncHistory();

private:
    typedef std::shared_ptr<PhysicalQueuesManager> spPhysicalStreamsManager;

    static common::StreamCreator* getStreamCreator(synDeviceType deviceType);

    bool createQueue(QueueInterface*&                 rpQueueInterface,
                     const BasicQueueInfo&            rBasicQueueInfo,
                     synDeviceType                    deviceType,
                     PhysicalQueuesManager*           pPhysicalStreamsManager,
                     uint32_t                         physicalQueueOffset,
                     uint32_t                         amountOfEnginesInArbGroup,
                     bool                             isReduced,
                     DeviceRecipeDownloaderContainer* pDeviceRecipeDownloaderContainer);

    synStatus createStream(internalStreamType               internalType,
                           const uint32_t                   flags,
                           bool                             isReduced,
                           DeviceRecipeDownloaderContainer* pDeviceRecipeDownloaderContainer,
                           uint32_t                         amountOfEnginesInArbGroup,
                           QueueInterface*&                 rpQueueInterface);

    const synDeviceType                      m_devType;
    const uint32_t                           m_amountOfEnginesInComputeArbGroup;
    spPhysicalStreamsManager                 m_pPhysicalStreamsManager;
    QueueInterface*                          m_pDmaDownQueueInterface;
    QueueInterfaceContainer                  m_queueInterfaceContainer;
    EventsPool                               m_eventsPool;
    WorkCompletionManagerInterface&          m_rWorkCompletionManager;
    DevMemoryAllocInterface&                 m_rDevMemAlloc;
    DeviceRecipeAddressesGeneratorInterface& m_rDevRecipeAddress;
};

#include "device_queues.hpp"

#include "defenders.h"
#include "infra/threads/single_execution_owner.hpp"
#include "internal/hccl_internal.h"
#include "physical_queues_manager.hpp"
#include "queue_collective_network.hpp"
#include "queue_compute_qman.hpp"
#include "queue_copy_qman.hpp"
#include "queue_creator.hpp"
#include "runtime/qman/common/data_chunk/data_chunks_allocator.hpp"
#include "runtime/qman/common/device_recipe_downloader_container.hpp"
#include "runtime/qman/common/submit_command_buffers.hpp"
#include "runtime/qman/gaudi/stream_creator.hpp"
#include "syn_singleton.hpp"
#include "types_exception.h"

#define VERIFY_TRAINING_SIGNAL_MGR_RTN_CODE(rtn)                                                                       \
    if (m_pPhysicalStreamsManager == nullptr)                                                                          \
    {                                                                                                                  \
        LOG_ERR(SYN_DEVICE, "{}: no instance of TSM", HLLOG_FUNC);                                                     \
        return rtn;                                                                                                    \
    }

#define VERIFY_TRAINING_SIGNAL_MGR() VERIFY_TRAINING_SIGNAL_MGR_RTN_CODE(synUnsupported)

static SubmitCommandBuffers submitter;

DeviceQueues::DeviceQueues(synDeviceType                            devType,
                           const uint32_t                           amountOfEnginesInComputeArbGroup,
                           DevMemoryAllocInterface&                 rDevMemAlloc,
                           DeviceRecipeAddressesGeneratorInterface& rDevRecipeAddress,
                           WorkCompletionManagerInterface&          rWorkCompletionManager)
: m_devType(devType),
  m_amountOfEnginesInComputeArbGroup(amountOfEnginesInComputeArbGroup),
  m_pDmaDownQueueInterface(nullptr),
  m_rWorkCompletionManager(rWorkCompletionManager),
  m_rDevMemAlloc(rDevMemAlloc),
  m_rDevRecipeAddress(rDevRecipeAddress)
{
}

void DeviceQueues::init(uint16_t numSyncObj)
{
    m_pPhysicalStreamsManager.reset(new PhysicalQueuesManager(getStreamCreator(m_devType)));
}

void DeviceQueues::finalize()
{
    m_queueInterfaceContainer.clearAll();

    m_pPhysicalStreamsManager.reset();
}

synStatus DeviceQueues::createEvent(synEventHandle* pEventHandle, const unsigned int flags)
{
    VERIFY_TRAINING_SIGNAL_MGR();

    bool rtn = m_eventsPool.getNewEvent(*pEventHandle, flags);

    if (rtn == false)
    {
        *pEventHandle = nullptr;
        return synAllResourcesTaken;
    }
    return synSuccess;
}

synStatus DeviceQueues::destroyEvent(synEventHandle eventHandle)
{
    VERIFY_TRAINING_SIGNAL_MGR();

    bool rtn = m_eventsPool.destroyEvent(eventHandle);

    if (rtn == false)
    {
        return synFail;
    }
    return synSuccess;
}

SlotMapItemSptr<QmanEvent> DeviceQueues::getEventSptr(synEventHandle eventHandle)
{
    return m_eventsPool.getEventSptr(eventHandle);
}

TrainingRetCode DeviceQueues::validateEventHandle(const QmanEvent* pEventHandle)
{
    VERIFY_TRAINING_SIGNAL_MGR_RTN_CODE(TRAINING_RET_CODE_INVALID_REQUEST);
    return TRAINING_RET_CODE_SUCCESS;
}

synStatus DeviceQueues::createStream(internalStreamType               internalStreamType,
                                     const uint32_t                   flags,
                                     bool                             isReduced,
                                     DeviceRecipeDownloaderContainer& rDeviceRecipeDownloaderContainer,
                                     QueueInterface*&                 rpQueueInterface)
{
    uint32_t amountOfEnginesInArbGroup = 0;

    if (internalStreamType == INTERNAL_STREAM_TYPE_COMPUTE)
    {
        amountOfEnginesInArbGroup = m_amountOfEnginesInComputeArbGroup;
    }

    return createStream(internalStreamType,
                        flags,
                        isReduced,
                        &rDeviceRecipeDownloaderContainer,
                        amountOfEnginesInArbGroup,
                        rpQueueInterface);
}

synStatus DeviceQueues::createStream(internalStreamType               internalType,
                                     const uint32_t                   flags,
                                     bool                             isReduced,
                                     DeviceRecipeDownloaderContainer* pDeviceRecipeDownloaderContainer,
                                     uint32_t                         amountOfEnginesInArbGroup,
                                     QueueInterface*&                 rpQueueInterface)
{
    VERIFY_TRAINING_SIGNAL_MGR();

    BasicQueueInfo basicQueueInfo {{0, 0}, INTERNAL_STREAM_TYPE_NUM, TRAINING_QUEUE_NUM, 0, nullptr};
    basicQueueInfo.queueType   = internalType;
    basicQueueInfo.logicalType = TRAINING_QUEUE_NUM;

    TrainingRetCode retCode = m_pPhysicalStreamsManager->createStream(basicQueueInfo, flags);
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_DEVICE, "{}: Cant create stream {} on device retCode={}", HLLOG_FUNC, internalType, retCode);
        return (retCode == TRAINING_RET_CODE_FULLY_USED)
                   ? synAllResourcesTaken
                   : ((retCode == TRAINING_RET_CODE_INVALID_REQUEST) ? synInvalidArgument : synFail);
    }

    QueueInterface* pQueueInterface = nullptr;
    bool            status          = createQueue(pQueueInterface,
                              basicQueueInfo,
                              m_devType,
                              m_pPhysicalStreamsManager.get(),
                              m_pPhysicalStreamsManager->getStreamPhysicalOffset(basicQueueInfo),
                              amountOfEnginesInArbGroup,
                              isReduced,
                              pDeviceRecipeDownloaderContainer);
    if (!status)
    {
        LOG_ERR(SYN_DEVICE, "{}: Can not initialize a stream on device", HLLOG_FUNC);
        m_pPhysicalStreamsManager->destroyStream(basicQueueInfo);
        return synFail;
    }

    status = m_queueInterfaceContainer.addStreamHandle(pQueueInterface);
    if (!status)
    {
        LOG_ERR(SYN_DEVICE, "{}: Can not add a stream on device", HLLOG_FUNC);
        pQueueInterface->finalize();
        delete pQueueInterface;
        m_pPhysicalStreamsManager->destroyStream(basicQueueInfo);
        return synFail;
    }

    rpQueueInterface = pQueueInterface;

    return synSuccess;
}

synStatus DeviceQueues::destroyStream(QueueInterface* pQueueInterface)
{
    if (pQueueInterface == nullptr)
    {
        LOG_DEBUG(SYN_API, "Stream-handle 0x{:x} had already been destroyed", (uint64_t)pQueueInterface);
        return synSuccess;
    }

    VERIFY_TRAINING_SIGNAL_MGR();

    QueueInterface* pQueueInterfaceLocal = pQueueInterface;
    BasicQueueInfo  basicQueueInfo       = pQueueInterfaceLocal->getBasicQueueInfo();

    bool status = m_queueInterfaceContainer.removeStreamHandle(pQueueInterface);
    if (!status)
    {
        LOG_ERR(SYN_DEVICE, "Can not remove stream-handle {}", basicQueueInfo.getDescription());
        return synFail;
    }

    pQueueInterface->finalize();

    delete pQueueInterface;

    m_pPhysicalStreamsManager->destroyStream(basicQueueInfo);

    return synSuccess;
}

synStatus DeviceQueues::destroyDownSynStream()
{
    std::deque<QueueInterface*>& deviceStreamsDB = m_queueInterfaceContainer.getQueueInterfaces();

    if (deviceStreamsDB.size() != 1)
    {
        LOG_ERR(SYN_DEVICE, "{}: There is more than 1 stream {}", HLLOG_FUNC, deviceStreamsDB.size());
        return synFail;
    }

    HB_ASSERT(m_pDmaDownQueueInterface != nullptr, "Synapse DMA Down Compute stream is not created");

    QueueInterface* pQueueInterface = deviceStreamsDB.front();
    if (pQueueInterface->getBasicQueueInfo().queueType != INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE)
    {
        LOG_ERR(SYN_DEVICE, "{}: Invalid stream type {}", HLLOG_FUNC, pQueueInterface->getBasicQueueInfo().queueType);
        return synFail;
    }

    synStatus status = destroyStream(pQueueInterface);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Can not destroy stream {}", HLLOG_FUNC, TO64(pQueueInterface));
        return status;
    }

    deviceStreamsDB.clear();

    m_pDmaDownQueueInterface = nullptr;

    return synSuccess;
}

void DeviceQueues::finalizeStreams()
{
    std::deque<QueueInterface*>& deviceStreamsDB = m_queueInterfaceContainer.getQueueInterfaces();

    for (QueueInterface* pQueueInterface : deviceStreamsDB)
    {
        pQueueInterface->finalize();
    }
}

synStatus DeviceQueues::destroyUserStreams()
{
    std::deque<QueueInterface*>& deviceStreamsDB = m_queueInterfaceContainer.getQueueInterfaces();

    for (QueueInterface* pQueueInterface : deviceStreamsDB)
    {
        if (pQueueInterface->getBasicQueueInfo().queueType != INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE)
        {
            synStatus status = destroyStream(pQueueInterface);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_DEVICE, "{}: Can not destroy stream {}", HLLOG_FUNC, TO64(pQueueInterface));
                return status;
            }
        }
    }

    return synSuccess;
}

synStatus DeviceQueues::createDownSynStream()
{
    HB_ASSERT(m_pDmaDownQueueInterface == nullptr, "Synapse DMA Down Compute stream already created");

    uint32_t flags                     = 0;
    uint32_t amountOfEnginesInArbGroup = 0;  // Ignore ARB engines verification
    return createStream(INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE,
                        flags,
                        false,
                        nullptr,
                        amountOfEnginesInArbGroup,
                        m_pDmaDownQueueInterface);
}

void DeviceQueues::notifyAllRecipeRemoval()
{
    m_queueInterfaceContainer.notifyAllRecipeRemoval();
};

synStatus DeviceQueues::getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const
{
    return m_queueInterfaceContainer.getDeviceTotalStreamMappedMemory(totalStreamMappedMemorySize);
}

QueueInterface* DeviceQueues::getDmaDownStream()
{
    return m_pDmaDownQueueInterface;
}

synStatus DeviceQueues::synchronizeAllStreams()
{
    if (!m_pPhysicalStreamsManager)
    {
        LOG_ERR(SYN_STREAM, "{}: no instance of TSM", HLLOG_FUNC);
        return synUnsupported;
    }

    InternalWaitHandlesVector streamWaitHandles;
    // Todo avoid using the non interface method and change physical_streams_manager.hpp to
    // physical_streams_manager_interface.hpp
    TrainingRetCode retCode = m_pPhysicalStreamsManager->getLastWaitHandles(streamWaitHandles);
    if (retCode == TRAINING_RET_CODE_NO_CHANGE)
    {
        return synSuccess;
    }

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    synStatus status =
        _SYN_SINGLETON_INTERNAL->waitAndReleaseStreamHandles(streamWaitHandles, SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT);

    ETL_ADD_LOG_T_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER, logId, SYN_STREAM, "{}: Synchronized device", HLLOG_FUNC);

    return status;
}

void DeviceQueues::logStreamsSyncHistory()
{
    LOG_DEBUG(SYN_DEV_FAIL, "Logging History to file {}", DFA_API_FILE);

    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LogManager::LogType::DFA_API_INFO));
}

bool DeviceQueues::createQueue(QueueInterface*&                 rpQueueInterface,
                               const BasicQueueInfo&            rBasicQueueInfo,
                               synDeviceType                    deviceType,
                               PhysicalQueuesManager*           pPhysicalStreamsManager,
                               uint32_t                         physicalQueueOffset,
                               uint32_t                         amountOfEnginesInArbGroup,
                               bool                             isReduced,
                               DeviceRecipeDownloaderContainer* pDeviceRecipeDownloaderContainer)
{
    try
    {
        switch (rBasicQueueInfo.queueType)
        {
            case INTERNAL_STREAM_TYPE_DMA_UP:
            case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
            case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
            case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
            case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
            {
                rpQueueInterface = new QueueCopyQman(rBasicQueueInfo,
                                                     physicalQueueOffset,
                                                     deviceType,
                                                     pPhysicalStreamsManager,
                                                     m_rWorkCompletionManager,
                                                     m_rDevMemAlloc);
                break;
            }
            case INTERNAL_STREAM_TYPE_COMPUTE:
            {
                CHECK_POINTER(SYN_DEVICE, m_pDmaDownQueueInterface, "DMA-Down Synapse Stream-handle", false);

                std::unique_ptr<StreamMasterHelperInterface> streamMasterHelper =
                    std::make_unique<StreamMasterHelper>(deviceType, physicalQueueOffset);
                rpQueueInterface = new QueueComputeQman(rBasicQueueInfo,
                                                        physicalQueueOffset,
                                                        amountOfEnginesInArbGroup,
                                                        isReduced,
                                                        deviceType,
                                                        pPhysicalStreamsManager,
                                                        m_rWorkCompletionManager,
                                                        submitter,
                                                        m_rDevMemAlloc,
                                                        m_rDevRecipeAddress,
                                                        *pDeviceRecipeDownloaderContainer,
                                                        *getDmaDownStream(),
                                                        std::move(streamMasterHelper));

                const synStatus status = ((QueueComputeQman*)rpQueueInterface)->initAllocators();
                if (status != synSuccess)
                {
                    LOG_ERR(SYN_DEVICE, "{}: initAllocators failed {}", HLLOG_FUNC, status);
                    delete rpQueueInterface;
                    rpQueueInterface = nullptr;
                    return false;
                }
                break;
            }
            case INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK:
            {
                rpQueueInterface = new QueueCollectiveNetwork(rBasicQueueInfo,
                                                              physicalQueueOffset,
                                                              deviceType,
                                                              pPhysicalStreamsManager);
                break;
            }
            default:
            {
                LOG_ERR(SYN_DEVICE,
                        "{}: Cant create pStream (stream and handle) for {}",
                        HLLOG_FUNC,
                        rBasicQueueInfo.getDescription());
                return false;
            }
        }
    }
    catch (const SynapseException& err)
    {
        LOG_ERR(SYN_DEVICE,
                "{}: Cant create pStream (stream and handle) for {}",
                HLLOG_FUNC,
                rBasicQueueInfo.getDescription());

        return false;
    }

    return true;
}

std::string DeviceQueues::getInternalStreamName(internalStreamType queueType)
{
    switch (queueType)
    {
        case INTERNAL_STREAM_TYPE_DMA_UP:
            return "DMA_UP";
        case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
            return "DMA_UP_PROFILER";
        case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
            return "DEV_TO_DEV";
        case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
            return "DMA_DOWN_USER";
        case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
            return "DMA_DOWN_SYNAPSE";
        case INTERNAL_STREAM_TYPE_COMPUTE:
            return "COMPUTE";
        case INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK:
            return "COLLECTIVE_NETWORK";
        default:
            return "Unknown Stream Type " + std::to_string(queueType);
    }
}

common::StreamCreator* DeviceQueues::getStreamCreator(synDeviceType deviceType)
{
    common::StreamCreator* pStreamCreator = nullptr;

    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            pStreamCreator = gaudi::StreamCreator::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Invalid device type {}", deviceType);
        }
    }

    return pStreamCreator;
}

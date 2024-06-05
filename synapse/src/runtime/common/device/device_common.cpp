#include "device_common.hpp"

#include "defenders.h"
#include "device_utils.hpp"

#include "global_statistics.hpp"
#include "graph_compiler/graph_traits.h"

#include "profiler_api.hpp"

#include "synapse_api_types.h"
#include "synapse_common_types.h"

#include "utils.h"

#include "runtime/common/osal/osal.hpp"

#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"

#include "runtime/common/queues/basic_queue_info.hpp"
#include "runtime/common/queues/queue_interface.hpp"

#include "runtime/common/streams/stream_job.hpp"
#include "runtime/common/streams/stream.hpp"

#include "runtime/qman/common/command_submission_builder.hpp"

#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"

#include "internal/hccl_internal.h"

#include "habana_global_conf_runtime.h"

#include <bitset>

const uint8_t DeviceCommon::s_apiIdMask = 0b11111;

DeviceCommon::DeviceCommon(synDeviceType                devType,
                           DevMemoryAlloc*              devMemoryAlloc,
                           const DeviceConstructInfo&   deviceConstructInfo,
                           bool                         setStreamAffinityByJobType,
                           const AffinityCountersArray& rMaxAffinities)
: m_devType(devType),
  m_osalInfo(deviceConstructInfo.deviceInfo),
  m_pciAddr(deviceConstructInfo.pciAddr),
  m_acquireTime(std::chrono::system_clock::now()),
  m_hlIdx(deviceConstructInfo.hlIdx),
  m_maxAffinities(rMaxAffinities),
  m_devMemoryAlloc(devMemoryAlloc),
  m_dfa(deviceConstructInfo, this),
  m_eventFdController(m_osalInfo.fd, this),
  m_assertAsyncBufferHostAddr(nullptr),
  m_assertAsyncBufferMappedAddr(0),
  m_streamsContainer(setStreamAffinityByJobType),
  m_apiId(0)
{
    LOG_INFO_T(SYN_DEVICE,
               "DeviceCommon {:#x} created devType {} deviceInfo.deviceType {} dramAddr {:#x} dramSize "
               "{:#x} hlIdx {} pciAddr {}",
               TO64(this),
               devType,
               deviceConstructInfo.deviceInfo.deviceType,
               deviceConstructInfo.deviceInfo.dramBaseAddress,
               deviceConstructInfo.deviceInfo.dramSize,
               deviceConstructInfo.hlIdx,
               deviceConstructInfo.pciAddr);
}

DeviceCommon::~DeviceCommon()
{
    LOG_INFO_T(SYN_DEVICE, "DeviceCommon {:#x} destroyed", TO64(this));
}

synStatus DeviceCommon::mapBufferToDevice(uint64_t           size,
                                          void*              buffer,
                                          bool               isUserRequest,
                                          uint64_t           reqVAAddress,
                                          const std::string& mappingDesc)
{
    return m_devMemoryAlloc->mapBufferToDevice(size, buffer, isUserRequest, reqVAAddress, mappingDesc);
}

synStatus DeviceCommon::unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize)
{
    return m_devMemoryAlloc->unmapBufferFromDevice(buffer, isUserRequest, bufferSize);
}

synStatus DeviceCommon::allocateMemory(uint64_t           size,
                                       uint32_t           flags,
                                       void**             buffer,
                                       bool               isUserRequest,
                                       uint64_t           reqVAAddress,
                                       const std::string& mappingDesc,
                                       uint64_t*          deviceVA)
{
    return m_devMemoryAlloc->allocateMemory(size, flags, buffer, isUserRequest, reqVAAddress, mappingDesc, deviceVA);
}

synStatus DeviceCommon::deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest)
{
    return m_devMemoryAlloc->deallocateMemory(pBuffer, flags, isUserRequest);
}

synStatus DeviceCommon::getDeviceLimitationInfo(synDeviceLimitationInfo& rDeviceLimitationInfo)
{
    synStatus status = OSAL::getInstance().getDeviceLimitationInfo(rDeviceLimitationInfo);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "getDeviceLimitationInfo failed with status {}", status);
        if (status == synDeviceReset)
        {
            notifyHlthunkFailure(DfaErrorCode::getDeviceInfoFailed);
        }
        return status;
    }

    return status;
}

bool DeviceCommon::_checkRecipeCacheOverlap(uint64_t  srcAddress,
                                            uint64_t  dstAddress,
                                            uint64_t  recipeCacheBaseAddress,
                                            uint64_t  recipeCacheLastAddress,
                                            uint64_t  size,
                                            synDmaDir dir,
                                            bool      isMemset)
{
    uint64_t baseAddressToValidate = dstAddress;
    uint64_t lastAddressToValidate = baseAddressToValidate + size - 1;
    if ((dir == HOST_TO_DRAM) || (dir == DRAM_TO_DRAM))
    {
        if (isAddressRangeOverlaps(baseAddressToValidate,
                                   lastAddressToValidate,
                                   recipeCacheBaseAddress,
                                   recipeCacheLastAddress))
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Invalid dst-address (overrides Recipe-Cache area)"
                    " baseAddressToValidate 0x{:x} lastAddressToValidate 0x{:x}"
                    " recipeCacheBaseAddress 0x{:x} recipeCacheLastAddress 0x{:x}",
                    HLLOG_FUNC,
                    baseAddressToValidate,
                    lastAddressToValidate,
                    recipeCacheBaseAddress,
                    recipeCacheLastAddress);
            return false;
        }
    }

    baseAddressToValidate = srcAddress;
    lastAddressToValidate = baseAddressToValidate + size - 1;
    if ((!isMemset) && ((dir == DRAM_TO_HOST) || (dir == DRAM_TO_DRAM)))
    {
        if (isAddressRangeOverlaps(baseAddressToValidate,
                                   lastAddressToValidate,
                                   recipeCacheBaseAddress,
                                   recipeCacheLastAddress))
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Invalid src-address (overrides Recipe-Cache area)"
                    " baseAddressToValidate 0x{:x} lastAddressToValidate 0x{:x}"
                    " recipeCacheBaseAddress 0x{:x} recipeCacheLastAddress 0x{:x}",
                    HLLOG_FUNC,
                    baseAddressToValidate,
                    lastAddressToValidate,
                    recipeCacheBaseAddress,
                    recipeCacheLastAddress);
            return false;
        }
    }

    return true;
}

synStatus DeviceCommon::createAndAddHangCommandBuffer(synCommandBuffer*& pSynCB,
                                                      uint32_t           cbIndex,
                                                      const void*        pBuffer,
                                                      uint64_t           commandBufferSize,
                                                      uint32_t           queueId,
                                                      bool               isForceMmuMapped)
{
    synStatus status = CommandSubmissionBuilder::getInstance()->createAndAddHangCommandBuffer(pSynCB,
                                                                                              cbIndex,
                                                                                              pBuffer,
                                                                                              commandBufferSize,
                                                                                              queueId,
                                                                                              isForceMmuMapped);
    if (status == synFailedToInitializeCb)
    {
        status = CommandSubmissionBuilder::getInstance()
                     ->createAndAddHangCommandBuffer(pSynCB, cbIndex, pBuffer, commandBufferSize, queueId);

        LOG_TRACE(SYN_DEVICE, "{}: Retry due to memory-allocation failure. New status - {}", HLLOG_FUNC, status);
    }

    return status;
}

generic::CommandBufferPktGenerator* DeviceCommon::_getCommandBufferGeneratorInstance(synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            return gaudi::CommandBufferPktGenerator::getInstance();
        }
        default:
        {
        }
    }

    HB_ASSERT(false, "unsupported device type");
    return nullptr;
}

synStatus DeviceCommon::createSynCommandBuffer(synCommandBuffer* pSynCB, uint32_t queueId, uint64_t commandBufferSize)
{
    synStatus status =
        CommandSubmissionBuilder::getInstance()->createSynCommandBuffer(pSynCB, queueId, commandBufferSize);

    if (status == synFailedToInitializeCb)
    {
        status = CommandSubmissionBuilder::getInstance()->createSynCommandBuffer(pSynCB,
                                                                                 queueId,
                                                                                 commandBufferSize);

        LOG_TRACE(SYN_DEVICE, "{}: Retry due to memory-allocation failure. New status - {}", HLLOG_FUNC, status);
    }

    return status;
}

synStatus DeviceCommon::waitAndReleaseCS(uint64_t  handle,
                                         uint64_t  timeout,
                                         bool      returnUponTimeout /* = false */,
                                         bool      collectStats /* = false */,
                                         uint64_t* userEventTime)
{
    synStatus status = synSuccess;
    uint32_t  waitStatus;
    unsigned  ioctlNotCompletedCntr = 0;
    uint64_t  timeStamp             = 0;

    do
    {
        int ret;

        timeStamp = 0;
        STAT_GLBL_START(timeStart);
        ret        = hlthunk_wait_for_cs_with_timestamp(m_osalInfo.fd, handle, timeout, &waitStatus, &timeStamp);
        int errno_ = errno;  // save errno before calling any other function
        if (collectStats) STAT_GLBL_COLLECT_TIME(timeStart, globalStatPointsEnum::waitForCs);

        if (ret < 0)
        {
            if (errno_ == ENODEV)
            {
                ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
                LOG_CRITICAL(SYN_OSAL, "Device is going to hard reset, need to close FD");
                status = synDeviceReset;
                notifyHlthunkFailure(DfaErrorCode::waitForCsFailed);
            }
            else if (errno_ == ETIMEDOUT)
            {
                LOG_ERR(SYN_OSAL, "Command submission timedout");
                notifyHlthunkFailure(DfaErrorCode::waitForCsFailed);
                status = synTimeout;
            }
            else if (errno_ == EIO)
            {
                ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
                LOG_ERR(SYN_OSAL, "Command submission aborted due to device reset");
                notifyHlthunkFailure(DfaErrorCode::waitForCsFailed);
                status = synDeviceReset;
            }
            else if (errno_ == EINVAL)
            {
                LOG_ERR(SYN_OSAL, "Request is invalid indication (the handle might had long been resolved)");
                status = synInvalidArgument;
            }
            else
            {
                ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
                LOG_ERR(SYN_OSAL, "Command submission aborted due to {}", errno_);
                notifyHlthunkFailure(DfaErrorCode::waitForCsFailed);
                status = synFail;
            }

            LOG_ERR(SYN_DEVICE, "set ioctl return error (ioctlNotCompletedCntr={})", ioctlNotCompletedCntr);
            return status;
        }

        if (waitStatus != HL_WAIT_CS_STATUS_COMPLETED)
        {
            if (returnUponTimeout)
            {
                LOG_TRACE(SYN_OSAL, "{} Operation not completed", HLLOG_FUNC);
                return synBusy;
            }

            ioctlNotCompletedCntr++;
        }
    } while (waitStatus != HL_WAIT_CS_STATUS_COMPLETED);

    LOG_DEBUG(SYN_DEVICE, "waitAndReleaseCS done with hadnle 0x{:x}", handle);

    if (userEventTime != nullptr)
    {
        *userEventTime = timeStamp;
    }

    return synSuccess;
}

synStatus DeviceCommon::profile(hl_debug_args* args)
{
    const synStatus status = ((hlthunk_debug(m_osalInfo.fd, args) >= 0) ? synSuccess : synFail);

    if (status != synSuccess)
    {
        notifyHlthunkFailure(DfaErrorCode::hlthunkDebugFailed);
    }

    return status;
}

synStatus DeviceCommon::getClockSyncInfo(hlthunk_time_sync_info* infoOut)
{
    return getClockSyncPerDieInfo(0, infoOut);
}

synStatus DeviceCommon::getClockSyncPerDieInfo(uint32_t dieIndex, hlthunk_time_sync_info* infoOut)
{
    const synStatus status = ((hlthunk_get_time_sync_per_die_info(m_osalInfo.fd, dieIndex, infoOut) >= 0) ? synSuccess : synFail);

    if (status != synSuccess)
    {
        notifyHlthunkFailure(DfaErrorCode::getTimeSyncInfoFailed);
    }

    return status;
}

synStatus DeviceCommon::getPllFrequency(uint32_t index, struct hlthunk_pll_frequency_info* freqOut)
{
    const synStatus status = ((hlthunk_get_pll_frequency(m_osalInfo.fd, index, freqOut) == 0) ? synSuccess : synFail);

    return status;
}

synStatus DeviceCommon::waitAndReleaseStreamHandles(const InternalWaitHandlesVector& streamWaitHandles,
                                                    uint64_t                         timeout,
                                                    bool                             returnUponTimeout /* = false */)
{
    for (auto currentHandle : streamWaitHandles)
    {
        uint64_t waitHandle = currentHandle.handle;
        if (waitHandle == std::numeric_limits<uint64_t>::max())
        {
            continue;
        }

        // We could "reset" that wait-handle (in the relevant DB), but we will go with KISS
        synStatus status = waitAndReleaseCS(waitHandle, timeout, returnUponTimeout, false, nullptr);
        if (status != synSuccess)
        {
            return status;
        }
    }

    return synSuccess;
}

synStatus DeviceCommon::createStream(QueueType queueType, unsigned int flags, synStreamHandle& rStreamHandle)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);
    return createStreamGeneric(flags, rStreamHandle);
}

synStatus DeviceCommon::destroyStream(synStreamHandle streamHandle)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);
    return destroyStreamGeneric(streamHandle);
}

synStatus DeviceCommon::getDeviceAttribute(const synDeviceAttribute* deviceAttr,
                                           const unsigned            querySize,
                                           uint64_t*                 retVal)
{
    return extractDeviceAttributes(deviceAttr, querySize, retVal, m_osalInfo, nullptr, this);
}

synStatus DeviceCommon::getPCIBusId(char* pPciBusId, const int len)
{
    synStatus status = OSAL::getInstance().getPCIBusId(pPciBusId, len);
    if (status == synDeviceReset)
    {
        notifyHlthunkFailure(DfaErrorCode::getPciBusIdFailed);
    }

    return status;
}

/*
 ***************************************************************************************************
 *   @brief notifyHlthunkFailure() This function is called from relevant hlthunk calls that fail.
 *   @param  DfaErrorCode - indicates where it failed
 *   @return None
 *
 ***************************************************************************************************
 */
void DeviceCommon::notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo)
{
    m_dfa.notifyHlthunkFailure(errCode, dfaExtraInfo);
}

/*
 ***************************************************************************************************
 *   @brief notifyDeviceFailure() This function is called whenever the code detects critical error.
 *          currently, when backgrnoud-work detects an error (==g2/3 timeout)
 *
 *   @param errCode - indicated the critical error type
 *   @return none
 *
 ***************************************************************************************************
 */
void DeviceCommon::notifyDeviceFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    m_dfa.notifyDeviceFailure(errCode, dfaExtraInfo);
}

/*
 ***************************************************************************************************
 *   @brief notifyEventFd() This function is called when the event fd is triggered
 *
 *   @param errCode - indicated the critical error type
 *   @return none
 *
 ***************************************************************************************************
 */
void DeviceCommon::notifyEventFd(uint64_t events)
{
    m_dfa.notifyEventFd(events);
}

/*
 ***************************************************************************************************
 *   @brief getDfaStatus() returns device DfaStatus (by calling the dfa class)
 *
 *   @param errCode - indicated the critical error type
 *   @return DfaStatus
 *
 ***************************************************************************************************
 */
DfaStatus DeviceCommon::getDfaStatus()
{
    return m_dfa.getStatus();
}

synStatus DeviceCommon::startEventFdThread()
{
    uint32_t    tpcBufferSize = 4096;
    std::string name {"assertAsyncBuffer"};
    synStatus   status = m_devMemoryAlloc->allocateMemory(tpcBufferSize,
                                                        synMemFlags::synMemHost,
                                                        &m_assertAsyncBufferHostAddr,
                                                        false,
                                                        0,
                                                        name,
                                                        &m_assertAsyncBufferMappedAddr);
    if (status != synSuccess)
    {
        return status;
    }
    memset(m_assertAsyncBufferHostAddr, 0, tpcBufferSize);

    m_dfa.setAssertAsyncBufferAddress(m_assertAsyncBufferHostAddr);

    status = m_eventFdController.start();
    if (status != synSuccess)
    {
        return status;
    }
    return synSuccess;
}

synStatus DeviceCommon::stopEventFdThread()
{
    synStatus status = synSuccess;
    if (m_assertAsyncBufferHostAddr != nullptr)
    {
        status = m_devMemoryAlloc->deallocateMemory(m_assertAsyncBufferHostAddr, synMemFlags::synMemHost, false);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "deallocateMemory failed with synStatus {}", status);
        }
    }

    status = m_eventFdController.stop();
    if (status != synSuccess)
    {
        return status;
    }
    return status;
}

synStatus DeviceCommon::streamQuery(synStreamHandle streamHandle)
{
    CHECK_POINTER(SYN_DEVICE, streamHandle, "streamHandle", synInvalidArgument);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    return streamSptr->query();
}

synStatus DeviceCommon::streamWaitEvent(synStreamHandle streamHandle, synEventHandle eventHandle, unsigned int flags)
{
    auto eventSptr = loadAndValidateEvent(eventHandle, __FUNCTION__);
    if (eventSptr == nullptr)
    {
        return synInvalidEventHandle;
    }
    return streamWaitEventInterface(streamHandle, eventSptr.get(), flags);
}

synStatus DeviceCommon::streamWaitEventInterface(synStreamHandle streamHandle,
                                                 EventInterface* pEventInterface,
                                                 unsigned int    flags)
{
    CHECK_POINTER(SYN_DEVICE, pEventInterface, "eventHandle", synInvalidArgument);
    return streamGenericWaitEvent(streamHandle, *pEventInterface, flags);
}

synStatus DeviceCommon::synchronizeAllStreams()
{
    if (GCFG_INIT_HCCL_ON_ACQUIRE.value() == true)
    {
        hcclResult_t hcclStatus = hcclSuccess;
        LOG_TRACE(SYN_STREAM, "{}: Calling hcclSynchronizeAllStreams", HLLOG_FUNC);
        hcclStatus = hcclSynchronizeAllStreams();

        if (hcclStatus != hcclSuccess && hcclStatus != hcclInvalidArgument)
        {
            LOG_ERR(SYN_STREAM, "{}: Calling hcclSynchronizeAllStreams failed. ", HLLOG_FUNC);
            return synFail;
        }
    }

    // Note: we are not taking the lock here since StreamsContainer::synchronizeAllStreams() is protected.
    // Nevertheless, I'd take it here and remove it from there for the sake of alignment.
    return synchronizeAllStreamsGeneric();
}

synStatus DeviceCommon::eventRecord(synEventHandle eventHandle, synStreamHandle streamHandle)
{
    auto eventSptr = loadAndValidateEvent(eventHandle, __FUNCTION__);
    if (eventSptr == nullptr)
    {
        return synInvalidEventHandle;
    }
    return eventRecord(eventSptr.get(), streamHandle);
}

synStatus DeviceCommon::synchronizeEvent(synEventHandle eventHandle)
{
    auto eventSptr = loadAndValidateEvent(eventHandle, __FUNCTION__);
    if (eventSptr == nullptr)
    {
        return synInvalidEventHandle;
    }
    return synchronizeEvent(eventSptr.get());
}

synStatus DeviceCommon::eventQuery(synEventHandle eventHandle)
{
    auto eventSptr = loadAndValidateEvent(eventHandle, __FUNCTION__);
    if (eventSptr == nullptr)
    {
        return synInvalidEventHandle;
    }
    return eventQuery(eventSptr.get());
}

synStatus DeviceCommon::synchronizeStream(synStreamHandle streamHandle)
{
    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    synStatus status = streamSptr->synchronize();
    if (status == synDeviceReset)
    {
        notifyHlthunkFailure(DfaErrorCode::streamSyncFailed);
    }
    return status;
}

synStatus DeviceCommon::launchWithExternalEvents(const synStreamHandle         streamHandle,
                                                 const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                 const uint32_t                enqueueTensorsAmount,
                                                 uint64_t                      workspaceAddress,
                                                 const synRecipeHandle         pRecipeHandle,
                                                 synEventHandle*               eventHandleList,
                                                 uint32_t                      numberOfEvents,
                                                 uint32_t                      flags)
{
    if (m_devType != pRecipeHandle->deviceAgnosticRecipeHandle.m_deviceType)
    {
        LOG_ERR(SYN_API,
                "The device type {} does not match the recipe device type {}",
                m_devType,
                pRecipeHandle->deviceAgnosticRecipeHandle.m_deviceType);
        return synInvalidArgument;
    }

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    if (numberOfEvents > 0 && eventHandleList == nullptr)
    {
        LOG_ERR(SYN_API,
                "Number of events is {} but eventHandleList is nullptr",
                numberOfEvents);
         return synInvalidEventHandle;
    }

    // keep EventSptr's in order to ensure they are not destroyed during launch execution
    EventWithMappedTensorDB events;
    events.reserve(numberOfEvents);
    for (unsigned i = 0; i < numberOfEvents; ++i)
    {
        auto eventSptr = getEventSptr(eventHandleList[i]);
        if (!eventSptr)
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Failed to launch, event 0x{:x} not found (index:{})",
                    HLLOG_FUNC,
                    (uint64_t)eventHandleList[i],
                    i);
            return synInvalidEventHandle;
        }
        EventWithMappedTensorSptr eventWithTensorSptr =
            SlotMapItemDynamicCast<EventWithMappedTensor>(std::move(eventSptr));
        if (!eventWithTensorSptr)
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Failed to launch, event 0x{:x} (index:{}) cannot be casted to EventWithTensorMapping",
                    HLLOG_FUNC,
                    (uint64_t)eventHandleList[i],
                    i);
            return synInvalidEventHandle;
        }
        if (eventWithTensorSptr->getSequenceOffset() == SEQUENCE_OFFSET_NOT_USED)
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Failed to launch, event 0x{:x} (index:{}) sequence id is not set",
                    HLLOG_FUNC,
                    (uint64_t)eventHandleList[i],
                    i);
            return synInvalidEventHandle;
        }

        events.push_back(std::move(eventWithTensorSptr));
    }

    return launch(streamSptr.get(),
                  enqueueTensorsInfo,
                  enqueueTensorsAmount,
                  QueueComputeUtils::getAlignedWorkspaceAddress(workspaceAddress),
                  pRecipeHandle,
                  events,
                  flags);
}

synStatus DeviceCommon::memcopy(const synStreamHandle  streamHandle,
                                internalMemcopyParams& memcpyParams,
                                const internalDmaDir   direction,
                                bool                   isUserRequest)
{
    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    std::unique_ptr<StreamJob> job = std::make_unique<MemcopyJob>(memcpyParams, direction, isUserRequest, generateApiId());
    return m_streamsContainer.addJob(streamSptr.get(), job);
}

synStatus DeviceCommon::memSet(const synStreamHandle streamHandle,
                               uint64_t              pDeviceMem,
                               const uint32_t        value,
                               const size_t          numOfElements,
                               const size_t          elementSize)
{
    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    std::unique_ptr<StreamJob> job = std::make_unique<MemsetJob>(pDeviceMem, value, numOfElements, elementSize, generateApiId());
    return m_streamsContainer.addJob(streamSptr.get(), job);
}

synStatus DeviceCommon::createStreamGeneric(uint32_t flags, synStreamHandle& rStreamHandle)
{
    synEventHandle     eventHandle = nullptr;
    const unsigned int eventFlags  = 0;

    synStatus eventStatus = createEvent(&eventHandle, eventFlags);
    if (eventStatus != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "failed to create generic stream event {}", eventStatus);
        return eventStatus;
    }

    auto eventSptr = loadAndValidateEvent(eventHandle, __FUNCTION__);
    if (eventSptr == nullptr)
    {
        LOG_ERR(SYN_DEVICE, "failed to load stream event");
        return synInvalidEventHandle;
    }

    EventInterface& rEventInterface = *eventSptr.get();

    synStreamHandle streamHandle = nullptr;

    synStatus streamStatus = m_streamsContainer.createStream(eventHandle, rEventInterface, &streamHandle);
    if (streamStatus != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "failed to create generic stream status {}", streamStatus);
        eventStatus = destroyEvent(eventHandle);
        if (eventStatus != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "failed to destroy generic stream event {}", eventStatus);
        }
        return streamStatus;
    }

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        LOG_ERR(SYN_DEVICE, "failed to load stream");
        return synInvalidArgument;
    }

    rStreamHandle = streamHandle;

    return synSuccess;
}

synStatus DeviceCommon::destroyStreamGeneric(synStreamHandle streamHandle)
{
    synEventHandle eventHandle = nullptr;

    {
        auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
        if (streamSptr == nullptr)
        {
            return synInvalidArgument;
        }
        eventHandle = streamSptr->getEventHandle();
    }

    synStatus status = m_streamsContainer.destroyStream(streamHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "failed to destroy generic stream status {}", status);
        return status;
    }

    status = destroyEvent(eventHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "failed to create stream event {}", status);
        return status;
    }

    return synSuccess;
}

synStatus DeviceCommon::synchronizeAllStreamsGeneric()
{
    return m_streamsContainer.synchronizeAllStreams();
}

synStatus DeviceCommon::setStreamAffinity(synStreamHandle streamHandle, uint64_t streamAffinityMask)
{
    VERIFY_IS_NULL_POINTER(SYN_DEVICE, streamHandle, "Stream handle");

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    uint64_t  deviceAffinityMask;
    synStatus status = getDeviceAffinityMaskRange(deviceAffinityMask);
    if (status != synSuccess) return status;

    if (!(0 < streamAffinityMask && streamAffinityMask <= deviceAffinityMask))
    {
        LOG_ERR(SYN_STREAM,
                "setStreamAffinity called with non exist affinity value for streamHandle {:#x} "
                "deviceAffinityMask {:#x} streamAffinityMask {:#x}",
                TO64(streamHandle),
                deviceAffinityMask,
                streamAffinityMask);
        return synFail;
    }

    uint64_t existingAffinityMask;
    m_streamsContainer.getAffinities(existingAffinityMask);
    if ((streamAffinityMask & existingAffinityMask) != streamAffinityMask)
    {
        LOG_INFO(SYN_STREAM,
                 "Uninitialized affinity was set by the user. Adding all uninitialized affinities streamHandle {:#x} "
                 "deviceAffinityMask {:#x} existingAffinityMask {:#x} streamAffinityMask {:#x}",
                 TO64(streamHandle),
                 deviceAffinityMask,
                 existingAffinityMask,
                 streamAffinityMask);

        const std::bitset<sizeof(uint64_t) * 8> existingAffinityBitMap(existingAffinityMask);
        const unsigned                          existingCounter = existingAffinityBitMap.count();
        AffinityCountersArray                   affinityArr     = m_maxAffinities;
        for (auto& typeCounter : affinityArr)
        {
            if (typeCounter > existingCounter)
            {
                typeCounter -= existingCounter;
            }
            else
            {
                typeCounter = 0;
            }
        }

        status = addStreamAffinities(affinityArr, true);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: addStreamAffinities failed with status {} ", HLLOG_FUNC, status);
            return status;
        }
    }

    return m_streamsContainer.setStreamAffinity(streamSptr.get(), streamAffinityMask);
}

synStatus DeviceCommon::getStreamAffinity(synStreamHandle streamHandle, uint64_t* streamAffinityMask) const
{
    VERIFY_IS_NULL_POINTER(SYN_DEVICE, streamHandle, "Stream handle");

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    return m_streamsContainer.getStreamAffinity(streamSptr.get(), *streamAffinityMask);
}

synStatus DeviceCommon::getDeviceAffinityMaskRange(uint64_t& rDeviceAffinityMaskRange) const
{
    const unsigned maxAffinity = *std::max_element(m_maxAffinities.begin(), m_maxAffinities.end());

    rDeviceAffinityMaskRange = 0;

    for (unsigned iter = 0; iter < maxAffinity; iter++)
    {
        rDeviceAffinityMaskRange |= 1 << iter;
    }

    return synSuccess;
}

synStatus DeviceCommon::getDeviceNextStreamAffinity(uint64_t& rDeviceAffinityMaskRange)
{
    return m_streamsContainer.getDeviceNextStreamAffinity(rDeviceAffinityMaskRange);
}

synStatus DeviceCommon::syncHCLStreamHandle(synStreamHandle streamHandle)
{
    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    std::unique_ptr<StreamJob> job = std::make_unique<EventNetwork>();
    return m_streamsContainer.addJob(streamSptr.get(), job);
}

synStatus DeviceCommon::flushWaitsOnCollectiveStream(synStreamHandle streamHandle)
{
    CHECK_POINTER(SYN_DEVICE, streamHandle, "streamHandle", synInvalidArgument);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    return streamSptr->flushWaitsOnCollectiveQueue();
}

uint32_t DeviceCommon::getNetworkStreamPhysicalQueueOffset(synStreamHandle streamHandle)
{
    HB_ASSERT_PTR(streamHandle);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    return streamSptr->getPhysicalQueueOffset(QUEUE_TYPE_NETWORK_COLLECTIVE);
}

hcl::hclStreamHandle DeviceCommon::getNetworkStreamHclStreamHandle(synStreamHandle streamHandle)
{
    HB_ASSERT_PTR(streamHandle);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return nullptr;
    }

    return streamSptr->getHclStreamHandle(QUEUE_TYPE_NETWORK_COLLECTIVE);
}

EventSptr DeviceCommon::loadAndValidateEvent(synEventHandle eventHandle, const char* functionName)
{
    auto eventSptr = getEventSptr(eventHandle);
    if (eventSptr == nullptr)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed on event handle {:x} verification, event handle is incorrect, probably event was destroyed",
                functionName,
                (SMHandle)eventHandle);
    }
    return eventSptr;
}

SlotMapItemSptr<Stream> DeviceCommon::loadAndValidateStream(synStreamHandle streamHandle, const char* functionName) const
{
    auto streamSptr = m_streamsContainer.getStreamSptr(streamHandle);
    if (streamSptr == nullptr)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed on stream handle {:x} verification, handle is incorrect, probably stream was destroyed",
                functionName,
                (SMHandle)streamHandle);
    }
    return streamSptr;
}

synStatus DeviceCommon::getInternalStreamTypes(QueueType queueType, internalStreamType* internalType)
{
    auto mapItr = s_streamToInternalStreamMap.find(queueType);
    if (mapItr == s_streamToInternalStreamMap.end())
    {
        return synInvalidArgument;
    }

    *internalType = mapItr->second;
    return synSuccess;
}

/**
 * used for logging via SCAL/HCL
 * because SCAL doesn't know SYN_DEV_FAIL log.
 * In HCL it gives us control over where it is logged
 * @param logLevel wanted log level
 * @param msg log message
 */
void DeviceCommon::dfaLogFunc(int logLevel, const char* msg)
{
    SYN_LOG(synapse::LogManager::LogType::SYN_DEV_FAIL, logLevel, "{}", msg);
}

void DeviceCommon::dfaLogFuncErr(int logLevel, const char* msg)
{
    if (logLevel >= SPDLOG_LEVEL_ERROR)
    {
        SYN_LOG(synapse::LogManager::LogType::SYN_DEV_FAIL, logLevel, "{}", msg);
    }
}

const std::unordered_map<unsigned, internalStreamType> DeviceCommon::s_streamToInternalStreamMap = {
    {QUEUE_TYPE_COPY_DEVICE_TO_HOST, INTERNAL_STREAM_TYPE_DMA_UP},
    {QUEUE_TYPE_COPY_HOST_TO_DEVICE, INTERNAL_STREAM_TYPE_DMA_DOWN_USER},
    {QUEUE_TYPE_COPY_DEVICE_TO_DEVICE, INTERNAL_STREAM_TYPE_DEV_TO_DEV},
    {QUEUE_TYPE_COMPUTE, INTERNAL_STREAM_TYPE_COMPUTE},
    {QUEUE_TYPE_NETWORK_COLLECTIVE, INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK},
    {QUEUE_TYPE_RESERVED_1, INTERNAL_STREAM_TYPE_DMA_UP_PROFILER}};

synStatus DeviceCommon::addStreamAffinities(const AffinityCountersArray& rAffinityArr, bool isReduced)
{
    QueueInterfacesArrayVector queueHandles;
    synStatus                status = createStreamQueues(rAffinityArr, isReduced, queueHandles);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Device createStreamQueues failed with status {}", status);
        synStatus statusDestroy = destroyStreamQueues(rAffinityArr, queueHandles);
        if (statusDestroy != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "Device destroyStreamQueues failed with statusDestroy {}", statusDestroy);
        }
        return status;
    }

    status = m_streamsContainer.addStreamAffinities(queueHandles);
    if (status != synSuccess) return status;

    return synSuccess;
}

synStatus DeviceCommon::removeAllStreamAffinities()
{
    QueueInterfacesArrayVector queueHandles;
    synStatus                  status = m_streamsContainer.removeAffinities(queueHandles);
    if (status != synSuccess) return status;

    status = destroyStreamQueues(m_maxAffinities, queueHandles);
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: destroyStreamQueues failed with status {}", HLLOG_FUNC, status);
        return status;
    }

    return synSuccess;
}

synStatus DeviceCommon::createStreamQueues(AffinityCountersArray       affinityArr,
                                           bool                        isReduced,
                                           QueueInterfacesArrayVector& rQueueInterfaces)
{
    LOG_INFO_T(SYN_DEVICE, "{}", HLLOG_FUNC);

    const unsigned maxAffinity = *std::max_element(affinityArr.begin(), affinityArr.end());

    for (unsigned affinityIndex = 0; affinityIndex < maxAffinity; affinityIndex++)
    {
        rQueueInterfaces.push_back({nullptr, nullptr, nullptr, nullptr, nullptr});
        for (unsigned typeIndex = 0; typeIndex < QUEUE_TYPE_MAX_USER_TYPES; typeIndex++)
        {
            if (affinityArr[typeIndex] > 0)
            {
                synStatus status =
                    createStreamQueue((QueueType)typeIndex, 0, isReduced, rQueueInterfaces[affinityIndex][typeIndex]);
                if (status != synSuccess)
                {
                    return status;
                }

                affinityArr[typeIndex]--;
            }
            else if (affinityIndex != 0)
            {
                rQueueInterfaces[affinityIndex][typeIndex] = rQueueInterfaces[affinityIndex - 1][typeIndex];
            }
            else
            {
                rQueueInterfaces[affinityIndex][typeIndex] = nullptr;
            }
        }
    }

    return synSuccess;
}

synStatus DeviceCommon::destroyStreamQueues(AffinityCountersArray       affinityArr,
                                            QueueInterfacesArrayVector& rQueueInterfaces)
{
    LOG_INFO_T(SYN_DEVICE, "{}", HLLOG_FUNC);

    for (unsigned affinityIndex = 0; affinityIndex < rQueueInterfaces.size(); affinityIndex++)
    {
        for (unsigned typeIndex = 0; typeIndex < QUEUE_TYPE_MAX_USER_TYPES; typeIndex++)
        {
            if ((affinityArr[typeIndex] > 0) && (rQueueInterfaces[affinityIndex][typeIndex] != nullptr))
            {
                synStatus status = destroyStreamQueue(rQueueInterfaces[affinityIndex][typeIndex]);
                if (status != synSuccess)
                {
                    return status;
                }

                affinityArr[typeIndex]--;
            }
        }
    }

    return synSuccess;
}

void DeviceCommon::testingOnlySetBgFreq(std::chrono::milliseconds period)
{
   m_eventFdController.testingOnlySetBgFreq(period);
}

synStatus DeviceCommon::getDeviceInfo(synDeviceInfoV2& rDeviceInfo) const
{
    rDeviceInfo.sramBaseAddress = m_osalInfo.sramBaseAddress;
    rDeviceInfo.dramBaseAddress = m_osalInfo.dramBaseAddress;
    rDeviceInfo.sramSize        = m_osalInfo.sramSize;
    rDeviceInfo.dramSize        = m_osalInfo.dramSize;
    rDeviceInfo.tpcEnabledMask  = m_osalInfo.tpcEnabledMask;
    rDeviceInfo.dramEnabled     = m_osalInfo.dramEnabled;
    rDeviceInfo.deviceId        = m_osalInfo.deviceId;
    rDeviceInfo.fd              = m_osalInfo.fd;
    rDeviceInfo.deviceType      = m_osalInfo.deviceType;
    rDeviceInfo.deviceIndex     = m_hlIdx;

    return synSuccess;
}
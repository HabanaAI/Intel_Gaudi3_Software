#pragma once

#include "define_synapse_common.hpp"
#include "runtime/common/common_types.hpp"
#include "dfa_base.hpp"
#include "runtime/common/queues/event_interface.hpp"
#include "recipe.h"
#include "synapse_common_types.h"
#include "host_to_virtual_address_mapper.hpp"
#include "infra/containers/slot_map.hpp"
#include "hcl_public_streams.h"

#include <cstdint>

using EventSptr = SlotMapItemSptr<EventInterface>;

struct hl_debug_args;
struct hlthunk_time_sync_info;
class QueueInterface;

class DeviceInterface
{
public:
    virtual ~DeviceInterface() = default;

    virtual synStatus acquire(const uint16_t numSyncObj) = 0;

    virtual synStatus release(std::atomic<bool>& rDeviceBeingReleased) = 0;

    virtual synStatus mapBufferToDevice(uint64_t           size,
                                        void*              buffer,
                                        bool               isUserRequest,
                                        uint64_t           reqVAAddress,
                                        const std::string& mappingDesc) = 0;

    virtual synStatus unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize) = 0;

    // need to refactor allocateMemory()
    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) = 0;

    virtual synStatus deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest) = 0;

    virtual synDeviceType getDevType() const = 0;

    virtual uint32_t getDevHlIdx() const = 0;

    virtual synStatus getDramMemInfo(uint64_t& free, uint64_t& total) const = 0;

    virtual eMappingStatus getDeviceVirtualAddress(bool         isUserRequest,
                                                   void*        hostAddress,
                                                   uint64_t     bufferSize,
                                                   uint64_t*    pDeviceVA,
                                                   bool*        pIsExactKeyFound = nullptr) = 0;

    virtual synStatus getDeviceInfo(synDeviceInfo& rDeviceInfo) const = 0;
    virtual synStatus getDeviceInfo(synDeviceInfoV2& rDeviceInfo) const = 0;

    virtual synStatus getDeviceLimitationInfo(synDeviceLimitationInfo& rDeviceLimitationInfo) = 0;

    virtual synStatus
    createSynCommandBuffer(synCommandBuffer* pSynCB, uint32_t queueId, uint64_t commandBufferSize) = 0;

    virtual synStatus waitAndReleaseCS(uint64_t  handle,
                                       uint64_t  timeout,
                                       bool      returnUponTimeout = false,
                                       bool      collectStats      = false,
                                       uint64_t* usrEventTime      = nullptr) = 0;

    virtual synStatus profile(hl_debug_args* args) = 0;

    virtual synStatus getClockSyncInfo(hlthunk_time_sync_info* infoOut) = 0;

    virtual synStatus getPllFrequency(uint32_t index, struct hlthunk_pll_frequency_info* freqOut) = 0;

    virtual synStatus waitAndReleaseStreamHandles(const InternalWaitHandlesVector& streamWaitHandles,
                                                  uint64_t                         timeout,
                                                  bool                             returnUponTimeout) = 0;

    virtual synStatus createStream(QueueType queueType, unsigned int flags, synStreamHandle& rStreamHandle) = 0;

    virtual synStatus createStreamGeneric(uint32_t flags, synStreamHandle& rStreamHandle) = 0;

    virtual synStatus destroyStream(synStreamHandle streamHandle) = 0;

    virtual synStatus streamQuery(synStreamHandle streamHandle) = 0;
    virtual synStatus streamWaitEvent(synStreamHandle streamHandle, synEventHandle eventHandle, unsigned int flags) = 0;

    virtual synStatus synchronizeAllStreams() = 0;

    virtual synStatus synchronizeEvent(synEventHandle eventHandle)           = 0;
    virtual synStatus synchronizeEvent(const EventInterface* eventInterface) = 0;

    virtual synStatus eventQuery(synEventHandle eventHandle)                = 0;
    virtual synStatus eventQuery(const EventInterface* eventInterface)      = 0;
    virtual synStatus synchronizeStream(const synStreamHandle streamHandle) = 0;

    virtual synStatus createEvent(synEventHandle* pEventHandle, const unsigned int flags) = 0;

    virtual synStatus destroyEvent(synEventHandle eventHandle) = 0;

    virtual EventSptr getEventSptr(synEventHandle eventHandle) = 0;

    virtual synStatus eventRecord(synEventHandle eventHandle, synStreamHandle streamHandle)  = 0;

    virtual synStatus getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const = 0;

    virtual synStatus launchWithExternalEvents(const synStreamHandle         streamHandle,
                                               const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                               const uint32_t                enqueueTensorsAmount,
                                               uint64_t                      workspaceAddress,
                                               const synRecipeHandle         pRecipeHandle,
                                               synEventHandle*               eventHandleList,
                                               uint32_t                      numberOfEvents,
                                               uint32_t                      flags) = 0;

    virtual synStatus memcopy(const synStreamHandle  streamHandle,
                              internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest) = 0;

    virtual synStatus memSet(const synStreamHandle streamHandle,
                             uint64_t              pDeviceMem,
                             const uint32_t        value,
                             const size_t          numOfElements,
                             const size_t          elementSize) = 0;

    virtual void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle) = 0;

    virtual synStatus getDeviceAttribute(const synDeviceAttribute* deviceAttr,
                                         const unsigned            querySize,
                                         uint64_t*                 retVal) = 0;

    virtual synStatus getPCIBusId(char* pPciBusId, const int len) = 0;

    virtual void notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo = {}) = 0;

    virtual void notifyDeviceFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo = {}) = 0;

    virtual void notifyEventFd(uint64_t events) = 0;

    virtual void addDfaObserver(DfaObserver* pObserver) = 0;

    virtual DfaStatus getDfaStatus() = 0;

    virtual void bgWork() = 0;
    virtual void debugCheckWorkStatus() = 0;

    virtual synStatus getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                      synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const = 0;

    virtual bool isAssertAsyncNoticed() const = 0;

    virtual synStatus setStreamAffinity(synStreamHandle streamHandle, uint64_t streamAffinityMask) = 0;

    virtual synStatus getStreamAffinity(synStreamHandle streamHandle, uint64_t* streamAffinityMask) const = 0;

    virtual synStatus getDeviceAffinityMaskRange(uint64_t& rDeviceAffinityMaskRange) const = 0;

    virtual synStatus getDeviceNextStreamAffinity(uint64_t& nextDeviceAffinity) = 0;

    virtual synStatus
    kernelsPrintf(const InternalRecipeHandle& rInternalRecipeHandle, uint64_t wsAddr, void* hostBuff) = 0;

    virtual synStatus syncHCLStreamHandle(synStreamHandle streamHandle) = 0;

    virtual synStatus flushWaitsOnCollectiveStream(synStreamHandle streamHandle) = 0;

    virtual uint32_t getNetworkStreamPhysicalQueueOffset(synStreamHandle streamHandle) = 0;

    virtual hcl::hclStreamHandle getNetworkStreamHclStreamHandle(synStreamHandle streamHandle) = 0;

    virtual bool isDirectModeUserDownloadStream() const = 0;

    virtual uint8_t generateApiId() = 0;

    virtual synStatus getClockSyncPerDieInfo(uint32_t dieIndex, hlthunk_time_sync_info* infoOut) = 0;
};

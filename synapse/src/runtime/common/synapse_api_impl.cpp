#include <chrono>
#include <thread>
#include <sstream>
#include <version.h>
#include "defs.h"
#include "synapse_api.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "syn_singleton.hpp"
#include "profiler_api.hpp"
#include "api.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

#define VERIFY_IS_NULL_POINTER(pointer, name)                                                                          \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_WARN(SYN_API, "{}: got null pointer for {} ", HLLOG_FUNC, name);                                           \
        return synInvalidArgument;                                                                                     \
    }

synStatus SYN_API_CALL synStreamCreateGeneric(synStreamHandle*  pStreamHandle,
                                              const synDeviceId deviceId,
                                              const uint32_t    flags)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}  flags 0x{:x}", deviceId, flags);

    status = _SYN_SINGLETON_->createStream(pStreamHandle, deviceId, flags);

    VERIFY_IS_NULL_POINTER(pStreamHandle, "pStreamHandle");
    LOG_SYN_API("returned streamHandle 0x{:x}", TO64(*pStreamHandle));

    API_EXIT_STATUS_TIMED(status, synStreamCreateP)
}

synStatus SYN_API_CALL synStreamDestroy(const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("StreamHandle 0x{:x}", TO64(streamHandle));
    status = _SYN_SINGLETON_->destroyStream(streamHandle);
    API_EXIT_STATUS_TIMED(status, synStreamDestroyP);
}

synStatus SYN_API_CALL synStreamWaitEvent(const synStreamHandle streamHandle,
                                          synEventHandle        eventHandle,
                                          const uint32_t        flags)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("StreamHandle 0x{:x} eventHandle 0x{:x} flags 0x{:x}", TO64(streamHandle), TO64(eventHandle), flags);
    STAT_GLBL_START(StreamWaitEventUser);
    status = _SYN_SINGLETON_->streamWaitEvent(streamHandle, eventHandle, flags);
    STAT_GLBL_COLLECT_TIME(StreamWaitEventUser, globalStatPointsEnum::StreamWaitEventUser);
    API_EXIT_STATUS_TIMED(status, synStreamWaitEventP);
}

synStatus SYN_API_CALL synStreamSynchronize(const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("StreamHandle 0x{:x}", TO64(streamHandle));
    status = _SYN_SINGLETON_->synchronizeStream(streamHandle);
    API_EXIT_STATUS_TIMED(status, synStreamSynchronizeP);
}

synStatus SYN_API_CALL synStreamQuery(const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("StreamHandle 0x{:x}", TO64(streamHandle));
    status = _SYN_SINGLETON_->streamQuery(streamHandle);
    API_EXIT_STATUS_TIMED(status, synStreamQueryP);
}

synStatus SYN_API_CALL synEventCreate(synEventHandle* pEventHandle, const synDeviceId deviceId, const uint32_t flags)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {} flags 0x{:x}", deviceId, flags);
    status = _SYN_SINGLETON_->createEvent(pEventHandle, deviceId, flags);

    VERIFY_IS_NULL_POINTER(pEventHandle, "pEventHandle");
    LOG_SYN_API("returned eventHandle 0x{:x}", TO64(*pEventHandle));
    API_EXIT_STATUS_TIMED(status, synEventCreateP);
}

synStatus SYN_API_CALL synEventDestroy(synEventHandle eventHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("eventHandle 0x{:x}", TO64(eventHandle));
    status = _SYN_SINGLETON_->destroyEvent(eventHandle);
    API_EXIT_STATUS_TIMED(status, synEventDestroyP);
}

synStatus SYN_API_CALL synEventRecord(synEventHandle eventHandle, const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("eventHandle 0x{:x} streamHandle 0x{:x}", TO64(eventHandle), TO64(streamHandle));
    STAT_GLBL_START(synEventRecordUser);
    status = _SYN_SINGLETON_->eventRecord(eventHandle, streamHandle);
    STAT_GLBL_COLLECT_TIME(synEventRecordUser, globalStatPointsEnum::synEventRecordUser);
    API_EXIT_STATUS_TIMED(status, synEventRecordP);
}

synStatus SYN_API_CALL synEventQuery(const synEventHandle eventHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("eventHandle 0x{:x}", TO64(eventHandle));
    status = _SYN_SINGLETON_->eventQuery(eventHandle);
    API_EXIT_STATUS_TIMED(status, synEventQueryP);
}

synStatus SYN_API_CALL synEventSynchronize(const synEventHandle eventHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("eventHandle 0x{:x}", TO64(eventHandle));
    status = _SYN_SINGLETON_->synchronizeEvent(eventHandle);
    API_EXIT_STATUS_TIMED(status, synEventSynchronizeP);
}

synStatus SYN_API_CALL synEventElapsedTime(uint64_t*            pNanoSeconds,
                                           const synEventHandle eventHandleStart,
                                           const synEventHandle eventHandleEnd)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("eventHandleStart 0x{:x} eventHandleEnd 0x{:x}", TO64(eventHandleStart), TO64(eventHandleEnd));
    status = _SYN_SINGLETON_->eventElapsedTime(pNanoSeconds, eventHandleStart, eventHandleEnd);
    API_EXIT_STATUS_TIMED(status, synEventElapsedTimeP);
}

synStatus SYN_API_CALL synEventMapTensor(synEventHandle*            eventHandles,
                                         size_t                     numOfEvents,
                                         const synLaunchTensorInfo* launchTensorsInfo,
                                         const synRecipeHandle      recipeHandle)
{
    auto status = synEventMapTensorExt(eventHandles, numOfEvents, (synLaunchTensorInfoExt* )launchTensorsInfo, recipeHandle);
    return status;
}

synStatus SYN_API_CALL synEventMapTensorExt(synEventHandle*                eventHandles,
                                            size_t                         numOfEvents,
                                            const synLaunchTensorInfoExt*  launchTensorsInfoExt,
                                            const synRecipeHandle          recipeHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("eventHandles 0x{:x} tensorName {}", TO64(eventHandles), launchTensorsInfoExt->tensorName);
    status = _SYN_SINGLETON_->eventMapTensorBaseExt(eventHandles, numOfEvents, launchTensorsInfoExt, recipeHandle);
    API_EXIT_STATUS_TIMED(status, synEventMapTensorP);
}

synStatus SYN_API_CALL synTensorExtExtractExecutionOrder(const synRecipeHandle recipeHandle,
                                                         uint32_t              numOfEvents,
                                                         uint64_t*             tensorIds)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle 0x{:x} numOfEvents {}", TO64(recipeHandle), numOfEvents);
    status = _SYN_SINGLETON_->externalTensorsExtractExecutionOrder(recipeHandle, numOfEvents, tensorIds);
    API_EXIT_STATUS_TIMED(status, synTensorExtExtractExecutionOrderP);
}

synStatus SYN_API_CALL synRecipeDestroy(synRecipeHandle hRecipe)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("hRecipe 0x{:x}", TO64(hRecipe));
    status = _SYN_SINGLETON_->recipeDestroy(hRecipe);
    API_EXIT_STATUS_TIMED(status, synRecipeDestroyP);
}

synStatus SYN_API_CALL synSectionCreate(synSectionHandle*    pSectionHandle,
                                        uint64_t             sectionDescriptor,
                                        const synGraphHandle graph)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("sectionDescriptor 0x{:x}, graph 0x{:x}", sectionDescriptor, TO64(graph));
    status = _SYN_SINGLETON_->sectionCreate(pSectionHandle, sectionDescriptor, graph);

    VERIFY_IS_NULL_POINTER(pSectionHandle, "pSectionHandle");
    LOG_SYN_API("returned sectionHandle 0x{:x}", TO64(*pSectionHandle));
    API_EXIT_STATUS_TIMED(status, synSectionCreateP);
}

synStatus SYN_API_CALL synSectionSetGroup(synSectionHandle sectionHandle, uint64_t group)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}, group {}", TO64(sectionHandle), group);
    status = _SYN_SINGLETON_->sectionGroupSet(sectionHandle, group);
    API_EXIT_STATUS_TIMED(status, synSectionSetGroupP);
}

synStatus SYN_API_CALL synSectionGetGroup(synSectionHandle sectionHandle, uint64_t* group)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}", TO64(sectionHandle));
    status = _SYN_SINGLETON_->sectionGroupGet(sectionHandle, group);
    API_EXIT_STATUS_TIMED(status, synSectionGetGroupP);
}

synStatus SYN_API_CALL synSectionSetPersistent(synSectionHandle sectionHandle, bool sectionIsPersistent)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}, persistent {}", TO64(sectionHandle), sectionIsPersistent);
    status = _SYN_SINGLETON_->sectionPersistentSet(sectionHandle, sectionIsPersistent);
    API_EXIT_STATUS_TIMED(status, synSectionSetPersistentP);
}

synStatus SYN_API_CALL synSectionGetPersistent(synSectionHandle sectionHandle, bool* sectionIsPersistent)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}", TO64(sectionHandle));
    status = _SYN_SINGLETON_->sectionPersistentGet(sectionHandle, sectionIsPersistent);
    API_EXIT_STATUS_TIMED(status, synSectionGetPersistentP);
}

synStatus SYN_API_CALL synSectionSetConst(synSectionHandle sectionHandle, bool sectionIsConst)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}, is const {}", TO64(sectionHandle), sectionIsConst);
    status = _SYN_SINGLETON_->sectionConstSet(sectionHandle, sectionIsConst);
    API_EXIT_STATUS_TIMED(status, synSectionSetConstP);
}

synStatus SYN_API_CALL synSectionGetConst(synSectionHandle sectionHandle, bool* sectionIsConst)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}", TO64(sectionHandle));
    status = _SYN_SINGLETON_->sectionConstGet(sectionHandle, sectionIsConst);
    API_EXIT_STATUS_TIMED(status, synSectionGetConstP);
}

synStatus SYN_API_CALL synRecipeSectionGetProp(const synRecipeHandle  pRecipeHandle,
                                               const synSectionId     sectionId,
                                               const synSectionProp   prop,
                                               uint64_t*              propertyPtr)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle 0x{:x} section {}", TO64(pRecipeHandle), sectionId);
    status = _SYN_SINGLETON_->sectionGetProp(pRecipeHandle, sectionId, prop, propertyPtr);
    API_EXIT_STATUS_TIMED(status, synRecipeSectionGetPropP);
}

synStatus SYN_API_CALL synSectionSetRMW(synSectionHandle sectionHandle, bool sectionIsRMW)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}, RMW {}", TO64(sectionHandle), sectionIsRMW);
    status = _SYN_SINGLETON_->sectionRMWSet(sectionHandle, sectionIsRMW);
    API_EXIT_STATUS_TIMED(status, synSectionSetRMWP);
}

synStatus SYN_API_CALL synSectionGetRMW(synSectionHandle sectionHandle, bool* sectionIsRMW)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("section 0x{:x}", TO64(sectionHandle));
    status = _SYN_SINGLETON_->sectionRMWGet(sectionHandle, sectionIsRMW);
    API_EXIT_STATUS_TIMED(status, synSectionGetRMWP);
}

synStatus SYN_API_CALL synRecipeSectionHostBuffersClear(synRecipeHandle     recipeHandle,
                                                        const synSectionId* sectionIds,
                                                        size_t              numOfSections)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle {:#x} sectionIds {:#x} numOfSections {}",
                TO64(recipeHandle), TO64(sectionIds), numOfSections);
    status = _SYN_SINGLETON_->sectionsClearHostBuffer(recipeHandle, sectionIds, numOfSections);
    API_EXIT_STATUS_TIMED(status, synRecipeSectionHostBuffersClear);
}

synStatus SYN_API_CALL synSectionDestroy(synSectionHandle hSection)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("Section 0x{:x}", TO64(hSection));
    status = _SYN_SINGLETON_->sectionDestroy(hSection);
    API_EXIT_STATUS_TIMED(status, synSectionDestroyP);
}

synStatus SYN_API_CALL synRecipeSerialize(const synRecipeHandle recipeHandle, const char* recipeFileName)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle 0x{:x}", TO64(recipeHandle));
    status = _SYN_SINGLETON_->recipeSerialize(recipeHandle, recipeFileName);
    API_EXIT_STATUS_TIMED(status, synRecipeSerializeP);
}

synStatus SYN_API_CALL synRecipeDeSerialize(synRecipeHandle* pRecipeHandle, const char* recipeFileName)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle 0x{:x}", TO64(pRecipeHandle));
    status = _SYN_SINGLETON_->recipeDeSerialize(pRecipeHandle, recipeFileName);
    API_EXIT_STATUS_TIMED(status, synRecipeDeSerializeP);
}

synStatus SYN_API_CALL synRecipeGetAttribute(uint64_t*                 retVal,
                                             const synRecipeAttribute* recipeAttr,
                                             const unsigned            querySize,
                                             const synRecipeHandle     recipeHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle 0x{:x}", TO64(recipeHandle));
    status = _SYN_SINGLETON_->recipeGetAttribute(retVal, recipeAttr, querySize, recipeHandle);
    API_EXIT_STATUS_TIMED(status, synRecipeGetAttributeP);
}

synStatus SYN_API_CALL synLaunch(const synStreamHandle      streamHandle,
                                 const synLaunchTensorInfo* launchTensorsInfo,
                                 const uint32_t             numberTensors,
                                 uint64_t                   pWorkspace,
                                 const synRecipeHandle      pRecipeHandle,
                                 uint32_t                   flags)
{
   auto status = synLaunchExt(streamHandle, (synLaunchTensorInfoExt*)launchTensorsInfo, numberTensors, pWorkspace, pRecipeHandle, flags);
    return status;
}

synStatus SYN_API_CALL synLaunchExt(const synStreamHandle         streamHandle,
                                    const synLaunchTensorInfoExt* launchTensorsInfoExt,
                                    const uint32_t                numberTensors,
                                    uint64_t                      pWorkspace,
                                    const synRecipeHandle         pRecipeHandle,
                                    uint32_t                      flags)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("streamHandle 0x{:x} numberTensors {} pWorkspace 0x{:x} pRecipeHandle 0x{:x} flags 0x{:x}",
                TO64(streamHandle),
                numberTensors,
                pWorkspace,
                TO64(pRecipeHandle),
                flags);

    STAT_GLBL_START(timeStart);
    status = _SYN_SINGLETON_->enqueue(streamHandle,
                                      launchTensorsInfoExt,
                                      numberTensors,
                                      nullptr,  // TODO (next commit) - remove to match the SynSin API
                                      0,        // TODO (next commit) - remove to match the SynSin API
                                      pWorkspace,
                                      pRecipeHandle,
                                      flags);

    STAT_GLBL_COLLECT_TIME(timeStart, globalStatPointsEnum::LaunchUser);
    if (status != synSuccess)
    {
        LOG_SYN_API("done status {}", status);
    }
    API_EXIT_STATUS_TIMED(status, synLaunchP);
}

synStatus SYN_API_CALL synLaunchWithExternalEvents(const synStreamHandle      streamHandle,
                                                   const synLaunchTensorInfo* launchTensorsInfo,
                                                   const uint32_t             numberTensors,
                                                   uint64_t                   pWorkspace,
                                                   const synRecipeHandle      pRecipeHandle,
                                                   synEventHandle*            pEventHandleList,
                                                   const uint32_t             numberOfEvents,
                                                   uint32_t                   flags)
{
   auto status = synLaunchWithExternalEventsExt(streamHandle,
                                                 (synLaunchTensorInfoExt*)launchTensorsInfo,
                                                 numberTensors,
                                                 pWorkspace,
                                                 pRecipeHandle,
                                                 pEventHandleList,
                                                 numberOfEvents,
                                                 flags);
    return status;
}

synStatus SYN_API_CALL synLaunchWithExternalEventsExt(const synStreamHandle         streamHandle,
                                                      const synLaunchTensorInfoExt* launchTensorsInfoExt,
                                                      const uint32_t                numberTensors,
                                                      uint64_t                      pWorkspace,
                                                      const synRecipeHandle         pRecipeHandle,
                                                      synEventHandle*               pEventHandleList,
                                                      const uint32_t                numberOfEvents,
                                                      uint32_t                      flags)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("streamHandle 0x{:x} numberTensors {} pWorkspace 0x{:x} pRecipeHandle 0x{:x}, "
                "pEventHandleList 0x{:x}, numberOfEvents {}, flags 0x{:x}",
                TO64(streamHandle),
                numberTensors,
                pWorkspace,
                TO64(pRecipeHandle),
                TO64(pEventHandleList),
                numberOfEvents,
                flags);

    STAT_GLBL_START(timeStart);

    status = _SYN_SINGLETON_->enqueueWithExternalEventsExt(streamHandle,
                                                           launchTensorsInfoExt,
                                                           numberTensors,
                                                           pWorkspace,
                                                           pRecipeHandle,
                                                           pEventHandleList,
                                                           numberOfEvents,
                                                           flags);

    STAT_GLBL_COLLECT_TIME(timeStart, globalStatPointsEnum::LaunchUser);
    if (status != synSuccess)
    {
        LOG_SYN_API("done status {}", status);
    }
    API_EXIT_STATUS_TIMED(status, synLaunchWithExternalEventsP);
}

synStatus SYN_API_CALL synWorkspaceGetSize(uint64_t* pWorkspaceSize, const synRecipeHandle recipeHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("recipeHandle 0x{:x}", TO64(recipeHandle));
    status = _SYN_SINGLETON_->getTopologyWorkspaceSize(pWorkspaceSize, recipeHandle);
    API_EXIT_STATUS_TIMED(status, synWorkspaceGetSizeP);
}

synStatus SYN_API_CALL synMemCopyAsync(const synStreamHandle streamHandle,
                                       const uint64_t        src,
                                       const uint64_t        size,
                                       const uint64_t        dst,
                                       const synDmaDir       direction)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("streamHandle 0x{:x} src 0x{:x} size 0x{:x} dst 0x{:x} direction {}",
                TO64(streamHandle),
                src,
                size,
                dst,
                direction);

    STAT_GLBL_START(memcpyAsyncUser);
    status = _SYN_SINGLETON_->memcpyAsync(streamHandle, &src, &size, &dst, direction);
    STAT_GLBL_COLLECT_TIME(memcpyAsyncUser, globalStatPointsEnum::memcpyAsyncUser);

    API_EXIT_STATUS_TIMED(status, synMemCopyAsyncP);
}

synStatus SYN_API_CALL synMemCopyAsyncMultiple(const synStreamHandle streamHandle,
                                               const uint64_t*       src,
                                               const uint64_t*       size,
                                               const uint64_t*       dst,
                                               const synDmaDir       direction,
                                               const uint64_t        numCopies)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("streamHandle 0x{:x} direction {} numCopies {}", TO64(streamHandle), direction, numCopies);

    STAT_GLBL_START(memcpyAsyncUser);
    status = _SYN_SINGLETON_->memcpyAsync(streamHandle, src, size, dst, direction, numCopies);
    STAT_GLBL_COLLECT_TIME(memcpyAsyncUser, globalStatPointsEnum::memcpyAsyncUser);

    API_EXIT_STATUS_TIMED(status, synMemCopyAsyncMultipleP);
}

synStatus SYN_API_CALL synDeviceGetCount(uint32_t* pCount)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->deviceGetCount(pCount);
    API_EXIT_STATUS_TIMED(status, synDeviceGetCountP);
}

synStatus SYN_API_CALL synDeviceGetCountByDeviceType(uint32_t* pCount, const synDeviceType deviceType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceType {}", deviceType);
    status = _SYN_SINGLETON_->deviceGetCountByDeviceType(pCount, deviceType);
    API_EXIT_STATUS_TIMED(status, synDeviceGetCountByDeviceTypeP);
}

synStatus SYN_API_CALL synDeviceCount(uint32_t count[synDeviceTypeSize])
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->deviceCount(count);
    API_EXIT_STATUS_TIMED(status, synDeviceCountP);
}

synStatus SYN_API_CALL synDeviceAcquireByDeviceType(synDeviceId* pDeviceId, const synDeviceType deviceType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceType {}", deviceType);
    VERIFY_IS_NULL_POINTER(pDeviceId, "Device-ID");
    status = _SYN_SINGLETON_->acquireDevice(pDeviceId, "", deviceType, INVALID_MODULE_ID);
    API_EXIT_STATUS_TIMED(status, synDeviceAcquireByDeviceTypeP);
}

synStatus SYN_API_CALL synDeviceGetModuleIDs(uint32_t *pDeviceModuleIds, uint32_t*  size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->deviceGetModuleIds(pDeviceModuleIds, size);
    API_EXIT_STATUS_TIMED(status, synDeviceGetModuleIDsP);
}

synStatus SYN_API_CALL synDeviceAcquireByModuleId(synDeviceId* pDeviceId, const synModuleId moduleId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("moduleId {}", moduleId);
    status = _SYN_SINGLETON_->acquireDevice(pDeviceId, "", synDeviceTypeInvalid, moduleId);
    API_EXIT_STATUS_TIMED(status, synDeviceAcquireByModuleIdP);
}

synStatus SYN_API_CALL synDeviceAcquire(synDeviceId* pDeviceId, const char* pciBus)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->acquireDevice(pDeviceId, pciBus, synDeviceTypeInvalid, INVALID_MODULE_ID);
    API_EXIT_STATUS_TIMED(status, synDeviceAcquireP);
}

synStatus SYN_API_CALL synDeviceSynchronize(const synDeviceId deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);
    status = _SYN_SINGLETON_->synchronizeAllStreams(deviceId);
    API_EXIT_STATUS_TIMED(status, synDeviceSynchronizeP);
}

extern const char* SYNAPSE_SHA1_VERSION;

synStatus SYN_API_CALL synDriverGetVersion(char* pDriverVersion, const int len)
{
    LOG_SYN_API();
    VERIFY_IS_NULL_POINTER(pDriverVersion, "pDriverVersion");

    std::string version =
        fmt::format("{}.{}.{}.{}", HL_DRIVER_MAJOR, HL_DRIVER_MINOR, HL_DRIVER_PATCHLEVEL, SYNAPSE_SHA1_VERSION);

    std::strncpy(pDriverVersion, version.c_str(), len);
    if (version.length() >= len)
    {
        pDriverVersion[len - 1] = '\0';
        LOG_WARN(SYN_OSAL, "{}: Given pDriverVersion length is shorter than real device name", HLLOG_FUNC);
    }
    return synSuccess;
}

synStatus SYN_API_CALL synDeviceGetName(char* pName, const int len, const synDeviceId deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);
    status = _SYN_SINGLETON_->getDeviceName(pName, len, deviceId);
    API_EXIT_STATUS_TIMED(status, synDeviceGetNameP);
}

synStatus SYN_API_CALL synDeviceGetPCIBusId(char* pPciBusId, const int len, const synDeviceId deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);
    status = _SYN_SINGLETON_->deviceGetPCIBusId(pPciBusId, len, deviceId);
    API_EXIT_STATUS_TIMED(status, synDeviceGetPCIBusIdP);
}

synStatus SYN_API_CALL synTensorCreate(synTensor*                 pTensor,
                                       const synTensorDescriptor* descriptor,
                                       const synSectionHandle     pSectionHandle,
                                       const uint64_t             sectionOffset)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->createTensor(descriptor, pTensor, pSectionHandle, sectionOffset);
    API_EXIT_STATUS_TIMED(status, synTensorCreateP);
}

synStatus SYN_API_CALL synConstTensorCreate(synTensor* pTensor, const synTensorDescriptor* descriptor)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->createTensor(descriptor, pTensor, nullptr, 0, true);
    API_EXIT_STATUS_TIMED(status, synConstTensorCreateP);
}

synStatus SYN_API_CALL synTensorRetrieveInfosByName(const synRecipeHandle pRecipeHandle,
                                                    const uint32_t        numOfTensors,
                                                    TensorMetadataInfo*   tensorsMetadataInfo)
{
   auto status = synTensorRetrieveInfosByNameExt(pRecipeHandle, numOfTensors, (TensorMetadataInfoExt*)tensorsMetadataInfo);
   LOG_SYN_API("pRecipeHandle {} numOfTensors {}", TO64(pRecipeHandle), numOfTensors);

   return status;
}

synStatus SYN_API_CALL synTensorRetrieveInfosByNameExt(const synRecipeHandle    pRecipeHandle,
                                                       const uint32_t           numOfTensors,
                                                       TensorMetadataInfoExt*   tensorsMetadataInfoExt)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("pRecipeHandle {} numOfTensors {}", TO64(pRecipeHandle), numOfTensors);
    status =
        _SYN_SINGLETON_->tensorRetrieveMetadatasInfosByNameExt(pRecipeHandle, numOfTensors, tensorsMetadataInfoExt);
    API_EXIT_STATUS_TIMED(status, synTensorRetrieveInfosByNameExtP);
}

synStatus SYN_API_CALL synTensorRetrievePersistentAmount(const synRecipeHandle pRecipeHandle, uint32_t* numOfTensors)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->tensorRetrievePersistentAmount(pRecipeHandle, *numOfTensors);
    API_EXIT_STATUS_TIMED(status, synTensorRetrievePersistentAmountP);
}

synStatus SYN_API_CALL synTensorRetrieveNames(const synRecipeHandle pRecipeHandle,
                                              char                  tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                                              const uint32_t        numOfTensors)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->tensorRetrieveNames(pRecipeHandle, tensorsName, numOfTensors);
    API_EXIT_STATUS_TIMED(status, synTensorRetrieveNamesP);
}

synStatus SYN_API_CALL synTensorRetrieveLaunchAmount(const synRecipeHandle pRecipeHandle, uint32_t* numOfTensors)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->tensorRetrieveLaunchAmount(pRecipeHandle, *numOfTensors);
    API_EXIT_STATUS_TIMED(status, synTensorRetrieveLaunchAmountP);
}

synStatus SYN_API_CALL synTensorRetrieveLaunchIds(const synRecipeHandle pRecipeHandle,
                                                  uint64_t*             tensorsIds,
                                                  const uint32_t        numOfTensors)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->tensorRetrieveLaunchIds(pRecipeHandle, tensorsIds, numOfTensors);
    API_EXIT_STATUS_TIMED(status, synTensorRetrieveLaunchIdsP);
}

synStatus SYN_API_CALL synTensorRetrieveLaunchInfoById(const synRecipeHandle         pRecipeHandle,
                                                       const uint32_t                numOfTensors,
                                                       synRetrievedLaunchTensorInfo* tensorsLaunchInfo)
{
   auto status = synTensorRetrieveLaunchInfoByIdExt(pRecipeHandle, numOfTensors, (synRetrievedLaunchTensorInfoExt*)tensorsLaunchInfo);
   return status;
}

synStatus SYN_API_CALL synTensorRetrieveLaunchInfoByIdExt(const synRecipeHandle            pRecipeHandle,
                                                          const uint32_t                   numOfTensors,
                                                          synRetrievedLaunchTensorInfoExt* tensorsLaunchInfoExt)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("pRecipeHandle {} numOfTensors {}", TO64(pRecipeHandle), numOfTensors);
    status = _SYN_SINGLETON_->tensorRetrieveLaunchInfoByIdExt(pRecipeHandle, numOfTensors, tensorsLaunchInfoExt);
    API_EXIT_STATUS_TIMED(status, synTensorRetrieveLaunchInfoByIdExtP);
}

synStatus SYN_API_CALL synTensorRetrieveIds(const synRecipeHandle pRecipeHandle,
                                            const char**          tensorNames,
                                            uint64_t*             tensorIds,
                                            const uint32_t        numOfTensors)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->tensorRetrieveIds(pRecipeHandle, tensorNames, tensorIds, numOfTensors);
    API_EXIT_STATUS_TIMED(status, synTensorRetrieveIdsP);
}

synStatus SYN_API_CALL synTensorDestroy(const synTensor tensor)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("tensor 0x{:x}", TO64(tensor));
    status = _SYN_SINGLETON_->destroyTensor(tensor);
    API_EXIT_STATUS_TIMED(status, synTensorDestroyP);
}

synStatus SYN_API_CALL synNodeCreate(const synGraphHandle graphHandle,
                                     const synTensor*     pInputsTensorList,
                                     const synTensor*     pOutputsTensorList,
                                     const uint32_t       numberInputs,
                                     const uint32_t       numberOutputs,
                                     const void*          pUserParams,
                                     const unsigned       paramsSize,
                                     const char*          pGuid,
                                     const char*          pName,
                                     const char**         inputLayouts,
                                     const char**         outputLayouts)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} numberInputs {} numberOutputs {}", TO64(graphHandle), numberInputs, numberOutputs);

    std::string nodeName(pName ? pName : "");
    status = _SYN_SINGLETON_->createGenericNode(graphHandle,
                                                pInputsTensorList,
                                                pOutputsTensorList,
                                                numberInputs,
                                                numberOutputs,
                                                pUserParams,
                                                paramsSize,
                                                pGuid,
                                                inputLayouts,
                                                outputLayouts,
                                                nodeName);
    API_EXIT_STATUS_TIMED(status, synNodeCreateP);
}

synStatus SYN_API_CALL synNodeCreateWithId(const synGraphHandle graphHandle,
                                           const synTensor*     pInputsTensorList,
                                           const synTensor*     pOutputsTensorList,
                                           const uint32_t       numberInputs,
                                           const uint32_t       numberOutputs,
                                           const void*          pUserParams,
                                           const unsigned       paramsSize,
                                           const char*          pGuid,
                                           const char*          pName,
                                           synNodeId*           nodeUniqueId,
                                           const char**         inputLayouts,
                                           const char**         outputLayouts)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} numberInputs {} numberOutputs {}", TO64(graphHandle), numberInputs, numberOutputs);

    std::string nodeName(pName ? pName : "");
    status = _SYN_SINGLETON_->createGenericNodeWithId(graphHandle,
                                                      pInputsTensorList,
                                                      pOutputsTensorList,
                                                      numberInputs,
                                                      numberOutputs,
                                                      pUserParams,
                                                      paramsSize,
                                                      pGuid,
                                                      inputLayouts,
                                                      outputLayouts,
                                                      nodeName,
                                                      nodeUniqueId);
    API_EXIT_STATUS_TIMED(status, synNodeCreateWithIdP);
}

synStatus SYN_API_CALL synNodeTypeSetPrecision(const synGraphHandle graphHandle,
                                               const char*          guid,
                                               synDataType          precision)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} Set Node Type {} Precision {}", TO64(graphHandle), guid, precision);

    status = _SYN_SINGLETON_->nodeTypeSetUserPrecision(graphHandle, guid, precision);
    API_EXIT_STATUS_TIMED(status, synNodeTypeSetPrecisionP);
}

synStatus SYN_API_CALL synNodeDependencySet(const synGraphHandle graphHandle,
                                            const synNodeId*     pBlockingNodesIdList,
                                            const synNodeId*     pBlockedNodesIdList,
                                            const uint32_t       numberblocking,
                                            const uint32_t       numberblocked)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} numberblocking {} numberblocked {}",
                TO64(graphHandle),
                numberblocking,
                numberblocked);
    status = _SYN_SINGLETON_->createControlDependency(graphHandle,
                                                      pBlockingNodesIdList,
                                                      pBlockedNodesIdList,
                                                      numberblocking,
                                                      numberblocked);
    API_EXIT_STATUS_TIMED(status, synNodeDependencySetP);
}

synStatus SYN_API_CALL synNodeSetDeterministic(const synGraphHandle graphHandle,
                                               const synNodeId      nodeId,
                                               const bool           useDeterministic)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} synNodeId {} useDeterministic {}", TO64(graphHandle), nodeId, useDeterministic);
    status = _SYN_SINGLETON_->nodeSetDeterministic(graphHandle, nodeId, useDeterministic);
    API_EXIT_STATUS_TIMED(status, synNodeSetDeterministicP);
}

synStatus SYN_API_CALL synNodeGetDeterministic(const synGraphHandle graphHandle,
                                               const synNodeId      nodeId,
                                               bool*                pUseDeterministic)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} synNodeId {}", TO64(graphHandle), nodeId);
    status = _SYN_SINGLETON_->nodeGetDeterministic(graphHandle, nodeId, pUseDeterministic);
    API_EXIT_STATUS_TIMED(status, synNodeGetDeterministicP);
}

synStatus SYN_API_CALL synNodeSetRoundingMode(const synGraphHandle  graphHandle,
                                              const synNodeId       nodeId,
                                              const synRoundingMode roundingMode)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} synNodeId {} SetRoundingMode {}", TO64(graphHandle), nodeId, roundingMode);
    status = _SYN_SINGLETON_->nodeSetRoundingMode(graphHandle, nodeId, roundingMode);
    API_EXIT_STATUS_TIMED(status, synNodeSetRoundingModeP);
}

synStatus SYN_API_CALL synNodeGetRoundingMode(const synGraphHandle graphHandle,
                                              const synNodeId      nodeId,
                                              synRoundingMode*     pRoundingMode)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x} synNodeId {}", TO64(graphHandle), nodeId);
    status = _SYN_SINGLETON_->nodeGetRoundingMode(graphHandle, nodeId, pRoundingMode);
    API_EXIT_STATUS_TIMED(status, synNodeGetRoundingModeP);
}

synStatus SYN_API_CALL synGraphCompile(synRecipeHandle*     pRecipeHandle,
                                       const synGraphHandle graphHandle,
                                       const char*          pRecipeName,
                                       const char*          pBuildLog)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->compileGraph(pRecipeHandle, graphHandle, pRecipeName, pBuildLog);
    API_EXIT_STATUS_TIMED(status, synGraphCompileP);
}

synStatus SYN_API_CALL synGraphCreate(synGraphHandle* pGraphHandle, const synDeviceType deviceType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceType {}", deviceType);
    status = _SYN_SINGLETON_->createGraph(pGraphHandle, deviceType, CompilationMode::Graph);
    API_EXIT_STATUS_TIMED(status, synGraphCreateP);
}

/*DEPRECATED*/
synStatus SYN_API_CALL synGraphSetAttribute(synGraphHandle           graphHandle,
                                            const synGraphAttribute* attributes,
                                            const uint64_t*          values,
                                            uint32_t                 size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->graphSetAttribute(graphHandle, attributes, values, size);
    API_EXIT_STATUS_TIMED(status, synGraphSetAttributeP);
}

synStatus SYN_API_CALL synGraphSetAttributes(synGraphHandle              graphHandle,
                                             const synGraphAttribute*    attributes,
                                             const synGraphAttributeVal* values,
                                             uint32_t                    size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->graphSetAttributes(graphHandle, attributes, values, size);
    API_EXIT_STATUS_TIMED(status, synGraphSetAttributesP);
}

/*DEPRECATED*/
synStatus SYN_API_CALL synGraphGetAttribute(synGraphHandle           graphHandle,
                                            const synGraphAttribute* attributes,
                                            uint64_t*                values,
                                            uint32_t                 size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->graphGetAttribute(graphHandle, attributes, values, size);
    API_EXIT_STATUS_TIMED(status, synGraphGetAttributeP);
}

synStatus SYN_API_CALL synGraphGetAttributes(synGraphHandle           graphHandle,
                                             const synGraphAttribute* attributes,
                                             synGraphAttributeVal*    values,
                                             uint32_t                 size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->graphGetAttributes(graphHandle, attributes, values, size);
    API_EXIT_STATUS_TIMED(status, synGraphGetAttributesP);
}

synStatus SYN_API_CALL synGraphCreateEager(synGraphHandle* pGraphHandle, const synDeviceType deviceType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceType {}", deviceType);
    status = _SYN_SINGLETON_->createGraph(pGraphHandle, deviceType, CompilationMode::Eager);
    API_EXIT_STATUS_TIMED(status, synGraphCreateEagerP);
}

synStatus SYN_API_CALL synGraphDestroy(const synGraphHandle graphHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->destroyGraph(graphHandle);
    API_EXIT_STATUS_TIMED(status, synGraphDestroyP);
}

synStatus SYN_API_CALL synGraphGetDeviceType(const synGraphHandle graphHandle, synDeviceType* deviceType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("graphHandle 0x{:x}", TO64(graphHandle));
    status = _SYN_SINGLETON_->getGraphDeviceType(graphHandle, deviceType);
    API_EXIT_STATUS_TIMED(status, synGraphGetDeviceTypeP);
}

synStatus SYN_API_CALL synMemsetD32Async(uint64_t              pDeviceMem,
                                         const uint32_t        value,
                                         const size_t          numOfElements,
                                         const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("pDeviceMem 0x{:x} value 0x{:x} numOfElements 0x{:x} streamHandle 0x{:x}",
                pDeviceMem,
                value,
                numOfElements,
                TO64(streamHandle));
    status = _SYN_SINGLETON_->memsetAsync(streamHandle, pDeviceMem, value, numOfElements, sizeof(uint32_t));
    API_EXIT_STATUS_TIMED(status, synMemsetD32AsyncP);
}

synStatus SYN_API_CALL synMemsetD8Async(uint64_t              pDeviceMem,
                                        const unsigned char   value,
                                        const size_t          numOfElements,
                                        const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("pDeviceMem 0x{:x} value 0x{:x} numOfElements 0x{:x} streamHandle 0x{:x}",
                pDeviceMem,
                value,
                numOfElements,
                TO64(streamHandle));
    status = _SYN_SINGLETON_->memsetAsync(streamHandle, pDeviceMem, value, numOfElements, sizeof(uint8_t));
    API_EXIT_STATUS_TIMED(status, synMemsetD8AsyncP);
}

synStatus SYN_API_CALL synMemsetD16Async(uint64_t              pDeviceMem,
                                         const uint16_t        value,
                                         const size_t          numOfElements,
                                         const synStreamHandle streamHandle)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("pDeviceMem 0x{:x} value 0x{:x} numOfElements 0x{:x} streamHandle 0x{:x}",
                pDeviceMem,
                value,
                numOfElements,
                TO64(streamHandle));
    status = _SYN_SINGLETON_->memsetAsync(streamHandle, pDeviceMem, value, numOfElements, sizeof(uint16_t));
    API_EXIT_STATUS_TIMED(status, synMemsetD16AsyncP);
}

synStatus SYN_API_CALL synHostMalloc(const synDeviceId deviceId,
                                     const uint64_t    size,
                                     const uint32_t    flags,
                                     void**            buffer)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("deviceId {} size 0x{:x} flags 0x{:x}", deviceId, size, flags);
    if (flags != 0)
    {
        LOG_ERR(SYN_API, "{} called with flags={} ,must be equal to zero", HLLOG_FUNC, flags);
        status = synFail;
    }
    else
    {
        status = _SYN_SINGLETON_->allocateDeviceMemory(deviceId, size, synMemFlags::synMemHost, buffer, 0);
    }
    API_EXIT_STATUS_TIMED(status, synHostMallocP);
}

synStatus SYN_API_CALL synHostFree(const synDeviceId deviceId, const void* buffer, const uint32_t flags)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("deviceId {} buffer 0x{:x} flags 0x{:x}", deviceId, TO64(buffer), flags);
    if (flags != 0)
    {
        LOG_ERR(SYN_API, "{} called with flags={} ,must be equal to zero", HLLOG_FUNC, flags);
        status = synFail;
    }
    else
    {
        status = _SYN_SINGLETON_->deallocateDeviceMemory(deviceId, const_cast<void*>(buffer), synMemFlags::synMemHost);
    }
    API_EXIT_STATUS_TIMED(status, synHostFreeP);
}

synStatus SYN_API_CALL synHostMap(const synDeviceId deviceId, const uint64_t size, const void* buffer)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("deviceId {} size 0x{:x} buffer 0x{:x}", deviceId, size, TO64(buffer));

    status = _SYN_SINGLETON_->mapBufferToDevice(deviceId, size, const_cast<void*>(buffer), 0);
    API_EXIT_STATUS_TIMED(status, synHostMapP);
}

synStatus SYN_API_CALL synHostUnmap(const synDeviceId deviceId, const void* buffer)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("deviceId {} buffer 0x{:x}", deviceId, TO64(buffer));
    status = _SYN_SINGLETON_->unmapBufferFromDevice(deviceId, const_cast<void*>(buffer));
    API_EXIT_STATUS_TIMED(status, synHostUnmapP);
}

synStatus SYN_API_CALL synDeviceMalloc(const synDeviceId deviceId,
                                       const uint64_t    size,
                                       uint64_t          requestedAddress,
                                       const uint32_t    flags,
                                       uint64_t*         buffer)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("deviceId {} size 0x{:x} requestedAddress 0x{:x} flags 0x{:x}",
                deviceId,
                size,
                requestedAddress,
                flags);

    if (flags != 0)
    {
        LOG_ERR(SYN_API, "{} called with flags={} ,must be equal to zero", HLLOG_FUNC, flags);
        status = synFail;
    }
    else
    {
        status = _SYN_SINGLETON_->allocateDeviceMemory(deviceId,
                                                       size,
                                                       synMemFlags::synMemDevice,
                                                       reinterpret_cast<void**>(buffer),
                                                       requestedAddress);
    }
    API_EXIT_STATUS_TIMED(status, synDeviceMallocP);
}

synStatus SYN_API_CALL synDeviceFree(const synDeviceId deviceId, const uint64_t buffer, const uint32_t flags)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("deviceId {} buffer 0x{:x} flags 0x{:x}", deviceId, buffer, flags);
    if (flags != 0)
    {
        LOG_ERR(SYN_API, "{} called with flags={} ,must be equal to zero", HLLOG_FUNC, flags);
        status = synFail;
    }
    else
    {
        status = _SYN_SINGLETON_->deallocateDeviceMemory(deviceId,
                                                         reinterpret_cast<void*>(buffer),
                                                         synMemFlags::synMemDevice);
    }
    API_EXIT_STATUS_TIMED(status, synDeviceFreeP);
}

synStatus SYN_API_CALL synDeviceRelease(const synDeviceId deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);
    status = _SYN_SINGLETON_->releaseDevice(deviceId);
    API_EXIT_STATUS_TIMED(status, synDeviceReleaseP);
}

synStatus SYN_API_CALL synDeviceGetMemoryInfo(const synDeviceId deviceId, uint64_t* free, uint64_t* total)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);

    VERIFY_IS_NULL_POINTER(free, "free");
    VERIFY_IS_NULL_POINTER(total, "total");

    status = _SYN_SINGLETON_->getDeviceDramMemoryInfo(deviceId, *free, *total);
    API_EXIT_STATUS_TIMED(status, synDeviceGetMemoryInfoP);
}

synStatus SYN_API_CALL synDeviceGetInfo(const synDeviceId deviceId, synDeviceInfo* pDeviceInfo)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);

    VERIFY_IS_NULL_POINTER(pDeviceInfo, "Device-info");

    status = _SYN_SINGLETON_->getDeviceInfo(deviceId, pDeviceInfo);
    API_EXIT_STATUS_TIMED(status, synDeviceGetInfoP);
}

synStatus SYN_API_CALL synDeviceGetInfoV2(const synDeviceId deviceId, synDeviceInfoV2* pDeviceInfo)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);

    VERIFY_IS_NULL_POINTER(pDeviceInfo, "Device-info");

    status = _SYN_SINGLETON_->getDeviceInfo(deviceId, pDeviceInfo);
    API_EXIT_STATUS_TIMED(status, synDeviceGetInfoP);
}

synStatus SYN_API_CALL synDeviceGetAttribute(uint64_t*                 retVal,
                                             const synDeviceAttribute* deviceAttr,
                                             const unsigned            querySize,
                                             const synDeviceId         deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("querySize {} deviceId {}", querySize, deviceId);
    status = _SYN_SINGLETON_->getDeviceAttribute(deviceId, deviceAttr, querySize, retVal);
    API_EXIT_STATUS_TIMED(status, synDeviceGetAttributeP);
}

synStatus SYN_API_CALL synDeviceGetAttributeByModuleId( uint64_t*                 retVal,
                                                        const synDeviceAttribute* deviceAttr,
                                                        const unsigned            querySize,
                                                        const synModuleId         moduleId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("querySize {} moduleId {}", querySize, moduleId);
    status = _SYN_SINGLETON_->getDeviceAttributesByModuleId(moduleId, deviceAttr, querySize, retVal);
    API_EXIT_STATUS_TIMED(status, synDeviceGetAttributeByModuleIdP);
}

synStatus SYN_API_CALL synDeviceTypeGetAttribute(uint64_t*                 retVal,
                                                 const synDeviceAttribute* deviceAttr,
                                                 const unsigned            querySize,
                                                 const synDeviceType       deviceType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("querySize {} deviceType {}", querySize, deviceType);
    status = _SYN_SINGLETON_->getDeviceTypeAttribute(deviceType, deviceAttr, querySize, retVal);
    API_EXIT_STATUS_TIMED(status, synDeviceTypeGetAttributeP);
}

synStatus SYN_API_CALL synConfigurationSet(const char* configurationName, const char* configurationValue)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->setCfg(configurationName, configurationValue);
    API_EXIT_STATUS_TIMED(status, synConfigurationSetP);
}

synStatus SYN_API_CALL synConfigurationGet(const char* configurationName, char* configurationValue, uint64_t size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->getCfg(configurationName, configurationValue, size);
    API_EXIT_STATUS_TIMED(status, synConfigurationGetP);
}

synStatus SYN_API_CALL synTensorHandleCreate(synTensor*     tensor,
                                             synGraphHandle graph,
                                             synTensorType  type,
                                             const char*    tensorName)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->createTensor(tensor, graph, type, tensorName);
    API_EXIT_STATUS_TIMED(status, synTensorHandleCreateP);
}

synStatus SYN_API_CALL synTensorAssignToSection(synTensor tensor, synSectionHandle section, uint64_t byteOffset)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorAssignToSection(tensor, section, byteOffset);
    API_EXIT_STATUS_TIMED(status, synTensorAssignToSectionP);
}

synStatus SYN_API_CALL synTensorSetSectionOffset(synTensor           tensor,
                                                 uint64_t            byteOffset)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetSectionOffset(tensor, byteOffset);
    API_EXIT_STATUS_TIMED(status, synTensorSetSectionOffsetP);
}

synStatus SYN_API_CALL
synTensorSetHostPtr(synTensor tensor, void* hostPtr, uint64_t size, synDataType dataType, bool copyBuffer)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    HB_ASSERT(copyBuffer, "copyBuffer must be true, depracated argument");
    status = _SYN_SINGLETON_->tensorSetHostPtr(tensor, hostPtr, size, dataType, copyBuffer);
    API_EXIT_STATUS_TIMED(status, synTensorSetHostPtrP);
}

synStatus SYN_API_CALL synTensorSetQuantizationData(synTensor               tensor,
                                                    synQuantizationProperty prop,
                                                    void*                   propVal,
                                                    uint64_t                propSize)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetQuantizationData(tensor, prop, propVal, propSize);
    API_EXIT_STATUS_TIMED(status, synTensorSetQuantizationDataP);
}

synStatus SYN_API_CALL synTensorSetExternal(synTensor tensor, bool isExternal)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetIsExternal(tensor, isExternal);
    API_EXIT_STATUS_TIMED(status, synTensorSetExternalP);
}

synStatus SYN_API_CALL synTensorSetGeometryExt(synTensor                   tensor,
                                               const synTensorGeometryExt* geometry,
                                               synGeometryType             geometryType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetGeometryExt(tensor, geometry, geometryType);
    API_EXIT_STATUS_TIMED(status, synTensorSetGeometryExtP);
}

synStatus SYN_API_CALL synTensorSetGeometry(synTensor                tensor,
                                            const synTensorGeometry* geometry,
                                            synGeometryType          geometryType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetGeometryExt(tensor, (synTensorGeometryExt*)geometry, geometryType);
    API_EXIT_STATUS_TIMED(status, synTensorSetGeometryP);
}

synStatus SYN_API_CALL synTensorSetDeviceDataType(synTensor tensor, synDataType deviceDataType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetDeviceDataType(tensor, deviceDataType);
    API_EXIT_STATUS_TIMED(status, synTensorSetDeviceDataTypeP);
}

synStatus SYN_API_CALL synTensorSetDeviceLayout(synTensor tensor, const synTensorDeviceLayout* layout)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetDeviceLayout(tensor, layout);
    API_EXIT_STATUS_TIMED(status, synTensorSetDeviceLayoutP);
}

synStatus SYN_API_CALL synTensorSetPermutation(synTensor tensor, const synTensorPermutation* permutation)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetPermutation(tensor, permutation);
    API_EXIT_STATUS_TIMED(status, synTensorSetPermutationP);
}

synStatus SYN_API_CALL synTensorSetDeviceFullLayout(synTensor tensor, const synTensorDeviceFullLayout* layout)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorSetDeviceFullLayout(tensor, layout);
    API_EXIT_STATUS_TIMED(status, synTensorSetDeviceFullLayoutP);
}

synStatus SYN_API_CALL synTensorGetSection(synTensor tensor, synSectionHandle* section, uint64_t* byteOffset)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API()
    status = _SYN_SINGLETON_->tensorGetSection(tensor, section, byteOffset);
    API_EXIT_STATUS_TIMED(status, synTensorGetSectionP);
}

synStatus SYN_API_CALL synTensorSetAllowPermutation(synTensor tensor, int8_t allow)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API()
    status = _SYN_SINGLETON_->tensorSetAllowPermutation(tensor, allow);
    API_EXIT_STATUS_TIMED(status, synTensorSetAllowPermutationP);
}

synStatus SYN_API_CALL synTensorGetAllowPermutation(synTensor tensor, int8_t* allow)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API()
    status = _SYN_SINGLETON_->tensorGetAllowPermutation(tensor, allow);
    API_EXIT_STATUS_TIMED(status, synTensorGetAllowPermutationP);
}

synStatus SYN_API_CALL synTensorGetHostPtr(synTensor tensor, void** hostPtr, uint64_t* size, synDataType* dataType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API()
    status = _SYN_SINGLETON_->tensorGetHostPtr(tensor, hostPtr, size, dataType);
    API_EXIT_STATUS_TIMED(status, synTensorGetHostPtrP);
}

synStatus SYN_API_CALL synTensorGetQuantizationData(synTensor               tensor,
                                                    synQuantizationProperty prop,
                                                    void*                   propVal,
                                                    uint64_t                propSize)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API()
    status = _SYN_SINGLETON_->tensorGetQuantizationData(tensor, prop, propVal, propSize);
    API_EXIT_STATUS_TIMED(status, synTensorGetQuantizationDataP);
}

synStatus SYN_API_CALL synTensorGetGeometryExt(synTensor             tensor,
                                               synTensorGeometryExt* geometry,
                                               synGeometryType       geometryType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetGeometryExt(tensor, geometry, geometryType);
    API_EXIT_STATUS_TIMED(status, synTensorGetGeometryExtP);
}

synStatus SYN_API_CALL synTensorGetGeometry(synTensor tensor, synTensorGeometry* geometry, synGeometryType geometryType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetGeometryExt(tensor, (synTensorGeometryExt*)geometry, geometryType);
    API_EXIT_STATUS_TIMED(status, synTensorGetGeometryP);
}

synStatus SYN_API_CALL synTensorGetPermutation(const synTensor tensor, synTensorPermutation* permutation)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetPermutation(tensor, permutation);
    API_EXIT_STATUS_TIMED(status, synTensorGetPermutationP);
}

synStatus SYN_API_CALL synTensorGetDeviceFullLayout(synTensor tensor, synTensorDeviceFullLayout* layout)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetDeviceFullLayout(tensor, layout);
    API_EXIT_STATUS_TIMED(status, synTensorGetDeviceFullLayoutP);
}

synStatus SYN_API_CALL synTensorGetExternal(const synTensor tensor, bool* isExternal)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetIsExternal(tensor, isExternal);
    API_EXIT_STATUS_TIMED(status, synTensorGetExternalP);
}

synStatus SYN_API_CALL synTensorGetDeviceDataType(synTensor tensor, synDataType* deviceDataType)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetDeviceDataType(tensor, deviceDataType);
    API_EXIT_STATUS_TIMED(status, synTensorGetDeviceDataTypeP);
}

synStatus SYN_API_CALL synTensorGetDeviceLayout(synTensor tensor, synTensorDeviceLayout* layout)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetDeviceLayout(tensor, layout);
    API_EXIT_STATUS_TIMED(status, synTensorGetDeviceLayoutP);
}

synStatus SYN_API_CALL synTensorGetName(synTensor tensor, const uint64_t size, char* name)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetName(tensor, size, name);
    API_EXIT_STATUS_TIMED(status, synTensorGetNameP);
}

synStatus SYN_API_CALL synTensorGetType(synTensor tensor, synTensorType* type)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->tensorGetType(tensor, type);
    API_EXIT_STATUS_TIMED(status, synTensorGetTypeP);
}

synStatus SYN_API_CALL synTPCLibraryGetVersionSize(uint32_t* size)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->getTPCLibraryVersionSize(size);
    API_EXIT_STATUS_TIMED(status, synTPCLibraryGetVersionSizeP);
}

synStatus SYN_API_CALL synTPCLibraryGetVersions(const char** libs, uint32_t* versions)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->getTPCLibraryVersions(libs, versions);
    API_EXIT_STATUS_TIMED(status, synTPCLibraryGetVersionsP);
}

synStatus SYN_API_CALL synGraphDuplicate(synGraphHandle      graphHandle,
                                         synGraphHandle*     newGraphHandle,
                                         synTensorHandleMap* tensorsMap,
                                         uint32_t*           numTensors,
                                         synNodeHandleMap*   nodesMap,
                                         uint32_t*           numNodes)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->duplicateGraph(graphHandle, newGraphHandle, tensorsMap, numTensors, nodesMap, numNodes);
    API_EXIT_STATUS_TIMED(status, synGraphDuplicateP);
}

synStatus SYN_API_CALL synGraphInferShapes(synGraphHandle graphHandle)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API();
    status = _SYN_SINGLETON_->inferGraphShapes(graphHandle);
    API_EXIT_STATUS_TIMED(status, synGraphInferShapesP);
}

synStatus SYN_API_CALL synStreamSetAffinity(const synDeviceId       deviceId,
                                            const synStreamHandle   pStreamHandle,
                                            uint64_t                streamAffinityMask)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}, pStreamHandle 0x{:x}, streamAffinityMask {}",
                deviceId,
                TO64(pStreamHandle),
                streamAffinityMask);
    status = _SYN_SINGLETON_->setStreamAffinity(deviceId, pStreamHandle, streamAffinityMask);
    API_EXIT_STATUS_TIMED(status, synStreamSetAffinityP);
}

synStatus SYN_API_CALL synStreamGetAffinity(const synDeviceId       deviceId,
                                            const synStreamHandle   pStreamHandle,
                                            uint64_t*               streamAffinityMask)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}, pStreamHandle 0x{:x}", deviceId, TO64(pStreamHandle));
    status = _SYN_SINGLETON_->getStreamAffinity(deviceId, pStreamHandle, streamAffinityMask);
    API_EXIT_STATUS_TIMED(status, synStreamGetAffinityP);
}

synStatus SYN_API_CALL synDeviceGetAffinityMaskRange(const synDeviceId  deviceId,
                                                     uint64_t*          deviceAffinityMaskRange)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);
    status = _SYN_SINGLETON_->getDeviceAffinityMaskRange(deviceId, deviceAffinityMaskRange);
    API_EXIT_STATUS_TIMED(status, synDeviceGetAffinityMaskRangeP);
}

synStatus SYN_API_CALL synDeviceGetNextStreamAffinity(const synDeviceId deviceId, uint64_t* nextDeviceAffinity)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {}", deviceId);
    status = _SYN_SINGLETON_->getDeviceNextStreamAffinity(deviceId, nextDeviceAffinity);
    API_EXIT_STATUS_TIMED(status, synDeviceGetNextStreamAffinity);
}

synStatus SYN_API_CALL synNodeSetUserParams(const synGraphHandle graphHandle,
                                            const synNodeId      nodeId,
                                            const void*          userParams,
                                            const unsigned       paramsSize)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("graphHandle 0x{:x} synNodeId {} paramsSize {} userParams 0x{:x}",
                TO64(graphHandle),
                nodeId,
                paramsSize,
                TO64(userParams));
    status = _SYN_SINGLETON_->nodeSetParams(graphHandle, nodeId, userParams, paramsSize);
    API_EXIT_STATUS_TIMED(status, synNodeSetUserParamsP);
}

synStatus SYN_API_CALL synNodeGetUserParams(const synGraphHandle graphHandle,
                                            const synNodeId      nodeId,
                                            void*                userParams,
                                            unsigned*            paramsSize)
{
    API_ENTRY_STATUS_TIMED();
    LOG_SYN_API("graphHandle 0x{:x} synNodeId {}", TO64(graphHandle), nodeId);
    if (userParams != nullptr)
    {
        LOG_SYN_API("paramsSize 0x{:x}", TO64(paramsSize));
    }
    else
    {
        LOG_SYN_API("{} will set paramsSize to point to the buffer size that needs to be allocated for userParams",
                    HLLOG_FUNC);
    }
    status = _SYN_SINGLETON_->nodeGetParams(graphHandle, nodeId, userParams, paramsSize);
    API_EXIT_STATUS_TIMED(status, synNodeGetUserParamsP);
}

synStatus SYN_API_CALL synStatusGetBriefDescription(synStatus status, char* statusDescription, size_t len)
{
    return synSingleton::convertStatusToString(status, statusDescription, len);
}
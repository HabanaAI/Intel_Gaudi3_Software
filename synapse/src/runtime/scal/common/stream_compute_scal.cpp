#include "stream_compute_scal.hpp"

#include "habana_global_conf_runtime.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "scal_event.hpp"
#include "synapse_common_types.h"
#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "defenders.h"
#include "device/device_mem_alloc.hpp"

#include "log_manager.h"

#include "runtime/scal/common/entities/scal_completion_group.hpp"
#include "runtime/scal/common/entities/scal_memory_pool.hpp"
#include "runtime/scal/common/entities/scal_stream_compute_interface.hpp"
#include "global_statistics.hpp"

#include "profiler_api.hpp"

QueueComputeScal::QueueComputeScal(const BasicQueueInfo&           rBasicQueueInfo,
                                   ScalStreamComputeInterface*     pScalStream,
                                   const ComputeCompoundResources* pComputeResources,
                                   synDeviceType                   deviceType,
                                   const ScalDevSpecificInfo&      rDevSpecificInfo,
                                   DevMemoryAllocInterface&        devMemoryAlloc)
: QueueBaseScalCommon(rBasicQueueInfo, pScalStream),
  m_deviceType(deviceType),
  m_rDevSpecificInfo(rDevSpecificInfo),
  m_memMgrs(m_scalStream->getName(), devMemoryAlloc),
  m_launchTracker(GCFG_NUM_OF_CSDC_TO_CHK.value()),
  m_devMemoryAlloc(devMemoryAlloc),
  m_computeResources(*pComputeResources)
{
    // Currently, we will have internal-signal for the TX-Command
    // Hence, in case that is the compute-stream, this will need to be avoided
    HB_ASSERT(!(pComputeResources->m_pTxCommandsStream->isComputeStream()),
              "scal_stream_copy_xxx::memcopy is needed to be updated to supprt this mode");
}

QueueComputeScal::~QueueComputeScal() {}

synStatus QueueComputeScal::eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
#ifdef DISABLE_SYNC_ON_DEV
    synchronizeStream(streamHandle);
    return synSuccess;
#endif
    ScalEvent& rScalEvent = dynamic_cast<ScalEvent&>(rEventInterface);
    rScalEvent.clearState();
    rScalEvent.pStreamIfScal = this;

    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    return m_scalStream->eventRecord(true, rScalEvent);
}

synStatus QueueComputeScal::eventWait(const EventInterface& rEventInterface,
                                      const unsigned int    flags,
                                      synStreamHandle       streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());

#ifdef DISABLE_SYNC_ON_DEV
    return synSuccess;
#endif

    const ScalEvent& tmpScalEvent = dynamic_cast<const ScalEvent&>(rEventInterface);
    // defend against overriding the event while we're working on it
    ScalEvent rcScalEvent = tmpScalEvent;

    ScalLongSyncObject longSo = rcScalEvent.longSo;
    if (rcScalEvent.isOnHclStream())
    {
        longSo.m_index       = rcScalEvent.hclSyncInfo.long_so_index;
        longSo.m_targetValue = rcScalEvent.hclSyncInfo.targetValue;
    }

    synStatus status;
    {
        std::lock_guard<std::timed_mutex> lock(m_userOpLock);
        status = m_scalStream->longSoWaitOnDevice(longSo, true);
    }

    return status;
}

synStatus QueueComputeScal::query()
{
    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    return m_scalStream->longSoWaitForLast(true, 0, __FUNCTION__);
}

synStatus QueueComputeScal::synchronize(synStreamHandle streamHandle, bool isUserRequest)
{
    return waitForLastLongSo(isUserRequest);
}

// this function can be called ONLY after successfull recipeLauncher->launch(launchInfo);
void QueueComputeScal::updateSfgEventsScalStream(EventWithMappedTensorDB& events)
{
    for (auto& event : events)
    {
        // validation is done beforehand
        auto scalEvent           = static_cast<ScalEvent*>(event.get());
        scalEvent->pStreamIfScal = this;
    }
}

/*
 ***************************************************************************************************
 *   @brief init() inits the recipeBuffs (used to move/cache the recipe in the HBM)
 *
 *   @param  Node
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::init(PreAllocatedStreamMemoryAll& preAllocatedInfo)
{
    if (m_initDone == true)
    {
        LOG_CRITICAL_T(SYN_STREAM,
                       "Stream {:x} m_scalStream {:x} steam {} already init",
                       TO64(this),
                       TO64(m_scalStream),
                       m_basicQueueInfo.getDescription());
        return synFail;
    }

    m_hbmGlblSize    = preAllocatedInfo.global.Size;
    m_hbmGlblAddr    = preAllocatedInfo.global.AddrDev;
    m_arcHbmSize     = preAllocatedInfo.shared.Size;
    m_arcHbmAddrCore = preAllocatedInfo.shared.AddrCore;
    m_arcHbmAddrDev  = preAllocatedInfo.shared.AddrDev;

    {
        synStatus status = initMemMgrs();
        if (status != synSuccess)
        {
            return status;
        }
    }

    m_initDone = true;
    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief initMemMgrs() inits all the memory managers
 *
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::initMemMgrs()
{
    m_memMgrs.hbmGlblMemMgr.init(m_hbmGlblAddr, m_hbmGlblSize);
    m_memMgrs.arcHbmMemMgr.init(m_arcHbmAddrCore, m_arcHbmAddrDev, m_arcHbmSize);
    return m_memMgrs.mappedMemMgr.init();
}

/*
 ***************************************************************************************************
 * get hbm global memory for one program (simultaneously HBM_BUFFS_AMOUNT can reside in this memory)
 *
 * @param maxGlbHbmMemorySize
 * @return chunk size for one program
 ***************************************************************************************************
 */
uint64_t QueueComputeScal::getHbmGlblMaxRecipeSize(uint64_t maxGlbHbmMemorySize)
{
    return HbmGlblMemMgr::getMaxRecipeSize(maxGlbHbmMemorySize);
}

/*
 ***************************************************************************************************
 * get hbm shared memory for one program (simultaneously HBM_BUFFS_AMOUNT can reside in this memory)
 *
 * @param maxSharedHbmMemorySize
 * @return chunk size for one program
 ***************************************************************************************************
 */
uint64_t QueueComputeScal::getHbmSharedMaxRecipeSize(uint64_t maxArcHbmMemorySize)
{
    return ArcHbmMemMgr::getMaxRecipeSize(maxArcHbmMemorySize);
}

/*
 ***************************************************************************************************
 * get getDynamicShapeProcessor
 * @return DynamicRecipe
 ***************************************************************************************************
 */
DynamicRecipe* QueueComputeScal::getDynamicShapeProcessor(const LaunchInfo& launchInfo)
{
    uint64_t recipeId        = launchInfo.pRecipeHandle->recipeSeqNum;
    auto     dsdProcessorItr = m_dynamicShapeProcessor.find(recipeId);

    if (dsdProcessorItr != m_dynamicShapeProcessor.end())
    {
        LOG_DEBUG_T(SYN_STREAM, "DSD processor for recipe 0x{:x} already on stream", (uint64_t)recipeId);
    }
    else
    {
        const DeviceAgnosticRecipeInfo& deviceAgnostcInfo   = launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle;
        const RecipeAddrPatcher*        pRecipeAddrPatcher  = &deviceAgnostcInfo.m_recipeStaticInfoScal.recipeAddrPatcher;
        std::unique_ptr<DynamicRecipe>  dsdProcessor        =
            std::make_unique<DynamicRecipe>(launchInfo.pRecipeHandle->basicRecipeHandle,
                                            deviceAgnostcInfo,
                                            &deviceAgnostcInfo.m_recipeStaticInfoScal.recipeDsdPpInfo.getDsdDCPatchingInfo(),
                                            pRecipeAddrPatcher);
        m_dynamicShapeProcessor[recipeId] = std::move(dsdProcessor);
    }
    return m_dynamicShapeProcessor[recipeId].get();
}

synStatus QueueComputeScal::launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                                   uint32_t                      launchTensorsAmount,
                                   uint64_t                      workspaceAddress,
                                   InternalRecipeHandle*         pRecipeHandle,
                                   uint64_t                      assertAsyncMappedAddress,
                                   uint32_t                      flags,
                                   EventWithMappedTensorDB&      events,
                                   uint8_t                       apiId)
{
    CHECK_POINTER(SYN_STREAM, pRecipeHandle, "pRecipeHandle", synFail);
    basicRecipeInfo& basicRecipeHandle = pRecipeHandle->basicRecipeHandle;
    HB_ASSERT_PTR(basicRecipeHandle.recipe);

    const uint64_t recipeHandle = (uint64_t)pRecipeHandle;
    const uint64_t recipeSeqNum = pRecipeHandle->recipeSeqNum;
    const uint64_t seqId        = QueueComputeUtils::getLaunchSeqId();
    const uint64_t inFlight     = m_launchTracker.size();

    STAT_GLBL_COLLECT(seqId, seqId);
    STAT_GLBL_COLLECT(pRecipeHandle->recipeSeqNum, recipeId);

    PROFILER_COLLECT_TIME()

    LOG_DEBUG_T(
        SYN_STREAM,
        "{}: Stream {:#x} recipeHandle {:#x} recipeSeqNum {} seqId {} inFlight {} workspaceAddress {:#x} apiId {}",
        HLLOG_FUNC,
        TO64(this),
        recipeHandle,
        recipeSeqNum,
        seqId,
        inFlight,
        workspaceAddress,
        apiId);

    if (QueueComputeUtils::isRecipeEmpty(*pRecipeHandle))
    {
        return synSuccess;
    }

    synStatus status = QueueComputeUtils::prepareLaunch(m_basicQueueInfo,
                                                        launchTensorsInfo,
                                                        launchTensorsAmount,
                                                        pRecipeHandle,
                                                        events,
                                                        flags);
    if (status != synSuccess)
    {
        return status;
    }

    LaunchInfo launchInfo(launchTensorsInfo,
                          launchTensorsAmount,
                          workspaceAddress,
                          assertAsyncMappedAddress,
                          pRecipeHandle,
                          flags,
                          m_rDevSpecificInfo,
                          events,
                          apiId);

    status = launch(launchInfo);

    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_STREAM, "SCAL launch failed with status {}", status);
        return status;
    }

    ProfilerApi::setHostProfilerApiId(apiId);

    const std::string profilerStr(
        "enqueueWithExternalEvents recipeHandle " + fmt::format(FMT_COMPILE("{:#x}"), recipeHandle) + " recipeSeqNum " +
        std::to_string(recipeSeqNum) + " seqId " + std::to_string(seqId) + " inFlight " + std::to_string(inFlight));
    PROFILER_MEASURE_TIME(profilerStr.c_str());

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief launch() Does the launch on the stream. It gets the HBM addresses from the recipeBuffs
 *   and sends the needed commands to scal to do the work.
 *   Based on some habana global config flags it also:
 *   read the recipe from the HBM and compares it to the recipe from the GC
 *   Dumps the recipe to the log
 *   Skips sending the commands to scal (for testing)
 *
 *   @param  LaunchInfo - all information needed for this specific launch
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::launch(const LaunchInfo& launchInfo)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);

    STAT_GLBL_START(scalChkCompletion);
    uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
    m_launchTracker.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
    STAT_GLBL_COLLECT_TIME(scalChkCompletion, globalStatPointsEnum::scalChkCompletion);

    DynamicRecipe* dsdProcessor = nullptr;
    bool           isDsd        = RecipeUtils::isDsd(launchInfo.pRecipeHandle->basicRecipeHandle);

    if (isDsd)
    {
        dsdProcessor = getDynamicShapeProcessor(launchInfo);
    }

    ScalStreamComputeInterface* pComputeScalStream = dynamic_cast<ScalStreamComputeInterface*>(m_scalStream);
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pComputeScalStream, "pComputeScalStream");

    std::unique_ptr<RecipeLauncher> recipeLauncher = std::make_unique<RecipeLauncher>(pComputeScalStream,
                                                                                      &m_computeResources,
                                                                                      m_devMemoryAlloc,
                                                                                      m_memMgrs,
                                                                                      m_launchTracker,
                                                                                      launchInfo.pRecipeHandle,
                                                                                      dsdProcessor,
                                                                                      ++m_runningId,
                                                                                      launchInfo.m_apiId);

    synStatus status = recipeLauncher->launch(launchInfo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: recipeLauncher launch failed, recipe seq {}",
                HLLOG_FUNC,
                launchInfo.pRecipeHandle->recipeSeqNum);
        // in case of failure after enqueue, add launcher to tracker anyway
        if (status == synFailedSectionValidation)
        {
            m_launchTracker.add(std::move(recipeLauncher));
        }
        return status;
    }

    updateSfgEventsScalStream(launchInfo.events);
    m_launchTracker.add(std::move(recipeLauncher));

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief getMappedMemorySize() returns the size of mapped memory used by this stream
 *
 *   The size returned is of the global-hbm memory size (mapped memory mgr), not arc-shared memory
 *
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = m_memMgrs.mappedMemMgr.getMappedMemorySize();
    return synSuccess;
}

void QueueComputeScal::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);

    m_launchTracker.checkRecipeCompletion(rRecipeHandle);

    RecipeSeqId recipeSeqId(rRecipeHandle.recipeSeqNum);
    m_memMgrs.mappedMemMgr.removeId(recipeSeqId);

    auto recipeProcessorsIter = m_dynamicShapeProcessor.find(rRecipeHandle.recipeSeqNum);
    if (recipeProcessorsIter != m_dynamicShapeProcessor.end())
    {
        m_dynamicShapeProcessor.erase(recipeProcessorsIter);
    }
}

void QueueComputeScal::notifyAllRecipeRemoval()
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);

    m_launchTracker.checkForCompletionAll();

    m_memMgrs.mappedMemMgr.removeAllId();

    m_dynamicShapeProcessor.clear();
}

/**
 * log all recipes related to the current stream
 * @param oldestRecipeOnly if true, logs only the oldest recipe (=the one that failed). otherwise, logs all related
 * recipes.
 */
void QueueComputeScal::dfaUniqStreamInfo(bool               oldestRecipeOnly,
                                         uint64_t           currentLongSo,
                                         bool               dumpRecipe,
                                         const std::string& callerMsg)
{
    if (!oldestRecipeOnly)
    {
        m_launchTracker.dfaLogRecipesDesc(oldestRecipeOnly, currentLongSo, dumpRecipe, callerMsg, true);
    }

    LOG_INFO(SYN_DEV_FAIL, "- Related recipe(s) -");

    m_launchTracker.dfaLogRecipesDesc(oldestRecipeOnly, currentLongSo, dumpRecipe, callerMsg, false);
}

synStatus QueueComputeScal::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                            std::vector<tensor_info_t>& tensorInfoArray) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "recipeHandle");
    uint64_t recipeId        = recipeHandle->recipeSeqNum;
    auto     dsdProcessorItr = m_dynamicShapeProcessor.find(recipeId);

    if (dsdProcessorItr == m_dynamicShapeProcessor.end())
    {
        LOG_DEBUG_T(SYN_STREAM, "DSD processor for recipe 0x{:x} doesn't exist", (uint64_t)recipeId);
        return synInvalidArgument;
    }

    tensorInfoArray = dsdProcessorItr->second.get()->getDynamicShapesTensorInfoArray();
    return synSuccess;
}

std::set<ScalStreamCopyInterface*> QueueComputeScal::dfaGetQueueScalStreams()
{
    std::set<ScalStreamCopyInterface*> set;

    set.insert(m_scalStream);
    set.insert(m_computeResources.m_pTxCommandsStream);
    set.insert(m_computeResources.m_pDev2DevCommandsStream);
    set.insert(m_computeResources.m_pRxCommandsStream);

    return set;
}

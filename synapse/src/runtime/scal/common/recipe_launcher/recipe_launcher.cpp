#include "recipe_launcher.hpp"
#include "defs.h"
#include "habana_global_conf_runtime.h"
#include "mapped_memory_sections_utils.hpp"
#include "mem_mgrs.hpp"
#include "recipe.h"
#include "synapse_common_types.h"
#include "device/device_mem_alloc.hpp"
#include "runtime/common/recipe/patching/host_address_patcher.hpp"
#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/recipe_patch_processor.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"
#include "runtime/common/recipe/recipe_logger.hpp"
#include "runtime/scal/common/entities/scal_stream_copy_interface.hpp"
#include "runtime/scal/common/entities/scal_stream_compute_interface.hpp"
#include "runtime/scal/common/scal_event.hpp"
#include "log_manager.h"
#include "runtime/scal/common/infra/scal_utils.hpp"
#include "runtime/scal/common/patching/recipe_addr_patcher.hpp"
#include <limits>
#include "infra/memory_utils.h"
#include "global_statistics.hpp"
/*
 ***************************************************************************************************
 *   @brief RecipeLauncher() - constructor, keeps some needed pointer/references needed for the launch
 *
 *   @param  stream, sync information (kept between RecipeLauncher), memory allocator and more
 *   @param  runningId - unique Id given by the caller
 *   @return None
 *
 ***************************************************************************************************
 */
RecipeLauncher::RecipeLauncher(ScalStreamComputeInterface*     pComputeScalStream,
                               const ComputeCompoundResources* pComputeResources,
                               DevMemoryAllocInterface&        devMemoryAlloc,
                               MemMgrs&                        memMgrs,
                               LaunchTrackerInterface&         rLaunchTracker,
                               const InternalRecipeHandle*     pRecipeHandle,
                               DynamicRecipe*                  pDynamicRecipeProcessor,
                               uint64_t                        runningId,
                               uint8_t                         apiId)
: m_devMemoryAlloc(devMemoryAlloc),
  m_rLaunchTracker(rLaunchTracker),
  m_pRecipeHandle(pRecipeHandle),
  m_wsInfo(),
  m_memMgrs(memMgrs),
  m_pComputeScalStream(pComputeScalStream),
  m_pComputeResources(pComputeResources),
  m_longSoCopy(LongSoEmpty),
  m_longSoCompute(LongSoEmpty),
  m_entryIds({RecipeSeqId(pRecipeHandle->recipeSeqNum), runningId}),
  m_isDsd(RecipeUtils::isDsd(m_pRecipeHandle->basicRecipeHandle)),
  m_pDynamicRecipeProcessor(pDynamicRecipeProcessor),
  m_isIH2DRecipe(RecipeUtils::isIH2DRecipe(m_pRecipeHandle->basicRecipeHandle.recipe)),
  m_apiId(apiId)
{
    HB_ASSERT(pComputeScalStream->getStaticMonitorPayloadInfo(m_computeWorkCompletionAddress, m_computeWorkCompletionValue) == true,
              "Failed to get Compute's Monitor payload-address");

    LOG_TRACE(SYN_STREAM, "m_computeWorkCompletionAddress {:#x} m_computeWorkCompletionValue {:#x}",
                   m_computeWorkCompletionAddress, m_computeWorkCompletionValue);

    pComputeScalStream->longSoRecord(false, m_longSoHbmBuff);
}

synStatus validateEvents(EventWithMappedTensorDB& events)
{
    // set longSo for ext tensors mapped events
    for (auto& event : events)
    {
        ScalEvent* scalEvent = dynamic_cast<ScalEvent*>(event.get());
        if (scalEvent == nullptr)
        {
            LOG_ERR(SYN_STREAM, "{}: failed to convert event {} to ScalEvent", HLLOG_FUNC, event->toString());
            return synInvalidEventHandle;
        }
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief launch() - This function (by calling others) put the work on the device.
 *   It calls the following steps:
 *   Download to device HBM (including patching)
 *   For debug - re-read and compares to original recipe
 *   enqueue to device (sends packets with recipe location)
 *   deletes the pre_launch info (m_pre)
 *
 *   @param  launchInfo - needed launch information (tensors, workspace addr, flags, etc.)
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::launch(const LaunchInfo& launchInfo)
{
    LOG_TRACE(SYN_STREAM, "Start on recipe {:x} runningId {:x}", m_entryIds.recipeId.val, m_entryIds.runningId);

    // events validation should be out of unrecoverable region
    synStatus status = validateEvents(launchInfo.events);
    if (status != synSuccess)
    {
        return status;
    }

    status = prepareForDownload(launchInfo);
    if (status != synSuccess)
    {
        clearMemMgrs();
        return status;
    }

    // download to device, get the address last pdma sends unfence
    status = downloadToDev(launchInfo);
    if (status != synSuccess)
    {
        validateSectionsInfo(launchInfo);
        return status;
    }

    /*
     * =================================================================================================
     * UNRECOVERABLE REGION : START
     *
     * ATTENTION : from downloadToDev to scalEnqueue we can fail only if something really awful happened
     *             e.g. addCommand failed
     *             reason - there is no way to correctly recover from an error
     */

    const uint32_t fenceTarget = 1;

    // wait for downloadToDev to finish
    status = getComputeScalStream()->addStreamFenceWait(fenceTarget, false, true);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "addFenceWait on local fence failed with status {}", status);
        return status;
    }

    // In case we perform CompareAfterDownLoad, than we will wait-on-host, as part of the operation
    // Hence, we will not need to wait for it prior of the enqueue.
    if (ScalUtils::isCompareAfterDownLoad())
    {
        synStatus compareStatus = debugCompareRecipeOnDev(launchInfo.pRecipeHandle, "Compare after copy");
        HB_ASSERT(compareStatus == synSuccess, "Compare after copy failed with compareStatus {}", compareStatus);
    }

    uint64_t nbExtTensors =
        launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getNumberOfExternalTensors();
    updateSfgLongSos(nbExtTensors, launchInfo.events);

    STAT_GLBL_START(scalEnqueueStat);
    status = scalEnqueue(nbExtTensors);  // enqueue, event record for next launch
    if (status != synSuccess)
    {
        // if enqueue failed, the assumption is that we didn't reach the device
        // therefore no synFailedAfterEnqueue
        return status;
    }
    /*
     * UNRECOVERABLE REGION : END
     * =================================================================================================
     */

    STAT_GLBL_COLLECT_TIME(scalEnqueueStat, globalStatPointsEnum::scalEnqueue);

    STAT_GLBL_START(tensorsValidate);
    if (launchInfo.pRecipeHandle->basicRecipeHandle.recipe->patch_points_nr != 0)
    {
        status = validateSectionsInfo(launchInfo);
        if (status != synSuccess)
        {
            return status;
        }
    }
    STAT_GLBL_COLLECT_TIME(tensorsValidate, globalStatPointsEnum::tensorsValidate);

    if (RecipeUtils::isKernelPrintf(*launchInfo.pRecipeHandle))
    {
        handleKernelsPrintf(launchInfo);
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief checkCompletion() - checks if the oldest launch has finished. If is done, it notifies the
 *                              mapped memory. It is called from the LaunchTracker
 *
 *   @param  timeout - how much time to wait
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::checkCompletionCopy(uint64_t timeout)
{
    LOG_TRACE(SYN_STREAM, "RecipeLauncher {} checkCompletion with timeout {}", getDescription(), timeout);

    if (m_longSoCopy != LongSoEmpty)
    {
        synStatus status = getTxCommandsScalStream()->longSoWait(m_longSoCopy, timeout, __FUNCTION__);
        if (status != synSuccess)
        {
            return status;
        }

        LOG_DEBUG(SYN_STREAM, "RecipeLauncher {} Copy successfully completed", getDescription());
        m_longSoCopy = LongSoEmpty;

        if (ScalUtils::isCompareAfterDownloadPost())
        {
            synStatus compareStatus = debugCompareRecipeOnDev(m_pRecipeHandle, "Compare after copy post");
            HB_ASSERT(compareStatus == synSuccess,
                      "Compare after copy post failed with compareStatus {}",
                      compareStatus);
        }

        m_memMgrs.mappedMemMgr.unuseId(m_entryIds);
    }

    return synSuccess;
}

synStatus RecipeLauncher::checkCompletionCompute(uint64_t timeout)
{
    LOG_TRACE(SYN_STREAM, "RecipeLauncher {} checkCompletion with timeout {}", getDescription(), timeout);

    // Check copy completion on each call to compute completion in order to ensure that copy resources are already
    // released.
    synStatus status = checkCompletionCopy(timeout);
    if (status != synSuccess)
    {
        return status;
    }

    if (m_longSoCompute != LongSoEmpty)
    {
        status = getComputeScalStream()->longSoWait(m_longSoCompute, timeout, __FUNCTION__);
        if (status != synSuccess)
        {
            return status;
        }

        LOG_DEBUG(SYN_STREAM, "RecipeLauncher {} Compute successfully completed", getDescription());
        m_longSoCompute = LongSoEmpty;

        if (ScalUtils::isCompareAfterLaunch())
        {
            synStatus compareStatus = debugCompareRecipeOnDev(m_pRecipeHandle, "Compare after compute");
            HB_ASSERT(compareStatus == synSuccess, "Compare after compute failed with compareStatus {}", compareStatus);
        }
    }

    return synSuccess;
}

synStatus RecipeLauncher::validateSectionsInfo(const LaunchInfo& launchInfo)
{
    synStatus status = RecipePatchProcessor::validateSectionsInfo(
        launchInfo.pRecipeHandle->basicRecipeHandle,
        launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo,
        launchInfo.pEnqueueTensorsInfoAmount,
        launchInfo.pEnqueueTensorsInfo,
        launchInfo.launchFlags,
        m_wsInfo,
        launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo.m_sectionsInfo,
        m_devMemoryAlloc);

    return status;
}

/*
 ***************************************************************************************************
 *   @brief getDescription() - For logging, returns a string description of the class
 *
 *   @param  None
 *   @return string (description of class)
 *
 ***************************************************************************************************
 */
std::string RecipeLauncher::getDescription() const
{
    return fmt::format("SoCopy {:#x},{:#x} SoCompute {:#x},{:#x} (recipe, runningId): ({:#x}, {:#x})",
                       m_longSoCopy.m_index,
                       m_longSoCopy.m_targetValue,
                       m_longSoCompute.m_index,
                       m_longSoCompute.m_targetValue,
                       m_entryIds.recipeId.val,
                       m_entryIds.runningId);
}

static std::string sectionId2str(int i)
{
    switch (i)
    {
        case SectionType::PATCHABLE:     return "PATCHABLE    ";
        case SectionType::PROGRAM_DATA:  return "PROGRAM_DATA ";
        case SectionType::NON_PATCHABLE: return "NON_PATCHABLE";
        case SectionType::DYNAMIC:       return "DYNAMIC      ";
        default :                        return "ECB LIST     ";
    }
}

/**
 * Get description of the recipe formatted as strings.
 * @return a vector of strings, each string contains a different field about this recipe.
 */
bool RecipeLauncher::dfaLogDescription(bool               oldestRecipeOnly,
                                       uint64_t           currentLongSo,
                                       bool               dumpRecipe,
                                       const std::string& callerMsg,
                                       bool               forTools) const
{
    if (forTools)
    {
        LOG_INFO(SYN_DEV_FAIL,
                 "#recipe 0x{:x} expected SoCopy {:#x},{:#x} SoCompute {:#x},{:#x}",
                 m_pRecipeHandle->recipeSeqNum,
                 m_longSoCopy.m_index,
                 m_longSoCopy.m_targetValue,
                 m_longSoCompute.m_index,
                 m_longSoCompute.m_targetValue);
        return true;
    }

    // sometimes recipe name isn't available (for example, on eager mode), so replace it with a string
    if (m_longSoCompute.m_targetValue <= currentLongSo)
    {
        if (!oldestRecipeOnly)
        {
            LOG_INFO(SYN_DEV_FAIL,
                     "longSo still not cleared but already done {:x} <= {:x}",
                     m_longSoCompute.m_targetValue,
                     currentLongSo);
        }
        return false;
    }
    LOG_INFO(SYN_DEV_FAIL, "sync info:\t\t{}", getDescription());
    LOG_INFO(SYN_DEV_FAIL, "Recipe Hbm info: glbl {:x} arc {:x}", m_sections.m_glbHbmAddr, m_sections.m_arcHbmAddr);

    auto& sections = m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.recipeSections;

    for (int i = 0; i < sections.size(); i++)
    {
        LOG_INFO(SYN_DEV_FAIL, "Hbm offset for {}: {:x}", sectionId2str(i), sections[i].offsetHbm);
    }

    RecipeManager::dfaLogRecipeInfo(*m_pRecipeHandle);

    if (dumpRecipe)
    {
        RecipeLogger::dfaDumpRecipe(m_pRecipeHandle, true, callerMsg);

        if (GCFG_DFA_SAVE_RECIPE.value())
        {
            std::string fullFilePath;
            synapse::LogManager::getLogsFolderPath(fullFilePath);

            std::string fileName = fullFilePath + "/" + getComputeScalStream()->getName() + "-longSo" + std::to_string(currentLongSo) + ".recipe";

            synStatus status = RecipeManager::recipeSerialize(m_pRecipeHandle, fileName.c_str());
            if (status != synSuccess)
            {
                LOG_ERR(SYN_DEV_FAIL, "Failed to serialize failed recipe to {} with status {}", fileName, status);
            }
        }
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @brief updateSfgLongSos() - update longSo in the events an for the whole recipe launch
 *
 *   @param  nbExtTensors - number of external tensors
 *   @param  events       - events mapped to external tensors
 *
 ***************************************************************************************************
 */
void RecipeLauncher::updateSfgLongSos(uint64_t nbExtTensors, EventWithMappedTensorDB& events)
{
    // set longSo for ext tensors mapped events
    for (auto& event : events)
    {
        // validation is performed beforehand (validateEvents)
        // here we are sure that all the events are valid
        ScalEvent* scalEvent = static_cast<ScalEvent*>(event.get());
        scalEvent->longSo    = getComputeScalStream()->getTargetLongSo(event->getSequenceOffset() + 1);
    }

    // advance logSo by the number of external tensors regardless of the amount of events supplied by the user.
    // if 0, do not call the function to avoid and entry in the longSo entries update table
    if (nbExtTensors > 0)
    {
        m_longSoCompute = getComputeScalStream()->getIncrementedLongSo(true, nbExtTensors);
        LOG_TRACE(SYN_PROGRESS, "{:20} : {:>8x} : {:>8x} : {}/{}",
                 getComputeScalStream()->getName(),
                 m_longSoCompute.m_index,
                 m_longSoCompute.m_targetValue,
                 HLLOG_FUNC,
                 __LINE__);
    }
}

/*
 ***************************************************************************************************
 *   @brief scalEnqueue() - Sends the request to the device (after recipe already on HBM)
 *   It does the following steps:
 *   Waits for previous launch to finish
 *   Sends packets to sets the bases addresses
 *   Sends packets with the ecb lists
 *   Records the launch (so the next launch can wait on)
 *
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::scalEnqueue(uint64_t nbExtTensors)
{
    const RecipeStaticInfoScal& rRecipeStaticInfoScal =
        m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal;

    // take global fence
    const uint32_t globalFenceTarget = 1;
    getComputeScalStream()->addGlobalFenceWait(globalFenceTarget, false);

    const EngineGrpArr& computeEngineGrpArr = rRecipeStaticInfoScal.m_computeEngineGrpArr; // without CME
    synStatus status = addBaseAddresses(computeEngineGrpArr);
    if (status != synSuccess)
    {
        return status;
    }

    status = addEcbListWithBarrierAndSend(nbExtTensors);
    if (status != synSuccess)
    {
        return status;
    }

    // release global fence
    getComputeScalStream()->addGlobalFenceInc(true);

    // set the longSo on the HBM allocators (record an event to get the current longSo)
    ScalLongSyncObject scalLongSyncObject;
    getComputeScalStream()->longSoRecord(true, scalLongSyncObject);
    m_memMgrs.arcHbmMemMgr.setLongSo(scalLongSyncObject.m_targetValue);
    m_memMgrs.hbmGlblMemMgr.setLongSo(scalLongSyncObject.m_targetValue);

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief analyzeTensors() - Analyze tensors and process the recipe.
 *
 *   Get memory addresses from HBM and mapped memory
 *   Process recipe and analyze tensors
 *
 *   @param  launchInfo
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::analyzeTensors(const LaunchInfo& launchInfo)
{
    STAT_GLBL_START(tensorsAnalyze);
    m_wsInfo.scratchPadAddress  = launchInfo.workspaceAddress;
    m_wsInfo.programDataAddress = getHbmAddr(PROGRAM_DATA);
    m_wsInfo.programCodeAddress = getHbmAddr(NON_PATCHABLE);
    m_wsInfo.assertAsyncMappedAddress = launchInfo.m_assertAsyncMappedAddress;

    LOG_DEBUG(
        SYN_STREAM,
        "{} scratchPadAddress {:x} programDataAddress {:x} programCodeAddress {:x} assertAsyncMappedAddress {:x}",
        getDescription(),
        m_wsInfo.scratchPadAddress,
        m_wsInfo.programDataAddress,
        m_wsInfo.programCodeAddress,
        m_wsInfo.assertAsyncMappedAddress);

    patching::HostAddressPatchingInformation* hostAddrPatchInfo = m_sections.m_hostAddrPatchInfo;
    ValidSectionAddresses validSectionAddresses {launchInfo.devSpecificInfo.dramBaseAddr,
                                                 launchInfo.devSpecificInfo.dramEndAddr};

    synStatus status = RecipePatchProcessor::process(launchInfo.pRecipeHandle->basicRecipeHandle,
                                                     launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo,
                                                     launchInfo.pEnqueueTensorsInfo,
                                                     launchInfo.pEnqueueTensorsInfoAmount,
                                                     launchInfo.launchFlags,
                                                     m_wsInfo,
                                                     *hostAddrPatchInfo,
                                                     m_devMemoryAlloc,
                                                     m_tensorIdx2userIdx,
                                                     true /* isInitAndCompletionRequired */,
                                                     true /* shouldResolveTensorsIndices */,
                                                     &validSectionAddresses);
    // NOTE: at this point hostAddrPatchInfo doesn't hold the sectionTypes anymore because process() is calling
    //       patchingCompletion. I think we should solve it inside process() - maybe call patching completion
    //       before we start the process or something similar

    if (status != synSuccess)
    {
        return (status == synFail) ? synFailedSectionValidation : status;
    }
    STAT_GLBL_COLLECT_TIME(tensorsAnalyze, globalStatPointsEnum::tensorsAnalyze);

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief runSifPreDownload() - On DSD recipe run SIF before downloading to device as part of patching process.
 *                                Also allocate new program data buffer for IH2D recipe.
 *
 *   Allocate new program data buffer for IH2D recipe.
 *   Run SIF before downloading to device for DSD recipe.
 *
 *   @param  launchInfo
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::runSifPreDownload(const LaunchInfo& launchInfo)
{
    if (m_isIH2DRecipe)
    {
        STAT_GLBL_START(ih2dBufferMemAlloc);

        uint64_t recipeProgramDataSize  = m_pRecipeHandle->basicRecipeHandle.recipe->program_data_blobs_size;
        uint8_t* pRecipeProgramData     = (uint8_t*)(m_pRecipeHandle->basicRecipeHandle.recipe->program_data_blobs_buffer);

        m_sections.m_ih2dBuffer = std::make_unique<uint8_t []>(recipeProgramDataSize);
        memcpy(m_sections.m_ih2dBuffer.get(), pRecipeProgramData, recipeProgramDataSize);

        LOG_DEBUG(SYN_MEM_ALLOC,
                       "process IH2D recipe - allocating new buffer in addr: {:x} for program data with size: {:x}, "
                       "addr in recipe: {:x}",
                       TO64(m_sections.m_ih2dBuffer.get()),
                       recipeProgramDataSize,
                       TO64(pRecipeProgramData));

        STAT_GLBL_COLLECT_TIME(ih2dBufferMemAlloc, globalStatPointsEnum::ih2dBufferMemAlloc);
    }

    if (m_isDsd)
    {
        STAT_GLBL_START(patchingDsdSif);
        bool status = m_pDynamicRecipeProcessor->runSifOnAllNodes(launchInfo.pEnqueueTensorsInfo,
                                                                  launchInfo.pEnqueueTensorsInfoAmount,
                                                                  m_tensorIdx2userIdx, m_isIH2DRecipe ? (uint64_t)m_sections.m_ih2dBuffer.get() : 0);

        if (!status)
        {
            LOG_ERR(SYN_STREAM, "Failed to perform DSD SIF patching");
            m_pDynamicRecipeProcessor->patchAbort();
            return synFailedDynamicPatching;
        }
        STAT_GLBL_COLLECT_TIME(patchingDsdSif, globalStatPointsEnum::DsdSif);
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief prepareForDownload() - Analyze tensors and process the recipe, also run SIF if recipe is DSD
 *                                 and allocate new program data buffer for IH2D recipe.
 *
 *   Get memory addresses from HBM and mapped memory
 *   Process recipe and analyze tensors
 *   If IH2D allocate new program data buffer
 *   If DSD run SIF on all recipe nodes
 *
 *   @param  launchInfo
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::prepareForDownload(const LaunchInfo& launchInfo)
{
    STAT_GLBL_START(scalGetHbms);

    // Get address on hbm-global to be used
    uint64_t longSoGlbHbm = m_memMgrs.hbmGlblMemMgr.getAddr(
        launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.m_glbHbmSizeTotal,
        m_sections.m_glbHbmAddr);

    // get address for hbm-arc to be used
    uint64_t longSoArcHbm = m_memMgrs.arcHbmMemMgr.getAddr(
        launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.m_arcHbmSize,
        m_sections.m_arcHbmAddr,
        m_sections.m_arcHbmCoreAddr);

    // wait for the longSo that indicates the buffer is free
    uint64_t maxLongSo            = std::max(longSoGlbHbm, longSoArcHbm);
    m_longSoHbmBuff.m_targetValue = maxLongSo;

    LOG_DEBUG(SYN_PROG_DWNLD, "waiting for max({:x},{:x})", longSoGlbHbm, longSoArcHbm);
    synStatus status = getTxCommandsScalStream()->longSoWaitOnDevice(m_longSoHbmBuff, false);
    if (status != synSuccess)
    {
        return status;
    }

    STAT_GLBL_COLLECT_TIME(scalGetHbms, globalStatPointsEnum::scalGetHbms);

    STAT_GLBL_START(scalGetMapped);
    status = getMappedMemoryInfo();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "getMappedMemoryInfo failed with status {}", status);
        return status;
    }
    STAT_GLBL_COLLECT_TIME(scalGetMapped, globalStatPointsEnum::scalGetMapped);

    status = analyzeTensors(launchInfo);
    if (status != synSuccess)
    {
        return status;
    }

    status = runSifPreDownload(launchInfo);
    if (status != synSuccess)
    {
        return status;
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief downloadToDev() - Downloads the recipe parts (sections) from mapped memory to HBM-global and arc
 *
 *   Get memory on hbm-global to be used
 *   Get memory on hbm-arc to be used
 *   If needed:
 *      copy recipe to mapped memory
 *      patch
 *   Download from mapped memory to hbm-global + arc
 *
 *   @param  launchInfo
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::downloadToDev(const LaunchInfo& launchInfo)
{
    STAT_GLBL_START(scalMemcpy2Mapped);
    MappedMemorySectionsUtils::memcpyToMapped(
        m_sections,
        m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.recipeSections,
        m_isDsd,
        m_isIH2DRecipe);
    STAT_GLBL_COLLECT_TIME(scalMemcpy2Mapped, globalStatPointsEnum::scalMemcpy2Mapped);

    synStatus status = patch(launchInfo);
    if (status != synSuccess)
    {
        clearMemMgrs();
        return status;
    }

    // Now everything in the Mapped memory, download it
    STAT_GLBL_START(scalPdmaDownload);
    status = pdmaDownload(m_sections);
    if (status != synSuccess)
    {
        return status;
    }
    STAT_GLBL_COLLECT_TIME(scalPdmaDownload, globalStatPointsEnum::scalPdmaDownload);

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief debugCompareRecipeOnDev() - For debug: Reads the recipe from the HBM and compares to given recipe
 *
 *   Note: this is for debug only. The function is trying to not use anything but the given recipe
 *
 *   Steps:
 *   Get the recipe section information from the original recipe
 *   Allocate mapped memory to read to
 *   Read each section from hbm to mapped memory
 *   Compare the data that was read with the original recipe. For patchable section we calculate the sections addresses
 *       from the patching points and compare the section addresses with the expected values
 *
 *   @param  recipeHandle, msg (given by the caller)
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::debugCompareRecipeOnDev(const InternalRecipeHandle* recipeHandle, const std::string msg)
{
    const recipe_t* recipe = recipeHandle->basicRecipeHandle.recipe;
    // the scheduler is stuck on fence so no need to
    // wait for PDMA download completion on host

    // NOTE: I am not using the sections from the other code because this is a check and I don't want to rely on the
    // other code
    LOG_DEBUG(SYN_PROG_DWNLD, "Start checking recipe {:x} {}", TO64(recipe), msg);

    // Get information from the recipe
    struct DebugSections
    {
        uint64_t       size;
        uint64_t       devAddr;
        const uint8_t* recipeAddr;
        std::string    name;
    };

    std::vector<DebugSections> sections(ECB_LIST_FIRST + recipe->arc_jobs_nr * 2);

    // Build DB
    sections[PATCHABLE] = {.size       = recipe->patching_blobs_buffer_size,
                           .devAddr    = getHbmAddr(PATCHABLE),
                           .recipeAddr = (uint8_t*)recipe->patching_blobs_buffer,
                           .name       = "patchable"};

    sections[NON_PATCHABLE] = {.size    = recipe->execution_blobs_buffer_size,
                               .devAddr = getHbmAddr(NON_PATCHABLE),
                               (uint8_t*)recipe->execution_blobs_buffer,
                               .name = "non-patchable"};

    sections[DYNAMIC] = {.size       = recipe->dynamic_blobs_buffer_size,
                         .devAddr    = getHbmAddr(DYNAMIC),
                         .recipeAddr = (uint8_t*)recipe->dynamic_blobs_buffer,
                         .name       = "dynamic"};

    sections[PROGRAM_DATA] = {.size       = recipe->program_data_blobs_size,
                              .devAddr    = getHbmAddr(PROGRAM_DATA),
                              .recipeAddr = (uint8_t*)recipe->program_data_blobs_buffer,
                              .name       = "program-data"};

    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        sections[ECB_LIST_FIRST + 2 * i] = {.size       = recipe->arc_jobs[i].dynamic_ecb.cmds_size,
                                            .devAddr    = getHbmAddr((SectionType)(ECB_LIST_FIRST + 2 * i)),
                                            .recipeAddr = (uint8_t*)recipe->arc_jobs[i].dynamic_ecb.cmds,
                                            .name       = "arc-dynamic" + std::to_string(i)};

        sections[ECB_LIST_FIRST + 2 * i + 1] = {.size       = recipe->arc_jobs[i].static_ecb.cmds_size,
                                                .devAddr    = getHbmAddr((SectionType)(ECB_LIST_FIRST + 2 * i + 1)),
                                                .recipeAddr = (uint8_t*)recipe->arc_jobs[i].static_ecb.cmds,
                                                .name       = "arc-static" + std::to_string(i)};
    }

    // Calculate total size
    uint64_t neededSize = 0;
    for (const auto& single : sections)
    {
        neededSize += single.size;
        LOG_TRACE(SYN_API, "needed size {:x} added for {} {:x}", neededSize, single.name, single.size);
    }

    // Allocate mapped memory
    void* hostAddr;
    uint64_t mappedAddr = 0;

    std::string name {"compareRecipeOnDev"};
    synStatus   status =
        m_devMemoryAlloc.allocateMemory(neededSize, synMemFlags::synMemHost, &hostAddr, false, 0, name, &mappedAddr);

    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_PROG_DWNLD,
                          "Failed to allocate mapped memory for recipe compare status {} size 0x{:x} {}",
                          status,
                          neededSize,
                          getDescription());
        return status;
    }
    memset(hostAddr, 0x55, neededSize);
    internalMemcopyParams memcpyParams;

    // upload from dev memory to mapped memory
    {
        LOG_DEBUG(SYN_PROG_DWNLD, "Going to upload recipe");
        uint64_t offset    = 0;

        for (size_t sectionIter = 0; sectionIter < sections.size(); sectionIter++)
        {
            if (sections[sectionIter].size > 0)
            {
                memcpyParams.push_back({sections[sectionIter].devAddr, mappedAddr + offset, sections[sectionIter].size});
                offset += sections[sectionIter].size;
            }
        }
    }

    // Have to wait for prev to finish. If not, the pdma up might increase the longSo and the
    // user will think his work was done (but only the pdma up was done).
    uint64_t lastHbmGlbl   = m_memMgrs.hbmGlblMemMgr.getLastLongSo();
    uint64_t lastHbmArc    = m_memMgrs.arcHbmMemMgr.getLastLongSo();
    uint64_t lastMaxLongSo = std::max(lastHbmGlbl, lastHbmArc);

    m_longSoHbmBuff.m_targetValue = lastMaxLongSo;
    status = getRxCommandsScalStream()->longSoWaitOnDevice(m_longSoHbmBuff, false);
    if (status != synSuccess)
    {
        return status;
    }

    ScalLongSyncObject longSo(LongSoEmpty);
    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};
    status = getRxCommandsScalStream()->memcopy(ResourceStreamType::SYNAPSE_DMA_UP,
                                                memcpyParams,
                                                false,
                                                true,
                                                m_apiId,
                                                longSo,
                                                0,
                                                memcopySyncInfo);
    if (status != synSuccess)
    {
        return status;
    }
    // wait on host
    const uint64_t timeoutMicroSec = (uint64_t)((int64_t)SCAL_FOREVER);
    status                         = getRxCommandsScalStream()->longSoWait(longSo, timeoutMicroSec, __FUNCTION__);
    if (status != synSuccess)
    {
        return status;
    }

    // compare read recipe with original recipe
    {
        uint64_t offset     = 0;
        uint8_t* hostAddr8p = (uint8_t*)hostAddr;

        for (const auto& single : sections)
        {
            if (single.name != "patchable")
            {
                int res = memcmpSafe(hostAddr8p + offset, single.recipeAddr, single.size);
                if (res != 0)
                {
                    LOG_CRITICAL(SYN_PROG_DWNLD,
                                      "{} recipe {:x} not the same for {} addrs compare {:x} {:x} size {:x}",
                                      msg,
                                      TO64(recipe),
                                      single.name,
                                      TO64(hostAddr8p + offset),
                                      TO64(single.recipeAddr),
                                      single.size);

                    for (int j = 0; j < 8; j++)
                    {
                        LOG_DEBUG(SYN_PROG_DWNLD,
                                       "actual/expected: {:x} val {:x}/{:x} addr {:x}/{:x}",
                                       j,
                                       ((uint64_t*)(hostAddr8p + offset))[j],
                                       ((uint64_t*)single.recipeAddr)[j],
                                       TO64(&((uint64_t*)(hostAddr8p + offset))[j]),
                                       TO64(&((uint64_t*)single.recipeAddr)[j]));
                    }

                    int cnt = 0;
                    for (int j = 0; j < single.size / 8; j++)
                    {
                        if (((uint64_t*)(hostAddr8p + offset))[j] != ((uint64_t*)single.recipeAddr)[j])
                        {
                            LOG_DEBUG(SYN_PROG_DWNLD,
                                           "actual/expected: {:x} val {:x}/{:x} addr {:x}/{:x}",
                                           j,
                                           ((uint64_t*)(hostAddr8p + offset))[j],
                                           ((uint64_t*)single.recipeAddr)[j],
                                           TO64(&((uint64_t*)(hostAddr8p + offset))[j]),
                                           TO64(&((uint64_t*)single.recipeAddr)[j]));

                            if (cnt++ >= 16) break;
                        }
                    }

                    status = synFail;
                }
                else
                {
                    LOG_DEBUG(SYN_PROG_DWNLD, "recipe {:x} checked OK for {}", TO64(recipe), single.name);
                }
            }
            else  // if (!patchable)
            {
                bool res = true;
                if (recipe->patch_points_nr > 0)
                {
                    MemoryMappedAddrVec dummyDc = {{hostAddr8p + offset, 0}};

                    HostAddressPatchingInformation* hostAddrPatchInfo = m_sections.m_hostAddrPatchInfo;
                    const RecipeAddrPatcher&        recipeAddrPatcher =
                        recipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.recipeAddrPatcher;

                    res = recipeAddrPatcher.verifySectionsAddrFromDc(dummyDc,
                                                                     single.size,
                                                                     hostAddrPatchInfo->getSectionsToHostAddressDB(),
                                                                     hostAddrPatchInfo->getSectionsDbSize());
                }

                if (!res)
                {
                    LOG_CRITICAL(SYN_PROG_DWNLD,
                                      "{} recipe {:x} not the same for {}. See previous errors for more info",
                                      msg,
                                      TO64(recipe),
                                      single.name);
                    status = synFail;
                }
                else
                {
                    LOG_DEBUG(SYN_PROG_DWNLD, "recipe {:x} checked OK for {}", TO64(recipe), single.name);
                }
            }

            offset += single.size;
        }
    }
    m_devMemoryAlloc.deallocateMemory(hostAddr, synMemFlags::synMemHost, false);

    return status;
}

/*
 ***************************************************************************************************
 *   @brief getMappedMemoryInfo() - get mapped memory from MappedMemoryMgr to be used. If memory
 *                                  not available, waits for a launch to finish and tries again
 *
 *   @param  None
 *   @return void
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::getMappedMemoryInfo()
{
    bool done = false;

    while (done == false)
    {
        m_memMgrs.mappedMemMgr.getAddrForId(m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal,
                                            m_entryIds,
                                            m_sections);

        if (m_sections.anyBusyInMapped())
        {
            STAT_GLBL_START(CompletionNeedResources);
            const synStatus status = m_rLaunchTracker.waitForCompletionCopy();
            STAT_GLBL_COLLECT_TIME(CompletionNeedResources, globalStatPointsEnum::CompletionNeedResources);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_STREAM, "waitForCompletionCopy failed with status {}", status);
                return status;
            }
        }
        else
        {
            uint64_t statVal = 0;
            if (m_sections.m_inMappedNoPatch == OUT)
            {
                statVal |= 0x1;
            }
            if (m_sections.m_inMappedPatch == OUT)
            {
                statVal |= 0x2;
            }
            STAT_GLBL_COLLECT(statVal, mappedOutBits);
            done = true;
        }
    }  // while (!done)
    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief getMemDownloadParamsCountForPdmaDownload()
 *    get the amount of items for a subsequent mem download operation
 *
 *   @param  sizeDc - size of the data-chunks
 *   @param  offsetMapped - offset in mapped memory (data chunks) to start from
 *   @param  size - how much to copy
 *   @return amount of items for memory download operation
 *
 ***************************************************************************************************
 */
unsigned RecipeLauncher::getMemDownloadParamsCountForPdmaDownload(uint64_t sizeDc, uint64_t offsetMapped, uint64_t size)
{
    uint32_t firstDc = offsetMapped / sizeDc;
    uint32_t lastDc  = (offsetMapped + size - 1) / sizeDc;
    return lastDc - firstDc + 1;
}

/*
 ***************************************************************************************************
 *   @brief getMemDownloadParamsForPdmaDownload()
 *    fills up parameters of pdma download for subsequent mem download operation
 *
 *   @param  hbmAddr
 *   @param  mapped - a vector of the data-chunks
 *   @param  sizeDc - size of the data-chunks
 *   @param  offsetMapped - offset in mapped memory (data chunks) to start from
 *   @param  size - how much to copy
 *   @param  memDownloadParams - memory download parameters
 *
 ***************************************************************************************************
 */
void RecipeLauncher::getMemDownloadParamsForPdmaDownload(uint64_t                   hbmAddr,
                                                         const MemoryMappedAddrVec& mapped,
                                                         uint64_t                   sizeDc,
                                                         uint64_t                   offsetMapped,
                                                         uint64_t                   size,
                                                         internalMemcopyParams&     memDownloadParams)
{
    uint32_t firstDc = offsetMapped / sizeDc;
    uint32_t lastDc  = (offsetMapped + size - 1) / sizeDc;

    memDownloadParams.reserve(memDownloadParams.size() + lastDc - firstDc + 1);

    uint64_t copied = 0;
    for (uint32_t i = 0, dc = firstDc; dc <= lastDc; dc++, i++)
    {
        uint32_t offsetStart = (dc == firstDc) ? (offsetMapped % sizeDc) : 0;
        uint32_t offsetEnd   = (dc == lastDc) ? (offsetMapped + size) % sizeDc : sizeDc;

        if (offsetEnd == 0) offsetEnd = sizeDc;

        uint64_t currentSize = offsetEnd - offsetStart;

        memDownloadParams.push_back(
            {.src  = mapped[dc].devAddr + offsetStart, /* sections.sections.m_nonPatchableMappedAddr.devAddr */
             .dst  = hbmAddr + copied,
             .size = currentSize /* sections.m_gaudiXrecipeInfo.m_glbHbmSizeNoPatch    */});

        LOG_DEBUG(SYN_STREAM,
                       "download to glbl index {} dc {:x} {:x}->{:x} size {:x} addr {:x}",
                       i,
                       dc,
                       mapped[dc].devAddr + offsetStart,
                       hbmAddr + copied,
                       currentSize,
                       TO64(mapped[dc].hostAddr + offsetStart));

        copied += currentSize;
    }
}

/*
 ***************************************************************************************************
 *   @brief scalMemDownload() - Downloads memory data from memDownloadParams to device
 *                              it's using getTxCommandsScalStream()->scalMemcopy
 *                              logs the total amount of downloaded memory
 *
 *   @param  memDownloadParams parameters of memory areas to download
 *   @param  send - do a submit
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::scalMemDownload(internalMemcopyParams const& memcopyParams)
{
    ScalLongSyncObject longSo(LongSoEmpty);

    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {
        .m_pdmaSyncMechanism     = ScalStreamCopyInterface::PDMA_TX_SYNC_FENCE_ONLY,
        .m_workCompletionAddress = m_computeWorkCompletionAddress,
        .m_workCompletionValue   = m_computeWorkCompletionValue};

    synStatus status = getTxCommandsScalStream()->memcopy(ResourceStreamType::SYNAPSE_DMA_DOWN,
                                                          memcopyParams,
                                                          false, /* isUserRequest */
                                                          true,  /* send */
                                                          m_apiId,
                                                          longSo,
                                                          0, /* overrideMemsetVal */
                                                          memcopySyncInfo);
    if (unlikely(LOG_LEVEL_AT_LEAST_DEBUG(SYN_STREAM)))
    {
        uint64_t size = 0;
        for (auto const& copyParam : memcopyParams)
        {
            size += copyParam.size;
        }
        LOG_DEBUG(SYN_STREAM, "Download (memcopy size {}) status {}", size, status);
    }

    status = getTxCommandsScalStream()->addBarrierOrEmptyPdma(m_longSoCopy);
    if (status != synSuccess)
    {
        return status;
    }

    LOG_DEBUG(SYN_STREAM, "{}", getDescription());

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief pdmaDownload() - Creates a pdma requests to copy a recipe section from mapped memory.
 *                           It copies:
 *                           non-patchable to glbl-hbm
 *                           non-patchable to arc-hbm
 *                           patchable to glb-hbm
 *   It requests an un-fence on the last request
 *   @param  rSections - recipe sections information
 *   @param  send - do a submit
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::pdmaDownload(const MemorySectionsScal& rSections)
{
    unsigned  memDwnldParamsCount = 0;

    const RecipeStaticInfoScal& rRecipeStaticInfoScal =
        m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal;

    if (m_isIH2DRecipe)
    {
        // NON-PATCHABLE
        memDwnldParamsCount =
            getMemDownloadParamsCountForPdmaDownload(rSections.m_nonPatchableDcSize,
                                                     rRecipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetMapped,
                                                     rRecipeStaticInfoScal.recipeSections[NON_PATCHABLE].size);
        // PRG_DATA
        memDwnldParamsCount +=
            getMemDownloadParamsCountForPdmaDownload(rSections.m_patchableDcSize,
                                                     rRecipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetMapped,
                                                     rRecipeStaticInfoScal.recipeSections[PROGRAM_DATA].size);
    }
    else
    {
        // PRG_DATA + NON_PATCHABLE
        memDwnldParamsCount =
            getMemDownloadParamsCountForPdmaDownload(rSections.m_nonPatchableDcSize,
                                                     0,
                                                     rRecipeStaticInfoScal.m_glbHbmSizeNoPatch);
    }
    if (m_isDsd)
    {
        memDwnldParamsCount +=
            getMemDownloadParamsCountForPdmaDownload(rSections.m_patchableDcSize,
                                                     rRecipeStaticInfoScal.recipeSections[DYNAMIC].offsetMapped,
                                                     rRecipeStaticInfoScal.recipeSections[DYNAMIC].size);

        memDwnldParamsCount +=
            getMemDownloadParamsCountForPdmaDownload(rSections.m_nonPatchableDcSize,
                                                     rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST].offsetMapped,
                                                     rRecipeStaticInfoScal.m_ecbListsTotalSize);
    }
    else
    {
        // DYNAMIC (if any) + ECB
        memDwnldParamsCount +=
            getMemDownloadParamsCountForPdmaDownload(rSections.m_nonPatchableDcSize,
                                                     rRecipeStaticInfoScal.recipeSections[FIRST_IN_ARC].offsetMapped,
                                                     rRecipeStaticInfoScal.m_arcHbmSize);
    }

    // if there's no patchable dc, signal since it's the last cb
    const bool hasPatchable = (rRecipeStaticInfoScal.recipeSections[PATCHABLE].size != 0);
    if (hasPatchable)
    {
        memDwnldParamsCount +=
            getMemDownloadParamsCountForPdmaDownload(rSections.m_patchableDcSize,
                                                     rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped,
                                                     rRecipeStaticInfoScal.recipeSections[PATCHABLE].size);
    }

    internalMemcopyParams memcopyParams;
    memcopyParams.reserve(memDwnldParamsCount);

    // NON-PATCHABLE
    if (m_isIH2DRecipe)
    {
        getMemDownloadParamsForPdmaDownload(rSections.m_glbHbmAddr +
                                                rRecipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetHbm,
                                            rSections.m_nonPatchableMappedAddr,
                                            rSections.m_nonPatchableDcSize,
                                            rRecipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetMapped,
                                            rRecipeStaticInfoScal.recipeSections[NON_PATCHABLE].size,
                                            memcopyParams);
    }
    // PRG_DATA + NON-PATCHABLE
    else
    {
        getMemDownloadParamsForPdmaDownload(rSections.m_glbHbmAddr,
                                            rSections.m_nonPatchableMappedAddr,
                                            rSections.m_nonPatchableDcSize,
                                            0,
                                            rRecipeStaticInfoScal.m_glbHbmSizeNoPatch,
                                            memcopyParams);
    }

    // DYNAMIC + ECB
    if (m_isDsd)
    {
        getMemDownloadParamsForPdmaDownload(rSections.m_arcHbmAddr,
                                            rSections.m_patchableMappedAddr,
                                            rSections.m_patchableDcSize,
                                            rRecipeStaticInfoScal.recipeSections[DYNAMIC].offsetMapped,
                                            rRecipeStaticInfoScal.recipeSections[DYNAMIC].size,
                                            memcopyParams);
        getMemDownloadParamsForPdmaDownload(rSections.m_arcHbmAddr +
                                                rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST].offsetHbm,
                                            rSections.m_nonPatchableMappedAddr,
                                            rSections.m_nonPatchableDcSize,
                                            rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST].offsetMapped,
                                            rRecipeStaticInfoScal.m_ecbListsTotalSize,
                                            memcopyParams);
    }
    else
    {
        getMemDownloadParamsForPdmaDownload(rSections.m_arcHbmAddr,
                                            rSections.m_nonPatchableMappedAddr,
                                            rSections.m_nonPatchableDcSize,
                                            rRecipeStaticInfoScal.recipeSections[FIRST_IN_ARC].offsetMapped,
                                            rRecipeStaticInfoScal.m_arcHbmSize,
                                            memcopyParams);
    }

    // PRG_DATA
    if (m_isIH2DRecipe)
    {
        getMemDownloadParamsForPdmaDownload(rSections.m_glbHbmAddr,
                                            rSections.m_patchableMappedAddr,
                                            rSections.m_patchableDcSize,
                                            rRecipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetMapped,
                                            rRecipeStaticInfoScal.recipeSections[PROGRAM_DATA].size,
                                            memcopyParams);
    }

    if (hasPatchable)
    {
        // PATCHABLE
        if (m_isDsd)
        {
            if (m_isIH2DRecipe)
            {
                getMemDownloadParamsForPdmaDownload(rSections.m_glbHbmAddr +
                                                        rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetHbm,
                                                    rSections.m_patchableMappedAddr,
                                                    rSections.m_patchableDcSize,
                                                    rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped,
                                                    rRecipeStaticInfoScal.recipeSections[PATCHABLE].size,
                                                    memcopyParams);
            }
            else
            {
                getMemDownloadParamsForPdmaDownload(rSections.m_glbHbmAddr +
                                                        rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST].offsetMapped,
                                                    rSections.m_patchableMappedAddr,
                                                    rSections.m_patchableDcSize,
                                                    rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped,
                                                    rRecipeStaticInfoScal.recipeSections[PATCHABLE].size,
                                                    memcopyParams);
            }
        }
        else
        {
            getMemDownloadParamsForPdmaDownload(rSections.m_glbHbmAddr +
                                                    rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetHbm,
                                                rSections.m_patchableMappedAddr,
                                                rSections.m_patchableDcSize,
                                                rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped,
                                                rRecipeStaticInfoScal.recipeSections[PATCHABLE].size,
                                                memcopyParams);
        }
    }

    synStatus status = synSuccess;
    if (memcopyParams.size() > 0)
    {
        status = scalMemDownload(memcopyParams);
    }
    return status;
}

/*
 ***************************************************************************************************
 *   @brief pdmaDownload() - Creates a pdma requests to copy a recipe section from mapped memory.
 *                           It is using pdmaDownloadDc() utility for that.
 *                           It copies:
 *                           non-patchable to glbl-hbm
 *                           non-patchable to arc-hbm
 *                           patchable to glb-hbm
 *   It requests an un-fence on the last request
 *   @param  rSections - recipe sections information
 *   @param  send - do submit
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::patch(const LaunchInfo& launchInfo)
{
    const recipe_t* recipe = launchInfo.pRecipeHandle->basicRecipeHandle.recipe;
    if (recipe->patch_points_nr == 0)
    {
        LOG_DEBUG(SYN_STREAM, "Recipe has no patch points");
        return synSuccess;
    }

    if (m_isDsd)
    {
        STAT_GLBL_START(patchingDsdSmf);
        std::vector<uint64_t> dataChunksHostAddresses(m_sections.m_patchableMappedAddr.size());

        int idx = 0;
        for (auto addr : m_sections.m_patchableMappedAddr)
        {
            dataChunksHostAddresses[idx] = (uint64_t)addr.hostAddr;
            idx++;
        }

        bool res = m_pDynamicRecipeProcessor->runSmfOnAllNodes(dataChunksHostAddresses);
        if (!res)
        {
            LOG_ERR(SYN_STREAM, "Failed to perform DSD SMF patching");
            m_pDynamicRecipeProcessor->patchAbort();
            return synFailedDynamicPatching;
        }
        STAT_GLBL_COLLECT_TIME(patchingDsdSmf, globalStatPointsEnum::DsdSmf);
    }

    STAT_GLBL_START(patchingAll);
    patching::HostAddressPatchingInformation* hostAddrPatchInfo = m_sections.m_hostAddrPatchInfo;
    const uint64_t* sectionAddrDb = hostAddrPatchInfo->getSectionsToHostAddressDB();

    const RecipeAddrPatcher& recipeAddrPatcher = m_isDsd ?
                                                 m_pDynamicRecipeProcessor->getRecipeAddrPatcher() :
                                                 launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.recipeAddrPatcher;

    recipeAddrPatcher.patchAll(sectionAddrDb, m_sections.m_patchableMappedAddr);
    STAT_GLBL_COLLECT_TIME(patchingAll, globalStatPointsEnum::patchingAll);

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief handleBaseAddresses() sends all bases addresses as part of the launch
 *
 *   @param  LaunchAddr - addresses where all the parts of the recipe are
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::addBaseAddresses(const EngineGrpArr& engineGrpArr)
{
    bool isNopKernelRequired = m_pRecipeHandle->basicRecipeHandle.recipe->valid_nop_kernel;

    const uint16_t maxNumOfRecipeBaseElements = 4;
    uint16_t       numofRecipeBaseElements =
        isNopKernelRequired ? maxNumOfRecipeBaseElements : (maxNumOfRecipeBaseElements - 1);

    const uint64_t tpcNopKernelBaseAddress =
        getHbmAddr(PROGRAM_DATA) + m_pRecipeHandle->basicRecipeHandle.recipe->nop_kernel_offset;

    uint64_t recipeBaseAddresses[maxNumOfRecipeBaseElements] = {getHbmAddr(PATCHABLE),
                                                                getHbmAddr(NON_PATCHABLE),
                                                                getHbmAddr(DYNAMIC),
                                                                tpcNopKernelBaseAddress};

    uint16_t recipeBaseIndices[maxNumOfRecipeBaseElements] = {PATCHING_ADDR_BASE,
                                                              EXECUTE_ADDR_BASE,
                                                              DYNAMIC_ADDR_BASE,
                                                              NOP_KERNEL_ADDR_BASE};

    synStatus status = getComputeScalStream()->addUpdateRecipeBaseAddresses(engineGrpArr,
                                                                            numofRecipeBaseElements,
                                                                            recipeBaseAddresses,
                                                                            recipeBaseIndices,
                                                                            false);

    return status;
}

/*
 ***************************************************************************************************
 *   @brief handleEcbList() sends all the dcb lists as part of the launch
 *
 *   @param  LaunchAddr - addresses where all the parts of the recipe are
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus RecipeLauncher::addEcbListWithBarrierAndSend(uint64_t nbExtTensors)
{
    const recipe_t*             recipe    = m_pRecipeHandle->basicRecipeHandle.recipe;
    const MemorySectionsScal&   rSections = m_sections;
    const RecipeStaticInfoScal& rRecipeStaticInfoScal =
        m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal;
    McidInfo mcidInfo{ .mcidDegradeCount = recipe->max_used_mcid_degrade, .mcidDiscardCount = recipe->max_used_mcid_discard };

    synStatus status = getComputeScalStream()->addAllocBarrierForLaunch(true /*allocSoSet*/, true/*relSoSet*/, mcidInfo, false);
    if (status != synSuccess)
    {
        return status;
    }

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        const uint32_t staticAddrCore =
            rSections.m_arcHbmCoreAddr + rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST + 2 * i + 1].offsetHbm;
        const uint32_t staticSize = rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST + 2 * i + 1].size;
        const uint32_t staticSinglePhysicalEngineOffset = recipe->arc_jobs[i].static_ecb.cmds_eng_offset;

        const uint32_t dynamicAddrCore =
            rSections.m_arcHbmCoreAddr + rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST + 2 * i].offsetHbm;
        const uint32_t dynamicSize = rRecipeStaticInfoScal.recipeSections[ECB_LIST_FIRST + 2 * i].size;

        LOG_DEBUG(
            SYN_STREAM,
            "{} engine:{} staticAddrCore {:x} dynamicAddrCore {:x} staticSinglePhysicalEngineOffset {} staticSize {} "
            "dynamicSize {}",
            HLLOG_FUNC,
            recipe->arc_jobs[i].logical_engine_id,
            staticAddrCore,
            dynamicAddrCore,
            staticSinglePhysicalEngineOffset,
            staticSize,
            dynamicSize);

        bool shouldUseGcNopKernel =
            (recipe->valid_nop_kernel) && (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC);

        status = getComputeScalStream()->addDispatchComputeEcbList(recipe->arc_jobs[i].logical_engine_id,
                                                                   staticAddrCore,
                                                                   staticSize,
                                                                   staticSinglePhysicalEngineOffset,
                                                                   dynamicAddrCore,
                                                                   dynamicSize,
                                                                   shouldUseGcNopKernel,
                                                                   false);
        if (status != synSuccess)
        {
            return status;
        }
    }

    status = getComputeScalStream()->addDispatchBarrier(ResourceStreamType::COMPUTE,
                                                        true,
                                                        true,
                                                        m_longSoCompute,
                                                        nbExtTensors);
    if (status != synSuccess)
    {
        return status;
    }

    LOG_DEBUG(SYN_STREAM, "{}", getDescription());

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief getHbmAddr() utility to get hte hbm addr from the section tyep
 *
 *   @param  sectionNum - what section
 *   @return hbm address (glbl or arc)
 *
 ***************************************************************************************************
 */
uint64_t RecipeLauncher::getHbmAddr(SectionType sectionNum) const
{
    const RecipeSingleSectionVec& rSections =
        m_pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal.recipeSections;

    switch ((int)sectionNum)
    {
        case PATCHABLE:
            return m_sections.m_glbHbmAddr + rSections[PATCHABLE].offsetHbm;
        case PROGRAM_DATA:
            return m_sections.m_glbHbmAddr + rSections[PROGRAM_DATA].offsetHbm;
        case NON_PATCHABLE:
            return m_sections.m_glbHbmAddr + rSections[NON_PATCHABLE].offsetHbm;
        case DYNAMIC:
            return m_sections.m_arcHbmAddr + rSections[DYNAMIC].offsetHbm;
    }

    if (sectionNum >= rSections.size())
    {
        LOG_ERR(SYN_STREAM, "Bad sectionNum {:x} >= {:x}", TO64(sectionNum), rSections.size());
        return 0;
    }

    return m_sections.m_arcHbmAddr + rSections[sectionNum].offsetHbm;
}

/*
 ***************************************************************************************************
 *   @brief handleKernelsPrintf() - Copies the kernelsPrintf data from the program data to the workspace
 *   This function is called immediately after the launch
 *   1) Waits on the host for the launch to finish
 *   2) Copies the print from the program data to the workspace
 *   3) Waits for the copy to finish
 *
 *   @param
 *   @return void
 *
 ***************************************************************************************************
 */
void RecipeLauncher::handleKernelsPrintf(const LaunchInfo& launchInfo)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    const recipe_t* recipe = launchInfo.pRecipeHandle->basicRecipeHandle.recipe;

    synStatus status;
    // if prev cmd was wait, then add a barrier/pdma.
    // (in here - no need to lock m_userOpLock in QueueBaseScal, because the caller already locked it)
    if (getComputeScalStream()->prevCmdIsWait())
    {
        ScalLongSyncObject longSo;
        status = getComputeScalStream()->addBarrierOrEmptyPdma(longSo);
        if (status != synSuccess)
        {
            LOG_ERR_T(SYN_STREAM, "Failed to add a barrier after previous stream cmd was wait");
            return;
        }
    }
    // for launch to finish
    status = getComputeScalStream()->longSoWaitForLast(true, (uint64_t)SCAL_FOREVER, __FUNCTION__);
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_STREAM, "Failed to wait for kernelsPrintf");
        return;
    }

    // Copy to ws
    uint64_t wsPrintAddr =
        launchInfo.workspaceAddress + RecipeUtils::getKernelPrintOffsetInWs(launchInfo.pRecipeHandle);
    uint64_t prgDataAddr = getHbmAddr(PROGRAM_DATA);

    uint32_t  printAddrNr    = recipe->debug_profiler_info.printf_addr_nr;
    uint64_t* printAddrArr   = recipe->debug_profiler_info.printf_addr;
    uint64_t  singleBuffSize = GCFG_TPC_PRINTF_TENSOR_SIZE.value();

    internalMemcopyParams memcopyParams;

    for (int i = 0; i < printAddrNr; i++)
    {
        internalMemcopyParamEntry entry = {.src  = prgDataAddr + maskOutMemoryID(printAddrArr[i]),
                                           .dst  = wsPrintAddr + (i * singleBuffSize),
                                           .size = singleBuffSize};
        memcopyParams.push_back(entry);
    }

    ScalLongSyncObject longSo;
    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};
    status = getDev2DevCommandsScalStream()->memcopy(ResourceStreamType::SYNAPSE_DEV_TO_DEV,
                                                     memcopyParams,
                                                     true,
                                                     true,
                                                     m_apiId,
                                                     longSo,
                                                     false,
                                                     memcopySyncInfo);
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_STREAM, "Failed to memcopy prgData->ws");
        return;
    }

    status = getDev2DevCommandsScalStream()->longSoWaitForLast(true, (uint64_t)SCAL_FOREVER, __FUNCTION__);
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_STREAM, "Failed to wait for kernelsPrintf copy prgData->ws");
        return;
    }

    return;
}

void RecipeLauncher::clearMemMgrs()
{
    m_memMgrs.hbmGlblMemMgr.unuseIdOnError();
    m_memMgrs.arcHbmMemMgr.unuseIdOnError();
    m_memMgrs.mappedMemMgr.unuseIdOnError(m_entryIds);
}

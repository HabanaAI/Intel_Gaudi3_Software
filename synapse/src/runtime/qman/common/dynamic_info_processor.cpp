#include "dynamic_info_processor.hpp"

#include "command_submission.hpp"
#include "command_submission_builder.hpp"
#include "command_submission_data_chunks.hpp"
#include "data_chunk/data_chunk.hpp"
#include "defenders.h"
#include "debug_define.hpp"
#include "event_triggered_logger.hpp"
#include "gaudi/gaudi.h"
#include "habana_global_conf_runtime.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "recipe_static_information.hpp"
#include "rt_exceptions.h"
#include "runtime/common/queues/basic_queue_info.hpp"
#include "arb_master_helper.hpp"
#include "stream_master_helper.hpp"
#include "stream_dc_downloader.hpp"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "types.h"
#include "utils.h"
#include "recipe.h"
#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "runtime/qman/common/submit_command_buffers_interface.hpp"
#include "profiler_api.hpp"

#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"

#include "runtime/common/common_types.hpp"

#include "runtime/common/device/device_mem_alloc.hpp"
#include "runtime/qman/common/device_recipe_addresses_generator_interface.hpp"

#include "runtime/common/recipe/patching/define.hpp"
#include "runtime/common/recipe/patching/host_address_patcher.hpp"
#include "runtime/common/recipe/recipe_patch_processor.hpp"

#include <limits.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <list>
#include <memory>
#include <unordered_map>

using namespace patching;

void _getCommandsDataChunksHandles(std::vector<uint64_t>& commandsDcsHandles, const DataChunksDB& commandsDataChunks)
{
    commandsDcsHandles.resize(commandsDataChunks.size());

    uint32_t chunkIndex = 0;
    for (auto dataChunk : commandsDataChunks)
    {
        uint64_t dcAddress             = dataChunk->getHandle();
        commandsDcsHandles[chunkIndex] = dcAddress;

        chunkIndex++;
    }
}

DynamicInfoProcessor::DynamicInfoProcessor(synDeviceType                      deviceType,
                                           const BasicQueueInfo&              rBasicQueueInfo,
                                           uint32_t                           physicalQueueOffset,
                                           uint32_t                           amountOfEnginesInArbGroup,
                                           uint64_t                           recipeId,
                                           const basicRecipeInfo&             rBasicRecipeInfo,
                                           const DeviceAgnosticRecipeInfo&    rDeviceAgnosticRecipeInfo,
                                           const RecipeStaticInfo&            rRecipeInfo,
                                           const ArbMasterHelper&             rArbMasterHelper,
                                           const StreamMasterHelperInterface& rStreamMasterHelper,
                                           const StreamDcDownloader&          rStreamDcDownloader,
                                           SubmitCommandBuffersInterface&     rSubmitter,
                                           DevMemoryAllocInterface&           rDevMemAlloc)
: m_deviceType(deviceType),
  m_rBasicQueueInfo(rBasicQueueInfo),
  m_physicalQueueOffset(physicalQueueOffset),
  m_amountOfEnginesInArbGroup(amountOfEnginesInArbGroup),
  m_rRecipe(*rBasicRecipeInfo.recipe),
  m_recipeId(recipeId),
  m_rBasicRecipeInfo(rBasicRecipeInfo),
  m_rDeviceAgnosticRecipeInfo(rDeviceAgnosticRecipeInfo),
  m_rRecipeInfo(rRecipeInfo),
  m_lastPatchedWorkspaceAddress(INITIAL_WORKSPACE_ADDRESS),
  m_lastDownloadWorkspaceAddress(INITIAL_WORKSPACE_ADDRESS),
  m_lastPatchedProgramDataHandle(INVALID_HANDLE_VALUE),
  m_lastSobjAddress(SIG_HANDLE_INVALID),
  m_rArbMasterHelper(rArbMasterHelper),
  m_rStreamMasterHelper(rStreamMasterHelper),
  m_rStreamDcDownloader(rStreamDcDownloader),
  m_currentExecutionHandle(CS_DC_FIRST_NON_RESERVED_EXECUTION_HANDLE),
  m_lastProgramDataHandle(INVALID_HANDLE_VALUE),
  m_lastProgramCodeHandle(INVALID_HANDLE_VALUE),
  m_dynamicRecipe(nullptr),
  m_qmansDef(nullptr),
  m_staticBlobsAddressesInWs {{}, INVALID_HANDLE_VALUE},
  m_rSubmitter(rSubmitter),
  m_rDevMemAlloc(rDevMemAlloc)
{
    switch (m_deviceType)
    {
        case synDeviceGaudi:
        {
            m_qmansDef = gaudi::QmansDefinition::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type");
        }
    }

    m_dataChunksHostAddresses.reserve(
        m_rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.getMaxProgramCommandsChunksAmount());

    LOG_DEBUG(
        SYN_STREAM,
        "Creating stream Info processor and Adding patch scheme PATCHING_TYPE_HOST_ADDRESSES to patch manager 0x{:x}",
        (uint64_t)this);

    if (RecipeUtils::isDsd(m_rBasicRecipeInfo))
    {
        const DataChunkSmPatchPointsInfo* pSmPatchPointsDataChunksInfo = m_rRecipeInfo.getSmPatchingPointsDcLocation();

        const DataChunkPatchPointsInfo* pPatchPointsDataChunksInfo =
            m_rRecipeInfo.getPatchingPointsDcLocation(EXECUTION_STAGE_ENQUEUE, PP_TYPE_ID_ALL);
        data_chunk_patch_point_t* dataChunksPatchPoints = (pPatchPointsDataChunksInfo != nullptr) ?
                                                          pPatchPointsDataChunksInfo->m_dataChunkPatchPoints : nullptr;

        m_dynamicRecipe =
            std::unique_ptr<DynamicRecipe>(new DynamicRecipe(m_rBasicRecipeInfo,
                                                             m_rDeviceAgnosticRecipeInfo,
                                                             pSmPatchPointsDataChunksInfo,
                                                             dataChunksPatchPoints));
    }
}

DynamicInfoProcessor::~DynamicInfoProcessor() {}

synStatus DynamicInfoProcessor::enqueue(const synLaunchTensorInfoExt* launchTensorsInfo,
                                        uint32_t                      launchTensorsAmount,
                                        CommandSubmissionDataChunks*& pCsDataChunks,
                                        uint64_t                      scratchPadAddress,
                                        uint64_t                      programDataDeviceAddress,
                                        uint64_t                      programCodeHandle,
                                        uint64_t                      programDataHandle,
                                        bool                          programCodeInCache,
                                        bool                          programDataInCache,
                                        uint64_t                      assertAsyncMappedAddress,
                                        uint32_t                      flags,
                                        uint64_t&                     csHandle,
                                        eAnalyzeValidateStatus        analyzeValidateStatus,
                                        uint32_t                      sigHandleId,
                                        uint32_t                      sigHandleSobjBaseAddressOffset,
                                        eCsDcProcessorStatus&         csDcProcessingStatus)
{
    synStatus status = _enqueue(launchTensorsInfo,
                                launchTensorsAmount,
                                pCsDataChunks,
                                scratchPadAddress,
                                programDataDeviceAddress,
                                programCodeHandle,
                                programDataHandle,
                                programCodeInCache,
                                programDataInCache,
                                assertAsyncMappedAddress,
                                flags,
                                sigHandleId,
                                sigHandleSobjBaseAddressOffset,
                                csHandle,
                                csDcProcessingStatus,
                                analyzeValidateStatus);

    if ((analyzeValidateStatus & ANALYZE_VALIDATE_STATUS_DO_VALIDATE) > 0)
    {
        synStatus validationStatus = _validateNewTensors(launchTensorsInfo,
                                                         launchTensorsAmount,
                                                         flags,
                                                         scratchPadAddress,
                                                         programCodeHandle,
                                                         programDataDeviceAddress,
                                                         programCodeInCache,
                                                         programDataInCache);
        analyzeValidateStatus      = ANALYZE_VALIDATE_STATUS_NOT_REQUIRED;
        if (status == synSuccess)
        {
            return validationStatus;
        }
    }

    return status;
}

synStatus DynamicInfoProcessor::_enqueue(const synLaunchTensorInfoExt* launchTensorsInfo,
                                         uint32_t                      launchTensorsAmount,
                                         CommandSubmissionDataChunks*& pCsDataChunks,
                                         uint64_t                      scratchPadAddress,
                                         uint64_t                      programDataDeviceAddress,
                                         uint64_t                      programCodeHandle,
                                         uint64_t                      programDataHandle,
                                         bool                          programCodeInCache,
                                         bool                          programDataInCache,
                                         uint64_t                      assertAsyncMappedAddress,
                                         uint32_t                      flags,
                                         uint32_t                      sigHandleId,
                                         uint32_t                      sigHandleSobjBaseAddressOffset,
                                         uint64_t&                     csHandle,
                                         eCsDcProcessorStatus&         csDcProcessingStatus,
                                         eAnalyzeValidateStatus&       analyzeValidateStatus)
{
    STAT_GLBL_START(dipRecipeEnqueue);

    LOG_TRACE(SYN_STREAM,
              "{}: launchTensorsAmount {} last-Execution-WS-Address 0x{:x} last-Download-WS-Address 0x{:x}",
              HLLOG_FUNC,
              launchTensorsAmount,
              m_lastPatchedWorkspaceAddress,
              m_lastDownloadWorkspaceAddress);

    LOG_TRACE(SYN_STREAM,
              "{}: last-patched Program-Data-Handle 0x{:x} last-download Program-Data-Handle 0x{:x}",
              HLLOG_FUNC,
              m_lastPatchedProgramDataHandle,
              programDataHandle);

    bool               operationStatus    = true;
    CommandSubmission* pCommandSubmission = nullptr;
    eCsDcExecutionType csDcExecutionType;

    CHECK_POINTER(SYN_STREAM, pCsDataChunks, "CS Data-Chunks", synFail);
    csDcExecutionType = pCsDataChunks->getCsDcExecutionType();
    LOG_DEBUG(SYN_STREAM, "{}: csDcExecutionType {}", HLLOG_FUNC, csDcExecutionType);
    pCommandSubmission = pCsDataChunks->getCommandSubmissionInstance();
    if (pCommandSubmission != nullptr)
    {
        pCommandSubmission->setEncapsHandleId(sigHandleId);
    }

    STAT_GLBL_START(dipSendCommandSubmission);

    // Re-usage of old command-submission (Gaudi-Demo or stored CS-DC)
    if (csDcExecutionType == CS_DC_EXECUTION_TYPE_CS_READY)
    {
        LOG_DEBUG(SYN_STREAM, "{}: pCommandSubmission exists", HLLOG_FUNC);
        HB_ASSERT(pCommandSubmission != nullptr, "Command Submission is null-pointer");

        if ((g_preSubmissionCsInspection) && (csDcExecutionType == CS_DC_EXECUTION_TYPE_CS_READY))
        {
            checkForCsUndefinedOpcode(pCsDataChunks, pCommandSubmission, m_amountOfEnginesInArbGroup);
        }

        operationStatus = _sendCommandSubmission(pCommandSubmission, csHandle, nullptr, 0);
        if (!operationStatus)
        {
            LOG_ERR(SYN_STREAM, "{}: cannot send-job", HLLOG_FUNC);
            _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                    csDcExecutionType,
                                                    EXECUTION_STAGE_ENQUEUE,
                                                    false,
                                                    csDcProcessingStatus);
            delete pCommandSubmission;

            return synFailedToSubmitWorkload;
        }

        LOG_DEBUG(SYN_STREAM, "setWaitForEventHandle csHandle 0x{:x}", csHandle);
        pCsDataChunks->addWaitForEventHandle(csHandle);

        _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                csDcExecutionType,
                                                EXECUTION_STAGE_ENQUEUE,
                                                true,
                                                csDcProcessingStatus);

        STAT_GLBL_COLLECT_TIME(dipSendCommandSubmission, globalStatPointsEnum::dipSendCommandSubmission);
        STAT_GLBL_COLLECT_TIME(dipRecipeEnqueue, globalStatPointsEnum::dipRecipeEnqueue);
        return synSuccess;
    }

    STAT_GLBL_COLLECT_TIME(dipSendCommandSubmission, globalStatPointsEnum::dipSendCommandSubmission);

    const bool isPatchingRequired = _isPatchingRequired(*pCsDataChunks,
                                                        launchTensorsAmount,
                                                        scratchPadAddress,
                                                        programDataHandle,
                                                        sigHandleSobjBaseAddressOffset,
                                                        EXECUTION_STAGE_ENQUEUE);

    HostAddressPatchingInformation* pPatchingInformation = nullptr;

    if (isPatchingRequired)
    {
        if ((!getHostAddressPatchingInfo(pCsDataChunks, pPatchingInformation)) || (pPatchingInformation == nullptr))
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to get patching-information", HLLOG_FUNC);

            _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                    csDcExecutionType,
                                                    EXECUTION_STAGE_ENQUEUE,
                                                    false,
                                                    csDcProcessingStatus);
            return synFail;
        }
    }

    const bool isDsd = RecipeUtils::isDsd(m_rBasicRecipeInfo);

    STAT_GLBL_START(dipAnalyzeNewTensors);

    bool hasNewAddress = false;
    if ((isPatchingRequired) && (launchTensorsAmount != 0))
    {
        if ((analyzeValidateStatus & ANALYZE_VALIDATE_STATUS_DO_ANALYZE) > 0)
        {
            synStatus status = _analyzeNewTensors(pCsDataChunks,
                                                  launchTensorsInfo,
                                                  launchTensorsAmount,
                                                  scratchPadAddress,
                                                  programCodeHandle,
                                                  programDataDeviceAddress,
                                                  csDcExecutionType,
                                                  EXECUTION_STAGE_ENQUEUE,
                                                  hasNewAddress,
                                                  programCodeInCache,
                                                  programDataInCache,
                                                  csDcProcessingStatus,
                                                  sigHandleSobjBaseAddressOffset,
                                                  assertAsyncMappedAddress,
                                                  flags);

            analyzeValidateStatus = ANALYZE_VALIDATE_STATUS_DO_VALIDATE;

            if (status != synSuccess)
            {
                return status;
            }
        }
    }

    STAT_GLBL_COLLECT_TIME(dipAnalyzeNewTensors, globalStatPointsEnum::dipAnalyzeNewTensors);

    STAT_GLBL_START(dipGetCommandsBufferDataChunks);
    DataChunksDB          commandsDataChunks = pCsDataChunks->getCommandsBufferDataChunks();
    std::vector<uint64_t> commandsDcsHandles;
    STAT_GLBL_COLLECT_TIME(dipGetCommandsBufferDataChunks, globalStatPointsEnum::dipGetCommandsBufferDataChunks);

    STAT_GLBL_START(dipCopyPatchableBlobs);
    const uint8_t* patchableBlobs          = (uint8_t*)m_rRecipe.patching_blobs_buffer;
    uint64_t       patchingBlobsBufferSize = m_rRecipe.patching_blobs_buffer_size;
    // DSD patching requires a clean start at every launch (copy original patchable blobs to data chunks)
    if ((!pCsDataChunks->isCopiedToDc()) || isDsd)
    {
        if (patchingBlobsBufferSize)
        {
            StreamDcDownloader::downloadPatchableBlobsBuffer(patchableBlobs,
                                                             patchingBlobsBufferSize,
                                                             commandsDataChunks,
                                                             commandsDcsHandles);
        }
        pCsDataChunks->setCopiedToDc();
    }
    else
    {
        _getCommandsDataChunksHandles(commandsDcsHandles, commandsDataChunks);
    }
    STAT_GLBL_COLLECT_TIME(dipCopyPatchableBlobs, globalStatPointsEnum::dipCopyPatchableBlobs);

    STAT_GLBL_START(dipCopyPrgToDc);
    if (csDcExecutionType == CS_DC_EXECUTION_TYPE_NOT_READY)
    {
        DataChunksDB cpDmaDataChunks = pCsDataChunks->getCpDmaDataChunks();

        staticBlobsDeviceAddresses* pStaticBlobsAddresses = nullptr;

        if (!programCodeInCache)
        {
            _storeStaticBlobsDeviceAddresses(programCodeHandle);
            pStaticBlobsAddresses = &(m_staticBlobsAddressesInWs.staticBlobsAddresses);
        }

        if (m_rRecipe.execute_jobs_nr > 0)
        {
            uint64_t dcSizeCommand = 0;

            operationStatus = m_rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.getDcSizeCommand(dcSizeCommand);
            if (!operationStatus)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to retrieve chunk-size", HLLOG_FUNC);
                _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                        csDcExecutionType,
                                                        EXECUTION_STAGE_ENQUEUE,
                                                        false,
                                                        csDcProcessingStatus);
                return synFail;
            }

            HB_ASSERT(dcSizeCommand != 0, "Invalid Chunk size value (zero)");
            HB_ASSERT((dcSizeCommand % 32) == 0, "dcSizeCommand is not a multiplication of 32 bytes");

            m_rStreamDcDownloader.downloadProgramCodeBuffer(m_rArbMasterHelper.getArbMasterQmanId(),
                                                            *pCsDataChunks,
                                                            commandsDcsHandles,
                                                            cpDmaDataChunks,
                                                            dcSizeCommand,
                                                            (&m_rRecipe),
                                                            m_rDeviceAgnosticRecipeInfo,
                                                            m_rRecipeInfo,
                                                            m_rRecipe.blobs,
                                                            EXECUTION_STAGE_ENQUEUE,
                                                            patchableBlobs,
                                                            pStaticBlobsAddresses);
        }
    }

    STAT_GLBL_COLLECT_TIME(dipCopyPrgToDc, globalStatPointsEnum::dipCopyPrgToDc);

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    dataChunksReady(commandsDataChunks, false);
    dataChunksReady(commandsDataChunks);
#endif

    // staged submission data
    PatchPointPointerPerSection currPatchPoint;

    SectionTypesToPatch sectionTypesToPatch;
    bool                isPatchOnDataChunkRequired = (isPatchingRequired) && (hasNewAddress);
    if (isPatchOnDataChunkRequired)
    {
        PROFILER_COLLECT_TIME()

        // Expecting to have no real gain from storing the dataChunksHostAddresses and re-use
        storeDataChunksHostAddresses(*pCsDataChunks);

        if (isDsd)
        {
            PROFILER_COLLECT_TIME()
            STAT_GLBL_START(patch);

            bool res = true;

            if (!RecipeUtils::isIH2DRecipe(&m_rRecipe))
            {
                res = m_dynamicRecipe->runSifOnAllNodes(launchTensorsInfo,
                                                        launchTensorsAmount,
                                                        m_tensorIdx2userIdx,
                                                        0 /* programDataHostAddress - NA*/);
                if (!res)
                {
                    LOG_DSD_ERR("Failed DSD sif");
                    return synFailedDynamicPatching;
                }
            }

            PROFILER_MEASURE_TIME("dynamic patching")
            STAT_GLBL_COLLECT_TIME(patch, globalStatPointsEnum::DsdPatchDynamicRecipe);

            currPatchPoint[PP_TYPE_ID_ALL] = m_dynamicRecipe->getPatchPoints();
            sectionTypesToPatch.insert(PP_TYPE_ID_ALL);
        }
        else
        {
            // check if patching of program data is required and if so - add it for mandatory patching
            _conditionalAddProgramDataSectionForPatching(pCsDataChunks, programDataHandle);
            sectionTypesToPatch = _setupStartingPatchPointsPerSection(pCsDataChunks, currPatchPoint, EXECUTION_STAGE_ENQUEUE);
        }
        // validate HostAddressInDcInformation
        HB_ASSERT(pCsDataChunks->getHostAddressInDcInformation().isInitialized(),
                  "Host-Address patching-info is not initialized");
        if (!pCsDataChunks->getHostAddressInDcInformation().validateAllSectionsAddressSet())
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to execute patching."
                    "Some patch-point IDs are not set",
                    HLLOG_FUNC);
            pPatchingInformation->patchingAbort();
            return synFailedSectionValidation;
        }

        PROFILER_MEASURE_TIME("preparePatchingInfo");
    }

    const std::vector<uint32_t>*   stagesNodes = nullptr;
    const std::vector<StagedInfo>* stagesInfo  = nullptr;
    const std::vector<uint32_t>    dummyNode   = {m_rRecipe.node_nr - 1};

    bool isStagedSubmission = isPatchOnDataChunkRequired      &&
                              (m_deviceType == synDeviceGaudi && GCFG_ENABLE_STAGED_SUBMISSION.value());
    if (isStagedSubmission)
    {
        stagesNodes = &pCsDataChunks->getDataChunksStagesNodes();
        stagesInfo  = &pCsDataChunks->getDataChunksStagesInfo();
        HB_ASSERT(stagesNodes->size() > 0, "dataChunksStagesNodes is not initialized");
    }
    else
    {
        stagesNodes = &dummyNode;
    }

    bool isCSsetInCSDC = false;
    uint32_t firstNodeIndex = 0;

    // stage submission loop
    for (unsigned int stageIdx = 0; stageIdx < (*stagesNodes).size(); ++stageIdx)
    {
        PROFILER_COLLECT_TIME()

        uint32_t stageNodesIndex = (*stagesNodes)[stageIdx];
        if (isPatchOnDataChunkRequired)
        {
            if (isDsd)
            {
                bool res = m_dynamicRecipe->runSmfOnNodes(m_dataChunksHostAddresses, firstNodeIndex, stageNodesIndex + 1);
                if (!res)
                {
                    LOG_DSD_ERR("Failed DSD SMF on nodes  {} - {}", firstNodeIndex, stageNodesIndex);
                    m_dynamicRecipe->patchAbort();

                    return synFailedDynamicPatching;
                }
                firstNodeIndex = stageNodesIndex + 1;
            }

            synStatus status = _patchOnDc(pCsDataChunks,
                                          scratchPadAddress,
                                          csDcExecutionType,
                                          EXECUTION_STAGE_ENQUEUE,
                                          currPatchPoint,
                                          sectionTypesToPatch,
                                          stageNodesIndex,
                                          sigHandleSobjBaseAddressOffset,
                                          csDcProcessingStatus);

            if (status != synSuccess)
            {
                return synFailedStaticPatching;
            }
        }

        csDcExecutionType = CS_DC_EXECUTION_TYPE_CP_DMA_READY;

        bool isCsCreationRequired = (pCommandSubmission == nullptr);
        if (isCsCreationRequired)
        {
            uint64_t internalPqEntriesAmount;
            if (!m_rRecipeInfo.getCpDmaChunksAmount(EXECUTION_STAGE_ENQUEUE,
                                                    internalPqEntriesAmount,
                                                    programCodeInCache))
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to get CP-DMA Data-Chunks amount", HLLOG_FUNC);
                return synFail;
            }

            uint64_t numOfPqsForInternalQueues = internalPqEntriesAmount;
            // Using DC's PrimeQueue entries m_rRecipeInfo->getNumOfExternalQueuesCbs();
            uint64_t numOfPqsForExternalQueues = 0;

            CommandSubmissionBuilder::getInstance()->buildShellCommandSubmission(pCommandSubmission,
                                                                                 numOfPqsForInternalQueues,
                                                                                 numOfPqsForExternalQueues);
            CHECK_POINTER(SYN_STREAM, pCommandSubmission, "Command submission", synFail);

            if (m_deviceType == synDeviceGaudi)
            {
                // Handle external-queues
                // since GC does not (currently) create work for engine id 2 (DMA_0_2),
                // we need to create a fake job to initialize things
                PROFILER_COLLECT_TIME()
                if (GCFG_ENABLE_STAGED_SUBMISSION.value())
                {
                    // fence clear and fence set needs to be separated:
                    //   clear - will be sent on first stage submission - in order to start execution on first
                    //   submission set   - will be sent on last
                    pCommandSubmission->addPrimeQueueEntry(
                        PQ_ENTRY_TYPE_EXTERNAL_EXECUTION,
                        m_qmansDef->getStreamMasterQueueIdForCompute(),
                        m_rStreamMasterHelper.getStreamMasterFenceClearBufferSize(),
                        m_rStreamMasterHelper.getStreamMasterFenceClearBufferHandle());

                    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION,
                                                           m_qmansDef->getStreamMasterQueueIdForCompute(),
                                                           m_rStreamMasterHelper.getStreamMasterFenceBufferSize(),
                                                           m_rStreamMasterHelper.getStreamMasterFenceBufferHandle());
                }
                else
                {
                    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION,
                                                           m_qmansDef->getStreamMasterQueueIdForCompute(),
                                                           m_rStreamMasterHelper.getStreamMasterBufferSize(),
                                                           m_rStreamMasterHelper.getStreamMasterBufferHandle());
                }

                PROFILER_MEASURE_TIME("addPQExternalEntry");
            }

            operationStatus = _createCommandSubmission(*pCommandSubmission,
                                                       *pCsDataChunks,
                                                       numOfPqsForInternalQueues,
                                                       EXECUTION_STAGE_ENQUEUE);
            if (!operationStatus)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to create command-submission", HLLOG_FUNC);
                delete pCommandSubmission;

                return synFailedToInitializeCb;
            }
        }

        // Preferring simplicity over dealing with a case where the DCs are ready, but the CS failed to be created (no
        // real benefit)
        csDcExecutionType = CS_DC_EXECUTION_TYPE_CS_READY;  // For cleanup purpose

        STAT_GLBL_START(dipSendCommandSubmission);

        // Gaudi-2 (?)
        if (m_deviceType == synDeviceGaudi)
        {
            if (isStagedSubmission)
            {
                // Use temp cs handle for all staged CSs, but it is important to return the **first** cs handle
                // such that it will be saved in the stream inflight Cs DB
                uint64_t          tmpCsHandle = 0;
                const StagedInfo& stagedInfo  = (*stagesInfo)[stageIdx];

                if (!stagedInfo.hasWork)
                {
                    continue;
                }

                if ((g_preSubmissionCsInspection) && stagedInfo.isLastSubmission)
                {
                    checkForCsUndefinedOpcode(pCsDataChunks, pCommandSubmission, m_amountOfEnginesInArbGroup);
                }

                pCommandSubmission->setEncapsHandleId(sigHandleId);
                bool currStatus = _sendCommandSubmission(pCommandSubmission, tmpCsHandle, &stagedInfo, stageIdx);
                if (stagedInfo.isFirstSubmission)
                {
                    pCommandSubmission->setFirstStageCSHandle(tmpCsHandle);
                    // save the first handle in csHandle since this returns to the stream
                    // LKD saves and watches the completion of the first cs (handle) only!
                    csHandle = tmpCsHandle;
                }

                if (!currStatus)  // if failed, continue going so hopefully we won't have stuck operations on the
                                  // device.
                {
                    operationStatus = currStatus;
                    LOG_CRITICAL(SYN_STREAM,
                                 "{}: Cant send-staged job, node 0x{:x} out of 0x{:x}",
                                 HLLOG_FUNC,
                                 stageNodesIndex,
                                 m_rRecipe.node_nr);
                }
            }
            else
            {
                if (g_preSubmissionCsInspection)
                {
                    checkForCsUndefinedOpcode(pCsDataChunks, pCommandSubmission, m_amountOfEnginesInArbGroup);
                }

                pCommandSubmission->setEncapsHandleId(sigHandleId);
                operationStatus = _sendCommandSubmission(pCommandSubmission, csHandle, nullptr, 0);
            }
        }
        else
        {
            pCommandSubmission->setCalledFrom("enqueue");

            if (g_preSubmissionCsInspection)
            {
                checkForCsUndefinedOpcode(pCsDataChunks, pCommandSubmission, m_amountOfEnginesInArbGroup);
            }

            pCommandSubmission->setEncapsHandleId(sigHandleId);
            operationStatus = _sendCommandSubmission(pCommandSubmission, csHandle, nullptr, 0);
        }
        STAT_GLBL_COLLECT_TIME(dipSendCommandSubmission, globalStatPointsEnum::dipSendCommandSubmission);

        if (!operationStatus)
        {
            LOG_ERR(SYN_STREAM, "{}: cannot send-job", HLLOG_FUNC);
            pPatchingInformation->patchingAbort();
            _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                    csDcExecutionType,
                                                    EXECUTION_STAGE_ENQUEUE,
                                                    false,
                                                    csDcProcessingStatus);

            if (!isCSsetInCSDC)
            {
                // We prefer to store it over deleting it, so we can parse it
                pCsDataChunks->setCommandSubmissionInstance(pCommandSubmission);
            }

            return synFailedToSubmitWorkload;
        }

        if (isCsCreationRequired)
        {
            pCsDataChunks->setCommandSubmissionInstance(pCommandSubmission);
            isCSsetInCSDC = true;
        }
        if (GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())
        {
            char desc[50];
            sprintf(desc, "stage lastNodeIdx=%d,", stageNodesIndex);
            PROFILER_MEASURE_TIME(desc);
        }
    }  // end of stage submission loop
    m_lastSobjAddress = sigHandleSobjBaseAddressOffset;
    pCsDataChunks->addWaitForEventHandle(csHandle);

    if (isPatchingRequired)
    {
        pPatchingInformation->patchingCompletion();
    }

    _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                            csDcExecutionType,
                                            EXECUTION_STAGE_ENQUEUE,
                                            true,
                                            csDcProcessingStatus);

    STAT_GLBL_COLLECT_TIME(dipRecipeEnqueue, globalStatPointsEnum::dipRecipeEnqueue);
    return synSuccess;
}

void DynamicInfoProcessor::_storeStaticBlobsDeviceAddresses(uint64_t programCodeHandle)
{
    if (m_staticBlobsAddressesInWs.programCodeAddrInWS == programCodeHandle)
    {
        // There is no need to update CP_DMAs
        return;
    }

    staticBlobsDeviceAddresses& staticBlobsAddrMap = m_staticBlobsAddressesInWs.staticBlobsAddresses;

    uint64_t blobsNumber  = m_rRecipe.blobs_nr;
    blob_t*  pCurrentBlob = m_rRecipe.blobs;
    for (uint64_t blobIndex = 0; blobIndex < blobsNumber; blobIndex++, pCurrentBlob++)
    {
        uint64_t blobSize = pCurrentBlob->size;

        // in gaudi demo all the blobs are tagged as patchable blobs
        if (pCurrentBlob->blob_type.requires_patching)
        {
            continue;
        }
        staticBlobsAddrMap[blobIndex] = programCodeHandle;
        LOG_TRACE(SYN_STREAM,
                  "{}: blob index:{} static blob addr: 0x{:x}, size: {}",
                  HLLOG_FUNC,
                  blobIndex,
                  programCodeHandle,
                  blobSize);

        programCodeHandle += blobSize;
    }
}

bool DynamicInfoProcessor::notifyCsCompleted(CommandSubmissionDataChunks* pCsDataChunks,
                                             uint64_t                     waitForEventHandle,
                                             bool                         csFailed)
{
    std::unique_lock<std::mutex> mutex(m_csdcMutex);

    if (m_usedCsDcStageTypeDb.size() == 0)
    {
        LOG_ERR(SYN_STREAM, "{}: Inflight CSDC database is empty during completion call", HLLOG_FUNC);
        return false;
    }

    const eExecutionStage                   stage = pCsDataChunks->getExecutionStage();
    UsedCsDcStageTypeDB::iterator           usedCsDcStageTypeIter;
    CommandSubmissionDataChunksDB&          rUsedCsDataChunksDb    = m_usedCsDataChunksDb[stage];
    CommandSubmissionDataChunksDB::iterator usedCsDataChunksDbIter = rUsedCsDataChunksDb.begin();
    for (usedCsDcStageTypeIter = m_usedCsDcStageTypeDb.begin(); usedCsDcStageTypeIter != m_usedCsDcStageTypeDb.end();
         ++usedCsDcStageTypeIter)
    {
        if (stage != *usedCsDcStageTypeIter)
        {
            continue;
        }

        CommandSubmissionDataChunks* pCsDataChunksUsed = *usedCsDataChunksDbIter;

        if (pCsDataChunksUsed == pCsDataChunks)
        {
            if (pCsDataChunksUsed->containsHandle(waitForEventHandle))
            {
                break;
            }
        }

        ++(usedCsDataChunksDbIter);

        if (usedCsDataChunksDbIter == rUsedCsDataChunksDb.end())
        {
            LOG_ERR(SYN_STREAM,
                    "{}: failed to find CS since invalid location reached stage {} {:#x}",
                    HLLOG_FUNC,
                    stage,
                    TO64(pCsDataChunks));
            return false;
        }
    }

    if (usedCsDcStageTypeIter == m_usedCsDcStageTypeDb.end())
    {
        LOG_ERR(SYN_STREAM, "{}: failed to find CS {:#x}", HLLOG_FUNC, TO64(pCsDataChunks));
        return false;
    }

    if (!pCsDataChunks->popWaitForEventHandle(waitForEventHandle))
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to pop wait-for-event handle {}", HLLOG_FUNC, waitForEventHandle);
        return false;
    }

    if ((!pCsDataChunks->isWaitForEventHandleSet()) && (!csFailed))
    {
        uint64_t executionHandle = pCsDataChunks->getExecutionHandle();
        m_freeCsDataChunksDb[stage][executionHandle].push_back(pCsDataChunks);
    }

    rUsedCsDataChunksDb.erase(usedCsDataChunksDbIter);
    m_usedCsDcStageTypeDb.erase(usedCsDcStageTypeIter);

    return true;
}

uint32_t DynamicInfoProcessor::releaseCommandSubmissionDataChunks(uint32_t numOfElementsToRelease,
                                                                  CommandSubmissionDataChunksVec& releasedElements,
                                                                  bool                            keepOne)
{
    std::unique_lock<std::mutex> mutex(m_csdcMutex);

    const uint32_t numOfFreeElements =
        m_freeCsDataChunksDb[EXECUTION_STAGE_ACTIVATE].size() + m_freeCsDataChunksDb[EXECUTION_STAGE_ENQUEUE].size();
    const uint32_t numOfUsedElements = m_usedCsDcStageTypeDb.size();
    if (numOfFreeElements == 0)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: Free-Elements DB is empty, Used-Elements DB has {} elements",
                  HLLOG_FUNC,
                  numOfUsedElements);
        return 0;
    }

    uint32_t                                    leftElementsToRelease       = numOfElementsToRelease;
    uint32_t                                    totalAmountOfElements       = numOfFreeElements + numOfUsedElements;
    freeCommandSubmissionDataChunksDB::iterator currentExecutionHandleCsDcIter;
    freeCommandSubmissionDataChunksDB*          pCurrentFreeCsDataChunksDb = nullptr;
    // Todo [SW-86401] Based reuse CSDC release logic also on stage
    for (uint32_t stage = EXECUTION_STAGE_ACTIVATE; stage <= EXECUTION_STAGE_ENQUEUE; stage++)
    {
        auto freeCsDcIter = m_freeCsDataChunksDb[stage].begin();

        auto currentExecutionHandleCsDcLocalIter = m_freeCsDataChunksDb[stage].find(m_currentExecutionHandle);
        auto freeCsDcEndIter                     = m_freeCsDataChunksDb[stage].end();

        bool isBreak = false;
        while (freeCsDcIter != freeCsDcEndIter)
        {
            uint32_t numOfReleasedElements = (numOfElementsToRelease - leftElementsToRelease);
            if (keepOne && (totalAmountOfElements - numOfReleasedElements == 1))
            {
                isBreak = true;
                break;
            }

            if (freeCsDcIter == currentExecutionHandleCsDcLocalIter)
            {
                pCurrentFreeCsDataChunksDb     = &m_freeCsDataChunksDb[stage];
                currentExecutionHandleCsDcIter = currentExecutionHandleCsDcLocalIter;
                freeCsDcIter++;
                continue;
            }

            CommandSubmissionDataChunksDB& currentHandleFreeCsDataChunks = freeCsDcIter->second;

            _releaseCommandSubmissionDataChunks(leftElementsToRelease, releasedElements, currentHandleFreeCsDataChunks);

            if (currentHandleFreeCsDataChunks.size() == 0)
            {
                freeCsDcIter = m_freeCsDataChunksDb[stage].erase(freeCsDcIter);
            }

            if (leftElementsToRelease == 0)
            {
                LOG_TRACE(SYN_STREAM,
                          "{}: Released (non-current only) {} elements free DB size {}",
                          HLLOG_FUNC,
                          numOfReleasedElements,
                          m_freeCsDataChunksDb[stage].size());
                return numOfElementsToRelease;
            }
        }
        if (isBreak)
        {
            break;
        }
    }

    // we get here only if we didn't release enough and one of the options:
    // 1) keepOne = true, free = 1, use == 0 -> do not release
    // 2) keepOne = true, use  > 0           -> release
    // 3) keepOne = false                    -> release

    // Release from CURRENT execution-handle:
    // In case there are used-elements
    if ((!keepOne || (m_usedCsDcStageTypeDb.size() > 0)) && (pCurrentFreeCsDataChunksDb != nullptr))
    {
        CommandSubmissionDataChunksDB& currentHandleFreeCsDataChunks = currentExecutionHandleCsDcIter->second;

        _releaseCommandSubmissionDataChunks(leftElementsToRelease, releasedElements, currentHandleFreeCsDataChunks);

        if (currentHandleFreeCsDataChunks.empty())
        {
            currentExecutionHandleCsDcIter = pCurrentFreeCsDataChunksDb->erase(currentExecutionHandleCsDcIter);
        }
    }

    LOG_TRACE(SYN_STREAM,
              "{}: Released {} elements free DB size {} used DB size {}",
              HLLOG_FUNC,
              releasedElements.size(),
              numOfFreeElements - (numOfElementsToRelease - leftElementsToRelease),
              numOfUsedElements);
    return (numOfElementsToRelease - leftElementsToRelease);
}

CommandSubmissionDataChunks*
DynamicInfoProcessor::getAvailableCommandSubmissionDataChunks(eExecutionStage executionStage, bool isCurrent)
{
    std::unique_lock<std::mutex> mutex(m_csdcMutex);

    CommandSubmissionDataChunksDB& requestedExecStageUsedCsDcDB = m_usedCsDataChunksDb[executionStage];
    if ((isCurrent) && (!requestedExecStageUsedCsDcDB.empty()))
    {
        auto usedCsDataChunkIter = requestedExecStageUsedCsDcDB.back();
        HB_ASSERT((usedCsDataChunkIter->getExecutionHandle() == m_currentExecutionHandle),
                  "Last stored CS-DC does not match current execution-handle");

        return usedCsDataChunkIter;
    }

    freeCommandSubmissionDataChunksDB& requestedExecStageFreeCsDcDB = m_freeCsDataChunksDb[executionStage];
    auto freeCsDataChunkIter = requestedExecStageFreeCsDcDB.find(m_currentExecutionHandle);
    if (!isCurrent)
    {
        freeCsDataChunkIter = requestedExecStageFreeCsDcDB.begin();
    }

    auto freeCsDataChunkEndIter = requestedExecStageFreeCsDcDB.end();

    CommandSubmissionDataChunks* freeCsDataChunkElement = nullptr;

    if (freeCsDataChunkIter != freeCsDataChunkEndIter)
    {
        CommandSubmissionDataChunksDB& csDataChunksDb = freeCsDataChunkIter->second;
        LOG_DEBUG(SYN_STREAM, "{}: Found free CS-DC (is current execution-handle {})", HLLOG_FUNC, isCurrent);
        freeCsDataChunkElement = csDataChunksDb.front();
        csDataChunksDb.pop_front();

        if (csDataChunksDb.empty())
        {
            requestedExecStageFreeCsDcDB.erase(freeCsDataChunkIter);
        }
    }

    return freeCsDataChunkElement;
}

void DynamicInfoProcessor::incrementExecutionHandle()
{
    if (m_currentExecutionHandle == std::numeric_limits<uint64_t>::max())
    {
        m_currentExecutionHandle = CS_DC_FIRST_NON_RESERVED_EXECUTION_HANDLE;
    }

    m_currentExecutionHandle++;
}

uint64_t DynamicInfoProcessor::getExecutionHandle()
{
    return m_currentExecutionHandle;
}

bool DynamicInfoProcessor::_createCommandSubmission(CommandSubmission&           commandSubmission,
                                                    CommandSubmissionDataChunks& commandSubmissionDataChunks,
                                                    uint64_t                     numOfPqsForInternalQueues,
                                                    eExecutionStage              stage)
{
    PROFILER_COLLECT_TIME()

    HB_ASSERT(stage < EXECUTION_STAGE_LAST, "Illegal execution stage");

    unsigned externalQueuesCbsCounter = 0;

    uint64_t  pqIndex    = 0;  // For Internal-queues
    synStatus status     = synSuccess;
    bool      operStatus = true;

    const uint64_t jobs_nr =
        (stage == EXECUTION_STAGE_ENQUEUE) ? m_rRecipe.execute_jobs_nr : m_rRecipe.activate_jobs_nr;
    job_t* pCurrentJob = (stage == EXECUTION_STAGE_ENQUEUE) ? m_rRecipe.execute_jobs : m_rRecipe.activate_jobs;

    if (jobs_nr == 0)
    {
        LOG_DEBUG(SYN_STREAM, "No CS creation is required - there are no jobs");
        return true;
    }

    uint64_t arbMasterEngineId = m_rArbMasterHelper.getArbMasterQmanId();
    job_t*   pArbMasterJob     = nullptr;

    for (uint64_t jobIndex = 0; jobIndex < jobs_nr; jobIndex++, pCurrentJob++)
    {
        uint64_t engineId = pCurrentJob->engine_id;
        if (m_qmansDef->isWorkCompletionQueueId(engineId))
        {
            continue;
        }

        if (m_deviceType == synDeviceGaudi)
        {
            if (m_qmansDef->isStreamMasterQueueIdForCompute(engineId))
            {
                // Handle external-queues
                operStatus = createExternalQueueCB(commandSubmission);
                if (!operStatus)
                {
                    status = synFail;
                    LOG_ERR(SYN_STREAM, "{}: Failed to create external-queue CB for engineId {}", HLLOG_FUNC, engineId);
                    break;
                }
                externalQueuesCbsCounter++;
                continue;
            }
        }

        if (arbMasterEngineId == engineId)
        {
            pArbMasterJob = pCurrentJob;
            continue;
        }

        // Handle internal-queues
        HB_ASSERT((pqIndex < numOfPqsForInternalQueues),
                  "Failed to build CS due to mismatch of internal-queues' amount");

        operStatus =
            _createInternalQueuePqCommand(commandSubmission, commandSubmissionDataChunks, pqIndex, pCurrentJob);
        if (!operStatus)
        {
            status = synFail;
            LOG_ERR(SYN_STREAM, "{}: Failed to create internal-queue CB for engineId {}", HLLOG_FUNC, engineId);
            break;
        }
    }

    // Handling of the ARB-Master job, as the last job
    do
    {
        if (pArbMasterJob == nullptr)
        {
            LOG_ERR(SYN_STREAM, "Did not find an ARB-Master job");
            status = synFail;
            break;
        }

        // TODO - Use a method for avoiding code duplication

        // Handle internal-queues
        HB_ASSERT((pqIndex < numOfPqsForInternalQueues),
                  "Failed to build CS due to mismatch of internal-queues' amount");

        operStatus =
            _createInternalQueuePqCommand(commandSubmission, commandSubmissionDataChunks, pqIndex, pArbMasterJob);
        if (!operStatus)
        {
            status = synFail;
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to create internal-queue CB for engineId {}",
                    HLLOG_FUNC,
                    arbMasterEngineId);
            break;
        }
    } while (0);  // do once

    operStatus = (status == synSuccess);
    if (!operStatus)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to create Command-Buffers for external-queues", HLLOG_FUNC);
        status = CommandSubmissionBuilder::getInstance()->destroyCmdSubmissionSynCBs(commandSubmission);
        if (status != synSuccess)
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Failed to destroy command-submission", HLLOG_FUNC);
        }
    }

    PROFILER_MEASURE_TIME("createCS");

    return operStatus;
}

bool DynamicInfoProcessor::createExternalQueueCB(CommandSubmission& commandSubmission)
{
    const std::array<DataChunksDB, RecipeStaticInfo::EXTERNAL_QMANS_AMOUNT>& allEnginesDataChunks =
        m_rRecipeInfo.retrieveExternalQueueBlobsDataChunks();

    for (uint16_t queueIndex = 0; queueIndex < RecipeStaticInfo::EXTERNAL_QMANS_AMOUNT; queueIndex++)
    {
        const DataChunksDB& dataChunks = allEnginesDataChunks[queueIndex];

        for (auto singleDataChunk : dataChunks)
        {
            commandSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION,
                                                 queueIndex,
                                                 singleDataChunk->getUsedSize(),
                                                 singleDataChunk->getHandle());
        }
    }

    return true;
}

bool DynamicInfoProcessor::_createInternalQueuePqCommand(CommandSubmission&           commandSubmission,
                                                         CommandSubmissionDataChunks& commandSubmissionDataChunks,
                                                         uint64_t&                    pqIndex,
                                                         const job_t*                 pCurrentJob)
{
    const CommandSubmissionBuilder* pCmdSubmissionBuilder = CommandSubmissionBuilder::getInstance();

    uint64_t engineId = pCurrentJob->engine_id;
    // Add PQs of current program
    try
    {
        const cpDmaSingleEngineInfo& engineCpDmaInfo =
            commandSubmissionDataChunks.getSingleEngineCpDmaMapping(engineId);
        for (auto& singleCpDmasDataChunk : engineCpDmaInfo)
        {
            primeQueueCommand pq = {0};
            pq.queueIndex        = engineId;
            pq.address           = singleCpDmasDataChunk.deviceVirtualAddress;
            pq.size              = singleCpDmasDataChunk.size;

            bool operStatus = pCmdSubmissionBuilder->addPqCmdForInternalQueue(commandSubmission,
                                                                              pqIndex,
                                                                              engineId,
                                                                              pq.size,
                                                                              pq.address);
            if (!operStatus)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to add PQ-command {}", HLLOG_FUNC, pqIndex);
                return false;
            }

            LOG_TRACE(SYN_STREAM,
                      "{}: New PQ (at index {}) - engineId {} size 0x{:x} address 0x{:x}",
                      HLLOG_FUNC,
                      pqIndex,
                      pq.queueIndex,
                      pq.size,
                      pq.address);
            pqIndex++;
        }
    }
    catch (std::out_of_range& err)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to find engineId {} to CP-DMAs mapping DB", HLLOG_FUNC, engineId);
        return false;
    }

    return true;
}

bool DynamicInfoProcessor::_sendCommandSubmission(CommandSubmission*& pCommandSubmission,
                                                  uint64_t&           csHandle,
                                                  const StagedInfo*   pStagedInfo,
                                                  unsigned int        stageIdx)
{
    HB_ASSERT(pCommandSubmission != nullptr, "Got nullptr for CommandSubmission");

    synStatus status = m_rSubmitter.submitCommandBuffers(m_rBasicQueueInfo,
                                                         m_physicalQueueOffset,
                                                         *pCommandSubmission,
                                                         &csHandle,
                                                         m_physicalQueueOffset,
                                                         pStagedInfo,
                                                         stageIdx);

    if (status != synSuccess)
    {
        return false;
    }

    LOG_DEBUG(SYN_STREAM,
              "Stream {} recipe 0x{:x} submitted with csHandle 0x{:x}",
              m_rBasicQueueInfo.getDescription(),
              TO64((&m_rRecipe)),
              csHandle);

    return true;
}

void DynamicInfoProcessor::_insertCsDataChunkToCacheUponCompletion(CommandSubmissionDataChunks* pCsDataChunks,
                                                                   eCsDcExecutionType           csDcExecutionType,
                                                                   eExecutionStage              executionStage,
                                                                   bool                         isSucceeded,
                                                                   eCsDcProcessorStatus&        csDcProcessingStatus)
{
    HB_ASSERT(executionStage < EXECUTION_STAGE_LAST, "Illegal execution stage");

    if (pCsDataChunks == nullptr)
    {
        return;
    }

    switch (csDcExecutionType)
    {
        case CS_DC_EXECUTION_TYPE_CS_READY:
            pCsDataChunks->setExecutionHandle(m_currentExecutionHandle);
            break;

        case CS_DC_EXECUTION_TYPE_CP_DMA_READY:
            pCsDataChunks->setExecutionHandle(CS_DC_RESERVED_EXECUTION_HANDLE);
            break;

        case CS_DC_EXECUTION_TYPE_NOT_READY:
            pCsDataChunks->setExecutionHandle(CS_DC_INVALID_EXECUTION_HANDLE);
            break;
    }

    pCsDataChunks->setCsDcExecutionType(csDcExecutionType);

    {
        std::unique_lock<std::mutex> mutex(m_csdcMutex);

        if (isSucceeded)
        {
            m_usedCsDataChunksDb[executionStage].push_back(pCsDataChunks);
            m_usedCsDcStageTypeDb.push_back(executionStage);
            csDcProcessingStatus = CS_DC_PROCESSOR_STATUS_STORED_AND_SUBMITTED;
        }
        else
        {
            if (csDcExecutionType == CS_DC_EXECUTION_TYPE_NOT_READY)
            {
                csDcProcessingStatus = CS_DC_PROCESSOR_STATUS_FAILED;
            }
            else
            {
                m_freeCsDataChunksDb[executionStage][pCsDataChunks->getExecutionHandle()].push_back(pCsDataChunks);
                csDcProcessingStatus = CS_DC_PROCESSOR_STATUS_STORED_ONLY;
            }
        }
    }
}

void DynamicInfoProcessor::_releaseCommandSubmissionDataChunks(
    uint32_t&                       leftElementsToRelease,
    CommandSubmissionDataChunksVec& releasedElements,
    CommandSubmissionDataChunksDB&  currentHandleFreeCsDataChunks)
{
    uint32_t numOfHandleElements          = currentHandleFreeCsDataChunks.size();
    uint32_t numOfHandleElementsToRelease = std::min(leftElementsToRelease, numOfHandleElements);
    while (numOfHandleElementsToRelease != 0)
    {
        releasedElements.push_back(currentHandleFreeCsDataChunks.front());
        currentHandleFreeCsDataChunks.pop_front();
        leftElementsToRelease--;
        numOfHandleElementsToRelease--;
    }
}

void DynamicInfoProcessor::getProgramCodeHandle(uint64_t& programCodeHandle) const
{
    programCodeHandle = m_lastProgramCodeHandle;
}

void DynamicInfoProcessor::setProgramCodeHandle(uint64_t programCodeHandle)
{
    m_lastProgramCodeHandle = programCodeHandle;
}

void DynamicInfoProcessor::setProgramCodeAddrInWS(uint64_t programCodeAddrInWS)
{
    m_staticBlobsAddressesInWs.programCodeAddrInWS = programCodeAddrInWS;
}

void DynamicInfoProcessor::getProgramDataHandle(uint64_t& programDataHandle) const
{
    programDataHandle = m_lastProgramDataHandle;
}

void DynamicInfoProcessor::setProgramDataHandle(uint64_t programDataHandle)
{
    m_lastProgramDataHandle = programDataHandle;
}

synStatus DynamicInfoProcessor::_patchOnDc(CommandSubmissionDataChunks*& pCsDataChunks,
                                           uint64_t                      scratchPadAddress,
                                           eCsDcExecutionType            csDcExecutionType,
                                           eExecutionStage               executionStage,
                                           PatchPointPointerPerSection&  currPatchPoints,
                                           SectionTypesToPatch&          sectionTypesToPatch,
                                           uint32_t                      lastStageNode,
                                           uint32_t                      sobBaseAddr,
                                           eCsDcProcessorStatus&         csDcProcessingStatus)
{
    HB_ASSERT_PTR(pCsDataChunks);

    PROFILER_COLLECT_TIME()
    HostAddressPatchingInformation* pPatchingInformation = &(pCsDataChunks->getHostAddressInDcInformation());
    synStatus                       status               = synSuccess;

    STAT_GLBL_START(allSectionsPatch);

    do  // once
    {
        bool                            operationStatus            = true;
        const DataChunkPatchPointsInfo* pPatchPointsDataChunksInfo = nullptr;
        uint32_t                        sobBaseAddressToPatch      = 0;
        if (m_deviceType == synDeviceGaudi)
        {
            sobBaseAddressToPatch = SIG_HANDLE_INVALID != sobBaseAddr
                                        ? (uint32_t)(CFG_BASE + sobBaseAddr)
                                        : m_rDeviceAgnosticRecipeInfo.m_signalFromGraphInfo.m_lkdSobLowPartAddress;
        }
        else if (SIG_HANDLE_INVALID != sobBaseAddr)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Unable to use SFG feature on device type {}, only Gaudi is currently supported",
                    HLLOG_FUNC,
                    m_deviceType);
            status = synFail;
            break;
        }

        for (auto& type : sectionTypesToPatch)
        {
            LOG_TRACE(SYN_STREAM, "{}: Patching type {}", HLLOG_FUNC, type);
            if (currPatchPoints[type] == nullptr)  // this happened on a very small graph. A tensor was on section type
                                                   // 0, but no patch points for section type 0
            {
                continue;
            }

            pPatchPointsDataChunksInfo = m_rRecipeInfo.getPatchingPointsDcLocation(executionStage, type);
            operationStatus = HostAddressPatchingExecuter::executePatchingInDataChunks(
                pPatchingInformation->getSectionsToHostAddressDB(),
                currPatchPoints[type],
                m_dataChunksHostAddresses,
                pPatchPointsDataChunksInfo->m_singleChunkSize,
                lastStageNode,
                sobBaseAddressToPatch);
            if (!operationStatus)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to patch (full) recipe", HLLOG_FUNC);
                status = synFail;
                break;
            }
        }
        // Need to make sure the SOB the driver used is the same as in CS-DC
        if ((sectionTypesToPatch.empty() ||
             (sectionTypesToPatch.size() > 0 && *(sectionTypesToPatch.begin()) != PP_TYPE_ID_ALL)) &&
            (pCsDataChunks->isNewSobjAddress(sobBaseAddr)))
        {  // patching only of sobj patching is required and was not already done as included in PP_TYPE_ID_ALL
            LOG_TRACE(SYN_STREAM, "{}: Patching sobj with address {}", HLLOG_FUNC, sobBaseAddr);

            pPatchPointsDataChunksInfo           = m_rRecipeInfo.getSobjPatchingPointsDcLocation();
            const data_chunk_patch_point_t* dcPp = pPatchPointsDataChunksInfo->m_dataChunkPatchPoints;

            if (m_deviceType != synDeviceGaudi)
            {
                LOG_ERR(SYN_STREAM,
                        "{}: Unable to use SFG feature on device type {}, only Gaudi is currently supported",
                        HLLOG_FUNC,
                        m_deviceType);
                status = synFail;
                break;
            }
            operationStatus = HostAddressPatchingExecuter::executePatchingInDataChunks(
                pPatchingInformation->getSectionsToHostAddressDB(),
                dcPp,
                m_dataChunksHostAddresses,
                pPatchPointsDataChunksInfo->m_singleChunkSize,
                lastStageNode,
                sobBaseAddressToPatch);

            if (!operationStatus)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to patch (full) recipe", HLLOG_FUNC);
                status = synFail;
                break;
            }
        }
        m_lastPatchedWorkspaceAddress = scratchPadAddress;
    } while (0);  // Do once

    STAT_GLBL_COLLECT_TIME(allSectionsPatch, globalStatPointsEnum::patchingAll);

    if (status != synSuccess)
    {
        pPatchingInformation->patchingAbort();
        _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                csDcExecutionType,
                                                executionStage,
                                                false,
                                                csDcProcessingStatus);
        LOG_ERR(SYN_STREAM, "{}: Failed to patch recipe", HLLOG_FUNC);
    }

    PROFILER_MEASURE_TIME("patchOnDc");

    return status;
}

synStatus DynamicInfoProcessor::_validateNewTensors(const synLaunchTensorInfoExt* launchTensorsInfo,
                                                    uint32_t                      launchTensorsAmount,
                                                    uint32_t                      flags,
                                                    uint64_t                      scratchPadAddress,
                                                    uint64_t                      programCodeHandle,
                                                    uint64_t                      sectionAddressForData,
                                                    bool                          programCodeInCache,
                                                    bool                          programDataInCache)
{
    PROFILER_COLLECT_TIME()

    LOG_DEBUG(SYN_STREAM, "{}: launchTensorsAmount {}", HLLOG_FUNC, launchTensorsAmount);

    STAT_GLBL_START(validate);


    WorkSpacesInformation workspaceInfo;
    workspaceInfo.scratchPadAddress  = scratchPadAddress;
    workspaceInfo.programCodeAddress = programCodeHandle;
    workspaceInfo.programDataAddress = sectionAddressForData;
    workspaceInfo.programCodeInCache = programCodeInCache;
    workspaceInfo.programDataInCache = programDataInCache;

    synStatus status =
        RecipePatchProcessor::validateSectionsInfo(m_rBasicRecipeInfo,
                                                   m_rDeviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                   launchTensorsAmount,
                                                   launchTensorsInfo,
                                                   flags,
                                                   workspaceInfo,
                                                   m_rDeviceAgnosticRecipeInfo.m_recipeTensorInfo.m_sectionsInfo,
                                                   m_rDevMemAlloc);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to validate tensors", HLLOG_FUNC);
        return status;
    }

    STAT_GLBL_COLLECT_TIME(validate, globalStatPointsEnum::tensorsValidate);
    PROFILER_MEASURE_TIME("validateTensors")

    return synSuccess;
}

synStatus DynamicInfoProcessor::_analyzeNewTensors(CommandSubmissionDataChunks*& pCsDataChunks,
                                                   const synLaunchTensorInfoExt* launchTensorsInfo,
                                                   uint32_t                      launchTensorsAmount,
                                                   uint64_t                      scratchPadAddress,
                                                   uint64_t                      programCodeHandle,
                                                   uint64_t                      sectionAddressForData,
                                                   eCsDcExecutionType            csDcExecutionType,
                                                   eExecutionStage               executionStage,
                                                   bool&                         hasNewAddress,
                                                   bool                          programCodeInCache,
                                                   bool                          programDataInCache,
                                                   eCsDcProcessorStatus&         csDcProcessingStatus,
                                                   uint32_t                      sobjAddress,
                                                   uint64_t                      assertAsyncMappedAddress,
                                                   uint32_t                      flags)
{
    LOG_DEBUG(SYN_STREAM, "{}: launchTensorsAmount {}", HLLOG_FUNC, launchTensorsAmount);

    PROFILER_COLLECT_TIME()

    HostAddressPatchingInformation* pPatchingInformation = nullptr;
    if ((!getHostAddressPatchingInfo(pCsDataChunks, pPatchingInformation)) || (pPatchingInformation == nullptr))
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to get patching-information", HLLOG_FUNC);
        return synFail;
    }

    STAT_GLBL_START(analyze);
    ValidSectionAddresses* pValidAddresses = nullptr;
    ValidSectionAddresses  validSectionAddresses;
    m_rDevMemAlloc.getValidAddressesRange(validSectionAddresses.lowestValidAddress,
                                          validSectionAddresses.highestValidAddress);
    if (m_deviceType != synDeviceGaudi2)
    {
        pValidAddresses = &validSectionAddresses;
    }

    WorkSpacesInformation workspaceInfo;
    workspaceInfo.scratchPadAddress  = scratchPadAddress;
    workspaceInfo.programCodeAddress = programCodeHandle;
    workspaceInfo.programDataAddress = sectionAddressForData;
    workspaceInfo.programCodeInCache = programCodeInCache;
    workspaceInfo.programDataInCache = programDataInCache;
    workspaceInfo.assertAsyncMappedAddress = assertAsyncMappedAddress;

    synStatus status = RecipePatchProcessor::process(
        m_rBasicRecipeInfo,
        m_rDeviceAgnosticRecipeInfo.m_recipeTensorInfo,
        launchTensorsInfo,
        launchTensorsAmount,
        flags,
        workspaceInfo,
        *pPatchingInformation,
        m_rDevMemAlloc,
        m_tensorIdx2userIdx,
        false,
        !RecipeUtils::isIH2DRecipe(&m_rRecipe),
        pValidAddresses);

    STAT_GLBL_COLLECT_TIME(analyze, globalStatPointsEnum::tensorsAnalyze);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to analyze tensors", HLLOG_FUNC);

        pPatchingInformation->patchingAbort();
        _insertCsDataChunkToCacheUponCompletion(pCsDataChunks,
                                                csDcExecutionType,
                                                executionStage,
                                                false,
                                                csDcProcessingStatus);

        return (status == synFail) ? synFailedSectionValidation : status;
    }
    bool isNewSobjAddress = false;
    isNewSobjAddress      = pCsDataChunks->isNewSobjAddress(sobjAddress);

    hasNewAddress =
        pPatchingInformation->hasNewSectionAddress() || RecipeUtils::isDsd(m_rBasicRecipeInfo) || isNewSobjAddress;
    if (!hasNewAddress)
    {
        pPatchingInformation->patchingCompletion();
    }
    if (GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())
    {
        char desc[50] = {};
        snprintf(desc, sizeof(desc), "%s count=%d", "analyzeTensors", launchTensorsAmount);
        PROFILER_MEASURE_TIME(desc)
    }

    return synSuccess;
}

bool DynamicInfoProcessor::_isPatchingRequired(const CommandSubmissionDataChunks& rCsDataChunks,
                                               uint32_t                           launchTensorsAmount,
                                               uint64_t                           scratchPadAddress,
                                               uint64_t                           programDataHandle,
                                               uint64_t                           sobjAddress,
                                               eExecutionStage                    stage) const
{
    // recipe values correctness was already checked in RecipeVerification::verifyPatching
    const uint32_t activate_patch_points_nr = m_rRecipe.activate_patch_points_nr;
    const uint32_t enqueue_patch_points_nr  = m_rRecipe.patch_points_nr - m_rRecipe.activate_patch_points_nr;
    const uint64_t patchingBlobsBufferSize  = m_rRecipe.patching_blobs_buffer_size;

    LOG_TRACE(SYN_STREAM,
              "launchTensorsAmount {} scratchPadAddress {:#x} programDataHandle {:#x} sobjAddress {:#x} stage {} "
              "activate_patch_points_nr {} enqueue_patch_points_nr {} patchingBlobsBufferSize {}",
              launchTensorsAmount,
              scratchPadAddress,
              programDataHandle,
              sobjAddress,
              stage,
              activate_patch_points_nr,
              enqueue_patch_points_nr,
              patchingBlobsBufferSize);

    uint32_t patch_points_nr;

    switch (stage)
    {
        case EXECUTION_STAGE_ACTIVATE:
        {
            patch_points_nr = activate_patch_points_nr;
            break;
        }
        case EXECUTION_STAGE_ENQUEUE:
        {
            patch_points_nr = enqueue_patch_points_nr;
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal execution stage");
            return false;
        }
    }

    if (patch_points_nr == 0)
    {
        return false;
    }

    bool isPersistentTensorsPatchingRequired = (launchTensorsAmount != 0);
    bool isScratchpadPatchingRequired        = (m_lastPatchedWorkspaceAddress != m_lastDownloadWorkspaceAddress);
    bool isProgramDataPatchingRequired       = (m_lastPatchedProgramDataHandle != programDataHandle);
    bool isSobjPatchingRequired              = (m_lastSobjAddress != sobjAddress);

    isScratchpadPatchingRequired  = rCsDataChunks.isNewScratchpadHandle(scratchPadAddress);
    isProgramDataPatchingRequired = rCsDataChunks.isNewProgramDataHandle(programDataHandle);
    isSobjPatchingRequired        = rCsDataChunks.isNewSobjAddress(sobjAddress);

    bool isPatchingRequired = isPersistentTensorsPatchingRequired || isScratchpadPatchingRequired ||
                              isProgramDataPatchingRequired || isSobjPatchingRequired;

    LOG_DEBUG(SYN_STREAM,
              "Stage {} patch_points_nr {} patching required={}",
              stage,
              patch_points_nr,
              isPatchingRequired);
    return isPatchingRequired;
}

bool DynamicInfoProcessor::getHostAddressPatchingInfo(CommandSubmissionDataChunks*&    pCsDataChunks,
                                                      HostAddressPatchingInformation*& pPatchingInformation)
{
    if (unlikely(pCsDataChunks == nullptr))
    {
        return false;
    }

    pPatchingInformation = &(pCsDataChunks->getHostAddressInDcInformation());
    return true;
}

void DynamicInfoProcessor::_conditionalAddProgramDataSectionForPatching(CommandSubmissionDataChunks* pCsDataChunks,
                                                                        uint64_t                     programDataHandle)
{
    HostAddressPatchingInformation& hostAddressPatchingInfo = pCsDataChunks->getHostAddressInDcInformation();

    uint64_t csDcProgramHandle = 0;
    pCsDataChunks->getProgramDataHandle(csDcProgramHandle);
    bool isProgramDataPatchingRequired = (csDcProgramHandle != programDataHandle);
    // Intentionally separated
    // TODO - Break the CsDcExecutionType into two separate patching-handles
    //        One for the Commands DCs and one for the Jobs DCs
    isProgramDataPatchingRequired |= pCsDataChunks->getCsDcExecutionType() == CS_DC_EXECUTION_TYPE_NOT_READY;
    if (isProgramDataPatchingRequired)
    {
        hostAddressPatchingInfo.markSectionTypeForPatching(
            m_rDeviceAgnosticRecipeInfo.m_recipeTensorInfo.m_sectionToSectionType[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA]);
        m_lastPatchedProgramDataHandle = programDataHandle;
    }
}

SectionTypesToPatch& DynamicInfoProcessor::_setupStartingPatchPointsPerSection(CommandSubmissionDataChunks* pCsDataChunks,
                                                                               PatchPointPointerPerSection& currPatchPoint,
                                                                               eExecutionStage              stage)
{
    // set current patch point pointer to track the stages of submission
    HostAddressPatchingInformation& hostAddressPatchingInfo     = pCsDataChunks->getHostAddressInDcInformation();
    SectionTypesToPatch&            sectionTypesToPatch         = hostAddressPatchingInfo.getSectionTypesQueuedForPatching();

    uint64_t amountOfsectionTypesToPatch = sectionTypesToPatch.size();
    uint64_t amountOfDummyPpsAll         = STAGED_SUBMISSION_PATCH_POINT_ADDITION;
    uint64_t amountOfPpsAll              = (m_rRecipeInfo.getPatchingPointsDcAmount(stage, PP_TYPE_ID_ALL) -
                                            STAGED_SUBMISSION_PATCH_POINT_ADDITION);
    bool     isPatchAll                  = hostAddressPatchingInfo.isFirstPatching();

    const uint64_t minimalAmountOfPPs                 = 2500;
    const float    maxPercentageOfPpsForSelectingPBSG = 20.0; // PBSG - Patch By Sections-Groups

    float    percentageOfPatching   = 0;
    uint64_t amountOfDummyPpsByType = 0;
    uint64_t amountOfPpsByType      = 0;

    if (amountOfsectionTypesToPatch == 0)
    {
        LOG_TRACE(SYN_STREAM, "No sections-types to patch");
        return sectionTypesToPatch;
    }

    if (isPatchAll)
    {
        LOG_TRACE(SYN_STREAM, "First patching => Patch all");
    }
    else if ((amountOfsectionTypesToPatch > 1) && (amountOfPpsAll < minimalAmountOfPPs))
    {
        isPatchAll = true;
        LOG_TRACE(SYN_STREAM, "Graph not have enough PPs ({}) => Patch all", amountOfPpsAll);
    }

    // In case there is a single SG, we will not patch all. Hence, no need to check for the PPs percentage boundary
    if ((!isPatchAll) && (amountOfsectionTypesToPatch > 1))
    {
        // Calculate total amount of PPs required to be patched
        for (auto& type : sectionTypesToPatch)
        {
            const DataChunkPatchPointsInfo* ppDcLoc = m_rRecipeInfo.getPatchingPointsDcLocation(stage, type);

            if (ppDcLoc == nullptr)
            {  // This happened for a tensor in section type 0, but no patch points for that section type
                currPatchPoint[type] = nullptr;
            }
            else
            {
                currPatchPoint[type]   =  ppDcLoc->m_dataChunkPatchPoints;
                uint64_t amountOfPps   =  m_rRecipeInfo.getPatchingPointsDcAmount(stage, type);
                amountOfDummyPpsByType += STAGED_SUBMISSION_PATCH_POINT_ADDITION;
                amountOfPpsByType      += (amountOfPps - STAGED_SUBMISSION_PATCH_POINT_ADDITION);
                LOG_TRACE(SYN_STREAM, "{}: stage {} types {} amount {}", HLLOG_FUNC, stage, type, amountOfPps);
            }
        }

        percentageOfPatching = (amountOfPpsAll != 0) ? ((float) (amountOfPpsByType * 100) / amountOfPpsAll) : 0;
        if (percentageOfPatching >= maxPercentageOfPpsForSelectingPBSG)
        {
            isPatchAll = true;
            LOG_TRACE(SYN_STREAM,
                      "Percentage of PPs ({}) is above limie ({}) => Patch all",
                      percentageOfPatching,
                      maxPercentageOfPpsForSelectingPBSG);
        }
    }

    if ((isPatchAll) || (amountOfsectionTypesToPatch == 1))
    {
        uint64_t patchPointsType = *(sectionTypesToPatch.begin());

        if (isPatchAll)
        {
            patchPointsType = PP_TYPE_ID_ALL;

            sectionTypesToPatch.clear();
            sectionTypesToPatch.insert(patchPointsType);
        }

        const DataChunkPatchPointsInfo* ppDcLoc = m_rRecipeInfo.getPatchingPointsDcLocation(stage, patchPointsType);

        if (ppDcLoc == nullptr)
        {  // This happened for a tensor in section type 0, but no patch points for that section type
            currPatchPoint[patchPointsType] = nullptr;
        }
        else
        {
            currPatchPoint[patchPointsType] = ppDcLoc->m_dataChunkPatchPoints;
        }
    }

    LOG_DEBUG(SYN_STREAM,
              "{}: sections types changed {} isPatchAll = {} percentageOfPatching = {} "
              "(#PPs: ByType {} All {}, #Dummies: ByType {} All {}) patchable_size {}",
              HLLOG_FUNC,
              amountOfsectionTypesToPatch,
              isPatchAll ? "yes" : "no",
              percentageOfPatching,
              amountOfPpsByType,
              amountOfPpsAll,
              amountOfDummyPpsByType,
              amountOfDummyPpsAll,
              m_rRecipe.patching_blobs_buffer_size);

    return sectionTypesToPatch;
}

bool DynamicInfoProcessor::resolveTensorsIndices(std::vector<uint32_t>*&       pTensorIdx2userIdx,
                                                 const uint32_t                launchTensorsAmount,
                                                 const synLaunchTensorInfoExt* launchTensorsInfo)
{
    pTensorIdx2userIdx = m_tensorIdx2userIdx;
    return RecipePatchProcessor::resolveTensorsIndices(m_tensorIdx2userIdx, m_rBasicRecipeInfo,
                                                       launchTensorsAmount, launchTensorsInfo);
}

bool DynamicInfoProcessor::isAnyInflightCsdc()
{
    std::unique_lock<std::mutex> mutex(m_csdcMutex);
    LOG_DEBUG(SYN_STREAM,
              "{}: inflight cs Num {}, mpRecipe: 0x{:x}",
              HLLOG_FUNC,
              m_usedCsDcStageTypeDb.size(),
              (uint64_t)(&m_rRecipe));
    return m_usedCsDcStageTypeDb.size() != 0;
}

void DynamicInfoProcessor::storeDataChunksHostAddresses(CommandSubmissionDataChunks& rCsDataChunks)
{
    const commandsDataChunksDB& currentCommandsDataChunks = rCsDataChunks.getCommandsBufferDataChunks();
    const uint64_t              numOfCommandsDataChunks   = currentCommandsDataChunks.size();
    m_dataChunksHostAddresses.resize(numOfCommandsDataChunks);
    auto commandsDataChunksIter = currentCommandsDataChunks.begin();

    for (uint64_t i = 0; i < numOfCommandsDataChunks; i++, commandsDataChunksIter++)
    {
        m_dataChunksHostAddresses[i] = (uint64_t)(*commandsDataChunksIter)->getChunkBuffer();
    }
}

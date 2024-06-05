#include "stream_dc_downloader.hpp"
#include "command_submission_data_chunks.hpp"
#include "data_chunk/data_chunk.hpp"
#include "data_chunk/data_chunks_cache.hpp"
#include "defenders.h"
#include "define_synapse_common.hpp"
#include "recipe_static_information.hpp"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "utils.h"
#include "profiler_api.hpp"
#include "command_submission.hpp"
#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"

// This is the size in all platforms - validation is still needed
#define CP_DMA_PACKET_SIZE 16

#define CUR_DATA_CHUNK _getChunk(cpDmaDataChunks, curDataChunkIdx)

StreamDcDownloader::StreamDcDownloader(synDeviceType deviceType, uint32_t physicalQueueOffset)
: m_deviceType(deviceType),
  m_physicalQueueOffset(physicalQueueOffset),
  m_packetGenerator(nullptr),
  m_qmansDef(nullptr),
  m_useArbitration(false)
{
    switch (m_deviceType)
    {
        case synDeviceGaudi:
        {
            m_packetGenerator = gaudi::CommandBufferPktGenerator::getInstance();
            m_qmansDef        = gaudi::QmansDefinition::getInstance();
            m_useArbitration  = true;
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type");
        }
    }
}

void StreamDcDownloader::_resetDataChunks(DataChunksDB& dataChunks)
{
    for (auto& pdataChunk : dataChunks)
    {
        pdataChunk->resetUsedChunkArea();
    }
}

void StreamDcDownloader::_updateStageDataChunksInfo(DataChunksDB&            cpDmaDataChunks,
                                                    uint32_t&                prevDataChunkIdx,
                                                    uint32_t                 curDataChunkIdx,
                                                    uint32_t&                curStageIdx,
                                                    std::vector<StagedInfo>& stagesInfo)
{
    StagedInfo& prevStage        = (curStageIdx == 0) ? stagesInfo[0] : stagesInfo[curStageIdx - 1];
    const auto& prevOffsetSizeDB = prevStage.offsetSizeInDc;
    StagedInfo& curStageInfo     = stagesInfo[curStageIdx];

    for (uint32_t dcIndex = prevDataChunkIdx; dcIndex <= curDataChunkIdx; dcIndex++)
    {
        uint64_t prevOffset    = prevOffsetSizeDB[dcIndex].offset + prevOffsetSizeDB[dcIndex].size;
        uint64_t chunkUsedSize = cpDmaDataChunks[dcIndex]->getUsedSize();
        HB_ASSERT((chunkUsedSize >= prevOffset), "invalid chunk offset");

        uint32_t sizeAdded = chunkUsedSize - prevOffset;

        curStageInfo.addDataChunkInfo(dcIndex, prevOffset, sizeAdded);

    }  // end(dcIndex)

    prevDataChunkIdx = curDataChunkIdx;
    curStageIdx++;
}

void StreamDcDownloader::downloadProgramCodeBuffer(uint64_t                        arbMasterBaseQmanId,
                                                   CommandSubmissionDataChunks&    csDataChunks,
                                                   std::vector<uint64_t>&          patchableBlobsDcAddresses,
                                                   DataChunksDB&                   cpDmaDataChunks,
                                                   uint64_t                        dcSizeCommand,
                                                   const recipe_t*                 recipeHandle,
                                                   const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                   const RecipeStaticInfo&         rRecipeStaticInfo,
                                                   const blob_t*                   blobsForCpDma,
                                                   eExecutionStage                 stage,
                                                   const uint8_t*                  patchableBlobsBuffer,
                                                   staticBlobsDeviceAddresses*     staticBlobsDevAddresses) const
{
    PROFILER_COLLECT_TIME()

    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    uint64_t arbCommandSize           = 0;
    uint64_t arbSetCmdHostAddress     = 0;
    uint64_t arbReleaseCmdHostAddress = 0;

    uint32_t curDataChunkIdx  = 0;
    uint32_t prevDataChunkIdx = 0;

    bool useArbitration = m_useArbitration && (stage == EXECUTION_STAGE_ENQUEUE);

    const blob_t* blobs = blobsForCpDma;

    _resetDataChunks(cpDmaDataChunks);
    if (useArbitration)
    {
        arbCommandSize = m_packetGenerator->getArbitrationCommandSize();
        bool status    = rRecipeStaticInfo.getArbitrationSetHostAddress(arbSetCmdHostAddress);
        UNUSED(status);
        HB_ASSERT((status == true), "Failed to get Arb-Set command");
        status = rRecipeStaticInfo.getArbitrationReleaseHostAddress(arbReleaseCmdHostAddress);
        HB_ASSERT((status == true), "Failed to get Arb-Release command");
    }

    const uint64_t jobs_nr =
        (stage == EXECUTION_STAGE_ENQUEUE) ? recipeHandle->execute_jobs_nr : recipeHandle->activate_jobs_nr;
    job_t* pCurrentJob = (stage == EXECUTION_STAGE_ENQUEUE) ? recipeHandle->execute_jobs : recipeHandle->activate_jobs;

    // init stage submission dbs
    std::vector<uint32_t> stagesNodes;
    bool stageSubmissionEnabled = GCFG_ENABLE_STAGED_SUBMISSION.value() && stage == EXECUTION_STAGE_ENQUEUE;
    if (stageSubmissionEnabled)
    {
        stagesNodes = rDeviceAgnosticRecipeInfo.m_recipeStageInfo.m_stagesNodes;
    }
    else
    {
        stagesNodes.push_back(recipeHandle->node_nr - 1);
    }

    std::vector<StagedInfo> stagesInfo;
    stagesInfo.resize(stagesNodes.size());
    for (auto& itrStageInfo : stagesInfo)
    {
        itrStageInfo.offsetSizeInDc.resize(cpDmaDataChunks.size());
    }

    if (stagesInfo.empty())
    {
        return;
    }

    stagesInfo.front().isFirstSubmission = true;
    stagesInfo.back().isLastSubmission   = true;

    job_t* pArbMasterJob = nullptr;

    for (uint64_t jobIndex = 0; jobIndex < jobs_nr; jobIndex++, pCurrentJob++)
    {
        uint64_t engineId     = pCurrentJob->engine_id;
        uint64_t programIndex = pCurrentJob->program_idx;

        if (m_qmansDef->isNonInternalCommandsDcQueueId(engineId))
        {
            continue;
        }

        bool isArbMasterQman = (engineId == arbMasterBaseQmanId);
        if (isArbMasterQman)
        {
            pArbMasterJob = pCurrentJob;
            continue;
        }

        _copySingleProgramToDataChunks(recipeHandle,
                                       blobs,
                                       blobsForCpDma,
                                       staticBlobsDevAddresses,
                                       csDataChunks,
                                       cpDmaDataChunks,
                                       patchableBlobsDcAddresses,
                                       rDeviceAgnosticRecipeInfo,
                                       rRecipeStaticInfo,
                                       stagesNodes,
                                       stagesInfo,
                                       engineId,
                                       programIndex,
                                       arbCommandSize,
                                       arbSetCmdHostAddress,
                                       arbReleaseCmdHostAddress,
                                       curDataChunkIdx,
                                       prevDataChunkIdx,
                                       m_physicalQueueOffset,
                                       stage,
                                       useArbitration,
                                       false, /* syncWithStreamMaster */
                                       false, /* isArbMasterQman */
                                       stageSubmissionEnabled);
    }  // for(jobIndex)

    HB_ASSERT_PTR(pArbMasterJob);

    if (pArbMasterJob != nullptr)
    {
        // In the Gaudi new sync scheme, we need to wrap the arbitration-master engine with Sync with Stream-Master QMAN
        bool syncBetweenStreamMasterAndArbMaster = (m_deviceType == synDeviceGaudi);
        bool syncWithStreamMaster                = syncBetweenStreamMasterAndArbMaster;

        _copySingleProgramToDataChunks(recipeHandle,
                                       blobs,
                                       blobsForCpDma,
                                       staticBlobsDevAddresses,
                                       csDataChunks,
                                       cpDmaDataChunks,
                                       patchableBlobsDcAddresses,
                                       rDeviceAgnosticRecipeInfo,
                                       rRecipeStaticInfo,
                                       stagesNodes,
                                       stagesInfo,
                                       pArbMasterJob->engine_id,
                                       pArbMasterJob->program_idx,
                                       arbCommandSize,
                                       arbSetCmdHostAddress,
                                       arbReleaseCmdHostAddress,
                                       curDataChunkIdx,
                                       prevDataChunkIdx,
                                       m_physicalQueueOffset,
                                       stage,
                                       useArbitration,
                                       syncWithStreamMaster,
                                       true, /* isArbMasterQman */
                                       stageSubmissionEnabled);
    }

    HB_ASSERT(curDataChunkIdx == cpDmaDataChunks.size(), "Wrong amount of DC allocated");
    csDataChunks.setDataChunksStagesInfo(stagesNodes, stagesInfo);
    PROFILER_MEASURE_TIME("copyProgramToDC")
}

void StreamDcDownloader::downloadPatchableBlobsBuffer(const uint8_t*         patchableBlobsBuffer,
                                                      uint64_t               patchableBlobsBufferSize,
                                                      DataChunksDB&          commandsDataChunks,
                                                      std::vector<uint64_t>& commandsDcsAddresses)
{
    LOG_TRACE(SYN_STREAM, "{}", HLLOG_FUNC);
    PROFILER_COLLECT_TIME()

    uint8_t* currentDataPointer = (uint8_t*)patchableBlobsBuffer;
    uint64_t copyRequestSize    = patchableBlobsBufferSize;

    if (commandsDataChunks.empty())
    {
        return;
    }

    auto commandsDataChunkItr    = commandsDataChunks.begin();
    auto commandsDataChunkEndItr = commandsDataChunks.end();
    UNUSED(commandsDataChunkEndItr);

    DataChunk* pCommandsDataChunk = *commandsDataChunkItr;
    HB_ASSERT((pCommandsDataChunk != nullptr), "Null-pointer as Commands' Data-Chunk");

    _resetDataChunks(commandsDataChunks);

    while (copyRequestSize != 0)
    {
        HB_ASSERT((commandsDataChunkItr != commandsDataChunkEndItr),
                  "No more Data Chunks to handle copy of Commands blobs");

        if (pCommandsDataChunk->getFreeSize() == 0)
        {
            commandsDataChunkItr++;
            HB_ASSERT((commandsDataChunkItr != commandsDataChunkEndItr),
                      "No more Data Chunks to handle copy of Commands blobs");
            pCommandsDataChunk = *commandsDataChunkItr;
        }
        uint64_t dcAddress = pCommandsDataChunk->getHandle();

        uint64_t copyExecutedSize = 0;
        bool     success = pCommandsDataChunk->fillChunkData(copyExecutedSize, currentDataPointer, copyRequestSize);
        UNUSED(success);
        HB_ASSERT(success, "fillChunkData failed");
        currentDataPointer += copyExecutedSize;
        copyRequestSize -= copyExecutedSize;

        // store the DC address
        commandsDcsAddresses.push_back(dcAddress);
    }

    PROFILER_MEASURE_TIME("copyPatchBlobsBuffToDc")
}

void StreamDcDownloader::_copyStaticBlobsCpDmaToDc(DataChunksDB&                cpDmaDataChunks,
                                                   uint32_t&                    curDataChunkIdx,
                                                   const blob_t*                pCurrentBlob,
                                                   uint64_t                     blobIndex,
                                                   CommandSubmissionDataChunks& csDataChunks,
                                                   uint64_t                     engineId,
                                                   const RecipeStaticInfo&      rRecipeStaticInfo,
                                                   staticBlobsDeviceAddresses*  staticBlobsDevAddresses) const
{
    const uint64_t                cpDmaCommandSize = m_packetGenerator->getCpDmaSize();
    BlobCpDmaHostAddresses        cpDmaHostAddresses;
    const BlobCpDmaHostAddresses* pCpDmaHostAddresses = nullptr;
    static char                   pCpDmaCommand[CP_DMA_PACKET_SIZE];

    // The staticBlobsDevAddresses is for WS related addresses
    if (staticBlobsDevAddresses == nullptr)
    {
        rRecipeStaticInfo.getProgramCodeBlobCpDmaAddress(blobIndex, pCpDmaHostAddresses);
        LOG_TRACE(SYN_STREAM, "{}: static blob index:{} size:{}", HLLOG_FUNC, blobIndex, pCurrentBlob->size);
    }
    else  // WS
    {
        HB_ASSERT(CP_DMA_PACKET_SIZE >= cpDmaCommandSize, "CP_DMA_PACKET_SIZE is too small !!");
        char*    pPacket     = pCpDmaCommand;
        uint64_t blobDevAddr = staticBlobsDevAddresses->at(blobIndex);
        LOG_TRACE(SYN_STREAM,
                  "{}: static blob index:{} size:{}, addr: 0x{:x}",
                  HLLOG_FUNC,
                  blobIndex,
                  pCurrentBlob->size,
                  blobDevAddr);
        m_packetGenerator->generateCpDma(pPacket, pCurrentBlob->size, 0, 0, 0, blobDevAddr, 0);

        cpDmaHostAddresses.push_back((uint64_t)pCpDmaCommand);
        pCpDmaHostAddresses = &cpDmaHostAddresses;
    }

    for (auto blobCpDmaAddrEntry : *pCpDmaHostAddresses)
    {
        _fillDataChunksAndStore(cpDmaDataChunks,
                                curDataChunkIdx,
                                (void*)blobCpDmaAddrEntry,
                                cpDmaCommandSize,
                                engineId,
                                csDataChunks);
    }
}

void StreamDcDownloader::_addPatchableBlobsSingleCpDmaToDc(const blob_t*                pCurrentBlob,
                                                           DataChunksDB&                cpDmaDataChunks,
                                                           uint32_t&                    curDataChunkIdx,
                                                           uint64_t                     engineId,
                                                           uint64_t                     blobIndex,
                                                           CommandSubmissionDataChunks& csDataChunks,
                                                           std::vector<uint64_t>&       patchableBlobsDcAddresses,
                                                           const RecipeStaticInfo&      rRecipeStaticInfo) const
{
    uint64_t copyRequestSize = pCurrentBlob->size;

    const uint64_t cpDmaCommandSize = m_packetGenerator->getCpDmaSize();
    LOG_TRACE(SYN_STREAM,
              "{}: blob index: {}, blob size:{} blob data:{:x}",
              HLLOG_FUNC,
              blobIndex,
              copyRequestSize,
              (uint64_t)pCurrentBlob->data);

    const patchableBlobOffsetInDc patchableBlobOffset = rRecipeStaticInfo.getPatchableBlobOffset(blobIndex);
    uint32_t                      dcIndex             = patchableBlobOffset.dcIndex;
    uint64_t                      blobsOffsetInDc     = patchableBlobOffset.offsetInDc;
    uint64_t                      blobsAddressInDc    = patchableBlobsDcAddresses.at(dcIndex) + blobsOffsetInDc;

    HB_ASSERT(cpDmaCommandSize <= CUR_DATA_CHUNK->getChunkSize(), "invalid data chunk size");

    if (CUR_DATA_CHUNK->getFreeSize() < cpDmaCommandSize)
    {
        _addSingleCpDmaMapping(CUR_DATA_CHUNK, engineId, csDataChunks);
        curDataChunkIdx++;
    }

    char* nextAddrInChunk = (char*)(CUR_DATA_CHUNK->getNextChunkAddress());
    m_packetGenerator->generateDefaultCpDma(nextAddrInChunk, copyRequestSize, blobsAddressInDc);
    CUR_DATA_CHUNK->updateUsedSize(cpDmaCommandSize);
}

void StreamDcDownloader::_addPatchableBlobsMultipleCpDmaToDc(const blob_t*                pCurrentBlob,
                                                             DataChunksDB&                cpDmaDataChunks,
                                                             uint32_t&                    curDataChunkIdx,
                                                             uint64_t                     engineId,
                                                             uint64_t                     blobIndex,
                                                             CommandSubmissionDataChunks& csDataChunks,
                                                             std::vector<uint64_t>&       patchableBlobsDcAddresses,
                                                             const uint8_t*               patchableBlobsBuffer,
                                                             uint64_t                     dcSizeCommand) const
{
    uint64_t copyRequestSize = pCurrentBlob->size;

    const uint64_t cpDmaCommandSize = m_packetGenerator->getCpDmaSize();
    LOG_TRACE(SYN_STREAM, "{}: blob index: {}, blob size:{}", HLLOG_FUNC, blobIndex, copyRequestSize);

    // check if the blob will be broken or not
    uint64_t offsetInPatchBuff = (uint64_t)(pCurrentBlob->data) - (uint64_t)patchableBlobsBuffer;
    uint32_t dcIndex           = offsetInPatchBuff / dcSizeCommand;
    uint64_t blobsOffsetInDc   = offsetInPatchBuff % dcSizeCommand;
    uint64_t endDCAddress      = patchableBlobsDcAddresses[dcIndex] + dcSizeCommand;
    uint64_t blobsAddressInDc  = patchableBlobsDcAddresses[dcIndex] + blobsOffsetInDc;

    uint64_t currentCpDmaSize = copyRequestSize;

    do
    {
        uint64_t leftSizeToCopy = 0;
        uint64_t leftSizeInDc   = endDCAddress - blobsAddressInDc;

        if (copyRequestSize > leftSizeInDc)
        {
            // represents the size left to copy in the next loop
            leftSizeToCopy = copyRequestSize - leftSizeInDc;
        }
        currentCpDmaSize = copyRequestSize - leftSizeToCopy;

        LOG_TRACE(SYN_STREAM,
                  "{}: engineId {} CP_DMA for blob - Index {} Host-Address: 0x{:x} mapped addr: 0x{:x} size:{}",
                  HLLOG_FUNC,
                  engineId,
                  blobIndex,
                  (uint64_t)(pCurrentBlob->data),
                  (uint64_t)blobsAddressInDc,
                  currentCpDmaSize);

        _fillDataChunksAndStoreForPatchable(cpDmaDataChunks,
                                            curDataChunkIdx,
                                            (uint64_t)blobsAddressInDc,
                                            currentCpDmaSize,
                                            cpDmaCommandSize,
                                            engineId,
                                            csDataChunks);

        copyRequestSize = leftSizeToCopy;
        if (dcIndex + 1 < patchableBlobsDcAddresses.size())
        {
            blobsAddressInDc = patchableBlobsDcAddresses[dcIndex + 1];
            endDCAddress     = blobsAddressInDc + dcSizeCommand;
            dcIndex++;
        }
    } while (copyRequestSize > 0);
}

void StreamDcDownloader::_addSingleCpDmaMapping(DataChunk*                   pCpDmaDataChunk,
                                                uint64_t                     engineId,
                                                CommandSubmissionDataChunks& csDataChunks)
{
    uint64_t nextCpDmaVirtualAddress = pCpDmaDataChunk->getHandle();
    void*    nextCpDmaHostAddress    = pCpDmaDataChunk->getChunkBuffer();

    // change this function in th CSDC class to map between engineId
    csDataChunks.addSingleCpDmaDataChunksMapping(engineId,
                                                 (uint64_t)nextCpDmaHostAddress,
                                                 (uint64_t)nextCpDmaVirtualAddress,
                                                 pCpDmaDataChunk->getUsedSize());
}

// if blobIndex is equal to -1 then it is a copy of an ARB packet
void StreamDcDownloader::_fillDataChunksAndStore(DataChunksDB&                cpDmaDataChunks,
                                                 uint32_t&                    curDataChunkIdx,
                                                 void*                        cpDmaPktAddr,
                                                 uint64_t                     pktSize,
                                                 uint64_t                     engineId,
                                                 CommandSubmissionDataChunks& csDataChunks) const
{
    HB_ASSERT(pktSize <= CUR_DATA_CHUNK->getChunkSize(), "invalid data chunk size");

    if (CUR_DATA_CHUNK->getFreeSize() < pktSize)
    {
        _addSingleCpDmaMapping(CUR_DATA_CHUNK, engineId, csDataChunks);
        curDataChunkIdx++;
    }

    CUR_DATA_CHUNK->fillChunkData(cpDmaPktAddr, pktSize);
}

void StreamDcDownloader::_fillDataChunksAndStoreForPatchable(DataChunksDB&                cpDmaDataChunks,
                                                             uint32_t&                    curDataChunkIdx,
                                                             uint64_t                     blobsAddressInDc,
                                                             uint64_t                     currentCpDmaSize,
                                                             uint64_t                     pktSize,
                                                             uint64_t                     engineId,
                                                             CommandSubmissionDataChunks& csDataChunks) const
{
    HB_ASSERT(pktSize <= CUR_DATA_CHUNK->getChunkSize(), "invalid chunk size");

    if (CUR_DATA_CHUNK->getFreeSize() < pktSize)
    {
        _addSingleCpDmaMapping(CUR_DATA_CHUNK, engineId, csDataChunks);
        curDataChunkIdx++;
    }

    char* nextAddrInChunk = (char*)(CUR_DATA_CHUNK->getNextChunkAddress());
    m_packetGenerator->generateDefaultCpDma(nextAddrInChunk, currentCpDmaSize, blobsAddressInDc);
    CUR_DATA_CHUNK->updateUsedSize(pktSize);
}

void StreamDcDownloader::_copySingleProgramToDataChunks(const recipe_t*                 recipeHandle,
                                                        const blob_t*                   blobs,
                                                        const blob_t*                   blobsForCpDma,
                                                        staticBlobsDeviceAddresses*     staticBlobsDevAddresses,
                                                        CommandSubmissionDataChunks&    csDataChunks,
                                                        DataChunksDB&                   cpDmaDataChunks,
                                                        std::vector<uint64_t>&          patchableBlobsDcAddresses,
                                                        const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                        const RecipeStaticInfo&         rRecipeStaticInfo,
                                                        std::vector<uint32_t>&          stagesNodes,
                                                        std::vector<StagedInfo>&        stagesInfo,
                                                        uint64_t                        engineId,
                                                        uint64_t                        programIndex,
                                                        uint64_t                        arbCommandSize,
                                                        uint64_t                        arbSetCmdHostAddress,
                                                        uint64_t                        arbReleaseCmdHostAddress,
                                                        uint32_t&                       curDataChunkIdx,
                                                        uint32_t&                       prevDataChunkIdx,
                                                        uint32_t                        physicalQueueOffset,
                                                        eExecutionStage                 stage,
                                                        bool                            useArbitration,
                                                        bool                            syncWithStreamMaster,
                                                        bool                            isArbMasterQman,
                                                        bool                            stageSubmissionEnabled) const
{
    // in the new sync scheme, we need to preface with a Fence Msg
    // Add Wait for sync on arb-master from stream-master
    if (syncWithStreamMaster)
    {
        // create the Fence packet
        // Direction is:  ARB-Master -> Stream-Master
        char*    pFencePacketBuffer    = nullptr;
        uint64_t fencePacketBufferSize = 0;
        m_packetGenerator->generateFenceCommand(pFencePacketBuffer, fencePacketBufferSize);
        CUR_DATA_CHUNK->fillChunkData((void*)pFencePacketBuffer, fencePacketBufferSize);

        delete[] pFencePacketBuffer;
    }

    // Add Arb-Request
    if (useArbitration)
    {
        CUR_DATA_CHUNK->fillChunkData((void*)arbSetCmdHostAddress, arbCommandSize);
    }

    std::vector<uint64_t>::const_iterator currentPatchingBlobsDeviceAddrIter;

    // Add PRG-Code content (CP-DMAs)
    program_t currentProgram = recipeHandle->programs[programIndex];
    uint64_t  programLength  = currentProgram.program_length;

    uint32_t nodeNr      = (stageSubmissionEnabled) ? recipeHandle->node_nr : 1;
    uint32_t curStageIdx = 0;
    uint64_t nodeBlobIdx = 0;
    uint32_t node        = 0;

    for (; node < nodeNr; node++)
    {
        uint64_t programBlobsNr =
            stageSubmissionEnabled ? recipeHandle->node_exe_list[node].program_blobs_nr[programIndex] : programLength;

        if (stageSubmissionEnabled && (node == stagesNodes[curStageIdx] + 1))  // stage is over
        {
            _updateStageDataChunksInfo(cpDmaDataChunks, prevDataChunkIdx, curDataChunkIdx, curStageIdx, stagesInfo);
        }

        for (; nodeBlobIdx < programBlobsNr; nodeBlobIdx++)
        {
            uint64_t      blobIndex    = currentProgram.blob_indices[nodeBlobIdx];
            const blob_t* pCurrentBlob = &blobs[blobIndex];

            if (!pCurrentBlob->blob_type.requires_patching)
            {
                _copyStaticBlobsCpDmaToDc(cpDmaDataChunks,
                                          curDataChunkIdx,
                                          pCurrentBlob,
                                          blobIndex,
                                          csDataChunks,
                                          engineId,
                                          rRecipeStaticInfo,
                                          staticBlobsDevAddresses);
                continue;
            }

            _addPatchableBlobsSingleCpDmaToDc(pCurrentBlob,
                                              cpDmaDataChunks,
                                              curDataChunkIdx,
                                              engineId,
                                              blobIndex,
                                              csDataChunks,
                                              patchableBlobsDcAddresses,
                                              rRecipeStaticInfo);
        }  // for nodeBlobIdx
    }      // for(Node)

    HB_ASSERT((!stageSubmissionEnabled) || (node == stagesNodes[curStageIdx] + 1), "lastNode must be a stage barrier");

    // Add work completion context (on arb-master only)
    if (isArbMasterQman)
    {
        uint64_t   workCompletionProgramIndex;
        const bool isSet =
            rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.getWorkCompletionProgramIndex(stage,
                                                                                       workCompletionProgramIndex);
        if (isSet)
        {
            program_t* workCompletionCurrentProgram = &(recipeHandle->programs[workCompletionProgramIndex]);
            uint64_t   workCompletionProgramLength  = workCompletionCurrentProgram->program_length;

            LOG_TRACE(SYN_STREAM, "{}: engineId {} work completion added", HLLOG_FUNC, engineId);
            for (int j = 0; j < workCompletionProgramLength; j++)
            {
                uint64_t      blobIndex    = workCompletionCurrentProgram->blob_indices[j];
                const blob_t* pCurrentBlob = &blobsForCpDma[blobIndex];

                HB_ASSERT((pCurrentBlob != nullptr), "Current blob is null pointer");

                _fillDataChunksAndStore(cpDmaDataChunks,
                                        curDataChunkIdx,
                                        pCurrentBlob->data,
                                        pCurrentBlob->size,
                                        engineId,
                                        csDataChunks);
            }
        }
    }

    // Add Arb-Release
    if (useArbitration)
    {
        _fillDataChunksAndStore(cpDmaDataChunks,
                                curDataChunkIdx,
                                (void*)arbReleaseCmdHostAddress,
                                arbCommandSize,
                                engineId,
                                csDataChunks);
    }

    // Add Signal from arb-master to stream-master
    if (syncWithStreamMaster)
    {
        // create the FenceClear (MsgLong) packet
        // Direction is:  ARB-Master -> Stream-Master
        uint64_t streamMasterStreamId = m_qmansDef->getStreamMasterQueueIdForCompute() + physicalQueueOffset;

        char*    pFenceClearPacketToStreamMasterPacketBuffer = nullptr;
        uint64_t fenceClearPacketToStreamMasterBufferSize;

        m_packetGenerator->generateFenceClearCommand(pFenceClearPacketToStreamMasterPacketBuffer,
                                                     fenceClearPacketToStreamMasterBufferSize,
                                                     streamMasterStreamId);

        _fillDataChunksAndStore(cpDmaDataChunks,
                                curDataChunkIdx,
                                (void*)pFenceClearPacketToStreamMasterPacketBuffer,
                                fenceClearPacketToStreamMasterBufferSize,
                                engineId,
                                csDataChunks);
        delete[] pFenceClearPacketToStreamMasterPacketBuffer;
    }

    // update the last stage
    if (stageSubmissionEnabled)
    {
        _updateStageDataChunksInfo(cpDmaDataChunks, prevDataChunkIdx, curDataChunkIdx, curStageIdx, stagesInfo);
    }

    if (CUR_DATA_CHUNK->getUsedSize() != 0)
    {
        _addSingleCpDmaMapping(CUR_DATA_CHUNK, engineId, csDataChunks);
        curDataChunkIdx++;
        prevDataChunkIdx = curDataChunkIdx;
    }
}

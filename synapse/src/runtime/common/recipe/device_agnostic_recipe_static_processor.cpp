#include "device_agnostic_recipe_static_processor.hpp"
#include "device_agnostic_recipe_static_info.hpp"
#include "global_statistics.hpp"
#include "recipe.h"
#include "log_manager.h"
#include "habana_global_conf_runtime.h"
#include "defs.h"
#include "utils.h"
#include "runtime/qman/common/master_qmans_definition_interface.hpp"
#include "runtime/qman/gaudi/device_gaudi.hpp"
#include "runtime/qman/common/command_buffer_packet_generator.hpp"

synStatus DeviceAgnosticRecipeStaticProcessor::process(const recipe_t&                 rRecipe,
                                                       DeviceAgnosticRecipeStaticInfo& rRecipeInfo,
                                                       synDeviceType                   deviceType)
{
    const uint64_t dcSizeCommand = GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP.value() * 1024;

    bool ret = calculatePatchableBufferSizeAndChunksAmount(rRecipe, rRecipeInfo, dcSizeCommand);
    if (!ret)
    {
        LOG_ERR(SYN_RECIPE, "{}: Failed to during commands-buffer chunks' amount calculation", HLLOG_FUNC);
        return synFail;
    }

    setWorkCompletionProgramIndex(rRecipe, rRecipeInfo, deviceType);

    ret = validateWorkCompletionQueue(rRecipe, deviceType);
    if (!ret)
    {
        LOG_ERR(SYN_RECIPE, "{}: Failed to validateWorkCompletionQueue", HLLOG_FUNC);
        return synFail;
    }

    return synSuccess;
}

// TODO - split the DC amount between the different stages
bool DeviceAgnosticRecipeStaticProcessor::calculatePatchableBufferSizeAndChunksAmount(
    const recipe_t&                 rRecipe,
    DeviceAgnosticRecipeStaticInfo& rRecipeInfo,
    uint64_t                        dcSizeCommand)
{
    uint64_t& rPatchableBlobsChunkBufferSize       = rRecipeInfo.m_patchingBlobsChunksSize;
    uint64_t& rPatchableBlobsChunkDataChunksAmount = rRecipeInfo.m_patchingBlobsChunksDataChunksAmount;

    rPatchableBlobsChunkBufferSize = rRecipe.patching_blobs_buffer_size;
    if (rPatchableBlobsChunkBufferSize == 0)
    {
        rPatchableBlobsChunkDataChunksAmount = 0;
    }
    else
    {
        rPatchableBlobsChunkDataChunksAmount = 1 + ((rPatchableBlobsChunkBufferSize - 1) / dcSizeCommand);
    }

    const blob_t* pCurrentRecipeBlob = rRecipe.blobs;
    for (uint64_t blobIndex = 0; blobIndex < rRecipe.blobs_nr; blobIndex++, pCurrentRecipeBlob++)
    {
        if (!pCurrentRecipeBlob->blob_type.requires_patching)
        {
            continue;
        }

        uint32_t currentBlobsSize = pCurrentRecipeBlob->size;
        if (currentBlobsSize > dcSizeCommand)
        {
            LOG_CRITICAL(SYN_RECIPE,
                         "Current Blob {} (with size 0x{:x}) does not fit a single Chunk (with size of 0x{:x})",
                         blobIndex,
                         currentBlobsSize,
                         dcSizeCommand);
            return false;
        }
    }

    // Todo SW-76379 splitting DCs per execution-stage
    rRecipeInfo.setProgramCommandsChunksAmount(EXECUTION_STAGE_ENQUEUE, rPatchableBlobsChunkDataChunksAmount);
    rRecipeInfo.setProgramCommandsChunksAmount(EXECUTION_STAGE_ACTIVATE, rPatchableBlobsChunkDataChunksAmount);
    rRecipeInfo.setDcSizeCommand(dcSizeCommand);
    STAT_GLBL_COLLECT(dcSizeCommand, dcSizeCommand);
    STAT_GLBL_COLLECT(rPatchableBlobsChunkDataChunksAmount, dcAmountCommand);
    STAT_GLBL_COLLECT(rPatchableBlobsChunkBufferSize, accumulateBlobsChunksSize);

    LOG_DEBUG_T(SYN_RECIPE,
                "Required Commands Data-Chunks {} (chunks-size 0x{:x})",
                rPatchableBlobsChunkDataChunksAmount,
                dcSizeCommand);
    return true;
}

void DeviceAgnosticRecipeStaticProcessor::setWorkCompletionProgramIndex(const recipe_t&                 rRecipe,
                                                                        DeviceAgnosticRecipeStaticInfo& rRecipeInfo,
                                                                        synDeviceType                   deviceType)
{
    std::map<uint64_t, bool> internalQueuesPrograms;
    static const size_t      numOfExecutionStages(EXECUTION_STAGE_LAST);
    const job_t*             pCurrentJob[numOfExecutionStages];
    uint32_t                 currentJobsNr[numOfExecutionStages];
    eExecutionStage          currentExecutionStage[numOfExecutionStages];
    pCurrentJob[EXECUTION_STAGE_ENQUEUE]            = rRecipe.execute_jobs;
    currentJobsNr[EXECUTION_STAGE_ENQUEUE]          = rRecipe.execute_jobs_nr;
    currentExecutionStage[EXECUTION_STAGE_ENQUEUE]  = EXECUTION_STAGE_ENQUEUE;
    pCurrentJob[EXECUTION_STAGE_ACTIVATE]           = rRecipe.activate_jobs;
    currentJobsNr[EXECUTION_STAGE_ACTIVATE]         = rRecipe.activate_jobs_nr;
    currentExecutionStage[EXECUTION_STAGE_ACTIVATE] = EXECUTION_STAGE_ACTIVATE;

    QmanDefinitionInterface* pQmansDef = getQmansDefinition(deviceType);

    for (size_t stageIdx = 0; stageIdx < numOfExecutionStages; stageIdx++)
    {
        for (uint64_t jobIndex = 0; jobIndex < currentJobsNr[stageIdx]; jobIndex++, pCurrentJob[stageIdx]++)
        {
            const uint32_t engineId     = pCurrentJob[stageIdx]->engine_id;
            const uint32_t programIndex = pCurrentJob[stageIdx]->program_idx;

            if ((deviceType == synDeviceGaudi) && (pQmansDef->isStreamMasterQueueIdForCompute(engineId)))
            {
                continue;
            }

            if (internalQueuesPrograms.find(programIndex) != internalQueuesPrograms.end())
            {
                continue;
            }

            internalQueuesPrograms[programIndex] = true;

            if (pQmansDef->isWorkCompletionQueueId(engineId))
            {
                // GC sends the completion work blob on this queue
                rRecipeInfo.setWorkCompletionProgramIndex(currentExecutionStage[stageIdx], programIndex);
            }
        }
    }
}

bool DeviceAgnosticRecipeStaticProcessor::validateWorkCompletionQueue(const recipe_t& rRecipe, synDeviceType deviceType)
{
    const uint32_t                      amountOfEnginesInArbGroup = getAmountOfEnginesInArbGroup(deviceType);
    QmanDefinitionInterface*            pQmansDef                 = getQmansDefinition(deviceType);
    generic::CommandBufferPktGenerator* pCommandBufferPktGenerator =
        generic::CommandBufferPktGenerator::getCommandBufferPktGenerator(deviceType);

    uint16_t numOfEngines           = 0;
    bool     isWorkCompletionExists = false;

    std::array<unsigned, ENGINES_ID_SIZE_MAX> isEngineJobExists = {};

    // TODO - same for the activate part
    for (unsigned i = 0; i < rRecipe.execute_jobs_nr; i++)
    {
        if (rRecipe.execute_jobs[i].engine_id == pQmansDef->getWorkCompletionQueueId())
        {
            isWorkCompletionExists = true;
            continue;
        }

        const uint32_t qmanId = pCommandBufferPktGenerator->getQmanId(rRecipe.execute_jobs[i].engine_id);
        if (isEngineJobExists[qmanId] == 0)
        {
            isEngineJobExists[qmanId] = 1;
            numOfEngines++;
        }
    }

    if (!isWorkCompletionExists)
    {
        LOG_ERR(SYN_RECIPE, "Work completion queue had not been defined");
        return false;
    }

    if (numOfEngines != amountOfEnginesInArbGroup)
    {
        LOG_ERR(SYN_RECIPE,
                "Mismatch between amount of engines recipe {} arb-group {}",
                numOfEngines,
                amountOfEnginesInArbGroup);
        return false;
    }

    return true;
}

uint32_t DeviceAgnosticRecipeStaticProcessor::getAmountOfEnginesInArbGroup(synDeviceType deviceType)
{
    uint32_t amountOfEnginesInArbGroup = 0;

    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            amountOfEnginesInArbGroup = DeviceGaudi::getAmountOfEnginesInComputeArbGroupAquire();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Device is not supported {}", deviceType);
        }
    }
    return amountOfEnginesInArbGroup;
}

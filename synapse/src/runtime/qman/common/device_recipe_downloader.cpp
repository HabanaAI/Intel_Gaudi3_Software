#include "device_recipe_downloader.hpp"

#include "device_recipe_addresses_generator_interface.hpp"
#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "profiler_api.hpp"
#include "runtime/qman/common/recipe_program_buffer.hpp"

#include "runtime/common/common_types.hpp"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/qman/common/master_qmans_definition_interface.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "runtime/qman/common/command_buffer_packet_generator.hpp"
#include "runtime/qman/common/device_downloader_interface.hpp"
#include "runtime/qman/common/device_mapper_interface.hpp"
#include "runtime/qman/common/static_info_processor.hpp"
#include "recipe/recipe_utils.hpp"
DeviceRecipeDownloader::DeviceRecipeDownloader(synDeviceType                             deviceType,
                                               const QmanDefinitionInterface&            rQmansDefinition,
                                               const generic::CommandBufferPktGenerator& rCmdBuffPktGenerator,
                                               DeviceDownloaderInterface&                rDeviceDownloader,
                                               DeviceMapperInterface&                    rDeviceMapper,
                                               const DeviceAgnosticRecipeInfo&           rDeviceAgnosticRecipeInfo,
                                               const basicRecipeInfo&                    rBasicRecipeInfo,
                                               uint64_t                                  recipeId)
: m_deviceType(deviceType),
  m_rQmansDefinition(rQmansDefinition),
  m_rCmdBuffPktGenerator(rCmdBuffPktGenerator),
  m_rDeviceDownloader(rDeviceDownloader),
  m_rDeviceMapper(rDeviceMapper),
  m_rBasicRecipeInfo(rBasicRecipeInfo),
  m_rDeviceAgnosticRecipeInfo(rDeviceAgnosticRecipeInfo),
  m_recipeStaticInfo(),
  m_recipeId(recipeId)
{
}

synStatus DeviceRecipeDownloader::processRecipe(uint64_t               workspaceAddress,
                                                uint64_t               dcSizeCpDma,
                                                uint64_t               dcSizeCommand,
                                                bool                   programCodeInCache,
                                                uint64_t               programCodeHandle,
                                                uint64_t               programCodeDeviceAddress,
                                                std::vector<uint64_t>& rProgramCodeDeviceAddresses)
{
    return _processRecipe(m_deviceType,
                          m_rDeviceMapper,
                          dcSizeCpDma,
                          dcSizeCommand,
                          m_rBasicRecipeInfo,
                          m_rDeviceAgnosticRecipeInfo,
                          workspaceAddress,
                          &m_recipeStaticInfo,
                          &m_processingSingleExecutionOwner,
                          rProgramCodeDeviceAddresses,
                          programCodeDeviceAddress,
                          programCodeInCache,
                          programCodeHandle);
}

synStatus DeviceRecipeDownloader::downloadExecutionBufferCache(bool                   programCodeInCache,
                                                               uint64_t               programCodeHandle,
                                                               std::vector<uint64_t>& rProgramCodeDeviceAddresses)
{
    return _downloadExecutionBufferCache(m_recipeId,
                                         &m_rDeviceDownloader,
                                         m_rBasicRecipeInfo,
                                         &m_prgCodeDwlToCacheSingleExecutionOwner,
                                         rProgramCodeDeviceAddresses,
                                         programCodeInCache,
                                         programCodeHandle);
}

synStatus DeviceRecipeDownloader::downloadProgramDataBufferCache(uint64_t              workspaceAddress,
                                                                 bool                  programDataInCache,
                                                                 uint64_t              programDataHandle,
                                                                 uint64_t              programDataDeviceAddress,
                                                                 SpRecipeProgramBuffer programDataRecipeBuffer)
{
    return _downloadProgramDataBufferToCache(m_rDeviceDownloader,
                                             m_rBasicRecipeInfo,
                                             workspaceAddress,
                                             &m_recipeStaticInfo,
                                             &m_prgDataDwlToCacheSingleExecutionOwner,
                                             programDataInCache,
                                             programDataHandle,
                                             programDataDeviceAddress,
                                             programDataRecipeBuffer);
}

synStatus DeviceRecipeDownloader::downloadExecutionBufferWorkspace(QueueInterface* pComputeStream,
                                                                   bool            programCodeInCache,
                                                                   uint64_t        programCodeHandle,
                                                                   uint64_t        programCodeDeviceAddress,
                                                                   bool&           rIsDownloadWorkspaceProgramCode)
{
    rIsDownloadWorkspaceProgramCode = false;

    if (programCodeInCache)
    {
        return synSuccess;
    }

    if (programCodeDeviceAddress == INVALID_DEVICE_ADDR)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid PRG CODE address.", HLLOG_FUNC);
        return synFail;
    }

    const uint64_t programCodeSize = m_rBasicRecipeInfo.recipe->execution_blobs_buffer_size;
    if (programCodeSize == 0)
    {
        LOG_TRACE(SYN_STREAM, "Stream {:#x} PRG data is 0", TO64(this));
        return synSuccess;
    }

    LOG_DEBUG(SYN_STREAM,
              "Stream {:#x} Need to download program code to workspace, programCodeDeviceAddress:{:#x}",
              TO64(this),
              programCodeDeviceAddress);

    PROFILER_COLLECT_TIME()
    STAT_GLBL_START(downloadPrgCode);

    const uint64_t        programCodeHostAddress = (uint64_t)m_rBasicRecipeInfo.recipe->execution_blobs_buffer;
    internalMemcopyParams memcpyParams           = {
        {.src = programCodeHostAddress, .dst = programCodeDeviceAddress, .size = programCodeSize}};

    const synStatus status =
        m_rDeviceDownloader.downloadProgramCodeBuffer(m_recipeId, pComputeStream, memcpyParams, programCodeSize);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to copy program code to workspace programCodeHostAddress {:#x} "
                "programCodeDeviceAddress {:#x} programCodeSize {:#x}",
                HLLOG_FUNC,
                programCodeHostAddress,
                programCodeDeviceAddress,
                programCodeSize);
        return status;
    }

    rIsDownloadWorkspaceProgramCode = true;

    LOG_DEBUG(SYN_STREAM,
              "Stream {:#x} patch needed, programCodeDeviceAddress {:#x}",
              TO64(this),
              programCodeDeviceAddress);

    STAT_GLBL_COLLECT_TIME(downloadPrgCode, globalStatPointsEnum::downloadPrgCode);
    PROFILER_MEASURE_TIME("downloadExecutionBufferWorkspace")

    return synSuccess;
}

synStatus DeviceRecipeDownloader::downloadProgramDataBufferWorkspace(QueueInterface* pComputeStream,
                                                                     bool            programDataInCache,
                                                                     uint64_t        programDataHandle,
                                                                     uint64_t        programDataDeviceAddress,
                                                                     bool&           rIsDownloadWorkspaceProgramData,
                                                                     SpRecipeProgramBuffer programDataRecipeBuffer)
{
    rIsDownloadWorkspaceProgramData = false;

    if (programDataInCache)
    {
        return synSuccess;
    }

    if (programDataDeviceAddress == INVALID_DEVICE_ADDR)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid PRG data address.", HLLOG_FUNC);
        return synFail;
    }

    const uint64_t programDataSize = m_rBasicRecipeInfo.recipe->program_data_blobs_size;

    if (programDataSize == 0)
    {
        LOG_TRACE(SYN_STREAM, "Stream {:#x} PRG data is 0", TO64(this));
        return synSuccess;
    }

    LOG_DEBUG(SYN_STREAM,
              "Stream {:#x} Need to download program data to workspace, programDataDeviceAddress:{:#x}",
              TO64(this),
              programDataDeviceAddress);

    PROFILER_COLLECT_TIME()
    STAT_GLBL_START(downloadPrgData);

    internalMemcopyParams memcpyParams({{.src  = (uint64_t)programDataRecipeBuffer->getBuffer(),
                                         .dst  = programDataDeviceAddress,
                                         .size = programDataSize}});

    const synStatus status =
        m_rDeviceDownloader.downloadProgramDataBuffer(pComputeStream, memcpyParams, &programDataRecipeBuffer);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to copy program data to workspace programDataHostAddress {:#x} "
                "programDataDeviceAddress {:#x} programDataSize {:#x}",
                HLLOG_FUNC,
                (uint64_t)programDataRecipeBuffer->getBuffer(),
                programDataDeviceAddress,
                programDataSize);

        return status;
    }

    rIsDownloadWorkspaceProgramData = true;

    LOG_DEBUG(SYN_STREAM,
              "Stream {:#x} patch needed, programDataDeviceAddress {:#x}",
              TO64(this),
              programDataDeviceAddress);

    STAT_GLBL_COLLECT_TIME(downloadPrgData, globalStatPointsEnum::downloadPrgData);
    PROFILER_MEASURE_TIME("downloadProgramDataBufferWorkspace")

    return synSuccess;
}

void DeviceRecipeDownloader::notifyRecipeDestroy()
{
    StaticInfoProcessor::destroyProcessor(m_rDeviceMapper, m_recipeStaticInfo);
}

synStatus DeviceRecipeDownloader::_processRecipe(synDeviceType                   deviceType,
                                                 DeviceMapperInterface&          rDeviceMapper,
                                                 uint64_t                        dcSizeCpDma,
                                                 uint64_t                        dcSizeCommand,
                                                 const basicRecipeInfo&          rBasicRecipeInfo,
                                                 const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                 uint64_t                        workspaceAddress,
                                                 RecipeStaticInfo*               pRecipeInfo,
                                                 SingleExecutionOwner*           pSingleExecutionOwner,
                                                 std::vector<uint64_t>&          rProgramCodeDeviceAddresses,
                                                 uint64_t                        programCodeDeviceAddress,
                                                 bool                            programCodeInCache,
                                                 uint64_t                        programCodeHandle)
{
    bool shouldForce      = !programCodeInCache;
    bool isOwnershipTaken = pSingleExecutionOwner->takeOwnership(
        shouldForce ? SingleExecutionOwner::FORCE_OPERATION_OWNERSHIP_ID : programCodeHandle);
    if (!isOwnershipTaken)
    {
        return synSuccess;
    }

    bool ret = StaticInfoProcessor::allocateResourcesAndProcessRecipe(&rDeviceMapper,
                                                                      rBasicRecipeInfo,
                                                                      rDeviceAgnosticRecipeInfo,
                                                                      deviceType,
                                                                      *pRecipeInfo,
                                                                      rProgramCodeDeviceAddresses,
                                                                      programCodeDeviceAddress,
                                                                      programCodeInCache,
                                                                      workspaceAddress,
                                                                      dcSizeCpDma,
                                                                      dcSizeCommand);

    synStatus status      = (ret) ? synSuccess : synAllResourcesTaken;
    bool      shouldAbort = (shouldForce) || (!ret);
    pSingleExecutionOwner->releaseOwnership(shouldAbort ? SingleExecutionOwner::ABORT_OPERATION_OWNERSHIP_ID
                                                        : programCodeHandle);

    return status;
}

synStatus DeviceRecipeDownloader::_downloadExecutionBufferCache(uint64_t                   recipeId,
                                                                DeviceDownloaderInterface* pDeviceDownloader,
                                                                const basicRecipeInfo&     rBasicRecipeInfo,
                                                                SingleExecutionOwner*      pSingleExecutionOwner,
                                                                std::vector<uint64_t>&     rProgramCodeDeviceAddresses,
                                                                bool                       programCodeInCache,
                                                                uint64_t                   programCodeHandle)
{
    if (!programCodeInCache)
    {
        return synSuccess;
    }

    bool shouldForce      = false;
    bool isOwnershipTaken = pSingleExecutionOwner->takeOwnership(
        shouldForce ? SingleExecutionOwner::FORCE_OPERATION_OWNERSHIP_ID : programCodeHandle);
    if (!isOwnershipTaken)
    {
        return synSuccess;
    }

    synStatus status = synSuccess;

    PROFILER_COLLECT_TIME()
    STAT_GLBL_START(downloadPrgCode);
    if (rBasicRecipeInfo.recipe->execution_blobs_buffer_size > 0)
    {
        internalMemcopyParams memcpyParams;
        uint64_t              hostBufferSize;
        _generateDownloadProgramCodeCacheMemCopyParams(*rBasicRecipeInfo.recipe,
                                                       rProgramCodeDeviceAddresses,
                                                       memcpyParams,
                                                       hostBufferSize);

        const bool status =
            pDeviceDownloader->downloadProgramCodeBuffer(recipeId, nullptr, memcpyParams, hostBufferSize);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to copy program-code to device", HLLOG_FUNC);
            // Todo deallocate resources taken in allocateDeviceResources
        }
    }
    STAT_GLBL_COLLECT_TIME(downloadPrgCode, globalStatPointsEnum::downloadPrgCode);
    PROFILER_MEASURE_TIME("downPCAndUpdateRecipe")

    bool shouldAbort = (shouldForce) || (status != synSuccess);
    pSingleExecutionOwner->releaseOwnership(shouldAbort ? SingleExecutionOwner::ABORT_OPERATION_OWNERSHIP_ID
                                                        : programCodeHandle);

    return status;
}

void DeviceRecipeDownloader::_generateDownloadProgramCodeCacheMemCopyParams(
    const recipe_t&              rRecipe,
    const std::vector<uint64_t>& rProgramCodeDeviceAddresses,
    internalMemcopyParams&       rMemcpyParams,
    uint64_t&                    hostBufferSize)
{
    uint64_t* pExecutionBlobsBuffer    = rRecipe.execution_blobs_buffer;
    uint64_t  executionBlobsBufferSize = rRecipe.execution_blobs_buffer_size;
    uint64_t  blockSize                = 1024 * GCFG_RECIPE_CACHE_BLOCK_SIZE.value();

    hostBufferSize = rRecipe.execution_blobs_buffer_size;

    std::vector<uint64_t>::const_iterator blocksAddrIterator = rProgramCodeDeviceAddresses.begin();
    std::vector<uint64_t>::const_iterator blockAddrEnd       = rProgramCodeDeviceAddresses.end();

    // Allocate mapped buffer
    uint8_t* pStaticBlobsBuffer = (uint8_t*)pExecutionBlobsBuffer;

    rMemcpyParams.reserve(rMemcpyParams.size() + rProgramCodeDeviceAddresses.size());
    const uint32_t numOfBlocks    = rProgramCodeDeviceAddresses.size();
    const uint32_t leftover       = executionBlobsBufferSize % blockSize;
    bool           contiguousArea = true;
    if (numOfBlocks > 1)
    {
        uint32_t areaStartIdx = 0;
        auto     addr         = rProgramCodeDeviceAddresses[0];
        for (uint32_t j = 1; j < rProgramCodeDeviceAddresses.size(); ++j)
        {
            if (addr + (j - areaStartIdx) * blockSize != rProgramCodeDeviceAddresses[j])
            {
                contiguousArea = false;
                break;
            }
        }
    }
    if (contiguousArea)
    {
        rMemcpyParams.push_back({.src  = (uint64_t)pStaticBlobsBuffer,
                                 .dst  = rProgramCodeDeviceAddresses[0],
                                 .size = executionBlobsBufferSize});
    }
    else
    {
        uint32_t i;
        for (i = 1, blocksAddrIterator = rProgramCodeDeviceAddresses.begin(); blocksAddrIterator != blockAddrEnd;
             blocksAddrIterator++, i++)
        {
            if (i == numOfBlocks && leftover != 0)
            {
                blockSize = leftover;
            }

            rMemcpyParams.push_back(
                {.src = (uint64_t)pStaticBlobsBuffer, .dst = *blocksAddrIterator, .size = blockSize});

            pStaticBlobsBuffer += blockSize;
        }
    }
}

synStatus DeviceRecipeDownloader::_downloadProgramDataBufferToCache(const DeviceDownloaderInterface& rDownloader,
                                                                    const basicRecipeInfo&           rBasicRecipeInfo,
                                                                    uint64_t                         workspaceAddress,
                                                                    RecipeStaticInfo*                pRecipeInfo,
                                                                    SingleExecutionOwner* pSingleExecutionOwner,
                                                                    bool                  programDataInCache,
                                                                    uint64_t              programDataHandle,
                                                                    uint64_t              programDataDeviceAddress,
                                                                    SpRecipeProgramBuffer programDataRecipeBuffer)
{
    if (!programDataInCache)
    {
        return synSuccess;
    }

    bool isKernelPrintf = RecipeUtils::isKernelPrintf(rBasicRecipeInfo);

    /* we want to download again so the TPC printf buffer will be cleared*/
    uint64_t ownershipId = programDataHandle;
    if (isKernelPrintf)
    {
        LOG_TRACE(SYN_STREAM, "{} tpc kernel includes printf, download it to recipeCache, programDataHandle {}", HLLOG_FUNC, programDataHandle);
        ownershipId = SingleExecutionOwner::FORCE_OPERATION_OWNERSHIP_ID;
    }
    const bool isOwnershipTaken = pSingleExecutionOwner->takeOwnership(ownershipId);
    if (!isOwnershipTaken)
    {
        return synSuccess;
    }

    synStatus status;

    recipe_t*      pRecipe              = rBasicRecipeInfo.recipe;
    const uint64_t programDataBlobsSize = pRecipe->program_data_blobs_size;
    if (programDataBlobsSize > 0)
    {
        internalMemcopyParams memcpyParams({{.src  = (uint64_t)programDataRecipeBuffer->getBuffer(),
                                             .dst  = programDataDeviceAddress,
                                             .size = programDataBlobsSize}});

        status = rDownloader.downloadProgramDataBuffer(nullptr, memcpyParams, &programDataRecipeBuffer);
    }
    else
    {
        status = synSuccess;
    }

    bool shouldAbort = (status != synSuccess);
    pSingleExecutionOwner->releaseOwnership(shouldAbort ? SingleExecutionOwner::ABORT_OPERATION_OWNERSHIP_ID
                                                        : programDataHandle);

    return status;
}

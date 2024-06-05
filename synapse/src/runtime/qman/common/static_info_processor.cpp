#include "static_info_processor.hpp"

#include "recipe_package_types.hpp"
#include "defs.h"
#include "command_submission_builder.hpp"
#include "defenders.h"
#include "define_synapse_common.hpp"
#include "habana_global_conf_runtime.h"
#include "device_mapper_interface.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_static_info.hpp"
#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "recipe_static_information.hpp"
#include "synapse_runtime_logging.h"
#include "types.h"
#include "global_statistics.hpp"

#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"

#include "runtime/qman/common/master_qmans_definition_interface.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"

#include "runtime/common/recipe/patching/define.hpp"

#include "recipe.h"
#include "gaudi/gaudi_packets.h"
#include "profiler_api.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"

#include "drm/habanalabs_accel.h"

#include <cstring>

#define DEBUG_GET_DC_LOCATION_MACRO
#ifndef DEBUG_GET_DC_LOCATION_MACRO
#define GET_DC_LOCATION_FROM_BLOB_LOCATION(currentDataChunkIndex,                                                      \
                                           offsetInDataChunk,                                                          \
                                           blobIndex,                                                                  \
                                           blobs,                                                                      \
                                           dwOffsetInBlob,                                                             \
                                           blobsAmount,                                                                \
                                           blobsBufferAddress,                                                         \
                                           dcSizeCommand,                                                              \
                                           programCommandsChunksAmount)                                                \
    {                                                                                                                  \
        HB_ASSERT_DEBUG_ONLY(blobIndex < blobsAmount, "Invalid blob index");                                           \
        const blob_t& currentBlob = blobs[blobIndex];                                                                  \
                                                                                                                       \
        uint64_t patchPointBlobAddress = (uint64_t)(((const uint32_t*)currentBlob.data) + dwOffsetInBlob);             \
                                                                                                                       \
        uint64_t patchPointBlobRelativeAddress = patchPointBlobAddress - blobsBufferAddress;                           \
        currentDataChunkIndex                  = patchPointBlobRelativeAddress / dcSizeCommand;                        \
                                                                                                                       \
        offsetInDataChunk = patchPointBlobRelativeAddress - (currentDataChunkIndex * dcSizeCommand);                   \
                                                                                                                       \
        HB_ASSERT_DEBUG_ONLY((currentDataChunkIndex < programCommandsChunksAmount),                                    \
                             "Failed to convert Blobs' PPs into Data-Chunks' PPs");                                    \
    }

#else

void GET_DC_LOCATION_FROM_BLOB_LOCATION(uint64_t&            currentDataChunkIndex,
                                        uint64_t&            offsetInDataChunk,
                                        uint64_t             blobIndex,
                                        const blob_t* const& blobs,
                                        uint64_t             dwOffsetInBlob,
                                        uint64_t             blobsAmount,
                                        uint64_t             blobsBufferAddress,
                                        uint64_t             dcSizeCommand,
                                        uint64_t             programCommandsChunksAmount)
{
    HB_ASSERT_DEBUG_ONLY(blobIndex < blobsAmount, "Invalid blob index {} {}", blobIndex, blobsAmount);
    const blob_t& currentBlob = blobs[blobIndex];

    uint64_t patchPointBlobAddress = (uint64_t)(((const uint32_t*)currentBlob.data) + dwOffsetInBlob);

    uint64_t patchPointBlobRelativeAddress = patchPointBlobAddress - blobsBufferAddress;
    currentDataChunkIndex                  = patchPointBlobRelativeAddress / dcSizeCommand;

    offsetInDataChunk = patchPointBlobRelativeAddress - (currentDataChunkIndex * dcSizeCommand);

    HB_ASSERT_DEBUG_ONLY((currentDataChunkIndex < programCommandsChunksAmount),
                         "Failed to convert Blobs' PPs into Data-Chunks' PPs");
}
#endif

constexpr uint64_t INVALID_PP_INDEX = std::numeric_limits<uint64_t>::max();


bool StaticInfoProcessor::allocateResourcesAndProcessRecipe(DeviceMapperInterface*          pDeviceMapper,
                                                            const basicRecipeInfo&          rBasicRecipeHandle,
                                                            const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                            synDeviceType                   deviceType,
                                                            RecipeStaticInfo&               rRecipeInfo,
                                                            std::vector<uint64_t>&          rProgramCodeDeviceAddresses,
                                                            uint64_t                        sectionAddressForProgram,
                                                            bool                            programCodeInCache,
                                                            uint64_t                        workspaceAddress,
                                                            uint64_t                        dcSizeCpDma,
                                                            uint64_t                        dcSizeCommand)
{
    if (!rRecipeInfo.isInitialized())
    {
        const bool status = processRecipeInfo(rBasicRecipeHandle,
                                              rDeviceAgnosticRecipeInfo,
                                              rRecipeInfo,
                                              *pDeviceMapper,
                                              deviceType,
                                              dcSizeCpDma,
                                              dcSizeCommand,
                                              rProgramCodeDeviceAddresses,
                                              sectionAddressForProgram,
                                              programCodeInCache,
                                              workspaceAddress);

        if (!status)
        {
            return false;
        }
    }
    else
    {
        if (programCodeInCache)
        {
            const bool status = createCpDmaForStaticBlobsAndStore(deviceType,
                                                                  (*rBasicRecipeHandle.recipe),
                                                                  rRecipeInfo,
                                                                  rProgramCodeDeviceAddresses,
                                                                  *pDeviceMapper,
                                                                  sectionAddressForProgram);

            if (!status)
            {
                return false;
            }
        }
    }
    return true;
}

void StaticInfoProcessor::destroyProcessor(const DeviceMapperInterface& rDeviceMapper, RecipeStaticInfo& rRecipeInfo)
{
    if (!rRecipeInfo.isInitialized())
    {
        return;
    }

    LOG_DEBUG(SYN_STREAM, "Destroying RecipeInfo 0x{:x}", (uint64_t)&rRecipeInfo);

    if (!GCFG_ENABLE_MAPPING_IN_STREAM_COPY.value())
    {
        if (!unmapAndClearMappedBuffers(rDeviceMapper, rRecipeInfo))
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Failed to clear program-data blobs", HLLOG_FUNC);
        }
    }

    freeCpDmaStaticBlobsBuffer(rRecipeInfo);

    freeArbitrationPackets(rRecipeInfo);

    _clearPatchPointsDcInfoDbs(&rRecipeInfo);

    rRecipeInfo.setInitialized(false);
}

bool StaticInfoProcessor::processRecipeInfo(const basicRecipeInfo&          rBasicRecipeHandle,
                                            const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                            RecipeStaticInfo&               rRecipeInfo,
                                            const DeviceMapperInterface&    rDeviceMapper,
                                            synDeviceType                   deviceType,
                                            uint64_t                        dcSizeCpDma,
                                            uint64_t                        dcSizeCommand,
                                            std::vector<uint64_t>&          rProgramCodeDeviceAddresses,
                                            uint64_t                        sectionAddressForProgram,
                                            bool                            programCodeInCache,
                                            uint64_t                        workspaceAddress)
{
    const recipe_t&      rRecipe           = *rBasicRecipeHandle.recipe;
    shape_plane_graph_t* pShapePlanRecipe  = rBasicRecipeHandle.shape_plan_recipe;
    const uint64_t patchingBlobsChunksSize = rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.m_patchingBlobsChunksSize;
    const uint64_t patchingBlobsChunksDataChunksAmount =
        rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.m_patchingBlobsChunksDataChunksAmount;

    LOG_DEBUG(SYN_STREAM, "recipe {:#x} init start", TO64(&rRecipe));
    STAT_GLBL_START(statProcessRecipe);
    PROFILER_COLLECT_TIME()

    bool status = true;

    bool isInitDBs                       = false;
    bool isArbitrationPacketsHandled     = false;
    bool isCpDmaStaticBlobsBufferHandled = false;

    do
    {
        rRecipeInfo.initDBs(rRecipe);
        isInitDBs = true;

        status = allocateArbitrationPackets(
            &rRecipeInfo,
            deviceType);  // create 2*arb cmds, sets m_arbSetCmdHostAddress, m_arbReleaseCmdHostAddress
        if (!status)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to allocate arbitration packets", HLLOG_FUNC);
            break;
        }
        isArbitrationPacketsHandled = true;

        if (!GCFG_ENABLE_MAPPING_IN_STREAM_COPY.value())
        {
            status = mapAndSetProgramData(rRecipe, rDeviceMapper, rRecipeInfo);
            if (!status)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to handle program-data blobs", HLLOG_FUNC);
                break;
            }
        }

        // Calculate the max amount possible needed for allocating the CP_DMAs buffer
        allocateCpDmaStaticBlobsBuffer(rRecipe, rRecipeInfo, deviceType);
        isCpDmaStaticBlobsBufferHandled = true;

        if (programCodeInCache)
        {
            status = createCpDmaForStaticBlobsAndStore(deviceType,
                                                       rRecipe,
                                                       rRecipeInfo,
                                                       rProgramCodeDeviceAddresses,
                                                       rDeviceMapper,
                                                       sectionAddressForProgram);
            if (!status)
            {
                break;
            }
        }

        bool status = calculateProgramChunksAmount(deviceType, rRecipe, rDeviceAgnosticRecipeInfo, rRecipeInfo, dcSizeCpDma);
        if (!status)
        {
            return status;
        }

        uint64_t cpDmaChunksAmountActivateCached    = 0;
        uint64_t cpDmaChunksAmountActivateNotCached = 0;


        uint64_t cpDmaChunksAmountEnqueueCached    = 0;
        uint64_t cpDmaChunksAmountEnqueueNotCached = 0;
        rRecipeInfo.getCpDmaChunksAmount(EXECUTION_STAGE_ENQUEUE, cpDmaChunksAmountEnqueueCached, true);
        rRecipeInfo.getCpDmaChunksAmount(EXECUTION_STAGE_ENQUEUE, cpDmaChunksAmountEnqueueNotCached, false);

        LOG_DEBUG(
            SYN_STREAM,
            "Required Data-Chunks (dcSizeCpDma 0x{:x}) for CP-DMA cached {} not cached {} cached {} not cached {})",
            dcSizeCpDma,
            cpDmaChunksAmountActivateCached,
            cpDmaChunksAmountActivateNotCached,
            cpDmaChunksAmountEnqueueCached,
            cpDmaChunksAmountEnqueueNotCached);

        status = calcPatchableBlobsOffsetsAndStore(pShapePlanRecipe,
                                                   deviceType,
                                                   rDeviceAgnosticRecipeInfo,
                                                   rRecipeInfo,
                                                   rRecipe,
                                                   dcSizeCommand,
                                                   patchingBlobsChunksSize,
                                                   patchingBlobsChunksDataChunksAmount);

        if (!status)
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Failed to calc patchable blobs offsets ", HLLOG_FUNC);
            break;
        }

        if (!GCFG_ENABLE_MAPPING_IN_STREAM_COPY.value())
        {
            status = mapAndSetProgramCode(rRecipe, rDeviceMapper, rRecipeInfo);
            if (!status)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to handle program-code blobs", HLLOG_FUNC);
                break;
            }
        }

    } while (0);  // Do once

    if (status)
    {
        rRecipeInfo.setInitialized(true);
        LOG_DEBUG(SYN_STREAM, "recipe {:#x} done init", TO64(&rRecipe));
    }
    else
    {
        if (isCpDmaStaticBlobsBufferHandled)
        {
            freeCpDmaStaticBlobsBuffer(rRecipeInfo);
        }

        if (isArbitrationPacketsHandled)
        {
            freeArbitrationPackets(rRecipeInfo);
        }

        if (isInitDBs)
        {
            rRecipeInfo.clearDBs();
        }

        unmapAndClearMappedBuffers(rDeviceMapper, rRecipeInfo);

        _clearPatchPointsDcInfoDbs(&rRecipeInfo);
    }

    STAT_GLBL_COLLECT_TIME(statProcessRecipe, globalStatPointsEnum::recipeInit);
    PROFILER_MEASURE_TIME("processRecipeInfo")

    return status;
}

void StaticInfoProcessor::allocateCpDmaStaticBlobsBuffer(const recipe_t&   rRecipe,
                                                         RecipeStaticInfo& rRecipeInfo,
                                                         synDeviceType     deviceType)
{
    uint64_t staticBlobsAmount = 0;

    uint64_t numberOfBlob = rRecipe.blobs_nr;
    blob_t*  pCurrentBlob = rRecipe.blobs;
    uint64_t blobIndex    = 0;
    for (; blobIndex < numberOfBlob; ++blobIndex, pCurrentBlob++)
    {
        if (!pCurrentBlob->blob_type.requires_patching)
        {
            staticBlobsAmount++;
        }
    }

    generic::CommandBufferPktGenerator* generator        = _getPacketGenerator(deviceType);
    const uint64_t                      cpDmaCommandSize = generator->getCpDmaSize();

    uint64_t cacheBlocksRequired =
        std::ceil(rRecipe.execution_blobs_buffer_size / (GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * (float)1024));

    char* pCpDmaBufferAddress;
    if (staticBlobsAmount + cacheBlocksRequired > 1)
    {
        uint64_t maxCpDmaForStaticBlobs = staticBlobsAmount + cacheBlocksRequired - 1;
        pCpDmaBufferAddress             = new char[maxCpDmaForStaticBlobs * cpDmaCommandSize];
    }
    else
    {
        pCpDmaBufferAddress = nullptr;
    }

    rRecipeInfo.setCpDmaStaticBlobsBuffer(pCpDmaBufferAddress);
}

void StaticInfoProcessor::freeCpDmaStaticBlobsBuffer(RecipeStaticInfo& rRecipeInfo)
{
    char* pCpDmaBufferAddress;
    if (rRecipeInfo.getCpDmaStaticBlobsBuffer(pCpDmaBufferAddress))
    {
        rRecipeInfo.clearCpDmaStaticBlobsBuffer();
        delete[] pCpDmaBufferAddress;
    }
}

bool StaticInfoProcessor::mapAndSetProgramCode(const recipe_t&              rRecipe,
                                               const DeviceMapperInterface& rDeviceMapper,
                                               RecipeStaticInfo&            rRecipeInfo)
{
    PROFILER_COLLECT_TIME()

    std::string mappingDesc("program-Code");
    void*       hostVA = nullptr;

    if (!rDeviceMapper.mapBufferToDevice((uint8_t*)rRecipe.execution_blobs_buffer,
                                         rRecipe.execution_blobs_buffer_size,
                                         mappingDesc,
                                         &hostVA))
    {
        LOG_ERR(SYN_STREAM, "Failed to map program code");
        return false;
    }

    rRecipeInfo.setProgramCodeMappedAddress((blobAddressType)rRecipe.execution_blobs_buffer);

    PROFILER_MEASURE_TIME("mapPCBuffer");

    return true;
}

bool StaticInfoProcessor::mapAndSetProgramData(const recipe_t&              rRecipe,
                                               const DeviceMapperInterface& rDeviceMapper,
                                               RecipeStaticInfo&            rRecipeInfo)
{
    PROFILER_COLLECT_TIME()

    std::string mappingDesc("program-Data");
    void*       hostVA = nullptr;

    uint64_t programDataBufferSize = rRecipe.program_data_blobs_size;
    if (programDataBufferSize == 0)
    {
        return true;
    }

    if (!rDeviceMapper.mapBufferToDevice((uint8_t*)rRecipe.program_data_blobs_buffer,
                                         rRecipe.program_data_blobs_size,
                                         mappingDesc,
                                         &hostVA))
    {
        LOG_ERR(SYN_STREAM, "Failed to map program data");
        return false;
    }

    rRecipeInfo.setProgramDataMappedAddress((blobAddressType)rRecipe.program_data_blobs_buffer);

    LOG_DEBUG(SYN_STREAM,
              "Create Program data section (addr 0x{:x}) and map it to device (addr 0x{:x}, for RecipeInfo 0x{:x})",
              (uint64_t)rRecipe.program_data_blobs_buffer,
              (uint64_t)hostVA,
              (uint64_t)&rRecipeInfo);

    PROFILER_MEASURE_TIME("createAndMapPDSection")

    return true;
}

bool StaticInfoProcessor::unmapAndClearMappedBuffers(const DeviceMapperInterface& rDeviceMapper,
                                                     RecipeStaticInfo&            rRecipeInfo)
{
    blobAddressType mappedAddress;

    if (rRecipeInfo.getProgramDataMappedAddress(mappedAddress))
    {
        if (!rDeviceMapper.unmapBufferFromDevice((void*)mappedAddress))
        {
            LOG_CRITICAL(SYN_STREAM, "Failed to unmap program-data entry (0x{:x})", (uint64_t)mappedAddress);
            return false;
        }

        rRecipeInfo.clearProgramDataMappedAddress();
    }

    if (rRecipeInfo.getProgramCodeMappedAddress(mappedAddress))
    {
        if (!rDeviceMapper.unmapBufferFromDevice((void*)mappedAddress))
        {
            LOG_CRITICAL(SYN_STREAM, "Failed to unmap program-code entry (0x{:x})", (uint64_t)mappedAddress);
            return false;
        }

        rRecipeInfo.clearProgramCodeMappedAddress();
    }

    return true;
}

void StaticInfoProcessor::_clearPatchPointsDcInfoDbs(RecipeStaticInfo* pRecipeInfo)
{
    for (uint8_t executionStage = EXECUTION_STAGE_ACTIVATE; executionStage < EXECUTION_STAGE_LAST; executionStage++)
    {
        clearStagePatchPointsDcInfoDbs(pRecipeInfo, (eExecutionStage)executionStage);
    }
}

generic::CommandBufferPktGenerator* StaticInfoProcessor::_getPacketGenerator(synDeviceType deviceType)
{
    generic::CommandBufferPktGenerator* generator = nullptr;

    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            generator = gaudi::CommandBufferPktGenerator::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type");
        }
    }
    return generator;
}

/*
 ***************************************************************************************************
 * Function: _buildDcPatchPointsDatabases
 * @brief This function creates (allocates and build) the relevant DBs to be used for patching
 *        over Data-Chunks
 *        1) A DB which consists of the patch_points DB which holds the original PP informaiton,
 *        and the PP location in the Data-Chunks (instead of its location in a Blob)
 *        2) A DB which defines the amount of PP per each DC
 * @output bool (is operation succeeded)
 *
 ***************************************************************************************************
 */
bool StaticInfoProcessor::_buildDcPatchPointsOnBlobsChunksDatabases(
    shape_plane_graph_t*            pShapePlanRecipe,
    synDeviceType                   deviceType,
    const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
    RecipeStaticInfo&               rRecipeInfo,
    const recipe_t&                 rRecipe,
    uint64_t                        dcSizeCommand,
    uint64_t                        patchingBlobsChunksSize,
    uint64_t                        patchingBlobsChunksDataChunksAmount,
    bool                            isDsd,
    const blob_t**                  blobs,
    uint64_t* const                 blobsBuffer)
{
    PROFILER_COLLECT_TIME()

    bool status = true;

    const DeviceAgnosticRecipeStaticInfo& deviceAgnosticRecipeInfo = rDeviceAgnosticRecipeInfo.m_recipeStaticInfo;

    blob_t*  patchOnDataChunksBlobs = nullptr;
    uint8_t* patchingBlobsBuffer    = nullptr;

    patchOnDataChunksBlobs = rRecipe.blobs;
    patchingBlobsBuffer    = (uint8_t*)rRecipe.patching_blobs_buffer;

    uint32_t firstStage = EXECUTION_STAGE_ENQUEUE;

    uint32_t firstEnquePpIndex = rRecipe.activate_patch_points_nr;
    std::vector<uint64_t> firstEnqueueIndexOnSgPps(rRecipe.section_groups_nr, 0);
    if (firstEnquePpIndex != 0)
    {
        firstEnqueueIndexOnSgPps.assign(rRecipe.section_groups_nr, INVALID_PP_INDEX);

        section_group_t* pCurrSectionsGroupPps = rRecipe.section_groups_patch_points;

        for (uint64_t sectionGroupIndex = 0;
             sectionGroupIndex < rRecipe.section_groups_nr;
             sectionGroupIndex++, pCurrSectionsGroupPps++)
        {
            uint32_t  numOfSgPatchPoints     = pCurrSectionsGroupPps->patch_points_nr;
            uint32_t* pCurrSgPatchPointIndex = pCurrSectionsGroupPps->patch_points_index_list;

            for (uint32_t i = 0; i < numOfSgPatchPoints; i++, pCurrSgPatchPointIndex++)
            {
                if (*pCurrSgPatchPointIndex >= firstEnquePpIndex)
                {
                    firstEnqueueIndexOnSgPps[sectionGroupIndex] = i;
                    break;
                }
            }
        }
    }

    for (uint32_t stage = firstStage; stage < EXECUTION_STAGE_LAST; stage++)
    {
        eExecutionStage executionStage              = (eExecutionStage)stage;
        uint64_t        patchingBlobsBufferAddress  = (uint64_t)patchingBlobsBuffer;
        uint64_t        programCommandsChunksAmount = 0;

        status = deviceAgnosticRecipeInfo.getProgramCommandsChunksAmount(executionStage, programCommandsChunksAmount);
        if (!status)
        {
            break;
        }

        for (uint64_t i = 0; i < rRecipe.section_groups_nr; i++)
        {
            if (!_emplaceDcPatchPointsOnBlobsChunksDatabases(rRecipe,
                                                             rRecipeInfo,
                                                             (eExecutionStage)stage,
                                                             &rRecipe.section_groups_patch_points[i],
                                                             firstEnqueueIndexOnSgPps[i],
                                                             patchOnDataChunksBlobs,
                                                             patchingBlobsBuffer,
                                                             dcSizeCommand,
                                                             programCommandsChunksAmount,
                                                             false))
            {
                return false;
            }
        }

        if (!_emplaceDcPatchPointsOnBlobsChunksDatabases(rRecipe,
                                                         rRecipeInfo,
                                                         (eExecutionStage)stage,
                                                         nullptr,
                                                         INVALID_PP_INDEX,
                                                         patchOnDataChunksBlobs,
                                                         patchingBlobsBuffer,
                                                         dcSizeCommand,
                                                         programCommandsChunksAmount,
                                                         false))
        {
            return false;
        }

        if (rRecipe.sobj_section_group_patch_points.patch_points_nr != 0 &&
            !_emplaceDcPatchPointsOnBlobsChunksDatabases(rRecipe,
                                                         rRecipeInfo,
                                                         (eExecutionStage)stage,
                                                         &rRecipe.sobj_section_group_patch_points,
                                                         INVALID_PP_INDEX,
                                                         patchOnDataChunksBlobs,
                                                         patchingBlobsBuffer,
                                                         dcSizeCommand,
                                                         programCommandsChunksAmount,
                                                         true))
        {
            return false;
        }

        if (isDsd && (stage == EXECUTION_STAGE_ENQUEUE))
        {
            status = _buildSmPatchPointsOnDataChunkDb(rRecipe,
                                                      pShapePlanRecipe,
                                                      rRecipeInfo,
                                                      programCommandsChunksAmount,
                                                      dcSizeCommand,
                                                      patchOnDataChunksBlobs,
                                                      rRecipe.blobs_nr,
                                                      patchingBlobsBufferAddress);
            if (!status)
            {
                break;
            }
        }
    }
    if (blobs != nullptr)
    {
        *blobs = patchOnDataChunksBlobs;
    }
    if (blobsBuffer != nullptr)
    {
        *blobsBuffer = (uint64_t)patchingBlobsBuffer;
    }

    PROFILER_MEASURE_TIME("buildDcPPOnBlobsDB")

    return status;
}

bool StaticInfoProcessor::_emplaceDcPatchPointsOnBlobsChunksDatabases(const recipe_t&        rRecipe,
                                                                      RecipeStaticInfo&      rRecipeInfo,
                                                                      eExecutionStage        stage,
                                                                      const section_group_t* pSectionType,
                                                                      uint64_t               sgPpFirstEnqueueIndex,
                                                                      const blob_t*          patchOnDataChunksBlobs,
                                                                      const uint8_t*         patchingBlobsBuffer,
                                                                      uint64_t               dcSizeCommand,
                                                                      uint64_t               programCommandsChunksAmount,
                                                                      bool                   isSobj)
{
    bool     isActivate          = (stage == EXECUTION_STAGE_ACTIVATE);
    uint32_t activateAmountOfPPs = rRecipe.activate_patch_points_nr;

    uint64_t             type;
    uint64_t             ppNum;
    uint32_t*            ppList;
    // The relevant PPs database:
    //  For PP_ALL               => Pass the exact PP-list, as all of them will be emplaced
    //  For PP By Sections-Group => Pass all of the PP-list,
    //                              and then pick the relevant PPs, according to the SG's PPs-list
    uint32_t             patch_points_nr;
    const patch_point_t* patch_points;


    if (pSectionType != nullptr)
    {
        type            = pSectionType->section_group;

        if (isSobj)
        {
            ppNum  = isActivate ? 0 : pSectionType->patch_points_nr;
            ppList = isActivate ? nullptr : pSectionType->patch_points_index_list;
        }
        else if (isActivate)
        {
            ppList = pSectionType->patch_points_index_list;
            ppNum  = (sgPpFirstEnqueueIndex != INVALID_PP_INDEX) ?
                     sgPpFirstEnqueueIndex :
                     pSectionType->patch_points_nr;
        }
        else
        {
            ppList = &pSectionType->patch_points_index_list[sgPpFirstEnqueueIndex];
            ppNum  = (sgPpFirstEnqueueIndex != INVALID_PP_INDEX) ?
                     pSectionType->patch_points_nr - sgPpFirstEnqueueIndex :
                     0;
        }

        patch_points_nr = rRecipe.patch_points_nr;
        patch_points    = rRecipe.patch_points;
    }
    else
    {
        const uint32_t       stage_patch_points_nr = isActivate ?
                                                        activateAmountOfPPs :
                                                        rRecipe.patch_points_nr - activateAmountOfPPs;
        const patch_point_t* stage_patch_points    = isActivate ?
                                                        rRecipe.patch_points :
                                                        &rRecipe.patch_points[activateAmountOfPPs];

        type            = patching::PP_TYPE_ID_ALL;
        ppNum           = stage_patch_points_nr;
        ppList          = nullptr;

        patch_points_nr = stage_patch_points_nr;
        patch_points    = stage_patch_points;
    }

    LOG_DEBUG(SYN_STREAM,
              "Using patching information stage {} type {} ppNum {} ppList {:#x} patch_points_nr {} patch_points {:#x}",
              stage,
              type,
              ppNum,
              TO64(ppList),
              patch_points_nr,
              TO64(patch_points));

    DataChunkPatchPointsInfo* pPatchPointsDataChunksInfoDb = allocatePatchPointsDcInfoDbs(rRecipeInfo,
                                                                                          (eExecutionStage)stage,
                                                                                          type,
                                                                                          programCommandsChunksAmount,
                                                                                          ppNum,
                                                                                          dcSizeCommand,
                                                                                          isSobj);
    if (pPatchPointsDataChunksInfoDb == nullptr)
    {
        return false;
    }

    _updatePatchPointsInfoDb(pPatchPointsDataChunksInfoDb,
                             programCommandsChunksAmount,
                             dcSizeCommand,
                             pPatchPointsDataChunksInfoDb->m_dataChunkPatchPoints,
                             ppList,
                             ppNum,
                             patch_points,
                             patch_points_nr,
                             patchOnDataChunksBlobs,
                             rRecipe.blobs_nr,
                             (uint64_t)patchingBlobsBuffer);

    return true;
}

bool StaticInfoProcessor::_updatePatchPointsInfoDb(DataChunkPatchPointsInfo* const& pPatchPointsDataChunksInfoDb,
                                                   uint64_t                         programCommandsChunksAmount,
                                                   uint64_t                         dcSizeCommand,
                                                   data_chunk_patch_point_t*        dataChunkPatchPoints,
                                                   const uint32_t*                  sectionGroupPpIndices,
                                                   const uint64_t                   typePatchPointAmount,
                                                   const patch_point_t* const&      patchPoints,
                                                   uint64_t                         totalPatchPointsAmount,
                                                   const blob_t* const&             blobs,
                                                   uint64_t                         blobsAmount,
                                                   uint64_t                         blobsBufferAddress)
{
    data_chunk_patch_point_t* pCurrNewPatchPoint = dataChunkPatchPoints;

    HB_ASSERT_PTR(pPatchPointsDataChunksInfoDb);

    std::function<uint64_t(uint64_t)> getPatchPointFromTypeDb = [sectionGroupPpIndices](uint64_t i) {
        return sectionGroupPpIndices[i];
    };
    std::function<uint64_t(uint64_t)> getPatchPointFromFullDb = [](uint64_t i) { return i; };

    uint64_t currentDataChunkIndex = 0;
    uint64_t ppOffsetInDataChunk   = 0;
    auto getPatchPointIndex = (sectionGroupPpIndices == nullptr) ? getPatchPointFromFullDb : getPatchPointFromTypeDb;
    auto patchPointAmount   = (sectionGroupPpIndices == nullptr) ? totalPatchPointsAmount : typePatchPointAmount;

    for (uint64_t i = 0; i < patchPointAmount; i++, pCurrNewPatchPoint++)
    {
        const uint64_t       ppIndex         = getPatchPointIndex(i);
        const patch_point_t& pCurrPatchPoint = patchPoints[ppIndex];
        uint32_t             ppBlobIndex     = pCurrPatchPoint.blob_idx;

        GET_DC_LOCATION_FROM_BLOB_LOCATION(currentDataChunkIndex,
                                           ppOffsetInDataChunk,
                                           ppBlobIndex,
                                           blobs,
                                           pCurrPatchPoint.dw_offset_in_blob,
                                           blobsAmount,
                                           blobsBufferAddress,
                                           dcSizeCommand,
                                           programCommandsChunksAmount);

        pCurrNewPatchPoint->offset_in_data_chunk = ppOffsetInDataChunk;
        pCurrNewPatchPoint->data_chunk_index     = currentDataChunkIndex;

        // copy all other fields
        pCurrNewPatchPoint->type                                 = pCurrPatchPoint.type;
        pCurrNewPatchPoint->memory_patch_point.section_idx       = pCurrPatchPoint.memory_patch_point.section_idx;
        pCurrNewPatchPoint->memory_patch_point.effective_address = pCurrPatchPoint.memory_patch_point.effective_address;
        pCurrNewPatchPoint->node_exe_index                       = pCurrPatchPoint.node_exe_index;
    }

    // add fictive patch point indicating last stage
    dataChunkPatchPoints[patchPointAmount]                = {};
    dataChunkPatchPoints[patchPointAmount].node_exe_index = INVALID_NODE_INDEX;

    currentDataChunkIndex++;

    return true;
}

bool StaticInfoProcessor::_updateSmPatchPointsInfoDb(const recipe_t&                    rRecipe,
                                                     shape_plane_graph_t*               pShapePlanRecipe,
                                                     DataChunkSmPatchPointsInfo* const& pSmPatchPointsDataChunksInfoDb,
                                                     const blob_t* const&               blobs,
                                                     uint64_t                           programCommandsChunksAmount,
                                                     uint64_t                           dcSizeCommand,
                                                     uint64_t                           blobsBufferAddress)
{
    HB_ASSERT_PTR(pSmPatchPointsDataChunksInfoDb);

    data_chunk_sm_patch_point_t* dataChunkSmPatchPoints = pSmPatchPointsDataChunksInfoDb->m_dataChunkSmPatchPoints;
    HB_ASSERT_PTR(dataChunkSmPatchPoints);

    HB_ASSERT_PTR(pShapePlanRecipe);

    uint64_t ppOffsetInDataChunk   = 0;
    uint64_t currentDataChunkIndex = 0;
    // Not used. Required for the common method, which is being called

    data_chunk_sm_patch_point_t* pCurrDataChunkSmPatchPoint = dataChunkSmPatchPoints;
    uint32_t                     blobsAmount                = rRecipe.blobs_nr;
    const shape_plane_node_t*    pCurrentNode               = pShapePlanRecipe->sp_nodes;
    uint32_t                     amountOfShapePlaneNodes    = pShapePlanRecipe->sp_node_nr;
    for (uint32_t nodeIndex = 0; nodeIndex < amountOfShapePlaneNodes; nodeIndex++, pCurrentNode++)
    {
        sm_patch_point_t* pCurrentBlobSmPatchPoint = pCurrentNode->node_patch_points;
        uint32_t          amountOfSmPatchPoints    = pCurrentNode->node_patch_points_nr;

        for (uint32_t smPpIndex = 0; smPpIndex < amountOfSmPatchPoints;
             smPpIndex++, pCurrentBlobSmPatchPoint++, pCurrDataChunkSmPatchPoint++)
        {
            EFieldType fieldType = pCurrentBlobSmPatchPoint->patch_point_type;

            if (fieldType != FIELD_DYNAMIC_ADDRESS)
            {
                uint64_t ppBlobIndex = pCurrentBlobSmPatchPoint->blob_idx;

                GET_DC_LOCATION_FROM_BLOB_LOCATION(currentDataChunkIndex,
                                                   ppOffsetInDataChunk,
                                                   ppBlobIndex,
                                                   blobs,
                                                   pCurrentBlobSmPatchPoint->dw_offset_in_blob,
                                                   blobsAmount,
                                                   blobsBufferAddress,
                                                   dcSizeCommand,
                                                   programCommandsChunksAmount);

                LOG_DEBUG(SYN_STREAM,
                          "SM Patch-point (node-index {} pp-index {})"
                          " (blob-index {} offset-in-blob 0x{:x}) -> (DC-Index {}, offset-in-DC 0x{:x})",
                          nodeIndex,
                          smPpIndex,
                          ppBlobIndex,
                          pCurrentBlobSmPatchPoint->dw_offset_in_blob,
                          currentDataChunkIndex,
                          ppOffsetInDataChunk);

                pCurrDataChunkSmPatchPoint->data_chunk_index     = currentDataChunkIndex;
                pCurrDataChunkSmPatchPoint->offset_in_data_chunk = ppOffsetInDataChunk;
            }
            else
            {
                pCurrDataChunkSmPatchPoint->patch_point_idx_high = pCurrentBlobSmPatchPoint->patch_point_idx_high;
                pCurrDataChunkSmPatchPoint->patch_point_idx_low  = pCurrentBlobSmPatchPoint->patch_point_idx_low;
            }

            // copy all other fields
            pCurrDataChunkSmPatchPoint->patch_point_type = pCurrentBlobSmPatchPoint->patch_point_type;
            pCurrDataChunkSmPatchPoint->patch_size_dw    = pCurrentBlobSmPatchPoint->patch_size_dw;
            pCurrDataChunkSmPatchPoint->roi_idx          = pCurrentBlobSmPatchPoint->roi_idx;

            // pointer to tables in the origin SMF-PP
            pCurrDataChunkSmPatchPoint->p_smf_id       = &(pCurrentBlobSmPatchPoint->smf_id);
            pCurrDataChunkSmPatchPoint->p_pp_metdata   = pCurrentBlobSmPatchPoint->metadata;
            pCurrDataChunkSmPatchPoint->is_unskippable = pCurrentBlobSmPatchPoint->is_unskippable;
        }
    }

    return true;
}

bool StaticInfoProcessor::_buildSmPatchPointsOnDataChunkDb(const recipe_t&      rRecipe,
                                                           shape_plane_graph_t* pShapePlanRecipe,
                                                           RecipeStaticInfo&    rRecipeInfo,
                                                           uint64_t             programCommandsChunksAmount,
                                                           uint64_t             dcSizeCommand,
                                                           const blob_t* const& blobs,
                                                           uint64_t             blobsAmount,
                                                           uint64_t             patchingBlobsBufferAddress)
{
    LOG_TRACE(SYN_STREAM, "Update SM-patching information");
    uint32_t patchPointsAmount = _calculateTotalAmountOfSmPatchPoints(pShapePlanRecipe);
    if (patchPointsAmount == 0)
    {
        LOG_DEBUG(SYN_STREAM,
                  "no PP in _buildSmPatchPointsOnDataChunkDb");  // this happened in a small graph
                                                                 // continue and allocate or you get an exception
    }

    bool status = rRecipeInfo.allocateSmPatchingPointsDcLocation(patchPointsAmount, dcSizeCommand);
    if (!status)
    {
        LOG_ERR(SYN_STREAM, "Failed to allocate SM-patching information");
        return false;
    }

    DataChunkSmPatchPointsInfo* pSmPatchPointsDataChunksInfoDb = rRecipeInfo.refSmPatchingPointsDcLocation();

    if (pSmPatchPointsDataChunksInfoDb == nullptr)
    {
        LOG_ERR(SYN_STREAM, "Failed to allocate SM-patching information");
        return false;
    }

    return _updateSmPatchPointsInfoDb(rRecipe,
                                      pShapePlanRecipe,
                                      pSmPatchPointsDataChunksInfoDb,
                                      blobs,
                                      programCommandsChunksAmount,
                                      dcSizeCommand,
                                      patchingBlobsBufferAddress);
}

bool StaticInfoProcessor::calcPatchableBlobsOffsetsAndStore(shape_plane_graph_t*            pShapePlanRecipe,
                                                            synDeviceType                   deviceType,
                                                            const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                            RecipeStaticInfo&               rRecipeInfo,
                                                            const recipe_t&                 rRecipe,
                                                            uint64_t                        dcSizeCommand,
                                                            uint64_t                        patchingBlobsChunksSize,
                                                            uint64_t patchingBlobsChunksDataChunksAmount)
{
    const blob_t* blobs       = rRecipe.blobs;
    uint64_t      blobsBuffer = (uint64_t)rRecipe.patching_blobs_buffer;

    const bool isDsd  = (pShapePlanRecipe != nullptr);
    bool       status = _buildDcPatchPointsOnBlobsChunksDatabases(pShapePlanRecipe,
                                                            deviceType,
                                                            rDeviceAgnosticRecipeInfo,
                                                            rRecipeInfo,
                                                            rRecipe,
                                                            dcSizeCommand,
                                                            patchingBlobsChunksSize,
                                                            patchingBlobsChunksDataChunksAmount,
                                                            isDsd,
                                                            &blobs,
                                                            &blobsBuffer);

    if (!status)
    {
        return false;
    }

    _calcPatchableBlobsOffsetsAndStore(blobs, blobsBuffer, rRecipe.blobs_nr, dcSizeCommand, rRecipeInfo);

    return true;
}

void StaticInfoProcessor::_calcPatchableBlobsOffsetsAndStore(const blob_t*&    pRecipeBlobsElements,
                                                             uint64_t          blobsBuffer,
                                                             uint64_t          blobsAmount,
                                                             uint64_t          dcSizeCommand,
                                                             RecipeStaticInfo& rRecipeInfo)
{
    const blob_t* pCurrentRecipeBlob = pRecipeBlobsElements;

    std::vector<patchableBlobOffsetInDc>& patchableOffsetsDB     = rRecipeInfo.getPatchableBlobsOffsetsDB();
    auto                                  patchableOffsetsDBiter = patchableOffsetsDB.begin();

    for (uint64_t blobIndex = 0; blobIndex < blobsAmount; blobIndex++, pCurrentRecipeBlob++, patchableOffsetsDBiter++)
    {
        if (pCurrentRecipeBlob->blob_type.requires_patching)
        {
            uint64_t offsetInPatchBuff = (uint64_t)(pCurrentRecipeBlob->data) - blobsBuffer;
            uint32_t dcIndex           = offsetInPatchBuff / dcSizeCommand;
            uint64_t blobsOffsetInDc   = offsetInPatchBuff - (dcIndex * dcSizeCommand);

            patchableOffsetsDBiter->offsetInDc = blobsOffsetInDc;
            patchableOffsetsDBiter->dcIndex    = dcIndex;
        }
    }
}

uint32_t StaticInfoProcessor::_calculateTotalAmountOfSmPatchPoints(shape_plane_graph_t* pShapePlanRecipe)
{
    if (pShapePlanRecipe == nullptr)
    {
        return 0;
    }

    uint32_t amountOfSmPatchPoints = 0;

    const shape_plane_node_t* pCurrentNode = pShapePlanRecipe->sp_nodes;
    for (uint32_t nodeIndex = 0; nodeIndex < pShapePlanRecipe->sp_node_nr; nodeIndex++, pCurrentNode++)
    {
        amountOfSmPatchPoints += pCurrentNode->node_patch_points_nr;
    }

    return amountOfSmPatchPoints;
}

bool StaticInfoProcessor::createCpDmaForStaticBlobsAndStore(synDeviceType                deviceType,
                                                            const recipe_t&              rRecipe,
                                                            RecipeStaticInfo&            rRecipeInfo,
                                                            std::vector<uint64_t>&       rProgramCodeDeviceAddresses,
                                                            const DeviceMapperInterface& rDeviceMapper,
                                                            uint64_t                     sectionAddressForProgram)
{
    if (rRecipe.execution_blobs_buffer_size > 0)
    {
        const bool status =
            storeNonPatchableBlobsDeviceAddresses(rDeviceMapper, rRecipe, rRecipeInfo, rProgramCodeDeviceAddresses);

        if (!status)
        {
            return false;
        }

        STAT_GLBL_START(createStaticDma);
        _createCpDmaForStaticBlobsAndStore(rRecipe, &rRecipeInfo, deviceType);
        STAT_GLBL_COLLECT_TIME(createStaticDma, globalStatPointsEnum::createStaticDma);
    }

    return true;
}

bool StaticInfoProcessor::calculateProgramChunksAmount(synDeviceType                   deviceType,
                                                       const recipe_t&                 rRecipe,
                                                       const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                       RecipeStaticInfo&               rRecipeInfo,
                                                       uint64_t                        dcSizeCpDma)
{
    STAT_GLBL_START(calcDcNum);
    bool status = _calculateProgramChunksAmount(deviceType,
                                                rRecipe,
                                                rDeviceAgnosticRecipeInfo,
                                                rRecipeInfo,
                                                dcSizeCpDma,
                                                EXECUTION_STAGE_ENQUEUE);
    if (!status)
    {
        return status;
    }

    STAT_GLBL_COLLECT_TIME(calcDcNum, globalStatPointsEnum::calcDcNum);

    return true;
}

bool StaticInfoProcessor::_calculateProgramChunksAmount(synDeviceType                   deviceType,
                                                        const recipe_t&                 rRecipe,
                                                        const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                        RecipeStaticInfo&               rRecipeInfo,
                                                        uint64_t                        dcSizeCpDma,
                                                        eExecutionStage                 stage)
{
    generic::CommandBufferPktGenerator* packetGenerator = nullptr;
    QmanDefinitionInterface*            qmansDef        = nullptr;

    uint64_t arbCommandSize    = 0;
    uint64_t fenceSetPktSize   = 0;
    uint64_t fenceClearPktSize = 0;

    uint64_t arbMasterEngineId = 0;

    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            packetGenerator = gaudi::CommandBufferPktGenerator::getInstance();
            qmansDef        = gaudi::QmansDefinition::getInstance();

            arbCommandSize = packetGenerator->getArbitrationCommandSize();

            fenceSetPktSize   = packetGenerator->getFenceSetPacketCommandSize();
            fenceClearPktSize = packetGenerator->getFenceClearPacketCommandSize();

            arbMasterEngineId = qmansDef->getArbitratorMasterQueueIdForCompute();

            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type");
        }
    }

    const uint64_t cpDmaCommandSize = packetGenerator->getCpDmaSize();
    HB_ASSERT((arbCommandSize <= cpDmaCommandSize), "Arbitration packet-size shouldn't be larger than CP-Dma packet");
    HB_ASSERT(cpDmaCommandSize <= dcSizeCpDma, "invalid cpDmaCommandSize {}", cpDmaCommandSize);
    HB_ASSERT((dcSizeCpDma % cpDmaCommandSize) == 0, "invalid dcSizeCpDma {}", dcSizeCpDma);

    CachedAndNot totalCpDmaChunksAmount = {0, 0};
    CachedAndNot maxProgramDcSize       = {0, 0};
    CachedAndNot maxProgramWcDcSize     = {0, 0};
    CachedAndNot totalProgramDcSize     = {0, 0};
    CachedAndNot totalProgramWcDcSize   = {0, 0};

    programsCpDmaPacketsDB    programCpDmaAmountDB;
    blobsCpDmaPacketsAmountDB blobsCpDmaAmountDB;

    job_t*         pCurrentJob = (stage == EXECUTION_STAGE_ENQUEUE) ? rRecipe.execute_jobs : rRecipe.activate_jobs;
    const uint64_t jobs_nr = (stage == EXECUTION_STAGE_ENQUEUE) ? rRecipe.execute_jobs_nr : rRecipe.activate_jobs_nr;

    uint64_t internalCommandsDcQueueIdCounter   = 0;
    uint64_t internalCommandsDcQueueIdCounterWc = 0;

    for (uint64_t jobIndex = 0; jobIndex < jobs_nr; jobIndex++, pCurrentJob++)
    {
        uint64_t engineId     = pCurrentJob->engine_id;
        uint64_t programIndex = pCurrentJob->program_idx;

        if (qmansDef->isNonInternalCommandsDcQueueId(engineId))
        {
            continue;
        }

        if (programCpDmaAmountDB.find(programIndex) == programCpDmaAmountDB.end())
        {
            bool status = _calculateProgramCpDmaPktAmount(rRecipe,
                                                          programIndex,
                                                          programCpDmaAmountDB,
                                                          blobsCpDmaAmountDB,
                                                          dcSizeCpDma,
                                                          cpDmaCommandSize,
                                                          rRecipeInfo);
            if (!status)
            {
                return status;
            }
        }

        // all CP_DMAs in current program and ARB_REQ + ARB_REL
        CachedAndNot currentProgramDcSize;
        currentProgramDcSize.cached = (programCpDmaAmountDB[programIndex].cached * cpDmaCommandSize) + arbCommandSize;
        currentProgramDcSize.notCached =
            (programCpDmaAmountDB[programIndex].notCached * cpDmaCommandSize) + arbCommandSize;

        bool isArbMasterEngineId = (arbMasterEngineId == engineId);
        bool isWorkCompletion    = isArbMasterEngineId;

        if (!isWorkCompletion)
        {
            internalCommandsDcQueueIdCounter++;

            // consume hole in first DC left after ARB packet
            if (arbCommandSize > 0)
            {
                if (currentProgramDcSize.cached > dcSizeCpDma)
                {
                    currentProgramDcSize.cached += cpDmaCommandSize - arbCommandSize;
                }
                if (currentProgramDcSize.notCached > dcSizeCpDma)
                {
                    currentProgramDcSize.notCached += cpDmaCommandSize - arbCommandSize;
                }

                // ARB clear
                currentProgramDcSize.cached += arbCommandSize;
                currentProgramDcSize.notCached += arbCommandSize;
            }
        }
        else
        {
            internalCommandsDcQueueIdCounterWc++;
            currentProgramDcSize.cached += fenceSetPktSize;
            currentProgramDcSize.notCached += fenceSetPktSize;

            // consume hole in first DC left after ARB and Fence packets
            if ((fenceSetPktSize + arbCommandSize) != cpDmaCommandSize)
            {
                if (currentProgramDcSize.cached > dcSizeCpDma)
                {
                    currentProgramDcSize.cached += cpDmaCommandSize - fenceSetPktSize - arbCommandSize;
                }
                if (currentProgramDcSize.notCached > dcSizeCpDma)
                {
                    currentProgramDcSize.notCached += cpDmaCommandSize - fenceSetPktSize - arbCommandSize;
                }
            }

            uint64_t   workCompletionProgramIndex;
            const bool isSet =
                rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.getWorkCompletionProgramIndex(stage,
                                                                                           workCompletionProgramIndex);
            if (isSet)
            {
                program_t* workCompletionCurrentProgram = &(rRecipe.programs[workCompletionProgramIndex]);
                uint64_t   workCompletionProgramLength  = workCompletionCurrentProgram->program_length;
                LOG_TRACE(SYN_STREAM, "{}: work completion program length {}", HLLOG_FUNC, workCompletionProgramLength);
                for (int j = 0; j < workCompletionProgramLength; j++)
                {
                    uint64_t blobIndex    = workCompletionCurrentProgram->blob_indices[j];
                    blob_t*  pCurrentBlob = &rRecipe.blobs[blobIndex];
                    HB_ASSERT((pCurrentBlob != nullptr), "Current blob is null pointer");
                    const bool isLegalBlobSize = (dcSizeCpDma >= pCurrentBlob->size);
                    if (!isLegalBlobSize)
                    {
                        LOG_CRITICAL(
                            SYN_STREAM,
                            "Illegal Blob Size jobIndex {} engineId {} programIndex {} dcSizeCpDma {} blobSize {}",
                            jobIndex,
                            engineId,
                            programIndex,
                            dcSizeCpDma,
                            pCurrentBlob->size);
                        HB_ASSERT(false, "Completion program doesn't fit in a single DC");
                    }

                    CachedAndNot currentDcFreeSize;
                    currentDcFreeSize.cached    = dcSizeCpDma - (currentProgramDcSize.cached % dcSizeCpDma);
                    currentDcFreeSize.notCached = dcSizeCpDma - (currentProgramDcSize.notCached % dcSizeCpDma);

                    if (currentDcFreeSize.cached < pCurrentBlob->size)
                    {
                        // consume free size as the blob will be placed in a new DC
                        currentProgramDcSize.cached += currentDcFreeSize.cached;
                    }

                    if (currentDcFreeSize.notCached < pCurrentBlob->size)
                    {
                        // consume free size as the blob will be placed in a new DC
                        currentProgramDcSize.notCached += currentDcFreeSize.notCached;
                    }

                    currentProgramDcSize.cached += pCurrentBlob->size;
                    currentProgramDcSize.notCached += pCurrentBlob->size;
                }
            }

            CachedAndNot currentDcFreeSize;
            currentDcFreeSize.cached    = dcSizeCpDma - (currentProgramDcSize.cached % dcSizeCpDma);
            currentDcFreeSize.notCached = dcSizeCpDma - (currentProgramDcSize.notCached % dcSizeCpDma);

            // if one of the packets doesn't fit in the last DC, it will require an additional DC
            // This calculation is good enough, since it's accurate in resolution of DC amount
            if (currentDcFreeSize.cached < arbCommandSize + fenceClearPktSize)
            {
                // consume free size as the ARB+FENCE will be placed in a new DC
                currentProgramDcSize.cached += currentDcFreeSize.cached;
            }
            if (currentDcFreeSize.notCached < arbCommandSize + fenceClearPktSize)
            {
                // consume free size as the ARB+FENCE will be placed in a new DC
                currentProgramDcSize.notCached += currentDcFreeSize.notCached;
            }

            currentProgramDcSize.cached += arbCommandSize + fenceClearPktSize;
            currentProgramDcSize.notCached += arbCommandSize + fenceClearPktSize;
        }

        CachedAndNot addedChunks;
        addedChunks.cached    = ((currentProgramDcSize.cached - 1) / dcSizeCpDma) + 1;
        addedChunks.notCached = ((currentProgramDcSize.notCached - 1) / dcSizeCpDma) + 1;

        if ((addedChunks.cached) > 1 || (addedChunks.notCached > 1))
        {
            LOG_TRACE(SYN_STREAM,
                      "engineId: {} has more than one CP_DMA DCs: cached case {} not cached case {}",
                      engineId,
                      addedChunks.cached,
                      addedChunks.notCached);
        }

        if (!isWorkCompletion)
        {
            if (maxProgramDcSize.cached < currentProgramDcSize.cached)
            {
                maxProgramDcSize.cached = currentProgramDcSize.cached;
            }

            if (maxProgramDcSize.notCached < currentProgramDcSize.notCached)
            {
                maxProgramDcSize.notCached = currentProgramDcSize.notCached;
            }

            totalProgramDcSize.cached += currentProgramDcSize.cached;
            totalProgramDcSize.notCached += currentProgramDcSize.notCached;
        }
        else
        {
            if (maxProgramWcDcSize.cached < currentProgramDcSize.cached)
            {
                maxProgramWcDcSize.cached = currentProgramDcSize.cached;
            }

            if (maxProgramWcDcSize.notCached < currentProgramDcSize.notCached)
            {
                maxProgramWcDcSize.notCached = currentProgramDcSize.notCached;
            }

            totalProgramWcDcSize.cached += currentProgramDcSize.cached;
            totalProgramWcDcSize.notCached += currentProgramDcSize.notCached;
        }

        totalCpDmaChunksAmount.cached += addedChunks.cached;
        totalCpDmaChunksAmount.notCached += addedChunks.notCached;
    }  // for(JobIndex)

    rRecipeInfo.setCpDmaChunksAmount(stage, totalCpDmaChunksAmount);
    STAT_GLBL_COLLECT(dcSizeCpDma, dcSizeCpDma);
    STAT_GLBL_COLLECT(totalCpDmaChunksAmount.cached, dcAmountCpDmaCached);
    STAT_GLBL_COLLECT(totalCpDmaChunksAmount.notCached, dcAmountCpDmaNotCached);
    STAT_GLBL_COLLECT(maxProgramDcSize.cached, maxProgramDcSizeCached);
    STAT_GLBL_COLLECT(maxProgramDcSize.notCached, maxProgramDcSizeNotCached);
    STAT_GLBL_COLLECT(maxProgramWcDcSize.cached, maxProgramWcDcSizeCached);
    STAT_GLBL_COLLECT(maxProgramWcDcSize.notCached, maxProgramWcDcSizeNotCached);

    if (internalCommandsDcQueueIdCounter > 0)
    {
        STAT_GLBL_COLLECT(totalProgramDcSize.cached / internalCommandsDcQueueIdCounter, avgProgramDcSizeCached);
        STAT_GLBL_COLLECT(totalProgramDcSize.notCached / internalCommandsDcQueueIdCounter, avgProgramDcSizeNotCached);
    }
    else
    {
        STAT_GLBL_COLLECT(0, avgProgramDcSizeCached);
        STAT_GLBL_COLLECT(0, avgProgramDcSizeNotCached);
    }

    if (internalCommandsDcQueueIdCounterWc > 0)
    {
        STAT_GLBL_COLLECT(totalProgramWcDcSize.cached / internalCommandsDcQueueIdCounterWc, avgProgramWcDcSizeCached);
        STAT_GLBL_COLLECT(totalProgramWcDcSize.notCached / internalCommandsDcQueueIdCounterWc, avgProgramWcDcSizeNotCached);
    }
    else
    {
        STAT_GLBL_COLLECT(0, avgProgramWcDcSizeCached);
        STAT_GLBL_COLLECT(0, avgProgramWcDcSizeNotCached);
    }

    return true;
}

/*
1) Calculating the amount of CP-DMAs per blob, and summarize over a program

2) Aggregating the total amount of blobs' data
Later, that number will deduct the total amount of DC for the blobs' data

3) Non-patchable blobs are stored on recipe and not on DC. Hence:
a) Does not contribute to the blobs' DC amount
b) Contribute a single CP-DMA packet, pointing to that blob
*/
bool StaticInfoProcessor::_calculateProgramCpDmaPktAmount(const recipe_t&            rRecipe,
                                                          uint64_t                   programIndex,
                                                          programsCpDmaPacketsDB&    programCpDmaAmountDB,
                                                          blobsCpDmaPacketsAmountDB& blobsCpDmaAmountDB,
                                                          uint64_t                   dcSizeCpDma,
                                                          uint64_t                   cpDmaPacketSize,
                                                          RecipeStaticInfo&          rRecipeInfo)
{
    CachedAndNot programCpDmaPktsAmount = {0, 0};
    program_t    currentProgram         = rRecipe.programs[programIndex];

    uint64_t blobIdx = 0;

    uint32_t nodeNr = GCFG_ENABLE_STAGED_SUBMISSION.value() ? rRecipe.node_nr : 1;
    for (uint32_t node = 0; node < nodeNr; node++)
    {
        uint64_t programBlobsNr = GCFG_ENABLE_STAGED_SUBMISSION.value()
                                      ? rRecipe.node_exe_list[node].program_blobs_nr[programIndex]
                                      : rRecipe.programs[programIndex].program_length;
        for (; blobIdx < programBlobsNr; blobIdx++)
        {
            uint64_t blob = currentProgram.blob_indices[blobIdx];
            if (blobsCpDmaAmountDB.find(blob) != blobsCpDmaAmountDB.end())
            {
                programCpDmaPktsAmount.cached += blobsCpDmaAmountDB[blob].cached;
                programCpDmaPktsAmount.notCached += blobsCpDmaAmountDB[blob].notCached;
            }
            else
            {
                blob_t currentBlob = rRecipe.blobs[blob];
                if (!currentBlob.blob_type.requires_patching)
                {
                    uint8_t cpDmaStaticBlobAmount = std::numeric_limits<uint8_t>::max();
                    bool status = rRecipeInfo.getProgramCodeBlobCpDmaAmount(blob, cpDmaStaticBlobAmount);
                    if (!status)
                    {
                        return status;
                    }
                    blobsCpDmaAmountDB[blob].cached    = cpDmaStaticBlobAmount;
                    blobsCpDmaAmountDB[blob].notCached = cpDmaStaticBlobAmount ? 1 : 0;

                    programCpDmaPktsAmount.cached += cpDmaStaticBlobAmount;
                    programCpDmaPktsAmount.notCached += cpDmaStaticBlobAmount ? 1 : 0;
                }
                else
                {
                    uint64_t blobsOffsetInDc =
                        ((uint64_t)currentBlob.data - (uint64_t)rRecipe.patching_blobs_buffer) % dcSizeCpDma;
                    uint64_t sizeWithAlignment            = currentBlob.size + blobsOffsetInDc;
                    uint64_t currentBlobCpDmaChunksAmount = ((sizeWithAlignment - 1) / dcSizeCpDma) + 1;

                    blobsCpDmaAmountDB[blob].cached    = currentBlobCpDmaChunksAmount;
                    blobsCpDmaAmountDB[blob].notCached = currentBlobCpDmaChunksAmount;
                    programCpDmaPktsAmount.cached += currentBlobCpDmaChunksAmount;
                    programCpDmaPktsAmount.notCached += currentBlobCpDmaChunksAmount;
                }
            }
        }  // for(blobCount)
    }  // for(node)
    programCpDmaAmountDB[programIndex] = programCpDmaPktsAmount;

    return true;
}

void StaticInfoProcessor::_createCpDmaForStaticBlobsAndStore(const recipe_t&   pRecipe,
                                                             RecipeStaticInfo* pRecipeInfo,
                                                             synDeviceType     deviceType)
{
    generic::CommandBufferPktGenerator* generator = nullptr;
    if (deviceType == synDeviceGaudi)
    {
        generator = gaudi::CommandBufferPktGenerator::getInstance();
    }
    else
    {
        HB_ASSERT(false, "Illegal device type");
    }

    const uint64_t cpDmaCommandSize = generator->getCpDmaSize();

    const HostAndDevAddrVec host2DevAddrVec = pRecipeInfo->getProgramCodeBlobsToDeviceAddress();

    pRecipeInfo->clearProgramCodeToCpDmaAddressDatabase(pRecipe.blobs_nr);

    char*      pCpDmaBufferAddress;
    const bool getStatus = pRecipeInfo->getCpDmaStaticBlobsBuffer(pCpDmaBufferAddress);
    HB_ASSERT(getStatus, "{}: cpDmaStaticBlobsBuffer is not set!", __FUNCTION__);

    for (uint64_t blobIdx = 0; blobIdx < pRecipe.blobs_nr; blobIdx++)
    {
        const HostAndDevAddr& host2DevAddr = host2DevAddrVec[blobIdx];

        uint64_t staticBlobHostAddress = host2DevAddr.hostAddr;
        if (staticBlobHostAddress == 0)
        {
            continue;
        }

        LOG_TRACE(SYN_STREAM,
                  "{}: blob idx {} CP_DMA addr 0x{:x} blobHostAddress 0x{:x} device addr 0x{:x} sizeInCpDma {:x}",
                  HLLOG_FUNC,
                  blobIdx,
                  TO64(pCpDmaBufferAddress),
                  staticBlobHostAddress,
                  host2DevAddr.devAddrAndSize.devAddr,
                  host2DevAddr.devAddrAndSize.size);

        char* packet = pCpDmaBufferAddress;
        generator->generateDefaultCpDma(packet, host2DevAddr.devAddrAndSize.size, host2DevAddr.devAddrAndSize.devAddr);
        pRecipeInfo->setProgramCodeBlobCpDmaAddress(blobIdx, (uint64_t)pCpDmaBufferAddress);
        pCpDmaBufferAddress += cpDmaCommandSize;

        for (auto& extraDevAddrAndSize : host2DevAddr.extraDevAddrAndSize)
        {
            uint64_t blobDeviceAddr = extraDevAddrAndSize.devAddr;
            uint64_t blobDeviceSize = extraDevAddrAndSize.size;

            LOG_TRACE(SYN_STREAM,
                      "{}: CP_DMA addr: 0x{:x} blobHostAddress: 0x{:x}, device addr: 0x{:x}, sizeInCpDma:0x{:x}",
                      HLLOG_FUNC,
                      TO64(pCpDmaBufferAddress),
                      staticBlobHostAddress,
                      blobDeviceAddr,
                      blobDeviceSize);

            char* packet = pCpDmaBufferAddress;
            generator->generateDefaultCpDma(packet, blobDeviceSize, blobDeviceAddr);
            pRecipeInfo->setProgramCodeBlobCpDmaAddress(blobIdx, (uint64_t)pCpDmaBufferAddress);
            pCpDmaBufferAddress += cpDmaCommandSize;
        }
    }
}

bool StaticInfoProcessor::allocateArbitrationPackets(RecipeStaticInfo* pRecipeInfo, synDeviceType deviceType)
{
    generic::CommandBufferPktGenerator* pCmdBuffPktGenerator;
    if (deviceType == synDeviceGaudi)
    {
        pCmdBuffPktGenerator = gaudi::CommandBufferPktGenerator::getInstance();
    }
    else
    {
        return false;
    }

    uint64_t sizeOfArbitrationCommand = pCmdBuffPktGenerator->getArbitrationCommandSize();

    synStatus status     = synSuccess;
    char*     pTmpPacket = nullptr;

    uint8_t* pArbitrationSetCommand = new uint8_t[sizeOfArbitrationCommand];

    pTmpPacket = (char*)pArbitrationSetCommand;
    status     = pCmdBuffPktGenerator->generateArbitrationCommand(pTmpPacket, sizeOfArbitrationCommand, false);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "Failed to generate arbitration-set command");
        delete[] pArbitrationSetCommand;
        return false;
    }

    uint8_t* pArbitrationReleaseCommand = new uint8_t[sizeOfArbitrationCommand];

    pTmpPacket = (char*)pArbitrationReleaseCommand;
    status     = pCmdBuffPktGenerator->generateArbitrationCommand(pTmpPacket, sizeOfArbitrationCommand, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "Failed to generate arbitration-release command");
        delete[] pArbitrationSetCommand;
        delete[] pArbitrationReleaseCommand;
        return false;
    }

    pRecipeInfo->setArbitrationSetCommand((uint64_t)pArbitrationSetCommand);
    pRecipeInfo->setArbitrationReleaseCommand((uint64_t)pArbitrationReleaseCommand);
    LOG_DEBUG(SYN_STREAM,
              "Allocate arbitration set/release cmd 0x{:x}, 0x{:x}",
              (uint64_t)pArbitrationSetCommand,
              (uint64_t)pArbitrationReleaseCommand);
    return true;
}

void StaticInfoProcessor::freeArbitrationPackets(RecipeStaticInfo& rRecipeInfo)
{
    uint64_t arbCommandHostAddress = 0;

    if (rRecipeInfo.getArbitrationSetHostAddress(arbCommandHostAddress))
    {
        rRecipeInfo.clearArbitrationSetCommand();
        delete[](uint8_t*) arbCommandHostAddress;
    }

    if (rRecipeInfo.getArbitrationReleaseHostAddress(arbCommandHostAddress))
    {
        rRecipeInfo.clearArbitrationReleaseCommand();
        delete[](uint8_t*) arbCommandHostAddress;
    }
}

DataChunkPatchPointsInfo* const StaticInfoProcessor::allocatePatchPointsDcInfoDbs(RecipeStaticInfo& rRecipeInfo,
                                                                                  eExecutionStage   executionStage,
                                                                                  uint64_t          patchPointsTypeId,
                                                                                  uint32_t          chunksAmount,
                                                                                  uint64_t          patchPointsAmount,
                                                                                  uint32_t          dcSizeCommand,
                                                                                  bool              isSobj)
{
    DataChunkPatchPointsInfo* pPatchPointsDataChunksInfoDb = nullptr;

    try
    {
        pPatchPointsDataChunksInfoDb = new DataChunkPatchPointsInfo();

        // adding a dummy data_chunk_patch_point_t to mark the last patch point
        pPatchPointsDataChunksInfoDb->m_dataChunkPatchPoints =
            new data_chunk_patch_point_t[patchPointsAmount + STAGED_SUBMISSION_PATCH_POINT_ADDITION];

        pPatchPointsDataChunksInfoDb->m_ppsPerDataChunkDbSize = chunksAmount;
        pPatchPointsDataChunksInfoDb->m_singleChunkSize       = dcSizeCommand;

        rRecipeInfo.setPatchingPointsDcLocation(executionStage,
                                                patchPointsTypeId,
                                                pPatchPointsDataChunksInfoDb,
                                                patchPointsAmount + STAGED_SUBMISSION_PATCH_POINT_ADDITION,
                                                isSobj);
    }
    catch (std::bad_alloc& err)
    {
        delete pPatchPointsDataChunksInfoDb;
        LOG_ERR(SYN_STREAM, "Failed to allocate PPs per DCs due to {}", err.what());
        return nullptr;
    }

    return pPatchPointsDataChunksInfoDb;
}

void StaticInfoProcessor::clearStagePatchPointsDcInfoDbs(RecipeStaticInfo* pRecipeInfo, eExecutionStage executionStage)
{
    pRecipeInfo->deleteStagePatchingPointsDcs(executionStage);
}

bool StaticInfoProcessor::storeNonPatchableBlobsDeviceAddresses(const DeviceMapperInterface& rDeviceMapper,
                                                                const recipe_t&              rRecipe,
                                                                RecipeStaticInfo&            rRecipeInfo,
                                                                std::vector<uint64_t>& rProgramCodeDeviceAddresses)
{
    blob_t*   blobs                    = rRecipe.blobs;
    uint64_t  numberOfBlob             = rRecipe.blobs_nr;
    uint64_t* pExecutionBlobsBuffer    = rRecipe.execution_blobs_buffer;
    uint64_t  executionBlobsBufferSize = rRecipe.execution_blobs_buffer_size;
    uint64_t  blockSize                = 1024 * GCFG_RECIPE_CACHE_BLOCK_SIZE.value();

    std::vector<uint64_t>::const_iterator blocksAddrIterator = rProgramCodeDeviceAddresses.begin();
    std::vector<uint64_t>::const_iterator blockAddrEnd       = rProgramCodeDeviceAddresses.end();

    // Allocate mapped buffer
    uint8_t* pStaticBlobsBuffer = (uint8_t*)pExecutionBlobsBuffer;

    rRecipeInfo.clearProgramCodeBlobsDeviceAddressDatabase(numberOfBlob);

    // Clear and add mapping are done for **CS parser** purpose, only supported in case PRG is in Cache
    // Clear mapping is needed here since on this flow the recipe cache blocks have changed
    // and the old mapping should be erased before insterting a new one
    rRecipeInfo.clearProgramCodeBlockMapping();

    LOG_DEBUG(SYN_PROG_DWNLD, "Total size of execution blobs buffer to copy: {}", executionBlobsBufferSize);

    // Iterate over all execution blobs and_add ProgramCodeBlob device address
    uint64_t blockSizeLeft       = blockSize;
    uint8_t* currentAddrInBuffer = pStaticBlobsBuffer;

    STAT_GLBL_START(calcBlobDeviceAddr);
    for (uint64_t blobIndex = 0; blobIndex < numberOfBlob; blobIndex++)
    {
        blob_t* pCurrentBlob = &blobs[blobIndex];

        if (pCurrentBlob->blob_type.requires_patching)
        {
            continue;
        }

        if (!_addProgramCodeBlobDeviceAddress(rRecipe,
                                              currentAddrInBuffer,
                                              blobIndex,
                                              blockAddrEnd,
                                              blocksAddrIterator,
                                              blockSizeLeft,
                                              &rRecipeInfo,
                                              blockSize))
        {
            LOG_ERR(SYN_PROG_DWNLD, "{}: Failed to copy blobs to mapped buffer", HLLOG_FUNC);
            return false;
        }

        currentAddrInBuffer += pCurrentBlob->size;
    }
    STAT_GLBL_COLLECT_TIME(calcBlobDeviceAddr, globalStatPointsEnum::calcBlobDeviceAddr);

    uint32_t       i;
    const uint32_t numOfBlocks = rProgramCodeDeviceAddresses.size();
    const uint32_t leftover    = executionBlobsBufferSize % blockSize;

    for (i = 1, blocksAddrIterator = rProgramCodeDeviceAddresses.begin(); blocksAddrIterator != blockAddrEnd;
         blocksAddrIterator++, i++)
    {
        if (i == numOfBlocks && leftover != 0)
        {
            blockSize = leftover;
        }

        // Only supported in case PRG is in Cache
        if (!rRecipeInfo.addProgramCodeBlockMapping(*blocksAddrIterator, (uint64_t)pStaticBlobsBuffer, blockSize))
        {
            LOG_ERR(SYN_PROG_DWNLD, "{}: Failed to add mapping for program-code blocks", HLLOG_FUNC);
        }

        pStaticBlobsBuffer += blockSize;
    }

    return true;
}

bool StaticInfoProcessor::_addProgramCodeBlobDeviceAddress(const recipe_t&                        rRecipe,
                                                           uint8_t*                               currentAddrInBuffer,
                                                           uint64_t                               blobIdx,
                                                           std::vector<uint64_t>::const_iterator& blockAddrEnd,
                                                           std::vector<uint64_t>::const_iterator& blocksAddrIterator,
                                                           uint64_t&                              blockSizeLeft,
                                                           RecipeStaticInfo*                      pRecipeInfo,
                                                           uint64_t                               blockSize)
{
    const blob_t* pCurrentBlob   = &rRecipe.blobs[blobIdx];
    uint64_t      blobSize       = pCurrentBlob->size;
    uint64_t      leftSizeToCopy = blobSize;
    uint64_t      blobData       = (uint64_t)pCurrentBlob->data;

    while (leftSizeToCopy > 0)
    {
        if (blocksAddrIterator == blockAddrEnd)
        {
            LOG_ERR(SYN_PROG_DWNLD, "blocksAddrIterator equals to the end of list");
            return false;
        }
        uint64_t alignmentInBlock = blockSize - blockSizeLeft;
        uint64_t addrInBlock      = *blocksAddrIterator + alignmentInBlock;

        if (leftSizeToCopy > blockSizeLeft)
        {
            if (blockSizeLeft != 0)
            {
                // update m_programCodeBlobsToDeviceAddress: vec[blob]=map(addr,size)
                pRecipeInfo->addProgramCodeBlobDeviceAddress(blobIdx, blobData, addrInBlock, blockSizeLeft);
                leftSizeToCopy -= blockSizeLeft;
                currentAddrInBuffer += blockSizeLeft;
            }

            blocksAddrIterator++;
            blockSizeLeft = blockSize;
            continue;
        }

        pRecipeInfo->addProgramCodeBlobDeviceAddress(blobIdx, blobData, addrInBlock, leftSizeToCopy);
        blockSizeLeft -= leftSizeToCopy;
        break;
    }
    return true;
}

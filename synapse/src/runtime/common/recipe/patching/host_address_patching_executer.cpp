#include "host_address_patching_executer.hpp"

#include "define.hpp"
#include "defenders.h"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "utils.h"
#include "recipe.h"
#include "profiler_api.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"

using namespace patching;

synStatus HostAddressPatchingExecuter::verify(const recipe_t& rRecipe, uint64_t maxSectionId)
{
    const uint64_t maxSectionsDbSize = maxSectionId + 1;
    synDeviceType  deviceType        = (synDeviceType)RecipeUtils::getConfVal(&rRecipe, gc_conf_t::DEVICE_TYPE).value();

    bool patchingStatus = verifyPatchingInformation(rRecipe.patch_points,
                                                    rRecipe.patch_points_nr,
                                                    rRecipe.blobs,
                                                    rRecipe.blobs_nr,
                                                    maxSectionsDbSize,
                                                    deviceType);
    if (!patchingStatus)
    {
        LOG_ERR(SYN_RECIPE, "{}: Full patching information is invalid", HLLOG_FUNC);
        return synFail;
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 * Function: executePatching
 * @brief This function performs patching over DC-oriented patching-points
 * @output boolean
 *
 * The function goes over each DC's PPs
 * It patch it according to its DC-Location and other (recipe-original) PP's description
 * The last PP, over each DC, might be split between two DCs. Hence, a separate special handling
 * is being executed for that scenario [patch each part separately]
 ***************************************************************************************************
 */
bool HostAddressPatchingExecuter::executePatchingInDataChunks(const uint64_t* const            newAddressToBeSetDb,
                                                              const data_chunk_patch_point_t*& currPatchPoint,
                                                              const std::vector<uint64_t>&     dataChunksHostAddresses,
                                                              const uint64_t                   dataChunkSize,
                                                              const uint32_t                   lastNodeIndexToPatch,
                                                              const uint32_t                   sobBaseAddress)
{
    PROFILER_COLLECT_TIME()

    LOG_TRACE(SYN_PATCHING, "{}: Start patching-execution", HLLOG_FUNC);
    bool     shouldLog           = LOG_LEVEL_AT_LEAST_TRACE(SYN_PATCHING);
    unsigned currPatchPointIndex = 0;  // used only if shouldLog
    // node_exe_index index 0 is reserved for general patch points, executable nodes start at index 1.
    int numberOfPatchedPoints = 0;
    while (currPatchPoint->node_exe_index <= lastNodeIndexToPatch + 1)
    {
        // calculate patch point address
        auto dataChunkIdx =
            currPatchPoint->data_chunk_index;  // current stage data chunk - use in case that DCs are not sorted
        const uint64_t currentDataChunkHostAddress = dataChunksHostAddresses[dataChunkIdx];
        uint64_t       offsetInDataChunk           = currPatchPoint->offset_in_data_chunk;

        // calculate patch point value
        uint32_t* pPatchedLocation = (uint32_t*)(currentDataChunkHostAddress + offsetInDataChunk);
        ptrToInt  patchedPointValue;
        if (unlikely(currPatchPoint->type == patch_point_t::SOB_PATCH_POINT))
        {
            *(pPatchedLocation)   = sobBaseAddress;
            patchedPointValue.u64 = *(pPatchedLocation);
        }
        else  // memory patch point
        {
            uint64_t sectionAddress = newAddressToBeSetDb[currPatchPoint->memory_patch_point.section_idx];
            patchedPointValue.u64   = sectionAddress + currPatchPoint->memory_patch_point.effective_address;

            // actual patching
            *pPatchedLocation = patchedPointValue.u32[(currPatchPoint->type & 1)];  // will be overridden in some cases
            if (currPatchPoint->type == patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT)
            {
                *(pPatchedLocation + 1) = patchedPointValue.u32[1];
            }
        }
        if (unlikely(shouldLog))  // TODO need to make this better
        {
            LOG_TRACE(
                SYN_PATCHING,
                "{}: Patching (DC {} PP-Index in Stage {}) DC-Address 0x{:x} Offset-in-DC 0x{:x},"
                " section {}, node {}, effective-address 0x{:x} patched-location 0x{:x} patch-type {} value 0x{:x}",
                HLLOG_FUNC,
                dataChunkIdx,
                currPatchPointIndex,
                currentDataChunkHostAddress,
                offsetInDataChunk,
                currPatchPoint->memory_patch_point.section_idx,
                currPatchPoint->node_exe_index,
                currPatchPoint->memory_patch_point.effective_address,
                (uint64_t)currentDataChunkHostAddress + offsetInDataChunk,
                currPatchPoint->type,
                patchedPointValue.u64);
            currPatchPointIndex++;
        }
        currPatchPoint++;
        numberOfPatchedPoints++;
    }
    LOG_TRACE(SYN_PATCHING,
              "{}: Patching execution successfully completed {} patch-points",
              HLLOG_FUNC,
              currPatchPointIndex);

    if (GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())
    {
        char desc[50] = {};
        snprintf(desc, sizeof(desc), "%s count=%d", "executePatching", numberOfPatchedPoints);
        PROFILER_MEASURE_TIME(desc)
    }

    return true;
}

// The executor will go over the PPs to locate the PP-location (where patching should be done)
// and use the PP-Id, from each PP to retrieve the relevant patching-information entry,
// which contains information about how to patch (what is the value to patch)
bool HostAddressPatchingExecuter::executePatchingInBuffer(
    const HostAddressPatchingInformation* pHostAddressPatchingInfo,
    const patch_point_t*                  patchingPoints,
    const uint64_t                        numberOfPatchingPoints,
    blob_t*                               blobs,
    const uint64_t                        numberOfBlobs,
    const uint32_t                        sobBaseAddr)
{
    LOG_TRACE(SYN_PATCHING, "{}: Start patching-execution", HLLOG_FUNC);

    CHECK_POINTER(SYN_PATCHING, pHostAddressPatchingInfo, "Patching-Info", false);

    if (pHostAddressPatchingInfo == nullptr)
    {
        LOG_ERR(SYN_PATCHING,
                "{}: Failed to execute patching."
                " Unexpected patching-information parameter type",
                HLLOG_FUNC);

        return false;
    }

    HB_ASSERT(pHostAddressPatchingInfo->isInitialized(), "Host-Address patching-info is not initialized");

    if (patchingPoints == nullptr)
    {
        return true;
    }

    if (!pHostAddressPatchingInfo->validateAllSectionsAddressSet())
    {
        LOG_ERR(SYN_PATCHING,
                "{}: Failed to execute patching."
                "Some patch-point IDs are not set",
                HLLOG_FUNC);

        return false;
    }

    const uint64_t* newAddressToBeSetDb = pHostAddressPatchingInfo->getSectionsToHostAddressDB();

    bool log = LOG_LEVEL_AT_LEAST_TRACE(SYN_PATCHING);

    const patch_point_t* pCurrentPatchedPoint = patchingPoints;
    for (uint64_t patchPointIdx = 0; patchPointIdx < numberOfPatchingPoints; patchPointIdx++, pCurrentPatchedPoint++)
    {
        uint64_t blobSectionIdx = pCurrentPatchedPoint->memory_patch_point.section_idx;

        ptrToInt  patchedPointValue;
        blob_t*   pCurrentBlob         = blobs + pCurrentPatchedPoint->blob_idx;
        uint32_t* pPatchedBlobLocation = ((uint32_t*)pCurrentBlob->data) + pCurrentPatchedPoint->dw_offset_in_blob;

        if (unlikely(pCurrentPatchedPoint->type == patch_point_t::SOB_PATCH_POINT))
        {
            *pPatchedBlobLocation = sobBaseAddr;
            patchedPointValue.u64 = *pPatchedBlobLocation;
        }
        else
        {
            uint64_t sectionAddress = newAddressToBeSetDb[blobSectionIdx];
            patchedPointValue.u64   = sectionAddress;
            patchedPointValue.u64 += pCurrentPatchedPoint->memory_patch_point.effective_address;
            *pPatchedBlobLocation = patchedPointValue.u32[(pCurrentPatchedPoint->type & 1)];
            if (pCurrentPatchedPoint->type == patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT)
            {
                *(pPatchedBlobLocation + 1) = patchedPointValue.u32[1];
            }
        }

        if (log)
        {
            LOG_TRACE(SYN_PATCHING,
                      "{}: Patching (PP {}), blob {} blob-offset 0x{:x} section {}"
                      " effective-address 0x{:x} patched-location 0x{:x} patch-type {} value 0x{:x}",
                      HLLOG_FUNC,
                      patchPointIdx,
                      pCurrentPatchedPoint->blob_idx,
                      pCurrentPatchedPoint->dw_offset_in_blob,
                      blobSectionIdx,
                      pCurrentPatchedPoint->memory_patch_point.effective_address,
                      (uint64_t)pPatchedBlobLocation,
                      pCurrentPatchedPoint->type,
                      patchedPointValue.u64);
        }
    }

    LOG_TRACE(SYN_PATCHING, "{}: Patching-execution completed successfully", HLLOG_FUNC);
    return true;
}

bool HostAddressPatchingExecuter::verifyPatchingInformation(const patch_point_t* patchingPoints,
                                                            uint64_t             numberOfPatchingPoints,
                                                            const blob_t*        blobs,
                                                            uint64_t             numberOfBlobs,
                                                            uint64_t             sectionsDbSize,
                                                            synDeviceType        deviceType)
{
    LOG_DEBUG_T(SYN_PATCHING,
               "Verify patching info, {} patching points for {} blobs",
               numberOfPatchingPoints,
               numberOfBlobs);

    if (numberOfPatchingPoints == 0)
    {
        return true;
    }

    if (numberOfBlobs == 0)
    {
        LOG_ERR(SYN_PATCHING,
                "{}: numberOfPatchingPoints {} numberOfBlobs is zero",
                HLLOG_FUNC,
                numberOfPatchingPoints);
        return false;
    }

    CHECK_POINTER_RET(SYN_PATCHING, blobs, "blobs", false);
    CHECK_POINTER_RET(SYN_PATCHING, patchingPoints, "patchingPoints", false);

    const patch_point_t* pCurrentPatchedPoint = patchingPoints;
    for (uint64_t patchPointIdx = 0; patchPointIdx < numberOfPatchingPoints; patchPointIdx++, pCurrentPatchedPoint++)
    {
        if (pCurrentPatchedPoint->type > patch_point_t::SOB_PATCH_POINT)
        {
            LOG_ERR(SYN_PATCHING, "{}: Unsupported patching-type ({})", HLLOG_FUNC, pCurrentPatchedPoint->type);
            return false;  // In the future we may want to define that we can skip them instead
        }

        if (pCurrentPatchedPoint->blob_idx >= numberOfBlobs)
        {
            LOG_ERR(SYN_PATCHING,
                    "{}: Invalid blob-index (index {}, max {})",
                    HLLOG_FUNC,
                    pCurrentPatchedPoint->blob_idx,
                    numberOfBlobs);
            return false;
        }

        const blob_t* pCurrentBlob = blobs + pCurrentPatchedPoint->blob_idx;

        if ((deviceType == synDeviceGaudi && !pCurrentBlob->blob_type.requires_patching) ||
            (deviceType == synDeviceGaudi2 && pCurrentBlob->blob_type.static_exe))
        {
            LOG_ERR(SYN_PATCHING,
                    "{}: Patch-point {} points to a blob {} that does not require patching",
                    HLLOG_FUNC,
                    patchPointIdx,
                    pCurrentPatchedPoint->blob_idx);
            return false;
        }

        if (pCurrentBlob->data == nullptr)
        {
            LOG_ERR(SYN_PATCHING,
                    "{}: Null pointer passed as blob's ({}) data",
                    HLLOG_FUNC,
                    pCurrentPatchedPoint->blob_idx);
            return false;
        }

        if (pCurrentPatchedPoint->dw_offset_in_blob > pCurrentBlob->size)
        {
            LOG_ERR(SYN_PATCHING,
                    "{}: Out-of-range blob-offset (blob-size {} required {})",
                    HLLOG_FUNC,
                    pCurrentBlob->size,
                    pCurrentPatchedPoint->dw_offset_in_blob);
            return false;
        }

        if (pCurrentPatchedPoint->type != patch_point_t::SOB_PATCH_POINT &&
            pCurrentPatchedPoint->memory_patch_point.section_idx >= sectionsDbSize)
        {
            LOG_ERR(SYN_PATCHING,
                    "{}: blob-patch-point's section-index {} is OOB (sectionsDbSize {})",
                    HLLOG_FUNC,
                    pCurrentPatchedPoint->memory_patch_point.section_idx,
                    sectionsDbSize);
            return false;
        }
    }

    return true;
}

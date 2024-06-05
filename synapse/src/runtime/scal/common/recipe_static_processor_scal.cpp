#include "recipe_static_processor_scal.hpp"

#include "defs.h"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "math_utils.h"
#include "recipe.h"

#include "runtime/common/recipe/recipe_utils.hpp"

#include "runtime/scal/common/infra/scal_utils.hpp"

#include "runtime/scal/common/recipe_launcher/mapped_mem_mgr.hpp"

#include "runtime/scal/common/recipe_static_info_scal.hpp"

#include "runtime/scal/gaudi2/entities/recipe_reader_helper.hpp"
#include "runtime/scal/gaudi3/entities/recipe_reader_helper.hpp"

synStatus DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceType          deviceType,
                                                           const basicRecipeInfo& rBasicRecipeInfo,
                                                           RecipeStaticInfoScal&  rRecipeStaticInfoScal)
{
    synStatus status = processtRecipeSections(deviceType,
                                              *rBasicRecipeInfo.recipe,
                                              rRecipeStaticInfoScal,
                                              RecipeUtils::isDsd(rBasicRecipeInfo),
                                              RecipeUtils::isIH2DRecipe(rBasicRecipeInfo.recipe));
    if (status != synSuccess)
    {
        return status;
    }
    uint64_t dcSize = MappedMemMgr::getDcSize();
    bool     res    = rRecipeStaticInfoScal.recipeAddrPatcher.init(*rBasicRecipeInfo.recipe, dcSize);
    if (!res)
    {
        return synFail;
    }

    res = rRecipeStaticInfoScal.recipeDsdPpInfo.init(rBasicRecipeInfo, rRecipeStaticInfoScal, dcSize);
    if (!res)
    {
        return synFail;
    }

    return status;
}

synStatus DeviceAgnosticRecipeStaticProcessorScal::processtRecipeSections(synDeviceType         deviceType,
                                                                          const recipe_t&       rRecipe,
                                                                          RecipeStaticInfoScal& rRecipeStaticInfoScal,
                                                                          bool                  isDsd,
                                                                          bool                  isIH2DRecipe)
{
    HB_ASSERT((deviceType == synDeviceGaudi3) || (deviceType == synDeviceGaudi2), "Device-type not supported");
    const common::RecipeReaderHelper* pRecipeReaderHelper = nullptr;
    if (deviceType == synDeviceGaudi2)
    {
        pRecipeReaderHelper = gaudi2::RecipeReaderHelper::getInstance();
    }
    else if (deviceType == synDeviceGaudi3)
    {
        pRecipeReaderHelper = gaudi3::RecipeReaderHelper::getInstance();
    }
    const uint32_t programAlignment = pRecipeReaderHelper->getDynamicEcbListBufferSize();

    auto& sec = rRecipeStaticInfoScal.recipeSections;

    sec.resize(rRecipe.arc_jobs_nr * 2 + ECB_LIST_FIRST);

    // patchable (HBM global)
    sec[PATCHABLE].recipeAddr = (uint8_t*)rRecipe.patching_blobs_buffer;
    sec[PATCHABLE].size       = rRecipe.patching_blobs_buffer_size;
    sec[PATCHABLE].align      = programAlignment;

    // HBM global
    sec[PROGRAM_DATA].recipeAddr = (uint8_t*)rRecipe.program_data_blobs_buffer;
    sec[PROGRAM_DATA].size       = rRecipe.program_data_blobs_size;
    sec[PROGRAM_DATA].align      = programAlignment; // program data is always:
                                                     // 1. the first on the HBM (HbmGlblMemMgr make sure chunks are aligned to PRG_DATA_ALIGN)
                                                     // 2. copied first in a group of sections
                                                     // 1+2 -> There is no special aligment needed

    sec[NON_PATCHABLE].recipeAddr = (uint8_t*)rRecipe.execution_blobs_buffer;
    sec[NON_PATCHABLE].size       = rRecipe.execution_blobs_buffer_size;
    sec[NON_PATCHABLE].align      = programAlignment;

    // ARC_HBM
    sec[DYNAMIC].recipeAddr = (uint8_t*)rRecipe.dynamic_blobs_buffer;
    sec[DYNAMIC].size       = rRecipe.dynamic_blobs_buffer_size;

    // In DSD we also patch the dynamic section, make sure it starts on a new data-chunk. GC makes sure no blob crosses
    // PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES. We make sure that (DC-size % PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES == 0) in the
    // init of the mapped-mem-mgr
    // Note, Dynamic section is always first in the arc-hbm, so the alignment doesn't effect the hbm, only the mapped-memory
    sec[DYNAMIC].align      = MappedMemMgr::getDcSize();


    for (int i = 0; i < rRecipe.arc_jobs_nr; i++)
    {
        sec[ECB_LIST_FIRST + 2 * i].recipeAddr = rRecipe.arc_jobs[i].dynamic_ecb.cmds;
        sec[ECB_LIST_FIRST + 2 * i].size       = rRecipe.arc_jobs[i].dynamic_ecb.cmds_size;
        sec[ECB_LIST_FIRST + 2 * i].align      = programAlignment;

        sec[ECB_LIST_FIRST + 1 + 2 * i].recipeAddr = rRecipe.arc_jobs[i].static_ecb.cmds;
        sec[ECB_LIST_FIRST + 1 + 2 * i].size       = rRecipe.arc_jobs[i].static_ecb.cmds_size;
        sec[ECB_LIST_FIRST + 1 + 2 * i].align      = programAlignment;
    }

    /*******    Mapped, patchable area  *************/
    uint64_t offsetMapped = 0;

    sec[PATCHABLE].offsetMapped = 0;
    offsetMapped += sec[PATCHABLE].size;

    if (isDsd)
    {
        offsetMapped              = round_to_multiple(offsetMapped, sec[DYNAMIC].align);
        sec[DYNAMIC].offsetMapped = offsetMapped;
        offsetMapped += sec[DYNAMIC].size;
    }

    if (isIH2DRecipe)
    {
        offsetMapped                   = round_to_multiple(offsetMapped, sec[PROGRAM_DATA].align);
        sec[PROGRAM_DATA].offsetMapped = offsetMapped;
    }

    /*******     Mapped, non-patchable area   *************/
    offsetMapped = 0;  // start from 0 (different area in mapped memory)

    if (!isIH2DRecipe)
    {
        sec[PROGRAM_DATA].offsetMapped = offsetMapped;
        offsetMapped += sec[PROGRAM_DATA].size;
    }

    offsetMapped                    = round_to_multiple(offsetMapped, sec[NON_PATCHABLE].align);
    sec[NON_PATCHABLE].offsetMapped = offsetMapped;
    offsetMapped += sec[NON_PATCHABLE].size;

    if (!isDsd)
    {
        offsetMapped              = round_to_multiple(offsetMapped, sec[DYNAMIC].align);
        sec[DYNAMIC].offsetMapped = offsetMapped;
        offsetMapped += sec[DYNAMIC].size;
    }

    for (int i = 0; i < rRecipe.arc_jobs_nr * 2; i++)
    {
        int secId               = ECB_LIST_FIRST + i;
        offsetMapped            = round_to_multiple(offsetMapped, sec[secId].align);
        sec[secId].offsetMapped = offsetMapped;
        offsetMapped += sec[secId].size;
    }

    /*******  Offset in HBM  *************/
    sec[PROGRAM_DATA].offsetHbm = 0; // NOTE: We don't align the PROGRAM_DATA as we expect it to be the first
                                     // in the HBM (which is aligned). If this is not the case, make sure to align it to
                                     // DeviceAgnosticRecipeStaticProcessorScal::PRG_DATA_ALIGN

    uint64_t offsetHBM = sec[PROGRAM_DATA].offsetHbm;

    offsetHBM += sec[PROGRAM_DATA].size;

    offsetHBM                    = round_to_multiple(offsetHBM, sec[NON_PATCHABLE].align);
    sec[NON_PATCHABLE].offsetHbm = offsetHBM;
    offsetHBM += sec[NON_PATCHABLE].size;

    offsetHBM                = round_to_multiple(offsetHBM, sec[PATCHABLE].align);
    sec[PATCHABLE].offsetHbm = offsetHBM;
    offsetHBM += sec[PATCHABLE].size;

    /******* Offset in ARC HBM *************/
    uint64_t offsetArcHBM  = 0;
    sec[DYNAMIC].offsetHbm = offsetArcHBM;
    offsetArcHBM += sec[DYNAMIC].size;

    for (int i = 0; i < rRecipe.arc_jobs_nr * 2; i++)
    {
        int secId            = ECB_LIST_FIRST + i;
        offsetArcHBM         = round_to_multiple(offsetArcHBM, sec[secId].align);
        sec[secId].offsetHbm = offsetArcHBM;
        offsetArcHBM += sec[secId].size;
    }

    // log the values
    if (HLLOG_UNLIKELY(hl_logger::anyLogLevelAtLeast(synapse::LogManager::LogType::SYN_RECIPE, HLLOG_LEVEL_DEBUG)))
    {
        for (int i = 0; i < sec.size(); i++)
        {
            LOG_DEBUG(SYN_RECIPE,
                      "rRecipeStaticInfoScal: for {}-{}: recipeAddr {:x} size {:x} Offset mapped {:x} HBM {:x}",
                      i,
                      ScalSectionTypeToString((SectionType)i),
                      TO64(sec[i].recipeAddr),
                      sec[i].size,
                      sec[i].offsetMapped,
                      sec[i].offsetHbm);
        }
    }

    rRecipeStaticInfoScal.m_glbHbmSizeNoPatch = sec[NON_PATCHABLE].offsetMapped + sec[NON_PATCHABLE].size;
    if(isIH2DRecipe)
    {
        uint64_t glbHbmSizeOffset = round_to_multiple(sec[PROGRAM_DATA].size, sec[NON_PATCHABLE].align) + sec[NON_PATCHABLE].size;
        rRecipeStaticInfoScal.m_glbHbmSizeTotal = round_to_multiple(glbHbmSizeOffset, sec[PATCHABLE].align) + sec[PATCHABLE].size;
    }
    else
    {
        rRecipeStaticInfoScal.m_glbHbmSizeTotal =
            round_to_multiple(rRecipeStaticInfoScal.m_glbHbmSizeNoPatch, sec[PATCHABLE].align) + sec[PATCHABLE].size;
    }
    rRecipeStaticInfoScal.m_arcHbmSize =
        sec[sec.size() - 1].offsetHbm + sec[sec.size() - 1].size - sec[DYNAMIC].offsetHbm;

    rRecipeStaticInfoScal.m_mappedSizeNoPatch = sec[sec.size() - 1].offsetMapped + sec[sec.size() - 1].size;

    if (isDsd)
    {
        if (isIH2DRecipe)
        {
            rRecipeStaticInfoScal.m_mappedSizePatch = sec[PROGRAM_DATA].offsetMapped + sec[PROGRAM_DATA].size;
        }
        else
        {
            rRecipeStaticInfoScal.m_mappedSizePatch = sec[DYNAMIC].offsetMapped + sec[DYNAMIC].size;
        }
    }
    else
    {
        rRecipeStaticInfoScal.m_mappedSizePatch = sec[PATCHABLE].size;
    }

    if (rRecipe.arc_jobs_nr != 0)
    {
        const RecipeSingleSection& firstECBSection = sec[ECB_LIST_FIRST];
        const RecipeSingleSection& lastECBSection  = sec.back();
        rRecipeStaticInfoScal.m_ecbListsTotalSize =
            lastECBSection.offsetHbm + lastECBSection.size - firstECBSection.offsetHbm;
    }
    else
    {
        rRecipeStaticInfoScal.m_ecbListsTotalSize = 0;
    }

    LOG_DEBUG(SYN_RECIPE,
              "rRecipeStaticInfoScal: hbmsize NoPatch {:x}, Total {:x}, arcGbmSize {:x}, mappedNoPatch {:x}, "
              "mappedPatch {:x}, "
              "ecbListsTotalSize {:x}",
              rRecipeStaticInfoScal.m_glbHbmSizeNoPatch,
              rRecipeStaticInfoScal.m_glbHbmSizeTotal,
              rRecipeStaticInfoScal.m_arcHbmSize,
              rRecipeStaticInfoScal.m_mappedSizeNoPatch,
              rRecipeStaticInfoScal.m_mappedSizePatch,
              rRecipeStaticInfoScal.m_ecbListsTotalSize);

    // get the engine array for the compute
    auto& numGrps = rRecipeStaticInfoScal.m_computeEngineGrpArr.numEngineGroups;
    auto& engArr  = rRecipeStaticInfoScal.m_computeEngineGrpArr.eng;

    rRecipeStaticInfoScal.m_computeEngineGrpArr.numEngineGroups = 0;

    bool shouldAddTpcGroup = rRecipe.valid_nop_kernel;
    for (int i = 0; i < rRecipe.arc_jobs_nr; i++)
    {
        uint8_t eng = ScalUtils::convertLogicalEngineIdTypeToScalEngineGroupType(rRecipe.arc_jobs[i].logical_engine_id);
        if (eng != SCAL_CME_GROUP)
        {
            engArr[numGrps] = eng;

            if (eng == SCAL_TPC_COMPUTE_GROUP)
            {
                shouldAddTpcGroup = false;
            }
            numGrps++;
        }
    }

    if (shouldAddTpcGroup)
    {
        engArr[numGrps] = SCAL_TPC_COMPUTE_GROUP;
        numGrps++;
    }

    return synSuccess;
}

const uint32_t DeviceAgnosticRecipeStaticProcessorScal::PRG_DATA_ALIGN;

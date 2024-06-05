#include <habana_global_conf_runtime.h>
#include <utils.h>
#include "memory_management/memory_protection.hpp"
#include "log_manager.h"
#include "recipe_dsd_pp_info.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/scal/common/recipe_static_info_scal.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"

/*
 ***************************************************************************************************
 *   @brief init() - translates the recipe patch-points to a new database that is more efficient
 *                   when doing the patching. Needs to be called once
 *
 *   Note: we have one set of patch-points that includes all.
 *         And N sets of patch points, one per section type
 *
 *   @param  recipe, data-chunk size (that we later do the patching on)
 *   @return bool (false-fail, true-OK)
 *
 ***************************************************************************************************
 */
bool RecipeDsdPpInfo::init(const basicRecipeInfo&      rBasicRecipeInfo,
                           const RecipeStaticInfoScal& rRecipeStaticInfoScal,
                           uint32_t                    dcSize)
{
    if (!RecipeUtils::isDsd(rBasicRecipeInfo))
    {
        return true;
    }

    if (dcSize == 0)
    {
        LOG_ERR(SYN_RECIPE, "dcSize is 0, illegal");
        return false;
    }

    // calculate Total Amount Of Sm Patch Points
    shape_plane_graph_t*      shapePlanRecipe       = rBasicRecipeInfo.shape_plan_recipe;
    uint32_t                  amountOfSmPatchPoints = 0;
    const shape_plane_node_t* pCurrentNode          = shapePlanRecipe->sp_nodes;
    for (uint32_t nodeIndex = 0; nodeIndex < shapePlanRecipe->sp_node_nr; nodeIndex++, pCurrentNode++)
    {
        amountOfSmPatchPoints += pCurrentNode->node_patch_points_nr;
    }

    m_pSmPatchPointsDataChunksInfo.init(amountOfSmPatchPoints);
    m_pSmPatchPointsDataChunksInfo.m_singleChunkSize = dcSize;

    pCurrentNode = shapePlanRecipe->sp_nodes;

    unsigned curPPIndex = 0;
    for (uint32_t nodeIndex = 0; nodeIndex < shapePlanRecipe->sp_node_nr; nodeIndex++, pCurrentNode++)
    {
        sm_patch_point_t* pCurrentBlobSmPatchPoint = pCurrentNode->node_patch_points;

        for (uint32_t smPpIndex = 0; smPpIndex < pCurrentNode->node_patch_points_nr;
             smPpIndex++, pCurrentBlobSmPatchPoint++)
        {
            setDsdDcPp(m_pSmPatchPointsDataChunksInfo.m_dataChunkSmPatchPoints[curPPIndex],
                       *pCurrentBlobSmPatchPoint,
                       *rBasicRecipeInfo.recipe,
                       dcSize,
                       rRecipeStaticInfoScal);
            curPPIndex++;
        }
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @brief setDsdDcPp() - utility function used by init
 *
 *   @param
 *   @return None
 *
 ***************************************************************************************************
 */
void RecipeDsdPpInfo::setDsdDcPp(data_chunk_sm_patch_point_t& dcPp,
                                 sm_patch_point_t&            recipePp,
                                 const recipe_t&              recipe,
                                 uint32_t                     dcSize,
                                 const RecipeStaticInfoScal&  rRecipeStaticInfoScal)
{
    EFieldType fieldType = recipePp.patch_point_type;
    if (fieldType != FIELD_DYNAMIC_ADDRESS)
    {
        uint32_t recipeBlobIdx = recipePp.blob_idx;
        blob_t*  curBlob       = &recipe.blobs[recipeBlobIdx];
        uint8_t* ppBlobAddr    = nullptr;

        uint64_t offset = 0;
        if (curBlob->blob_type.dynamic_exe)
        {
            ppBlobAddr = (uint8_t*)(((uint32_t*)recipe.blobs[recipeBlobIdx].data) + recipePp.dw_offset_in_blob);
            offset     = ppBlobAddr - (uint8_t*)recipe.dynamic_blobs_buffer;
            offset += rRecipeStaticInfoScal.recipeSections[DYNAMIC].offsetMapped;
        }
        else if (curBlob->blob_type.requires_patching)
        {
            ppBlobAddr = (uint8_t*)(((uint32_t*)recipe.blobs[recipeBlobIdx].data) + recipePp.dw_offset_in_blob);
            offset     = ppBlobAddr - (uint8_t*)recipe.patching_blobs_buffer;
            offset += rRecipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped;
        }
        else
        {
            HB_ASSERT(false, " blob should also be with requires_patching");
        }

        dcPp.data_chunk_index     = offset / dcSize;
        dcPp.offset_in_data_chunk = offset - (static_cast<uint64_t>(dcPp.data_chunk_index) * dcSize);
    }
    else
    {
        // This is permitted for Gaudi and Gaudi3 but not for Gaudi2.
        // There is no good way to find out whether we are doing Gaudi2 here.
        // TODO add something like
        // HB_ASSERT(isGaudi2, "should not enter here in gaudi2, no support for pp-patching");
        dcPp.patch_point_idx_high = recipePp.patch_point_idx_high;
        dcPp.patch_point_idx_low  = recipePp.patch_point_idx_low;
    }

    dcPp.patch_point_type = recipePp.patch_point_type;
    dcPp.patch_size_dw    = recipePp.patch_size_dw;
    dcPp.roi_idx          = recipePp.roi_idx;

    // pointer to tables in the origin SMF-PP
    dcPp.p_smf_id       = &(recipePp.smf_id);
    dcPp.p_pp_metdata   = recipePp.metadata;
    dcPp.is_unskippable = recipePp.is_unskippable;
}

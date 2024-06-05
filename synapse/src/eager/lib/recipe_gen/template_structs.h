#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"
#include "utils/general_utils.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// This file contains reusable structure for template creation of various engines.
// Note that none of the classes or structures in this file should define virtual methods or destructors.
// Keep this in mind when using these structures, and avoid dynamically allocating any of them.

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// struct PatchPoints
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType patchPointsNr>
struct PatchPoints final
{
    // External access to template parameters
    static constexpr TensorsNrType getPatchPointsNr() { return patchPointsNr; }

    patch_point_t patch_points[patchPointsNr];

    void init(BlobsNrType patchingBlobIdx)
    {
        for (TensorsNrType i = 0; i < patchPointsNr; ++i)
        {
            patch_points[i].blob_idx          = patchingBlobIdx;
            patch_points[i].dw_offset_in_blob = (i + 1) * asicRegsPerEntry;
            patch_points[i].node_exe_index    = 1;
            patch_points[i].type              = patch_point_t::EPatchPointType::SIMPLE_DDW_MEM_PATCH_POINT;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct NodeExeList
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
struct NodeExeList final
{
    node_program_t node_exe_list;
    POINTER_TYPE(node_program_t, program_blobs_nr) program_blobs_nr[tensorsNr];
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WorkspaceSizes
///////////////////////////////////////////////////////////////////////////////////////////////////

// 3 uint64_t for workspace, program and program data
struct WorkspaceSizes final
{
    POINTER_TYPE(recipe_t, workspace_sizes) workspace_sizes[3];
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct RecipeConfParams
///////////////////////////////////////////////////////////////////////////////////////////////////

struct RecipeConfParams final
{
    gc_conf_t recipe_conf_params[5];

    void
    init(synDeviceType deviceType, uint64_t tpcEnginesMask, uint64_t numMmeEngines, uint64_t internalDmaEnginesMask)
    {
        // Somehow runtime failed when I tried to shuffle the following order, and perhaps all those configs are
        // essential:
        recipe_conf_params[0].conf_id    = gc_conf_t::recipeCompileParams::DEVICE_TYPE;
        recipe_conf_params[0].conf_value = deviceType;
        recipe_conf_params[1].conf_id    = gc_conf_t::recipeCompileParams::TIME_STAMP;
        recipe_conf_params[1].conf_value = 0;
        recipe_conf_params[2].conf_id    = gc_conf_t::recipeCompileParams::TPC_ENGINE_MASK;
        recipe_conf_params[2].conf_value = tpcEnginesMask;
        recipe_conf_params[3].conf_id    = gc_conf_t::recipeCompileParams::MME_NUM_OF_ENGINES;
        recipe_conf_params[3].conf_value = numMmeEngines;
        recipe_conf_params[4].conf_id    = gc_conf_t::recipeCompileParams::DMA_ENGINE_MASK;
        recipe_conf_params[4].conf_value = internalDmaEnginesMask;
    }
};

}  // namespace eager_mode
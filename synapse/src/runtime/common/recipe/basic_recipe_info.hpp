#pragma once

#include <cstdint>
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include <atomic>

struct recipe_t;
struct shape_plane_graph_t;
class RecipeAllocator;

struct RecipeStats
{
    std::atomic<uint64_t> numbSuccessfulLaunch;
};

struct basicRecipeInfo
{
    synStatus tensorRetrieveMetadatasInfosByName(const uint32_t         numOfTensors,
                                                 TensorMetadataInfoExt* tensorsMetadataInfo) const;

    synStatus tensorRetrievePersistentAmount(uint32_t& numOfTensors) const;

    synStatus tensorRetrieveNames(char tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE], const uint32_t numOfTensors) const;

    synStatus tensorRetrieveLaunchInfoById(const uint32_t                   numOfTensors,
                                           synRetrievedLaunchTensorInfoExt* tensorsMetadataInfo) const;

    synStatus getHostMemorySize(uint64_t& retVal) const;
    synStatus getPersistentTensorsMemorySize(uint64_t& retVal) const;
    synStatus getConstSectionMemorySize(uint64_t& retVal) const;
    synStatus getRecipeHbmMemorySize(uint64_t& retVal) const;

    // For Gaudi recipe_t support
    recipe_t*            recipe;
    shape_plane_graph_t* shape_plan_recipe;
    const char*          recipeDebugInfo;
    uint32_t             recipeDebugInfoSize;
    RecipeAllocator*     recipeAllocator;
    RecipeStats          recipeStats;
};

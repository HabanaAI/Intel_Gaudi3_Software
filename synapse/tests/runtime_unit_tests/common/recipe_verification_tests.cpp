
#include "define_synapse_common.hpp"
#include "recipe.h"
#include "graph_entries_container.hpp"
#include "habana_global_conf_runtime.h"
#include "runtime/common/recipe/recipe_verification.hpp"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "recipe_allocator.h"
#include "test_dummy_recipe.hpp"

#include "runtime/scal/common/recipe_static_processor_scal.hpp"
#include "runtime/scal/common/recipe_static_info_scal.hpp"

#include "runtime/scal/gaudi2/device_gaudi2scal.hpp"

#include <gtest/gtest.h>

TEST(UTRecipeVerification, testVerifyProgramCodeBlobs)
{
    recipe_t recipe;

    const uint16_t numOfBlobs          = 3;
    const uint64_t patchingBufferSize  = 0x40;
    const uint64_t executionBufferSize = 0x40;
    const uint64_t dynamicBufferSize   = 0x40;

    blob_t*   recipesBlobs    = new blob_t[numOfBlobs];
    uint64_t* patchingBuffer  = new uint64_t[patchingBufferSize];
    uint64_t* executionBuffer = new uint64_t[executionBufferSize];
    uint32_t* dynamicBuffer   = new uint32_t[dynamicBufferSize];

    // Patchable buffer
    recipe.patching_blobs_buffer      = patchingBuffer;
    recipe.patching_blobs_buffer_size = patchingBufferSize;

    // Execution buffer
    recipe.execution_blobs_buffer = executionBuffer;
    ;
    recipe.execution_blobs_buffer_size = executionBufferSize;

    // Dynamic buffer
    recipe.dynamic_blobs_buffer      = dynamicBuffer;
    recipe.dynamic_blobs_buffer_size = dynamicBufferSize;

    recipe.blobs_nr = numOfBlobs;
    recipe.blobs    = recipesBlobs;

    // Good case
    recipesBlobs[0].blob_type_all = blob_t::PATCHING;
    recipesBlobs[0].data          = (void*)recipe.patching_blobs_buffer;
    recipesBlobs[0].size          = recipe.patching_blobs_buffer_size;

    recipesBlobs[1].blob_type_all = blob_t::EXE;
    recipesBlobs[1].data          = (void*)recipe.execution_blobs_buffer;
    recipesBlobs[1].size          = recipe.execution_blobs_buffer_size;

    recipesBlobs[2].blob_type_all = blob_t::DYNAMIC;
    recipesBlobs[2].data          = (void*)recipe.dynamic_blobs_buffer;
    recipesBlobs[2].size          = recipe.dynamic_blobs_buffer_size;
    bool status                   = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, true);

    // Bad case 1 - Patching-Blobs buffer base-address is OOB
    recipesBlobs[0].data = (void*)(recipe.patching_blobs_buffer - 1);
    status               = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, false);
    recipesBlobs[0].data = (void*)(recipe.patching_blobs_buffer + 1);

    // Bad case 2 - Patching-Blobs buffer last-address is OOB
    recipesBlobs[0].size++;
    status = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, false);
    recipesBlobs[0].size--;

    // Bad case 3 - Execution-Buffer base-address is OOB
    recipesBlobs[1].data = (void*)(recipe.execution_blobs_buffer - 1);
    status               = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, false);
    recipesBlobs[1].data = (void*)(recipe.execution_blobs_buffer + 1);

    // Bad case 4 - Execution-Blobs buffer last-address is OOB
    recipesBlobs[1].size++;
    status = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, false);
    recipesBlobs[1].size--;

    // Bad case 5 - Dynamic-Blobs buffer base-address is OOB
    recipesBlobs[2].data = (void*)(recipe.dynamic_blobs_buffer - 1);
    status               = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, false);
    recipesBlobs[2].data = (void*)(recipe.dynamic_blobs_buffer + 1);

    // Bad case 6 - Dynamic-Blobs buffer last-address is OOB
    recipesBlobs[2].size++;
    status = RecipeVerification::verifyProgramCodeBlobs(&recipe);
    ASSERT_EQ(status, false);
    recipesBlobs[2].size--;

    delete[] recipesBlobs;
    delete[] patchingBuffer;
    delete[] executionBuffer;
    delete[] dynamicBuffer;
}

TEST(UTRecipeVerification, testVerifyProgramDataBlobsAll)
{
    const int blobSize = 0x100;

    recipe_t recipe;
    memset(&recipe, 0, sizeof(recipe));

    recipe.program_data_blobs_buffer = (char*)0x2000;
    recipe.program_data_blobs_size   = 0x1000;
    recipe.program_data_blobs_nr     = 10;

    recipe.program_data_blobs = new program_data_blob_t[recipe.program_data_blobs_nr];
    recipe.workspace_nr       = MEMORY_ID_RESERVED_FOR_PROGRAM + 1;
    recipe.workspace_sizes    = new uint64_t[MEMORY_ID_RESERVED_FOR_PROGRAM + 1];

    recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = recipe.program_data_blobs_size;

    for (uint i = 0; i < recipe.program_data_blobs_nr; i++)
    {
        program_data_blob_t& curr = recipe.program_data_blobs[i];
        curr.size                 = blobSize;
        curr.offset_in_section    = i * blobSize;
        if (i >= 5) curr.offset_in_section += 0x10;  // have a small gap after #5.
        curr.data        = recipe.program_data_blobs_buffer + curr.offset_in_section;
        curr.section_idx = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;
    }

    bool rtn;

    // Good case
    rtn = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, true);

    // overlap with next
    recipe.program_data_blobs[3].size = blobSize + 0x10;
    rtn                               = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, false);
    recipe.program_data_blobs[3].size = blobSize;

    // First outside buffer (and mismatch in data/offset)
    recipe.program_data_blobs[0].data -= 0x10;
    rtn = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, false);
    recipe.program_data_blobs[0].data += 0x10;

    // Last is past buffer
    recipe.program_data_blobs[recipe.program_data_blobs_nr - 1].size = 0xF00;
    rtn = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, false);
    recipe.program_data_blobs[recipe.program_data_blobs_nr - 1].size = blobSize;

    // Size of all blobs != 0
    recipe.program_data_blobs[8].size = 0;
    rtn                               = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, false);
    recipe.program_data_blobs[8].size = blobSize;

    // All blobs are in section MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;
    recipe.program_data_blobs[8].section_idx = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA + 1;
    rtn                                      = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, false);
    recipe.program_data_blobs[8].section_idx = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;

    // workspace size != buffer size
    recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] += 1;
    rtn = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, false);
    recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] -= 1;

    // Verify all is good again
    rtn = RecipeVerification::verifyProgramDataBlobs(&recipe, true);
    ASSERT_EQ(rtn, true);

    delete[] recipe.program_data_blobs;
    delete[] recipe.workspace_sizes;
}

TEST(UTRecipeVerification, checkEmptyGraphVerification)
{
    GlobalConfManager::instance().setDeviceType(synDeviceTypeInvalid);
    HabanaGraphPtr pGraph = GraphFactory::createGraph(synDeviceGaudi2, CompilationMode::Graph);

    bool res = pGraph->compile();
    ASSERT_EQ(res, true) << "Failed to compile graph";

    RecipeAllocator recipeAllocator;
    recipe_t*       pRecipe = pGraph->serializeDataPlane(&recipeAllocator);

    shape_plane_graph_t* pShapePlanRecipe;
    if (pGraph->isDynamicShape())
    {
        pShapePlanRecipe = pGraph->serializeShapePlane(&recipeAllocator);
    }
    else
    {
        pShapePlanRecipe = nullptr;
    }

    res = RecipeVerification::verifyRecipe(pRecipe, pShapePlanRecipe);
    ASSERT_EQ(res, true) << "Failed to verifyRecipe";

    GlobalConfManager::instance().setDeviceType(synDeviceTypeInvalid);
    recipeAllocator.freeAll();
}

TEST(UTRecipeVerification, checkRecipeCacheSize)
{
    HabanaGraphPtr pGraph = GraphFactory::createGraph(synDeviceGaudi, CompilationMode::Graph);

    ASSERT_TRUE(pGraph->compile()) << "Failed to compile graph";

    RecipeAllocator recipeAllocator;
    recipe_t*       pRecipe = pGraph->serializeDataPlane(&recipeAllocator);

    auto initialRecipeCacheSize = GCFG_RECIPE_CACHE_SIZE.value();

    ASSERT_TRUE(RecipeVerification::verifyRecipeCacheSize(pRecipe)) << "Failed to verifyRecipeCacheSize";
    ASSERT_TRUE(RecipeVerification::verifyRecipe(pRecipe, nullptr)) << "Failed to verifyRecipe";
    GCFG_RECIPE_CACHE_SIZE.setValue(128);

    ASSERT_FALSE(RecipeVerification::verifyRecipeCacheSize(pRecipe))
        << "should fail verifyRecipeCacheSize for a small cache size";
    ASSERT_FALSE(RecipeVerification::verifyRecipe(pRecipe, nullptr))
        << "should fail verifyRecipe for a small cache size";
    GCFG_RECIPE_CACHE_SIZE.setValue(initialRecipeCacheSize);
    recipeAllocator.freeAll();
}

TEST(UTRecipeVerification, checkHbmMemorySize)
{
    GlobalConfManager::instance().setDeviceType(synDeviceTypeInvalid);

    uint64_t dcSize       = 256 * 1024;
    uint64_t patchSize    = dcSize;
    uint64_t execSize     = dcSize;
    uint64_t dynamicSize  = dcSize;
    uint64_t prgDataSize  = dcSize;
    uint64_t ecbListsSize = 0x800;

    TestDummyRecipe dummyRecipe = TestDummyRecipe(RECIPE_TYPE_NORMAL, patchSize, execSize, dynamicSize, prgDataSize, ecbListsSize, 0, synDeviceGaudi2);

    recipe_t*            pRecipe = dummyRecipe.getRecipe();
    basicRecipeInfo      basicRecipeInfo = {};
    basicRecipeInfo.recipe = pRecipe;

    RecipeStaticInfoScal recipeStaticInfoScal;

    auto status =
        DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2, basicRecipeInfo, recipeStaticInfoScal);
    ASSERT_EQ(status, synSuccess);

    ASSERT_TRUE(RecipeVerification::verifyScalMemorySizes(recipeStaticInfoScal)) << "Failed to verifyHbmMemorySize";
    ASSERT_TRUE(RecipeVerification::verifyRecipe(pRecipe, nullptr)) << "Failed to verifyRecipe";

    auto prevValue = pRecipe->program_data_blobs_size;

    // Make one section in glbl-hbm bigger than max recipe size
    pRecipe->program_data_blobs_size = DeviceGaudi2scal::getHbmGlblMaxRecipeSize() + 1;

    status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2, basicRecipeInfo, recipeStaticInfoScal);
    ASSERT_EQ(status, synSuccess);

    // pRecipe
    ASSERT_FALSE(RecipeVerification::verifyScalMemorySizes(recipeStaticInfoScal)) << "should fail for a large recipe";

    // restore recipe
    pRecipe->program_data_blobs_size = prevValue;

    // Make one section in shared-hbm bigger than max recipe size
    pRecipe->arc_jobs[pRecipe->arc_jobs_nr - 1].static_ecb.cmds_size = DeviceGaudi2scal::getHbmSharedMaxRecipeSize() + 1;
    status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2, basicRecipeInfo, recipeStaticInfoScal);
    ASSERT_EQ(status, synSuccess);

    ASSERT_FALSE(RecipeVerification::verifyScalMemorySizes(recipeStaticInfoScal)) << "should fail for a large recipe";
}
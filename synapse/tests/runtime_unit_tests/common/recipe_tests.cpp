#include "habana_graph_mock.hpp"
#include "node_factory.h"
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"
#include "synapse_api.h"
#include "tensor_tests.hpp"

#include "runtime/common/habana_global_conf_runtime.h"

#include "recipe_allocator.h"

#include "hpp/synapse.hpp"
#include "hpp/syn_context.hpp"
#include "hpp/syn_recipe.hpp"

#include <gtest/gtest.h>

class UTRecipeTest : public ::testing::Test
{
public:
    static void getTopologyWorkspaceSizeGaudi2(const std::vector<uint64_t>& rWorkspaceSizes, uint64_t& rWorkspaceSize);

    static void check_new_apis(synDeviceType deviceType);
};

void UTRecipeTest::getTopologyWorkspaceSizeGaudi2(const std::vector<uint64_t>& rWorkspaceSizes,
                                                  uint64_t&                    rWorkspaceSize)
{
    RecipeManager         rMng;
    HabanaGraphMock       graph(synDeviceGaudi2, rWorkspaceSizes);
    const std::string     fileName("myRecipe.txt");
    InternalRecipeHandle* pRecipeHndl = nullptr;

    synStatus status =
        rMng.addRecipeHandleAndCompileGraph(&graph, false, nullptr, 0, fileName.c_str(), nullptr, pRecipeHndl);
    ASSERT_EQ(synSuccess, status);

    rWorkspaceSize = pRecipeHndl->deviceAgnosticRecipeHandle.m_workspaceSize;

    bool removeStatus = rMng.removeRecipeHandle(pRecipeHndl);
    ASSERT_EQ(true, removeStatus);
}

void UTRecipeTest::check_new_apis(synDeviceType deviceType)
{
    syn::Context context;

    synStatus status                    = synSuccess;
    unsigned sizes[SYN_MAX_TENSOR_DIM] = {16, 16, 16, 16};

    // Prepare some descriptors - having a minimal memcpy - just to check APIs
    std::vector<synTensor> inputs(1);
    std::vector<synTensor> outputs(1);

    synGraphHandle graphA;
    status = synGraphCreate(&graphA, deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create graph";
    const unsigned batch = 1;
    const unsigned wOFM  = 128;
    const unsigned nIFM  = 3;
    const unsigned wIFM  = wOFM;
    const unsigned hIFM  = nIFM;

    const std::vector<unsigned> inTensorDims  = {nIFM, wIFM, hIFM, batch};
    const std::vector<unsigned> outTensorDims = {nIFM, wIFM, hIFM, batch};
    // const char* in_layouts[] = {"A_0_2"};
    // const char* out_layouts[] = {"B_0_2"};
    inputs[0] =
        UTTensorTest::createInferenceTensor(inTensorDims, INPUT_TENSOR, syn_type_single, sizes, "A_0_2", graphA);
    outputs[0] =
        UTTensorTest::createInferenceTensor(outTensorDims, OUTPUT_TENSOR, syn_type_single, sizes, "B_0_2", graphA);
    if (deviceType != synDeviceGaudi)
    {
        ASSERT_EQ(synTensorSetExternal(outputs[0], true), synSuccess);
    }
    char nodeGuid[] = "memcpy";
    EXPECT_EQ(synNodeCreate(graphA,
                            inputs.data(),
                            outputs.data(),
                            inputs.size(),
                            outputs.size(),
                            nullptr,
                            0,
                            nodeGuid,
                            "",
                            nullptr,
                            nullptr),
              synSuccess);
    synRecipeHandle recipeHandle;

    const std::string recipeFileName("/tmp/recipe.txt");
    status = synGraphCompile(&recipeHandle, graphA, recipeFileName.c_str(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";
    auto     recipe = recipeHandle->basicRecipeHandle.recipe;
    uint32_t numOfTensors;
    ASSERT_EQ(synTensorRetrievePersistentAmount(recipeHandle, &numOfTensors), synSuccess)
        << "Failed to retrieve amount of tensors";
    ASSERT_EQ(recipe->persist_tensors_nr, numOfTensors);
    ASSERT_EQ(2, numOfTensors);
    char tensorsName[numOfTensors][ENQUEUE_TENSOR_NAME_MAX_SIZE];

    ASSERT_EQ(synTensorRetrieveNames(recipeHandle, tensorsName, numOfTensors), synSuccess)
        << "Failed to retrieve tensor names";
    static const size_t numOfAttributes(7);
    synRecipeAttribute  recipeAttr[numOfAttributes] = {RECIPE_ATTRIBUTE_WORKSPACE_SIZE,
                                                      RECIPE_ATTRIBUTE_NUM_PERSISTENT_TENSORS,
                                                      RECIPE_ATTRIBUTE_HOST_MEM_SIZE,
                                                      RECIPE_ATTRIBUTE_NUM_EXTERNAL_TENSORS,
                                                      RECIPE_ATTRIBUTE_PERSISTENT_TENSORS_SIZE,
                                                      RECIPE_ATTRIBUTE_CONST_SECTIONS_SIZE,
                                                      RECIPE_ATTRIBUTE_DEVICE_MEMORY_SIZE};
    uint64_t            recipeAttributeOutValues[numOfAttributes];
    ASSERT_EQ(synRecipeGetAttribute((recipeAttributeOutValues), recipeAttr, numOfAttributes, recipeHandle), synSuccess)
        << "Failed to retrieve recipe size";
    ASSERT_EQ(recipeHandle->deviceAgnosticRecipeHandle.m_workspaceSize, recipeAttributeOutValues[0]);
    ASSERT_EQ(recipe->persist_tensors_nr, recipeAttributeOutValues[1]);
    // recipeSize might change when code change - these are reasonable limits
    ASSERT_LT(recipeAttributeOutValues[2], 10000) << "Wrong recipe size";
    ASSERT_GT(recipeAttributeOutValues[2], 1000) << "Wrong recipe size";
    ASSERT_EQ(deviceType != synDeviceGaudi ? 1 : 0, recipeAttributeOutValues[3]);

    ASSERT_LT(recipeAttributeOutValues[4], 700) << "Wrong persistent tensors size";
    ASSERT_GT(recipeAttributeOutValues[4], 600) << "Wrong persistent tensors size";

    ASSERT_EQ(recipeAttributeOutValues[5], 0) << "Wrong const section size";

    ASSERT_LT(recipeAttributeOutValues[6], 1500) << "Wrong HBM memory size";
    ASSERT_GT(recipeAttributeOutValues[6], 500) << "Wrong HBM memory size";

    TensorMetadataInfoExt tensorMetadataInfo[numOfTensors];

    for (size_t i = 0; i < numOfTensors; i++)
    {
        ASSERT_EQ(strcmp(tensorsName[i], recipe->tensors[i].name), 0);
        std::string strTensorName(tensorsName[i]);
        tensorMetadataInfo[i].tensorName = strTensorName.c_str();
        status                           = synTensorRetrieveInfosByNameExt(recipeHandle, i + 1, tensorMetadataInfo);
        ASSERT_EQ(status, synSuccess) << "Failed to get tensors' info";
        ASSERT_EQ(tensorMetadataInfo[i].zp, recipe->tensors[i].zp) << "Failed to get tensors' zp";
        ASSERT_EQ(tensorMetadataInfo[i].scale, recipe->tensors[i].scale) << "Failed to get tensors' scale";
        ASSERT_EQ(tensorMetadataInfo[i].elementType, recipe->tensors[i].elementType)
            << "Failed to get tensors' elementType";
        ASSERT_EQ(tensorMetadataInfo[i].batchSize, recipe->tensors[i].batchSize) << "Failed to get tensors' batchSize";
        ASSERT_EQ(tensorMetadataInfo[i].dimensions, recipe->tensors[i].dimensions)
            << "Failed to get tensors' dimensions";
    }
    synGraphDestroy(graphA);
    UTTensorTest::destroySections();
}

TEST_F(UTRecipeTest, recipe_serialize_deserialize)
{
    // Initialization
    syn::Context context;

    // Disable recipe validation as this test does not suppose to check recipe correctness only serialization
    GCFG_RUNTIME_SKIP_RECIPE_VALIDATION.setValue(true);

    // create a recipe
    recipe_t recipe = {0};

    recipe.version_major = 1;
    recipe.version_minor = 1;

    recipe.blobs_nr = 2;
    blob_t blobs[2];
    recipe.blobs                  = blobs;
    recipe.blobs[0].blob_type_all = blob_t::EXE;
    recipe.blobs[0].size          = 4;
    uint8_t data0[4]              = {8, 8, 9, 9};
    recipe.blobs[0].data          = data0;
    recipe.blobs[1].blob_type_all = blob_t::PATCHING;
    recipe.blobs[1].size          = 10;
    uint8_t data1[10]             = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    recipe.blobs[1].data          = data1;

    recipe.execution_blobs_buffer_size = 4;
    recipe.execution_blobs_buffer      = (uint64_t*)data0;
    recipe.patching_blobs_buffer_size  = 10;
    recipe.patching_blobs_buffer       = (uint64_t*)data1;

    recipe.programs_nr = 1;
    program_t programs[1];
    recipe.programs                   = programs;
    recipe.programs[0].program_length = 2;
    uint64_t blob_indices[2]          = {1, 2};
    recipe.programs[0].blob_indices   = blob_indices;

    recipe.execute_jobs_nr = 1;
    job_t jobs[1];
    recipe.execute_jobs                = jobs;
    recipe.execute_jobs[0].engine_id   = 4;
    recipe.execute_jobs[0].program_idx = 1;

    const uint64_t        multiViewsAmount = 1;
    persist_tensor_info_t multiViewsTensors[multiViewsAmount];
    multiViewsTensors[0].name                   = "test_multi_view";
    multiViewsTensors[0].section_idx            = 1;
    multiViewsTensors[0].offset_in_section      = 16;
    multiViewsTensors[0].layout                 = nullptr;
    multiViewsTensors[0].permutation[0]         = std::numeric_limits<uint8_t>::max();
    multiViewsTensors[0].permutation[1]         = 88;
    multiViewsTensors[0].multi_views_indices    = nullptr;
    multiViewsTensors[0].multi_views_indices_nr = 0;
    recipe.permute_tensors_views_nr             = multiViewsAmount;
    recipe.permute_tensors_views                = multiViewsTensors;

    uint32_t tensorsMultiViewsIndices[1] = {0};
    recipe.persist_tensors_nr = 1;
    persist_tensor_info_t tensors[1];
    recipe.tensors                           = tensors;
    recipe.tensors[0].name                   = "test_tensor";
    recipe.tensors[0].section_idx            = 1;
    recipe.tensors[0].offset_in_section      = 16;
    recipe.tensors[0].layout                 = nullptr;
    recipe.tensors[0].permutation[0]         = std::numeric_limits<uint8_t>::max();
    recipe.tensors[0].permutation[1]         = 88;
    recipe.tensors[0].multi_views_indices    = tensorsMultiViewsIndices;
    recipe.tensors[0].multi_views_indices_nr = 1;

    recipe.program_data_blobs_nr = 1;
    program_data_blob_t data_blobs[1];
    recipe.program_data_blobs                      = data_blobs;
    recipe.program_data_blobs[0].size              = 3;
    char data[3]                                   = {5, 5, 5};
    recipe.program_data_blobs[0].data              = data;
    recipe.program_data_blobs[0].offset_in_section = 64;
    recipe.program_data_blobs[0].section_idx       = 1;

    recipe.program_data_blobs_size   = 3;
    recipe.program_data_blobs_buffer = data;

    recipe.patch_points_nr = 1;
    patch_point_t patch_points[1];
    recipe.patch_points                                         = patch_points;
    recipe.patch_points[0].type                                 = patch_point_t::SIMPLE_DW_LOW_MEM_PATCH_POINT;
    recipe.patch_points[0].blob_idx                             = 1;
    recipe.patch_points[0].dw_offset_in_blob                    = 0;
    recipe.patch_points[0].memory_patch_point.effective_address = 100;
    recipe.patch_points[0].memory_patch_point.section_idx       = 1;

    recipe.sections_nr = 1;

    recipe.workspace_nr = 3;
    uint64_t workspaceSizes[3];
    recipe.workspace_sizes    = workspaceSizes;
    recipe.workspace_sizes[0] = 128;

    RecipeAllocator recipeAlloc;
    recipe.dynamic_blobs_buffer_size = 128;
    recipe.dynamic_blobs_buffer      = (uint32_t*)recipeAlloc.allocate(128);

    recipe.arc_jobs_nr = 2;
    recipe.arc_jobs    = (arc_job_t*)recipeAlloc.allocate(2 * sizeof(arc_job_t));

    recipe.arc_jobs[0].logical_engine_id     = Recipe::EngineType::TPC;
    recipe.arc_jobs[0].engines_filter        = 2;
    recipe.arc_jobs[0].static_ecb.cmds_size  = 10;
    uint8_t staticData0[10]                  = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    recipe.arc_jobs[0].static_ecb.cmds       = staticData0;
    uint8_t dynamicData0[16]                 = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 11, 12, 13, 16, 17, 20};
    recipe.arc_jobs[0].static_ecb.cmds       = staticData0;
    recipe.arc_jobs[0].dynamic_ecb.cmds_size = 16;
    recipe.arc_jobs[0].dynamic_ecb.cmds      = dynamicData0;

    recipe.arc_jobs[1].logical_engine_id     = Recipe::EngineType::MME;
    recipe.arc_jobs[1].engines_filter        = 3;
    recipe.arc_jobs[1].static_ecb.cmds_size  = 4;  // 128;
    uint8_t staticData1[4]                   = {8, 8, 9, 9};
    recipe.arc_jobs[1].static_ecb.cmds       = staticData1;
    recipe.arc_jobs[1].dynamic_ecb.cmds_size = 0;
    recipe.arc_jobs[1].dynamic_ecb.cmds      = nullptr;

    recipe.recipe_conf_nr = 1;
    recipe.recipe_conf_params =
        reinterpret_cast<gc_conf_t*>(recipeAlloc.allocate(recipe.recipe_conf_nr * sizeof(gc_conf_t)));
    recipe.recipe_conf_params[0].conf_id    = gc_conf_t::DEVICE_TYPE;
    recipe.recipe_conf_params[0].conf_value = synDeviceGaudi2;

    recipe.permute_tensors_views    = nullptr;
    recipe.permute_tensors_views_nr = 0;

    // serialize recipe to file
    InternalRecipeHandle handle {};
    handle.basicRecipeHandle.recipe = &recipe;

    synRecipeHandle recipeHandle = &handle;

    std::string recipeFileName("/tmp/recipe.txt");

    synStatus status = synRecipeSerialize(recipeHandle, recipeFileName.c_str());
    ASSERT_EQ(status, synSuccess) << "Failed to Serialize recipe";

    // deserialize recipe from file
    synRecipeHandle handleAfterDeSerialize;
    status = synRecipeDeSerialize(&handleAfterDeSerialize, recipeFileName.c_str());
    ASSERT_EQ(status, synSuccess) << "Failed to DeSerialize recipe";

    recipe_t* pRecipe                 = &recipe;
    recipe_t* pRecipeAfterDeSerialize = handleAfterDeSerialize->basicRecipeHandle.recipe;

    // check equality
    ASSERT_EQ(pRecipe->version_major, pRecipeAfterDeSerialize->version_major);
    ASSERT_EQ(pRecipe->version_minor, pRecipeAfterDeSerialize->version_minor);

    ASSERT_EQ(pRecipe->blobs_nr, pRecipeAfterDeSerialize->blobs_nr);
    for (uint64_t blob_index = 0; blob_index < pRecipe->blobs_nr; blob_index++)
    {
        blob_t* pCurrBlob                 = &pRecipe->blobs[blob_index];
        blob_t* pCurrBlobAfterDeSerialize = &pRecipeAfterDeSerialize->blobs[blob_index];

        ASSERT_EQ(pCurrBlob->blob_type.requires_patching, pCurrBlobAfterDeSerialize->blob_type.requires_patching);

        ASSERT_EQ(0, memcmp(&pCurrBlob->size, &pCurrBlobAfterDeSerialize->size, sizeof(uint32_t)));

        ASSERT_EQ(0,
                  memcmp((uint8_t*)pCurrBlob->data,
                         (uint8_t*)pCurrBlobAfterDeSerialize->data,
                         sizeof(uint8_t) * pCurrBlob->size));
    }

    ASSERT_EQ(pRecipe->programs_nr, pRecipeAfterDeSerialize->programs_nr);
    for (uint64_t program_index = 0; program_index < pRecipe->programs_nr; program_index++)
    {
        program_t* pCurrProgram                 = &pRecipe->programs[program_index];
        program_t* pCurrProgramAfterDeSerialize = &pRecipeAfterDeSerialize->programs[program_index];

        ASSERT_EQ(pCurrProgram->program_length, pCurrProgramAfterDeSerialize->program_length);

        ASSERT_EQ(0,
                  memcmp(pCurrProgram->blob_indices,
                         pCurrProgramAfterDeSerialize->blob_indices,
                         sizeof(uint64_t) * pCurrProgram->program_length));
    }
    ASSERT_EQ(pRecipe->activate_jobs_nr, pRecipeAfterDeSerialize->activate_jobs_nr);
    for (uint64_t job_index = 0; job_index < pRecipe->activate_jobs_nr; job_index++)
    {
        ASSERT_EQ(0,
                  memcmp(&pRecipe->activate_jobs[job_index],
                         &pRecipeAfterDeSerialize->activate_jobs[job_index],
                         sizeof(uint32_t) * 2));
    }

    ASSERT_EQ(pRecipe->execute_jobs_nr, pRecipeAfterDeSerialize->execute_jobs_nr);
    for (uint64_t job_index = 0; job_index < pRecipe->execute_jobs_nr; job_index++)
    {
        ASSERT_EQ(0,
                  memcmp(&pRecipe->execute_jobs[job_index],
                         &pRecipeAfterDeSerialize->execute_jobs[job_index],
                         sizeof(uint64_t)));
    }

    ASSERT_EQ(pRecipe->permute_tensors_views_nr, pRecipeAfterDeSerialize->permute_tensors_views_nr);
    for (uint64_t view_index = 0; view_index < pRecipe->permute_tensors_views_nr; view_index++)
    {
        persist_tensor_info_t* pCurrTensor = &pRecipe->permute_tensors_views[view_index];
        persist_tensor_info_t* pCurrTensorAfterDeSerialize =
            &pRecipeAfterDeSerialize->permute_tensors_views[view_index];

        ASSERT_EQ(0, strcmp(pCurrTensor->name, pCurrTensorAfterDeSerialize->name));
        ASSERT_EQ(pCurrTensor->layout, pCurrTensorAfterDeSerialize->layout);

        ASSERT_EQ(0, memcmp(&pCurrTensor->section_idx, &pCurrTensorAfterDeSerialize->section_idx, sizeof(uint32_t)));
        ASSERT_EQ(
            0,
            memcmp(&pCurrTensor->offset_in_section, &pCurrTensorAfterDeSerialize->offset_in_section, sizeof(uint64_t)));
        ASSERT_EQ(0,
                  memcmp(&pCurrTensor->permutation,
                         &pCurrTensorAfterDeSerialize->permutation,
                         sizeof(pCurrTensor->permutation)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->size, &pCurrTensorAfterDeSerialize->size, sizeof(uint64_t)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->elementType, &pCurrTensorAfterDeSerialize->elementType, sizeof(uint32_t)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->zp, &pCurrTensorAfterDeSerialize->zp, sizeof(double)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->scale, &pCurrTensorAfterDeSerialize->scale, sizeof(double)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->dimensions, &pCurrTensorAfterDeSerialize->dimensions, sizeof(uint32_t)));
        for (size_t idx = 0; idx < SYN_MAX_TENSOR_DIM; idx++)
        {
            ASSERT_EQ(0,
                      memcmp(&pCurrTensor->dimensionsSize[idx],
                             &pCurrTensorAfterDeSerialize->dimensionsSize[idx],
                             sizeof(uint32_t)));
        }
        ASSERT_EQ(pCurrTensorAfterDeSerialize->isInput, pCurrTensor->isInput);
        ASSERT_EQ(0, memcmp(&pCurrTensor->batchSize, &pCurrTensorAfterDeSerialize->batchSize, sizeof(uint32_t)));

        ASSERT_EQ(pCurrTensor->multi_views_indices_nr, pCurrTensorAfterDeSerialize->multi_views_indices_nr);
        ASSERT_EQ(pCurrTensor->multi_views_indices_nr, 0);
    }

    ASSERT_EQ(pRecipe->persist_tensors_nr, pRecipeAfterDeSerialize->persist_tensors_nr);
    for (uint64_t tensor_index = 0; tensor_index < pRecipe->persist_tensors_nr; tensor_index++)
    {
        persist_tensor_info_t* pCurrTensor                 = &pRecipe->tensors[tensor_index];
        persist_tensor_info_t* pCurrTensorAfterDeSerialize = &pRecipeAfterDeSerialize->tensors[tensor_index];

        ASSERT_EQ(0, strcmp(pCurrTensor->name, pCurrTensorAfterDeSerialize->name));
        ASSERT_EQ(pCurrTensor->layout, pCurrTensorAfterDeSerialize->layout);

        ASSERT_EQ(0, memcmp(&pCurrTensor->section_idx, &pCurrTensorAfterDeSerialize->section_idx, sizeof(uint32_t)));
        ASSERT_EQ(
            0,
            memcmp(&pCurrTensor->offset_in_section, &pCurrTensorAfterDeSerialize->offset_in_section, sizeof(uint64_t)));
        ASSERT_EQ(0,
                  memcmp(&pCurrTensor->permutation,
                         &pCurrTensorAfterDeSerialize->permutation,
                         sizeof(pCurrTensor->permutation)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->size, &pCurrTensorAfterDeSerialize->size, sizeof(uint64_t)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->elementType, &pCurrTensorAfterDeSerialize->elementType, sizeof(uint32_t)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->zp, &pCurrTensorAfterDeSerialize->zp, sizeof(double)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->scale, &pCurrTensorAfterDeSerialize->scale, sizeof(double)));
        ASSERT_EQ(0, memcmp(&pCurrTensor->dimensions, &pCurrTensorAfterDeSerialize->dimensions, sizeof(uint32_t)));
        for (size_t idx = 0; idx < SYN_MAX_TENSOR_DIM; idx++)
        {
            ASSERT_EQ(0,
                      memcmp(&pCurrTensor->dimensionsSize[idx],
                             &pCurrTensorAfterDeSerialize->dimensionsSize[idx],
                             sizeof(uint32_t)));
        }
        ASSERT_EQ(pCurrTensorAfterDeSerialize->isInput, pCurrTensor->isInput);
        ASSERT_EQ(0, memcmp(&pCurrTensor->batchSize, &pCurrTensorAfterDeSerialize->batchSize, sizeof(uint32_t)));

        ASSERT_EQ(pCurrTensor->multi_views_indices_nr, pCurrTensorAfterDeSerialize->multi_views_indices_nr);
        ASSERT_EQ(0,
                  memcmp(pCurrTensor->multi_views_indices,
                         pCurrTensorAfterDeSerialize->multi_views_indices,
                         pCurrTensor->multi_views_indices_nr * sizeof(uint32_t)));
    }

    ASSERT_EQ(pRecipe->program_data_blobs_nr, pRecipeAfterDeSerialize->program_data_blobs_nr);
    for (uint64_t blob_index = 0; blob_index < pRecipe->program_data_blobs_nr; blob_index++)
    {
        program_data_blob_t* pCurrBlob                 = &pRecipe->program_data_blobs[blob_index];
        program_data_blob_t* pCurrBlobAfterDeSerialize = &pRecipeAfterDeSerialize->program_data_blobs[blob_index];

        ASSERT_EQ(pCurrBlob->size, pCurrBlobAfterDeSerialize->size);

        ASSERT_EQ(0, memcmp(pCurrBlob->data, pCurrBlobAfterDeSerialize->data, sizeof(char) * pCurrBlob->size));

        ASSERT_EQ(
            0,
            memcmp(&pCurrBlob->offset_in_section, &pCurrBlobAfterDeSerialize->offset_in_section, sizeof(uint64_t)));

        ASSERT_EQ(0, memcmp(&pCurrBlob->section_idx, &pCurrBlobAfterDeSerialize->section_idx, sizeof(uint16_t)));
    }

    ASSERT_EQ(pRecipe->patch_points_nr, pRecipeAfterDeSerialize->patch_points_nr);
    for (uint64_t patch_index = 0; patch_index < pRecipe->patch_points_nr; patch_index++)
    {
        patch_point_t* pCurrPatch                 = &pRecipe->patch_points[patch_index];
        patch_point_t* pCurrPatchAfterDeSerialize = &pRecipeAfterDeSerialize->patch_points[patch_index];

        ASSERT_EQ(pCurrPatch->type, pCurrPatchAfterDeSerialize->type);

        ASSERT_EQ(0, memcmp(&pCurrPatch->blob_idx, &pCurrPatchAfterDeSerialize->blob_idx, sizeof(uint64_t)));

        ASSERT_EQ(
            0,
            memcmp(&pCurrPatch->dw_offset_in_blob, &pCurrPatchAfterDeSerialize->dw_offset_in_blob, sizeof(uint32_t)));
        ASSERT_EQ(0,
                  memcmp(&pCurrPatch->memory_patch_point.effective_address,
                         &pCurrPatchAfterDeSerialize->memory_patch_point.effective_address,
                         sizeof(uint64_t)));
        ASSERT_EQ(0,
                  memcmp(&pCurrPatch->memory_patch_point.section_idx,
                         &pCurrPatchAfterDeSerialize->memory_patch_point.section_idx,
                         sizeof(uint16_t)));

        ASSERT_EQ(0,
                  memcmp(&pCurrPatch->node_exe_index, &pCurrPatchAfterDeSerialize->node_exe_index, sizeof(uint32_t)));
    }

    ASSERT_EQ(pRecipe->sections_nr, pRecipeAfterDeSerialize->sections_nr);

    ASSERT_EQ(pRecipe->workspace_nr, pRecipeAfterDeSerialize->workspace_nr);

    ASSERT_EQ(0,
              memcmp(pRecipe->workspace_sizes,
                     pRecipeAfterDeSerialize->workspace_sizes,
                     sizeof(uint64_t) * pRecipe->workspace_nr));

    // arcs related feilds
    ASSERT_EQ(pRecipe->dynamic_blobs_buffer_size, pRecipeAfterDeSerialize->dynamic_blobs_buffer_size);

    ASSERT_EQ(0, memcmp(pRecipe->dynamic_blobs_buffer, pRecipeAfterDeSerialize->dynamic_blobs_buffer, 128));

    ASSERT_EQ(pRecipe->arc_jobs_nr, pRecipeAfterDeSerialize->arc_jobs_nr);

    for (uint64_t arc_job_index = 0; arc_job_index < recipe.arc_jobs_nr; arc_job_index++)
    {
        arc_job_t* pJobBefore = &pRecipe->arc_jobs[arc_job_index];
        arc_job_t* pJobAfter  = &pRecipeAfterDeSerialize->arc_jobs[arc_job_index];

        ASSERT_EQ(pJobBefore->logical_engine_id, pJobAfter->logical_engine_id);
        ASSERT_EQ(pJobBefore->engines_filter, pJobAfter->engines_filter);

        ASSERT_EQ(pJobBefore->static_ecb.cmds_size, pJobAfter->static_ecb.cmds_size);
        if (pJobBefore->static_ecb.cmds_size)
        {
            ASSERT_EQ(0,
                      memcmp((uint8_t*)pJobBefore->static_ecb.cmds,
                             (uint8_t*)pJobAfter->static_ecb.cmds,
                             sizeof(uint8_t) * pJobBefore->static_ecb.cmds_size));
        }

        ASSERT_EQ(pJobBefore->dynamic_ecb.cmds_size, pJobAfter->dynamic_ecb.cmds_size);
        if (pJobBefore->dynamic_ecb.cmds_size)
        {
            ASSERT_EQ(0,
                      memcmp((uint8_t*)pJobBefore->dynamic_ecb.cmds,
                             (uint8_t*)pJobAfter->dynamic_ecb.cmds,
                             sizeof(uint8_t) * pJobBefore->dynamic_ecb.cmds_size));
        }
    }

    ASSERT_EQ(pRecipe->recipe_conf_nr, pRecipeAfterDeSerialize->recipe_conf_nr);
    for (uint64_t recipe_conf_index = 0; recipe_conf_index < recipe.recipe_conf_nr; recipe_conf_index++)
    {
        ASSERT_EQ(recipe.recipe_conf_params[recipe_conf_index].conf_id,
                  pRecipeAfterDeSerialize->recipe_conf_params[recipe_conf_index].conf_id);
        ASSERT_EQ(recipe.recipe_conf_params[recipe_conf_index].conf_value,
                  pRecipeAfterDeSerialize->recipe_conf_params[recipe_conf_index].conf_value);
    }

    // cleanup
    status = synRecipeDestroy(handleAfterDeSerialize);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy recipe handle after deserialize";

}

TEST_F(UTRecipeTest, check_new_apis_gaudi)
{
    check_new_apis(synDeviceGaudi);
}

TEST_F(UTRecipeTest, check_new_apis_gaudi2)
{
    check_new_apis(synDeviceGaudi2);
}

TEST_F(UTRecipeTest, check_tensor_layout)
{
    syn::Context context;

    synStatus status = synSuccess;
    // Prepare some descriptors
    std::vector<synTensor> inputs(2);
    std::vector<synTensor> outputs(1);

    synGraphHandle graphA;
    status = synGraphCreate(&graphA, synDeviceGaudi);
    ASSERT_EQ(status, synSuccess) << "Failed to create Gaudi graph";

    const unsigned H = 5;
    const unsigned W = 5;
    const unsigned C = 2;
    const unsigned N = 1;
    const unsigned K = 2;
    const unsigned R = 2;
    const unsigned S = 2;

    const unsigned weightsStride  = 1;
    const unsigned weightsPadding = 1;
    const unsigned outW           = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const unsigned outH           = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const unsigned outC           = K;

    synConvolutionParams params;
    params.dH   = weightsStride;
    params.dW   = weightsStride;
    params.kH   = S;
    params.kW   = R;
    params.padT = weightsPadding;
    params.padB = weightsPadding;
    params.padL = weightsPadding;

    params.padR = weightsPadding;
    params.dilH = 1;
    params.dilW = 1;

    const std::vector<unsigned> inTensorDims              = {C, W, H, N};
    const std::vector<unsigned> weightTensorDims          = {K, C, S, R};
    const std::vector<unsigned> outTensorDims             = {outC, outW, outH, N};
    unsigned                    sizes[SYN_MAX_TENSOR_DIM] = {16, 16, 16, 16};

    synDataType dataType         = syn_type_single;
    unsigned    numberOfElements = multiplyElements(std::begin(weightTensorDims), std::end(weightTensorDims));
    // 4 bits quantized weight tensor will be packed later with two 4bits elements in a byte
    unsigned dataSize = safeBitsToByte(numberOfElements * dataTypeToSizeInBits(dataType));
    char*    data     = new char[dataSize];

    inputs[0] = UTTensorTest::createInferenceTensor(inTensorDims, INPUT_TENSOR, dataType, sizes, "ifm", graphA);
    inputs[1] =
        UTTensorTest::createWeightTensor(graphA, INPUT_TENSOR, weightTensorDims, dataType, "weights", data, dataSize);
    outputs[0] = UTTensorTest::createInferenceTensor(outTensorDims, OUTPUT_TENSOR, dataType, sizes, "ofm", graphA);

    const char* conv2D_in_layouts[]  = {"CWHN", "KCSR", "", "CWHN", ""};
    const char* conv2D_out_layouts[] = {"CWHN"};

    EXPECT_EQ(synNodeCreate(graphA,
                            inputs.data(),
                            outputs.data(),
                            inputs.size(),
                            outputs.size(),
                            &params,
                            sizeof(synConvolutionParams),
                            "spatial_convolution",
                            "convNode",
                            conv2D_in_layouts,
                            conv2D_out_layouts),
              synSuccess);

    synRecipeHandle recipeHandle;

    const std::string recipeFileName("/tmp/recipe.txt");
    status = synGraphCompile(&recipeHandle, graphA, recipeFileName.c_str(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";
    auto     recipe = recipeHandle->basicRecipeHandle.recipe;
    uint32_t numOfTensors;
    ASSERT_EQ(synTensorRetrievePersistentAmount(recipeHandle, &numOfTensors), synSuccess)
        << "Failed to retrieve amount of tensors";
    ASSERT_EQ(recipe->persist_tensors_nr, numOfTensors);
    char tensorsName[numOfTensors][ENQUEUE_TENSOR_NAME_MAX_SIZE];

    ASSERT_EQ(synTensorRetrieveNames(recipeHandle, tensorsName, numOfTensors), synSuccess)
        << "Failed to retrieve tensor names";

    TensorMetadataInfo tensorMetadataInfo[numOfTensors];
    for (size_t i = 0; i < numOfTensors; i++)
    {
        ASSERT_EQ(strcmp(tensorsName[i], recipe->tensors[i].name), 0);
        std::string strTensorName(tensorsName[i]);
        char*       tensorNameChar = new char[strTensorName.size() + 1];
        strTensorName.copy(tensorNameChar, strTensorName.size());
        tensorNameChar[strTensorName.size()] = '\0';
        tensorMetadataInfo[i].tensorName     = tensorNameChar;
    }

    for (size_t i = 0; i < numOfTensors; i++)
    {
        status = synTensorRetrieveInfosByName(recipeHandle, numOfTensors, tensorMetadataInfo);
        ASSERT_EQ(status, synSuccess) << "Failed to get tensors' info";
        ASSERT_EQ(tensorMetadataInfo[i].zp, recipe->tensors[i].zp) << "Failed to get tensors' zp";
        ASSERT_EQ(tensorMetadataInfo[i].scale, recipe->tensors[i].scale) << "Failed to get tensors' scale";
        ASSERT_EQ(tensorMetadataInfo[i].elementType, recipe->tensors[i].elementType)
            << "Failed to get tensors' elementType";
        ASSERT_EQ(tensorMetadataInfo[i].batchSize, recipe->tensors[i].batchSize) << "Failed to get tensors' batchSize";
        ASSERT_EQ(tensorMetadataInfo[i].dimensions, recipe->tensors[i].dimensions)
            << "Failed to get tensors' dimensions";
        if (recipe->tensors[i].layout == nullptr)
        {
            ASSERT_EQ(std::string(tensorMetadataInfo[i].layout), std::string("NotAvailable"))
                << "Failed to get tensors' layouts";
        }
        else
        {
            ASSERT_EQ(std::string(tensorMetadataInfo[i].layout), std::string(recipe->tensors[i].layout))
                << "Failed to get tensors' layouts";
        }
    }

    for (size_t i = 0; i < numOfTensors; i++)
    {
        delete[] tensorMetadataInfo[i].tensorName;
    }
    delete[] data;
    synGraphDestroy(graphA);
    UTTensorTest::destroySections();
}

class UTRecipeLaunchTensorsInfoTest
: public UTRecipeTest
, public testing::WithParamInterface<unsigned>
{
};

INSTANTIATE_TEST_SUITE_P(, UTRecipeLaunchTensorsInfoTest, ::testing::Values(1, 7));

TEST_P(UTRecipeLaunchTensorsInfoTest, get_launch_tensors_info)
{
    syn::Context ctx;

    size_t H     = 3;
    size_t W     = 2;
    size_t C     = 2;
    size_t MAX_B = 7;
    size_t MIN_B = GetParam();

    const std::vector<TSize> maxDataSize = {C, W, H, MAX_B};
    const std::vector<TSize> minDataSize  = {C, W, H, MIN_B};
    const std::vector<TSize> maxShapeSize = {1, C * W, H, MAX_B};
    const std::vector<TSize> minShapeSize = {1, C * W, H, MIN_B};

    syn::Graph   graph   = ctx.createGraph(synDeviceGaudi);
    syn::Section section = graph.createSection();
    section.setPersistent(true);

    std::string inputTensorName = "input";
    syn::Tensor inputTensor     = graph.createTensor(DATA_TENSOR_DYNAMIC, inputTensorName);
    inputTensor.setGeometry(maxDataSize, synGeometryMaxSizes);
    inputTensor.setGeometry(minDataSize, synGeometryMinSizes);
    inputTensor.setDeviceLayout({}, syn_type_single);
    inputTensor.assignToSection(section, 0);

    std::string outputTensorName = "output";
    syn::Tensor outputTensor     = graph.createTensor(DATA_TENSOR_DYNAMIC, outputTensorName);
    outputTensor.setGeometry(maxShapeSize, synGeometryMaxSizes);
    outputTensor.setGeometry(minShapeSize, synGeometryMinSizes);
    outputTensor.setDeviceLayout({}, syn_type_single);
    outputTensor.assignToSection(section, inputTensor.getSizeInBytes());

    std::string shapeTensorName = "shape";
    syn::Tensor shapeTensor     = graph.createTensor(OUTPUT_DESCRIBING_SHAPE_TENSOR, shapeTensorName);
    shapeTensor.setGeometry(maxShapeSize, synGeometryMaxSizes);
    shapeTensor.setGeometry(minShapeSize, synGeometryMinSizes);
    shapeTensor.setDeviceLayout({}, syn_type_int32);
    shapeTensor.assignToSection(section, 0);

    syn::Node node = graph.createNode({inputTensor, shapeTensor},
                                      {outputTensor},
                                      {},
                                      NodeFactory::reshapeNodeTypeName,
                                      "reshape",
                                      {},
                                      {});

    syn::Recipe recipe = graph.compile("get_launch_tensors_info");

    uint32_t  numOfTensors = 0;
    synStatus sts          = synTensorRetrieveLaunchAmount(recipe.handle(), &numOfTensors);
    ASSERT_EQ(synSuccess, sts);

    ASSERT_EQ(3, numOfTensors);

    uint64_t ids[numOfTensors];
    sts = synTensorRetrieveLaunchIds(recipe.handle(), ids, numOfTensors);
    ASSERT_EQ(synSuccess, sts);

    synRetrievedLaunchTensorInfo infos[numOfTensors];
    for (size_t i = 0; i < numOfTensors; ++i)
    {
        infos[i].tensorId = ids[i];
    }

    sts = synTensorRetrieveLaunchInfoById(recipe.handle(), numOfTensors, infos);
    ASSERT_EQ(synSuccess, sts);

    synRetrievedLaunchTensorInfo invalidInfo;
    invalidInfo.tensorId = numOfTensors + 1;
    sts                  = synTensorRetrieveLaunchInfoById(recipe.handle(), 1, &invalidInfo);
    ASSERT_EQ(synSuccess, sts);
    ASSERT_EQ(TENSOR_TYPE_INVALID, invalidInfo.tensorType);

    std::map<std::string, synRetrievedLaunchTensorInfo> name2Info;
    for (size_t i = 0; i < numOfTensors; ++i)
    {
        name2Info[infos[i].tensorName] = infos[i];
    }
    ASSERT_EQ(1, name2Info.count("input"));
    ASSERT_EQ(1, name2Info.count("output"));
    ASSERT_EQ(1, name2Info.count("shape"));

    auto inputInfo  = name2Info.at("input");
    auto outputInfo = name2Info.at("output");
    auto shapeInfo  = name2Info.at("shape");

    ASSERT_EQ(true, inputInfo.isInput);
    ASSERT_EQ(DATA_TENSOR_DYNAMIC, inputInfo.tensorType);
    ASSERT_EQ(syn_type_single, inputInfo.tensorDataType);
    ASSERT_EQ(0, inputInfo.tensorOffsetInSection);
    ASSERT_EQ(maxDataSize.size(), inputInfo.tensorDims);

    ASSERT_EQ(false, outputInfo.isInput);
    ASSERT_EQ(DATA_TENSOR_DYNAMIC, outputInfo.tensorType);
    ASSERT_EQ(syn_type_single, outputInfo.tensorDataType);
    ASSERT_EQ(inputTensor.getSizeInBytes(), outputInfo.tensorOffsetInSection);
    ASSERT_EQ(maxShapeSize.size(), outputInfo.tensorDims);

    ASSERT_EQ(true, shapeInfo.isInput);
    ASSERT_EQ(OUTPUT_DESCRIBING_SHAPE_TENSOR, shapeInfo.tensorType);
    ASSERT_EQ(syn_type_int32, shapeInfo.tensorDataType);
    ASSERT_EQ(0, shapeInfo.tensorOffsetInSection);
    ASSERT_EQ(maxShapeSize.size(), shapeInfo.tensorDims);
}

TEST_F(UTRecipeTest, serialize_recipe_name)
{
    syn::Context ctx;
    syn::Graph   graph          = ctx.createGraph(synDeviceGaudi);
    std::string  recipeName     = "serialize_recipe_name";
    std::string  recipeFileName = recipeName + ".recipe";
    syn::Recipe  recipe         = graph.compile(recipeName);
    recipe.serialize(recipeFileName);

    syn::Recipe loadedRecipe(recipeFileName);
    std::string loadedName = loadedRecipe.handle()->basicRecipeHandle.recipe->name;
    EXPECT_EQ(recipeName, loadedName);
}

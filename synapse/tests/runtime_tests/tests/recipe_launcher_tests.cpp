#include "define_synapse_common.hpp"
#include "defs.h"
#include "habana_global_conf.h"
#include "infra/habana_global_conf_common.h"
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include "runtime/scal/common/recipe_static_processor_scal.hpp"
#include "scal_internal/pkt_macros.hpp"
#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_dummy_recipe.hpp"

// specs
struct gaudi2firmware
{
#include "gaudi2_arc_eng_packets.h"  // XXX_ECB_LIST_BUFF_SIZE
};

struct gaudi3firmware
{
#include "gaudi3/gaudi3_arc_eng_packets.h"  // XXX_ECB_LIST_BUFF_SIZE
};

class RecipeLauncherTests : public SynBaseTest
{
public:
    RecipeLauncherTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }

    static void
    buildDummyRecipe(recipe_t* recipe, synDeviceType deviceType, size_t nopCommandSize, uint8_t* pNopCommandAddr);
    static void destroyDummyRecipe(recipe_t* recipe);
    static void createBuff(uint64_t** buff, uint64_t sizeGiven, uint64_t val);
    static void prepareTensorInfo(synRecipeHandle recipe, synLaunchTensorInfo* tensorInfo, uint32_t totalNumOfTensors);
};

REGISTER_SUITE(RecipeLauncherTests, ALL_TEST_PACKAGES);

void RecipeLauncherTests::createBuff(uint64_t** buff, uint64_t sizeGiven, uint64_t val)
{
    uint32_t size = sizeGiven / sizeof(uint64_t);

    *buff = new uint64_t[size];

    for (int i = 0; i < size; i++)
    {
        (*buff)[i] = (val << 32) + i + 1;
    }
}

void RecipeLauncherTests::prepareTensorInfo(synRecipeHandle      recipe,
                                            synLaunchTensorInfo* tensorInfo,
                                            uint32_t             totalNumOfTensors)
{
    const char* tensorNames[totalNumOfTensors] = {};
    uint64_t    tensorIds[totalNumOfTensors];
    uint32_t    i = 0;

    for (i = 0; i < totalNumOfTensors; ++i)
    {
        tensorNames[i] = tensorInfo[i].tensorName;
    }
    ASSERT_EQ(synTensorRetrieveIds(recipe, tensorNames, tensorIds, totalNumOfTensors), synSuccess);
    for (i = 0; i < totalNumOfTensors; i++)
    {
        tensorInfo[i].tensorId = tensorIds[i];
    }
}

void RecipeLauncherTests::destroyDummyRecipe(recipe_t* recipe)
{
    delete[] recipe->execution_blobs_buffer;
    delete[] recipe->patching_blobs_buffer;
    delete[] recipe->dynamic_blobs_buffer;
    delete[] recipe->program_data_blobs_buffer;
    delete[] recipe->arc_jobs;
    delete[] recipe->patch_points;
    delete[] recipe->blobs;
    delete[] recipe->tensors;
    delete[] recipe->workspace_sizes;
    delete[] recipe->recipe_conf_params;
}

void RecipeLauncherTests::buildDummyRecipe(recipe_t*     recipe,
                                           synDeviceType deviceType,
                                           size_t        nopCommandSize,
                                           uint8_t*      pNopCommandAddr)
{
    memset(recipe, 0, sizeof(*recipe));

    uint64_t execBlobsSize   = 0x1000;
    uint64_t patchingSize    = 0x2000;
    uint64_t dynamicSize     = 0x3000;
    uint64_t programDataSize = 0x2800;

    recipe->execution_blobs_buffer_size = execBlobsSize;
    createBuff(&(recipe->execution_blobs_buffer), execBlobsSize, 1);

    recipe->patching_blobs_buffer_size = patchingSize;
    createBuff(&(recipe->patching_blobs_buffer), patchingSize, 2);

    recipe->dynamic_blobs_buffer_size = dynamicSize;
    createBuff((uint64_t**)&(recipe->dynamic_blobs_buffer), dynamicSize, 3);

    recipe->program_data_blobs_size = programDataSize;
    createBuff((uint64_t**)&(recipe->program_data_blobs_buffer), programDataSize, 4);

    recipe->arc_jobs_nr = Recipe::EngineType::ROT;  // rotator is not included at the moment
    if (deviceType == synDeviceGaudi3)
    {
        recipe->arc_jobs_nr = Recipe::EngineType::DMA;  // EDMA is not included at the moment
    }
    recipe->arc_jobs = new arc_job_t[recipe->arc_jobs_nr] {};

    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        recipe->arc_jobs[i].logical_engine_id = (Recipe::EngineType)i;
        recipe->arc_jobs[i].engines_filter    = 0;

        recipe->arc_jobs[i].static_ecb.cmds_size = nopCommandSize;
        recipe->arc_jobs[i].static_ecb.cmds      = pNopCommandAddr;

        recipe->arc_jobs[i].dynamic_ecb.cmds_size = nopCommandSize;
        recipe->arc_jobs[i].dynamic_ecb.cmds      = pNopCommandAddr;
    }
    TestDummyRecipe::createValidEcbLists(recipe);

    const uint32_t numPpBlobs = 4;
    const uint32_t size       = 0x10;

    recipe->patch_points_nr = numPpBlobs;
    recipe->blobs_nr        = numPpBlobs;

    recipe->patch_points = new patch_point_t[numPpBlobs] {};
    recipe->blobs        = new blob_t[numPpBlobs] {};

    uint64_t offset = 0;
    for (int i = 0; i < numPpBlobs; i++)
    {
        recipe->blobs[i].size = size;
        recipe->blobs[i].data = (uint8_t*)recipe->patching_blobs_buffer + offset;
        offset += size;

        recipe->patch_points[i].type                                 = (patch_point_t::EPatchPointType)(i % 3);
        int blobIdx                                                  = i;
        recipe->patch_points[i].blob_idx                             = blobIdx;
        recipe->blobs[blobIdx].blob_type.requires_patching           = true;
        recipe->patch_points[i].dw_offset_in_blob                    = i;
        recipe->patch_points[i].memory_patch_point.effective_address = 0x1111111111111111 * (i + 1);
        recipe->patch_points[i].memory_patch_point.section_idx       = (i < 2) ? i : i + 1;
    }

    recipe->permute_tensors_views_nr = 0;

    recipe->persist_tensors_nr                = 2;
    recipe->tensors                           = new persist_tensor_info_t[recipe->persist_tensors_nr] {};
    recipe->tensors[0].section_idx            = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    recipe->tensors[0].name                   = "aaa";
    recipe->tensors[0].size                   = 0x8;
    recipe->tensors[0].multi_views_indices_nr = 0;

    recipe->tensors[1].section_idx            = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;
    recipe->tensors[1].name                   = "bbb";
    recipe->tensors[1].size                   = 0x8;
    recipe->tensors[1].multi_views_indices_nr = 0;

    recipe->node_nr = 1;

    recipe->recipe_conf_nr     = 2;
    recipe->recipe_conf_params = new gc_conf_t[recipe->recipe_conf_nr];

    recipe->recipe_conf_params[0].conf_id    = gc_conf_t::DEVICE_TYPE;
    recipe->recipe_conf_params[0].conf_value = deviceType;

    recipe->recipe_conf_params[1].conf_id    = gc_conf_t::TPC_ENGINE_MASK;
    recipe->recipe_conf_params[1].conf_value = GCFG_TPC_ENGINES_ENABLED_MASK.value();
}

TEST_F_SYN(RecipeLauncherTests, recipeLauncherBasicTest)
{
    uint64_t dynamicComputeEcbListBuffSize = 0;
    uint64_t staticComputeEcbListBuffSize  = 0;
    uint8_t* pNopCommandAddr               = nullptr;
    size_t   nopCommandSize                = 0;
    size_t   cmdSize                       = 0;
    if (m_deviceType == synDeviceGaudi2)
    {
        dynamicComputeEcbListBuffSize = g2fw::DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
        staticComputeEcbListBuffSize  = g2fw::STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
        cmdSize                       = EngEcbNopPkt<g2fw>::getSize();
    }
    else if (m_deviceType == synDeviceGaudi3)
    {
        dynamicComputeEcbListBuffSize = g3fw::DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
        staticComputeEcbListBuffSize  = g3fw::STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
        cmdSize                       = EngEcbNopPkt<g3fw>::getSize();
    }
    else
    {
        HB_ASSERT(false, " Device-type not supported");
    }

    nopCommandSize = std::max(staticComputeEcbListBuffSize, dynamicComputeEcbListBuffSize);

    HB_ASSERT(cmdSize <= nopCommandSize, "Invalid cmdSize 0x{:x} CMD_SIZE_MAX 0x{:x}", cmdSize, nopCommandSize);
    pNopCommandAddr = new uint8_t[nopCommandSize];

    recipe_t recipe;
    buildDummyRecipe(&recipe, m_deviceType, nopCommandSize, pNopCommandAddr);

    InternalRecipeHandle internalRecipeHandle;
    basicRecipeInfo&     basicRecipeI = internalRecipeHandle.basicRecipeHandle;
    basicRecipeI.recipe               = &recipe;
    basicRecipeI.shape_plan_recipe    = nullptr;

    synDeviceId deviceId;
    synStatus   status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to acquire device";

    uint64_t dummyWorkspace;
    ASSERT_EQ(synDeviceMalloc(deviceId, 0x1000, 0, 0, &dummyWorkspace), synSuccess);

    static const uint32_t totalNumOfTensors = 2;
    synLaunchTensorInfo   tensorInfo[totalNumOfTensors] {};
    tensorInfo[0].pTensorAddress = dummyWorkspace + 0x10000;  // just a valid addr
    tensorInfo[0].tensorName     = "aaa";

    tensorInfo[1].pTensorAddress = dummyWorkspace + 0x20000;  // just a valid addr
    tensorInfo[1].tensorName     = "bbb";

    prepareTensorInfo(&internalRecipeHandle, tensorInfo, totalNumOfTensors);

    tensorInfo[0].tensorId = 0;  // we avoid compilation so we can't get tensorId through GC
    tensorInfo[1].tensorId = 1;

    recipe.workspace_nr                                         = 3;
    recipe.workspace_sizes                                      = new uint64_t[recipe.workspace_nr] {};
    recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]    = 0x100;
    recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = 0x180;
    recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM]      = 0;

    internalRecipeHandle.deviceAgnosticRecipeHandle.m_deviceType = m_deviceType;

    internalRecipeHandle.deviceAgnosticRecipeHandle.m_workspaceSize =
        recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE];

    ASSERT_EQ(RecipeTensorsProcessor::process(basicRecipeI,
                                              internalRecipeHandle.deviceAgnosticRecipeHandle.m_recipeTensorInfo,
                                              internalRecipeHandle.deviceAgnosticRecipeHandle.m_recipeDsdStaticInfo),
              synSuccess)
        << "RecipeTensorsProcessor failed to process";

    ASSERT_EQ(DeviceAgnosticRecipeStaticProcessorScal::process(
                  m_deviceType,
                  basicRecipeI,
                  internalRecipeHandle.deviceAgnosticRecipeHandle.m_recipeStaticInfoScal),
              synSuccess)
        << "RecipeStaticProcessorScal failed to process";

    LOG_DEBUG(SYN_RT_TEST, "stream create");
    synStreamHandle computeStream;
    ASSERT_EQ(synStreamCreateGeneric(&computeStream, deviceId, 0), synSuccess) << "Failed to create computeStream";

    LOG_DEBUG(SYN_RT_TEST, "stream launch recipe");
    ASSERT_EQ(synLaunch(computeStream, tensorInfo, totalNumOfTensors, dummyWorkspace, &internalRecipeHandle, 0),
              synSuccess)
        << "Failed to synLaunch";

    ASSERT_EQ(synStreamSynchronize(computeStream), synSuccess) << "Failed to synchronize compute-stream";

    LOG_DEBUG(SYN_RT_TEST, "stream destroy");
    ASSERT_EQ(synStreamDestroy(computeStream), synSuccess) << "Failed to destroy computeStream";

    ASSERT_EQ(synDeviceFree(deviceId, dummyWorkspace, 0), synSuccess) << "Failed to release workspace memory";

    ASSERT_EQ(synDeviceRelease(deviceId), synSuccess) << "Failed to release device";

    destroyDummyRecipe(&recipe);

    delete[] pNopCommandAddr;
}

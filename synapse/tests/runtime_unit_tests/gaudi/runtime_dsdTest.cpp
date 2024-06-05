#include "common/dsd_recipe.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include "synapse_api.h"

//            Node0
//            ---         node1
//   In0----->| |  mid0  ---
//   In0----->| |------> | |
//   Shape0-->| |        | |         node2
//            ---        | |  mid1  ---
//   In1---------------->| |------->| |
//   Shape1------------->| |        | |
//                       ---        | |
//   In2--------------------------->| |--------->Out0 (tensor3)
//   Shape2------------------------>| |
//                                  ---
//
//   Launch tensors = {In0, In1, In2, out0, Shape0, Shape1, Shape2}
//   2 patch points per node
//   Tensors: 0-2 (in) 3(out) 4-5(Internal) 6-8(Shape)
//   Static tensors: In1, Shape2
//

using tensorType = shape_plane_basic_node_t::EShapePlanceTensorDb;

class UTGaudiRtDynamicShapesTest : public DsdRecipe
{
public:
    bool updateSmPatchPointsInfoDb(const recipe_t&                    rRecipe,
                                   shape_plane_graph_t*               pShapePlanRecipe,
                                   DataChunkSmPatchPointsInfo* const& pSmPatchPointsDataChunksInfo,
                                   const blob_t* const&               blobs,
                                   uint64_t                           programCommandsChunksAmount,
                                   uint64_t                           dcSizeCommand,
                                   uint64_t                           blobsBufferAddress);
};

TEST_F(UTGaudiRtDynamicShapesTest, dynamicPatching)
{
    initDynamicPatchingTest();

    uint64_t programCommandsChunksAmount = 0;
    bool     status =
        deviceAgnosticRecipeInfo.m_recipeStaticInfo.getProgramCommandsChunksAmount(EXECUTION_STAGE_ENQUEUE,
                                                                                   programCommandsChunksAmount);
    ASSERT_EQ(status, true) << "Failed to get PRG Commands' Chunk amount";

    uint64_t dcSizeCommand = 0;
    status                 = deviceAgnosticRecipeInfo.m_recipeStaticInfo.getDcSizeCommand(dcSizeCommand);
    ASSERT_EQ(status, true) << "Failed to get PRG Commands' Chunk amount";

    std::unique_ptr<SingleDataChunkHostBuffer[]> dataChunksHostBuffers(
        new SingleDataChunkHostBuffer[programCommandsChunksAmount]);
    for (uint16_t i = 0; i < programCommandsChunksAmount; i++)
    {
        dataChunksHostBuffers[i].reset(new uint8_t[dcSizeCommand]);
    }
    std::vector<uint64_t> dataChunksHostAddresses;
    dataChunksHostAddresses.resize(programCommandsChunksAmount);
    for (uint16_t i = 0; i < programCommandsChunksAmount; i++)
    {
        dataChunksHostAddresses[i] = (uint64_t)dataChunksHostBuffers[i].get();
    }

    const DataChunkPatchPointsInfo* pPatchPointsDataChunksInfo =
        recipeStaticInfo.getPatchingPointsDcLocation(EXECUTION_STAGE_ENQUEUE, PP_TYPE_ID_ALL);

    const DataChunkSmPatchPointsInfo* pSmPatchPointsDataChunksInfo = recipeStaticInfo.getSmPatchingPointsDcLocation();

    const blob_t* blobsChunkBlobs          = blobs;
    uint8_t*      patchingBlobsChunkBuffer = (uint8_t*)blobData;

    {
        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        // All is good
        {
            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, true) << "Should pass";
            verifyBlobs("First run", blobsChunkBlobs);
        }

        // good tensor size (max and min)
        {
            launchTensors[1].tensorSize[1] = MAX_TENSOR_SIZE;
            launchTensors[1].tensorSize[2] = MIN_TENSOR_SIZE;
            launchTensors[1].tensorSize[4] = 0x100000;  // should be ignored, dim is 3

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, true) << "Should pass, tensor size set to min / max";

            launchTensors[1].tensorSize[1] = SET_TENSOR_SIZE;
            launchTensors[1].tensorSize[2] = SET_TENSOR_SIZE;
            launchTensors[1].tensorSize[4] = SET_TENSOR_SIZE;  // should be ignored, dim is 3
        }

        // Bad input tensor size (too big)
        {
            launchTensors[2].tensorSize[2] = MAX_TENSOR_SIZE + 1;

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, bad input tensor size";

            launchTensors[2].tensorSize[2] = SET_TENSOR_SIZE;
        }

        // Bad input tensor size (too small)
        {
            launchTensors[2].tensorSize[2] = MIN_TENSOR_SIZE - 1;

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, bad input tensor size (too small)";

            launchTensors[2].tensorSize[2] = SET_TENSOR_SIZE;
        }
    }

    // Bad non-persistent tensor size
    {
        spg.sp_tensors[NUM_NODES].max_dims[1] = 9;  // MIN_TENSOR_SIZE;

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, bad non-persistent tensor size";

        spg.sp_tensors[NUM_NODES].max_dims[1] = MAX_TENSOR_SIZE;
    }

    // Tensor not set
    {
        uint64_t temp                    = spg.sp_nodes[0].input_tensors[0];
        spg.sp_nodes[0].input_tensors[0] = NUM_NODES + 1;

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, Tensor not set";

        spg.sp_nodes[0].input_tensors[0] = temp;
    }

    // Persist Tensor calculated size != given size
    {
        launchTensors[NUM_PERSIST_TENSORS - 1].tensorSize[1] = SET_TENSOR_SIZE - 1;

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, persistent tensor calculated size ! given size";

        launchTensors[NUM_PERSIST_TENSORS - 1].tensorSize[1] = SET_TENSOR_SIZE;
    }

    // SIF returns error
    {
        spg.sp_nodes[1].input_tensors_nr = NUM_NODE_INPUTS + 1;

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, SIF func returns error";

        spg.sp_nodes[1].input_tensors_nr = NUM_NODE_INPUTS;
    }

    {
        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);
        // Bad shape tensor size ( > max)
        {
            launchTensors[NUM_PERSIST_TENSORS + 1].tensorSize[1] = MAX_TENSOR_SIZE + 1;
            ;

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, bad shape tensor size (>max)";

            launchTensors[NUM_PERSIST_TENSORS + 1].tensorSize[1] = SET_SHAPE_SIZE;
        }

        // Bad shape tensor size (found in SIF)
        {
            launchTensors[NUM_PERSIST_TENSORS + 1].tensorSize[1] = SET_SHAPE_SIZE - 1;

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, bad shape tensor size";

            launchTensors[NUM_PERSIST_TENSORS + 1].tensorSize[1] = SET_SHAPE_SIZE;
        }
    }

    // Missing input tensor
    {
        recipeTensor[1].isInput = false;

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, missing input tensor";

        recipeTensor[1].isInput = true;
        status                  = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
    }

#ifdef EXTRA_CHECKING
    // Tensor set twice
    {
        spg.sp_nodes[1].output_tensors[0]--;

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, same tensor set twice";

        spg.sp_nodes[1].output_tensors[0]++;
    }
#endif

    // Check node_match_output_tensors_nr.
    // 1) Modify the recipe to have only the last node
    // 2) set copy in node0 from in0, in1 to mid0, mid1
    // 3) verify no failures -> mid1 was set for the last node by the copy (mid0 isn't tested)
    {
        spg.sp_nodes[0].node_match_output_tensors_nr = 2;
        uint32_t source[]                            = {0, 1};
        uint32_t dest[]                              = {4, 5};

        spg.sp_nodes[0].output_src_tensors = source;
        spg.sp_nodes[0].output_dst_tensors = dest;

        auto func0 = spg.sp_nodes[0].basic_nodes[0].sif_id.sm_func_index;
        auto func1 = spg.sp_nodes[1].basic_nodes[0].sif_id.sm_func_index;

        spg.sp_nodes[0].basic_nodes[0].sif_id.sm_func_index = INVALID_SHAPE_FUNC_ID;
        spg.sp_nodes[1].basic_nodes[0].sif_id.sm_func_index = INVALID_SHAPE_FUNC_ID;

        spg.sp_tensors[1].infer_info.geometry.maxSizes[0]++;

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);

        // verify normal node_match_output_tensors
        {
            DynamicRecipe dynamicRecipe(recipeInfo,
                                        deviceAgnosticRecipeInfo,
                                        pSmPatchPointsDataChunksInfo,
                                        pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, true) << "expected to pass, node_match_output_tensors";
        }

        // verify fails, mid1 is not copied
        {
            spg.sp_nodes[0].node_match_output_tensors_nr = 1;

            DynamicRecipe dynamicRecipe(recipeInfo,
                                        deviceAgnosticRecipeInfo,
                                        pSmPatchPointsDataChunksInfo,
                                        pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, mid1 is not set";
            spg.sp_nodes[0].node_match_output_tensors_nr = 2;
        }

        // verify can't set same tensor twice from node_match_output_tensors
        {
            dest[0] = 5;  // dest is now {5 ,5}; should fail on duplicate setting

            DynamicRecipe dynamicRecipe(recipeInfo,
                                        deviceAgnosticRecipeInfo,
                                        pSmPatchPointsDataChunksInfo,
                                        pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, duplicate setting of tensor 5";
        }

        // verify can set static shape tensor once
        {
            source[0] = 7;  // source = {7, 1}
            dest[0]   = 8;  // dest   = {8, 5}, should be OK

            DynamicRecipe dynamicRecipe(recipeInfo,
                                        deviceAgnosticRecipeInfo,
                                        pSmPatchPointsDataChunksInfo,
                                        pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, true) << "Should pass, setting static shape tensor";
        }

        // Should fail, setting non-static shape tensor again
        {
            dest[0] = 7;  // should be fail

            DynamicRecipe dynamicRecipe(recipeInfo,
                                        deviceAgnosticRecipeInfo,
                                        pSmPatchPointsDataChunksInfo,
                                        pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, false) << "Should fail, setting non-static shape tensor again";
        }

        spg.sp_tensors[1].infer_info.geometry.maxSizes[0]--;
        spg.sp_nodes[1].basic_nodes[0].sif_id.sm_func_index = func1;
        spg.sp_nodes[0].basic_nodes[0].sif_id.sm_func_index = func0;
        spg.sp_nodes[0].node_match_output_tensors_nr        = 0;
    }

    // Test patching of effective adder in patch points
    {
        sm_patch_point_t* node0PP     = &spn[0].node_patch_points[0];
        sm_patch_point_t  node0PP0org = node0PP[0];
        sm_patch_point_t  node0PP1org = node0PP[1];

        node0PP[0].patch_point_type     = FIELD_DYNAMIC_ADDRESS;
        node0PP[0].patch_point_idx_low  = 0;
        node0PP[0].patch_point_idx_high = -1;
        node0PP[0].patch_size_dw        = 2;

        node0PP[1].patch_point_type     = FIELD_DYNAMIC_ADDRESS;
        node0PP[1].patch_point_idx_low  = 1;
        node0PP[1].patch_point_idx_high = 2;
        node0PP[1].patch_size_dw        = 2;

        DataChunkSmPatchPointsInfo* pMutableSmPatchPointsDataChunksInfo =
            const_cast<DataChunkSmPatchPointsInfo*>(pSmPatchPointsDataChunksInfo);

        bool res = updateSmPatchPointsInfoDb(*recipeInfo.recipe,
                                             recipeInfo.shape_plan_recipe,
                                             pMutableSmPatchPointsDataChunksInfo,
                                             blobsChunkBlobs,
                                             programCommandsChunksAmount,
                                             dcSizeCommand,
                                             (uint64_t)patchingBlobsChunkBuffer);
        ASSERT_EQ(res, true) << "Should succeed updating SM PP Info DB";

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        res = patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, true) << "Should pass";

        data_chunk_patch_point_t* modifiedPP = dynamicRecipe.getPatchPoints();
        verifyAddrPP(modifiedPP, 3, "two patching points");

        node0PP[1].patch_point_type = FIELD_ADDRESS_PART_FULL;  // just not FIELD_DYNAMIC_ADDRESS

        res = patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, true) << "Should pass";

        verifyAddrPP(modifiedPP, 1, "one patching point");

        node0PP[0] = node0PP0org;
        node0PP[1] = node0PP1org;
    }

    // Do not give shape1, copy it from shape0
    {
        uint32_t db_index = spgTensors[7].tensor_db_index;  // backup for recovery

        synLaunchTensorInfoExt launch5 = launchTensors[5];  // backup for recovery

        spgTensors[7].tensor_db_index = INVALID_TENSOR_INDEX;  // Shape1 is internal
        launchTensors[5]              = launchTensors[4];      // duplicate input (removes Shape1 from input)

        spg.sp_nodes[0].node_match_output_tensors_nr = 1;
        uint32_t source[]                            = {6};  // shape0
        uint32_t dest[]                              = {7};  // shape1

        spg.sp_nodes[0].output_src_tensors = source;
        spg.sp_nodes[0].output_dst_tensors = dest;

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, true) << "Should pass";

        spgTensors[7].tensor_db_index                = db_index;  // recover
        launchTensors[5]                             = launch5;   // recover
        spg.sp_nodes[0].node_match_output_tensors_nr = 0;

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
    }

    // Do not give shape1 (internally), should fail
    {
        uint32_t db_index = spgTensors[7].tensor_db_index;  // backup for recovery

        synLaunchTensorInfoExt launch5 = launchTensors[5];  // backup for recovery

        spgTensors[7].tensor_db_index = INVALID_TENSOR_INDEX;  // Shape1 is internal
        launchTensors[5]              = launchTensors[4];      // duplicate input

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);

        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, false) << "Should fail, tensor shape1 not given";

        spgTensors[7].tensor_db_index = db_index;  // recover
        launchTensors[5]              = launch5;   // recover

        status = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                        deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
    }

    // All is good again
    {
        DynamicRecipe dynamicRecipe(recipeInfo,
                                    deviceAgnosticRecipeInfo,
                                    pSmPatchPointsDataChunksInfo,
                                    pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

        bool res =
            patch(dynamicRecipe, patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCommand, tensorIdx2userIdx);
        ASSERT_EQ(res, true) << "Expected to pass again";
        verifyBlobs("Last run", blobsChunkBlobs);
    }
};

TEST_F(UTGaudiRtDynamicShapesTest, dynamicFuserPatching)
{
    initDynamicPatchingTest(true);

    uint64_t programCommandsChunksAmount = 0;
    bool     status =
        deviceAgnosticRecipeInfo.m_recipeStaticInfo.getProgramCommandsChunksAmount(EXECUTION_STAGE_ENQUEUE,
                                                                                   programCommandsChunksAmount);
    ASSERT_EQ(status, true) << "Failed to get PRG Commands' Chunk amount";

    uint64_t dcSizeCommand = 0;
    status                 = deviceAgnosticRecipeInfo.m_recipeStaticInfo.getDcSizeCommand(dcSizeCommand);
    ASSERT_EQ(status, true) << "Failed to get PRG Commands' Chunk amount";

    std::unique_ptr<SingleDataChunkHostBuffer[]> dataChunksHostBuffers(
        new SingleDataChunkHostBuffer[programCommandsChunksAmount]);
    for (uint16_t i = 0; i < programCommandsChunksAmount; i++)
    {
        dataChunksHostBuffers[i].reset(new uint8_t[dcSizeCommand]);
    }
    std::vector<uint64_t> dataChunksHostAddresses;
    dataChunksHostAddresses.resize(programCommandsChunksAmount);
    for (uint16_t i = 0; i < programCommandsChunksAmount; i++)
    {
        dataChunksHostAddresses[i] = (uint64_t)dataChunksHostBuffers[i].get();
    }

    const DataChunkPatchPointsInfo* pPatchPointsDataChunksInfo =
        recipeStaticInfo.getPatchingPointsDcLocation(EXECUTION_STAGE_ENQUEUE, PP_TYPE_ID_ALL);

    const DataChunkSmPatchPointsInfo* pSmPatchPointsDataChunksInfo = recipeStaticInfo.getSmPatchingPointsDcLocation();

    const blob_t* blobsChunkBlobs          = blobs;
    uint8_t*      patchingBlobsChunkBuffer = (uint8_t*)blobData;

    {
        {
            LOG_INFO(SYN_API, "------ starting fuser test -------");
            DynamicRecipe dynamicRecipe(recipeInfo,
                                        deviceAgnosticRecipeInfo,
                                        pSmPatchPointsDataChunksInfo,
                                        pPatchPointsDataChunksInfo->m_dataChunkPatchPoints);

            bool res = patch(dynamicRecipe,
                             patchingBlobsChunkBuffer,
                             dataChunksHostAddresses,
                             dcSizeCommand,
                             tensorIdx2userIdx);
            ASSERT_EQ(res, true) << "Should pass";
            verifyBlobs("First run", blobsChunkBlobs);
        }
    }
}

TEST_F(UTGaudiRtDynamicShapesTest, dynamicPatchingVerifyRecipe)
{
    initDynamicPatchingTest();

    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();

    // All is good
    {
        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, true) << "Should pass";
    }

    // Bad data tensor db
    {
        uint32_t temp                     = spg.sp_tensors[1].tensor_db_index;
        spg.sp_tensors[1].tensor_db_index = (uint32_t)recipe.persist_tensors_nr;
        bool res                          = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad data tensor DB";
        spg.sp_tensors[1].tensor_db_index = temp;
    }

    // Bad shape tensor db
    {
        uint32_t temp                                          = spg.sp_tensors[FIRST_SHAPE_TENSOR + 1].tensor_db_index;
        spg.sp_tensors[FIRST_SHAPE_TENSOR + 1].tensor_db_index = (uint32_t)spg.shape_tensors_list_nr;
        bool res                                               = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad shape tensor DB";
        spg.sp_tensors[FIRST_SHAPE_TENSOR + 1].tensor_db_index = temp;
    }

    // Bad input Tensor
    {
        uint64_t temp                    = spg.sp_nodes[1].input_tensors[1];
        spg.sp_nodes[1].input_tensors[1] = spg.sp_tensors_nr;
        bool res                         = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad input tensor";
        spg.sp_nodes[1].input_tensors[1] = temp;
    }

    // Bad output Tensor
    {
        uint64_t temp                     = spg.sp_nodes[2].output_tensors[0];
        spg.sp_nodes[2].output_tensors[0] = spg.sp_tensors_nr + 1;
        bool res                          = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad output tensor";
        spg.sp_nodes[2].output_tensors[0] = temp;
    }

    // Bad SIF
    {
        sm_function_id_t id {};
        id.sm_func_index = 2;
        sfr.registerSIF(id, nullptr, "null sif", GC_SIF_VERSION);

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad sif function (null)";

        sfr.registerSIF(id, sif2_1, "sif function " + std::to_string(id.sm_func_index), GC_SIF_VERSION);
    }

    // Bad SMF
    {
        uint64_t temp = spg.sp_nodes[1].node_patch_points[0].smf_id.sm_func_index;
        spg.sp_nodes[1].node_patch_points[0].smf_id.sm_func_index = INVALID_SHAPE_FUNC_ID;

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad SMF id";

        spg.sp_nodes[1].node_patch_points[0].smf_id.sm_func_index = temp;
    }

    // Bad SMF function
    {
        sm_function_id_t smf_id = spg.sp_nodes[2].node_patch_points[1].smf_id;
        smf_t            smf    = sfr.getSMF(smf_id);

        sfr.registerSMFTestingOnly((ShapeFuncID)smf_id.sm_func_index, nullptr, "nullptr");

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad smf function (null)";

        sfr.registerSMFTestingOnly((ShapeFuncID)smf_id.sm_func_index, smf, "recovered func");
    }

    // Bad blob
    {
        uint64_t temp                                 = spg.sp_nodes[0].node_patch_points[0].blob_idx;
        spg.sp_nodes[0].node_patch_points[0].blob_idx = recipe.blobs_nr;

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad blob";

        spg.sp_nodes[0].node_patch_points[0].blob_idx = temp;
    }

    // Bad blob size
    {
        uint64_t temp                                          = spg.sp_nodes[1].node_patch_points[1].dw_offset_in_blob;
        spg.sp_nodes[1].node_patch_points[1].dw_offset_in_blob = BLOB_DATA_SIZE;  // BLOB_DATA_SIZE already in dw

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad blob size";

        spg.sp_nodes[1].node_patch_points[1].dw_offset_in_blob = temp;
    }

    // requires_patching not set
    {
        recipe.blobs[1].blob_type_all = blob_t::EXE;
        bool res                      = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, flag requires patching not set";

        recipe.blobs[1].blob_type_all = blob_t::PATCHING;
    }

    // Bad roi input index
    {
        spg.sp_nodes[1].node_patch_points[1].roi_idx = 1;  // Bad roi index, 1 >= 1

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad roi input index";

        spg.sp_nodes[1].node_patch_points[1].roi_idx = 0;
    }

    // roi input and output number of tensors is 0 (output is already 0 before the test)
    {
        roiInfo.roi_in_tensor_nr = 0;  // now both #input, #output = 0

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, number of roi in and out are both 0";

        roiInfo.roi_in_tensor_nr = NUM_NODE_INPUTS;
    }

    // Bad PP patching
    {
        auto tempLow  = spg.sp_nodes[1].node_patch_points[0].patch_point_idx_low;
        auto tempHigh = spg.sp_nodes[1].node_patch_points[0].patch_point_idx_high;

        spg.sp_nodes[1].node_patch_points[0].patch_point_type     = FIELD_DYNAMIC_ADDRESS;
        spg.sp_nodes[1].node_patch_points[0].patch_point_idx_low  = -1;
        spg.sp_nodes[1].node_patch_points[0].patch_point_idx_high = -1;

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "patch point patching but no pp idx";

        spg.sp_nodes[1].node_patch_points[0].patch_point_idx_low  = 0;
        spg.sp_nodes[1].node_patch_points[0].patch_point_idx_high = recipe.patch_points_nr;

        res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "bad pp patching index";

        spg.sp_nodes[1].node_patch_points[0].patch_point_idx_low  = tempLow;
        spg.sp_nodes[1].node_patch_points[0].patch_point_idx_high = tempHigh;

        spg.sp_nodes[1].node_patch_points[0].patch_point_type = FIELD_ADDRESS_PART_FULL;
    }

    // Bad output_source_tensors
    {
        spg.sp_nodes[1].node_match_output_tensors_nr = 2;
        uint32_t source[]                            = {1, spg.sp_tensors_nr};
        uint32_t dest[]                              = {2, 3};

        spg.sp_nodes[1].output_src_tensors    = source;
        spg.sp_nodes[1].output_dst_tensors    = dest;
        bool res                              = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad output_source_tensor index";

        spg.sp_nodes[1].node_match_output_tensors_nr = 0;
    }

    // Bad output_dest_tensors
    {
        spg.sp_nodes[1].node_match_output_tensors_nr = 2;
        uint32_t source[]                            = {1, 0};
        uint32_t dest[]                              = {2, spg.sp_tensors_nr};

        spg.sp_nodes[1].output_src_tensors    = source;
        spg.sp_nodes[1].output_dst_tensors    = dest;
        bool res                              = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad output_dest_tensor index";

        spg.sp_nodes[1].node_match_output_tensors_nr = 0;
    }

    // Internal shpae tensor has an invalid db index
    {
        uint32_t temp                     = spg.sp_tensors[7].tensor_db_index;
        spg.sp_tensors[7].tensor_db_index = INVALID_TENSOR_INDEX;

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, true) << "Should pass, internal shape tensor has INVALID_TENSOR_INDEX";

        spg.sp_tensors[7].tensor_db_index = temp;
    }

    // bad tensor type
    {
        tensor_info_t::ETensorType temp = spg.sp_tensors[7].tensor_type;
        spg.sp_tensors[7].tensor_type   = (tensor_info_t::ETensorType)100;

        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, false) << "Should fail, bad tensor type";

        spg.sp_tensors[7].tensor_type = temp;
    }

    // All is good again (verify that I fixed every error I put it)
    {
        bool res = RecipeVerification::verifyDynamicRecipe(&recipe, &spg);
        ASSERT_EQ(res, true) << "Should pass again";
    }
}

class UTGaudiRtDSDRecipe : public ::testing::Test
{
};

TEST_F(UTGaudiRtDSDRecipe, shape_func_repository_destroy)
{
    synStatus status = synInitialize();
    ASSERT_EQ(status, synSuccess);

    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();

    auto sifBefore = sfr.getAllSifTestingOnly();
    auto smfBefore = sfr.getAllSmfTestingOnly();

    LOG_INFO_T(SYN_API, "---- test shape_func_repository_destroy - call synDestroy");
    status = synDestroy();
    ASSERT_EQ(status, synSuccess);
    // m_numOfAcquiredDevices = 0; // the device have been released during synDestroy

    LOG_INFO_T(SYN_API, "---- test shape_func_repository_destroy - call synInitialize");
    status = synInitialize();
    ASSERT_EQ(status, synSuccess);

    auto sifAfter = sfr.getAllSifTestingOnly();
    auto smfAfter = sfr.getAllSmfTestingOnly();

    ASSERT_EQ(sifBefore.size(), sifAfter.size()) << "sif maps are not equal";
    ASSERT_EQ(smfBefore.size(), smfAfter.size()) << "smf maps are not equal";

    status = synDestroy();
    ASSERT_EQ(status, synSuccess);
}

bool UTGaudiRtDynamicShapesTest::updateSmPatchPointsInfoDb(
    const recipe_t&                    rRecipe,
    shape_plane_graph_t*               pShapePlanRecipe,
    DataChunkSmPatchPointsInfo* const& pSmPatchPointsDataChunksInfo,
    const blob_t* const&               blobs,
    uint64_t                           programCommandsChunksAmount,
    uint64_t                           dcSizeCommand,
    uint64_t                           blobsBufferAddress)
{
    return StaticInfoProcessor::_updateSmPatchPointsInfoDb(rRecipe,
                                                           pShapePlanRecipe,
                                                           pSmPatchPointsDataChunksInfo,
                                                           blobs,
                                                           programCommandsChunksAmount,
                                                           dcSizeCommand,
                                                           blobsBufferAddress);
}
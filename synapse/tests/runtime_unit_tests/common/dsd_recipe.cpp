#include "dsd_recipe.hpp"

#include "device_agnostic_recipe_static_processor.hpp"
#include "graph_compiler/habana_global_conf.h"
#include "recipe_processing_utils.hpp"
#include "runtime/common/habana_global_conf_runtime.h"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include "synapse_api.h"
#include "tpc_kernel_lib_interface.h"


void DsdRecipe::SetUp()
{
    synStatus status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "synInitialize failed!";
}

void DsdRecipe::TearDown()
{
    for (uint8_t executionStage = EXECUTION_STAGE_ACTIVATE; executionStage < EXECUTION_STAGE_LAST; executionStage++)
    {
        recipeStaticInfo.deleteStagePatchingPointsDcs((eExecutionStage)executionStage);
    }
    synStatus status = synDestroy();
    ASSERT_EQ(status, synSuccess) << "Failed to destroy synapse";
}

void DsdRecipe::createRecipeTensors()
{
    tensorName[0]              = "In0";
    recipeTensor[0].isInput    = true;
    recipeTensor[0].dimensions = DIMS;
    tensorName[1]              = "In1";
    recipeTensor[1].isInput    = true;
    recipeTensor[1].dimensions = DIMS;
    tensorName[2]              = "In2";
    recipeTensor[2].isInput    = true;
    recipeTensor[2].dimensions = DIMS;

    tensorName[3]              = "Out";
    recipeTensor[3].isInput    = false;
    recipeTensor[3].dimensions = DIMS;

    for (int i = 0; i < NUM_PERSIST_TENSORS; i++)
    {
        recipeTensor[i].name                   = tensorName[i].c_str();
        recipeTensor[i].multi_views_indices_nr = 0;
    }
}

void DsdRecipe::createRecipeBlobs()
{
    recipe.blobs_nr = NUM_BLOBS;
    recipe.blobs    = &blobs[0];

    for (int idx = 0; idx < NUM_BLOBS; idx++)
    {
        blobs[idx].size = BLOB_DATA_SIZE * sizeof(uint32_t);
    }
}

void DsdRecipe::createSpgNodes()
{
    roiInfo.roi_in_tensor_nr  = NUM_NODE_INPUTS;
    roiInfo.roi_in_tensors    = &tensorRoi;
    roiInfo.roi_out_tensor_nr = 0;  // do not use outputs

    for (uint i = 0; i < NUM_NODES; i++)
    {
        shape_plane_node_t& node = spg.sp_nodes[i];

        node.node_match_output_tensors_nr = 0;

        node.input_tensors_nr  = NUM_NODE_INPUTS;
        node.input_tensors     = nodeIn[i];
        node.output_tensors_nr = NUM_NODE_OUTPUTS;
        node.output_tensors    = nodeOut[i];

        node.activation_rois_nr = 1;
        node.activation_rois    = &roiInfo;

        memset(node.nodeData, 0, sizeof(node.nodeData));

        node.node_patch_points_nr = 2;
        node.node_patch_points    = smPP[i];

        for (uint j = 0; j < NUM_PP_NODE; j++)
        {
            node.node_patch_points[j].smf_id.sm_funcid  = j;
            node.node_patch_points[j].smf_id.sm_tableid = LIB_ID_RESERVED_FOR_GC_SMF;
            node.node_patch_points[j].blob_idx          = i;
            node.node_patch_points[j].dw_offset_in_blob = j * 4;
            node.node_patch_points[j].patch_size_dw     = j + 1;
            node.node_patch_points[j].roi_idx           = 0;
            node.node_patch_points[j].metadata          = nullptr;
        }

        node.basic_nodes[0].sif_id.sm_func_index = i;

        node.basic_nodes[0].sif_params_nr = 1;
        node.basic_nodes[0].sif_params    = &sif_params[i];
        node.basic_nodes[0].sif_params[0] = i * 3;

        std::string name = "node" + std::to_string(i);
        strcpy(node.node_name, name.c_str());
    }

    shape_plane_node_t* n = spg.sp_nodes;

    n[0].input_tensors[0]  = 0;
    n[0].input_tensors[1]  = 0;
    n[0].input_tensors[2]  = 6;
    n[0].output_tensors[0] = 4;
    n[1].input_tensors[0]  = 1;
    n[1].input_tensors[1]  = 4;
    n[1].input_tensors[2]  = 7;
    n[1].output_tensors[0] = 5;
    n[2].input_tensors[0]  = 2;
    n[2].input_tensors[1]  = 5;
    n[2].input_tensors[2]  = 8;
    n[2].output_tensors[0] = 3;

    for (uint i = 0; i < NUM_NODES; i++)
    {
        shape_plane_node_t& node = spg.sp_nodes[i];

        node.node_db_tensors_nr = 0;

        strcpy(node.basic_nodes[0].node_name, node.node_name);

        node.basic_nodes[0].input_tensors_nr = node.input_tensors_nr;
        node.basic_nodes[0].input_tensors    = node.input_tensors;  // points to the same array
        for (int in = 0; in < node.input_tensors_nr; in++)
        {
            node.basic_nodes[0].input_tensors_db[in] = shape_plane_basic_node_t::GRAPH_TENSOR_DB;
        }
        node.basic_nodes[0].output_tensors_nr = node.output_tensors_nr;
        node.basic_nodes[0].output_tensors    = node.output_tensors;  // points to the same array
        for (int out = 0; out < node.output_tensors_nr; out++)
        {
            node.basic_nodes[0].output_tensors_db[out] = shape_plane_basic_node_t::GRAPH_TENSOR_DB;
        }
    }
}

void DsdRecipe::createSpgTensors()
{
    spg.shape_tensors_list_nr = NUM_SHAPE_TENSORS;

    for (uint i = 0; i < NUM_TENSORS; i++)
    {
        tensor_info_t& curr = spgTensors[i];

        memset(&curr.infer_info, 0, sizeof(curr.infer_info));

        curr.infer_info.geometry.dims = DIMS;
        for (int j = 0; j < MAX_DIMENSIONS_NUM; j++)
        {
            curr.min_dims[j] = MIN_TENSOR_SIZE;  // This is overridden below for some tensors
            curr.max_dims[j] = MAX_TENSOR_SIZE;  // This is overridden below for some tensors
            curr.strides[j]  = 2;
        }
        curr.data_type = 0;
    }

    tensor_info_t* t = spgTensors;

    t[0].tensor_type      = tensor_info_t::PERSISTENT_TENSOR;
    t[0].tensor_db_index  = 0;  // In0
    t[1].tensor_type      = tensor_info_t::PERSISTENT_TENSOR;
    t[1].tensor_db_index  = 1;  // In1
    t[2].tensor_type      = tensor_info_t::PERSISTENT_TENSOR;
    t[2].tensor_db_index  = 2;  // In2
    t[3].tensor_type      = tensor_info_t::PERSISTENT_TENSOR;
    t[3].user_tensor_type = DATA_TENSOR_DYNAMIC;
    t[3].tensor_db_index  = 3;  // Out

    t[4].tensor_type     = tensor_info_t::INTERNAL_TENSOR;
    t[4].tensor_db_index = INVALID_TENSOR_INDEX;  // Mid0
    t[5].tensor_type     = tensor_info_t::INTERNAL_TENSOR;
    t[5].tensor_db_index = INVALID_TENSOR_INDEX;  // Mid1

    t[6].tensor_type     = tensor_info_t::SHAPE_TENSOR;
    t[6].tensor_db_index = 0;  // Shape0
    t[7].tensor_type     = tensor_info_t::SHAPE_TENSOR;
    t[7].tensor_db_index = 1;  // Shape1
    t[8].tensor_type     = tensor_info_t::SHAPE_TENSOR;
    t[8].tensor_db_index = 2;  // Shape2

    // Tensors 0 (In1) and tensor 8 (Shape2) are static, need to set their sizes
    for (int j = 0; j < MAX_DIMENSIONS_NUM; j++)
    {
        t[1].min_dims[j]         = SET_TENSOR_SIZE;
        t[1].max_dims[j]         = SET_TENSOR_SIZE;
        t[1].infer_info.geometry.maxSizes[j] = SET_TENSOR_SIZE;  // Make it static DATA
        t[8].min_dims[j]         = SET_SHAPE_SIZE;
        t[8].max_dims[j]         = SET_SHAPE_SIZE;
        t[8].infer_info.geometry.maxSizes[j] = SET_SHAPE_SIZE;  // Make it static SHAPE
    }
}

void DsdRecipe::createIn()
{
    synLaunchTensorInfoExt* t = launchTensors;

    t[0].tensorName     = "In0";
    t[0].pTensorAddress = 1;
    t[0].tensorType     = DATA_TENSOR_DYNAMIC;
    t[1].tensorName     = "In1";
    t[1].pTensorAddress = 2;
    t[1].tensorType     = DATA_TENSOR;
    t[2].tensorName     = "In2";
    t[2].pTensorAddress = 3;
    t[2].tensorType     = DATA_TENSOR_DYNAMIC;
    t[3].tensorName     = "Out";
    t[3].pTensorAddress = 4;
    t[3].tensorType     = DATA_TENSOR_DYNAMIC;

    for (uint i = 0; i < NUM_PERSIST_TENSORS; i++)
    {
        synLaunchTensorInfoExt& curr = launchTensors[i];
        for (int j = 0; j < HABANA_DIM_MAX; j++)
        {
            curr.tensorSize[j] = SET_TENSOR_SIZE;
        }
    }

    t[4].tensorName     = "Shape0";
    t[4].pTensorAddress = 5;
    t[4].tensorType     = SHAPE_TENSOR;
    t[5].tensorName     = "Shape1";
    t[5].pTensorAddress = 6;
    t[5].tensorType     = SHAPE_TENSOR;
    t[6].tensorName     = "Shape2";
    t[6].pTensorAddress = 7;
    t[6].tensorType     = SHAPE_TENSOR;

    for (uint i = NUM_PERSIST_TENSORS; i < NUM_PERSIST_TENSORS + NUM_SHAPE_TENSORS; i++)
    {
        synLaunchTensorInfoExt& curr = launchTensors[i];
        for (int j = 0; j < HABANA_DIM_MAX; j++)
        {
            curr.tensorSize[j] = SET_SHAPE_SIZE;
        }
    }
}

void DsdRecipe::createBlobs()
{
    for (uint i = 0; i < NUM_BLOBS; i++)
    {
        blobs[i].data          = blobData[i];
        blobs[i].blob_type_all = blob_t::PATCHING;
        for (int j = 0; j < BLOB_DATA_SIZE; j++)
        {
            blobData[i][j] = i << 16 | j;
        }
    }
    recipe.blobs                      = blobs;
    recipe.patching_blobs_buffer      = (uint64_t*)blobData;
    recipe.patching_blobs_buffer_size = NUM_BLOBS * BLOB_DATA_SIZE;
}

void DsdRecipe::registerFunctions()
{
    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();

    for (uint i = 0; i < NUM_NODES; i++)
    {
        sm_function_id_t id {};
        id.sm_func_index = i;
        sfr.registerSIF(id, sif2_1, "sif function " + std::to_string(id.sm_func_index), GC_SIF_VERSION);
    }

    for (int i = 0; i < NUM_PP_NODE; i++)
    {
        sfr.registerSMFTestingOnly((ShapeFuncID)i, smf0, "smf0");
    }
}

void DsdRecipe::verifyBlobs(std::string msg, const blob_t* blobsChunksBlobs)
{
    const blob_t* verifiedBlobs = nullptr;

    if (blobsChunksBlobs != nullptr)
    {
        verifiedBlobs = blobsChunksBlobs;
    }
    else
    {
        verifiedBlobs = recipe.blobs;
        ;
    }

    for (int i = 0; i < NUM_BLOBS; i++)
    {
        uint32_t* blobsData = (uint32_t*)verifiedBlobs[i].data;

        for (int j = 0; j < BLOB_DATA_SIZE; j++)
        {
            bool special = false;
            for (int pp = 0; pp < NUM_PP_NODE; pp++)
            {
                if (j == pp * 4)  // we patched it
                {
                    int ppSize = pp + 1;
                    for (int dw = 0; dw < ppSize; dw++)
                    {
                        ASSERT_EQ(blobsData[j + dw], (dw + 1) * 256)
                            << msg << " Wrong value in blob (expected to change) " << i << " " << j << " " << pp << " "
                            << dw << " " << blobsData[j + dw] << " " << (dw + 1) * 256;
                    }
                    j += (ppSize - 1);
                    special = true;
                }
            }
            if (special) continue;
            ASSERT_EQ((blobsData[j]), (i << 16 | j)) << msg << " Wrong value in blob (expected unchanged) " << i << " "
                                                     << j << " " << blobsData[j] << " " << (i << 16 | j);
        }
    }
}

void DsdRecipe::verifyAddrPP(data_chunk_patch_point_t* pp, int n, std::string msg)
{
    for (int i = 0; i < NUM_ADDR_PP; i++)
    {
        if (i < n)
        {
            uint64_t expected = i < n ? (uint64_t {0x200} << 32) + 0x100 : 0;
            ASSERT_EQ(pp[i].memory_patch_point.effective_address, expected)
                << msg << " Bad value in effective_address after patching " << i << " "
                << pp[i].memory_patch_point.effective_address << " " << expected;
            pp[i].memory_patch_point.effective_address = 0;
        }
    }
}

void DsdRecipe::initTensorMap()
{
    tensorIdx2userIdx[0].clear();
    tensorIdx2userIdx[0].resize(NUM_PERSIST_TENSORS, INVALID_TENSOR_INDEX);
    for (uint i = 0; i < NUM_PERSIST_TENSORS; i++)
    {
        tensorIdx2userIdx[0][i] = i;  // persistent tensors are in user given tensors [0-NUM_PERSIST_TENSOR) 0-3
    }

    tensorIdx2userIdx[1].clear();
    tensorIdx2userIdx[1].resize(NUM_SHAPE_TENSORS, INVALID_TENSOR_INDEX);
    for (uint i = 0; i < NUM_SHAPE_TENSORS; i++)
    {
        tensorIdx2userIdx[1][i] =
            NUM_PERSIST_TENSORS + i;  // Shape tensors are in user given tensors from NUM_PERSIST_TENSOR 4-6
    }
}

void DsdRecipe::initDynamicPatchingTest(bool addFuser)
{
    recipeInfo.recipe               = &recipe;
    recipe.permute_tensors_views_nr = 0;
    recipe.persist_tensors_nr       = NUM_PERSIST_TENSORS;
    recipe.tensors                  = &recipeTensor[0];
    recipe.patch_points             = addrPP;
    recipe.patch_points_nr          = NUM_ADDR_PP;
    recipe.recipe_conf_params       = gcConf;

    RecipeProcessingUtils::getExecutionJobs(synDeviceGaudi, recipe.execute_jobs_nr, recipe.execute_jobs);

    uint64_t blobIndices[1];
    blobIndices[0] = 0;

    uint32_t  programsNr = 1;        // The number of programs on the recipe.
    program_t programs[programsNr];  // The recipe's programs.
    programs[0].program_length = 1;
    programs[0].blob_indices   = &blobIndices[0];

    recipe.programs_nr = programsNr;
    recipe.programs    = &programs[0];

    gc_conf_t* pFiller    = recipe.recipe_conf_params;
    pFiller->conf_id      = gc_conf_t::DEVICE_TYPE;
    pFiller->conf_value   = synDeviceGaudi;
    pFiller++;
    pFiller->conf_id      = gc_conf_t::TPC_ENGINE_MASK;
    pFiller->conf_value   = 0xFF;
    pFiller++;
    recipe.recipe_conf_nr = std::distance(recipe.recipe_conf_params, pFiller);

    createRecipeBlobs();
    createRecipeTensors();

    recipeInfo.shape_plan_recipe                = &spg;
    recipeInfo.shape_plan_recipe->sp_node_nr    = NUM_NODES;
    recipeInfo.shape_plan_recipe->sp_nodes      = &spn[0];
    recipeInfo.shape_plan_recipe->sp_tensors_nr = NUM_TENSORS;
    recipeInfo.shape_plan_recipe->sp_tensors    = &spgTensors[0];

    for (int i = 0; i < NUM_NODES; i++)
    {
        spn[i].basic_nodes_nr = 1;
        spn[i].basic_nodes    = &spbn[i];

        spbn[i].input_tensors_db  = tensorInType[i];
        spbn[i].output_tensors_db = tensorOutType[i];
    }

    createSpgNodes();

    createSpgTensors();
    createIn();
    createBlobs();
    createShapeTensors();

    registerFunctions();
    initTensorMap();

    if (addFuser) setFuser();

    synStatus status = DeviceAgnosticRecipeStaticProcessor::process(recipe,
                                                                    deviceAgnosticRecipeInfo.m_recipeStaticInfo,
                                                                    synDeviceGaudi);
    ASSERT_EQ(status, synSuccess) << "DeviceAgnosticRecipeStaticProcessor::process failed";
    bool ret = RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(recipeInfo,
                                                                      deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                      deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
    ASSERT_EQ(ret, true) << "processShapePlanRecipe failed";
    const uint64_t dcSizeCommand = GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP.value() * 1024;
    ret                          = recipeStaticInfo.allocateSmPatchingPointsDcLocation(NUM_ADDR_PP, dcSizeCommand);
    ASSERT_EQ(ret, true) << "allocateSmPatchingPointsDcLocation failed";
    bool isDsd                        = (recipeInfo.shape_plan_recipe != nullptr);
    bool shouldAllocateDcsPatchPoints = false;
    shouldAllocateDcsPatchPoints      = true;

    if (shouldAllocateDcsPatchPoints)
    {
        ret = StaticInfoProcessor::debugBuildDcPatchPointsOnBlobsChunksDatabases(
            recipeInfo.shape_plan_recipe,
            synDeviceGaudi,
            deviceAgnosticRecipeInfo,
            recipeStaticInfo,
            recipe,
            dcSizeCommand,
            deviceAgnosticRecipeInfo.m_recipeStaticInfo.m_patchingBlobsChunksSize,
            deviceAgnosticRecipeInfo.m_recipeStaticInfo.m_patchingBlobsChunksDataChunksAmount,
            isDsd);
        ASSERT_EQ(ret, true) << "debugBuildDcPatchPointsOnBlobsChunksDatabases failed";
    }
}

void DsdRecipe::createShapeTensors()
{
    for (int i = 0; i < NUM_SHAPE_TENSORS; i++)
    {
        shapeTensorsName[i]      = "Shape" + std::to_string(i);
        shapeTensorsInfo[i].name = shapeTensorsName[i].c_str();
    }
    spg.shape_tensors = &shapeTensorsInfo[0];
}

// At the moment, for simplicity, we defined the DC-Size to be equal to the BC-Size
// Hence, a simpler copy operation is good enough
void DsdRecipe::copyBetweenBlobsChunksAndDataChunks(uint8_t*                     patchingBlobsChunkBuffer,
                                                    const std::vector<uint64_t>& dataChunksHostAddresses,
                                                    uint64_t                     dcSizeCpDma,
                                                    bool                         isToDataChunk)
{
    uint8_t* pCurrentBlobsChunkLocation = patchingBlobsChunkBuffer;

    uint64_t accomulateCopySize = 0;

    for (int i = 0; i < dataChunksHostAddresses.size(); i++)
    {
        uint64_t copySize = dcSizeCpDma;
        if (accomulateCopySize + copySize > TOTAL_DATA_SIZE)
        {
            copySize = TOTAL_DATA_SIZE - accomulateCopySize;
        }

        if (isToDataChunk)
        {
            std::memcpy((uint8_t*)dataChunksHostAddresses[i], pCurrentBlobsChunkLocation, copySize);
            pCurrentBlobsChunkLocation += dcSizeCpDma;
        }
        else
        {
            std::memcpy(pCurrentBlobsChunkLocation, (uint8_t*)dataChunksHostAddresses[i], copySize);
            pCurrentBlobsChunkLocation += dcSizeCpDma;
        }

        accomulateCopySize += copySize;
    }
}

bool DsdRecipe::patch(DynamicRecipe&               dynamicRecipe,
                      uint8_t*&                    patchingBlobsChunkBuffer,
                      const std::vector<uint64_t>& dataChunksHostAddresses,
                      uint64_t                     dcSizeCpDma,
                      const std::vector<uint32_t>* tensorIdx2userIdx)
{
    uint32_t NUM_USER_TENSORS = ARRAY_SIZE(launchTensors);  // Persistent + shape tensors

    copyBetweenBlobsChunksAndDataChunks(patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCpDma, true);
    synLaunchTensorInfoExt launchTensorInfo[NUM_USER_TENSORS];
    for (size_t i = 0; i < NUM_USER_TENSORS; i++)
    {
        launchTensorInfo[i].tensorName     = launchTensors[i].tensorName;
        launchTensorInfo[i].pTensorAddress = launchTensors[i].pTensorAddress;
        launchTensorInfo[i].tensorType     = launchTensors[i].tensorType;
        launchTensorInfo[i].tensorId       = launchTensors[i].tensorId;
        memcpy(launchTensorInfo[i].tensorSize, launchTensors[i].tensorSize, sizeof(TSize) * HABANA_DIM_MAX);
    }

    bool res = dynamicRecipe.patch(launchTensorInfo, NUM_USER_TENSORS, dataChunksHostAddresses, tensorIdx2userIdx);
    if (!res)
    {
        return res;
    }

    copyBetweenBlobsChunksAndDataChunks(patchingBlobsChunkBuffer, dataChunksHostAddresses, dcSizeCpDma, false);
    return res;
}

void DsdRecipe::setFuser()
{
    // See at beginning of file for description
    // link pointer as needed

    for (int i = 0; i < 2; i++)
    {
        fuserBasicNode[i].input_tensors     = fuserIn[i];
        fuserBasicNode[i].output_tensors    = fuserOut[i];
        fuserBasicNode[i].input_tensors_db  = fuserInType[i];
        fuserBasicNode[i].output_tensors_db = fuserOutType[i];
        fuserBasicNode[i].sif_params        = fuserSifParams[i];

        strcpy(fuserBasicNode[i].node_name, ("subNode" + std::to_string(i)).c_str());
    }

    spn[1].basic_nodes_nr     = 2;
    spn[1].basic_nodes        = fuserBasicNode;
    spn[1].node_db_tensors_nr = 2;
    spn[1].node_db_tensors    = fuserTensorDb;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < MAX_DIMENSIONS_NUM; j++)
        {
            fuserTensorDb[i].min_dims[j] = MIN_TENSOR_SIZE;  // This is overridden below for some tensors
            fuserTensorDb[i].max_dims[j] = MAX_TENSOR_SIZE;  // This is overridden below for some tensors
            fuserTensorDb[i].strides[j]  = 2;
        }
        fuserTensorDb[i].infer_info.geometry.dims = DIMS;
        fuserTensorDb[i].data_type       = 0;
        fuserTensorDb[i].tensor_db_index = -1;
    }

    for (int i = 0; i < 2; i++)
    {
        fuserBasicNode[i].input_tensors_nr     = 3;
        fuserBasicNode[i].sif_id.sm_func_index = 0;
        fuserBasicNode[i].sif_params_nr        = 1;
        fuserBasicNode[i].sif_params[0]        = 0;
        fuserBasicNode[i].sif_version          = GC_SIF_VERSION;
    }

    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();

    sm_function_id_t id {};
    id.sm_func_index = 10;
    sfr.registerSIF(id, sif2_2, "sif function " + std::to_string(id.sm_func_index), GC_SIF_VERSION);
    fuserBasicNode[0].sif_id.sm_func_index = 10;

    // node 0 in/out
    fuserBasicNode[0].output_tensors_nr    = 2;
    fuserBasicNode[0].input_tensors[0]     = 4;
    fuserBasicNode[0].input_tensors_db[0]  = tensorType::GRAPH_TENSOR_DB;  // mid0
    fuserBasicNode[0].input_tensors[1]     = 1;
    fuserBasicNode[0].input_tensors_db[1]  = tensorType::GRAPH_TENSOR_DB;  // In 1
    fuserBasicNode[0].input_tensors[2]     = 6;
    fuserBasicNode[0].input_tensors_db[2]  = tensorType::GRAPH_TENSOR_DB;  // shape0
    fuserBasicNode[0].output_tensors[0]    = 0;
    fuserBasicNode[0].output_tensors_db[0] = tensorType::NODE_TENSOR_DB;  // temp0
    fuserBasicNode[0].output_tensors[1]    = 1;
    fuserBasicNode[0].output_tensors_db[1] = tensorType::NODE_TENSOR_DB;  // temp1

    // node 1 in/out
    fuserBasicNode[1].output_tensors_nr    = 1;
    fuserBasicNode[1].input_tensors[0]     = 0;
    fuserBasicNode[1].input_tensors_db[0]  = tensorType::NODE_TENSOR_DB;  // temp0
    fuserBasicNode[1].input_tensors[1]     = 1;
    fuserBasicNode[1].input_tensors_db[1]  = tensorType::NODE_TENSOR_DB;  // temp1
    fuserBasicNode[1].input_tensors[2]     = 6;
    fuserBasicNode[1].input_tensors_db[2]  = tensorType::GRAPH_TENSOR_DB;  // shape0
    fuserBasicNode[1].output_tensors[0]    = 5;
    fuserBasicNode[1].output_tensors_db[0] = tensorType::GRAPH_TENSOR_DB;  // mid1

    // return to original state
    // spn[1].basic_nodes_nr = 1;
}

tpc_lib_api::GlueCodeReturn
sif2_n(const tpc_lib_api::ShapeInferenceParams* inputParams, tpc_lib_api::ShapeInferenceOutput* outputData, int numOut)
{
    auto numIn = DsdRecipe::NUM_NODE_INPUTS;
    if (inputParams->inputTensorsNr != numIn)
    {
        LOG_INFO(SYN_API, "sif2_1 bad num of input tensors. {} != {}", inputParams->inputTensorsNr, numIn);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    if (inputParams->outputTensorsNr != numOut)
    {
        LOG_INFO(SYN_API, "sif2_1 bad num of output tensors. {} != {}", inputParams->outputTensorsNr, numOut);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // calc average of the two inputs
    uint64_t avgSizes[tpc_lib_api::MAX_TENSOR_DIM] {};
    for (int i = 0; i < 2; i++)
    {
        uint64_t* sizes = inputParams->inputTensors[i]->geometry.maxSizes;
        for (int dim = 0; dim < SYN_MAX_TENSOR_DIM; dim++)
        {
            avgSizes[dim] += sizes[dim] / 2;
        }
        LOG_DSD_INFO("sif2_1 input {}: 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}",
                     i,
                     sizes[0],
                     sizes[1],
                     sizes[2],
                     sizes[3],
                     sizes[4]);
    }

    // Verify shape tensor
    for (int dim = 0; dim < SYN_MAX_TENSOR_DIM; dim++)
    {
        uint32_t expected = DsdRecipe::SET_SHAPE_SIZE;
        if (inputParams->inputTensors[2]->geometry.maxSizes[dim] != expected)
        {
            LOG_DSD_ERR("Bad Shape Tensor size dim {} size 0x{:x} expected 0x{:x}",
                        dim,
                        inputParams->inputTensors[2]->geometry.maxSizes[dim],
                        expected);
            return tpc_lib_api::GLUE_FAILED;
        }
    }

    for (int out = 0; out < numOut; out++)
    {
        memcpy(outputData->outputTensors[out]->geometry.maxSizes, avgSizes, sizeof(outputData->outputTensors[0]->geometry.maxSizes));
    }
    LOG_DSD_INFO("sif2_1 outputs 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}",
                 outputData->outputTensors[0]->geometry.maxSizes[0],
                 outputData->outputTensors[0]->geometry.maxSizes[1],
                 outputData->outputTensors[0]->geometry.maxSizes[2],
                 outputData->outputTensors[0]->geometry.maxSizes[3],
                 outputData->outputTensors[0]->geometry.maxSizes[4]);
    outputData->outputTensors[0]->geometry.dims = inputParams->inputTensors[0]->geometry.dims;

    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn sif2_1(tpc_lib_api::DeviceId deviceId,const tpc_lib_api::ShapeInferenceParams* inputParams,
                                         tpc_lib_api::ShapeInferenceOutput* outputData)
{
    return sif2_n(inputParams, outputData, DsdRecipe::NUM_NODE_OUTPUTS);
}

tpc_lib_api::GlueCodeReturn sif2_2(tpc_lib_api::DeviceId deviceId, const tpc_lib_api::ShapeInferenceParams* inputParams,
                                         tpc_lib_api::ShapeInferenceOutput* outputData)
{
    return sif2_n(inputParams, outputData, 2);
}

void smf0(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    assert(params->inputTensorsNr == DsdRecipe::NUM_NODE_INPUTS);
    assert(params->outputTensorsNr == DsdRecipe::NUM_NODE_OUTPUTS);

    outputs->outputShouldBypass = (uint32_t) false;

    outputs->outPatchValuesNr = params->inPatchValuesNr;
    for (uint i = 0; i < outputs->outPatchValuesNr; i++)
    {
        outputs->outputPatchValues[i] = (i + 1) * 256;
    }
}
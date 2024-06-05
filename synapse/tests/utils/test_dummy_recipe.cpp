#include "test_dummy_recipe.hpp"
#include "infra/habana_global_conf_common.h"
#include "define_synapse_common.hpp"
#include "recipe_allocator.h"
#include "runtime/scal/common/recipe_static_info_scal.hpp"
#include "scal_internal/pkt_macros.hpp"
#include <utils.h>

uint64_t TestDummyRecipe::m_s_runningId = 0;

/*
 ***************************************************************************************************
 *   @brief TestDummyRecipe() is used for testing. It builds a recipe with needed buffers and
 *                        fills them with running values to help debug if needed
 *
 ***************************************************************************************************
 */
TestDummyRecipe::TestDummyRecipe(DummyRecipeType recipeType,
                                 uint64_t        patchingSize,
                                 uint64_t        execBlobsSize,
                                 uint64_t        dynamicSize,
                                 uint64_t        programDataSize,
                                 uint64_t        ecbListSize,
                                 size_t          numberOfSobPps,
                                 synDeviceType   deviceType)
: m_isDsd(false), m_isIH2DRecipe(false)
{
    switch (recipeType)
    {
        case RECIPE_TYPE_NORMAL:
            m_isDsd        = false;
            m_isIH2DRecipe = false;
            break;
        case RECIPE_TYPE_DSD:
            m_isDsd        = true;
            m_isIH2DRecipe = false;
            break;
        case RECIPE_TYPE_DSD_AND_IH2D:
            m_isDsd        = true;
            m_isIH2DRecipe = true;
            break;
        default:
            LOG_ERR_T(SYN_RECIPE, "Unsupported type of recipe requested {}", (unsigned)deviceType);
            break;
    }

    uint64_t ecbListSize64 = ecbListSize / sizeof(ecbListSize64);

    m_s_runningId++;
    m_internalRecipeHandle.recipeSeqNum = m_s_runningId;
    m_seqId                             = m_s_runningId;

    LOG_INFO(SYN_API,
             "new dummyRecipe {:x}. Sizes patchingSize {:x} execBlobsSize {:x} dynamicSize {:x} programDataSize {:x} "
             "ecbListSize {:x}",
             m_seqId,
             patchingSize,
             execBlobsSize,
             dynamicSize,
             programDataSize,
             ecbListSize);

    memset(&m_recipe, 0, sizeof(m_recipe));
    // memset(&m_internalRecipeHandle, 0, sizeof(m_internalRecipeHandle));
    m_internalRecipeHandle.basicRecipeHandle.recipe            = &m_recipe;
    m_internalRecipeHandle.basicRecipeHandle.recipeAllocator   = new RecipeAllocator();
    m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe = nullptr;

    // node_nr should be > 0, if 0, launch returns success without doing anything
    m_internalRecipeHandle.basicRecipeHandle.recipe->node_nr = 1;

    m_recipe.patching_blobs_buffer_size = patchingSize;
    createBuff(&(m_recipe.patching_blobs_buffer), patchingSize, PATCHABLE + 1);

    m_recipe.program_data_blobs_size = programDataSize;
    createBuff((uint64_t**)&(m_recipe.program_data_blobs_buffer), programDataSize, PROGRAM_DATA + 1);

    m_recipe.execution_blobs_buffer_size = execBlobsSize;
    createBuff(&(m_recipe.execution_blobs_buffer), execBlobsSize, NON_PATCHABLE + 1);

    m_recipe.dynamic_blobs_buffer_size = dynamicSize;
    createBuff(&(m_recipe.dynamic_blobs_buffer), dynamicSize, DYNAMIC + 1);

    std::vector<Recipe::EngineType> arcsTemp {Recipe::TPC, Recipe::MME};
    if (deviceType == synDeviceGaudi2)
    {
        arcsTemp.push_back(Recipe::DMA);  // gaudi3 doesn't support DMA yet
        arcsTemp.push_back(Recipe::ROT);  // gaudi3 doesn't support ROT yet
    }

    m_recipe.arc_jobs_nr = arcsTemp.size();  // rotator is not included at the moment
    m_recipe.arc_jobs    = new arc_job_t[m_recipe.arc_jobs_nr];

    for (int i = 0; i < m_recipe.arc_jobs_nr; i++)
    {
        m_recipe.arc_jobs[i].logical_engine_id = arcsTemp[i];
        m_recipe.arc_jobs[i].engines_filter    = 0;

        SingleBuff64 bufferStatic(ecbListSize64);
        m_recipe.arc_jobs[i].static_ecb.cmds_size       = bufferStatic.size() * sizeof(bufferStatic[0]);
        m_recipe.arc_jobs[i].static_ecb.cmds_eng_offset = 0;
        fillBuff(bufferStatic, (ECB_LIST_FIRST + 1) + i * 2);
        m_recipe.arc_jobs[i].static_ecb.cmds = (uint8_t*)bufferStatic.data();
        m_ecbBuffers.push_back(std::move(bufferStatic));

        SingleBuff64 bufferDynamic(ecbListSize64);
        m_recipe.arc_jobs[i].dynamic_ecb.cmds_size       = bufferDynamic.size() * sizeof(bufferDynamic[0]);
        m_recipe.arc_jobs[i].dynamic_ecb.cmds_eng_offset = 0;
        fillBuff(bufferDynamic, (ECB_LIST_FIRST + 1) + i * 2 + 1);
        m_recipe.arc_jobs[i].dynamic_ecb.cmds = (uint8_t*)bufferDynamic.data();
        m_ecbBuffers.push_back(std::move(bufferDynamic));
    }

    // Tensors
    m_tensors.resize(2);

    m_recipe.permute_tensors_views_nr = 0;
    m_recipe.permute_tensors_views    = nullptr;

    m_recipe.persist_tensors_nr = m_tensors.size();
    m_recipe.tensors            = m_tensors.data();

    m_recipe.tensors[0].section_idx            = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    m_recipe.tensors[0].name                   = "aaa";
    m_recipe.tensors[0].size                   = 0x10;
    m_recipe.tensors[0].multi_views_indices_nr = 0;

    m_recipe.tensors[1].section_idx            = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;
    m_recipe.tensors[1].name                   = "bbb";
    m_recipe.tensors[1].size                   = 0x10;
    m_recipe.tensors[1].multi_views_indices_nr = 0;

    m_recipe.sections_nr = 4;

    // Patch points + Blobs
    const uint32_t numPpBlobs = patchingSize / BLOB_SIZE;

    m_recipe.patch_points_nr = numPpBlobs;
    m_patchPoints.resize(numPpBlobs);
    m_recipe.patch_points = m_patchPoints.data();

    m_recipe.blobs_nr = numPpBlobs;
    m_recipe.blobs    = new blob_t[numPpBlobs];

    uint64_t offset = 0;
    for (int i = 0; i < numberOfSobPps; i++)
    {
        m_recipe.blobs[i].size = BLOB_SIZE;
        m_recipe.blobs[i].data = (uint8_t*)m_recipe.patching_blobs_buffer + offset;
        offset += BLOB_SIZE;
        m_recipe.blobs[i].blob_type_all = blob_t::PATCHING;

        m_recipe.patch_points[i].type                            = patch_point_t::SOB_PATCH_POINT;
        m_recipe.patch_points[i].blob_idx                        = i;
        m_recipe.patch_points[i].dw_offset_in_blob               = 0;  // last 4 bytes in blob
        m_recipe.patch_points[i].sob_patch_point.tensor_db_index = 0;
    }
    for (int i = numberOfSobPps; i < numPpBlobs; i++)
    {
        m_recipe.blobs[i].size          = BLOB_SIZE;
        m_recipe.blobs[i].data          = (uint8_t*)m_recipe.patching_blobs_buffer + offset;
        m_recipe.blobs[i].blob_type_all = blob_t::PATCHING;
        offset += BLOB_SIZE;

        m_recipe.patch_points[i].type              = patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT;
        m_recipe.patch_points[i].blob_idx          = i;
        m_recipe.patch_points[i].dw_offset_in_blob = (BLOB_SIZE - 8) / sizeof(uint32_t);  // last 8 bytes in blob
        m_recipe.patch_points[i].memory_patch_point.effective_address = 0;

        m_recipe.patch_points[i].memory_patch_point.section_idx = 1;
    }

    // Ws Size
    m_wsSizes.resize(MEMORY_ID_RESERVED_FOR_PROGRAM + 1);
    m_wsSizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]    = wsScratchpadSize;
    m_wsSizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = wsPrgDataSize;
    m_recipe.workspace_nr                          = m_wsSizes.size();
    m_recipe.workspace_sizes                       = m_wsSizes.data();

    // conf_params
    m_gc_conf.resize(2);
    m_gc_conf[0].conf_id    = gc_conf_t::DEVICE_TYPE;
    m_gc_conf[0].conf_value = deviceType;

    m_gc_conf[1].conf_id    = gc_conf_t::TPC_ENGINE_MASK;
    m_gc_conf[1].conf_value = GCFG_TPC_ENGINES_ENABLED_MASK.value();

    m_recipe.recipe_conf_nr     = m_gc_conf.size();
    m_recipe.recipe_conf_params = m_gc_conf.data();

    if (isDsd())
    {
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe                = new shape_plane_graph_t();
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->version_major = 1;
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->version_minor = 1;

        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_node_nr = 0;
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_nodes   = nullptr;

        m_dsdTensors.resize(m_tensors.size());
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors_nr = m_dsdTensors.size();
        ;
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors = m_dsdTensors.data();

        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors[0].tensor_info_name = "aaa";
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors[0].tensor_type =
            tensor_info_t::PERSISTENT_TENSOR;
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors[0].tensor_db_index = 0;

        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors[1].tensor_info_name = "bbb";
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors[1].tensor_type =
            tensor_info_t::PERSISTENT_TENSOR;
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->sp_tensors[0].tensor_db_index = 1;

        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->shape_tensors_list_nr = 0;
        m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe->shape_tensors         = nullptr;
    }

    if (isIH2DRecipe())
    {
        m_recipe.h2di_tensors_nr = 1;
    }

    delete m_internalRecipeHandle.basicRecipeHandle.recipeAllocator;
}

void TestDummyRecipe::createSingleValidEcbList(ecb_t list)
{
    if (list.cmds_size == 0) return;

    assert(list.cmds_size % 0x80 == 0);

    const uint32_t ecbSizePacketSize = EngEcbSizePkt<g2fw>::getSize();
    const uint32_t nopPacketSize     = EngEcbNopPkt<g2fw>::getSize();

    uint8_t* buff = list.cmds;

    EngEcbSizePkt<G2Packets>::build(buff, true, 0, true, list.cmds_size);
    buff += EngEcbSizePkt<G2Packets>::getSize();

    EngEcbNopPkt<G2Packets>::build(buff, false, 0, false, (0x80 - ecbSizePacketSize - nopPacketSize) / 4);

    uint32_t end = list.cmds_size / 0x80;
    for (int i = 1; i < end; i++)
    {
        buff          = list.cmds + 0x80 * i;
        bool switchCq = (i == (end - 1)) ? true : false;

        EngEcbNopPkt<G2Packets>::build(buff, false, 0, switchCq, (0x80 - nopPacketSize) / 4);
    }
}

void TestDummyRecipe::createValidEcbLists()
{
    createValidEcbLists(&m_recipe);
}

void TestDummyRecipe::createValidEcbLists(recipe_t* recipe)
{
    // for every ecb list, create a size packet and then nop packets with skip to fill it to 0x80
    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        createSingleValidEcbList(recipe->arc_jobs[i].dynamic_ecb);
        createSingleValidEcbList(recipe->arc_jobs[i].static_ecb);
    }
}

TestDummyRecipe::~TestDummyRecipe()
{
    LOG_INFO(SYN_API, "delete dummyRecipe {:x}", m_seqId);

    delete[] m_recipe.execution_blobs_buffer;
    delete[] m_recipe.patching_blobs_buffer;
    delete[] m_recipe.dynamic_blobs_buffer;
    delete[] m_recipe.program_data_blobs_buffer;
    delete[] m_recipe.blobs;

    delete[] m_recipe.arc_jobs;

    if (isDsd())
    {
        delete m_internalRecipeHandle.basicRecipeHandle.shape_plan_recipe;
    }
}

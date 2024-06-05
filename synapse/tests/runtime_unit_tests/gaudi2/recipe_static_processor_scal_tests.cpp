#include "test_dummy_recipe.hpp"
#include "synapse_test.hpp"
#include "runtime/scal/common/recipe_launcher/mapped_mem_mgr.hpp"

#include "runtime/scal/common/recipe_static_info_scal.hpp"
#include "runtime/scal/common/recipe_static_processor_scal.hpp"

#include "gaudi2_arc_eng_packets.h"

TEST(UTDeviceAgnosticRecipeStaticProcessorScal, checkProcess)
{
    TestDummyRecipe dummyRecipe(RECIPE_TYPE_NORMAL, 0x1003, 0x1002, 0x3004, 0x2005, 0x800);

    const recipe_t* recipe = dummyRecipe.getRecipe();

    RecipeStaticInfoScal recipeStaticInfoScal;
    const synStatus      status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                                  *dummyRecipe.getBasicRecipeInfo(),
                                                                                   recipeStaticInfoScal);
    uint64_t             dcSize = MappedMemMgr::getDcSize();

    ASSERT_EQ(status, synSuccess);

    const uint32_t programAlignment = DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;

    // Global HBM patch - on his own DC, starts from 0
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].size, 0x1003);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].offsetHbm, 0x2100 + 0x1100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].recipeAddr, (uint8_t*)recipe->patching_blobs_buffer);

    // Global HBM non-patch - assumes DC is aligned to 0x2000 (1<<13)
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetMapped, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].size, 0x2005);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetHbm, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].recipeAddr, (uint8_t*)recipe->program_data_blobs_buffer);

    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetMapped, 0x2100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].size, 0x1002);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetHbm, 0x2100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].recipeAddr, (uint8_t*)recipe->execution_blobs_buffer);

    // Arc HBM
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].offsetMapped, dcSize);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].size, 0x3004);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].offsetHbm, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].align, dcSize);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].recipeAddr, (uint8_t*)recipe->dynamic_blobs_buffer);

    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        int idx = ECB_LIST_FIRST + i * 2;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].offsetMapped, dcSize + 0x3100 + (i * 2) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].size, 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].offsetHbm, 0x3100 + (i * 2) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].align, programAlignment) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].recipeAddr, (uint8_t*)recipe->arc_jobs[i].dynamic_ecb.cmds) << "for idx " << idx;

        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].offsetMapped, dcSize + 0x3100 + (i * 2 + 1) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].size, 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].offsetHbm, 0x3100 + (i * 2 + 1) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].align, programAlignment) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].recipeAddr,
                  (uint8_t*)recipe->arc_jobs[i].static_ecb.cmds) << "for idx " << idx;
    }

    const uint64_t mappedNoPatch = 0x2005 /*prgData*/ +
                                   0xFB  /* align to 2100 */ + 0x1002 /* exec */ +
                                   0x3CEFE /* align to 40000==dcSize */ + 0x3004 /* dyn */  +
                                   0xFC + (recipe->arc_jobs_nr * 2 - 1) * 0x800 + 0x800;

    ASSERT_EQ(recipeStaticInfoScal.m_mappedSizeNoPatch, mappedNoPatch);
    ASSERT_EQ(recipeStaticInfoScal.m_glbHbmSizeTotal, 0x2100 + 0x1100 + 0x1003);
    ASSERT_EQ(recipeStaticInfoScal.m_arcHbmSize, 0x3100 + (recipe->arc_jobs_nr * 2 - 1) * 0x800 + 0x800);
}

TEST(UTDeviceAgnosticRecipeStaticProcessorScal, checkDsdProcess)
{
    TestDummyRecipe dummyRecipe(RECIPE_TYPE_DSD, 0x1003, 0x1002, 0x3004, 0x2005, 0x800, 0);

    const recipe_t* recipe = dummyRecipe.getRecipe();

    RecipeStaticInfoScal recipeStaticInfoScal;
    const synStatus      status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                                   *dummyRecipe.getBasicRecipeInfo(),
                                                                                   recipeStaticInfoScal);
    uint64_t             dcSize = MappedMemMgr::getDcSize();

    ASSERT_EQ(status, synSuccess);

    const uint32_t programAlignment = DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;

    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].size, 0x1003);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].offsetHbm, 0x2100 + 0x1100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].recipeAddr, (uint8_t*)recipe->patching_blobs_buffer);

    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetMapped, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].size, 0x2005);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetHbm, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].recipeAddr, (uint8_t*)recipe->program_data_blobs_buffer);

    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetMapped, 0x2100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].size, 0x1002);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetHbm, 0x2100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].recipeAddr, (uint8_t*)recipe->execution_blobs_buffer);

    // Arc HBM
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].offsetMapped, dcSize); // patchable 1003 + align to 0x4000
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].size, 0x3004);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].offsetHbm, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].align, dcSize);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].recipeAddr, (uint8_t*)recipe->dynamic_blobs_buffer);

    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        int idx = ECB_LIST_FIRST + i * 2;
        // offset in mapped: prgData + non-patchable = 0x2005, 1002 -> 2100 + 1100=0x3200
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].offsetMapped, 0x3200 + (i * 2) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].size, 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].offsetHbm, 0x3100 + (i * 2) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].align, programAlignment) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].recipeAddr, (uint8_t*)recipe->arc_jobs[i].dynamic_ecb.cmds) << "for idx " << idx;

        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].offsetMapped, 0x3200 + (i * 2 + 1) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].size, 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].offsetHbm, 0x3100 + (i * 2 + 1) * 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].align, programAlignment) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].recipeAddr,
                  (uint8_t*)recipe->arc_jobs[i].static_ecb.cmds) << "for idx " << idx;
    }

    ASSERT_EQ(recipeStaticInfoScal.m_mappedSizeNoPatch,
              0x2005 /*prgData*/ +
              0xFB  /* align to 2100 */ + 0x1002 /* exec */ +
              0xFE /* align to 3200 */ + (recipe->arc_jobs_nr * 2 - 1) * 0x800 + 0x800);

    ASSERT_EQ(recipeStaticInfoScal.m_glbHbmSizeTotal, 0x2100 + 0x1100 + 0x1003);
    ASSERT_EQ(recipeStaticInfoScal.m_arcHbmSize, 0x3100 + (recipe->arc_jobs_nr * 2 - 1) * 0x800 + 0x800);
}

TEST(UTDeviceAgnosticRecipeStaticProcessorScal, checkIH2DProcess)
{
    TestDummyRecipe dummyRecipe(RECIPE_TYPE_DSD_AND_IH2D, 0x1003, 0x1002, 0x3004, 0x2005, 0x800, 0);

    const recipe_t* recipe = dummyRecipe.getRecipe();

    RecipeStaticInfoScal recipeStaticInfoScal;
    const synStatus      status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                                   *dummyRecipe.getBasicRecipeInfo(),
                                                                                   recipeStaticInfoScal);
    uint64_t             dcSize = MappedMemMgr::getDcSize();

    ASSERT_EQ(status, synSuccess);

    const uint32_t programAlignment = DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;

    // Global HBM patch - on his own DC, starts from 0
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].offsetMapped, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].size, 0x1003);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].offsetHbm, 0x2100 + 0x1100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PATCHABLE].recipeAddr, (uint8_t*)recipe->patching_blobs_buffer);

    // patchable + dynamic -> 1100 + align (40000==data chunk) = 40000 + 3004 (dynamic) + align (100) -> 0x43100
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetMapped, 0x43100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].size, 0x2005);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].offsetHbm, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[PROGRAM_DATA].recipeAddr, (uint8_t*)recipe->program_data_blobs_buffer);

    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetMapped, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].size, 0x1002);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].offsetHbm, 0x2100);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].align, programAlignment);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[NON_PATCHABLE].recipeAddr, (uint8_t*)recipe->execution_blobs_buffer);

    // Arc HBM
    // patchable (1003) + align (4000) -> 4000
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].offsetMapped, dcSize);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].size, 0x3004);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].offsetHbm, 0);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].align, dcSize);
    ASSERT_EQ(recipeStaticInfoScal.recipeSections[DYNAMIC].recipeAddr, (uint8_t*)recipe->dynamic_blobs_buffer);

    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        int idx = ECB_LIST_FIRST + i * 2;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].offsetMapped, 0x1100 + (i * 2) * 0x800);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].size, 0x800) << "for idx " << idx;
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].offsetHbm, 0x3100 + (i * 2) * 0x800);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].align, programAlignment);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx].recipeAddr, (uint8_t*)recipe->arc_jobs[i].dynamic_ecb.cmds);

        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].offsetMapped, 0x1100 + (i * 2 + 1) * 0x800);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].size, 0x800);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].offsetHbm, 0x3100 + (i * 2 + 1) * 0x800);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].align, programAlignment);
        ASSERT_EQ(recipeStaticInfoScal.recipeSections[idx + 1].recipeAddr,
                  (uint8_t*)recipe->arc_jobs[i].static_ecb.cmds);
    }

    ASSERT_EQ(recipeStaticInfoScal.m_mappedSizeNoPatch,
              0x1100 + (recipe->arc_jobs_nr * 2 - 1) * 0x800 + 0x800);

    // patchable (0x1003) + align(40000) -> 40000 + dynamic (3004) = 43004 + align (100) -> 43100 + prgData (2005) -> 45105
    ASSERT_EQ(recipeStaticInfoScal.m_mappedSizePatch, 0x45105);
    ASSERT_EQ(recipeStaticInfoScal.m_glbHbmSizeTotal, 0x2100 + 0x1100 + 0x1003);
    ASSERT_EQ(recipeStaticInfoScal.m_arcHbmSize, 0x3100 + (recipe->arc_jobs_nr * 2 - 1) * 0x800 + 0x800);
}
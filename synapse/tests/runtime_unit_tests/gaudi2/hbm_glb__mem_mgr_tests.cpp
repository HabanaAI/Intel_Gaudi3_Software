#include "test_dummy_recipe.hpp"
#include "mapped_mem_mgr_tests_utils.hpp"
#include "synapse_test.hpp"

#include "runtime/scal/common/recipe_launcher/hbm_global_mem_mgr.hpp"
#include "runtime/scal/common/recipe_launcher/mem_mgrs_types.hpp"

#include "runtime/scal/common/recipe_static_processor_scal.hpp"

class UTGaudi2HbmGlbMemMgrTest : public ::testing::Test
{
};

TEST_F(UTGaudi2HbmGlbMemMgrTest, hbmGlbMemMgr_basic_test)
{
    HbmGlblMemMgr gmm;

    constexpr uint16_t CHUNKS = NBuffAllocator::NUM_BUFF;

    uint64_t                   memSize = 16 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> addrDev(new uint8_t[memSize + 127]());
    uint8_t*                   addrDevAlign = round_to_multiple(addrDev.get(), 128);

    gmm.init((uint64_t)addrDevAlign, memSize);

    std::deque<RecipeAndSections> recipes;

    for (int i = 0; i < 100; i++)
    {
        if (recipes.size() == CHUNKS)
        {
            auto x = recipes.front();

            bool res =
                MappedMemMgrTestUtils::testingCheckGlbSections(x.pRecipeStaticInfoScal->recipeSections, *x.pSections);
            ASSERT_EQ(res, true);

            delete x.pDummyRecipe;
            delete x.pRecipeStaticInfoScal;
            delete x.pSections;

            recipes.pop_front();
        }

        TestDummyRecipe*          dummyRecipe           = new TestDummyRecipe;
        RecipeStaticInfoScal* pRecipeStaticInfoScal = new RecipeStaticInfoScal();
        const synStatus       status                = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                                  *dummyRecipe->getBasicRecipeInfo(),
                                                                                  *pRecipeStaticInfoScal);
        ASSERT_EQ(status, synSuccess);
        MemorySectionsScal* pSections = new MemorySectionsScal();

        recipes.push_back({dummyRecipe, pRecipeStaticInfoScal, pSections});
        auto longSo = gmm.getAddr(pRecipeStaticInfoScal->m_glbHbmSizeTotal, pSections->m_glbHbmAddr);
        UNUSED(longSo);
        gmm.setLongSo(i + 1);

        MappedMemMgrTestUtils::testingCopyToGlb(pRecipeStaticInfoScal->recipeSections, *pSections);
    }

    while (!recipes.empty())
    {
        auto& x = recipes.front();

        bool res =
            MappedMemMgrTestUtils::testingCheckGlbSections(x.pRecipeStaticInfoScal->recipeSections, *x.pSections);
        ASSERT_EQ(res, true);

        delete x.pDummyRecipe;
        delete x.pRecipeStaticInfoScal;
        delete x.pSections;

        recipes.pop_front();
    }

    // check with throw an exception for recipe that is too big
    LOG_INFO(SYN_API, "-----hbmGlbMemMgr_basic_test: recipe too big-----");
    TestDummyRecipe          tempDummyRecipe(RECIPE_TYPE_NORMAL, 0, 0x1000000000);
    RecipeStaticInfoScal recipeStaticInfoScal;
    const synStatus      status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                              *tempDummyRecipe.getBasicRecipeInfo(),
                                                                              recipeStaticInfoScal);
    ASSERT_EQ(status, synSuccess);

    MemorySectionsScal sections;
    uint64_t           longSo;
    UNUSED(longSo);
    EXPECT_ANY_THROW(longSo = gmm.getAddr(recipeStaticInfoScal.m_glbHbmSizeTotal, sections.m_glbHbmAddr)) <<
       "Trying to allocate size " << recipeStaticInfoScal.m_glbHbmSizeTotal << "should throw";
}

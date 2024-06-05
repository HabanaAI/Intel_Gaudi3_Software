#include "synapse_test.hpp"
#include "runtime/scal/common/recipe_launcher/mem_mgrs_types.hpp"
#include "test_dummy_recipe.hpp"
#include "runtime/scal/common/recipe_launcher/arc_hbm_mem_mgr.hpp"
#include "mapped_mem_mgr_tests_utils.hpp"
#include "runtime/scal/common/recipe_static_processor_scal.hpp"

class UTGaudi2ArcHbmMemMgrTest : public ::testing::Test
{
};

TEST_F(UTGaudi2ArcHbmMemMgrTest, arcHbmMemMgr_basic_test)
{
    ArcHbmMemMgr amm;

    constexpr uint16_t CHUNKS = NBuffAllocator::NUM_BUFF;

    uint64_t                   memSize = 16 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> addrDev(new uint8_t[memSize + 127]());
    uint8_t*                   addrDevAlign = round_to_multiple(addrDev.get(), 128);

    const uint64_t addrCore       = 0x10000;
    const uint64_t dev2coreOffset = (uint64_t)addrDevAlign - addrCore;

    amm.init(addrCore, (uint64_t)addrDevAlign, memSize);

    std::deque<RecipeAndSections> recipes;

    for (int i = 0; i < 100; i++)
    {
        if (recipes.size() == CHUNKS)
        {
            auto x = recipes.front();

            bool res = MappedMemMgrTestUtils::testingCheckArcSections(x.pRecipeStaticInfoScal->recipeSections,
                                                                      dev2coreOffset,
                                                                      *x.pSections);
            ASSERT_EQ(res, true);
            delete x.pSections;
            delete x.pRecipeStaticInfoScal;
            delete x.pDummyRecipe;
            recipes.pop_front();
        }

        TestDummyRecipe*     dummyRecipe     = new TestDummyRecipe;
        basicRecipeInfo* basicRecipeInfo = dummyRecipe->getBasicRecipeInfo();

        RecipeStaticInfoScal* pRecipeStaticInfoScal = new RecipeStaticInfoScal();
        const synStatus       status =
            DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2, *basicRecipeInfo, *pRecipeStaticInfoScal);
        ASSERT_EQ(status, synSuccess);

        MemorySectionsScal* pSections = new MemorySectionsScal();

        recipes.push_back({dummyRecipe, pRecipeStaticInfoScal, pSections});
        auto rtn = amm.getAddr(pRecipeStaticInfoScal->m_arcHbmSize,
                    pSections->m_arcHbmAddr,
                    pSections->m_arcHbmCoreAddr);
        UNUSED(rtn);
        amm.setLongSo(i + 1);

        MappedMemMgrTestUtils::testingCopyToArc(pRecipeStaticInfoScal->recipeSections, *pSections);
    }

    while (!recipes.empty())
    {
        auto& x = recipes.front();

        bool res = MappedMemMgrTestUtils::testingCheckArcSections(x.pRecipeStaticInfoScal->recipeSections,
                                                                  dev2coreOffset,
                                                                  *x.pSections);
        ASSERT_EQ(res, true);

        delete x.pSections;
        delete x.pRecipeStaticInfoScal;
        delete x.pDummyRecipe;

        recipes.pop_front();
    }

    // check with throw an exception for recipe that is too big
    TestDummyRecipe          tempDummyRecipe(RECIPE_TYPE_NORMAL, 0x2000, 0x1000, 0x30000000);
    RecipeStaticInfoScal recipeStaticInfoScal;
    const synStatus      status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                              *tempDummyRecipe.getBasicRecipeInfo(),
                                                                              recipeStaticInfoScal);
    ASSERT_EQ(status, synSuccess);

    MemorySectionsScal sections;
    uint64_t           longSo;
    UNUSED(longSo);
    EXPECT_ANY_THROW(longSo = amm.getAddr(recipeStaticInfoScal.m_arcHbmSize,
                                 sections.m_arcHbmAddr,
                                 sections.m_arcHbmCoreAddr)) <<
                                 "Trying to allocate size " << recipeStaticInfoScal.m_arcHbmSize << " should throw";
}

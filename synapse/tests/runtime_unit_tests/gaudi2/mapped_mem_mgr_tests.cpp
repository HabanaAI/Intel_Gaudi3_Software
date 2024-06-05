#include "global_statistics.hpp"
#include "test_dummy_recipe.hpp"
#include "habana_global_conf_runtime.h"
#include "host_buffers_mapper.hpp"
#include "mapped_mem_mgr_tests_utils.hpp"
#include "scoped_configuration_change.h"
#include "synapse_test.hpp"

#include "runtime/scal/common/recipe_launcher/mapped_mem_mgr.hpp"
#include "runtime/scal/common/recipe_launcher/mapped_memory_sections_utils.hpp"

#include "runtime/scal/common/recipe_static_processor_scal.hpp"

#include "gaudi2_arc_eng_packets.h"

class UTGaudi2MappedMemMgrTest : public ::testing::TestWithParam<DummyRecipeType>
{
};

class DummyDevAlloc : public DevMemoryAllocInterface
{
public:
    DummyDevAlloc(uint64_t offset) : m_offset(offset), m_hostAddrToVirtualAddrMapper(name) {}

    virtual synStatus allocate() override
    {
        assert(0);
        return synFail;
    }
    virtual synStatus release() override
    {
        assert(0);
        return synFail;
    }

    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) override
    {
        uint8_t* buff = new uint8_t[size];

        assert(flags == synMemFlags::synMemHost);

        *buffer   = buff;
        *deviceVA = (uint64_t)buff - m_offset;

        return synSuccess;
    }

    virtual synStatus deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest) override
    {
        delete[](uint8_t*) pBuffer;
        return synSuccess;
    }

    virtual eMappingStatus getDeviceVirtualAddress(bool      isUserRequest,
                                                   void*     hostAddress,
                                                   uint64_t  bufferSize,
                                                   uint64_t* pDeviceVA,
                                                   bool*     pIsExactKeyFound = nullptr) override
    {
        assert(0);
        return eMappingStatus::HATVA_MAPPING_STATUS_FAILURE;
    }

    virtual synStatus mapBufferToDevice(uint64_t           size,
                                        void*              buffer,
                                        bool               isUserRequest,
                                        uint64_t           reqVAAddress,
                                        const std::string& mappingDesc) override
    {
        assert(0);
        return synFail;
    };

    virtual synStatus unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize) override
    {
        assert(0);
        return synFail;
    }
    virtual synStatus getDramMemInfo(uint64_t& free, uint64_t& total) const override
    {
        assert(0);
        return synFail;
    }
    virtual void getValidAddressesRange(uint64_t& lowestValidAddress, uint64_t& highestValidAddress) const override
    {
        assert(0);
    }
    virtual void dfaLogMappedMem() const override { assert(0); }

    virtual synStatus destroyHostAllocations(bool isUserAllocations) override
    {
        assert(0);
        return synFail;
    }

private:
    uint64_t m_offset;
    const std::string           name {"xxx"};
    HostAddrToVirtualAddrMapper m_hostAddrToVirtualAddrMapper;
};

// Create 10 recipes, allocate in loop until busy, release oldest and allocate again
TEST_P(UTGaudi2MappedMemMgrTest, mappedMemMgr_basic_test)
{
    synInitialize();  // for stats

    LOG_INFO(SYN_API, "-----mappedMemMgr_basic_test-----");
    bool isDsd = GetParam() != RECIPE_TYPE_NORMAL;
    bool isIH2DRecipe = GetParam() == RECIPE_TYPE_DSD_AND_IH2D;

    DummyDevAlloc dummyDevAlloc(0);
    MappedMemMgr  mmm("mappedMemMgr_basic_test", dummyDevAlloc);

    // 16 DC - each recipe uses 2, so 8 recipes can fit.
    ScopedConfigurationChange arcSupportMode("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange dataChunksAmount("STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP", "16");

    mmm.init();

    std::vector<RecipeAndSections> recipes;

    // Create 10 recipes
    int const NUM_RECIPES = 10;
    for (int i = 0; i < NUM_RECIPES; i++)
    {
        TestDummyRecipe*      pDummyRecipe          = new TestDummyRecipe(GetParam());
        RecipeStaticInfoScal* pRecipeStaticInfoScal = new RecipeStaticInfoScal();
        DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                         *pDummyRecipe->getBasicRecipeInfo(),
                                                         *pRecipeStaticInfoScal);
        MemorySectionsScal* pSections = new MemorySectionsScal();
        if (isIH2DRecipe)
        {
            uint64_t recipeProgramDataSize = pDummyRecipe->getRecipe()->program_data_blobs_size;
            uint8_t* pRecipeProgramData = (uint8_t*)(pDummyRecipe->getRecipe()->program_data_blobs_buffer);

            pSections->m_ih2dBuffer = std::make_unique<uint8_t []>(recipeProgramDataSize);
            memcpy(pSections->m_ih2dBuffer.get(), pRecipeProgramData, recipeProgramDataSize);
        }

        recipes.push_back({pDummyRecipe, pRecipeStaticInfoScal, pSections});
    }

    int  lastUnused = 0;
    bool hadBusy    = false;  // This is to make sure we get to the busy case. If not, change the NUM_RECIPES and/or
                              // the size of each recipe (see DummyRecipe constructor)

    for (int i = 0; i < NUM_RECIPES; i++)
    {
        LOG_INFO(SYN_API, "----loop {}-----", i);
        RecipeSeqId           recipeSeqId(recipes[i].pDummyRecipe->getRecipeSeqId());
        RecipeStaticInfoScal* pRecipeStaticInfoScal = recipes[i].pRecipeStaticInfoScal;
        MemorySectionsScal*   pSections             = recipes[i].pSections;

        // try to get memory for recipe i
        mmm.getAddrForId(*pRecipeStaticInfoScal, {recipeSeqId, i}, *pSections);

        bool busy = pSections->anyBusyInMapped();

        if (busy)
        {
            hadBusy = true;
            // if busy, unuse one recipe and try again
            int numInMapped = mmm.getNumRecipes();
            ASSERT_GT(numInMapped, 0) << " i is " << i;  // verify at least one recipe in mapped memory

            bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
                recipes[lastUnused].pRecipeStaticInfoScal->recipeSections,
                lastUnused,
                *recipes[lastUnused].pSections,
                isDsd,
                isIH2DRecipe);  // check data is still there
            ASSERT_EQ(cmp, true) << "Failed cmp for i " << i;

            recipeSeqId = RecipeSeqId {recipes[lastUnused].pDummyRecipe->getRecipeSeqId()};
            mmm.unuseId({recipeSeqId, lastUnused});
            lastUnused++;

            recipeSeqId.val = recipes[i].pDummyRecipe->getRecipeSeqId();
            mmm.getAddrForId(*pRecipeStaticInfoScal, {recipeSeqId, i}, *pSections);  // should pass now
            busy = pSections->anyBusyInMapped();
            ASSERT_EQ(busy, false);
        }

        MappedMemorySectionsUtils::memcpyToMapped(*pSections, pRecipeStaticInfoScal->recipeSections, isDsd, isIH2DRecipe);
        MappedMemMgrTestUtils::testingFillMappedPatchable(pRecipeStaticInfoScal->recipeSections, i, *pSections);
        if (isDsd)
        {
            MappedMemMgrTestUtils::testingFillMappedDsdPatchable(pRecipeStaticInfoScal->recipeSections, i, *pSections);
        }

        bool res = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(pRecipeStaticInfoScal->recipeSections,
                                                                                 i,
                                                                                 *pSections,
                                                                                 isDsd,
                                                                                 isIH2DRecipe);
        ASSERT_EQ(res, true);
    }
    ASSERT_EQ(hadBusy, true) << "Test is not checking busy case";

    // unuse the rest
    for (int i = lastUnused; i < NUM_RECIPES; i++)
    {
        // check data is still there
        bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
            recipes[i].pRecipeStaticInfoScal->recipeSections,
            i,
            *recipes[i].pSections,
            isDsd,
            isIH2DRecipe);
        ASSERT_EQ(cmp, true) << "Failed cmp for i " << i;
        RecipeSeqId recipeSeqId(recipes[i].pDummyRecipe->getRecipeSeqId());
        mmm.unuseId({recipeSeqId, i});
        mmm.removeId(recipeSeqId);
    }

    ASSERT_EQ(mmm.getNumRecipes(), 0) << "mmm should be empty";

    // check with throw an exception for recipe that is too big
    MemorySectionsScal*   pSections             = recipes[0].pSections;
    RecipeStaticInfoScal* pRecipeStaticInfoScal = recipes[0].pRecipeStaticInfoScal;
    pRecipeStaticInfoScal->m_mappedSizePatch    = MappedMemMgr::testingOnlyInitialMappedSize() + 1;

    LOG_INFO(SYN_API, "-----Trying to allocate too big {:x}", pRecipeStaticInfoScal->m_mappedSizePatch);
    RecipeSeqId recipeSeqId(1);
    mmm.getAddrForId(*pRecipeStaticInfoScal, {recipeSeqId, 0}, *pSections);
    bool busy = pSections->anyBusyInMapped();
    ASSERT_EQ(busy, true);

    for (auto& oneRecipe : recipes)
    {
        delete oneRecipe.pDummyRecipe;
        delete oneRecipe.pRecipeStaticInfoScal;
        delete oneRecipe.pSections;
    }

    ASSERT_EQ(mmm.getNumRecipes(), 0) << "mmm should be empty";

    synDestroy();
}

// create one recipe, allocate until busy, when busy, release the oldest one and allocate
TEST_P(UTGaudi2MappedMemMgrTest, mappedMemMgr_same_recipe_test)
{
    synInitialize();  // for stats
    DummyDevAlloc          dummyDevAlloc(0x1000);
    bool isDsd                   = GetParam() != RECIPE_TYPE_NORMAL;
    bool isIH2DRecipe            = GetParam() == RECIPE_TYPE_DSD_AND_IH2D;


    MappedMemMgr mmm("mappedMemMgr_same_recipe_test", dummyDevAlloc);

    ScopedConfigurationChange arcSupportMode("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange dataChunksAmount("STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP", "8");

    mmm.init();

    TestDummyRecipe          dummyRecipe(GetParam());
    RecipeStaticInfoScal recipeStaticInfoScal;
    const synStatus      status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                              *dummyRecipe.getBasicRecipeInfo(),
                                                                              recipeStaticInfoScal);
    ASSERT_EQ(status, synSuccess);

    int  lastUnused = 0;
    bool hadBusy    = false;  // This is to make sure we get to the busy case. If not, change the NUM_RECIPES and/or
    // the size of each recipe (see TestDummyRecipe constructor)

    const int                        LOOPS = 10;
    std::vector<MemorySectionsScal*> pSections(LOOPS);

    for (int i = 0; i < LOOPS; i++)
    {
        LOG_INFO(SYN_API, "----loop {}-----", i);

        pSections[i] = new MemorySectionsScal();

        RecipeSeqId recipeSeqId(dummyRecipe.getRecipeSeqId());

        if (isIH2DRecipe)
        {
            uint64_t recipeProgramDataSize = dummyRecipe.getRecipe()->program_data_blobs_size;
            uint8_t* pRecipeProgramData = (uint8_t*)(dummyRecipe.getRecipe()->program_data_blobs_buffer);

            pSections[i]->m_ih2dBuffer = std::make_unique<uint8_t []>(recipeProgramDataSize);
            memcpy(pSections[i]->m_ih2dBuffer.get(), pRecipeProgramData, recipeProgramDataSize);
        }

        mmm.getAddrForId(recipeStaticInfoScal, {recipeSeqId, i}, *pSections[i]);

        bool busy = pSections[i]->anyBusyInMapped();

        if (busy)
        {
            hadBusy = true;
            // if busy, unuse one recipe and try again
            int numInMapped = mmm.getNumRecipes();
            ASSERT_GT(numInMapped, 0);  // verify at least one recipe in mapped memory

            bool cmp =
                MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(recipeStaticInfoScal.recipeSections,
                                                                              lastUnused,
                                                                              *pSections[lastUnused],
                                                                              isDsd,
                                                                              isIH2DRecipe);  // check data is still there
            ASSERT_EQ(cmp, true) << "Failed cmp for i " << i;
            mmm.unuseId({recipeSeqId, lastUnused});
            lastUnused++;

            mmm.getAddrForId(recipeStaticInfoScal, {recipeSeqId, i}, *pSections[i]);  // should pass now
            busy = pSections[i]->anyBusyInMapped();
            ASSERT_EQ(busy, false);
        }

        MappedMemorySectionsUtils::memcpyToMapped(*pSections[i], recipeStaticInfoScal.recipeSections, isDsd, isIH2DRecipe);
        MappedMemMgrTestUtils::testingFillMappedPatchable(recipeStaticInfoScal.recipeSections, i, *pSections[i]);
        if (isDsd)
        {
            MappedMemMgrTestUtils::testingFillMappedDsdPatchable(recipeStaticInfoScal.recipeSections, i, *pSections[i]);
        }
    }
    ASSERT_EQ(hadBusy, true) << "Test is not checking busy case";

    RecipeSeqId recipeSeqId(dummyRecipe.getRecipeSeqId());

    for (int i = lastUnused; i < LOOPS; i++)
    {
        // check data is still there
        bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(recipeStaticInfoScal.recipeSections,
                                                                                 i,
                                                                                 *pSections[i],
                                                                                 isDsd,
                                                                                 isIH2DRecipe);
        ASSERT_EQ(cmp, true) << "Failed cmp for i " << i;
        mmm.unuseId({recipeSeqId, i});
    }

    mmm.removeId(recipeSeqId);

    for (int i = 0; i < LOOPS; i++)
    {
        delete pSections[i];
    }

    ASSERT_EQ(mmm.getNumRecipes(), 0) << "mmm should be empty";
    synDestroy();
}

// create one recipe, allocate until busy, when busy, release the oldest one and allocate
TEST_P(UTGaudi2MappedMemMgrTest, mappedMemMgr_two_recipes_test)
{
    synInitialize();  // for stats
    uint64_t               map2devOffset = 0x10000;
    DummyDevAlloc          dummyDevAlloc(map2devOffset);
    MappedMemMgr           mmm("mappedMemMgr_two_recipes_test", dummyDevAlloc);

    ScopedConfigurationChange arcSupportMode("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange dataChunksAmount("STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP", "8");

    mmm.init();

    std::vector<RecipeAndSections> recipes;

    // Create 10 recipes
    int const NUM_RECIPES           = 10;
    int const NUM_DIFFERENT_RECIPES = 2;

    // Fill the array below with one "special" recipe (normal/dsd/ih2d) and one non-dsd recipe
    TestDummyRecipe      dummyRecipeArr[NUM_DIFFERENT_RECIPES] = {TestDummyRecipe(GetParam())};
    RecipeStaticInfoScal recipeStaticInfoScalArr[NUM_DIFFERENT_RECIPES];

    synStatus status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                        *dummyRecipeArr[0].getBasicRecipeInfo(),
                                                                        recipeStaticInfoScalArr[0]);
    ASSERT_EQ(status, synSuccess);

    status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                              *dummyRecipeArr[1].getBasicRecipeInfo(),
                                                              recipeStaticInfoScalArr[1]);
    ASSERT_EQ(status, synSuccess);

    // create the 10 recipes (based on the dummyRecipeArr)
    for (int i = 0; i < NUM_RECIPES; i++)
    {
        TestDummyRecipe*      dummyRecipe           = &dummyRecipeArr[i % NUM_DIFFERENT_RECIPES];
        RecipeStaticInfoScal* pRecipeStaticInfoScal = &recipeStaticInfoScalArr[i % NUM_DIFFERENT_RECIPES];
        MemorySectionsScal*   pSections             = new MemorySectionsScal();

        if (dummyRecipe->isIH2DRecipe())
        {
            uint64_t recipeProgramDataSize = dummyRecipe->getRecipe()->program_data_blobs_size;
            uint8_t* pRecipeProgramData = (uint8_t*)(dummyRecipe->getRecipe()->program_data_blobs_buffer);

            pSections->m_ih2dBuffer = std::make_unique<uint8_t []>(recipeProgramDataSize);
            memcpy(pSections->m_ih2dBuffer.get(), pRecipeProgramData, recipeProgramDataSize);
        }

        recipes.push_back({dummyRecipe, pRecipeStaticInfoScal, pSections});
    }

    int  lastUnused = 0;
    bool hadBusy    = false;  // This is to make sure we get to the busy case. If not, change the NUM_RECIPES and/or
    // the size of each recipe (see TestDummyRecipe constructor)

    for (int i = 0; i < NUM_RECIPES; i++)
    {
        LOG_INFO(SYN_API, "----loop {}-----", i);

        RecipeSeqId           recipeSeqId(recipes[i].pDummyRecipe->getRecipeSeqId());
        RecipeStaticInfoScal* pRecipeStaticInfoScal = recipes[i].pRecipeStaticInfoScal;
        MemorySectionsScal*   pSections             = recipes[i].pSections;

        // try to get memory for recipe i
        mmm.getAddrForId(*pRecipeStaticInfoScal, {recipeSeqId, i}, *pSections);

        bool busy = pSections->anyBusyInMapped();

        if (busy)
        {
            hadBusy = true;
            // if busy, unuse one recipe and try again
            int numInMapped = mmm.getNumRecipes();
            ASSERT_GT(numInMapped, 0);  // verify at least one recipe in mapped memory
            // check data is still there
            bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
                recipes[lastUnused].pRecipeStaticInfoScal->recipeSections,
                lastUnused,
                *recipes[lastUnused].pSections,
                recipes[lastUnused].pDummyRecipe->isDsd(),
                recipes[lastUnused].pDummyRecipe->isIH2DRecipe());
            ASSERT_EQ(cmp, true) << "Failed cmp for i " << i;

            RecipeSeqId tempRecipeSeqId((uint64_t)recipes[lastUnused].pDummyRecipe->getRecipeSeqId());
            mmm.unuseId({tempRecipeSeqId, lastUnused});
            lastUnused++;

            mmm.getAddrForId(*pRecipeStaticInfoScal, {recipeSeqId, i}, *pSections);  // should pass now
            busy = pSections->anyBusyInMapped();
            ASSERT_EQ(busy, false);
        }

        bool offsetOK = MappedMemMgrTestUtils::testingVerifyMappedToDevOffset(map2devOffset, *pSections);
        ASSERT_EQ(offsetOK, true);

        MappedMemorySectionsUtils::memcpyToMapped(*pSections, pRecipeStaticInfoScal->recipeSections, recipes[i].pDummyRecipe->isDsd(), recipes[i].pDummyRecipe->isIH2DRecipe());
        MappedMemMgrTestUtils::testingFillMappedPatchable(pRecipeStaticInfoScal->recipeSections, i, *pSections);
        if (recipes[i].pDummyRecipe->isDsd())
        {
            MappedMemMgrTestUtils::testingFillMappedDsdPatchable(pRecipeStaticInfoScal->recipeSections, i, *pSections);
        }

        bool res = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(pRecipeStaticInfoScal->recipeSections,
                                                                                 i,
                                                                                 *pSections,
                                                                                 recipes[i].pDummyRecipe->isDsd(),
                                                                                 recipes[i].pDummyRecipe->isIH2DRecipe());
        ASSERT_EQ(res, true);
    }
    ASSERT_EQ(hadBusy, true) << "Test is not checking busy case";

    // unuse the rest
    for (int i = lastUnused; i < NUM_RECIPES; i++)
    {
        // check data is still there
        bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
            recipeStaticInfoScalArr[i % NUM_DIFFERENT_RECIPES].recipeSections,
            i,
            *recipes[i].pSections,
            recipes[i].pDummyRecipe->isDsd(),
            recipes[i].pDummyRecipe->isIH2DRecipe());
        ASSERT_EQ(cmp, true) << "Failed cmp for i " << i;
        RecipeSeqId recipeSeqId(recipes[i].pDummyRecipe->getRecipeSeqId());
        mmm.unuseId({recipeSeqId, i});
    }

    mmm.removeAllId();

    for (auto& recipeEntry : recipes)
    {
        delete recipeEntry.pSections;
    }

    ASSERT_EQ(mmm.getNumRecipes(), 0) << "mmm should be empty";

    synDestroy();
}

// This tests creates 100 recipes (50 random aligned to dcSize/2, 50 random around dcSize/2*N)
// It then loops for 2000 times
//    picks a random recipe
//    tries to get mapped memory for it
//    If not busy -> fill it with the buffers from the recipe
//    if busy     -> unuse few recipes, check before unuse the data in the mapped memory and try again
TEST(UTGaudi2MappedMemMgrTest, mappedMemMgr_random_recipes)
{
    synInitialize();  // for stats
    uint64_t               map2devOffset = 0x10000;
    DummyDevAlloc          dummyDevAlloc(map2devOffset);
    MappedMemMgr           mmm("mappedMemMgr_two_recipes_test", dummyDevAlloc);

    mmm.init();

    uint64_t      dcSize = MappedMemMgr::getDcSize();
    const uint8_t maxDc  = 10;  // not too big, so it won't take too long

    int const NUM_RECIPES1  = 50;
    int const NUM_RECIPES2  = 50;
    int const TOTAL_RECIPES = NUM_RECIPES1 + NUM_RECIPES2;
    int const NUM_RUNS      = 2000;

    std::vector<std::unique_ptr<TestDummyRecipe>> dummyRecipes(TOTAL_RECIPES);
    std::vector<RecipeStaticInfoScal>         recipeStaticInfoScalArr(TOTAL_RECIPES);
    std::vector<MemorySectionsScal>           sections(NUM_RUNS);

    // random functions for buffers sizes
    auto randomFunc = [&](int i) {
        return (i < NUM_RECIPES1) ? (1 + std::rand() % (maxDc * 2)) * dcSize / 2 - 0x20 + (std::rand() % 9) * 8
                                  : (1 + std::rand() % (maxDc * 2)) * dcSize / 2;
    };
    auto randomFuncList = [&](int i) {
        return (i < NUM_RECIPES1) ? (1 + std::rand() % 0xFFF) * DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE : 0x8000;
    };

    // create recipes
    for (int i = 0; i < TOTAL_RECIPES; i++)
    {
        uint64_t patchSize    = randomFunc(i);
        uint64_t execSize     = randomFunc(i);
        uint64_t dynamicSize  = randomFunc(i);
        uint64_t prgDataSize  = randomFunc(i);
        uint64_t ecbListsSize = randomFuncList(i);
        bool     isDsd        = randomFunc(i) % 2;  // aligned to DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE
        bool     isIH2DRecipe;
        DummyRecipeType recipeType;

        if (isDsd)
        {
            isIH2DRecipe = randomFunc(i) % 2;
            recipeType = isIH2DRecipe ? RECIPE_TYPE_DSD_AND_IH2D : RECIPE_TYPE_DSD;
        }
        else
        {
            isIH2DRecipe = false;
            recipeType = RECIPE_TYPE_NORMAL;
        }

        dummyRecipes[i] = std::unique_ptr<TestDummyRecipe>(
            new TestDummyRecipe(recipeType, patchSize, execSize, dynamicSize, prgDataSize, ecbListsSize));

        synStatus status = DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                                            *dummyRecipes[i]->getBasicRecipeInfo(),
                                                                            recipeStaticInfoScalArr[i]);
        ASSERT_EQ(status, synSuccess);
    }

    LOG_INFO(SYN_API, "-----Finished buiding recipes, starting");

    struct RecipeAndId
    {
        uint64_t recipe;
        uint64_t runningId;
    };

    std::deque<RecipeAndId> current;

    // run random recipes
    for (int i = 0; i < NUM_RUNS; i++)
    {
        uint64_t idx = std::rand() % TOTAL_RECIPES;

        bool done = false;

        while (!done)
        {
            // get mapped memory for the recipe
            RecipeSeqId recipeSeqId(dummyRecipes[idx]->getRecipeSeqId());
            mmm.getAddrForId(recipeStaticInfoScalArr[idx], {recipeSeqId, i}, sections[i]);

            bool busy = sections[i].anyBusyInMapped();

            if (!busy)
            {
                // Got memory, fill it with data
                current.push_back({idx, i});
                bool offsetOK = MappedMemMgrTestUtils::testingVerifyMappedToDevOffset(map2devOffset, sections[i]);
                ASSERT_EQ(offsetOK, true);

                bool isDsd        = dummyRecipes[idx]->isDsd();
                bool isIH2DRecipe = dummyRecipes[idx]->isIH2DRecipe();
                MappedMemorySectionsUtils::memcpyToMapped(sections[i],
                                                          recipeStaticInfoScalArr[idx].recipeSections,
                                                          isDsd,
                                                          isIH2DRecipe);
                MappedMemMgrTestUtils::testingFillMappedPatchable(recipeStaticInfoScalArr[idx].recipeSections,
                                                                  idx,
                                                                  sections[i]);
                if (isDsd)
                {
                    MappedMemMgrTestUtils::testingFillMappedDsdPatchable(recipeStaticInfoScalArr[idx].recipeSections,
                                                                         idx,
                                                                         sections[i]);
                }

                // the below check should of course pass. Adding it here to help with debug in case something goes wrong
                bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
                    recipeStaticInfoScalArr[idx].recipeSections,
                    idx,
                    sections[i],
                    isDsd,
                    isIH2DRecipe);
                ASSERT_EQ(cmp, true) << "Failed cmp for i, just after the copy" << i;
                done = true;
            }
            else
            {
                // got busy, release some previous recipes
                ASSERT_GE(current.size(), 0) << "Must free, should have something running";

                // release random number of recipes
                uint64_t toFree = std::rand() % (current.size() + 1);

                for (int j = 0; j < toFree; j++)
                {
                    RecipeAndId recipeAndId = current.front();
                    current.pop_front();

                    uint64_t recipe    = recipeAndId.recipe;
                    uint64_t runningId = recipeAndId.runningId;

                    LOG_TRACE(SYN_API, "--testing code: going to un-ues {:x}/{:x}", recipe, runningId);

                    // check data still in buffers before the release
                    bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
                        recipeStaticInfoScalArr[recipe].recipeSections,
                        recipe,
                        sections[runningId],
                        false,
                        false);
                    ASSERT_EQ(cmp, true) << "Failed cmp for runningId before un-use " << runningId;

                    recipeSeqId.val = dummyRecipes[recipe]->getRecipeSeqId();
                    mmm.unuseId({recipeSeqId, recipeAndId.runningId});
                }
            }
        }
    }  // for(i)

    // Now release and check the rest
    for (int j = 0; j < current.size(); j++)
    {
        RecipeAndId recipeAndId = current.front();
        current.pop_front();

        uint64_t recipe    = recipeAndId.recipe;
        uint64_t runningId = recipeAndId.runningId;

        LOG_TRACE(SYN_API, "--testing code: going to un-ues at the end {:x}/{:x}", recipe, runningId);

        // check data still in buffers before the release
        bool cmp = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(
            recipeStaticInfoScalArr[recipe].recipeSections,
            recipe,
            sections[runningId],
            false,
            false);
        ASSERT_EQ(cmp, true) << "Failed cmp for runningId before un-use " << runningId;

        RecipeSeqId recipeSeqId(dummyRecipes[recipe]->getRecipeSeqId());
        mmm.unuseId({recipeSeqId, recipeAndId.runningId});
    }

    synDestroy();
}

TEST(UTGaudi2MappedMemMgrTest, releaseEntryOrSegment)
{
    synInitialize();  // for stats

    uint64_t      map2devOffset = 0x10000;
    DummyDevAlloc dummyDevAlloc(map2devOffset);

    MappedMemMgr  mmm("mappedMemMgr_basic_test", dummyDevAlloc);

    mmm.init();

    // Create 2 recipes
    TestDummyRecipe          dummyRecipeMain(RECIPE_TYPE_NORMAL);
    RecipeStaticInfoScal recipeStaticInfoScalMain {};
    DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                     *dummyRecipeMain.getBasicRecipeInfo(),
                                                     recipeStaticInfoScalMain);
    //
    TestDummyRecipe          dummyRecipeSecondary(RECIPE_TYPE_NORMAL);
    RecipeStaticInfoScal recipeStaticInfoScalSecondary {};
    DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                     *dummyRecipeSecondary.getBasicRecipeInfo(),
                                                     recipeStaticInfoScalSecondary);

    MemorySectionsScal sections {};

    // Allocate all sections for main recipe (2 sections for non-patchable, all the others for patchable)
    unsigned    entryIndex = 0;
    RecipeSeqId mainRecipeSeqId(1);
    while (entryIndex < 10000)
    {
        EntryIds entryIds{.recipeId = mainRecipeSeqId, .runningId = entryIndex};

        mmm.getAddrForId(recipeStaticInfoScalMain, entryIds, sections);
        if (sections.anyBusyInMapped())
        {
            // Used up all resources
            break;
        }

        entryIndex++;
    }

    RecipeSeqId secondaryRecipeSeqId(2);
    EntryIds secondaryEntryIds{.recipeId = secondaryRecipeSeqId, .runningId = 0};

    // Try to allocate memory for another recipe, should fail
    mmm.getAddrForId(recipeStaticInfoScalSecondary, secondaryEntryIds, sections);
    ASSERT_EQ(sections.anyBusyInMapped(), true) << "Unexpectedly succeeded allocating resources for the second recipe";

    // Release 1 segment (patchable), and retry (should fail, we need 3 per recipe - 2 fon non-patchable + 1 for patchable)
    mmm.unuseId({mainRecipeSeqId, 0});
    mmm.getAddrForId(recipeStaticInfoScalSecondary, secondaryEntryIds, sections);
    ASSERT_EQ(sections.anyBusyInMapped(), true) << "Unexpectedly a single resource is enough for the second recipe";

    // Release 1 more segments (patchable), and retry (should fail, we need 3 per recipe - 2 fon non-patchable + 1 for patchable)
    mmm.unuseId({mainRecipeSeqId, 1});
    mmm.getAddrForId(recipeStaticInfoScalSecondary, secondaryEntryIds, sections);
    ASSERT_EQ(sections.anyBusyInMapped(), true) << "Unexpectedly a single resource is enough for the second recipe";

    // Release 1 segments (patchable), and retry (should pass, we already freed 3 sections,
    // we need 3 per recipe - 2 fon non-patchable + 1 for patchable)
    mmm.unuseId({mainRecipeSeqId, 2});
    mmm.getAddrForId(recipeStaticInfoScalSecondary, secondaryEntryIds, sections);
    ASSERT_EQ(sections.anyBusyInMapped(), false) << "Failed to allocate resources for the second recipe";

    synDestroy();
}

INSTANTIATE_TEST_SUITE_P(UTGaudi2MemMgr,
                         UTGaudi2MappedMemMgrTest,
                         ::testing::Values(RECIPE_TYPE_NORMAL, RECIPE_TYPE_DSD, RECIPE_TYPE_DSD_AND_IH2D),
                         [](const ::testing::TestParamInfo<DummyRecipeType>& info) {
                            // Test name suffix - either DSD, DSD and IH2D or regular version
                            const DummyRecipeType recipeType = info.param;
                            std::string name;
                            switch (recipeType)
                            {
                                case RECIPE_TYPE_NORMAL:
                                    name = "regular_recipe";
                                    break;
                                case RECIPE_TYPE_DSD:
                                    name = "dsd_recipe";
                                    break;
                                case RECIPE_TYPE_DSD_AND_IH2D:
                                    name = "dsd_ih2d_recipe";
                                    break;
                                default:
                                    name = "NULL";
                                    break;
                             }

                             return name;
                         });

// The test below tries to simulate an eager case. We create one recipe but run it multiple times with
// a new id each time. Then we remove the recipe every time. We also measure how long the mapped-memory allocation takes
class UTGaudi2MappedMemMgrEagerTest
{
};

// This test is running a lot of different small recipes to simlulate an eager scenario
TEST(UTGaudi2MappedMemMgrEagerTest, DISABLED_mappedMemMgr_eager_perf)
{
    synInitialize();  // for stats

    LOG_INFO(SYN_API, "-----mappedMemMgr_basic_test-----");
    bool isDsd = false;
    bool isIH2DRecipe = false;

    DummyDevAlloc dummyDevAlloc(0);
    MappedMemMgr  mmm("mappedMemMgr_basic_test", dummyDevAlloc);

    mmm.init();

    // Create 1 recipes
    TestDummyRecipe          dummyRecipe(RECIPE_TYPE_NORMAL);
    RecipeStaticInfoScal RecipeStaticInfoScal {};
    DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                     *dummyRecipe.getBasicRecipeInfo(),
                                                     RecipeStaticInfoScal);
    MemorySectionsScal sections {};

    int REPEATS = 10000;
    for (int i = 0; i < REPEATS; i++)
    {
        LOG_INFO(SYN_API, "----loop {}-----", i);

        RecipeSeqId recipeSeqId {i + 1};

        // try to get memory for recipe i
        STAT_GLBL_START(scalGetMapped);
        mmm.getAddrForId(RecipeStaticInfoScal, {recipeSeqId, i}, sections);
        STAT_GLBL_COLLECT_TIME(scalGetMapped, globalStatPointsEnum::scalGetMapped);

        MappedMemorySectionsUtils::memcpyToMapped(sections, RecipeStaticInfoScal.recipeSections, isDsd, isIH2DRecipe);
        MappedMemMgrTestUtils::testingFillMappedPatchable(RecipeStaticInfoScal.recipeSections, i, sections);

        bool res = MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(RecipeStaticInfoScal.recipeSections,
                                                                                 i,
                                                                                 sections,
                                                                                 isDsd,
                                                                                 isIH2DRecipe);
        ASSERT_EQ(res, true);

        STAT_GLBL_START(scalGetMapped2);
        mmm.unuseId({recipeSeqId, i});
        // We do not remove the recipe to simulate the eager case. In eager, we try to remove the recipe but because
        // we don't check it is completed we always fail and keep it. So skipping the removeId below
        // mmm.removeId(recipeSeqId);

        STAT_GLBL_COLLECT_TIME(scalGetMapped2, globalStatPointsEnum::scalGetMapped);
        STAT_GLBL_COLLECT(1, LaunchUser);  // this indicates to the stat code that a cycle was done
    }

    synDestroy();
}

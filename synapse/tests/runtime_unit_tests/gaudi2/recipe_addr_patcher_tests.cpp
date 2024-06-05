#include "test_dummy_recipe.hpp"
#include "synapse_test.hpp"

#include "runtime/scal/common/patching/recipe_addr_patcher.hpp"

class UTrecipePatchingInfo : public ::testing::Test
{
public:
    ~UTrecipePatchingInfo();

    void basicTestFlow(uint32_t dcSize);
    void patchingTime128K(uint8_t numDc);

    const int      TOTAL_BLOB     = 100;
    const int      BLOBS_WITH_PP  = TOTAL_BLOB / 2;
    const int      SECTION_OFFSET = 0x10;
    const uint32_t PATCHING_SIZE  = TOTAL_BLOB * TestDummyRecipe::BLOB_SIZE;

    void createDummySectionAddr(uint32_t numSections);
    void createRecipe();
    void copyToDummyDc(uint32_t dcSize);
    void deleteDc();

    std::unique_ptr<TestDummyRecipe> m_dummyRecipe;
    recipe_t*                    m_recipe;
    basicRecipeInfo*             m_basicRecipeInfo;
    std::vector<uint64_t>        m_sectionAddr;
    MemoryMappedAddrVec          m_dcAddr;
};

UTrecipePatchingInfo::~UTrecipePatchingInfo()
{
    deleteDc();
}

void UTrecipePatchingInfo::deleteDc()
{
    for (auto x : m_dcAddr)
    {
        delete[] x.hostAddr;
    }
}

void UTrecipePatchingInfo::createRecipe()
{
    m_dummyRecipe     = std::unique_ptr<TestDummyRecipe>(new TestDummyRecipe(RECIPE_TYPE_NORMAL, PATCHING_SIZE));
    m_recipe          = m_dummyRecipe->getRecipe();
    m_basicRecipeInfo = m_dummyRecipe->getBasicRecipeInfo();

    assert(BLOBS_WITH_PP == m_recipe->patch_points_nr / 2);  // This to verify the test itself

    auto recipePP = m_recipe->patch_points;
    int  pp       = 0;

    for (int i = 0; i < BLOBS_WITH_PP; i++)
    {
        recipePP[pp].blob_idx                             = i;
        recipePP[pp].type                                 = (patch_point_t::EPatchPointType)(i % 3);
        recipePP[pp].dw_offset_in_blob                    = i;
        recipePP[pp].memory_patch_point.effective_address = 0x100 + i;
        recipePP[pp].memory_patch_point.section_idx       = SECTION_OFFSET + i;
        pp++;

        recipePP[pp].blob_idx                             = i;
        recipePP[pp].type                                 = (patch_point_t::EPatchPointType)(i % 3);
        recipePP[pp].dw_offset_in_blob                    = TestDummyRecipe::BLOB_SIZE / 4 - 1 - i;  // dw is in 4 bytes
        recipePP[pp].memory_patch_point.effective_address = 0x200 + i;
        recipePP[pp].memory_patch_point.section_idx       = SECTION_OFFSET + i;
        pp++;
    }
}

void UTrecipePatchingInfo::createDummySectionAddr(uint32_t numSections)
{
    m_sectionAddr.resize(numSections);

    for (uint64_t i = 0; i < m_sectionAddr.size(); i++)
    {
        m_sectionAddr[i] = (i << 32) + i + 1;
    }
}

void UTrecipePatchingInfo::copyToDummyDc(uint32_t dcSize)
{
    // delete if we have previous ones
    deleteDc();

    uint32_t numDc = (m_recipe->patching_blobs_buffer_size + dcSize - 1) / dcSize;
    m_dcAddr.resize(numDc);

    for (int i = 0; i < numDc; i++)
    {
        m_dcAddr[i].hostAddr = new uint8_t[dcSize];

        uint32_t copySize = dcSize;
        if (i == (numDc - 1))
        {
            copySize = m_recipe->patching_blobs_buffer_size % dcSize;
            if (copySize == 0)
            {
                copySize = dcSize;
            }
        }

        memcpy(m_dcAddr[i].hostAddr, (uint8_t*)m_recipe->patching_blobs_buffer + dcSize * i, copySize);
    }
}

// The test class:
// creates 100 patch points on 50 blobs.
// Copies the patachble section to dummy data-chunks
// Does the patching
// Verifies the patching
void UTrecipePatchingInfo::basicTestFlow(uint32_t dcSize)
{
    createRecipe();
    createDummySectionAddr(SECTION_OFFSET + BLOBS_WITH_PP);
    copyToDummyDc(dcSize);

    RecipeAddrPatcher rpi;
    rpi.init(*m_basicRecipeInfo->recipe, dcSize);
    rpi.patchAll(m_sectionAddr.data(), m_dcAddr);

    bool ok = rpi.verifySectionsAddrFromDc(m_dcAddr, dcSize, m_sectionAddr.data(), m_sectionAddr.size());
    ASSERT_EQ(ok, true) << "Patchable buff is bad, check error log";
}

// One data-chunk
TEST_F(UTrecipePatchingInfo, basic_all)
{
    basicTestFlow(PATCHING_SIZE);
}

// One data-chunk, not full
TEST_F(UTrecipePatchingInfo, basic_all_dc_not_full)
{
    basicTestFlow(PATCHING_SIZE + 8 * TestDummyRecipe::BLOB_SIZE);
}

// 4 data-chunk, all are full
TEST_F(UTrecipePatchingInfo, basic_all_multi_dc)
{
    basicTestFlow(PATCHING_SIZE / 4);
}

// 4 data-chunk, last one is not full
TEST_F(UTrecipePatchingInfo, basic_all_multi_dc_no_full)
{
    basicTestFlow((PATCHING_SIZE + 8 * TestDummyRecipe::BLOB_SIZE) / 4);
}

/*********************************************************************************************/
void UTrecipePatchingInfo::patchingTime128K(uint8_t numDc)
{
    const uint32_t numPp       = 128 * 1024;
    const uint32_t numSections = 100;

    uint64_t patchingSize = numPp * sizeof(uint64_t);  // 128K * 8 bytes, patch every 8 bytes to cover ddw (8 bytes)

    m_dummyRecipe     = std::unique_ptr<TestDummyRecipe>(new TestDummyRecipe(RECIPE_TYPE_NORMAL, patchingSize));
    m_recipe          = m_dummyRecipe->getRecipe();
    m_basicRecipeInfo = m_dummyRecipe->getBasicRecipeInfo();

    m_dummyRecipe->m_patchPoints.resize(numPp);
    m_recipe->patch_points_nr = numPp;
    m_recipe->patch_points    = m_dummyRecipe->m_patchPoints.data();

    uint32_t numBlobs = m_recipe->blobs_nr;

    ASSERT_EQ(patchingSize, numBlobs * TestDummyRecipe::BLOB_SIZE) << "The test itself has a bug";

    auto recipePP = m_recipe->patch_points;
    int  pp       = 0;

    for (int blob = 0; blob < numBlobs; blob++)
    {
        for (int blobPp = 0; blobPp < TestDummyRecipe::BLOB_SIZE / sizeof(uint64_t); blobPp++)
        {
            recipePP[pp].blob_idx                             = blob;
            recipePP[pp].type                                 = (patch_point_t::EPatchPointType)((blobPp + blob) % 3);
            recipePP[pp].dw_offset_in_blob                    = blobPp * 2;  // patch every 8 bytes = 2 * dw
            recipePP[pp].memory_patch_point.effective_address = 0x100 + blob;
            recipePP[pp].memory_patch_point.section_idx       = (SECTION_OFFSET + blobPp) % numSections;
            pp++;
        }
    }

    createDummySectionAddr(numSections);

    uint64_t dcSize = patchingSize / numDc;

    copyToDummyDc(dcSize);

    RecipeAddrPatcher rpi;
    rpi.init(*m_basicRecipeInfo->recipe, dcSize);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    rpi.patchAll(m_sectionAddr.data(), m_dcAddr);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
              << "[ns]" << std::endl;

    bool ok = rpi.verifySectionsAddrFromDc(m_dcAddr, dcSize, m_sectionAddr.data(), m_sectionAddr.size());
    ASSERT_EQ(ok, true) << "Patchable buff is bad, check error log";
}

// 1M patch points, all in one DC
TEST_F(UTrecipePatchingInfo, time_128K_singleDc)
{
    patchingTime128K(1);
}

// 1M patch points, in 2 DC
TEST_F(UTrecipePatchingInfo, time_128K_2Dc)
{
    patchingTime128K(2);
}

// 1M patch points, in 4 DC
TEST_F(UTrecipePatchingInfo, time_128K_4Dc)
{
    patchingTime128K(4);
}

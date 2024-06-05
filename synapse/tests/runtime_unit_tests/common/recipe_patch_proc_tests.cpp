#include <gtest/gtest.h>
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include <habana_global_conf_runtime.h>
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"
#include "define_synapse_common.hpp"
#include "test_dummy_recipe.hpp"
#include "runtime/common/recipe/recipe_patch_processor.hpp"
#include "../common/dev_memory_alloc_mock.hpp"
#include "runtime/common/syn_logging.h"

using Tensors          = std::vector<synLaunchTensorInfoExt>;
using TensorModifyFunc = std::function<void(Tensors&)>;

// NOTE: This UT is for gaudi2 only as it is using RecipePatchProcessor::process
//       And doesn't test yet tensors related to DSD

class UTrecipePatchProcTest : public ::testing::Test
{
public:
    void setupRecipe();
    void setupTest();
    void checkAddr(const patching::HostAddressPatchingInformation& hostAddrPatchInfo);
    void setupLaunchTensors(bool shuffle, TensorModifyFunc);
    void runTest(bool             withValidate,
                 bool             shuffle,
                 TensorModifyFunc tensorModifyFunc,
                 synStatus        expectedProcess,
                 bool             expectedValidate);
    void runAllCases(
        synStatus        expectedProcess,
        bool             expectedValidate,
        TensorModifyFunc tensorModifyFunc = [](Tensors&) {});
    uint32_t toSectionIdx(uint32_t tensorNum);
    uint32_t toIdxInSection(uint32_t tensorNum);

    static const int          TENSORS_IN_SECTION     = 3;
    static const int          NUM_SECTIONS           = 40;
    const int                 NUM_TENSORS            = NUM_SECTIONS * TENSORS_IN_SECTION;
    const int                 NUM_SECTION_TYPES      = 10;
    uint64_t                  scratchPadAddr         = 0x11223344;
    uint64_t                  prgDataAddr            = 0x55664000;
    uint64_t                  prgCodeAddr            = 0x66778000;
    uint64_t                  assertAsyncAddr        = 0x67891000;
    uint64_t                  SECTION_TO_ADDR_FACTOR = 0x1000000;
    uint64_t                  baseTensorSize         = 0x100;

    uint64_t                     constSectionNumber = 0;
    std::vector<const_section_t> constSecVec;

    std::unique_ptr<TestDummyRecipe> m_dummyRecipe;
    std::vector<std::string>     m_tensorName;

    basicRecipeInfo          m_recipeInfo {};
    DeviceAgnosticRecipeInfo m_deviceAgnosticRecipeInfo {};
    WorkSpacesInformation    m_wsInfo {};
    Tensors                  m_launchTensors;

    uint64_t offsetInSection[NUM_SECTIONS + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR][TENSORS_IN_SECTION] {};

    ValidSectionAddresses validSectionAddresses {0x100000, 0xFF00000000000000};

    bool shouldAvoidCheckSectionAddr = false;
};

uint32_t UTrecipePatchProcTest::toSectionIdx(uint32_t tensorNum)
{
    return (tensorNum % NUM_SECTIONS) + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
}

uint32_t UTrecipePatchProcTest::toIdxInSection(uint32_t tensorNum)
{
    return tensorNum / NUM_SECTIONS;
}

void UTrecipePatchProcTest::setupRecipe()
{
    m_dummyRecipe = std::unique_ptr<TestDummyRecipe>(new TestDummyRecipe);

    m_dummyRecipe->m_tensors.resize(NUM_TENSORS);
    m_tensorName.resize(NUM_TENSORS);

    recipe_t* recipe                 = m_dummyRecipe->getRecipe();
    recipe->permute_tensors_views_nr = 0;
    recipe->persist_tensors_nr       = m_dummyRecipe->m_tensors.size();
    recipe->tensors                  = m_dummyRecipe->m_tensors.data();

    auto& tensors = m_dummyRecipe->m_tensors;

    for (int i = 0; i < NUM_TENSORS; i++)
    {
        uint64_t tensorSize = (i + 1) * baseTensorSize;
        tensors[i].size     = tensorSize;
        m_tensorName[i]     = std::to_string(i);
        tensors[i].name     = m_tensorName[i].c_str();

        uint32_t sectionIdx          = toSectionIdx(i);
        uint32_t idxInSection        = toIdxInSection(i);
        tensors[i].section_idx       = sectionIdx;
        tensors[i].offset_in_section = offsetInSection[sectionIdx][idxInSection];
        if (idxInSection < (TENSORS_IN_SECTION - 1))
        {
            offsetInSection[sectionIdx][idxInSection + 1] = offsetInSection[sectionIdx][idxInSection] + tensorSize;
        }

        tensors[i].isInput                = i % 2;
        tensors[i].section_type           = tensors[i].section_idx % NUM_SECTION_TYPES;
        tensors[i].multi_views_indices_nr = 0;
    }

    if (constSectionNumber > 0)
    {
        recipe->const_sections_nr = constSectionNumber;
        recipe->const_sections    = new const_section_t[constSectionNumber];
        for (int i = 0; i < constSectionNumber; ++i)
        {
            recipe->const_sections[i] = constSecVec[i];
        }
    }
}

void UTrecipePatchProcTest::setupLaunchTensors(bool shuffle, TensorModifyFunc tensorModifyFunc)
{
    // create launch tensors
    m_launchTensors.resize(NUM_TENSORS);

    for (int i = 0; i < NUM_TENSORS; i++)
    {
        m_launchTensors[i].tensorId =
            i;  // always set, to support tensorModifyFunc function, later zeroed out if needed
        m_launchTensors[i].tensorName = nullptr;
        m_launchTensors[i].tensorType = DATA_TENSOR;

        uint64_t sectionIdx   = toSectionIdx(i);
        uint32_t idxInSection = toIdxInSection(i);

        m_launchTensors[i].pTensorAddress =
            SECTION_TO_ADDR_FACTOR * sectionIdx + offsetInSection[sectionIdx][idxInSection];
    }

    if (shuffle)
    {
        std::random_shuffle(m_launchTensors.begin(), m_launchTensors.end());
    }

    if (tensorModifyFunc)
    {
        tensorModifyFunc(m_launchTensors);
    }
}

void UTrecipePatchProcTest::setupTest()
{
    GCFG_CHECK_SECTION_OVERLAP.setValue(true);

    setupRecipe();

    m_recipeInfo.recipe = m_dummyRecipe->getRecipe();

    recipe_t* pRecipe = m_recipeInfo.recipe;

    synStatus status = RecipeTensorsProcessor::process(m_recipeInfo,
                                                       m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                       m_deviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
    ASSERT_EQ(status, synSuccess);

    assert((prgDataAddr % (1 << 13)) == 0);  // test error: tensor analyze aligns the prgDataAddr to 1<<13, fix test
    m_wsInfo.scratchPadAddress        = scratchPadAddr;
    m_wsInfo.programDataAddress       = prgDataAddr;
    m_wsInfo.programCodeAddress       = prgCodeAddr;
    m_wsInfo.assertAsyncMappedAddress = assertAsyncAddr;

    pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]    = (scratchPadAddr == 0) ? 0 : 1;
    pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = (prgDataAddr    == 0) ? 0 : 1;
    pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM]      = (prgCodeAddr    == 0) ? 0 : 1;

    RecipeTensorsProcessor::setSectionsInfo(m_recipeInfo, m_deviceAgnosticRecipeInfo.m_recipeTensorInfo);
}

void UTrecipePatchProcTest::checkAddr(const patching::HostAddressPatchingInformation& hostAddrPatchInfo)
{
    if (shouldAvoidCheckSectionAddr)
    {
        return;
    }

    uint32_t        dbSize = hostAddrPatchInfo.getSectionsDbSize();
    const uint64_t* dbAddr = hostAddrPatchInfo.getSectionsToHostAddressDB();

    ASSERT_EQ(dbSize, NUM_SECTIONS + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    ASSERT_EQ(dbAddr[MEMORY_ID_RESERVED_FOR_WORKSPACE], scratchPadAddr);
    ASSERT_EQ(dbAddr[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA], prgDataAddr);

    for (int i = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR; i < dbSize; i++)
    {
        ASSERT_EQ(dbAddr[i], i * SECTION_TO_ADDR_FACTOR) << "failed for section  " << i;
    }
}

void UTrecipePatchProcTest::runTest(bool             withValidate,
                                    bool             shuffle,
                                    TensorModifyFunc tensorModifyFunc,
                                    synStatus        expectedProcess,
                                    bool             expectedValidate)
{
    // build dummy recipe
    //    setupTest();

    // create hostAddrPatchInfo
    patching::HostAddressPatchingInformation hostAddrPatchInfo;
    DevMemoryAllocMock                       devMemoryAlloc;

    // analyze tensors
    for (int i = 0; i < 3; i++)
    {
        setupLaunchTensors(shuffle, tensorModifyFunc);

        std::vector<uint32_t> tensorIdx2userIdx[tensor_info_t::ETensorType::INTERNAL_TENSOR];

        synStatus status = RecipePatchProcessor::process(m_recipeInfo,
                                                         m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                         m_launchTensors.data(),
                                                         m_launchTensors.size(),
                                                         0,
                                                         m_wsInfo,
                                                         hostAddrPatchInfo,
                                                         devMemoryAlloc,
                                                         tensorIdx2userIdx,
                                                         true /* isInitAndCompletionRequired */,
                                                         true /* shouldResolveTensorsIndices */,
                                                         &validSectionAddresses);
        ASSERT_EQ(status, expectedProcess);

        if (status == synSuccess)
        {
            if (expectedValidate == true)  // check addr only for the good case
            {
                checkAddr(hostAddrPatchInfo);
            }

            if (withValidate)
            {
                synStatus res = RecipePatchProcessor::validateSectionsInfo(
                    m_recipeInfo,
                    m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                    m_launchTensors.size(),
                    m_launchTensors.data(),
                    0,
                    m_wsInfo,
                    m_deviceAgnosticRecipeInfo.m_recipeTensorInfo.m_sectionsInfo,
                    devMemoryAlloc);

                if (expectedValidate)
                {
                    ASSERT_EQ(res, synSuccess);
                }
                else
                {
                    ASSERT_NE(res, synSuccess);
                }
            }
        }
    }

    // change addresses
    SECTION_TO_ADDR_FACTOR = 0x2000000;

    for (int i = 0; i < 3; i++)
    {
        setupLaunchTensors(shuffle, tensorModifyFunc);

        std::vector<uint32_t> tensorIdx2userIdx[tensor_info_t::ETensorType::INTERNAL_TENSOR];

        synStatus status = RecipePatchProcessor::process(m_recipeInfo,
                                                         m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                         m_launchTensors.data(),
                                                         m_launchTensors.size(),
                                                         0,
                                                         m_wsInfo,
                                                         hostAddrPatchInfo,
                                                         devMemoryAlloc,
                                                         tensorIdx2userIdx,
                                                         true /* isInitAndCompletionRequired */,
                                                         true /* shouldResolveTensorsIndices */,
                                                         &validSectionAddresses);
        ASSERT_EQ(status, expectedProcess);
        if (status == synSuccess)
        {
            if (expectedValidate == true)  // check addr only for the good case
            {
                checkAddr(hostAddrPatchInfo);
            }

            if (withValidate)
            {
                synStatus res = RecipePatchProcessor::validateSectionsInfo(
                    m_recipeInfo,
                    m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                    m_launchTensors.size(),
                    m_launchTensors.data(),
                    0,
                    m_wsInfo,
                    m_deviceAgnosticRecipeInfo.m_recipeTensorInfo.m_sectionsInfo,
                    devMemoryAlloc);

                if (expectedValidate)
                {
                    ASSERT_EQ(res, synSuccess);
                }
                else
                {
                    ASSERT_NE(res, synSuccess);
                }
            }
        }
    }

    for (int i = 1; i < 5; i++)
    {
        // change addresses
        SECTION_TO_ADDR_FACTOR = i * 0x1000000;

        setupLaunchTensors(shuffle, tensorModifyFunc);
        std::vector<uint32_t> tensorIdx2userIdx[tensor_info_t::ETensorType::INTERNAL_TENSOR];

        synStatus status = RecipePatchProcessor::process(m_recipeInfo,
                                                         m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                         m_launchTensors.data(),
                                                         m_launchTensors.size(),
                                                         0,
                                                         m_wsInfo,
                                                         hostAddrPatchInfo,
                                                         devMemoryAlloc,
                                                         tensorIdx2userIdx,
                                                         true /* isInitAndCompletionRequired */,
                                                         true /* shouldResolveTensorsIndices */,
                                                         &validSectionAddresses);
        ASSERT_EQ(status, expectedProcess);
        if (status == synSuccess)
        {
            if (expectedValidate == true)  // check addr only for the good case
            {
                checkAddr(hostAddrPatchInfo);
            }

            if (withValidate)
            {
                synStatus res = RecipePatchProcessor::validateSectionsInfo(
                    m_recipeInfo,
                    m_deviceAgnosticRecipeInfo.m_recipeTensorInfo,
                    m_launchTensors.size(),
                    m_launchTensors.data(),
                    0,
                    m_wsInfo,
                    m_deviceAgnosticRecipeInfo.m_recipeTensorInfo.m_sectionsInfo,
                    devMemoryAlloc);

                if (expectedValidate)
                {
                    ASSERT_EQ(res, synSuccess);
                }
                else
                {
                    ASSERT_NE(res, synSuccess);
                }
            }
        }
    }
}

void UTrecipePatchProcTest::runAllCases(synStatus        expectedProcess,
                                        bool             expectedValidate,
                                        TensorModifyFunc tensorModifyFunc)
{
    for (bool withValidate : {false, true})
    {
        for (bool shuffle : {false, true})
        {
            LOG_INFO(SYN_API,
                     "======{} withValidate {}, shuffle {}======",
                     ::testing::UnitTest::GetInstance()->current_test_info()->name(),
                     withValidate,
                     shuffle);
            runTest(withValidate, shuffle, tensorModifyFunc, expectedProcess, expectedValidate);
        }
    }
}

// Test the good case
TEST_F(UTrecipePatchProcTest, good_case)
{
    setupTest();
    runAllCases(synSuccess, true);
}

// Missing vector
TEST_F(UTrecipePatchProcTest, missing_tensor)
{
    setupTest();

    auto removeTensor = [this](Tensors& tensors) { tensors.erase(tensors.begin() + NUM_TENSORS / 2); };

    runAllCases(synSuccess, false, removeTensor);
}

// One tensor in the section has a different addr than the other tensors in the same section
TEST_F(UTrecipePatchProcTest, incosistent_addr)
{
    setupTest();

    auto changeAddr = [this](Tensors& tensors) { tensors[NUM_TENSORS / 2].pTensorAddress += 10; };

    runAllCases(synFail, false, changeAddr);

    // run again the good case - verify we recovered correctly
    runAllCases(synSuccess, true);
}

TEST_F(UTrecipePatchProcTest, zero_addr_workspace_address)
{
    uint64_t originalWorkspaceAddress = 0;

    originalWorkspaceAddress = scratchPadAddr;
    scratchPadAddr           = 0;
    setupTest();
    runAllCases(synSuccess, true, nullptr);
    scratchPadAddr = originalWorkspaceAddress;

    originalWorkspaceAddress = prgDataAddr;
    prgDataAddr              = 0;
    setupTest();
    runAllCases(synSuccess, true, nullptr);
    prgDataAddr = originalWorkspaceAddress;

    originalWorkspaceAddress = prgCodeAddr;
    prgCodeAddr              = 0;
    setupTest();
    runAllCases(synSuccess, true, nullptr);
    prgCodeAddr = originalWorkspaceAddress;

    originalWorkspaceAddress = assertAsyncAddr;
    assertAsyncAddr          = 0;
    setupTest();
    runAllCases(synSuccess, true, nullptr);
    assertAsyncAddr = originalWorkspaceAddress;
}

// Tensor has an addr of 0 (valid, not sure what is the case, I think something in DSD)
TEST_F(UTrecipePatchProcTest, zero_addr_tensor_while_section_is_set)
{
    setupTest();

    auto zeroAddr = [this](Tensors& tensors) { tensors[NUM_TENSORS / 2].pTensorAddress = 0; };

    runAllCases(synSuccess, true, zeroAddr);
}

// All tensors in a section get zero-addr, and hence the section will not be set and test should fail
TEST_F(UTrecipePatchProcTest, zero_addr_for_all_tensors_in_section)
{
    setupTest();

    auto zeroAddrForAllTensorsInSection = [this](Tensors& tensors)
    {
        unsigned sectionToChange = NUM_SECTIONS / 2;
        for (unsigned i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                tensors[i].pTensorAddress = 0;
            }
        }
    };

    runAllCases(synFail, true, zeroAddrForAllTensorsInSection);
}

// Verify the tensor offset - there is a check in the code "tensorOffsetInSection > launchTensorsInfo[i].pTensorAddress"
// To check it, I move the offset of a specific section by 10 and then set the address to 1
TEST_F(UTrecipePatchProcTest, big_offsetInSection)
{
    setupTest();

    auto bigOffsetInSection = [this](Tensors& tensors) {
        int sectionToChange = NUM_SECTIONS / 2 + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
        for (int i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                recipe_t* recipe = m_dummyRecipe->getRecipe();
                recipe->tensors[tensorId].offset_in_section += 10;
            }
        }
        tensors[NUM_TENSORS / 2].pTensorAddress = 1;
    };

    runAllCases(synFail, true, bigOffsetInSection);
}

TEST_F(UTrecipePatchProcTest, duplicate_name)
{
    setupTest();

    auto duplicateName = [this](Tensors& tensors) { tensors.push_back(tensors[NUM_TENSORS / 2]); };

    runAllCases(synSuccess, true, duplicateName);
}

TEST_F(UTrecipePatchProcTest, overlap_same_addr)
{
    setupTest();

    auto overlapSameAddr = [this](Tensors& tensors) {
        int sectionToChange = NUM_SECTIONS / 2 + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
        for (int i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                tensors[i].pTensorAddress -= SECTION_TO_ADDR_FACTOR;
            }
        }
    };

    runAllCases(synSuccess, false, overlapSameAddr);
}

TEST_F(UTrecipePatchProcTest, overlap)
{
    setupTest();

    auto overlap = [this](Tensors& tensors) {
        int sectionToChange = NUM_SECTIONS / 2 + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
        for (int i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                tensors[i].pTensorAddress -= SECTION_TO_ADDR_FACTOR + 0x10;
            }
        }
    };

    runAllCases(synSuccess, false, overlap);
}

TEST_F(UTrecipePatchProcTest, verifyAddrInHbm)
{
    setupTest();

    auto badAddrHigh = [this](Tensors& tensors) {
        int sectionToChange = NUM_SECTIONS / 2 + MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
        for (int i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                tensors[i].pTensorAddress += validSectionAddresses.highestValidAddress;  // make it bad
            }
        }
    };

    runAllCases(synFail,false, badAddrHigh);

    setupTest();
    validSectionAddresses.lowestValidAddress = validSectionAddresses.highestValidAddress - 0x1000;

    runAllCases(synFail, false, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyScratchPadInHbm)
{
    scratchPadAddr = validSectionAddresses.lowestValidAddress - 1;
    setupTest();

    runAllCases(synFail, false, nullptr);

    scratchPadAddr = validSectionAddresses.highestValidAddress + 1;
    setupTest();

    runAllCases(synFail, false, nullptr);

    scratchPadAddr = validSectionAddresses.lowestValidAddress;
    setupTest();

    runAllCases(synSuccess, true, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyPrgCodeAddrAddrInHbm)
{
    prgCodeAddr = validSectionAddresses.lowestValidAddress - 1;
    setupTest();

    runAllCases(synFail, false, nullptr);

    prgCodeAddr = validSectionAddresses.highestValidAddress + 1;
    setupTest();

    runAllCases(synFail, false, nullptr);

    prgCodeAddr = validSectionAddresses.lowestValidAddress;
    setupTest();

    runAllCases(synSuccess, true, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyConstSections_MarkLastAsConst)
{
    constSectionNumber = 3;

    for (uint16_t i = 0; i < constSectionNumber; ++i)
    {
        const_section_t constSec = {nullptr, baseTensorSize, NUM_SECTIONS + 1 + i};
        constSecVec.push_back(constSec);
    }
    setupTest();
    runAllCases(synSuccess, true, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyConstSectionsTenso_rReduceSize)
{
    constSectionNumber = 1;

    uint64_t constSectionSize = baseTensorSize;
    for (uint16_t i = 0; i < constSectionNumber; ++i)
    {
        // start with second section
        const_section_t constSec = {nullptr, constSectionSize, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i + 1};
        constSecVec.push_back(constSec);
    }
    setupTest();
    runAllCases(synSuccess, true, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyConstSectionsTensor_IncreaseSize)
{
    constSectionNumber = 1;

    uint64_t constSectionSize = baseTensorSize * NUM_TENSORS * SECTION_TO_ADDR_FACTOR * 2;
    for (uint16_t i = 0; i < constSectionNumber; ++i)
    {
        // start with second section
        const_section_t constSec = {nullptr, constSectionSize, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i + 1};
        constSecVec.push_back(constSec);
    }
    setupTest();
    runAllCases(synSuccess, false, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyConstSections_ZeroSize)
{
    constSectionNumber = 1;

    uint64_t constSectionSize = 0;
    for (uint16_t i = 0; i < constSectionNumber; ++i)
    {
        // start with second section
        const_section_t constSec = {nullptr, constSectionSize, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i + 1};
        constSecVec.push_back(constSec);
    }
    setupTest();

    runAllCases(synSuccess, true, nullptr);
}

TEST_F(UTrecipePatchProcTest, VerifyConstSections_MissingTensor)
{
    constSectionNumber = 1;

    uint64_t constSectionSize = baseTensorSize;
    for (uint16_t i = 0; i < constSectionNumber; ++i)
    {
        // start with second section
        const_section_t constSec = {nullptr, constSectionSize, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i + 1};
        constSecVec.push_back(constSec);
    }
    setupTest();

    auto missingTensor = [this](Tensors& tensors) {
        int sectionToChange = 5;
        for (int i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                tensors[i].tensorId++;  // make it point to next section
                tensors[i].pTensorAddress = tensors[i + 1].pTensorAddress;
            }
        }
    };

    // no shuffle, should fail on missing tensor beacuse the const section isn't zero
    runTest(true, false, missingTensor, synFail, false);
}

TEST_F(UTrecipePatchProcTest, VerifyConstSections_ZeroSizeMissingTensor)
{
    constSectionNumber = 1;

    uint64_t constSectionSize = 0;
    uint64_t sectionId        = 5;

    for (uint16_t i = 0; i < constSectionNumber; ++i)
    {
        // start with second section
        const_section_t constSec = {nullptr, constSectionSize, sectionId};
        constSecVec.push_back(constSec);
    }
    setupTest();

    auto missingTensor = [this](Tensors& tensors) {
        int sectionToChange = 5;
        for (int i = 0; i < tensors.size(); i++)
        {
            uint32_t tensorId = tensors[i].tensorId;
            if (toSectionIdx(tensorId) == sectionToChange)
            {
                tensors[i].tensorId++;  // make it point to next section
                tensors[i].pTensorAddress = tensors[i + 1].pTensorAddress;
            }
        }
    };

    // no shuffle, should passe const section isn zero
    shouldAvoidCheckSectionAddr = true;
    runTest(true, false, missingTensor, synSuccess, true);
    shouldAvoidCheckSectionAddr = false;
}

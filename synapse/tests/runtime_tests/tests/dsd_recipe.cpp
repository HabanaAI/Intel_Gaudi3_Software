#include "syn_base_test.hpp"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "smf/shape_func_registry.h"
#include "../infra/test_types.hpp"
#include "test_recipe_dsd_gemm.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "dsd_recipes.h"

class DsdRecipeTest : public DsdRecipeTestBase
{
public:
    DsdRecipeTest() : DsdRecipeTestBase() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
    bool m_gaudi3DSD;

public:
    void SetUp() override
    {
        DsdRecipeTestBase::SetUp();
    }

    void TearDown() override
    {
        SynBaseTest::TearDown();
    }
};

REGISTER_SUITE(DsdRecipeTest, ALL_TEST_PACKAGES);

TEST_F_SYN(DsdRecipeTest, shape_func_repository_destroy)
{
    synDeviceId deviceId;
    synStatus   status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to acquire device";

    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();

    auto sifBefore = sfr.getAllSifTestingOnly();
    auto smfBefore = sfr.getAllSmfTestingOnly();

    LOG_DEBUG(SYN_RT_TEST, "---- test shape_func_repository_destroy - call synDestroy");
    status = synDestroy();
    ASSERT_EQ(status, synSuccess) << "synDestroy failed";
    LOG_DEBUG(SYN_RT_TEST, "---- test shape_func_repository_destroy - call synInitialize");
    status = synInitialize();
    ASSERT_EQ(status, synSuccess);

    auto sifAfter = sfr.getAllSifTestingOnly();
    auto smfAfter = sfr.getAllSmfTestingOnly();

    ASSERT_EQ(sifBefore.size(), sifAfter.size()) << "sif maps are not equal";
    ASSERT_EQ(smfBefore.size(), smfAfter.size()) << "smf maps are not equal";
}

/*
 ***************************************************************************************************
 *   @brief The test below checks the good and bad paths of anyalyze tensor for DATA_TENSOR, DATA_TENSOR_DYNAMIC
 *
 ***************************************************************************************************
 */
TEST_F_SYN(DsdRecipeTest, testAnalyzeTensors)
{
    unsigned op1Actual = 64;
    unsigned op2Actual = op1Actual * 2;

    synConfigurationSet("CHECK_SECTION_OVERLAP_CHECK", "true");

    TestRecipeDsdGemm recipe(m_deviceType,
                             false /* isDynamic */,
                             false /* isSharedInputSection */,
                             {256, 128},
                             {256, 128},
                             {256, 256},
                             {128, 256},
                             {256, 128},
                             {128, 128});
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    // 1. Should pass
    {
        TestTensorsDimensions testTensorsDimensions;
        {
            TensorDimensions tensorDimensions =
                {op2Actual, recipe.getDynamicTensorMaxDimSizeInput(1 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(true, 1, tensorDimensions);
        }

        {
            TensorDimensions tensorDimensions =
                {op2Actual, recipe.getDynamicTensorMaxDimSizeOutput(0 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(false, 0, tensorDimensions);
        }

        TestLauncher       launcher(device);
        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0}, testTensorsDimensions);

        TestLauncher::execute(stream, recipe, recipeLaunchParams);

        stream.synchronize();
    }

    // 2. Address out of HBM range.
    // The test is here and not in SynGaudiRtRecipe because there it is a nop node and the simulator doesn't crash
    {
        TestTensorsDimensions testTensorsDimensions;
        LOG_INFO_T(SYN_API, "-----Out of HBM range");

        {
            TensorDimensions tensorDimensions =
                {op2Actual, recipe.getDynamicTensorMaxDimSizeInput(0 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(true, 1, tensorDimensions);
        }

        recipe.generateRecipe();

        TestLauncher   launcher(device);

        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
                                             testTensorsDimensions);
        synLaunchTensorInfoExt tensorLaunchInfo;
        const synLaunchTensorInfoExt* origTensorInfo = recipeLaunchParams.getSynLaunchTensorInfoVec().data();
        std::memcpy(&tensorLaunchInfo, (void*)origTensorInfo, sizeof(synLaunchTensorInfoExt));

        tensorLaunchInfo.pTensorAddress = std::numeric_limits<uint64_t>::max();

        synStatus status = synLaunchExt(stream.operator synStreamHandle(),
                                        &tensorLaunchInfo,
                                        1, //numberTensors
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0); //flags

        ASSERT_EQ(status, synFailedSectionValidation);

        stream.synchronize();
    }
    // 3. Bad tensor type, should fail
    {
        LOG_INFO_T(SYN_API, "-----Bad tensor type1");

        TestTensorsDimensions testTensorsDimensions;

        {
            TensorDimensions tensorDimensions =
                            {op2Actual, recipe.getDynamicTensorMaxDimSizeOutput(0 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(false, 0, tensorDimensions);
        }

        recipe.generateRecipe();

        TestLauncher   launcher(device);

        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe,{TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
        testTensorsDimensions);

        synLaunchTensorInfoExt tensorLaunchInfo;
        const synLaunchTensorInfoExt* origTensorInfo = recipeLaunchParams.getSynLaunchTensorInfoVec().data();
        std::memcpy(&tensorLaunchInfo, (void*)origTensorInfo, sizeof(synLaunchTensorInfoExt));

        tensorLaunchInfo.tensorType = DATA_TENSOR_DYNAMIC;

        synStatus status = synLaunchExt(stream.operator synStreamHandle(),
                                        &tensorLaunchInfo,
                                        1, //numberTensors
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0); //flags

        ASSERT_EQ(status, synFailedSectionValidation);

        stream.synchronize();
    }

    // 4. Bad tensor type, should fail
    {
        LOG_INFO_T(SYN_API, "-----Bad tensor type2");
        TestTensorsDimensions testTensorsDimensions;

        {
            TensorDimensions tensorDimensions =
                            {op2Actual, recipe.getDynamicTensorMaxDimSizeOutput(0 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(false, 0, tensorDimensions);
        }

        recipe.generateRecipe();

        TestLauncher   launcher(device);

        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe,{TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
        testTensorsDimensions);

        synLaunchTensorInfoExt tensorLaunchInfo;
        const synLaunchTensorInfoExt* origTensorInfo = recipeLaunchParams.getSynLaunchTensorInfoVec().data();
        std::memcpy(&tensorLaunchInfo, (void*)origTensorInfo, sizeof(synLaunchTensorInfoExt));

        tensorLaunchInfo.tensorType = SHAPE_TENSOR;

        synStatus status = synLaunchExt(stream.operator synStreamHandle(),
                                        &tensorLaunchInfo,
                                        1, //numberTensors
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0); //flags

        ASSERT_EQ(status, synFailedSectionValidation);

        stream.synchronize();
    }

    // 5. Using DEVICE_SHAPE_TENSOR instead of DATA_TENSOR, should fail too
    {
        LOG_INFO_T(SYN_API, "-----DEVICE_SHAPE_TENSOR instead of DATA_TENSOR");
        TestTensorsDimensions testTensorsDimensions;

        {
            TensorDimensions tensorDimensions =
                            {op2Actual, recipe.getDynamicTensorMaxDimSizeOutput(0 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(false, 0, tensorDimensions);
        }

        recipe.generateRecipe();

        TestLauncher   launcher(device);

        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe,{TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
        testTensorsDimensions);

        synLaunchTensorInfoExt tensorLaunchInfo;
        const synLaunchTensorInfoExt* origTensorInfo = recipeLaunchParams.getSynLaunchTensorInfoVec().data();
        std::memcpy(&tensorLaunchInfo, (void*)origTensorInfo, sizeof(synLaunchTensorInfoExt));

        tensorLaunchInfo.tensorType = DEVICE_SHAPE_TENSOR;

        synStatus status = synLaunchExt(stream.operator synStreamHandle(),
                                        &tensorLaunchInfo,
                                        1, //numberTensors
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0); //flags

        ASSERT_EQ(status, synFailedSectionValidation);

        stream.synchronize();
    }

    // 6. Should pass
    {
        TestTensorsDimensions testTensorsDimensions;

        {
            TensorDimensions tensorDimensions =
                {op2Actual, recipe.getDynamicTensorMaxDimSizeInput(1 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(true, 1, tensorDimensions);
        }

        {
            TensorDimensions tensorDimensions =
                {op2Actual, recipe.getDynamicTensorMaxDimSizeOutput(0 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};
            testTensorsDimensions.setDimensions(false, 0, tensorDimensions);
        }

        recipe.generateRecipe();

        TestLauncher launcher(device);

        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0}, testTensorsDimensions);

        TestLauncher::execute(stream, recipe, recipeLaunchParams);

        stream.synchronize();
    }
}

/*
 ***************************************************************************************************
 *   @brief The test below randomized DSD GEMM recipe inputs and validate results
 *
 ***************************************************************************************************
 */
TEST_F_SYN(DsdRecipeTest, randomizeAndValidate)
{
    unsigned op1Actual = 64;
    unsigned op2Actual = op1Actual * 2;

    TestRecipeDsdGemm recipe(m_deviceType,
                             true /* isDynamic */,
                             false /* isSharedInputSection */,
                             {256, 128},
                             {256, 64},
                             {256, 256},
                             {128, 256},
                             {256, 128},
                             {128, 64});
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestTensorsDimensions testTensorsDimensions;

    unsigned index = 0;

    {
        TensorDimensions tensorDimensions =
            {recipe.getDynamicTensorMaxDimSizeInput(0 /* tensorIndex */, 0 /* dimIndex */), op1Actual, 0, 0, 0};

        testTensorsDimensions.setDimensions(true, 0, tensorDimensions);

        index++;
    }

    {
        TensorDimensions tensorDimensions =
            {op2Actual, recipe.getDynamicTensorMaxDimSizeInput(1 /* tensorIndex */, 1 /* dimIndex */), 0, 0, 0};

        testTensorsDimensions.setDimensions(true, 1, tensorDimensions);

        index++;
    }

    {
        TensorDimensions tensorDimensions = {op2Actual, op1Actual, 0, 0, 0};

        testTensorsDimensions.setDimensions(false, 0, tensorDimensions);

        index++;  // NA
    }

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe,
                                          {{TensorInitOp::RANDOM_WITH_NEGATIVE, 1}, {TensorInitOp::CONST, 0}},
                                          testTensorsDimensions);

    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    recipe.setExecutionDynamicSize(op1Actual, op2Actual);

    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());
}

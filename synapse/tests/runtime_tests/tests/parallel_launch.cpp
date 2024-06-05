#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_utils.h"
#include "test_recipe_gemm.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_recipe_addf32.hpp"
#include "test_recipe_addf32_gemm.hpp"
#include "test_recipe_gemm_addf32.hpp"

class SynCommonParallelLaunch : public SynBaseTest
{
public:
    SynCommonParallelLaunch() : SynBaseTest()
    {
        m_deviceType = synDeviceTypeInvalid;

        if (getenv("SYN_DEVICE_TYPE") != nullptr)
        {
            m_deviceType = (synDeviceType)std::stoi(getenv("SYN_DEVICE_TYPE"));
        }

        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi2");
            m_deviceType = synDeviceGaudi2;
        }
        setSupportedDevices({synDeviceGaudi2});
    }

    void execute(TestRecipeBase& recipe)
    {
        recipe.generateRecipe();

        TestDevice device(m_deviceType);

        TestStream stream = device.createStream();

        TestLauncher launcher(device);

        RecipeLaunchParams recipeLaunchParam =
            std::move(launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_POSITIVE, 0}));

        TestLauncher::execute(stream, recipe, recipeLaunchParam);

        stream.synchronize();

        recipe.validateResults(recipeLaunchParam.getLaunchTensorMemory());
    }

    void execute(TestRecipeBase& recipe1, TestRecipeBase& recipe2)
    {
        recipe1.generateRecipe();
        recipe2.generateRecipe();

        TestDevice device(m_deviceType);

        auto streamDown     = device.createStream();
        auto streamCompute1 = device.createStream();
        auto streamCompute2 = device.createStream();
        auto streamUp       = device.createStream();

        // TODO: prepare tensors https://jira.habana-labs.com/browse/SW-150330
        // std::shared_ptr<DataProvider>  dataProvider1  = std::make_shared<RandDataProvider>(0, 100.0,
        // recipe1.m_tensors); std::shared_ptr<DataCollector> dataCollector1 =
        // std::make_shared<DataCollector>(recipe1.m_recipe);
        //
        // std::shared_ptr<DataProvider>  dataProvider2  = std::make_shared<RandDataProvider>(0, 100.0,
        // recipe2.m_tensors); std::shared_ptr<DataCollector> dataCollector2 =
        // std::make_shared<DataCollector>(recipe2.m_recipe);

        TestLauncher launcher(device);

        RecipeLaunchParams recipeLaunchParam1 =
            std::move(launcher.createRecipeLaunchParams(recipe1, {TensorInitOp::RANDOM_POSITIVE, 0}));
        RecipeLaunchParams recipeLaunchParam2 =
            std::move(launcher.createRecipeLaunchParams(recipe2, {TensorInitOp::RANDOM_POSITIVE, 0}));
        TestLauncher::download(streamDown, recipe1, recipeLaunchParam1);
        TestLauncher::download(streamDown, recipe2, recipeLaunchParam2);
        device.synchronize();

        for (int i = 0; i < 10; i++)
        {
            TestLauncher::launch(streamCompute1, recipe1, recipeLaunchParam1);

            TestLauncher::launch(streamCompute2, recipe2, recipeLaunchParam2);
        }
        device.synchronize();

        TestLauncher::upload(streamUp, recipe1, recipeLaunchParam1);
        TestLauncher::upload(streamUp, recipe2, recipeLaunchParam2);
        device.synchronize();

        recipe1.validateResults(recipeLaunchParam1.getLaunchTensorMemory());
        recipe2.validateResults(recipeLaunchParam2.getLaunchTensorMemory());
    }
};

REGISTER_SUITE(SynCommonParallelLaunch, ALL_TEST_PACKAGES);

TEST_F_SYN(SynCommonParallelLaunch, gemm_only)
{
    std::vector<TSize> sizes = {64, 64};
    TestRecipeGemm     recipe(m_deviceType, sizes, false);
    execute(recipe);
}

TEST_F_SYN(SynCommonParallelLaunch, tpc_only)
{
    std::vector<TSize> sizes = {64, 64};
    TestRecipeAddf32   recipe(m_deviceType, sizes, false);
    execute(recipe);
}

TEST_F_SYN(SynCommonParallelLaunch, tpcGemm_only)
{
    std::vector<TSize>   sizes = {64, 64};
    TestRecipeAddf32Gemm recipe(m_deviceType, sizes, false);
    execute(recipe);
}

TEST_F_SYN(SynCommonParallelLaunch, tpcGemm_only_sections)
{
    std::vector<TSize>   sizes = {64, 64};
    TestRecipeAddf32GemmSections recipe(m_deviceType, sizes, false);
    execute(recipe);
}

TEST_F_SYN(SynCommonParallelLaunch, gemmTpc_only)
{
    std::vector<TSize>   sizes = {64, 64};
    TestRecipeGemmAddf32 recipe(m_deviceType, sizes, false);
    execute(recipe);
}

TEST_F_SYN(SynCommonParallelLaunch, tpcGemmMulti)
{
    std::vector<TSize>   sizes = {64, 64};  // {1536, 1536};
    TestRecipeAddf32Gemm recipe(m_deviceType, sizes, false);

    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    auto streamDown    = device.createStream();
    auto streamCompute = device.createStream();
    auto streamUp      = device.createStream();

    // TODO: prepare tensors https://jira.habana-labs.com/browse/SW-150330
    // std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0, recipe.m_tensors);
    // std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(recipe.m_recipe);

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParam =
        std::move(launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0}));

    TestLauncher::download(streamDown, recipe, recipeLaunchParam);
    device.synchronize();

    for (int i = 0; i < 10; i++)
    {
        TestLauncher::launch(streamCompute, recipe, recipeLaunchParam);
    }
    device.synchronize();

    TestLauncher::upload(streamUp, recipe, recipeLaunchParam);
    device.synchronize();

    recipe.validateResults(recipeLaunchParam.getLaunchTensorMemory());
}

TEST_F_SYN(SynCommonParallelLaunch, tpcGemm_tpc)
{
    std::vector<TSize>   sizes = {64, 64};  // {1536, 1536};
    TestRecipeAddf32Gemm recipe1(m_deviceType, sizes, false);
    TestRecipeAddf32     recipe2(m_deviceType, sizes, false);
    execute(recipe1, recipe2);
}

TEST_F_SYN(SynCommonParallelLaunch, gemmTpc_tpc)
{
    std::vector<TSize> sizesGemm = {64, 64};    //{1536, 1536};
    std::vector<TSize> sizesTpc  = {128, 128};  //{4096, 4096};

    TestRecipeGemmAddf32 recipe1(m_deviceType, sizesGemm, false);
    TestRecipeAddf32     recipe2(m_deviceType, sizesTpc, false);
    execute(recipe1, recipe2);
}

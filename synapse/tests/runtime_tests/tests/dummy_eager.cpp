//
// Created by esdeor on 10/19/22.
//

#include "syn_base_test.hpp"
#include "../recipes/test_recipe_addf32.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

class SynCommonDummyEager
: public SynBaseTest
, public testing::WithParamInterface<bool>
{
public:
    SynCommonDummyEager() { setSupportedDevices({synDeviceGaudi2}); }

    static constexpr int N = 100;
};

REGISTER_SUITE(SynCommonDummyEager, ALL_TEST_PACKAGES);

INSTANTIATE_TEST_SUITE_P(, SynCommonDummyEager, ::testing::Values(false, true));

// This test runs a lot of recipes. Each recipe has a different seqNum (but all are the same).
// It is used to simulate an eager scenario. It is used to simulate an eager scenario. It does N*compile + N*launch
TEST_P(SynCommonDummyEager, DISABLED_NcompileNLaunch)  // the test is for development only, no need to run on CI
{
    const bool eager = GetParam();

    const std::vector<TSize> sizes = {256};
    TestRecipeAddf32         recipeMaster(m_deviceType, sizes, eager);
    recipeMaster.generateRecipe();

    TestDevice device(m_deviceType);
    TestStream streamDown    = device.createStream();
    TestStream streamCompute = device.createStream();

    TestLauncher launcher(device);

    // TODO: prepare tensors https://jira.habana-labs.com/browse/SW-150330
    //    std::shared_ptr<DataProvider> dataProvider =
    //        std::make_shared<RandDataProvider>(0, 100.0, recipeMaster.m_tensors);
    //    std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(recipeMaster.m_recipe);

    // only one set of tensors for all recipes
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipeMaster, {TensorInitOp::RANDOM_POSITIVE, 0});
    TestLauncher::download(streamDown, recipeMaster, recipeLaunchParams);
    device.synchronize();

    std::vector<TestRecipeAddf32> addf32Recipes;
    addf32Recipes.reserve(N);

    for (int i = 0; i < N; i++)
    {
        addf32Recipes.emplace_back(m_deviceType, sizes, eager);
        addf32Recipes[i].compileGraph();
    }

    for (int i = 0; i < N; i++)
    {
        if (i % 1000 == 0) printf("%d\n", i);
        TestLauncher::launch(streamCompute, addf32Recipes[i], recipeLaunchParams);
    }
    device.synchronize();
}

// This test runs a lot of recipes. Each recipe has a different seqNum (but all are the same).
// It is used to simulate an eager scenario. It does N*(compile + run)
TEST_P(SynCommonDummyEager, DISABLED_Ncompile_and_launch)  // the test is for development only, no need to run on CI
{
    const bool eager = GetParam();

    const std::vector<TSize> size = {256};

    TestRecipeAddf32 recipeMaster(m_deviceType, size, eager);

    recipeMaster.compileGraph();

    // TODO: prepare tensors https://jira.habana-labs.com/browse/SW-150330
    //    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0,
    //    recipeMaster.m_tensors); std::shared_ptr<DataCollector> dataCollector =
    //    std::make_shared<DataCollector>(recipeMaster.m_recipe);

    TestDevice device(m_deviceType);

    auto         streamDown    = device.createStream();
    auto         streamCompute = device.createStream();
    TestLauncher launcher(device);

    // only one set of tensors for all recipes
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipeMaster, {TensorInitOp::RANDOM_POSITIVE, 0});
    TestLauncher::download(streamDown, recipeMaster, recipeLaunchParams);
    device.synchronize();

    std::vector<TestRecipeAddf32> addf32Recipes;
    addf32Recipes.reserve(N);

    for (int i = 0; i < N; i++)
    {
        if (i % 1000 == 0) printf("%d\n", i);

        addf32Recipes.emplace_back(m_deviceType, size, eager);
        addf32Recipes.back().compileGraph();
        TestLauncher::launch(streamCompute, addf32Recipes.back(), recipeLaunchParams);
    }

    device.synchronize();
}

class SynCommonDummyEagerThreaded : public SynCommonDummyEager
{
public:
    void launcher(TestStream& streamCompute, std::vector<TestRecipeAddf32>& recipes, RecipeLaunchParams& launchParams);

protected:
    int                     m_toRun;
    bool                    m_ready;
    bool                    m_processed;
    std::mutex              m_mtx;
    std::condition_variable m_cv;
};

REGISTER_SUITE(SynCommonDummyEagerThreaded, ALL_TEST_PACKAGES);

void SynCommonDummyEagerThreaded::launcher(TestStream&                    streamCompute,
                                           std::vector<TestRecipeAddf32>& recipes,
                                           RecipeLaunchParams&            launchParams)
{
    while (true)
    {
        int toRun;

        {
            std::unique_lock lk(m_mtx);
            m_cv.wait(lk, [this] { return m_ready; });
            toRun       = m_toRun;
            m_ready     = false;
            m_processed = true;
        }
        TestLauncher::launch(streamCompute, recipes[toRun], launchParams);

        m_cv.notify_one();

        if (toRun == (N - 1)) return;
    }
}

INSTANTIATE_TEST_SUITE_P(, SynCommonDummyEagerThreaded, ::testing::Values(false, true));

// This test runs a lot of recipes. Each recipe has a different seqNum (but all are the same).
// It is used to simulate an eager scenario. It does N*(compile thread1 + run thread2)
TEST_P(SynCommonDummyEagerThreaded,
       DISABLED_Ncompile_and_launch_threaded)  // the test is for development only, no need to run on CI
{
    const bool eager = GetParam();

    const std::vector<TSize> size = {256};

    TestRecipeAddf32 recipeMaster(m_deviceType, size, eager);

    recipeMaster.compileGraph();

    // TODO: prepare tensors https://jira.habana-labs.com/browse/SW-150330
    //    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0,
    //    recipeMaster.m_tensors); std::shared_ptr<DataCollector> dataCollector =
    //    std::make_shared<DataCollector>(recipeMaster.m_recipe);
    TestDevice device(m_deviceType);

    auto streamDown    = device.createStream();
    auto streamCompute = device.createStream();

    TestLauncher launcher(device);

    // only one set of tensors for all recipes
    RecipeLaunchParams launchParams =
        std::move(launcher.createRecipeLaunchParams(recipeMaster, {TensorInitOp::RANDOM_POSITIVE, 0}));
    TestLauncher::download(streamDown, recipeMaster, launchParams);
    device.synchronize();

    std::vector<TestRecipeAddf32> recipes;
    recipes.reserve(N);

    printf("start thread\n");
    std::thread worker(&SynCommonDummyEagerThreaded::launcher,
                       this,
                       std::ref(streamCompute),
                       std::ref(recipes),
                       std::ref(launchParams));

    for (int i = 0; i < N; i++)
    {
        if ((i % 1000) == 0)
        {
            printf("%d\n", i);
        }

        {
            std::lock_guard lk(m_mtx);
            recipes.emplace_back(m_deviceType, size, eager);
            recipes.back().compileGraph();
            m_ready = true;
            m_toRun = i;
        }
        m_cv.notify_one();

        // wait for the worker
        {
            std::unique_lock lk(m_mtx);
            m_cv.wait(lk, [this] { return m_processed; });
            m_processed = false;
        }
    }

    device.synchronize();

    worker.join();
}

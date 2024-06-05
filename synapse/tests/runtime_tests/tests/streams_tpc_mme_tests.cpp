#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

#include "test_recipe_tpc.hpp"
#include "test_recipe_gemm.hpp"

class StreamsTpcMmeTest : public SynBaseTest
{
public:
    StreamsTpcMmeTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    void recipesLauncher(TestDevice&          device,
                         TestStream&          streamA,
                         TestStream&          streamB,
                         TestRecipeInterface& recipeA,
                         TestRecipeInterface& recipeB,
                         unsigned             numOfLaunches,
                         unsigned             numOfTpcLaunches,
                         unsigned             numOfMmeLaunches);
};

REGISTER_SUITE(StreamsTpcMmeTest, ALL_TEST_PACKAGES);

void StreamsTpcMmeTest::recipesLauncher(TestDevice&          device,
                                        TestStream&          streamA,
                                        TestStream&          streamB,
                                        TestRecipeInterface& recipeA,
                                        TestRecipeInterface& recipeB,
                                        unsigned             numOfLaunches,
                                        unsigned             numOfTpcLaunches,
                                        unsigned             numOfMmeLaunches)
{
    TestLauncher launcher(device);

    RecipeLaunchParams recipeALaunchParams = launcher.createRecipeLaunchParams(recipeA, {TensorInitOp::CONST, -25});

    RecipeLaunchParams recipeBLaunchParams = launcher.createRecipeLaunchParams(recipeB, {TensorInitOp::CONST, 0xEE});

    // Copy all inputs to device
    launcher.download(streamA, recipeA, recipeALaunchParams);
    launcher.download(streamB, recipeB, recipeBLaunchParams);

    // synLaunch
    for (unsigned i = 0; i < numOfLaunches; i++)
    {
        // TPC recipe launch
        for (unsigned j = 0; j < numOfTpcLaunches; j++)
        {
            launcher.launch(streamA, recipeA, recipeALaunchParams);
        }

        // MME recipe launch
        for (unsigned j = 0; j < numOfMmeLaunches; j++)
        {
            launcher.launch(streamB, recipeB, recipeBLaunchParams);
        }
    }

    // Copy all outputs to host and synchronize
    launcher.upload(streamA, recipeA, recipeALaunchParams);
    launcher.upload(streamB, recipeB, recipeBLaunchParams);

    streamA.synchronize();
    if (streamA != streamB)
    {
        streamB.synchronize();
    }

    // Validate results and free memory allocations
    recipeA.validateResults(recipeALaunchParams.getLaunchTensorMemory());
    recipeB.validateResults(recipeBLaunchParams.getLaunchTensorMemory());
}

TEST_F_SYN(StreamsTpcMmeTest, single_stream)
{
    TestDevice device(m_deviceType);
    TestStream stream = device.createStream();

    TestRecipeTpc tpcRecipe(m_deviceType);
    tpcRecipe.generateRecipe();

    TestRecipeGemm mmeRecipe(m_deviceType, {16, 16});
    mmeRecipe.generateRecipe();

    recipesLauncher(device, stream, stream, tpcRecipe, mmeRecipe, 10, 2, 1);
}

TEST_F_SYN(StreamsTpcMmeTest, multi_stream)
{
    TestDevice device(m_deviceType);
    TestStream tpcStream = device.createStream();
    TestStream mmeStream = device.createStream();

    TestRecipeTpc tpcRecipe(m_deviceType);
    tpcRecipe.generateRecipe();

    TestRecipeGemm mmeRecipe(m_deviceType, {16, 16});
    mmeRecipe.generateRecipe();

    recipesLauncher(device, tpcStream, mmeStream, tpcRecipe, mmeRecipe, 2, 1, 1);
}

#include "syn_base_test.hpp"
#include "test_recipe_dma.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_recipe_launch_params.hpp"

class DmaTest : public SynBaseTest
{
public:
    DmaTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); };

    ~DmaTest() override = default;

protected:
    static void download(const TestStream&            rStream,
                         const TestRecipeInterface&   rRecipe,
                         const RecipeLaunchParamsVec& rRecipeLaunchParams);

    static void upload(const TestStream&            rStream,
                       const TestRecipeInterface&   rRecipe,
                       const RecipeLaunchParamsVec& rRecipeLaunchParams);

    void basic_dma_test(uint64_t numberOfLaunches);
};

REGISTER_SUITE(DmaTest, ALL_TEST_PACKAGES);

void DmaTest::download(const TestStream&            rStream,
                       const TestRecipeInterface&   rRecipe,
                       const RecipeLaunchParamsVec& rRecipeLaunchParamsVec)
{
    const uint64_t numberOfLaunches = rRecipeLaunchParamsVec.size();
    uint64_t       src[numberOfLaunches];
    uint64_t       size[numberOfLaunches];
    uint64_t       dst[numberOfLaunches];
    const unsigned tensorIndex = 0;
    for (uint64_t index = 0; index < numberOfLaunches; index++)
    {
        src[index]  = (uint64_t)rRecipeLaunchParamsVec[index].getHostInput(tensorIndex).getBuffer();
        size[index] = rRecipe.getTensorSizeInput(tensorIndex);
        dst[index]  = rRecipeLaunchParamsVec[index].getDeviceInput(tensorIndex).getBuffer();
    }

    rStream.memcopyAsyncMultiple(&src[0], &size[0], &dst[0], HOST_TO_DRAM, numberOfLaunches);
}

void DmaTest::upload(const TestStream&            rStream,
                     const TestRecipeInterface&   rRecipe,
                     const RecipeLaunchParamsVec& rRecipeLaunchParamsVec)
{
    const uint64_t numberOfLaunches = rRecipeLaunchParamsVec.size();
    uint64_t       src[numberOfLaunches];
    uint64_t       size[numberOfLaunches];
    uint64_t       dst[numberOfLaunches];
    const unsigned tensorIndex = 0;
    for (uint64_t index = 0; index < numberOfLaunches; index++)
    {
        src[index]  = rRecipeLaunchParamsVec[index].getDeviceOutput(tensorIndex).getBuffer();
        size[index] = rRecipe.getTensorSizeOutput(tensorIndex);
        dst[index]  = (uint64_t)rRecipeLaunchParamsVec[index].getHostOutput(tensorIndex).getBuffer();
    }

    rStream.memcopyAsyncMultiple(&src[0], &size[0], &dst[0], DRAM_TO_HOST, numberOfLaunches);
}

void DmaTest::basic_dma_test(uint64_t numberOfLaunches)
{
    TestRecipeDma recipe(m_deviceType, 16 * 1024U, 1024U, 0xEE, false, syn_type_uint8);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    RecipeLaunchParamsVec launchVec;
    for (int i = 0; i < numberOfLaunches; ++i)
    {
        launchVec.emplace_back(
            launcher.createRecipeLaunchParams(recipe, TensorInitInfo {TensorInitOp::RANDOM_WITH_NEGATIVE, 0}));
    }

    download(stream, recipe, launchVec);

    for (int i = 0; i < numberOfLaunches; ++i)
    {
        TestLauncher::launch(stream, recipe, launchVec[i]);
    }

    upload(stream, recipe, launchVec);

    stream.synchronize();

    for (int i = 0; i < numberOfLaunches; ++i)
    {
        recipe.validateResults(launchVec[i].getLaunchTensorMemory());
    }
}

TEST_F_SYN(DmaTest, basic_dma)
{
    basic_dma_test(1);
}

TEST_F_SYN(DmaTest, basic_dma_x10)
{
    basic_dma_test(10);
}

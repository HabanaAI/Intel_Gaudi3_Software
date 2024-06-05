#include "syn_base_test.hpp"
#include "test_recipe_relu.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

class SynGaudiTestInfraNoDram : public SynBaseTest
{
public:
    SynGaudiTestInfraNoDram() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2}); }
};

REGISTER_SUITE(SynGaudiTestInfraNoDram, synTestPackage::CI);

TEST_F_SYN(SynGaudiTestInfraNoDram, relu_forward)
{
    TestRecipeRelu recipe(m_deviceType);
    TestDevice     device(m_deviceType);
    TestLauncher   launcher(device);
    TestStream     stream = device.createStream();

    recipe.generateRecipe();

    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());
}
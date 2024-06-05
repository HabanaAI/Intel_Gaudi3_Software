#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_utils.h"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_recipe_split_shape_node.hpp"

class SynCommonShapeNodesTests : public SynBaseTest
{
public:
    SynCommonShapeNodesTests() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
};

REGISTER_SUITE(SynCommonShapeNodesTests, ALL_TEST_PACKAGES);

TEST_F_SYN(SynCommonShapeNodesTests, split_shape_node_no_persistent_tensors)
{
    TestDevice device(m_deviceType);

    TestRecipeSplitShapeNode recipe(m_deviceType);
    recipe.generateRecipe();

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();
}

#include "syn_base_test.hpp"
#include "test_recipe_gemm.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "habana_global_conf_runtime.h"
#include "test_device.hpp"
#include "test_launcher.hpp"

class MmeTest : public SynBaseTest
{
public:
    MmeTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }

protected:
    void gemmExecute();
};

REGISTER_SUITE(MmeTest, ALL_TEST_PACKAGES);

void MmeTest::gemmExecute()
{
    TestRecipeGemm recipe(m_deviceType, {16, 16});
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});
    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());
}

TEST_F_SYN(MmeTest, basic_mme)
{
    gemmExecute();
}

TEST_F_SYN(MmeTest, basic_mme_with_debug_pratial)
{
    GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD);
    gemmExecute();
}

TEST_F_SYN(MmeTest, basic_mme_with_debug_full)
{
    GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD |
                                                  COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD_POST |
                                                  COMPARE_RECIPE_ON_DEVICE_AFTER_LAUNCH);
    gemmExecute();
}

#include "recipe_handle_impl.hpp"

#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "syn_singleton.hpp"
#include "test_recipe_dsd_gemm.hpp"
#include "test_recipe_dsd_dma.hpp"
#include "dsd_recipes.h"

class DynamicMultiTests
: public DsdRecipeTestBase
, public testing::WithParamInterface<bool>
{
};

REGISTER_SUITE(DynamicMultiTests, ALL_TEST_PACKAGES);

INSTANTIATE_TEST_SUITE_P(, DynamicMultiTests, ::testing::Values(true, false));

TEST_P(DynamicMultiTests, multiGraphDSDtests)
{
    synConfigurationSet("EXP_FLAGS", "true");
    synConfigurationSet("CHECK_SECTION_OVERLAP_CHECK", "true");
    synConfigurationSet("GAUDI3_DSD", "true");

    if (GetParam())
    {
        synConfigurationSet("ENABLE_STAGED_SUBMISSION", "true");
    }

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    TestStream   stream = device.createStream();

    TestRecipeDsdGemm dsdGemmRecipe(m_deviceType,
                                    true /* isDynamic */,
                                    false /* isSharedInputSection */,
                                    {256, 128} /* op1Max */,
                                    {256, 64} /* op1Min */,
                                    {256, 256} /* op2Max */,
                                    {128, 256} /* op2Min */,
                                    {256, 128} /* outMax */,
                                    {128, 64} /* outMin */);
    dsdGemmRecipe.generateRecipe();

    TestRecipeDsdDma dsdDmaRecipe(m_deviceType,
                                  {4, 64, 2, 10} /* inputMax */,
                                  {4, 64, 2, 2} /* inputMin */,
                                  {4, 64, 2, 10} /* outputMax */,
                                  {4, 64, 2, 2} /* outputMin */);
    dsdDmaRecipe.generateRecipe();

    const std::pair<int, int> dsdPairs[4] = {{64, 2}, {128, 10}, {103, 5}, {90, 3}};

    for (const auto& dsdPair : dsdPairs)
    {
        TestTensorsDimensions gemmTestTensorsDimensions;
        TestTensorsDimensions dmaTestTensorsDimensions;

        unsigned gemmRecipeOp1Actual = dsdPair.first;
        unsigned gemmRecipeOp2Actual = gemmRecipeOp1Actual * 2;
        unsigned dmaRecipeOp1Actual  = dsdPair.second;

        {
            TensorDimensions tensorDims = {256, gemmRecipeOp1Actual};
            gemmTestTensorsDimensions.setDimensions(true, 0, tensorDims);
        }

        {
            TensorDimensions tensorDims = {gemmRecipeOp2Actual, 256};
            gemmTestTensorsDimensions.setDimensions(true, 1, tensorDims);
        }

        {
            TensorDimensions tensorDims = {gemmRecipeOp2Actual, gemmRecipeOp1Actual};
            gemmTestTensorsDimensions.setDimensions(false, 0, tensorDims);
        }

        {
            TensorDimensions tensorDims = {4, 64, 2, dmaRecipeOp1Actual};
            dmaTestTensorsDimensions.setDimensions(true, 0, tensorDims);
        }

        {
            TensorDimensions tensorDims = {4, 64, 2, dmaRecipeOp1Actual};
            dmaTestTensorsDimensions.setDimensions(false, 0, tensorDims);
        }

        RecipeLaunchParams recipeLaunchParamsGemm =
            launcher.createRecipeLaunchParams(dsdGemmRecipe,
                                              {TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
                                              gemmTestTensorsDimensions);

        RecipeLaunchParams recipeLaunchParamsDma =
            launcher.createRecipeLaunchParams(dsdDmaRecipe,
                                              {TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
                                              dmaTestTensorsDimensions);

        TestLauncher::execute(stream, dsdGemmRecipe, recipeLaunchParamsGemm);
        TestLauncher::execute(stream, dsdDmaRecipe, recipeLaunchParamsDma);

        stream.synchronize();

        dsdGemmRecipe.setExecutionDynamicSize(gemmRecipeOp1Actual, gemmRecipeOp2Actual);
        dsdDmaRecipe.setExecutionDynamicSize(dmaRecipeOp1Actual);

        dsdGemmRecipe.validateResults(recipeLaunchParamsGemm.getLaunchTensorMemory());
        dsdDmaRecipe.validateResults(recipeLaunchParamsDma.getLaunchTensorMemory());
    }
}

TEST_F_SYN(DynamicMultiTests, simpleTensorSize)
{
    synConfigurationSet("EXP_FLAGS", "true");
    synConfigurationSet("ENABLE_PROFILER", "true");

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    TestStream   stream = device.createStream();

    TestRecipeDsdGemm dsdFirstGemmRecipe(m_deviceType,
                                         true /* isDynamic */,
                                         false /* isSharedInputSection */,
                                         {256, 128} /* op1Max */,
                                         {256, 64} /* op1Min */,
                                         {256, 256} /* op2Max */,
                                         {128, 256} /* op2Min */,
                                         {256, 128} /* outMax */,
                                         {128, 64} /* outMin */);
    dsdFirstGemmRecipe.generateRecipe();
    synRecipeHandle originalRecipe = dsdFirstGemmRecipe.getRecipe();

    TestRecipeDsdGemm dsdSecondGemmRecipe(m_deviceType,
                                          true /* isDynamic */,
                                          false /* isSharedInputSection */,
                                          {256, 128} /* op1Max */,
                                          {256, 64} /* op1Min */,
                                          {256, 256} /* op2Max */,
                                          {128, 256} /* op2Min */,
                                          {256, 128} /* outMax */,
                                          {128, 64} /* outMin */);
    ASSERT_TRUE(dsdSecondGemmRecipe.recipeDeserialize()) << "Failed to deserialize recipe";
    synRecipeHandle deserializedRecipe = dsdSecondGemmRecipe.getRecipe();

    uint64_t numOfShapePlanTensors = originalRecipe->basicRecipeHandle.shape_plan_recipe->sp_tensors_nr;
    for (uint64_t i = 0; i < numOfShapePlanTensors; i++)
    {
        ASSERT_NE(originalRecipe->basicRecipeHandle.shape_plan_recipe->sp_tensors[i].tensor_info_name, nullptr);
        ASSERT_NE(deserializedRecipe->basicRecipeHandle.shape_plan_recipe->sp_tensors[i].tensor_info_name, nullptr);

        int cmp = strcmp(originalRecipe->basicRecipeHandle.shape_plan_recipe->sp_tensors[i].tensor_info_name,
                         deserializedRecipe->basicRecipeHandle.shape_plan_recipe->sp_tensors[i].tensor_info_name);
        ASSERT_EQ(cmp, 0);
    }

    unsigned              gemmRecipeOp1Actual = 64;
    unsigned              gemmRecipeOp2Actual = gemmRecipeOp1Actual * 2;
    TestTensorsDimensions gemmTestTensorsDimensions;

    {
        TensorDimensions tensorDims = {256, gemmRecipeOp1Actual};
        gemmTestTensorsDimensions.setDimensions(true, 0, tensorDims);
    }

    {
        TensorDimensions tensorDims = {gemmRecipeOp2Actual, 256};
        gemmTestTensorsDimensions.setDimensions(true, 1, tensorDims);
    }

    {
        TensorDimensions tensorDims = {gemmRecipeOp2Actual, gemmRecipeOp1Actual};
        gemmTestTensorsDimensions.setDimensions(false, 0, tensorDims);
    }

    RecipeLaunchParams recipeLaunchParamsGemm =
        launcher.createRecipeLaunchParams(dsdFirstGemmRecipe,
                                          {TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
                                          gemmTestTensorsDimensions);

    launcher.execute(stream, dsdFirstGemmRecipe, recipeLaunchParamsGemm);
    stream.synchronize();

    dsdFirstGemmRecipe.setExecutionDynamicSize(gemmRecipeOp1Actual, gemmRecipeOp2Actual);
    dsdFirstGemmRecipe.validateResults(recipeLaunchParamsGemm.getLaunchTensorMemory());

    std::vector<tensor_info_t> tensorInfoArr;
    _SYN_SINGLETON_INTERNAL->getDynamicShapesTensorInfoArrayV2(stream, originalRecipe, tensorInfoArr);

    ASSERT_EQ(tensorInfoArr.size(), originalRecipe->basicRecipeHandle.shape_plan_recipe->sp_tensors_nr)
        << "Tensor vector size mismatch";

    for (int i = 0; i < tensorInfoArr.size(); i++)
    {
        ASSERT_NE(tensorInfoArr[i].tensor_info_name, nullptr);
        LOG_TRACE(SYN_RT_TEST,
                  "{}: tensorInfoArr[{}], \n\tsizes: ({}, {}), \n\tmin/max: [{},{}], [{}, {}], \n\ttensor_db_index {}, "
                  "\n\ttype: {},\n\tpermutation: ({},{}),\n\tname: {};",
                  __func__,
                  i,
                  tensorInfoArr[i].infer_info.geometry.maxSizes[0],
                  tensorInfoArr[i].infer_info.geometry.maxSizes[1],
                  tensorInfoArr[i].min_dims[0],
                  tensorInfoArr[i].max_dims[0],
                  tensorInfoArr[i].min_dims[1],
                  tensorInfoArr[i].max_dims[1],
                  tensorInfoArr[i].tensor_db_index,
                  tensorInfoArr[i].tensor_type,
                  tensorInfoArr[i].permutation[0],
                  tensorInfoArr[i].permutation[1],
                  tensorInfoArr[i].tensor_info_name);

        EXPECT_EQ(tensorInfoArr[i].permutation[0], 0);
        EXPECT_EQ(tensorInfoArr[i].permutation[1], 1);
    }

    uint64_t t_type, t_size_0, t_size_1;

    // predicate to find tensor by type and inferred sizes
    auto is_equal = [&](tensor_info_t t) {
        return (t.tensor_type == t_type) && (t.infer_info.geometry.maxSizes[0] == t_size_0) && (t.infer_info.geometry.maxSizes[1] == t_size_1);
    };

    // looking for the persistent tensors (not necessarily in order)
    t_type       = 0;
    t_size_0     = 256;
    t_size_1     = 64;
    auto findRes = std::find_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    EXPECT_NE(findRes, tensorInfoArr.end());

    t_size_0 = 128;
    t_size_1 = 256;
    findRes  = std::find_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    EXPECT_NE(findRes, tensorInfoArr.end());

    t_size_0 = 128;
    t_size_1 = 64;
    findRes  = std::find_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    EXPECT_NE(findRes, tensorInfoArr.end());

    // looking for the internal tensors (not necessarily in order)
    t_type         = 2;
    auto findCount = std::count_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    if (m_deviceType == synDeviceGaudi) EXPECT_EQ(findCount, 2);
    else if (m_deviceType == synDeviceGaudi2)
        EXPECT_EQ(findCount, 1);
    else
        LOG_TRACE(SYN_RT_TEST, "{}: not supporting device Type {}", __func__, m_deviceType);

    t_size_1  = 256;
    findCount = std::count_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    if (m_deviceType == synDeviceGaudi) EXPECT_EQ(findCount, 3);
    else if (m_deviceType == synDeviceGaudi2)
        EXPECT_EQ(findCount, 1);
    else
        LOG_TRACE(SYN_RT_TEST, "{}: not supporting device Type {}", __func__, m_deviceType);

    t_size_0  = 0;
    findCount = std::count_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    if (m_deviceType == synDeviceGaudi) EXPECT_EQ(findCount, 2);
    else if (m_deviceType == synDeviceGaudi2)
        EXPECT_EQ(findCount, 0);
    else
        LOG_TRACE(SYN_RT_TEST, "{}: not supporting device Type {}", __func__, m_deviceType);

    t_size_1  = 64;
    findCount = std::count_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    if (m_deviceType == synDeviceGaudi) EXPECT_EQ(findCount, 1);
    else if (m_deviceType == synDeviceGaudi2)
        EXPECT_EQ(findCount, 0);
    else
        LOG_TRACE(SYN_RT_TEST, "{}: not supporting device Type {}", __func__, m_deviceType);

    t_size_0  = 256;
    findCount = std::count_if(begin(tensorInfoArr), end(tensorInfoArr), is_equal);
    if (m_deviceType == synDeviceGaudi) EXPECT_EQ(findCount, 2);
    else if (m_deviceType == synDeviceGaudi2)
        EXPECT_EQ(findCount, 1);
    else
        LOG_TRACE(SYN_RT_TEST, "{}: not supporting device Type {}", __func__, m_deviceType);
    // total of 13 tensors found, as expected For Gaudi1 and 6 for Gaudi2
}

TEST_F_SYN(DynamicMultiTests, analyzeDsdTensor)
{
    GCFG_CHECK_SECTION_OVERLAP.setValue(true);

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    TestStream   stream = device.createStream();

    TestRecipeDsdGemm dsdGemmRecipe(m_deviceType,
                                    true /* isDynamic */,
                                    true /* isSharedInputSection */,
                                    {256, 128} /* op1Max */,
                                    {256, 64} /* op1Min */,
                                    {256, 256} /* op2Max */,
                                    {128, 256} /* op2Min */,
                                    {256, 128} /* outMax */,
                                    {128, 64} /* outMin */);
    dsdGemmRecipe.generateRecipe();
    ASSERT_NE(dsdGemmRecipe.getRecipe()->basicRecipeHandle.recipe, nullptr);
    ASSERT_NE(dsdGemmRecipe.getRecipe()->basicRecipeHandle.shape_plan_recipe, nullptr);

    TestTensorsDimensions gemmTestTensorsDimensions;

    // Set actual size to half, now should pass as the seoond input tensor is half the size and the output tensor is not overlapping
    unsigned gemmRecipeOp1Actual = 64;
    unsigned gemmRecipeOp2Actual = gemmRecipeOp1Actual * 2;

    {
        TensorDimensions tensorDims = {256, gemmRecipeOp1Actual};
        gemmTestTensorsDimensions.setDimensions(true, 0, tensorDims);
    }

    {
        TensorDimensions tensorDims = {gemmRecipeOp2Actual, 256};
        gemmTestTensorsDimensions.setDimensions(true, 1, tensorDims);
    }

    {
        TensorDimensions tensorDims = {gemmRecipeOp2Actual, gemmRecipeOp1Actual};
        gemmTestTensorsDimensions.setDimensions(false, 0, tensorDims);
    }

    RecipeLaunchParams launchParams = launcher.createRecipeLaunchParams(dsdGemmRecipe,
                                                                        {TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
                                                                        gemmTestTensorsDimensions);

    launcher.execute(stream, dsdGemmRecipe, launchParams);
    stream.synchronize();

    dsdGemmRecipe.setExecutionDynamicSize(gemmRecipeOp1Actual, gemmRecipeOp2Actual);
    dsdGemmRecipe.validateResults(launchParams.getLaunchTensorMemory());
}

TEST_F_SYN(DynamicMultiTests, analyzeDsdTensorOverlap)
{
    GCFG_CHECK_SECTION_OVERLAP.setValue(true);

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    TestStream   stream = device.createStream();

    TestRecipeDsdGemm dsdGemmRecipe(m_deviceType,
                                    true       /* isDynamic */,
                                    true       /* isSharedInputSection */,
                                    {256, 128} /* op1Max */,
                                    {256, 64}  /* op1Min */,
                                    {256, 256} /* op2Max */,
                                    {128, 256} /* op2Min */,
                                    {256, 128} /* outMax */,
                                    {128, 64}  /* outMin */);
    dsdGemmRecipe.generateRecipe();
    ASSERT_NE(dsdGemmRecipe.getRecipe()->basicRecipeHandle.recipe, nullptr);
    ASSERT_NE(dsdGemmRecipe.getRecipe()->basicRecipeHandle.shape_plan_recipe, nullptr);

    unsigned gemmRecipeOp1Actual = 128;
    unsigned gemmRecipeOp2Actual = gemmRecipeOp1Actual * 2;
    TestTensorsDimensions  gemmTestTensorsDimensions;

    {
        TensorDimensions tensorDims = {256, gemmRecipeOp1Actual};
        gemmTestTensorsDimensions.setDimensions(true, 0, tensorDims);
    }

    {
        TensorDimensions  tensorDims = {gemmRecipeOp2Actual, 256};
        gemmTestTensorsDimensions.setDimensions(true, 1, tensorDims);
    }

    {
        TensorDimensions tensorDims = {gemmRecipeOp2Actual, gemmRecipeOp1Actual};
        gemmTestTensorsDimensions.setDimensions(false, 0, tensorDims);
    }

    RecipeLaunchParams launchParams = launcher.createRecipeLaunchParams(dsdGemmRecipe,
                                                                        {TensorInitOp::RANDOM_WITH_NEGATIVE, 0},
                                                                        gemmTestTensorsDimensions);

    std::vector<synLaunchTensorInfoExt> launchTensorsInfo = launchParams.getSynLaunchTensorInfoVec();
    // set the output tensor address to the middle of the input second tensor. Should fail on overlap
    launchTensorsInfo[2].pTensorAddress = launchTensorsInfo[1].pTensorAddress +
                                          dsdGemmRecipe.getTensorInfoInput(1)->m_tensorSize / 2;

    synStatus status = synLaunchExt(stream.operator synStreamHandle(),
                                    launchTensorsInfo.data(),
                                    dsdGemmRecipe.getTensorInfoVecSize(),
                                    launchParams.getWorkspace(),
                                    dsdGemmRecipe.getRecipe(),
                                    0);
    ASSERT_EQ(status, synFailedSectionValidation);
}

TEST_F_SYN(DynamicMultiTests, DsdRecipeSerialization)
{
    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    TestStream   stream = device.createStream();

    TestRecipeDsdGemm dsdGemmRecipe(m_deviceType,
                                    true /* isDynamic */,
                                    false /* isSharedInputSection */,
                                    {256, 128} /* op1Max */,
                                    {256, 64} /* op1Min */,
                                    {256, 256} /* op2Max */,
                                    {128, 256} /* op2Min */,
                                    {256, 128} /* outMax */,
                                    {128, 64} /* outMin */);
    dsdGemmRecipe.generateRecipe();

    synRecipeHandle originalRecipe = dsdGemmRecipe.getRecipe();

    synStatus status = synRecipeSerialize(originalRecipe, "DsdRecipeSerialization");
    ASSERT_EQ(status, synSuccess) << "Could not serialize original recipe";

    originalRecipe->basicRecipeHandle.shape_plan_recipe->sp_nodes[0].basic_nodes[0].sif_version++;

    status = synRecipeSerialize(originalRecipe, "DsdRecipeSerialization");
    ASSERT_EQ(status, synSuccess) << "Could not serialize original recipe 2";

    synRecipeHandle copyRecipe {};
    status = synRecipeDeSerialize(&copyRecipe, "DsdRecipeSerialization");
    ASSERT_EQ(status, synUnsupported) << "Should fail on bad sif version";
}

#include "test_device.hpp"
#include "test_recipe_relu.hpp"
#include "test_recipe_rnd_relu.hpp"
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "test_recipe_broadcast.hpp"
#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_tensors_container.hpp"

class FlowTest : public SynBaseTest
{
public:
    FlowTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    void collectAllPatchPointsNodeIds(synRecipeHandle recipeHandle);

    void validatePatchPointsNodeIds(synRecipeHandle recipeHandle);

    enum elementType_e
    {
        ELEMENT_TYPE_8,
        ELEMENT_TYPE_16,
        ELEMENT_TYPE_32
    };

    void testLinDmaMemset(uint32_t      numberOfElements,
                          uint64_t      validationValue,
                          uint32_t      elementSize,
                          elementType_e elementType);
    void testBroadcast(TestSizes size, TestSizes broadcastOutSize, uint32_t dims, bool isFcdMultipleDimsBroadcast);
    void checkBuffer(void* buffer, uint32_t numberOfElements, uint32_t expectedValue, elementType_e elementType);

protected:
    std::map<uint32_t, uint64_t> m_nodeIdToPPs;
};

REGISTER_SUITE(FlowTest, synTestPackage::CI, synTestPackage::SIM, synTestPackage::ASIC, synTestPackage::ASIC_CI);

void FlowTest::collectAllPatchPointsNodeIds(synRecipeHandle recipeHandle)
{
    // collect all patch points and their associated node execution indices - we will verify it after deserialization
    recipe_t* currRecipe = recipeHandle->basicRecipeHandle.recipe;

    for (uint64_t patch_index = 0; patch_index < currRecipe->patch_points_nr; patch_index++)
    {
        uint32_t nodeId = currRecipe->patch_points[patch_index].node_exe_index;
        if (m_nodeIdToPPs.find(nodeId) == m_nodeIdToPPs.end())
        {
            m_nodeIdToPPs[nodeId] = 0;
        }
        m_nodeIdToPPs[nodeId]++;
    }

    // make sure we have more than the default node execution index "0"
    ASSERT_TRUE(m_nodeIdToPPs.size() > 1);

    // verify each node is associated with at least one patch point
    std::map<uint32_t, uint64_t>::iterator it;
    for (it = m_nodeIdToPPs.begin(); it != m_nodeIdToPPs.end(); ++it)
    {
        ASSERT_TRUE(it->second > 0);
    }
}

void FlowTest::validatePatchPointsNodeIds(synRecipeHandle recipeHandle)
{
    std::map<uint32_t, uint64_t> nodeIdToPPsAfterDeserialization;
    recipe_t*                    currRecipe = recipeHandle->basicRecipeHandle.recipe;

    for (uint64_t patch_index = 0; patch_index < currRecipe->patch_points_nr; patch_index++)
    {
        uint32_t nodeId = currRecipe->patch_points[patch_index].node_exe_index;
        if (nodeIdToPPsAfterDeserialization.find(nodeId) == nodeIdToPPsAfterDeserialization.end())
        {
            nodeIdToPPsAfterDeserialization[nodeId] = 0;
        }
        nodeIdToPPsAfterDeserialization[nodeId]++;
    }

    ASSERT_EQ(m_nodeIdToPPs.size(), nodeIdToPPsAfterDeserialization.size());

    std::map<uint32_t, uint64_t>::iterator it;
    for (it = nodeIdToPPsAfterDeserialization.begin(); it != nodeIdToPPsAfterDeserialization.end(); ++it)
    {
        ASSERT_TRUE(m_nodeIdToPPs.find(it->first) != m_nodeIdToPPs.end());
        ASSERT_EQ(it->second, m_nodeIdToPPs[it->first]);
    }
}

// This test is using the new API (with temporal API for its enqueue).
// Hence, it does not use the Gaudi tests's INFRA
TEST_F_SYN(FlowTest, full_relu_flow)
{
    TestRecipeRelu recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::NONE, 0});

    void*     data   = nullptr;
    synStatus status = synHostMalloc(device.getDeviceId(), recipe.getTensorSizeInput(0), 0, &data);

    float* typed_data = static_cast<float*>(data);

    bool inputFileReadStatus =
        recipe.read_file(recipe.getPathPrefix() + "worker_0_bn1_output", typed_data, recipe.getTensorSizeInput(0));
    ASSERT_TRUE(inputFileReadStatus) << "Failed to read input-file";
    stream.memcopyAsync((uint64_t)data,
                        recipe.getTensorSizeInput(0),
                        recipeLaunchParams.getDeviceInput(0).getBuffer(),
                        HOST_TO_DRAM);

    stream.launch(recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                  recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                  recipeLaunchParams.getWorkspace(),
                  recipe.getRecipe(),
                  0);

    stream.memcopyAsync(recipeLaunchParams.getDeviceOutput(0).getBuffer(),
                        recipe.getTensorSizeInput(0),
                        (uint64_t)recipeLaunchParams.getHostOutput(0).getBuffer(),
                        DRAM_TO_HOST);

    stream.synchronize();

    status = synDeviceSynchronize(device.getDeviceId());
    ASSERT_EQ(status, synSuccess) << "synDeviceSynchronize fails when event already concluded";

    status = synStreamQuery(synStreamHandle(stream));
    ASSERT_EQ(status, synSuccess) << "StreamQuery fails when event already concluded";

    recipe.validateResultsWithFile(recipeLaunchParams.getLaunchTensorMemory());
}

TEST_F_SYN(FlowTest, invalid_non_presistant_inputs)
{
    synStatus       status = synSuccess;
    synGraphHandle  graphHandle;
    synRecipeHandle recipeHandle;

    status = synGraphCreate(&graphHandle, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph (A)";

    const TSize inputTensorSizes[]  = {64, 112, 112, 64};
    const TSize outputTensorSizes[] = {64, 112, 112, 64};

    const char* AtensorName = "A_0_1";
    const char* BtensorName = "B_0_1";

    TestTensorsContainer in_tensor(1 /* numOfTensors */), out_tensor(1 /* numOfTensors */);

    TestRecipeInterface::createTrainingTensor(in_tensor,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              inputTensorSizes,
                                              false,
                                              AtensorName,
                                              graphHandle,
                                              nullptr,
                                              false,
                                              0,
                                              nullptr,
                                              DATA_TENSOR,
                                              nullptr /* minTensorSize */);
    TestRecipeInterface::createTrainingTensor(out_tensor,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              outputTensorSizes,
                                              true,
                                              BtensorName,
                                              graphHandle,
                                              nullptr,
                                              false,
                                              0,
                                              nullptr,
                                              DATA_TENSOR,
                                              nullptr /* minTensorSize */);

    status = synNodeCreate(graphHandle,
                           in_tensor.tensors(),
                           out_tensor.tensors(),
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_f32",
                           "worker_0_relu",
                           nullptr,
                           nullptr);

    ASSERT_EQ(status, synSuccess) << "Failed to create Convolution Node";

    status = synGraphCompile(&recipeHandle, graphHandle, "invalid_non_presistant_inputs", 0);
    ASSERT_EQ(status, synFail) << "Compilation should fail in this case";

    ASSERT_EQ(synGraphDestroy(graphHandle), synSuccess) << "Failed to destroy graph";
}

void FlowTest::checkBuffer(void* buffer, uint32_t numberOfElements, uint32_t expectedValue, elementType_e elementType)
{
    if (elementType == ELEMENT_TYPE_8)
    {
        for (uint32_t i = 0; i < numberOfElements; i++)
        {
            ASSERT_EQ(((uint8_t*)buffer)[i], expectedValue) << "Got wrong value from device";
        }
    }
    else if (elementType == ELEMENT_TYPE_16)
    {
        for (uint32_t i = 0; i < numberOfElements; i++)
        {
            ASSERT_EQ(((uint16_t*)buffer)[i], expectedValue) << "Got wrong value from device";
        }
    }
    else if (elementType == ELEMENT_TYPE_32)
    {
        for (uint32_t i = 0; i < numberOfElements; i++)
        {
            ASSERT_EQ(((uint32_t*)buffer)[i], expectedValue) << "Got wrong value from device";
        }
    }
    else
    {
        ASSERT_TRUE(false) << "wrong element type " << elementType;
    }
}

TEST_F_SYN(FlowTest, lin_DMA_large_tensor_ASIC_CI)
{
    uint32_t size         = 1130537984;
    void*    hostBuffer   = nullptr;
    void*    deviceBuffer = nullptr;

    synStatus status;

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    status = synHostMalloc(device.getDeviceId(), size, 0, &hostBuffer);
    EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory";

    memset(hostBuffer, 1, size * sizeof(uint8_t));

    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, (uint64_t*)&deviceBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    stream.memcopyAsync((uint64_t)hostBuffer, size, (uint64_t)deviceBuffer, HOST_TO_DRAM);

    stream.synchronize();

    memset(hostBuffer, 0, size * sizeof(uint8_t));

    stream.memcopyAsync((uint64_t)deviceBuffer, size, (uint64_t)hostBuffer, DRAM_TO_HOST);

    stream.synchronize();

    checkBuffer(hostBuffer, size, 1, ELEMENT_TYPE_8);
}

void FlowTest::testLinDmaMemset(uint32_t      numberOfElements,
                                uint64_t      validationValue,
                                uint32_t      elementSize,
                                elementType_e elementType)
{
    uint32_t  size         = (numberOfElements + 1) * elementSize;
    void*     hostBuffer   = nullptr;
    void*     deviceBuffer = nullptr;
    synStatus status;

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    status = synHostMalloc(device.getDeviceId(), size, 0, &hostBuffer);
    EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory";

    memset(hostBuffer, 0, size);

    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, (uint64_t*)&deviceBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";
    switch (elementType)
    {
        case ELEMENT_TYPE_8:
            status = synMemsetD8Async((uint64_t)deviceBuffer, 0, numberOfElements + 1, stream);
            break;
        case ELEMENT_TYPE_16:
            status = synMemsetD16Async((uint64_t)deviceBuffer, 0, numberOfElements + 1, stream);
            break;
        case ELEMENT_TYPE_32:
            status = synMemsetD32Async((uint64_t)deviceBuffer, 0, numberOfElements + 1, stream);
            break;
        default:
            ASSERT_TRUE(false) << "wrong element type " << elementType;
    }
    ASSERT_EQ(status, synSuccess) << "Failed to memset Device HBM memory";

    stream.synchronize();

    stream.memcopyAsync((uint64_t)deviceBuffer, size, (uint64_t)hostBuffer, DRAM_TO_HOST);

    stream.synchronize();

    checkBuffer(hostBuffer, numberOfElements + 1, 0, elementType);

    switch (elementType)
    {
        case ELEMENT_TYPE_8:
            status = synMemsetD8Async((uint64_t)deviceBuffer, validationValue, numberOfElements, stream);
            break;
        case ELEMENT_TYPE_16:
            status = synMemsetD16Async((uint64_t)deviceBuffer, validationValue, numberOfElements, stream);
            break;
        case ELEMENT_TYPE_32:
            status = synMemsetD32Async((uint64_t)deviceBuffer, validationValue, numberOfElements, stream);
            break;
        default:
            ASSERT_TRUE(false) << "wrong element type " << elementType;
    }
    ASSERT_EQ(status, synSuccess) << "Failed to memset Device HBM memory";

    stream.synchronize();

    stream.memcopyAsync((uint64_t)deviceBuffer, size, (uint64_t)hostBuffer, DRAM_TO_HOST);

    stream.synchronize();

    checkBuffer(hostBuffer, numberOfElements, validationValue, elementType);
    switch (elementType)
    {
        case ELEMENT_TYPE_8:
            ASSERT_EQ(((uint8_t*)hostBuffer)[numberOfElements], 0) << "Got overrun";
            break;
        case ELEMENT_TYPE_16:
            ASSERT_EQ(((uint16_t*)hostBuffer)[numberOfElements], 0) << "Got overrun";
            break;
        case ELEMENT_TYPE_32:
            ASSERT_EQ(((uint32_t*)hostBuffer)[numberOfElements], 0) << "Got overrun";
            break;
        default:
            ASSERT_TRUE(false) << "wrong element type " << elementType;
    }

    status = synDeviceFree(device.getDeviceId(), (uint64_t)deviceBuffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Free device buffer";

    status = synHostFree(device.getDeviceId(), hostBuffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Free Host buffer";
}

TEST_F_SYN(FlowTest, lin_DMA_memsetD8)
{
    uint32_t numberOfElements = 1000000;
    uint8_t  validationValue  = 0xA5;
    uint32_t elementSize      = sizeof(uint8_t);
    testLinDmaMemset(numberOfElements, validationValue, elementSize, ELEMENT_TYPE_8);
}

TEST_F_SYN(FlowTest, lin_DMA_memsetD16)
{
    uint32_t numberOfElements = 200000000;
    uint16_t validationValue  = 0xA57E;
    uint32_t elementSize      = sizeof(uint16_t);
    testLinDmaMemset(numberOfElements, validationValue, elementSize, ELEMENT_TYPE_16);
}

TEST_F_SYN(FlowTest, lin_DMA_memsetD32)
{
    uint32_t numberOfElements = 100000000;
    uint32_t validationValue  = 0xA57E183C;
    uint32_t elementSize      = sizeof(uint32_t);
    testLinDmaMemset(numberOfElements, validationValue, elementSize, ELEMENT_TYPE_32);
}

TEST_F_SYN(FlowTest, lin_DMA_memcpy_multiple_ASIC_CI)
{
    const uint32_t basic_size = 1024 * 1024;  // 1M
    const uint32_t numChunks  = 30;

    uint64_t  size[numChunks], sizeWithCheck[numChunks];
    void*     hostRandomBuffer[numChunks] = {0};
    void*     hostBuffer[numChunks]       = {0};
    void*     deviceBuffer[numChunks]     = {0};
    synStatus status;

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    for (unsigned i = 0; i < numChunks; i++)
    {
        size[i]          = (i + 1) * basic_size;
        sizeWithCheck[i] = size[i] + 1;  // additional byte to check for overrun
    }

    for (unsigned i = 0; i < numChunks; i++)
    {
        status = synHostMalloc(device.getDeviceId(), sizeWithCheck[i], 0, &hostBuffer[i]);
        EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory";

        memset((void*)hostBuffer[i], 0, size[i]);

        status = synHostMalloc(device.getDeviceId(), sizeWithCheck[i], 0, &hostRandomBuffer[i]);
        EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory";

        memset((void*)hostRandomBuffer[i], 0x5A + i, size[i] + 1);

        status = synDeviceMalloc(device.getDeviceId(), sizeWithCheck[i], 0, 0, (uint64_t*)&deviceBuffer[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

        status = synMemsetD8Async((uint64_t)deviceBuffer[i], 0, sizeWithCheck[i], stream);
        ASSERT_EQ(status, synSuccess) << "Failed to memset Device HBM memory";
    }

    stream.synchronize();

    const uint64_t* randomPtr = (uint64_t*)hostRandomBuffer;
    const uint64_t* devPtr    = (uint64_t*)deviceBuffer;

    // copy to device
    status = synMemCopyAsyncMultiple(stream, randomPtr, size, devPtr, HOST_TO_DRAM, numChunks);
    ASSERT_EQ(status, synSuccess) << "Failed to copy to the device";

    stream.synchronize();

    // copy back to host
    const uint64_t* hostPtr = (uint64_t*)hostBuffer;
    status                  = synMemCopyAsyncMultiple(stream, devPtr, sizeWithCheck, hostPtr, DRAM_TO_HOST, numChunks);

    ASSERT_EQ(status, synSuccess) << "Failed to copy to the host";

    stream.synchronize();

    for (unsigned i = 0; i < numChunks; i++)
    {
        uint8_t* hostValues = (uint8_t*)hostBuffer[i];
        uint8_t* randValues = (uint8_t*)hostRandomBuffer[i];

        for (unsigned j = 0; j < sizeWithCheck[i]; j++)
        {
            if (j == sizeWithCheck[i] - 1)
            {
                // last
                ASSERT_EQ(hostValues[j], 0) << "Got wrong value from device last index in " << i;
            }
            else
            {
                ASSERT_EQ(hostValues[j], randValues[j]) << "Got wrong value from device " << i << "- " << j;
            }
        }
    }
}

void FlowTest::testBroadcast(TestSizes size, TestSizes broadcastOutSize, uint32_t dims, bool isFcdMultipleDimsBroadcast)
{
    TestRecipeBroadcast recipe(m_deviceType, size, broadcastOutSize, dims, isFcdMultipleDimsBroadcast);

    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    TestLauncher::download(stream, recipe, recipeLaunchParams);
    TestLauncher::launch(stream, recipe, recipeLaunchParams);
    TestLauncher::upload(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());
}

TEST_F_SYN(FlowTest, fcd_broadcast)
{
    TestSizes size             = {1, 1, 1, 1, 1};
    TestSizes broadcastOutSize = {10, 1, 1, 1, 1};
    testBroadcast(size, broadcastOutSize, 1, false);
}

TEST_F_SYN(FlowTest, 4dim_broadcast)
{
    TestSizes size             = {1, 1, 1, 1, 1};
    TestSizes broadcastOutSize = {10, 10, 10, 10, 1};
    testBroadcast(size, broadcastOutSize, 4, false);
}

TEST_F_SYN(FlowTest, fcd_broadcast_4dim_tensor)
{
    TestSizes size             = {1, 10, 10, 1, 1};
    TestSizes broadcastOutSize = {10, 10, 10, 1, 1};
    testBroadcast(size, broadcastOutSize, 4, true);
}

TEST_F_SYN(FlowTest, fcd_broadcast_5dim_tensor)
{
    TestSizes size             = {1, 10, 10, 2, 2};
    TestSizes broadcastOutSize = {10, 10, 10, 2, 2};
    testBroadcast(size, broadcastOutSize, 5, true);
}

TEST_F_SYN(FlowTest, workspace_no_reuse_test)
{
    TestRecipeRndRelu recipe(m_deviceType);
    recipe.generateRecipe();

    uint8_t validationVal = 0xFF;

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    TestEvent event = device.createEvent(0);

    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    synLaunchTensorInfoExt outTensor    = {"output",
                                        recipeLaunchParams.getDeviceOutput(0).getBuffer(),
                                        DATA_TENSOR,
                                        {0, 0, 0, 0, 0}};
    synLaunchTensorInfoExt conTensors[] = {outTensor};

    stream.launch(conTensors, 1, recipeLaunchParams.getWorkspace(), recipe.getRecipe(), 0);

    uint64_t workspaceAddr, workspaceSize;
    uint64_t workspaceAddr2;

    workspaceAddr    = recipeLaunchParams.getWorkspace();
    synStatus status = synWorkspaceGetSize(&workspaceSize, recipe.getRecipe());
    ASSERT_EQ(status, synSuccess) << "Failed to synWorkspaceGetSize";

    status = synMemsetD8Async(workspaceAddr, validationVal, workspaceSize, stream);
    ASSERT_EQ(status, synSuccess) << "Failed to memset workspace";

    status = synDeviceMalloc(device.getDeviceId(), workspaceSize, 0, 0, &workspaceAddr2);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace";

    stream.launch(conTensors, 1, workspaceAddr2, recipe.getRecipe(), 0);

    void* hostWorkspaceBuffer;
    status = synHostMalloc(device.getDeviceId(), workspaceSize, 0, &hostWorkspaceBuffer);
    ASSERT_EQ(synSuccess, status) << "Failed to allocate device memory";

    stream.memcopyAsync(workspaceAddr, workspaceSize, (uint64_t)hostWorkspaceBuffer, DRAM_TO_HOST);
    stream.synchronize();

    auto pVal = reinterpret_cast<uint8_t*>(hostWorkspaceBuffer);
    for (int i = 0; i < workspaceSize; i++)
    {
        ASSERT_EQ(pVal[i], validationVal) << "Launch used wrong workspace";
    }
}

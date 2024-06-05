#include "node_factory.h"

#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "syn_singleton.hpp"
#include "test_recipe_tpc.hpp"
#include "test_recipe_dma.hpp"
#include "test_recipe_dma_gemm.hpp"

#include <future>

class MultipleThreadsTests
: public SynBaseTest
, public testing::WithParamInterface<bool>
{
public:
    MultipleThreadsTests() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    synStatus synDeviceMallocCall(TestDevice*    device,
                                  const uint64_t size,
                                  uint64_t       requestedAddress,
                                  const uint32_t flags,
                                  uint64_t*      buffer);
    synStatus synHostMallocCall(TestDevice* device, const uint64_t size, const uint32_t flags, void** buffer);

    synStatus synHostMapCall(TestDevice* device, const uint64_t size, const void* buffer);
    synStatus synHostUnmapCall(TestDevice* device, const void* buffer);

    synStatus synDeviceFreeCall(TestDevice* device, const uint64_t buffer, const uint32_t flags);
    synStatus synHostFreeCall(TestDevice* device, const void* buffer, const uint32_t flags);

    void downloadTensorData(TestStream* testStreamDown, void* data, uint64_t tensorAddr, unsigned sizeBytes);
    void uploadTensorData(TestStream* testStreamUp, uint64_t tensorAddr, void* data, unsigned sizeBytes);
    void memcpyTensorData(TestStream* testStream, uint64_t src, uint64_t dst, unsigned sizeBytes, synDmaDir direction);

    bool isSyncRequired(synStreamHandle streamA, synStreamHandle streamB) { return streamA != streamB; }

    void compile_and_launch_simple(TestDevice*   device,
                                   TestStream*   testStreamDown,
                                   TestStream*   testStreamUp,
                                   TestStream*   testStreamCompute,
                                   TestLauncher* launcher,
                                   uint32_t      threadID);

    void launch_simple(synStreamHandle      stream,
                       TestRecipeInterface* recipe,
                       RecipeLaunchParams*  launchParams,
                       bool                 ignoreResult);

    void stream_destroy_simple(synStreamHandle* streamHandlesList, unsigned streamIndex, unsigned numOfDestroyStream);

    void compile_launch_destory_recipe(TestDevice* device,
                                       TestStream* testStreamUp,
                                       TestStream* testStreamCompute,
                                       uint32_t    threadID,
                                       uint64_t    numberOfTimes);

    void alloc_and_free_memory(TestDevice* device, uint32_t threadIndex);
};

REGISTER_SUITE(MultipleThreadsTests, ALL_TEST_PACKAGES);

synStatus MultipleThreadsTests::synDeviceMallocCall(TestDevice*    device,
                                                    const uint64_t size,
                                                    uint64_t       requestedAddress,
                                                    const uint32_t flags,
                                                    uint64_t*      buffer)
{
    return synDeviceMalloc(device->getDeviceId(), size, requestedAddress, flags, buffer);
}

synStatus
MultipleThreadsTests::synHostMallocCall(TestDevice* device, const uint64_t size, const uint32_t flags, void** buffer)
{
    return synHostMalloc(device->getDeviceId(), size, flags, (void**)buffer);
}

synStatus MultipleThreadsTests::synHostMapCall(TestDevice* device, const uint64_t size, const void* buffer)
{
    return synHostMap(device->getDeviceId(), size, buffer);
}

synStatus MultipleThreadsTests::synHostUnmapCall(TestDevice* device, const void* buffer)
{
    return synHostUnmap(device->getDeviceId(), buffer);
}

synStatus MultipleThreadsTests::synDeviceFreeCall(TestDevice* device, const uint64_t buffer, const uint32_t flags)
{
    return synDeviceFree(device->getDeviceId(), buffer, flags);
}

synStatus MultipleThreadsTests::synHostFreeCall(TestDevice* device, const void* buffer, const uint32_t flags)
{
    return synHostFree(device->getDeviceId(), buffer, flags);
}

void MultipleThreadsTests::downloadTensorData(TestStream* testStreamDown,
                                              void*       data,
                                              uint64_t    tensorAddr,
                                              unsigned    sizeBytes)
{
    memcpyTensorData(testStreamDown, (uint64_t)data, tensorAddr, sizeBytes, HOST_TO_DRAM);
}

void MultipleThreadsTests::uploadTensorData(TestStream* testStreamUp,
                                            uint64_t    tensorAddr,
                                            void*       data,
                                            unsigned    sizeBytes)
{
    memcpyTensorData(testStreamUp, tensorAddr, (uint64_t)data, sizeBytes, DRAM_TO_HOST);
}

void MultipleThreadsTests::memcpyTensorData(TestStream* testStream,
                                            uint64_t    src,
                                            uint64_t    dst,
                                            unsigned    sizeBytes,
                                            synDmaDir   direction)
{
    testStream->memcopyAsync(src, sizeBytes, dst, direction);
    testStream->synchronize();
}

void MultipleThreadsTests::compile_launch_destory_recipe(TestDevice* device,
                                                         TestStream* testStreamUp,
                                                         TestStream* testStreamCompute,
                                                         uint32_t    threadID,
                                                         uint64_t    numberOfTimes)
{
    unsigned elemNumb =
        (threadID % 2 == 0) ? GCFG_MAX_CONST_TENSOR_SIZE_BYTES.value() / dataTypeSizeInBytes(syn_type_float) : 8000;

    // unique_ptr is used since there is no proper move ctor for TestRecipeBase
    std::vector<std::unique_ptr<TestRecipeDma>> recipes;
    TestLauncher                                launcher(*device);

    synEventHandle eventHandle;
    synStatus      status = synEventCreate(&eventHandle, device->getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create event";

    // First iteration, generate first recipe launch params and use it as a reference for other recipes
    recipes.emplace_back(
        std::make_unique<TestRecipeDma>(m_deviceType, elemNumb, 1U, 0.7, true, syn_type_float, threadID, 0));
    recipes.back()->compileGraph();

    RecipeLaunchParams launchParams =
        launcher.createRecipeLaunchParams(*recipes.back(), {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});
    testStreamCompute->launch(launchParams.getSynLaunchTensorInfoVec().data(),
                              launchParams.getSynLaunchTensorInfoVec().size(),
                              launchParams.getWorkspace(),
                              recipes.back()->getRecipe(),
                              0);

    for (int i = 1; i < numberOfTimes; ++i)
    {
        recipes.emplace_back(
            std::make_unique<TestRecipeDma>(m_deviceType, elemNumb, 1U, 0.7, true, syn_type_float, threadID, i));
        recipes.back()->compileGraph();

        SynLaunchTensorInfoVec synLaunchTensorInfo;
        launcher.generateLaunchTensorsWithTensorsMemory(*recipes.back(),
                                                        launchParams.getLaunchTensorMemory(),
                                                        synLaunchTensorInfo,
                                                        {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

        testStreamCompute->launch(synLaunchTensorInfo.data(),
                                  synLaunchTensorInfo.size(),
                                  launchParams.getWorkspace(),
                                  recipes.back()->getRecipe(),
                                  0);

        if (recipes.size() > 20)
        {
            recipes.pop_back();
        }
    }

    if (isSyncRequired(*testStreamCompute, *testStreamUp))
    {
        status = synEventRecord(eventHandle, *testStreamCompute);
        ASSERT_EQ(status, synSuccess) << "Failed record-event (enqueue)";

        status = synStreamWaitEvent(*testStreamUp, eventHandle, 0);
        ASSERT_EQ(status, synSuccess) << "Failed stream-wait-event (completion of enqueue)";
    }
    testStreamUp->memcopyAsync(launchParams.getDeviceOutput(0).getBuffer(),
                               recipes[recipes.size() - 1]->getTensorSizeOutput(0),
                               (uint64_t)(launchParams).getHostOutput(0).getBuffer(),
                               DRAM_TO_HOST);
    testStreamUp->synchronize();

    recipes[recipes.size() - 1]->validateResults(launchParams.getLaunchTensorMemory());

    status = synEventDestroy(eventHandle);
    ASSERT_EQ(status, synSuccess) << "Failed destroy event";
}

void MultipleThreadsTests::compile_and_launch_simple(TestDevice*   device,
                                                     TestStream*   testStreamDown,
                                                     TestStream*   testStreamUp,
                                                     TestStream*   testStreamCompute,
                                                     TestLauncher* launcher,
                                                     uint32_t      threadID)
{
    synEventHandle eventHandle;
    synStatus      status = synEventCreate(&eventHandle, device->getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create event";

    unsigned          matrixSize = (threadID % 5) + 2;
    TestRecipeDmaGemm recipeDmaGemm(m_deviceType, matrixSize, matrixSize);
    recipeDmaGemm.generateRecipe();
    RecipeLaunchParams recipeLaunchParams =
        launcher->createRecipeLaunchParams(recipeDmaGemm, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    for (unsigned inputIndex = 0; inputIndex < recipeDmaGemm.getTensorInfoVecSizeInput(); inputIndex++)
    {
        const void* pHostInput = recipeLaunchParams.getHostInput(inputIndex).getBuffer();
        ASSERT_NE(pHostInput, nullptr) << "Failed to get host-buffer (for launchIndex " << 0 << " inputIndex "
                                       << inputIndex << ")";

        const auto& deviceInput = recipeLaunchParams.getDeviceInput(inputIndex);

        // Copy data from host to device
        testStreamDown->memcopyAsync((uint64_t)pHostInput,
                                     recipeDmaGemm.getTensorSizeInput(inputIndex),
                                     deviceInput.getBuffer(),
                                     HOST_TO_DRAM);
    }

    if (isSyncRequired(*testStreamDown, *testStreamCompute))
    {
        status = synEventRecord(eventHandle, *testStreamDown);
        ASSERT_EQ(status, synSuccess) << "Failed record-event (copy to the device)";

        status = synStreamWaitEvent(*testStreamCompute, eventHandle, 0);
        ASSERT_EQ(status, synSuccess) << "Failed stream-wait-event (completion of copy to the device)";
    }

    testStreamCompute->launch(recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipeDmaGemm.getRecipe(),
                              0);

    if (isSyncRequired(*testStreamCompute, *testStreamUp))
    {
        status = synEventRecord(eventHandle, *testStreamCompute);
        ASSERT_EQ(status, synSuccess) << "Failed record-event (enqueue)";

        status = synStreamWaitEvent(*testStreamUp, eventHandle, 0);
        ASSERT_EQ(status, synSuccess) << "Failed stream-wait-event (completion of enqueue)";
    }

    for (unsigned outputIndex = 0; outputIndex < recipeDmaGemm.getTensorInfoVecSizeOutput(); outputIndex++)
    {
        const void* pHostOutput = recipeLaunchParams.getHostOutput(outputIndex).getBuffer();
        ASSERT_NE(pHostOutput, nullptr) << "Failed to get host-buffer (for launchIndex " << 0 << " outputIndex "
                                        << outputIndex << ")";

        const auto& deviceOutput = recipeLaunchParams.getDeviceOutput(outputIndex);

        // Copy data from host to device
        testStreamUp->memcopyAsync(deviceOutput.getBuffer(),
                                   recipeDmaGemm.getTensorSizeOutput(outputIndex),
                                   (uint64_t)pHostOutput,
                                   DRAM_TO_HOST);
    }

    // waiting for the completion of last operation
    testStreamUp->synchronize();
    recipeDmaGemm.validateResults(recipeLaunchParams.getLaunchTensorMemory());

    status = synEventDestroy(eventHandle);
    ASSERT_EQ(status, synSuccess) << "Failed destroy event";
}

void MultipleThreadsTests::launch_simple(synStreamHandle      stream,
                                         TestRecipeInterface* recipe,
                                         RecipeLaunchParams*  launchParams,
                                         bool                 ignoreResult)
{
    synStatus status = synLaunchExt(stream,
                                    launchParams->getSynLaunchTensorInfoVec().data(),
                                    launchParams->getSynLaunchTensorInfoVec().size(),
                                    launchParams->getWorkspace(),
                                    recipe->getRecipe(),
                                    0);
    if (!ignoreResult)
    {
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";
    }
}

void MultipleThreadsTests::stream_destroy_simple(synStreamHandle* streamHandlesList,
                                                 unsigned         streamIndex,
                                                 unsigned         numOfDestroyStream)
{
    for (unsigned i = streamIndex; i < streamIndex + numOfDestroyStream; i++)
    {
        synStreamDestroy(streamHandlesList[i]);
    }
}

void MultipleThreadsTests::alloc_and_free_memory(TestDevice* device, uint32_t threadIndex)
{
    synStatus      status;
    const uint64_t size  = 100;
    const uint32_t flags = 0;

    unsigned numberOfIter = 50;
    unsigned iter         = 0;
    while (iter < numberOfIter)
    {
        if (threadIndex % 2 == 0)
        {
            void* hostBuffer;

            synStatus status = synHostMalloc(device->getDeviceId(), size, flags, &hostBuffer);
            EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory for persistent tensor";

            status = synHostFree(device->getDeviceId(), hostBuffer, flags);
            EXPECT_EQ(status, synSuccess) << "Failed to free host memory for persistent tensor";
        }
        else
        {
            uint64_t deviceAddr;
            uint64_t requestedAddress = 0;

            status = synDeviceMalloc(device->getDeviceId(), size, flags, requestedAddress, &deviceAddr);
            EXPECT_EQ(status, synSuccess) << "Failed to allocate device memory for persistent tensor";

            status = synDeviceFree(device->getDeviceId(), deviceAddr, flags);
            EXPECT_EQ(status, synSuccess) << "Failed to free device memory for persistent tensor";
        }

        iter++;
    }
}

INSTANTIATE_TEST_SUITE_P(, MultipleThreadsTests, ::testing::Values(true, false));

TEST_P(MultipleThreadsTests, memory_tests_L2)
{
    TestDevice device(m_deviceType);

    synDeviceInfo  deviceInfo;
    synStatus      status          = synSuccess;
    const uint64_t allocationSize  = 100;
    const unsigned numberOfThreads = 110;

    std::vector<std::future<synStatus>> deviceMallocThreads;
    std::vector<std::future<synStatus>> hostMallocThreads;

    std::vector<std::future<synStatus>> hostMapThreads;
    std::vector<std::future<synStatus>> hostUnMapThreads;

    std::vector<std::future<synStatus>> deviceFreeThreads;
    std::vector<std::future<synStatus>> hostFreeThreads;

    status = synDeviceGetInfo(device.getDeviceId(), &deviceInfo);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    LOG_TRACE(SYN_API,
              "dramSize {} deviceId {} deviceType {} dramBaseAddress {} dramEnabled {} sramSize {} tpcEnabledMask {} "
              "dramBaseAddress {}",
              deviceInfo.dramSize,
              deviceInfo.deviceId,
              deviceInfo.deviceType,
              deviceInfo.dramBaseAddress,
              deviceInfo.dramEnabled,
              deviceInfo.sramSize,
              deviceInfo.tpcEnabledMask,
              deviceInfo.dramBaseAddress);

    uint64_t deviceArrayBuffer[numberOfThreads];
    void*    hostArrayBuffer[numberOfThreads];

    // Allocating memory on device/host on multi-threads
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<synStatus> th1 = std::async(&MultipleThreadsTests::synDeviceMallocCall,
                                                this,
                                                &device,
                                                allocationSize,
                                                0,
                                                0,
                                                &deviceArrayBuffer[threadIndex]);
        deviceMallocThreads.push_back(std::move(th1));

        std::future<synStatus> th2 = std::async(&MultipleThreadsTests::synHostMallocCall,
                                                this,
                                                &device,
                                                allocationSize,
                                                0,
                                                &hostArrayBuffer[threadIndex]);
        hostMallocThreads.push_back(std::move(th2));
    }

    // Checking return values for allocating on device/host on multi-threads
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        status = deviceMallocThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc failed";

        status = hostMallocThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synHostMalloc failed";
    }

    // Checking for logical values for the memory allocated
    for (unsigned i = 0; i < numberOfThreads; i++)
        for (unsigned k = i + 1; k < numberOfThreads; k++)
            ASSERT_NE(deviceArrayBuffer[i], deviceArrayBuffer[k]) << "Allocating same address by synDeviceMalloc";

    for (unsigned i = 0; i < numberOfThreads; i++)
        for (unsigned k = i + 1; k < numberOfThreads; k++)
            ASSERT_NE(hostArrayBuffer[i], hostArrayBuffer[k]) << "Allocating host address by synHostMalloc";

    // Mapping host memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<synStatus> th1 = std::async(&MultipleThreadsTests::synHostMapCall,
                                                this,
                                                &device,
                                                allocationSize,
                                                hostArrayBuffer[threadIndex]);
        hostMapThreads.push_back(std::move(th1));
    }

    // Checking Mapping host memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        status = hostMapThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc failed";
    }

    // Checking if Virtual addresses make sense
    uint64_t* hostVirtualAddress[numberOfThreads];
    // saving the host address before mapping so we could free later
    uint64_t* hostAddress[numberOfThreads];
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        uint64_t* hostVA             = new uint64_t();
        hostAddress[threadIndex]     = hostVA;
        eMappingStatus mappingStatus = _SYN_SINGLETON_INTERNAL->_getDeviceVirtualAddress(true,
                                                                                         hostArrayBuffer[threadIndex],
                                                                                         allocationSize,
                                                                                         hostVA);
        ASSERT_EQ(mappingStatus, HATVA_MAPPING_STATUS_FOUND) << "Faied to find mapping";

        hostVirtualAddress[threadIndex] = hostVA;
    }

    for (unsigned i = 0; i < numberOfThreads; i++)
        for (unsigned k = i + 1; k < numberOfThreads; k++)
            ASSERT_NE(hostVirtualAddress[i], hostVirtualAddress[k]) << "Virtual adderss is the same";

    // UnMap host memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads / 2; threadIndex++)
    {
        std::future<synStatus> th1 =
            std::async(&MultipleThreadsTests::synHostUnmapCall, this, &device, hostArrayBuffer[threadIndex]);
        hostUnMapThreads.push_back(std::move(th1));
    }

    // Checking UnMap host memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads / 2; threadIndex++)
    {
        status = hostUnMapThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc failed";
    }

    // Freeing device memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<synStatus> th1 =
            std::async(&MultipleThreadsTests::synDeviceFreeCall, this, &device, deviceArrayBuffer[threadIndex], 0);
        deviceFreeThreads.push_back(std::move(th1));
    }

    // Freeing host memory that was not unmapped
    for (unsigned threadIndex = numberOfThreads / 2 + 1; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<synStatus> th2 =
            std::async(&MultipleThreadsTests::synHostFreeCall, this, &device, hostArrayBuffer[threadIndex], 0);
        hostFreeThreads.push_back(std::move(th2));
    }

    // Freeing host memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        if (hostAddress[threadIndex] != nullptr)
        {
            delete hostAddress[threadIndex];
        }
    }

    // Checking Freeing device memory
    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        status = deviceFreeThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc failed";
    }

    // Checking Freeing host memory
    for (unsigned threadIndex = 0; threadIndex < hostFreeThreads.size(); threadIndex++)
    {
        status = hostFreeThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc failed";
    }
}

TEST_P_SYN(MultipleThreadsTests, memcpy_tests)
{
    TestDevice device(m_deviceType);

    std::shared_ptr<TestStream> testStreamDown;
    std::shared_ptr<TestStream> testStreamUp;

    bool isGenericStreamMode = GetParam();
    if (isGenericStreamMode)
    {
        testStreamDown = std::make_shared<TestStream>(device.createStream());
        testStreamUp   = testStreamDown;
    }
    else
    {
        testStreamDown = std::make_shared<TestStream>(device.createStream());
        testStreamUp   = std::make_shared<TestStream>(device.createStream());
    }

    unsigned       inputSize        = 40 * 1024;
    unsigned       inputSizeInBytes = inputSize * sizeof(float);
    const unsigned numberOfThreads  = 105;

    std::vector<std::future<synStatus>> deviceMallocThreads;
    std::vector<std::future<synStatus>> outputMallocThreads;
    std::vector<std::future<void>>      memcpyThreads;

    void*     data[numberOfThreads];
    uint64_t  inputDram[numberOfThreads];
    void*     outputArray[numberOfThreads];
    synStatus status = synSuccess;

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<synStatus> th1 = std::async(&MultipleThreadsTests::synHostMallocCall,
                                                this,
                                                &device,
                                                inputSizeInBytes,
                                                0,
                                                &data[threadIndex]);
        deviceMallocThreads.push_back(std::move(th1));

        std::future<synStatus> th2 = std::async(&MultipleThreadsTests::synHostMallocCall,
                                                this,
                                                &device,
                                                inputSizeInBytes,
                                                0,
                                                &outputArray[threadIndex]);
        outputMallocThreads.push_back(std::move(th2));

        std::future<synStatus> th3 = std::async(&MultipleThreadsTests::synDeviceMallocCall,
                                                this,
                                                &device,
                                                inputSizeInBytes,
                                                0,
                                                0,
                                                &inputDram[threadIndex]);
        deviceMallocThreads.push_back(std::move(th3));
    }

    for (unsigned threadIndex = 0; threadIndex < deviceMallocThreads.size(); threadIndex++)
    {
        status = deviceMallocThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc/synHostMalloc failed";
    }
    deviceMallocThreads.clear();

    status = synFail;
    for (unsigned threadIndex = 0; threadIndex < outputMallocThreads.size(); threadIndex++)
    {
        status = outputMallocThreads[threadIndex].get();
        ASSERT_EQ(status, synSuccess) << "synDeviceMalloc/synHostMalloc failed";
    }
    outputMallocThreads.clear();

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        float* typed_data = static_cast<float*>(data[threadIndex]);
        for (int i = 0; i < inputSize; ++i)
        {
            typed_data[i] = ((float)i - 0.5);
        }
    }

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&MultipleThreadsTests::downloadTensorData,
                                              this,
                                              testStreamDown.get(),
                                              (void*)data[threadIndex],
                                              inputDram[threadIndex],
                                              inputSizeInBytes);
        memcpyThreads.push_back(std::move(thread));
    }

    for (unsigned threadIndex = 0; threadIndex < memcpyThreads.size(); threadIndex++)
    {
        memcpyThreads[threadIndex].wait();
    }
    memcpyThreads.clear();

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&MultipleThreadsTests::uploadTensorData,
                                              this,
                                              testStreamUp.get(),
                                              inputDram[threadIndex],
                                              outputArray[threadIndex],
                                              inputSizeInBytes);
        memcpyThreads.push_back(std::move(thread));
    }

    for (unsigned threadIndex = 0; threadIndex < memcpyThreads.size(); threadIndex++)
    {
        memcpyThreads[threadIndex].wait();
    }
    memcpyThreads.clear();

    for (int i = 0; i < numberOfThreads; ++i)
    {
        int n;
        n = memcmp(outputArray[i], data[i], inputSizeInBytes);
        if (n != 0)
        {
            std::cout << "Fail on i = " << i << std::endl;
            ASSERT_TRUE(n == 0 && "Validation failed");
        }
    }
}

TEST_P(MultipleThreadsTests, parallel_test_a_simple)
{
    const unsigned                 numberOfThreads = 20;
    std::vector<std::future<void>> basicTestThreads;

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);

    std::shared_ptr<TestStream> streamHandleDown;
    std::shared_ptr<TestStream> streamHandleUp;
    std::shared_ptr<TestStream> streamHandleCompute;

    bool isGenericStreamMode = GetParam();
    if (isGenericStreamMode)
    {
        streamHandleDown = std::make_shared<TestStream>(device.createStream());
        streamHandleUp = streamHandleCompute = streamHandleDown;
    }
    else
    {
        streamHandleDown    = std::make_shared<TestStream>(device.createStream());
        streamHandleUp      = std::make_shared<TestStream>(device.createStream());
        streamHandleCompute = std::make_shared<TestStream>(device.createStream());
    }

    for (uint32_t threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&MultipleThreadsTests::compile_and_launch_simple,
                                              this,
                                              &device,
                                              streamHandleDown.get(),
                                              streamHandleUp.get(),
                                              streamHandleCompute.get(),
                                              &launcher,
                                              threadIndex);
        basicTestThreads.push_back(std::move(thread));
    }

    for (unsigned threadIndex = 0; threadIndex < basicTestThreads.size(); threadIndex++)
    {
        basicTestThreads[threadIndex].wait();
    }
}

TEST_P(MultipleThreadsTests, compile_launch_destory_recipe)
{
    const unsigned numberOfThreads             = 20;
    const unsigned numberOfIterationsPerThread = 50;

    std::vector<std::future<void>> basicTestThreads;

    TestDevice device(m_deviceType);

    std::shared_ptr<TestStream> testStreamUp;
    std::shared_ptr<TestStream> testStreamCompute;

    bool isGenericStreamMode = GetParam();
    if (isGenericStreamMode)
    {
        testStreamUp      = std::make_shared<TestStream>(device.createStream());
        testStreamCompute = testStreamUp;
    }
    else
    {
        testStreamUp      = std::make_shared<TestStream>(device.createStream());
        testStreamCompute = std::make_shared<TestStream>(device.createStream());
    }

    for (uint32_t threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&MultipleThreadsTests::compile_launch_destory_recipe,
                                              this,
                                              &device,
                                              testStreamUp.get(),
                                              testStreamCompute.get(),
                                              threadIndex,
                                              numberOfIterationsPerThread);
        basicTestThreads.push_back(std::move(thread));
    }
    for (unsigned threadIndex = 0; threadIndex < basicTestThreads.size(); threadIndex++)
    {
        basicTestThreads[threadIndex].wait();
    }
}

TEST_P(MultipleThreadsTests, parallel_user_allocations)
{
    const unsigned                 numberOfThreads = 40;
    std::vector<std::future<void>> basicTestThreads;

    TestDevice device(m_deviceType);

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&MultipleThreadsTests::alloc_and_free_memory, this, &device, threadIndex);
        basicTestThreads.push_back(std::move(thread));
    }

    for (unsigned threadIndex = 0; threadIndex < basicTestThreads.size(); threadIndex++)
    {
        basicTestThreads[threadIndex].wait();
    }
}

// pathing-info issue
TEST_P(MultipleThreadsTests, one_compile_parallel_launch)
{
    std::vector<std::future<void>> launchThreads;
    const unsigned                 numberOfThreads = 110;

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);

    std::shared_ptr<TestStream> streamHandleDown;
    std::shared_ptr<TestStream> streamHandleUp;
    std::shared_ptr<TestStream> streamHandleCompute;

    TestRecipeDmaGemm recipeDmaGemm(m_deviceType, 1, 1);
    recipeDmaGemm.generateRecipe();
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipeDmaGemm, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    bool isGenericStreamMode = GetParam();
    if (isGenericStreamMode)
    {
        streamHandleDown = std::make_shared<TestStream>(device.createStream());
        streamHandleUp = streamHandleCompute = streamHandleDown;
    }
    else
    {
        streamHandleDown    = std::make_shared<TestStream>(device.createStream());
        streamHandleUp      = std::make_shared<TestStream>(device.createStream());
        streamHandleCompute = std::make_shared<TestStream>(device.createStream());
    }

    for (unsigned inputIndex = 0; inputIndex < recipeDmaGemm.getTensorInfoVecSizeInput(); inputIndex++)
    {
        const void* pHostInput = recipeLaunchParams.getHostInput(inputIndex).getBuffer();
        ASSERT_NE(pHostInput, nullptr) << "Failed to get host-buffer (for launchIndex " << 0 << " inputIndex "
                                       << inputIndex << ")";

        const auto& deviceInput = recipeLaunchParams.getDeviceInput(inputIndex);

        // Copy data from host to device
        streamHandleDown->memcopyAsync((uint64_t)pHostInput,
                                       recipeDmaGemm.getTensorSizeInput(inputIndex),
                                       deviceInput.getBuffer(),
                                       HOST_TO_DRAM);
    }

    // Wait for download to finish
    streamHandleDown->synchronize();

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        // synStreamHandle handle = *streamHandleCompute;
        std::future<void> thread = std::async(&MultipleThreadsTests::launch_simple,
                                              this,
                                              (synStreamHandle)(*streamHandleCompute),
                                              &recipeDmaGemm,
                                              &(recipeLaunchParams),
                                              false);
        launchThreads.push_back(std::move(thread));
    }

    for (unsigned threadIndex = 0; threadIndex < launchThreads.size(); threadIndex++)
    {
        launchThreads[threadIndex].wait();
    }

    // Wait for enqueues to finish
    streamHandleCompute->synchronize();

    for (unsigned outputIndex = 0; outputIndex < recipeDmaGemm.getTensorInfoVecSizeOutput(); outputIndex++)
    {
        const void* pHostOutput = recipeLaunchParams.getHostOutput(outputIndex).getBuffer();
        ASSERT_NE(pHostOutput, nullptr) << "Failed to get host-buffer (for launchIndex " << 0 << " outputIndex "
                                        << outputIndex << ")";

        const auto& deviceOutput = recipeLaunchParams.getDeviceOutput(outputIndex);

        // Copy data from device to host
        streamHandleUp->memcopyAsync(deviceOutput.getBuffer(),
                                     recipeDmaGemm.getTensorSizeOutput(outputIndex),
                                     (uint64_t)pHostOutput,
                                     DRAM_TO_HOST);
    }

    // Wait for everything to finish by blocking on the copy from device to host
    streamHandleUp->synchronize();
    recipeDmaGemm.validateResults(recipeLaunchParams.getLaunchTensorMemory());
}

TEST_P(MultipleThreadsTests, multiple_streams_parallel_launch_and_destroy)
{
    const unsigned numberOfLaunchThreads  = 1000;
    const unsigned numberOfDestroyThreads = 10;

    TestRecipeTpc recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    synStreamHandle streamHandles[numberOfLaunchThreads];

    for (unsigned i = 0; i < numberOfLaunchThreads; i++)
    {
        // Create a stream for each recipe
        ASSERT_EQ(synStreamCreateGeneric(&streamHandles[i], device.getDeviceId(), 0), synSuccess)
            << "Failed to create stream";
    }

    TestLauncher launcher(device);

    std::vector<RecipeLaunchParams> recipeLaunchParamsVec;
    for (unsigned i = 0; i < numberOfLaunchThreads; i++)
    {
        auto recipeLaunchParam = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 25});

        recipeLaunchParamsVec.push_back(std::move(recipeLaunchParam));
    }

    std::vector<std::future<void>> basicTestThreads;

    // Start compute and destroy threads
    unsigned numOfStreamsToDestroy = numberOfLaunchThreads / numberOfDestroyThreads;
    unsigned streamIndex           = 0;

    for (uint32_t threadIndex = 0; threadIndex < numberOfLaunchThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&MultipleThreadsTests::launch_simple,
                                              this,
                                              streamHandles[threadIndex],
                                              &recipe,
                                              &recipeLaunchParamsVec[threadIndex],
                                              true);
        basicTestThreads.push_back(std::move(thread));

        if (threadIndex % numOfStreamsToDestroy == 0)
        {
            std::future<void> thread = std::async(&MultipleThreadsTests::stream_destroy_simple,
                                                  this,
                                                  streamHandles,
                                                  streamIndex,
                                                  numOfStreamsToDestroy);
            basicTestThreads.push_back(std::move(thread));
            streamIndex += numOfStreamsToDestroy;
        }
    }

    for (unsigned threadIndex = 0; threadIndex < basicTestThreads.size(); threadIndex++)
    {
        basicTestThreads[threadIndex].wait();
    }
}

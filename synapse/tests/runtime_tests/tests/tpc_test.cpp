#include "syn_base_test.hpp"
#include "test_recipe_tpc.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

class TpcTest : public SynBaseTest
{
public:
    TpcTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }

protected:
    static synStatus query_events(synEventHandle dmaToHostDone, synEventHandle tpcComputeDone);
};

REGISTER_SUITE(TpcTest, ALL_TEST_PACKAGES);

synStatus TpcTest::query_events(synEventHandle dmaToHostDone, synEventHandle tpcComputeDone)
{
    synStatus status;

    bool     computeJobDone = false;
    bool     dmaJobDone     = false;
    uint64_t iter           = 0;
    while (!computeJobDone || !dmaJobDone)
    {
        if (!dmaJobDone)
        {
            status = synEventQuery(dmaToHostDone);
            if (status == synSuccess)
            {
                std::cout << "DMA job done\n";
                dmaJobDone = true;
            }
        }
        if (!computeJobDone)
        {
            status = synEventQuery(tpcComputeDone);
            if (status == synSuccess)
            {
                std::cout << "compute job done\n";
                computeJobDone = true;
            }
        }
        sleep(1);
        iter++;
        std::cout << "query_events() iter " << iter << "status: DMA " << dmaJobDone << " Compute: " << computeJobDone
                  << std::endl;
    }
    return status;
}

TEST_F_SYN(TpcTest, basic_tpc)
{
    TestRecipeTpc recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 25});
    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());
}

TEST_F_SYN(TpcTest, dma_before_tpc_test)
{
    TestRecipeTpc recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    // Input and output need to be mapped to the device as they are copied from / to
    const uint64_t dmaSize = 16 * 1024 * 1024;
    uint8_t*       pDmaInput;
    uint8_t*       pDmaOutput;
    auto           status = synHostMalloc(device.getDeviceId(), dmaSize, 0, (void**)&pDmaInput);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), dmaSize, 0, (void**)&pDmaOutput);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Init input with random values and zero-out the output
    // std::pair<uint8_t, uint8_t> range = {-0, 100};
    // fillWithRandom<uint8_t>(input, size, range);
    memset(pDmaInput, 0xEE, dmaSize);
    memset(pDmaOutput, 0, dmaSize);

    // Create DMA streams
    synStreamHandle dmaInStream;
    status = synStreamCreateGeneric(&dmaInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    synStreamHandle dmaOutStream;
    status = synStreamCreateGeneric(&dmaOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t dmaDeviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), dmaSize, 0, 0, &dmaDeviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    synEventHandle dmaToHostDone;
    status = synEventCreate(&dmaToHostDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // Copy data from host to device
    status = synMemCopyAsync(dmaInStream, (uint64_t)pDmaInput, dmaSize, dmaDeviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
    std::cout << "basic_dma_start done\n";

    // Associate an event with its completion
    status = synEventRecord(dmaToHostDone, dmaInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 25});
    auto const& launchParams = recipeLaunchParams;

    // Create streams
    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;

    for (unsigned inputIndex = 0; inputIndex < recipe.getTensorInfoVecSizeInput(); inputIndex++)
    {
        const auto& hostInput   = launchParams.getHostInput(inputIndex);
        const auto& deviceInput = launchParams.getDeviceInput(inputIndex);

        // Copy data from host to device
        status = synMemCopyAsync(streamHandle,
                                 (uint64_t)hostInput.getBuffer(),
                                 recipe.getTensorSizeInput(inputIndex),
                                 deviceInput.getBuffer(),
                                 HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
    }

    ASSERT_EQ(synLaunchExt(streamHandle,
                           launchParams.getSynLaunchTensorInfoVec().data(),
                           launchParams.getSynLaunchTensorInfoVec().size(),
                           launchParams.getWorkspace(),
                           recipe.getRecipe(),
                           0),
              synSuccess)
        << "Failed to launch graph";

    synEventHandle tpcComputeDone;
    status = synEventCreate(&tpcComputeDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // Associate an event with its completion
    status = synEventRecord(tpcComputeDone, streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";
    std::cout << "compute job dma of input sent\n";

    for (unsigned outputIndex = 0; outputIndex < recipe.getTensorInfoVecSizeOutput(); outputIndex++)
    {
        const auto& hostOutput   = launchParams.getHostOutput(outputIndex);
        const auto& deviceOutput = launchParams.getDeviceOutput(outputIndex);

        // Copy data from device to host
        status = synMemCopyAsync(streamHandle,
                                 deviceOutput.getBuffer(),
                                 recipe.getTensorSizeOutput(outputIndex),
                                 (uint64_t)hostOutput.getBuffer(),
                                 DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";
    }

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    recipe.validateResults(launchParams.getLaunchTensorMemory());

    std::cout << "compute job launched\n";

    status = query_events(dmaToHostDone, tpcComputeDone);
    ASSERT_EQ(status, synSuccess) << "Failed to query_events";

    status = synEventDestroy(dmaToHostDone);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy dmaToHostDone";

    status = synEventDestroy(tpcComputeDone);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy tpcComputeDone";

    // Wait for everything to finish by blocking on the copy from host to device
    std::cout << "basic_dma_end begins\n";
    status = synStreamSynchronize(dmaInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Copy data from device to host
    status = synMemCopyAsync(dmaOutStream, dmaDeviceAddress, dmaSize, (uint64_t)pDmaOutput, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(dmaOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    synDeviceFree(device.getDeviceId(), dmaDeviceAddress, 0);

    synStreamDestroy(dmaInStream);
    synStreamDestroy(dmaOutStream);

    // Check results
    ASSERT_EQ(memcmp(pDmaInput, pDmaOutput, dmaSize), 0) << "Wrong results";

    synHostFree(device.getDeviceId(), (void*)pDmaInput, 0);
    synHostFree(device.getDeviceId(), (void*)pDmaOutput, 0);
}

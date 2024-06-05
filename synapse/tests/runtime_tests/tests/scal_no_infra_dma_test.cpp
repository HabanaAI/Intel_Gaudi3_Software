#include "syn_base_test.hpp"
#include "syn_singleton.hpp"
#include "scoped_configuration_change.h"
#include "habana_global_conf_runtime.h"
#include "test_recipe_dma.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "runtime/scal/gaudi2/device_gaudi2scal.hpp"
#include "runtime/scal/gaudi3/device_gaudi3scal.hpp"

class SynScalNoInfraDma : public SynBaseTest
{
public:
    SynScalNoInfraDma() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }
    //
    void open_close_collective_stream();
    void basic_dma_disable_timeout();
    void basic_dma_with_device_memset(unsigned mode);
    void basic_dma_no_compilation();
    void basic_dma(int             NUM_LAUNCH,
                   int             errorOnIdx             = -1,
                   bool            simulateComputeTimeout = false,
                   scal_timeouts_t timeouts = scal_timeouts_t {SCAL_TIMEOUT_NOT_SET, SCAL_TIMEOUT_NOT_SET});
    void basic_ccb_management();
    void basic_dev2dev();
    void testGetDeviceDramMemoryInfo();
    void basic_ccb_management_multiple_memcpy();
    void d2d_ccb_test2();
    void d2d_ccb_test2_impl(const uint64_t size, const uint64_t maxElem, const uint64_t loops);
};

REGISTER_SUITE(SynScalNoInfraDma, ALL_TEST_PACKAGES);

class SynScalNoInfraDmaAsic : public SynScalNoInfraDma
{
public:
    SynScalNoInfraDmaAsic() : SynScalNoInfraDma() {}
};

REGISTER_SUITE(SynScalNoInfraDmaAsic, synTestPackage::ASIC, synTestPackage::ASIC_CI);

void SynScalNoInfraDma::open_close_collective_stream()
{
    TestDevice device(m_deviceType);

    synStreamHandle stream;
    auto            status = synStreamCreateGeneric(&stream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "synStreamCreateGeneric failed";

    status = synStreamDestroy(stream);
    ASSERT_EQ(status, synSuccess) << "synStreamDestroy failed";
}

TEST_F_SYN(SynScalNoInfraDma, open_close_collective_stream)
{
    open_close_collective_stream();
}

/**
 * test running with a short timeout and "timeout disabled"
 */
void SynScalNoInfraDma::basic_dma_disable_timeout()
{
    TestDevice device(m_deviceType);

    // set short timeout value to 5 micro seconds and disable timeout
    ScopedEnvChange timeoutEnv("SCAL_TIMEOUT_VALUE", "5");             // set a very small timeout
    ScopedEnvChange timeoutSecEnv("SCAL_TIMEOUT_VALUE_SECONDS", "0");  // set a very small timeout
    ScopedEnvChange timeoutDisEnv("SCAL_DISABLE_TIMEOUT", "1");        // disable timeout

    // Host buffer for input
    uint8_t*       input;
    const uint64_t size = 16 * 1024 * 1024;

    // Input need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";

    // std::pair<uint8_t, uint8_t> range = {-0, 100};
    // fillWithRandom<uint8_t>(input, size, range);
    memset(input, 0xEE, size);

    // Create streams
    synStreamHandle copyInStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";

    // Device-side (HBM) buffers for input
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    // Copy data from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // Wait for everything to finish by blocking on the copy from host to device and count duration
    LOG_DEBUG(SYN_API, "{}: synStreamSynchronize started", HLLOG_FUNC);
    status = synStreamSynchronize(copyInStream);
    // check that the sync FAILED.
    ASSERT_EQ(status, synSuccess) << "synStreamSynchronize failed";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);
    synStreamDestroy(copyInStream);
    synHostFree(device.getDeviceId(), (void*)input, 0);
}

void SynScalNoInfraDma::basic_dma_with_device_memset(unsigned mode)
{
    TestDevice device(m_deviceType);

    // Host buffers for input & output
    uint8_t *      input, *output;
    const uint64_t size = 16 * 1024 * 1024;

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Init input with random values and zero-out the output
    // std::pair<uint8_t, uint8_t> range = {-0, 100};
    // fillWithRandom<uint8_t>(input, size, range);
    memset(input, 0, size);  // we will override parts of it later
    memset(output, 0, size);

    // Create streams
    synStreamHandle copyInStream, copyInStream2, copyOutStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";

    status = synStreamCreateGeneric(&copyInStream2, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";

    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
    uint32_t v = 0;
    // memset data on the device
    if (mode == 32)
    {
        v = 0x12345678;
        std::fill_n((uint32_t*)input, size >> 2, v);
        status = synMemsetD32Async((uint64_t)deviceAddress, v, size >> 2, copyInStream);
    }
    else if (mode == 16)
    {
        v = 0x1234;
        std::fill_n((uint16_t*)input, size >> 1, v);
        status = synMemsetD16Async((uint64_t)deviceAddress, v, size >> 1, copyInStream);
    }
    else if (mode == 8)
    {
        v = 0x99;
        memset(input, v, size);
        status = synMemsetD8Async((uint64_t)deviceAddress, v, size, copyInStream);
    }
    else if (mode == 7)
    {
        // test copying size - 1 values

        // 1st copy full array of input
        status = synMemCopyAsync(copyInStream2, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
        status = synStreamSynchronize(copyInStream2);
        ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";
        // then memset size-1 values
        v = 0x99;
        memset(input, v, size - 1);
        status = synMemsetD8Async((uint64_t)deviceAddress, v, size - 1, copyInStream);
    }
    else
    {
        status = synInvalidArgument;
    }
    ASSERT_EQ(status, synSuccess) << "Failed to memset Device HBM memory";
    /*
        status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
    */
    // Wait for everything to finish by blocking on the copy from host to device
    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, deviceAddress, size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyInStream2);
    synStreamDestroy(copyOutStream);

    // Check results
    ASSERT_EQ(memcmp(input, output, size), 0) << "Wrong results";

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_no_compilation)
{
    TestDevice device(m_deviceType);

    // Allocate buffers on host
    uint8_t *      input, *output;
    const uint64_t size   = 16 * 1024 * 1024;
    auto           status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Set the input buffer with 0xEE and zero-out the output buffer
    memset(input, 0xEE, size);
    memset(output, 0, size);

    // Allocate buffer on device
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    // Create stream
    synStreamHandle streamHandle;
    status = synStreamCreateGeneric(&streamHandle, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Copy data from host to device
    status = synMemCopyAsync(streamHandle, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // Copy data from device to host
    status = synMemCopyAsync(streamHandle, deviceAddress, size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    // Destroy stream
    synStreamDestroy(streamHandle);

    // Release buffer on device
    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    // Check results
    ASSERT_EQ(memcmp(input, output, size), 0) << "Wrong results";

    // Release buffer on host
    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_disable_timeout)
{
    basic_dma_disable_timeout();
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_with_device_memset_d32)
{
    basic_dma_with_device_memset(32);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_with_device_memset_d16)
{
    basic_dma_with_device_memset(16);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_with_device_memset_d8)
{
    basic_dma_with_device_memset(8);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_with_device_memset_d7)
{
    basic_dma_with_device_memset(7);
}

void SynScalNoInfraDma::basic_dma(int NUM_LAUNCH, int errorOnIdx, bool simulateComputeTimeout, scal_timeouts_t timeouts)
{
    TestRecipeDma recipe(m_deviceType, 16 * 1024U, 1024U, 0xEE, false, syn_type_uint8);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    // NOTE Z  = 16 * 1024U;   and default is  Z = 1 * 1024U;

    // Execution
    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    auto                             devScal         = dynamic_cast<common::DeviceScal*>(deviceInterface.get());
    ASSERT_NE(devScal, nullptr) << "cannot get devScal";
    devScal->setTimeouts(timeouts, false);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_POSITIVE, 0});

    // Host buffers for input & output
    uint8_t *   input[NUM_LAUNCH], *output[NUM_LAUNCH];
    uint64_t    tensorSize               = recipe.getTensorSizeInput(0);
    uint32_t    numOfTensor              = 2;
    const char* tensorNames[numOfTensor] = {"input_0_0", "output_0_0"};

    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        // Input and output need to be mapped to the device as they are copied from / to
        synStatus status = synHostMalloc(device.getDeviceId(), tensorSize, 0, (void**)&input[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input" << i;
        status = synHostMalloc(device.getDeviceId(), tensorSize, 0, (void**)&output[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output" << i;

        // Init input with random values and zero-out the output
        memset(input[i], 0xEE + i, tensorSize);
        memset(output[i], 0x00, tensorSize);
    }

    // Create streams
    synStreamHandle copyInStream, copyOutStream, computeStream;
    synStatus       status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&computeStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create compute stream";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t pDeviceInput[NUM_LAUNCH], pDeviceOutput[NUM_LAUNCH];
    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        status = synDeviceMalloc(device.getDeviceId(), tensorSize, 0, 0, &pDeviceInput[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
        status = synDeviceMalloc(device.getDeviceId(), tensorSize, 0, 0, &pDeviceOutput[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";
    }

    // Associate the tensors with the device memory so compute knows where to read from / write to
    synLaunchTensorInfo persistentTensorInfo[NUM_LAUNCH][numOfTensor];
    uint64_t            tensorIds[numOfTensor];
    ASSERT_EQ(synTensorRetrieveIds(recipe.getRecipe(), tensorNames, tensorIds, numOfTensor), synSuccess);

    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        persistentTensorInfo[i][0].tensorName     = "input";  // Must match the name supplied at tensor creation
        persistentTensorInfo[i][0].pTensorAddress = pDeviceInput[i];
        persistentTensorInfo[i][0].tensorType     = DATA_TENSOR;
        memset(&persistentTensorInfo[i][0].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
        persistentTensorInfo[i][0].tensorId = tensorIds[0];

        persistentTensorInfo[i][1].tensorName     = "output";  // Must match the name supplied at tensor creation
        persistentTensorInfo[i][1].pTensorAddress = pDeviceOutput[i];
        persistentTensorInfo[i][1].tensorType     = DATA_TENSOR;
        memset(&persistentTensorInfo[i][1].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
        persistentTensorInfo[i][1].tensorId = tensorIds[1];
    }

    synEventHandle copyDone, computeDone;
    status = synEventCreate(&copyDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&computeDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    if (!simulateComputeTimeout)  // if we simulate compute timeout, don't do any operation before it (because
                                  // that operation will timeout and not the compute
    {
        for (int i = 0; i < NUM_LAUNCH; i++)
        {
            // Copy data from host to device
            status = synMemCopyAsync(copyInStream, (uint64_t)input[i], tensorSize, pDeviceInput[i], HOST_TO_DRAM);
            ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

            // Sync on device

            // Associate an event with its completion
            status = synEventRecord(copyDone, copyInStream);
            ASSERT_EQ(status, synSuccess) << "Failed to record event";

            // Compute waits for the copy to finish
            status = synStreamWaitEvent(computeStream, copyDone, 0);
            ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";
        }
    }
    // Schedule compute
    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        int numTensors = (i == errorOnIdx) ? 1 : 2;
        status         = synLaunch(computeStream,
                           persistentTensorInfo[i],
                           numTensors,
                           recipeLaunchParams.getWorkspace(),
                           recipe.getRecipe(),
                           0);

        synStatus expected = (i == errorOnIdx) ? synFailedSectionValidation : synSuccess;

        ASSERT_EQ(status, expected) << "Failed to launch graph";
    }

    // Associate an event with its completion
    status = synEventRecord(computeDone, computeStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // Copy waits for compute to finish
    status = synStreamWaitEvent(copyOutStream, computeDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    if (simulateComputeTimeout)
    {
        std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
        common::DeviceScal*              devScal         = dynamic_cast<common::DeviceScal*>(deviceInterface.get());

        devScal->testingOnlySetBgFreq(std::chrono::milliseconds(1));

        // This triggers DFA. To make sure we don't crash the device, wait until work is done on device
        while ((status = synEventQuery(computeDone)) == synBusy)
        {
            sleep(1);
        }
        return;
    }

    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        // Copy data from device to host
        status = synMemCopyAsync(copyOutStream, pDeviceOutput[i], tensorSize, (uint64_t)output[i], DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

        // Wait for everything to finish by blocking on the copy from device to host
        status = synStreamSynchronize(copyOutStream);
        ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

        if (i == errorOnIdx) continue;
        // Check results
        ASSERT_EQ(memcmp(input[i], output[i], tensorSize), 0) << "Wrong results";
    }

    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        synHostFree(device.getDeviceId(), (void*)input[i], 0);
        synHostFree(device.getDeviceId(), (void*)output[i], 0);
    }

    synEventDestroy(copyDone);
    synEventDestroy(computeDone);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);
    synStreamDestroy(computeStream);

    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        synDeviceFree(device.getDeviceId(), pDeviceInput[i], 0);
        synDeviceFree(device.getDeviceId(), pDeviceOutput[i], 0);
    }
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma)
{
    basic_dma(1);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_lack_data_chunks)
{
    // We are running 10 times. One DC for non-patchable, we need another 10 for patchable. I'll allocate only 8
    ScopedConfigurationChange expFlags("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange dataChunksAmount("STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP", "8");

    basic_dma(10);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dma_multi_data_chunks)
{
    // We are running 10 times. One DC for non-patchable, we need another 10 for patchable. I'll allocate only 8
    ScopedConfigurationChange expFlags("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange dataChunksMin("STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP", "16");
    uint64_t                  keep     = PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES;
    PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES = 256;

    basic_dma(1);

    PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES = keep;
}

// Run multiple times a recipe (dma) but inject a patching error on one of the launches
TEST_F_SYN(SynScalNoInfraDma, basic_dma_multi_with_patch_error)
{
    basic_dma(4, 1);
}

void SynScalNoInfraDma::basic_ccb_management()
{
    TestDevice device(m_deviceType);

    // Host buffers for input & output
    uint8_t *      input, *output;
    const uint64_t size           = 1 * 1024;
    const uint64_t numberOfInputs = 4000;

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size * numberOfInputs, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size * numberOfInputs, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    for (int i = 0; i < numberOfInputs; ++i)
    {
        memset(input + (size * i), i, size);
        memset(output + (size * i), 0, size);
    }

    // Create streams
    synStreamHandle copyInStream, copyOutStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size * numberOfInputs, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    for (int i = 0; i < numberOfInputs; ++i)
    {
        // Copy data from host to device
        status =
            synMemCopyAsync(copyInStream, (uint64_t)(input + i * size), size, deviceAddress + i * size, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
    }
    // Wait for everything to finish by blocking on the copy from host to device
    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, deviceAddress, numberOfInputs * size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);

    // Check results
    ASSERT_EQ(memcmp(input, output, size * numberOfInputs), 0) << "Wrong results";

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);
}

void SynScalNoInfraDma::basic_ccb_management_multiple_memcpy()
{
    TestDevice device(m_deviceType);

    // Host buffers for input & output
    uint8_t *      input, *output;
    const uint64_t size           = 8;
    const uint32_t innerMemCopies = 32768;

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size * innerMemCopies, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size * innerMemCopies, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    for (int i = 0; i < innerMemCopies; ++i)
    {
        memset(input + (size * i), i, size);
        memset(output + (size * i), 0, size);
    }

    // Create streams
    synStreamHandle copyInStream, copyOutStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size * innerMemCopies, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    std::unique_ptr<uint64_t[]> inputsVec(new uint64_t[innerMemCopies]);
    std::unique_ptr<uint64_t[]> outputsVec(new uint64_t[innerMemCopies]);
    std::unique_ptr<uint64_t[]> sizesVec(new uint64_t[innerMemCopies]);
    std::unique_ptr<uint64_t[]> deviceAddressesVec(new uint64_t[innerMemCopies]);

    for (int i = 0; i < innerMemCopies; ++i)
    {
        inputsVec[i]          = (uint64_t)input + (size * i);
        outputsVec[i]         = (uint64_t)output + (size * i);
        sizesVec[i]           = size;
        deviceAddressesVec[i] = deviceAddress + (size * i);
    }

    // Copy data from host to device
    status = synMemCopyAsyncMultiple(copyInStream,
                                     inputsVec.get(),
                                     sizesVec.get(),
                                     deviceAddressesVec.get(),
                                     HOST_TO_DRAM,
                                     innerMemCopies);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // Wait for everything to finish by blocking on the copy from host to device
    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Copy data from device to host
    status = synMemCopyAsyncMultiple(copyOutStream,
                                     deviceAddressesVec.get(),
                                     sizesVec.get(),
                                     outputsVec.get(),
                                     DRAM_TO_HOST,
                                     innerMemCopies);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);

    // Check results
    ASSERT_EQ(memcmp(input, output, size * innerMemCopies), 0) << "Wrong results";

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);
}
void SynScalNoInfraDma::d2d_ccb_test2()
{
    constexpr uint64_t size    = 0x1000;
    constexpr uint64_t maxElem = 256;
    constexpr uint64_t loops   = 500;

    d2d_ccb_test2_impl(size, maxElem, loops);
}

void SynScalNoInfraDma::d2d_ccb_test2_impl(const uint64_t size, const uint64_t maxElem, const uint64_t loops)
{
    std::srand(1);


    TestDevice device(m_deviceType);

    synStreamHandle streamD2D;
    auto            status = synStreamCreateGeneric(&streamD2D, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";

    // Device-side (HBM) buffers for input and output

    uint64_t deviceAddressSrc, deviceAddressDst;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddressSrc);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddressDst);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";

    std::vector<uint64_t> srcVec(maxElem, deviceAddressSrc);
    std::vector<uint64_t> dstVec(maxElem, deviceAddressDst);
    std::vector<uint64_t> sizeVec(maxElem, size);

    for (int i = 0; i < loops; i++)
    {
        uint32_t numElem = std::rand() % maxElem + 1;

        // Copy data between device buffers
        status =
            synMemCopyAsyncMultiple(streamD2D, srcVec.data(), sizeVec.data(), dstVec.data(), DRAM_TO_DRAM, numElem);
    }

    // Wait for everything to finish by blocking on the stream
    status = synStreamSynchronize(streamD2D);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    synDeviceFree(device.getDeviceId(), deviceAddressSrc, 0);
    synDeviceFree(device.getDeviceId(), deviceAddressDst, 0);

    synStreamDestroy(streamD2D);
}

TEST_F_SYN(SynScalNoInfraDma, d2d_ccb_test)
{
    d2d_ccb_test2();
}

TEST_F_SYN(SynScalNoInfraDmaAsic, d2d_ccb_test_pi_overflow_ASIC, {synDeviceGaudi3})
{
    constexpr uint64_t size    = 0x10;
    constexpr uint64_t maxElem = 50000;
    constexpr uint64_t loops   = 15000;

    // takes about 4 min on G3 ASIC
    // does 2 overflows
    d2d_ccb_test2_impl(size, maxElem, loops);
}


// Run multiple pdma commands to get the ccb full twice
// the main motivation is to test that commands are not given overwritten
TEST_F_SYN(SynScalNoInfraDma, basic_ccb_management)
{
    basic_ccb_management();
}

TEST_F_SYN(SynScalNoInfraDma, basic_ccb_management_multiple_memcpy)
{
    basic_ccb_management_multiple_memcpy();
}
class SynScalNoInfraDmaM
: public SynScalNoInfraDma
, public testing::WithParamInterface<unsigned>
{
};

INSTANTIATE_TEST_SUITE_P(, SynScalNoInfraDmaM, ::testing::Values(2, 8, 12));

TEST_P(SynScalNoInfraDmaM, basic_dma_multi)
{
    basic_dma(GetParam());
}

void SynScalNoInfraDma::testGetDeviceDramMemoryInfo()
{
    TestDevice device(m_deviceType);

    uint64_t freeStart  = std::numeric_limits<uint64_t>::max();
    uint64_t totalStart = std::numeric_limits<uint64_t>::max();
    auto     status     = synDeviceGetMemoryInfo(device.getDeviceId(), &freeStart, &totalStart);
    ASSERT_EQ(status, synSuccess);

    uint64_t currMem = freeStart;

    struct AddrSize
    {
        uint64_t addr;
        uint64_t size;
    };

    std::vector<AddrSize> devAddr;
    for (int i = 0; i < 10; i++)
    {
        uint64_t size = (i + 1) * 0x1000;

        uint64_t deviceAddress;
        status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

        currMem -= size;
        devAddr.push_back({deviceAddress, size});

        uint64_t free  = -1;
        uint64_t total = -1;

        status = synDeviceGetMemoryInfo(device.getDeviceId(), &free, &total);
        ASSERT_EQ(status, synSuccess);

        ASSERT_EQ(total, totalStart);
        ASSERT_EQ(free, currMem);
    }

    for (int i = 0; i < devAddr.size(); i++)
    {
        status = synDeviceFree(device.getDeviceId(), devAddr[i].addr, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

        currMem += devAddr[i].size;

        uint64_t free  = std::numeric_limits<uint64_t>::max();
        uint64_t total = std::numeric_limits<uint64_t>::max();

        status = synDeviceGetMemoryInfo(device.getDeviceId(), &free, &total);
        ASSERT_EQ(status, synSuccess);

        ASSERT_EQ(total, totalStart);
        ASSERT_EQ(free, currMem);
    }
}

TEST_F_SYN(SynScalNoInfraDma, testGetDeviceDramMemoryInfo)
{
    testGetDeviceDramMemoryInfo();
}

// This test:
// 1) copies data to deviceAddressIn (host->device)
// 2) dev-to-dev copy from deviceAddressIn to deviceAddressOut (device->device)
// 3) copies data back to host from deviceAddressOut and compares the data (device->host)

void SynScalNoInfraDma::basic_dev2dev()
{
    // Host buffers for input & output
    uint8_t *      input, *output;
    const uint64_t size = 16 * 1024 * 1024;

    TestDevice device(m_deviceType);

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Init input with random values and zero-out the output
    // std::pair<uint8_t, uint8_t> range = {-0, 100};
    // fillWithRandom<uint8_t>(input, size, range);
    memset(input, 0xEE, size);
    memset(output, 0, size);

    // Create streams
    synStreamHandle copyInStream, copyOutStream, dev2devStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";
    status = synStreamCreateGeneric(&dev2devStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from device to device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddressIn;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddressIn);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory in";

    uint64_t deviceAddressOut;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddressOut);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory out";

    // Copy data from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddressIn, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // Wait for everything to finish by blocking on the copy from host to device
    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    status = synMemCopyAsync(dev2devStream, deviceAddressIn, size, deviceAddressOut, DRAM_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Wait for everything to finish by blocking on the copy device to device
    status = synStreamSynchronize(dev2devStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for dev2dev stream";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, deviceAddressOut, size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    synDeviceFree(device.getDeviceId(), deviceAddressIn, 0);
    synDeviceFree(device.getDeviceId(), deviceAddressOut, 0);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);
    synStreamDestroy(dev2devStream);

    // Check results
    ASSERT_EQ(memcmp(input, output, size), 0) << "Wrong results";

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);
}

TEST_F_SYN(SynScalNoInfraDma, basic_dev2dev_test)
{
    basic_dev2dev();
}

TEST_F_SYN(SynScalNoInfraDma, DISABLED_basic_dma_simulate_timeout)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GCFG_TERMINATE_SYNAPSE_UPON_DFA.setValue((uint64_t)DfaSynapseTerminationState::disabled);

    LOG_ERR_T(SYN_API, "-----THIS TEST IS EXPECTED TO LOG ERRORS-----");

    // set a very small timeout
    basic_dma(10, -1, true, scal_timeouts_t {1, SCAL_TIMEOUT_NOT_SET});
}

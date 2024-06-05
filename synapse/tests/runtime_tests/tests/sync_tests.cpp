#include "syn_base_test.hpp"
#include "syn_singleton.hpp"
#include "runtime/scal/common/scal_event.hpp"
#include "scal_internal/pkt_macros.hpp"
#include "test_recipe_dma.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_utils.h"
#include "habana_global_conf_runtime.h"

class NoInfraStreamsSync : public SynBaseTest
{
public:
    NoInfraStreamsSync() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }

    void basic_dma_read_write_sync_test();
    void basic_event_after_wait_sync_test();
    void basic_dma_sync_test();
    void basic_dma_compute_sync_test();
    void stream_event_wait_cyclic_buffer_quarter_signal();
};

REGISTER_SUITE(NoInfraStreamsSync, ALL_TEST_PACKAGES);

/*
 ***************************************************************************************************
 *   @brief basic_dma_read_write_sync_test() - test dma_down and dma_up streams synchronization,
 *   use two dma operaions to chcek read after write works correctly
 *
 ***************************************************************************************************
 */
void NoInfraStreamsSync::basic_dma_read_write_sync_test()
{
    TestDevice device(m_deviceType);

    // Host buffers for input & output
    uint8_t *      input, *output, *input2, *output2;
    const uint64_t size = 2 * 1024 * 1024;

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input2);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output2);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Init input with random values and zero-out the output
    std::pair<uint8_t, uint8_t> range = {0, 50};
    fillWithRandom<uint8_t>(input, size, range);
    memset(output, 0, size);

    // Init second input with random values and zero-out the second output
    std::pair<uint8_t, uint8_t> range2 = {50, 100};
    fillWithRandom<uint8_t>(input2, size, range2);
    memset(output2, 0, size);

    // Create streams
    synStreamHandle copyInStream, copyOutStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    synEventHandle copyDown, copyUp;
    status = synEventCreate(&copyDown, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&copyUp, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // Copy first input from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";

    // Associate an event with its completion
    status = synEventRecord(copyDown, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // DMA up waits for the DMA down to finish
    status = synStreamWaitEvent(copyOutStream, copyDown, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, deviceAddress, size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Associate an event with its completion
    status = synEventRecord(copyUp, copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // DMA down waits for the dma up finish
    status = synStreamWaitEvent(copyInStream, copyUp, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy second input from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input2, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";

    // Associate an event with its completion
    status = synEventRecord(copyDown, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // DMA up waits for the copy to finish
    status = synStreamWaitEvent(copyOutStream, copyDown, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, deviceAddress, size, (uint64_t)output2, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Check results
    ASSERT_EQ(memcmp(input, output, size), 0) << "Wrong first results";
    ASSERT_EQ(memcmp(input2, output2, size), 0) << "Wrong second results";

    // Waiting for the completion of all operations on the device
    status = synDeviceSynchronize(device.getDeviceId());
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-the device";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    synEventDestroy(copyDown);
    synEventDestroy(copyUp);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);

    synHostFree(device.getDeviceId(), (void*)input2, 0);
    synHostFree(device.getDeviceId(), (void*)output2, 0);
}

/*
 ***************************************************************************************************
 *   @brief basic_dma_read_write_sync_test() - test dma_down and dma_up streams synchronization,
 *   the stream uses 2 DMA down streams and 1 DMA up stream.
 *   The up stream should wait on event that capture the state of both DMA down streams
 ***************************************************************************************************
 */
void NoInfraStreamsSync::basic_event_after_wait_sync_test()
{
    TestDevice device(m_deviceType);

    // Host buffers for input & output
    uint8_t *      input, *output, *input2, *output2;
    const uint64_t size = 2 * 1024 * 1024;

    const uint64_t offsetToCheck = size - 1000;

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input2);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output2);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Init input with random values and zero-out the output
    std::pair<uint8_t, uint8_t> range = {0, 50};
    fillWithRandom<uint8_t>(input, size, range);
    memset(output, 0, size);

    // Init second input with random values and zero-out the second output
    std::pair<uint8_t, uint8_t> range2 = {50, 100};
    fillWithRandom<uint8_t>(input2, size, range2);
    memset(output2, 0, size);

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

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress2;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress2);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    synEventHandle copyDown, copyDown2, copyUp;
    status = synEventCreate(&copyDown, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&copyDown2, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&copyUp, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // Copy first input from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";

    // Copy first input from host to device
    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Copy second input from host to device
    status = synMemCopyAsync(copyInStream2, (uint64_t)input2, size, deviceAddress2, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";

    // Associate an event with its completion
    status = synEventRecord(copyDown, copyInStream2);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // copyInStream should wait to copyInStream2 to finish
    status = synStreamWaitEvent(copyInStream, copyDown, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // copyDown2 should capture the state after the synStreamWaitEvent
    status = synEventRecord(copyDown2, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // DMA up should waits for both DMA down to finish
    status = synStreamWaitEvent(copyOutStream, copyDown2, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy last bytes of second input from device to host
    uint64_t deviceAddresToRead = deviceAddress2 + offsetToCheck;
    status = synMemCopyAsync(copyOutStream, deviceAddresToRead, size - offsetToCheck, (uint64_t)output2, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Copy last bytes of first input from device to host
    deviceAddresToRead = deviceAddress + offsetToCheck;
    status = synMemCopyAsync(copyOutStream, deviceAddresToRead, size - offsetToCheck, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    // Check results
    ASSERT_EQ(memcmp(input + offsetToCheck, output, size - offsetToCheck), 0) << "Wrong first results";
    ASSERT_EQ(memcmp(input2 + offsetToCheck, output2, size - +offsetToCheck), 0) << "Wrong second results";

    // Waiting for the completion of all operations on the device
    status = synDeviceSynchronize(device.getDeviceId());
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-the device";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    synEventDestroy(copyDown);
    synEventDestroy(copyDown2);
    synEventDestroy(copyUp);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyInStream2);
    synStreamDestroy(copyOutStream);

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);

    synHostFree(device.getDeviceId(), (void*)input2, 0);
    synHostFree(device.getDeviceId(), (void*)output2, 0);
}

/*
 ***************************************************************************************************
 *   @brief basic_dma_sync_test() - test dma_down and dma_up streams synchronization,
 *   the test modifed the event used for the synchronization to control the exact
 *   dma_down job that will trigger the dma_up work
 *
 ***************************************************************************************************
 */
void NoInfraStreamsSync::basic_dma_sync_test()
{
    TestDevice device(m_deviceType);

    // Host buffers for input & output
    uint8_t *      input, *output;
    const uint64_t size = 1 * 1024 * 1024;

    // Input and output need to be mapped to the device as they are copied from / to
    auto status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(device.getDeviceId(), size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    // Init input with random values and zero-out the output
    std::pair<uint8_t, uint8_t> range = {0, 100};
    fillWithRandom<uint8_t>(input, size, range);
    memset(output, 0, size);

    // Create streams
    synStreamHandle copyInStream, copyOutStream;
    status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress;
    status = synDeviceMalloc(device.getDeviceId(), size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    synEventHandle copyDown, copyUp;
    status = synEventCreate(&copyDown, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&copyUp, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // Copy data from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";

    // Associate an event with its completion
    status = synEventRecord(copyDown, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // modify event to look on next future job
    ScalEvent* scalEvent = dynamic_cast<ScalEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(copyDown));
    ASSERT_NE(scalEvent, nullptr);
    scalEvent->longSo.m_targetValue += 1;

    // DMA up waits for the copy to finish
    status = synStreamWaitEvent(copyOutStream, copyDown, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, deviceAddress, size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Associate an event with its completion
    status = synEventRecord(copyUp, copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // modify back the event to the good state
    scalEvent->longSo.m_targetValue -= 1;

    // make sure dma down job finished and dma up didn't started
    int maxIter = 20;
    int iter    = 0;
    while (iter < maxIter)
    {
        status = synEventQuery(copyDown);
        if (status == synSuccess) break;
        sleep(1);
        iter++;
    }
    ASSERT_NE(iter, maxIter) << "DMA JOB didn't finished in time 20 seconds";

    sleep(2);

    // event should return busy
    status = synEventQuery(copyUp);
    ASSERT_EQ(status, synBusy) << "Failed to record event";

    // stream should return busy
    status = synStreamQuery(copyOutStream);
    ASSERT_EQ(status, synBusy) << "Failed to record event";

    // start second job Copy data from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    status = synStreamSynchronize(copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy in stream";

    // Check results
    ASSERT_EQ(memcmp(input, output, size), 0) << "Wrong results";

    // Waiting for the completion of all operations on the device
    status = synDeviceSynchronize(device.getDeviceId());
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-the device";

    synDeviceFree(device.getDeviceId(), deviceAddress, 0);

    synEventDestroy(copyDown);
    synEventDestroy(copyUp);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);

    synHostFree(device.getDeviceId(), (void*)input, 0);
    synHostFree(device.getDeviceId(), (void*)output, 0);
}

/*
 ***************************************************************************************************
 *   @brief basic_dma_sync_test() - test dma_down, compute and dma_up streams synchronization,
 *   the test modifed the event used for the synchronization to control the exact
 *   dma_down job that will trigger the compute work, then doing the same scheme
 *   that the dma_up will start only after the second compute job
 *
 ***************************************************************************************************
 */
void NoInfraStreamsSync::basic_dma_compute_sync_test()
{
    TestRecipeDma recipe(m_deviceType, 16 * 1024U, 1024U, 0xEE, false, syn_type_uint8);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    // Execution
    const uint64_t     NUM_LAUNCH = 2;
    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_POSITIVE, 0});

    // Host buffers for input & output
    uint8_t *input[NUM_LAUNCH], *output[NUM_LAUNCH];

    for (int i = 0; i < NUM_LAUNCH; i++)
    {
        // Input and output need to be mapped to the device as they are copied from / to
        synStatus status = synHostMalloc(device.getDeviceId(), recipe.getTensorSizeInput(0), 0, (void**)&input[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input" << i;
        status = synHostMalloc(device.getDeviceId(), recipe.getTensorSizeOutput(0), 0, (void**)&output[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output" << i;

        // Init input with random values and zero-out the output
        memset(input[i], 0xEE + i, recipe.getTensorSizeInput(0));
        memset(output[i], 0x00, recipe.getTensorSizeOutput(0));
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
        status = synDeviceMalloc(device.getDeviceId(), recipe.getTensorSizeInput(0), 0, 0, &pDeviceInput[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
        status = synDeviceMalloc(device.getDeviceId(), recipe.getTensorSizeOutput(0), 0, 0, &pDeviceOutput[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";
    }

    // Associate the tensors with the device memory so compute knows where to read from / write to
    const uint32_t tensorsAmount = 2;
    uint32_t       i             = 0;

    uint64_t    tensorIds[tensorsAmount];
    const char* tensorNames[tensorsAmount] = {"input", "output"};

    synLaunchTensorInfoExt persistentTensorInfo[NUM_LAUNCH][tensorsAmount];

    ASSERT_EQ(synTensorRetrieveIds(recipe.getRecipe(), tensorNames, tensorIds, tensorsAmount), synSuccess);

    for (i = 0; i < NUM_LAUNCH; i++)
    {
        persistentTensorInfo[i][0].tensorName     = tensorNames[0];  // Must match the name supplied at tensor creation
        persistentTensorInfo[i][0].pTensorAddress = pDeviceInput[i];
        persistentTensorInfo[i][0].tensorType     = DATA_TENSOR;
        memset(&persistentTensorInfo[i][0].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
        persistentTensorInfo[i][0].tensorId = tensorIds[0];

        persistentTensorInfo[i][1].tensorName     = tensorNames[1];  // Must match the name supplied at tensor creation
        persistentTensorInfo[i][1].pTensorAddress = pDeviceOutput[i];
        persistentTensorInfo[i][1].tensorType     = DATA_TENSOR;
        memset(&persistentTensorInfo[i][1].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
        persistentTensorInfo[i][1].tensorId = tensorIds[1];
    }
    synEventHandle copyDown, computeDone, copyUp;
    status = synEventCreate(&copyDown, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&computeDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&copyUp, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // work on first input buffer
    i = 0;

    // Copy data from host to device
    status =
        synMemCopyAsync(copyInStream, (uint64_t)input[i], recipe.getTensorSizeInput(0), pDeviceInput[i], HOST_TO_DRAM);

    // Associate an event with its completion
    status = synEventRecord(copyDown, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // modify event to look on next future job
    ScalEvent* scalEvent = dynamic_cast<ScalEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(copyDown));
    ASSERT_NE(scalEvent, nullptr);
    scalEvent->longSo.m_targetValue += 1;

    // Compute waits for the copy to finish
    // it won't start until the second dma down job
    status = synStreamWaitEvent(computeStream, copyDown, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Schedule compute
    status = synLaunchExt(computeStream,
                          persistentTensorInfo[i],
                          2,
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0);
    ASSERT_EQ(status, synSuccess) << "Failed to launch graph";

    status = synEventRecord(computeDone, computeStream);
    ASSERT_EQ(status, synSuccess) << "Failed to event record";

    // modify event to look on next future job
    scalEvent = dynamic_cast<ScalEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(computeDone));
    ASSERT_NE(scalEvent, nullptr);
    scalEvent->longSo.m_targetValue += 1;

    // Copy waits for compute to finish
    status = synStreamWaitEvent(copyOutStream, computeDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream,
                             pDeviceOutput[i],
                             recipe.getTensorSizeOutput(0),
                             (uint64_t)output[i],
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    status = synEventRecord(copyUp, copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // make sure first dma down job finished
    scalEvent = dynamic_cast<ScalEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(copyDown));
    ASSERT_NE(scalEvent, nullptr);
    scalEvent->longSo.m_targetValue -= 1;

    int maxIter = 60;
    int iter    = 0;
    while (iter < maxIter)
    {
        status = synEventQuery(copyDown);
        if (status == synSuccess) break;
        sleep(1);
        iter++;
    }
    ASSERT_NE(iter, maxIter) << "DMA JOB didn't finished in time 20 seconds";

    sleep(2);

    // compute shoudn't start yet, event should return busy
    scalEvent = dynamic_cast<ScalEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(computeDone));
    ASSERT_NE(scalEvent, nullptr);
    scalEvent->longSo.m_targetValue -= 1;

    status = synEventQuery(computeDone);
    ASSERT_EQ(status, synBusy) << "Failed to event query";

    // stream should return busy
    status = synStreamQuery(computeStream);
    ASSERT_EQ(status, synBusy) << "Failed to stream query";

    // dma up shoudn't start yet, event should return busy
    status = synEventQuery(copyUp);
    ASSERT_EQ(status, synBusy) << "Failed to event query";

    // stream should return busy
    status = synStreamQuery(copyOutStream);
    ASSERT_EQ(status, synBusy) << "Failed to stream query";

    // work on second input to
    i = 1;

    // Copy data from host to device should trigger compute stream
    status =
        synMemCopyAsync(copyInStream, (uint64_t)input[i], recipe.getTensorSizeInput(0), pDeviceInput[i], HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // make sure compute job finished
    maxIter = 20;
    iter    = 0;
    while (iter < maxIter)
    {
        status = synEventQuery(computeDone);
        if (status == synSuccess) break;
        sleep(1);
        iter++;
    }
    ASSERT_NE(iter, maxIter) << "Compute JOB didn't finished in time 20 seconds";

    sleep(2);

    // dma up shoudn't start yet, event should return busy
    status = synEventQuery(copyUp);
    ASSERT_EQ(status, synBusy) << "Failed to event query";

    // stream should return busy
    status = synStreamQuery(copyOutStream);
    ASSERT_EQ(status, synBusy) << "Failed to stream query";

    // Schedule compute
    status = synLaunchExt(computeStream,
                          persistentTensorInfo[i],
                          tensorsAmount,
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0);
    ASSERT_EQ(status, synSuccess) << "Failed to launch graph";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    // check first input result
    i = 0;
    ASSERT_EQ(memcmp(input[i], output[i], recipe.getTensorSizeInput(0)), 0) << "Wrong results";

    // Waiting for the completion of all operations on the device
    status = synDeviceSynchronize(device.getDeviceId());
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-the device";

    for (uint64_t launchIter = 0; launchIter < NUM_LAUNCH; launchIter++)
    {
        synHostFree(device.getDeviceId(), (void*)input[launchIter], 0);
        synHostFree(device.getDeviceId(), (void*)output[launchIter], 0);
    }

    synEventDestroy(copyDown);
    synEventDestroy(copyUp);
    synEventDestroy(computeDone);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);
    synStreamDestroy(computeStream);

    for (uint64_t launchIter = 0; launchIter < NUM_LAUNCH; launchIter++)
    {
        synDeviceFree(device.getDeviceId(), pDeviceInput[launchIter], 0);
        synDeviceFree(device.getDeviceId(), pDeviceOutput[launchIter], 0);
    }
}

void NoInfraStreamsSync::stream_event_wait_cyclic_buffer_quarter_signal()
{
    std::variant<G2Packets, G3Packets> gaudiDevicePackets;

    switch (m_deviceType)
    {
        case synDeviceGaudi2:
            gaudiDevicePackets = G2Packets();
            break;

        case synDeviceGaudi3:
            gaudiDevicePackets = G3Packets();
            break;

        default:
            bool supportedDevice = (m_deviceType == synDeviceGaudi2 || m_deviceType == synDeviceGaudi3);
            ASSERT_EQ(true, supportedDevice) << "Unsupported dev";
    }

    TestRecipeDma recipe(m_deviceType, 16 * 1024U, 1024U, 0xEE, false, syn_type_uint8);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_POSITIVE, 0});

    // Execution

    // Create streams
    synStreamHandle copyInStream, copyOutStream, computeStream;
    synStatus       status = synStreamCreateGeneric(&copyInStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&computeStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create compute stream";
    status = synStreamCreateGeneric(&copyOutStream, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    synEventHandle copyDone, computeDone;
    status = synEventCreate(&copyDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&computeDone, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    ScalEvent* scalEvent = dynamic_cast<ScalEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(computeDone));
    ASSERT_NE(scalEvent, nullptr);

    // Copy data from host to device
    const auto& hostInput   = recipeLaunchParams.getHostInput(0);
    const auto& deviceInput = recipeLaunchParams.getDeviceInput(0);

    // Copy data from host to device
    status = synMemCopyAsync(copyInStream,
                             (uint64_t)hostInput.getBuffer(),
                             recipe.getTensorSizeInput(0),
                             deviceInput.getBuffer(),
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // Sync on device

    // Associate an event with its completion
    status = synEventRecord(copyDone, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // Compute waits for the copy to finish
    // First synStreamWaitEvent will trigger an alloc/dispatch barrier to increment the longSo
    // for the quarter-barrier scheme on the stream cyclic buffer.
    // synStreamWaitEvent is called one quarter times to reach the next quarter,
    // creating another alloc/dispatch barrier to verify it is working.
    size_t streamWaitEventPktsSize = 0;

    std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            streamWaitEventPktsSize += LbwWritePkt<T>::getSize();
        },
        gaudiDevicePackets);

    std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            if constexpr (!std::is_same_v<T, G2Packets>)
            {
                streamWaitEventPktsSize += AcpFenceWaitPkt<T>::getSize();
            }
            else
            {
                streamWaitEventPktsSize += FenceWaitPkt<T>::getSize();
            }
        },
        gaudiDevicePackets);

    uint32_t ccbSize         = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
    uint32_t ccbChunkAmount  = GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
    uint32_t numOfIterations = (ccbSize / ccbChunkAmount) / (uint32_t)streamWaitEventPktsSize;
    for (unsigned i = 0; i < numOfIterations; i++)
    {
        if (i == 1)
        {
            status = synEventRecord(computeDone, computeStream);
            ASSERT_EQ(status, synSuccess) << "Failed to record event";
            if (GCFG_ENABLE_CHECK_EVENT_REUSE.value())
            {
                //  test that another event record, without any wait between them, indeed fails, as expected
                status = synEventRecord(computeDone, computeStream);
                ASSERT_EQ(status, synResourceBadUsage)
                    << "should have Failed to record event the 2nd time with no wait";
            }
            // Verify the longSo target value is 1:
            // The first synEventRecord which inserted as first command at the beginning of the first cyclic buffer
            // quarter.
            ASSERT_EQ(scalEvent->longSo.m_targetValue, 1);
        }
        status = synStreamWaitEvent(computeStream, copyDone, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";
    }
    // synEventRecord has a new guard mechanism against recording on a none waited event. Without the next line,
    // synEventRecord will fail
    status = synEventQuery(computeDone);
    status = synEventRecord(computeDone, computeStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";
    // Verify the longSo target value is 3:
    // The synEventRecord which inserted as first command at the beginning of the second cyclic buffer quarter.
    // The synEventRecord after synStreamWaitEvent will alloc and dispatch a barrier also.
    ASSERT_EQ(scalEvent->longSo.m_targetValue, 3);

    status = synLaunchExt(computeStream,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0);
    ASSERT_EQ(status, synSuccess) << "Failed to launch graph";

    // Associate an event with its completion

    // synEventRecord has a new guard mechanism against recording on a none waited event. Without the next line,
    // synEventRecord will fail
    status = synEventQuery(computeDone);
    status = synEventRecord(computeDone, computeStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // Verify the longSo target value is 2 since multi op stream we moved to a different one.
    // The synLaunch dispatch barrier incrementation.
    ASSERT_EQ(scalEvent->longSo.m_targetValue, 2);

    // Copy waits for compute to finish
    status = synStreamWaitEvent(copyOutStream, computeDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy data from device to host
    const auto& hostOutput   = recipeLaunchParams.getHostOutput(0);
    const auto& deviceOutput = recipeLaunchParams.getDeviceOutput(0);

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream,
                             deviceOutput.getBuffer(),
                             recipe.getTensorSizeOutput(0),
                             (uint64_t)hostOutput.getBuffer(),
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    // Check results
    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());

    synEventDestroy(copyDone);
    synEventDestroy(computeDone);

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);
    synStreamDestroy(computeStream);
}

TEST_F_SYN(NoInfraStreamsSync, DISABLED_basic_dma_compute_sync_scal)
{
    basic_dma_compute_sync_test();
}

TEST_F_SYN(NoInfraStreamsSync, basic_dma_sync_scal)
{
    basic_dma_sync_test();
}

TEST_F_SYN(NoInfraStreamsSync, basic_dma_read_write_sync)
{
    basic_dma_read_write_sync_test();
}

TEST_F_SYN(NoInfraStreamsSync, basic_event_after_wait_sync)
{
    basic_event_after_wait_sync_test();
}

TEST_F_SYN(NoInfraStreamsSync, stream_event_wait_cyclic_buffer_quarter_signal)
{
    stream_event_wait_cyclic_buffer_quarter_signal();
}

#include "syn_base_test.hpp"
#include <thread>
#include "synapse_api.h"
#include "test_utils.h"

class RtMultiDev2Host : public SynBaseTest
{
public:
    RtMultiDev2Host() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
};

REGISTER_SUITE(RtMultiDev2Host, ALL_TEST_PACKAGES);

static void prepareTensorInfo(synRecipeHandle recipe, synLaunchTensorInfo* tensorInfo, uint32_t totalNumOfTensors)
{
    std::vector<const char*> tensorNames(totalNumOfTensors, nullptr);
    uint64_t                 tensorIds[totalNumOfTensors];
    uint32_t                 i = 0;

    for (i = 0; i < totalNumOfTensors; ++i)
    {
        tensorNames[i] = tensorInfo[i].tensorName;
    }
    ASSERT_EQ(synTensorRetrieveIds(recipe, tensorNames.data(), tensorIds, totalNumOfTensors), synSuccess);
    for (i = 0; i < totalNumOfTensors; i++)
    {
        tensorInfo[i].tensorId = tensorIds[i];
    }
}

/* Test: dma_multi_d2h
 * ===================
 * This test creates a recipe with one dma node. It then runs this node NUM_RUNS times.
 * After each synLaunch it creates NUM_THREADS threads to check the result. First thread
 * checks all the range, the second one a smaller range and so on
 * synchronization between the streams is done in the most efficient way
 */
void chkData(int             idx,
             synStreamHandle copyOutStream,
             synEventHandle  computeDone,
             uint64_t        pDeviceOutput,
             uint64_t        tensorSize,
             uint8_t*        output,
             uint8_t*        input,
             int             deviceId)
{
    uint64_t shift       = idx * 0x100;
    uint8_t* shiftIn     = input + shift;
    uint8_t* shiftOut    = output + shift;
    uint64_t shiftSize   = tensorSize - shift;
    uint64_t shiftDevOut = pDeviceOutput + shift;

    synEventHandle eventUp;
    synEventCreate(&eventUp, deviceId, 0);

    // Copy data from device to host
    synStatus status = synStreamWaitEvent(copyOutStream, computeDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    status = synMemCopyAsync(copyOutStream, shiftDevOut, shiftSize, (uint64_t)shiftOut, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    status = synEventRecord(eventUp, copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    status = synEventSynchronize(eventUp);
    ASSERT_EQ(status, synSuccess) << "synEventSync";

    LOG_TRACE(SYN_API, "Testing results");
    for (int i = 0; i < 32; i++)
    {
        ASSERT_EQ(shiftIn[shiftSize - 1 - i], shiftOut[shiftSize - 1 - i]) << "bad results at the end " << i;
    }

    ASSERT_EQ(memcmp(shiftIn, shiftOut, shiftSize), 0) << "Wrong results";

    status = synEventDestroy(eventUp);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy eventUp";
}

// See description above
TEST_F_SYN(RtMultiDev2Host, dma_multi_d2h)
{
    synStatus status;

    synGraphHandle graph;
    status = synGraphCreate(&graph, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create graph";

    // Create input and output sections for managing device memory
    synSectionHandle inSection, outSection;
    status = synSectionCreate(&inSection, 0, graph);
    ASSERT_EQ(status, synSuccess) << "Failed to create input section";
    status = synSectionCreate(&outSection, 0, graph);
    ASSERT_EQ(status, synSuccess) << "Failed to create output section";

    // Tensor sizes
    const unsigned dims                           = 4U;
    const unsigned Z                              = 16 * 1024U;
    const unsigned W                              = 1024U;
    const unsigned H                              = 1U;
    const unsigned batch                          = 1U;
    unsigned       tensorDimSizes[HABANA_DIM_MAX] = {Z, W, H, batch};
    uint64_t       tensorSize                     = Z * W * H * batch * sizeof(uint8_t);

    // Tensors
    synTensor           in_tensor, out_tensor;
    synTensorDescriptor desc;
    uint32_t            numOfTensor              = 2;
    const char*         tensorNames[numOfTensor] = {"input", "output"};

    desc.m_dataType = syn_type_uint8;
    desc.m_dims     = dims;
    desc.m_name     = tensorNames[0];
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i)
    {
        desc.m_sizes[i] = tensorDimSizes[i];
    }

    status = synTensorCreate(&in_tensor, &desc, inSection, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create input tensor";

    desc.m_name = tensorNames[1];
    for (unsigned i = 0; i < dims; ++i)
    {
        desc.m_sizes[i] = tensorDimSizes[i];
    }

    status = synTensorCreate(&out_tensor, &desc, outSection, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create output tensor";

    // Create DMA node
    status = synNodeCreate(graph,  // associated graph
                           &in_tensor,
                           &out_tensor,  // input/output tensor vectors
                           1,
                           1,  // input/output tensor vector sizes
                           nullptr,
                           0,  // user params
                           "memcpy",
                           "memcpy_node",  // guid and node name
                           nullptr,
                           nullptr);  // input/output layouts
    ASSERT_EQ(status, synSuccess) << "Failed to create node";

    // Compile the graph to get an executable recipe
    synRecipeHandle recipe;
    status = synGraphCompile(&recipe, graph, "my_graph", nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";

    // Execution
    constexpr int NUM_THREADS = 10;
    constexpr int NUM_RUNS    = 2;
    // Host buffers for input & output
    uint8_t *input, *output[NUM_RUNS][NUM_THREADS];

    synDeviceId deviceId;
    status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to acquire device";

    // Input and output need to be mapped to the device as they are copied from / to
    status = synHostMalloc(deviceId, tensorSize, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";

    for (int run = 0; run < NUM_RUNS; run++)
    {
        for (int thr = 0; thr < NUM_THREADS; thr++)
        {
            status = synHostMalloc(deviceId, tensorSize, 0, (void**)&output[run][thr]);
            ASSERT_EQ(status, synSuccess)
                << "Could not allocate host memory for output on run " << run << " thread " << thr;
        }
    }

    // Init input with random values and zero-out the output
    uint64_t* input64 = (uint64_t*)input;
    for (int i = 0; i < tensorSize / sizeof(uint64_t); i++)
    {
        input64[i] = i + 1;
    }

    for (int run = 0; run < NUM_RUNS; run++)
    {
        for (int thr = 0; thr < NUM_THREADS; thr++)
        {
            memset(output[run][thr], 0x00, tensorSize);
        }
    }

    // Create streams
    synStreamHandle copyInStream, copyOutStream, computeStream;
    status = synStreamCreateGeneric(&copyInStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&computeStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create compute stream";
    status = synStreamCreateGeneric(&copyOutStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t pDeviceInput, pDeviceOutput[NUM_RUNS];
    status = synDeviceMalloc(deviceId, tensorSize, 0, 0, &pDeviceInput);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    for (int run = 0; run < NUM_RUNS; run++)
    {
        status = synDeviceMalloc(deviceId, tensorSize, 0, 0, &pDeviceOutput[run]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";
    }
    // Workspace for all the non-user-managed memory
    uint64_t workspaceSize;
    status = synWorkspaceGetSize(&workspaceSize, recipe);
    ASSERT_EQ(status, synSuccess) << "Failed to query required workspace size";

    uint64_t pWorkspace[NUM_RUNS] {};
    if (workspaceSize > 0)
    {
        for (int run = 0; run < NUM_RUNS; run++)
        {
            status = synDeviceMalloc(deviceId, workspaceSize, 0, 0, &pWorkspace[run]);
            ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory" << run;
        }
    }

    // Associate the tensors with the device memory so compute knows where to read from / write to
    synLaunchTensorInfo persistentTensorInfo[NUM_RUNS][numOfTensor];
    uint64_t            tensorIds[numOfTensor];
    ASSERT_EQ(synTensorRetrieveIds(recipe, tensorNames, tensorIds, numOfTensor), synSuccess);

    for (int run = 0; run < NUM_RUNS; run++)
    {
        persistentTensorInfo[run][0].tensorName     = "input";  // Must match the name supplied at tensor creation
        persistentTensorInfo[run][0].pTensorAddress = pDeviceInput;
        persistentTensorInfo[run][0].tensorType     = DATA_TENSOR;
        memset(&persistentTensorInfo[run][0].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
        persistentTensorInfo[run][0].tensorId = tensorIds[0];

        persistentTensorInfo[run][1].tensorName     = "output";  // Must match the name supplied at tensor creation
        persistentTensorInfo[run][1].pTensorAddress = pDeviceOutput[run];
        persistentTensorInfo[run][1].tensorType     = DATA_TENSOR;
        memset(&persistentTensorInfo[run][1].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
        persistentTensorInfo[run][1].tensorId = tensorIds[1];
    }

    synEventHandle copyDone, computeDone[NUM_RUNS];
    status = synEventCreate(&copyDone, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    for (int run = 0; run < NUM_RUNS; run++)
    {
        status = synEventCreate(&computeDone[run], deviceId, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create event";
    }
    // Copy data from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input, tensorSize, pDeviceInput, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    // Sync on device

    // Associate an event with its completion
    status = synEventRecord(copyDone, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // Compute waits for the copy to finish
    status = synStreamWaitEvent(computeStream, copyDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Schedule compute
    int numTensors = 2;

    std::vector<std::thread> threadVector;

    for (int run = 0; run < NUM_RUNS; run++)
    {
        status = synLaunch(computeStream, persistentTensorInfo[run], numTensors, pWorkspace[run], recipe, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";

        status = synEventRecord(computeDone[run], computeStream);
        ASSERT_EQ(status, synSuccess) << "Failed to record event";

        for (unsigned thread = 0; thread < NUM_THREADS; thread++)
        {
            std::thread th(&chkData,
                           thread,
                           copyOutStream,
                           computeDone[run],
                           pDeviceOutput[run],
                           tensorSize,
                           output[run][thread],
                           input,
                           deviceId);
            threadVector.push_back(std::move(th));
        }
    }

    for (auto& th : threadVector)
    {
        th.join();
    }

    synRecipeDestroy(recipe);

    synHostFree(deviceId, (void*)input, 0);
    for (int run = 0; run < NUM_RUNS; run++)
    {
        for (int i = 0; i < NUM_THREADS; i++)
        {
            synHostFree(deviceId, (void*)output[run][i], 0);
        }
    }

    synTensorDestroy(in_tensor);
    synTensorDestroy(out_tensor);

    synSectionDestroy(inSection);
    synSectionDestroy(outSection);

    synEventDestroy(copyDone);
    for (int run = 0; run < NUM_RUNS; run++)
    {
        synEventDestroy(computeDone[run]);
    }

    synStreamDestroy(copyInStream);
    synStreamDestroy(copyOutStream);
    synStreamDestroy(computeStream);

    synDeviceFree(deviceId, pDeviceInput, 0);
    for (int run = 0; run < NUM_RUNS; run++)
    {
        synDeviceFree(deviceId, pDeviceOutput[run], 0);

        if (workspaceSize > 0)
        {
            synDeviceFree(deviceId, pWorkspace[run], 0);
        }
    }

    status = synGraphDestroy(graph);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy graph";
}

/* Test: gaudi2_tpc_test_multi_d2h
 * ===================
 * This test creates a recipe NUM_THREADS * TPC node (sum). The on einput to each node is the same tensor, the second
 * input is different between the nodes. It then does one synLaunch and creates NUM_THREADS threads to check the results
 * synchronization between the streams is done in the most efficient way
 */
static void chkRes(int             idx,
                   synStreamHandle copyOutStream,
                   synEventHandle  computeDone,
                   uint64_t        pDeviceOutput,
                   uint64_t        tensorSize,
                   float*          output,
                   uint64_t        tensorSizeInElements,
                   float*          expectedRes,
                   int             devIdx)
{
    // Copy waits for compute to finish
    synStatus status = synStreamWaitEvent(copyOutStream, computeDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Copy data from device to host
    status = synMemCopyAsync(copyOutStream, pDeviceOutput, tensorSize, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    synEventHandle outDoneEvent;
    status = synEventCreate(&outDoneEvent, devIdx, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event" << idx;

    status = synEventRecord(outDoneEvent, copyOutStream);
    ASSERT_EQ(status, synSuccess) << "Failed to event record" << idx;
    // Wait for everything to finish by blocking on the copy from device to host
    sleep(1);
    status = synEventSynchronize(outDoneEvent);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream" << idx;

    /*------------------------------------- Results validation ---------------------------*/
    int res = memcmp(output, expectedRes, tensorSizeInElements * sizeof(float));

    ASSERT_EQ(res, 0);
}

#define inTensor1Name "input1"
#define inTensor2Name "input2"
#define outTensorName "output"

// See description abouve
TEST_F_SYN(RtMultiDev2Host, gaudi2_tpc_test_multi_d2h)
{
    constexpr int NUM_THREADS = 5;

    synStatus status;

    /*------------------------------------- Create graph ---------------------------*/
    synGraphHandle graph;
    status = synGraphCreate(&graph, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create graph";

    // Create input and output sections for managing device memory
    synSectionHandle inSection1;
    status = synSectionCreate(&inSection1, 0, graph);
    ASSERT_EQ(status, synSuccess) << "Failed to create input section";
    synSectionHandle inSection2[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
        status = synSectionCreate(&inSection2[i], 0, graph);
        ASSERT_EQ(status, synSuccess) << "Failed to create input section";
    }

    synSectionHandle outSection[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
        status = synSectionCreate(&outSection[i], 0, graph);
        ASSERT_EQ(status, synSuccess) << "Failed to create output section";
    }

    constexpr unsigned dims                           = 4;
    constexpr unsigned N                              = 4096;
    constexpr unsigned W                              = 4;
    constexpr unsigned H                              = 4;
    constexpr unsigned B                              = 1;
    unsigned           tensorDimSizes[HABANA_DIM_MAX] = {N, W, H, B};
    constexpr uint64_t tensorSizeInElements           = N * W * H * B;
    constexpr uint64_t tensorSize                     = tensorSizeInElements * sizeof(float);

    // Tensors
    synTensor           inTensor1;
    synTensor           inTensor2[NUM_THREADS];
    synTensor           outTensor[NUM_THREADS];
    synTensorDescriptor desc;

    std::vector<std::array<float, tensorSizeInElements>> expectedRes(NUM_THREADS);

    desc.m_dataType = syn_type_single;
    desc.m_dims     = dims;
    desc.m_name     = inTensor1Name;
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i)
    {
        desc.m_sizes[i] = tensorDimSizes[i];
    }

    status = synTensorCreate(&inTensor1, &desc, inSection1, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create input1 tensor";

    std::string nameIn[NUM_THREADS];
    std::string nameOut[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
        nameIn[i]  = inTensor2Name + std::to_string(i);
        nameOut[i] = outTensorName + std::to_string(i);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        desc.m_name = nameIn[i].c_str();
        status      = synTensorCreate(&inTensor2[i], &desc, inSection2[i], 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create input2 tensor";
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        desc.m_name = nameOut[i].c_str();
        status      = synTensorCreate(&outTensor[i], &desc, outSection[i], 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create output tensor";
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        synTensor inputs[2]  = {inTensor1, inTensor2[i]};
        synTensor outputs[1] = {outTensor[i]};

        // Create add_f32 node
        status = synNodeCreate(graph,  // associated graph
                               inputs,
                               outputs,  // input/output tensor vectors
                               2,
                               1,  // input/output tensor vector sizes
                               nullptr,
                               0,  // user params
                               "add_fwd_f32",
                               "addNode",  // guid and node name
                               nullptr,
                               nullptr);  // input/output layouts
        ASSERT_EQ(status, synSuccess) << "Failed to create node";
    }
    /*------------------------------------- Compile and execute graph ---------------------------*/
    // Compile the graph to get an executable recipe
    synRecipeHandle recipe;
    status = synGraphCompile(&recipe, graph, "my_graph", nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";

    // Execution
    // Host buffers for input & output
    float* input1;
    float* input2[NUM_THREADS];
    float* output[NUM_THREADS];

    synDeviceId deviceId;
    status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to acquire device";

    // Input and output need to be mapped to the device as they are copied from / to
    status = synHostMalloc(deviceId, tensorSize, 0, (void**)&input1);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input1";

    for (int i = 0; i < NUM_THREADS; i++)
    {
        status = synHostMalloc(deviceId, tensorSize, 0, (void**)&input2[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input2";
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        status = synHostMalloc(deviceId, tensorSize, 0, (void**)&output[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output" << i;
    }

    // Init input with random values and zero-out the output
    std::pair<float, float> range = {-50.0f, 50.0f};
    fillWithRandom<float>(input1, tensorSizeInElements, range);

    for (int i = 0; i < NUM_THREADS; i++)
    {
        fillWithRandom<float>(input2[i], tensorSizeInElements, range);
        for (int elem = 0; elem < tensorSizeInElements; elem++)
        {
            input2[i][elem] += i;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        for (int elem = 0; elem < tensorSizeInElements; elem++)
        {
            expectedRes[i][elem] = input1[elem] + input2[i][elem];
        }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        memset(output[i], 0, tensorSize);
    }
    // Create streams
    synStreamHandle copyInStream;
    synStreamHandle copyOutStream;
    synStreamHandle computeStream;
    status = synStreamCreateGeneric(&copyInStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&computeStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create compute stream";
    status = synStreamCreateGeneric(&copyOutStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    // Device-side (HBM) buffers for input and output
    uint64_t pDeviceInput1;
    uint64_t pDeviceInput2[NUM_THREADS];
    uint64_t pDeviceOutput[NUM_THREADS];
    status = synDeviceMalloc(deviceId, tensorSize, 0, 0, &pDeviceInput1);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    for (int i = 0; i < NUM_THREADS; i++)
    {
        status = synDeviceMalloc(deviceId, tensorSize, 0, 0, &pDeviceInput2[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
        status = synDeviceMalloc(deviceId, tensorSize, 0, 0, &pDeviceOutput[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";
    }

    // Workspace for all the non-user-managed memory
    uint64_t workspaceSize;
    status = synWorkspaceGetSize(&workspaceSize, recipe);
    ASSERT_EQ(status, synSuccess) << "Failed to query required workspace size";

    uint64_t pWorkspace = 0;
    if (workspaceSize > 0)
    {
        status = synDeviceMalloc(deviceId, workspaceSize, 0, 0, &pWorkspace);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";
    }

    // Associate the tensors with the device memory so compute knows where to read from / write to
    std::vector<synLaunchTensorInfo> launchTensors(1 + NUM_THREADS * 2);

    launchTensors[0].pTensorAddress = pDeviceInput1;
    launchTensors[0].tensorName     = inTensor1Name;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        launchTensors[1 + i].pTensorAddress = pDeviceInput2[i];
        launchTensors[1 + i].tensorName     = nameIn[i].c_str();
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        launchTensors[1 + NUM_THREADS + i].pTensorAddress = pDeviceOutput[i];
        launchTensors[1 + NUM_THREADS + i].tensorName     = nameOut[i].c_str();
    }

    uint32_t totalNumOfTensors = launchTensors.size();
    prepareTensorInfo(recipe, launchTensors.data(), totalNumOfTensors);

    synEventHandle copyDone;
    synEventHandle computeDone;
    status = synEventCreate(&copyDone, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    status = synEventCreate(&computeDone, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create event";

    // Copy data from host to device
    status = synMemCopyAsync(copyInStream, (uint64_t)input1, tensorSize, pDeviceInput1, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    for (int i = 0; i < NUM_THREADS; i++)
    {
        status = synMemCopyAsync(copyInStream, (uint64_t)input2[i], tensorSize, pDeviceInput2[i], HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
    }

    // Associate an event with its completion
    status = synEventRecord(copyDone, copyInStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    // Compute waits for the copy to finish
    status = synStreamWaitEvent(computeStream, copyDone, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

    // Schedule compute
    status = synLaunch(computeStream, launchTensors.data(), launchTensors.size(), pWorkspace, recipe, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to launch graph";

    // Associate an event with its completion
    status = synEventRecord(computeDone, computeStream);
    ASSERT_EQ(status, synSuccess) << "Failed to record event";

    std::vector<std::thread> threadVector;
    for (unsigned thread = 0; thread < NUM_THREADS; thread++)
    {
        std::thread th(&chkRes,
                       thread,
                       copyOutStream,
                       computeDone,
                       pDeviceOutput[thread],
                       tensorSize,
                       output[thread],
                       tensorSizeInElements,
                       &expectedRes[thread][0],
                       deviceId);
        threadVector.push_back(std::move(th));
    }

    for (auto& th : threadVector)
    {
        th.join();
    }

    /*------------------------------------- Destroy ---------------------------*/

    ASSERT_EQ(synRecipeDestroy(recipe), synSuccess);

    ASSERT_EQ(synHostFree(deviceId, (void*)input1, 0), synSuccess);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        ASSERT_EQ(synHostFree(deviceId, (void*)input2[i], 0), synSuccess);
        ASSERT_EQ(synHostFree(deviceId, (void*)output[i], 0), synSuccess);
    }

    synTensorDestroy(inTensor1);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        ASSERT_EQ(synTensorDestroy(inTensor2[i]), synSuccess);
        ASSERT_EQ(synTensorDestroy(outTensor[i]), synSuccess);
    }

    ASSERT_EQ(synSectionDestroy(inSection1), synSuccess);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        ASSERT_EQ(synSectionDestroy(inSection2[i]), synSuccess);
        ASSERT_EQ(synSectionDestroy(outSection[i]), synSuccess);
    }

    ASSERT_EQ(synEventDestroy(copyDone), synSuccess);
    ASSERT_EQ(synEventDestroy(computeDone), synSuccess);

    ASSERT_EQ(synStreamDestroy(copyInStream), synSuccess);
    ASSERT_EQ(synStreamDestroy(copyOutStream), synSuccess);
    ASSERT_EQ(synStreamDestroy(computeStream), synSuccess);

    ASSERT_EQ(synDeviceFree(deviceId, pDeviceInput1, 0), synSuccess);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        ASSERT_EQ(synDeviceFree(deviceId, pDeviceInput2[i], 0), synSuccess);
        ASSERT_EQ(synDeviceFree(deviceId, pDeviceOutput[i], 0), synSuccess);
    }
    if (workspaceSize > 0)
    {
        synDeviceFree(deviceId, pWorkspace, 0);
    }
    synGraphDestroy(graph);
}

#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "synapse_api.h"
#include "global_conf_test_setter.h"
#include "hpp/synapse.hpp"

class SynGaudi2NoInfraReduction : public SynTest
{
public:
    SynGaudi2NoInfraReduction()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi2");
            m_deviceType = synDeviceGaudi2;
        }
        setSupportedDevices({synDeviceGaudi2});
    }

    virtual void SetUpTest() override;
    virtual void TearDownTest() override;
};

void SynGaudi2NoInfraReduction::SetUpTest()
{
    ReleaseDevice(); // the test manages device acquire
    if (!shouldRunTest()) GTEST_SKIP() << m_testConfig.skipReason();
    SetTestFileName();
    m_setupStatus = true;
}

void SynGaudi2NoInfraReduction::TearDownTest()
{
    printProfileInformation();
    CleanTestIntermediatesFiles();
}

TEST_F_GC(SynGaudi2NoInfraReduction, basic_dma_with_reduction, {synDeviceGaudi2})
{
    // Tin1 reductionOp Tin2 => Tout
    int       NUM_RUNS              = 4;
    const int reductionOp[NUM_RUNS] = {0 /* ADD */, 1 /* SUB */, 2 /* MIN */, 3 /* MAX */};

    syn::Context         context;
    GlobalConfTestSetter conf1("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GlobalConfTestSetter conf2("ENABLE_INTERNAL_NODES", "true");

    synStatus status = synSuccess;
    for (size_t RUN = 0; RUN < NUM_RUNS; RUN++)
    {
        synGraphHandle graph;
        status = synGraphCreate(&graph, synDeviceGaudi2);
        ASSERT_EQ(status, synSuccess) << "Failed to create graph";

        // Create input and output sections for managing device memory
        synSectionHandle in1Section, in2Section, outSection;
        status = synSectionCreate(&in1Section, 0, graph);
        ASSERT_EQ(status, synSuccess) << "Failed to create input section";
        status = synSectionCreate(&in2Section, 0, graph);
        ASSERT_EQ(status, synSuccess) << "Failed to create input section";
        status = synSectionCreate(&outSection, 0, graph);
        ASSERT_EQ(status, synSuccess) << "Failed to create output section";

        // Tensor sizes
        const unsigned dims                               = 4U;
        const unsigned Z                                  = 1U;
        const unsigned W                                  = 4U;
        const unsigned H                                  = 4U;
        const unsigned batch                              = 1U;
        unsigned       tensorDimSizes[SYN_MAX_TENSOR_DIM] = {Z, W, H, batch};
        uint64_t       tensorSize                         = Z * W * H * batch * sizeof(float);

        // Tensors
        synTensor           in1_tensor, in2_tensor, memcpy1_tensor, memcpy2_tensor, out_tensor;
        synTensorDescriptor desc;
        uint32_t            numOfTensor              = 5;
        const char*         tensorNames[numOfTensor] = {"input1", "input2", "memcpy1", "memcpy2", "output"};

        desc.m_dataType = syn_type_float;
        desc.m_dims     = dims;
        desc.m_name     = tensorNames[0];
        memset(desc.m_strides, 0, sizeof(desc.m_strides));
        for (unsigned i = 0; i < dims; ++i)
        {
            desc.m_sizes[i] = tensorDimSizes[i];
        }

        status = synTensorCreate(&in1_tensor, &desc, in1Section, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create input1 tensor";

        desc.m_name = tensorNames[1];
        status      = synTensorCreate(&in2_tensor, &desc, in2Section, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create input1 tensor";

        desc.m_name = tensorNames[2];
        status      = synTensorCreate(&memcpy1_tensor, &desc, nullptr, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create intermediate1 tensor";

        desc.m_name = tensorNames[3];
        status      = synTensorCreate(&memcpy2_tensor, &desc, nullptr, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create intermediate2 tensor";

        desc.m_name = tensorNames[4];
        status      = synTensorCreate(&out_tensor, &desc, outSection, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create output tensor";

        // Create DMA nodes
        status = synNodeCreate(graph,
                               &in1_tensor,
                               &memcpy1_tensor,
                               1,
                               1,
                               nullptr,
                               0,
                               "memcpy",
                               "memcpy1_node",
                               nullptr,
                               nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create DMA node";

        status = synNodeCreate(graph,
                               &in2_tensor,
                               &memcpy2_tensor,
                               1,
                               1,
                               nullptr,
                               0,
                               "memcpy",
                               "memcpy2_node",
                               nullptr,
                               nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create DMA node";

        // Create reduction node
        int       reduction_op      = reductionOp[RUN];
        synTensor reductionInput[2] = {memcpy1_tensor, memcpy2_tensor};
        status                      = synNodeCreate(graph,
                               reductionInput,
                               &out_tensor,
                               2,
                               1,
                               &reduction_op,
                               sizeof(reduction_op),
                               "reduction",
                               "reduction_node",
                               nullptr,
                               nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create reduction node";

        // Compile the graph to get an executable recipe
        synRecipeHandle recipe;
        status = synGraphCompile(&recipe, graph, "my_graph", nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to compile graph";

        // Execution
        synDeviceId devId;
        status = synDeviceAcquireByDeviceType(&devId, synDeviceGaudi2);
        ASSERT_EQ(status, synSuccess) << "No available Gaudi2 devices! Did you forget to run the simulator?";

        // Host buffers for input & output
        float *input1, *input2, *output, *expected;

        // Input and output need to be mapped to the device as they are copied from / to
        status = synHostMalloc(devId, tensorSize, 0, (void**)&input1);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
        status = synHostMalloc(devId, tensorSize, 0, (void**)&input2);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
        status = synHostMalloc(devId, tensorSize, 0, (void**)&output);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";
        status = synHostMalloc(devId, tensorSize, 0, (void**)&expected);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

        // Init input with some values and zero-out the output
        std::fill_n(input1, tensorSize / sizeof(float), 2.0);
        std::fill_n(input2, tensorSize / sizeof(float), 1.0);
        std::fill_n(output, tensorSize / sizeof(float), 0.0);
        switch (reductionOp[RUN])
        {
            case 0:  // Tin1 + Tin2
                std::fill_n(expected, tensorSize / sizeof(float), 2.0 + 1.0);
                break;
            case 1:  // Tin1 - Tin2
                std::fill_n(expected, tensorSize / sizeof(float), 2.0 - 1.0);
                break;
            case 2:  // MIN(Tin1, Tin2)
                std::fill_n(expected, tensorSize / sizeof(float), std::min(2.0, 1.0));
                break;
            case 3:  // MAX(Tin1, Tin2)
                std::fill_n(expected, tensorSize / sizeof(float), std::max(2.0, 1.0));
                break;
            default:
                break;
        }

        // Create streams
        synStreamHandle copyInStream, copyOutStream, computeStream;
        status = synStreamCreateGeneric(&copyInStream, devId, 0);
        ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
        status = synStreamCreateGeneric(&computeStream, devId, 0);
        ASSERT_EQ(status, synSuccess) << "Could not create compute stream";
        status = synStreamCreateGeneric(&copyOutStream, devId, 0);
        ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

        // Device-side (HBM) buffers for input and output
        uint64_t pDeviceInput1, pDeviceInput2, pDeviceOutput;
        status = synDeviceMalloc(devId, tensorSize, 0, 0, &pDeviceInput1);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
        status = synDeviceMalloc(devId, tensorSize, 0, 0, &pDeviceInput2);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
        status = synDeviceMalloc(devId, tensorSize, 0, 0, &pDeviceOutput);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";

        // Workspace for all the non-user-managed memory
        uint64_t workspaceSize;
        status = synWorkspaceGetSize(&workspaceSize, recipe);
        ASSERT_EQ(status, synSuccess) << "Failed to query required workspace size";

        uint64_t pWorkspace = 0;
        if (workspaceSize > 0)
        {
            status = synDeviceMalloc(devId, workspaceSize, 0, 0, &pWorkspace);
            ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";
        }

        // Associate the tensors with the device memory so compute knows where to read from / write to
        synLaunchTensorInfo persistentTensorInfo[numOfTensor];
        uint64_t            tensorIds[numOfTensor];
        ASSERT_EQ(synTensorRetrieveIds(recipe, tensorNames, tensorIds, numOfTensor), synSuccess);

        persistentTensorInfo[0].tensorName     = "input1";  // Must match the name supplied at tensor creation
        persistentTensorInfo[0].pTensorAddress = pDeviceInput1;
        persistentTensorInfo[0].tensorType     = DATA_TENSOR;
        memcpy(&persistentTensorInfo[0].tensorSize[0], tensorDimSizes, SYN_MAX_TENSOR_DIM * sizeof(uint32_t));
        persistentTensorInfo[0].tensorId       = tensorIds[0];
        persistentTensorInfo[1].tensorName     = "input2";  // Must match the name supplied at tensor creation
        persistentTensorInfo[1].pTensorAddress = pDeviceInput2;
        persistentTensorInfo[1].tensorType     = DATA_TENSOR;
        memcpy(&persistentTensorInfo[1].tensorSize[0], tensorDimSizes, SYN_MAX_TENSOR_DIM * sizeof(uint32_t));
        persistentTensorInfo[1].tensorId       = tensorIds[1];
        persistentTensorInfo[2].tensorName     = "output";  // Must match the name supplied at tensor creation
        persistentTensorInfo[2].pTensorAddress = pDeviceOutput;
        persistentTensorInfo[2].tensorType     = DATA_TENSOR;
        memcpy(&persistentTensorInfo[2].tensorSize[0], tensorDimSizes, SYN_MAX_TENSOR_DIM * sizeof(uint32_t));
        persistentTensorInfo[2].tensorId = tensorIds[4];

        synEventHandle copyDone, computeDone;
        status = synEventCreate(&copyDone, devId, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create event";

        status = synEventCreate(&computeDone, devId, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create event";

        // Copy data from host to device
        status = synMemCopyAsync(copyInStream, (uint64_t)input1, tensorSize, pDeviceInput1, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
        status = synMemCopyAsync(copyInStream, (uint64_t)input2, tensorSize, pDeviceInput2, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

        // Sync on device

        // Associate an event with its completion
        status = synEventRecord(copyDone, copyInStream);
        ASSERT_EQ(status, synSuccess) << "Failed to record event";

        // Compute waits for the copy to finish
        status = synStreamWaitEvent(computeStream, copyDone, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

        // Schedule compute
        status = synLaunch(computeStream, persistentTensorInfo, 3, pWorkspace, recipe, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";

        // Associate an event with its completion
        status = synEventRecord(computeDone, computeStream);
        ASSERT_EQ(status, synSuccess) << "Failed to record event";

        // Copy waits for compute to finish
        status = synStreamWaitEvent(copyOutStream, computeDone, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to stream wait event";

        // Copy data from device to host
        status = synMemCopyAsync(copyOutStream, pDeviceOutput, tensorSize, (uint64_t)output, DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

        // Wait for everything to finish by blocking on the copy from device to host
        status = synStreamSynchronize(copyOutStream);
        ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

        // Check results
        ASSERT_EQ(memcmp(expected, output, tensorSize), 0) << "Wrong results";

        synRecipeDestroy(recipe);

        synHostFree(devId, (void*)input1, 0);
        synHostFree(devId, (void*)input2, 0);
        synHostFree(devId, (void*)output, 0);

        synEventDestroy(copyDone);
        synEventDestroy(computeDone);

        synStreamDestroy(copyInStream);
        synStreamDestroy(copyOutStream);
        synStreamDestroy(computeStream);

        synDeviceFree(devId, pDeviceInput1, 0);
        synDeviceFree(devId, pDeviceInput2, 0);
        synDeviceFree(devId, pDeviceOutput, 0);
        if (workspaceSize > 0)
        {
            synDeviceFree(devId, pWorkspace, 0);
        }

        synTensorDestroy(in1_tensor);
        synTensorDestroy(in2_tensor);
        synTensorDestroy(memcpy1_tensor);
        synTensorDestroy(memcpy2_tensor);
        synTensorDestroy(out_tensor);

        synSectionDestroy(in1Section);
        synSectionDestroy(in2Section);
        synSectionDestroy(outSection);

        status = synDeviceRelease(devId);
        ASSERT_EQ(status, synSuccess) << "Failed to release device";

        status = synGraphDestroy(graph);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy graph";
    }
}
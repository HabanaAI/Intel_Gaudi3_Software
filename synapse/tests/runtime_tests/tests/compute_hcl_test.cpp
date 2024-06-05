#include "syn_base_test.hpp"
#include "../../utils/cpu_calculator.h"
#include "test_utils.h"
#include "internal/hccl_api_wrapper.inl"
#include "scoped_configuration_change.h"
#include "tensor_validator.inl"
#include "test_recipe_hcl.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

// This class Is built to run on 4-8 devices (or simulators)
// to have two ranks for the collective to perform allReduce on.

#define MASTER_RANK_ID 0

class SynGaudiMultiDevicesSimple
: public SynBaseTest
, public ::testing::WithParamInterface<uint64_t>
{
public:
    SynGaudiMultiDevicesSimple();
    ~SynGaudiMultiDevicesSimple() = default;

    void SetUp() override;
    void TearDown() override {}

    void startProcesses(int numberOfDevices, void (SynGaudiMultiDevicesSimple::*testFunction)(uint64_t, uint64_t));
    void runComputeAndHCL(uint64_t rank, uint64_t numberOfDevices);
    void runAllReduceAndAllGather(uint64_t rank, uint64_t numberOfDevices);
    void runCollective(uint64_t rank, uint64_t numberOfDevices);
    void getDeviceCountTest(uint64_t rank, uint64_t numberOfDevices);
    void initSynapseAndHCL();

    static constexpr unsigned NUM_INPUTS  = 1;
    static constexpr unsigned NUM_OUTPUTS = 2;

protected:
    void      prepareHCLComm(unsigned numberOfDevices, unsigned rank, hcclComm_t* hclComm);
    synTensor createTensor(const TSize          dims,
                           const synDataType    dataType,
                           const TSize*         tensorSize,
                           const char*          name,
                           synSectionHandle&    pSectionHandle,
                           const synGraphHandle graphHandle,
                           const bool           isPersist = true);
    void      runAllReduce(uint32_t        deviceId,
                           uint64_t        rank,
                           synStreamHandle streamHandle,
                           hcclComm_t&     hclCommunicator,
                           uint64_t        numberOfDevices);
    void      runAllGather(uint32_t        deviceId,
                           uint64_t        rank,
                           synStreamHandle streamHandle,
                           hcclComm_t&     hclCommunicator,
                           uint64_t        numberOfDevices);

    const unsigned MAX_DEVICES = 8;

private:
    void (SynGaudiMultiDevicesSimple::*TestFunction)(uint64_t, uint64_t);

    std::unique_ptr<ScopedConfigurationChange> m_enableExperimental;
    std::unique_ptr<ScopedConfigurationChange> m_HcclCommId;
};

REGISTER_SUITE(SynGaudiMultiDevicesSimple, ALL_TEST_PACKAGES);

SynGaudiMultiDevicesSimple::SynGaudiMultiDevicesSimple()
{
    LOG_INFO(SYN_API, "{}", HLLOG_FUNC);
    if (m_deviceType == synDeviceTypeInvalid)
    {
        LOG_WARN(SYN_RT_TEST,
                 "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi");
        m_deviceType = synDeviceGaudi;
    }
    setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
}

// Override Setup to acquire device after fork
void SynGaudiMultiDevicesSimple::SetUp()
{
    /*
    This is the parent process which verifies if it is possible to run the test
    by checking the current available devices on the VM and if they are supported for this test.
    Each child process will synInitialize and acquire a device independently during the test.
    This is why we call synInitialize() -> synDeviceCount() -> synDestroy()
    */
    synStatus status = synInitialize();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RT_TEST, "synInitialize failed with rc: {}", status);
        return;
    }
    uint32_t deviceNum = synDeviceTypeSize - synDeviceGaudi;
    uint32_t deviceCount[deviceNum];
    status = synDeviceCount(deviceCount);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RT_TEST, "synDeviceCount failed with rc: {}", status);
        return;
    }
    status = synDestroy();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RT_TEST, "synDestroy failed with rc: {}", status);
        return;
    }

    uint32_t deviceType   = synDeviceGaudi;
    uint32_t numOfDevices = 0;
    for (; deviceType < deviceNum; deviceType++)
    {
        if (numOfDevices < deviceCount[deviceType])
        {
            m_deviceType = (synDeviceType)deviceType;
            numOfDevices = deviceCount[deviceType];
            break;
        }
    }

    if (deviceType == deviceNum)
    {
        GTEST_SKIP() << "No devices found on the VM";
    }

    if (numOfDevices < MAX_DEVICES)
    {
        GTEST_SKIP() << "Not enough devices to run HCL-Compute test";
    }
}

// HCL Init
void SynGaudiMultiDevicesSimple::prepareHCLComm(unsigned numberOfDevices, unsigned rank, hcclComm_t* hclComm)
{
    hcclResult_t result;
    hcclUniqueId uniqueId {};

    // Check if the current rank is the master
    if (rank == MASTER_RANK_ID)
    {
        result = hcclGetUniqueId(&uniqueId);
        ASSERT_EQ(result, hcclSuccess) << "Failed to get unique HCL ID";
    }

    // Add the current rank to the communicator
    result = hcclCommInitRank(hclComm, numberOfDevices, uniqueId, rank);
    ASSERT_EQ(result, hcclSuccess) << "Failed to init HCL rank";
}

synTensor SynGaudiMultiDevicesSimple::createTensor(const TSize          dims,
                                                   const synDataType    dataType,
                                                   const TSize*         tensorSize,
                                                   const char*          name,
                                                   synSectionHandle&    pSectionHandle,
                                                   const synGraphHandle graphHandle,
                                                   const bool           isPersist)
{
    synTensor tensor;
    synStatus status;

    if (!pSectionHandle)
    {
        uint64_t sectionDescriptor = 0;
        status                     = synSectionCreate(&pSectionHandle, sectionDescriptor, graphHandle);
        if (status != synSuccess)
        {
            return nullptr;
        }
    }

    status = synTensorHandleCreate(&tensor, graphHandle, DATA_TENSOR, name);  // << "Failed to create tensor handle";
    if (status != synSuccess)
    {
        return nullptr;
    }

    if (isPersist)
    {
        status = synSectionSetPersistent(pSectionHandle, true);
        if (status != synSuccess)
        {
            return nullptr;
        }

        status = synTensorAssignToSection(tensor, pSectionHandle, 0);
        if (status != synSuccess)
        {
            return nullptr;
        }
    }

    synTensorGeometry geometry;
    geometry.dims = dims;
    for (int i = 0; i < dims; i++)
        geometry.sizes[i] = tensorSize[i];
    status = synTensorSetGeometry(tensor, &geometry, synGeometryMaxSizes);
    if (status != synSuccess)
    {
        return nullptr;
    }

    synTensorDeviceFullLayout deviceLayout;
    deviceLayout.deviceDataType = dataType;

    std::fill_n(deviceLayout.strides, sizeof(deviceLayout.strides) / sizeof(decltype(deviceLayout.strides[0])), 0);

    status = synTensorSetDeviceFullLayout(tensor, &deviceLayout);
    if (status != synSuccess)
    {
        return nullptr;
    }

    return tensor;
}

void SynGaudiMultiDevicesSimple::runAllReduce(uint32_t        deviceId,
                                              uint64_t        rank,
                                              synStreamHandle streamHandle,
                                              hcclComm_t&     hclCommunicator,
                                              uint64_t        numberOfDevices)
{
    uint64_t count       = 1024;
    uint64_t sizeInBytes = count * sizeof(float);

    float* inData;
    float* outData;

    uint64_t bufferOnDevice;
    uint64_t bufferOnDevice2;

    ASSERT_EQ(synSuccess, synHostMalloc(deviceId, sizeInBytes, 0, (void**)&inData));
    ASSERT_EQ(synSuccess, synHostMalloc(deviceId, sizeInBytes, 0, (void**)&outData));

    ASSERT_EQ(synSuccess, synDeviceMalloc(deviceId, sizeInBytes, 0, 0, &bufferOnDevice));
    ASSERT_EQ(synSuccess, synDeviceMalloc(deviceId, sizeInBytes, 0, 0, &bufferOnDevice2));

    for (uint64_t i = 0; i < count; i++)
    {
        inData[i] = rank + 1;
    }

    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, (uint64_t)inData, sizeInBytes, bufferOnDevice, HOST_TO_DRAM));

    ASSERT_EQ(hcclSuccess,
              hcclAllReduce(reinterpret_cast<void*>(bufferOnDevice),
                            reinterpret_cast<void*>(bufferOnDevice2),
                            count,
                            hcclFloat,
                            hcclSum,
                            hclCommunicator,
                            streamHandle));

    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, bufferOnDevice2, sizeInBytes, (uint64_t)outData, DRAM_TO_HOST));
    ASSERT_EQ(synSuccess, synStreamSynchronize(streamHandle)) << "Failed synchronization after memcopy";

    uint64_t expectedResult = (numberOfDevices * (numberOfDevices + 1)) / 2;

    for (uint64_t i = 0; i < count; i++)
    {
        ASSERT_EQ(expectedResult, outData[i])
            << "failed in different send and receive buffers of allReduce on " + std::to_string(i);
    }

    synDeviceFree(deviceId, bufferOnDevice, 0);
    synDeviceFree(deviceId, bufferOnDevice2, 0);

    synHostFree(deviceId, inData, 0);
    synHostFree(deviceId, outData, 0);
}

void SynGaudiMultiDevicesSimple::runAllGather(uint32_t        deviceId,
                                              uint64_t        rank,
                                              synStreamHandle streamHandle,
                                              hcclComm_t&     hclCommunicator,
                                              uint64_t        numberOfDevices)
{
    uint64_t count       = 16;
    uint64_t sizeInBytes = count * sizeof(float);

    float* inData;
    float* outData;

    uint64_t bufferOnDevice;
    uint64_t bufferOnDevice2;

    ASSERT_EQ(synSuccess, synHostMalloc(deviceId, sizeInBytes, 0, (void**)&inData));
    ASSERT_EQ(synSuccess, synHostMalloc(deviceId, sizeInBytes, 0, (void**)&outData));

    ASSERT_EQ(synSuccess, synDeviceMalloc(deviceId, sizeInBytes, 0, 0, &bufferOnDevice));
    ASSERT_EQ(synSuccess, synDeviceMalloc(deviceId, sizeInBytes, 0, 0, &bufferOnDevice2));

    for (uint64_t i = 0; i < count; i++)
    {
        inData[i] = rank + 1;
    }

    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, (uint64_t)inData, sizeInBytes, bufferOnDevice, HOST_TO_DRAM));

    uint64_t smallCount = count / numberOfDevices;

    ASSERT_EQ(hcclSuccess,
              hcclAllGather(reinterpret_cast<void*>(bufferOnDevice),
                            reinterpret_cast<void*>(bufferOnDevice2),
                            smallCount,
                            hcclFloat,
                            hclCommunicator,
                            streamHandle));

    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, bufferOnDevice2, sizeInBytes, (uint64_t)outData, DRAM_TO_HOST));
    ASSERT_EQ(synSuccess, synStreamSynchronize(streamHandle)) << "Failed synchronization after memcopy";

    uint64_t expectedResult;

    for (uint64_t i = 0; i < count; i++)
    {
        expectedResult = (i / smallCount) + 1;
        ASSERT_EQ(expectedResult, outData[i])
            << "failed in different send and receive buffers of allGather on " + std::to_string(i);
    }
    synDeviceFree(deviceId, bufferOnDevice, 0);
    synDeviceFree(deviceId, bufferOnDevice2, 0);

    synHostFree(deviceId, inData, 0);
    synHostFree(deviceId, outData, 0);
}

// Process creation util - aligned with HCL's python testing scripts
void SynGaudiMultiDevicesSimple::startProcesses(int numberOfDevices,
                                                void (SynGaudiMultiDevicesSimple::*testFunction)(uint64_t, uint64_t))
{
    // Need to set up N devices, and set one of them to be HCL master of puppets
    uint64_t pid    = 0;
    uint64_t rank   = 0;
    uint64_t ppid   = getpid();
    int      status = 0;

    TestFunction = testFunction;

    for (int i = 0; i < numberOfDevices; i++)
    {
        pid = fork();
        if (pid == 0)
        {
            rank = i;
            break;
        }
    }

    if (pid == 0)
    {
        (this->*TestFunction)(rank, numberOfDevices);
    }
    else
    {
        while (wait(NULL) > 0)
            ;
        ASSERT_EQ(status, 0) << "Test run has errors!";
    }

    if (getpid() != ppid)
    {
        exit(testing::Test::HasFailure());
    }
}

void SynGaudiMultiDevicesSimple::runComputeAndHCL(uint64_t rank, uint64_t numberOfDevices)
{
    initSynapseAndHCL();

    TestRecipeHcl recipe(m_deviceType, false /* isSfgGraph */);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    hcclComm_t hclCommunicator;
    prepareHCLComm(numberOfDevices, rank, &hclCommunicator);

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;

    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());

    runAllReduce(device.getDeviceId(), rank, streamHandle, hclCommunicator, numberOfDevices);
    runAllGather(device.getDeviceId(), rank, streamHandle, hclCommunicator, numberOfDevices);
}

void SynGaudiMultiDevicesSimple::runAllReduceAndAllGather(uint64_t rank, uint64_t numberOfDevices)
{
    initSynapseAndHCL();

    TestDevice device(m_deviceType);

    hcclComm_t hclCommunicator;
    prepareHCLComm(numberOfDevices, rank, &hclCommunicator);

    synStreamHandle streamHandle;
    ASSERT_EQ(synSuccess, synStreamCreateGeneric(&streamHandle, device.getDeviceId(), 0));

    uint64_t count       = 41943040;  // 40MB
    uint64_t sizeInBytes = count * sizeof(float);

    float* inData;
    float* outData;

    uint64_t bufferOnDevice;
    uint64_t bufferOnDevice2;
    uint64_t bufferOnDevice3;

    ASSERT_EQ(synSuccess, synHostMalloc(device.getDeviceId(), sizeInBytes, 0, (void**)&inData));
    ASSERT_EQ(synSuccess, synHostMalloc(device.getDeviceId(), sizeInBytes, 0, (void**)&outData));

    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), sizeInBytes, 0, 0, &bufferOnDevice));
    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), sizeInBytes, 0, 0, &bufferOnDevice2));

    for (uint64_t i = 0; i < count; i++)
    {
        inData[i] = rank + 1;
    }

    uint64_t smallCount = count / numberOfDevices;

    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, (uint64_t)inData, sizeInBytes, bufferOnDevice, HOST_TO_DRAM));

    ASSERT_EQ(hcclSuccess,
              hcclAllGather(reinterpret_cast<void*>(bufferOnDevice),
                            reinterpret_cast<void*>(bufferOnDevice2),
                            smallCount,
                            hcclFloat,
                            hclCommunicator,
                            streamHandle));
    ASSERT_EQ(synSuccess,
              synMemCopyAsync(streamHandle,
                              bufferOnDevice2,
                              sizeInBytes,
                              (uint64_t)outData,
                              DRAM_TO_HOST));  // Buffer in fractions

    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), sizeInBytes, 0, 0, &bufferOnDevice3));
    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, bufferOnDevice2, sizeInBytes, bufferOnDevice3, DRAM_TO_DRAM));

    ASSERT_EQ(hcclSuccess,
              hcclAllReduce(reinterpret_cast<void*>(bufferOnDevice3),
                            reinterpret_cast<void*>(bufferOnDevice2),
                            count,
                            hcclFloat,
                            hcclSum,
                            hclCommunicator,
                            streamHandle));

    ASSERT_EQ(synSuccess, synMemCopyAsync(streamHandle, bufferOnDevice2, sizeInBytes, (uint64_t)outData, DRAM_TO_HOST));

    ASSERT_EQ(synSuccess, synStreamSynchronize(streamHandle)) << "Failed synchronization after memcopy";

    uint64_t expectedResult;
    for (uint64_t i = 0; i < count; i++)
    {
        expectedResult = numberOfDevices * ((i / smallCount) + 1);
        ASSERT_EQ(expectedResult, outData[i])
            << "failed in different send and receive buffers of allReduce on " + std::to_string(i);
    }

    synDeviceFree(device.getDeviceId(), bufferOnDevice, 0);
    synDeviceFree(device.getDeviceId(), bufferOnDevice2, 0);
    synDeviceFree(device.getDeviceId(), bufferOnDevice3, 0);

    synHostFree(device.getDeviceId(), inData, 0);
    synHostFree(device.getDeviceId(), outData, 0);
}

void SynGaudiMultiDevicesSimple::runCollective(uint64_t rank, uint64_t numberOfDevices)
{
    initSynapseAndHCL();

    TestDevice device(m_deviceType);

    hcclComm_t hclCommunicator;
    prepareHCLComm(numberOfDevices, rank, &hclCommunicator);

    synStreamHandle streamHandle;
    ASSERT_EQ(synSuccess, synStreamCreateGeneric(&streamHandle, device.getDeviceId(), 0)) << "Failed create stream";

    synConvolutionParams params = {};

    std::array<TSize, MAX_DIMENSIONS_NUM> xTensorSizes = {32, 28, 28, 32, 1};
    std::array<TSize, MAX_DIMENSIONS_NUM> wTensorSizes = {1, xTensorSizes[0], params.kW, params.kH, 1};
    std::array<TSize, MAX_DIMENSIONS_NUM> yTensorSizes = {
        1,
        convOutputDimSize(xTensorSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
        convOutputDimSize(xTensorSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
        xTensorSizes[3],
        1};

    unsigned yTensorElements =
        std::accumulate(yTensorSizes.begin(), yTensorSizes.end(), 1, std::multiplies<unsigned int>());

    uint64_t  yTensorBufferDevice;
    uint64_t  yTensorReluBufferDevice;
    Bfloat16* yTensorBufferHost;

    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), yTensorElements * 2, 0, 0, &yTensorBufferDevice));
    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), yTensorElements * 2, 0, 0, &yTensorReluBufferDevice));
    ASSERT_EQ(synSuccess, synHostMalloc(device.getDeviceId(), yTensorElements * 2, 0, (void**)&yTensorBufferHost));

    // Main rank
    if (rank == 0)
    {
        synGraphHandle  graphRelu;
        synRecipeHandle recipeHandleRelu;
        const char*     reluRecipePath = "relu.pb";

        randomBufferValues(MEM_INIT_RANDOM_WITH_NEGATIVE,
                           syn_type_bf16,
                           getNumberOfElements(yTensorSizes.data(), 4),
                           yTensorBufferHost,
                           false);

        ((char*)yTensorBufferHost)[0] = 0x13;
        ((char*)yTensorBufferHost)[1] = 0x37;

        ASSERT_EQ(synSuccess, synGraphCreate(&graphRelu, m_deviceType));
        std::vector<synTensor> reluInputs(1);
        std::vector<synTensor> reluOutputs(1);

        synSectionHandle pSectionHandle_yTensor     = nullptr;
        synSectionHandle pSectionHandle_yTensorRelu = nullptr;
        uint64_t         sectionDescriptor          = 0;

        ASSERT_EQ(synSuccess, synSectionCreate(&pSectionHandle_yTensor, sectionDescriptor, graphRelu));
        ASSERT_EQ(synSuccess, synSectionCreate(&pSectionHandle_yTensorRelu, sectionDescriptor, graphRelu));

        reluInputs[0] =
            createTensor(4U, syn_type_bf16, yTensorSizes.data(), "yTensor", pSectionHandle_yTensor, graphRelu);
        ASSERT_NE(reluInputs[0], nullptr) << "Failed to create tensor";
        reluOutputs[0] =
            createTensor(4U, syn_type_bf16, yTensorSizes.data(), "yTensorRelu", pSectionHandle_yTensorRelu, graphRelu);
        ASSERT_NE(reluOutputs[0], nullptr) << "Failed to create tensor";

        ASSERT_EQ(synSuccess,
                  synNodeCreate(graphRelu,
                                reluInputs.data(),
                                reluOutputs.data(),
                                1,
                                1,
                                nullptr,
                                0,
                                "relu_fwd_bf16",
                                "",
                                nullptr,
                                nullptr));

        ASSERT_EQ(synSuccess, synGraphCompile(&recipeHandleRelu, graphRelu, reluRecipePath, 0));

        uint64_t topologyReluWorkspaceSize   = 0;
        uint64_t topologyReluWorkspaceBuffer = 0;
        ASSERT_EQ(synSuccess, synWorkspaceGetSize(&topologyReluWorkspaceSize, recipeHandleRelu));
        ASSERT_EQ(synSuccess,
                  synDeviceMalloc(device.getDeviceId(), topologyReluWorkspaceSize, 0, 0, &topologyReluWorkspaceBuffer));

        ASSERT_EQ(synSuccess,
                  synMemCopyAsync(streamHandle,
                                  (uint64_t)yTensorBufferHost,
                                  yTensorElements * 2,
                                  yTensorBufferDevice,
                                  HOST_TO_DRAM));

        synLaunchTensorInfo reluTensorsList[2] = {
            {"yTensor", yTensorBufferDevice, DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
            {"yTensorRelu", yTensorReluBufferDevice, DATA_TENSOR, {0, 0, 0, 0, 0}, 0}};

        ASSERT_EQ(synSuccess,
                  synLaunch(streamHandle,
                            reluTensorsList,
                            2,
                            topologyReluWorkspaceBuffer,
                            recipeHandleRelu,
                            SYN_FLAGS_TENSOR_NAME));

        // Main rank Cleanup
        ASSERT_EQ(synSuccess, synDeviceFree(device.getDeviceId(), topologyReluWorkspaceBuffer, 0));
        ASSERT_EQ(synSuccess, synGraphDestroy(graphRelu));
        ASSERT_EQ(synSuccess, synRecipeDestroy(recipeHandleRelu));
    }

    ASSERT_EQ(hcclSuccess,
              hcclBroadcast(reinterpret_cast<void*>(yTensorBufferDevice),
                            reinterpret_cast<void*>(yTensorBufferDevice),
                            yTensorElements * 2,
                            hcclBfloat16,
                            0,
                            hclCommunicator,
                            streamHandle));
    synStreamSynchronize(streamHandle);

    ASSERT_EQ(hcclSuccess,
              hcclBroadcast(reinterpret_cast<void*>(yTensorReluBufferDevice),
                            reinterpret_cast<void*>(yTensorReluBufferDevice),
                            yTensorElements * 2,
                            hcclBfloat16,
                            0,
                            hclCommunicator,
                            streamHandle));
    synStreamSynchronize(streamHandle);

    ASSERT_EQ(synSuccess,
              synMemCopyAsync(streamHandle,
                              yTensorBufferDevice,
                              yTensorElements * 2,
                              (uint64_t)yTensorBufferHost,
                              DRAM_TO_HOST));
    ASSERT_EQ(synSuccess, synStreamSynchronize(streamHandle));

    ASSERT_EQ(0x13, ((char*)yTensorBufferHost)[0]);
    ASSERT_EQ(0x37, ((char*)yTensorBufferHost)[1]);

    unsigned xTensorElements =
        std::accumulate(xTensorSizes.begin(), xTensorSizes.end(), 1, std::multiplies<unsigned int>());
    unsigned wTensorElements =
        std::accumulate(wTensorSizes.begin(), wTensorSizes.end(), 1, std::multiplies<unsigned int>());

    uint64_t  xTensorBufferDevice;
    uint64_t  wTensorBufferDevice;
    Bfloat16* xTensorBufferHost;
    Bfloat16* wTensorBufferHost;

    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), xTensorElements * 2, 0, 0, &xTensorBufferDevice));
    ASSERT_EQ(synSuccess, synDeviceMalloc(device.getDeviceId(), wTensorElements * 4, 0, 0, &wTensorBufferDevice));
    ASSERT_EQ(synSuccess, synHostMalloc(device.getDeviceId(), xTensorElements * 2, 0, (void**)&xTensorBufferHost));
    ASSERT_EQ(synSuccess, synHostMalloc(device.getDeviceId(), wTensorElements * 4, 0, (void**)&wTensorBufferHost));

    synGraphHandle  graphDeDw;
    synRecipeHandle recipeHandleDeDw;
    const char*     dedwRecipePath = "dedw.pb";

    randomBufferValues(MEM_INIT_RANDOM_WITH_NEGATIVE,
                       syn_type_bf16,
                       getNumberOfElements(xTensorSizes.data(), 4),
                       xTensorBufferHost,
                       false);

    memset((void*)wTensorBufferHost, 0, wTensorElements * 4);

    ASSERT_EQ(synSuccess, synGraphCreate(&graphDeDw, m_deviceType));
    std::vector<synTensor> dedwInputs(2);
    std::vector<synTensor> dedwOutputs(1);

    synSectionHandle pSectionHandle_yTensorDeDw = nullptr;
    synSectionHandle pSectionHandle_xTensor     = nullptr;
    synSectionHandle pSectionHandle_wTensor     = nullptr;
    uint64_t         sectionDescriptor          = 0;

    ASSERT_EQ(synSuccess, synSectionCreate(&pSectionHandle_yTensorDeDw, sectionDescriptor, graphDeDw));
    ASSERT_EQ(synSuccess, synSectionCreate(&pSectionHandle_xTensor, sectionDescriptor, graphDeDw));
    ASSERT_EQ(synSuccess, synSectionCreate(&pSectionHandle_wTensor, sectionDescriptor, graphDeDw));

    dedwInputs[0] =
        createTensor(4U, syn_type_bf16, yTensorSizes.data(), "yTensorDeDw", pSectionHandle_yTensorDeDw, graphDeDw);
    ASSERT_NE(dedwInputs[0], nullptr) << "Failed to create tensor";
    dedwInputs[1] = createTensor(4U, syn_type_bf16, xTensorSizes.data(), "xTensor", pSectionHandle_xTensor, graphDeDw);
    ASSERT_NE(dedwInputs[1], nullptr) << "Failed to create tensor";
    dedwOutputs[0] =
        createTensor(4U, syn_type_float, wTensorSizes.data(), "wTensor", pSectionHandle_wTensor, graphDeDw);
    ASSERT_NE(dedwOutputs[0], nullptr) << "Failed to create tensor";

    ASSERT_EQ(synSuccess,
              synNodeCreate(graphDeDw,
                            dedwInputs.data(),
                            dedwOutputs.data(),
                            2,
                            1,
                            &params,
                            0,
                            "dedw",
                            "",
                            nullptr,
                            nullptr));

    ASSERT_EQ(synSuccess, synGraphCompile(&recipeHandleDeDw, graphDeDw, dedwRecipePath, 0));

    uint64_t topologyDeDwWorkspaceSize   = 0;
    uint64_t topologyDeDwWorkspaceBuffer = 0;
    ASSERT_EQ(synSuccess, synWorkspaceGetSize(&topologyDeDwWorkspaceSize, recipeHandleDeDw));
    ASSERT_EQ(synSuccess,
              synDeviceMalloc(device.getDeviceId(), topologyDeDwWorkspaceSize, 0, 0, &topologyDeDwWorkspaceBuffer));

    ASSERT_EQ(synSuccess,
              synMemCopyAsync(streamHandle,
                              (uint64_t)xTensorBufferHost,
                              xTensorElements * 2,
                              xTensorBufferDevice,
                              HOST_TO_DRAM));

    synLaunchTensorInfo dedwTensorsList[3] = {{"yTensorDeDw", yTensorReluBufferDevice, DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
                                              {"xTensor", xTensorBufferDevice, DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
                                              {"wTensor", wTensorBufferDevice, DATA_TENSOR, {0, 0, 0, 0, 0}, 0}};

    ASSERT_EQ(synSuccess,
              synLaunch(streamHandle,
                        dedwTensorsList,
                        3,
                        topologyDeDwWorkspaceBuffer,
                        recipeHandleDeDw,
                        SYN_FLAGS_TENSOR_NAME));

    ASSERT_EQ(synSuccess,
              synMemCopyAsync(streamHandle,
                              wTensorBufferDevice,
                              wTensorElements * 4,
                              (uint64_t)wTensorBufferHost,
                              DRAM_TO_HOST));
    ASSERT_EQ(synSuccess, synStreamSynchronize(streamHandle));

    Bfloat16* yData = new Bfloat16[yTensorElements];
    char*     xData = (char*)xTensorBufferHost;
    float*    wData = new float[wTensorElements];

    memcpy(yData, yTensorBufferHost, yTensorElements * sizeof(*yData));

    synTensorDescriptor yDesc = TestRecipeBase::getTensorDescriptor(syn_type_bf16,
                                                                    yTensorSizes.data(),
                                                                    4,
                                                                    "yTensorDeDw",
                                                                    nullptr,
                                                                    nullptr,
                                                                    false,
                                                                    nullptr,
                                                                    synTensorType::DATA_TENSOR);
    synTensorDescriptor xDesc = TestRecipeBase::getTensorDescriptor(syn_type_bf16,
                                                                    xTensorSizes.data(),
                                                                    4,
                                                                    "xTensor",
                                                                    nullptr,
                                                                    nullptr,
                                                                    false,
                                                                    nullptr,
                                                                    synTensorType::DATA_TENSOR);
    synTensorDescriptor wDesc = TestRecipeBase::getTensorDescriptor(syn_type_float,
                                                                    wTensorSizes.data(),
                                                                    4,
                                                                    "wTensor",
                                                                    nullptr,
                                                                    nullptr,
                                                                    false,
                                                                    nullptr,
                                                                    synTensorType::DATA_TENSOR);

    calculateRelu(yData, yTensorElements);
    calculateDEDW(yDesc, (char*)yData, xDesc, xData, wDesc, (char*)wData, params, m_deviceType);

    float* result = (float*)wTensorBufferHost;
    validateResult(wData, result, wTensorElements);

    delete[] wData;
    delete[] yData;

    // Cleanup
    ASSERT_EQ(synSuccess, synDeviceFree(device.getDeviceId(), yTensorBufferDevice, 0));
    ASSERT_EQ(synSuccess, synDeviceFree(device.getDeviceId(), yTensorReluBufferDevice, 0));
    ASSERT_EQ(synSuccess, synDeviceFree(device.getDeviceId(), xTensorBufferDevice, 0));
    ASSERT_EQ(synSuccess, synDeviceFree(device.getDeviceId(), wTensorBufferDevice, 0));
    ASSERT_EQ(synSuccess, synDeviceFree(device.getDeviceId(), topologyDeDwWorkspaceBuffer, 0));

    ASSERT_EQ(synSuccess, synHostFree(device.getDeviceId(), yTensorBufferHost, 0));
    ASSERT_EQ(synSuccess, synHostFree(device.getDeviceId(), xTensorBufferHost, 0));
    ASSERT_EQ(synSuccess, synHostFree(device.getDeviceId(), wTensorBufferHost, 0));

    ASSERT_EQ(synSuccess, synStreamDestroy(streamHandle));

    ASSERT_EQ(synSuccess, synDeviceRelease(device.getDeviceId()));

    ASSERT_EQ(synSuccess, synGraphDestroy(graphDeDw));

    ASSERT_EQ(synSuccess, synRecipeDestroy(recipeHandleDeDw));

    ASSERT_EQ(synSuccess, synDestroy());
}

void SynGaudiMultiDevicesSimple::getDeviceCountTest(uint64_t rank, uint64_t numberOfDevices)
{
    ASSERT_EQ(synInitialize(), synSuccess) << "Failed to initialize synapse";

    uint32_t count;
    ASSERT_EQ(synDeviceGetCount(&count), synSuccess) << "Failed on synDeviceGetCount";
    ASSERT_LE(count, MAX_DEVICES);
}

/* This function is required for a multi process testing environment.
   Since each process is independent, synInitialize is required for each starting process.
*/
void SynGaudiMultiDevicesSimple::initSynapseAndHCL()
{
    synStatus status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "synInitialize failed!";

    m_enableExperimental = std::make_unique<ScopedConfigurationChange>("ENABLE_EXPERIMENTAL_FLAGS", "true");
    // Initialize a global unique ID - to avoid the need to manually spread it among gaudi's
    m_HcclCommId         = std::make_unique<ScopedConfigurationChange>("HCCL_COMM_ID", "127.0.0.1:5656");
}

// Run once with 4 devices and once with 8 devices
INSTANTIATE_TEST_SUITE_P(, SynGaudiMultiDevicesSimple, ::testing::Values(4, 8));

// Test running compute and HCL jobs on the same Stream
TEST_P_SYN(SynGaudiMultiDevicesSimple, compute_and_hcl)
{
    int numberOfDevices = GetParam();
    startProcesses(numberOfDevices, &SynGaudiMultiDevicesSimple::runComputeAndHCL);
}

// Test HCL jobs on Stream with bigger buffers
// and device to device copy between HCL actions
TEST_P_SYN(SynGaudiMultiDevicesSimple, all_reduce_and_all_gather)
{
    int numberOfDevices = GetParam();
    startProcesses(numberOfDevices, &SynGaudiMultiDevicesSimple::runAllReduceAndAllGather);
}

TEST_P_SYN(SynGaudiMultiDevicesSimple, collective_graph)
{
    int numberOfDevices = GetParam();
    startProcesses(numberOfDevices, &SynGaudiMultiDevicesSimple::runCollective);
}

// Test running compute and HCL jobs on the same Stream
TEST_P_SYN(SynGaudiMultiDevicesSimple, device_count_test)
{
    int numberOfDevices = GetParam();
    startProcesses(numberOfDevices, &SynGaudiMultiDevicesSimple::getDeviceCountTest);
}
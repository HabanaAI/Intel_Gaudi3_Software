#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "log_manager.h"
#include "synapse_common_types.h"
#include "test_recipe_resnet.hpp"
#include "tensor_validator.inl"
#include <future>
#include "math_utils.h"

static const size_t CL_SIZE = 128;
static size_t       alignSizeToCL(size_t size)
{
    return CL_SIZE * div_round_up(size, CL_SIZE);
}

class ResnetSfgTest : public SynBaseTest
{
public:
    ResnetSfgTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2}); }
    ~ResnetSfgTest() override = default;

    void SetUp() override;

    void execute(TestDevice&         rDevice,
                 TestStream&         stream,
                 TestRecipeResnet*   pRecipe,
                 RecipeLaunchParams& rRecipeLaunchParams,
                 synEventHandle*     eventHandles             = nullptr,
                 size_t              numOfEvents              = 0,
                 size_t              externalTensorsIndices[] = nullptr);

    void setInputTensorDeviceBuffer(const TestDevice&   rDevice,
                                    const TensorInfo&   rTensorInfo,
                                    const TensorMemory& rTensorMemory,
                                    const TestStream&   rStream);

    void setOutputTensorDeviceBuffer(const TestDevice&   rDevice,
                                     const TensorInfo&   rTensorInfo,
                                     const TensorMemory& rTensorMemory,
                                     const TestStream&   rStream);

    void validateResults(const TestDevice* pDevice,
                         const TestStream* pStream,
                         TestRecipeResnet* pRecipe,
                         const TensorInfo* pTensorInfo,
                         TensorMemory*     pTensorMemory,
                         synEventHandle*   eventHandle);

protected:
    std::string m_pathPrefix = "";
};

REGISTER_SUITE(ResnetSfgTest, ALL_TEST_PACKAGES);

void ResnetSfgTest::SetUp()
{
    const char* envSoftwareLfsData = std::getenv("SOFTWARE_LFS_DATA");
    ASSERT_TRUE(envSoftwareLfsData) << "SOFTWARE_LFS_DATA is not set!";
    std::string softwareLfsData = envSoftwareLfsData;
    m_pathPrefix                = softwareLfsData.append("/demos/gaudi/functional/");

    SynBaseTest::SetUp();
}

void ResnetSfgTest::setInputTensorDeviceBuffer(const TestDevice&   rDevice,
                                               const TensorInfo&   rTensorInfo,
                                               const TensorMemory& rTensorMemory,
                                               const TestStream&   rStream)
{
    const size_t bufferSize = alignSizeToCL(rTensorInfo.m_tensorSize);
    ASSERT_EQ(bufferSize, rTensorInfo.m_tensorSize) << "Invalid tensor size " << rTensorInfo.m_tensorSize;

    TestHostBufferMalloc buffer = rDevice.allocateHostBuffer(bufferSize, 0);

    // When running on PLDM, reading uninitialized memory may cause HW interrupts since the mem is not scrubbed.
    // Allocating each tensor in CL alignment and initializing all the buffer to 0 should prevent this.
    buffer.fill(0);

    const std::string bufferFileName = m_pathPrefix + rTensorInfo.m_tensorName;
    const bool        file_res       = buffer.read_file(bufferFileName);
    ASSERT_TRUE(file_res);

    rStream.memcopyAsync((uint64_t)buffer.getBuffer(),
                         bufferSize,
                         rTensorMemory.getTestDeviceBuffer().getBuffer(),
                         HOST_TO_DRAM);

    rStream.synchronize();
}

void ResnetSfgTest::setOutputTensorDeviceBuffer(const TestDevice&   rDevice,
                                                const TensorInfo&   rTensorInfo,
                                                const TensorMemory& rTensorMemory,
                                                const TestStream&   rStream)
{
    TestHostBufferMalloc buffer = rDevice.allocateHostBuffer(rTensorInfo.m_tensorSize, 0);

    buffer.fill(0);

    rStream.memcopyAsync((uint64_t)buffer.getBuffer(),
                         rTensorInfo.m_tensorSize,
                         rTensorMemory.getTestDeviceBuffer().getBuffer(),
                         HOST_TO_DRAM);

    rStream.synchronize();
}

void ResnetSfgTest::validateResults(const TestDevice* pDevice,
                                    const TestStream* pStream,
                                    TestRecipeResnet* pRecipe,
                                    const TensorInfo* pTensorInfo,
                                    TensorMemory*     pTensorMemory,
                                    synEventHandle*   eventHandle)
{
    const TensorInfo& rTensorInfo   = *pTensorInfo;
    TensorMemory&     rTensorMemory = *pTensorMemory;

    if (eventHandle != nullptr)
    {
        ASSERT_EQ(synSuccess, synStreamWaitEvent(*pStream, *eventHandle, 0));
    }

    TestHostBufferMalloc hostBuffer  = pDevice->allocateHostBuffer(rTensorInfo.m_tensorSize, 0);
    void*                pHostBuffer = hostBuffer.getBuffer();
    if (pHostBuffer == nullptr)
    {
        LOG_WARN(SYN_RT_TEST, "Result compare skipped due to missing host buffer: {}", rTensorInfo.m_tensorName);
        return;
    }

    pStream->memcopyAsync(rTensorMemory.getTestDeviceBuffer().getBuffer(),
                          rTensorInfo.m_tensorSize,
                          (uint64_t)pHostBuffer,
                          DRAM_TO_HOST);

    LOG_TRACE(SYN_RT_TEST, "{}: calling streamSynchronize for tensor {}", __FUNCTION__, rTensorInfo.m_tensorName);
    pStream->synchronize();

    TestHostBufferMalloc hostBufferRef = pDevice->allocateHostBuffer(rTensorInfo.m_tensorSize, 0);
    hostBufferRef.read_file(m_pathPrefix + rTensorInfo.m_tensorName);
    void* pHostBufferRef = hostBufferRef.getBuffer();
    if (pHostBufferRef == nullptr)
    {
        LOG_WARN(SYN_RT_TEST, "Result compare skipped due to missing host buffer ref: {}", rTensorInfo.m_tensorName);
        return;
    }

    ::validateResult(static_cast<const float*>(pHostBufferRef),
                     static_cast<const float*>(pHostBuffer),
                     rTensorInfo.m_tensorSize / sizeof(float),
                     rTensorInfo.m_tensorName);
}

void ResnetSfgTest::execute(TestDevice&         rDevice,
                            TestStream&         stream,
                            TestRecipeResnet*   pRecipe,
                            RecipeLaunchParams& rRecipeLaunchParams,
                            synEventHandle*     eventHandles,
                            size_t              numOfEvents,
                            size_t              externalTensorsIndices[])
{
    for (int i = 0; i < pRecipe->getTensorInfoVecSizeInput(); i++)
    {
        setInputTensorDeviceBuffer(rDevice,
                                   *pRecipe->getTensorInfoInput(i),
                                   rRecipeLaunchParams.getLaunchTensorMemory().m_tensorInfoVecInputs[i],
                                   stream);
    }
    if (eventHandles != nullptr)
    {
        for (int i = 0; i < pRecipe->getTensorInfoVecSizeOutput(); i++)
        {
            setOutputTensorDeviceBuffer(rDevice,
                                        *pRecipe->getTensorInfoOutput(i),
                                        rRecipeLaunchParams.getLaunchTensorMemory().m_tensorInfoVecOutputs[i],
                                        stream);
        }
    }
    synStatus status = synLaunchWithExternalEventsExt(stream,
                                                      rRecipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                                                      rRecipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                                                      rRecipeLaunchParams.getWorkspace(),
                                                      pRecipe->getRecipe(),
                                                      eventHandles,
                                                      numOfEvents,
                                                      0);
    ASSERT_EQ(synSuccess, status);

    if (eventHandles != nullptr)
    {
        std::future<void> futureArr[numOfEvents];
        for (size_t eventIdx = 0; eventIdx < numOfEvents; eventIdx++)
        {
            std::string tName = rRecipeLaunchParams.getSynLaunchTensorInfoVec()
                                    .data()[pRecipe->getExternalTensorIds()[eventIdx]]
                                    .tensorName;

            futureArr[eventIdx] =
                std::async(std::launch::async,
                           &ResnetSfgTest::validateResults,
                           this,
                           &rDevice,
                           &stream,
                           pRecipe,
                           pRecipe->getTensorInfo(tName),
                           &const_cast<LaunchTensorMemory*>(&rRecipeLaunchParams.getLaunchTensorMemory())
                                ->m_tensorInfoVecOutputs[eventIdx],
                           &eventHandles[eventIdx]);
        }
        for (size_t eventIdx = 0; eventIdx < numOfEvents; eventIdx++)
        {
            futureArr[eventIdx].wait();
        }
    }

    stream.synchronize();

    for (int i = 0; i < pRecipe->getTensorInfoVecSizeOutput(); i++)
    {
        validateResults(
            &rDevice,
            &stream,
            pRecipe,
            pRecipe->getTensorInfoOutput(i),
            &const_cast<LaunchTensorMemory*>(&rRecipeLaunchParams.getLaunchTensorMemory())->m_tensorInfoVecOutputs[i],
            nullptr);
    }
}

TEST_F_SYN(ResnetSfgTest, resnet_sfg_test, {synTestPackage::ASIC})
{
    size_t    numOfExternalTensors(3);
    size_t    numOfIterations(2);
    synStatus status;

    TestRecipeResnet recipe(m_deviceType, true);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestLauncher launcher(device);
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    ASSERT_EQ(recipe.generateSfg(), synSuccess) << "Failed to generate SFG details for recipe";

    // Create streams
    TestStream stream = device.createStream();

    std::vector<TestEvent>      testEvents;
    std::vector<synEventHandle> events;
    for (int i = 0; i < numOfExternalTensors; i++)
    {
        testEvents.push_back(device.createEvent(0));
        events.push_back(synEventHandle(testEvents[i]));
    }

    for (size_t eventIdx = 0; eventIdx < numOfExternalTensors; eventIdx++)
    {
        const SynLaunchTensorInfoVec& currLaunchTensors = recipeLaunchParams.getSynLaunchTensorInfoVec();

        size_t tensorExternalId = recipe.getExternalTensorIds()[eventIdx];

        const synLaunchTensorInfoExt* pLaunchTensorsInfo = &currLaunchTensors[tensorExternalId];

        status = synEventMapTensorExt(&events[eventIdx], 1, pLaunchTensorsInfo, recipe.getRecipe());
        ASSERT_EQ(status, synSuccess) << "Failed to map event tensor";
    }

    execute(device, stream, &recipe, recipeLaunchParams, nullptr, 0, nullptr);
    execute(device,
            stream,
            &recipe,
            recipeLaunchParams,
            events.data(),
            numOfExternalTensors,
            recipe.getExternalTensorIds().data());
    execute(device, stream, &recipe, recipeLaunchParams, nullptr, 0, nullptr);
    for (size_t i = 0; i < numOfIterations; i++)
    {
        execute(device,
                stream,
                &recipe,
                recipeLaunchParams,
                events.data(),
                numOfExternalTensors,
                recipe.getExternalTensorIds().data());
    }
}

TEST_F_SYN(ResnetSfgTest, resnet_sfg_test_reuse_events)
{
    const size_t numOfExternalTensors(3), numOfIterations(2);

    TestRecipeResnet recipe(m_deviceType, true);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestLauncher launcher(device);
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    ASSERT_EQ(recipe.generateSfg(), synSuccess) << "Failed to generate SFG details for recipe";

    // Create streams
    synStatus  status;
    TestStream stream = device.createStream();

    std::vector<TestEvent>      testEvents;
    std::vector<synEventHandle> events;
    for (int i = 0; i < numOfExternalTensors; i++)
    {
        testEvents.push_back(device.createEvent(0));
        events.push_back(synEventHandle(testEvents[i]));
    }

    for (size_t eventIdx = 0; eventIdx < numOfExternalTensors; eventIdx++)
    {
        const SynLaunchTensorInfoVec& currLaunchTensors = recipeLaunchParams.getSynLaunchTensorInfoVec();

        size_t tensorExternalId = recipe.getExternalTensorIds()[eventIdx];

        const synLaunchTensorInfoExt* pLaunchTensorsInfo = &currLaunchTensors[tensorExternalId];

        status = synEventMapTensorExt(&events[eventIdx], 1, pLaunchTensorsInfo, recipe.getRecipe());
        ASSERT_EQ(status, synSuccess) << "Failed to map event tensor";
    }

    execute(device, stream, &recipe, recipeLaunchParams, nullptr, 0, nullptr);
    execute(device,
            stream,
            &recipe,
            recipeLaunchParams,
            events.data(),
            numOfExternalTensors,
            recipe.getExternalTensorIds().data());

    status = synEventRecord(events[numOfExternalTensors - 1], stream);
    ASSERT_EQ(status, synSuccess) << "Failed to synEventRecord";

    execute(device, stream, &recipe, recipeLaunchParams, nullptr, 0, nullptr);
    for (size_t i = 0; i < numOfIterations; i++)
    {
        execute(device,
                stream,
                &recipe,
                recipeLaunchParams,
                events.data(),
                numOfExternalTensors - 1,
                recipe.getExternalTensorIds().data());
    }

    status = synStreamWaitEvent(stream, events[numOfExternalTensors - 1], 0);
    ASSERT_EQ(status, synSuccess) << "Failed to synStreamWaitEvent";
}

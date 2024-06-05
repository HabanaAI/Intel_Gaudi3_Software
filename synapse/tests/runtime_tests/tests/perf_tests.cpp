#include "syn_base_test.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_processor.hpp"
#include "timer.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "infra/type_utils.h"
#include "graph_entries_container.hpp"
#include "synapse_common_types.h"
#include "test_device.hpp"
#include "synapse_api.h"

class SynScalPerfTests : public SynBaseTest
{
public:
    SynScalPerfTests();
    virtual ~SynScalPerfTests() = default;

    void basicDma(int numStreams, bool waitForDownload);

private:
    synDeviceType m_deviceType;
};

SynScalPerfTests::SynScalPerfTests() : m_deviceType(synDeviceTypeInvalid)
{
    if (getenv("SYN_DEVICE_TYPE") != nullptr)
    {
        m_deviceType = (synDeviceType)std::stoi(getenv("SYN_DEVICE_TYPE"));
    }
    setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});
}

// Copy a 4 buffers host->dev->host using different number of streams
void SynScalPerfTests::basicDma(int numStreams, bool waitForDownload)
{
    const uint64_t SIZE = 1024ULL * 1024ULL * 1024ULL / 2 / numStreams;  // This size fits into huge pages

    assert(SIZE % sizeof(uint64_t) == 0);

    TestDevice device(m_deviceType);

    // create streams (up/down)
    std::vector<TestStream>            streamDown;
    std::vector<TestStream>            streamUp;
    std::vector<TestHostBufferMalloc>  inBuff;
    std::vector<TestHostBufferMalloc>  outBuff;
    std::vector<TestDeviceBufferAlloc> devBuff;

    // create streams
    for (int i = 0; i < numStreams; i++)
    {
        streamDown.emplace_back(device.createStream());
        streamUp.emplace_back(device.createStream());
    }

    // create buffers
    for (int i = 0; i < numStreams; i++)
    {
        inBuff.emplace_back(device.allocateHostBuffer(SIZE, 0));
        outBuff.emplace_back(device.allocateHostBuffer(SIZE, 0));
        devBuff.emplace_back(device.allocateDeviceBuffer(SIZE, 0));

        // set values in the buffer
        for (int j = 0; j < SIZE / sizeof(uint64_t); j++)
        {
            ((uint64_t*)inBuff[i].getBuffer())[j]  = j + 1;
            ((uint64_t*)outBuff[i].getBuffer())[j] = 0;
        }
    }

    // Copy the buffer using N streams
    for (int i = 0; i < numStreams; i++)
    {
        streamDown[i].memcopyAsync(inBuff[i], devBuff[i]);
    }

    if (waitForDownload)
    {
        for (int i = 0; i < numStreams; i++)
        {
            auto start = TimeTools::timeNow();
            streamDown[i].synchronize();
            auto timeNs = TimeTools::timeFromNs(start);
            printf("stream %d download took %ld (%ld Gb/s)\n", i, timeNs, SIZE / timeNs);
        }
        usleep(100000);
    }

    // Do the upload
    for (int i = 0; i < numStreams; i++)
    {
        TestEvent eventDown = device.createEvent();

        streamDown[i].eventRecord(eventDown);

        streamUp[i].eventWait(eventDown);
        streamUp[i].memcopyAsync(devBuff[i], outBuff[i]);
    }

    // wait for upload
    for (int i = 0; i < numStreams; i++)
    {
        auto start = TimeTools::timeNow();
        streamUp[i].synchronize();
        auto timeNs = TimeTools::timeFromNs(start);
        if (waitForDownload)
        {
            printf("up");
        }
        else
        {
            printf("up+down:");
        }
        printf("stream %d took %ld (%ld Gb/s)\n", i, timeNs, SIZE / timeNs);
    }

    // check data
    for (int i = 0; i < numStreams; i++)
    {
        for (int item = 0; item < inBuff[i].getSize() / sizeof(uint64_t); item++)
        {
            if (((uint64_t*)inBuff[i].getBuffer())[item] != ((uint64_t*)outBuff[i].getBuffer())[item])
            {
                ASSERT_EQ(((uint64_t*)inBuff[i].getBuffer())[item], ((uint64_t*)outBuff[i].getBuffer())[item])
                    << "Failed on stream " << i << " item " << item;
            }
        }
    }
}

REGISTER_SUITE(SynScalPerfTests, ALL_TEST_PACKAGES);

class SynScalPerfTestsM
: public SynScalPerfTests
, public testing::WithParamInterface<unsigned>
{
};

INSTANTIATE_TEST_SUITE_P(, SynScalPerfTestsM, ::testing::Values(1, 2, 4));

TEST_P(SynScalPerfTestsM, DISABLED_basicDma)  // disable: no need to run on CI every time
{
    basicDma(GetParam(), true);
}

TEST_F_SYN(SynScalPerfTests, DISABLED_dmaUpAndDown)  // disable: no need to run on CI every time
{
    basicDma(4, false);
}

REGISTER_SUITE(SynScalPerfTestsM, ALL_TEST_PACKAGES);

/******************************************************************************************/

class SynScalPrgDownloadPerfTests : public SynBaseTest
{
public:
    SynScalPrgDownloadPerfTests();
    virtual ~SynScalPrgDownloadPerfTests() = default;

    void runTest();

private:
    void buildBasicDmaRecipe(synDeviceType devType);
    void fakeBigRecipe();
    void recoverRecipe();
    void createHostBuffers();
    void createDevBuffers();
    void allocateWorkspace();
    void createStreams();
    void setupLaunchTensors();
    void releaseDestroy();

    static const uint32_t NUM_OF_TENSOR = 2;
    static const uint64_t DMA_SIZE =
        1024ULL * 1024ULL * 1024ULL / 2;  // limit to 0.5 because the number of huge pages (1000)
    static const int      LOOPS          = 2;
    static const uint64_t EXEC_FAKE_SIZE = 0x3000000;

    const char* m_tensorNames[NUM_OF_TENSOR] = {"input", "output"};

    std::vector<uint32_t> m_tensorDims {1024U, 1, 1U, 1U};

    synDeviceId     m_devId;
    synRecipeHandle m_recipeHandle;
    synRecipeHandle m_recipeHandleBig;

    uint64_t                   m_orgSize;
    uint64_t*                  m_orgBuff;
    std::unique_ptr<uint8_t[]> m_bigBuff;

    uint64_t m_tensorSize;

    uint8_t* m_recipeIn;
    uint8_t* m_recipeOut;
    uint8_t* m_dmaIn;
    uint8_t* m_dmaOut;

    uint64_t m_devRecipeIn;
    uint64_t m_devRecipeOut;
    uint64_t m_devDmaIn;
    uint64_t m_devDmaOut;

    uint64_t m_workspace = 0;

    synStreamHandle m_copyInStream;
    synStreamHandle m_copyOutStream;
    synStreamHandle m_computeStream;
    synStreamHandle m_copyDev2Dev;

    synLaunchTensorInfo m_persistentTensorInfo[NUM_OF_TENSOR];

    synDeviceType m_deviceType;
};

REGISTER_SUITE(SynScalPrgDownloadPerfTests, ALL_TEST_PACKAGES);

SynScalPrgDownloadPerfTests::SynScalPrgDownloadPerfTests() : m_deviceType(synDeviceTypeInvalid)
{
    if (getenv("SYN_DEVICE_TYPE") != nullptr)
    {
        m_deviceType = (synDeviceType)std::stoi(getenv("SYN_DEVICE_TYPE"));
    }

    setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});
}

void SynScalPrgDownloadPerfTests::releaseDestroy()
{
    synRecipeDestroy(m_recipeHandle);
    synRecipeDestroy(m_recipeHandleBig);

    synHostFree(m_devId, (void*)m_recipeIn, 0);
    synHostFree(m_devId, (void*)m_recipeOut, 0);
    synHostFree(m_devId, (void*)m_dmaIn, 0);
    synHostFree(m_devId, (void*)m_dmaOut, 0);

    synStreamDestroy(m_copyInStream);
    synStreamDestroy(m_copyOutStream);
    synStreamDestroy(m_computeStream);
    synStreamDestroy(m_copyDev2Dev);

    synDeviceFree(m_devId, m_devRecipeIn, 0);
    synDeviceFree(m_devId, m_devRecipeOut, 0);
    synDeviceFree(m_devId, m_devDmaIn, 0);
    synDeviceFree(m_devId, m_devDmaOut, 0);

    if (m_workspace != 0)
    {
        synDeviceFree(m_devId, m_workspace, 0);
    }

    synStatus status = synDeviceRelease(m_devId);
    ASSERT_EQ(status, synSuccess) << "Failed to release device";

    status = synDestroy();
    ASSERT_EQ(status, synSuccess) << "Failed to destroy synapse";
}

void SynScalPrgDownloadPerfTests::setupLaunchTensors()
{
    uint64_t tensorIds[NUM_OF_TENSOR];
    ASSERT_EQ(synTensorRetrieveIds(m_recipeHandleBig, m_tensorNames, tensorIds, NUM_OF_TENSOR), synSuccess);

    m_persistentTensorInfo[0].tensorName     = "input";  // Must match the name supplied at tensor creation
    m_persistentTensorInfo[0].pTensorAddress = m_devRecipeIn;
    m_persistentTensorInfo[0].tensorType     = DATA_TENSOR;
    memset(&m_persistentTensorInfo[0].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
    m_persistentTensorInfo[0].tensorId       = tensorIds[0];
    m_persistentTensorInfo[1].tensorName     = "output";  // Must match the name supplied at tensor creation
    m_persistentTensorInfo[1].pTensorAddress = m_devRecipeOut;
    m_persistentTensorInfo[1].tensorType     = DATA_TENSOR;
    memset(&m_persistentTensorInfo[1].tensorSize[0], 0, HABANA_DIM_MAX * sizeof(TSize));
    m_persistentTensorInfo[1].tensorId = tensorIds[1];
}

void SynScalPrgDownloadPerfTests::createStreams()
{
    synStatus status = synStreamCreateGeneric(&m_copyInStream, m_devId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&m_computeStream, m_devId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create compute stream";
    status = synStreamCreateGeneric(&m_copyOutStream, m_devId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";
    status = synStreamCreateGeneric(&m_copyDev2Dev, m_devId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy device to device";
}

void SynScalPrgDownloadPerfTests::allocateWorkspace()
{
    uint64_t  workspaceSize;
    synStatus status = synWorkspaceGetSize(&workspaceSize, m_recipeHandleBig);
    ASSERT_EQ(status, synSuccess) << "Failed to query required workspace size";

    if (workspaceSize > 0)
    {
        status = synDeviceMalloc(m_devId, workspaceSize, 0, 0, &m_workspace);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";
    }
}

void SynScalPrgDownloadPerfTests::createDevBuffers()
{
    synStatus status = synDeviceMalloc(m_devId, m_tensorSize, 0, 0, &m_devRecipeIn);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
    status = synDeviceMalloc(m_devId, m_tensorSize, 0, 0, &m_devRecipeOut);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";

    status = synDeviceMalloc(m_devId, DMA_SIZE, 0, 0, &m_devDmaIn);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";
    status = synDeviceMalloc(m_devId, DMA_SIZE, 0, 0, &m_devDmaOut);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate output buffer in device memory";
}

void SynScalPrgDownloadPerfTests::createHostBuffers()
{
    synStatus status = synHostMalloc(m_devId, m_tensorSize, 0, (void**)&m_recipeIn);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(m_devId, m_tensorSize, 0, (void**)&m_recipeOut);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    memset(m_recipeIn, 0xEE, m_tensorSize);
    memset(m_recipeOut, 0x00, m_tensorSize);

    status = synHostMalloc(m_devId, DMA_SIZE, 0, (void**)&m_dmaIn);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(m_devId, DMA_SIZE, 0, (void**)&m_dmaOut);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";

    memset(m_dmaIn, 0xEE, DMA_SIZE);
    memset(m_dmaOut, 0x00, DMA_SIZE);
}

void SynScalPrgDownloadPerfTests::fakeBigRecipe()
{
    recipe_t* recipe  = m_recipeHandle->basicRecipeHandle.recipe;
    uint64_t  orgSize = recipe->execution_blobs_buffer_size;
    uint64_t* orgBuff = recipe->execution_blobs_buffer;

    m_bigBuff                           = std::unique_ptr<uint8_t[]>(new uint8_t[EXEC_FAKE_SIZE]);
    recipe->execution_blobs_buffer_size = EXEC_FAKE_SIZE;
    recipe->execution_blobs_buffer      = reinterpret_cast<uint64_t*>(m_bigBuff.get());

    memcpy(m_bigBuff.get(), orgBuff, orgSize);

    synRecipeSerialize(m_recipeHandle, "temp");
    synRecipeDeSerialize(&m_recipeHandleBig, "temp");
}

void SynScalPrgDownloadPerfTests::recoverRecipe()
{
    recipe_t* recipe = m_recipeHandle->basicRecipeHandle.recipe;

    recipe->execution_blobs_buffer_size = m_orgSize;
    recipe->execution_blobs_buffer      = m_orgBuff;
}

void SynScalPrgDownloadPerfTests::buildBasicDmaRecipe(synDeviceType devType)
{
    synStatus status;

    synGraphHandle graph;
    status = synGraphCreate(&graph, devType);
    ASSERT_EQ(status, synSuccess) << "Failed to create graph";

    // Create input and output sections for managing device memory
    synSectionHandle inSection, outSection;

    status = synSectionCreate(&outSection, 0, graph);
    ASSERT_EQ(status, synSuccess) << "Failed to create output section";
    status = synSectionCreate(&inSection, 0, graph);
    ASSERT_EQ(status, synSuccess) << "Failed to create input section";

    // Tensor sizes
    const unsigned dims                               = m_tensorDims.size();
    const unsigned Z                                  = m_tensorDims[0];
    const unsigned W                                  = m_tensorDims[1];
    const unsigned H                                  = m_tensorDims[2];
    const unsigned batch                              = m_tensorDims[3];
    unsigned       tensorDimSizes[SYN_MAX_TENSOR_DIM] = {Z, W, H, batch};

    // Tensors
    synTensor           in_tensor, out_tensor;
    synTensorDescriptor desc;
    const uint32_t      numOfTensor              = 2;
    const char*         tensorNames[numOfTensor] = {"input", "output"};

    desc.m_dataType = syn_type_uint32;
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
    status = synGraphCompile(&m_recipeHandle, graph, "my_graph", nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";

    synTensorDestroy(in_tensor);
    synTensorDestroy(out_tensor);

    synSectionDestroy(inSection);
    synSectionDestroy(outSection);

    status = synGraphDestroy(graph);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy graph";
}

void SynScalPrgDownloadPerfTests::runTest()
{
    synStatus status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "Failed to init synapse";

    buildBasicDmaRecipe(m_deviceType);

    fakeBigRecipe();  // fake size as 0x400000

    m_tensorSize = getActualTensorSize<uint32_t>(m_tensorDims.size(), m_tensorDims.data(), syn_type_int32);
    // Execution
    status = synDeviceAcquireByDeviceType(&m_devId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "No available devices! Did you forget to run the simulator?";

    createHostBuffers();

    createStreams();

    createDevBuffers();

    allocateWorkspace();

    setupLaunchTensors();

    printf("---Copy in 0x%lX bytes---\n", DMA_SIZE);
    for (int i = 0; i < LOOPS; i++)
    {
        auto startTime = TimeTools::timeNow();
        status         = synMemCopyAsync(m_copyInStream, (uint64_t)m_dmaIn, DMA_SIZE, m_devDmaIn, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy in";
        ASSERT_EQ(synStreamSynchronize(m_copyInStream), synSuccess) << "Fail sync on copy in";
        auto timeNs = TimeTools::timeFromNs(startTime);
        printf("loop %d took %ld (%ld Gb/s)\n", i, timeNs, DMA_SIZE / timeNs);
        usleep(100000);
    }

    printf("---Copy out 0x%lX bytes---\n", DMA_SIZE);
    for (int i = 0; i < LOOPS; i++)
    {
        auto startTime = TimeTools::timeNow();
        status         = synMemCopyAsync(m_copyOutStream, m_devDmaIn, DMA_SIZE, (uint64_t)m_dmaOut, DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed to copy out";
        ASSERT_EQ(synStreamSynchronize(m_copyOutStream), synSuccess) << "Fail sync on copy out";
        auto timeNs = TimeTools::timeFromNs(startTime);
        printf("copy %d took %ld (%ld Gb/s)\n", i, timeNs, DMA_SIZE / timeNs);
        usleep(100000);
    }

    printf("---Copy dev2dev 0x%lX bytes---\n", DMA_SIZE);
    for (int i = 0; i < LOOPS; i++)
    {
        auto startTime = TimeTools::timeNow();
        status         = synMemCopyAsync(m_copyDev2Dev, m_devDmaIn, DMA_SIZE, m_devDmaOut, DRAM_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy dev";
        ASSERT_EQ(synStreamSynchronize(m_copyDev2Dev), synSuccess) << "Fail sync on copy out";
        auto timeNs = TimeTools::timeFromNs(startTime);
        printf("copy %d took %ld (%ld Gb/s)\n", i, timeNs, DMA_SIZE / timeNs);
        usleep(100000);
    }

    // Copy tensor from host to device
    status = synMemCopyAsync(m_copyInStream, (uint64_t)m_recipeIn, m_tensorSize, m_devRecipeIn, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    ASSERT_EQ(synStreamSynchronize(m_copyInStream), synSuccess) << "Fail sync on recipe tensor in";

    printf("---Launch (big recipe (exec blobs 0x%lX), very little work) dma of only 0x%lX bytes---\n"
           "***ignore first one, it includes memcpy, a lot of time is building the dma requests\n",
           EXEC_FAKE_SIZE,
           m_tensorSize);
    for (int i = 0; i < LOOPS + 1; i++)
    {
        auto startTime = TimeTools::timeNow();
        status = synLaunch(m_computeStream, m_persistentTensorInfo, NUM_OF_TENSOR, m_workspace, m_recipeHandleBig, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";
        ASSERT_EQ(synStreamSynchronize(m_computeStream), synSuccess) << "Fail sync on copy out";
        auto timeNs = TimeTools::timeFromNs(startTime);
        if (i == 0) printf("Ignore: ");
        printf("launch %d (download to device + work) took %ld\n", i, timeNs);
        usleep(100000);
    }

    printf("---dma in and launch at the same time, expect launch to take the same time as before---\n");
    for (int i = 0; i < LOOPS; i++)
    {
        auto startTimeDma = TimeTools::timeNow();
        status            = synMemCopyAsync(m_copyInStream, (uint64_t)m_dmaIn, DMA_SIZE, m_devDmaIn, HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy in";

        auto startTimeLaunch = TimeTools::timeNow();
        status = synLaunch(m_computeStream, m_persistentTensorInfo, NUM_OF_TENSOR, m_workspace, m_recipeHandleBig, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";
        ASSERT_EQ(synStreamSynchronize(m_computeStream), synSuccess) << "Fail sync on copy out";
        auto timeNsLaunch = TimeTools::timeFromNs(startTimeLaunch);
        printf("launch %d (download to device + work) took %ld\n", i, timeNsLaunch);

        ASSERT_EQ(synStreamSynchronize(m_copyInStream), synSuccess) << "Fail sync on copy in";
        auto timeNsDma = TimeTools::timeFromNs(startTimeDma);
        printf("loop %d took %ld (%ld Gb/s)\n", i, timeNsDma, DMA_SIZE / timeNsDma);
        usleep(100000);
    }

    printf("---dma out and launch at the same time, expect launch to take the same time as before---\n");
    for (int i = 0; i < LOOPS; i++)
    {
        auto startTimeDma = TimeTools::timeNow();
        status            = synMemCopyAsync(m_copyOutStream, m_devDmaIn, DMA_SIZE, (uint64_t)m_dmaOut, DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed to copy out";

        auto startTimeLaunch = TimeTools::timeNow();
        status = synLaunch(m_computeStream, m_persistentTensorInfo, NUM_OF_TENSOR, m_workspace, m_recipeHandleBig, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";
        ASSERT_EQ(synStreamSynchronize(m_computeStream), synSuccess) << "Fail sync on copy out";
        auto timeNsLaunch = TimeTools::timeFromNs(startTimeLaunch);
        printf("launch %d (download to device + work) took %ld\n", i, timeNsLaunch);

        ASSERT_EQ(synStreamSynchronize(m_copyOutStream), synSuccess) << "Fail sync on copy out";
        auto timeNsDma = TimeTools::timeFromNs(startTimeDma);
        printf("copy %d took %ld (%ld Gb/s)\n", i, timeNsDma, DMA_SIZE / timeNsDma);
        usleep(100000);
    }

    printf("---dma dev2dev and launch at the same time, expect launch to take the same time as before---\n");
    for (int i = 0; i < LOOPS; i++)
    {
        auto startTimeDma = TimeTools::timeNow();
        status            = synMemCopyAsync(m_copyDev2Dev, m_devDmaIn, DMA_SIZE, m_devDmaOut, DRAM_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy out";

        auto startTimeLaunch = TimeTools::timeNow();
        status = synLaunch(m_computeStream, m_persistentTensorInfo, NUM_OF_TENSOR, m_workspace, m_recipeHandleBig, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to launch graph";
        ASSERT_EQ(synStreamSynchronize(m_computeStream), synSuccess) << "Fail sync on copy out";
        auto timeNsLaunch = TimeTools::timeFromNs(startTimeLaunch);
        printf("launch %d (download to device + work) took %ld\n", i, timeNsLaunch);

        ASSERT_EQ(synStreamSynchronize(m_copyDev2Dev), synSuccess) << "Fail sync on copy out";
        auto timeNsDma = TimeTools::timeFromNs(startTimeDma);
        printf("copy %d took %ld (%ld Gb/s)\n", i, timeNsDma, DMA_SIZE / timeNsDma);
        usleep(100000);
    }

    recoverRecipe();

    releaseDestroy();
}

TEST_F_SYN(SynScalPrgDownloadPerfTests, DISABLED_fakeBigRecipe)  // disable: no need to run on CI every time
{
    runTest();
}

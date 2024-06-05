#include "infra/gc_synapse_test.h"
#include "scoped_configuration_change.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>
#include <utility>

namespace
{
class SynGaudiEagerAPITests : public SynTest
{
public:
    SynGaudiEagerAPITests()
    {
        m_testConfig.m_numOfTestDevices = 1;
        m_testConfig.m_compilationMode  = COMP_BOTH_MODE_TESTS;
        m_testConfig.m_supportedDeviceTypes.clear();
        setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});
        setTestPackage(TEST_PACKAGE_EAGER);
    }

    void runMultiThreadedDuplicateAddTest(bool isEager, bool shrinkTensors, bool expandTensors);
};
}  // namespace

static void createTensorInfo(synRecipeHandle recipe, synLaunchTensorInfo* tensorInfo, uint32_t totalNumOfTensors)
{
    std::vector<const char*> tensorNames;
    tensorNames.reserve(totalNumOfTensors);
    for (unsigned i = 0; i < totalNumOfTensors; ++i)
    {
        tensorNames.push_back(tensorInfo[i].tensorName);
    }
    std::vector<uint64_t> tensorIds(totalNumOfTensors);
    ASSERT_EQ(synTensorRetrieveIds(recipe, tensorNames.data(), tensorIds.data(), totalNumOfTensors), synSuccess);
    for (unsigned i = 0; i < totalNumOfTensors; i++)
    {
        tensorInfo[i].tensorId = tensorIds[i];
    }
}

void SynGaudiEagerAPITests::runMultiThreadedDuplicateAddTest(bool isEager, bool shrinkTensors, bool expandTensors)
{
    static constexpr size_t               NUM_THREADS     = 1;
    static constexpr size_t               TENSOR_NUM_DIMS = 2;
    static constexpr size_t               ORIG_DIM_SIZE   = 16;
    std::array<unsigned, TENSOR_NUM_DIMS> origDimSizes    = {ORIG_DIM_SIZE, ORIG_DIM_SIZE};
    synGraphHandle                        graphA;

    synStatus status = isEager ? synGraphCreateEager(&graphA, m_deviceType) : synGraphCreate(&graphA, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

    synTensor in1 = createTrainingTensor(TENSOR_NUM_DIMS, syn_type_single, origDimSizes.data(), true, "in1", graphA);
    synTensor in2 = createTrainingTensor(TENSOR_NUM_DIMS, syn_type_single, origDimSizes.data(), true, "in2", graphA);
    synTensor out = createTrainingTensor(TENSOR_NUM_DIMS, syn_type_single, origDimSizes.data(), true, "out", graphA);
    static constexpr size_t                  NUM_INPUT_TENSORS  = 2;
    static constexpr size_t                  NUM_OUTPUT_TENSORS = 1;
    std::array<synTensor, NUM_INPUT_TENSORS> inTensors          = {in1, in2};
    status                                                      = synNodeCreate(graphA,
                           inTensors.data(),
                           &out,
                           NUM_INPUT_TENSORS,
                           NUM_OUTPUT_TENSORS,
                           nullptr,
                           0,
                           "add_fwd_f32",
                           "",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create add Node";

    synStreamHandle streamHandleDownload;
    synStreamHandle streamHandleCompute;
    synStreamHandle streamHandleUpload;
    uint32_t        deviceId = 0;

    status = synStreamCreateGeneric(&streamHandleDownload, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create Download (host->device copy) stream";

    status = synStreamCreateGeneric(&streamHandleCompute, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create Compute stream";

    status = synStreamCreateGeneric(&streamHandleUpload, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create Upload (device->host copy) stream";

    auto duplicateFunc = [&]() {
        std::array<unsigned, TENSOR_NUM_DIMS> newDimSizes;
        auto                                  newDimSize    = ORIG_DIM_SIZE;
        auto                                  dimSizeChange = std::rand() % 16;
        if (shrinkTensors) newDimSize -= dimSizeChange;
        else if (expandTensors)
            newDimSize += dimSizeChange;
        newDimSizes.fill(newDimSize);
        uint64_t tensorSizeInElements =
            std::accumulate(newDimSizes.begin(), newDimSizes.end(), 1u, [](const auto& a, const auto& b) {
                return a * b;
            });
        uint64_t                        tensorSizeInBytes = sizeof(float) * tensorSizeInElements;
        synStatus                       status            = synSuccess;
        uint32_t                        nodesMapSize      = 1;
        uint32_t                        tensorsMapSize    = 3;
        synGraphHandle                  graphB;
        std::vector<synTensorHandleMap> tensorsMap(tensorsMapSize);
        std::vector<synNodeHandleMap>   nodesMap(nodesMapSize);

        // test the tensor count and node count retrieval functionality
        using SynMappingPair = std::pair<synTensorHandleMap*, synNodeHandleMap*>;
        for (const auto& mappingPair : {SynMappingPair(nullptr, nodesMap.data()),
                                        SynMappingPair(tensorsMap.data(), nullptr),
                                        SynMappingPair(nullptr, nullptr)})
        {
            uint32_t numNodes   = 0;
            uint32_t numTensors = 0;
            status = synGraphDuplicate(graphA, &graphB, mappingPair.first, &numTensors, mappingPair.second, &numNodes);
            ASSERT_EQ(status, synSuccess) << "graph duplication tensors and nodes count retrieval failed";
            ASSERT_EQ(nodesMapSize, numNodes) << "unexpected nodes count in graph";
            ASSERT_EQ(tensorsMapSize, numTensors) << "unexpected tensors count in graph";
        }

        status = synGraphDuplicate(graphA, &graphB, tensorsMap.data(), &tensorsMapSize, nodesMap.data(), &nodesMapSize);
        ASSERT_EQ(status, synSuccess) << "graph duplication failed";

        // update the geometry
        synTensorGeometry geometry;
        geometry.dims = TENSOR_NUM_DIMS;
        for (size_t i = 0; i < geometry.dims; i++)
            geometry.sizes[i] = newDimSizes[i];
        for (const auto& [origTensor, dupTensor] : tensorsMap)
        {
            static_cast<void>(origTensor);  // W/A for unused variable
            synTensorSetGeometry(dupTensor, &geometry, synGeometryMaxSizes);
        }

        synRecipeHandle recipeHandleB;
        status = synGraphCompile(&recipeHandleB, graphB, GetTestFileName().c_str(), nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed on graphB compilation";

        // Cleanups

        status = synGraphDestroy(graphB);
        ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph B";

        // Running
        uint64_t inputDeviceBuffer = 0;
        status                     = synDeviceMalloc(deviceId, tensorSizeInBytes, 0, 0, &inputDeviceBuffer);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate HBM memory for tensor";

        uint64_t inputDeviceBuffer2 = 0;
        status                      = synDeviceMalloc(deviceId, tensorSizeInBytes, 0, 0, &inputDeviceBuffer2);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate HBM memory for tensor";

        uint64_t outputDeviceBuffer = 0;
        status                      = synDeviceMalloc(deviceId, tensorSizeInBytes, 0, 0, &outputDeviceBuffer);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate HBM memory for tensor";

        // Calculate workspace
        uint64_t workspaceSize = 0;
        status                 = synWorkspaceGetSize(&workspaceSize, recipeHandleB);
        ASSERT_EQ(status, synSuccess) << "Failed to WorkspaceGetSize";

        uint64_t workspaceAddress;
        status = synDeviceMalloc(deviceId, workspaceSize, 0, 0, &workspaceAddress);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate HBM memory for workspace";

        ////// create and memcpy down input tensors /////
        float *pDataVal, *pDataVal2;
        status = synHostMalloc(deviceId, tensorSizeInBytes, 0, (void**)(&pDataVal));
        ASSERT_EQ(status, synSuccess) << "Failed to create synHostMalloc";
        status = synHostMalloc(deviceId, tensorSizeInBytes, 0, (void**)(&pDataVal2));
        ASSERT_EQ(status, synSuccess) << "Failed to create synHostMalloc";

        float* pOutputVal;
        status = synHostMalloc(deviceId, tensorSizeInBytes, 0, (void**)(&pOutputVal));
        ASSERT_EQ(status, synSuccess) << "Failed to synHostMalloc";
        std::array<synLaunchTensorInfo, 3> tensors {
            {{"in1", inputDeviceBuffer}, {"in2", inputDeviceBuffer2}, {"out", outputDeviceBuffer}}};

        createTensorInfo(recipeHandleB, tensors.data(), tensors.size());

        for (size_t i = 0; i < tensorSizeInElements; i++)
        {
            pDataVal[i]  = i;
            pDataVal2[i] = 2 * i;
        }
        for (auto [pDataVal, deviceBuffer] :
             {std::pair(pDataVal, inputDeviceBuffer), std::pair(pDataVal2, inputDeviceBuffer2)})
        {
            status = synMemCopyAsync(streamHandleDownload,
                                     (uint64_t)pDataVal,
                                     tensorSizeInBytes,
                                     deviceBuffer,
                                     HOST_TO_DRAM);
            ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync HOST_TO_DRAM";
            status = synStreamSynchronize(streamHandleDownload);
            ASSERT_EQ(status, synSuccess) << "Failed to synStreamSynchronize";
        }

        status = synLaunch(streamHandleCompute, tensors.data(), tensors.size(), workspaceAddress, recipeHandleB, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
        status = synStreamSynchronize(streamHandleCompute);
        ASSERT_EQ(status, synSuccess) << "Failed to synchronize stream (compute)";

        // Results in Host for compare with Reference
        status = synMemCopyAsync(streamHandleUpload,
                                 outputDeviceBuffer,
                                 tensorSizeInBytes,
                                 (uint64_t)pOutputVal,
                                 DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed to synMemCopyAsync";
        status = synStreamSynchronize(streamHandleUpload);
        ASSERT_EQ(status, synSuccess) << "Failed to synchronize stream (DRAM_TO_HOST)";

        // Validating
        for (size_t i = 0; i < tensorSizeInElements; i++)
        {
            ASSERT_EQ(pOutputVal[i], 3 * i)
                << "wrong value at index i " << i << " expected " << 3 * i << " actual " << pOutputVal[i] << std::endl;
        }

        status = synRecipeDestroy(recipeHandleB);
        ASSERT_EQ(status, synSuccess) << "Failed destroy RecipeB handle";
    };

    std::vector<std::thread> duplicateThreads;
    duplicateThreads.reserve(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        duplicateThreads.emplace_back(std::thread(duplicateFunc));
    }
    for (auto& duplicateThread : duplicateThreads)
    {
        duplicateThread.join();
    }

    // Cleanups
    status = synGraphDestroy(graphA);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph A";
}

TEST_F_GC(SynGaudiEagerAPITests, graph_duplication_add_multi_threaded, {synDeviceGaudi2, synDeviceGaudi3})
{
    runMultiThreadedDuplicateAddTest(false, false, false);
}

TEST_F_GC(SynGaudiEagerAPITests, eager_graph_duplication_add_multi_threaded, {synDeviceGaudi2, synDeviceGaudi3})
{
    runMultiThreadedDuplicateAddTest(true, false, false);
}

TEST_F_GC(SynGaudiEagerAPITests, graph_duplication_add_multi_threaded_shrink, {synDeviceGaudi2, synDeviceGaudi3})
{
    runMultiThreadedDuplicateAddTest(false, true, false);
}

TEST_F_GC(SynGaudiEagerAPITests, eager_graph_duplication_add_multi_threaded_shrink, {synDeviceGaudi2, synDeviceGaudi3})
{
    runMultiThreadedDuplicateAddTest(true, true, false);
}

TEST_F_GC(SynGaudiEagerAPITests, graph_duplication_add_multi_threaded_expand, {synDeviceGaudi2, synDeviceGaudi3})
{
    runMultiThreadedDuplicateAddTest(false, false, true);
}

TEST_F_GC(SynGaudiEagerAPITests, eager_graph_duplication_add_multi_threaded_expand, {synDeviceGaudi2, synDeviceGaudi3})
{
    runMultiThreadedDuplicateAddTest(true, false, true);
}

TEST_F_GC(SynGaudiEagerAPITests, graph_duplication_with_shape_inference)
{
    synGraphHandle graphA;
    synStatus      status = synGraphCreateEager(&graphA, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

    constexpr unsigned         dims      = 4;
    std::array<unsigned, dims> sizes     = {16, 16, 16, 16};
    std::array<unsigned, dims> new_sizes = {8, 8, 8, 8};

    synTensor in = createTrainingTensor(sizes.size(), syn_type_single, sizes.data(), true, "in", graphA);

    synTensor intermediate =
        createTrainingTensor(sizes.size(), syn_type_single, sizes.data(), false, "intermediate", graphA);

    synTensor out = createTrainingTensor(sizes.size(), syn_type_single, sizes.data(), true, "out", graphA);

    // nodes are added out of order for the test:
    status = synNodeCreate(graphA, &intermediate, &out, 1, 1, nullptr, 0, "relu_fwd_f32", "2nd_node", nullptr, nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create 2nd relu Node";
    status = synNodeCreate(graphA, &in, &intermediate, 1, 1, nullptr, 0, "relu_fwd_f32", "1st_node", nullptr, nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create 1st relu Node";

    uint32_t                        nodesMapSize   = 2;
    uint32_t                        tensorsMapSize = 3;
    synGraphHandle                  graphB;
    std::vector<synTensorHandleMap> tensorsMap(tensorsMapSize);
    std::vector<synNodeHandleMap>   nodesMap(nodesMapSize);

    status = synGraphDuplicate(graphA, &graphB, tensorsMap.data(), &tensorsMapSize, nodesMap.data(), &nodesMapSize);
    ASSERT_EQ(status, synSuccess) << "graph duplication failed";

    {
        synTensorGeometry geometry;
        geometry.dims = new_sizes.size();
        for (size_t i = 0; i < geometry.dims; i++)
        {
            geometry.sizes[i] = new_sizes[i];
        }

        for (const auto& [origTensor, dupTensor] : tensorsMap)
        {
            // set both persistent tensors (input and output) and leave the intermediate up for shape-inference
            if (origTensor == in || origTensor == out)
            {
                status = synTensorSetGeometry(dupTensor, &geometry, synGeometryMaxSizes);
                ASSERT_EQ(status, synSuccess) << "Failed to set geometry";
            }
        }
    }

    status = synGraphInferShapes(graphB);
    ASSERT_EQ(status, synSuccess) << "Failed on graphB shape inference";

    synRecipeHandle recipeHandleB;
    status = synGraphCompile(&recipeHandleB, graphB, GetTestFileName().c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed on graphB compilation";

    // Cleanups
    status = synGraphDestroy(graphA);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph A";

    status = synGraphDestroy(graphB);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph B";

    status = synRecipeDestroy(recipeHandleB);
    ASSERT_EQ(status, synSuccess) << "Failed destroy RecipeB handle";
}
#include "syn_base_test.hpp"
#include "../../src/include/tensor.h"
#include "perf_lib_layer_params.h"
#include "test_utils.h"
#include "platform/gaudi/utils.hpp"
#include "gaudi/gaudi_packets.h"
#include "test_config.hpp"
#include "test_tensors_container.hpp"
#include "test_recipe_interface.hpp"

using namespace std::literals::chrono_literals;

class SynApiNoDeviceTest : public SynBaseTest
{
public:
    SynApiNoDeviceTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    ~SynApiNoDeviceTest() override = default;

    enum class graphDuplicationFailure
    {
        SMALL_TENSOR_MAP_ARRAY,
        BIG_TENSOR_MAP_ARRAY,
        SMALL_NODE_MAP_ARRAY,
        BIG_NODE_MAP_ARRAY,
        COMPILED_GRAPH,
        NON_EXISTENT_GRAPH
    };

    void TestGraphDuplicationFailure(graphDuplicationFailure failureType);
};

REGISTER_SUITE(SynApiNoDeviceTest, ALL_TEST_PACKAGES);

TEST_F_SYN(SynApiNoDeviceTest, section_group_semantic_checks)
{
    synSectionHandle hSection {};
    ASSERT_EQ(synFail, synSectionCreate(&hSection, 0, nullptr));

    synGraphHandle hGraph {};
    ASSERT_EQ(synSuccess, synGraphCreate(&hGraph, m_deviceType));

    ASSERT_EQ(synSuccess, synSectionCreate(&hSection, 0, hGraph));
    ASSERT_EQ(synInvalidArgument, synSectionCreate(nullptr, 0, hGraph));

    ASSERT_EQ(synSuccess, synSectionSetGroup(hSection, 100));
    ASSERT_EQ(synFail, synSectionSetGroup(nullptr, 100));
    ASSERT_EQ(synUnsupported, synSectionSetGroup(hSection, 300));

    uint64_t group = 0;
    ASSERT_EQ(synSuccess, synSectionGetGroup(hSection, &group));
    ASSERT_EQ(100, group);
    ASSERT_EQ(synFail, synSectionGetGroup(nullptr, &group));
    ASSERT_EQ(synInvalidArgument, synSectionGetGroup(hSection, nullptr));

    TSize                sizes[] = {1, 2, 3, 4};
    TestTensorsContainer testTensorInfo(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(testTensorInfo,
                                              0 /* tensorIndex */,
                                              4,
                                              syn_type_float,
                                              sizes,
                                              true,
                                              "tensor",
                                              hGraph,
                                              &hSection,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, testTensorInfo.tensor(0));

    ASSERT_EQ(synFail, synSectionSetGroup(hSection, 101))
        << "Expected to fail in setting section group after tensor creation with the section.";
    group = 0;
    ASSERT_EQ(synSuccess, synSectionGetGroup(hSection, &group));
    ASSERT_EQ(100, group);

    synGraphDestroy(hGraph);
}

TEST_F_SYN(SynApiNoDeviceTest, section_persistency_semantic_checks)
{
    synGraphHandle hGraph {};
    ASSERT_EQ(synSuccess, synGraphCreate(&hGraph, m_deviceType));

    synSectionHandle hSection {};
    ASSERT_EQ(synSuccess, synSectionCreate(&hSection, 0, hGraph));

    bool persistency = true;
    ASSERT_EQ(synSuccess, synSectionGetPersistent(hSection, &persistency));
    ASSERT_TRUE(persistency);
    ASSERT_EQ(synFail, synSectionGetPersistent(nullptr, &persistency));
    ASSERT_EQ(synInvalidArgument, synSectionGetPersistent(hSection, nullptr));

    ASSERT_EQ(synSuccess, synSectionSetPersistent(hSection, false));
    ASSERT_EQ(synFail, synSectionSetPersistent(nullptr, false));
    ASSERT_EQ(synSuccess, synSectionGetPersistent(hSection, &persistency));
    ASSERT_FALSE(persistency);

    ASSERT_EQ(synSuccess, synSectionSetPersistent(hSection, true));
    ASSERT_EQ(synSuccess, synSectionGetPersistent(hSection, &persistency));
    ASSERT_TRUE(persistency);

    TSize                sizes[] = {1, 2, 3, 4};
    TestTensorsContainer testTensorInfo(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(testTensorInfo,
                                              0 /* tensorIndex */,
                                              4,
                                              syn_type_float,
                                              sizes,
                                              true,
                                              "tensor",
                                              hGraph,
                                              &hSection,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, testTensorInfo.tensor(0));

    ASSERT_EQ(synFail, synSectionSetPersistent(hSection, false)) << "Persistency modified after tensor creation";
    ASSERT_EQ(synSuccess, synSectionGetPersistent(hSection, &persistency));
    ASSERT_TRUE(persistency);

    synGraphDestroy(hGraph);
}

TEST_F_SYN(SynApiNoDeviceTest, section_rmw_semantic_checks)
{
    synGraphHandle hGraph {};
    ASSERT_EQ(synSuccess, synGraphCreate(&hGraph, m_deviceType));

    synSectionHandle hSection {};
    ASSERT_EQ(synSuccess, synSectionCreate(&hSection, 0, hGraph));

    bool rmw = true;
    ASSERT_EQ(synSuccess, synSectionGetRMW(hSection, &rmw));
    ASSERT_FALSE(rmw);
    ASSERT_EQ(synFail, synSectionGetRMW(nullptr, &rmw));
    ASSERT_EQ(synInvalidArgument, synSectionGetRMW(hSection, nullptr));

    ASSERT_EQ(synSuccess, synSectionSetRMW(hSection, true));
    ASSERT_EQ(synSuccess, synSectionGetRMW(hSection, &rmw));
    ASSERT_TRUE(rmw);

    unsigned            sizes[] = {1, 2, 3, 4};
    synTensorDescriptor desc {};

    // input
    desc.m_dataType = syn_type_float;
    desc.m_dims     = 4;
    desc.m_name     = "tensor";
    std::copy(sizes, sizes + 4, desc.m_sizes);

    synTensor tensor;

    if (m_deviceType == synDeviceGaudi)
    {
        ASSERT_EQ(synInvalidArgument, synTensorCreate(&tensor, &desc, hSection, 0))
            << "Should not be able to create a tensor in a persistent RMW section in Gaudi";
    }

    ASSERT_EQ(synSuccess, synSectionSetPersistent(hSection, false));
    ASSERT_EQ(synSuccess, synTensorCreate(&tensor, &desc, hSection, 0));

    ASSERT_EQ(synFail, synSectionSetRMW(hSection, false)) << "RMW modified after tensor creation";
    ASSERT_EQ(synSuccess, synSectionGetRMW(hSection, &rmw));
    ASSERT_TRUE(rmw);

    synTensorDestroy(tensor);
    synSectionDestroy(hSection);
    synGraphDestroy(hGraph);
}

TEST_F_SYN(SynApiNoDeviceTest, create_multiple_graphs_and_tensors)
{
    static const size_t                                                            NUM_OF_ITERATIONS        = 3;
    static const size_t                                                            NUM_OF_TENSORS_PER_GRAPH = 5;
    std::vector<synGraphHandle>                                                    hGraph(NUM_OF_ITERATIONS);
    std::vector<synSectionHandle>                                                  hSection(NUM_OF_ITERATIONS);
    std::array<std::array<synTensor, NUM_OF_ITERATIONS>, NUM_OF_TENSORS_PER_GRAPH> tensors;
    std::vector<synTensorDescriptor>                                               desc(NUM_OF_ITERATIONS);

    for (size_t graphIndex = 0; graphIndex < NUM_OF_ITERATIONS; graphIndex++)
    {
        ASSERT_EQ(synSuccess, synGraphCreate(&hGraph[graphIndex], m_deviceType));

        ASSERT_EQ(synSuccess, synSectionCreate(&hSection[graphIndex], 0, hGraph[graphIndex]));

        ASSERT_EQ(synSuccess, synSectionSetRMW(hSection[graphIndex], true));

        unsigned sizes[] = {1, 2, 3, 4};
        ASSERT_EQ(synSuccess, synSectionSetPersistent(hSection[graphIndex], false));

        // input
        desc[graphIndex].m_dataType = syn_type_float;
        desc[graphIndex].m_dims     = 4;
        std::copy(sizes, sizes + 4, desc[graphIndex].m_sizes);
        for (size_t tensorIndex = 0; tensorIndex < NUM_OF_TENSORS_PER_GRAPH; tensorIndex++)
        {
            std::string tensorName  = "tensor" + std::to_string(tensorIndex);
            desc[graphIndex].m_name = tensorName.c_str();
            ASSERT_EQ(synSuccess,
                      synTensorCreate(&tensors[graphIndex][tensorIndex], &desc[graphIndex], hSection[graphIndex], 0));
        }
    }
    // test const tensor creation
    synTensor           constTensor;
    synTensorDescriptor constDesc;
    constDesc.m_name          = "const_tensor";
    constDesc.m_isQuantized   = true;
    constDesc.m_dataType      = syn_type_float;
    const unsigned bufferSize = 10;
    auto           buffer     = std::vector<float>(bufferSize);
    constDesc.m_ptr           = buffer.data();
    ASSERT_EQ(synSuccess, synConstTensorCreate(&constTensor, &constDesc));

    uint32_t constGraphId = reinterpret_cast<Tensor*>(constTensor)->getGraphID();
    for (size_t graphIndex = 0; graphIndex < NUM_OF_ITERATIONS; graphIndex++)
    {
        // const tensor is created with dummy graph id - different from current graphID
        ASSERT_NE(constGraphId, graphIndex);
        for (size_t tensorIndex = 0; tensorIndex < NUM_OF_TENSORS_PER_GRAPH - 1; tensorIndex++)
        {
            ASSERT_EQ(synSuccess, synTensorDestroy(tensors[graphIndex][tensorIndex]));
        }
        ASSERT_EQ(synSuccess, synGraphDestroy(hGraph[graphIndex]));
        ASSERT_EQ(synSuccess, synSectionDestroy(hSection[graphIndex]));
    }
    ASSERT_EQ(synSuccess, synTensorDestroy(constTensor));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_null_values)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor   testTensor;
    std::string tensorName = "testTensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

    synQuantMetadata     quantMetadata;
    synQuantFlags        quantFlags;
    synQuantDynamicRange dynamicRange;

    ASSERT_EQ(synInvalidArgument,
              synTensorGetQuantizationData(nullptr, SYN_QUANT_METADATA, &quantMetadata, sizeof(synQuantMetadata)));
    ASSERT_EQ(synInvalidArgument,
              synTensorGetQuantizationData(testTensor, SYN_QUANT_METADATA, nullptr, sizeof(synQuantMetadata)));

    ASSERT_EQ(synInvalidArgument,
              synTensorGetQuantizationData(nullptr, SYN_QUANT_FLAGS, &quantFlags, sizeof(synQuantFlags)));
    ASSERT_EQ(synInvalidArgument,
              synTensorGetQuantizationData(testTensor, SYN_QUANT_FLAGS, nullptr, sizeof(synQuantFlags)));

    ASSERT_EQ(
        synInvalidArgument,
        synTensorGetQuantizationData(nullptr, SYN_QUANT_DYNAMIC_RANGE, &dynamicRange, sizeof(synQuantDynamicRange)));
    ASSERT_EQ(synInvalidArgument,
              synTensorGetQuantizationData(testTensor, SYN_QUANT_DYNAMIC_RANGE, nullptr, sizeof(synQuantDynamicRange)));

    synDataType deviceDataType;
    ASSERT_EQ(synInvalidArgument, synTensorGetDeviceDataType(nullptr, &deviceDataType));
    ASSERT_EQ(synInvalidArgument, synTensorGetDeviceDataType(testTensor, nullptr));
    ASSERT_EQ(synInvalidArgument, synTensorSetDeviceDataType(nullptr, syn_type_float));

    synTensorGeometryExt geometry;
    ASSERT_EQ(synInvalidArgument, synTensorGetGeometryExt(nullptr, &geometry, synGeometryMaxSizes));
    ASSERT_EQ(synInvalidArgument, synTensorGetGeometryExt(testTensor, nullptr, synGeometryMaxSizes));
    ASSERT_EQ(synInvalidArgument, synTensorSetGeometryExt(nullptr, &geometry, synGeometryMaxSizes));
    ASSERT_EQ(synInvalidArgument, synTensorSetGeometryExt(testTensor, nullptr, synGeometryMaxSizes));

    char*       buffer     = nullptr;
    uint64_t    bufferSize = 0;
    synDataType dataType   = syn_type_na;
    ASSERT_EQ(synInvalidArgument, synTensorGetHostPtr(testTensor, nullptr, &bufferSize, &dataType));
    ASSERT_EQ(synInvalidArgument, synTensorGetHostPtr(nullptr, (void**)&buffer, &bufferSize, &dataType));

    synSectionHandle section;
    uint64_t         sectionOffset = 0;
    ASSERT_EQ(synInvalidArgument, synTensorGetSection(testTensor, nullptr, &sectionOffset));
    ASSERT_EQ(synInvalidArgument, synTensorGetSection(nullptr, &section, &sectionOffset));

    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_device_dtype_geometry_external)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor   testTensor;
    std::string tensorName = "testTensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

    synDataType deviceDataType;
    // verify get uninitialized property;
    ASSERT_EQ(synObjectNotInitialized, synTensorGetDeviceDataType(testTensor, &deviceDataType));

    // set properties
    synDataType type = syn_type_float;

    unsigned index = 0;
    uint64_t maxSizes[HABANA_DIM_MAX];
    std::fill(std::begin(maxSizes), std::end(maxSizes), 1);
    maxSizes[index++] = 3;
    maxSizes[index++] = 3;
    maxSizes[index++] = 3;
    maxSizes[index++] = 1;
    maxSizes[index++] = 1;
    //
    index = 0;
    uint64_t minSizes[HABANA_DIM_MAX];
    std::fill(std::begin(minSizes), std::end(minSizes), 1);
    minSizes[index++] = 3;
    minSizes[index++] = 3;
    minSizes[index++] = 1;
    minSizes[index++] = 1;
    minSizes[index++] = 1;

    size_t   sizesSize = HABANA_DIM_MAX * sizeof(uint64_t);
    uint32_t dims      = 4;

    synTensorGeometryExt maxGeometry;
    maxGeometry.dims = dims;
    memcpy(maxGeometry.sizes, maxSizes, sizesSize);
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(testTensor, &maxGeometry, synGeometryMaxSizes));

    synTensorGeometryExt minGeometry;
    minGeometry.dims = dims;
    memcpy(minGeometry.sizes, minSizes, sizesSize);
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(testTensor, &minGeometry, synGeometryMinSizes));

    ASSERT_EQ(synSuccess, synTensorSetDeviceDataType(testTensor, type));

    bool isExternal = true;
    if (m_deviceType != synDeviceGaudi)
    {
        ASSERT_EQ(synSuccess, synTensorSetExternal(testTensor, isExternal));
    }

    // get properties
    ASSERT_EQ(synSuccess, synTensorGetDeviceDataType(testTensor, &deviceDataType));
    ASSERT_EQ(type, deviceDataType);

    synTensorGeometryExt testMaxGeometry;
    ASSERT_EQ(synSuccess, synTensorGetGeometryExt(testTensor, &testMaxGeometry, synGeometryMaxSizes));
    ASSERT_EQ(0, memcmp(maxSizes, testMaxGeometry.sizes, sizesSize));
    ASSERT_EQ(dims, testMaxGeometry.dims);

    synTensorGeometryExt testMinGeometry;
    ASSERT_EQ(synSuccess, synTensorGetGeometryExt(testTensor, &testMinGeometry, synGeometryMinSizes));
    ASSERT_EQ(0, memcmp(minSizes, testMinGeometry.sizes, sizesSize));
    ASSERT_EQ(dims, testMinGeometry.dims);

    bool resIsExternal = false;
    if (m_deviceType != synDeviceGaudi)
    {
        ASSERT_EQ(synSuccess, synTensorGetExternal(testTensor, &resIsExternal));
        ASSERT_EQ(isExternal, resIsExternal);
    }
    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_permutation)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor   testTensor;
    std::string tensorName = "testTensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

    synTensorPermutation permResult;
    ASSERT_EQ(synObjectNotInitialized, synTensorGetPermutation(testTensor, &permResult));

    synTensorPermutation perm;
    perm.dims = 4;

    // invalid permutation (should include all dims from 0-3)
    perm.permutation[0] = 2;
    perm.permutation[1] = 5;
    perm.permutation[2] = 1;
    perm.permutation[3] = 3;
    ASSERT_EQ(synInvalidArgument, synTensorSetPermutation(testTensor, &perm));

    // valid permutation
    perm.permutation[0] = 2;
    perm.permutation[1] = 0;
    perm.permutation[2] = 1;
    perm.permutation[3] = 3;
    ASSERT_EQ(synSuccess, synTensorSetPermutation(testTensor, &perm));

    ASSERT_EQ(synSuccess, synTensorGetPermutation(testTensor, &permResult));

    ASSERT_EQ(permResult.dims, perm.dims);
    for (unsigned i = 0; i < perm.dims; i++)
    {
        ASSERT_EQ(permResult.permutation[i], perm.permutation[i]);
    }

    // unset permutation by passing dims=0
    perm.dims = 0;
    ASSERT_EQ(synSuccess, synTensorSetPermutation(testTensor, &perm));
    ASSERT_EQ(synObjectNotInitialized, synTensorGetPermutation(testTensor, &permResult));

    // set the permutation again
    perm.dims = 4;
    ASSERT_EQ(synSuccess, synTensorSetPermutation(testTensor, &perm));

    ASSERT_EQ(synSuccess, synTensorSetDeviceDataType(testTensor, syn_type_float));

    uint64_t             maxSizes[HABANA_DIM_MAX] = {3, 5, 10, 2, 1};
    synTensorGeometryExt maxGeometry;
    maxGeometry.dims = 4;
    memcpy(maxGeometry.sizes, maxSizes, sizeof(maxSizes));
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(testTensor, &maxGeometry, synGeometryMaxSizes));

    ns_ConstantKernel::Params params = {0};
    // can't set a permutation on a non-persistent tensor, so creating the node which uses the tensor should fail
    ASSERT_EQ(synInvalidArgument,
              synNodeCreate(graphHandle,
                            nullptr,
                            &testTensor,
                            0,
                            1,
                            &params,
                            sizeof(params),
                            "constant_f32",
                            "testNode",
                            nullptr,
                            nullptr));

    // set the tensor as persistent
    synSectionHandle section;
    ASSERT_EQ(synSuccess, synSectionCreate(&section, 0, graphHandle));
    ASSERT_EQ(synSuccess, synTensorAssignToSection(testTensor, section, 0));

    // setting geometry with different number of dims than the permutation
    maxGeometry.dims = 5;
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(testTensor, &maxGeometry, synGeometryMaxSizes));

    // dims in tensor geometry doesn't match dims in permutation, so creating the node which uses the tensor should fail
    ASSERT_EQ(synInvalidArgument,
              synNodeCreate(graphHandle,
                            nullptr,
                            &testTensor,
                            0,
                            1,
                            &params,
                            sizeof(params),
                            "constant_f32",
                            "testNode",
                            nullptr,
                            nullptr));

    maxGeometry.dims = 4;
    // setting geometry with correct number of dims (like the permutation)
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(testTensor, &maxGeometry, synGeometryMaxSizes));

    ASSERT_EQ(synSuccess,
              synNodeCreate(graphHandle,
                            nullptr,
                            &testTensor,
                            0,
                            1,
                            &params,
                            sizeof(params),
                            "constant_f32",
                            "testNode",
                            nullptr,
                            nullptr));

    // can't set a permutation (or any other tensor attribute) after the tensor is finalized (used in some node)
    ASSERT_EQ(synObjectAlreadyInitialized, synTensorSetPermutation(testTensor, &perm));

    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_name)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor   testTensor;
    std::string tensorName = "testTensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

    size_t nameSize       = tensorName.size() + 1;
    char*  testTensorName = new char[nameSize];
    ASSERT_EQ(synInvalidArgument, synTensorGetName(testTensor, nameSize - 1, testTensorName));  // size is too large
    ASSERT_EQ(synSuccess, synTensorGetName(testTensor, nameSize, testTensorName));
    ASSERT_EQ(0, strcmp(tensorName.c_str(), testTensorName));
    delete[] testTensorName;
    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_type)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor     testTensor;
    synTensorType setType = DATA_TENSOR;
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, setType, "data"));
    synTensorType getType;
    ASSERT_EQ(synSuccess, synTensorGetType(testTensor, &getType));
    ASSERT_EQ(setType, getType);

    setType = SHAPE_TENSOR;
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, setType, "shape"));
    ASSERT_EQ(synSuccess, synTensorGetType(testTensor, &getType));
    ASSERT_EQ(setType, getType);

    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, get_graph_device_type)
{
    synGraphHandle graphHandle;

    synStatus status = synGraphCreate(&graphHandle, m_deviceType);
    ASSERT_EQ(synSuccess, status);

    synDeviceType deviceType;
    status = synGraphGetDeviceType(graphHandle, &deviceType);
    ASSERT_EQ(synSuccess, status);
    ASSERT_EQ(m_deviceType, deviceType);

    status = synGraphDestroy(graphHandle);
    ASSERT_EQ(synSuccess, status);
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_qunatization_data)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor   testTensor;
    std::string tensorName = "testTensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

    // set basic properties that are required by quantization properties
    synTensorGeometryExt geometry   = {{3, 3, 3, 1, 1}, 4};
    uint64_t             size       = 27;
    synDataType          dtype      = syn_type_uint8;
    auto                 hostBuffer = std::vector<float>(size);

    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(testTensor, &geometry, synGeometryMaxSizes));
    ASSERT_EQ(synSuccess, synTensorSetDeviceDataType(testTensor, dtype));
    ASSERT_EQ(synSuccess, synTensorSetHostPtr(testTensor, hostBuffer.data(), size, dtype, true));

    /* test quantization properties */
    synQuantFlags             quantFlags;
    synQuantDynamicRange      dynamicRange;
    synPerChannelDynamicRange pcDynamicRange;
    synQuantMetadata          quantMetadata;
    synFpQuantMetadata        fpQuantMetadata;

    /* test SYN_QUANT_FLAGS */
    quantFlags.enablePerChannelQuant = 1;
    quantFlags.isSparsifiedWeights   = 1;
    quantFlags.isWeights             = 1;
    synQuantFlags quantFlagsTest;
    // test prop not set
    ASSERT_EQ(synObjectNotInitialized,
              synTensorGetQuantizationData(testTensor, SYN_QUANT_FLAGS, &quantFlagsTest, sizeof(synQuantFlags)));
    // set & get
    ASSERT_EQ(synSuccess,
              synTensorSetQuantizationData(testTensor, SYN_QUANT_FLAGS, &quantFlags, sizeof(synQuantFlags)));
    ASSERT_EQ(synFail,
              synTensorGetQuantizationData(testTensor, SYN_QUANT_FLAGS, &quantFlagsTest, sizeof(synQuantFlags) + 1));

    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor, SYN_QUANT_FLAGS, &quantFlagsTest, sizeof(synQuantFlags)));
    ASSERT_EQ(quantFlagsTest.enablePerChannelQuant, quantFlags.enablePerChannelQuant);
    ASSERT_EQ(quantFlagsTest.isSparsifiedWeights, quantFlags.isSparsifiedWeights);
    ASSERT_EQ(quantFlagsTest.isWeights, quantFlags.isWeights);

    /* test SYN_QUANT_DYNAMIC_RANGE */
    dynamicRange.max = 3;
    dynamicRange.min = dynamicRange.max - 1;
    synQuantDynamicRange dynamicRangeTest;
    // test prop not set
    ASSERT_EQ(synObjectNotInitialized,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_DYNAMIC_RANGE,
                                           &dynamicRangeTest,
                                           sizeof(synQuantDynamicRange)));
    // test set fail with invalid dyn range
    dynamicRange.min = dynamicRange.max + 1;
    ASSERT_EQ(
        synInvalidArgument,
        synTensorSetQuantizationData(testTensor, SYN_QUANT_DYNAMIC_RANGE, &dynamicRange, sizeof(synQuantDynamicRange)));
    // set & get
    dynamicRange.min = dynamicRange.max - 1;
    ASSERT_EQ(
        synSuccess,
        synTensorSetQuantizationData(testTensor, SYN_QUANT_DYNAMIC_RANGE, &dynamicRange, sizeof(synQuantDynamicRange)));
    ASSERT_EQ(synFail,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_DYNAMIC_RANGE,
                                           &dynamicRangeTest,
                                           sizeof(synQuantDynamicRange) - 1));
    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_DYNAMIC_RANGE,
                                           &dynamicRangeTest,
                                           sizeof(synQuantDynamicRange)));
    ASSERT_EQ(dynamicRangeTest.max, dynamicRange.max);
    ASSERT_EQ(dynamicRangeTest.min, dynamicRange.min);

    /* test SYN_QUANT_PC_DYNAMIC_RANGE */
    unsigned             numChannels   = 3;
    synQuantDynamicRange rangesArray[] = {{2, 3}, {3, 4}, {4, 5}};
    pcDynamicRange.ranges              = rangesArray;
    pcDynamicRange.numChannels         = 3;
    synPerChannelDynamicRange pcDynamicRangeTest;
    pcDynamicRangeTest.ranges = nullptr;

    // test prop not set
    ASSERT_EQ(synObjectNotInitialized,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRangeTest,
                                           sizeof(synPerChannelDynamicRange)));

    // test set fail with invalid dyn range
    pcDynamicRange.ranges[0].min = pcDynamicRange.ranges[0].max + 1;
    ASSERT_EQ(synInvalidArgument,
              synTensorSetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRange,
                                           sizeof(synPerChannelDynamicRange)));

    // set & get
    pcDynamicRange.ranges[0].min = pcDynamicRange.ranges[0].max - 1;
    ASSERT_EQ(synSuccess,
              synTensorSetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRange,
                                           sizeof(synPerChannelDynamicRange)));

    ASSERT_EQ(synFail,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRangeTest,
                                           sizeof(synPerChannelDynamicRange) - 1));
    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRangeTest,
                                           sizeof(synPerChannelDynamicRange)));

    // verify returned num channels is correct and array is still null and size is still 0
    ASSERT_EQ(pcDynamicRangeTest.numChannels, numChannels);
    ASSERT_EQ(pcDynamicRangeTest.ranges, nullptr);
    unsigned              rangesSize      = pcDynamicRangeTest.numChannels * sizeof(synQuantDynamicRange);
    synQuantDynamicRange* rangesArrayTest = new synQuantDynamicRange[pcDynamicRangeTest.numChannels];
    pcDynamicRangeTest.ranges             = rangesArrayTest;

    // test get fail with size of just struct
    ASSERT_EQ(synFail,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRangeTest,
                                           sizeof(synPerChannelDynamicRange)));
    // now get also the expBiasScale array
    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_PC_DYNAMIC_RANGE,
                                           &pcDynamicRangeTest,
                                           sizeof(synPerChannelDynamicRange) + rangesSize));

    // verify ranges correct values
    for (int i = 0; i < pcDynamicRange.numChannels; i++)
    {
        ASSERT_EQ(pcDynamicRangeTest.ranges[i].max, pcDynamicRange.ranges[i].max);
        ASSERT_EQ(pcDynamicRangeTest.ranges[i].min, pcDynamicRange.ranges[i].min);
    }

    /* test SYN_QUANT_METADATA */
    quantMetadata.dataType          = dtype;
    quantMetadata.numZPScales       = numChannels;
    synQuantZPScale zpScaleChannel1 = {2, 0.1};
    synQuantZPScale zpScaleChannel2 = {3, 0.02};
    synQuantZPScale zpScaleChannel3 = {127, -0.03};
    synQuantZPScale zpScaleArray[]  = {zpScaleChannel1, zpScaleChannel2, zpScaleChannel3};
    quantMetadata.zpScales          = zpScaleArray;
    synQuantMetadata quantMetadataTest;
    // test prop not set
    ASSERT_EQ(
        synObjectNotInitialized,
        synTensorGetQuantizationData(testTensor, SYN_QUANT_METADATA, &quantMetadataTest, sizeof(synQuantMetadata)));
    // test set get
    quantMetadata.dataType = dtype;
    ASSERT_EQ(synSuccess,
              synTensorSetQuantizationData(testTensor, SYN_QUANT_METADATA, &quantMetadata, sizeof(synQuantMetadata)));

    // test get with wrong data type
    quantMetadataTest.dataType = syn_type_int16;
    quantMetadataTest.zpScales = nullptr;
    ASSERT_EQ(
        synInvalidArgument,
        synTensorGetQuantizationData(testTensor, SYN_QUANT_METADATA, &quantMetadataTest, sizeof(synQuantMetadata)));

    quantMetadataTest.dataType = dtype;
    // test get fail with wrong size
    ASSERT_EQ(
        synFail,
        synTensorGetQuantizationData(testTensor, SYN_QUANT_METADATA, &quantMetadataTest, sizeof(synQuantMetadata) - 1));
    ASSERT_EQ(
        synSuccess,
        synTensorGetQuantizationData(testTensor, SYN_QUANT_METADATA, &quantMetadataTest, sizeof(synQuantMetadata)));
    // verify returned num channels are correct and array still null and size still 0
    ASSERT_EQ(quantMetadataTest.numZPScales, numChannels);
    ASSERT_EQ(quantMetadataTest.dataType, dtype);
    ASSERT_EQ(quantMetadataTest.zpScales, nullptr);

    unsigned         zpScaleSize      = quantMetadataTest.numZPScales * sizeof(synQuantZPScale);
    synQuantZPScale* zpScaleArrayTest = new synQuantZPScale[zpScaleSize];
    quantMetadataTest.zpScales        = zpScaleArrayTest;

    // test get fail with size of just struct
    ASSERT_EQ(
        synFail,
        synTensorGetQuantizationData(testTensor, SYN_QUANT_METADATA, &quantMetadataTest, sizeof(synQuantMetadata)));
    // now get also the zpScale array
    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor,
                                           SYN_QUANT_METADATA,
                                           &quantMetadataTest,
                                           sizeof(synQuantMetadata) + zpScaleSize));
    // test zpScale array correct values
    for (unsigned i = 0; i < numChannels; i++)
    {
        ASSERT_EQ(quantMetadataTest.zpScales[i].scale, quantMetadata.zpScales[i].scale);
        ASSERT_EQ(quantMetadataTest.zpScales[i].zp, quantMetadata.zpScales[i].zp);
    }
    delete[] zpScaleArrayTest;

    /* test SYN_FP_QUANT_METADATA */
    dtype                                = syn_type_fp8_143;
    fpQuantMetadata.dataType             = dtype;
    fpQuantMetadata.numFpQuantParams     = numChannels;
    synFpQuantParam expBiasScaleChannel1 = {2, 3};
    synFpQuantParam expBiasScaleChannel2 = {3, 7};
    synFpQuantParam expBiasScaleChannel3 = {127, 11};
    synFpQuantParam expBiasScaleArray[]  = {expBiasScaleChannel1, expBiasScaleChannel2, expBiasScaleChannel3};
    fpQuantMetadata.fpQuantParams        = expBiasScaleArray;
    synFpQuantMetadata fpQuantMetadataTest;
    // test prop not set
    ASSERT_EQ(synObjectNotInitialized,
              synTensorGetQuantizationData(testTensor,
                                           SYN_FP_QUANT_METADATA,
                                           &fpQuantMetadataTest,
                                           sizeof(synFpQuantMetadata)));
    // test set get
    fpQuantMetadata.dataType = dtype;
    ASSERT_EQ(
        synSuccess,
        synTensorSetQuantizationData(testTensor, SYN_FP_QUANT_METADATA, &fpQuantMetadata, sizeof(synFpQuantMetadata)));
    // test get with wrong data type
    fpQuantMetadataTest.dataType      = syn_type_int16;
    fpQuantMetadataTest.fpQuantParams = nullptr;
    ASSERT_EQ(synInvalidArgument,
              synTensorGetQuantizationData(testTensor,
                                           SYN_FP_QUANT_METADATA,
                                           &fpQuantMetadataTest,
                                           sizeof(synFpQuantMetadata)));
    fpQuantMetadataTest.dataType = dtype;
    // test get fail with wrong size
    ASSERT_EQ(synFail,
              synTensorGetQuantizationData(testTensor,
                                           SYN_FP_QUANT_METADATA,
                                           &fpQuantMetadataTest,
                                           sizeof(synFpQuantMetadata) - 1));
    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor,
                                           SYN_FP_QUANT_METADATA,
                                           &fpQuantMetadataTest,
                                           sizeof(synFpQuantMetadata)));
    // verify returned num channels are correct and array still null and size still 0
    ASSERT_EQ(fpQuantMetadataTest.numFpQuantParams, numChannels);
    ASSERT_EQ(fpQuantMetadataTest.dataType, dtype);
    ASSERT_EQ(fpQuantMetadataTest.fpQuantParams, nullptr);
    unsigned         expBiasScaleSize      = fpQuantMetadataTest.numFpQuantParams * sizeof(synFpQuantParam);
    synFpQuantParam* expBiasScaleArrayTest = new synFpQuantParam[fpQuantMetadataTest.numFpQuantParams];
    fpQuantMetadataTest.fpQuantParams      = expBiasScaleArrayTest;
    // test get fail with size of just struct
    ASSERT_EQ(synFail,
              synTensorGetQuantizationData(testTensor,
                                           SYN_FP_QUANT_METADATA,
                                           &fpQuantMetadataTest,
                                           sizeof(synFpQuantMetadata)));
    // now get also the expBiasScale array
    ASSERT_EQ(synSuccess,
              synTensorGetQuantizationData(testTensor,
                                           SYN_FP_QUANT_METADATA,
                                           &fpQuantMetadataTest,
                                           sizeof(synFpQuantMetadata) + expBiasScaleSize));
    // test expBiasScale array correct values
    for (unsigned i = 0; i < numChannels; i++)
    {
        ASSERT_EQ(fpQuantMetadataTest.fpQuantParams[i].scale, fpQuantMetadata.fpQuantParams[i].scale);
        ASSERT_EQ(fpQuantMetadataTest.fpQuantParams[i].expBias, fpQuantMetadata.fpQuantParams[i].expBias);
    }

    delete[] expBiasScaleArrayTest;
    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_host_buffer)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor   testTensor;
    std::string tensorName = "testTensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

    // set basic properties
    uint64_t    numElements = 27;
    synDataType dtype       = syn_type_uint8;
    ASSERT_EQ(synSuccess, synTensorSetDeviceDataType(testTensor, dtype));

    std::vector<float> hostBuffer(numElements);
    fillWithRandom(hostBuffer.data(), numElements);
    synDataType bufferDataType    = syn_type_float;
    uint64_t    bufferSizeInBytes = numElements * sizeof(float);

    char*       testBuffer         = nullptr;
    uint64_t    testBufferSize     = 0;
    synDataType testBufferDataType = syn_type_na;
    // test prop not set
    ASSERT_EQ(synObjectNotInitialized,
              synTensorGetHostPtr(testTensor, (void**)&testBuffer, &testBufferSize, &testBufferDataType));
    // test set get
    ASSERT_EQ(synSuccess, synTensorSetHostPtr(testTensor, hostBuffer.data(), bufferSizeInBytes, bufferDataType, true));
    // fail since data type null
    ASSERT_EQ(synInvalidArgument, synTensorGetHostPtr(testTensor, (void**)&testBuffer, &testBufferSize, nullptr));
    // fail since size null
    ASSERT_EQ(synInvalidArgument, synTensorGetHostPtr(testTensor, (void**)&testBuffer, nullptr, &testBufferDataType));
    ASSERT_EQ(synSuccess, synTensorGetHostPtr(testTensor, (void**)&testBuffer, &testBufferSize, &testBufferDataType));
    ASSERT_EQ(testBufferDataType, bufferDataType);
    ASSERT_EQ(testBufferSize, bufferSizeInBytes);
    float* floatBuffer = reinterpret_cast<float*>(testBuffer);  // cast so we can compare
    ASSERT_NE(hostBuffer.data(), floatBuffer);                  // shouldn't be equal since buffer was copied
    ASSERT_NE(testBuffer, nullptr);
    for (unsigned i = 0; i < numElements; i++)
    {
        EXPECT_EQ(hostBuffer.data()[i], floatBuffer[i]);
    }
    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
}

TEST_F_SYN(SynApiNoDeviceTest, test_tensor_api_section)
{
    invokeMultiThread(8, 300, [&]() {
        synGraphHandle graphHandle;
        ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

        synTensor   testTensor;
        std::string tensorName = "testTensor";
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&testTensor, graphHandle, DATA_TENSOR, tensorName.c_str()));

        // test prop not set
        synSectionHandle testSection;
        uint64_t         testSectionOffset = 0;
        ASSERT_EQ(synObjectNotInitialized, synTensorGetSection(testTensor, &testSection, &testSectionOffset));
        // test assign and get
        synSectionHandle section;
        ASSERT_EQ(synSuccess, synSectionCreate(&section, 0, graphHandle));
        uint64_t sectionOffset = 10;
        ASSERT_EQ(synSuccess, synTensorAssignToSection(testTensor, section, sectionOffset));
        ASSERT_EQ(synInvalidArgument,
                  synTensorGetSection(testTensor, &testSection, nullptr));  // fail since offset is null
        ASSERT_EQ(synSuccess, synTensorGetSection(testTensor, &testSection, &testSectionOffset));
        ASSERT_NE(testSection, nullptr);
        ASSERT_EQ(section, testSection);  // should be same since point to same section
        ASSERT_EQ(sectionOffset, testSectionOffset);
        ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
        ASSERT_EQ(synSuccess, synSectionDestroy(section));
    });
}

TEST_F_SYN(SynApiNoDeviceTest, test_concurrent_graphs_api_creation)
{
    invokeMultiThread(8, 300, [&]() {
        synGraphHandle graphHandle;
        ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));
        ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
    });
}

TEST_F_SYN(SynApiNoDeviceTest, test_RMW_tensor_out_of_section)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor in;
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&in, graphHandle, DATA_TENSOR, "in"));
    // set geometry and device layout for total size in bytes
    uint64_t             maxSizes[HABANA_DIM_MAX] = {3, 3, 3, 1, 1};
    size_t               sizesSize                = HABANA_DIM_MAX * sizeof(uint64_t);
    uint32_t             dims                     = 4;
    synTensorGeometryExt maxGeometry;
    maxGeometry.dims = dims;
    memcpy(maxGeometry.sizes, maxSizes, sizesSize);
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(in, &maxGeometry, synGeometryMaxSizes));
    ASSERT_EQ(synSuccess, synTensorSetDeviceDataType(in, syn_type_int8));
    // set the section
    synSectionHandle section;
    ASSERT_EQ(synSuccess, synSectionCreate(&section, 0, graphHandle));
    ASSERT_EQ(synSuccess, synSectionSetPersistent(section, false));  // apperantley section is persistent by default
    ASSERT_EQ(synSuccess, synSectionSetRMW(section, true));
    // set offset such that in tensor exceeds max section size
    uint64_t sectionOffset = GCFG_RMW_SECTION_MAX_SIZE_BYTES.value();
    ASSERT_EQ(synSuccess, synTensorAssignToSection(in, section, sectionOffset));
    // add another tensor and try to create the node
    synTensor out;
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&out, graphHandle, DATA_TENSOR, "out"));
    ASSERT_EQ(synSuccess, synTensorSetGeometryExt(out, &maxGeometry, synGeometryMaxSizes));
    ASSERT_EQ(synSuccess, synTensorSetDeviceDataType(out, syn_type_int8));
    // creating node should fail
    ASSERT_EQ(synFail, synNodeCreate(graphHandle, &in, &out, 1, 1, nullptr, 0, "abs_fwd", "", nullptr, nullptr));
    // clean objects
    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
    ASSERT_EQ(synSuccess, synSectionDestroy(section));
}

TEST_F_SYN(SynApiNoDeviceTest, recompile_failure)
{
    invokeMultiThread(8, 800ms, [&]() {
        synGraphHandle graphA;
        synStatus      status = synGraphCreate(&graphA, m_deviceType);
        ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

        TSize sizes[HABANA_DIM_MAX] = {16, 16, 16, 16};

        TestTensorsContainer tensorInput(1 /* numOfTensors */);
        TestRecipeInterface::createTrainingTensor(tensorInput,
                                                  0 /* tensorIndex */,
                                                  4U,
                                                  syn_type_single,
                                                  sizes,
                                                  true,
                                                  "A_0_2",
                                                  graphA,
                                                  nullptr,
                                                  false /* isConstSection */,
                                                  0,
                                                  nullptr /* hostBuffer */,
                                                  DATA_TENSOR,
                                                  nullptr);
        ASSERT_NE(nullptr, tensorInput.tensor(0));

        TestTensorsContainer tensorOutput(1 /* numOfTensors */);
        TestRecipeInterface::createTrainingTensor(tensorOutput,
                                                  0 /* tensorIndex */,
                                                  4U,
                                                  syn_type_single,
                                                  sizes,
                                                  true,
                                                  "B_0_2",
                                                  graphA,
                                                  nullptr,
                                                  false /* isConstSection */,
                                                  0,
                                                  nullptr /* hostBuffer */,
                                                  DATA_TENSOR,
                                                  nullptr);
        ASSERT_NE(nullptr, tensorOutput.tensor(0));

        status = synNodeCreate(graphA,
                               tensorInput.tensors(),
                               tensorOutput.tensors(),
                               1,
                               1,
                               nullptr,
                               0,
                               "memcpy",
                               "",
                               nullptr,
                               nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create memcpy Node";

        const std::string testName(getTestName());
        synRecipeHandle   recipeHandle;
        status = synGraphCompile(&recipeHandle, graphA, testName.c_str(), nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed on first graph compilation";
        status = synGraphCompile(&recipeHandle, graphA, testName.c_str(), nullptr);
        ASSERT_NE(status, synSuccess) << "Passed second graph compilation (should fail)";

        // Cleanups
        status = synGraphDestroy(graphA);
        ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph";

        status = synRecipeDestroy(recipeHandle);
        ASSERT_EQ(status, synSuccess) << "Failed destroy Recipe handle";
    });
}

TEST_F_SYN(SynApiNoDeviceTest, check_undefined_opcode_in_buffer)
{
    uint64_t bufferSize = sizeof(packet_lin_dma) + sizeof(packet_cp_dma) + sizeof(packet_wreg_bulk) +
                          sizeof(packet_wreg32) + sizeof(packet_msg_long) + sizeof(packet_msg_short) +
                          sizeof(packet_repeat) + sizeof(packet_msg_prot) + sizeof(packet_fence) + sizeof(packet_nop) +
                          sizeof(packet_stop) + sizeof(packet_arb_point) + sizeof(packet_wait) +
                          sizeof(packet_load_and_exe) + sizeof(packet_nop);

    uint8_t*    pBuffer      = new uint8_t[bufferSize];
    uint8_t*    pTmpBuffer   = pBuffer;
    const void* pConstBuffer = (const void*)pBuffer;

    generatePacketOpCode<packet_lin_dma, PACKET_LIN_DMA>(pTmpBuffer);
    generatePacketOpCode<packet_cp_dma, PACKET_CP_DMA>(pTmpBuffer);        // Resetting the CP-DMA-ed' buffer-size
    generatePacketOpCode<packet_wreg_bulk, PACKET_WREG_BULK>(pTmpBuffer);  // Resetting the Bulk' buffer-size
    generatePacketOpCode<packet_wreg32, PACKET_WREG_32>(pTmpBuffer);
    generatePacketOpCode<packet_msg_long, PACKET_MSG_LONG>(pTmpBuffer);
    generatePacketOpCode<packet_msg_short, PACKET_MSG_SHORT>(pTmpBuffer);
    generatePacketOpCode<packet_repeat, PACKET_REPEAT>(pTmpBuffer);
    generatePacketOpCode<packet_msg_prot, PACKET_MSG_PROT>(pTmpBuffer);
    generatePacketOpCode<packet_fence, PACKET_FENCE>(pTmpBuffer);
    generatePacketOpCode<packet_nop, PACKET_NOP>(pTmpBuffer);
    generatePacketOpCode<packet_stop, PACKET_STOP>(pTmpBuffer);
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    generatePacketOpCode<packet_load_and_exe, PACKET_LOAD_AND_EXE>(pTmpBuffer);

    ASSERT_EQ(true,
              gaudi::checkForUndefinedOpcode(pConstBuffer,
                                             bufferSize - sizeof(packet_nop),
                                             PKT_VAIDATION_LOGGING_MODE_UPON_FAILURE));

    generatePacketOpCode<packet_nop, 0x1f>(pTmpBuffer);
    ASSERT_EQ(false, gaudi::checkForUndefinedOpcode(pConstBuffer, bufferSize, PKT_VAIDATION_LOGGING_MODE_UPON_FAILURE));

    delete[] pBuffer;
}

TEST_F_SYN(SynApiNoDeviceTest, get_tpc_libs_versions)
{
    unsigned libsVersionsMapSize = 0;
    ASSERT_EQ(synTPCLibraryGetVersionSize(&libsVersionsMapSize), synSuccess)
        << "Failed to get kernels libs versions size";
    ASSERT_NE(libsVersionsMapSize, 0) << "Failed to get kernels libs versions. No versions were found";

    const char* libsPaths[libsVersionsMapSize];
    unsigned    libsVersions[libsVersionsMapSize];
    ASSERT_EQ(synTPCLibraryGetVersions(libsPaths, libsVersions), synSuccess) << "Failed to get kernels libs versions";

    for (unsigned i = 0; i < libsVersionsMapSize; i++)
    {
        LOG_DEBUG(SYN_RT_TEST, "Found TPC library path {} with version {}", libsPaths[i], libsVersions[i]);
    }
}

void SynApiNoDeviceTest::TestGraphDuplicationFailure(graphDuplicationFailure failureType)
{
    synGraphHandle graphA;
    synStatus      status = synGraphCreate(&graphA, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

    TSize sizes[HABANA_DIM_MAX] = {16, 16, 16, 16};

    TestTensorsContainer tensorInput(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(tensorInput,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              sizes,
                                              true,
                                              "A_0_2",
                                              graphA,
                                              nullptr,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, tensorInput.tensor(0));

    TestTensorsContainer tensorOutput(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(tensorOutput,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              sizes,
                                              true,
                                              "B_0_2",
                                              graphA,
                                              nullptr,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, tensorOutput.tensor(0));

    status = synNodeCreate(graphA,
                           tensorInput.tensors(),
                           tensorOutput.tensors(),
                           1,
                           1,
                           nullptr,
                           0,
                           "memcpy",
                           "",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create memcpy Node";

    uint32_t                        nodesMapSize       = 1;
    uint32_t                        tensorsMapSize     = 2;
    bool                            failPreCompile     = true;
    bool                            failPreDestruction = true;
    synGraphHandle                  graphB;
    std::vector<synTensorHandleMap> tensorsMap(tensorsMapSize);
    std::vector<synNodeHandleMap>   nodesMap(nodesMapSize);
    switch (failureType)
    {
        case graphDuplicationFailure::BIG_NODE_MAP_ARRAY:
            nodesMapSize++;
            break;
        case graphDuplicationFailure::BIG_TENSOR_MAP_ARRAY:
            tensorsMapSize++;
            break;
        case graphDuplicationFailure::SMALL_NODE_MAP_ARRAY:
            nodesMapSize = 0;
            break;
        case graphDuplicationFailure::SMALL_TENSOR_MAP_ARRAY:
            tensorsMapSize = 1;
            break;
        case graphDuplicationFailure::COMPILED_GRAPH:
            failPreCompile = false;
            break;
        case graphDuplicationFailure::NON_EXISTENT_GRAPH:
            failPreCompile = failPreDestruction = false;
            break;
    }

    if (failPreCompile)
    {
        status = synGraphDuplicate(graphA, &graphB, tensorsMap.data(), &tensorsMapSize, nodesMap.data(), &nodesMapSize);
        ASSERT_NE(status, synSuccess) << "graph duplication succeeded for wrong mapping argument size";
    }

    const std::string testName(getTestName());
    synRecipeHandle   recipeHandle;
    status = synGraphCompile(&recipeHandle, graphA, testName.c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed on graph compilation";

    if (failPreDestruction)
    {
        status = synGraphDuplicate(graphA, &graphB, tensorsMap.data(), &tensorsMapSize, nodesMap.data(), &nodesMapSize);
        ASSERT_NE(status, synSuccess) << "graph duplication succeeded for a compiled graph";
    }

    // Cleanups
    status = synGraphDestroy(graphA);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph";

    if (!failPreDestruction && failureType != graphDuplicationFailure::NON_EXISTENT_GRAPH)
    {
        status = synGraphDuplicate(graphA, &graphB, tensorsMap.data(), &tensorsMapSize, nodesMap.data(), &nodesMapSize);
        ASSERT_NE(status, synSuccess) << "graph duplication succeeded for a destroyed graph";
    }

    status = synRecipeDestroy(recipeHandle);
    ASSERT_EQ(status, synSuccess) << "Failed destroy Recipe handle";
}

TEST_F_SYN(SynApiNoDeviceTest, compiled_graph_duplication_failure)
{
    TestGraphDuplicationFailure(graphDuplicationFailure::COMPILED_GRAPH);
}

TEST_F_SYN(SynApiNoDeviceTest, destroyed_graph_duplication_failure)
{
    TestGraphDuplicationFailure(graphDuplicationFailure::NON_EXISTENT_GRAPH);
}

TEST_F_SYN(SynApiNoDeviceTest, big_tensor_map_size_graph_duplication_failure)
{
    TestGraphDuplicationFailure(graphDuplicationFailure::BIG_TENSOR_MAP_ARRAY);
}

TEST_F_SYN(SynApiNoDeviceTest, small_tensor_map_size_graph_duplication_failure)
{
    TestGraphDuplicationFailure(graphDuplicationFailure::SMALL_TENSOR_MAP_ARRAY);
}

TEST_F_SYN(SynApiNoDeviceTest, big_node_map_size_graph_duplication_failure)
{
    TestGraphDuplicationFailure(graphDuplicationFailure::BIG_NODE_MAP_ARRAY);
}

TEST_F_SYN(SynApiNoDeviceTest, small_node_map_size_graph_duplication_failure)
{
    TestGraphDuplicationFailure(graphDuplicationFailure::SMALL_NODE_MAP_ARRAY);
}

TEST_F_SYN(SynApiNoDeviceTest, graph_duplication_good_path)
{
    synGraphHandle graphA;
    synStatus      status = synGraphCreate(&graphA, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

    TSize sizes[HABANA_DIM_MAX] = {16, 16, 16, 16};

    TestTensorsContainer tensorInput(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(tensorInput,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              sizes,
                                              true,
                                              "in",
                                              graphA,
                                              nullptr,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, tensorInput.tensor(0));

    TestTensorsContainer tensorOutput(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(tensorOutput,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              sizes,
                                              true,
                                              "out",
                                              graphA,
                                              nullptr,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, tensorOutput.tensor(0));

    TestTensorsContainer tensorExtra(1 /* numOfTensors */);
    TestRecipeInterface::createTrainingTensor(tensorExtra,
                                              0 /* tensorIndex */,
                                              4U,
                                              syn_type_single,
                                              sizes,
                                              true,
                                              "extra",
                                              graphA,
                                              nullptr,
                                              false /* isConstSection */,
                                              0,
                                              nullptr /* hostBuffer */,
                                              DATA_TENSOR,
                                              nullptr);
    ASSERT_NE(nullptr, tensorExtra.tensor(0));

    status = synNodeCreate(graphA,
                           tensorInput.tensors(),
                           tensorOutput.tensors(),
                           1,
                           1,
                           nullptr,
                           0,
                           "memcpy",
                           "",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create memcpy Node";

    uint32_t                        nodesMapSize   = 1;
    uint32_t                        tensorsMapSize = 3;
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

    const std::string testName(getTestName());
    synRecipeHandle   recipeHandleA;
    status = synGraphCompile(&recipeHandleA, graphA, testName.c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed on graphA compilation";

    synRecipeHandle recipeHandleB;
    status = synGraphCompile(&recipeHandleB, graphB, testName.c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed on graphB compilation";

    // Cleanups
    status = synGraphDestroy(graphA);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph A";

    status = synGraphDestroy(graphB);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph B";

    status = synRecipeDestroy(recipeHandleA);
    ASSERT_EQ(status, synSuccess) << "Failed destroy RecipeA handle";

    status = synRecipeDestroy(recipeHandleB);
    ASSERT_EQ(status, synSuccess) << "Failed destroy RecipeB handle";
}

#include <graph_compiler/habana_nodes/node_factory.h>
#include "gc_gaudi_test_infra.h"

class SynGaudiShapeTensorOperationsTest : public SynGaudiTestInfra
{
public:
    SynGaudiShapeTensorOperationsTest()
    {
        setTestPackage(TEST_PACKAGE_DSD);
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2});
    }
};

TEST_F_GC(SynGaudiShapeTensorOperationsTest, memset_should_write_according_to_the_shape_tensor)
{
    std::vector<unsigned> maxSizes = {8, 256};
    std::vector<unsigned> minSizes = {8, 128};

    auto output = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ONES,
                                      nullptr,
                                      maxSizes.data(),
                                      maxSizes.size(),
                                      syn_type_float,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      minSizes.data());
    auto shape =
        createShapeTensor(INPUT_TENSOR, maxSizes.data(), minSizes.data(), maxSizes.size(), syn_type_uint32, nullptr);

    addNodeToGraph(NodeFactory::memsetNodeTypeName, {shape}, {output});

    compileTopology();

    std::vector<unsigned> actualSizes = maxSizes;
    actualSizes.back() -= 50;
    setActualSizes(shape, actualSizes.data());
    setActualSizes(output, actualSizes.data());

    runTopology(0, true);

    auto pResult = static_cast<float*>(m_hostBuffers[output]);
    auto expZeros      = multiplyElements(actualSizes);
    auto totalElements = multiplyElements(maxSizes);

    unsigned idx = 0;
    for (; idx < expZeros; idx++)
    {
        ASSERT_FLOAT_EQ(pResult[idx], 0.0f);
    }
    for (; idx < totalElements; idx++)
    {
        ASSERT_FLOAT_EQ(pResult[idx], 1.0f);
    }
}

TEST_F_GC(SynGaudiShapeTensorOperationsTest, reshape_should_resize_according_to_shape_tensor)
{
    std::vector<unsigned> inputMaxSize = {512, 1, 1, 1};
    std::vector<unsigned> inputMinSize = {256, 1, 1, 1};
    std::vector<unsigned> addMaxSize   = {8, 64, 1, 1};
    std::vector<unsigned> addMinSize   = {8, 32, 1, 1};

    auto input = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     inputMaxSize.data(),
                                     inputMaxSize.size(),
                                     syn_type_float,
                                     nullptr,
                                     "input",
                                     0,
                                     0,
                                     nullptr,
                                     inputMinSize.data());
    auto shape = createShapeTensor(INPUT_TENSOR, addMaxSize.data(), addMinSize.data(), addMaxSize.size());
    auto reshaped = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, addMaxSize.data(), addMaxSize.size(), syn_type_float, nullptr, addMinSize.data());

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {input, shape}, {reshaped});

    auto addIn0 = connectOutputTensorToInputTensor(reshaped);
    auto addIn1 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      addMaxSize.data(),
                                      addMaxSize.size(),
                                      syn_type_float,
                                      nullptr,
                                      "addIn1",
                                      0,
                                      0,
                                      nullptr,
                                      addMinSize.data());
    auto addOut = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      addMaxSize.data(),
                                      addMaxSize.size(),
                                      syn_type_float,
                                      nullptr,
                                      "addOut",
                                      0,
                                      0,
                                      nullptr,
                                      addMinSize.data());

    addNodeToGraph("add_fwd_f32", {addIn0, addIn1}, {addOut});

    compileTopology();

    std::vector<unsigned> actualSize      = {8, 43, 1, 1};
    std::vector<unsigned> actualInputSize = {multiplyElements(actualSize), 1, 1, 1};
    setActualSizes(input, actualInputSize.data());
    setActualSizes(shape, actualSize.data());
    setActualSizes(addIn1, actualSize.data());
    setActualSizes(addOut, actualSize.data());

    runTopology(0, true);

    auto pResult = static_cast<float*>(m_hostBuffers[addOut]);
    auto pIn1 = static_cast<float*>(m_hostBuffers[input]);
    auto pIn2 = static_cast<float*>(m_hostBuffers[addIn1]);

    unsigned actualElements = multiplyElements(actualSize);
    unsigned maxElements = multiplyElements(inputMaxSize);

    unsigned idx = 0;
    for (; idx < actualElements; idx++)
    {
        ASSERT_FLOAT_EQ(pResult[idx], pIn1[idx] + pIn2[idx]);
    }
    for (; idx < maxElements; idx++)
    {
        ASSERT_FLOAT_EQ(pResult[idx], 0.0f);
    }
}
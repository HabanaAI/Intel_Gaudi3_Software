#include "graph_factory.h"
#include "graph_optimizer_test.h"
#include "generic_graph_test.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "gaudi2_graph.h"
#include "passes/calc_quantization_info.cpp"
#include "synapse_api.h"
#include <data_types/bfloat16.h>

class ScaleCalcTest : public GenericGraphTest
{
protected:
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_CALC_DYNAMIC_RANGE, "true");
        synDeviceType deviceType = GetParam();
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
    }
};

TEST_P(ScaleCalcTest, fp8_calc_scale)
{
    using Tensor1BufferDtype = float;
    using Tensor2BufferDtype = bfloat16;

    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    std::vector<TSize>    sizes          = {1, 2, 2, 1};
    const TSize           elementsNum    = 4;
    constexpr std::size_t input1BuffSize = sizeof(Tensor1BufferDtype) * elementsNum;
    constexpr std::size_t input2BuffSize = sizeof(Tensor2BufferDtype) * elementsNum;

    std::vector<Tensor1BufferDtype> input1 = {-15000, 7800, 19200, 900};
    std::vector<Tensor2BufferDtype> input2 = {0.1, -0.7, -0.9375, 0.8};

    auto inputBuffer1 = std::make_unique<char[]>(input1BuffSize);
    std::memcpy(inputBuffer1.get(), input1.data(), input1BuffSize);

    auto inputBuffer2 = std::make_unique<char[]>(input2BuffSize);
    std::memcpy(inputBuffer2.get(), input2.data(), input2BuffSize);

    TensorPtr t1 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t1->setName("t1", true);
    t1->setAsStaticParam(true);
    t1->setTensorBuffer(inputBuffer1.get(),
                        input1.size() * sizeof(Tensor1BufferDtype),
                        asSynType<Tensor1BufferDtype>());

    TensorPtr t2 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t2->setName("t2", true);
    t2->setTensorBuffer(inputBuffer2.get(),
                        input2.size() * sizeof(Tensor2BufferDtype),
                        asSynType<Tensor2BufferDtype>());
    DynamicRange dynamicRange;
    dynamicRange.max   = 0.8;
    dynamicRange.min   = -0.9375;
    dynamicRange.isSet = true;
    ASSERT_TRUE(t2->setDynamicRange(dynamicRange));

    TensorPtr t3 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t3->setName("t3", true);

    synGEMMParams params {};
    pNode         gemm = NodeFactory::createNode({t1, t2}, {t3}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm));

    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_EQ((*m_graph).getTensors().size(), 3);
    ASSERT_TRUE(calcDynamicRange(*m_graph));
    ASSERT_TRUE(calcQuantizationInfo(*m_graph));
    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).scale(), 5);
    ASSERT_FLOAT_EQ(t2->getQuantizationParams(syn_type_fp8_143).scale(), 1);
    ASSERT_EQ(t1->getQuantizationParams(syn_type_fp8_152).scale(), 1);
}

TEST_P(ScaleCalcTest, fp8_calc_PC_scale)
{
    setGlobalConfForTest(GCFG_PER_CHANNEL_SCALING, "true");

    using Tensor1BufferDtype = float;
    using Tensor2BufferDtype = bfloat16;

    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    std::vector<TSize>    sizes          = {2, 2, 2, 1};
    const TSize           elementsNum    = 8;
    constexpr std::size_t input1BuffSize = sizeof(Tensor1BufferDtype) * elementsNum;
    constexpr std::size_t input2BuffSize = sizeof(Tensor2BufferDtype) * elementsNum;

    std::vector<Tensor1BufferDtype> input1 = {-15000, -150, 7800, 78, 19200, 192, 900, 9};
    std::vector<Tensor2BufferDtype> input2 = {0.1, 0.1, -0.7, -0.7, -0.9375, -0.9375, 0.8, 0.8};

    auto inputBuffer1 = std::make_unique<char[]>(input1BuffSize);
    std::memcpy(inputBuffer1.get(), input1.data(), input1BuffSize);

    auto inputBuffer2 = std::make_unique<char[]>(input2BuffSize);
    std::memcpy(inputBuffer2.get(), input2.data(), input2BuffSize);

    TensorPtr t1 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t1->setName("t1", true);
    t1->setAsStaticParam(true);
    t1->setTensorBuffer(inputBuffer1.get(),
                        input1.size() * sizeof(Tensor1BufferDtype),
                        asSynType<Tensor1BufferDtype>(),
                        false);

    TensorPtr t2 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t2->setName("t2", true);
    t2->setTensorBuffer(inputBuffer2.get(),
                        input2.size() * sizeof(Tensor2BufferDtype),
                        asSynType<Tensor2BufferDtype>(),
                        false);
    DynamicRange dynamicRange;
    dynamicRange.max   = 0.8;
    dynamicRange.min   = -0.9375;
    dynamicRange.isSet = true;
    ASSERT_TRUE(t2->setDynamicRange(dynamicRange));

    TensorPtr t3 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t3->setName("t3", true);

    synGEMMParams params {};
    pNode         gemm = NodeFactory::createNode({t2, t1}, {t3}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm));

    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_EQ((*m_graph).getTensors().size(), 3);
    ASSERT_TRUE(calcDynamicRange(*m_graph));
    ASSERT_TRUE(calcQuantizationInfo(*m_graph));
    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).scale(0), 80);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).expBias(0), QuantizationData::S_EXP_BIAS_143_DEFAULT);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).scale(1), 0.8);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).expBias(1), QuantizationData::S_EXP_BIAS_143_DEFAULT);
    ASSERT_FLOAT_EQ(t2->getQuantizationParams(syn_type_fp8_143).scale(), 1);
    ASSERT_EQ(t1->getQuantizationParams(syn_type_fp8_152).scale(), 1);
}

TEST_P(ScaleCalcTest, fp8_calc_PC_scale_user_dynamic_range)
{
    setGlobalConfForTest(GCFG_PER_CHANNEL_SCALING, "true");

    using Tensor2BufferDtype = bfloat16;

    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    const unsigned        channelsNum    = 2;
    std::vector<TSize>    sizes          = {channelsNum, 2, 2, 1};
    const TSize           elementsNum    = 8;
    constexpr std::size_t input2BuffSize = sizeof(Tensor2BufferDtype) * elementsNum;

    std::vector<Tensor2BufferDtype> input2 = {0.1, 0.1, -0.7, -0.7, -0.9375, -0.9375, 0.8, 0.8};

    auto inputBuffer2 = std::make_unique<char[]>(input2BuffSize);
    std::memcpy(inputBuffer2.get(), input2.data(), input2BuffSize);

    TensorPtr t1 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t1->setName("t1", true);
    PerChannelDynamicRange perChannelDynamicRange;
    perChannelDynamicRange.numChannels   = channelsNum;
    perChannelDynamicRange.ranges        = {{-15000, 19200}, {-150, 192}};
    perChannelDynamicRange.isSet = true;
    ASSERT_TRUE(t1->setPerChannelDynamicRange(perChannelDynamicRange));

    TensorPtr t2 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t2->setName("t2", true);
    t2->setTensorBuffer(inputBuffer2.get(),
                        input2.size() * sizeof(Tensor2BufferDtype),
                        asSynType<Tensor2BufferDtype>(),
                        false);
    DynamicRange dynamicRange;
    dynamicRange.max   = 0.8;
    dynamicRange.min   = -0.9375;
    dynamicRange.isSet = true;
    ASSERT_TRUE(t2->setDynamicRange(dynamicRange));

    TensorPtr t3 = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    t3->setName("t3", true);

    synGEMMParams params {};
    pNode         gemm = NodeFactory::createNode({t2, t1}, {t3}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm));

    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_EQ((*m_graph).getTensors().size(), 3);
    ASSERT_TRUE(calcDynamicRange(*m_graph));
    ASSERT_TRUE(calcQuantizationInfo(*m_graph));
    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).scale(0), 80);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).expBias(0), QuantizationData::S_EXP_BIAS_143_DEFAULT);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).scale(1), 0.8);
    ASSERT_FLOAT_EQ(t1->getQuantizationParams(syn_type_fp8_143).expBias(1), QuantizationData::S_EXP_BIAS_143_DEFAULT);
    ASSERT_FLOAT_EQ(t2->getQuantizationParams(syn_type_fp8_143).scale(), 1);
    ASSERT_EQ(t1->getQuantizationParams(syn_type_fp8_152).scale(), 1);
}

INSTANTIATE_TEST_SUITE_P(,
                         ScaleCalcTest,
                         ::testing::Values(synDeviceGaudi2, synDeviceGaudi3),
                         GenericGraphTest::GetName());


#include "code_generation/tensor_size_validator.h"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "graph_optimizer_test.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "types.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <memory>

uint64_t quarterOfOneGb = ((uint64_t)1 << 28);
uint64_t oneGb          = ((uint64_t)1 << 30);
uint64_t twoGb          = ((uint64_t)1 << 31);
uint64_t fourGb         = ((uint64_t)1 << 32);

using TensorValidationTestParams = std::tuple<uint64_t,      // chosen size
                                              uint64_t,      // default size
                                              BundleEngine,  // chosen Engine
                                              bool,          // expect assert
                                              unsigned,      // dim
                                              synDataType    // the tensors data type
                                              >;

class SynCodeGenTensorValidation
: public GraphOptimizerTest
, public ::testing::WithParamInterface<TensorValidationTestParams>
{
public:
    uint64_t     m_ChosenSize;
    uint64_t     m_DefaultSize;
    BundleEngine m_Engine;
    bool         m_ExpectFailure;
    unsigned     m_Dim;
    synDataType  m_DataType;

    void readParams()
    {
        m_ChosenSize    = std::get<0>(GetParam());
        m_DefaultSize   = std::get<1>(GetParam());
        m_Engine        = std::get<2>(GetParam());
        m_ExpectFailure = std::get<3>(GetParam());
        m_Dim           = std::get<4>(GetParam());
        m_DataType      = std::get<5>(GetParam());
    }

    TensorPtr
    createTensor(SizeVector shape, synDataType dataType, const char* name = nullptr, const TStride* strides = nullptr)
    {
        auto tensor = std::make_shared<Tensor>(shape.size(), shape.data(), dataType, nullptr, strides);
        if (name)
        {
            tensor->setName(name);
        }
        return tensor;
    }
};

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_MME_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(fourGb),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_MME),
                                            ::testing::Values(true),
                                            ::testing::Values(1, 2, 3, 4, 5),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_MME_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(fourGb - 1),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_MME),
                                            ::testing::Values(false),
                                            ::testing::Values(1, 2, 3, 4, 5),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_MME_FLOAT32_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(twoGb),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_MME),
                                            ::testing::Values(true),
                                            ::testing::Values(1, 2, 3, 4, 5),
                                            ::testing::Values(syn_type_float)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_MME_FLOAT32_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(twoGb - 1),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_MME),
                                            ::testing::Values(false),
                                            ::testing::Values(1, 2, 3, 4, 5),
                                            ::testing::Values(syn_type_float)));

INSTANTIATE_TEST_SUITE_P(DISABLED_GAUDI2_GAUDI3_DMA_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(twoGb),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_DMA),
                                            ::testing::Values(true),
                                            ::testing::Values(1, 2, 3, 4),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_DMA_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(twoGb - 1),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_DMA),
                                            ::testing::Values(false),
                                            ::testing::Values(1, 2, 3, 4),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_DIM0_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values((((uint64_t)1) << 42)),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(false),
                                            ::testing::Values(1),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_DIM0_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values((((uint64_t)1) << 42) + 1),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(true),
                                            ::testing::Values(1),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_DIM1_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(oneGb + 1),
                                            ::testing::Values(2),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(true),
                                            ::testing::Values(2),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_DIM1_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(oneGb),
                                            ::testing::Values(2),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(false),
                                            ::testing::Values(2),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_DIM_EXTENDED_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values((((uint64_t)1) << 46) + 1),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(true),
                                            ::testing::Values(2, 3, 4, 5),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_DIM_EXTENDED_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values((((uint64_t)1) << 46)),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(false),
                                            ::testing::Values(2, 3, 4, 5),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(twoGb + 1),
                                            ::testing::Values(2),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(true),
                                            ::testing::Values(2, 3, 4),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(twoGb),
                                            ::testing::Values(1),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(false),
                                            ::testing::Values(2, 3, 4),
                                            ::testing::Values(syn_type_bf16)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_INT64_NO_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(quarterOfOneGb - 1),
                                            ::testing::Values(2),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(false),
                                            ::testing::Values(1, 2, 3 /*, 4*/),
                                            ::testing::Values(syn_type_int64)));

INSTANTIATE_TEST_SUITE_P(GAUDI2_GAUDI3_TPC_INT64_FAIL,
                         SynCodeGenTensorValidation,
                         ::testing::Combine(::testing::Values(quarterOfOneGb + 1),
                                            ::testing::Values(2),
                                            ::testing::Values(ENGINE_TPC),
                                            ::testing::Values(true),
                                            ::testing::Values(3, 4),
                                            ::testing::Values(syn_type_int64)));

// TODO adding this test for Gaudi/Greco, transpose for DMA

TEST_P(SynCodeGenTensorValidation, codegen_tensor_validation)
{
    readParams();
    HabanaGraph* graph;
    Gaudi2Graph  gaudi2Graph;
    graph                = &gaudi2Graph;
    uint64_t defaultSize = m_DefaultSize;
    uint64_t         nInput       = m_Dim == 1 ? m_ChosenSize : defaultSize;
    uint64_t         wInput       = m_Dim == 2 ? m_ChosenSize : defaultSize;
    uint64_t         hInput       = m_Dim == 3 ? m_ChosenSize : defaultSize;
    uint64_t         dInput       = m_Dim == 4 ? m_ChosenSize : defaultSize;
    uint64_t         batch        = m_Dim == 5 ? m_ChosenSize : defaultSize;
    uint64_t         nOutput      = defaultSize;
    uint64_t         kernelWidth  = defaultSize;
    uint64_t         kernelHeight = defaultSize;
    uint64_t         kernelDepth  = defaultSize;
    const SizeVector xSize        = {nInput, wInput, hInput, dInput, batch};
    const SizeVector wSize        = {nOutput, nInput, kernelWidth, kernelHeight, kernelDepth};
    uint64_t         wOutput      = (wInput - kernelWidth + 3) / 2;
    uint64_t         hOutput      = (hInput - kernelHeight + 3) / 2;
    uint64_t         dOutput      = (dInput - kernelDepth + 3) / 2;

    const SizeVector ySize = {nOutput, wOutput, hOutput, dOutput, batch};
    auto    x = createTensor(xSize, m_DataType, "inputA");
    auto    w = createTensor(wSize, m_DataType, "inputB");
    auto    y = createTensor(ySize, m_DataType, "output");
    NodePtr node;
    if (m_Engine == ENGINE_MME)
    {
        synConvolution3DParams convParams;
        convParams.kernel[CONV_KERNEL_WIDTH]  = kernelWidth;
        convParams.kernel[CONV_KERNEL_HEIGHT] = kernelHeight;
        convParams.kernel[CONV_KERNEL_DEPTH]  = kernelDepth;
        node = NodeFactory::createNode({x, w}, {y}, &convParams, NodeFactory::convolutionNodeTypeName, "CONV");
    }
    else if (m_Engine == ENGINE_DMA)
    {
        auto xcopy = createTensor(xSize, m_DataType, "output");
        node       = NodeFactory::createNode({x}, {xcopy}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "memcpy1");
    }
    else if (m_Engine == ENGINE_TPC)
    {
        TensorVector in, out;
        if (m_Dim != 2)  // special case, to bypase the non degenerate dims and validate this specific tensor
        {
            in  = {createTensor(xSize, m_DataType, "input")};
            out = {createTensor(xSize, m_DataType, "output")};
        }
        else
        {
            auto reducedDimTensor = {nInput, wInput, hInput};
            in                    = {createTensor(reducedDimTensor, m_DataType, "input")};
            out                   = {createTensor(reducedDimTensor, m_DataType, "output")};
        }
        node = NodeFactory::createGenericTPCNode(in, out, nullptr, "relu_fwd_f32", "reluIn");
    }

    GraphEditor::addNode(*graph, node);
    if (m_ExpectFailure)
    {
        ASSERT_FALSE(TensorSizeValidator(*graph).validateTensors(*graph));
    }
    else
    {
        ASSERT_TRUE(TensorSizeValidator(*graph).validateTensors(*graph));
    }
}

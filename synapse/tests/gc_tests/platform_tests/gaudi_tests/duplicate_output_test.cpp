#include "gtest/gtest.h"
#include <memory>
#include "../gaudi_tests/gc_gaudi_test_infra.h"
#include "data_type_utils.h"
#include "defs.h"
#include "node_annotation.h"
#include "scoped_configuration_change.h"
#include "syn_data_type_type_conversions.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "test_utils.h"
#include "types.h"
#include <syn_singleton.hpp>
#include "node_factory.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "compilation_hal_reader.h"

typedef enum
{
    OUT_0_IDX = 1,
    OUT_1_IDX = 0,
} mmeOutputsIdx;

class SynGaudi2DuplicateOutput
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<int /* stride */,
                                                int /* ifmC */,
                                                int /* ifm spatial size*/,
                                                int /* ifmB */,
                                                int /* ofmK */,
                                                bool /* m_out0InHbm */,
                                                bool /* m_out1InHbm */,
                                                ERepefenceOp /* op */>>
{
public:
    template<typename DType>
    void                 runMmeTest();
    void                 initTestParams();
    bool                 m_out1InHbm;
    bool                 m_out0InHbm;
    synConvolutionParams m_convParams;
    TestSizes            m_xSize, m_wSize, m_ySize;
    ERepefenceOp         m_op;

private:
    void addExtraOutputAndSetInSram();
};

void SynGaudi2DuplicateOutput::addExtraOutputAndSetInSram()
{
    HabanaGraph* pGraph = synSingleton::getInstanceInternal()->getGraph(m_graphs[0].graphHandle);
    NodeSet      nodes  = pGraph->getNodes();
    ASSERT_EQ(nodes.size(), 2) << "The graph should contain only 2 nodes at this point: one mme, one tpc.";
    TensorPtr  out1;
    BundleInfo info(0, BundleType::MME);
    for (NodePtr node : nodes)
    {
        std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
        if (mmeNode)
        {
            // Add an extra output for the mme node:
            GraphEditor::removeNode(*pGraph, node);
            auto out0 = node->getOutput(0);
            out1      = out0->clone();
            out1->setName("oTensor");
            node->addOutput(out1);

            // Set each output in sram, if necessary:
            if (!m_out0InHbm)
            {
                out0->setTensorInSram();
            }
            if (!m_out1InHbm)
            {
                out1->setTensorInSram();
            }
            node->getNodeAnnotation().bundleInfo = info;
            GraphEditor::addNode(*pGraph, node);
        }
    }
    for (NodePtr node : nodes)
    {
        // Replace the initial dummy input with the new oTensor
        if (node->getNodeType() == Node::TYPE_USER)
        {
            GraphEditor::removeNode(*pGraph, node);
            node->replaceInput(0, out1);
            info.operationIndex++;
            node->getNodeAnnotation().bundleInfo = info;
            GraphEditor::addNode(*pGraph, node);
        }
    }
}

void SynGaudi2DuplicateOutput::initTestParams()
{
    m_convParams.kH         = 3;
    m_convParams.kW         = 3;
    m_convParams.dW         = std::get<0>(GetParam());
    m_convParams.dH         = std::get<0>(GetParam());
    unsigned ifmC           = std::get<1>(GetParam());
    unsigned ifmSpatialSize = std::get<2>(GetParam());
    unsigned batch          = std::get<3>(GetParam());
    unsigned ofmK           = std::get<4>(GetParam());
    m_out0InHbm             = std::get<5>(GetParam());
    m_out1InHbm             = std::get<6>(GetParam());
    m_op                    = std::get<7>(GetParam());

    m_xSize = {ifmC, ifmSpatialSize, ifmSpatialSize, batch, 1};
    m_wSize = {ofmK, ifmC, m_convParams.kW, m_convParams.kH, 1};
    m_ySize = {ofmK,
               convOutputDimSize(m_xSize[1],
                                 m_convParams.kW,
                                 m_convParams.dW,
                                 m_convParams.padL + m_convParams.padR,
                                 m_convParams.dilW),
               convOutputDimSize(m_xSize[2],
                                 m_convParams.kH,
                                 m_convParams.dH,
                                 m_convParams.padT + m_convParams.padB,
                                 m_convParams.dilH),
               batch,
               1};
}

template<typename DType>
void SynGaudi2DuplicateOutput::runMmeTest()
{
    // Disable all bundling algorithms, as the test does the bundling instead of the compiler.
    ScopedConfigurationChange disableSramSlicer("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    ScopedConfigurationChange diablePipelineManagement("ENABLE_PIPELINE_MANAGEMENT", "false");
    ScopedConfigurationChange disableCguidExtractor("COMPLEX_GUID_EXTRACTOR_MODE", "0");

    initTestParams();
    CompilationHalReader::setHalReader(Gaudi2HalReader::instance());
    // std::cout << m_out0InHbm << ", " << m_out1InHbm << std::endl;
    synDataType   dtype         = asSynType<DType>();
    TensorUsage   xUsage        = m_op == REFERENCE_OP_DEDX ? OUTPUT_TENSOR : INPUT_TENSOR;
    TensorUsage   wUsage        = m_op == REFERENCE_OP_DEDW ? OUTPUT_TENSOR : INPUT_TENSOR;
    TensorUsage   yUsage        = m_op == REFERENCE_OP_FWD ? OUTPUT_TENSOR : INPUT_TENSOR;
    unsigned      xTensorIndex  = createTensors(1,
                                          xUsage,
                                          xUsage == OUTPUT_TENSOR ? m_out0InHbm : true /*isPersistent*/,
                                          "xTensor",
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          (unsigned*)m_xSize.data(),
                                          DEFAULT_SIZES,
                                          dtype)[0];
    unsigned      wTensorIndex  = createTensors(1,
                                          wUsage,
                                          wUsage == OUTPUT_TENSOR ? m_out0InHbm : true /*isPersistent*/,
                                          "wTensor",
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          (unsigned*)m_wSize.data(),
                                          DEFAULT_SIZES,
                                          dtype)[0];
    unsigned      yTensorIndex  = createTensors(1,
                                          yUsage,
                                          yUsage == OUTPUT_TENSOR ? m_out0InHbm : true /*isPersistent*/,
                                          "yTensor",
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          (unsigned*)m_ySize.data(),
                                          DEFAULT_SIZES,
                                          dtype)[0];
    std::string   guid          = "spatial_convolution";
    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};
    TestSizes     outDimSizes   = m_ySize;
    if (m_op == REFERENCE_OP_DEDW)
    {
        guid             = "dedw";
        inputIndices[0]  = yTensorIndex;
        inputIndices[1]  = xTensorIndex;
        outputIndices[0] = wTensorIndex;
        outDimSizes      = m_wSize;
    }
    else if (m_op == REFERENCE_OP_DEDX)
    {
        guid             = "dedx";
        inputIndices[0]  = yTensorIndex;
        inputIndices[1]  = wTensorIndex;
        outputIndices[0] = xTensorIndex;
        outDimSizes      = m_xSize;
    }

    addNodeToGraph(guid.c_str(), inputIndices, outputIndices, (void*)&m_convParams, sizeof(m_convParams));

    std::map<mmeOutputsIdx, unsigned> finalOutputIndices;

    // "Copy" (using relu to include TPC in the test) oTensor from workspace to a persistent tensor, for
    // validation and stress purposes
    unsigned dupOutTensorIndex = createTensors(1,
                                               OUTPUT_TENSOR,
                                               true /*isPersistent*/,
                                               "reluOut1Data",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               (unsigned*)outDimSizes.data(),
                                               DEFAULT_SIZES,
                                               dtype)[0];
    const std::string guidStr           = fmt::format("relu_fwd_{}", getDtypeSuffixFromSynDataType(dtype));
    addNodeToGraph(guidStr.c_str(), {outputIndices[0]}, {dupOutTensorIndex});
    finalOutputIndices[OUT_1_IDX] = dupOutTensorIndex;

    addExtraOutputAndSetInSram();

    // Add memcpy to Y tensor if needed:
    if (!m_out0InHbm)
    {
        unsigned dupOutTensorIndex = createTensors(1,
                                                   OUTPUT_TENSOR,
                                                   true /*isPersistent*/,
                                                   "memcpyOutData",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   (unsigned*)outDimSizes.data(),
                                                   DEFAULT_SIZES,
                                                   dtype)[0];

        addNodeToGraph("memcpy", {outputIndices[0]}, {dupOutTensorIndex});
        finalOutputIndices[OUT_0_IDX] = dupOutTensorIndex;
    }
    else
    {
        finalOutputIndices[OUT_0_IDX] = outputIndices[0];
    }

    compileAndRun();

    unsigned firstOutTensorIndex = m_out0InHbm ? finalOutputIndices[OUT_0_IDX] : finalOutputIndices[OUT_1_IDX];
    synTensorDescriptor xDesc    = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc    = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc    = m_tensorDescs[yTensorIndex];
    void*               xData    = m_hostBuffers[xTensorIndex];
    void*               wData    = m_hostBuffers[wTensorIndex];
    void*               yData    = m_hostBuffers[yTensorIndex];
    DType*              hbmData  = castHostOutBuffer<DType>(firstOutTensorIndex);

    CoordArray wrongIdx = {0};
    bool       ret      = checkMmeOp(xDesc,
                          m_op == REFERENCE_OP_DEDX ? (char*)hbmData : (char*)xData,
                          wDesc,
                          m_op == REFERENCE_OP_DEDW ? (char*)hbmData : (char*)wData,
                          yDesc,
                          m_op == REFERENCE_OP_FWD ? (char*)hbmData : (char*)yData,
                          m_convParams,
                          m_op,
                          wrongIdx,
                          m_deviceType);
    ASSERT_EQ(ret, true) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',');

    // Check extra output tensor:
    dupOutTensorIndex   = m_out0InHbm ? finalOutputIndices[OUT_1_IDX] : finalOutputIndices[OUT_0_IDX];
    auto*    dupOutData = castHostOutBuffer<DType>(dupOutTensorIndex);
    unsigned length     = multiplyElements(outDimSizes.begin(), outDimSizes.end());
    for (int i = 0; i < length; i++)
    {
        ASSERT_FLOAT_EQ(hbmData[i], dupOutData[i]);
    }
}
INSTANTIATE_TEST_SUITE_P(
    duplicate_output_single,
    SynGaudi2DuplicateOutput,
    ::testing::Combine(::testing::ValuesIn({1}),            // stride
                       ::testing::ValuesIn({16}),           // ifmC
                       ::testing::ValuesIn({64}),           // ifm spatial size
                       ::testing::ValuesIn({4}),            // ifmB
                       ::testing::ValuesIn({16}),           // ofmK
                       ::testing::ValuesIn({true, false}),  // m_out0InHbm
                       ::testing::ValuesIn({true, false}),  // m_out1InHbm
                       ::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW})));

INSTANTIATE_TEST_SUITE_P(
    duplicate_output_full_DAILY,
    SynGaudi2DuplicateOutput,
    ::testing::Combine(::testing::ValuesIn({2, 3}),         // stride
                       ::testing::ValuesIn({8, 32}),        // ifmC
                       ::testing::ValuesIn({12, 45}),       // ifm spatial size
                       ::testing::ValuesIn({1, 6}),         // ifmB
                       ::testing::ValuesIn({8, 23}),        // ofmK
                       ::testing::ValuesIn({true, false}),  // m_out0InHbm
                       ::testing::ValuesIn({true, false}),  // m_out1InHbm
                       ::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW})));

TEST_P_GC(SynGaudi2DuplicateOutput, conv_tests_double_output_f32_L2, {synDeviceGaudi2})
{
    runMmeTest<float>();
}

TEST_P_GC(SynGaudi2DuplicateOutput, DISABLED_conv_tests_double_output_bf16_L2, {synDeviceGaudi2})
{
    runMmeTest<bfloat16>();
}

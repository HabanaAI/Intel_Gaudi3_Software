#include "access_pattern_generator.h"
#include "compilation_hal_reader.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"

using namespace gc::access_pattern;

using TensorDimToIndexSpaceDimMapping = TPCCustomIndexSpaceMappingNode::TensorDimToIndexSpaceDimMapping;
struct GlueCodeAccessPatternDimMappingTestParams
{
    unsigned                                     tensorRank;
    unsigned                                     nodeResolutionRank;
    std::vector<TensorDimToIndexSpaceDimMapping> dimMappingForInputs;
    std::vector<TensorDimToIndexSpaceDimMapping> dimMappingForOutputs;
};

class GlueCodeAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<GlueCodeAccessPatternDimMappingTestParams>
{
public:
    NodePtr createTpcNode()
    {
        TPCCustomIndexSpaceMappingNode::Params params;
        params.tensorRank           = m_tensorRank;
        params.nodeResolutionRank   = m_nodeResolutionRank;
        params.dimMappingForInputs  = m_dimMappingForInputs;
        params.dimMappingForOutputs = m_dimMappingForOutputs;
        return TPCCustomIndexSpaceMappingNode::create(params);
    }

    void validateTensorIndexSpaceMapping(const NodeAccessPatternPtr             nodeAp,
                                         const TensorPtr&                       tensor,
                                         const TensorDimToIndexSpaceDimMapping& dimMapping)
    {
        for (Dim tensorDim = 0; tensorDim < m_tensorRank; tensorDim++)
        {
            Dim indexSpaceDim = nodeAp->getIndexSpaceDim(tensor, tensorDim);
            ASSERT_LT(indexSpaceDim, nodeAp->getNodeResolution().size());
            if (dimMapping.empty() || dimMapping[tensorDim].second)
            {
                // For all-required tensor dims - the last node resolution dim should be returned
                ASSERT_EQ(indexSpaceDim, nodeAp->getNodeResolution().size() - 1);
            }
            else
            {
                ASSERT_EQ(indexSpaceDim, dimMapping[tensorDim].first);
            }
        }
    }

    const unsigned                                     m_tensorRank           = GetParam().tensorRank;
    const unsigned                                     m_nodeResolutionRank   = GetParam().nodeResolutionRank;
    const std::vector<TensorDimToIndexSpaceDimMapping> m_dimMappingForInputs  = GetParam().dimMappingForInputs;
    const std::vector<TensorDimToIndexSpaceDimMapping> m_dimMappingForOutputs = GetParam().dimMappingForOutputs;
};

TEST_P(GlueCodeAccessPatternDimMappingTest, glue_code_access_pattern_dims_mapping_test)
{
    NodePtr node = createTpcNode();

    const auto& nodeAp = node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    // A new dummy dim should be added at the end of the node resolution -
    // will be used to map the all-required tensor dims.
    ASSERT_EQ(nodeAp->getNodeResolution().size(), m_nodeResolutionRank + 1);
    ASSERT_EQ(nodeAp->getNodeResolution().back(), 1);

    ASSERT_EQ(node->getNumInputs(), m_dimMappingForInputs.size());

    for (auto i = 0; i < node->getNumInputs(); i++)
    {
        const auto& input = node->getInput(i);
        ASSERT_TRUE(input);
        validateTensorIndexSpaceMapping(nodeAp, input, m_dimMappingForInputs[i]);
    }

    ASSERT_EQ(node->getNumOutputs(), m_dimMappingForOutputs.size());

    for (auto i = 0; i < node->getNumOutputs(); i++)
    {
        const auto& output = node->getOutput(i);
        ASSERT_TRUE(output);
        validateTensorIndexSpaceMapping(nodeAp, output, m_dimMappingForOutputs[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    glue_code_access_pattern_dims_mapping,
    GlueCodeAccessPatternDimMappingTest,
    ::testing::Values(GlueCodeAccessPatternDimMappingTestParams {2,  // tensorsRank
                                                                 4,  // nodeResolutionRank
                                                                 // dimMappingForInputs:
                                                                 {{{0, false}, {1, false}}, {{2, false}, {3, false}}},
                                                                 // dimMappingForOutputs:
                                                                 {{{3, false}, {2, false}}, {{1, false}, {0, false}}}},
                      GlueCodeAccessPatternDimMappingTestParams {2,  // tensorsRank
                                                                 4,  // nodeResolutionRank
                                                                 // dimMappingForInputs:
                                                                 {{{0, true}, {1, true}}, {{2, true}, {3, true}}},
                                                                 // dimMappingForOutputs:
                                                                 {{{3, true}, {2, true}}, {{1, true}, {0, true}}}},
                      GlueCodeAccessPatternDimMappingTestParams {3,  // tensorsRank
                                                                 2,  // nodeResolutionRank
                                                                 // dimMappingForInputs:
                                                                 {{}, {}, {}, {}, {}, {}},
                                                                 // dimMappingForOutputs:
                                                                 {{}, {}}},
                      GlueCodeAccessPatternDimMappingTestParams {
                          5,  // tensorsRank
                          3,  // nodeResolutionRank
                          // dimMappingForInputs:
                          {{{2, false}, {1, true}, {0, false}, {0, false}, {2, false}},
                           {{1, false}, {1, false}, {1, false}, {1, false}, {1, true}},
                           {},
                           {{0, true}, {0, false}, {1, false}, {1, false}, {2, false}},
                           {}},
                          // dimMappingForOutputs:
                          {{{2, false}, {1, true}, {0, false}, {0, false}, {2, false}},
                           {{1, false}, {1, false}, {1, false}, {1, false}, {1, true}},
                           {},
                           {{0, true}, {0, false}, {1, false}, {1, false}, {2, false}},
                           {}}},
                      GlueCodeAccessPatternDimMappingTestParams {3,  // tensorsRank
                                                                 9,  // nodeResolutionRank
                                                                 // dimMappingForInputs:
                                                                 {{},
                                                                  {{3, false}, {4, false}, {3, false}},
                                                                  {{2, true}, {2, true}, {2, true}},
                                                                  {{5, true}, {6, false}, {1, false}},
                                                                  {},
                                                                  {{5, true}, {7, false}, {1, false}}},
                                                                 // dimMappingForOutputs:
                                                                 {{{3, false}, {4, true}, {3, false}}}}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GemmAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::tuple<bool,      // transpose A
                                                  bool,      // transpose B
                                                  unsigned,  // num batch dims
                                                  bool,      // is masked BGEMM
                                                  bool,      // is broadcast BGEMM
                                                  bool,      // true - operand A is broadcasted, false - operand B is
                                                             // broadcasted
                                                  bool       // is broadcast dim present
                                                  >>
{
public:
    GemmAccessPatternDimMappingTest()
    {
        std::tie(m_transposeA,
                 m_transposeB,
                 m_numBatchDims,
                 m_isMaskedBgemm,
                 m_isBroadcastBgemm,
                 m_isOpABroadcasted,
                 m_isBroadcastDimPresent) = GetParam();
    }

    TensorPtr
    createTensor(std::vector<TSize> shape, bool transposed, bool isMask, const std::vector<TSize>& batchDims) const
    {
        if (transposed)
        {
            std::swap(shape[0], shape[1]);
        }
        if (!batchDims.empty())
        {
            if (isMask)  // 2 batch dimensions
            {
                shape.push_back(1);               // Dim 2 of the mask is 1
                shape.push_back(batchDims[1]);    // Dim 3 of the mask is the external batch dim
            }
            else
            {
                shape.insert(shape.end(), batchDims.begin(), batchDims.end());
            }
        }
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
    }

    std::vector<TSize> initBatchDims(TSize batchSize, bool isOperandBroadcasted) const
    {
        std::vector<TSize> batchDims;
        if (isOperandBroadcasted)
        {
            if (m_isBroadcastDimPresent)
            {
                batchDims.resize(m_numBatchDims, 1);
            }
        }
        else
        {
            batchDims.resize(m_numBatchDims, batchSize);
        }
        return batchDims;
    }

    bool     m_transposeA;
    bool     m_transposeB;
    unsigned m_numBatchDims;
    bool     m_isMaskedBgemm;
    bool     m_isBroadcastBgemm;
    bool     m_isOpABroadcasted;
    bool     m_isBroadcastDimPresent;

    Gaudi2Graph                      m_graph;
    const CompilationHalReaderSetter m_chrSetter {&m_graph};
};

TEST_P(GemmAccessPatternDimMappingTest, gemm_access_pattern_dims_mapping_test)
{
    const TSize heightA        = 128;
    const TSize commonDim      = 128;
    const TSize widthB         = 128;
    const TSize masksCommonDim = 128;
    const TSize batchSize      = 8;

    std::string   guid = (m_numBatchDims == 0) ? NodeFactory::gemmNodeTypeName : NodeFactory::batchGemmNodeTypeName;
    synGEMMParams params(m_transposeA, m_transposeB);

    TensorVector inputs;
    inputs.push_back(createTensor({commonDim, heightA},
                                  params.transpose_a,
                                  false,
                                  initBatchDims(batchSize, m_isBroadcastBgemm && m_isOpABroadcasted)));
    inputs.push_back(createTensor({widthB, commonDim},
                                  params.transpose_b,
                                  false,
                                  initBatchDims(batchSize, m_isBroadcastBgemm && !m_isOpABroadcasted)));
    if (m_isMaskedBgemm)
    {
        // Validate the test params are correct
        ASSERT_EQ(m_numBatchDims, 2);
        ASSERT_FALSE(m_isBroadcastBgemm);

        guid = NodeFactory::maskedBatchGemmNodeTypeName;
        inputs.push_back(
            createTensor({masksCommonDim, heightA}, params.transpose_a, true, initBatchDims(batchSize, false)));
        inputs.push_back(
            createTensor({widthB, masksCommonDim}, params.transpose_b, true, initBatchDims(batchSize, false)));
    }
    TensorPtr output = createTensor({widthB, heightA}, false, false, initBatchDims(batchSize, false));

    NodePtr node = NodeFactory::createNode(inputs, {output}, &params, guid.c_str(), "GEMM");

    const auto& nodeAp = node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    for (const auto& tensor : node->getOperands())
    {
        for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
        {
            ASSERT_LT(nodeAp->getIndexSpaceDim(tensor, tensorDim), nodeAp->getNodeResolution().size());
        }
    }

    const auto& opA = node->getInput(0);
    const auto& opB = node->getInput(1);
    const auto& out = node->getOutput(0);

    ASSERT_EQ(nodeAp->getIndexSpaceDim(opA, m_transposeA ? 1 : 0), nodeAp->getIndexSpaceDim(opB, m_transposeB ? 0 : 1));
    ASSERT_EQ(nodeAp->getIndexSpaceDim(opA, m_transposeA ? 0 : 1), nodeAp->getIndexSpaceDim(out, 1));
    ASSERT_EQ(nodeAp->getIndexSpaceDim(opB, m_transposeB ? 1 : 0), nodeAp->getIndexSpaceDim(out, 0));

    for (auto i = 0; i < m_numBatchDims; i++)
    {
        const auto batchDim = DIM_GEMM_BATCH + i;
        if (!m_isBroadcastBgemm)  // No broadcast
        {
            ASSERT_EQ(nodeAp->getIndexSpaceDim(opA, batchDim), nodeAp->getIndexSpaceDim(opB, batchDim));
            ASSERT_EQ(nodeAp->getIndexSpaceDim(opA, batchDim), nodeAp->getIndexSpaceDim(out, batchDim));
        }
        else if (m_isOpABroadcasted)  // Broadcast BGEMM - operand A is broadcasted
        {
            ASSERT_EQ(nodeAp->getIndexSpaceDim(opB, batchDim), nodeAp->getIndexSpaceDim(out, batchDim));
            if (opA->getDim() > batchDim)  // broadcast dim exists
            {
                ASSERT_EQ(opA->getSizeInElements(batchDim), 1);
                // Broadcast dims should be mapped to a different index-space dim with size of 1
                Dim broadcastIndexSpaceDim = nodeAp->getIndexSpaceDim(opA, batchDim);
                ASSERT_NE(broadcastIndexSpaceDim, nodeAp->getIndexSpaceDim(out, batchDim));
                ASSERT_EQ(nodeAp->getNodeResolution()[broadcastIndexSpaceDim], 1);
            }
        }
        else  // Broadcast BGEMM - operand B is broadcasted
        {
            ASSERT_EQ(nodeAp->getIndexSpaceDim(opA, batchDim), nodeAp->getIndexSpaceDim(out, batchDim));
            if (opB->getDim() > batchDim)  // broadcast dim exists
            {
                ASSERT_EQ(opB->getSizeInElements(batchDim), 1);
                // Broadcast dims should be mapped to a different index-space dim with size of 1
                Dim broadcastIndexSpaceDim = nodeAp->getIndexSpaceDim(opB, batchDim);
                ASSERT_NE(broadcastIndexSpaceDim, nodeAp->getIndexSpaceDim(out, batchDim));
                ASSERT_EQ(nodeAp->getNodeResolution()[broadcastIndexSpaceDim], 1);
            }
        }
    }

    if (m_isMaskedBgemm)
    {
        const auto& maskA = node->getInput(TENSOR_AUX_BGEMM_MASK_A);
        const auto& maskB = node->getInput(TENSOR_AUX_BGEMM_MASK_B);

        ASSERT_EQ(nodeAp->getIndexSpaceDim(maskA, m_transposeA ? 1 : 0),
                  nodeAp->getIndexSpaceDim(maskB, m_transposeB ? 0 : 1));
        ASSERT_EQ(nodeAp->getIndexSpaceDim(maskA, m_transposeA ? 0 : 1), nodeAp->getIndexSpaceDim(out, 1));
        ASSERT_EQ(nodeAp->getIndexSpaceDim(maskB, m_transposeB ? 1 : 0), nodeAp->getIndexSpaceDim(out, 0));

        // The first (internal) batch dim for the masks is num identical masks, the rest are the same as the output
        ASSERT_EQ(nodeAp->getIndexSpaceDim(maskA, 2), nodeAp->getIndexSpaceDim(maskB, 2));
        ASSERT_EQ(nodeAp->getIndexSpaceDim(maskA, 3), nodeAp->getIndexSpaceDim(maskB, 3));
        ASSERT_EQ(nodeAp->getIndexSpaceDim(maskA, 3), nodeAp->getIndexSpaceDim(out, 3));
    }
}

INSTANTIATE_TEST_SUITE_P(
    gemm_access_pattern_dims_mapping,
    GemmAccessPatternDimMappingTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false, true),  // transpose B
                       ::testing::Values(0),            // num batch dims
                       ::testing::Values(false),        // is masked BGEMM
                       ::testing::Values(false),        // is broadcast BGEMM
                       ::testing::Values(false),    // true - operand A is broadcated, false - operand B is broadcated
                       ::testing::Values(false)));  // is broadcast dim present

INSTANTIATE_TEST_SUITE_P(
    bgemm_access_pattern_dims_mapping,
    GemmAccessPatternDimMappingTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false, true),  // transpose B
                       ::testing::Values(1, 2, 3),      // num batch dims
                       ::testing::Values(false),        // is masked BGEMM
                       ::testing::Values(false),        // is broadcast BGEMM
                       ::testing::Values(false),    // true - operand A is broadcated, false - operand B is broadcated
                       ::testing::Values(false)));  // is broadcast dim present

INSTANTIATE_TEST_SUITE_P(
    masked_bgemm_access_pattern_dims_mapping,
    GemmAccessPatternDimMappingTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false, true),  // transpose B
                       ::testing::Values(2),            // num batch dims
                       ::testing::Values(true),         // is masked BGEMM
                       ::testing::Values(false),        // is broadcast BGEMM
                       ::testing::Values(false),    // true - operand A is broadcated, false - operand B is broadcated
                       ::testing::Values(false)));  // is broadcast dim present

INSTANTIATE_TEST_SUITE_P(
    broadcast_bgemm_access_pattern_dims_mapping,
    GemmAccessPatternDimMappingTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false, true),  // transpose B
                       ::testing::Values(1, 2, 3),      // num batch dims
                       ::testing::Values(false),        // is masked BGEMM
                       ::testing::Values(true),         // is broadcast BGEMM
                       ::testing::Values(true,
                                         false),  // true - operand A is broadcated, false - operand B is broadcated
                       ::testing::Values(true, false)));  // is broadcast dim present

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvAccessPatternDimMappingTestParams
{
    std::string guid;
};

class ConvAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<ConvAccessPatternDimMappingTestParams>
{
protected:
    // Creates a convolution node with default params.
    void createNode()
    {
        SizeVector xSize, wSize, ySize;
        if (is3DConv())
        {
            xSize = {m_ifmCSize, m_ifmSpatialSize, m_ifmSpatialSize, m_ifmSpatialSize, m_batchSize};
            wSize = {m_ofmKSize, m_ifmCSize, 1, 1, 1};
            ySize = {m_ofmKSize,
                     convOutputDimSize(xSize[1], 1, 1, 0, 1),
                     convOutputDimSize(xSize[2], 1, 1, 0, 1),
                     convOutputDimSize(xSize[3], 1, 1, 0, 1),
                     m_batchSize};
        }
        else
        {
            xSize = {m_ifmCSize, m_ifmSpatialSize, m_ifmSpatialSize, m_batchSize};
            wSize = {m_ofmKSize, m_ifmCSize, 1, 1};
            ySize = {m_ofmKSize,
                     convOutputDimSize(xSize[1], 1, 1, 0, 1),
                     convOutputDimSize(xSize[2], 1, 1, 0, 1),
                     m_batchSize};
        }
        m_xOperand = std::make_shared<Tensor>(xSize.size(), xSize.data(), syn_type_float);
        m_wOperand = std::make_shared<Tensor>(wSize.size(), wSize.data(), syn_type_float);
        m_yOperand = std::make_shared<Tensor>(ySize.size(), ySize.data(), syn_type_float);

        TensorVector inputs;
        TensorPtr    output;
        if ((m_guid == NodeFactory::convolutionNodeTypeName) || (m_guid == NodeFactory::convolution3DNodeTypeName))
        {
            inputs.push_back(m_xOperand);
            inputs.push_back(m_wOperand);
            output = m_yOperand;
        }
        else if ((m_guid == NodeFactory::deDwNodeTypeName) || (m_guid == NodeFactory::deDw3DNodeTypeName))
        {
            inputs.push_back(m_yOperand);
            inputs.push_back(m_xOperand);
            output = m_wOperand;
        }
        else  // ((m_guid == NodeFactory::deDxNodeTypeName) || (m_guid == NodeFactory::deDx3DNodeTypeName))
        {
            inputs.push_back(m_yOperand);
            inputs.push_back(m_wOperand);
            output = m_xOperand;
        }

        if (is3DConv())
        {
            synConvolution3DParams params3D;
            m_node = NodeFactory::createNode(inputs, {output}, &params3D, m_guid.c_str(), "3DCONV");
        }
        else
        {
            synConvolutionParams params;
            m_node = NodeFactory::createNode(inputs, {output}, &params, m_guid.c_str(), "CONV");
        }
        ASSERT_TRUE(m_node);
    }

    bool is3DConv() const
    {
        return (m_guid == NodeFactory::convolution3DNodeTypeName) || (m_guid == NodeFactory::deDw3DNodeTypeName) ||
               (m_guid == NodeFactory::deDx3DNodeTypeName);
    }

    const std::string m_guid           = GetParam().guid;
    const TSize       m_batchSize      = 16;
    const TSize       m_ifmCSize       = 128;
    const TSize       m_ofmKSize       = 512;
    const TSize       m_ifmSpatialSize = 32;

    NodePtr   m_node;
    TensorPtr m_xOperand;
    TensorPtr m_wOperand;
    TensorPtr m_yOperand;

    Gaudi2Graph                      m_graph;
    const CompilationHalReaderSetter m_chrSetter {&m_graph};
};

TEST_P(ConvAccessPatternDimMappingTest, conv_access_pattern_dims_mapping_test)
{
    createNode();

    const auto& nodeAp = m_node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    for (const auto& tensor : m_node->getOperands())
    {
        for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
        {
            ASSERT_LT(nodeAp->getIndexSpaceDim(tensor, tensorDim), nodeAp->getNodeResolution().size());
        }
    }

    ASSERT_EQ(nodeAp->getIndexSpaceDim(m_xOperand, is3DConv() ? DIM_B_FOR_5D_TENSOR : DIM_B),
              nodeAp->getIndexSpaceDim(m_yOperand, is3DConv() ? DIM_B_FOR_5D_TENSOR : DIM_B));
    if (is3DConv())
    {
        ASSERT_EQ(nodeAp->getIndexSpaceDim(m_xOperand, DIM_D_FOR_5D_TENSOR),
                  nodeAp->getIndexSpaceDim(m_yOperand, DIM_D_FOR_5D_TENSOR));
    }
    ASSERT_EQ(nodeAp->getIndexSpaceDim(m_xOperand, DIM_H), nodeAp->getIndexSpaceDim(m_yOperand, DIM_H));
    ASSERT_EQ(nodeAp->getIndexSpaceDim(m_xOperand, DIM_W), nodeAp->getIndexSpaceDim(m_yOperand, DIM_W));
    ASSERT_EQ(nodeAp->getIndexSpaceDim(m_xOperand, DIM_C), nodeAp->getIndexSpaceDim(m_wOperand, WEIGHT_DIM_C));
    ASSERT_EQ(nodeAp->getIndexSpaceDim(m_yOperand, DIM_C), nodeAp->getIndexSpaceDim(m_wOperand, WEIGHT_DIM_K));
}

INSTANTIATE_TEST_SUITE_P(conv_access_pattern_dims_mapping,
                         ConvAccessPatternDimMappingTest,
                         ::testing::Values(ConvAccessPatternDimMappingTestParams {NodeFactory::convolutionNodeTypeName},
                                           ConvAccessPatternDimMappingTestParams {
                                               NodeFactory::convolution3DNodeTypeName}));

INSTANTIATE_TEST_SUITE_P(dedx_access_pattern_dims_mapping,
                         ConvAccessPatternDimMappingTest,
                         ::testing::Values(ConvAccessPatternDimMappingTestParams {NodeFactory::deDxNodeTypeName},
                                           ConvAccessPatternDimMappingTestParams {NodeFactory::deDx3DNodeTypeName}));

INSTANTIATE_TEST_SUITE_P(dedw_access_pattern_dims_mapping,
                         ConvAccessPatternDimMappingTest,
                         ::testing::Values(ConvAccessPatternDimMappingTestParams {NodeFactory::deDwNodeTypeName},
                                           ConvAccessPatternDimMappingTestParams {NodeFactory::deDw3DNodeTypeName}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ReshapeAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::tuple<SizeVector,  // input sizes
                                                  SizeVector,  // output sizes
                                                  bool>>       // has shape tensor
{
protected:
    ReshapeAccessPatternDimMappingTest() { std::tie(m_inSizes, m_outSizes, m_hasShapeTensor) = GetParam(); }

    NodePtr createNode() const
    {
        TensorVector inputs = {createOperand(m_inSizes)};
        if (m_hasShapeTensor)
        {
            inputs.push_back(createShapeOperand(m_outSizes));
        }
        return NodeFactory::createNode(inputs,
                                       {createOperand(m_outSizes)},
                                       nullptr,
                                       NodeFactory::reshapeNodeTypeName,
                                       "Reshape");
    }

    TensorPtr createOperand(const SizeVector& shape) const
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
    }

    TensorPtr createShapeOperand(const SizeVector& shape) const
    {
        return std::make_shared<Tensor>(shape.size(),
                                        shape.data(),
                                        syn_type_uint32,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        nullptr,
                                        SHAPE_TENSOR);
    }

    Dim findLastNoneDegenerateDim(const TensorPtr& tensor) const
    {
        for (Dim i = tensor->getDim() - 1; i > 0; i--)
        {
            if (tensor->getSizeInElements(i) > 1)
            {
                return i;
            }
        }
        return 0;
    }

    SizeVector m_inSizes;
    SizeVector m_outSizes;
    bool       m_hasShapeTensor;
};

TEST_P(ReshapeAccessPatternDimMappingTest, reshape_access_pattern_dims_mapping_test)
{
    NodePtr node = createNode();

    const auto& nodeAp = node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    // Currently we have 2 dims in the node resolution: the first is used to "track" the mapped dim
    // (outer-most non-degenerate dim) and the second is used to map the rest of the dims
    // (all-required access pattern).
    ASSERT_EQ(nodeAp->getNodeResolution().size(), 2);

    for (const auto& tensor : node->getOperands())
    {
        Dim mappedDim = findLastNoneDegenerateDim(tensor);
        for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
        {
            ASSERT_EQ(nodeAp->getIndexSpaceDim(tensor, tensorDim), (tensorDim == mappedDim) ? 0 : 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(reshape_access_pattern_dims_mapping_output_flattened,
                         ReshapeAccessPatternDimMappingTest,
                         ::testing::Combine(::testing::Values(SizeVector {1024, 1, 1, 128},
                                                              SizeVector {1024, 128, 1, 1},
                                                              SizeVector {1024, 1, 128, 1}),
                                            ::testing::Values(SizeVector {1024, 128}),
                                            ::testing::Values(false, true)));

INSTANTIATE_TEST_SUITE_P(reshape_access_pattern_dims_mapping_input_flattened,
                         ReshapeAccessPatternDimMappingTest,
                         ::testing::Combine(::testing::Values(SizeVector {1024, 128}),
                                            ::testing::Values(SizeVector {1024, 1, 1, 128},
                                                              SizeVector {1024, 128, 1, 1},
                                                              SizeVector {1024, 1, 128, 1}),
                                            ::testing::Values(false, true)));

INSTANTIATE_TEST_SUITE_P(
    reshape_access_pattern_dims_mapping_no_flatten,
    ReshapeAccessPatternDimMappingTest,
    ::testing::Values(std::make_tuple(SizeVector {1024, 1, 1, 128}, SizeVector {1024, 128, 1, 1}, false),
                      std::make_tuple(SizeVector {1024, 1, 1, 128}, SizeVector {1024, 128, 1, 1}, true),
                      std::make_tuple(SizeVector {1024, 128, 1, 1}, SizeVector {1024, 1, 1, 128}, false),
                      std::make_tuple(SizeVector {1024, 128, 1, 1}, SizeVector {1024, 1, 1, 128}, true),
                      std::make_tuple(SizeVector {10, 20, 30, 40}, SizeVector {10, 20, 30, 40}, false),
                      std::make_tuple(SizeVector {10, 20, 30, 40}, SizeVector {10, 20, 30, 40}, true),
                      std::make_tuple(SizeVector {10, 20, 30, 40}, SizeVector {40, 30, 20, 10}, false),
                      std::make_tuple(SizeVector {10, 20, 30, 40}, SizeVector {40, 30, 20, 10}, true)));

INSTANTIATE_TEST_SUITE_P(
    reshape_access_pattern_dims_mapping_with_aggregation,
    ReshapeAccessPatternDimMappingTest,
    ::testing::Values(std::make_tuple(SizeVector {1024, 32, 64}, SizeVector {1024, 32 * 64}, true),
                      std::make_tuple(SizeVector {1024, 6, 32, 64}, SizeVector {1024, 6 * 32 * 64}, true),
                      std::make_tuple(SizeVector {1024, 32 * 64}, SizeVector {1024, 32, 64}, true),
                      std::make_tuple(SizeVector {1024, 3 * 32 * 64}, SizeVector {1024, 3, 32, 64}, true)));

INSTANTIATE_TEST_SUITE_P(
    reshape_access_pattern_dims_mapping_all_required,
    ReshapeAccessPatternDimMappingTest,
    ::testing::Values(std::make_tuple(SizeVector {32 * 32, 32 * 32}, SizeVector {32, 32 * 32, 32}, false),
                      std::make_tuple(SizeVector {32 * 32, 32 * 32}, SizeVector {32, 32 * 32, 32}, true)));

INSTANTIATE_TEST_SUITE_P(
    reshape_access_pattern_dims_mapping_all_dims_degenerated,
    ReshapeAccessPatternDimMappingTest,
    ::testing::Combine(::testing::Values(SizeVector {1}, SizeVector {1, 1}, SizeVector {1, 1, 1, 1}),
                       ::testing::Values(SizeVector {1}, SizeVector {1, 1}, SizeVector {1, 1, 1}),
                       ::testing::Values(false, true)));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TransposeAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::vector<Dim>>  // transpose permutation
{
protected:
    TransposeAccessPatternDimMappingTest()
    {
        const auto& permutation = GetParam();
        for (Dim dim : permutation)
        {
            m_permutation.push_back(TransposePermutationDim(dim));
        }
    }

    NodePtr createNode()
    {
        m_inputShape.resize(m_permutation.size());

        TSize firstDimSize = 10;

        synTransposeParams params;
        params.tensorDim = m_permutation.size();
        for (auto i = 0; i < m_permutation.size(); i++)
        {
            params.permutation[i] = m_permutation[i];
            m_outputShape.push_back(firstDimSize++);            // Output shape will be {10,11,12,...}
            m_inputShape[m_permutation[i]] = m_outputShape[i];  // Input shape is initialized according to permutation
        }

        auto input  = std::make_shared<Tensor>(m_inputShape.size(), m_inputShape.data(), syn_type_float);
        auto output = std::make_shared<Tensor>(m_outputShape.size(), m_outputShape.data(), syn_type_float);

        return NodeFactory::createNode({input},
                                       {output},
                                       &params,
                                       sizeof(params),
                                       NodeFactory::transposeNodeTypeName,
                                       "Transpose");
    }

    Dim findMatchingOutputDim(Dim inputDim) const
    {
        HB_ASSERT(inputDim < m_inputShape.size(), "Invalid input dim");
        for (Dim i = 0; i < m_outputShape.size(); i++)
        {
            if (m_inputShape[inputDim] == m_outputShape[i])
            {
                return i;  // Output is initialized with unique sizes
            }
        }
        HB_ASSERT(false, "Invalid shapes for transpose node");
        return 0;
    }

    TransposePermutationArray m_permutation;
    std::vector<TSize>        m_inputShape;
    std::vector<TSize>        m_outputShape;
};

TEST_P(TransposeAccessPatternDimMappingTest, transpose_access_pattern_dims_mapping_test)
{
    NodePtr node = createNode();

    const auto& nodeAp = node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    ASSERT_EQ(nodeAp->getNodeResolution().size(), node->getOutput(0)->getDim());

    ASSERT_EQ(node->getInput(0)->getDim(), node->getOutput(0)->getDim());

    for (Dim tensorDim = 0; tensorDim < node->getOutput(0)->getDim(); tensorDim++)
    {
        // 1:1 mapping for the output tensor.
        ASSERT_EQ(nodeAp->getIndexSpaceDim(node->getOutput(0), tensorDim), tensorDim);

        // Matching dims (same size) should be mapped to the same index-space dim.
        ASSERT_EQ(nodeAp->getIndexSpaceDim(node->getInput(0), tensorDim),
                  nodeAp->getIndexSpaceDim(node->getOutput(0), findMatchingOutputDim(tensorDim)));
    }
}

INSTANTIATE_TEST_SUITE_P(transpose_access_pattern_dims_mapping_identity_permutation,
                         TransposeAccessPatternDimMappingTest,
                         ::testing::Values(std::vector<Dim> {0},
                                           std::vector<Dim> {0, 1, 2},
                                           std::vector<Dim> {0, 1, 2, 3, 4}));

INSTANTIATE_TEST_SUITE_P(transpose_access_pattern_dims_mapping_with_permutation,
                         TransposeAccessPatternDimMappingTest,
                         ::testing::Values(std::vector<Dim> {1, 0},
                                           std::vector<Dim> {0, 2, 1},
                                           std::vector<Dim> {1, 2, 0},
                                           std::vector<Dim> {0, 3, 1, 2},
                                           std::vector<Dim> {3, 1, 2, 0},
                                           std::vector<Dim> {3, 2, 1, 0},
                                           std::vector<Dim> {3, 4, 0, 2, 1},
                                           std::vector<Dim> {4, 3, 2, 1, 0},
                                           std::vector<Dim> {3, 2, 4, 0, 1}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ExpandDimsAccessPatternDimMappingTestParams
{
    Dim      inputRank;
    unsigned expandDim;
};

class ExpandDimsAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<ExpandDimsAccessPatternDimMappingTestParams>
{
protected:
    NodePtr createNode()
    {
        TSize inputDimSize = 100;
        for (auto i = 0; i < m_inputRank; i++)
        {
            m_inputShape.push_back(inputDimSize++);  // Input shape will be {100,101,102...} according to rank
        }
        auto input = std::make_shared<Tensor>(m_inputShape.size(), m_inputShape.data(), syn_type_float);

        m_outputShape = m_inputShape;
        m_outputShape.insert(m_outputShape.begin() + m_expandDim, 1);
        auto output = std::make_shared<Tensor>(m_outputShape.size(), m_outputShape.data(), syn_type_float);

        return NodeFactory::createNode({input},
                                       {output},
                                       &m_expandDim,
                                       NodeFactory::expandDimsNodeTypeName,
                                       "ExpandDims");
    }

    Dim findMatchingOutputDim(Dim inputDim) const
    {
        EXPECT_LT(inputDim, m_inputShape.size());
        for (Dim i = 0; i < m_outputShape.size(); i++)
        {
            if (m_inputShape[inputDim] == m_outputShape[i])
            {
                return i;  // Input is initialized with unique sizes
            }
        }
        HB_ASSERT(false, "Invalid shapes for expand-dims node");
        return 0;
    }

    const Dim      m_inputRank = GetParam().inputRank;
    const unsigned m_expandDim = GetParam().expandDim;

    std::vector<TSize> m_inputShape;
    std::vector<TSize> m_outputShape;
};

TEST_P(ExpandDimsAccessPatternDimMappingTest, expand_dims_access_pattern_dims_mapping_test)
{
    NodePtr node = createNode();

    const auto& nodeAp = node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    ASSERT_EQ(node->getInput(0)->getDim() + 1, node->getOutput(0)->getDim());

    // The expand dims node shape is similar to the output tensor, with 1:1 mapping.
    ASSERT_EQ(nodeAp->getNodeResolution().size(), node->getOutput(0)->getDim());
    for (Dim tensorDim = 0; tensorDim < node->getOutput(0)->getDim(); tensorDim++)
    {
        ASSERT_EQ(nodeAp->getIndexSpaceDim(node->getOutput(0), tensorDim), tensorDim);
    }

    // The expand dims input is missing the expanded dim from the node dims, and has 1:1 mapping for the rest of the
    // dims
    for (Dim tensorDim = 0; tensorDim < node->getInput(0)->getDim(); tensorDim++)
    {
        ASSERT_EQ(nodeAp->getIndexSpaceDim(node->getInput(0), tensorDim),
                  nodeAp->getIndexSpaceDim(node->getOutput(0), findMatchingOutputDim(tensorDim)));
        ASSERT_NE(nodeAp->getIndexSpaceDim(node->getInput(0), tensorDim),
                  nodeAp->getIndexSpaceDim(node->getOutput(0), m_expandDim));
    }
}

INSTANTIATE_TEST_SUITE_P(expand_dims_access_pattern_dims_mapping,
                         ExpandDimsAccessPatternDimMappingTest,  // {input rank, expand dim}
                         ::testing::Values(ExpandDimsAccessPatternDimMappingTestParams {1, 0},
                                           ExpandDimsAccessPatternDimMappingTestParams {1, 1},
                                           ExpandDimsAccessPatternDimMappingTestParams {2, 0},
                                           ExpandDimsAccessPatternDimMappingTestParams {2, 1},
                                           ExpandDimsAccessPatternDimMappingTestParams {2, 2},
                                           ExpandDimsAccessPatternDimMappingTestParams {3, 0},
                                           ExpandDimsAccessPatternDimMappingTestParams {3, 1},
                                           ExpandDimsAccessPatternDimMappingTestParams {3, 2},
                                           ExpandDimsAccessPatternDimMappingTestParams {3, 3},
                                           ExpandDimsAccessPatternDimMappingTestParams {4, 0},
                                           ExpandDimsAccessPatternDimMappingTestParams {4, 1},
                                           ExpandDimsAccessPatternDimMappingTestParams {4, 2},
                                           ExpandDimsAccessPatternDimMappingTestParams {4, 3},
                                           ExpandDimsAccessPatternDimMappingTestParams {4, 4}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SqueezeAccessPatternDimMappingTestParams
{
    std::vector<TSize>      inputShape;
    std::optional<unsigned> squeezeDim;
    bool padOutputWithOnes;  // Add "1" padding at the end of the output (will have the same rank as the input)
};

class SqueezeAccessPatternDimMappingTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<SqueezeAccessPatternDimMappingTestParams>
{
protected:
    NodePtr createNode()
    {
        auto input = std::make_shared<Tensor>(m_inputShape.size(), m_inputShape.data(), syn_type_float);

        m_outputShape = m_inputShape;
        if (m_squeezeDim.has_value())
        {
            EXPECT_FALSE(m_padOutputWithOnes);  // Not supported in this mode
            EXPECT_EQ(m_inputShape.at(m_squeezeDim.value()), 1);
            m_outputShape.erase(m_outputShape.begin() + m_squeezeDim.value());
        }
        else
        {
            // Remove all ones from output shape
            m_outputShape.erase(std::remove(m_outputShape.begin(), m_outputShape.end(), 1), m_outputShape.end());
            if (m_padOutputWithOnes)
            {
                for (auto i = m_outputShape.size(); i < m_inputShape.size(); i++)
                {
                    m_outputShape.push_back(1);
                }
            }
        }
        auto output = std::make_shared<Tensor>(m_outputShape.size(), m_outputShape.data(), syn_type_float);

        if (m_squeezeDim.has_value())
        {
            synSqueezeParams params;
            params.axis = m_squeezeDim.value();
            return NodeFactory::createNode({input}, {output}, &params, NodeFactory::squeezeNodeTypeName, "Squeeze");
        }
        return NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::squeezeNodeTypeName, "Squeeze");
    }

    Dim findMatchingInputDim(Dim outputDim) const
    {
        EXPECT_LT(outputDim, m_outputShape.size());
        EXPECT_NE(m_outputShape[outputDim], 1);  // Handle real dims only (not squeezed/padded)
        for (Dim i = 0; i < m_inputShape.size(); i++)
        {
            if (m_inputShape[i] == m_outputShape[outputDim])
            {
                return i;
            }
        }
        HB_ASSERT(false, "Invalid shapes for squeeze node");
        return 0;
    }

    const std::vector<TSize>      m_inputShape        = GetParam().inputShape;
    const std::optional<unsigned> m_squeezeDim        = GetParam().squeezeDim;
    const bool                    m_padOutputWithOnes = GetParam().padOutputWithOnes;

    std::vector<TSize> m_outputShape;
};

TEST_P(SqueezeAccessPatternDimMappingTest, squeeze_access_pattern_dims_mapping_test)
{
    NodePtr node = createNode();

    const auto& nodeAp = node->getNodeAccessPattern();
    ASSERT_TRUE(nodeAp);

    // Validate that the node geometry is 1 for all the squeezed dims and the padding output dims.
    // For real tensor dims (not squeezed/padded) - the mapped index-space dims in input and output should match.

    for (Dim tensorDim = 0; tensorDim < node->getInput(0)->getDim(); tensorDim++)
    {
        Dim indexSpaceDim = nodeAp->getIndexSpaceDim(node->getInput(0), tensorDim);
        ASSERT_LT(indexSpaceDim, nodeAp->getNodeResolution().size());
        if (node->getInput(0)->getSizeInElements(tensorDim) == 1)
        {
            ASSERT_EQ(nodeAp->getNodeResolution().at(indexSpaceDim), 1);
        }
    }

    for (Dim tensorDim = 0; tensorDim < node->getOutput(0)->getDim(); tensorDim++)
    {
        Dim indexSpaceDim = nodeAp->getIndexSpaceDim(node->getOutput(0), tensorDim);
        ASSERT_LT(indexSpaceDim, nodeAp->getNodeResolution().size());
        if (node->getOutput(0)->getSizeInElements(tensorDim) == 1)
        {
            ASSERT_EQ(nodeAp->getNodeResolution().at(indexSpaceDim), 1);
        }
        else
        {
            ASSERT_EQ(nodeAp->getIndexSpaceDim(node->getOutput(0), tensorDim),
                      nodeAp->getIndexSpaceDim(node->getInput(0), findMatchingInputDim(tensorDim)));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(squeeze_access_pattern_dims_mapping_single_squeezed_dim,
                         SqueezeAccessPatternDimMappingTest,
                         ::testing::Values(SqueezeAccessPatternDimMappingTestParams {{1, 123}, 0, false},
                                           SqueezeAccessPatternDimMappingTestParams {{123, 1}, 1, false},
                                           SqueezeAccessPatternDimMappingTestParams {{1, 128, 43, 1}, 0, false},
                                           SqueezeAccessPatternDimMappingTestParams {{1, 128, 43, 1}, 3, false},
                                           SqueezeAccessPatternDimMappingTestParams {{1, 128, 1, 43, 1}, 2, false},
                                           SqueezeAccessPatternDimMappingTestParams {{31, 1, 2, 65, 221}, 1, false},
                                           SqueezeAccessPatternDimMappingTestParams {{31, 876, 1, 65}, 2, false},
                                           SqueezeAccessPatternDimMappingTestParams {{1, 1, 1, 1}, 1, false}));

INSTANTIATE_TEST_SUITE_P(
    squeeze_access_pattern_dims_mapping_multiple_squeezed_dims,
    SqueezeAccessPatternDimMappingTest,
    ::testing::Values(SqueezeAccessPatternDimMappingTestParams {{128, 1, 1}, std::nullopt, false},
                      SqueezeAccessPatternDimMappingTestParams {{1, 1, 128}, std::nullopt, false},
                      SqueezeAccessPatternDimMappingTestParams {{1, 128, 1}, std::nullopt, false},
                      SqueezeAccessPatternDimMappingTestParams {{1, 128, 1, 64, 1}, std::nullopt, false},
                      SqueezeAccessPatternDimMappingTestParams {{1, 1, 1, 64, 1}, std::nullopt, false},
                      SqueezeAccessPatternDimMappingTestParams {{48, 128, 1, 1, 96}, std::nullopt, false}));

INSTANTIATE_TEST_SUITE_P(
    squeeze_access_pattern_dims_mapping_with_padding_in_output,
    SqueezeAccessPatternDimMappingTest,
    ::testing::Values(SqueezeAccessPatternDimMappingTestParams {{128, 1, 1}, std::nullopt, true},
                      SqueezeAccessPatternDimMappingTestParams {{1, 1, 128}, std::nullopt, true},
                      SqueezeAccessPatternDimMappingTestParams {{1, 128, 1}, std::nullopt, true},
                      SqueezeAccessPatternDimMappingTestParams {{1, 128, 1, 64, 1}, std::nullopt, true},
                      SqueezeAccessPatternDimMappingTestParams {{1, 1, 1, 64, 1}, std::nullopt, true},
                      SqueezeAccessPatternDimMappingTestParams {{48, 128, 1, 1, 96}, std::nullopt, true}));
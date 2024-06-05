#include <math.h>
#include "habana_pass.h"
#include "quantization_data.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "node.h"
#include "sim_graph.h"
#include "infra/global_conf_manager.h"
#include "graph_optimizer_test.h"
#include "generic_graph_test.h"
#include "test_utils.h"
#include <graph_compiler/passes/quantization_utils.h>
#include <graph_compiler/habana_nodes/node_factory.h>
#include <graph_compiler/habana_nodes/transpose_utils.h>
#include "graph_compiler/passes/cast_nodes_handler.h"
#include "gaudi2_graph.h"
#include "types.h"
#include "tpc_node.h"

using namespace gc;

class UpdateMMEPrecisionTest : public GenericGraphTest
{
protected:
    void        SetUp() override
    {
        GraphOptimizerTest::SetUp();
        synDeviceType deviceType = GetParam();
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
        setGlobalConfForTest(GCFG_UPDATE_GRAPH_OUTPUT_MME, "true");
    }

    void TearDown() override
    {
        m_graph.reset();
        GraphOptimizerTest::TearDown();
    }
};


TEST_P(UpdateMMEPrecisionTest, mme_with_float_inputs)
{
    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    setGlobalConfForTest(GCFG_ALLOW_DEFAULT_QUANT_PARAMS, "false");

    synConvolutionParams params;
    const TSize          kW = 1, kH = 5, dW = 1, dH = 1;
    const TSize          nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    // o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = {nIFM, wIFM, hIFM, 1};
    const TSize o_sizes[] = {nOFM, wOFM, hOFM, 1};
    const TSize w_sizes[] = {nOFM, nIFM, kW, kH};

    TensorPtr    ifm    = std::make_shared<Tensor>(4U, i_sizes, syn_type_float);
    TensorPtr    weight = std::make_shared<Tensor>(4U, w_sizes, syn_type_float);
    TensorPtr    ofm    = std::make_shared<Tensor>(4U, o_sizes, syn_type_float);
    TensorVector inputs {ifm, weight, nullptr};
    TensorVector outputs {ofm};

    NodePtr node = NodeFactory::createNode(inputs, outputs, &params, "spatial_convolution", "");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, node)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNumNodes(), 1) << "Input graph contains 1 node";
    ASSERT_TRUE(updateMMENodePrecision(*m_graph)) << "Failed to detach bias from mme node";

    ASSERT_EQ((*m_graph).getNumNodes(), 1) << "Graph not should not be changed";

    const auto& nodes = (*m_graph).getExeSortedNodes();

    NodePtr mmeNode = nodes[0];
    ASSERT_EQ(mmeNode->getGUID(), "spatial_convolution");
    ASSERT_EQ(mmeNode->getInput(0)->getElementType(), syn_type_float);
    ASSERT_EQ(mmeNode->getInput(1)->getElementType(), syn_type_float);
    ASSERT_EQ(mmeNode->getOutput(0)->getElementType(), syn_type_float);
}

TEST_P(UpdateMMEPrecisionTest, mme_with_float_inputs_use_default_quant_params)
{
    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);

    synConvolutionParams params;
    const TSize          kW = 1, kH = 5, dW = 1, dH = 1;
    const TSize          nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    // o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = {nIFM, wIFM, hIFM, 1};
    const TSize o_sizes[] = {nOFM, wOFM, hOFM, 1};
    const TSize w_sizes[] = {nOFM, nIFM, kW, kH};

    TensorPtr    ifm    = std::make_shared<Tensor>(4U, i_sizes, syn_type_float);
    TensorPtr    weight = std::make_shared<Tensor>(4U, w_sizes, syn_type_float);
    TensorPtr    ofm    = std::make_shared<Tensor>(4U, o_sizes, syn_type_float);
    TensorVector inputs {ifm, weight, nullptr};
    TensorVector outputs {ofm};

    NodePtr node = NodeFactory::createNode(inputs, outputs, &params, "spatial_convolution", "");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, node)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNumNodes(), 1) << "Input graph contains 1 node";
    ASSERT_TRUE(updateMMENodePrecision(*m_graph)) << "Failed to detach bias from mme node";

    ASSERT_EQ((*m_graph).getNumNodes(), 3) << "Expecting graph of 3 nodes, 2 cast nodes and 1 MME node";

    const auto& nodes = (*m_graph).getExeSortedNodes();

    NodePtr castIfmNode = nodes[0];
    ASSERT_TRUE(castIfmNode->isCast());
    ASSERT_EQ(castIfmNode->getGUID(), "cast_f32_to_hf8");
    ASSERT_EQ(castIfmNode->getInput(0), ifm);
    ASSERT_EQ(castIfmNode->getOutput(0)->getElementType(), syn_type_fp8_143);
    auto castIfmAsTPCNode = std::dynamic_pointer_cast<TPCNode>(castIfmNode);
    ASSERT_TRUE(castIfmAsTPCNode != nullptr);
    ASSERT_EQ(castIfmAsTPCNode->getParamsSize(), sizeof(ns_CastKernel::ParamsV3));
    NodePtr castWgtNode = nodes[1];
    ASSERT_TRUE(castWgtNode->isCast());
    ASSERT_EQ(castWgtNode->getGUID(), "cast_f32_to_hf8");
    ASSERT_EQ(castWgtNode->getInput(0), weight);
    ASSERT_EQ(castWgtNode->getOutput(0)->getElementType(), syn_type_fp8_143);
    auto castWgtAsTPCNode = std::dynamic_pointer_cast<TPCNode>(castWgtNode);
    ASSERT_TRUE(castWgtAsTPCNode != nullptr);
    ASSERT_EQ(castWgtAsTPCNode->getParamsSize(), sizeof(ns_CastKernel::ParamsV3));

    NodePtr mmeNode = nodes[2];
    ASSERT_EQ(mmeNode->getGUID(), "spatial_convolution");
    ASSERT_EQ(mmeNode->getInput(0)->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(mmeNode->getInput(1)->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(mmeNode->getOutput(0)->getElementType(), syn_type_float);
}

TEST_P(UpdateMMEPrecisionTest, mme_with_float_inputs_with_dynamic_ranges)
{
    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);

    synConvolutionParams params;
    const TSize          kW = 1, kH = 5, dW = 1, dH = 1;
    const TSize          nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    // o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = {nIFM, wIFM, hIFM, 1};
    const TSize o_sizes[] = {nOFM, wOFM, hOFM, 1};
    const TSize w_sizes[] = {nOFM, nIFM, kW, kH};

    TensorPtr    ifm    = std::make_shared<Tensor>(4U, i_sizes, syn_type_float);
    TensorPtr    weight = std::make_shared<Tensor>(4U, w_sizes, syn_type_float);
    TensorPtr    ofm    = std::make_shared<Tensor>(4U, o_sizes, syn_type_float);
    TensorVector inputs {ifm, weight, nullptr};
    TensorVector outputs {ofm};

    DynamicRange dynamicRangeIfm;
    dynamicRangeIfm.min   = 0;
    dynamicRangeIfm.max   = 1;
    dynamicRangeIfm.isSet = true;
    ifm->setDynamicRange(dynamicRangeIfm);

    DynamicRange dynamicRangeWeight;
    dynamicRangeWeight.min   = 0;
    dynamicRangeWeight.max   = 1;
    dynamicRangeWeight.isSet = true;
    weight->setDynamicRange(dynamicRangeWeight);

    NodePtr node = NodeFactory::createNode(inputs, outputs, &params, "spatial_convolution", "");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, node)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNumNodes(), 1) << "Input graph contains 1 node";
    ASSERT_TRUE(updateMMENodePrecision(*m_graph)) << "Failed to detach bias from mme node";

    ASSERT_EQ((*m_graph).getNumNodes(), 3) << "Expecting graph of 3 nodes, 2 cast nodes and 1 MME node";

    const auto& nodes = (*m_graph).getExeSortedNodes();

    NodePtr castIfmNode = nodes[0];
    ASSERT_TRUE(castIfmNode->isCast());
    ASSERT_EQ(castIfmNode->getGUID(), "cast_f32_to_hf8");
    ASSERT_EQ(castIfmNode->getInput(0), ifm);
    ASSERT_EQ(castIfmNode->getOutput(0)->getElementType(), syn_type_fp8_143);

    NodePtr castWgtNode = nodes[1];
    ASSERT_TRUE(castWgtNode->isCast());
    ASSERT_EQ(castWgtNode->getGUID(), "cast_f32_to_hf8");
    ASSERT_EQ(castWgtNode->getInput(0), weight);
    ASSERT_EQ(castWgtNode->getOutput(0)->getElementType(), syn_type_fp8_143);

    NodePtr mmeNode = nodes[2];
    ASSERT_EQ(mmeNode->getGUID(), "spatial_convolution");
    ASSERT_EQ(mmeNode->getInput(0)->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(mmeNode->getInput(1)->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(mmeNode->getOutput(0)->getElementType(), syn_type_float);
}

TEST_P(UpdateMMEPrecisionTest, mme_with_fp8_152_profile_precision)
{
    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    setGlobalConfForTest(GCFG_PROFILE_PRECISION, "f8");

    synConvolutionParams params;
    const TSize          kW = 1, kH = 5, dW = 1, dH = 1;
    const TSize          nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    // o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = {nIFM, wIFM, hIFM, 1};
    const TSize o_sizes[] = {nOFM, wOFM, hOFM, 1};
    const TSize w_sizes[] = {nOFM, nIFM, kW, kH};

    TensorPtr    ifm    = std::make_shared<Tensor>(4U, i_sizes, syn_type_float);
    TensorPtr    weight = std::make_shared<Tensor>(4U, w_sizes, syn_type_float);
    TensorPtr    ofm    = std::make_shared<Tensor>(4U, o_sizes, syn_type_float);
    TensorVector inputs {ifm, weight, nullptr};
    TensorVector outputs {ofm};

    NodePtr node = NodeFactory::createNode(inputs, outputs, &params, "spatial_convolution", "");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, node)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNumNodes(), 1) << "Input graph contains 1 node";
    ASSERT_TRUE(updateMMENodePrecision(*m_graph)) << "Failed to detach bias from mme node";

    ASSERT_EQ((*m_graph).getNumNodes(), 3) << "Expecting graph of 3 nodes, 2 cast nodes and 1 MME node";

    const auto& nodes = (*m_graph).getExeSortedNodes();

    NodePtr castIfmNode = nodes[0];
    ASSERT_TRUE(castIfmNode->isCast());
    ASSERT_EQ(castIfmNode->getGUID(), "cast_f32_to_f8");
    ASSERT_EQ(castIfmNode->getInput(0), ifm);
    ASSERT_EQ(castIfmNode->getOutput(0)->getElementType(), syn_type_fp8_152);

    NodePtr castWgtNode = nodes[1];
    ASSERT_TRUE(castWgtNode->isCast());
    ASSERT_EQ(castWgtNode->getGUID(), "cast_f32_to_f8");
    ASSERT_EQ(castWgtNode->getInput(0), weight);
    ASSERT_EQ(castWgtNode->getOutput(0)->getElementType(), syn_type_fp8_152);

    NodePtr mmeNode = nodes[2];
    ASSERT_EQ(mmeNode->getGUID(), "spatial_convolution");
    ASSERT_EQ(mmeNode->getInput(0)->getElementType(), syn_type_fp8_152);
    ASSERT_EQ(mmeNode->getInput(1)->getElementType(), syn_type_fp8_152);
    ASSERT_EQ(mmeNode->getOutput(0)->getElementType(), syn_type_float);
}

// this test simulates KV$ in fp8 from GPT-J
// original pre-graph :
// MME(bf16 output)->reshape->cast_bf16_to_hf8->tpc_kernel_hf8->cast_hf8_to_bf16->reshape->MME
// expected post-graph :
// MME(hf8 output)->reshape->tpc_kernel_hf8->reshape->MME
TEST_P(UpdateMMEPrecisionTest, mme_with_fp8_output)
{
    setGlobalConfForTest(GCFG_UPDATE_MME_OUTPUT_PRECISION_FILTER, "v_proj");

    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    synDataType preGraphDataType = syn_type_bf16;
    synDataType profilePrecision = getSynDataTypeFromString(GCFG_PROFILE_PRECISION.getValueStr());//getSynDataTypeFromString(m_profilePrecision);

    synConvolutionParams params;
    const TSize          kW = 1, kH = 5, dW = 1, dH = 1;
    const TSize          nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    const TSize          wIFM = ((wOFM - 1) * dW) + kW;
    const TSize          hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = {nIFM, wIFM, hIFM, 1};
    const TSize o_sizes[] = {nOFM, wOFM, hOFM, 1};
    const TSize w_sizes[] = {nOFM, nIFM, kW, kH};

    TensorPtr firstIfm    = std::make_shared<Tensor>(4U, i_sizes, preGraphDataType);
    TensorPtr firstWeight = std::make_shared<Tensor>(4U, w_sizes, preGraphDataType);
    TensorPtr firstOfm    = std::make_shared<Tensor>(4U, o_sizes, preGraphDataType);

    // first MME node - need to name it v_proj so casting on its output will work
    NodePtr firstMME =
        NodeFactory::createNode({firstIfm, firstWeight, nullptr}, {firstOfm}, &params, "spatial_convolution", "v_proj");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, firstMME));
    // first reshape
    TensorPtr firstReshapeOut = std::make_shared<Tensor>(4U, o_sizes, preGraphDataType);
    NodePtr   firstReshape    = NodeFactory::createNode({firstOfm},
                                                        {firstReshapeOut},
                                                   nullptr,
                                                   NodeFactory::reshapeNodeTypeName,
                                                   "firstReshape");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, firstReshape));
    // cast to fp8
    TensorPtr   castOutputFp8 = std::make_shared<Tensor>(4U, o_sizes, profilePrecision);
    std::string castGuid      = getCastGUID(preGraphDataType, profilePrecision);
    NodePtr     castBf16ToFp8 =
        NodeFactory::createNode({firstReshapeOut}, {castOutputFp8}, nullptr, 0, castGuid, "firstCast");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, castBf16ToFp8));
    // memcpy fp8 - simulates fp8 kernel (as we don't have one to test currently)
    TensorPtr memcpyOut = std::make_shared<Tensor>(4U, o_sizes, profilePrecision);
    NodePtr   memcpyFp8 =
        NodeFactory::createNode({castOutputFp8}, {memcpyOut}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, memcpyFp8));
    // cast from fp8
    TensorPtr   castOutputBf16  = std::make_shared<Tensor>(4U, o_sizes, preGraphDataType);
    std::string reverseCastGuid = getCastGUID(profilePrecision, preGraphDataType);
    NodePtr     castFp8ToBf16 =
        NodeFactory::createNode({memcpyOut}, {castOutputBf16}, nullptr, 0, reverseCastGuid, "secondCast");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, castFp8ToBf16));
    // second reshape
    TensorPtr secondReshapeOut = std::make_shared<Tensor>(4U, o_sizes, preGraphDataType);
    NodePtr   secondReshape    = NodeFactory::createNode({castOutputBf16},
                                                         {secondReshapeOut},
                                                    nullptr,
                                                    NodeFactory::reshapeNodeTypeName,
                                                    "secondReshape");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, secondReshape));
    // second MME node
    TensorPtr weight2   = std::make_shared<Tensor>(4U, w_sizes, preGraphDataType);
    TensorPtr ofm2      = std::make_shared<Tensor>(4U, o_sizes, preGraphDataType);
    NodePtr   secondMME = NodeFactory::createNode({secondReshapeOut, weight2, nullptr},
                                                  {ofm2},
                                                &params,
                                                "spatial_convolution",
                                                "secondMME");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, secondMME));

    // verify number of casts in pre-graph
    unsigned    preGraphCasts = 0;
    const auto& preGraphNodes = (*m_graph).getExeSortedNodes();
    for (auto n : preGraphNodes)
    {
        if (!n->isCast()) continue;
        preGraphCasts++;
    }
    ASSERT_EQ(preGraphCasts, 2);
    ASSERT_TRUE(updateMMENodePrecision(*m_graph));
    // verify number of casts after update mme precision pass -
    // expecting 7 casts: 2 surrounding memcpy, 1 on first MME output and 4 on all MME inputs
    unsigned    middleGraphCasts = 0;
    const auto& middleGraphNodes = (*m_graph).getExeSortedNodes();
    for (auto n : middleGraphNodes)
    {
        if (!n->isCast()) continue;
        middleGraphCasts++;
    }
    ASSERT_EQ(middleGraphCasts, 7);
    // verify first MME output has cast
    ASSERT_TRUE((*m_graph).getTensorConsumers(firstMME->getOutput(0)).front()->isCast());
    auto castOfmAsTPCNode = std::dynamic_pointer_cast<TPCNode>((*m_graph).getTensorConsumers(firstMME->getOutput(0)).front());
    ASSERT_TRUE(castOfmAsTPCNode != nullptr);
    ASSERT_EQ(castOfmAsTPCNode->getParamsSize(), sizeof(ns_CastKernel::ParamsV3));
    // verify second MME input has cast
    ASSERT_TRUE((*m_graph).getTensorProducer(secondMME->getInput(0))->isCast());

    ASSERT_TRUE(propagateCastNodes(*m_graph));
    ASSERT_TRUE(removeContiguousCastNodes(*m_graph));

    // verify number of casts after remove contiguous cast pass -
    // expecting only 3 casts,
    // the following casts should be removed: 2 surrounding memcpy, 1 on first MME output and 1 on second MME input
    unsigned    postGraphCasts = 0;
    const auto& postGraphNodes = (*m_graph).getExeSortedNodes();
    for (auto n : postGraphNodes)
    {
        if (!n->isCast()) continue;
        postGraphCasts++;
    }
    ASSERT_EQ(postGraphCasts, 3);
    // verify first MME output doesn't have cast
    ASSERT_FALSE((*m_graph).getTensorConsumers(firstMME->getOutput(0)).front()->isCast());
    // verify second MME input doesn't have cast
    ASSERT_FALSE((*m_graph).getTensorProducer(secondMME->getInput(0))->isCast());
}



// this test simulates another sub graph in KV$ in fp8 from GPT-J.
// original pre-graph , transpose output has 2 consumers:
// MME(bf16 output)->transpose->cast_bf16_to_hf8->tpc_kernel_hf8
//                            ->broadcast->MME
// expected post-graph :
// MME(hf8 output)->transpose->tpc_kernel_hf8
//                           ->broadcast->MME
TEST_P(UpdateMMEPrecisionTest, mme_with_fp8_output_propagate_multi_consumers)
{
    setGlobalConfForTest(GCFG_UPDATE_MME_OUTPUT_PRECISION_FILTER, "v_proj");

    (*m_graph).setInferenceMode(true);
    (*m_graph).setQuantizationEnabled(true);
    synDataType preGraphDataType = syn_type_bf16;
    synDataType profilePrecision = getSynDataTypeFromString(GCFG_PROFILE_PRECISION.getValueStr());//getSynDataTypeFromString(m_profilePrecision);

    unsigned             dim = 4;
    synConvolutionParams params;
    const TSize          kW = 1, kH = 5, dW = 1, dH = 1;
    const TSize          nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    const TSize          wIFM = ((wOFM - 1) * dW) + kW;
    const TSize          hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = {nIFM, wIFM, hIFM, 1};
    const TSize o_sizes[] = {nOFM, wOFM, hOFM, 1};
    const TSize w_sizes[] = {nOFM, nIFM, kW, kH};

    TensorPtr firstIfm    = std::make_shared<Tensor>(dim, i_sizes, preGraphDataType);
    TensorPtr firstWeight = std::make_shared<Tensor>(dim, w_sizes, preGraphDataType);
    TensorPtr firstOfm    = std::make_shared<Tensor>(dim, o_sizes, preGraphDataType);

    // first MME node - need to name it v_proj so casting on its output will work
    NodePtr firstMME =
        NodeFactory::createNode({firstIfm, firstWeight, nullptr}, {firstOfm}, &params, "spatial_convolution", "v_proj");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, firstMME));
    // transpose
    synTransposeParamsNDims transposeParams = permutationToParams(gc::Permutation(dim));
    transposeParams.tensorDim               = dim;
    TensorPtr transposeOut                  = std::make_shared<Tensor>(dim, o_sizes, preGraphDataType);
    NodePtr   transpose                     = NodeFactory::createNode({firstOfm},
                                                                      {transposeOut},
                                                &transposeParams,
                                                sizeof(synTransposeParamsNDims),
                                                NodeFactory::transposeNodeTypeName,
                                                "transpose");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, transpose));
    // cast to fp8 (transpose output first consumer)
    TensorPtr castOutputFp8 = std::make_shared<Tensor>(dim, o_sizes, profilePrecision);
    NodePtr castBf16ToFp8 = CastNodeHandler::createCastNode(transposeOut, castOutputFp8, "firstCast", (*m_graph).getDeviceId());
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, castBf16ToFp8));
    // memcpy fp8 - simulates fp8 kernel (as we don't have one to test currently)
    TensorPtr memcpyOut = std::make_shared<Tensor>(dim, o_sizes, profilePrecision);
    NodePtr   memcpyFp8 =
        NodeFactory::createNode({castOutputFp8}, {memcpyOut}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, memcpyFp8));
    // reshape (transpose output second consumer)
    TensorPtr broadcastOut = std::make_shared<Tensor>(dim, o_sizes, preGraphDataType);
    NodePtr   broadcast    = NodeFactory::createNode({transposeOut},
                                                     {broadcastOut},
                                                nullptr,
                                                NodeFactory::broadcastNodeTypeName,
                                                "broadcast");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, broadcast));
    // second MME node
    TensorPtr weight2 = std::make_shared<Tensor>(dim, w_sizes, preGraphDataType);
    TensorPtr ofm2    = std::make_shared<Tensor>(dim, o_sizes, preGraphDataType);
    NodePtr   secondMME =
        NodeFactory::createNode({broadcastOut, weight2, nullptr}, {ofm2}, &params, "spatial_convolution", "secondMME");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, secondMME));
    // verify number of casts in pre-graph
    unsigned    preGraphCasts = 0;
    const auto& preGraphNodes = (*m_graph).getExeSortedNodes();
    for (auto n : preGraphNodes)
    {
        if (!n->isCast()) continue;
        preGraphCasts++;
    }
    ASSERT_EQ(preGraphCasts, 1);
    ASSERT_TRUE(updateMMENodePrecision(*m_graph));
    // Current graph state - casts were added on all MME inputs and first MME outputs:
    // cast_bf16_to_hf8->MME(hf8 output)->cast_hf8_to_bf16->transpose->cast_bf16_to_hf8->tpc_kernel_hf8
    //                                                               ->broadcast->cast_bf16_to_hf8->MME
    // verify number of casts after update mme precision pass -
    // expecting 6 casts: 1 before memcpy, 1 on first MME output and 4 on all MME inputs
    unsigned    postUpdatePrecisionCasts = 0;
    const auto& postUpdatePrecisionNodes = (*m_graph).getExeSortedNodes();
    for (auto n : postUpdatePrecisionNodes)
    {
        if (!n->isCast()) continue;
        postUpdatePrecisionCasts++;
    }
    ASSERT_EQ(postUpdatePrecisionCasts, 6);
    // verify first MME output has cast_hf8_to_bf16
    NodePtr castAfterFirstMME = (*m_graph).getTensorConsumers(firstMME->getOutput(0)).front();
    ASSERT_TRUE(castAfterFirstMME->isCast());
    ASSERT_TRUE(castAfterFirstMME->getInput(0)->getElementType() == profilePrecision);
    ASSERT_TRUE(castAfterFirstMME->getOutput(0)->getElementType() == preGraphDataType);
    // verify second MME input has cast_bf16_to_hf8
    NodePtr castBeforeSecondMME = (*m_graph).getTensorProducer(secondMME->getInput(0));
    ASSERT_TRUE(castBeforeSecondMME->isCast());
    ASSERT_TRUE(castBeforeSecondMME->getInput(0)->getElementType() == preGraphDataType);
    ASSERT_TRUE(castBeforeSecondMME->getOutput(0)->getElementType() == profilePrecision);
    ASSERT_TRUE((*m_graph).getTensorProducer(castBeforeSecondMME->getInput(0)) == broadcast);
    ASSERT_TRUE(propagateCastNodes(*m_graph));
    // Current graph state - the cast from before second MME was propagated above broadcast:
    // cast_bf16_to_hf8->MME(hf8 output)->cast_hf8_to_bf16->transpose->cast_bf16_to_hf8->tpc_kernel_hf8
    //                                                                 cast_bf16_to_hf8->broadcast->MME
    unsigned    postPropagateCasts = 0;
    const auto& postPropagateNodes = (*m_graph).getExeSortedNodes();
    for (auto n : postPropagateNodes)
    {
        if (!n->isCast()) continue;
        postPropagateCasts++;
    }
    ASSERT_EQ(postPropagateCasts, 6);
    // verify the cast before second MME was propagated above broadcast to be produced from transpose.
    // The transpose output now has 2 consumers.
    ASSERT_TRUE((*m_graph).getTensorProducer(castBeforeSecondMME->getInput(0)) == transpose);
    ASSERT_EQ((*m_graph).getTensorConsumers(transpose->getOutput(0)).size(), 2);
    ASSERT_TRUE(commonSubExpressionElimination(*m_graph));
    // Current graph state - one of the casts produced by the transpose was eliminated :
    // cast_bf16_to_hf8->MME(hf8 output)->cast_hf8_to_bf16->transpose->cast_bf16_to_hf8->tpc_kernel_hf8
    //                                                                                 ->broadcast->MME
    unsigned    postCSECasts = 0;
    const auto& postCSENodes = (*m_graph).getExeSortedNodes();
    for (auto n : postCSENodes)
    {
        if (!n->isCast()) continue;
        postCSECasts++;
    }
    ASSERT_EQ(postCSECasts, 5);
    ASSERT_EQ((*m_graph).getTensorConsumers(transpose->getOutput(0)).size(), 1);
    ASSERT_TRUE(propagateCastNodes(*m_graph));
    // Current graph state - cast was propagated above transpose :
    // cast_bf16_to_hf8->MME(hf8 output)->cast_hf8_to_bf16->cast_bf16_to_hf8->transpose->tpc_kernel_hf8
    //                                                                                 ->broadcast->MME
    ASSERT_TRUE(removeContiguousCastNodes(*m_graph));
    // Current graph state - contiguous casts between first MME and transpose were removed:
    // cast_bf16_to_hf8->MME(hf8 output)->transpose->tpc_kernel_hf8
    //                                             ->broadcast->MME
    // verify number of casts after remove contiguous cast pass -
    // expecting only 3 casts,
    // the following casts should be removed: 1 before memcpy, 1 on first MME output and 1 on second MME input
    unsigned    postGraphCasts = 0;
    const auto& postGraphNodes = (*m_graph).getExeSortedNodes();
    for (auto n : postGraphNodes)
    {
        if (!n->isCast()) continue;
        postGraphCasts++;
    }
    ASSERT_EQ(postGraphCasts, 3);
    // verify first MME output doesn't have cast
    ASSERT_FALSE((*m_graph).getTensorConsumers(firstMME->getOutput(0)).front()->isCast());
    ASSERT_TRUE((*m_graph).getTensorConsumers(firstMME->getOutput(0)).front() == transpose);
}

INSTANTIATE_TEST_SUITE_P(,
                         UpdateMMEPrecisionTest,
                         ::testing::Values(synDeviceGaudi2, synDeviceGaudi3),
                         GenericGraphTest::GetName());
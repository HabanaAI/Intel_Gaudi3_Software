#include "graph_optimizer_test.h"
#include <node_factory.h>
#include <gaudi_graph.h>

#include "tpc_node.h"
#include "dedw_node.h"
#include "dedx_node.h"
#include "platform/gaudi/graph_compiler/passes.h"

class GaudiNodeGraphValidationTest : public GraphOptimizerTest {};

TEST_F(GaudiNodeGraphValidationTest, tpc_kernel_invalid_for_gaudi)
{
    GaudiGraph graph;

    const char *invalidGuid = "asdfdsa";
    TensorVector invalidInputs;
    TensorVector invalidOutputs;
    std::nullptr_t invalidUserParams = nullptr;

    NodePtr invalidKernelNode =
        NodeFactory::createGenericTPCNode(invalidInputs, invalidOutputs, invalidUserParams, invalidGuid);
    ASSERT_TRUE(invalidKernelNode) << "Failed to create node";
    ASSERT_FALSE(GraphEditor::addNode(graph, invalidKernelNode))
        << "Trying to add the node to the graph was supposed to fail";
}

TEST_F(GaudiNodeGraphValidationTest, tpc_kernel_valid_for_gaudi)
{
    GaudiGraph graph;

    const char* validGuid = "relu_fwd_f32";
    TensorVector validInputs;
    TensorVector validOutputs;
    std::nullptr_t validUserParams = nullptr;

    NodePtr validKernelNode = NodeFactory::createGenericTPCNode(validInputs, validOutputs, validUserParams, validGuid);
    ASSERT_TRUE(validKernelNode) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(graph, validKernelNode)) << "Failed to add node to graph";
}

TEST_F(GaudiNodeGraphValidationTest, beam_search_valid_for_gaudi)
{
    GaudiGraph    graph;
    synBeamParams params;
    const TSize   i_sizes[] = {2, 2, 8200};
    const TSize   validCountSizes[] = {2, 2};
    TensorPtr     scores    = std::make_shared<Tensor>(3U, i_sizes, syn_type_float);
    TensorPtr     indices    = std::make_shared<Tensor>(3U, i_sizes, syn_type_int32);
    TensorPtr     validCount    = std::make_shared<Tensor>(2U, validCountSizes, syn_type_int32);

    NodePtr node   = NodeFactory::createNode({scores, indices, validCount}, TensorVector(), &params, NodeFactory::beamSearchNodeTypeName,"");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(graph, node)) << "Failed to add node to graph";
}

TEST_F(GaudiNodeGraphValidationTest, concat_valid_for_gaudi)
{
    GaudiGraph graph;
    TSize isize[] = { 1 , 2 };
    TSize osize[] = { 2 , 2 };

    TensorPtr in1 = std::make_shared<Tensor>(2U, isize, syn_type_fixed);
    TensorPtr in2 = std::make_shared<Tensor>(2U, isize, syn_type_fixed);
    TensorPtr out = std::make_shared<Tensor>(2U, osize, syn_type_fixed);
    NodePtr   node = NodeFactory::createNode({ in1, in2 }, {out}, nullptr, NodeFactory::concatenateNodeTypeName, "");

    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(graph, node)) << "Failed to add node to graph";
}

TEST_F(GaudiNodeGraphValidationTest, reduction_valid_for_gaudi)
{
    GaudiGraph graph;
    TensorPtr  in        = std::make_shared<Tensor>(syn_type_bf16);
    TensorPtr  out       = std::make_shared<Tensor>(syn_type_bf16);
    NodePtr    node      = NodeFactory::createNode({in}, {out}, nullptr, NodeFactory::reductionNodeTypeName, "");

    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(graph, node)) << "Failed to add node to graph";
}

TEST_F(GaudiNodeGraphValidationTest, non_valid_mme_node)
{
    GaudiGraph g;
    synConvolutionParams params;
    const TSize kW   = 1, kH = 5, dW = 1, dH =1;
    const TSize nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = { nIFM, wIFM, hIFM, 1 };
    const TSize o_sizes[] = { nOFM, wOFM, hOFM, 1 };
    const TSize w_sizes[] = { nOFM, nIFM, kW, kH };
    const TSize b_sizes[] = { 1 };

    TensorPtr    ifm    = std::make_shared<Tensor>(4U, i_sizes, syn_type_bf16);
    TensorPtr    weight = std::make_shared<Tensor>(4U, w_sizes, syn_type_bf16);
    TensorPtr    bias   = std::make_shared<Tensor>(1U, b_sizes, syn_type_bf16);
    TensorPtr    ofm    = std::make_shared<Tensor>(4U, o_sizes, syn_type_bf16);
    TensorVector inputs {ifm, weight, bias};
    TensorVector outputs {ofm};

    NodePtr node = NodeFactory::createNode(inputs, outputs, &params, "spatial_convolution", "");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "Failed to add node to graph";
    // pass should fail since we got MME node with bias.
    ASSERT_FALSE(validateMMENodes(g)) << "Running validateMMENodes pass should fail";
}

TEST_F(GaudiNodeGraphValidationTest, valid_mme_node_with_bias)
{
    GaudiGraph g;
    synConvolutionParams params;
    const TSize kW   = 1, kH = 5, dW = 1, dH =1;
    const TSize nOFM = 1, wOFM = 5, hOFM = 5, nIFM = 1;
    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    const TSize i_sizes[] = { nIFM, wIFM, hIFM, 1 };
    const TSize o_sizes[] = { nOFM, wOFM, hOFM, 1 };
    const TSize w_sizes[] = { nOFM, nIFM, kW, kH };
    const TSize b_sizes[] = { 1 };

    TensorPtr    ifm    = std::make_shared<Tensor>(4U, i_sizes, syn_type_float);
    TensorPtr    weight = std::make_shared<Tensor>(4U, w_sizes, syn_type_float);
    TensorPtr    bias   = std::make_shared<Tensor>(1U, b_sizes, syn_type_float);
    TensorPtr    ofm    = std::make_shared<Tensor>(4U, o_sizes, syn_type_float);
    TensorVector inputs {ifm, weight, bias};
    TensorVector outputs {ofm};

    NodePtr node = NodeFactory::createNode(inputs, outputs, &params, "spatial_convolution", "");
    ASSERT_TRUE(node) << "Failed to create node";
    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "Failed to add node to graph";
    // pass should fail since we got MME node with bias.
    ASSERT_TRUE(addMmeBias(g)) << "Failed to detach bias from mme node";
    ASSERT_TRUE(validateMMENodes(g));

    const auto& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.front(), node) << "Expectes the 1st node to be the convolution";
    ASSERT_EQ(nodes.front()->getInput(TENSOR_BIAS), nullptr);
    ASSERT_EQ(nodes.back()->getGUID(), "add_fwd_f32") << "Expects an add node to be added";
    ASSERT_EQ(nodes.back()->getOutput(0), ofm);
}

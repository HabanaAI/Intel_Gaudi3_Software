#include "habana_nodes.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "sram_management_fe_test.h"
#include "graph_compiler/passes/sram_management/flatten_mme.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "graph_compiler/passes/sram_management/slicing_utils.h"
#include "scoped_configuration_change.h"

namespace gaudi
{
class FlattenMMETest : public SRAMManagementTest
{
protected:
    NodePtr
    createAndAddConvNode(TensorPtr& ifm, TensorPtr& weights, TensorPtr& output, synConvolutionParams* params = nullptr)
    {
        synConvolutionParams defaultConvParams;
        if (!params) params = &defaultConvParams;
        NodePtr node =
            NodeFactory::createNode({ifm, weights}, {output}, params, NodeFactory::convolutionNodeTypeName, "conv");
        GraphEditor::addNode(getGraph(), node);
        return node;
    }
    NodePtr createAndAddReshapeNode(TensorPtr& input, TensorPtr& output)
    {
        NodePtr node = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
        GraphEditor::addNode(getGraph(), node);
        return node;
    }

    GaudiGraph& getGraph() {return m_graph;}
private:
    GaudiGraph      m_graph;
};

struct FlattenDynamicMMETestParams
{
    TSize              m, k, n, in0Batch1, in0Batch2;
    std::vector<TSize> in0minSizes;
    std::vector<TSize> in1minSizes;
    std::vector<TSize> outminSizes;
    bool               supported;
};

class FlattenDynamicMMETest
: public SRAMManagementTest
, public testing::WithParamInterface<FlattenDynamicMMETestParams>
{
public:
protected:
    NodePtr createAndAddBgemmNode(TensorPtr& in1, TensorPtr& in2, TensorPtr& out)
    {
        NodePtr node = NodeFactory::createNode({in1, in2}, {out}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
        GraphEditor::addNode(getGraph(), node);
        return node;
    }
    GaudiGraph& getGraph() { return m_graph; }

protected:
    GaudiGraph m_graph;
};

TEST_F(FlattenMMETest, pass_should_flatten_input_tensor)
{
    ScopedConfigurationChange flattenConv("ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "true");

    /* Pass should flatten the input tensor of the node as kernel is 1x1
     * output tensor of the node should be flatten as well
     */
    SizeArray inputShape = {1024, 256, 256, 256};
    SizeArray weightShape = {256, 1024, 1, 1};
    SizeArray outputShape = {256, 256, 256, 256};

    pTensor ifm = std::make_shared<Tensor>(4U, inputShape.data(), syn_type_float);
    pTensor weights = std::make_shared<Tensor>(4U, weightShape.data(), syn_type_float);
    pTensor ofm = std::make_shared<Tensor>(4U, outputShape.data(), syn_type_float);

    pNode node = createAndAddConvNode(ifm, weights, ofm);
    // execute pass
    MMENodeFlattener flattener(getGraph());
    flattener.execute();

    ASSERT_TRUE(SlicedOperandUtils::isTensor2D(node->getInput(0)));
    ASSERT_TRUE(SlicedOperandUtils::isTensor2D(node->getOutput(0)));
}

TEST_F(FlattenMMETest, flattenMME_pass_should_not_flatten_input_tensor_with_kernel)
{
    ScopedConfigurationChange flattenConv("ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "true");

    /* Pass should not flatten tensors of node that is not convertible to gemm
     * i.e. kernel size > 1x1
     */
    SizeArray inputShape = {1024, 256, 256, 256};
    SizeArray weightShape = {256, 1024, 3, 3};
    SizeArray outputShape = {256, 256, 256, 256};
    synConvolutionParams params;
    params.kH = 3; params.kW = 3;
    pTensor ifm = std::make_shared<Tensor>(4U, inputShape.data(), syn_type_float);
    pTensor weights = std::make_shared<Tensor>(4U, weightShape.data(), syn_type_float);
    pTensor ofm = std::make_shared<Tensor>(4U, outputShape.data(), syn_type_float);

    pNode node = createAndAddConvNode(ifm, weights, ofm, &params);
    // execute pass
    MMENodeFlattener flattener(getGraph());
    flattener.execute();

    ASSERT_FALSE(SlicedOperandUtils::isTensor2D(node->getInput(0)));
    ASSERT_FALSE(SlicedOperandUtils::isTensor2D(node->getOutput(0)));
}

TEST_F(FlattenMMETest, flattenMME_pass_should_choose_already_flatten_tensor)
{
    ScopedConfigurationChange flattenConv("ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "true");

    /* in this case there is a flatten tensor in the graph.
     * We expect the pass to connect the already flatten tensor to the conv input
     * instead of creating a reshape node.
     */
    SizeArray inputShape = {1024, 256, 256, 256};
    SizeArray flattenInputShape {1024, 256*256*256, 1, 1};
    SizeArray weightShape = {256, 1024, 1, 1};
    SizeArray outputShape = {256, 256, 256, 256};

    pTensor ifm = std::make_shared<Tensor>(4U, inputShape.data(), syn_type_float);
    pTensor flattenIfm = std::make_shared<Tensor>(4U, flattenInputShape.data(), syn_type_float);
    pTensor weights = std::make_shared<Tensor>(4U, weightShape.data(), syn_type_float);
    pTensor ofm = std::make_shared<Tensor>(4U, outputShape.data(), syn_type_float);

    pNode node = createAndAddConvNode(ifm, weights, ofm);
    pNode reshapeNode = createAndAddReshapeNode(ifm, flattenIfm);
    // execute pass
    MMENodeFlattener flattener(getGraph());
    flattener.execute();

    ASSERT_TRUE(SlicedOperandUtils::isTensor2D(node->getInput(0)));
    ASSERT_TRUE(SlicedOperandUtils::isTensor2D(node->getOutput(0)));
    ASSERT_TRUE(node->getInput(0) == reshapeNode->getOutput(0));
}

TEST_F(FlattenMMETest, flattenMME_pass_should_not_choose_2D_flatten_tensor)
{
    ScopedConfigurationChange flattenConv("ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "true");

    /* Pass should not pick the flatten tensor as the no. of dims is different then expected
     * it should create a different reshape
     */
    SizeArray inputShape = {1024, 256, 256, 256};
    SizeArray flattenInputShape {1024, 256*256*256};
    SizeArray weightShape = {256, 1024, 1, 1};
    SizeArray outputShape = {256, 256, 256, 256};

    pTensor ifm = std::make_shared<Tensor>(4U, inputShape.data(), syn_type_float);
    pTensor flattenIfm_2D = std::make_shared<Tensor>(2U, flattenInputShape.data(), syn_type_float);
    pTensor weights = std::make_shared<Tensor>(4U, weightShape.data(), syn_type_float);
    pTensor ofm = std::make_shared<Tensor>(4U, outputShape.data(), syn_type_float);

    pNode node = createAndAddConvNode(ifm, weights, ofm);
    pNode reshapeNode = createAndAddReshapeNode(ifm, flattenIfm_2D);
    // execute pass
    MMENodeFlattener flattener(getGraph());
    flattener.execute();

    ASSERT_TRUE(node->getInput(0) != reshapeNode->getOutput(0));
    ASSERT_TRUE(SlicedOperandUtils::isTensor2D(node->getInput(0)));
}

TEST_P(FlattenDynamicMMETest, dynamic_bgemm_flatten)
{
    std::vector<TSize> in0Sizes({GetParam().k, GetParam().m, GetParam().in0Batch1, GetParam().in0Batch2});
    std::vector<TSize> in1Sizes({GetParam().n, GetParam().k, 1, 1});  // Full boradcast
    std::vector<TSize> in0MinSizes(GetParam().in0minSizes);
    std::vector<TSize> in1MinSizes(GetParam().in1minSizes);
    std::vector<TSize> outSizes({GetParam().m, GetParam().n, GetParam().in0Batch1, GetParam().in0Batch2});
    std::vector<TSize> outMinSizes({GetParam().outminSizes});
    bool               supported = GetParam().supported;

    TensorPtr in1 = createTensor(in0Sizes, syn_type_bf16, true, in0MinSizes);
    TensorPtr in2 = createTensor(in1Sizes, syn_type_bf16, true, in1MinSizes);
    TensorPtr out = createTensor(outSizes, syn_type_bf16, true, outMinSizes);

    auto node = std::static_pointer_cast<BatchGemmNode>(createAndAddBgemmNode(in1, in2, out));
    ASSERT_EQ(node->canBeConvertedToGEMM(), supported);
}

INSTANTIATE_TEST_SUITE_P(dynamic_bgemm_flatten_test,
                         FlattenDynamicMMETest,
                         // m,k,n,in0Batch1,inBatch2, in0 minSize, in1 minSize,out minSize supported
                         ::testing::Values(
                             // Only outer dim is dynamic
                             FlattenDynamicMMETestParams {128,                  // M
                                                          1024,                 // K
                                                          1024,                 // N
                                                          16,                   // Batch1
                                                          32,                   // Batch2
                                                          {1024, 128, 16, 16},  // in0 min
                                                          {1024, 1024, 1, 1},   // in1 min
                                                          {1024, 128, 16, 16},  // out min
                                                          true},
                             // Two outer batch dims are dynamic
                             FlattenDynamicMMETestParams {128,
                                                          1024,
                                                          1024,
                                                          16,
                                                          32,
                                                          {1024, 128, 8, 16},
                                                          {1024, 1024, 1, 1},
                                                          {1024, 128, 8, 16},
                                                          false},
                             // Inner dim is dynamic
                             FlattenDynamicMMETestParams {128,
                                                          1024,
                                                          1024,
                                                          16,
                                                          32,
                                                          {1024, 64, 16, 32},
                                                          {1024, 1024, 1, 1},
                                                          {1024, 64, 16, 32},
                                                          false},
                             // Inner dim and outer dims are dynamic
                             FlattenDynamicMMETestParams {128,
                                                          1024,
                                                          1024,
                                                          16,
                                                          32,
                                                          {1024, 32, 16, 16},
                                                          {1024, 1024, 1, 1},
                                                          {1024, 32, 16, 16},
                                                          false}));
}  // namespace gaudi
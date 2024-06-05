#include <gtest/gtest.h>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "data_type_utils.h"
#include "scoped_configuration_change.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include <graph_compiler/habana_nodes/node_factory.h>

class Gaudi2TPCWDTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<std::vector<unsigned>, const char*, synDataType>>
{
};

TEST_P(Gaudi2TPCWDTest, DISABLED_dynamic_tpc_wd)
{
    // SW-75244: To Be Developed. Alternative: instead of unit test, we can have a synapse_test that
    // ensures the number of working engines is as expected, so it's testing the box size indirectly.
}

INSTANTIATE_TEST_SUITE_P(
    _,
    Gaudi2TPCWDTest,
    ::testing::Values(std::make_tuple(std::vector<unsigned int> {256, 32, 1, 1, 20}, "relu_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {256, 7, 1, 1, 12}, "relu_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {128, 128, 1, 1, 15}, "relu_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {100, 2, 3, 4, 1}, "relu_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {22, 24, 5, 1, 1}, "add_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {256, 2, 1, 1, 24}, "add_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {128, 32, 1, 1, 10}, "add_fwd_f32", syn_type_single),
                      std::make_tuple(std::vector<unsigned int> {5, 5, 5, 5, 5}, "add_fwd_f32", syn_type_single)));


TEST_F(GraphOptimizerTest, mandatory_first_split_dim_for_tpc)
{
    Gaudi2Graph g;

    const unsigned dims    = 5;
    const TSize    size1[] = {2304, 384, 6, 12, 4};
    const TSize    size2[] = {1, 384, 6, 12, 4};
    const int      param[] = {0,0,181,61,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0};

    TensorPtr in   = TensorPtr(new Tensor(dims, size1, syn_type_bf16));
    TensorPtr out1 = TensorPtr(new Tensor(dims, size1, syn_type_bf16));
    TensorPtr out2 = TensorPtr(new Tensor(dims, size2, syn_type_bf16));
    TensorPtr out3 = TensorPtr(new Tensor(dims, size2, syn_type_bf16));

    synMemoryDescriptor memDesc(true);
    in->setMemoryDescriptor(memDesc);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    out1->setMemoryDescriptor(memDesc);
    out1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    out2->setMemoryDescriptor(memDesc);
    out2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    out3->setMemoryDescriptor(memDesc);
    out3->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);

    NodePtr tpcNode = NodeFactory::createNode(
        {in},
        {out1, out2, out3},
        (UserParams)param,
        sizeof(param)/sizeof(int),
        "scaled_masked_triangular_softmax_fwd_bf16", // this guid has mandatory first split dim
        "node_name1");

    GraphEditor::addNode(g, tpcNode);
    ASSERT_EQ(g.compile(), true);
    std::shared_ptr<TPCNode> compiledNode = std::dynamic_pointer_cast<TPCNode>(g.getExeSortedNodes().front());
    ASSERT_TRUE(compiledNode != nullptr);
    ASSERT_TRUE(compiledNode->hasMandatorySplitDim());
    unsigned firstSplitDim = compiledNode->getMandatorySplitDim();
    std::list<NodeROI>* rois = g.GetNodeROIs(tpcNode);
    const TpcWdCtx& ctx = rois->front().tpcWdCtx.front();
    // we expect that after splitting, the mandatory dim will be 1, so every engine gets work for that dim
    ASSERT_EQ(ctx.boxSize[firstSplitDim], 1);
}

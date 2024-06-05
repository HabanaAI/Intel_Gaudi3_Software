#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "tpc_node.h"
#include "habana_global_conf.h"
#include "perf_lib_layer_params.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include "tile_utils.hpp"
#include "broadcast_node.h"

class TileOptimizationTest
: public GraphOptimizerTest
, public TileTestUtilsBase<SizeVector>
{
    void SetUp()
    {
        setGlobalConfForTest(GCFG_MAKE_BROADCAST_PHYSICAL, "false");
        GraphOptimizerTest::SetUp();
    }
};
class TileReplaceTest : public TileOptimizationTest
{
};
class TileNoReplaceTest : public TileOptimizationTest
{
};

TEST_P(TileReplaceTest, tile_replace_broadcast)
{
    GaudiGraph   g;
    auto         sizesIn  = ::testing::get<0>(GetParam());
    auto         sizesOut = ::testing::get<1>(GetParam());
    unsigned int dimIn    = sizesIn.size();
    unsigned int dimOut   = sizesOut.size();

    pTensor             tAddIn1(new Tensor(dimIn, sizesIn.data(), syn_type_float, nullptr, nullptr, false, true));
    pTensor             tAddIn2(new Tensor(dimIn, sizesIn.data(), syn_type_float, nullptr, nullptr, false, true));
    pTensor             tAddOut(new Tensor(dimIn, sizesIn.data(), syn_type_float));
    const TensorVector& addInputs {tAddIn1, tAddIn2};
    const TensorVector& addOutputs {tAddOut};

    ns_TileKernel::ParamsV2 params;
    calculateTileParams(sizesIn, sizesOut, params);

    NodePtr nodeAddFwd = NodeFactory::createNode(addInputs, addOutputs, nullptr, "add_f32", "add");

    GraphEditor::addNode(g, nodeAddFwd);
    synMemoryDescriptor add_fwd_memDesc(true);
    tAddIn1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tAddIn2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tAddIn1->setMemoryDescriptor(add_fwd_memDesc);
    tAddIn2->setMemoryDescriptor(add_fwd_memDesc);

    pTensor tTileOut(new Tensor(dimOut, sizesOut.data(), syn_type_float));

    const TensorVector& tileInputs {tAddOut};
    const TensorVector& tileOutputs {tTileOut};

    NodePtr nodeTileFwd = NodeFactory::createNode(tileInputs, tileOutputs, &params, "tile_fwd_f32", "tile");
    GraphEditor::addNode(g, nodeTileFwd);

    pTensor                  tReluOut(new Tensor(dimOut, sizesOut.data(), syn_type_float, nullptr, nullptr, true));
    const TensorVector&      reluInputs {tTileOut};
    const TensorVector&      reluOutputs {tReluOut};
    NodePtr nodeReluFwd = NodeFactory::createNode(reluInputs, reluOutputs, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(g, nodeReluFwd);

    tReluOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tReluOut->setMemoryDescriptor(add_fwd_memDesc);

    ASSERT_TRUE(g.compile()) << "compilation failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    auto validatedBroadcast = false;
    for (const NodePtr& node : nodes)
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_BROADCAST)
        {
            std::shared_ptr<LogicalBroadcastNode> broadcastNode = std::dynamic_pointer_cast<LogicalBroadcastNode>(node);

            ASSERT_NE(broadcastNode, nullptr);
            // node name can be modified by complex guid lib, so we lookup original name as substring
            ASSERT_NE(broadcastNode->getNodeName().find("tile"), std::string::npos);

            ASSERT_EQ(broadcastNode->getNumInputs(), 1);
            ASSERT_EQ(broadcastNode->getNumOutputs(), 1);

            ASSERT_EQ(node->getOutput(0)->getDim(), sizesOut.size());

            uint64_t productOfStrides = 1;
            for (unsigned i = 0; i < dimOut; i++)
            {
                productOfStrides *= node->getOutput(0)->getStrideInElements(i + 1);
            }

            ASSERT_EQ(0, productOfStrides);

            validatedBroadcast = true;
        }
    }

    ASSERT_TRUE(validatedBroadcast);
}

TEST_P(TileNoReplaceTest, tile_no_replace_broadcast)
{
    GaudiGraph   g;
    auto         sizesIn  = ::testing::get<0>(GetParam());
    auto         sizesOut = ::testing::get<1>(GetParam());
    unsigned int dim      = sizesOut.size();

    pTensor             tAddIn1(new Tensor(dim, sizesIn.data(), syn_type_float, nullptr, nullptr, false, true));
    pTensor             tAddIn2(new Tensor(dim, sizesIn.data(), syn_type_float, nullptr, nullptr, false, true));
    pTensor             tAddOut(new Tensor(dim, sizesIn.data(), syn_type_float));
    const TensorVector& addInputs {tAddIn1, tAddIn2};
    const TensorVector& addOutputs {tAddOut};

    ns_TileKernel::ParamsV2 params;
    calculateTileParams(sizesIn, sizesOut, params);

    NodePtr nodeAddFwd = NodeFactory::createNode(addInputs, addOutputs, nullptr, "add_f32", "add");

    GraphEditor::addNode(g, nodeAddFwd);
    synMemoryDescriptor add_fwd_memDesc(true);
    tAddIn1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tAddIn2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tAddIn1->setMemoryDescriptor(add_fwd_memDesc);
    tAddIn2->setMemoryDescriptor(add_fwd_memDesc);

    pTensor tTileOut(new Tensor(dim, sizesOut.data(), syn_type_float));

    const TensorVector& tileInputs {tAddOut};
    const TensorVector& tileOutputs {tTileOut};

    NodePtr nodeTileFwd = NodeFactory::createNode(tileInputs, tileOutputs, &params, "tile_fwd_f32", "tile");
    GraphEditor::addNode(g, nodeTileFwd);

    pTensor                  tReluOut(new Tensor(dim, sizesOut.data(), syn_type_float, nullptr, nullptr, true));
    const TensorVector&      reluInputs {tTileOut};
    const TensorVector&      reluOutputs {tReluOut};
    NodePtr nodeReluFwd = NodeFactory::createNode(reluInputs, reluOutputs, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(g, nodeReluFwd);

    tReluOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tReluOut->setMemoryDescriptor(add_fwd_memDesc);

    ASSERT_TRUE(g.compile()) << "compilation failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    auto validatedTile = false;
    for (const NodePtr& node : nodes)
    {
        // Check that tile node exist based on prefix
        if (node->getNodeName().rfind("tile", 0) == 0)
        {
            std::shared_ptr<TPCNode> tileNode = std::dynamic_pointer_cast<TPCNode>(node);

            ASSERT_NE(tileNode, nullptr);
            ASSERT_EQ(tileNode->getGUID(), "tile_fwd_f32");

            validatedTile = true;
        }
    }

    ASSERT_TRUE(validatedTile);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TileNoReplaceTest,
    ::testing::Values(::testing::make_tuple(SizeVector {1}, SizeVector {4}),
                      ::testing::make_tuple(SizeVector {1, 1}, SizeVector {4, 4}),
                      ::testing::make_tuple(SizeVector {1, 4}, SizeVector {4, 4}),
                      ::testing::make_tuple(SizeVector {1, 4, 4}, SizeVector {1, 4, 4}),
                      ::testing::make_tuple(SizeVector {1, 2, 3, 4}, SizeVector {4, 2, 3, 4}),
                      ::testing::make_tuple(SizeVector {1, 2, 1, 2}, SizeVector {3, 2, 3, 2}),
                      ::testing::make_tuple(SizeVector {1, 16, 1, 16}, SizeVector {16, 16, 16, 16}),
                      ::testing::make_tuple(SizeVector {1, 1, 1, 1, 1}, SizeVector {2, 2, 2, 2, 2}),
                      ::testing::make_tuple(SizeVector {1, 2, 1, 4, 1}, SizeVector {2, 2, 3, 4, 5})),
    TileNoReplaceTest::GetName());

INSTANTIATE_TEST_SUITE_P(
    ,
    TileReplaceTest,
    ::testing::Values(::testing::make_tuple(SizeVector {4, 1}, SizeVector {4, 4}),
                      ::testing::make_tuple(SizeVector {4, 1}, SizeVector {4, 4, 5}),
                      ::testing::make_tuple(SizeVector {2, 1, 2, 2}, SizeVector {2, 2, 2, 2}),
                      ::testing::make_tuple(SizeVector {2, 1, 1, 1}, SizeVector {2, 3, 4, 5}),
                      ::testing::make_tuple(SizeVector {2, 1, 2, 1}, SizeVector {2, 3, 2, 3}),
                      ::testing::make_tuple(SizeVector {3, 1, 1, 1}, SizeVector {3, 2, 2, 2, 2}),
                      ::testing::make_tuple(SizeVector {3, 1, 1, 1, 1}, SizeVector {3, 2, 2, 2, 2}),
                      ::testing::make_tuple(SizeVector {1, 2, 1, 4, 1}, SizeVector {1, 2, 3, 4, 5})),
    TileReplaceTest::GetName());
#include "graph_optimizer_test.h"

#include "gaudi2_graph.h"
#include "generic_graph_test.h"
#include "node_factory.h"
#include "tensor.h"
#include "types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

bool eliminateRedundantNodes(HabanaGraph& g);

namespace
{
class EliminateRedundantNodeTest : public GenericGraphTest
{
public:
    void setPersistent(TensorPtr& tensor)
    {
        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000);
    }
};
}  // anonymous namespace

TEST_P(EliminateRedundantNodeTest, eliminate_redundant_nodes_remove_cast_node)
{
    /*
     * Original graph has 3 nodes - gemm, gelu and cast.
     *
     * The eliminateRedundantNodes pass should remove the cast node because its output is non-persistent,
     * and doesn't have a consumer.
     * The other two node have a persistent output (gelu) or consumed output (gemm, by gelu),
     * and therefore are not removed.
     */
    SizeArray sizes    = {2048, 2048};
    auto      gemmIn1  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      gemmIn2  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      gemmOut  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      geluOut1 = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      geluOut2 = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    setPersistent(geluOut2);

    synGEMMParams gemmParams {};
    NodePtr       gemm =
        NodeFactory::createNode({gemmIn1, gemmIn2}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    NodePtr gelu = NodeFactory::createNode({gemmOut}, {geluOut1, geluOut2}, nullptr, 0, "gelu_fwd_f32", "GELU");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gelu)) << "Failed to add node to graph";

    auto    castIn  = std::make_shared<Tensor>(2, sizes.data(), syn_type_bf16);
    auto    castOut = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    NodePtr cast    = NodeFactory::createNode({castIn}, {castOut}, nullptr, 0, "cast_bf16_to_f32", "CAST");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, cast)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNodes().size(), 3) << "Expecting three nodes in pre-pass graph";
    bool ret = eliminateRedundantNodes(*m_graph);
    ASSERT_EQ(ret, true) << "Failed to run pass eliminateRedundantNodes";
    ASSERT_EQ((*m_graph).getNodes().size(), 2) << "Expecting two nodes in graph";
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        ASSERT_EQ(((node != nullptr) && (!node->isCast())), true) << "found unexpected cast node in the graph";
    }
}

TEST_P(EliminateRedundantNodeTest, eliminate_redundant_nodes_remove_all_nodes_except_one)
{
    /*
     * Original graph has 3 nodes - gemm, gelu and cast.
     *
     * The eliminateRedundantNodes pass should remove the cast and gelu nodes:
     *  cast is removed because its output is non-persistent, and isn't consumed
     *  gelu is then removed since its output is non-persistent and no longer consumed
     *  (was previously consumed by the cast node that was removed).
     *  The gemm node also has a non-persistent output which is no longer consumed,
     *  but it is the last node in the graph and therefore can't be removed.
     */
    SizeArray sizes    = {2048, 2048};
    auto      gemmIn1  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      gemmIn2  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      gemmOut  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      geluOut1 = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      geluOut2 = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      castOut  = std::make_shared<Tensor>(2, sizes.data(), syn_type_bf16);

    synGEMMParams gemmParams {};
    NodePtr       gemm =
        NodeFactory::createNode({gemmIn1, gemmIn2}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    NodePtr gelu = NodeFactory::createNode({gemmOut}, {geluOut1, geluOut2}, nullptr, 0, "gelu_fwd_f32", "GELU");
    NodePtr cast = NodeFactory::createNode({geluOut2}, {castOut}, nullptr, 0, "cast_f32_to_bf16", "CAST");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gelu)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, cast)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNodes().size(), 3) << "Expecting three nodes in pre-pass graph";
    bool ret = eliminateRedundantNodes(*m_graph);
    ASSERT_EQ(ret, true) << "Failed to run pass eliminateRedundantNodes";
    ASSERT_EQ((*m_graph).getNodes().size(), 1) << "Expecting one node in graph";
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        ASSERT_EQ(((node != nullptr) && (Node::isGemmNode(node))), true) << "found unexpected node in the graph";
    }
}

TEST_P(EliminateRedundantNodeTest, eliminate_redundant_nodes_dont_remove_nodes)
{
    /*
     * Original graph has 3 nodes - gemm, gelu and cast.
     *
     * The eliminateRedundantNodes pass should not remove any node, since they have a persistent output (cast)
     * or consumed output (gemm's output is consumed by gelu, gelu's output is consumed by cast).
     */
    SizeArray sizes    = {2048, 2048};
    auto      gemmIn1  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      gemmIn2  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      gemmOut  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      geluOut1 = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      geluOut2 = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      castOut  = std::make_shared<Tensor>(2, sizes.data(), syn_type_bf16);
    setPersistent(castOut);

    synGEMMParams gemmParams {};
    NodePtr       gemm =
        NodeFactory::createNode({gemmIn1, gemmIn2}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    NodePtr gelu = NodeFactory::createNode({gemmOut}, {geluOut1, geluOut2}, nullptr, 0, "gelu_fwd_f32", "GELU");
    NodePtr cast = NodeFactory::createNode({geluOut2}, {castOut}, nullptr, 0, "cast_f32_to_bf16", "CAST");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gelu)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, cast)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNodes().size(), 3) << "Expecting three nodes in pre-pass graph";
    bool ret = eliminateRedundantNodes(*m_graph);
    ASSERT_EQ(ret, true) << "Failed to run pass eliminateRedundantNodes";
    ASSERT_EQ((*m_graph).getNodes().size(), 3) << "Expecting three nodes in graph";
}

TEST_P(EliminateRedundantNodeTest, eliminate_redundant_nodes_dont_remove_nodes_with_rmw_output)
{
    /*
     * Original graph has 4 nodes - broadcast, cast, memcpy and memset
     *
     * The eliminateRedundantNodes pass should not remove any node, since they have a persistent output (memcpy),
     * an RMW output (memset), or consumed output (broadcast's output is consumed by cast, cast's output is consumed by
     * memcpy).
     */
    SizeArray sizes        = {2048, 2048};
    auto      broadcastIn  = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      broadcastOut = std::make_shared<Tensor>(2, sizes.data(), syn_type_float);
    auto      castOut      = std::make_shared<Tensor>(2, sizes.data(), syn_type_bf16);
    auto      memcpyOut    = std::make_shared<Tensor>(2, sizes.data(), syn_type_bf16);
    auto      memsetOut    = std::make_shared<Tensor>(2, sizes.data(), syn_type_bf16);

    uint64_t rmwSectionId = (*m_graph).getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);

    memsetOut->setTensorInSram();
    auto& nonPersistentSectionInfoMemset = memsetOut->getTensorAnnotation().nonPersistentSectionInfo;
    nonPersistentSectionInfoMemset.sectionId.set(rmwSectionId);
    nonPersistentSectionInfoMemset.offsetFromBase.set(0);

    castOut->setTensorInSram();
    auto& nonPersistentSectionInfoCast = castOut->getTensorAnnotation().nonPersistentSectionInfo;
    nonPersistentSectionInfoCast.sectionId.set(rmwSectionId);
    nonPersistentSectionInfoCast.offsetFromBase.set(0);

    setPersistent(memcpyOut);

    NodePtr broadcast = NodeFactory::createNode({broadcastIn},
                                                {broadcastOut},
                                                nullptr,
                                                NodeFactory::broadcastNodeTypeName,
                                                "BROADCAST");
    NodePtr cast      = NodeFactory::createNode({broadcastOut}, {castOut}, nullptr, 0, "cast_f32_to_bf16", "CAST");
    NodePtr memcpy =
        NodeFactory::createNode({castOut}, {memcpyOut}, nullptr, 0, NodeFactory::memcpyNodeTypeName, "MEMCPY");
    NodePtr memset = NodeFactory::createNode({}, {memsetOut}, nullptr, 0, NodeFactory::memsetNodeTypeName, "MEMSET");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, broadcast)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, cast)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, memcpy)) << "Failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, memset)) << "Failed to add node to graph";

    ASSERT_EQ((*m_graph).getNodes().size(), 4) << "Expecting four nodes in pre-pass graph";
    bool ret = eliminateRedundantNodes(*m_graph);
    ASSERT_EQ(ret, true) << "Failed to run pass eliminateRedundantNodes";
    ASSERT_EQ((*m_graph).getNodes().size(), 4) << "Expecting four nodes in graph";
}

INSTANTIATE_TEST_SUITE_P(,
                         EliminateRedundantNodeTest,
                         ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3),
                         EliminateRedundantNodeTest::GetName());

namespace
{
class EliminateRedundantNodesTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::pair<bool, size_t>>
{
protected:
    void setPersistent(TensorPtr& tensor)
    {
        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000);
    }
    Gaudi2Graph m_graph;
};
}  // anonymous namespace

TEST_P(EliminateRedundantNodesTest, eliminate_redundant_nodes_test)
{
    // isFlat: if true as all RELU's have the same input, otherwise chained RELU's.
    // keptNodes: number of kept nodes.
    auto [isFlat, keptNodes] = GetParam();

    constexpr std::array<TSize, 1> sizes             = {1};
    constexpr synDataType          synDataTypeFlavor = syn_type_bf16;

    constexpr size_t nodesNumber = 10;
    keptNodes                    = std::min(keptNodes, nodesNumber);

    std::vector<TensorPtr> tensorsArray(nodesNumber + 1);
    for (int i = 0; i < tensorsArray.size(); i++)
    {
        tensorsArray[i] = std::make_shared<Tensor>(sizes.size(), sizes.data(), synDataTypeFlavor);
        if (isFlat)
        {
            // Always want the root tensor to be persistent.
            // And all nodes except for the ones to be removed to have persistent outputs.
            if (i <= keptNodes)
            {
                setPersistent(tensorsArray[i]);
            }
        }
    }
    // Test the case of chained nodes with a single persistent output in the middle,
    // since the nodes up to it should be kept and the rest will to be removed.
    if (!isFlat) setPersistent(tensorsArray[keptNodes]);

    for (int i = 1; i < tensorsArray.size(); ++i)
    {
        NodePtr reluNode = NodeFactory::createNode({isFlat ? tensorsArray[0] : tensorsArray[i - 1]},
                                                   {tensorsArray[i]},
                                                   nullptr,
                                                   "relu_fwd_bf16",
                                                   fmt::format("relu_{}", i));
        ASSERT_TRUE(GraphEditor::addNode(m_graph, reluNode));
    }

    ASSERT_EQ(m_graph.getNumNodes(), nodesNumber);
    eliminateRedundantNodes(m_graph);

    // Nodes with persistent outputs are kept. And we don't leave an empty graph so one node is kept.
    ASSERT_EQ(m_graph.getNumNodes(), std::clamp(keptNodes, size_t {1}, nodesNumber));
}

INSTANTIATE_TEST_SUITE_P(,
                         EliminateRedundantNodesTest,
                         testing::Values(std::make_pair(false, 0),
                                         std::make_pair(false, 7),
                                         std::make_pair(false, 15),
                                         std::make_pair(true, 0),
                                         std::make_pair(true, 6),
                                         std::make_pair(true, 12)));
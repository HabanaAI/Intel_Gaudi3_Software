#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "habana_nodes.h"
#include "gaudi_graph.h"
#include "test_utils.h"
#include "syn_logging.h"
#include "graph_optimizer_test.h"
#include "quantization_data.h"
#include "synapse_test.hpp"

class RemoveCastTest : public GraphOptimizerTest
{
protected:
    struct NodeInGraph
    {
        const char*           guid;
        std::vector<unsigned> inputs;
        std::vector<unsigned> outputs;
        void*                 params;
        unsigned              sizeOfParams;
        NodeInGraph(const char*                  guid,
                    const std::vector<unsigned>& inputs,
                    const std::vector<unsigned>& outputs,
                    void*                        params       = nullptr,
                    unsigned                     sizeOfParams = 0)
        : guid(guid), inputs(inputs), outputs(outputs), params(params), sizeOfParams(sizeOfParams)
        {
        }
    };
    using Tensors = std::vector<std::pair<synDataType, TSize>>;
    using Nodes   = std::vector<NodeInGraph>;
    void init(const Tensors& tensors, const Nodes& nodes)
    {
        TensorVector generatedTensors;
        for (const auto& tensor : tensors)
        {
            TSize sizes[] = {1, tensor.second, 1, 1};
            generatedTensors.push_back(TensorPtr(new Tensor(4U, sizes, tensor.first)));
        }
        for (const auto& node : nodes)
        {
            TensorVector inputs;
            TensorVector outputs;
            for (const auto& input : node.inputs)
            {
                inputs.push_back(generatedTensors.at(input));
            }
            for (const auto& output : node.outputs)
            {
                outputs.push_back(generatedTensors.at(output));
            }
            GraphEditor::addNode(m_graph,
                                 NodeFactory::createNode(inputs,
                                                         outputs,
                                                         node.params,
                                                         node.sizeOfParams,
                                                         node.guid,
                                                         createNodeName(node.guid)));
        }
    }
    void validateNumOfCastNodes(unsigned expected)
    {
        unsigned castNodeCount = 0;
        for (const auto& node : m_graph.getExeSortedNodes())
        {
            if ((node != nullptr) && (node->isCast()))
            {
                ++castNodeCount;
            }
        }
        ASSERT_EQ(castNodeCount, expected)
            << "found " << castNodeCount << " cast nodes in the graph but expected to " << expected;
    }

    void validateDataTypes(synDataType dataTypeBeforeCast, synDataType dataTypeAfterCast) const
    {
        synDataType expectedDt = dataTypeBeforeCast;
        for (const auto& node : m_graph.getExeSortedNodes())
        {
            if (node->isCast())
            {
                ASSERT_EQ(node->getInput(0)->getElementType(), expectedDt);
                expectedDt = dataTypeAfterCast;
                ASSERT_EQ(node->getOutput(0)->getElementType(), expectedDt);
            }
            else
            {
                for (const auto& t : node->getOperands())
                {
                    ASSERT_EQ(t->getElementType(), expectedDt);
                }
            }
        }
    }

    void validateCastInPlace(unsigned place)
    {
        unsigned counter = 0;
        for (const auto& node : m_graph.getExeSortedNodes())
        {
            if (counter == place)
            {
                ASSERT_EQ(node != nullptr, true) << "invalid node";
                ASSERT_EQ(node->isCast(), true) << node->getNodeName() << " is not cast node";
            }
            ++counter;
        }
        ASSERT_LT(place, counter) << "place bigger than number of nodes in graph";
    }
    void run()
    {
        bool ret = removeContiguousCastNodes(m_graph);
        ASSERT_EQ(ret, true) << "failed to execute removeContiguousCastNodes pass";
    }

    void setPersistent(TensorPtr& tensor)
    {
        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000);
    }
    GaudiGraph m_graph;

private:
    std::string createNodeName(const char* guid)
    {
        static unsigned counter = 0;
        return std::string(guid) + "_" + std::to_string(++counter);
    }
};

TEST_F(RemoveCastTest, opposite_casts_with_tpc_fuser)
{
    /*
     * The test verify that both of the cast nodes are removed before reaching the TPC fuser pass.
     * If it's not the case, the mult node will be fused with them, and it will be missing from the graph.
     */

    const TSize       batchSize              = 2;
    const TSize       classSize              = 1001;
    const TSize       inSizes[]              = {batchSize, classSize};
    const synDataType synDataTypeFlavor      = syn_type_bf16;
    const synDataType synDataFloatTypeFlavor = syn_type_float;

    TensorPtr T0 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    TensorPtr T1 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    TensorPtr T2 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    TensorPtr T3 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    TensorPtr T4 = TensorPtr(new Tensor(2U, inSizes, synDataFloatTypeFlavor));
    TensorPtr T5 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    TensorPtr T6 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    TensorPtr T7 = TensorPtr(new Tensor(2U, inSizes, synDataTypeFlavor));
    setPersistent(T0);
    setPersistent(T1);
    setPersistent(T6);

    ns_SoftmaxCrossEntropy::Params userParams;
    userParams.mode      = CROSS_ENTROPY_MODE_MEAN;
    userParams.batchSize = batchSize;
    synSplitParams splitParams = {1};

    NodePtr sce   = NodeFactory::createNode({T0, T1}, {T2}, &userParams, "softmax_cross_entropy_bwd_bf16", "sce");
    NodePtr mult  = NodeFactory::createNode({T2, T2}, {T3}, nullptr, "mult_fwd_bf16", "mult");
    NodePtr cast1 = NodeFactory::createNode({T3}, {T4}, nullptr, "cast_bf16_to_f32", "cast1");
    NodePtr cast2 = NodeFactory::createNode({T4}, {T5}, nullptr, "cast_f32_to_bf16", "cast2");
    NodePtr split = NodeFactory::createNode({T5}, {T6, T7}, &splitParams, "split", "split");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, sce));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, mult));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, cast1));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, cast2));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, split));

    validateNumOfCastNodes(2);
    ASSERT_TRUE(m_graph.compile());
    validateNumOfCastNodes(0);

    const NodeVector& nodes = m_graph.getExeSortedNodes();
    ASSERT_TRUE(std::any_of(nodes.begin(),
                            nodes.end(),
                            [](const NodePtr& n) { return n != nullptr && n->getGUID() == "mult_fwd_bf16"; }) &&
                "mult node is missing");
}

TEST_F(RemoveCastTest, remove_opposite_casts)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                                     |
             |cast f32 to bf16|    +------------------>             |
             +--------+-------+                                     |
                      |            +------------------>             |
                      |                                             |
             +--------v-------+                                     |
             |cast bf16 to f32|                                     |
             +--------+-------+                            +--------+---------+
                      |                                    |                  |
           +----------+--------+                           |                  |
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_float, 128),   // Tensor 0
                       std::make_pair(syn_type_float, 128),   // Tensor 1
                       std::make_pair(syn_type_bf16, 128),    // Tensor 2
                       std::make_pair(syn_type_float, 128),   // Tensor 3
                       std::make_pair(syn_type_float, 128),   // Tensor 4
                       std::make_pair(syn_type_float, 128)};  // Tensor 5
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {1}, {2}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {5}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(0);
}

TEST_F(RemoveCastTest, merge_casts_success)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                                     |
             |cast i32 to f32 |    +------------------>             |
             +--------+-------+                             +-------v---------+
                      |            +------------------>     | cast i32 to i8  |
                      |                                     +-------+---------+
             +--------v-------+                                     |
             |cast f32 to i8  |                                     |
             +--------+-------+                            +--------+---------+
                      |                                    |                  |
           +----------+--------+                           |                  |
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_int32, 128),  // Tensor 0
                       std::make_pair(syn_type_int32, 128),  // Tensor 1
                       std::make_pair(syn_type_float, 128),  // Tensor 2
                       std::make_pair(syn_type_int8, 128),   // Tensor 3
                       std::make_pair(syn_type_int8, 128),   // Tensor 4
                       std::make_pair(syn_type_int8, 128)};  // Tensor 5
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {5}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(1);
}

TEST_F(RemoveCastTest, merge_casts_failure)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                             +-------v---------+
             |cast bf16 to f32|    +------------------>     | cast bf16 to f32|
             +--------+-------+                             +-------+---------+
                      |            +------------------>             |
                      |                                             |
             +--------v-------+                             +-------v---------+
             | cast f32 to i8 |                             | cast f32 to i8  |
             +--------+-------+                             +--------+--------+
                      |                                              |
           +----------+--------+                           +---------+--------+
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_bf16, 128),   // Tensor 0
                       std::make_pair(syn_type_bf16, 128),   // Tensor 1
                       std::make_pair(syn_type_float, 128),  // Tensor 2
                       std::make_pair(syn_type_int8, 128),   // Tensor 3
                       std::make_pair(syn_type_int8, 128),   // Tensor 4
                       std::make_pair(syn_type_int8, 128)};  // Tensor 5
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {5}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(2);
}

TEST_F(RemoveCastTest, remove_only_opposite_cast)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             +---------+
             +--------v-------+                            +--------v-------+ |
             |cast f32 to bf16|    +------------------>    |cast f32 to bf16| |
             +--------+-------+                            +--------+-------+ |
                      |            +------------------>             |         |
           +----------+                                             |         |
           | +--------v-------+                                     |         |
           | |cast bf16 to f32|                                     |         |
           | +--------+-------+                            +--------+         |
           |          |                                    |                  |
           |          +--------+                                              |
           |                   |                           |                  |
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_float, 128),   // Tensor 0
                       std::make_pair(syn_type_float, 128),   // Tensor 1
                       std::make_pair(syn_type_bf16, 128),    // Tensor 2
                       std::make_pair(syn_type_float, 128),   // Tensor 3
                       std::make_pair(syn_type_bf16, 128),    // Tensor 4
                       std::make_pair(syn_type_float, 128)};  // Tensor 5
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {1}, {2}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {5}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(1);
}

TEST_F(RemoveCastTest, remove_opposite_casts_around_logical)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                                     |
             |cast f32 to bf16|    +------------------>             |
             +--------+-------+                                     |
                      |            +------------------>             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    reshape     |                            |    reshape     |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    identity    |                            |    identity    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    reshape     |                            |    reshape     |
             +--------+-------+                            +--------+-------+
                      |                                             |
             +--------v-------+                                     |
             |cast bf16 to f32|                                     |
             +--------+-------+                            +--------+---------+
                      |                                    |                  |
           +----------+--------+                           |                  |
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_float, 128),   // Tensor 0
                       std::make_pair(syn_type_float, 128),   // Tensor 1
                       std::make_pair(syn_type_bf16, 128),    // Tensor 2
                       std::make_pair(syn_type_bf16, 128),    // Tensor 3
                       std::make_pair(syn_type_bf16, 128),    // Tensor 4
                       std::make_pair(syn_type_bf16, 128),    // Tensor 5
                       std::make_pair(syn_type_float, 128),   // Tensor 6
                       std::make_pair(syn_type_float, 128),   // Tensor 7
                       std::make_pair(syn_type_float, 128)};  // Tensor 8
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {1}, {2}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {5}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {7}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {8}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(0);
}

TEST_F(RemoveCastTest, merge_casts_around_logical_downcast)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------+-------+
             |cast i32 to f32 |    +------------------>    |cast i32 to i8  |
             +--------+-------+                            +--------+-------+
                      |            +------------------>             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    reshape     |                            |    reshape     |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    identity    |                            |    identity    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    reshape     |                            |    reshape     |
             +--------+-------+                            +--------+-------+
                      |                                             |
             +--------v-------+                                     |
             |cast f32 to i8  |                                     |
             +--------+-------+                            +--------+---------+
                      |                                    |                  |
           +----------+--------+                           |                  |
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_int32, 128),  // Tensor 0
                       std::make_pair(syn_type_int32, 128),  // Tensor 1
                       std::make_pair(syn_type_float, 128),  // Tensor 2
                       std::make_pair(syn_type_float, 128),  // Tensor 3
                       std::make_pair(syn_type_float, 128),  // Tensor 4
                       std::make_pair(syn_type_float, 128),  // Tensor 5
                       std::make_pair(syn_type_int8, 128),   // Tensor 6
                       std::make_pair(syn_type_int8, 128),   // Tensor 7
                       std::make_pair(syn_type_int8, 128)};  // Tensor 8
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {5}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {7}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {8}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(1);
    validateCastInPlace(1);
}

TEST_F(RemoveCastTest, merge_casts_around_logical_upcast)
{
    /*
                      +                                             +
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |      memcpy    |                            |      memcpy    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                                     |
             |cast i8 to f32  |    +------------------>             |
             +--------+-------+                                     |
                      |            +------------------>             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    reshape     |                            |    reshape     |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    identity    |                            |    identity    |
             +--------+-------+                            +--------+-------+
                      |                                             |
                      |                                             |
             +--------v-------+                            +--------v-------+
             |    reshape     |                            |    reshape     |
             +--------+-------+                            +--------+-------+
                      |                                             |
             +--------v-------+                            +----------------+
             |cast f32 to i32 |                            |cast i8 to i32  |
             +--------+-------+                            +----------------+
                      |                                             |
           +----------+--------+                           +--------+---------+
    +------v--------+  +-------v------+            +-------v-------+  +-------v------+
    |    memcpy     |  |     memcpy   |            |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+            +-------+-------+  +-------+------+
           |                   |                           |                  |
           v                   v                           v                  v
    */
    Tensors tensors = {std::make_pair(syn_type_int8, 128),    // Tensor 0
                       std::make_pair(syn_type_int8, 128),    // Tensor 1
                       std::make_pair(syn_type_float, 128),   // Tensor 2
                       std::make_pair(syn_type_float, 128),   // Tensor 3
                       std::make_pair(syn_type_float, 128),   // Tensor 4
                       std::make_pair(syn_type_float, 128),   // Tensor 5
                       std::make_pair(syn_type_int32, 128),   // Tensor 6
                       std::make_pair(syn_type_int32, 128),   // Tensor 7
                       std::make_pair(syn_type_int32, 128)};  // Tensor 8
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_i8_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph("cast_f32_to_i32", {5}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {7}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {8}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(1);
    validateCastInPlace(4);
}

TEST_F(RemoveCastTest, remove_opposite_casts_around_logical_multiconsumers)
{
    /*
                                                       +                                             +
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |      memcpy    |                            |      memcpy    |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                                     |
                                              |cast bf16 to f32|    +------------------>             |
                                              +--------+-------+                                     |
                                                       |            +------------------>             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |    reshape     |                            |    reshape     |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |    identity    |                            |    identity    |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |     split      |                            |     split      |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
            +------------------+--------------------+--+-----------------+                           |
            |                  |                    |                    |                           |
            |                  |                    |                    |
    +---------+---------+-----------------------+-----------------+
    +-------v--------+  +------v---------+  +-------v--------+  +--------v-------+         |                   | | |
    |cast f32 to bf16|  |cast f32 to bf16|  |cast f32 to bf16|  |cast f32 to bf16|         |                   | | |
    +------+---------+  +--------+-------+  +--------+-------+  +--------+-------+         |                   | | | |
    |                   |                   |                 |                   |                       | |
    +------v--------+    +-------v------+    +-------v-------+   +-------v------+     +----v----------+ +--v----------+
    +-------v--------+  +-----v--------+ |    memcpy     |    |     memcpy   |    |    memcpy     |   |     memcpy   |
    |    memcpy     |     |    memcpy   |    |     memcpy     |  |   memcpy     |
    +------+--------+    +-------+------+    +-------+-------+   +-------+------+     +------+--------+ +------+------+
    +--------+-------+  +------+-------+ |                     |                   |                   | | | | | v v v
    v                   v                     v                    v                 v
    */

    const unsigned dim     = 1;
    Tensors        tensors = {std::make_pair(syn_type_bf16, 128),   // Tensor 0
                       std::make_pair(syn_type_bf16, 128),   // Tensor 1
                       std::make_pair(syn_type_float, 128),  // Tensor 2
                       std::make_pair(syn_type_float, 128),  // Tensor 3
                       std::make_pair(syn_type_float, 128),  // Tensor 4
                       std::make_pair(syn_type_float, 32),   // Tensor 5
                       std::make_pair(syn_type_float, 32),   // Tensor 6
                       std::make_pair(syn_type_float, 32),   // Tensor 7
                       std::make_pair(syn_type_float, 32),   // Tensor 8
                       std::make_pair(syn_type_bf16, 32),    // Tensor 9
                       std::make_pair(syn_type_bf16, 32),    // Tensor 10
                       std::make_pair(syn_type_bf16, 32),    // Tensor 11
                       std::make_pair(syn_type_bf16, 32),    // Tensor 12
                       std::make_pair(syn_type_bf16, 32),    // Tensor 13
                       std::make_pair(syn_type_bf16, 32),    // Tensor 14
                       std::make_pair(syn_type_bf16, 32),    // Tensor 15
                       std::make_pair(syn_type_bf16, 32)};   // Tensor 16
    Nodes          nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::splitNodeTypeName, {4}, {5, 6, 7, 8}, (void*)&dim, sizeof(dim)));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {5}, {9}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {6}, {10}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {7}, {11}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {8}, {12}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {9}, {13}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {10}, {14}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {11}, {15}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {12}, {16}));
    init(tensors, nodes);
    validateNumOfCastNodes(5);
    run();
    validateNumOfCastNodes(0);
}

TEST_F(RemoveCastTest, remove_opposite_casts_around_logical_multiproducers)
{
    /*
           +                     +                   +                   +                     |                   | | |
           |                     |                   |                   |                     |                   | | |
           |                     |                   |                   |                     |                   | | |
    +------+--------+    +-------+------+    +-------+-------+   +-------+------+         +----v----------+
    +--v----------+    +-------v--------+  +-----v--------+ |    memcpy     |    |     memcpy   |    |    memcpy     |
    |     memcpy   |         |    memcpy     |     |    memcpy   |    |     memcpy     |  |   memcpy     |
    +------+--------+    +-------+------+    +-------+-------+   +-------+------+         +------+--------+
    +------+------+    +--------+-------+  +------+-------+ |                     |                   | | | | | |
     +-----+----------+  +-------+--------+  +-------v--------+  +-------+--------+              |                     |
    |                 | |cast f32 to bf16|  |cast f32 to bf16|  |cast f32 to bf16|  |cast f32 to bf16|              | |
    |                 |
     +-------+--------+  +------+---------+  +-------+--------+  +--------+-------+              |                     |
    |                 | |                  |                    |                    |
    +-+-------------------+--------------------+-----------------+ |                  |                    | | |
             +------------------+-----------------------------------------+                        |
                                                     |                                             |
                                                     |                                             |
                                                     |                                             |
                                                     |                                             |
                                                     |            ------------------->             |
                                                     |                                             |
                                                     |            +------------------>             |
                                                     |                                             |
                                            +--------v-------+                            +--------v-------+
                                            |     concat     |                            |     concat     |
                                            +--------+-------+                            +--------+-------+
                                                     |                                             |
                                                     |                                             |
                                            +--------v-------+                            +--------v-------+
                                            |     reshape    |                            |     reshape    |
                                            +--------+-------+                            +--------+-------+
                                                     |                                             |
                                                     |                                             |
                                            +--------+-------+                            +--------v-------+
                                            |    identity    |                            |    identity    |
                                            +--------+-------+                            +--------+-------+
                                                     |                                             |
                                            +--------v-------+                                     |
                                            |cast bf16 to f32|                                     |
                                            +--------+-------+                                     |
                                                     |                                             |
                                                     |                                             |
                                            +--------v-------+                            +--------v-------+
                                            |      memcpy    |                            |      memcpy    |
                                            +-------+--------+                            +--------+-------+
                                                    |                                              |
                                                    |                                              |
                                                    |                                              v
                                                    v
    */

    const unsigned dim     = 1;
    Tensors        tensors = {std::make_pair(syn_type_float, 32),    // Tensor 0
                       std::make_pair(syn_type_float, 32),    // Tensor 1
                       std::make_pair(syn_type_float, 32),    // Tensor 2
                       std::make_pair(syn_type_float, 32),    // Tensor 3
                       std::make_pair(syn_type_float, 32),    // Tensor 4
                       std::make_pair(syn_type_float, 32),    // Tensor 5
                       std::make_pair(syn_type_float, 32),    // Tensor 6
                       std::make_pair(syn_type_float, 32),    // Tensor 7
                       std::make_pair(syn_type_bf16, 32),     // Tensor 8
                       std::make_pair(syn_type_bf16, 32),     // Tensor 9
                       std::make_pair(syn_type_bf16, 32),     // Tensor 10
                       std::make_pair(syn_type_bf16, 32),     // Tensor 11
                       std::make_pair(syn_type_bf16, 128),    // Tensor 12
                       std::make_pair(syn_type_bf16, 128),    // Tensor 13
                       std::make_pair(syn_type_bf16, 128),    // Tensor 14
                       std::make_pair(syn_type_float, 128),   // Tensor 15
                       std::make_pair(syn_type_float, 128)};  // Tensor 16
    Nodes          nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {1}, {5}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {7}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {4}, {8}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {5}, {9}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {6}, {10}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {7}, {11}));
    nodes.push_back(NodeInGraph(NodeFactory::concatenateNodeTypeName, {8, 9, 10, 11}, {12}, (void*)&dim, sizeof(dim)));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {12}, {13}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {13}, {14}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {14}, {15}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {15}, {16}));
    init(tensors, nodes);
    validateNumOfCastNodes(5);
    run();
    validateNumOfCastNodes(0);
}

TEST_F(RemoveCastTest, merge_opposite_casts_around_logical_multiconsumers)
{
    /*
                                                       +                                             +
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |      memcpy    |                            |      memcpy    |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |cast i32 to f32 |    +------------------>    |cast i32 to i8  |
                                              +--------+-------+                            +--------+-------+
                                                       |            +------------------>             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |    reshape     |                            |    reshape     |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |    identity    |                            |    identity    |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
                                                       |                                             |
                                              +--------v-------+                            +--------v-------+
                                              |     split      |                            |     split      |
                                              +--------+-------+                            +--------+-------+
                                                       |                                             |
            +------------------+--------------------+--+-----------------+                           |
            |                  |                    |                    |                           |
            |                  |                    |                    |
    +---------+---------+-----------------------+-----------------+
    +-------v--------+  +------v---------+  +-------v--------+  +--------v-------+         |                   | | |
    |cast f32 to i8  |  |cast f32 to i8  |  |cast f32 to i8  |  |cast f32 to i8  |         |                   | | |
    +------+---------+  +--------+-------+  +--------+-------+  +--------+-------+         |                   | | | |
    |                   |                   |                 |                   |                       | |
    +------v--------+    +-------v------+    +-------v-------+   +-------v------+     +----v----------+ +--v----------+
    +-------v--------+  +-----v--------+ |    memcpy     |    |     memcpy   |    |    memcpy     |   |     memcpy   |
    |    memcpy     |     |    memcpy   |    |     memcpy     |  |   memcpy     |
    +------+--------+    +-------+------+    +-------+-------+   +-------+------+     +------+--------+ +------+------+
    +--------+-------+  +------+-------+ |                     |                   |                   | | | | | v v v
    v                   v                     v                    v                 v
    */
    const unsigned dim     = 1;
    Tensors        tensors = {std::make_pair(syn_type_int32, 128),  // Tensor 0
                       std::make_pair(syn_type_int32, 128),  // Tensor 1
                       std::make_pair(syn_type_float, 128),  // Tensor 2
                       std::make_pair(syn_type_float, 128),  // Tensor 3
                       std::make_pair(syn_type_float, 128),  // Tensor 4
                       std::make_pair(syn_type_float, 32),   // Tensor 5
                       std::make_pair(syn_type_float, 32),   // Tensor 6
                       std::make_pair(syn_type_float, 32),   // Tensor 7
                       std::make_pair(syn_type_float, 32),   // Tensor 8
                       std::make_pair(syn_type_int8, 32),    // Tensor 9
                       std::make_pair(syn_type_int8, 32),    // Tensor 10
                       std::make_pair(syn_type_int8, 32),    // Tensor 11
                       std::make_pair(syn_type_int8, 32),    // Tensor 12
                       std::make_pair(syn_type_int8, 32),    // Tensor 13
                       std::make_pair(syn_type_int8, 32),    // Tensor 14
                       std::make_pair(syn_type_int8, 32),    // Tensor 15
                       std::make_pair(syn_type_int8, 32)};   // Tensor 16
    Nodes          nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::splitNodeTypeName, {4}, {5, 6, 7, 8}, (void*)&dim, sizeof(dim)));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {5}, {9}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {6}, {10}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {7}, {11}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {8}, {12}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {9}, {13}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {10}, {14}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {11}, {15}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {12}, {16}));
    init(tensors, nodes);
    validateNumOfCastNodes(5);
    run();
    validateNumOfCastNodes(1);
    validateCastInPlace(1);
}

TEST_F(RemoveCastTest, merge_opposite_casts_around_logical_multiproducers)
{
    /*
           +                     +                   +                   +                     |                   | | |
           |                     |                   |                   |                     |                   | | |
           |                     |                   |                   |                     |                   | | |
    +------+--------+    +-------+------+    +-------+-------+   +-------+------+         +----v----------+
    +--v----------+    +-------v--------+  +-----v--------+ |    memcpy     |    |     memcpy   |    |    memcpy     |
    |     memcpy   |         |    memcpy     |     |    memcpy   |    |     memcpy     |  |   memcpy     |
    +------+--------+    +-------+------+    +-------+-------+   +-------+------+         +------+--------+
    +------+------+    +--------+-------+  +------+-------+ |                     |                   | | | | | |
     +-----+----------+  +-------+--------+  +-------v--------+  +-------+--------+              |                     |
    |                 | |cast i32 to f32 |  |cast i32 to f32 |  |cast i32 to f32 |  |cast i32 to f32 |              | |
    |                 |
     +-------+--------+  +------+---------+  +-------+--------+  +--------+-------+              |                     |
    |                 | |                  |                    |                    |
    +-+-------------------+--------------------+-----------------+ |                  |                    | | |
             +------------------+-----------------------------------------+                        |
                                                     |                                             |
                                                     |                                             |
                                                     |                                             |
                                                     |                                             |
                                                     |            ------------------->             |
                                                     |                                             |
                                                     |            +------------------>             |
                                                     |                                             |
                                            +--------v-------+                            +--------v-------+
                                            |     concat     |                            |     concat     |
                                            +--------+-------+                            +--------+-------+
                                                     |                                             |
                                                     |                                             |
                                            +--------v-------+                            +--------v-------+
                                            |     reshape    |                            |     reshape    |
                                            +--------+-------+                            +--------+-------+
                                                     |                                             |
                                                     |                                             |
                                            +--------+-------+                            +--------v-------+
                                            |    identity    |                            |    identity    |
                                            +--------+-------+                            +--------+-------+
                                                     |                                             |
                                            +--------v-------+                                     |
                                            |cast f32 to i8  |                            +--------v-------+
                                            +--------+-------+                            |cast i32 to i8  |
                                                     |                                    +--------+-------+
                                                     |                                             |
                                            +--------v-------+                            +--------v-------+
                                            |      memcpy    |                            |      memcpy    |
                                            +-------+--------+                            +--------+-------+
                                                    |                                              |
                                                    |                                              |
                                                    |                                              v
                                                    v
    */
    const unsigned dim     = 1;
    Tensors        tensors = {std::make_pair(syn_type_int32, 32),   // Tensor 0
                       std::make_pair(syn_type_int32, 32),   // Tensor 1
                       std::make_pair(syn_type_int32, 32),   // Tensor 2
                       std::make_pair(syn_type_int32, 32),   // Tensor 3
                       std::make_pair(syn_type_int32, 32),   // Tensor 4
                       std::make_pair(syn_type_int32, 32),   // Tensor 5
                       std::make_pair(syn_type_int32, 32),   // Tensor 6
                       std::make_pair(syn_type_int32, 32),   // Tensor 7
                       std::make_pair(syn_type_float, 32),   // Tensor 8
                       std::make_pair(syn_type_float, 32),   // Tensor 9
                       std::make_pair(syn_type_float, 32),   // Tensor 10
                       std::make_pair(syn_type_float, 32),   // Tensor 11
                       std::make_pair(syn_type_float, 128),  // Tensor 12
                       std::make_pair(syn_type_float, 128),  // Tensor 13
                       std::make_pair(syn_type_float, 128),  // Tensor 14
                       std::make_pair(syn_type_int8, 128),   // Tensor 15
                       std::make_pair(syn_type_int8, 128)};  // Tensor 16
    Nodes          nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {1}, {5}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {7}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {4}, {8}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {5}, {9}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {6}, {10}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {7}, {11}));
    nodes.push_back(NodeInGraph(NodeFactory::concatenateNodeTypeName, {8, 9, 10, 11}, {12}, (void*)&dim, sizeof(dim)));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {12}, {13}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {13}, {14}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {14}, {15}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {15}, {16}));
    init(tensors, nodes);
    validateNumOfCastNodes(5);
    run();
    validateNumOfCastNodes(1);
    validateCastInPlace(7);
}

TEST_F(RemoveCastTest, fail_remove_opposite_casts_around_logical_with_more_consumer)
{
    /*
                      +
                      |
                      |
             +--------v-------+
             |      memcpy    |
             +--------+-------+
                      |
                      |
             +--------v-------+
             |cast f32 to bf16|
             +--------+-------+
                      |
                      |
             +--------v-------+
             |    reshape     |
             +--------+-------+
                      |
                      |
             +--------v-------+
             |    identity    |
             +--------+-------+
                      |
           +----------+
           | +--------v-------+
           | |    reshape     |
           | +--------+-------+
           |          |
           | +--------v-------+
           | |cast bf16 to f32|
           | +--------+-------+
           |          |
           |          +--------+
    +------v--------+  +-------v------+
    |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+
           |                   |
           v                   v
    */
    Tensors tensors = {std::make_pair(syn_type_float, 128),  // Tensor 0
                       std::make_pair(syn_type_float, 128),  // Tensor 1
                       std::make_pair(syn_type_bf16, 128),   // Tensor 2
                       std::make_pair(syn_type_bf16, 128),   // Tensor 3
                       std::make_pair(syn_type_bf16, 128),   // Tensor 4
                       std::make_pair(syn_type_bf16, 128),   // Tensor 5
                       std::make_pair(syn_type_float, 128),  // Tensor 6
                       std::make_pair(syn_type_float, 128),  // Tensor 7
                       std::make_pair(syn_type_bf16, 128)};  // Tensor 8
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {1}, {2}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {5}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {6}, {7}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {4}, {8}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(2);
}

TEST_F(RemoveCastTest, fail_remove_opposite_casts_around_logical_with_more_producer)
{
    /*
           +
           |                        +
           |                        |
           |                        |
           |               +--------v-------+
           |               |      memcpy    |
           |               +--------+-------+
           |                        |
           |                        |
           |               +--------v-------+
           |               |cast f32 to bf16|
           |               +--------+-------+
           |                        |
           |                        |
           |               +--------v-------+
           |               |    reshape     |
           |               +--------+-------+
           |                        |
           |                        |
    +------v--------+      +--------v-------+
    |    memcpy     |      |    identity    |
    +------+--------+      +--------+-------+
           |                        |
           +------------------------+
                           +--------v-------+
                           |     concat     |
                           +--------+-------+
                                    |
                           +--------v-------+
                           |cast bf16 to f32|
                           +--------+-------+
                                    |
                                    |
                            +-------v------+
                            |     memcpy   |
                            +--------------+
    */
    const unsigned dim     = 1;
    Tensors        tensors = {std::make_pair(syn_type_float, 64),   // Tensor 0
                       std::make_pair(syn_type_float, 64),   // Tensor 1
                       std::make_pair(syn_type_bf16, 64),    // Tensor 2
                       std::make_pair(syn_type_bf16, 64),    // Tensor 3
                       std::make_pair(syn_type_bf16, 64),    // Tensor 4
                       std::make_pair(syn_type_bf16, 64),    // Tensor 5
                       std::make_pair(syn_type_bf16, 64),    // Tensor 6
                       std::make_pair(syn_type_bf16, 128),   // Tensor 7
                       std::make_pair(syn_type_float, 128),  // Tensor 8
                       std::make_pair(syn_type_bf16, 128)};  // Tensor 9
    Nodes          nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {6}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {1}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph(NodeFactory::concatenateNodeTypeName, {5, 6}, {7}, (void*)&dim, sizeof(dim)));
    nodes.push_back(NodeInGraph("cast_bf16_to_f32", {7}, {8}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {8}, {9}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(2);
}

TEST_F(RemoveCastTest, fail_merge_casts_with_more_consumer)
{
    /*
                      +
                      |
                      |
             +--------v-------+
             |      memcpy    |
             +--------+-------+
                      |
                      |
             +--------v-------+
             |cast i32 to f32 |
             +--------+-------+
                      |
           +----------+
           | +--------v-------+
           | |cast f32 to i8  |
           | +--------+-------+
           |          |
           |          +--------+
           |                   |
    +------v--------+  +-------v------+
    |    memcpy     |  |     memcpy   |
    +------+--------+  +-------+------+
           |                   |
           v                   v
    */
    Tensors tensors = {std::make_pair(syn_type_int32, 128),  // Tensor 0
                       std::make_pair(syn_type_int32, 128),  // Tensor 1
                       std::make_pair(syn_type_float, 128),  // Tensor 2
                       std::make_pair(syn_type_int8, 128),   // Tensor 3
                       std::make_pair(syn_type_float, 128),  // Tensor 4
                       std::make_pair(syn_type_int8, 128)};  // Tensor 5
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph("cast_i32_to_f32", {1}, {2}));
    nodes.push_back(NodeInGraph("cast_f32_to_i8", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {2}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {3}, {5}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(2);
}

TEST_F(RemoveCastTest, merge_casts_with_packing)
{
    Tensors tensors = {std::make_pair(syn_type_int8, 128),   // Tensor 0
                       std::make_pair(syn_type_int8, 128),   // Tensor 1
                       std::make_pair(syn_type_int8, 128),   // Tensor 2
                       std::make_pair(syn_type_float, 128),  // Tensor 3
                       std::make_pair(syn_type_float, 128),  // Tensor 4
                       std::make_pair(syn_type_float, 128),  // Tensor 5
                       std::make_pair(syn_type_bf16, 128),   // Tensor 6
                       std::make_pair(syn_type_bf16, 128),   // Tensor 7
                       std::make_pair(syn_type_bf16, 128),   // Tensor 8
                       std::make_pair(syn_type_bf16, 128)};  // Tensor 9
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {1}, {2}));
    nodes.push_back(NodeInGraph("cast_i8_to_f32", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {5}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {6}, {7}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {7}, {8}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {7}, {9}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(1);

    // Expecting one cast node : i8 -> bf16
    validateDataTypes(syn_type_int8, syn_type_bf16);
}

TEST_F(RemoveCastTest, merge_casts_with_packing_and_logical_nodes_between)
{
    Tensors tensors = {std::make_pair(syn_type_int8, 128),   // Tensor 0
                       std::make_pair(syn_type_int8, 128),   // Tensor 1
                       std::make_pair(syn_type_int8, 128),   // Tensor 2
                       std::make_pair(syn_type_float, 128),  // Tensor 3
                       std::make_pair(syn_type_float, 128),  // Tensor 4
                       std::make_pair(syn_type_float, 128),  // Tensor 5
                       std::make_pair(syn_type_float, 128),  // Tensor 6
                       std::make_pair(syn_type_float, 128),  // Tensor 7
                       std::make_pair(syn_type_float, 128),  // Tensor 8
                       std::make_pair(syn_type_bf16, 128),   // Tensor 9
                       std::make_pair(syn_type_bf16, 128),   // Tensor 10
                       std::make_pair(syn_type_bf16, 128),   // Tensor 11
                       std::make_pair(syn_type_bf16, 128)};  // Tensor 12
    Nodes   nodes;
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {0}, {1}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {1}, {2}));
    nodes.push_back(NodeInGraph("cast_i8_to_f32", {2}, {3}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {3}, {4}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {4}, {5}));
    nodes.push_back(NodeInGraph(NodeFactory::identityNodeTypeName, {5}, {6}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {6}, {7}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {7}, {8}));
    nodes.push_back(NodeInGraph("cast_f32_to_bf16", {8}, {9}));
    nodes.push_back(NodeInGraph(NodeFactory::reshapeNodeTypeName, {9}, {10}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {10}, {11}));
    nodes.push_back(NodeInGraph(NodeFactory::memcpyNodeTypeName, {10}, {12}));
    init(tensors, nodes);
    validateNumOfCastNodes(2);
    run();
    validateNumOfCastNodes(1);

    // Expecting one cast node : i8 -> bf16
    validateDataTypes(syn_type_int8, syn_type_bf16);
}

// Dedw node is sliced, which adds a reduction + cast node from fp32 to bf16 (cast is after the reduction). The next
// cast is opposite - bf16 to fp32 - so they should be removed.
TEST_F(RemoveCastTest, remove_opposite_casts_after_reduction)
{
    const TSize          dims = 4;
    const TSize          b = 1, h = 1, w = 1;
    const TSize          c = 1, k = 64;
    synConvolutionParams params;
    const TSize          xSizes[]   = {c, w, h, b};
    const TSize          dwSizes[]  = {k, c, params.kW, params.kH};
    const TSize          dySizes[]  = {k,
                             convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW),
                             convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH),
                             b};
    TSize                inputSize  = b * h * w * c * dataTypeSizeInBytes(syn_type_bf16);
    TSize                outputSize = params.kW * params.kH * k * c * dataTypeSizeInBytes(syn_type_bf16);

    setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, std::to_string(inputSize / 2 + outputSize));
    auto x  = std::make_shared<Tensor>(dims, xSizes, syn_type_bf16);
    auto dy = std::make_shared<Tensor>(dims, dySizes, syn_type_bf16);
    auto dw = std::make_shared<Tensor>(dims, dwSizes, syn_type_bf16);
    setPersistent(dy);
    setPersistent(x);

    // Dedw will be sliced, so a reduction will be added + cast from fp32 to bf16 after the reduction.
    pNode dedwNode = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");

    // This cast is expected to be optimized out (it is an opposite cast to the one created by the sram management)
    auto  castOut  = std::make_shared<Tensor>(dims, dwSizes, syn_type_single);
    pNode castNode = NodeFactory::createNode({dw}, {castOut}, nullptr, "cast_bf16_to_f32", "cast");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, dedwNode));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, castNode));

    validateNumOfCastNodes(1);
    ASSERT_TRUE(m_graph.compile());
    validateNumOfCastNodes(0);
}
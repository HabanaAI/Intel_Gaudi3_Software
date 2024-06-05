#include "habana_pass.h"
#include "node.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "types.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "transpose_utils.h"

class RemoveSequencesTest : public GraphOptimizerTest
{
protected:
    struct gcNode
    {
        const char*           guid;
        std::vector<unsigned> inputs;
        std::vector<unsigned> outputs;
        void*                 params;
        unsigned              sizeOfParams;
        gcNode(const char*                  guid,
               const std::vector<unsigned>& inputs,
               const std::vector<unsigned>& outputs,
               void*                        params       = nullptr,
               unsigned                     sizeOfParams = 0)
        : guid(guid), inputs(inputs), outputs(outputs), params(params), sizeOfParams(sizeOfParams)
        {
        }
    };
    struct gcTensor
    {
        synDataType dtype;
        TSize       size;
        bool        isPersistent;
        unsigned    sizeOfParams;
        bool        overlap;
        gcTensor(synDataType dtype, TSize size, bool isPersistent = false, bool isOverlap = false)
        : dtype(dtype), size(size), isPersistent(isPersistent), overlap(isOverlap)
        {
        }
    };
    using Tensors = std::vector<gcTensor>;
    using Nodes   = std::vector<gcNode>;
    using ControlEdges = std::vector<std::pair<unsigned, unsigned>>;
    void init(const Tensors& tensors, const Nodes& nodes, const ControlEdges& controlEdges = {})
    {
        uint64_t overlapSection = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;
        uint64_t sectionId      = overlapSection + 1;
        for (const auto& gcTensor : tensors)
        {
            TSize sizes[] = {gcTensor.size, gcTensor.size, gcTensor.size, gcTensor.size};
            auto  tensor  = TensorPtr(new Tensor(4U, sizes, gcTensor.dtype));
            if (gcTensor.isPersistent)
            {
                synMemoryDescriptor memDesc(true);
                tensor->setMemoryDescriptor(memDesc);
                tensor->setMemorySectionID(gcTensor.overlap ? overlapSection : sectionId++);
                tensor->setMemorySectionOffset(0);
            }
            m_generatedTensors.push_back(tensor);
        }
        for (const auto& node : nodes)
        {
            TensorVector inputs;
            TensorVector outputs;
            for (const auto& input : node.inputs)
            {
                inputs.push_back(m_generatedTensors.at(input));
            }
            for (const auto& output : node.outputs)
            {
                outputs.push_back(m_generatedTensors.at(output));
            }
            auto generatedNode = NodeFactory::createNode(inputs,
                                                         outputs,
                                                         node.params,
                                                         node.sizeOfParams,
                                                         node.guid,
                                                         createNodeName(node.guid));
            m_generatedNodes.push_back(generatedNode);
            GraphEditor::addNode(m_graph, generatedNode);
        }
        for (const auto& [blockingIdx, blockedIdx] : controlEdges)
        {
            m_graph.addControlDependency(m_generatedNodes[blockingIdx], m_generatedNodes[blockedIdx]);
        }
        registerMemoryCoherence(m_graph);
    }
    std::string createNodeName(const char* guid)
    {
        static unsigned counter = 0;
        return std::string(guid) + "_" + std::to_string(++counter);
    }

    template<typename NodeContainer>
    NodeVector findNodes(const NodeContainer& nodes, std::function<bool(const NodePtr&)> pred) const
    {
        NodeVector ret {};
        for (const auto& n : nodes)
        {
            if (n && pred(n)) ret.push_back(n);
        }
        return ret;
    }

    void validateNumOfNodes(const char* guid, unsigned expected, std::function<bool(const NodePtr&)> predicate)
    {
        const auto& foundNodes(findNodes(m_graph.getExeSortedNodes(), predicate));
        const auto  nodeCount(foundNodes.size());
        ASSERT_EQ(nodeCount, expected) << "found " << std::to_string(nodeCount) << " " << guid
                                       << " nodes in the graph but expected " << std::to_string(expected);
    }

    template<typename NodeContainer>
    NodeVector findNodesByGuid(const NodeContainer& nodes, const std::string& guid) const
    {
        return findNodes(nodes, [&guid](const auto& n) { return n && n->getGUID() == guid; });
    }

    NodeVector   m_generatedNodes;
    TensorVector m_generatedTensors;
    GaudiGraph   m_graph;
};

class RemoveTransposeSequencesTest : public RemoveSequencesTest
{
public:
    template<typename NodeContainer>
    NodeVector findTransposeNodes(const NodeContainer& nodes) const
    {
        return findNodes(nodes,
                         [](const auto& n) -> bool { return n && n->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE; });
    }

    void validateNumOfTransposeNodes(unsigned expected)
    {
        validateNumOfNodes("transpose", expected, [&](const NodePtr& n) {
            return n->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE;
        });
    }
    void validateNumOfMemcpyNodes(unsigned expected)
    {
        validateNumOfNodes("memcopy", expected, [&](const NodePtr& n) {
            return n->getNodeType() == Node::TYPE_MEMCOPY;
        });
    }
    void run()
    {
        ScopedConfigurationChange enableIdentityRemove("ENABLE_CONTIGUOUS_TRANSPOSE_REMOVAL", "true");
        ScopedConfigurationChange enableFusion("ENABLE_FUSING_CONTIGUOUS_TRANSPOSE_NODES", "true");
        bool ret = removeContiguousTransposes(m_graph);
        ASSERT_EQ(ret, true) << "failed to execute removeContiguousTransposes pass";
    }
};

// input: T0 -> T1 is identity.
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
// output:
// +---+    +---+
// | A |--->| B |
// +---+    +---+
TEST_F(RemoveTransposeSequencesTest, basic_transpose_seq)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15),   // Tensor 3
                       gcTensor(syn_type_float, 15)};  // Tensor 4

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                             // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {4}));                                             // B
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(0);
}

// input: T0 -> T1 is identity.
// t0, t4 are overlapping tensors (with t0 written first)
//
//  t1 +---+ t2 +----+ t3 +----+ t4 +---+ t5
// --->| A |--->| T0 |--->| T1 |--->| B |-->
//     +---+    +----+ |  +----+    +---+
//                     |
//                     |  +----+ t0
//                     `->| C  |--->
//                        +----+
// output: not changed
TEST_F(RemoveTransposeSequencesTest, transpose_seq_with_control)
{
    Tensors tensors = {gcTensor(syn_type_float, 15, true, true),  // Tensor 0
                       gcTensor(syn_type_float, 15),              // Tensor 1
                       gcTensor(syn_type_float, 15),              // Tensor 2
                       gcTensor(syn_type_float, 15),              // Tensor 3
                       gcTensor(syn_type_float, 15, true, true),  // Tensor 4
                       gcTensor(syn_type_float, 15)};             // Tensor 5

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {3}, {0}));                                             // C
    nodes.push_back(gcNode("relu_fwd_f32", {1}, {2}));                                             // A
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParams, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {4}, {5}));                                             // B
    ControlEdges controlEdges = {std::make_pair(0, 3)};

    init(tensors, nodes, controlEdges);
    run();
    validateNumOfTransposeNodes(2);
}

// input: T0 -> T1 is identity
// +---+    +----+    +----+
// | A |--->| T0 |--->| T1 |
// +---+    +----+    +----+
// output:
// +---+
// | A |
// +---+
TEST_F(RemoveTransposeSequencesTest, seq_output_is_graph_output)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15)};  // Tensor 3

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                             // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams, sizeof(synTransposeParams)));  // T1
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(0);

    NodePtr   A           = m_generatedNodes[0];
    TensorPtr outT1Before = m_generatedTensors[3];
    ASSERT_EQ(A->getOutput(0), outT1Before);
}

// input: T0 -> T1 is identity
// +----+    +----+
// | T0 |--->| T1 |
// +----+    +----+
// output:
// +--------+
// | memcpy |
// +--------+
TEST_F(RemoveTransposeSequencesTest, seq_io_are_graph_io)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15)};  // Tensor 2

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("transpose", {0}, {1}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));  // T1
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(0);
    validateNumOfMemcpyNodes(1);
}

// input: T0 -> T1 -> T2 is identity
// +----+    +----+    +----+   +----+   +----+
// | A  +--->| T0 +--->| T1 +-->| T2 +-->| B  |
// +----+    +----+    +-+--+   +----+   +----+
//                       |
//                       v
//                     +----+
//                     | C  |
//                     +----+
// output:
//   +-------------------+
//   |                   |
// +-+--+    +----+   +--v-+
// | A  +--->|T0+1|   | B  |
// +----+    +-+--+   +----+
//             |
//             v
//           +----+
//           | C  |
//           +----+
TEST_F(RemoveTransposeSequencesTest, middle_node_with_multiple_consumers)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15),   // Tensor 3
                       gcTensor(syn_type_float, 15),   // Tensor 4
                       gcTensor(syn_type_float, 15),   // Tensor 5
                       gcTensor(syn_type_float, 15)};  // Tensor 6

    Nodes              nodes;
    synTransposeParams transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParamsT2 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                               // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParamsT0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParamsT1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParamsT2, sizeof(synTransposeParams)));  // T2
    nodes.push_back(gcNode("relu_fwd_f32", {4}, {5}));                                               // B
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {6}));                                               // C
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(1);
}

// input: T0 -> T1 is identity, T0 output is persistent
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
// output:
//   +-------------------+
//   |                   |
// +-+--+    +----+   +--v-+
// | A  +--->| T0 |   | B  |
// +----+    +-+--+   +----+
TEST_F(RemoveTransposeSequencesTest, middle_node_with_persistent_output)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),        // Tensor 0
                       gcTensor(syn_type_float, 15),        // Tensor 1
                       gcTensor(syn_type_float, 15, true),  // Tensor 2
                       gcTensor(syn_type_float, 15),        // Tensor 3
                       gcTensor(syn_type_float, 15)};       // Tensor 4

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                             // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {4}));                                             // B
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(1);
}

// input: T0 -> T1 is identity, output of T0 and T1 is persistent
// +-----+    +----+    +----+    +------+
// | Add |--->| T0 |--->| T1 |--->| Relu |
// +-----+    +----+    +----+    +------+
//
//
// output:
//   +----------------------+
//   |                      |
// +-+----+    +----+   +---v---+
// | Add  +--->| T0 |   | Relu  |
// +------+    +-+--+   +-------+
TEST_F(RemoveTransposeSequencesTest, user_managed_sequence_output_and_intermediate_sanity)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),        // Tensor 0
                       gcTensor(syn_type_float, 15),        // Tensor 1
                       gcTensor(syn_type_float, 15, true),  // Tensor 2
                       gcTensor(syn_type_float, 15, true),  // Tensor 3
                       gcTensor(syn_type_float, 15)};       // Tensor 4

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    const std::string  addGuid("add_fwd_f32");
    const std::string  reluGuid("relu_fwd_f32");

    nodes.push_back(gcNode(addGuid.c_str(), {0}, {1}));                                            // Add
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(transposeParams)));     // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams, sizeof(transposeParams)));     // T1
    nodes.push_back(gcNode(reluGuid.c_str(), {3}, {4}));                                           // Relu
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(1);

    const auto graphNodes(m_graph.getNodes());

    const auto transposeNodes(findTransposeNodes(graphNodes));
    ASSERT_TRUE(transposeNodes.size() == 1) << "Expecting a single transpose node";
    const auto& transpose(transposeNodes.front());

    const auto reluNodes = findNodesByGuid(graphNodes, reluGuid);
    ASSERT_TRUE(reluNodes.size() == 1);
    const auto addNodes = findNodesByGuid(graphNodes, addGuid);
    ASSERT_TRUE(addNodes.size() == 1);

    const auto& relu(reluNodes.front());
    const auto& add(addNodes.front());

    const auto sequenceProducerConsumers(m_graph.getNodeConsumers(add));
    ASSERT_TRUE(sequenceProducerConsumers.size() == 2)
        << "Expecting transpose sequence producer (add) to be consumed by remaining transpose and relu";

    std::unordered_set<NodePtr> expectedSequenceProducerConsumers {relu, transpose};
    for (const auto& seqProducerConsumer : sequenceProducerConsumers)
    {
        ASSERT_TRUE(expectedSequenceProducerConsumers.find(seqProducerConsumer) !=
                    expectedSequenceProducerConsumers.end())
            << "Unexpected consumer " << seqProducerConsumer->getNodeTypeStr() << " of transpose sequence producer";
    }

    ASSERT_TRUE(transpose->getInput(0)->isUserManagedDram())
        << "Expecting input tensor of remaining transpose node to be persistent";
    ASSERT_TRUE(transpose->getOutput(0)->isUserManagedDram())
        << "Expecting output tensor of remaining transpose node to be persistent";
}

// input: T0 -> T1 -> T2 -> T3 is identity, T1 and T3 outputs are persistent
// +-----+    +----+    +----+    +----+    +----+    +------+
// | Add |--->| T0 |--->| T1 |--->| T2 |--->| T3 |--->| Relu |
// +-----+    +----+    +----+    +----+    +----+    +------+
//
//
// output:
//   +---------------------------------+
//   |                                 |
// +-+----+    +----+              +---v---+
// | Add  +--->|T0+1|              | Relu  |
// +------+    +----+              +-------+
TEST_F(RemoveTransposeSequencesTest, user_managed_sequence_output_and_non_adjacent_intermediate)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),        // Tensor 0
                       gcTensor(syn_type_float, 15),        // Tensor 1
                       gcTensor(syn_type_float, 15),        // Tensor 2
                       gcTensor(syn_type_float, 15),        // Tensor 3
                       gcTensor(syn_type_float, 15, true),  // Tensor 4
                       gcTensor(syn_type_float, 15),        // Tensor 5
                       gcTensor(syn_type_float, 15, true),  // Tensor 6
                       gcTensor(syn_type_float, 15)};       // Tensor 7

    Nodes nodes {};

    synTransposeParams transposeParams0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams1 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams transposeParams2 = {{TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams3 = {{TPD_Height, TPD_Channel, TPD_4Dim_Batch, TPD_Width}, 4};
    const std::string  addGuid("add_fwd_f32");
    const std::string  reluGuid("relu_fwd_f32");

    nodes.push_back(gcNode(addGuid.c_str(), {0, 1}, {2}));                                        // Add
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams0, sizeof(transposeParams0)));  // T0
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParams1, sizeof(transposeParams1)));  // T1
    nodes.push_back(gcNode("transpose", {4}, {5}, &transposeParams2, sizeof(transposeParams2)));  // T2
    nodes.push_back(gcNode("transpose", {5}, {6}, &transposeParams3, sizeof(transposeParams3)));  // T3
    nodes.push_back(gcNode(reluGuid.c_str(), {6}, {7}));                                          // Relu
    init(tensors, nodes);
    run();

    const auto graphNodes(m_graph.getNodes());

    const auto reluNodes(findNodesByGuid(graphNodes, reluGuid));
    ASSERT_TRUE(reluNodes.size() == 1) << "Expecting 1 relu node";

    const auto addNodes(findNodesByGuid(graphNodes, addGuid));
    ASSERT_TRUE(addNodes.size() == 1) << "Expecting 1 add node";

    const auto transposeNodes(findTransposeNodes(graphNodes));
    ASSERT_TRUE(transposeNodes.size() == 1) << "Expecting 1 transpose nodes";

    const auto& relu(reluNodes.front());
    const auto& add(addNodes.front());

    const auto sequenceProducerConsumers(m_graph.getNodeConsumers(add));
    ASSERT_TRUE(sequenceProducerConsumers.size() == 2)
        << "Expecting transpose sequence producer to be consumed by remaining transpose and relu";

    const auto& seqProducerConsumingTransposes(findTransposeNodes(sequenceProducerConsumers));
    ASSERT_TRUE(seqProducerConsumingTransposes.size() == 1)
        << "Expecting transpose T0+1 to consume identity sequence producer (add)";

    const auto& transpose0(seqProducerConsumingTransposes.front());
    const auto& transpose0ConsumingTransposes(findTransposeNodes(m_graph.getNodeConsumers(transpose0)));
    ASSERT_TRUE(transpose0ConsumingTransposes.size() == 0) << "Expecting transpose T0+1 to have exactly no consumers";
    ASSERT_TRUE(transpose0->getOutput(0)->isUserManagedDram()) << "Expecting transpose T1's output to be user managed";

    ASSERT_TRUE(relu->getInput(0)->isUserManagedDram())
        << "Expecting relu input (transpose sequence output tensor) to be persistent";
    ASSERT_TRUE(relu->getInput(0) == add->getOutput(0))
        << "Expecting sequence producer output tensor to be replaced by the persistent sequence output";
}

// input: T0 -> T1 -> T2 -> T3  are identity, T1 and T3 outputs are persistent.
//        T1's output has two consumers besides T2.
//
// +-----+    +----+    +----+    +----+    +----+    +------+
// | Add |--->| T0 |--->| T1 |--->| T2 |--->| T3 |--->| Relu |
// +-----+    +----+    +--+-+   +----+    +----+    +------+
//                         |
//                         +----------------+
//                         |                |
//                      +--v----+      +----v------+
//                      | Gelu  |      | LeakyRelu |
//                      +-------+      +-----------+
// output:
//   +-------------------------------+
//   |                               |
// +-+----+    +-----+           +---v---+
// | Add  +--->| T1+0|           | Relu  |
// +------+    +--+--+           +-------+
//                |
//                +----------------+
//                |                |
//             +--v----+      +----v------+
//             | Gelu  |      | LeakyRelu |
//             +-------+      +-----------+

TEST_F(RemoveTransposeSequencesTest,
       user_managed_sequence_output_and_non_adjacent_user_managed_intermediate_with_multiple_consumers)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),         // Tensor 0
                       gcTensor(syn_type_float, 15),         // Tensor 1
                       gcTensor(syn_type_float, 15),         // Tensor 2
                       gcTensor(syn_type_float, 15),         // Tensor 3
                       gcTensor(syn_type_float, 15, true),   // Tensor 4
                       gcTensor(syn_type_float, 15),         // Tensor 5
                       gcTensor(syn_type_float, 15, true),   // Tensor 6
                       gcTensor(syn_type_float, 15),         // Tensor 7
                       gcTensor(syn_type_float, 15, true),   // Tensor 8
                       gcTensor(syn_type_float, 15, true)};  // Tensor 9

    Nodes nodes;

    synTransposeParams transposeParams0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams1 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams transposeParams2 = {{TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams3 = {{TPD_Height, TPD_Channel, TPD_4Dim_Batch, TPD_Width}, 4};

    const std::string addGuid("add_fwd_f32");
    const std::string reluGuid("relu_fwd_f32");
    const std::string leakyReluGuid("leakyrelu_fwd_f32");
    const std::string geluGuid("gelu_fwd_f32");

    nodes.push_back(gcNode(addGuid.c_str(), {0, 1}, {2}));                                          // Add
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams0, sizeof(transposeParams0)));    // T0
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParams1, sizeof(transposeParams1)));    // T1
    nodes.push_back(gcNode(geluGuid.c_str(), {4}, {8}));                                            // Gelu
    nodes.push_back(gcNode(leakyReluGuid.c_str(), {4}, {9}));                                       // Leaky relu
    nodes.push_back(gcNode("transpose", {4}, {5}, &transposeParams2, sizeof(transposeParams2)));    // T2
    nodes.push_back(gcNode("transpose", {5}, {6}, &transposeParams3, sizeof(transposeParams3)));    // T3
    nodes.push_back(gcNode(reluGuid.c_str(), {6}, {7}));                                            // Relu
    init(tensors, nodes);
    run();

    validateNumOfTransposeNodes(1);
    const auto graphNodes(m_graph.getNodes());

    const auto reluNodes = findNodesByGuid(graphNodes, reluGuid);
    ASSERT_TRUE(reluNodes.size() == 1);

    const auto addNodes = findNodesByGuid(graphNodes, addGuid);
    ASSERT_TRUE(addNodes.size() == 1);

    const auto geluNodes = findNodesByGuid(graphNodes, geluGuid);
    ASSERT_TRUE(geluNodes.size() == 1);

    const auto leakyReluNodes = findNodesByGuid(graphNodes, leakyReluGuid);
    ASSERT_TRUE(leakyReluNodes.size() == 1);

    const auto& relu(reluNodes.front());
    const auto& leakyRelu(leakyReluNodes.front());
    const auto& gelu(geluNodes.front());
    const auto& add(addNodes.front());

    const auto sequenceProducerConsumers(m_graph.getNodeConsumers(add));
    ASSERT_TRUE(sequenceProducerConsumers.size() == 2)
        << "Expecting transpose sequence producer to be consumed by remaining transpose and relu";

    const auto& seqProducerConsumingTransposes(findTransposeNodes(sequenceProducerConsumers));
    const auto& transpose0(seqProducerConsumingTransposes.front());
    ASSERT_TRUE(seqProducerConsumingTransposes.size() == 1)
        << "Expecting transpose T0 to consume identity sequence producer (add)";

    {
        std::unordered_set<NodePtr> expectedSequenceProducerConsumers {transpose0, relu};
        for (const auto& seqProducerConsumer : sequenceProducerConsumers)
        {
            ASSERT_TRUE(expectedSequenceProducerConsumers.find(seqProducerConsumer) !=
                        expectedSequenceProducerConsumers.end())
                << "Unexpected consumer " << seqProducerConsumer->getNodeTypeStr() << " of transpose sequence producer";
        }
    }

    const auto fusedTransposeConsumers(m_graph.getNodeConsumers(transpose0));
    ASSERT_TRUE(fusedTransposeConsumers.size() == 2)
        << "Expecting transpose T1 to be consumed by exactly two nodes (gelu and leaky relu)";
    std::unordered_set<NodePtr> expectedTranspose1Consumers {gelu, leakyRelu};
    for (const auto& transpose1Consumer : fusedTransposeConsumers)
    {
        ASSERT_TRUE(expectedTranspose1Consumers.find(transpose1Consumer) != expectedTranspose1Consumers.end())
            << "Unexpected consumer " << transpose1Consumer->getNodeTypeStr() << " of transpose sequence producer";
    }

    const auto& fusedTransposeOutputs(transpose0->getOutputs());
    ASSERT_TRUE(fusedTransposeOutputs.size() == 1) << "Expecting transpose1 to have exactly one output";
    ASSERT_TRUE(fusedTransposeOutputs[0]->isUserManagedDram()) << "Expecting transpose1 output to be user managed";
    ASSERT_TRUE(relu->getInput(0)->isUserManagedDram())
        << "Expecting relu input (transpose sequence output tensor) to be user managed";
}

// input: T0 -> T1 is identity, T1 output is persistent
// +----+    +----+    +---+
// | T0 |--->| T1 |--->| B |
// +----+    +----+    +---+
// output:
// +--------+    +---+
// | memcpy |--->| B |
// +--------+    +---+
TEST_F(RemoveTransposeSequencesTest, last_node_with_persistent_output)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),        // Tensor 0
                       gcTensor(syn_type_float, 15),        // Tensor 1
                       gcTensor(syn_type_float, 15, true),  // Tensor 2
                       gcTensor(syn_type_float, 15)};       // Tensor 3

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("transpose", {0}, {1}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {2}, {3}));                                             // B
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(0);
    validateNumOfMemcpyNodes(1);
}

// input: T0 -> T1 is identity
//  +----+    +----+    +----+    +----+
//  | A  +--->| T0 +--->| T1 +--->| C  |
//  +----+    +----+    +-+--+    +----+
//                        |
//                        |
//                      +-v--+
//                      | B  |
//                      +----+
// output:
//   +----+     +----+
//   | A  +---> | C  |
//   +-+--+     +----+
//     |
//     v
//   +----+
//   | B  |
//   +----+
TEST_F(RemoveTransposeSequencesTest, last_node_with_multiple_consumers)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15),   // Tensor 3
                       gcTensor(syn_type_float, 15),   // Tensor 4
                       gcTensor(syn_type_float, 15)};  // Tensor 5

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                             // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {4}));                                             // B
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {5}));                                             // C
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(0);
}

// input: T1 -> T2 and T0 -> T1 -> T3 are identity sequences.
// +----+    +----+    +----+    +----+    +----+
// | A  +--->| T0 +--->| T1 +--->| T2 +--->| B  |
// +----+    +----+    +-+--+    +----+    +----+
//                       |
//                       |
//                     +-v--+    +----+    +----+
//                     | T3 +--->| T4 +--->| C  |
//                     +----+    +----+    +----+
// output:
//  +----+    +-+--+    +----+
//  | A  +--->| T0 +--->| B  |
//  +--+-+    +----+    +----+
//     |
//     v
//  +----+    +----+
//  | T4 +--->| C  |
//  +----+    +----+
TEST_F(RemoveTransposeSequencesTest, two_subsequences_are_identity_no_overlap)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15),   // Tensor 3
                       gcTensor(syn_type_float, 15),   // Tensor 4
                       gcTensor(syn_type_float, 15),   // Tensor 5
                       gcTensor(syn_type_float, 15),   // Tensor 6
                       gcTensor(syn_type_float, 15),   // Tensor 7
                       gcTensor(syn_type_float, 15)};  // Tensor 8

    Nodes              nodes;
    synTransposeParams transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParamsT2 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParamsT3 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                               // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParamsT0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParamsT1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParamsT2, sizeof(synTransposeParams)));  // T2
    nodes.push_back(gcNode("relu_fwd_f32", {4}, {5}));                                               // B
    nodes.push_back(gcNode("transpose", {3}, {6}, &transposeParamsT3, sizeof(synTransposeParams)));  // T3
    nodes.push_back(gcNode("transpose", {6}, {7}, &transposeParamsT2, sizeof(synTransposeParams)));  // T4
    nodes.push_back(gcNode("relu_fwd_f32", {7}, {8}));                                               // C
    init(tensors, nodes);
    NodeSet nodesBefore = m_graph.getNodes();
    run();
    validateNumOfTransposeNodes(2);

    NodePtr A  = m_generatedNodes[0];
    NodePtr T0 = m_generatedNodes[1];
    NodePtr B  = m_generatedNodes[4];
    NodePtr T4 = m_generatedNodes[6];
    NodePtr C  = m_generatedNodes[7];
    ASSERT_EQ(A->getOutput(0), T0->getInput(0));
    ASSERT_EQ(T0->getOutput(0), B->getInput(0));
    ASSERT_EQ(A->getOutput(0), T4->getInput(0));
    ASSERT_EQ(T4->getOutput(0), C->getInput(0));
    TensorPtr outABefore = m_generatedTensors[1];
    TensorPtr outBBefore = m_generatedTensors[5];
    TensorPtr outCBefore = m_generatedTensors[8];
    ASSERT_EQ(A->getOutput(0), outABefore);
    ASSERT_EQ(B->getOutput(0), outBBefore);
    ASSERT_EQ(C->getOutput(0), outCBefore);
}

// input: T0 is identity.
// +---+    +----+    +---+
// | A |--->| T0 |--->| B |
// +---+    +----+    +---+
// output:
// +---+    +---+
// | A |--->| B |
// +---+    +---+
TEST_F(RemoveTransposeSequencesTest, single_identity_transpose)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15)};  // Tensor 3

    Nodes              nodes;
    synTransposeParams transposeParams = {{TPD_Channel, TPD_Width, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams, sizeof(synTransposeParams)));
    nodes.push_back(gcNode("relu_fwd_f32", {2}, {3}));
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(0);
}

// input: T0-> T1, and T1->T2->T3 is identity.
// +----+    +----+    +----+    +----+    +----+    +----+
// | A  +--->| T0 +--->| T1 +--->| T2 +--->| T3 +--->| B  |
// +----+    +----+    +----+    +----+    +----+    +----+
// output:
// +----+    +----+    +----+
// | A  +--->| T0 +--->| B  |
// +----+    +----+    +----+
TEST_F(RemoveTransposeSequencesTest, overlapping_sequences)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15),   // Tensor 3
                       gcTensor(syn_type_float, 15),   // Tensor 4
                       gcTensor(syn_type_float, 15),   // Tensor 5
                       gcTensor(syn_type_float, 15)};  // Tensor 6

    Nodes              nodes;
    synTransposeParams transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParamsT2 = {{TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams transposeParamsT3 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                               // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParamsT0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParamsT1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParamsT2, sizeof(synTransposeParams)));  // T2
    nodes.push_back(gcNode("transpose", {4}, {5}, &transposeParamsT3, sizeof(synTransposeParams)));  // T3
    nodes.push_back(gcNode("relu_fwd_f32", {5}, {6}));                                               // B
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(1);
}

// input: T0 -> T1 is fusible but not identity.
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
// output:
// +---+    +----+    +---+
// | A |--->| T2 |--->| B |
// +---+    +----+    +---+
TEST_F(RemoveTransposeSequencesTest, basic_fuse_transpose_seq)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),   // Tensor 0
                       gcTensor(syn_type_float, 15),   // Tensor 1
                       gcTensor(syn_type_float, 15),   // Tensor 2
                       gcTensor(syn_type_float, 15),   // Tensor 3
                       gcTensor(syn_type_float, 15)};  // Tensor 4

    Nodes              nodes;
    synTransposeParams transposeParams0 = {{TPD_Width, TPD_Height, TPD_Channel, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                              // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {4}));                                              // B
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(1);
}

// input: T0 -> T1 is fusible but not identity, T0 output is persistent
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
// output:
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
TEST_F(RemoveTransposeSequencesTest, middle_node_with_persistent_output_fusion)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),        // Tensor 0
                       gcTensor(syn_type_float, 15),        // Tensor 1
                       gcTensor(syn_type_float, 15, true),  // Tensor 2
                       gcTensor(syn_type_float, 15),        // Tensor 3
                       gcTensor(syn_type_float, 15)};       // Tensor 4

    Nodes              nodes;
    synTransposeParams transposeParams0 = {{TPD_Width, TPD_Height, TPD_Channel, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};

    nodes.push_back(gcNode("relu_fwd_f32", {0}, {1}));                                              // A
    nodes.push_back(gcNode("transpose", {1}, {2}, &transposeParams0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(gcNode("relu_fwd_f32", {3}, {4}));                                              // B
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(2);
}

// input: T0 -> T1 -> T2 -> T3 are fusible but not identity, T1 and T3 outputs are persistent
// +-----+    +----+    +----+    +----+    +----+    +------+
// | Add |--->| T0 |--->| T1 |--->| T2 |--->| T3 |--->| Relu |
// +-----+    +----+    +----+    +----+    +----+    +------+
//
//
// output:
// +-----+    +----+    +----+    +------+
// | Add |--->|T0+1|--->|T2+3|--->| Relu |
// +-----+    +----+    +----+    +------+
TEST_F(RemoveTransposeSequencesTest, user_managed_sequence_output_and_non_adjacent_intermediate_fusible)
{
    Tensors tensors = {gcTensor(syn_type_float, 15),        // Tensor 0
                       gcTensor(syn_type_float, 15),        // Tensor 1
                       gcTensor(syn_type_float, 15),        // Tensor 2
                       gcTensor(syn_type_float, 15),        // Tensor 3
                       gcTensor(syn_type_float, 15, true),  // Tensor 4
                       gcTensor(syn_type_float, 15),        // Tensor 5
                       gcTensor(syn_type_float, 15, true),  // Tensor 6
                       gcTensor(syn_type_float, 15)};       // Tensor 7

    Nodes nodes {};

    synTransposeParams transposeParams0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams1 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams transposeParams2 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams transposeParams3 = {{TPD_Height, TPD_4Dim_Batch, TPD_Channel, TPD_Width}, 4};
    const std::string  addGuid("add_fwd_f32");
    const std::string  reluGuid("relu_fwd_f32");

    nodes.push_back(gcNode(addGuid.c_str(), {0, 1}, {2}));                                        // Add
    nodes.push_back(gcNode("transpose", {2}, {3}, &transposeParams0, sizeof(transposeParams0)));  // T0
    nodes.push_back(gcNode("transpose", {3}, {4}, &transposeParams1, sizeof(transposeParams1)));  // T1
    nodes.push_back(gcNode("transpose", {4}, {5}, &transposeParams2, sizeof(transposeParams2)));  // T2
    nodes.push_back(gcNode("transpose", {5}, {6}, &transposeParams3, sizeof(transposeParams3)));  // T3
    nodes.push_back(gcNode(reluGuid.c_str(), {6}, {7}));                                          // Relu
    init(tensors, nodes);
    run();
    validateNumOfTransposeNodes(2);
}
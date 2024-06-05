#include "slicer/bpt_handler.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "types.h"
#include "tensor_view_node.h"

using namespace gc::layered_brain;

class BPTHandlerTest : public GraphOptimizerTest
{
protected:
    TensorPtr createTensor()
    {
        return std::make_shared<Tensor>(m_tensorSizes.size(), m_tensorSizes.data(), syn_type_float);
    }

    NodePtr createNode()
    {
        NodePtr node = NodeFactory::createNode({createTensor(), createTensor()},
                                               {createTensor(), createTensor()},
                                               nullptr,
                                               TPCNode::NOP_KERNEL_NAME,
                                               "TPC" + std::to_string(m_nodeIdx++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        return node;
    }

    NodePtr addProducer(const TensorPtr& tensor)
    {
        NodePtr node = NodeFactory::createNode({createTensor()},
                                               {tensor},
                                               nullptr,
                                               TPCNode::NOP_KERNEL_NAME,
                                               "TPC" + std::to_string(m_nodeIdx++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        return node;
    }

    NodePtr addConsumer(const TensorPtr& tensor)
    {
        NodePtr node = NodeFactory::createNode({tensor},
                                               {createTensor()},
                                               nullptr,
                                               TPCNode::NOP_KERNEL_NAME,
                                               "TPC" + std::to_string(m_nodeIdx++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        return node;
    }

    void markGraphPersistentTensors()
    {
        for (auto& tensor : m_graph.getTensors())
        {
            const auto& producer  = m_graph.getTensorProducer(tensor);
            const auto  numConsumers = m_graph.getNumberOfTensorConsumers(tensor);
            // Mark graph inputs and outputs as persistent
            if (!producer || numConsumers == 0)
            {
                synMemoryDescriptor memDesc(true);
                tensor->setMemoryDescriptor(memDesc);
            }
        }
    }

    TensorPtr getTensorSlice(const TensorPtr& tensor) { return tensor->clone(false, false, false); }

    static constexpr BundleIdx BUNDLE_IDX = 4;
    Gaudi2Graph                m_graph;
    const std::vector<TSize>   m_tensorSizes = {128, 128};
    unsigned                   m_nodeIdx     = 0;
};

TEST_F(BPTHandlerTest, find_bpts_empty_bundle)
{
    NodePtr node = createNode();
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {});

    for (const auto& tensor : node->getOperands())
    {
        ASSERT_FALSE(bptHandler.isBPT(tensor));
    }
}

TEST_F(BPTHandlerTest, find_bpts_single_node_bundle_no_external_producers_and_consumers)
{
    NodePtr node = createNode();
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node});

    for (const auto& tensor : node->getOperands())
    {
        ASSERT_TRUE(bptHandler.isBPT(tensor));
    }
}

TEST_F(BPTHandlerTest, find_bpts_single_node_bundle_with_external_producers_and_cosumers)
{
    NodePtr node = createNode();
    addProducer(node->getInput(0));
    addConsumer(node->getOutput(1));
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node});

    for (const auto& tensor : node->getOperands())
    {
        ASSERT_TRUE(bptHandler.isBPT(tensor));
    }
}

TEST_F(BPTHandlerTest, find_bpts_bundle_with_producer_and_consumer)
{
    NodePtr node     = createNode();
    NodePtr producer = addProducer(node->getInput(0));
    NodePtr consumer = addConsumer(node->getOutput(1));
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node, producer, consumer});

    ASSERT_FALSE(bptHandler.isBPT(node->getInput(0)));
    ASSERT_TRUE(bptHandler.isBPT(node->getInput(1)));
    ASSERT_TRUE(bptHandler.isBPT(node->getOutput(0)));
    ASSERT_FALSE(bptHandler.isBPT(node->getOutput(1)));
    ASSERT_TRUE(bptHandler.isBPT(producer->getInput(0)));
    ASSERT_TRUE(bptHandler.isBPT(consumer->getOutput(0)));
}

TEST_F(BPTHandlerTest, find_bpts_bundle_with_intermediate_output_bpt)
{
    NodePtr node     = createNode();
    NodePtr producer = addProducer(node->getInput(0));
    // node->getOutput(1) has both internal and external consumers.
    NodePtr consumerInBundle = addConsumer(node->getOutput(1));
    NodePtr externalConsumer = addConsumer(node->getOutput(1));
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node, producer, consumerInBundle});

    ASSERT_FALSE(bptHandler.isBPT(node->getInput(0)));
    ASSERT_TRUE(bptHandler.isBPT(node->getInput(1)));
    ASSERT_TRUE(bptHandler.isBPT(node->getOutput(0)));
    ASSERT_TRUE(bptHandler.isBPT(node->getOutput(1)));  // Intermediate output BPT
    ASSERT_TRUE(bptHandler.isBPT(producer->getInput(0)));
    ASSERT_TRUE(bptHandler.isBPT(consumerInBundle->getOutput(0)));
    ASSERT_FALSE(bptHandler.isBPT(externalConsumer->getOutput(0)));
}

TEST_F(BPTHandlerTest, find_bpts_bundle_with_intermediate_persistent_output_bpt)
{
    NodePtr node = createNode();
    // node->getOutput(0) is persistent and consumed inside the bundle.
    synMemoryDescriptor memDesc(true);
    node->getOutput(0)->setMemoryDescriptor(memDesc);
    NodePtr consumer = addConsumer(node->getOutput(0));
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node, consumer});

    ASSERT_TRUE(bptHandler.isBPT(node->getInput(0)));
    ASSERT_TRUE(bptHandler.isBPT(node->getInput(1)));
    ASSERT_TRUE(bptHandler.isBPT(node->getOutput(0)));  // Intermediate output BPT
    ASSERT_TRUE(bptHandler.isBPT(node->getOutput(1)));
    ASSERT_TRUE(bptHandler.isBPT(consumer->getOutput(0)));
}

TEST_F(BPTHandlerTest, find_bpts_bundle_with_unconsumed_output)
{
    NodePtr node = createNode();
    markGraphPersistentTensors();
    // Consumer output is not persistent and not consumed in the graph
    NodePtr consumer = addConsumer(node->getOutput(0));

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node, consumer});

    ASSERT_TRUE(bptHandler.isBPT(node->getInput(0)));
    ASSERT_TRUE(bptHandler.isBPT(node->getInput(1)));
    ASSERT_TRUE(bptHandler.isBPT(node->getOutput(0)));
    ASSERT_TRUE(bptHandler.isBPT(consumer->getOutput(0)));  // Unconsumed WS tensor should be considered as BPT
}

TEST_F(BPTHandlerTest, create_aggregation_nodes_for_bpts)
{
    NodePtr node     = createNode();
    NodePtr producer = addProducer(node->getInput(0));
    NodePtr consumer = addConsumer(node->getOutput(1));
    markGraphPersistentTensors();

    BPTHandler bptHandler(m_graph, BUNDLE_IDX, {node, producer, consumer});

    // 1 slice for input BPT : node->getInput(1)
    bptHandler.addTensorSlice(node->getInput(1), getTensorSlice(node->getInput(1)), {0, 0});

    // 3 slices for output BPT : node->getOutput(0)
    bptHandler.addTensorSlice(node->getOutput(0), getTensorSlice(node->getOutput(0)), {0, 0});
    bptHandler.addTensorSlice(node->getOutput(0), getTensorSlice(node->getOutput(0)), {0, 1});
    bptHandler.addTensorSlice(node->getOutput(0), getTensorSlice(node->getOutput(0)), {0, 2});

    // 2 slices for input BPT : producer->getInput(0)
    bptHandler.addTensorSlice(producer->getInput(0), getTensorSlice(producer->getInput(0)), {0, 0});
    bptHandler.addTensorSlice(producer->getInput(0), getTensorSlice(producer->getInput(0)), {0, 1});

    // 4 slices for output BPT : consumer->getOutput(0)
    bptHandler.addTensorSlice(consumer->getOutput(0), getTensorSlice(consumer->getOutput(0)), {0, 0});
    bptHandler.addTensorSlice(consumer->getOutput(0), getTensorSlice(consumer->getOutput(0)), {0, 1});
    bptHandler.addTensorSlice(consumer->getOutput(0), getTensorSlice(consumer->getOutput(0)), {0, 2});
    bptHandler.addTensorSlice(consumer->getOutput(0), getTensorSlice(consumer->getOutput(0)), {0, 3});

    const auto& aggregationNodes = bptHandler.createForkAndJoinNodes();
    ASSERT_EQ(aggregationNodes.size(), 4);

    for (const auto& aggregationNode : aggregationNodes)
    {
        ASSERT_EQ(aggregationNode->getNodeType(), Node::TYPE_TENSOR_VIEW);
        ASSERT_TRUE(aggregationNode->getNodeAnnotation().bundleInfo.is_set());
        ASSERT_EQ(aggregationNode->getNodeAnnotation().bundleInfo->bundleIndex, BUNDLE_IDX);
        const auto& tensorViewNode = std::dynamic_pointer_cast<TensorViewNode>(aggregationNode);
        ASSERT_TRUE(tensorViewNode);
        if (tensorViewNode->realTensorIsInput())  // Fork for input BPT
        {
            ASSERT_EQ(tensorViewNode->getNumInputs(), 1);
            if (tensorViewNode->getInput(0) == node->getInput(1))
            {
                ASSERT_EQ(tensorViewNode->getNumOutputs(), 1);
            }
            else
            {
                ASSERT_EQ(tensorViewNode->getInput(0), producer->getInput(0));
                ASSERT_EQ(tensorViewNode->getNumOutputs(), 2);
            }
        }
        else  // Join for output BPT
        {
            ASSERT_EQ(tensorViewNode->getNumOutputs(), 1);
            if (tensorViewNode->getOutput(0) == node->getOutput(0))
            {
                ASSERT_EQ(tensorViewNode->getNumInputs(), 3);
            }
            else
            {
                ASSERT_EQ(tensorViewNode->getOutput(0), consumer->getOutput(0));
                ASSERT_EQ(tensorViewNode->getNumInputs(), 4);
            }
        }
    }
}
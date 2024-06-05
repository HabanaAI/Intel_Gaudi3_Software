#include "generic_graph_test.h"
#include "node_factory.h"
#include "node_utils.h"

class RemoveContinguousConvertsTest : public GenericGraphTest
{
    protected:
    virtual void SetUp() override
    {
        GenericGraphTest::SetUp();
    }
};

TEST_P(RemoveContinguousConvertsTest, remove_contiguous_converts_one_remove_fp8_to_bf16)
{
    SizeArray firstSize = {25, 1};
    SizeArray secondSize = {5, 5};
    SizeArray scaleSize = {1, 1};
    std::vector<float> scaleData = {1};
    char* scaleBuffer = reinterpret_cast<char*>(scaleData.data());

    TensorPtr t1 = std::make_shared<Tensor>(1U, firstSize.data(), syn_type_fp8_143);
    TensorPtr t2 = std::make_shared<Tensor>(2U, secondSize.data(), syn_type_fp8_143);
    TensorPtr t3 = std::make_shared<Tensor>(2U, secondSize.data(), syn_type_bf16);
    TensorPtr t4 = std::make_shared<Tensor>(2U, secondSize.data(), syn_type_fp8_143);
    TensorPtr t5 = std::make_shared<Tensor>(1U, firstSize.data(), syn_type_fp8_143);

    TensorPtr scale1 = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scale1->setAsStaticParam(true);
    scale1->setTensorBuffer(scaleBuffer, 1 * sizeof(float), syn_type_float);

    TensorPtr scale2 = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scale2->setTensorBuffer(scaleBuffer, 1 * sizeof(float), syn_type_float);
    scale2->setAsStaticParam(true);

    NodePtr   reshape1                 = NodeFactory::createNode({t1}, {t2}, nullptr,
                                                                 NodeFactory::reshapeNodeTypeName,
                                                                 "reshape1");
    NodePtr   convert_from_fp8_to_bf16 = NodeFactory::createNode({t2, scale1}, {t3}, nullptr,
                                                                 "convert_from_fp8_bf16",
                                                                 "convert_from_fp8_to_bf16");
    NodePtr   convert_from_bf16_to_fp8 = NodeFactory::createNode({t3, scale2}, {t4}, nullptr,
                                                                 "convert_to_fp8_bf16",
                                                                 "convert_from_bf16_to_fp8");
    NodePtr   reshape2                 = NodeFactory::createNode({t4}, {t5}, nullptr,
                                                                 NodeFactory::reshapeNodeTypeName,
                                                                 "reshape2");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, reshape1))
                << "Failed to add reshape1 node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, convert_from_fp8_to_bf16))
                << "Failed to add convert_from_fp8_to_bf16 node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, convert_from_bf16_to_fp8))
                << "Failed to add convert_from_bf16_to_fp8 node to graph";

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, reshape2))
                << "Failed to add reshape2 node to graph";

    removeContinguousConverts(*m_graph);

    ASSERT_EQ(m_graph->getNodes().size(), 2);
    for(auto node : m_graph->getNodes())
    {
        ASSERT_EQ(isConvertFp8Node(node), false);
    }
}

TEST_P(RemoveContinguousConvertsTest, remove_contiguous_converts_one_remove_bf16_to_fp8)
{
    SizeArray firstSize = {25, 1};
    SizeArray secondSize = {5, 5};
    SizeArray scaleSize = {1, 1};
    std::vector<float> scaleData = {1};
    char* scaleBuffer = reinterpret_cast<char*>(scaleData.data());

    TensorPtr t1 = std::make_shared<Tensor>(1U, firstSize.data(), syn_type_bf16);
    TensorPtr t2 = std::make_shared<Tensor>(2U, secondSize.data(), syn_type_bf16);
    TensorPtr t3 = std::make_shared<Tensor>(2U, secondSize.data(), syn_type_fp8_143);
    TensorPtr t4 = std::make_shared<Tensor>(2U, secondSize.data(), syn_type_bf16);
    TensorPtr t5 = std::make_shared<Tensor>(1U, firstSize.data(), syn_type_bf16);

    TensorPtr scale1 = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scale1->setAsStaticParam(true);
    scale1->setTensorBuffer(scaleBuffer, 1 * sizeof(float), syn_type_float);

    TensorPtr scale2 = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scale2->setTensorBuffer(scaleBuffer, 1 * sizeof(float), syn_type_float);
    scale2->setAsStaticParam(true);

    NodePtr   reshape1                 = NodeFactory::createNode({t1}, {t2}, nullptr,
                                                                 NodeFactory::reshapeNodeTypeName,
                                                                 "reshape1");
    NodePtr   convert_from_bf16_to_fp8 = NodeFactory::createNode({t2, scale1}, {t3}, nullptr,
                                                                 "convert_to_fp8_bf16",
                                                                 "convert_from_bf16_to_fp8");
    NodePtr   convert_from_fp8_to_bf16 = NodeFactory::createNode({t3, scale2}, {t4}, nullptr,
                                                                 "convert_from_fp8_bf16",
                                                                 "convert_from_fp8_to_bf16");
    NodePtr   reshape2                 = NodeFactory::createNode({t4}, {t5}, nullptr,
                                                                 NodeFactory::reshapeNodeTypeName,
                                                                 "reshape2");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, reshape1))
                << "Failed to add reshape1 node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, convert_from_bf16_to_fp8))
                << "Failed to add convert_from_bf16_to_fp8 node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, convert_from_fp8_to_bf16))
                << "Failed to add convert_from_fp8_to_bf16 node to graph";
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, reshape2))
                << "Failed to add reshape2 node to graph";

    removeContinguousConverts(*m_graph);

    ASSERT_EQ(m_graph->getNodes().size(), 2);
    for(auto node : m_graph->getNodes())
    {
        ASSERT_EQ(isConvertFp8Node(node), false);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         RemoveContinguousConvertsTest,
                         ::testing::Values(synDeviceGaudi2, synDeviceGaudi3),
                         GenericGraphTest::GetName());
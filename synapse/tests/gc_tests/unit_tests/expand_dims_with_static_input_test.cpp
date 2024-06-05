#include <gtest/gtest.h>
#include "tensor.h"
#include "test_utils.h"
#include "node_factory.h"
#include "cast_utils.hpp"
#include "graph_factory.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"

class ExpandDimsFoldingTest
: public GraphOptimizerTest
, public testing::WithParamInterface<synDeviceType>
{
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        synDeviceType deviceType = GetParam();
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
        m_graph->setInferenceMode(true);
    }

    void TearDown() override { GraphOptimizerTest::TearDown(); }

protected:
    std::unique_ptr<HabanaGraph> m_graph;
    bool                         m_isStaticInput     = false;
    bool                         m_isPersistentInput = false;
    synNodeId                    m_extpandDimsId;
    uint64_t                     m_expandDimsOutput;
    uint64_t                     m_expandDimsInput;

    void runTest();
};

void ExpandDimsFoldingTest::runTest()
{
    const unsigned width = 32, height = 32, batchSize = 10;
    SizeArray      inSizes            = {height, batchSize};
    SizeArray      expandDimsOutSizes = {1, height, batchSize};
    SizeArray      gemmmInSizes       = {width, 1, batchSize};
    SizeArray      gemmmOuSizes       = {width, height, batchSize};

    pTensor expandDimsIn  = pTensor(new Tensor(2, inSizes.data(), syn_type_float));
    pTensor expandDimsOut = pTensor(new Tensor(3, expandDimsOutSizes.data(), syn_type_float));
    if (m_isPersistentInput) setTensorAsPersistent(expandDimsIn, 0);
    m_expandDimsInput  = expandDimsIn->getId();
    m_expandDimsOutput = expandDimsOut->getId();
    unsigned expandDim = 0;
    if (m_isStaticInput)
    {
        expandDimsIn->setAsStaticParam(true);
        expandDimsIn->setAsDataTypeMatchData();
    }
    NodePtr expandDims =
        NodeFactory::createNode({expandDimsIn}, {expandDimsOut}, &expandDim, "expand_dims", "expandDims");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, expandDims));
    m_extpandDimsId = expandDims->getId();

    pTensor gemmIn  = pTensor(new Tensor(3, gemmmInSizes.data(), syn_type_float));
    pTensor gemmOut = pTensor(new Tensor(2, gemmmOuSizes.data(), syn_type_float));

    synGEMMParams gemmParams {};
    NodePtr       gemm = NodeFactory::createNode({gemmIn, expandDimsOut}, {gemmOut}, &gemmParams, "batch_gemm", "gemm");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, gemm));
}

TEST_P(ExpandDimsFoldingTest, remove_expand_dims_with_static_input)
{
    m_isStaticInput = true;
    runTest();
    ASSERT_TRUE(eliminateNodesWithStaticInputs(*m_graph));
    ASSERT_TRUE(m_graph->getNodeByID(m_extpandDimsId) == nullptr) << "Expand dims node should be removed";
    const TensorSet& tensors              = m_graph->getTensors();
    bool             isInputTensorInGraph = std::any_of(tensors.begin(), tensors.end(), [&](const TensorPtr& tensor) {
        return (tensor && tensor->getId() == m_expandDimsInput);
    });
    ASSERT_TRUE(!isInputTensorInGraph) << "Expecting original input tensor to be removed";
    auto outputTensor = std::find_if(tensors.begin(), tensors.end(), [&](const TensorPtr& tensor) {
        return (tensor && tensor->getId() == m_expandDimsOutput);
    });
    ASSERT_TRUE(outputTensor != tensors.end()) << "Expecting original output tensor to be the graph";
    ASSERT_TRUE((*outputTensor)->isStaticParam())
        << "Expecting output of expand dims node to be static after the node is removed";
}

TEST_P(ExpandDimsFoldingTest, dont_remove_expand_dims_with_persistent_input)
{
    m_isPersistentInput = true;
    runTest();
    ASSERT_TRUE(m_graph->getNodeByID(m_extpandDimsId) != nullptr) << "Expand dims node should not be removed";
}

TEST_P(ExpandDimsFoldingTest, dont_remove_expand_dims_with_not_static_input)
{
    runTest();
    ASSERT_TRUE(m_graph->getNodeByID(m_extpandDimsId) != nullptr) << "Expand dims node should not be removed";
}

INSTANTIATE_TEST_SUITE_P(, ExpandDimsFoldingTest, ::testing::Values(synDeviceGaudi, synDeviceGaudi2));
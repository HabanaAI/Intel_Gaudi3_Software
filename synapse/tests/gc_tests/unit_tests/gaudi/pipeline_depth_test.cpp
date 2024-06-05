#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"

#include "define_synapse_common.hpp"

class PipelineDepthTest : public GraphOptimizerTest
{
    virtual void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        // optimize tpc may change the node shape and change the number of pipeline levels a little.
        // disabling this for test verification
        setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    }
};

TEST_F(PipelineDepthTest, concat_relu_pipeline)
{
    setGlobalConfForTest(GCFG_DEFAULT_PIPELINE_DEPTH, "3");

    SizeArray singleSize = {100, 100, 100, 1, 1};
    constexpr uint32_t concatNumInputs = 5;
    TensorVector       concatInputs;
    TensorVector       splitOutputs;
    for (uint32_t i = 0; i < concatNumInputs; ++i)
    {
        pTensor concatInput(new Tensor(4, singleSize.data(), syn_type_bf16));
        concatInput->setMemoryDescriptor(synMemoryDescriptor(true));
        concatInput->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i);
        concatInputs.push_back(concatInput);
        pTensor splitOutput(new Tensor(4, singleSize.data(), syn_type_bf16));
        splitOutputs.push_back(splitOutput);
    }

    SizeArray reluSize = {100, 100, 100, concatNumInputs, 1};
    pTensor concatTOut(new Tensor(4, reluSize.data(), syn_type_bf16));
    pTensor reluTOut(new Tensor(4, reluSize.data(), syn_type_bf16));

    pTensor secondReluOut(new Tensor(4, singleSize.data(), syn_type_bf16));

    GaudiGraph graph;
    unsigned int concatDim = 3;
    pNode concat = NodeFactory::createNode(concatInputs,
                                           {concatTOut},
                                           &concatDim,
                                           sizeof(concatDim),
                                           NodeFactory::concatenateNodeTypeName,
                                           "concat");

    pNode relu = NodeFactory::createNode({concatTOut},
                                         {reluTOut},
                                         nullptr,
                                         0,
                                         "relu_fwd_bf16",
                                         "relu");

    pNode split = NodeFactory::createNode({reluTOut},
                                          splitOutputs,
                                          &concatDim,
                                          sizeof(concatDim),
                                          NodeFactory::splitNodeTypeName,
                                          "split");

    pNode secondRelu = NodeFactory::createNode({splitOutputs[0]},
                                               {secondReluOut},
                                               nullptr,
                                               0,
                                               "relu_fwd_bf16",
                                               "second_relu");

    ASSERT_TRUE(GraphEditor::addNode(graph, concat));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));
    ASSERT_TRUE(GraphEditor::addNode(graph, split));
    ASSERT_TRUE(GraphEditor::addNode(graph, secondRelu));
    ASSERT_TRUE(graph.compile());
    bool firstNode = true;
    for (auto node : graph.getExeSortedNodes())
    {
        if (HabanaGraph::runsOnTPC(node))
        {
            if (firstNode)
            {
                ASSERT_EQ(concatNumInputs, graph.GetNodeROIs(node)->size());
                firstNode = false;
            }
            else
            {
                ASSERT_EQ(GCFG_DEFAULT_PIPELINE_DEPTH.value(), graph.GetNodeROIs(node)->size());
            }
        }
    }
}

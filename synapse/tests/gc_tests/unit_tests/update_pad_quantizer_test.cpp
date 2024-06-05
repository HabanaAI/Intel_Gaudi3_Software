#include <gtest/gtest.h>
#include "utils.h"
#include "quantizer.h"
#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "perf_lib_layer_params.h"
#include <map>

class UpdatePadQuantizerTest : public GraphOptimizerTest
{
protected:
    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GCFG_ENABLE_SYNAPSE_QUANTIZATION.setValue(true);
    }

    virtual void TearDown()
    {
        GraphOptimizerTest::TearDown();
    }
};

TEST_F(UpdatePadQuantizerTest, inputPadMmeTest)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    pTensor input2      = std::make_shared<Tensor>(syn_type_int8);
    pTensor padOutput2  = std::make_shared<Tensor>(syn_type_int8);
    pTensor mme2Output2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor convW       = std::make_shared<Tensor>(syn_type_int8);

    ns_PadKernel::Params padParams;
    pNode padNode = NodeFactory::createGenericTPCNode({input2}, {padOutput2}, &padParams, "pad", "");
    synConvolution3DParams convParams;
    pNode convNode = NodeFactory::createNode({padOutput2, convW, nullptr, nullptr},
                                             {mme2Output2}, &convParams, NodeFactory::convolutionNodeTypeName, "");

    GraphEditor::addNode(g, padNode);
    GraphEditor::addNode(g, convNode);

    QuantizerPtr padQuantizer;
    std::shared_ptr<BackwardQuantizer> bwdQuantizer;
    padQuantizer = padNode->getQuantizer();
    bwdQuantizer = std::dynamic_pointer_cast<BackwardQuantizer>(padQuantizer);
    ASSERT_TRUE(bwdQuantizer == nullptr);
    ASSERT_TRUE(updatePadQuantizer(g));
    padQuantizer = padNode->getQuantizer();
    bwdQuantizer = std::dynamic_pointer_cast<BackwardQuantizer>(padQuantizer);
    ASSERT_TRUE(bwdQuantizer != nullptr);
}

TEST_F(UpdatePadQuantizerTest, mmePadMmeTest)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    pTensor input1      = std::make_shared<Tensor>(syn_type_int8);
    pTensor mme1Output1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor padOutput1  = std::make_shared<Tensor>(syn_type_int8);
    pTensor mme2Output1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor convW       = std::make_shared<Tensor>(syn_type_int8);

    synConvolution3DParams convParams;
    pNode convNode1 = NodeFactory::createNode({input1, convW, nullptr, nullptr},
                                              {mme1Output1}, &convParams, NodeFactory::convolutionNodeTypeName, "");
    ns_PadKernel::Params padParams;
    pNode padNode = NodeFactory::createGenericTPCNode({mme1Output1}, {padOutput1}, &padParams, "pad", "");
    pNode convNode2 = NodeFactory::createNode({padOutput1, convW, nullptr, nullptr},
                                             {mme2Output1}, &convParams, NodeFactory::convolutionNodeTypeName, "");

    GraphEditor::addNode(g, convNode1);
    GraphEditor::addNode(g, padNode);
    GraphEditor::addNode(g, convNode2);

    QuantizerPtr padQuantizer;
    std::shared_ptr<BackwardQuantizer> bwdQuantizer;
    padQuantizer = padNode->getQuantizer();
    bwdQuantizer = std::dynamic_pointer_cast<BackwardQuantizer>(padQuantizer);
    ASSERT_TRUE(bwdQuantizer == nullptr);
    ASSERT_TRUE(updatePadQuantizer(g));
    padQuantizer = padNode->getQuantizer();
    bwdQuantizer = std::dynamic_pointer_cast<BackwardQuantizer>(padQuantizer);
    ASSERT_TRUE(bwdQuantizer != nullptr);
}

TEST_F(UpdatePadQuantizerTest, nonInputPadMmeTest)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    pTensor input1      = std::make_shared<Tensor>(syn_type_int8);
    pTensor reluOutput1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor padOutput1  = std::make_shared<Tensor>(syn_type_int8);
    pTensor mme2Output1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor convW       = std::make_shared<Tensor>(syn_type_int8);

    synConvolution3DParams convParams;
    pNode reluNode = NodeFactory::createNode({input1},
                                              {reluOutput1}, &convParams, NodeFactory::reluNodeTypeName, "");
    ns_PadKernel::Params padParams;
    pNode padNode = NodeFactory::createGenericTPCNode({reluOutput1}, {padOutput1}, &padParams, "pad", "");
    pNode convNode = NodeFactory::createNode({padOutput1, convW, nullptr, nullptr},
                                              {mme2Output1}, &convParams, NodeFactory::convolutionNodeTypeName, "");

    GraphEditor::addNode(g, reluNode);
    GraphEditor::addNode(g, padNode);
    GraphEditor::addNode(g, convNode);

    QuantizerPtr padQuantizer;
    std::shared_ptr<BackwardQuantizer> bwdQuantizer;
    padQuantizer = padNode->getQuantizer();
    bwdQuantizer = std::dynamic_pointer_cast<BackwardQuantizer>(padQuantizer);
    ASSERT_TRUE(bwdQuantizer == nullptr);
    ASSERT_TRUE(updatePadQuantizer(g));
    padQuantizer = padNode->getQuantizer();
    bwdQuantizer = std::dynamic_pointer_cast<BackwardQuantizer>(padQuantizer);
    ASSERT_TRUE(bwdQuantizer == nullptr);
}
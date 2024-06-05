#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"

namespace gaudi
{
class LayerNormBwdTest
: public GraphOptimizerTest
, public testing::WithParamInterface<synDataType>
{
public:
    LayerNormBwdTest() : m_dataType(GetParam()) {}

    static const unsigned countNonAuxInputs(pNode node)
    {
        unsigned nonAuxInputs = 0;
        for (auto tensor : node->getInputs())
        {
            if (!tensor->isAuxTensor())
            {
                nonAuxInputs++;
            }
        }
        return nonAuxInputs;
    }

protected:
    synDataType m_dataType;
};

class LayerNormBwdSplitTest : public LayerNormBwdTest
{
public:
    void runSingleSplitTest();
};

class LayerNormBwdNoSplitTest : public LayerNormBwdTest
{
public:
    void runSingleNoSplitTest();
};

void LayerNormBwdSplitTest::runSingleSplitTest()
{
    // This test should check GC logic and graph not be dependent on suggested TPC manipulations
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    GaudiGraph g;
    TSize sizes[] = {1, 8, 2, 2};
    TSize sizes1Dim[] = {1, 1, 1, 1};
    pTensor    ifmTensor(new Tensor(4, sizes, m_dataType));
    pTensor    gradInTensor(new Tensor(4, sizes, m_dataType));
    pTensor meanTensor(new Tensor(4, sizes, syn_type_float));
    pTensor lstdTensor(new Tensor(4, sizes, syn_type_float));
    pTensor gammaTensor(new Tensor(1, sizes1Dim, syn_type_float));
    pTensor    gradOutTensor(new Tensor(4, sizes, m_dataType));
    pTensor gradBetaTensor(new Tensor(1, sizes1Dim, syn_type_float));
    pTensor gradGammaTensor(new Tensor(1, sizes1Dim, syn_type_float));

    ifmTensor->setDramOffset(0x1000);
    gradInTensor->setDramOffset(0x2000);
    meanTensor->setDramOffset(0x3000);
    lstdTensor->setDramOffset(0x4000);
    gammaTensor->setDramOffset(0x5000);
    gradOutTensor->setDramOffset(0x6000);
    gradBetaTensor->setDramOffset(0x7000);
    gradGammaTensor->setDramOffset(0x8000);

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    synMemoryDescriptor memDescPersist(true);
    ifmTensor->setMemoryDescriptor(memDescPersist);
    ifmTensor->setMemorySectionID(memSecId++);
    gradInTensor->setMemoryDescriptor(memDescPersist);
    gradInTensor->setMemorySectionID(memSecId++);
    meanTensor->setMemoryDescriptor(memDescPersist);
    meanTensor->setMemorySectionID(memSecId++);
    lstdTensor->setMemoryDescriptor(memDescPersist);
    lstdTensor->setMemorySectionID(memSecId++);
    gammaTensor->setMemoryDescriptor(memDescPersist);
    gammaTensor->setMemorySectionID(memSecId++);
    gradOutTensor->setMemoryDescriptor(memDescPersist);
    gradOutTensor->setMemorySectionID(memSecId++);
    gradBetaTensor->setMemoryDescriptor(memDescPersist);
    gradBetaTensor->setMemorySectionID(memSecId++);
    gradGammaTensor->setMemoryDescriptor(memDescPersist);
    gradGammaTensor->setMemorySectionID(memSecId++);

    ns_LayerNormKernel::Params params;
    params.eps = 0.1;
    params.epsValid = false;

    auto n = NodeFactory::createNode({ifmTensor, gradInTensor, meanTensor, lstdTensor, gammaTensor},
                                     {gradOutTensor, gradBetaTensor, gradGammaTensor},
                                     &params,
                                     "layer_norm_bwd_f32",
                                     "layerNorm");
    GraphEditor::addNode(g, n);
    bool ret = g.compile();

    //validations:
    ASSERT_TRUE(ret) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    // 2 memset, 2 Reduction, Stag1 and Stage2, optional DMA nodes for scalar pipe
    ASSERT_GE(nodes.size(), 6);

    // Check that layerNorm-bwd was split based on the node prefix
    auto layerNormStage1 = std::find_if(std::begin(nodes), std::end(nodes), [&](const NodePtr& node) {
        return (node->getGUID().rfind("layer_norm_stage1_bwd", 0) == 0);
    });

    ASSERT_TRUE(nodes.end() != layerNormStage1) << "Missing layer_norm_stage1_bwd_xxx";

    auto layerNormStage2 = std::find_if(std::begin(nodes), std::end(nodes), [&](const NodePtr& node) {
        return (node->getGUID().rfind("layer_norm_stage2_bwd", 0) == 0);
    });

    ASSERT_TRUE(nodes.end() != layerNormStage2) << "Missing layer_norm_stage2_bwd_xxx";
    ASSERT_EQ(countNonAuxInputs(*layerNormStage1), 5);
    ASSERT_EQ((*layerNormStage1)->getNumOutputs(), 3);

    pNode reduction1 = g.getTensorConsumers((*layerNormStage1)->getOutput(1)).front();
    pNode reduction2 = g.getTensorConsumers((*layerNormStage1)->getOutput(2)).front();

    ASSERT_EQ(reduction1->getNodeTypeStr(), "Reduction");
    ASSERT_EQ(reduction2->getNodeTypeStr(), "Reduction");

    ASSERT_EQ(countNonAuxInputs(*layerNormStage2), 2);
    ASSERT_EQ((*layerNormStage2)->getNumOutputs(), 2);
    ASSERT_EQ((*layerNormStage2)->getInput(0), reduction1->getOutput(0));
    ASSERT_EQ((*layerNormStage2)->getInput(1), reduction2->getOutput(0));
    ASSERT_EQ((*layerNormStage2)->getOutput(0), gradGammaTensor);
    ASSERT_EQ((*layerNormStage2)->getOutput(1), gradBetaTensor);
}

TEST_P(LayerNormBwdSplitTest, layer_norm_bwd_split)
{
    runSingleSplitTest();
}

INSTANTIATE_TEST_SUITE_P(layerNorm_bwd_dataType,
                         LayerNormBwdSplitTest,
                         ::testing::Values(syn_type_float, syn_type_bf16));

void LayerNormBwdNoSplitTest::runSingleNoSplitTest()
{
    setGlobalConfForTest(GCFG_SKIP_LAYER_NORM_BWD_SPLIT, "1");

    GaudiGraph g;
    TSize sizes4D[] = {1, 16, 5, 6};
    TSize sizes1D[] = {1, 1, 1, 1};
    pTensor    ifmTensor(new Tensor(4, sizes4D, m_dataType));
    pTensor    gradInTensor(new Tensor(4, sizes4D, m_dataType));
    pTensor meanTensor(new Tensor(4, sizes4D, syn_type_float));
    pTensor lstdTensor(new Tensor(4, sizes4D, syn_type_float));
    pTensor gammaTensor(new Tensor(1, sizes1D, syn_type_float));
    pTensor    gradOutTensor(new Tensor(4, sizes4D, m_dataType));
    pTensor gradBetaTensor(new Tensor(1, sizes1D, syn_type_float));
    pTensor gradGammaTensor(new Tensor(1, sizes1D, syn_type_float));

    ifmTensor->setDramOffset(0x1000);
    gradInTensor->setDramOffset(0x2000);
    meanTensor->setDramOffset(0x3000);
    lstdTensor->setDramOffset(0x4000);
    gammaTensor->setDramOffset(0x5000);
    gradOutTensor->setDramOffset(0x6000);
    gradBetaTensor->setDramOffset(0x7000);
    gradGammaTensor->setDramOffset(0x8000);

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    synMemoryDescriptor memDescPersist(true);
    ifmTensor->setMemoryDescriptor(memDescPersist);
    ifmTensor->setMemorySectionID(memSecId++);
    gradInTensor->setMemoryDescriptor(memDescPersist);
    gradInTensor->setMemorySectionID(memSecId++);
    meanTensor->setMemoryDescriptor(memDescPersist);
    meanTensor->setMemorySectionID(memSecId++);
    lstdTensor->setMemoryDescriptor(memDescPersist);
    lstdTensor->setMemorySectionID(memSecId++);
    gammaTensor->setMemoryDescriptor(memDescPersist);
    gammaTensor->setMemorySectionID(memSecId++);
    gradOutTensor->setMemoryDescriptor(memDescPersist);
    gradOutTensor->setMemorySectionID(memSecId++);
    gradBetaTensor->setMemoryDescriptor(memDescPersist);
    gradBetaTensor->setMemorySectionID(memSecId++);
    gradGammaTensor->setMemoryDescriptor(memDescPersist);
    gradGammaTensor->setMemorySectionID(memSecId++);

    ns_LayerNormKernel::Params params;
    params.eps = 0.1;
    params.epsValid = false;

    pNode node = NodeFactory::createNode({ifmTensor, gradInTensor, meanTensor, lstdTensor, gammaTensor},
                                         {gradOutTensor, gradBetaTensor, gradGammaTensor},
                                         &params,
                                         "layer_norm_bwd_f32",
                                         "layerNorm");
    GraphEditor::addNode(g, node);
    bool ret = g.compile();

    //validations:
    ASSERT_TRUE(ret) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    // Added 5 reshapes after tpc kernel optimization, optional DMA nodes for scalar pipe
    ASSERT_GE(nodes.size(), 6);

    pNode layerNormBwd;
    for (const pNode& n : nodes)
    {
        bool isReshapeNode = (dynamic_cast<ReshapeNode*>(n.get()) != nullptr);
        if (!isReshapeNode)
        {
            layerNormBwd = n;
        }
    }

    ASSERT_EQ(layerNormBwd->getGUID(), "layer_norm_bwd_f32");
    ASSERT_EQ(countNonAuxInputs(layerNormBwd), 5);
    ASSERT_EQ(layerNormBwd->getNumOutputs(), 3);
    ASSERT_EQ(layerNormBwd->getOutput(1), gradBetaTensor);
    ASSERT_EQ(layerNormBwd->getOutput(2), gradGammaTensor);
}

TEST_P(LayerNormBwdNoSplitTest, layer_norm_bwd_no_split)
{
    runSingleNoSplitTest();
}

INSTANTIATE_TEST_SUITE_P(layer_norm_no_split_float, LayerNormBwdNoSplitTest, ::testing::Values(syn_type_float));

} // namespace gaudi

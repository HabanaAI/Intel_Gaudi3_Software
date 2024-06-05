#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "perf_lib_layer_params.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

// Testing memsetNodeOutput pass
class GaudiMemsetOutputTensorTest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_F(GaudiMemsetOutputTensorTest, memsetBeforeExecutionDram)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");

    GaudiGraph g;

    const TSize inSize[] = {10, 1, 1, 1};
    const TSize outSize[] = {15, 1, 1, 1};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor gradTensor = pTensor(new Tensor(4U, inSize, syn_type_float));
    gradTensor->setName("gradTensor");
    gradTensor->setDramOffset(0x1000);
    gradTensor->setMemoryDescriptor(persistentMemoryDesc);
    gradTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    pTensor indicesTensor = pTensor(new Tensor(4U, inSize, syn_type_int32));
    indicesTensor->setName("indicesTensor");
    indicesTensor->setDramOffset(0x3000);
    indicesTensor->setMemoryDescriptor(persistentMemoryDesc);
    indicesTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    pTensor outputTensor = pTensor(new Tensor(4U, outSize, syn_type_float));
    outputTensor->setName("outputTensor");
    outputTensor->setDramOffset(0x5000);
    outputTensor->setMemoryDescriptor(persistentMemoryDesc);
    outputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    ns_SortBwd::Params params;
    params.axis = 0;

    pNode sortNode = NodeFactory::createGenericTPCNode({gradTensor, indicesTensor}, {outputTensor}, &params, "sort_bwd_f32", "sortNode");
    GraphEditor::addNode(g, sortNode);

    bool retVal = g.compile();
    ASSERT_EQ(retVal, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 4) << "Got " << nodes.size() << ", Expected 4";

    ASSERT_EQ((*std::next(nodes.begin(), 0))->getNodeTypeStr(), "DmaMemcpy");

    pNode memset_node = *std::next(nodes.begin(),    1);
    pNode sort_node = *std::next(nodes.begin(),      2);
    pNode reduction_node = *std::next(nodes.begin(), 3);

    ASSERT_EQ(memset_node->isMemset(), true);
    ASSERT_EQ(memset_node->getNumInputs(),  0);
    ASSERT_EQ(memset_node->getNumOutputs(), 1);

    ASSERT_EQ(sort_node->getNodeTypeStr(), "sort_bwd_f32");
    ASSERT_EQ(sort_node->getNumInputs(),  2);
    ASSERT_EQ(sort_node->getNumOutputs(), 1);

    ASSERT_EQ(reduction_node->getNodeTypeStr(), "Reduction");
    ASSERT_EQ(reduction_node->getNumInputs(),  2);
    ASSERT_EQ(reduction_node->getNumOutputs(), 1);

    ASSERT_EQ(reduction_node->getInput(0),  memset_node->getOutput(0));
    ASSERT_EQ(reduction_node->getInput(1),  sort_node->getOutput(0));
    ASSERT_EQ(reduction_node->getOutput(0), outputTensor);
}

TEST_F(GaudiMemsetOutputTensorTest, memsetBeforeExecutionSram)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");

    GaudiGraph g;

    const TSize inSize[] = {10, 1, 1, 1};
    const TSize outSize[] = {15, 1, 1, 1};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor gradTensor = pTensor(new Tensor(4U, inSize, syn_type_float));
    gradTensor->setName("gradTensor");
    gradTensor->setDramOffset(0x1000);
    gradTensor->setMemoryDescriptor(persistentMemoryDesc);
    gradTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    pTensor indicesTensor = pTensor(new Tensor(4U, inSize, syn_type_int32));
    indicesTensor->setName("indicesTensor");
    indicesTensor->setDramOffset(0x3000);
    indicesTensor->setMemoryDescriptor(persistentMemoryDesc);
    indicesTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    pTensor outputTensor = pTensor(new Tensor(4U, outSize, syn_type_float));
    outputTensor->setName("outputTensor");
    outputTensor->setTensorInSram();

    ns_SortBwd::Params params;
    params.axis = 0;

    pNode sortNode = NodeFactory::createGenericTPCNode({gradTensor, indicesTensor}, {outputTensor}, &params, "sort_bwd_f32", "sortNode");
    GraphEditor::addNode(g, sortNode);

    bool retVal = g.compile();
    ASSERT_EQ(retVal, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 4) << "Got " << nodes.size() << ", Expected 4";

    ASSERT_EQ((*std::next(nodes.begin(), 0))->getNodeTypeStr(), "DmaMemcpy");

    pNode memset_node = *std::next(nodes.begin(),    1);
    pNode sort_node = *std::next(nodes.begin(),      2);
    pNode reduction_node = *std::next(nodes.begin(), 3);

    ASSERT_EQ(memset_node->isMemset(), true);
    ASSERT_EQ(memset_node->getNumInputs(),  0);
    ASSERT_EQ(memset_node->getNumOutputs(), 1);

    ASSERT_EQ(sort_node->getNodeTypeStr(), "sort_bwd_f32");
    ASSERT_EQ(sort_node->getNumInputs(),  2);
    ASSERT_EQ(sort_node->getNumOutputs(), 1);

    ASSERT_EQ(reduction_node->getNodeTypeStr(), "Reduction");
    ASSERT_EQ(reduction_node->getNumInputs(),  2);
    ASSERT_EQ(reduction_node->getNumOutputs(), 1);

    ASSERT_EQ(reduction_node->getInput(0),  memset_node->getOutput(0));
    ASSERT_EQ(reduction_node->getInput(1),  sort_node->getOutput(0));
    ASSERT_EQ(reduction_node->getOutput(0), g.getGraphOutputs().front());
    ASSERT_EQ(g.getGraphOutputs().size(), 1);
}

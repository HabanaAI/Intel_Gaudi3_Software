#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "perf_lib_layer_params.h"


// Testing memsetNodeOutput  pass
class GaudiInputInplaceReuseTest : public GraphOptimizerTest {};

TEST_F(GaudiInputInplaceReuseTest, input_reuse_different_sections)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    setGlobalConfForTest(GCFG_ENABLE_INPUT_REUSE_AS_LOGICAL_NODE, "true");

    GaudiGraph g;

    TSize dataDims[4] = {1, 2, 2, 1};
    TSize idxDims[4] = {1, 1, 2, 1};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor inputData = pTensor(new Tensor(4U, dataDims, syn_type_bf16));
    inputData->setName("inputData");
    inputData->setDramOffset(0x1000);
    inputData->setMemoryDescriptor(persistentMemoryDesc);
    inputData->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    pTensor indicesTensor = pTensor(new Tensor(4U, idxDims, syn_type_int32));
    indicesTensor->setName("indicesTensor");
    indicesTensor->setDramOffset(0x3000);
    indicesTensor->setMemoryDescriptor(persistentMemoryDesc);
    indicesTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    pTensor updatesTensor = pTensor(new Tensor(4U, idxDims, syn_type_bf16));
    updatesTensor->setName("updatesTensor");
    updatesTensor->setDramOffset(0x5000);
    updatesTensor->setMemoryDescriptor(persistentMemoryDesc);
    updatesTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    pTensor outputTensor = pTensor(new Tensor(4U, dataDims, syn_type_bf16));
    outputTensor->setName("outputTensor");
    outputTensor->setDramOffset(0x7000);
    outputTensor->setMemoryDescriptor(persistentMemoryDesc);
    outputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);

    ns_ScatterKernel::Params params;
    params.axis = 1;

    pNode gatherNode = NodeFactory::createGenericTPCNode({inputData, indicesTensor, updatesTensor}, {outputTensor},
                                                         &params, "gather_bwd_bf16", "gatherNode");
    GraphEditor::addNode(g, gatherNode);

    bool retVal = g.compile();
    ASSERT_EQ(retVal, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 3) << "Got " << nodes.size() << ", Expected 3";

    gatherNode            = *std::next(nodes.begin(), 2);
    pNode memcpyIn        = g.getTensorProducer(gatherNode->getInput(0));
    pNode tpcSramMemcpyIn = g.getTensorProducer(gatherNode->getInput(1));

    ASSERT_NE(memcpyIn, nullptr);
    ASSERT_EQ(memcpyIn->getNodeType(), Node::TYPE_DMA);
    ASSERT_EQ(memcpyIn->getNumInputs(),  1);
    ASSERT_EQ(memcpyIn->getNumOutputs(), 1);
    ASSERT_EQ(memcpyIn->getInput(0),  inputData);

    ASSERT_NE(tpcSramMemcpyIn, nullptr);
    ASSERT_EQ(tpcSramMemcpyIn->getNodeType(), Node::TYPE_DMA);
    ASSERT_EQ(tpcSramMemcpyIn->getNumInputs(),  1);
    ASSERT_EQ(tpcSramMemcpyIn->getNumOutputs(), 1);
    ASSERT_EQ(tpcSramMemcpyIn->getInput(0),  indicesTensor);

    ASSERT_EQ(gatherNode->getNodeTypeStr(), "gather_bwd_bf16");
    ASSERT_EQ(gatherNode->getNumInputs(), 3);
    ASSERT_EQ(gatherNode->getNumOutputs(), 1);
}

TEST_F(GaudiInputInplaceReuseTest, input_reuse_shared_memory)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    setGlobalConfForTest(GCFG_ENABLE_INPUT_REUSE_AS_LOGICAL_NODE, "true");

    GaudiGraph g;

    TSize dataDims[4] = {1, 2, 2, 1};
    TSize idxDims[4] = {1, 1, 2, 1};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor inputData = pTensor(new Tensor(4U, dataDims, syn_type_bf16));
    inputData->setName("inputData");
    inputData->setDramOffset(0x1000);
    inputData->setMemoryDescriptor(persistentMemoryDesc);
    inputData->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    pTensor indicesTensor = pTensor(new Tensor(4U, idxDims, syn_type_int32));
    indicesTensor->setName("indicesTensor");
    indicesTensor->setDramOffset(0x3000);
    indicesTensor->setMemoryDescriptor(persistentMemoryDesc);
    indicesTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    pTensor updatesTensor = pTensor(new Tensor(4U, idxDims, syn_type_bf16));
    updatesTensor->setName("updatesTensor");
    updatesTensor->setDramOffset(0x5000);
    updatesTensor->setMemoryDescriptor(persistentMemoryDesc);
    updatesTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    pTensor outputTensor = pTensor(new Tensor(4U, dataDims, syn_type_bf16));
    outputTensor->setName("outputTensor");
    outputTensor->setDramOffset(0x1000);
    outputTensor->setMemoryDescriptor(persistentMemoryDesc);
    outputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    ns_ScatterKernel::Params params;
    params.axis = 1;

    pNode gatherNode = NodeFactory::createGenericTPCNode({inputData, indicesTensor, updatesTensor}, {outputTensor},
                                                         &params, "gather_bwd_bf16", "gatherNode");
    GraphEditor::addNode(g, gatherNode);

    bool retVal = g.compile();
    ASSERT_EQ(retVal, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 2) << "Got " << nodes.size() << ", Expected 2";

    pNode tpcSramMemcpyIn = *std::next(nodes.begin(), 0);
    pNode gather_node = *std::next(nodes.begin(), 1);

    ASSERT_EQ(tpcSramMemcpyIn->getNodeType(), Node::TYPE_DMA);
    ASSERT_EQ(tpcSramMemcpyIn->getNumInputs(),  1);
    ASSERT_EQ(tpcSramMemcpyIn->getNumOutputs(), 1);
    ASSERT_EQ(tpcSramMemcpyIn->getInput(0),  indicesTensor);
    ASSERT_EQ(tpcSramMemcpyIn->getOutput(0), gather_node->getInput(1));

    ASSERT_EQ(gather_node->getNodeTypeStr(), "gather_bwd_bf16");
    ASSERT_EQ(gather_node->getNumInputs(),  3);
    ASSERT_EQ(gather_node->getNumOutputs(), 1);
}

#include "gtest/gtest.h"
#include <graph_compiler/habana_nodes/node_factory.h>
#include "../graph_optimizer_test.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "perf_lib_layer_params.h"

class TpcScalarPipeTest : public GraphOptimizerTest
{

};

TEST_F(TpcScalarPipeTest, DISABLED_gather2d_indices_fit_in_sram)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    SizeArray dataDims = {2, 2};
    SizeArray idxDims = {1};
    SizeArray validCountDims = {1};

    pTensor inputData(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor inputIndices(new Tensor(1, idxDims.data(), syn_type_int32));
    pTensor validCount(new Tensor(1, validCountDims.data(), syn_type_int32));
    pTensor gatherOutput(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor output(new Tensor(2, dataDims.data(), syn_type_float));

    pNode gather2dNode = NodeFactory::createGenericTPCNode({inputData, inputIndices, validCount},{gatherOutput},
                                                           nullptr, "gather_with_valid_count_2d_f32", "gather2d");
    pNode addNode = NodeFactory::createGenericTPCNode({inputData, gatherOutput},{output},
                                                      nullptr, "add_fwd_f32", "add");

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    inputData->setDramOffset(0x1000);
    inputIndices->setDramOffset(0x3000);
    validCount->setDramOffset(0x6000);
    gatherOutput->setDramOffset(0x9000);
    output->setDramOffset(0x12000);

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    synMemoryDescriptor memDescPersist(true);
    inputData->setMemoryDescriptor(memDescPersist);
    inputData->setMemorySectionID(memSecId++);
    inputIndices->setMemoryDescriptor(memDescPersist);
    inputIndices->setMemorySectionID(memSecId++);
    validCount->setMemoryDescriptor(memDescPersist);
    validCount->setMemorySectionID(memSecId++);
    gatherOutput->setMemoryDescriptor(memDescPersist);
    gatherOutput->setMemorySectionID(memSecId++);
    output->setMemoryDescriptor(memDescPersist);
    output->setMemorySectionID(memSecId++);

    GaudiGraph g;
    ASSERT_TRUE(GraphEditor::addNode(g, gather2dNode));
    ASSERT_TRUE(GraphEditor::addNode(g, addNode));
    ASSERT_TRUE(g.compile());

    const NodeVector& execOrder = g.getExeSortedNodes();

    ASSERT_EQ(4, execOrder.size());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[0]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[1]->getNodeType());

    pNode gatherOutputNode = execOrder[2];
    ASSERT_EQ(gather2dNode->getNodeTypeStr(), gatherOutputNode->getNodeTypeStr());
    ASSERT_EQ(3, gatherOutputNode->getNumInputs());
    ASSERT_FALSE(gatherOutputNode->getInput(0)->inSram());
    ASSERT_TRUE(gatherOutputNode->getInput(1)->inSram());
    ASSERT_TRUE(gatherOutputNode->getInput(2)->inSram());

    ASSERT_EQ(gather2dNode->getInput(1), execOrder[0]->getInput(0));
    ASSERT_EQ(gatherOutputNode->getInput(1), execOrder[0]->getOutput(0));

    ASSERT_EQ(gather2dNode->getInput(2), execOrder[1]->getInput(0));
    ASSERT_EQ(gatherOutputNode->getInput(2), execOrder[1]->getOutput(0));

    pNode addOutputNode = execOrder[3];
    ASSERT_EQ(addNode->getNodeTypeStr(), addOutputNode->getNodeTypeStr());
    ASSERT_EQ(2, addOutputNode->getNumInputs());
    ASSERT_FALSE(addOutputNode->getInput(0)->inSram());
    ASSERT_FALSE(addOutputNode->getInput(1)->inSram());
    ASSERT_FALSE(addOutputNode->getOutput(0)->inSram());
}

TEST_F(TpcScalarPipeTest, DISABLED_gather2d_tensors_does_not_fit_in_sram)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    SizeArray dataDims = {2, 2};
    SizeArray idxDims = {1024 * 1024 * 6};
    SizeArray validCountDims = {1};

    pTensor inputData(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor inputIndices(new Tensor(1, idxDims.data(), syn_type_int32));
    pTensor validCount(new Tensor(1, validCountDims.data(), syn_type_int32));
    pTensor gatherOutput(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor output(new Tensor(2, dataDims.data(), syn_type_float));

    pNode gather2dNode = NodeFactory::createGenericTPCNode({inputData, inputIndices, validCount},{gatherOutput},
                                                           nullptr, "gather_with_valid_count_2d_f32", "gather2d");
    pNode addNode = NodeFactory::createGenericTPCNode({inputData, gatherOutput},{output},
                                                      nullptr, "add_fwd_f32", "add");

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    inputData->setDramOffset(0x1000);
    inputIndices->setDramOffset(0x3000);
    validCount->setDramOffset(0x6000);
    gatherOutput->setDramOffset(0x9000);
    output->setDramOffset(0x12000);

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    synMemoryDescriptor memDescPersist(true);
    inputData->setMemoryDescriptor(memDescPersist);
    inputData->setMemorySectionID(memSecId++);
    inputIndices->setMemoryDescriptor(memDescPersist);
    inputIndices->setMemorySectionID(memSecId++);
    validCount->setMemoryDescriptor(memDescPersist);
    validCount->setMemorySectionID(memSecId++);
    gatherOutput->setMemoryDescriptor(memDescPersist);
    gatherOutput->setMemorySectionID(memSecId++);
    output->setMemoryDescriptor(memDescPersist);
    output->setMemorySectionID(memSecId++);

    GaudiGraph g;
    ASSERT_TRUE(GraphEditor::addNode(g, gather2dNode));
    ASSERT_TRUE(GraphEditor::addNode(g, addNode));
    ASSERT_TRUE(g.compile());

    const NodeVector& execOrder = g.getExeSortedNodes();

    ASSERT_EQ(2, execOrder.size());

    pNode gatherOutputNode = execOrder[0];
    ASSERT_EQ(gather2dNode->getNodeTypeStr(), gatherOutputNode->getNodeTypeStr());
    ASSERT_EQ(4, gatherOutputNode->getNumInputs());
    ASSERT_FALSE(gatherOutputNode->getInput(0)->inSram());
    ASSERT_FALSE(gatherOutputNode->getInput(1)->inSram());
    ASSERT_FALSE(gatherOutputNode->getInput(2)->inSram());
    ASSERT_TRUE(gatherOutputNode->getInput(3)->inSram());
    ASSERT_TRUE(gatherOutputNode->getInput(3)->isAuxTensor());

    pNode addOutputNode = execOrder[1];
    ASSERT_EQ(addNode->getNodeTypeStr(), addOutputNode->getNodeTypeStr());
    ASSERT_EQ(2, addOutputNode->getNumInputs());
    ASSERT_FALSE(addOutputNode->getInput(0)->inSram());
    ASSERT_FALSE(addOutputNode->getInput(1)->inSram());
    ASSERT_FALSE(addOutputNode->getOutput(0)->inSram());
}

TEST_F(TpcScalarPipeTest, optimizer_sparse_fit_in_sram)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    ns_OptimizerSparseSGD::Params params;
    params.mom = 1;
    params.nesterov = false;

    SizeArray dataDims = {2, 2};
    SizeArray idxDims = {1};
    SizeArray validCountDims = {1};

    pTensor gradientIn(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor weightsIn(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor momentsIn(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor indicesIn(new Tensor(1, idxDims.data(), syn_type_int32));
    pTensor learningRateIn(new Tensor(1, idxDims.data(), syn_type_float));
    pTensor validCountIn(new Tensor(1, validCountDims.data(), syn_type_int32));

    pTensor weightsOut(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor momentsOut(new Tensor(2, dataDims.data(), syn_type_float));

    pNode sparseNode = NodeFactory::createGenericTPCNode({gradientIn, weightsIn, momentsIn, indicesIn,
                                                            learningRateIn, validCountIn}, {weightsOut, momentsOut},
                                                           &params,
                                                           "optimizer_sparse_sgd_with_valid_count_2d_f32",
                                                           "OptimizerSpars");

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    gradientIn->setDramOffset(0x1000);
    weightsIn->setDramOffset(0x2000);
    momentsIn->setDramOffset(0x3000);
    indicesIn->setDramOffset(0x4000);
    learningRateIn->setDramOffset(0x5000);
    validCountIn->setDramOffset(0x6000);

    weightsOut->setDramOffset(0x7000);
    momentsOut->setDramOffset(0x8000);

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    synMemoryDescriptor memDescPersist(true);
    gradientIn->setMemoryDescriptor(memDescPersist);
    gradientIn->setMemorySectionID(memSecId++);
    weightsIn->setMemoryDescriptor(memDescPersist);
    weightsIn->setMemorySectionID(memSecId++);
    momentsIn->setMemoryDescriptor(memDescPersist);
    momentsIn->setMemorySectionID(memSecId++);
    indicesIn->setMemoryDescriptor(memDescPersist);
    indicesIn->setMemorySectionID(memSecId++);
    learningRateIn->setMemoryDescriptor(memDescPersist);
    learningRateIn->setMemorySectionID(memSecId++);
    validCountIn->setMemoryDescriptor(memDescPersist);
    validCountIn->setMemorySectionID(memSecId++);

    weightsOut->setMemoryDescriptor(memDescPersist);
    weightsOut->setMemorySectionID(memSecId++);
    momentsOut->setMemoryDescriptor(memDescPersist);
    momentsOut->setMemorySectionID(memSecId++);

    GaudiGraph g;
    ASSERT_TRUE(GraphEditor::addNode(g, sparseNode));
    ASSERT_TRUE(g.compile());

    const NodeVector& execOrder = g.getExeSortedNodes();

    ASSERT_EQ(8, execOrder.size());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[0]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[1]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[2]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[3]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[4]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[6]->getNodeType());
    ASSERT_EQ(Node::eNodeType::TYPE_DMA, execOrder[7]->getNodeType());

    pNode sparseOutputNode = execOrder[5];
    ASSERT_EQ(sparseNode->getNodeTypeStr(), sparseOutputNode->getNodeTypeStr());
    ASSERT_EQ(6, sparseOutputNode->getNumInputs());
    ASSERT_FALSE(sparseOutputNode->getInput(0)->inSram());
    ASSERT_FALSE(sparseOutputNode->getInput(1)->inSram());
    ASSERT_FALSE(sparseOutputNode->getInput(2)->inSram());
    ASSERT_TRUE(sparseOutputNode->getInput(3)->inSram());
    ASSERT_TRUE(sparseOutputNode->getInput(4)->inSram());
    ASSERT_TRUE(sparseOutputNode->getInput(5)->inSram());

    ASSERT_FALSE(sparseOutputNode->getOutput(0)->inSram());
    ASSERT_FALSE(sparseOutputNode->getOutput(1)->inSram());
}

TEST_F(TpcScalarPipeTest, schedule_scalar_pipe_last)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    setGlobalConfForTest(GCFG_RUN_TPC_FUSER, "false");
    GaudiGraph g;
    unsigned dims = 2;
    TSize dataDims[2] = {2, 2};
    TSize idxDims[1] = {2};
    TSize validCountDims[1] = {2};
    synMemoryDescriptor memDescPersist(true);
    pTensor inputData = pTensor(new Tensor(2, dataDims, syn_type_float));
    pTensor inputIndices = pTensor(new Tensor(1, idxDims, syn_type_int32));
    pTensor validCount = pTensor(new Tensor(1, validCountDims, syn_type_int32));
    pTensor output = pTensor(new Tensor(2, dataDims, syn_type_float));

    // Set tensors as persistent
    inputData->setMemoryDescriptor(memDescPersist);
    inputData->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inputIndices->setMemoryDescriptor(memDescPersist);
    inputIndices->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    validCount->setMemoryDescriptor(memDescPersist);
    validCount->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    output->setMemoryDescriptor(memDescPersist);
    output->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);

    pNode gatherNode = NodeFactory::createNode({inputData, inputIndices, validCount},{output},
                                               nullptr, "gather_with_valid_count_2d_f32", "gatherNode");
    TensorVector tensors;
    for (int i = 0; i < 3; i++)
    {
        // Set tensors as persistent
        pTensor tens = pTensor(new Tensor(dims, dataDims, syn_type_float));
        tens->setMemoryDescriptor(memDescPersist);
        tens->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i + 4);

        tensors.push_back(tens);
    }

    pNode relu = NodeFactory::createGenericTPCNode({tensors[0]}, {tensors[1]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[1]}, {tensors[2]}, nullptr, "relu_fwd_f32", "relu2");

    ASSERT_TRUE(GraphEditor::addNode(g, gatherNode));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(g.compile());

    const NodeVector& execOrder = g.getExeSortedNodes();

    //verify execution order, gather node should be last
    ASSERT_EQ(5, execOrder.size());
    EXPECT_EQ(relu->getNodeTypeStr(), execOrder[0]->getNodeTypeStr());
    EXPECT_EQ(relu->getNodeTypeStr(), execOrder[1]->getNodeTypeStr());
    EXPECT_EQ(Node::eNodeType::TYPE_DMA, execOrder[2]->getNodeType());
    EXPECT_EQ(Node::eNodeType::TYPE_DMA, execOrder[3]->getNodeType());
    EXPECT_EQ(gatherNode->getNodeTypeStr(), execOrder[4]->getNodeTypeStr());

    pNode gatherExecNode = execOrder[4];
    ASSERT_EQ(3, gatherExecNode->getNumInputs());
    ASSERT_FALSE(gatherExecNode->getInput(0)->inSram());
    ASSERT_TRUE(gatherExecNode->getInput(1)->inSram());
    ASSERT_TRUE(gatherExecNode->getInput(2)->inSram());
}
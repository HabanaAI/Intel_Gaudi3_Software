#include "gaudi_graph.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"

#include "graph_optimizer_test.h"
#include "gtest/gtest.h"

class LogicalOperationsTest: public GraphOptimizerTest {};

TEST_F(LogicalOperationsTest, two_pass_invocation)
{
    GaudiGraph graph;

    SizeArray sizes = {25, 1, 1, 1, 1};
    TensorPtr t1 = std::make_shared<Tensor>(2U, sizes.data(), syn_type_bf16);
    TensorPtr t2 = std::make_shared<Tensor>(2U, sizes.data(), syn_type_bf16);
    sizes[1] = 2;
    TensorPtr t3 = std::make_shared<Tensor>(2U, sizes.data(), syn_type_bf16);
    unsigned concatDim = 1;
    NodePtr   concat = NodeFactory::createNode({t1, t2}, {t3}, &concatDim, sizeof(concatDim), NodeFactory::concatenateNodeTypeName, "concat");

    ASSERT_TRUE(GraphEditor::addNode(graph, concat)) << "Failed to add concat node to graph";

    ASSERT_TRUE(handleLogicalOps(graph)) << "Failed to run logical operations pass";

    sizes[0] = sizes[1] = 5;
    TensorPtr t1_reshaped = std::make_shared<Tensor>(2U, sizes.data(), syn_type_bf16);
    NodePtr   reshape= NodeFactory::createNode({t1_reshaped}, {t1}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "reshape");

    ASSERT_TRUE(GraphEditor::addNode(graph, reshape)) << "Failed to add reshape node to graph";

    ASSERT_TRUE(handleLogicalOps(graph)) << "Failed to run logical operations pass for the second time";

    ASSERT_EQ(graph.getNodes().size(), 2) << "Expect only reshape and concat nodes";
    for (const auto& node : graph.getNodes())
    {
        ASSERT_TRUE(node->isLogicalOperation()) << "Expect logical operation only";
    }
}

TEST_F(LogicalOperationsTest, swap_traspose_direction_to_avoid_memcpy)
{
    // Graph: nop->transpose->reshape->nop

    // Default alias direction for logical transpose is OUTPUT_TO_INPUT.
    // In this case, we are insert memcpy between transpose and reshape
    // We can swap the alias direction and avoid adding memcpy node. (transpose_in will be aliased to transpose_out).
    const TSize    FCD = 1;
    const TSize    WIDTH = 2;
    const TSize    HEIGHT = 3;
    const TSize    BATCH = 1;
    const unsigned dim_num = 4;

    const TSize input_dimensions[] = {FCD, WIDTH, HEIGHT, BATCH};
    const TSize transpose_output_dimensions[] = {FCD, HEIGHT, WIDTH, BATCH};
    const TSize reshape_output_dimensions[] = {FCD * HEIGHT, WIDTH * BATCH};

    TransposePermutationArray permutation({TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch});

    TensorPtr memcpy_out    = TensorPtr(new Tensor(dim_num, input_dimensions,  syn_type_float));
    TensorPtr transpose_out = TensorPtr(new Tensor(dim_num, transpose_output_dimensions,  syn_type_float));
    TensorPtr reshape_out   = TensorPtr(new Tensor(dim_num / 2, reshape_output_dimensions,  syn_type_float));

    NodePtr producer  = NodeFactory::createNode({}, {memcpy_out}, nullptr, NOP_KERNEL_NAME, "producer");
    NodePtr transpose = NodeFactory::createNode({memcpy_out}, {transpose_out}, &permutation, "transpose_logic", "transpose");
    NodePtr reshape   = NodeFactory::createNode({transpose_out}, {reshape_out}, nullptr, "reshape", "reshape");
    NodePtr consumer  = NodeFactory::createNode({reshape_out}, {}, nullptr, NOP_KERNEL_NAME, "consumer");

    GaudiGraph g;

    GraphEditor::addNode(g, producer);
    GraphEditor::addNode(g, transpose);
    GraphEditor::addNode(g, reshape);
    GraphEditor::addNode(g, consumer);

    ASSERT_TRUE(handleLogicalOps(g)) << "Failed to run handleLogicalOps pass";

    // Make sure there is no additional memcpy nodes are added in handleLogicalOps
    const NodeVector& nodes         = g.getExeSortedNodes();
    int memcpyCounter = 0;
    for (NodePtr node : nodes)
    {
        LOG_DEBUG(GO_TEST, "Node {} in graph", node->getNodeTypeStr());
        memcpyCounter += (node->getNodeType() == Node::TYPE_MEMCOPY);
    }
    ASSERT_EQ(memcpyCounter, 0) << "Additional memcpy added";
}

TEST_F(LogicalOperationsTest, shaped_strided_slice)
{
    GaudiGraph g;
    TSize      inputSizes[] = {1, 1};
    TensorPtr  input        = TensorPtr(new Tensor(2, inputSizes, syn_type_int32, nullptr, nullptr, false, true));

    synMemoryDescriptor input_memDesc(true);
    input->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    input->setMemoryDescriptor(input_memDesc);

    TSize sliceOutputSize[] = {1, 1};
    // not persistent
    TensorPtr sliceOutput = TensorPtr(new Tensor(2, sliceOutputSize, syn_type_int32, nullptr, nullptr, false, true));

    synSliceParams sliceParams     = {{0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 1, 0, 0, 0}, {1, 1, 0, 0, 0}};
    GraphEditor::addNode(g, NodeFactory::createNode({input}, {sliceOutput}, &sliceParams, "slice", "slice"));

    TSize reshapeSizes[] = {1};
    TensorPtr reshapeShapeTensor = TensorPtr(new Tensor (1, reshapeSizes, syn_type_int32, nullptr, nullptr, false, true, INVALID_BATCH_POS, nullptr, SHAPE_TENSOR));

    // not persistent
    TSize reshapeOutputSizes[] = {1};
    TensorPtr reshapeOutput = TensorPtr(new Tensor (1, reshapeOutputSizes, syn_type_int32, nullptr, nullptr, false, true));

    GraphEditor::addNode(
        g,
        NodeFactory::createNode({sliceOutput, reshapeShapeTensor}, {reshapeOutput}, nullptr, 0, "reshape", "reshape"));

    // persistent
    TSize gatherInputSizes[] = {4, 17};
    TensorPtr  gatherInput = TensorPtr(new Tensor(2, gatherInputSizes, syn_type_single, nullptr, nullptr, false, true));

    synMemoryDescriptor gatherInput_memDesc(true);
    gatherInput->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    gatherInput->setMemoryDescriptor(gatherInput_memDesc);

    // persistent
    TSize     gatherOutputSizes[] = {4, 1};
    TensorPtr gatherOutputTensor =
        TensorPtr(new Tensor(2, gatherOutputSizes, syn_type_single, nullptr, nullptr, true, false));

    synMemoryDescriptor gatherOutputTensor_memDesc(true);
    gatherOutputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    gatherOutputTensor->setMemoryDescriptor(gatherOutputTensor_memDesc);

    uint32_t gatherParams = 1;
    GraphEditor::addNode(g,
                         NodeFactory::createNode({gatherInput, reshapeOutput},
                                                 {gatherOutputTensor},
                                                 (void*)&gatherParams,
                                                 4,
                                                 "gather_fwd_f32",
                                                 "gather_fwd_f32"));

    g.compile();
}

TEST_F(LogicalOperationsTest, concat_transpoe_without_memcpy)
{
    // Disable TPC shape manipulations to avoid mem-copies due to reshapes on strided tensors.
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");

    GaudiGraph g;
    SizeArray concatInSize = {10, 15, 20, 1, 1};
    SizeArray concatOutSize = {10, 15, 20, 2, 1};
    SizeArray transposeSize = {10, 20, 15, 2, 1};
    SizeArray reshapeSize = {200, 30, 1, 1, 1};

    TensorPtr reluIn1 = TensorPtr(new Tensor(4, concatInSize.data(), syn_type_float));
    TensorPtr reluIn2 = TensorPtr(new Tensor(4, concatInSize.data(), syn_type_float));
    TensorPtr reluOut1 = TensorPtr(new Tensor(4, concatInSize.data(), syn_type_float));
    TensorPtr reluOut2 = TensorPtr(new Tensor(4, concatInSize.data(), syn_type_float));
    TensorPtr concatOut = TensorPtr(new Tensor(4, concatOutSize.data(), syn_type_float));
    TensorPtr transposeOut = TensorPtr(new Tensor(4, transposeSize.data(), syn_type_float));
    TensorPtr reshapeOut = TensorPtr(new Tensor(2, reshapeSize.data(), syn_type_float));
    TensorPtr reluFinalOut = TensorPtr(new Tensor(2, reshapeSize.data(), syn_type_float));

    synMemoryDescriptor persistentDesc(true);
    reluIn1->setMemoryDescriptor(persistentDesc);
    reluIn2->setMemoryDescriptor(persistentDesc);
    reluFinalOut->setMemoryDescriptor(persistentDesc);

    GraphEditor::addNode(g, NodeFactory::createNode({reluIn1}, {reluOut1}, nullptr, 0, "relu_fwd_f32", "relu1"));
    GraphEditor::addNode(g, NodeFactory::createNode({reluIn2}, {reluOut2}, nullptr, 0, "relu_fwd_f32", "relu2"));
    unsigned concatDim = 3;
    GraphEditor::addNode(g,
                         NodeFactory::createNode({reluOut1, reluOut2},
                                                 {concatOut},
                                                 &concatDim,
                                                 sizeof(concatDim),
                                                 NodeFactory::concatenateNodeTypeName,
                                                 "concat"));
    synTransposeParams transposeParam;
    transposeParam.permutation[0] = TPD_Channel;
    transposeParam.permutation[1] = TPD_Height;
    transposeParam.permutation[2] = TPD_Width;
    transposeParam.permutation[3] = TPD_4Dim_Batch;
    transposeParam.tensorDim = 4;
    GraphEditor::addNode(g,
                         NodeFactory::createNode({concatOut},
                                                 {transposeOut},
                                                 &transposeParam,
                                                 sizeof(transposeParam),
                                                 NodeFactory::transposeNodeTypeName,
                                                 "transpose"));
    GraphEditor::addNode(
        g,
        NodeFactory::createNode({transposeOut}, {reshapeOut}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "reshape"));
    GraphEditor::addNode(
        g,
        NodeFactory::createNode({reshapeOut}, {reluFinalOut}, nullptr, 0, "relu_fwd_f32", "reluFinal"));

    g.compile();

    uint32_t physicalNodes = 0;
    for (auto node : g.getNodes())
    {
        if (node->isLogicalOperation()) continue;
        ++physicalNodes;
    }

    // Expect 3 physical nodes - relu nodes
    ASSERT_EQ(physicalNodes, 3);
}

class ConcatToBroadcastTest :
    public LogicalOperationsTest,
    // params are concat dim and vector of pairs of sequences length and input index
    public testing::WithParamInterface<std::tuple<unsigned, std::vector<std::pair<unsigned, unsigned>>>>
{
protected:
    bool expectToConcatNode(const std::vector<std::pair<unsigned, unsigned>>& concatInputs)
    {
        return (concatInputs.size() != 1) || (concatInputs.at(0).first <= 2);
    }
    void validateConcatInputs(const TensorVector& newInputs,
                              const TensorPtr& input1,
                              const TensorPtr& input2,
                              const std::vector<std::pair<unsigned, unsigned>>& originalInputs)
    {
        TensorVector expectedInputs;
        for (auto p : originalInputs)
        {
            if (p.first > 2)
            {
                expectedInputs.push_back(nullptr);
            }
            else if (p.first == 2)
            {
                expectedInputs.push_back((p.second == 0) ? input1 : input2);
                expectedInputs.push_back((p.second == 0) ? input1 : input2);
            }
            else // p.first == 1
            {
                expectedInputs.push_back((p.second == 0) ? input1 : input2);
            }
        }
        ASSERT_EQ(expectedInputs.size(), newInputs.size()) << "there is mismatch in expected inputs calculation";
        for (unsigned i = 0; i < newInputs.size(); ++i)
        {
            if (expectedInputs.at(i) == input1)
            {
                ASSERT_EQ(*newInputs.at(i), *input1) << "mismatch in index: " << i;
            }
            else if (expectedInputs.at(i) == input2)
            {
                ASSERT_EQ(*newInputs.at(i), *input2) << "mismatch in index: " << i;
            }
            else
            {
                ASSERT_NE(*newInputs.at(i), *input1) << "mismatch in index: " << i;
                ASSERT_NE(*newInputs.at(i), *input2) << "mismatch in index: " << i;
            }
        }
    }
};

TEST_P(ConcatToBroadcastTest, concat_to_broadcast)
{
    GaudiGraph g;
    const auto& testParams = GetParam();
    unsigned concatDim = std::get<0>(testParams);
    std::vector<std::pair<unsigned, unsigned>> concatInputs = std::get<1>(testParams);
    TSize inSizes[][4] = {{4, 2, 3 ,5}, {4, 2, 3 ,5}};
    inSizes[0][concatDim] += 1;
    inSizes[1][concatDim] -= 1;
    unsigned concatOutputSize = std::accumulate(concatInputs.begin(),
                                                concatInputs.end(),
                                                (unsigned)0,
                                                [&](unsigned a, std::pair<unsigned, unsigned> b)
                                                {
                                                    return a + b.first * inSizes[b.second][concatDim];
                                                });
    TSize outSizes[] = {4, 2, 3 ,5};
    outSizes[concatDim] = concatOutputSize;
    synMemoryDescriptor memDescPersist(true);

    TensorPtr input1 = TensorPtr(new Tensor(4U, inSizes[0], syn_type_float));
    input1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    input1->setMemoryDescriptor(memDescPersist);
    TensorPtr input2 = TensorPtr(new Tensor(4U, inSizes[1], syn_type_float));
    input2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    input2->setMemoryDescriptor(memDescPersist);

    TensorPtr output = TensorPtr(new Tensor(4U, outSizes, syn_type_float));
    output->setMemoryDescriptor(memDescPersist);
    output->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    TensorVector inputsArray = {input1, input2};

    TensorVector inputs;
    unsigned     expectedBroadcasts = 0;
    for (auto size : concatInputs)
    {
        if (size.first > 2)
        {
            ++expectedBroadcasts;
        }
        for (auto i = 0; i < size.first; ++i)
        {
            inputs.push_back(inputsArray[size.second]);
        }
    }
    synConcatenateParams params = {concatDim};
    NodePtr concat = NodeFactory::createNode(inputs, {output}, &params,
                                             NodeFactory::concatenateNodeTypeName, "concat");

    GraphEditor::addNode(g, concat);
    ASSERT_TRUE(g.compile()) << "Compilation failed";

    const NodeVector& nodes         = g.getExeSortedNodes();
    uint64_t        numBroadcasts = 0;
    bool concatExists = false;

    for (auto node : nodes)
    {
        if (node->isDma())
        {
            std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(node);
            if (dmaNode->isBroadcast())
            {
                ++numBroadcasts;
            }
        }
        if (std::dynamic_pointer_cast<ConcatenateNode>(node))
        {
            concatExists = true;
            validateConcatInputs(node->getInputs(), input1, input2, concatInputs);
        }
    }
    ASSERT_LE(numBroadcasts, expectedBroadcasts * 2) << "Unexpected number of memcpy nodes in the graph";
    ASSERT_EQ(concatExists, expectToConcatNode(concatInputs));
}

INSTANTIATE_TEST_SUITE_P(
        ConcatToBroadcast,
        ConcatToBroadcastTest,
        ::testing::Combine(
            ::testing::Range<unsigned>(0, 4),
            ::testing::Values(
                std::vector<std::pair<unsigned, unsigned>>({{3, 0}, {2, 1}, {6, 0}, {1, 1}, {1, 0}, {6, 1}, {6, 0}}),
                std::vector<std::pair<unsigned, unsigned>>({{6, 0}, {6, 1}}),
                std::vector<std::pair<unsigned, unsigned>>({{6, 0}}),
                std::vector<std::pair<unsigned, unsigned>>({{2, 0}}),
                std::vector<std::pair<unsigned, unsigned>>({{200000, 0}}),
                std::vector<std::pair<unsigned, unsigned>>({{200000, 0}, {1, 1}}),
                std::vector<std::pair<unsigned, unsigned>>({{200000, 0}, {200000, 1}}),
                std::vector<std::pair<unsigned, unsigned>>({{1, 0}, {1, 1}, {1, 0}}),
                std::vector<std::pair<unsigned, unsigned>>({{100, 0}, {1, 1}, {100, 0}})
            )
        )
    );
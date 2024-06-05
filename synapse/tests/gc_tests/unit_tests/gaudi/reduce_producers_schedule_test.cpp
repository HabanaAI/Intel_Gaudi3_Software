#include <cstring>
#include <graph_compiler/habana_nodes/node_factory.h>
#include <string>
#include "../graph_optimizer_test.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "graph_editor.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"

class ReduceProducersScheduleTest : public GraphOptimizerTest
{
protected:
    static constexpr unsigned gconf_max_str = 128;
    char                      m_enable_max_path_scheduler[gconf_max_str];
};

TEST_F(ReduceProducersScheduleTest, memset_reduction_with_control_edges)
{
    GaudiGraph g;

    SizeArray memsetSize = {1};
    pTensor t1(new Tensor(1, memsetSize.data(), syn_type_float));
    pTensor t2(new Tensor(1, memsetSize.data(), syn_type_float));
    pTensor t3(new Tensor(1, memsetSize.data(), syn_type_float));
    pTensor t4(new Tensor(1, memsetSize.data(), syn_type_float));
    pTensor t5(new Tensor(1, memsetSize.data(), syn_type_float));
    pTensor   tBlocking(new Tensor(1, memsetSize.data(), syn_type_float));
    tBlocking->setMemoryDescriptor(synMemoryDescriptor(true));
    tBlocking->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tBlocking->setName("tBlocking");
    pTensor tBlocked(new Tensor(1, memsetSize.data(), syn_type_float));
    tBlocked->setMemoryDescriptor(synMemoryDescriptor(true));
    tBlocked->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tBlocked->setName("tBlocked");

    NodePtr blockingNode = NodeFactory::createNode({tBlocking}, {t1}, nullptr, NOP_KERNEL_NAME, "blocking-node");
    NodePtr tpcNode      = NodeFactory::createNode({t1}, {t2}, nullptr, NOP_KERNEL_NAME, "tpc-node");
    NodePtr blockedNode  = NodeFactory::createNode({t2}, {tBlocked}, nullptr, NOP_KERNEL_NAME, "blocked-node");
    NodePtr memsetNode = NodeFactory::createNode({}, {t3}, nullptr, 0, NodeFactory::memsetNodeTypeName, "memset");
    NodePtr reductionNode = NodeFactory::createNode({t3, t2}, {t4}, nullptr, 0, NodeFactory::reductionNodeTypeName, "reduction");
    NodePtr memcpyNode = NodeFactory::createNode({t4}, {t5}, nullptr, 0, NodeFactory::memcpyNodeTypeName, "memcpy");

    GraphEditor::addNode(g, blockingNode);
    GraphEditor::addNode(g, tpcNode);
    GraphEditor::addNode(g, blockedNode);

    g.addControlDependency({blockingNode}, {tpcNode});
    g.addControlDependency({tpcNode}, {blockedNode});

    registerMemoryCoherence(g);
    relaxCtrlDeps(g);
    GraphEditor::replaceNodes(g, {tpcNode}, {tpcNode, reductionNode, memsetNode, memcpyNode});
    handleCtrlEdgesForLogicalNodes(g);

    NodeVector execOrder = g.getExeSortedNodes();

    //verify exec order:
    GlobalConfManager::instance().getGlobalConf("ENABLE_MAX_PATH_SCHEDULE",
                                                m_enable_max_path_scheduler,
                                                sizeof(m_enable_max_path_scheduler));

    NodeVector expectedOrder;
    if (m_enable_max_path_scheduler == std::string("1"))
    {
        expectedOrder = {blockingNode, memsetNode, tpcNode, reductionNode, blockedNode, memcpyNode};
    }
    else
    {
        expectedOrder = {blockingNode, memsetNode, tpcNode, blockedNode, reductionNode, memcpyNode};
    }

    ASSERT_EQ(6, execOrder.size());
    ASSERT_TRUE(std::equal(execOrder.begin(), execOrder.end(), expectedOrder.begin()));

    //verify control edges correctness:
    ASSERT_EQ(blockingNode->getControlInputs().size(), 0);
    ASSERT_EQ(tpcNode->getControlInputs().size(), 1);  // blocking->tpc, memset->tpc
    ASSERT_EQ(memsetNode->getControlInputs().size(), 0);
    ASSERT_EQ(reductionNode->getControlInputs().size(), 0);
    ASSERT_EQ(memcpyNode->getControlInputs().size(), 0);   // blocking->memcpy
    ASSERT_EQ(blockedNode->getControlInputs().size(), 1);  // tpc->blocked, memcpy->blocked

    ASSERT_EQ(blockingNode->getControlOutputs().size(), 1);
    ASSERT_EQ(g.getNumberOfTensorConsumers(blockingNode->getControlOutputs()[0]), 1);  // blocking->blocked
    ASSERT_EQ(tpcNode->getControlOutputs().size(), 0);                                // tpc->blocked
    ASSERT_EQ(memsetNode->getControlOutputs().size(), 1);                             // memset->tpc
    ASSERT_EQ(reductionNode->getControlOutputs().size(), 0);
    ASSERT_EQ(memcpyNode->getControlOutputs().size(), 0);  // memcpy->blocked
    ASSERT_EQ(blockedNode->getControlOutputs().size(), 0);
}


TEST_F(ReduceProducersScheduleTest, reduction_memset_logical_nodes)
{
    ScopedConfigurationChange config("HANDLE_MEMORY_COHERENCE", "false");

    SizeArray memsetSize = {10, 1};
    SizeArray concatSize = {10, 2};
    pTensor   t1(new Tensor(2, memsetSize.data(), syn_type_float));
    pTensor   t2(new Tensor(2, memsetSize.data(), syn_type_float));
    pTensor   t3(new Tensor(2, concatSize.data(), syn_type_float));
    pTensor   t4(new Tensor(2, concatSize.data(), syn_type_float));
    pTensor   t5(new Tensor(2, concatSize.data(), syn_type_float));
    pNode                nonMemsetNode1 = NodeFactory::createNode({}, {t1}, nullptr, NOP_KERNEL_NAME, "non-memset1");
    pNode                nonMemsetNode2 = NodeFactory::createNode({}, {t2}, nullptr, NOP_KERNEL_NAME, "non-memset2");
    synConcatenateParams concatParams;
    concatParams.axis = 1;
    pNode nonMemsetNode3 =
        NodeFactory::createNode({t1, t2}, {t3}, &concatParams, NodeFactory::concatenateNodeTypeName, "concat");
    pNode memsetNode = NodeFactory::createNode({}, {t4}, nullptr, NodeFactory::memsetNodeTypeName, "memset");

    pNode reductionNode =
            NodeFactory::createNode({t4, t3}, {t5}, nullptr, NodeFactory::reductionNodeTypeName, "reduction");

    GaudiGraph g;
    GraphEditor::addNode(g, nonMemsetNode1);
    GraphEditor::addNode(g, memsetNode);
    GraphEditor::addNode(g, nonMemsetNode2);
    GraphEditor::addNode(g, nonMemsetNode3);
    GraphEditor::addNode(g, reductionNode);

    handleCtrlEdgesForLogicalNodes(g);

    const NodeVector& execOrder = g.getExeSortedNodes();
    ASSERT_EQ(5, execOrder.size());
    ASSERT_EQ(memsetNode, execOrder.front());
    ASSERT_EQ(reductionNode, execOrder.back());
}


TEST_F(ReduceProducersScheduleTest, memset_should_run_as_first_reduce_producer)
{
    ScopedConfigurationChange config("HANDLE_MEMORY_COHERENCE", "false");

    auto    t1             = std::make_shared<Tensor>(syn_type_bf16);
    auto    t2             = std::make_shared<Tensor>(syn_type_bf16);
    auto    t3             = std::make_shared<Tensor>(syn_type_bf16);
    auto    t4             = std::make_shared<Tensor>(syn_type_bf16);
    pNode   nonMemsetNode1 = NodeFactory::createNode({}, {t1}, nullptr, NOP_KERNEL_NAME, "non-memset1");
    pNode   nonMemsetNode2 = NodeFactory::createNode({}, {t2}, nullptr, NOP_KERNEL_NAME, "non-memset2");
    pNode memsetNode = NodeFactory::createNode({}, {t3}, nullptr, NodeFactory::memsetNodeTypeName, "memset");

    pNode reductionNode =
        NodeFactory::createNode({t1, t2, t3}, {t4}, nullptr, NodeFactory::reductionNodeTypeName, "reduction");

    GaudiGraph g;
    GraphEditor::addNode(g, nonMemsetNode1);
    GraphEditor::addNode(g, memsetNode);
    GraphEditor::addNode(g, nonMemsetNode2);
    GraphEditor::addNode(g, reductionNode);

    handleCtrlEdgesForLogicalNodes(g);

    const NodeVector& execOrder = g.getExeSortedNodes();
    ASSERT_EQ(4, execOrder.size());
    ASSERT_EQ(memsetNode, execOrder.front());
    ASSERT_EQ(reductionNode, execOrder.back());
}

// Test that if other nodes can be executed before memset, they are scheduled thus.
TEST_F(ReduceProducersScheduleTest, memset_should_run_as_late_as_possible)
{
    ScopedConfigurationChange config("HANDLE_MEMORY_COHERENCE", "false");

    auto t1 = std::make_shared<Tensor>(syn_type_bf16);
    auto t2 = std::make_shared<Tensor>(syn_type_bf16);
    auto t3 = std::make_shared<Tensor>(syn_type_bf16);
    auto t4 = std::make_shared<Tensor>(syn_type_bf16);
    auto t5 = std::make_shared<Tensor>(syn_type_bf16);
    auto t6 = std::make_shared<Tensor>(syn_type_bf16);
    auto t7 = std::make_shared<Tensor>(syn_type_bf16);

    pNode   preReduceProducer1 = NodeFactory::createNode({}, {t6}, nullptr, NOP_KERNEL_NAME, "preReduceProducer1");
    pNode   preReduceProducer2 = NodeFactory::createNode({}, {t5, t7}, nullptr, NOP_KERNEL_NAME, "preReduceProducer2");
    pNode   nonMemsetNode1     = NodeFactory::createNode({t5}, {t1}, nullptr, NOP_KERNEL_NAME, "non-memset1");
    pNode   nonMemsetNode2     = NodeFactory::createNode({t6, t7}, {t2}, nullptr, NOP_KERNEL_NAME, "non-memset2");

    pNode memsetNode = NodeFactory::createNode({}, {t3}, nullptr, NodeFactory::memsetNodeTypeName, "memset");

    pNode reductionNode =
              NodeFactory::createNode({t1, t2, t3}, {t4}, nullptr, NodeFactory::reductionNodeTypeName, "reduction");

    GaudiGraph g;
    GraphEditor::addNode(g, memsetNode);
    GraphEditor::addNode(g, preReduceProducer1);
    GraphEditor::addNode(g, preReduceProducer2);
    GraphEditor::addNode(g, nonMemsetNode1);
    GraphEditor::addNode(g, nonMemsetNode2);
    GraphEditor::addNode(g, reductionNode);

    handleCtrlEdgesForLogicalNodes(g);

    NodeVector execOrder = g.getExeSortedNodes();
    ASSERT_EQ(6, execOrder.size());
    ASSERT_EQ(memsetNode, execOrder[2]);
    ASSERT_EQ(reductionNode, execOrder.back());
}

TEST_F(ReduceProducersScheduleTest, sram_reduce_annotation_should_be_set_for_all_inputs_except_first)
{
    // Force execution order using the following topology:
    // Nodes A, B, C produce Tensors T1, T2, T3 which get reduced in node R:
    //
    // A ------------------------------------> T1 --|
    //   \---> T4  ---> B -------------------> T2 - R --->T6
    //                    \---> T5 ---> C ---> T3 --|
    // Expect T2 and T3 to be annotated for reduction and T1 not
    TSize sizes[] = {1};
    pTensor T1(new Tensor(1, sizes, syn_type_float));
    T1->setSramOffset(0x1000);
    pTensor T2(new Tensor(1, sizes, syn_type_float));
    T2->setSramOffset(0x1000);
    pTensor T3(new Tensor(1, sizes, syn_type_float));
    T3->setSramOffset(0x1000);
    pTensor T4(new Tensor(1, sizes, syn_type_float));
    T4->setSramOffset(0x2000);
    pTensor T5(new Tensor(1, sizes, syn_type_float));
    T5->setSramOffset(0x3000);
    pTensor T6(new Tensor(1, sizes, syn_type_float));
    T6->setSramOffset(0x1000);

    pTensor T7(new Tensor(1, sizes, syn_type_float));
    T7->setMemoryDescriptor(synMemoryDescriptor(true));
    T7->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    pTensor T8(new Tensor(1, sizes, syn_type_float));
    T8->setMemoryDescriptor(synMemoryDescriptor(true));
    T8->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    synSplitParams splitParams = {1};

    pNode A = NodeFactory::createNode({}, {T1, T4}, nullptr, NOP_KERNEL_NAME, "A");
    pNode B = NodeFactory::createNode({T4}, {T2, T5}, nullptr, NOP_KERNEL_NAME, "B");
    pNode C = NodeFactory::createNode({T5}, {T3}, nullptr, NOP_KERNEL_NAME, "C");

    pNode R = NodeFactory::createNode({T3, T1, T2}, {T6}, nullptr, NodeFactory::reductionNodeTypeName, "R");
    pNode S = NodeFactory::createNode({T6}, {T7, T8}, &splitParams, "split", "split");

    GaudiGraph g;
    ASSERT_TRUE(GraphEditor::addNode(g, R));
    ASSERT_TRUE(GraphEditor::addNode(g, A));
    ASSERT_TRUE(GraphEditor::addNode(g, B));
    ASSERT_TRUE(GraphEditor::addNode(g, C));
    ASSERT_TRUE(GraphEditor::addNode(g, S));

    ASSERT_TRUE(g.compile());

    ASSERT_FALSE(T1->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
    ASSERT_TRUE(T2->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
    ASSERT_TRUE(T3->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
}

TEST_F(ReduceProducersScheduleTest, sram_reduce_annotation_should_be_set_for_all_inputs_except_memset)
{
    // Force execution order using the following topology:
    // Nodes A, B, C produce Tensors T1, T2, T3 which get reduced in node R:
    //
    // A ------------------------------------> T1 --|
    //   \---> T4  ---> B -------------------> T2 - R --->T6
    //                    \---> T5 ---> C ---> T3 --|
    //                                             /
    //                            Memset ---> T7 -/
    //
    // Expect T1, T2, T3 to be annotated, but not T7.
    GaudiGraph g;

    TSize sizes[] = {1};

    auto T1 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T1->setSramOffset(0x1000);
    auto T2 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T2->setSramOffset(0x1000);
    auto T3 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T3->setSramOffset(0x1000);
    auto T4 = std::make_shared<Tensor>(syn_type_bf16);
    T4->setSramOffset(0x2000);
    auto T5 = std::make_shared<Tensor>(syn_type_bf16);
    T5->setSramOffset(0x3000);
    auto T6 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T6->setSramOffset(0x1000);

    auto T7 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T7->setSramOffset(0x1000 + g.getHALReader()->getSRAMBaseAddr());

    auto T8 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T8->setMemoryDescriptor(synMemoryDescriptor(true));
    T8->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    auto T9 = std::make_shared<Tensor>(1, sizes, syn_type_float);
    T9->setMemoryDescriptor(synMemoryDescriptor(true));
    T9->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    synSplitParams splitParams = {1};

    pNode A = NodeFactory::createNode({}, {T1, T4}, nullptr, NOP_KERNEL_NAME, "A");
    pNode B = NodeFactory::createNode({T4}, {T2, T5}, nullptr, NOP_KERNEL_NAME, "B");
    pNode C = NodeFactory::createNode({T5}, {T3}, nullptr, NOP_KERNEL_NAME, "C");

    pNode memsetNode = NodeFactory::createNode({}, {T7}, nullptr, NodeFactory::memsetNodeTypeName, "memset");

    pNode R = NodeFactory::createNode({T3, T1, T2, T7}, {T6}, nullptr, NodeFactory::reductionNodeTypeName, "R");

    pNode S = NodeFactory::createNode({T6}, {T8, T9}, &splitParams, "split", "split");

    ASSERT_TRUE(GraphEditor::addNode(g, R));
    ASSERT_TRUE(GraphEditor::addNode(g, A));
    ASSERT_TRUE(GraphEditor::addNode(g, B));
    ASSERT_TRUE(GraphEditor::addNode(g, C));
    ASSERT_TRUE(GraphEditor::addNode(g, memsetNode));
    ASSERT_TRUE(GraphEditor::addNode(g, S));

    ASSERT_TRUE(g.compile());

    ASSERT_TRUE(T1->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
    ASSERT_TRUE(T2->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
    ASSERT_TRUE(T3->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
    ASSERT_FALSE(T7->getTensorAnnotation().tensorReductionInfo.isReductionEnabled);
}

#pragma once

#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "params_file_manager.h"
#include "passes/base_regs_cache_manager.h"

static const int REDUCED_CACHE_SIZE_4 = 4;  // 4 entries cache
static const int REDUCED_CACHE_SIZE_5 = 5;  // 5 entries cache
static const int REDUCED_CACHE_SIZE_7 = 7;  // 7 entries cache

template<class GraphType, unsigned int TCacheSize>
class BaseRegsCacheManager_SingleLogicalEngine : public BaseRegsCacheManager<1, TCacheSize>
{
public:
    BaseRegsCacheManager_SingleLogicalEngine(GraphType& g) : BaseRegsCacheManager<1, TCacheSize>(g) {}

protected:
    uint64_t getCacheIndex(const NodePtr& node) override { return 0; }
};

static const int NUM_LOGICAL_ENGINES = 3;
template<class GraphType, unsigned int TCacheSize>
class BaseRegsCacheManager_MultiLogicalEngines : public BaseRegsCacheManager<NUM_LOGICAL_ENGINES, TCacheSize>
{
public:
    BaseRegsCacheManager_MultiLogicalEngines(GraphType& g) : BaseRegsCacheManager<NUM_LOGICAL_ENGINES, TCacheSize>(g) {}

protected:
    uint64_t getCacheIndex(const NodePtr& node) override
    {
        if (node->getNodeName().find("tpc") != std::string::npos) return 0;
        if (node->getNodeName().find("dma") != std::string::npos) return 1;
        if (node->getNodeName().find("mme") != std::string::npos) return 2;
        assert(0 && "ensure you node name contains tpc, dma or mme");
        return 0;
    }
};

template<class GraphType>
class GaudiBaseRegsCacheTestCommon : public GraphOptimizerTest
{
public:
    void basic()  // TODO enable, dma1_node not suppopretd yet
    {
        const unsigned tensor_dim = 1;
        const TSize    size       = 1;
        GraphType      g;
        TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        NodePtr        n = NodeFactory::createNode({i}, {o}, nullptr, NodeFactory::memcpyNodeTypeName, "dma1_node");
        synMemoryDescriptor memDesc(true);  // persistent

        // set some boguse addresses to the tensors and allocate host memory so we won't assert
        i->setDramOffset(getVirtualAddressForMemoryID(10, 0x1000));
        o->setDramOffset(getVirtualAddressForMemoryID(20, 0x2000));
        i->setMemorySectionID(10);
        o->setMemorySectionID(20);
        i->setMemoryDescriptor(memDesc);
        o->setMemoryDescriptor(memDesc);

        GraphEditor::addNode(g, n);
        ASSERT_TRUE(g.compile()) << "failed to compile graph";

        // the original semantic node was replaced during compilation by concrete DMA node, so get the node from the
        // graph
        NodePtr compiledNode = g.getExeSortedNodes().front();
        ASSERT_EQ(compiledNode->getNodeAnnotation().baseRegsCacheUpdate.size(), 2);
        ASSERT_EQ(compiledNode->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 0);
        ASSERT_EQ(compiledNode->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 10);
        ASSERT_EQ(compiledNode->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 1);
        ASSERT_EQ(compiledNode->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 20);
    }

    void single_logical_engine_graph()
    {
        const unsigned tensor_dim = 1;
        const TSize    size       = 1;
        GraphType      g;

        TensorPtr tin = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr tmid;
        TensorPtr tout;

        TensorPtr ts_1;
        TensorPtr ts_2;
        TensorPtr ts_3;
        TensorPtr ts_4;
        TensorPtr ts_5;

        NodePtr n1;
        NodePtr n2;

        for (unsigned i = 0; i < 4; ++i)
        {
            tmid = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
            tout = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

            ts_1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
            ts_2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
            ts_3 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
            ts_4 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
            ts_5 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

            tin->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
            tmid->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
            tout->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
            if (i % 2 == 0)
            {
                ts_1->setMemorySectionID(10);
                ts_2->setMemorySectionID(20);
                ts_3->setMemorySectionID(30);
                ts_4->setMemorySectionID(40);
            }
            else
            {
                ts_1->setMemorySectionID(50);
                ts_2->setMemorySectionID(60);
                ts_3->setMemorySectionID(70);
                ts_4->setMemorySectionID(80);
            }
            ts_5->setMemorySectionID(30);

            std::string nodeName1("node");
            nodeName1 += std::to_string(i * 2);
            n1 = NodeFactory::createNode({tin, ts_1, ts_2, ts_3, ts_4},
                                         {tmid},
                                         nullptr,
                                         NodeFactory::addNodeTypeName,
                                         nodeName1.c_str());
            std::string nodeName2("node");
            nodeName2 += std::to_string(i * 2 + 1);
            n2 = NodeFactory::createNode({tmid, ts_5},
                                         {tout},
                                         nullptr,
                                         NodeFactory::addNodeTypeName,
                                         nodeName2.c_str());

            GraphEditor::addNode(g, n1);
            GraphEditor::addNode(g, n2);

            tin = tout;
        }

        BaseRegsCacheManager_SingleLogicalEngine<GraphType, REDUCED_CACHE_SIZE_7> mngr(g);
        mngr.go();

        unsigned nodeIdx = 0;
        for (const auto& n : g.getExeSortedNodes())
        {
            if (nodeIdx == 0)
            {
                ASSERT_EQ(n->getNodeName().compare("node0"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), REDUCED_CACHE_SIZE_7);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 10);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].sectionID, 20);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].indexInCache, 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].sectionID, 30);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[4].indexInCache, 4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[4].sectionID, 40);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[5].indexInCache, 5);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[5].sectionID, 50);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[6].indexInCache, 6);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[6].sectionID, 60);
            }
            if (nodeIdx == 1)
            {
                ASSERT_EQ(n->getNodeName().compare("node1"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            if (nodeIdx == 2)
            {
                ASSERT_EQ(n->getNodeName().compare("node2"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 2);
                // baseRegsCacheUpdate is in ascending order with respect to the cache index
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 70);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 80);
            }
            if (nodeIdx == 3)
            {
                ASSERT_EQ(n->getNodeName().compare("node3"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            if (nodeIdx == 4)
            {
                ASSERT_EQ(n->getNodeName().compare("node4"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 20);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 40);
            }
            if (nodeIdx == 5)
            {
                ASSERT_EQ(n->getNodeName().compare("node5"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            if (nodeIdx == 6)
            {
                // residual
                ASSERT_EQ(n->getNodeName().compare("node6"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 70);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 80);
            }
            if (nodeIdx == 7)
            {
                ASSERT_EQ(n->getNodeName().compare("node7"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            nodeIdx++;
        }
    }

    void saturating_node()
    {
        const unsigned tensor_dim = 1;
        const TSize    size       = 1;
        GraphType      g;

        TensorPtr t1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t3 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

        t1->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t2->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t3->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);

        TensorPtr ts_10 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_20 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_30 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_40 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_50 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

        ts_10->setMemorySectionID(10);
        ts_20->setMemorySectionID(20);
        ts_30->setMemorySectionID(30);
        ts_40->setMemorySectionID(40);
        ts_50->setMemorySectionID(50);

        NodePtr n1 = NodeFactory::createNode({t1, ts_10, ts_20}, {t2}, nullptr, NodeFactory::addNodeTypeName, "node1");
        NodePtr n2 = NodeFactory::createNode({t2, ts_10, ts_20, ts_30, ts_40, ts_50},
                                             {t3},
                                             nullptr,
                                             NodeFactory::addNodeTypeName,
                                             "node2");

        GraphEditor::addNode(g, n1);
        GraphEditor::addNode(g, n2);

        BaseRegsCacheManager_SingleLogicalEngine<GraphType, REDUCED_CACHE_SIZE_5> mngr(g);
        mngr.go();

        unsigned nodeIdx = 0;
        for (const auto& n : g.getExeSortedNodes())
        {
            if (nodeIdx == 0)
            {
                ASSERT_EQ(n->getNodeName().compare("node1"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), REDUCED_CACHE_SIZE_5);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 10);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].sectionID, 20);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].indexInCache, 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].sectionID, 30);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[4].indexInCache, 4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[4].sectionID, 40);
            }
            if (nodeIdx == 1)
            {
                ASSERT_EQ(n->getNodeName().compare("node2"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            nodeIdx++;
        }
    }

    void multi_logical_engines_graph()
    {
        const unsigned tensor_dim = 1;
        const TSize    size       = 1;
        GraphType      g;

        TensorPtr t1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t3 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t4 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t5 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t6 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t7 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t8 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr t9 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

        t1->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t2->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t3->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t4->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t5->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t6->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t7->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t8->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        t9->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);

        t1->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 10));
        t2->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 20));
        t3->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 30));
        t4->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 40));
        t5->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 50));
        t6->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 60));
        t7->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 70));
        t8->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 80));
        t9->setDramOffset(getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE, 90));

        TensorPtr ts_1_10  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_2_20  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_3_20  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_4_30  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_5_10  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_6_20  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_7_30  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_8_10  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_9_30  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_10_40 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_11_40 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_12_50 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_13_60 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr ts_14_20 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

        synMemoryDescriptor memDesc(true);  // persistent

        ts_1_10->setMemoryDescriptor(memDesc);
        ts_2_20->setMemoryDescriptor(memDesc);
        ts_3_20->setMemoryDescriptor(memDesc);
        ts_4_30->setMemoryDescriptor(memDesc);
        ts_5_10->setMemoryDescriptor(memDesc);
        ts_6_20->setMemoryDescriptor(memDesc);
        ts_7_30->setMemoryDescriptor(memDesc);
        ts_8_10->setMemoryDescriptor(memDesc);
        ts_9_30->setMemoryDescriptor(memDesc);
        ts_10_40->setMemoryDescriptor(memDesc);
        ts_11_40->setMemoryDescriptor(memDesc);
        ts_12_50->setMemoryDescriptor(memDesc);
        ts_13_60->setMemoryDescriptor(memDesc);
        ts_14_20->setMemoryDescriptor(memDesc);

        ts_1_10->setMemorySectionID(10);
        ts_2_20->setMemorySectionID(20);
        ts_3_20->setMemorySectionID(20);
        ts_4_30->setMemorySectionID(30);
        ts_5_10->setMemorySectionID(10);
        ts_6_20->setMemorySectionID(20);
        ts_7_30->setMemorySectionID(30);
        ts_8_10->setMemorySectionID(10);
        ts_9_30->setMemorySectionID(30);
        ts_10_40->setMemorySectionID(40);
        ts_11_40->setMemorySectionID(40);
        ts_12_50->setMemorySectionID(50);
        ts_13_60->setMemorySectionID(60);
        ts_14_20->setMemorySectionID(20);

        ts_1_10->setDramOffset(getVirtualAddressForMemoryID(10, 100));
        ts_2_20->setDramOffset(getVirtualAddressForMemoryID(20, 200));
        ts_3_20->setDramOffset(getVirtualAddressForMemoryID(20, 300));
        ts_4_30->setDramOffset(getVirtualAddressForMemoryID(30, 400));
        ts_5_10->setDramOffset(getVirtualAddressForMemoryID(10, 500));
        ts_6_20->setDramOffset(getVirtualAddressForMemoryID(20, 600));
        ts_7_30->setDramOffset(getVirtualAddressForMemoryID(30, 700));
        ts_8_10->setDramOffset(getVirtualAddressForMemoryID(10, 800));
        ts_9_30->setDramOffset(getVirtualAddressForMemoryID(30, 900));
        ts_10_40->setDramOffset(getVirtualAddressForMemoryID(40, 1000));
        ts_11_40->setDramOffset(getVirtualAddressForMemoryID(40, 2000));
        ts_12_50->setDramOffset(getVirtualAddressForMemoryID(50, 3000));
        ts_13_60->setDramOffset(getVirtualAddressForMemoryID(60, 4000));
        ts_14_20->setDramOffset(getVirtualAddressForMemoryID(20, 5000));

        // The name of the node will determine its logical engine affiliation in
        // BaseRegsCacheManager_MultiLogicalEngines. We just need to make sure that nodes mimicing TPC nodes, will use
        // TPC GUID since we call graph.runsOnTPC(n) in BaseRegsCacheManager. For nodes mimicing DMA and MME nodes, we
        // just need to make sure their GUID is non-TPC so we use gemm here.
        NodePtr n1 = NodeFactory::createNode({t1, ts_1_10}, {t2}, nullptr, NOP_KERNEL_NAME, "tpc1");
        NodePtr n2 = NodeFactory::createNode({t2, ts_2_20}, {t3}, nullptr, NodeFactory::gemmNodeTypeName, "mme1");
        NodePtr n3 = NodeFactory::createNode({t3, ts_3_20}, {t4, ts_4_30}, nullptr, NOP_KERNEL_NAME, "tpc2");
        NodePtr n4 = NodeFactory::createNode({t4, ts_5_10, ts_6_20, ts_7_30},
                                             {t5},
                                             nullptr,
                                             NodeFactory::gemmNodeTypeName,
                                             "dma1");
        NodePtr n5 = NodeFactory::createNode({t5}, {t6, ts_8_10}, nullptr, NOP_KERNEL_NAME, "tpc3");
        NodePtr n6 =
            NodeFactory::createNode({t6, ts_9_30, ts_10_40}, {t7}, nullptr, NodeFactory::gemmNodeTypeName, "mme2");
        NodePtr n7 = NodeFactory::createNode({t7, ts_11_40, ts_12_50, ts_13_60},
                                             {t8},
                                             nullptr,
                                             NodeFactory::gemmNodeTypeName,
                                             "dma2");
        NodePtr n8 = NodeFactory::createNode({t8, ts_14_20}, {t9}, nullptr, NodeFactory::gemmNodeTypeName, "mme3");

        GraphEditor::addNode(g, n1);
        GraphEditor::addNode(g, n2);
        GraphEditor::addNode(g, n3);
        GraphEditor::addNode(g, n4);
        GraphEditor::addNode(g, n5);
        GraphEditor::addNode(g, n6);
        GraphEditor::addNode(g, n7);
        GraphEditor::addNode(g, n8);

        BaseRegsCacheManager_MultiLogicalEngines<GraphType, REDUCED_CACHE_SIZE_4> mngr(g);
        mngr.go();

        unsigned nodeIdx = 0;
        for (const auto& n : g.getExeSortedNodes())
        {
            if (nodeIdx == 0)
            {
                ASSERT_EQ(n->getNodeName().compare("tpc1"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), REDUCED_CACHE_SIZE_4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 1);  // program data section is always on TPC
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].sectionID, 10);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].indexInCache, 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].sectionID, 20);
            }
            if (nodeIdx == 1)
            {
                ASSERT_EQ(n->getNodeName().compare("mme1"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), REDUCED_CACHE_SIZE_4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 20);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].sectionID, 30);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].indexInCache, 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].sectionID, 40);
            }
            if (nodeIdx == 2)
            {
                ASSERT_EQ(n->getNodeName().compare("tpc2"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 30);
            }
            if (nodeIdx == 3)
            {
                ASSERT_EQ(n->getNodeName().compare("dma1"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), REDUCED_CACHE_SIZE_4);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 10);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].sectionID, 20);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].indexInCache, 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[3].sectionID, 30);
            }
            if (nodeIdx == 4)
            {
                ASSERT_EQ(n->getNodeName().compare("tpc3"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 10);
            }
            if (nodeIdx == 5)
            {
                ASSERT_EQ(n->getNodeName().compare("mme2"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            if (nodeIdx == 6)
            {
                ASSERT_EQ(n->getNodeName().compare("dma2"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].indexInCache, 1);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[0].sectionID, 40);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].indexInCache, 2);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[1].sectionID, 50);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].indexInCache, 3);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate[2].sectionID, 60);
            }
            if (nodeIdx == 7)
            {
                ASSERT_EQ(n->getNodeName().compare("mme3"), 0);
                ASSERT_EQ(n->getNodeAnnotation().baseRegsCacheUpdate.size(), 0);
            }
            nodeIdx++;
        }
    }
};

#include "graph_optimizer_test.h"
#include "habana_device_types.h"
#include "node_factory.h"
#include "tensor.h"
#include "platform/gaudi/graph_compiler/dma_dispatcher.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "infra/scoped_configuration_change.h"
#include "utils.h"
#include "gtest/gtest.h"

class GaudiSignalOutTest : public GraphOptimizerTest
{
protected:
    std::shared_ptr<GaudiGraph> createGraph(int howMany = 2)
    {
        TSize               sizes = 10;
        auto                g     = std::make_shared<GaudiGraph>();
        synMemoryDescriptor memDescPersist(true);
        auto                t1 = std::make_shared<Tensor>(1, &sizes, syn_type_single);
        t1->setMemoryDescriptor(memDescPersist);
        t1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
        for (int i = 0; i < 2; i++)
        {
            auto t2 = std::make_shared<Tensor>(1, &sizes, syn_type_single);
            t2->setTensorAsExternal(true);
            t2->setMemoryDescriptor(memDescPersist);
            t2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i + 1);
            pNode n1 = NodeFactory::createNode({t1}, {t2}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "");
            GraphEditor::addNode(*g, n1);
            t1 = t2;
        }
        return g;
    }
};

TEST_F(GaudiSignalOutTest, compile)
{
    auto g = createGraph();
    g->compile();
}

TEST_F(GaudiSignalOutTest, simple)
{
    auto g = createGraph();
    g->compile();
    // Patchable monitors
    auto nodes = g->getExeSortedNodes();
    int  i     = 0;
    const SyncConventions& syncConventions = g->getCodeGenerator()->getSyncObjectManager()->getSyncConventions();
    for (auto& node : nodes)
    {
        EXPECT_FALSE(node->getNodeAnnotation().syncScheme.front().patchableMonitors.empty());
        EXPECT_EQ(node->getNodeAnnotation().syncScheme.front().patchableMonitors.size(), 1);
        auto monitor = node->getNodeAnnotation().syncScheme.front().patchableMonitors[0];
        EXPECT_EQ(monitor.monObject.setupValue, 1);
        EXPECT_EQ(monitor.monObject.shouldInc, true);
        EXPECT_EQ(monitor.monObject.syncId, syncConventions.getSignalOutGroup() / syncConventions.getGroupSize());
        EXPECT_EQ(monitor.monObject.armValue, ++i);
        auto preSyncs = node->getNodeAnnotation().syncScheme[0].preSyncsAndMon;
        EXPECT_EQ(preSyncs.size(), 1);
        EXPECT_EQ(preSyncs.front().type, SyncOrMonitor::MONITOR_OBJ);
        EXPECT_EQ(preSyncs.front().monitor.signalSyncId, syncConventions.getSignalOutGroup());
        EXPECT_EQ(preSyncs.front().monitor.setupValue, 1);
        EXPECT_EQ(preSyncs.front().monitor.shouldInc, true);
        EXPECT_EQ(preSyncs.front().monitor.armValue, i);
        EXPECT_EQ(preSyncs.front().monitor.mask, (1 << node->getNodeAnnotation().syncScheme.size()) - 1);
    }
}

TEST_F(GaudiSignalOutTest, alot)
{
    auto g = createGraph(200);
    g->compile();
    // Patchable monitors
    auto nodes = g->getExeSortedNodes();
    int  i     = 0;
    const SyncConventions& syncConventions = g->getCodeGenerator()->getSyncObjectManager()->getSyncConventions();
    for (auto& node : nodes)
    {
        EXPECT_FALSE(node->getNodeAnnotation().syncScheme.front().patchableMonitors.empty());
        EXPECT_EQ(node->getNodeAnnotation().syncScheme.front().patchableMonitors.size(), 1);
        auto monitor = node->getNodeAnnotation().syncScheme.front().patchableMonitors[0];
        EXPECT_EQ(monitor.monObject.setupValue, 1);
        EXPECT_EQ(monitor.monObject.shouldInc, true);
        EXPECT_EQ(monitor.monObject.syncId, syncConventions.getSignalOutGroup() / syncConventions.getGroupSize());
        EXPECT_EQ(monitor.monObject.armValue, ++i);
        auto preSyncs = node->getNodeAnnotation().syncScheme[0].preSyncsAndMon;
        EXPECT_EQ(preSyncs.size(), 1);
        EXPECT_EQ(preSyncs.front().type, SyncOrMonitor::MONITOR_OBJ);
        EXPECT_EQ(preSyncs.front().monitor.signalSyncId, syncConventions.getSignalOutGroup());
        EXPECT_EQ(preSyncs.front().monitor.setupValue, 1);
        EXPECT_EQ(preSyncs.front().monitor.shouldInc, true);
        EXPECT_EQ(preSyncs.front().monitor.armValue, i);
        EXPECT_EQ(preSyncs.front().monitor.mask, (1 << node->getNodeAnnotation().syncScheme.size()) - 1);
    }
}

#include <memory>

#include "gaudi_types.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/command_queue.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

class DmaDescQueueForTest : public gaudi::DmaDescQueue
{
public:
    DmaDescQueueForTest() : gaudi::DmaDescQueue(0, 0, 0, false) {}

    QueueCommandPtr getExeCmd(pNode n)
    {
        gaudi::DmaDesc desc;
        memset(&desc, 0, sizeof(desc));
        // Avoid empty job seq
        desc.dst_tsize_0.val = n->getOutput(0)->getSizeInElements(0);
        DescriptorWrapper<gaudi::DmaDesc> descWrap(desc);
        return gaudi::DmaDescQueue::getExeCmd(n, descWrap);
    }

    void updateQueueStateAfterPush(pNode n) override {return gaudi::DmaDescQueue::updateQueueStateAfterPush(n);}

    bool canUseLinDmaPacket(pNode n) {return gaudi::DmaDescQueue::canUseLinDmaPacket(n);}
};

class GaudiTestDmaEngBarrier : public GraphOptimizerTest
{
public:
    GaudiTestDmaEngBarrier()
        : m_sizes{8},
          m_t1(new Tensor(1, m_sizes, syn_type_int8)),
          m_t2(new Tensor(1, m_sizes, syn_type_int8))
    {
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
        m_allocatedQueue = new gaudi::DmaDescQueue(0, 0, 0, false);
        m_queue = reinterpret_cast<DmaDescQueueForTest*>(m_allocatedQueue);
        memcpy1 = NodeFactory::createNode({m_t1}, {m_t2}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "memcpy1");
        memcpy2 = NodeFactory::createNode({m_t1}, {m_t2}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "memcpy2");
        memset1 = NodeFactory::createNode({}, {m_t1}, nullptr, NodeFactory::dmaMemsetNodeTypeName, "memset1");
        memset2 = NodeFactory::createNode({}, {m_t2}, nullptr, NodeFactory::dmaMemsetNodeTypeName, "memset2");
        std::dynamic_pointer_cast<DMANode>(memcpy1)->setParallelLevel(1);
        std::dynamic_pointer_cast<DMANode>(memcpy2)->setParallelLevel(1);
        std::dynamic_pointer_cast<DMANode>(memset1)->setParallelLevel(1);
        std::dynamic_pointer_cast<DMANode>(memset2)->setParallelLevel(1);
    }

    void updateNodePushed(pNode prevNode)
    {
        m_queue->updateQueueStateAfterPush(prevNode);
    }

    unsigned getNextCommandEngBarrier(pNode nextNode)
    {
        if (m_queue->canUseLinDmaPacket(nextNode))
        {
            return dynamic_cast<gaudi::DmaCommand&>(*m_queue->getExeCmd(nextNode)).getPacket().eng_barrier;
        }
        else
        {
            return dynamic_cast<gaudi::ExecuteDmaDesc&>(*m_queue->getExeCmd(nextNode)).getPacket().eng_barrier;
        }
    }

    ~GaudiTestDmaEngBarrier()
    {
        delete m_queue;
    }

    pNode memcpy1;
    pNode memcpy2;
    pNode memset1;
    pNode memset2;

protected:
    TSize m_sizes[1];
    pTensor m_t1;
    pTensor m_t2;
    CommandQueue* m_allocatedQueue;
    DmaDescQueueForTest* m_queue;
};

TEST_F(GaudiTestDmaEngBarrier, DmaDescQueue_set_eng_barrier_for_memcpy_after_memset)
{
    updateNodePushed(memset1);
    ASSERT_EQ(1, getNextCommandEngBarrier(memcpy1));
}

TEST_F(GaudiTestDmaEngBarrier, DmaDescQueue_set_eng_barrier_for_memset_after_memcpy)
{
    updateNodePushed(memcpy1);
    ASSERT_EQ(1, getNextCommandEngBarrier(memset1));
}

TEST_F(GaudiTestDmaEngBarrier, DmaDescQueue_clear_eng_barrier_for_memcpy_after_memcpy)
{
    updateNodePushed(memcpy1);
    ASSERT_EQ(0, getNextCommandEngBarrier(memcpy2));
}

TEST_F(GaudiTestDmaEngBarrier, DmaDescQueue_clear_eng_barrier_for_memset_after_memset)
{
    updateNodePushed(memset1);
    ASSERT_EQ(0, getNextCommandEngBarrier(memset2));
}

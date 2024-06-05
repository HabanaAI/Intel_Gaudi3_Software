#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "tensor.h"
#include "platform/gaudi/graph_compiler/dma_dispatcher.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "infra/scoped_configuration_change.h"

class TestGaudiGraph: public GaudiGraph
{
public:
    const QueueDispatcherMap& getDispatchers() { return m_codeGenerator->getDmaDispatchers(); }
};

class GaudiLinearDMATest : public GraphOptimizerTest
{
protected:
    bool findLinDMA(std::shared_ptr<Node> pNode);
    pNode generateNode(const uint64_t* strides, unsigned tensor_dim, const TSize* dims_sizes);
};

pNode GaudiLinearDMATest::generateNode(const uint64_t* strides, unsigned tensor_dim, const TSize* dims_sizes)
{
    pTensor      i = pTensor(new Tensor(tensor_dim, dims_sizes, syn_type_single, nullptr, strides));
    pTensor      o = pTensor(new Tensor(tensor_dim, dims_sizes, syn_type_single, nullptr, strides));
    pNode        n = NodeFactory::createNode({i}, {o}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "N1");

    synMemoryDescriptor memDescPersist(true);
    i->setMemoryDescriptor(memDescPersist);
    i->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o->setMemoryDescriptor(memDescPersist);
    o->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    return n;
}

bool GaudiLinearDMATest::findLinDMA(std::shared_ptr<Node> n)
{
    TestGaudiGraph g;

    GraphEditor::addNode(g, n);
    g.compile();

    bool foundLinDMA = false;
    for (const std::pair<const QueueDispatcherParams, QueueDispatcherPtr>& dispatcher : g.getDispatchers())
    {
        for (unsigned dmaEng = 0; dmaEng < dispatcher.second->getNumEngines(); ++dmaEng)
        {
            const CommandQueuePtr& cmdQueue = dispatcher.second->getQueue(dmaEng);
            for (auto& cmd : cmdQueue->getCommands(false))
            {
                gaudi::DmaDeviceInternal* dmaDevInt = dynamic_cast<gaudi::DmaDeviceInternal*>(cmd.get());
                if(dmaDevInt) {
                    foundLinDMA = true;
                }
            }
        }
    }
    return foundLinDMA;
}

TEST_F(GaudiLinearDMATest, verify_linear_dma_dense)
{
    ScopedConfigurationChange scopedChange("LIN_DMA_OPTIMIZATION_ENABLED", "true");

    const unsigned  tensor_dim      = 3;
    const TSize     dims_sizes[]    = {2 ,3, 4};

    /*
    * Using dense tensors.
    * First dimension stride is always 1 and therefore omitted from the strides array.
    * Following strides are always the product of previous logical dimensions sizes.
    */

    uint64_t strides[] = {sizeof(float),
                          dims_sizes[0] * sizeof(float),
                          dims_sizes[0] * dims_sizes[1] * sizeof(float),
                          dims_sizes[0] * dims_sizes[1] * dims_sizes[2] * sizeof(float)};

    pNode n = generateNode(strides, tensor_dim, dims_sizes);
    ASSERT_TRUE(findLinDMA(n));
}

TEST_F(GaudiLinearDMATest, verify_linear_dma_sparse)
{
    ScopedConfigurationChange scopedChange("LIN_DMA_OPTIMIZATION_ENABLED", "true");
    const unsigned  tensor_dim      = 3;
    const TSize     dims_sizes[]    = {2 ,3, 4};

    /*
     * Using non dense (sparse) tensors.
     * First dimension stride is always 1 and therefore omitted from the strides array.
     * First stride is 0, therefore engines treat the first FCD vector as logically replicated over 2nd dimension.
     * Following strides are always the product of previous logical dimensions sizes.
     */

    uint64_t strides[] = {sizeof(float),
                          0,
                          dims_sizes[0] * dims_sizes[1] * sizeof(float),
                          dims_sizes[0] * dims_sizes[1] * dims_sizes[2] * sizeof(float)};

    pNode n = generateNode(strides, tensor_dim, dims_sizes);
    ASSERT_FALSE(findLinDMA(n));
}

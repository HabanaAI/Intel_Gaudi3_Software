#include "platform/gaudi/graph_compiler/tpc_dispatcher.h"

#include "gaudi_code_generator.h"
#include "gaudi_graph.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "node_annotation.h"
#include "platform/gaudi/graph_compiler/command_queue.h"

namespace gaudi
{
TPCDispatcher::TPCDispatcher(unsigned activatedTpcEnginesMask, uint16_t sendSyncEventsMask, HabanaGraph* g)
: QueueDispatcher("TPC", true)
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    init(g->getHALReader()->getNumTpcEngines(), activatedTpcEnginesMask, g);
}

TPCDispatcher::~TPCDispatcher()
{
}

void TPCDispatcher::dispatchNode(const pNode &n, HabanaGraph *g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    TPCDescriptorsWrappers& descriptorsWrappers = downcaster<GaudiCodeGenerator>(g->getCodeGenerator().get())->getTPCNodeDescriptorsWrappers(n);
    HB_ASSERT( !descriptorsWrappers.empty(), "No TPC descriptors found for node");

    QueueDispatcher::dispatchNode(n, g, isSetup);

    unsigned numTpcEngines = getNumEngines();
    unsigned numDescs      = descriptorsWrappers.size();

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} TPC engines", numDescs, numTpcEngines);

    // Move descriptors list to mapping
    std::vector<std::vector<Settable<DescriptorWrapper<TpcDesc>>>> pipelineDescriptors;
    fillPipelineMapping<TpcDesc>(numDescs, descriptorsWrappers, pipelineDescriptors);

    assignPredicateToDescriptors(pipelineDescriptors);
    setInactiveQueueState<TpcDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<TpcDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

CommandQueue* TPCDispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    return new gaudi::TpcQueue(queueId, engineIdx, (bool) (engineIdx & m_sendSyncEventsMask), g->isDynamicShape());
}

} // namespace gaudi

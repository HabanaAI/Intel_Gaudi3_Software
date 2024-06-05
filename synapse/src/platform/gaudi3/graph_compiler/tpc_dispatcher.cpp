#include "platform/gaudi3/graph_compiler/tpc_dispatcher.h"

#include "gaudi3_graph.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "node_annotation.h"
#include "platform/gaudi3/graph_compiler/command_queue.h"

namespace gaudi3
{
TpcDispatcher::TpcDispatcher(unsigned activatedTpcEnginesMask, uint16_t sendSyncEventsMask, HabanaGraph* g)
: QueueDispatcher("TPC")
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    // Only one engine exists from GC stand-point with dynamic work distribution mode
    init(1, activatedTpcEnginesMask, g);
}

void TpcDispatcher::dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    const TpcDescriptorsWrappers& descriptorsWrappers = getWrappers(*g, n);
    HB_ASSERT(!descriptorsWrappers.empty(), "No TPC descriptors found for node");

    QueueDispatcher::dispatchNode(n, g, isSetup);

    unsigned numDescs = descriptorsWrappers.size();

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} TPC engines", numDescs, getNumEngines());

    // Move descriptors list to mapping
    std::vector<std::vector<Settable<DescriptorWrapper<TpcDesc>>>> pipelineDescriptors;
    fillPipelineMapping<TpcDesc>(numDescs, descriptorsWrappers, pipelineDescriptors);

    assignPredicateToDescriptors(pipelineDescriptors);  // required in order to predicate the tid (index space) part
    setInactiveQueueState<TpcDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<TpcDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

CommandQueue* TpcDispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    return new gaudi3::TpcQueue(queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

const TpcDescriptorsWrappers& TpcDispatcher::getWrappers(HabanaGraph& g, const NodePtr& n) const
{
    return static_cast<Gaudi3Graph&>(g).getTpcNodeDescriptorsWrappers(n);
}

}  // namespace gaudi3

#include "platform/gaudi2/graph_compiler/tpc_dispatcher.h"

#include "gaudi2_code_generator.h"
#include "gaudi2_graph.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "node_annotation.h"
#include "platform/gaudi2/graph_compiler/command_queue.h"

namespace gaudi2
{

TpcDispatcher::TpcDispatcher(unsigned activatedTpcEnginesMask, uint16_t sendSyncEventsMask, HabanaGraph* g)
: QueueDispatcher("TPC")
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    uint32_t numEngines = 1; // Only one engine exist from GC stand-point in Arc architecture
    init(numEngines, activatedTpcEnginesMask, g);
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

    assignPredicateToDescriptors(pipelineDescriptors); //required in order to predicate the tid (index space) part
    setInactiveQueueState<TpcDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<TpcDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

CommandQueue* TpcDispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    return new gaudi2::TpcQueue(queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

const TpcDescriptorsWrappers& TpcDispatcher::getWrappers(HabanaGraph& g, const NodePtr& n) const
{
    return static_cast<Gaudi2CodeGenerator&>(*g.getCodeGenerator().get()).getTpcNodeDescriptorsWrappers(n);
}

} // namespace gaudi2

#include "command_queue.h"
#include "gaudi_graph.h"
#include "gaudi_code_generator.h"

#include "mme_dispatcher.h"

namespace gaudi
{

MmeDispatcher::MmeDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g)
: QueueDispatcher("Gaudi MME")
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    // Both of MME engines must be active all the time
    static const uint8_t activatedEnginesMask = 0x3;
    init(g->getHALReader()->getNumMmeEngines(), activatedEnginesMask, g);
}

void MmeDispatcher::dispatchNode(const pNode& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    const MmeDescriptorsWrappers& descriptorsWrappers =
        downcaster<GaudiCodeGenerator>(g->getCodeGenerator().get())->getMmeNodeDescriptorsWrappers(n);
    HB_ASSERT(!descriptorsWrappers.empty(), "No MME descriptors found for node");

    if (descriptorsWrappers.size() % g->getHALReader()->getNumMmeEngines() != 0)
    {
        LOG_CRITICAL(QMAN, "MME descriptors doesn't spread on all queues");
        HB_ASSERT(false, "MME descriptors doesn't spread on all queues");
    }

    QueueDispatcher::dispatchNode(n, g, isSetup);

    unsigned numMmeEngines = getNumEngines();
    unsigned numDescs      = descriptorsWrappers.size();

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} MME engines", numDescs, numMmeEngines);

    // Move descriptors list to mapping
    std::vector<std::vector<Settable<DescriptorWrapper<MmeDesc>>>> pipelineDescriptors;
    fillPipelineMapping<MmeDesc>(numDescs, descriptorsWrappers, pipelineDescriptors, false);

    setInactiveQueueState<MmeDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<MmeDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

CommandQueue* MmeDispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    return new MmeQueue(queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

}

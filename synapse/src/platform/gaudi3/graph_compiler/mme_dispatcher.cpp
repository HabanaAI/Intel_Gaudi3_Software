#include "mme_dispatcher.h"

#include "command_queue.h"
#include "gaudi3_graph.h"
#include "hal_reader/gaudi3/hal_reader.h"

namespace gaudi3
{
MmeDispatcher::MmeDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g) : QueueDispatcher("MME")
{
    m_sendSyncEventsMask                      = sendSyncEventsMask;
    uint32_t             numEngines           = Gaudi3HalReader::instance()->getNumMmeEngines();
    static const uint8_t activatedEnginesMask = (1 << numEngines) - 1;  // Should be 0xFF
    init(numEngines, activatedEnginesMask, g);
}

void MmeDispatcher::dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    const MmeDescriptorsWrappers& descriptorsWrappers = getWrappers(*g, n);
    unsigned                      numDescs            = descriptorsWrappers.size();
    HB_ASSERT(!descriptorsWrappers.empty(), "No MME descriptors found for node");
    HB_ASSERT((numDescs % Gaudi3HalReader::instance()->getNumMmeEngines()) == 0,
              "MME descriptors don't spread evenly on all queues");

    QueueDispatcher::dispatchNode(n, g, isSetup);

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} MME engines", numDescs, getNumEngines());

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
    return new gaudi3::MmeQueue(queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

const MmeDescriptorsWrappers& MmeDispatcher::getWrappers(HabanaGraph& g, const NodePtr& n) const
{
    return static_cast<Gaudi3Graph&>(g).getMmeNodeDescriptorsWrappers(n);
}

}  // namespace gaudi3

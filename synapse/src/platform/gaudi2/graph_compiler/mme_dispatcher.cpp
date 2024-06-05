#include "mme_dispatcher.h"

#include "command_queue.h"
#include "gaudi2_code_generator.h"
#include "gaudi2_graph.h"
#include "hal_reader/gaudi2/hal_reader.h"

namespace gaudi2
{

MmeDispatcher::MmeDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g)
: QueueDispatcher("MME")
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    static const uint8_t activatedEnginesMask = 0x3; // both MME engines are always active
    init(Gaudi2HalReader::instance()->getNumMmeEngines(), activatedEnginesMask, g);
}

void MmeDispatcher::dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    const MmeDescriptorsWrappers& descriptorsWrappers = getWrappers(*g, n);
    unsigned numDescs = descriptorsWrappers.size();
    HB_ASSERT(!descriptorsWrappers.empty(), "No MME descriptors found for node");
    HB_ASSERT((numDescs % Gaudi2HalReader::instance()->getNumMmeEngines()) == 0, "MME descriptors don't spread evenly on all queues");

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
    return new gaudi2::MmeQueue(queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

const MmeDescriptorsWrappers& MmeDispatcher::getWrappers(HabanaGraph& g, const NodePtr& n) const
{
    return static_cast<Gaudi2CodeGenerator&>(*g.getCodeGenerator()).getMmeNodeDescriptorsWrappers(n);
}

} // namespace gaudi2

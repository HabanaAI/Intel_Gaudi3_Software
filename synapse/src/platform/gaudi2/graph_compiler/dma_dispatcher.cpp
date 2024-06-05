#include "platform/gaudi2/graph_compiler/dma_dispatcher.h"

#include "gaudi2_code_generator.h"
#include "gaudi2_graph.h"
#include "platform/gaudi2/graph_compiler/command_queue.h"

#include <bitset>

namespace gaudi2
{

DmaDispatcher::DmaDispatcher(uint8_t activatedDmaEnginesMask,
                             uint16_t sendSyncEventsMask,
                             unsigned dispatcherIndex,
                             HabanaGraph* g)
: QueueDispatcher("DMA"),
  m_numEngines(std::bitset<8>(activatedDmaEnginesMask).count()),
  m_dispatcherIndex(dispatcherIndex)
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    init(g->getHALReader()->getNumInternalDmaEngines(), activatedDmaEnginesMask, g);
    memset(&m_emptyJobDesc, 0, sizeof(m_emptyJobDesc));
}

void DmaDispatcher::dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    const DmaDescriptorsWrappers& descriptorsWrappers = getWrappers(*g, n);
    HB_ASSERT(descriptorsWrappers.size() > 0, "{}: No DMA descriptors found for node", n->getNodeName());

    QueueDispatcher::dispatchNode(n, g, isSetup);

    unsigned numDescs = descriptorsWrappers.size();

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} DMA engines", numDescs, getNumEngines());

    // Move descriptors list to mapping
    std::vector<std::vector<Settable<DescriptorWrapper<DmaDesc>>>> pipelineDescriptors;
    m_emptyJobDesc.ctx.idx.val = n->getContextId();
    fillPipelineMapping<DmaDesc>(numDescs, descriptorsWrappers, pipelineDescriptors, true, true, &m_emptyJobDesc);

    // For each pipeline, take FW context from the first descriptor (it was filled by the descriptor generator) and
    // copy it to the next numEngines-1 descriptors. This will allow us to dispatch in round robin fashion on the one
    // hand, while on the other ensuring the FW process only *one* sync scheme context in a pipeline, which is needed
    // in case we have multiple descriptors per engine in a single pipeline (like in transpose node).
    for (std::vector<Settable<DescriptorWrapper<DmaDesc>>>& pipelineDesc : pipelineDescriptors)
    {
        for (unsigned i = 1; i < m_numEngines; i++)
        {
            pipelineDesc[i].value().setFwCtx(pipelineDesc[0].value().getFwCtx());
        }
    }

    setInactiveQueueState<DmaDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<DmaDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

CommandQueue* DmaDispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    return new DmaDescQueue(DEVICE_DMA_LOGICAL_QUEUE, queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

const DmaDescriptorsWrappers& DmaDispatcher::getWrappers(HabanaGraph& g, const NodePtr& n) const
{
    return static_cast<Gaudi2CodeGenerator&>(*g.getCodeGenerator().get()).getDmaNodeDescriptorsWrappers(n);
}

} // namespace gaudi2

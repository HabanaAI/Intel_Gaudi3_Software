#include "platform/gaudi/graph_compiler/dma_dispatcher.h"

#include "gaudi_code_generator.h"
#include "gaudi_graph.h"
#include "platform/gaudi/graph_compiler/command_queue.h"

#include <bitset>

namespace gaudi
{

DMADispatcher::DMADispatcher(uint8_t activatedDmaEnginesMask,
                             uint16_t sendSyncEventsMask,
                             unsigned dispatcherIndex,
                             HabanaGraph* g)
: QueueDispatcher("DMA", true /*round robin mode*/),
  m_numEngines(std::bitset<8>(activatedDmaEnginesMask).count()),
  m_dispatcherIndex(dispatcherIndex)
{
    m_sendSyncEventsMask = sendSyncEventsMask;
    init(g->getHALReader()->getNumInternalDmaEngines(), activatedDmaEnginesMask, g);
}

DMADispatcher::~DMADispatcher()
{
}

void DMADispatcher::dispatchNode(const pNode &n, HabanaGraph *g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    DMADescriptorsWrappers& descriptorsWrappers =
        downcaster<GaudiCodeGenerator>(g->getCodeGenerator().get())->getDMANodeDescriptorsWrappers(n);
    HB_ASSERT(descriptorsWrappers.size() > 0, "{}: No DMA descriptors found for node", n->getNodeName());

    QueueDispatcher::dispatchNode(n, g, isSetup);

    unsigned numDmaEngines = getNumEngines();
    unsigned numDescs      = descriptorsWrappers.size();

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} DMA engines", numDescs, numDmaEngines);

    // Move descriptors list to mapping
    std::vector<std::vector<Settable<DescriptorWrapper<DmaDesc>>>> pipelineDescriptors;
    fillPipelineMapping<DmaDesc>(numDescs, descriptorsWrappers, pipelineDescriptors, true, true);

    setInactiveQueueState<DmaDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<DmaDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

void DMADispatcher::addEmptyJob(const NodePtr& n, uint32_t pipeLevel, CommandQueuePtr queue, bool isLastPipelineLevel)
{
    DmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.dst_base_hi.ctx_id_hi = (uint32_t)((n->getContextId() & 0xFF00) >> 8);
    DescriptorWrapper<DmaDesc> descWrapper(desc);
    std::vector<Settable<DescriptorWrapper<DmaDesc>>> pipeDescs;
    pipeDescs.push_back(descWrapper);

    DescCommandQueue<DmaDesc>* descQueue = downcaster<DescCommandQueue<DmaDesc>>(queue);
    descQueue->AddPartialNode(n,
                              pipeDescs.back().value(),
                              &pipeDescs,
                              pipeLevel,
                              false,
                              m_baseRegsCache,
                              isLastPipelineLevel,
                              nullptr,  // preSyncCmds
                              nullptr,  // postSyncCmds
                              true,
                              true);
}

CommandQueue* DMADispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    unsigned logicalQueue = baseDmaLogicalQueue(m_numEngines) + m_dispatcherIndex;

    return new DmaDescQueue(logicalQueue, queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

} // namespace gaudi

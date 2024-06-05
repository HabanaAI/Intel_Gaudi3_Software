#include "platform/gaudi3/graph_compiler/rotator_dispatcher.h"

#include "descriptor_generator.h"
#include "gaudi3_graph.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "node_annotation.h"
#include "patch_point_generator.h"
#include "platform/gaudi3/graph_compiler/command_queue.h"

namespace gaudi3
{
RotatorDispatcher::RotatorDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g) : QueueDispatcher("ROTATOR")
{
    m_sendSyncEventsMask                      = sendSyncEventsMask;
    uint32_t             numEngines           = Gaudi3HalReader::instance()->getNumRotatorEngines();
    static const uint8_t activatedEnginesMask = (1 << numEngines) - 1;  // Should be 0xF
    init(numEngines, activatedEnginesMask, g);

    m_inDramAddr  = g->getCodeGenerator()->getWorkspaceAllocator().Allocate(Gaudi3HalReader::instance()->getCacheLineSizeInBytes(),
                                                       Gaudi3HalReader::instance()->getCacheLineSizeInBytes());
    m_outDramAddr = g->getCodeGenerator()->getWorkspaceAllocator().Allocate(Gaudi3HalReader::instance()->getCacheLineSizeInBytes(),
                                                        Gaudi3HalReader::instance()->getCacheLineSizeInBytes());

    // create the empty job descriptor and mask
    gaudi3::DescriptorGenerator::generateRotatorEmptyJobDescriptor(m_emptyJobDesc,
                                                                   m_descMask,
                                                                   m_inDramAddr.value(),
                                                                   m_outDramAddr.value());
}

void RotatorDispatcher::dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    RotatorDescriptorsWrappers& descriptorsWrappers = downcaster<Gaudi3Graph>(g)->getRotateNodeDescriptorsWrappers(n);
    HB_ASSERT(!descriptorsWrappers.empty(), "No Rotator descriptors found for node");

    QueueDispatcher::dispatchNode(n, g, isSetup);

    unsigned numDescs   = descriptorsWrappers.size();
    unsigned numEngines = getNumEngines();

    LOG_DEBUG(QMAN, "Splitting {} descriptors to {} Rotator engines", numDescs, numEngines);

    m_emptyJobDesc.context_id.val = n->getContextId();

    // Move descriptors list to mapping
    std::vector<std::vector<Settable<DescriptorWrapper<RotatorDesc>>>> pipelineDescriptors;

    fillPipelineMapping<RotatorDesc>(numDescs, descriptorsWrappers, pipelineDescriptors, true, true, &m_emptyJobDesc);

    // For each pipeline, take FW context from the first descriptor (it was filled by the descriptor generator) and
    // copy it to the next numEngines-1 descriptors. This will allow us to dispatch in round robin fashion on the one
    // hand, while on the other ensuring the FW process only *one* sync scheme context in a pipeline, which is needed
    // in case we have multiple descriptors per engine in a single pipeline (like in rotator node).

    for (std::vector<Settable<DescriptorWrapper<RotatorDesc>>>& pipelineDesc : pipelineDescriptors)
    {
        for (unsigned i = 1; i < numEngines; i++)
        {
            pipelineDesc[i].value().setFwCtx(pipelineDesc[0].value().getFwCtx());
        }
    }

    setInactiveQueueState<RotatorDesc>(pipelineDescriptors);
    addPreSyncScheme(n, isSetup);
    addDescsToNode<RotatorDesc>(n, pipelineDescriptors, isSetup);
    addPostSyncScheme(n, isSetup);
}

CommandQueue* RotatorDispatcher::createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g)
{
    return new gaudi3::RotatorQueue(queueId, engineIdx, (bool)(engineIdx & m_sendSyncEventsMask));
}

void RotatorDispatcher::updateEmptyJobDescWrapper(void* wrapper)
{
    RotatorDescWrapper* descWrapper = (RotatorDescWrapper*)wrapper;

    descWrapper->getMask() = m_descMask;

    Gaudi3RotatorPatchPointGenerator ppGenerator;
    ppGenerator.generateEmptyJobRotatorPatchPoints(*descWrapper);
}

}  // namespace gaudi3

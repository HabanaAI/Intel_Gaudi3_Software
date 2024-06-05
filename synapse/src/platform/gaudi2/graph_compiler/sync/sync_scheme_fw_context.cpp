#include "sync_scheme_fw_context.h"
namespace gaudi2
{
SyncSchemeFwContext::SyncSchemeFwContext(const NodeUtility& nodeUtility) : m_nodeUtility(nodeUtility) {}

void SyncSchemeFwContext::reset(unsigned logicalId)
{
    m_arcLastDependencies[logicalId].clear();
    m_arcLastEmittedSignal[logicalId] = 0;
    // no need to reset the breakpoint since it is implemented using long sync-obj
}

// NOTE: The sync scheme data contains absolute values whereas the FW Context should contain only deltas
void SyncSchemeFwContext::fillArcDeltaSyncs(unsigned                  logicalId,               // input
                                            const ArcSyncInteraction& arcSyncs,                // input sync scheme
                                            unsigned*                 emittedSigValIncrement,  // output
                                            unsigned*                 virtSobBitmap,           // output
                                            virt_sob_ids_t&           virtSobIds)              // output
{
    HB_ASSERT_PTR(emittedSigValIncrement);
    HB_ASSERT_PTR(virtSobBitmap);

    *emittedSigValIncrement = 0;
    *virtSobBitmap = 0;
    virtSobIds = {0};

    // Handle emitted signal
    unsigned deltaEmittedSigVal = 0;
    if (arcSyncs.emittedSigVal.is_set())
    {
        HB_ASSERT(arcSyncs.emittedSigVal.value() >= m_arcLastEmittedSignal[logicalId], "unexpected emitted sigVal");
        deltaEmittedSigVal = arcSyncs.emittedSigVal.value() - m_arcLastEmittedSignal[logicalId];
        m_arcLastEmittedSignal[logicalId] = arcSyncs.emittedSigVal.value();  // update last
    }
    *emittedSigValIncrement = deltaEmittedSigVal;

    // Handle dependencies
    DependencyMap& lastDep = m_arcLastDependencies[logicalId];
    for (const auto& dep : arcSyncs.dependencies)
    {
        unsigned depLogicalId = dep.first;
        unsigned depSigVal    = dep.second;
        unsigned deltaDep     = depSigVal;
        bool     updateLast   = true;

        if (lastDep.find(depLogicalId) != lastDep.end())
        {
            if (depSigVal < lastDep[depLogicalId])  // this can happen if control-edges are used
            {
                deltaDep = 0;
                updateLast = false;
            }
            else
            {
                deltaDep = depSigVal - lastDep[depLogicalId];
            }
        }

        if (updateLast) lastDep[depLogicalId] = depSigVal;  // update last

        if (deltaDep)
        {
            if (depLogicalId == gaudi2::DEVICE_TPC_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_TPC);
                virtSobIds.tpc.raw = deltaDep;
            }
            else if (depLogicalId == gaudi2::DEVICE_MME_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_MME);
                virtSobIds.mme.raw = deltaDep;
            }
            else if (depLogicalId == gaudi2::DEVICE_DMA_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_EDMA);
                virtSobIds.edma.raw = deltaDep;
            }
            else if (depLogicalId == gaudi2::DEVICE_ROT_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_ROT);
                virtSobIds.rot.raw = deltaDep;
            }
            else
            {
                HB_ASSERT(0, "Unsupported logical ID");
            }
        }
    }

    // Handle breakpoint
    if (arcSyncs.breakpoint.is_set())
    {
        HB_ASSERT(arcSyncs.breakpoint.value() >= m_arcLastBreakpoint[logicalId], "unexpected breakpoint value");
        *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_DEBUG);
        virtSobIds.dbg.raw = arcSyncs.breakpoint.value() - m_arcLastBreakpoint[logicalId];
        m_arcLastBreakpoint[logicalId] = arcSyncs.breakpoint.value();  // update last
    }
}

void SyncSchemeFwContext::printArcDeltaSyncs(const NodePtr&        node,
                                             unsigned              pipelineLevel,
                                             unsigned              emittedSigValIncrement,
                                             unsigned              virtSobBitmap,
                                             const virt_sob_ids_t& virtSobIds) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(SYNC_SCHEME_DLT)) return;
    LOG_DEBUG(SYNC_SCHEME_DLT,
              "FW Context Syncs Dump> node: {}, id: {}, type: {}, logical queue id: {}",
              node->getNodeName(),
              node->getId(),
              node->getEngineTypeStr(),
              deviceTypeToLogicalQueue(m_nodeUtility.getNodeDeviceType(node)));
    LOG_DEBUG(SYNC_SCHEME_DLT, "  Pipeline Level #{}:", pipelineLevel);
    LOG_DEBUG(SYNC_SCHEME_DLT, "    Emitted signal increment: {}", emittedSigValIncrement);
    LOG_DEBUG(SYNC_SCHEME_DLT, "    Virtual SOB Dependencies (monitors):{}", virtSobBitmap == 0 ? " none" : "");
    if (virtSobBitmap & (1 << VIRTUAL_SOB_INDEX_TPC))
    {
        LOG_DEBUG(SYNC_SCHEME_DLT, "      TPC delta value: {}", virtSobIds.tpc.raw);
    }
    if (virtSobBitmap & (1 << VIRTUAL_SOB_INDEX_MME))
    {
        LOG_DEBUG(SYNC_SCHEME_DLT, "      MME delta value: {}", virtSobIds.mme.raw);
    }
    if (virtSobBitmap & (1 << VIRTUAL_SOB_INDEX_EDMA))
    {
        LOG_DEBUG(SYNC_SCHEME_DLT, "      DMA delta value: {}", virtSobIds.edma.raw);
    }
    if (virtSobBitmap & (1 << VIRTUAL_SOB_INDEX_ROT))
    {
        LOG_DEBUG(SYNC_SCHEME_DLT, "      ROT delta value: {}", virtSobIds.rot.raw);
    }
    if (virtSobBitmap & (1 << VIRTUAL_SOB_INDEX_DEBUG))
    {
        LOG_DEBUG(SYNC_SCHEME_DLT, "      Breakpoint delta value: {}", virtSobIds.dbg.raw);
    }
}

}
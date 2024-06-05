#include "sync_scheme_fw_context.h"

namespace gaudi3
{
SyncSchemeFwContext::SyncSchemeFwContext(const HabanaGraph& graph) : m_graph(graph) {}

void SyncSchemeFwContext::reset(unsigned logicalId)
{
    m_arcLastEmittedSignal[logicalId] = 0;

    if (logicalId == DEVICE_XPS_LOGICAL_QUEUE) m_xpsResetArrived = true;
    if (logicalId == DEVICE_MME_LOGICAL_QUEUE) m_mmeResetArrived = true;

    bool canResetMmeDep = (m_mmeResetArrived || !m_mmeInUse) && (m_xpsResetArrived || !m_xpsInUse);

    // Since the transpose (xps) and mme (gemm) are sharing the same qman and input port,
    // the history of dependencies is shared and we need to reset the common one.
    if (logicalId == DEVICE_XPS_LOGICAL_QUEUE) logicalId = DEVICE_MME_LOGICAL_QUEUE;

    if (logicalId != DEVICE_MME_LOGICAL_QUEUE)
    {
        m_arcLastDependencies[logicalId].clear();
    }
    else if (canResetMmeDep)
    {
        m_arcLastDependencies[logicalId].clear();
        m_xpsInUse = m_mmeInUse = m_mmeResetArrived = m_xpsResetArrived = false;
    }

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

    // Since the transpose (xps) and mme (gemm) are sharing the same qman and input port, the history of
    // previous dependencies and breakpoint should be shared in the delta calculations below.
    if (logicalId == DEVICE_XPS_LOGICAL_QUEUE) logicalId = DEVICE_MME_LOGICAL_QUEUE;

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
            if (depLogicalId == gaudi3::DEVICE_TPC_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_TPC);
                virtSobIds.tpc.raw = deltaDep;
            }
            else if (depLogicalId == gaudi3::DEVICE_MME_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_MME);
                virtSobIds.mme.raw = deltaDep;
            }
            else if (depLogicalId == gaudi3::DEVICE_ROT_LOGICAL_QUEUE)
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_ROT);
                virtSobIds.rot.raw = deltaDep;
            }
            else if (depLogicalId == gaudi3::DEVICE_XPS_LOGICAL_QUEUE)  // transpose
            {
                *virtSobBitmap |= (1 << VIRTUAL_SOB_INDEX_MME_XPOSE);
                virtSobIds.mme_xpose.raw = deltaDep;
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

void SyncSchemeFwContext::printArcDeltaSyncs(NodePtr               node,
                                             unsigned              pipelineLevel,
                                             unsigned              emittedSigValIncrement,
                                             unsigned              virtSobBitmap,
                                             const virt_sob_ids_t& virtSobIds) const
{
    LOG_DEBUG(SYNC_SCHEME_DLT,
              "FW Context Syncs Dump> node: {}, id: {}, type: {}, logical queue id: {}",
              node->getNodeName(),
              node->getId(),
              node->getEngineTypeStr(),
              deviceTypeToLogicalQueue(m_graph.getNodeUtility().getNodeDeviceType(node), *node));
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
    if (virtSobBitmap & (1 << VIRTUAL_SOB_INDEX_MME_XPOSE))
    {
        LOG_DEBUG(SYNC_SCHEME_DLT, "      Transpose delta value: {}", virtSobIds.mme_xpose.raw);
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
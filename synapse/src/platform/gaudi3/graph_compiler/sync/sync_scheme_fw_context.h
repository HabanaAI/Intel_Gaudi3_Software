#pragma once

#include "sync/sync_types.h"
#include "types.h"
#include "habana_graph.h"
#include "node_annotation.h"
#include "platform/gaudi3/graph_compiler/gaudi3_types.h"

namespace gaudi3
{
class SyncSchemeFwContext
{
public:
    SyncSchemeFwContext(const HabanaGraph& graph);

    // Fill the ARC's FW context sync-related fields with data provided by the sync scheme
    template<typename FwCtx> void fillArcSyncScheme(const NodePtr& node, unsigned pipelineLevel, FwCtx& ctx);

    // Copy only the sync scheme related fields from one context to another
    template<typename FwCtx> void copyArcSyncScheme(const FwCtx& fromCtx, FwCtx& toCtx);

private:
    void fillArcDeltaSyncs(unsigned                  logicalId,               // input
                           const ArcSyncInteraction& arcSyncs,                // input sync scheme
                           unsigned*                 emittedSigValIncrement,  // output
                           unsigned*                 virtSobBitmap,           // output
                           virt_sob_ids_t&           virtSobIds);             // output

    void printArcDeltaSyncs(NodePtr               node,
                            unsigned              pipelineLevel,
                            unsigned              emittedSigValIncrement,
                            unsigned              virtSobBitmap,
                            const virt_sob_ids_t& virtSobIds) const;

    void reset(unsigned logicalId);

    const HabanaGraph& m_graph;
    DependencyMap      m_arcLastDependencies[gaudi3::LOGICAL_QUEUE_MAX_ID];
    unsigned           m_arcLastEmittedSignal[gaudi3::LOGICAL_QUEUE_MAX_ID] = {0};
    unsigned           m_arcLastBreakpoint[gaudi3::LOGICAL_QUEUE_MAX_ID]    = {0};
    bool               m_xpsInUse = false;
    bool               m_mmeInUse = false;
    bool               m_xpsResetArrived = false;
    bool               m_mmeResetArrived = false;

};

// Fill the ARC's FW context sync-related fields with data provided by the sync scheme
template<typename FwCtx>
void SyncSchemeFwContext::fillArcSyncScheme(const NodePtr& node, unsigned pipelineLevel, FwCtx& ctx)
{
    HB_ASSERT(pipelineLevel < node->getNodeAnnotation().arcSyncScheme.size(), "missing arc syncs");

    ArcSyncInteraction arcSyncs  = node->getNodeAnnotation().arcSyncScheme[pipelineLevel];
    unsigned           logicalId = deviceTypeToLogicalQueue(m_graph.getNodeUtility().getNodeDeviceType(node), *node);

    unsigned emittedSigValIncrement = 0;
    unsigned virtSobBitmap          = 0;

    fillArcDeltaSyncs(logicalId, arcSyncs, &emittedSigValIncrement, &virtSobBitmap, ctx.virt_sob_ids);

    ctx.sig_inc_value      = emittedSigValIncrement;
    ctx.virtual_sob_bitmap = virtSobBitmap;

    printArcDeltaSyncs(node, pipelineLevel, ctx.sig_inc_value, ctx.virtual_sob_bitmap, ctx.virt_sob_ids);

    if (logicalId == DEVICE_XPS_LOGICAL_QUEUE) m_xpsInUse = true;
    if (logicalId == DEVICE_MME_LOGICAL_QUEUE) m_mmeInUse = true;

    // if SOBs reset is taking place at the end of this activation we need to reset the history of this logicalId
    if (arcSyncs.sobResetTotalNumEngs > 0) reset(logicalId);
}

// Copy only the sync scheme related fields from one context to another
template<typename FwCtx>
void SyncSchemeFwContext::copyArcSyncScheme(const FwCtx& fromCtx, FwCtx& toCtx)
{
    toCtx.sig_inc_value      = fromCtx.sig_inc_value;
    toCtx.virtual_sob_bitmap = fromCtx.virtual_sob_bitmap;
    toCtx.virt_sob_ids       = fromCtx.virt_sob_ids;
}

}  // namespace gaudi3
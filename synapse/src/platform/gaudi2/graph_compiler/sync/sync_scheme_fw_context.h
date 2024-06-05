#pragma once

#include "habana_nodes.h"
#include "node_utility.h"
#include "sync/sync_types.h"
#include "types.h"
#include "node_annotation.h"
#include "platform/gaudi2/graph_compiler/gaudi2_types.h"

namespace gaudi2
{
class SyncSchemeFwContext
{
public:
    SyncSchemeFwContext(const NodeUtility& nodeUtility);

    // Fill the ARC's FW context sync-related fields with data provided by the sync scheme
    template<typename FwCtx> void fillArcSyncScheme(const NodePtr& node, unsigned pipelineLevel, FwCtx& ctx);

private:
    void fillArcDeltaSyncs(unsigned                  logicalId,               // input
                           const ArcSyncInteraction& arcSyncs,                // input sync scheme
                           unsigned*                 emittedSigValIncrement,  // output
                           unsigned*                 virtSobBitmap,           // output
                           virt_sob_ids_t&           virtSobIds);             // output

    void printArcDeltaSyncs(const NodePtr&        node,
                            unsigned              pipelineLevel,
                            unsigned              emittedSigValIncrement,
                            unsigned              virtSobBitmap,
                            const virt_sob_ids_t& virtSobIds) const;

    void reset(unsigned logicalId);

    const NodeUtility& m_nodeUtility;
    DependencyMap      m_arcLastDependencies[gaudi2::LOGICAL_QUEUE_MAX_ID];
    unsigned           m_arcLastEmittedSignal[gaudi2::LOGICAL_QUEUE_MAX_ID] = {0};
    unsigned           m_arcLastBreakpoint[gaudi2::LOGICAL_QUEUE_MAX_ID]    = {0};
};

// Fill the ARC's FW context sync-related fields with data provided by the sync scheme
template<typename FwCtx>
void SyncSchemeFwContext::fillArcSyncScheme(const NodePtr& node, unsigned pipelineLevel, FwCtx& ctx)
{
    HB_ASSERT(pipelineLevel < node->getNodeAnnotation().arcSyncScheme.size(), "missing arc syncs");

    ArcSyncInteraction arcSyncs  = node->getNodeAnnotation().arcSyncScheme[pipelineLevel];
    unsigned           logicalId = deviceTypeToLogicalQueue(m_nodeUtility.getNodeDeviceType(node));

    unsigned emittedSigValIncrement = 0;
    unsigned virtSobBitmap          = 0;

    fillArcDeltaSyncs(logicalId, arcSyncs, &emittedSigValIncrement, &virtSobBitmap, ctx.virt_sob_ids);

    ctx.sig_inc_value      = emittedSigValIncrement;
    ctx.virtual_sob_bitmap = virtSobBitmap;

    printArcDeltaSyncs(node, pipelineLevel, ctx.sig_inc_value, ctx.virtual_sob_bitmap, ctx.virt_sob_ids);

    // if SOBs reset is taking place at the end of this activation we need to reset the history of this logicalId
    if (arcSyncs.sobResetTotalNumEngs > 0) reset(logicalId);
}

}
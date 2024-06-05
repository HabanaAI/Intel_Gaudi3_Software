#include "gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"

#include "platform/gaudi/graph_compiler/sync/debug_sync_scheme_manager.h"
#include "platform/gaudi/graph_compiler/sync/sync_scheme_manager.h"

namespace gaudi
{

bool allocateSyncs(GaudiGraph& g)
{
    std::shared_ptr<SyncSchemeManager> syncManager;
    if (g.isDebugMode())
    {
        syncManager.reset(new DebugSyncSchemeManagerGaudi(&g));
    }
    else
    {
        syncManager.reset(new SyncSchemeManagerGaudi(&g));
    }

    syncManager->runNodesSyncScheme();

    syncManager->printSyncScheme();

    return true;
}

}
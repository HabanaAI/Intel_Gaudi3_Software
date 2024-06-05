#include "gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "platform/gaudi3/graph_compiler/sync/sync_scheme_manager_arc.h"

namespace gaudi3
{
bool allocateSyncs(Gaudi3Graph& g)
{
    SyncSchemeManagerArcGaudi3 syncManager(&g);
    syncManager.go();
    syncManager.print();

    return true;
}

}  // namespace gaudi3

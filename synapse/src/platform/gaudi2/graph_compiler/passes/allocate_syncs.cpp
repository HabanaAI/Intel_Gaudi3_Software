#include "gaudi2_graph.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "platform/gaudi2/graph_compiler/sync/sync_scheme_manager_arc.h"

namespace gaudi2
{

bool allocateSyncs(Gaudi2Graph& g)
{
    SyncSchemeManagerArcGaudi2 syncManager(&g);
    syncManager.go();
    syncManager.print();
    return true;
}

} // namespace gaudi2
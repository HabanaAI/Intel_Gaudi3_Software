#include "gaudi3_graph.h"
#include "graph_compiler/tpc_kernel_loader.h"

namespace gaudi3
{
bool loadTpcKernels(Gaudi3Graph& g)
{
    TpcKernelLoader tpcKernelLoader(&g);
    return tpcKernelLoader.load();
}

bool allocateTpcKernels(Gaudi3Graph& g)
{
    TpcKernelLoader tpcKernelLoader(&g);
    return tpcKernelLoader.allocate();
}
}  // namespace gaudi3
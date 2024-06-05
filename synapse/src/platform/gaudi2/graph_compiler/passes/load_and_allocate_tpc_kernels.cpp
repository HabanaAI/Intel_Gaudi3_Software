#include "gaudi2_graph.h"
#include "graph_compiler/tpc_kernel_loader.h"

namespace gaudi2
{
    bool loadTpcKernels(Gaudi2Graph &g)
    {
        TpcKernelLoader tpcKernelLoader(&g);
        return tpcKernelLoader.load();
    }

    bool allocateTpcKernels(Gaudi2Graph &g)
    {
        TpcKernelLoader tpcKernelLoader(&g);
        return tpcKernelLoader.allocate(true);
    }
}
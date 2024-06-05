#include "graph_compiler/tpc_kernel_loader.h"

#include "platform/gaudi/graph_compiler/passes.h"

class GaudiGraph;

bool gaudi::loadTpcKernels(GaudiGraph& g)
{
    TpcKernelLoader tpcKernelLoader(&g);
    return tpcKernelLoader.load();
}

bool gaudi::allocateTpcKernels(GaudiGraph& g)
{
    TpcKernelLoader tpcKernelLoader(&g);
    return tpcKernelLoader.allocate();
}
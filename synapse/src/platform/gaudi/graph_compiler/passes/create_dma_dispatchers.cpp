#include "platform/gaudi/graph_compiler/passes.h"
#include "gaudi_graph.h"
#include "dma_dispatchers_creator.h"

namespace gaudi
{

bool createDMADispatchers(GaudiGraph &g)
{
    DmaDispatcherCreator creator;
    return creator.go(g);
}

} // namespace gaudi

#include <bitset>
#include "platform/gaudi2/graph_compiler/passes.h"
#include "gaudi2_graph.h"
#include "dma_dispatchers_creator.h"

namespace gaudi2
{

bool createDMADispatchers(Gaudi2Graph &g)
{
    DmaDispatcherCreator creator;
    return creator.go(g);
}

} // namespace gaudi2

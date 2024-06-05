#include <memory>

#include "passes.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "dma_inserter.h"

// Inject DMA nodes to copy from DRAM to SRAM model parameters that should be prefetched.
bool insertPrefetchDmaNodes(HabanaGraph& g)
{
    std::vector<DmaInsertionPoint> nodesToInsert;
    for (pTensor t : g.getGraphInputs())
    {
        if (t == nullptr) continue;         // no bias
        if (!t->isDenseLayout()) continue;  // Can't read/write strided tensors
        if (t->isStaticParam())
        {
            if (t->getTensorAnnotation().memorySpaceInfo.prefetchInfo.prefetch)
            {
                // if tensor should be prefetched, add another DMA node to copy from
                // dram to sram during execution
                nodesToInsert.push_back(DmaInsertionPoint(t, DMA_TYPE_PREFETCH_STATIC_TENSORS));
            }
            continue;
        }
    }
    addDmaNodes(g, nodesToInsert, false /*isSetup*/, "");
    return true;
}

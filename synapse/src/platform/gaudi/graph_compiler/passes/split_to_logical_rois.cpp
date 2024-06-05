#include "gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "roi_splitter.h"

namespace gaudi
{

    bool splitToLogicalROIs(GaudiGraph& g)
    {
        GaudiROISplitter splitter;
        return splitter.splitAllNodes(g);
    };
} //namespace gaudi

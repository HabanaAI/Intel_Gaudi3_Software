#include "gaudi2_graph.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "roi_splitter.h"

namespace gaudi2
{

    bool splitToLogicalROIs(Gaudi2Graph& g)
    {
        ROISplitter splitter;
        return splitter.splitAllNodes(g);
    };
} //namespace gaudi2

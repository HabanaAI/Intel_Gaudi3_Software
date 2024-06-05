#include "gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "roi_splitter.h"

namespace gaudi3
{
bool splitToLogicalROIs(Gaudi3Graph& g)
{
    ROISplitter splitter;
    return splitter.splitAllNodes(g);
};
}  // namespace gaudi3
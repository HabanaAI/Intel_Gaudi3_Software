#include "habana_nodes.h"
#include "habana_graph.h"
#include "habana_pass.h"


bool validateNodesLayout(HabanaGraph& g)
{
    const auto& sortedNodes = g.getExeSortedNodes();
    for (auto n : sortedNodes)
    {
        if (!n->validateNodeLayout())
        {
            LOG_ERR(DATA_LAYOUT, "Node {} has invalid layout", n->getNodeName());
            return false;
        }
    }
    return true;
}

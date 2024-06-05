#include "habana_pass.h"
#include "node_io_manager.h"
#include "habana_graph.h"
#include "graph_traits.h"

/*
 * Validate / Initialize Nodes with supported habana layouts
 *
 * if nodes are using unsupported layouts, fail compilation. if layout members are empty - initialize them properly.
 */

bool setHabanaLayouts(HabanaGraph& g)
{
    NodeVector nodes = g.getExeSortedNodes();
    for (NodePtr node : nodes)
    {
        auto&          ioManager  = node->getNodeIOManager();
        if (!ioManager.validateAndSetActualIOLayouts())
        {
            LOG_ERR(DATA_LAYOUT, "Failed setting Habana Layouts for Node: {}", node->getNodeName());
            return false;
        }
    }
    return true;
}
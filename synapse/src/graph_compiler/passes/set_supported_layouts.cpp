#include "habana_pass.h"
#include "habana_graph.h"
#include "node_io_manager.h"

bool setSupportedLayouts(HabanaGraph& g)
{
    NodeVector nodes = g.getExeSortedNodes();
    for (pNode node : nodes)
    {
        auto& nodeIOManager = node->getNodeIOManager();
        if (!nodeIOManager.setSupportedIOLayouts(g.getDeviceType()))
        {
            LOG_ERR(DATA_LAYOUT, "Setting supported IOLayouts failed. Node: {}", node->getNodeName());
            return false;
        }
    }

    return true;
}

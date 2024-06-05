#include "habana_graph.h"

static bool updateDynamicNodeInternalOperands(const pNode& node, synDeviceType deviceType)
{
    return node->inferOutputsSizes(deviceType, /*inferMax*/ false);
}

bool internalTensorsDynamicShape(HabanaGraph& graph)
{

    if (graph.getGraphAnnotation().partialGraph)
    {
        LOG_INFO(GC, "Not running shape inference for partial graphs");
        return true;
    }

    auto deviceType = graph.getDeviceType();
    for (const pNode& node : graph.getTopoSortedNodes())
    {
        if (node->isDynamicShape())
        {
            if (!updateDynamicNodeInternalOperands(node, deviceType))
            {
                LOG_ERR(GC, "Failure to update output shape for node: {}", node->getNodeName());
                return false;
            }
            if (!node->isDynamicShape())
            {
                LOG_DEBUG(GC, "Node {} went from dynamic to static post MIN inference", node->getNodeName());
            }
        }
        else if (GCFG_ENABLE_SIF_FOR_STATIC_NODES.value() == true)
        {
            return node->inferOutputsSizes(deviceType, /*inferMax*/ true);
        }
    }
    return true;
}

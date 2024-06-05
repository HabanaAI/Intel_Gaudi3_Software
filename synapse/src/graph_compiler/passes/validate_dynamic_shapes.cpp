
#include <habana_graph.h>
#include <habana_global_conf.h>

bool validateDynamicShapes(HabanaGraph& graph)
{
    if (!GCFG_GAUDI_DYNAMIC_SHAPE_VALIDATION_PASS_ENABLED.value())
    {
        return true;
    }

    auto deviceType = graph.getDeviceType();
    for (const pNode& node : graph.getExeSortedNodes())
    {
        if (node->isDynamicShape())
        {
            if (!node->validateDynamicShapes())
            {
                LOG_ERR(GC, "Failed to validate dynamic shapes for node {}", node->getNodeName());
                return false;
            }
        }
    }
    return true;
}

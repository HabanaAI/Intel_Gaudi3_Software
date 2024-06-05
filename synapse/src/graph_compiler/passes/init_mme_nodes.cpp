#include "habana_graph.h"
#include "mme_node.h"

bool InitMmeBrainIfc(HabanaGraph& g)
{
    synDeviceType deviceType = g.getDeviceType();
    for (auto& node : g.getNodes())
    {
        if (g.runsOnMME(node))
        {
            MmeNode& mmeNode = static_cast<MmeNode&>(*node);
            mmeNode.initMmeBrainIfc(deviceType);
        }
    }
    return true;
}

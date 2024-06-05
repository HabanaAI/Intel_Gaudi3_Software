#include "habana_graph.h"
#include "cast_nodes_handler.h"

bool castForTPCNodes(HabanaGraph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(GC,
                  "Inject TPC cast nodes is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  __FUNCTION__);
        return true;
    }
    bool            ret;
    const auto&     nodes = g.getNodes();
    CastNodeHandler castHandler;
    auto            device_id = g.getDeviceId();
    for (const NodePtr& n : nodes)
    {
        if (g.runsOnTPC(n))
        {
            LOG_DEBUG(GC, "{}: node {} candidate for cast injection", __FUNCTION__, n->getNodeName());
            ret = castHandler.createCastNodes(n, device_id);
            if (!ret)
            {
                LOG_ERR(GC, "{}: failed to inject cast for {} where needed", __FUNCTION__, n->getNodeName());
                return false;
            }
        }
    }
    ret = castHandler.plantCastNodes(g);
    castHandler.clear();
    return ret;
}
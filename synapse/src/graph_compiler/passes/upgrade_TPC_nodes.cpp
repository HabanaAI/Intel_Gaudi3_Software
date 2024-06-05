#include "defs.h"
#include "passes.h"
#include "habana_graph.h"
#include "tpc_node.h"


bool upgradeTPCNodesPrecision(HabanaGraph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(DATA_TYPES,
                  "Data type selection is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    const NodeSet& nodes = g.getNodes();
    for (const NodePtr& node : nodes)
    {
        if (!g.runsOnTPC(node)) continue;
        TPCNode* tpcNode = dynamic_cast<TPCNode*>(node.get());
        HB_ASSERT_PTR(tpcNode);

        if (!tpcNode->isInstantiated())
        {
            tpcNode->upgradeNodePrecisionIfMissingKernel();
        }
    }
    return true;
}
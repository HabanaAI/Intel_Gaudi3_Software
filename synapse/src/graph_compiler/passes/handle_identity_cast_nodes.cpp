
#include "habana_graph.h"
#include "passes.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "data_type_utils.h"

bool handleIdentityCastNodes(HabanaGraph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(DATA_TYPES,
                  "Data type selection is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    if (!GCFG_SYNAPSE_DATA_TYPE_SELECTION.value())
    {
        LOG_DEBUG(GC, "Data type selection is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    NodeVector nodesToRemove;
    const NodeSet& nodes = g.getNodes();

    for (const NodePtr& node : nodes)
    {
        if (!node->isCast()) continue;

        std::string_view fromDType = extractDtypeFromCastGUID(node->getGUID());
        std::string_view toDType   = extractDtypeFromGUID(node->getGUID());

        if (fromDType == toDType)
        {
            if (!isQuantDtype(fromDType))
            {
                nodesToRemove.push_back(node);
            }
        }
    }

    for (const NodePtr& node : nodesToRemove)
    {
        LOG_DEBUG(GC, "Attempting to remove redundant cast node '{}'", node->getNodeName());
        GraphEditor::removeOneToOneNode(g, node);
    }

    return true;
}

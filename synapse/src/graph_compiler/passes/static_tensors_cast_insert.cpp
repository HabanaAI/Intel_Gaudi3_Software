#include "synapse_common_types.h"
#include "types.h"

#include "cast_nodes_handler.h"
#include "habana_graph.h"
#include <algorithm>
#include <string>

bool staticTensorsCastInsert(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(CONST_FOLDING, "Not in inference mode, skipping {} pass", __FUNCTION__);
        return true;
    }

    TensorVector tensorsToCast;
    for (const TensorPtr& t : g.getTensors())
    {
        if (t->isStaticParam() && !t->isDataTypeMatchData())
        {
            tensorsToCast.push_back(t);
        }
    }

    for (const TensorPtr& t : tensorsToCast)
    {
        const NodeList consumers = g.getTensorConsumers(t);

        TensorPtr castOutput = createCastTensor(t, t->getElementType(), t->getName() + "_casted");
        t->changeDefaultElementType(t->getBufferDataType(), true);

        NodePtr castNode = CastNodeHandler::createCastNode(t, castOutput, t->getName() + "_cast", tpc_lib_api::DEVICE_ID_MAX);
        GraphEditor::addNode(g, castNode);

        for (const NodePtr& consumer : consumers)
        {
            GraphEditor::replaceTensor(g, consumer, t, castOutput);
        }
        LOG_DEBUG(CONST_FOLDING, "{}: added cast node with guid={} for static tensor {}", __FUNCTION__, castNode->getGUID(),
                  t->getName());
    }
    return true;
}
#include "habana_pass.h"
#include "habana_graph.h"

/* Find according to execution schedule the direct reduction input which is first produced by a real node*/
const TensorPtr findFirstRealReuctionInput(const HabanaGraph& g, const NodePtr& reduction)
{
    TensorPtr earliestTensor = reduction->getInput(0);
    uint32_t  minExeId       = std::numeric_limits<uint32_t>::max();
    for (const auto& directInput : reduction->getInputs())
    {
        for (const auto& nodeProducer : g.getRealProducers(directInput))
        {
            if (nodeProducer->getExecutionOrderedIndex() < minExeId)
            {
                minExeId       = nodeProducer->getExecutionOrderedIndex();
                earliestTensor = directInput;
            }
        }
    }
    return earliestTensor;
}

/* check that every consumer of the reduction input (except the reduction node)
    is executed before other reduction producers.
    otherwise, another reduction producer may write over the data before it is consumed properly
*/
void validateReductionInputConsumers(const HabanaGraph& g, const NodePtr& reduction, const TensorPtr& tensor)
{
    // if reduction is the only consumer, no need to check
    if (g.getNumberOfTensorConsumers(tensor) == 1) return;

    NodeSet realConsumers = g.getRealConsumersExcept(tensor, reduction);
    NodeSet realProducers;
    for (const TensorPtr& otherInput : reduction->getInputs())
    {
        if (!otherInput || otherInput == tensor) continue;
        NodeSet producers = g.getRealProducers(otherInput);
        realProducers.insert(producers.begin(), producers.end());
    }
    // check that every consumer of the reduction input (except the reduction node) is executed before other reduction
    // producers
    bool safeOrder = std::all_of(realProducers.begin(), realProducers.end(), [&](const NodePtr& producer) {
        return std::all_of(realConsumers.begin(), realConsumers.end(), [&](const NodePtr& consumer) {
            return producer == consumer || g.getNumberOfPaths(consumer, producer) > 0;
        });
    });
    HB_ASSERT(safeOrder,
              "Expecting reduction node {} to be the last consumer for all its inputs.",
              reduction->getNodeName());
}

bool setReductionMemset(HabanaGraph& g)
{
    std::set<NodePtr> encounteredReduces;

    for (NodePtr node : g.getExeSortedNodes())
    {
        // unmark reduction first input, so it will act as memset
        for (TensorPtr tensor : node->getOutputs())
        {
            const auto& consumers = g.getTensorConsumers(tensor);
            for (NodePtr consumer : consumers)
            {
                if (consumer->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
                {
                    validateReductionInputConsumers(g, consumer, tensor);
                    if (encounteredReduces.find(consumer) == encounteredReduces.end())
                    {
                        // 'node' is a producer for a reduction that wasn't previously encountered ('consumer')
                        // This node would set the memory value. The rest would add to it.
                        const TensorPtr& earliestTensor = findFirstRealReuctionInput(g, consumer);
                        encounteredReduces.insert(consumer);
                        earliestTensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled = false;
                    }
                }
            }
        }
    }

    return true;
}

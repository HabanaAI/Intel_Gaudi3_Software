#include "graph_annotation.h"
#include "habana_graph.h"

// Producer Node A and Consumer Node B will be called adjacent, if there is a path in the graph between Node A and Node
// B that is either direct or passes only through logical nodes.
bool areAdjacentNodes(HabanaGraph& g, const NodePtr& consumer, const NodePtr& targetProducer)
{
    const auto& directProducers = g.getNodeProducers(consumer);
    NodeVector  candidateProducers;
    std::copy(directProducers.begin(), directProducers.end(), std::back_inserter(candidateProducers));

    while (!candidateProducers.empty())
    {
        const auto& currProducer = candidateProducers.back();
        candidateProducers.pop_back();
        if (currProducer == targetProducer)
        {
            return true;
        }
        else if (currProducer->getNodeName() == targetProducer->getNodeName())
        {
            LOG_WARN(GC,
                     "Atomic node pair target and candidate node, share the same name: {}, but are not equal.",
                     targetProducer->getNodeName());
        }

        if (currProducer->isLogicalOperation())
        {
            const auto& newCandidates = g.getNodeProducers(currProducer);
            std::copy(newCandidates.begin(), newCandidates.end(), std::back_inserter(candidateProducers));
        }
    }
    return false;
}

// Atomic nodes are pairs of marked nodes that are required to be adjacent to each other when compilation is finished.
// The following function validates it after compilation is done.
// Possible hazard for not having this validation is unwanted nodes planted between extracted multinode nodes that
// operate in an atomic way. For example, check the following Jira ticket: [SW-111101]
bool validateAtomicNodes(HabanaGraph& g)
{
    const auto& atomicNodes = g.getGraphAnnotation().atomicNodes;
    for (const auto& [producer, consumer] : atomicNodes)
    {
        if (!areAdjacentNodes(g, consumer, producer))
        {
            LOG_CRITICAL(GC,
                         "Expected atomic nodes pair (producer node: {}, consumer node: {}) to be adjacent, but got "
                         "them separated",
                         producer->getNodeName(),
                         consumer->getNodeName());
            return false;
        }
    }
    return true;
}

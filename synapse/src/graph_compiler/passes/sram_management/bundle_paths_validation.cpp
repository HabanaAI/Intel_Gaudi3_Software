#include "bundle_paths_validation.h"
#include "bundle_plane_graph.h"

bool BundlePathsValidation::validateProducerPaths(const NodePtr&   candidateNode,
                                                  const TensorPtr& stitchedTensor,
                                                  const NodeSet&   acceptedNodes,
                                                  const NodeSet&   acceptedProducers)
{
    /**
    This function implements the design from SW-23360:
    3. If the candidate is a producer, then
        a. Validate no path from any accepted node to the candidate
        b. For each consumer of the stitched operand, if it is an accepted node, do
            i. Validate single path from the candidate to the consumer
            ii. For each accepted node with num of paths > 0 from the consumer, do
                1. Validate single path from the candidate to the node.
        c. Validate no path from the candidate to any accepted node not encountered in (3.b)
        d. Validate no path from the candidate to any other accepted producer.
        e. Validate single output of candidate that is consumed by the accepted nodes
    **/
    NodeSet      encounteredNodes;
    const BundlePlane* bp = m_graph.getBPGraph();

    for (const NodePtr& stitchedConsumer : m_graph.getTensorConsumers(stitchedTensor))
    {
        if (acceptedNodes.find(stitchedConsumer) == acceptedNodes.end()) continue;
        encounteredNodes.insert(stitchedConsumer);
        if (bp->getNumberOfPaths(candidateNode, stitchedConsumer) != 1) return false;  // 3bi
        for (const NodePtr& acceptedNode : acceptedNodes)                   // 3bii
        {
            if (bp->getNumberOfPaths(stitchedConsumer, acceptedNode) > 0)
            {
                encounteredNodes.insert(acceptedNode);
                if (bp->getNumberOfPaths(candidateNode, acceptedNode) != 1) return false;
            }
        }
    }
    for (const NodePtr& acceptedNode : acceptedNodes)
    {
        if (bp->getNumberOfPaths(acceptedNode, candidateNode) != 0) return false;  // 3a
        if (encounteredNodes.find(acceptedNode) == encounteredNodes.end())
        {
            if (bp->getNumberOfPaths(candidateNode, acceptedNode) != 0) return false;  // 3c
        }
    }
    for (const NodePtr& acceptedProducer : acceptedProducers)
    {
        if (bp->getNumberOfPaths(candidateNode, acceptedProducer) != 0) return false;  // 3d
    }
    for (const TensorPtr& candidateOutput : candidateNode->getOutputs())  // 3e
    {
        if (!candidateOutput || candidateOutput == stitchedTensor) continue;
        const auto& consumers = m_graph.getTensorConsumers(candidateOutput);
        if (std::any_of(consumers.begin(), consumers.end(), [&acceptedNodes](const NodePtr& n) {
                return acceptedNodes.find(n) != acceptedNodes.end();
            }))
        {
            return false;
        }
    }
    return true;
}

bool BundlePathsValidation::validateConsumerPaths(const NodePtr&   candidateNode,
                                                  const TensorPtr& stitchedTensor,
                                                  const NodeSet&   acceptedNodes)
{
    /**
    This function implements the design from SW-23360:
    4. If the candidate is a consumer
        a. Validate no path from the candidate to any accepted node
        b. If the stitched operand has an accepted producer, then
            i. Validate single path from the producer to the candidate
            ii. For each accepted node with num of paths > 0 to the producer, do
                1. Validate single path from the node to the consumer.
        c. Validate no path from any accepted node not encountered in (4.b) to the candidate
        d. Validate none of the candidate inputs != stitchedTensor are produced by accepted nodes
    **/
    NodeSet      encounteredNodes;
    const BundlePlane* bp       = m_graph.getBPGraph();
    const NodePtr      producer = m_graph.getTensorProducer(stitchedTensor);
    if (producer && (acceptedNodes.find(producer) != acceptedNodes.end()))
    {
        encounteredNodes.insert(producer);
        if (bp->getNumberOfPaths(producer, candidateNode) != 1) return false;  // 4bi
        for (const NodePtr& acceptedNode : acceptedNodes)
        {
            if (bp->getNumberOfPaths(acceptedNode, producer) > 0)  // 4bii
            {
                encounteredNodes.insert(acceptedNode);
                if (bp->getNumberOfPaths(acceptedNode, candidateNode) != 1) return false;
            }
        }
    }
    for (const NodePtr& acceptedNode : acceptedNodes)
    {
        if (bp->getNumberOfPaths(candidateNode, acceptedNode) != 0) return false;  // 4a
        if (encounteredNodes.find(acceptedNode) == encounteredNodes.end())
        {
            if (bp->getNumberOfPaths(acceptedNode, candidateNode) != 0) return false;  // 4c
        }
    }
    for (const auto& t : candidateNode->getInputs())  // 4d
    {
        if (!t || t == stitchedTensor) continue;
        if (const auto& producer = m_graph.getTensorProducer(t); producer)
        {
            if (acceptedNodes.find(producer) != acceptedNodes.end()) return false;
        }
    }
    return true;
}

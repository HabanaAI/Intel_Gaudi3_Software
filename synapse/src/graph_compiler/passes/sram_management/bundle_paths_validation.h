#pragma once

#include "habana_graph.h"

/**
    This class implements the design from SW-23360:
    3. If the candidate is a producer, then
        a. Validate no path from any accepted node to the candidate
        b. For each consumer of the stitched operand, if it is an accepeted node, do
            i. Validate single path from the candidate to the consumer
            ii. For each accepted node with num of paths > 0 from the consumer, do
                1. Validate single path from the candidate to the node.
        c. Validate no path from the candidate to any accepted node not encountered in (3.b)
        d. Validate no path from the candidate to any other accepted producer.
    4. Else // candidate is a consumer
        a. Validate no path from the candidate to any accepted node
        b. If the stitched operand has an accepted producer, then
            i. Validate single path from the producer to the candidate
            ii. For each accepted node with num of paths > 0 to the producer, do
                1. Validate single path from the node to the consumer.
        c. Validate no path from any accepted node not encountered in (4.b) to the candidate
    **/
class BundlePathsValidation
{
public:
    explicit BundlePathsValidation(const HabanaGraph& graph) : m_graph(graph) {}

    bool validateProducerPaths(const NodePtr&   candidateNode,
                               const TensorPtr& stitchedTensor,
                               const NodeSet&   acceptedNodes,
                               const NodeSet&   acceptedProducers);

    bool
    validateConsumerPaths(const NodePtr& candidateNode, const TensorPtr& stitchedTensor, const NodeSet& acceptedNodes);

private:
    const HabanaGraph& m_graph;
};
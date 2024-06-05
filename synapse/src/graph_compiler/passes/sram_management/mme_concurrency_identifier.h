#pragma once

#include "habana_graph.h"
#include "slicing_brain.h"
#include "reshape_aligner.h"

class MmeConcurrencyIdentifier
{
public:
    MmeConcurrencyIdentifier(HabanaGraph& graph) : m_graph(graph) {}

    // Main procedure
    bool scanGraph();       // Scan graph to identify concurrency. Pass runs before the Slicer

    static bool isCdConcurrencyEnabled(const HabanaGraph& g);
    static bool nodeHasCdConcurrencyAndNotFeedReduction(const HabanaGraph& g, NodePtr node);
    static bool allProducerNodesEligibleForCdConcurrency(const HabanaGraph& g, const NodePtr node);
    static void resetCdConcurrencyOfAllProducers(const HabanaGraph& g, const NodePtr node);

private:
    bool dataTypeSupportsCdConcurrency(synDataType outputDataType);
    bool dataTypeRequiresAccumulationFp32Reduction(synDataType outputDataType);
    bool nodeIsCandidateForCdConcurrency(const NodePtr& node);

    bool isMaxGemmNrSupportedForFp16() const;
    bool outputInReducibleMemory(const NodePtr& node);
    bool gaudiMmenodeIsCandidateForCdConcurrency(const NodePtr& node) const;
    void addFloat32OutputNodeAsNeeded(const NodePtr& node, unsigned outputIdx);

    HabanaGraph& m_graph;
};

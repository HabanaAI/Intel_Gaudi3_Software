#pragma once

#include "habana_graph.h"
#include "slicing_brain.h"
#include "reshape_aligner.h"

class MmeConcurrencyMemset
{
public:
    MmeConcurrencyMemset(HabanaGraph& graph) : m_graph(graph) {}

    // Main procedure
    bool addMemsetNodes();  // Add memset & reduction nodes as needed. Pass runs after the Slicer

private:
    void addMemsetNodeToReduction(const NodePtr& node);
    void addMemsetAndReductionNodes(const NodePtr node);
    void addMemsetAndReductionToOutput(const NodePtr dedwNode, unsigned outputIdx);
    void createMemsetNodeAndTensor(const TensorPtr& refTensor, NodePtr& memsetNode, TensorPtr& memsetTensor);

    HabanaGraph& m_graph;
};

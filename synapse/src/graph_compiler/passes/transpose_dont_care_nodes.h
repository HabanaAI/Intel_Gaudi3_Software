#pragma once

#include "node_layouts_handler.h"
#include "transpose_node.h"
#include "transpose_inserter.h"

class HabanaGraph;

using namespace gc;

class GraphModeTransposeDontCareNodesHandler
{
public:
    GraphModeTransposeDontCareNodesHandler(HabanaGraph& g) : m_graph(g) {}
    ~GraphModeTransposeDontCareNodesHandler() = default;

    bool singleIteration(bool forward = true);

private:
    Permutation extractPermutationsForNodeForward(const NodePtr& node) const;
    Permutation extractPermutationsForNodeBackward(const NodePtr& node) const;
    bool        extractPermutationsForNode(const NodePtr& node, bool forward, Permutation& perm) const;

    HabanaGraph& m_graph;
    NodeSet      m_wrappedNodes;
};

class EagerModeTransposeDontCareNodesHandler
{
public:
    EagerModeTransposeDontCareNodesHandler(HabanaGraph& graph, const NodePtr& node) : m_graph(graph), m_node(node) {}
    ~EagerModeTransposeDontCareNodesHandler() = default;

    bool                           canExtract();
    const TransposeNodeParamsVector& extract();
    NodePtr                          fixupNode();

protected:
    HabanaGraph&                     m_graph;
    const NodePtr&                   m_node;
    PermutationVector                m_inputTensorPermutations;
    PermutationVector                m_outputTensorPermutations;
    std::optional<TransposeInserter> m_transposeInserter;
};
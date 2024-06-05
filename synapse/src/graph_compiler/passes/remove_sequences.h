#pragma once

#include <memory>
#include <unordered_set>
#include "sequence.h"

using SequencesVector = std::vector<NodeVector>;
// Sequences.first=IdentitySequences, Sequences.second=NonIdentitySequences
using SequencesPair = std::pair<SequencesVector, SequencesVector>;

class SequenceRemover
{
public:
    SequenceRemover(HabanaGraph& g, const SequenceStrategy& strategy) : m_graph(g), m_strategy(strategy) {};
    bool removeSequences();
    bool isOutputMultipleConsumers(const TensorPtr& t) const;

private:
    SequencesPair           getRemovableSequences() const;
    const SequenceStrategy& getStrategy() const;
    void        handleIdentitySequence(const NodeVector& seq, std::unordered_set<NodePtr>& removedNodes /* output */);
    void        fuseNonIdentitySequences(const NodeVector& seq, std::unordered_set<NodePtr>& removedNodes /* output */);
    bool        isSequenceValidForRemoval(const NodeVector& seq, const std::unordered_set<NodePtr>& removedNodes) const;
    NodeVector  findRemovableSequenceNodes(const NodeVector& seq) const;
    void        disconnectIdentSeqFromGraph(const NodeVector& seq, const NodeList& toRemove);
    void        disconnectFusedSeqFromGraph(const NodeList& toRemove, const NodePtr& fusedNode);
    static void printSequences(const SequencesVector& seqList);
    static void printSequence(const NodeVector& seq);

    HabanaGraph&            m_graph;
    const SequenceStrategy& m_strategy;
};
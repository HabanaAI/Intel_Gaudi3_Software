
#include "remove_sequences.h"
#include "graph_editor.h"
#include "log_manager.h"
#include "node_factory.h"
#include "habana_graph.h"
#include "types.h"
#include <algorithm>
#include <sstream>

void SequenceRemover::printSequence(const NodeVector& seq)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(DATA_LAYOUT)) return;
    std::stringstream ss;
    for (const auto& n : seq)
    {
        ss << fmt::format("{} id {} {}", n->getNodeName(), n->getId(), (n != seq.back()) ? "->" : "");
    }
    LOG_DEBUG(DATA_LAYOUT, "{}", ss.str());
}

void SequenceRemover::printSequences(const SequencesVector& seqList)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(DATA_LAYOUT)) return;
    for (const auto& seq : seqList)
    {
        printSequence(seq);
    }
}

const SequenceStrategy& SequenceRemover::getStrategy() const
{
    return m_strategy;
}

SequencesPair SequenceRemover::getRemovableSequences() const
{
    SequencesVector allIdentitySequences {};  // List of all the identity sequences found
    SequencesVector allNonIdentFusibleSeqs {};
    std::unordered_map<NodePtr, SequencesVector>
        endNodeToSeq {};  // Maps a node to a vector of sequences that end with that node.

    for (const NodePtr& curr : m_graph.getTopoSortedNodes())
    {
        if (curr->getNumInputsDataTensors() != 1) continue;   // Only supporting nodes with one input
        if (curr->getNumOutputsDataTensors() != 1) continue;  // Only supporting nodes with one output
        // If the node is a valid sequence start (of m_strategy sequence)
        if (getStrategy().isSeqStart(curr))
        {
            // [CID: 48004] False positive - coverity ignores vecsmall Storage member initialization
            NodeVector newSeq;
            newSeq.push_back(curr);                // Create a sequence of length 1 with curr
            endNodeToSeq[curr].push_back(newSeq);  // Add newSeq to the vector of all the sequences ending with curr
            if (getStrategy().isIdentity(newSeq))
            {
                LOG_DEBUG(DATA_LAYOUT, "Sequence of single node {} is identity.", curr->getNodeName());
                allIdentitySequences.push_back(newSeq);
            }
        }
        // If the node is a valid sequence continuation (of m_strategy sequence)
        if (getStrategy().canContinueSeq(curr))
        {
            // Iterate curr's producers.
            // For a sequence ending with curr's producer, create a copy of seq, and add curr to the copy (that way
            // we keep all the subsequences of any sequence).
            // Add that new sequence to the mapping of all the sequences ending with curr.
            for (const NodePtr& prod : m_graph.getNodeProducers(curr))
            {
                const auto it = endNodeToSeq.find(prod);
                if (it != endNodeToSeq.end())
                {
                    for (const NodeVector& seq : it->second)
                    {
                        // [CID: 48004] False positive - coverity ignores vecsmall Storage member initialization
                        NodeVector seqCopy(seq);
                        seqCopy.push_back(curr);
                        if (getStrategy().isIdentity(seqCopy))
                        {
                            allIdentitySequences.push_back(seqCopy);
                        }
                        else if (getStrategy().isFusibleSequence(seqCopy) &&
                                 !isOutputMultipleConsumers(seqCopy.back()->getInput(0)))
                        {
                            allNonIdentFusibleSeqs.push_back(seqCopy);
                        }
                        endNodeToSeq[curr].push_back(std::move(seqCopy));
                    }
                }
            }
        }
    }
    if (!allIdentitySequences.empty())
    {
        LOG_DEBUG(DATA_LAYOUT, "Found identity sequences:");
        printSequences(allIdentitySequences);
    }
    return std::make_pair(allIdentitySequences, allNonIdentFusibleSeqs);
}

bool SequenceRemover::isOutputMultipleConsumers(const TensorPtr& t) const
{
    return t->isUserManagedDram() || m_graph.isOutputTensor(t) || m_graph.getNumberOfTensorConsumers(t) > 1;
}

// Disconnect the identity sequence in the following manner:
// 1. if the sequence output has to be preserved (aka is persistent or graph output):
// 1.1    if exists a sequence producer and its output is not user managed, replace its output with the sequence output
//        and remove the sequence
// 1.2    else, replace the sequence with memcopy
// 2. else
// 2.2    for every sequence consumer, replace its input with the sequence input, and remove the sequence
void SequenceRemover::disconnectIdentSeqFromGraph(const NodeVector& seq, const NodeList& toRemove)
{
    const TensorPtr seqInput  = (seq.front())->getInput(0);
    const TensorPtr seqOutput = (seq.back())->getOutput(0);
    HB_ASSERT(seqInput != nullptr && seqOutput != nullptr,
              "Currently only supporting sequences with one input and one output");

    const NodePtr  seqProducer  = m_graph.getTensorProducer(seqInput);
    const NodeList seqConsumers = m_graph.getTensorConsumers(seqOutput);

    // TODO SW-68630 - support seqInput and seqOutput in different memory allocations (sram/dram)
    if (m_graph.isOutputTensor(seqOutput) || seqOutput->isUserManagedDram())
    {
        if (seqProducer && !seqInput->isUserManagedDram())
        {
            // Update tensor attributes before deleting the old one
            seqOutput->mergeTensorsAttributes(seqInput);

            GraphEditor::removeNodes(m_graph, toRemove);
            GraphEditor::replaceTensor(m_graph, seqProducer, seqInput, seqOutput);

            // Handle the original sequence's input consumers
            for (const auto& seqInputConsumer : m_graph.getTensorConsumers(seqInput))
            {
                GraphEditor::replaceTensor(m_graph, seqInputConsumer, seqInput, seqOutput);
            }
        }
        else
        {
            LOG_DEBUG(DATA_LAYOUT, "Identity sequence has no producer, replacing it with memcpy node");
            // Replace the identity sequence with memcpy
            NodePtr memcpyNode = NodeFactory::createNode({seqInput},
                                                         {seqOutput},
                                                         nullptr,
                                                         NodeFactory::memcpyNodeTypeName,
                                                         seqInput->getName() + "_memcpy_internal");
            GraphEditor::replaceNodes(m_graph, toRemove, {memcpyNode});
        }
    }
    else
    {
        // Update tensor attributes before deleting the old one
        seqInput->mergeTensorsAttributes(seqOutput);
        GraphEditor::removeNodes(m_graph, toRemove);

        // Handle sequence consumer- replace its input with the sequence's producer's output
        for (const auto& chainConsumer : seqConsumers)
        {
            GraphEditor::replaceTensor(m_graph, chainConsumer, seqOutput, seqInput);
        }
    }
}

void SequenceRemover::disconnectFusedSeqFromGraph(const NodeList& toRemove, const NodePtr& fusedNode)
{
    LOG_DEBUG(DATA_LAYOUT,
              "{}: Replacing sequence of {} nodes with fused node {}",
              HLLOG_FUNC,
              toRemove.size(),
              fusedNode->getNodeName());
    GraphEditor::replaceNodes(m_graph, toRemove, {fusedNode});
}

// Returns true if the sequence is valid for removal:
// All the nodes in the sequence still exist (weren't removed before while handling another sequence)
bool SequenceRemover::isSequenceValidForRemoval(const NodeVector&                  seq,
                                                const std::unordered_set<NodePtr>& removedNodes) const
{
    for (const NodePtr& n : seq)
    {
        if (removedNodes.find(n) != removedNodes.end())
        {
            LOG_DEBUG(DATA_LAYOUT,
                      "Skipping identity sequence (start node: {} id {}, end node: {} id {}) - one of its' nodes was "
                      "previously removed.",
                      seq.front()->getNodeName(),
                      seq.front()->getId(),
                      seq.back()->getNodeName(),
                      seq.back()->getId());
            return false;
        }
    }
    return true;
}

NodeVector SequenceRemover::findRemovableSequenceNodes(const NodeVector& seq) const
{
    // Find the removable part of the sequence:
    // Traverse the sequence in reverse order (excluding the last node) until reaching the first node for which:
    // 1. Node has > 1 consumer OR
    // 2. Node output is isUserManagedDram OR
    // 3. Node output is a graph output
    // In addition, do not remove nodes that have complex control edges
    // [CID: 48004] False positive - coverity ignores vecsmall Storage member initialization
    NodeVector toRemove;
    for (auto it = seq.rbegin(); it != seq.rend(); ++it)
    {
        const NodePtr& curr    = *it;
        TensorPtr      currOut = curr->getOutput(0);

        if (it != seq.rbegin() && isOutputMultipleConsumers(currOut))
        {
            LOG_DEBUG(DATA_LAYOUT, "cannot remove candidate {} due to multiple consumers", curr->getNodeName());
            break;
        }
        if (!GraphEditor::canRemoveNodeControl(m_graph, curr))
        {
            LOG_DEBUG(DATA_LAYOUT, "cannot remove candidate {} due to complex control flow", curr->getNodeName());
            break;
        }
        toRemove.push_back(*it);
    }
    return toRemove;
}

void SequenceRemover::handleIdentitySequence(const NodeVector&            seq,
                                             std::unordered_set<NodePtr>& removedNodes /* output */)
{
    HB_ASSERT(getStrategy().isIdentity(seq), "Unknown state - handling only identity sequences.");
    HB_ASSERT(seq.size() > 0, "Sequence must have at least one node.");

    if (!isSequenceValidForRemoval(seq, removedNodes))
    {
        return;
    }

    NodeVector toRemove(findRemovableSequenceNodes(seq));
    if (toRemove.empty()) return;

    LOG_DEBUG(DATA_LAYOUT, "Removing identity sequence:");
    printSequence(toRemove);

    NodeList removeList(toRemove.begin(), toRemove.end());
    disconnectIdentSeqFromGraph(seq, removeList);
    removedNodes.insert(toRemove.begin(), toRemove.end());
}

void SequenceRemover::fuseNonIdentitySequences(const NodeVector&            seq,
                                               std::unordered_set<NodePtr>& removedNodes /* output */)
{
    HB_ASSERT(!getStrategy().isIdentity(seq), "Unknown state - fusing is supported only for non-identity sequences");
    HB_ASSERT(seq.size() > 1, "Sequence must have at least two nodes");

    if (!isSequenceValidForRemoval(seq, removedNodes))
    {
        return;
    }

    // verify that all of the sequence can be fused
    if (findRemovableSequenceNodes(seq).size() != seq.size()) return;

    const auto fusedNode = getStrategy().fuseSequence(seq);
    if (fusedNode.has_value())
    {
        LOG_DEBUG(DATA_LAYOUT, "Removing non-identity sequence:");
        printSequence(seq);
        LOG_DEBUG(DATA_LAYOUT, "New fused node: {}", fusedNode.value()->getNodeName());
        NodeList removeList(seq.begin(), seq.end());

        disconnectFusedSeqFromGraph(removeList, fusedNode.value());
        removedNodes.insert(seq.begin(), seq.end());
    }
}

bool SequenceRemover::removeSequences()
{
    auto sequences = getRemovableSequences();

    auto& identitySequences           = sequences.first;
    auto& nonIdentityFusibleSequences = sequences.second;

    // Stabilly sort sequences from longest to shortest
    std::stable_sort(identitySequences.begin(),
                     identitySequences.end(),
                     [](const NodeVector& seq1, const NodeVector& seq2) { return seq1.size() >= seq2.size(); });
    std::stable_sort(nonIdentityFusibleSequences.begin(),
                     nonIdentityFusibleSequences.end(),
                     [](const NodeVector& seq1, const NodeVector& seq2) { return seq1.size() >= seq2.size(); });
    std::unordered_set<NodePtr> removedNodes {};
    for (const auto& seq : identitySequences)
    {
        handleIdentitySequence(seq, removedNodes);
    }

    for (const auto& seq : nonIdentityFusibleSequences)
    {
        fuseNonIdentitySequences(seq, removedNodes);
    }

    return true;
}
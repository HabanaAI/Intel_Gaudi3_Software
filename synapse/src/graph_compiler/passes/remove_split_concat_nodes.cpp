#include "habana_graph.h"
#include "habana_pass.h"
#include "node_factory.h"

static inline unsigned getAggregationDim(const NodePtr& node)
{
    const auto& aggNode = std::dynamic_pointer_cast<AggregationNode>(node);
    return aggNode ? aggNode->getAggregationDim() : 0 /* FcdAggregationNode */;
}

static bool tryToRemoveOppositeConcatSplitSequence(HabanaGraph& g, const NodePtr& concatNode, const NodePtr& splitNode)
{
    const unsigned concatAggDim = getAggregationDim(concatNode);
    const unsigned splitAggDim  = getAggregationDim(splitNode);
    if (concatAggDim != splitAggDim) return false;  // split and concat are not opposite
    const unsigned& aggDim = concatAggDim;

    const auto& concatInputs = concatNode->getInputs();
    const auto& splitOutputs = splitNode->getOutputs();
    if (concatInputs.size() != splitOutputs.size()) return false;  // split and concat are not opposite

    for (unsigned i = 0; i < concatInputs.size(); ++i)
    {
        const auto& input  = concatInputs.at(i);
        const auto& output = splitOutputs.at(i);

        // If exists at least one mismatch in concat input and split output, do nothing and return
        if (input->getSizeInElements(aggDim) != output->getSizeInElements(aggDim)) return false;
    }

    // If we reach this line it mean that the split and concat are opposite
    GraphEditor::removeNodes<NodeVector>(g, {splitNode, concatNode});

    for (unsigned i = 0; i < concatInputs.size(); ++i)
    {
        const auto& input  = concatInputs.at(i);
        const auto& output = splitOutputs.at(i);

        // If both input and output are not persistent we just need one of them,
        // it needed for multi-depth opposite split-concat.
        if (!g.isUserManagedDram(input) && !g.isUserManagedDram(output))
        {
            GraphEditor::replaceTensor(g, output, input);
        }
        else  // We need to insert identity node
        {
            auto identityNode = NodeFactory::createNode({input},
                                                        {output},
                                                        nullptr,
                                                        NodeFactory::identityNodeTypeName,
                                                        input->getName() + "_to_" + output->getName());
            auto res          = GraphEditor::addNode(g, identityNode);
            HB_ASSERT(res, "Failed to add identity node");
        }
    }
    return true;
}

static inline bool isConcat(const NodePtr& n)
{
    return n && n->getNodeType() == Node::TYPE_INTERNAL_CONCAT;
}

static inline bool isSplit(const NodePtr& n)
{
    return n && n->getNodeType() == Node::TYPE_INTERNAL_SPLIT;
}

static uint64_t removeOppositeConcatSplit(HabanaGraph& g)
{
    uint64_t   removed = 0;
    NodeVector splitNodes;
    {  // Since "nodes" is invalidate when the graph is modified, we scope it and copy the split nodes
        const auto& nodes = g.getTopoSortedNodes();
        std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(splitNodes), isSplit);
    }

    for (const auto& splitNode : splitNodes)
    {
        const auto& sharedTensor = splitNode->getInput(0);
        if (g.isUserManagedDram(sharedTensor)) continue;  // The shared tensor is persistent, nothing to do

        const auto& concatNode = g.getTensorProducer(sharedTensor);
        if (!isConcat(concatNode)) continue;  // The producer of the input is not concat

        removed += tryToRemoveOppositeConcatSplitSequence(g, concatNode, splitNode) ? 1 : 0;
    }
    return removed;
}

// Remove concat -> split nodes if they are opposite. (but not opposite split -> concat)
bool removeOppositeConcatSplitSequence(HabanaGraph& g)
{
    if (!GCFG_ENABLE_OPPOSITE_CONCAT_SPLIT_REMOVAL.value())
    {
        return true;
    }
    auto n = removeOppositeConcatSplit(g);
    LOG_INFO(GC, "{} pairs of concat and split nodes removed", n);
    return true;
}

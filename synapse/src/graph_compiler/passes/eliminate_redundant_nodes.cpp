#include "graph_editor.h"
#include "habana_graph.h"

#include "types.h"

#include <cstddef>

static bool shouldEliminate(const HabanaGraph& g, const Node& node)
{
    // Since the nodes are removed as we go in reverse topological order,
    // All direct and future consumers which could be removed are gone by now
    // And it's enough to check if there's any non-removed consumers left.
    if (g.hasConsumer(node)) return false;

    // Check for persistent tensors or tensors that are part of a RMW section,
    // So their data may still be consumed by other nodes.
    for (const TensorPtr& tensor : node.getOutputs())
    {
        if (tensor == nullptr) continue;
        if (tensor->isPersistent())
        {
            LOG_TRACE(GC,
                      "Cannot eliminate node: {}, due to persistent output tensor: {}",
                      node.getNodeName(),
                      tensor->getName());
            return false;
        }
        if (tensor->isPartOfRMWSection())
        {
            LOG_TRACE(GC,
                      "Cannot eliminate node: {}, due to tensor in RMW section: {}",
                      node.getNodeName(),
                      tensor->getName());
            return false;
        }
    }
    return true;
}

/*
 * This pass is designed for "dead code elimination" - remove nodes that are not being used in data paths consumed by
 * the user, i.e. nodes without consumers or persistent output tensors.
 */
bool eliminateRedundantNodes(HabanaGraph& g)
{
    // creating a copy since the graph is modified in the loop
    const NodeVector graphNodes = g.getTopoSortedNodes();

    // go over all nodes in reversed topological order
    size_t nodesLeft = graphNodes.size();
    for (auto nodeIter = graphNodes.rbegin(); nodeIter != graphNodes.rend(); ++nodeIter)
    {
        if (nodesLeft < 2) break;  // avoid leaving an empty graph

        const NodePtr& node = *nodeIter;
        if (shouldEliminate(g, *node))
        {
            LOG_DEBUG(GC, "Eliminate redundant node: {}", node->getNodeName());
            GraphEditor::removeNode(g, node);
            --nodesLeft;
        }
    }

    LOG_DEBUG(GC,
              "Number of nodes without consumers and non-persistent outputs eliminated - {}",
              graphNodes.size() - nodesLeft);
    return true;
}
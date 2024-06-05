#include "habana_graph.h"
#include "habana_nodes.h"
#include "node_factory.h"

#include "habana_global_conf.h"
#include "habana_pass.h"
#include "graph_visualization.h"
#include "passes.h"
#include "graph_editor.h"

bool removeRedundantLogicalNodes(HabanaGraph& g)
{
    auto fn = [](const NodePtr& node){return node->isLogicalOperation();};
    NodeVector sortedNodes = g.getTopoSortedNodesCond(fn);

    for (auto n : sortedNodes)
    {
        if (!n->isDebug())
        {
            std::shared_ptr<LogicalOpNode> logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(n);
            HB_ASSERT_PTR(logicalNode);

             // don't remove logical operations that are counted for and already ran.
            if (logicalNode->getRunLogicalOperationDone()) continue;
            // Removing logical nodes that have no actual action, and may only cause memcpy node planting
            // Also check that graph has more than one node
            //
            // Also check that the node has real producers or consumers.
            if (logicalNode->isRedundantNode() && g.getNumNodes() > 1)
            {
                if (g.getNodeRealProducers(n, Node::TENSOR_TYPE_DATA).empty() &&
                    g.getNodeRealConsumers(n, Node::TENSOR_TYPE_DATA).empty())
                {
                    LOG_DEBUG(GC, "Skipping isolated logical node '{}'", n->getNodeName());
                    continue;
                }

                LOG_DEBUG(GC, "Attempting to remove redundant logical node '{}'", n->getNodeName());
                GraphEditor::removeOneToOneNode(g, n);

                if (!g.containsNode(n))  // if node successfully removed, remove it from atomic nodes vector
                {
                    g.getGraphAnnotation().removeAtomicNode(n);
                }
            }
        }
    }

    return true;
}

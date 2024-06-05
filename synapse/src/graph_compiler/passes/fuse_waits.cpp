#include "habana_graph.h"
#include "node_factory.h"
#include "graph_editor.h"

static void addDelayToNode(pNode node, unsigned waitCycles)
{
    LOG_DEBUG(GC, "adding {} cycles delay to node {}", waitCycles, node->getNodeName());
    node->getNodeAnnotation().waitCycles = waitCycles;
}

bool fuseWaits(HabanaGraph& g)
{
    NodeVector allNodes = g.getExeSortedNodes();
    NodeList   nodesToRemove;

    for (pNode n : allNodes)
    {
        if (!n->isWait()) continue;
        std::shared_ptr<WaitNode> waitNode = std::dynamic_pointer_cast<WaitNode>(n);

        NodeSet suspendedNodes = g.getBlockedNodes(n);
        HB_ASSERT(suspendedNodes.size(), "Wait node must be connected with ctrl tensor to at least one node");

        bool bDelayAdded = false;

        for (pNode suspendedNode : suspendedNodes)
        {
            if (suspendedNode->isLogicalOperation())
            {
                continue;
            }

            if (suspendedNode->getNodeAnnotation().waitCycles)
            {
                LOG_ERR(GC, "suspendedNode {} is already delayed", suspendedNode->getNodeName());
                return false;
            }

            addDelayToNode(suspendedNode, waitNode->getWaitCycles());
            bDelayAdded = true;
        }

        if (!bDelayAdded)
        {
            LOG_WARN(GC, "waitNode {} did not add delay to any physical node", waitNode->getNodeName());
        }

        g.removeNodeControlDependencies(waitNode);
        nodesToRemove.push_back(n);
    }

    GraphEditor::removeNodes(g, nodesToRemove);

    return true;
}
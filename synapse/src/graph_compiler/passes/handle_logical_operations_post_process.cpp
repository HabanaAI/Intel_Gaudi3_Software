#include "habana_graph.h"
#include "node.h"
#include "operand_reuse_logical_node.h"

// restore the original nodes
void restoreInputReuseNodes(HabanaGraph& g)
{
    const auto& nodes = g.getNodes();
    NodeVector  operandReuseNodes;
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(operandReuseNodes), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_OPERAND_REUSE_INTERNAL;
    });

    if (!operandReuseNodes.empty())
    {
        LOG_DEBUG(OPT_LOGICAL_OPS, "start to restore input reuse nodes");
        for (const auto& node : operandReuseNodes)
        {
            const auto& operandReuseNode = std::dynamic_pointer_cast<OperandReuseInternalLogicalNode>(node);
            HB_ASSERT_PTR(operandReuseNode);

            const auto& originalNode = operandReuseNode->getOriginalNode();
            originalNode->cloneConnectivityFromNode(*operandReuseNode);

            auto res = GraphEditor::replaceNodes(g, {operandReuseNode}, {originalNode});
            HB_ASSERT(res == REPLACE_NODE_SUCCESS, "Failed to replace node");

            LOG_TRACE(OPT_LOGICAL_OPS, "{} restored succesfuly", originalNode->getNodeName());
        }
    }
}

bool handleLogicalOpsPostProcess(HabanaGraph& g)
{
    restoreInputReuseNodes(g);
    return true;
}
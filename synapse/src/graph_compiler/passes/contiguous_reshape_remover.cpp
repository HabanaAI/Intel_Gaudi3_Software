#include "contiguous_reshape_remover.h"
#include "logical_op_node.h"
#include "graph_editor.h"
#include "habana_nodes.h"

void ContiguousReshapeRemover::removeContiguousReshapesForNode(pNode node)
{
    if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
    {
        //Don't fuse reshapes with shape tensor input
        if (node->isDynamicShape()) return;

        for (const auto& consumer : m_graph.getTensorConsumers(node->getOutput(0)))
        {
            fuseProducerAndConsumerReshape(node, consumer);
        }
    }
}
void ContiguousReshapeRemover::removeContiguousReshapesForGraph()
{
    NodeSet allNodes = m_graph.getNodes();
    for (auto node : allNodes)
    {
        removeContiguousReshapesForNode(node);
    }
}

bool ContiguousReshapeRemover::fuseProducerAndConsumerReshape(pNode producer, pNode consumer)
{
    if (m_graph.getTensorProducer(consumer->getInput(0)) != producer)
    {
        return false;
    }
    if (producer->getNodeType() != Node::TYPE_INTERNAL_RESHAPE ||
        consumer->getNodeType() != Node::TYPE_INTERNAL_RESHAPE)
    {
        return false;
    }
    bool userManagedDram = m_graph.isUserManagedDram(producer->getOutput(0));
    std::static_pointer_cast<LogicalOpNode>(consumer)->resetLogicalOp();
    std::static_pointer_cast<LogicalOpNode>(producer)->resetLogicalOp();
    GraphEditor::replaceInput(m_graph, consumer, 0, producer->getInput(0));

    if (m_graph.getNumberOfTensorConsumers(producer->getOutput(0)) == 0 && !userManagedDram)
    {
        GraphEditor::removeNode(m_graph, producer);
    }
    return true;
}

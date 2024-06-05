#include <graph_compiler/node_predicate_handler.h>
#include "habana_graph.h"
#include "utils.h"

NodePredicateHandler::NodePredicateHandler(HabanaGraph& graph)
: m_graph(graph)
{
}

void NodePredicateHandler::visit(Node* node)
{
    m_graph.turnOnPredicate(PREDICATE_ID_NODE_CREATED);
}

void NodePredicateHandler::visit(LogicalOpNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_LOGICAL_NODE_CREATED);
}

void NodePredicateHandler::visit(TPCNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_TPC_NODE_CREATED);
    if (node->isCast())
    {
        m_graph.turnOnPredicate(PREDICATE_ID_CAST_NODE_CREATED);
    }
    else if (isMemcpy(*node))
    {
        m_graph.turnOnPredicate(PREDICATE_ID_TPC_MEMCPY_NODE_CREATED);
    }
}

void NodePredicateHandler::visit(ReshapeNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_RESHAPE_NODE_CREATED);
}

void NodePredicateHandler::visit(DMANode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_DMA_NODE_CREATED);
}

void NodePredicateHandler::visit(TransposeNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED);
}

void NodePredicateHandler::visit(MemcpyNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_MEMCPY_NODE_CREATED);
}

void NodePredicateHandler::visit(MemsetNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_MEMCPY_NODE_CREATED);
}

void NodePredicateHandler::visit(ReductionNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_REDUCTION_NODE_CREATED);
}
void NodePredicateHandler::visit(BroadcastNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_BROADCAST_NODE_CREATED);
}
void NodePredicateHandler::visit(MmeNode* node)
{
    NodeVisitor::visit(node);
    m_graph.turnOnPredicate(PREDICATE_ID_MME_NODE_CREATED);
}
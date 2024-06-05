#pragma once

#include "habana_nodes/node_visitor.h"

class HabanaGraph;

class NodePredicateHandler : public NodeVisitor
{
public:
    NodePredicateHandler(HabanaGraph& graph);

    virtual void visit(Node* node);
    virtual void visit(LogicalOpNode* node);
    virtual void visit(TPCNode* node);
    virtual void visit(ReshapeNode* node);
    virtual void visit(DMANode* node);
    virtual void visit(TransposeNode* node);
    virtual void visit(MemcpyNode* node);
    virtual void visit(MemsetNode* node);
    virtual void visit(ReductionNode* node);
    virtual void visit(BroadcastNode* node);
    virtual void visit(MmeNode* node);

private:
    HabanaGraph& m_graph;
};

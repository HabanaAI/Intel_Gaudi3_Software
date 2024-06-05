#pragma once

#include "types.h"
#include "node_visitor.h"

class HabanaGraph;

/**
 * Calculate node pipeline depth
 * Assume pipeline depth going forward (producer dictate consumer pipeline depth)
 */
class NodePipelineDepth: public NodeVisitor
{
public:
    NodePipelineDepth(const HabanaGraph& graph);

    virtual void visit(Node* node);

    virtual void visit(LogicalOpNode* node);

    virtual void visit(ConcatenateNode* node);

    virtual void visit(SplitNode* node);

    uint32_t m_pipelineDepth;

private:
    const HabanaGraph& m_graph;
};


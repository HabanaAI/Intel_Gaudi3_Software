#pragma once

#include "node_visitor.h"

class DMANode;
class LogicalOpNode;
class MmeNode;
class Node;
class TPCNode;

namespace gc
{

/**
 * Base class
 * Select engine executer for each node
 */
class EngineSelector : public NodeVisitor
{
public:
    enum class EngineType
    {
        ENGINE_TPC,
        ENGINE_MME,
        ENGINE_DMA,
        ENGINE_ROTATOR,
        ENGINE_LOGICAL,

        ENGINE_UNKNOWN
    };

    EngineSelector();

    virtual ~EngineSelector();

    EngineType getSelectedEngine(Node& node);

protected:

    virtual void visit(MmeNode* node) override;

    virtual void visit(TPCNode* node) override;

    virtual void visit(DMANode* node) override;

    virtual void visit(RotateNode* node) override;

    virtual void visit(LogicalOpNode* node) override;

    EngineType m_selectedEngine;
};

}

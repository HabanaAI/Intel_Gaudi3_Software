#include "engine_selector.h"
#include "node.h"

class DMANode;
class LogicalOpNode;
class MmeNode;
class TPCNode;

namespace gc
{

EngineSelector::EngineSelector()
: m_selectedEngine(EngineType::ENGINE_UNKNOWN)
{
}

EngineSelector::~EngineSelector()
{
}

EngineSelector::EngineType EngineSelector::getSelectedEngine(Node& node)
{
    node.accept(this);

    return m_selectedEngine;
}

void EngineSelector::visit(MmeNode* node)
{
    m_selectedEngine = EngineType::ENGINE_MME;
}

void EngineSelector::visit(TPCNode* node)
{
    m_selectedEngine = EngineType::ENGINE_TPC;
}

void EngineSelector::visit(DMANode* node)
{
    m_selectedEngine = EngineType::ENGINE_DMA;
}

void EngineSelector::visit(RotateNode* node)
{
    m_selectedEngine = EngineType::ENGINE_ROTATOR;
}

void EngineSelector::visit(LogicalOpNode* node)
{
    m_selectedEngine = EngineType::ENGINE_LOGICAL;
}

}


#include "transpose_node.h"
#include "engine_selector.h"

namespace gaudi
{

void EngineSelector::visit(TransposeNode* node)
{
    if (node->permutation()[0] == TPD_Channel)
    {
        m_selectedEngine = EngineType::ENGINE_LOGICAL;
    }
    else
    {
        // Currently TPC engine
        // TODO: change to DMA [SW-10275]
        m_selectedEngine = EngineType::ENGINE_TPC;
    }
}

} /* namespace gaudi */

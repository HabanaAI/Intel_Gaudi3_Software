#include "spill_fill_manager.h"
#include "compilation_hal_reader.h"
#include "operation_slice.h"
#include "pipeline_management/fusion_handlers_factory.h"

std::vector<FusionHandlersFactoryPtr> SpillFillManager::getFusionHandlersFactories()
{
    return {m_tpcFusionHandlerFactory};
}

HandlersVector SpillFillManager::getHandlers(const NodePtr& directive)
{
    return FusionCandidatesSelector::selectHandlers(m_graph,
                                                    directive,
                                                    m_db.getCandidatesForDirective(directive),
                                                    getFusionHandlersFactories());
}

void SpillFillManager::fuse(const NodePtr& directive)
{
    for (auto& handler : getHandlers(directive))
    {
        if (handler->fuse(directive))
        {
            m_db.updateFused(directive, handler->getCandidate());
            break;
        }
    }
}

bool SpillFillManager::fuseAllSpillFillDirectives()
{
    for (const auto& directive : m_db.getDirectives())
    {
        fuse(directive);
    }
    // TODO SW-106239: enable fill fusion
    return true;
}

bool fuseSpillFillDirectives(HabanaGraph& g)
{
    if (!GCFG_ENABLE_SPILL_FILL_FUSION.value())
    {
        return true;
    }

    return SpillFillManager(g).fuseAllSpillFillDirectives();
}
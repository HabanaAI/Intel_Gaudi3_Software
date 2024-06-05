#pragma once

#include "habana_graph.h"
#include "fusion_db.h"
#include "fusion_candidates_selector.h"

class SpillFillManager
{
public:
    explicit SpillFillManager(HabanaGraph& g) : m_graph(g), m_db(g), m_tpcFusionHandlerFactory(new TpcFusionHandlersFactory()) {};
    bool fuseAllSpillFillDirectives();

private:
    void                     fuse(const NodePtr& directive);
    HandlersVector           getHandlers(const NodePtr& directive);
    std::vector<FusionHandlersFactoryPtr> getFusionHandlersFactories();

    HabanaGraph& m_graph;
    FusionCandidatesDb m_db;
    TpcFusionHandlersFactoryPtr m_tpcFusionHandlerFactory;
};
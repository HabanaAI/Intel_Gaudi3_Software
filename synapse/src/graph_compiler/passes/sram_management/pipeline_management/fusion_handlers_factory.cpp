#include "fusion_handlers_factory.h"

FusionHandlerPtr TpcFusionHandlersFactory::createForCandidate(HabanaGraph &g, const NodePtr &directive, const CandidateInfo &candidate)
{
    if (TpcFusionHandler::isValidForFusion(g, directive, candidate))
    {
        return std::make_shared<TpcFusionHandler>(g, candidate, m_tpcFusionDb);
    }
    return nullptr;
}

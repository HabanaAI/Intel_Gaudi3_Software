#include "fusion_candidates_selector.h"

HandlersVector FusionCandidatesSelector::selectHandlers(HabanaGraph&                                 g,
                                                        const NodePtr&                               directive,
                                                        const CandidatesInfoSet&                     candidates,
                                                        const std::vector<FusionHandlersFactoryPtr>& handlersFactory)
{
    HandlersVector handlers;
    for (const CandidateInfo& candidate : candidates)
    {
        for (const auto& factory : handlersFactory)
        {
            if (auto handler = factory->createForCandidate(g, directive, candidate))
            {
                handlers.push_back(handler);
            }
        }
    }
    return handlers;
}

#pragma once

#include "pipeline_management/fusion_handlers_factory.h"

using HandlersVector = std::vector<FusionHandlerPtr>;
class FusionCandidatesSelector
{
public:
    static HandlersVector selectHandlers(HabanaGraph&                                 g,
                                         const NodePtr&                               directive,
                                         const CandidatesInfoSet&                     candidates,
                                         const std::vector<FusionHandlersFactoryPtr>& handlersFactory);
};
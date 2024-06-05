#pragma once

#include "pipeline_management/fusion_candidates_collector.h"
#include "pipeline_management/fusion_handlers.h"

class FusionHandlersFactory
{
public:
    virtual FusionHandlerPtr createForCandidate(HabanaGraph& g, const NodePtr& directive, const CandidateInfo& candidate) = 0;
};


class TpcFusionHandlersFactory : public FusionHandlersFactory
{
public:
    TpcFusionHandlersFactory() {};
    FusionHandlerPtr createForCandidate(HabanaGraph& g, const NodePtr& directive, const CandidateInfo& candidate) override;

private:
    TpcRecompileDb m_tpcFusionDb;
};

using FusionHandlersFactoryPtr = std::shared_ptr<FusionHandlersFactory>;
using TpcFusionHandlersFactoryPtr = std::shared_ptr<FusionHandlersFactory>;

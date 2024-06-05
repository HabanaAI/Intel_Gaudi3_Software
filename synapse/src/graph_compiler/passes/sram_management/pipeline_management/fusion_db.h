#pragma once

#include "habana_graph.h"
#include "pipeline_management/fusion_candidates_collector.h"
#include "types.h"

class FusionCandidatesDb
{
public:
    FusionCandidatesDb() {};
    FusionCandidatesDb(const HabanaGraph& g);
    NodeVector        getDirectives() const;
    void              updateFused(const NodePtr& directive, const CandidateInfo& chosenCandidate);
    CandidatesInfoSet getCandidatesForDirective(const NodePtr& directive) const;

private:
    void addToDb(const NodePtr& directive, const CandidatesInfoSet& candidates);

    std::map<NodePtr, CandidatesInfoSet>               m_directiveToCandidates;
    std::map<CandidateInfo, NodeSet, CandidateInfoCmp> m_candidateToDirectives;
};
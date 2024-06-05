#include "fusion_db.h"
#include "defs.h"
#include "habana_graph.h"
#include "spill_fill_classifier.h"
#include "fusion_candidates_collector.h"
#include "types.h"

FusionCandidatesDb::FusionCandidatesDb(const HabanaGraph& g)
{
    SpillFillClassifier classifier(g);
    FusionCandidatesCollector collector;

    for (const auto& spill : classifier.getSpillDirectives())
    {
        const auto& candidates = collector.getSpillFusionCandidates(g, spill);
        addToDb(spill, CandidatesInfoSet(candidates.begin(), candidates.end()));
    }
}

void FusionCandidatesDb::addToDb(const NodePtr& directive, const CandidatesInfoSet& candidates)
{
    m_directiveToCandidates[directive] = candidates;
    for (const auto& candidate : candidates)
    {
        m_candidateToDirectives[candidate].insert(directive);
    }
}

NodeVector FusionCandidatesDb::getDirectives() const
{
    NodeVector directives;
    for (const auto& dirAndCand : m_directiveToCandidates)
    {
        directives.push_back(dirAndCand.first);
    }
    return directives;
}

CandidatesInfoSet FusionCandidatesDb::getCandidatesForDirective(const NodePtr& directive) const
{
    HB_ASSERT(m_directiveToCandidates.find(directive) != m_directiveToCandidates.end(), "Directive {} does not exist in the db", directive->getNodeName());
    return m_directiveToCandidates.at(directive);
}

void FusionCandidatesDb::updateFused(const NodePtr& directive, const CandidateInfo& chosenCandidate)
{
    HB_ASSERT(m_directiveToCandidates.find(directive) != m_directiveToCandidates.end(), "Directive {} does not exist in the db", directive->getNodeName());
    CandidatesInfoSet candidates = m_directiveToCandidates[directive];
    for (const auto& currDirective : m_candidateToDirectives[chosenCandidate])
    {
        HB_ASSERT(m_directiveToCandidates.find(currDirective) != m_directiveToCandidates.end(), "Directive {} does not exist in the db", directive->getNodeName());
        if (currDirective == directive)
        {
            m_directiveToCandidates[directive].clear();
            m_directiveToCandidates[directive].insert(chosenCandidate);
            continue;
        }
        m_directiveToCandidates[currDirective].erase(chosenCandidate);
    }
    for (const auto& candidate : candidates)
    {
        HB_ASSERT(m_candidateToDirectives.find(candidate) != m_candidateToDirectives.end(), "Candidate {} does not exist in the db", candidate.getNode()->getNodeName());
        if (candidate == chosenCandidate)
        {
            m_candidateToDirectives[candidate].clear();
            m_candidateToDirectives[candidate].insert(directive);
            continue;
        }
        m_candidateToDirectives[candidate].erase(directive);
    }
}
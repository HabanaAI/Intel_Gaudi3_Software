#include "bundle_candidate_finders.h"
#include "habana_graph.h"

namespace gc::layered_brain::bundler
{
DfsFinder::DfsFinder(const TensorPtr& operand, const HabanaGraph& graph) : CandidateFinder(graph)
{
    HB_ASSERT_PTR(operand);
}

std::optional<std::pair<ExpansionCandidate, ExpansionCandidate>> DfsFinder::next()
{
    SET_TEMP_LOG_CONTEXT("DfsFinder");
    while (!m_stack.empty())
    {
        auto& [link, candidates] = m_stack.top();
        while (!candidates.empty())
        {
            const auto candidate = candidates.front();
            candidates.pop();
            push(candidate);
            LOG_DEBUG(LB_BUNDLER, "next candidate: {}[{}]", candidate->getNodeName(), candidate->getNodeTypeStr());
            return std::make_pair(link, candidate);
        }
        m_stack.pop();
    }
    return std::nullopt;
}

void DfsFinder::rejectCandidate()
{
    if (m_stack.empty()) return;
    m_stack.pop();
}

DfsProducersFinder::DfsProducersFinder(const TensorPtr& operand, const HabanaGraph& graph) : DfsFinder(operand, graph)
{
    const auto prod = m_graph.getTensorProducer(operand);
    if (prod)
    {
        NodeQueue q;
        q.push(prod);
        m_stack.push({nullptr, q});
    }
}

void DfsProducersFinder::push(const NodePtr& prod)
{
    HB_ASSERT_PTR(prod);
    NodeQueue candidates;
    for (const auto& producer : m_graph.getNodeProducers(prod))
    {
        if (!producer) continue;
        candidates.push(producer);
    }
    m_stack.push(std::make_pair(prod, candidates));
}

void DfsConsumersFinder::push(const NodePtr& n)
{
    HB_ASSERT_PTR(n);
    NodeQueue candidates;
    for (const auto& consumer : m_graph.getNodeConsumers(n))
    {
        if (!consumer) continue;
        candidates.push(consumer);
    }
    m_stack.push(std::make_pair(n, candidates));
}

DfsConsumersFinder::DfsConsumersFinder(const TensorPtr& operand, const HabanaGraph& graph) : DfsFinder(operand, graph)
{
    NodeQueue  candidates;
    const auto consumers = m_graph.getTensorConsumers(operand);
    for (const auto& n : consumers)
    {
        if (!n) continue;
        candidates.push(n);
    }
    m_stack.push({nullptr, candidates});
}

}  // namespace gc::layered_brain::bundler
#pragma once

#include <stack>
#include <queue>
#include "bundle_candidate_finder.h"
#include "bundle_break_rules.h"

namespace gc::layered_brain::bundler
{
class DfsFinder : public CandidateFinder
{
public:
    DfsFinder(const TensorPtr& operand, const HabanaGraph& graph);
    std::optional<std::pair<ExpansionCandidate, ExpansionCandidate>> next() override;

    void rejectCandidate() override;

protected:
    using NodeQueue                             = std::queue<NodePtr>;
    virtual void push(const NodePtr& candidate) = 0;

    std::stack<std::pair<NodePtr, NodeQueue>> m_stack;
};

class DfsProducersFinder : public DfsFinder
{
public:
    DfsProducersFinder(const TensorPtr& operand, const HabanaGraph& graph);

private:
    void push(const NodePtr& candidate) override;
};

class DfsConsumersFinder : public DfsFinder
{
public:
    DfsConsumersFinder(const TensorPtr& operand, const HabanaGraph& graph);

private:
    void push(const NodePtr& prod) override;
};

}  // namespace gc::layered_brain::bundler

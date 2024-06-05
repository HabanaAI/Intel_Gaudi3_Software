#include "graph_traversal_generator.h"

GraphTraversalGenerator::GraphTraversalGenerator(const HabanaGraph& g, bool reverse, GraphTraversalComparator comp)
: m_g(g), m_comp(comp)
{
    // initialize
    for (const NodePtr& node : g.getNodes())
    {
        if (!node) continue;
        unsigned degree   = reverse ? g.getNodeConsumers(node).size() : g.getNodeProducers(node).size();
        m_inDegrees[node] = degree;
        if (degree == 0)
        {
            m_freeNodes.push_back(node);
        }
        m_blockedNodes[node] = reverse ? g.getNodeProducers(node) : g.getNodeConsumers(node);
    }
}

NodePtr GraphTraversalGenerator::getNext()
{
    if (empty()) return nullptr;

    auto    it       = std::min_element(m_freeNodes.begin(), m_freeNodes.end(), m_comp);
    NodePtr nextNode = *it;
    m_freeNodes.erase(it);

    for (const NodePtr& blocked : m_blockedNodes.at(nextNode))
    {
        if (--m_inDegrees[blocked] == 0)
        {
            m_freeNodes.push_back(blocked);
        }
    }
    return nextNode;
}

bool GraphTraversalGenerator::empty() const
{
    if (m_freeNodes.empty())
    {
        validateDone();
        return true;
    }
    return false;
}

void GraphTraversalGenerator::validateDone() const
{
    for (const auto& inDeg : m_inDegrees)
    {
        if (inDeg.second != 0)
        {
            HB_ASSERT(inDeg.second == 0, "not finished deg for {}", inDeg.first->getNodeName());
        }
    }
}
#include "key_nodes_finder.h"
#include "habana_graph.h"
#include "types.h"

using namespace gc::layered_brain;

uint64_t MultiMmeBundleKeyNodesFinder::getOperandsTotalSizeElements(const NodePtr& node) const
{
    const auto& operands = node->getOperands();
    return std::accumulate(operands.begin(), operands.end(), 0, [](uint64_t acc, const TensorPtr& t) {
        return acc + (t ? t->getTotalElements() : 0);
    });
}

unsigned MultiMmeBundleKeyNodesFinder::getNumConsumersInBundle(const NodePtr& node) const
{
    const auto& consumers = m_graph.getNodeConsumers(node);
    return std::count_if(consumers.begin(), consumers.end(), [&](const auto& consumer) {
        return std::find(m_bundleNodes.begin(), m_bundleNodes.end(), consumer) != m_bundleNodes.end();
    });
}

NodeVector MultiMmeBundleKeyNodesFinder::getSortedKeyNodes()
{
    SET_TEMP_LOG_CONTEXT("MultiMmeBundleKeyNodesFinder");
    NodeVector keyNodes;
    for (const auto& node : m_bundleNodes)
    {
        if (HabanaGraph::runsOnMME(node))
        {
            keyNodes.push_back(node);
        }
    }
    HB_ASSERT(!keyNodes.empty(), "Expected at least one MME node in the bundle");

    std::sort(keyNodes.begin(), keyNodes.end(), [&](const NodePtr& n1, const NodePtr& n2) {
        auto n1Consumers = getNumConsumersInBundle(n1);
        auto n2Consumers = getNumConsumersInBundle(n2);
        if (n1Consumers != n2Consumers)
        {
            return n1Consumers > n2Consumers;
        }
        auto n1OperandsSize = getOperandsTotalSizeElements(n1);
        auto n2OperandsSize = getOperandsTotalSizeElements(n2);
        if (n1OperandsSize != n2OperandsSize)
        {
            return n1OperandsSize > n2OperandsSize;
        }
        return n1->getId() < n2->getId();
    });

    LOG_DEBUG(LB_SLICER,
              "Found {} key nodes in bundle: {}",
              keyNodes.size(),
              toString(keyNodes, ',', [](const pNode& n) { return n->getNodeName(); }));
    return keyNodes;
}
#include "perforation_bvd_selector.h"
#include <queue>

using namespace gc::layered_brain;

NodePtr PerforationBVDSelector::getRootNode(const NodeSet&                                  nodes,
                                            const std::map<NodePtr, PerforationCandidates>& candidates) const
{
    HB_ASSERT(!nodes.empty(), "Expected a non empty nodes set");

    // Select a root node to start the bundle traversing according to this priority:

    // 1. MME node with preferred perforation dim, in case of multiple conflicting MME nodes in the bundle –
    // select the first one (this logic can be improved later)
    for (const auto& node : nodes)
    {
        if (candidates.at(node).mmeCandidate.has_value())
        {
            return node;
        }
    }

    // 2. TPC node with preferred perforation candidate
    for (const auto& node : nodes)
    {
        if (HabanaGraph::runsOnTPC(node) && !candidates.at(node).preferredCandidates.empty())
        {
            return node;
        }
    }

    // 3. Any node with valid perforation candidate
    for (const auto& node : nodes)
    {
        if (!candidates.at(node).validCandidates.empty())
        {
            return node;
        }
    }

    return *nodes.begin();
}

NodeSet PerforationBVDSelector::getNodeNeighbors(const NodePtr& node) const
{
    NodeSet neighbors;
    for (const auto& producer : m_graph.getNodeProducers(node))
    {
        if (producer && (std::find(m_bundleNodes.begin(), m_bundleNodes.end(), producer) != m_bundleNodes.end()))
        {
            neighbors.insert(producer);
        }
    }
    for (const auto& consumer : m_graph.getNodeConsumers(node))
    {
        if (consumer && (std::find(m_bundleNodes.begin(), m_bundleNodes.end(), consumer) != m_bundleNodes.end()))
        {
            neighbors.insert(consumer);
        }
    }
    return neighbors;
}

std::optional<BundleViewId>
PerforationBVDSelector::selectPerforationFromCandidates(const NodePtr&                     node,
                                                        const std::optional<BundleViewId>& currentPerforationDim,
                                                        const PerforationCandidates&       candidates) const
{
    // For MME nodes - take perforation from MME brain solution and igonore current perforation dim
    if (HabanaGraph::runsOnMME(node))
    {
        return candidates.mmeCandidate;
    }
    // Prefer current perforation BVD if it is one of the preferred candidates for the node
    if (currentPerforationDim.has_value() &&
        (std::find(candidates.preferredCandidates.begin(),
                   candidates.preferredCandidates.end(),
                   currentPerforationDim.value()) != candidates.preferredCandidates.end()))
    {
        return currentPerforationDim.value();
    }
    // Select a BVD from the preferred candidates that is common in most operations in the bundle
    if (!candidates.preferredCandidates.empty())
    {
        return candidates.preferredCandidates.front();
    }
    // Select a BVD from the valid candidates that is mapped to the most external tensor dim
    if (!candidates.validCandidates.empty())
    {
        return candidates.validCandidates.front();
    }
    return std::nullopt;  // No perforation candidates – empty dim will be returned
}

std::optional<BundleViewId>
PerforationBVDSelector::getPerforationDimForNode(const NodePtr&                     node,
                                                 const std::optional<BundleViewId>& currentPerforationDim,
                                                 const PerforationCandidates&       candidates) const
{
    const auto& perforationDim = selectPerforationFromCandidates(node, currentPerforationDim, candidates);

    if (perforationDim.has_value())
    {
        LOG_DEBUG(LB_SLICER,
                  "Set perforation BVD for node {} to {} (current bundle perforation BVD = [{}])",
                  node->getNodeName(),
                  perforationDim.value(),
                  currentPerforationDim.has_value() ? std::to_string(currentPerforationDim.value()) : "");
    }
    else
    {
        LOG_DEBUG(LB_SLICER, "No perforation BVD found for node {}", node->getNodeName());
    }

    return perforationDim;
}

void PerforationBVDSelector::validateCandidates(const std::map<NodePtr, PerforationCandidates>& candidates) const
{
    for (const auto& node : m_bundleNodes)
    {
        HB_ASSERT(candidates.find(node) != candidates.end(),
                  "Missing candidates for bundle node {}",
                  node->getNodeName());
    }
}

PerforationPerNode
PerforationBVDSelector::selectPerforationPerNode(const std::map<NodePtr, PerforationCandidates>& candidates) const
{
    validateCandidates(candidates);

    PerforationPerNode  perforationBVDPerNode;
    std::queue<NodePtr> nodesQueue;

    NodeSet nodesToHandle(m_bundleNodes.begin(), m_bundleNodes.end());
    while (!nodesToHandle.empty())
    {
        NodePtr rootNode = getRootNode(nodesToHandle, candidates);
        LOG_DEBUG(LB_SLICER, "Select perforation dim for bundle nodes, root node: {}", rootNode->getNodeName());
        perforationBVDPerNode[rootNode] = getPerforationDimForNode(rootNode, {}, candidates.at(rootNode));
        nodesToHandle.erase(rootNode);
        nodesQueue.push(rootNode);

        while (!nodesQueue.empty())
        {
            NodePtr nodeToHandle = nodesQueue.front();
            nodesQueue.pop();
            for (const auto& neighbor : getNodeNeighbors(nodeToHandle))
            {
                if (perforationBVDPerNode.find(neighbor) == perforationBVDPerNode.end())  // Not handled yet
                {
                    perforationBVDPerNode[neighbor] = getPerforationDimForNode(neighbor,
                                                                               perforationBVDPerNode.at(nodeToHandle),
                                                                               candidates.at(neighbor));
                    nodesToHandle.erase(neighbor);
                    nodesQueue.push(neighbor);
                }
            }
        }
    }

    HB_ASSERT(perforationBVDPerNode.size() == m_bundleNodes.size(), "Expected a perforation dim per node");
    return perforationBVDPerNode;
}
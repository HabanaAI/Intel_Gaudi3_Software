#pragma once

#include "bundle_view.h"
#include "habana_graph.h"
#include "strategy.h"
#include "perforation_bvd_candidates_finder.h"

namespace gc::layered_brain
{
// The perforation BVD selector is responsible to select a perforation BVD for every node in the bundle
// using the perforation candidates.
// For MME nodes with preferred perforation - the perforation dim is taken from the strategy.
// For the rest of the nodes (TPC nodes / MME without preferred perforation) - the algorithm starts with one of the
// nodes in the bundle and extend its perforation selection throughout the bundle in a BFS manner.
class PerforationBVDSelector
{
public:
    PerforationBVDSelector(const HabanaGraph& graph, const NodeVector& bundleNodes)
    : m_graph(graph), m_bundleNodes(bundleNodes) {};

    PerforationPerNode selectPerforationPerNode(const std::map<NodePtr, PerforationCandidates>& candidates) const;

private:
    void    validateCandidates(const std::map<NodePtr, PerforationCandidates>& candidates) const;
    NodePtr getRootNode(const NodeSet& nodes, const std::map<NodePtr, PerforationCandidates>& candidates) const;
    NodeSet getNodeNeighbors(const NodePtr& node) const;
    std::optional<BundleViewId>
                                selectPerforationFromCandidates(const NodePtr&                     node,
                                                                const std::optional<BundleViewId>& currentPerforationDim,
                                                                const PerforationCandidates&       candidates) const;
    std::optional<BundleViewId> getPerforationDimForNode(const NodePtr&                     node,
                                                         const std::optional<BundleViewId>& currentPerforationDim,
                                                         const PerforationCandidates&       candidates) const;
    const HabanaGraph&          m_graph;
    const NodeVector            m_bundleNodes;
};

}  // namespace gc::layered_brain
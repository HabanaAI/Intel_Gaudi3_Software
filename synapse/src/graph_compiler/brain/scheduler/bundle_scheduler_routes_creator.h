#pragma once

#include "bundle_scheduler_interfaces.h"

namespace gc::layered_brain
{
class SingleRouteCreator : public RoutesCreator
{
public:
    SingleRouteCreator(HabanaGraph& graph, const BundleNodes& bundle);
    RoutesTable getRoutesPerSlice(const std::vector<NodePtr>& routeEnds) override;

protected:
    TensorSet getSlicesForRouteEnd(const NodePtr& routeEnd);

    // Build the routes from an output sliced BPT to the bundle inputs.
    NodesRoute buildSlicedBPTRoute(const TensorPtr& bptSlice);

    // Breakes the route to 2 parts around an MME node - prefix route and postfix route, to be interleaved with other
    // slices when the thread is created. If there's no MME node - the prefix route is empty. The MME node is included
    // in the prefix route if it has no producers, or in the postfix route
    std::vector<NodesRoute> breakRouteOnMme(const NodesRoute& route);

    virtual NodesRoute getRouteToSlice(const TensorPtr& slice) = 0;

    bool isInBundle(const NodePtr& node) const;
    std::optional<NodePtr> getProducerInBundle(const TensorPtr& slice);

    HabanaGraph& m_graph;
    BundleNodes  m_bundle;
    BundleIndex  m_bundleIdx;
};

class DfsRoutesCreator : public SingleRouteCreator
{
public:
    DfsRoutesCreator(HabanaGraph& graph, const BundleNodes& bundle) : SingleRouteCreator(graph, bundle) {}

protected:
    // Use DFS post order to create the route to the output slice. Traversing backwards from the output.
    NodesRoute getRouteToSlice(const TensorPtr& slice) override;
    void       addInputsRoutes(const TensorVector& inputs, NodesRoute& routeToSlice);
    void       addSubRoute(NodesRoute& route, const NodesRoute& subRoute);
};

class BfsRoutesCreator : public SingleRouteCreator
{
public:
    BfsRoutesCreator(HabanaGraph& graph, const BundleNodes& bundle) : SingleRouteCreator(graph, bundle) {}

protected:
    // Use BFS with repetitions to create the route to the output slice. Traversing backwards from the output.
    NodesRoute getRouteToSlice(const TensorPtr& slice) override;
    void       addInputsToQueue(const TensorVector& inputs, std::list<NodePtr>& queue);
};

}  // namespace gc::layered_brain
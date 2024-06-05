#include "bundle_scheduler_routes_creator.h"
#include "habana_graph.h"

using namespace gc::layered_brain;

SingleRouteCreator::SingleRouteCreator(HabanaGraph& graph, const BundleNodes& bundle) : m_graph(graph), m_bundle(bundle)
{
    auto bundleIdx = getBundleIndex(m_bundle.front());
    HB_ASSERT(bundleIdx, "bundle nodes expected to have bundle index");
    m_bundleIdx = *bundleIdx;
}

RoutesTable SingleRouteCreator::getRoutesPerSlice(const std::vector<NodePtr>& routeEnds)
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);
    RoutesTable routesPerSlice;
    for (const auto& routeEnd : routeEnds)
    {
        for (const auto& slice : routeEnd->getInputs())
        {
            if (!slice) continue;
            auto route            = buildSlicedBPTRoute(slice);
            routesPerSlice[slice] = breakRouteOnMme(route);
        }
    }
    return routesPerSlice;
}

NodesRoute SingleRouteCreator::buildSlicedBPTRoute(const TensorPtr& bptSlice)
{
    LOG_TRACE(LB_SCHEDULER, "{}: {}", HLLOG_FUNC, bptSlice->getName());
    NodesRoute route = getRouteToSlice(bptSlice);
    route.print();
    return route;
}

std::vector<NodesRoute> SingleRouteCreator::breakRouteOnMme(const NodesRoute& route)
{
    const auto mmeIt =
        std::find_if(route.begin(), route.end(), [](const NodePtr& n) { return HabanaGraph::runsOnMME(n); });
    if (mmeIt != route.end())
    {
        // Break the route just before the MME node
        auto breakPointIt = mmeIt;
        if (mmeIt == route.begin())
        {
            // Include the MME in the prefix route if there are no producers
            breakPointIt = std::next(mmeIt, 1);
        }
        NodesRoute routePrefix {route.begin(), breakPointIt};
        NodesRoute routePostfix {breakPointIt, route.end()};
        return {routePrefix, routePostfix};
    }
    // If there's no MME node in the route - make it s postfix route (later scheduling)
    return {{}, route};
}

std::optional<NodePtr> SingleRouteCreator::getProducerInBundle(const TensorPtr& slice)
{
    NodePtr producer = m_graph.getTensorProducer(slice);
    // If there is no producer, or it's not in the bundle - empty route
    if (!producer || !isInBundle(producer) || Node::isForkNode(producer)) return std::nullopt;
    return producer;
}

bool SingleRouteCreator::isInBundle(const NodePtr& node) const
{
    const auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
    return (bundleInfo.is_set() && bundleInfo->bundleIndex == m_bundleIdx);
}

NodesRoute DfsRoutesCreator::getRouteToSlice(const TensorPtr& slice)
{
    NodesRoute routeToSlice;
    auto       producer = getProducerInBundle(slice);
    // If there is no producer, or it's not in the bundle - empty route
    if (!producer.has_value()) return {};

    // Traverse all the producer's inputs and control inputs.
    // Add the routes to the inputs to be executed before the producer
    addInputsRoutes(producer.value()->getInputs(), routeToSlice);
    addInputsRoutes(producer.value()->getControlInputs(), routeToSlice);
    // Finally add the producer to the route
    NodesRoute subRoute {1, producer.value()};
    addSubRoute(routeToSlice, subRoute);
    return routeToSlice;
}

void DfsRoutesCreator::addInputsRoutes(const TensorVector& inputs, NodesRoute& routeToSlice)
{
    // Traverse all the producer's inputs and collect the routes to be executed before the producer
    for (const TensorPtr& input : inputs)
    {
        if (!input) continue;
        auto routeToInput = getRouteToSlice(input);
        addSubRoute(routeToSlice, routeToInput);
    }
}

void DfsRoutesCreator::addSubRoute(NodesRoute& route, const NodesRoute& subRoute)
{
    if (subRoute.empty()) return;

    route.insert(route.end(), subRoute.begin(), subRoute.end());
}

NodesRoute BfsRoutesCreator::getRouteToSlice(const TensorPtr& slice)
{
    std::list<NodePtr> routeToSlice;  // insert visited nodes in reverse order to provide the route in correct order
    std::list<NodePtr> queue;

    auto producer = getProducerInBundle(slice);
    HB_ASSERT(producer.has_value(), "output BPT slice without a producer in bundle {}", slice->getName());
    queue.push_back(producer.value());

    while (!queue.empty())
    {
        auto currNode = queue.front();
        routeToSlice.push_front(currNode);
        queue.pop_front();
        addInputsToQueue(currNode->getInputs(), queue);
        addInputsToQueue(currNode->getControlInputs(), queue);
    }
    return NodesRoute {routeToSlice.begin(), routeToSlice.end()};
}

void BfsRoutesCreator::addInputsToQueue(const TensorVector& inputs, std::list<NodePtr>& queue)
{
    for (const TensorPtr& input : inputs)
    {
        auto producer = getProducerInBundle(input);
        // If the producer is in the bundle - insert it to queue. In classic BFS we should insert it only if it's not
        // already in the queue. However, in case of node with several dependents (data or control), we must add it
        // multiple times to make sure it is scheduled before its first dependent node. The multi appearances will be
        // ignored in scheduling
        if (producer.has_value())
        {
            queue.push_back(producer.value());
        }
    }
}
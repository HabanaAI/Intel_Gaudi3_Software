#include "bundle_scheduler_threads_creator.h"
#include <unordered_set>

using namespace gc::layered_brain;

// Gradually builds a thread from different routes
class ThreadMerger
{
public:
    ThreadMerger() {}
    void mergeRouteIntoThread(const NodesRoute& route)
    {
        for (const NodePtr& n : route)
        {
            bool elementIsNew = m_threadNodes.insert(n).second;
            if (elementIsNew)
            {
                m_thread.push_back(n);
            }
        }
    }
    NodesThread drain() { return std::move(m_thread); }

private:
    NodesThread                 m_thread;
    std::unordered_set<NodePtr> m_threadNodes;
};

ThreadsSequence SimpleThreadsCreator::getThreadsSequeceBySliceSets(const std::vector<TensorVector>& sliceSets,
                                                                   const RoutesTable&               routesPerSlice)
{
    ThreadsSequence threads;
    for (const auto& set : sliceSets)
    {
        NodesThread thread = getThreadPerSliceSet(set, routesPerSlice);
        threads.push_back(thread);
    }
    return threads;
}

NodesThread SimpleThreadsCreator::getThreadPerSliceSet(const TensorVector& sliceSet, const RoutesTable& routesPerSlice)
{
    ThreadMerger merger;
    std::vector<NodesRoute> orderedRoutes = sortSubRoutes(sliceSet, routesPerSlice);
    for (const auto& route : orderedRoutes)
    {
        merger.mergeRouteIntoThread(route);
    }
    return merger.drain();
}

// Sort the subroutes of all the slices in the slice set, to be interleaved. Every slice should have the same number of
// routes, such that route index i of each slice belongs to layer i. The sorting takes the route from each slice, which
// corresponds to the same layer, in increasing order. All subroutes of layer i for all slices will be placed before all
// subroutes of layer i+1.
std::vector<NodesRoute> SimpleThreadsCreator::sortSubRoutes(const TensorVector& sliceSet,
                                                            const RoutesTable&  routesPerSlice)
{
    // set the number of layers, and assert all slices have the same number of routes, to be interleaved by layers
    unsigned numLayers = 0;
    for (const auto& slice : sliceSet)
    {
        if (numLayers == 0)
        {
            numLayers = routesPerSlice.at(slice).size();
        }
        HB_ASSERT(routesPerSlice.at(slice).size() == numLayers,
                  "num routes ({}) != expected ({}) for slice {}",
                  routesPerSlice.at(slice).size(),
                  numLayers,
                  slice->getName());
    }
    // order the routes by layers for all slice sets
    std::vector<NodesRoute> orderedRoutes;
    orderedRoutes.reserve(numLayers * sliceSet.size());
    for (unsigned layerId = 0; layerId < numLayers; layerId++)
    {
        for (const auto& slice : sliceSet)
        {
            orderedRoutes.push_back(routesPerSlice.at(slice).at(layerId));
        }
    }
    return orderedRoutes;
}
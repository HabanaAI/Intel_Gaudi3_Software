#pragma once

#include "brain_data.h"
#include "layered_brain.h"
#include "layout.h"
#include "types.h"

namespace gc::layered_brain
{
class NodesRoute : public std::vector<NodePtr>
{
public:
    template<typename... T>
    NodesRoute(T... args) : std::vector<NodePtr>(args...)
    {
    }

    void print() const
    {
        if (LOG_LEVEL_AT_LEAST_DEBUG(LB_SCHEDULER))
        {
            LOG_DEBUG(LB_SCHEDULER, "Route: {}", toString(*this, '-', [](const NodePtr& n) {
                          return n->getNodeName();
                      }));
        }
    }
};

class NodesThread : public std::vector<NodePtr>
{
public:
    template<typename... T>
    NodesThread(T... args) : std::vector<NodePtr>(args...)
    {
    }

    void print() const
    {
        if (LOG_LEVEL_AT_LEAST_DEBUG(LB_SCHEDULER))
        {
            LOG_DEBUG(LB_SCHEDULER, "Thread: {}", toString(*this, ',', [](const NodePtr& n) {
                          return n->getNodeName();
                      }));
        }
    }
};

struct ReducedSliceInfo
{
    NodePtr                          sliceConsumer;
    std::unordered_set<BundleViewId> reducedBvdIds;
};

using SliceToReductionInfo = TensorToItemOrderedMap<ReducedSliceInfo>;
using RoutesTable          = TensorToItemOrderedMap<std::vector<NodesRoute>>;
using BvdTraversalPattern  = std::vector<BundleViewId>;
using ThreadsSequence      = std::vector<NodesThread>;

// Collects the bundle sliced nodes from the sliced graph, and creates the routes to produce each slice of each output
// BPT. The routes may include identical sub routes for nodes which participate in multiple output slices creation.
class RoutesCreator
{
public:
    virtual RoutesTable getRoutesPerSlice(const std::vector<NodePtr>& routeEnds) = 0;
    virtual ~RoutesCreator() {}
};

// Order the bundle views by efficiency considerations, and provide an optimal traversal pattern on the bundle output
// BPTs.
class BVDsTraversalSorter
{
public:
    virtual BvdTraversalPattern getBundleViewsByTraversalOrder(const BundleData& bundleData) = 0;
    virtual ~BVDsTraversalSorter() {}
};

// Orders the output BPT slices according to the walk pattern on the big output BPTs.
// Groups tiles of the output BPTs, which belong to the same BVD coordinate, to tiles sets
class OutputSlicesSequencer
{
public:
    virtual std::vector<TensorVector> getSliceSetsSequence(const BvdTraversalPattern&  traversalPattern,
                                                           const BundleData&           bundleData,
                                                           const SliceToReductionInfo& sliceToReducedBvds,
                                                           const std::vector<NodePtr>& routeEndNodes) = 0;
    virtual ~OutputSlicesSequencer() {}
};

// Merge routes of the output BPT slices, which belong to the same tiles set, to a single thread, ready to be scheduled.
// The created thread is aligned with the given order, to comply with memory capacity assumptions.
// The tiles sets may have different sizes, as different BPTs are sliced on different BVDs.
class ThreadsCreator
{
public:
    virtual ThreadsSequence getThreadsSequeceBySliceSets(const std::vector<TensorVector>& sliceSets,
                                                         const RoutesTable&               routesPerSlice) = 0;
    virtual ~ThreadsCreator() {}
};

// Schedules the bundle nodes according to the given routes and the routes order, derived from the output BPT slices
// order. Can schedule multiple routes in round-robin to maximize parallelism on multiple engines.
class ThreadsScheduler
{
public:
    virtual bool scheduleNodes(const ThreadsSequence&      threadsSequence,
                               const std::vector<NodePtr>& routeEnds,
                               unsigned                    pipelineDepth) = 0;
    virtual ~ThreadsScheduler() {}
};

}  // namespace gc::layered_brain
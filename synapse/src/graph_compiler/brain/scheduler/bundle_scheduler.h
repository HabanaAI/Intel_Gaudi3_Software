#pragma once

#include "bundle_scheduler_interfaces.h"
#include "tensor.h"

namespace gc::layered_brain
{
class BundleScheduler
{
public:
    BundleScheduler(HabanaGraph& graph, const BundleNodes& bundle);
    bool setBundleNodesSchedule(bool dryRun);

    static void printRoute(const NodesRoute& route);

private:
    HabanaGraph& m_graph;
    BundleNodes  m_bundle;
    BundleIndex  m_bundleIdx;

    std::unique_ptr<RoutesCreator>         getRoutesCreator(HabanaGraph& graph, const BundleNodes& bundle);
    std::unique_ptr<BVDsTraversalSorter>   getBvdsSorter();
    std::unique_ptr<OutputSlicesSequencer> getSlicesSequencer();
    std::unique_ptr<ThreadsCreator>        getThreadsCreator();
    std::unique_ptr<ThreadsScheduler>      getThreadsScheduler(HabanaGraph& graph, const BundleNodes& bundle);

    bool                             validateAllNodesScheduled() const;
    std::vector<NodePtr>             getRouteEndNodes(const BundleData& bundleData) const;
    std::unordered_set<BundleViewId> getReducedBvdIds(const NodePtr& node, const BundleData& bundleData) const;
    SliceToReductionInfo             createSliceToReductionInfoMap(const BundleData&  bundleData,
                                                                   const RoutesTable& routesTable) const;
    void                             validateCtrlDepForMemsets();
    static bool                      isRouteEnd(const NodePtr& n, const BundleData& bundleData);
};

}  // namespace gc::layered_brain
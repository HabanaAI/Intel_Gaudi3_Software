#pragma once

#include "bundle_scheduler_interfaces.h"
#include <optional>

namespace gc::layered_brain
{
class MultiThreadScheduler : public ThreadsScheduler
{
public:
    MultiThreadScheduler(HabanaGraph& graph, const BundleNodes& bundle);
    bool scheduleNodes(const ThreadsSequence&      threadsSequence,
                       const std::vector<NodePtr>& routeEnds,
                       unsigned                    pipelineDepth) override;

protected:
    struct ThreadState
    {
        NodesThread thread;
        enum class State
        {
            NOT_STARTED,
            STARTED,
        } state;
    };

    using ThreadsList = std::list<ThreadState>;

    void scheduleThreads(const ThreadsSequence& threadsSequence, unsigned pipelineDepth);

    unsigned getNextUnscheduledNodeIndex(const NodesThread& thread);

    bool shouldPrefetchNextThread(unsigned nextPendingThreadIdx, unsigned numThreads, const ThreadsList& activeThreads);
    bool preferSchedulingFromSameThread(const NodesThread& currentThread,
                                        const NodePtr&     lastScheduledNode,
                                        const ThreadsList& activeThreads);
    bool isNextNodeOnAllActiveThreadsDataDependent(const NodePtr& lastScheduledNode, const ThreadsList& activeThreads);
    bool isNextNodeOnThreadDataDependent(const NodesThread& thread, const NodePtr& lastScheduledNode);
    bool setUninitializedOperationIndex(const NodePtr&          node,
                                        std::optional<unsigned> threadIdx = std::nullopt,
                                        std::optional<unsigned> opIdx     = std::nullopt);
    void scheduleUnreachedBundleNodes();
    void scheduleJoins(const std::vector<NodePtr>& routeEnds);
    void scheduleForks();
    bool isInBundle(const NodePtr& node) const;

    HabanaGraph& m_graph;
    BundleNodes  m_bundle;
    unsigned     m_opIndex = 0;
    BundleIndex  m_bundleIdx;
    std::set<NodePtr> m_scheduledBigNodes;
};

}  // namespace gc::layered_brain
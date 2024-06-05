#include "bundle_scheduler_threads_scheduler.h"
#include "habana_graph.h"
#include "brain_conf.h"

using namespace gc::layered_brain;

MultiThreadScheduler::MultiThreadScheduler(HabanaGraph& graph, const BundleNodes& bundle)
: m_graph(graph), m_bundle(bundle)
{
    auto bundleIdx = getBundleIndex(m_bundle.front());
    HB_ASSERT(bundleIdx, "bundle nodes expected to have bundle index");
    m_bundleIdx = *bundleIdx;
}

bool MultiThreadScheduler::scheduleNodes(const ThreadsSequence&      threadsSequence,
                                         const std::vector<NodePtr>& routeEnds,
                                         unsigned                    pipelineDepth)
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);

    scheduleForks();
    scheduleThreads(threadsSequence, pipelineDepth);
    scheduleJoins(routeEnds);
    scheduleUnreachedBundleNodes();

    return true;
}

void MultiThreadScheduler::scheduleThreads(const ThreadsSequence& threadsSequence, unsigned pipelineDepth)
{
    // Since the thread is composed of DFS routes - the data dependency is resolved within the thread.
    // Nodes may repeat multiple times in a thread or in several threads. A node is handled on its first appearance
    // only.
    ThreadsList activeThreads;
    unsigned    nextPendingThreadIdx = 0;
    while (!activeThreads.empty() || nextPendingThreadIdx < threadsSequence.size())
    {
        // Fill the active threads to pipeline depth count
        // Add new threads before existing threads, while keeping the order of the thread sequence for the new threads
        auto newThreadsPos = activeThreads.begin();
        while (activeThreads.size() < pipelineDepth && nextPendingThreadIdx < threadsSequence.size())
        {
            LOG_TRACE(LB_SCHEDULER, "{}: Adding thread index {} to active", HLLOG_FUNC, nextPendingThreadIdx);
            threadsSequence.at(nextPendingThreadIdx).print();
            activeThreads.insert(newThreadsPos,
                                 {threadsSequence.at(nextPendingThreadIdx++), ThreadState::State::NOT_STARTED});
        }

        // schedule the active threads interleaved
        for (auto threadIt = activeThreads.begin(); threadIt != activeThreads.end();)
        {
            const NodesThread& thread    = threadIt->thread;
            auto               threadIdx = index_of(threadsSequence, thread);
            HB_ASSERT(threadIdx != -1, "Active thread must be found in threads sequence");
            unsigned nodeIdx         = getNextUnscheduledNodeIndex(thread);
            bool     schedSameThread = true;
            NodePtr  realScheduledNode;
            while (schedSameThread && (nodeIdx < thread.size()))
            {
                // Handle the next node in the thread which is placed in front
                LOG_TRACE(LB_SCHEDULER, "{}: schedule thread index {} node index {}", HLLOG_FUNC, threadIdx, nodeIdx);
                auto& node          = thread.at(nodeIdx);
                bool  nodeScheduled = setUninitializedOperationIndex(node, threadIdx);
                realScheduledNode   = (nodeScheduled && !node->isLogicalOperation()) ? node : realScheduledNode;
                threadIt->state     = realScheduledNode ? ThreadState::State::STARTED : threadIt->state;
                schedSameThread =
                    !realScheduledNode || preferSchedulingFromSameThread(thread, realScheduledNode, activeThreads);
                nodeIdx = getNextUnscheduledNodeIndex(thread);
            }
            if (nodeIdx == thread.size())
            {
                LOG_TRACE(LB_SCHEDULER, "{}: thread index {} done", HLLOG_FUNC, threadIdx);
                threadIt = activeThreads.erase(threadIt);
                if (shouldPrefetchNextThread(nextPendingThreadIdx, threadsSequence.size(), activeThreads)) break;
            }
            else
            {
                ++threadIt;
            }
        }
    }
}

bool MultiThreadScheduler::shouldPrefetchNextThread(unsigned           nextPendingThreadIdx,
                                                    unsigned           numThreads,
                                                    const ThreadsList& activeThreads)
{
    if (!GCFG_LAYERED_BRAIN_SCHEDULER_PREFETCH_NEXT_THREAD.value()) return false;
    // Check if there are more threads to schedule
    bool moreThreads = nextPendingThreadIdx < numThreads;
    // Check if all active threads started. Otherwise prefer to schedule form an existing thread instead of prefetching
    bool allActiveThreadStarted =
        std::all_of(activeThreads.begin(), activeThreads.end(), [&](const ThreadState& threadState) {
            return threadState.state == ThreadState::State::STARTED;
        });
    return moreThreads && allActiveThreadStarted;
}

bool MultiThreadScheduler::preferSchedulingFromSameThread(const NodesThread& currentThread,
                                                          const NodePtr&     lastScheduledNode,
                                                          const ThreadsList& activeThreads)
{
    if (!GCFG_LAYERED_BRAIN_PREFER_SCHEDULING_SAME_THREAD.value()) return false;
    // TODO SW-167409 - if there's a non data dependent thread - need to return it and jump to it in sched order
    return !isNextNodeOnThreadDataDependent(currentThread, lastScheduledNode) ||
           isNextNodeOnAllActiveThreadsDataDependent(lastScheduledNode, activeThreads);
}

bool MultiThreadScheduler::isNextNodeOnAllActiveThreadsDataDependent(const NodePtr&     lastScheduledNode,
                                                                     const ThreadsList& activeThreads)
{
    for (const auto& threadState : activeThreads)
    {
        if (!isNextNodeOnThreadDataDependent(threadState.thread, lastScheduledNode))
        {
            // there is an active thread for which the next node to schedule is independent of the last scheduled node
            // or the thread has ended and we can schedule a new thread instead of it, which is independent
            return false;
        }
    }
    // all active threads next node to schedule are dependent of the last scheduled node
    return true;
}

unsigned MultiThreadScheduler::getNextUnscheduledNodeIndex(const NodesThread& thread)
{
    unsigned nextNodeIdx = 0;
    while (nextNodeIdx < thread.size())
    {
        const auto& node       = thread.at(nextNodeIdx);
        const auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
        HB_ASSERT(bundleInfo.is_set(), "Bundle node without bundle info: {}", node->getNodeName());
        if (bundleInfo->operationIndex == 0)  // unset
        {
            break;
        }
        nextNodeIdx++;
    }
    return nextNodeIdx;
}

bool MultiThreadScheduler::isNextNodeOnThreadDataDependent(const NodesThread& thread, const NodePtr& lastScheduledNode)
{
    unsigned nextNodeIdx = getNextUnscheduledNodeIndex(thread);
    if (nextNodeIdx < thread.size())
    {
        const auto& nextNode  = thread.at(nextNodeIdx);
        const auto  consumers = m_graph.getNodeRealConsumers(lastScheduledNode, Node::TENSOR_TYPE_DATA);
        return (consumers.find(nextNode) != consumers.end());  // next node is a consumer
    }
    return false;
}

bool MultiThreadScheduler::setUninitializedOperationIndex(const NodePtr&          node,
                                                          std::optional<unsigned> threadIdx,
                                                          std::optional<unsigned> opIdx)
{
    bool  opIndexUpdated = false;
    auto& bundleInfo     = node->getNodeAnnotation().bundleInfo;
    HB_ASSERT(bundleInfo.is_set(), "Bundle node without bundle info: {}", node->getNodeName());
    if (bundleInfo->operationIndex == 0)  // unset
    {
        unsigned opIndex           = (opIdx.has_value()) ? *opIdx : ++m_opIndex;
        bundleInfo->operationIndex = m_opIndex;
        LOG_DEBUG(LB_SCHEDULER, "{}: op index {} set for {}", HLLOG_FUNC, opIndex, node->getNodeName());
        opIndexUpdated = true;
        if (threadIdx.has_value())
        {
            HB_ASSERT(!bundleInfo->threadIndex.has_value(), "Unscheduled node with threadIdx {}", node->getNodeName());
            bundleInfo->threadIndex = *threadIdx;
        }
    }
    // else - the node was scheduled by a previous iteration

    return opIndexUpdated;
}

void MultiThreadScheduler::scheduleUnreachedBundleNodes()
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);
    // TODO SW-120881 - handle unscheduled bundle nodes until there is a pass that removes them
    for (const auto& node : m_bundle)
    {
        bool nodeScheduled = setUninitializedOperationIndex(node, {}, std::numeric_limits<unsigned>::max());
        if (nodeScheduled)
        {
            LOG_WARN(LB_SCHEDULER, "Scheduling a node which can't be reached {}", node->getNodeName());
        }
    }
}

void MultiThreadScheduler::scheduleForks()
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);
    for (const NodePtr& node : m_bundle)
    {
        if (Node::isForkNode(node))
        {
            setUninitializedOperationIndex(node);
        }
    }
}

void MultiThreadScheduler::scheduleJoins(const std::vector<NodePtr>& routeEnds)
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);
    for (const auto& n : routeEnds)
    {
        if (Node::isJoinNode(n))
        {
            setUninitializedOperationIndex(n);
        }
    }
}

bool MultiThreadScheduler::isInBundle(const NodePtr& node) const
{
    const auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
    return (bundleInfo.is_set() && bundleInfo->bundleIndex == m_bundleIdx);
}
#include "memory_oriented_scheduler.h"
#include "habana_nodes/habana_nodes.h"
#include "habana_graph.h"
#include <stack>


static bool isDmaDownstream(const NodePtr& n)
{
    if (!n->isDma()) return false;
    return std::static_pointer_cast<DMANode>(n)->getDmaType() == DMA_TYPE::DMA_TYPE_DOWNSTREAM;
}

MemoryOrientedScheduler::MemoryOrientedScheduler(const HabanaGraph* graph) : Scheduler(graph)
{
    // initiate inner data-structures
    for (const NodePtr& n : m_graph->getNodes())
    {
        // initiate DFS data structure, all nodes are unvisited.
        m_whenVisited[n] = NEVER_VISITED;

        // calculate the barrier consumers for each node (nodes blocked by current node)
        for (const NodePtr& blockingNode : n->getNodeAnnotation().memorySpaceInfo.barriers)
        {
            m_barrierConsumers[blockingNode].insert(n);
        }
    }

    // calculate longest path from each node to outputs
    NodeList topologicalOrder = Scheduler::scheduleNodes();
    for (auto it = topologicalOrder.rbegin(); it != topologicalOrder.rend(); ++it)
    {
        m_maxPathLengthFromNode[*it] = 0;

        // inserting dram spills / fill should not change schedule priority, give them the same path length as the consumer
        bool skipPathLength = ((*it)->isDramSpill() || (*it)->isDramFill()) ||
                ((*it)->isLogicalOperation() && !(*it)->isDebug());  // don't treat logical nodes as real nodes.
        for (const NodePtr& consumer : m_graph->getNodeConsumers(*it))
        {
            m_maxPathLengthFromNode[*it] = skipPathLength?
                    std::max(m_maxPathLengthFromNode[*it], m_maxPathLengthFromNode[consumer]) :
                    std::max(m_maxPathLengthFromNode[*it], m_maxPathLengthFromNode[consumer] + 1) ;
        }
    }
}

NodeList MemoryOrientedScheduler::scheduleNodes()
{
    for (const NodePtr& n : m_graph->getNodes())
    {
        const auto& annotation = n->getNodeAnnotation();
        // this scheduler can't handle hard-set ranges.
        // (possible solution for the future - use a dummy node between 2 ranges)
        if (annotation.rangeIndex != 0 || annotation.sliceIndex != 0)
        {
            return Scheduler::scheduleNodes();
        }
    }
    return dfsSchedule();
}

bool DfsNodeScheduleComparator::operator()(const NodePtr& n1, const NodePtr& n2) const
{
    //ensure everything comes before null
    if (n1 == nullptr) return false;
    if (n2 == nullptr) return true;

    // prefer to perform "barrier-blocked" nodes later
    const NodeList& barriers = m_node->getNodeAnnotation().memorySpaceInfo.barriers;
    bool n1IsBarrier = std::find(barriers.begin(), barriers.end(), n1) != barriers.end();
    bool n2IsBarrier = std::find(barriers.begin(), barriers.end(), n2) != barriers.end();
    if (n1IsBarrier && !n2IsBarrier) return true;
    if (!n1IsBarrier && n2IsBarrier) return false;

    // execute the residual connections (nodes with shorter "longest path to output" last (push them first in dfs order).
    int n1MaxPath = m_maxPath.find(n1)->second;
    int n2MaxPath = m_maxPath.find(n2)->second;
    if (n1MaxPath > n2MaxPath) return false;
    if (n1MaxPath < n2MaxPath) return true;

    //prefer nodes that don't increase live tensor size
    int64_t n1DiffSize = 0;
    for (const TensorPtr& t : n1->getOutputs())
    {
        if (t == nullptr) continue;
        n1DiffSize += t->getDenseSizeInBytes();
    }
    for (const TensorPtr& t : n1->getInputs())
    {
        if (t == nullptr) continue;
        n1DiffSize -= t->getDenseSizeInBytes();
    }
    int64_t n2DiffSize = 0;
    for (const TensorPtr& t : n2->getOutputs())
    {
        if (t == nullptr) continue;
        n2DiffSize += t->getDenseSizeInBytes();
    }
    for (const TensorPtr& t : n2->getInputs())
    {
        if (t == nullptr) continue;
        n2DiffSize -= t->getDenseSizeInBytes();
    }
    if (n1DiffSize < n2DiffSize) return false;  // n1 makes data smaller
    if (n1DiffSize > n2DiffSize) return true;  // n2 makes data smaller

    // in dfs the "best" node to do next is the one we want scheduled last, so use the inverse of the default comparator
    return !NodeScheduleComparator::operator()(n1, n2);
}

namespace
{
    struct Root
    {
        NodePtr node;
        int longestPathDist;
    };
} // anonymous namespace

bool MemoryOrientedScheduler::allConsumersVisited(const NodePtr& n)
{
    for (const NodePtr& consumer : m_graph->getNodeConsumers(n))
    {
        if (m_whenVisited[consumer] == NEVER_VISITED) return false;
    }
    return true;
}

void MemoryOrientedScheduler::gatherLogicalInputs(NodeList& schedule, int rootCount)
{
    for (auto it = schedule.begin(); it != schedule.end(); it++)
    {
        for (const NodePtr& n : m_graph->getNodeProducers(*it))
        {
            if (m_whenVisited[n] == NEVER_VISITED && n->isLogicalOperation() && allConsumersVisited(n))
            {
                schedule.insert(it, n);
                m_whenVisited[n] = rootCount;
            }
        }
    }
}

NodeList MemoryOrientedScheduler::dfsScheduleFromRoot(const NodePtr& root, int rootCount, NodeSet& visitedConsumers)
{
    NodeList schedule;
    // push root node to stack.
    std::stack<NodePtr> dfsStack;
    dfsStack.push(root);
    m_whenVisited[root] = rootCount;

    // start dfs
    while (!dfsStack.empty())
    {
        // look at the top node
        NodePtr node = dfsStack.top();

        /*
         * get (best) next node consumers
         * reminder: the next node that is chosen will be scheduled AFTER the rest of the consumers
         */
        NodePtr nextNode = nullptr;
        const auto& comparator = DfsNodeScheduleComparator(node, m_maxPathLengthFromNode);
        // get barrier consumers - having a barrier is like a "invisible" edge.
        for (const NodePtr& barrierConsumer : m_barrierConsumers[node])
        {
            if (m_whenVisited[barrierConsumer] == NEVER_VISITED)  // not visited yet
            {
                nextNode = std::min<NodePtr>(nextNode, barrierConsumer, comparator);
            }
            else if (m_whenVisited[barrierConsumer] < rootCount) // visited in a previous root run.
            {
                visitedConsumers.insert(barrierConsumer);
            }
        }
        for (const NodePtr& consumer : m_graph->getNodeConsumers(node))
        {
            if (m_whenVisited[consumer] == NEVER_VISITED)  // not visited yet
            {
                nextNode = std::min<NodePtr>(nextNode, consumer, comparator);
            }
            else if (m_whenVisited[consumer] < rootCount) // visited in a previous root run.
            {
                visitedConsumers.insert(consumer);
            }
        }

        // if all consumers are visited, pop node and push in front of the topological order
        if (nextNode == nullptr)
        {
            dfsStack.pop();
            schedule.push_front(node);
        }
        else
        {
            dfsStack.push(nextNode);
            m_whenVisited[nextNode] = rootCount;
        }
    }

    gatherLogicalInputs(schedule, rootCount);
    return schedule;
}

// verify that we scheduled all nodes
void MemoryOrientedScheduler::verifySchedule(const NodeList& schedule) const
{
    if (m_graph->getNodes().size() != schedule.size())
    {
        for (const NodePtr& n : m_graph->getNodes())
        {
            if (std::find(schedule.begin(), schedule.end(), n) == schedule.end())
            {
                LOG_ERR(SCHEDULER, "Node {} wasn't scheduled", n->getNodeName());
            }
        }
        HB_ASSERT(0, "not all nodes were scheduled!");
    }

    for (const auto& it : m_whenVisited)
    {
        HB_ASSERT(it.second != NEVER_VISITED, "node {} wasn't scheduled!", it.first->getNodeName());
    }

    LOG_TRACE(SCHEDULER, "newly generated execution schedule:");
    for (const pNode& n : schedule)
    {
        LOG_TRACE(SCHEDULER, "{}", n->getNodeName());
    }
}

static NodePtr getSingleProducer(const Graph* graph, const NodePtr& node)
{ // get producer of node, and assert that it's the only producer
    HB_ASSERT(node->getNumInputs() == 1, "expected single input for dram spill");
    const TensorPtr input = node->getInput(0);
    const NodePtr& producer = graph->getTensorProducer(input);
    HB_ASSERT_PTR(producer);
    return producer;
}

static bool isNodeConsumer(const NodePtr& node, const NodePtr& consumer)
{ // is "node" a direct producer for "consumer"
    for (const TensorPtr& output : node->getOutputs())
    {
        for (const TensorPtr& input : consumer->getInputs())
        {
            if (input == output) return true;
        }
    }
    return false;
}

void MemoryOrientedScheduler::pushForwardDramSpills(NodeList& schedule)
{ // make sure dram spills happen as soon as possible (right after their producer)
    NodeList dramSpills;
    auto it = schedule.begin();
    auto previousIt = it++;
    while (it != schedule.end())
    {
        if ((*it)->isDramSpill())
        {
            // if the spill can be pushed forward, do it.
            NodePtr producer = getSingleProducer(m_graph, *it);
            // if the previous node is the producer, then no need to do anything.
            if (*previousIt != producer)
            {
                dramSpills.push_front(*it);
                it = schedule.erase(it);
                continue;
            }
        }
        it++;
        previousIt++;
    }

    for (const NodePtr& spillNode : dramSpills)
    {
        NodePtr producer = getSingleProducer(m_graph, spillNode);
        auto producerIt = std::find(schedule.begin(), schedule.end(), producer);
        schedule.insert(++producerIt, spillNode);
    }
}

void MemoryOrientedScheduler::postponeDramFills(NodeList& schedule)
{ // make sure dram fills happen as late as possible (right before their first consumer)
    NodeList dramFills;
    auto it = schedule.begin();
    while (it != schedule.end())
    {
        if ((*it)->isDramFill())
        {
            HB_ASSERT((*it)->getNumOutputs() == 1, "expected single output for dram fill");
            // if the next node is allready a consumer no need to do anything.
            if (!isNodeConsumer(*it, *std::next(it)))
            {
                dramFills.push_front(*it);
                it = schedule.erase(it);
                continue;
            }
        }
        it++;
    }

    for (const NodePtr& fillNode : dramFills)
    {
        NodeList consumersList = m_graph->getTensorConsumers(fillNode->getOutput(0));
        NodeSet consumersSet(consumersList.begin(), consumersList.end());
        auto consumerIt = schedule.begin();
        for (; consumerIt != schedule.end(); consumerIt++)
        {   // find the first consumer
            if (consumersSet.count(*consumerIt)) break;
        }
        // insert dram fill right before the first consumer
        schedule.insert(consumerIt, fillNode);
    }
}

// create memory-oriented schedule
NodeList MemoryOrientedScheduler::dfsSchedule()
{
    NodeList schedule;

    // get initial free list - all nodes without producers, sort them by longest path to output.
    auto cmpRoot = [](const Root& a, const Root& b)
    {
        return (a.longestPathDist == b.longestPathDist) ?
               a.node->getId() < b.node->getId() : (a.longestPathDist > b.longestPathDist);
    };
    std::set<Root, decltype(cmpRoot)> roots(cmpRoot);
    std::set<Root, decltype(cmpRoot)> upStreamRoots(cmpRoot);

    const auto hGraph = dynamic_cast<const HabanaGraph*>(m_graph);
    HB_ASSERT(hGraph != nullptr, "Expecting that graph is of type HabanaGraph");
    bool ioInDram = hGraph->getGraphAnnotation().memoryStrategyParams.dramInfo.IOsInDram;
    for (const NodePtr& rootNode : m_graph->getRootNodes())
    {
        if (ioInDram && isDmaDownstream(rootNode))
        { // when io is in Dram the Upstream nodes are not really roots - we can perform them eagerly and treat their consumers as roots.
            upStreamRoots.insert({rootNode, m_maxPathLengthFromNode[rootNode]});
            for (const NodePtr& rootConsumer : m_graph->getNodeConsumers(rootNode))
            {
                Root root = {rootConsumer, m_maxPathLengthFromNode[rootConsumer]};
                LOG_TRACE(SCHEDULER, "setting {} as root with longest path of {}", root.node->getNodeName(),
                          root.longestPathDist);
                roots.insert(root);
            }
        }
        else
        {
            Root root = {rootNode, m_maxPathLengthFromNode[rootNode]};
            LOG_TRACE(SCHEDULER, "setting {} as root with longest path of {}", root.node->getNodeName(),
                      root.longestPathDist);
            roots.insert(root);
        }
    }
    for (const Root& r : upStreamRoots)
    {
        schedule.push_back(r.node);
        m_whenVisited[r.node] = 0;
    }

    // for each root, create a schedule by using DFS topological sort
    int rootCount = 1;
    for (const Root& r : roots)
    {
        const NodePtr& root = r.node;
        if (m_whenVisited[root] != NEVER_VISITED) continue; // this can happen if the root is a consumer of a upstream node

        NodeSet visitedConsumers; // consumers for the current schedule (that were already scheduled in previous roots).
        NodeList rootSchedule = dfsScheduleFromRoot(root, rootCount, /* OUT */ visitedConsumers);
        rootCount++;

        if (schedule.empty() || visitedConsumers.empty())
        {
            // either this is the first root, or there are no consumers previously visited
            schedule.splice(schedule.end(), rootSchedule);
        }
        else // consumers for this root's schedule were already scheduled
        {
            // find first schedule consumer of the new root's schedule
            auto it = schedule.begin();
            for (; it != schedule.end(); it++)
            {
                if (visitedConsumers.count(*it)) break;
            }
            HB_ASSERT(it != schedule.end(), "visited nodes not found in schedule.");

            // insert the current root's schedule before that consumer
            schedule.splice(it, rootSchedule);
        }
    }
    // postpone dram fill nodes, and push forward dram spill nodes
    postponeDramFills(schedule);
    pushForwardDramSpills(schedule);


    // verify that all went well.
    verifySchedule(schedule);
    return schedule;
}

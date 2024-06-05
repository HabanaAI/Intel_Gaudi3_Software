#include "pass_manager.h"

#include "graph_visualization.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "log_manager.h"
#include "types_exception.h"
#include "utils/quant_info_dumper.h"

#include <lemon/connectivity.h>
#include <lemon/list_graph.h>

#include <queue>

typedef lemon::ListDigraph                  DirectedGraph;
typedef DirectedGraph::Node                 UntypedNode;
typedef DirectedGraph::NodeMap<pPass> NodeMap;


// DependencyGraphContainer defined and presented here to insure that
// PassManager only is familiar with the Lemon library
class DependencyGraphContainer
{

public:
    DependencyGraphContainer() :
    m_map(m_dependencyGraph)
    {}

    ~DependencyGraphContainer()
    {}

    DirectedGraph m_dependencyGraph; // Pure structure of a directed graph - the nodes and arcs and their connections.
                                     // It is un-typed. Actual computation will be performed on this DS

    NodeMap       m_map;             // All data that are assigned to the items of the graph (e.g. node labels, arc costs or capacities)
                                     // must be stored separately using lemon-maps

    struct UntypedNodeGroup
    {
        UntypedNode source;
        UntypedNode sink;
    };

    UntypedNodeGroup m_untypedNodes[PASS_ID_MAX_ID];  // Holds untyped nodes s.t. each node represents a pass
};

void PassManager::updateProgress(){
    m_currentProgress += 1;
}

void PassManager::printProgress(std::string_view msg) const
{
    int percent = m_currentProgress * 100 / m_registeredPassesCounter;
    LOG_DEBUG(PASS_MANAGER, "[{}%] {} - Done", percent, msg);
}

PassManager::PassManager()
: m_registeredPasses(),
  m_passGroupMapping(),
  m_registeredPassesCounter(0),
  m_graphVisualDebug(nullptr),
  m_state(PassMgrState::INIT),
  m_legacyMode(false)
{
    m_dependencyGraphContainer = std::make_unique<DependencyGraphContainer>();
    std::fill(m_registeredPasses.begin(), m_registeredPasses.end(), nullptr);
    std::fill(m_predicateState.begin(), m_predicateState.end(), false);
    std::fill(m_passGroupMapping.begin(), m_passGroupMapping.end(), PASS_ID_MAX_ID);
}

PassManager::~PassManager()
{
}

const std::vector<pPass>& PassManager::getExecutionList() const
{
    return m_executionList;
}

const std::vector<pPass>& PassManager::getExecutionOrder() const
{
    return m_actualExecutionOrder;
}

bool PassManager::reRunPass(PassId id)
{
    if (id >= PASS_ID_MAX_ID)
    {
        LOG_ERR(PASS_MANAGER, "Id {} is out of bounds! Ignoring rerun request.", id);
        return false;
    }

    pPass pass = m_registeredPasses[id];
    if (!pass)
    {
        LOG_WARN(PASS_MANAGER, "Unregistered pass {}. Ignoring the registration!", id);
        return false;
    }

    if (m_passRun[id])
    {
        // If the pass already ran we can deduce that its dependencies fulfilled
        if (m_reRunPriorityQueue.isPassEnqueued(pass))
        {
            LOG_TRACE(PASS_MANAGER, "Pass {} is in dynamic rerun execution list.", id);
        }
        else
        {
            LOG_TRACE(PASS_MANAGER, "Adding pass {} to dynamic rerun execution priority queue.", id);
            m_reRunPriorityQueue.push(pass);
        }
    }
    else
    {
        // The pass will be executed according to static execution order
        LOG_TRACE(PASS_MANAGER,"The pass {} is in execution list, PassManager will run it according to execution order.", id);
    }

    return true;
}

bool PassManager::turnOnPredicate(PredicateId id)
{
    if (state() != PassMgrState::RUNNING)
    {
        return false;
    }

    if (m_predicateTable.size() == 0)
    {
        LOG_WARN(PASS_MANAGER, "Predicate table is empty!");
        return false;
    }

    if (!m_predicateState[id])
    {
        LOG_TRACE(PASS_MANAGER, "Turning on predicate {}", id);
        m_predicateState[id] = true;
    }

    return true;
}

bool PassManager::handleActivePredicates()
{
    for (auto& pred: m_predicateTable)
    {

        if (m_predicateState[pred.first]) // Active predicate (was turned-on)
        {
            m_predicateState[pred.first] = false;

            LOG_TRACE(PASS_MANAGER, "Handle active predicate {}.", pred.first);

            // Run over all passes that bound to the given predicate
            for (PassId passId: pred.second)
            {
                auto search = m_passRun.find(passId);

                if (search != m_passRun.end())
                {
                    if (m_passRun[passId])
                    {
                        // If the pass already ran we may rerun it since its dependencies fulfilled
                        reRunPass(passId);
                    }
                    else
                    {
                        // The pass will be executed according to static execution order
                        LOG_TRACE(PASS_MANAGER, "Pass {} will be executed according to static execution order.",
                                  passId);
                    }
                }
                else
                {
                    LOG_ERR(PASS_MANAGER, "Unfamiliar pass: {}", passId);
                }
            }
        }
    }

    return true;
}

bool PassManager::registerPassToPredicate(pPass pass)
{
    // Register the pass to predicates that triggers its execution
    PassId    passId = pass->getId();

    for (PredicateId predId : pass->getPredicateSet())
    {
        // operator[] performing an insertion if predId does not already exist
        // same pass can't be mapped more than once to a predicate, m_predicateTable maps predId(key) to a set
        // Binding: predicate -> pass
        m_predicateTable[predId].insert(passId);

        LOG_TRACE(PASS_MANAGER, "Pass {} bound to predicate {}", passId, predId);
    }

    return true;
}

PassManager::PassManager(const PassManager& other)
{
    m_state                 = other.m_state;
    m_legacyMode            = other.m_legacyMode;
    m_testReplacementPasses = other.m_testReplacementPasses;
    if (state() == PassMgrState::INIT)
    {
        if (GCFG_ENABLE_GVD.value() || GCFG_ENABLE_PARTIAL_GVD.value() )
        {
            LOG_INFO(PASS_MANAGER, "GCFG_ENABLE_GVD is set");
            m_graphVisualDebug = std::make_unique<GraphVisualization>("_GVD_", true);
        }
        m_dependencyGraphContainer = std::make_unique<DependencyGraphContainer>();
    }
    else
    {
        m_graphVisualDebug = nullptr;
        // no use in a dependency graph if execution list had already been calculated
        m_dependencyGraphContainer = nullptr;
    }

    m_passRun                 = other.m_passRun;
    m_predicateState          = other.m_predicateState;
    m_predicateTable          = other.m_predicateTable;
    m_actualExecutionOrder    = other.m_actualExecutionOrder;
    m_reRunPriorityQueue      = other.m_reRunPriorityQueue;
    m_executionList           = other.m_executionList;
    m_registeredPassesCounter = other.m_registeredPassesCounter;
    m_passGroupMapping        = other.m_passGroupMapping;
    m_registeredPasses        = other.m_registeredPasses;
}

std::unique_ptr<PassManager> PassManager::clone() const
{
    return std::make_unique<PassManager>(*this);
}

bool PassManager::registerPass(pPass p)
{
    LOG_TRACE(PASS_MANAGER, "Registering pass. Pass ID {}: {}", p->getId(), p->getName());

    {
        const auto it = m_testReplacementPasses.find(p->getId());
        if (it != m_testReplacementPasses.end())
        {
            const auto& newPass = it->second.first;
            LOG_TRACE(PASS_MANAGER,
                      "Using test replacement pass {} id {} deps [{}] for pass {} id {} deps [{}]",
                      newPass->getName(),
                      newPass->getId(),
                      toString(newPass->getDependencySet(), ','),
                      p->getName(),
                      p->getId(),
                      toString(p->getDependencySet(), ','));

            HB_ASSERT(p->getId() == newPass->getId() && p->getDependencySet() == newPass->getDependencySet(),
                      "Pass {} cannot be replaced with pass {} due to Id or dependency set mismatch",
                      p->getName(),
                      newPass->getName());

            p = newPass;
            it->second.second = true;   // replacement used
        }
    }

    if (m_registeredPasses[p->getId()])
    {
        LOG_ERR(PASS_MANAGER, "Already registered this pass type! Pid:{}", p->getId());
        return false;
    }

    if (p->getId() > PASS_ID_MAX_ID || p->getId() <= PASS_ID_INVALID_ID)
    {
        LOG_ERR(PASS_MANAGER, "Invalid pass id! Pid: {}", p->getId());
        return false;
    }

    if (!m_legacyMode && p->getPriority() != PASS_DEF_PRIO)
    {
        LOG_ERR(PASS_MANAGER, "Pass static priorities are deprecated! Pid: {}", p->getId());
        return false;
    }

    m_passRun[p->getId()] = false;

    m_registeredPasses[p->getId()] = p;
    m_registeredPassesCounter++;

    if (p->isPassGroup())
    {
        // register group member ids
        auto pGroup = std::dynamic_pointer_cast<PassGroup>(p);
        HB_ASSERT_PTR(pGroup);
        for (PassId groupMember : pGroup->getGroupPasses())
        {
            PassId currentGroup = m_passGroupMapping[groupMember];
            if (currentGroup != PASS_ID_MAX_ID && (currentGroup != GROUP_ID_NO_GROUP || !m_legacyMode))
            {
                LOG_ERR(PASS_MANAGER,
                        "Cannot register pass {} into group {}, since it is already registered into {}",
                        groupMember,
                        p->getId(),
                        currentGroup);
                return false;
            }
            m_passGroupMapping[groupMember] = p->getId();
        }
        m_registeredPassesCounter++;  // if this is a pass-group, it contains start and sink nodes
    }

    return true;
}

void PassManager::addReplacementPass(pPass newPass)
{
    HB_ASSERT_PTR(newPass);
    LOG_TRACE(PASS_MANAGER, "Marking pass id {} for replacment with a custom pass", newPass->getId());
    m_testReplacementPasses[newPass->getId()] = {newPass, false};
}

void PassManager::addPassToDependencyGraph(const pPass& p)
{
    // 1.1 Add new node to the dependency graph (source and sink)
    UntypedNode untypedNodeSrc  = m_dependencyGraphContainer->m_dependencyGraph.addNode();
    UntypedNode untypedNodeSink = untypedNodeSrc;
    if (p->isPassGroup())  // if this is a sub-group, need to add also the sink node to the graph
    {
        untypedNodeSink                                    = m_dependencyGraphContainer->m_dependencyGraph.addNode();
        m_dependencyGraphContainer->m_map[untypedNodeSink] = p;
        m_dependencyGraphContainer->m_dependencyGraph.addArc(untypedNodeSrc, untypedNodeSink);
    }

    // 1.2 Binding: untyped node -> pass
    m_dependencyGraphContainer->m_map[untypedNodeSrc] = p;

    // 1.3 Save the node (untyped)
    m_dependencyGraphContainer->m_untypedNodes[p->getId()] = {untypedNodeSrc, untypedNodeSink};
}

bool PassManager::addPassDependenciesToGraph(const pPass& p)
{
    // Get the untyped node that bound to current pass
    UntypedNode target = m_dependencyGraphContainer->m_untypedNodes[p->getId()].source;

    LOG_TRACE(PASS_MANAGER, "Setting dependencies of node {}", p->getName());
    // 2.1 Set dependencies
    for (PassId id : p->getDependencySet())
    {
        // ID validation
        if (!m_registeredPasses[id]) continue;
        if (id >= PASS_ID_MAX_ID)
        {
            LOG_CRITICAL(PASS_MANAGER, "PassID: {} is not registered to pass manager.", id);
            return false;
        }
        // same group validation
        if (m_passGroupMapping[p->getId()] != m_passGroupMapping[id])
        {
            LOG_CRITICAL(PASS_MANAGER,
                         "PassID: {} is registered to group {} while its dependency {} is registered to {}",
                         p->getName(),
                         m_passGroupMapping[p->getId()],
                         m_registeredPasses[id]->getName(),
                         m_passGroupMapping[id]);
            return false;
        }
        // Get source node that bound
        UntypedNode source = m_dependencyGraphContainer->m_untypedNodes[id].sink;
        // Create directed arc
        m_dependencyGraphContainer->m_dependencyGraph.addArc(source, target);
    }
    return true;
}

bool PassManager::addPassGroupDependenciesToGraph(const pPass& p)
{
    PassId groupId = m_passGroupMapping[p->getId()];
    if (!m_registeredPasses[groupId])
    {
        LOG_CRITICAL(PASS_MANAGER, "Pass group with ID: {} is not registered to pass manager.", groupId);
        return false;
    }

    UntypedNode currentPassSource = m_dependencyGraphContainer->m_untypedNodes[p->getId()].source;
    UntypedNode currentPassSink   = m_dependencyGraphContainer->m_untypedNodes[p->getId()].sink;
    UntypedNode groupStart        = m_dependencyGraphContainer->m_untypedNodes[groupId].source;
    UntypedNode groupEnd          = m_dependencyGraphContainer->m_untypedNodes[groupId].sink;

    // add current pass to start after group start, and end before group end
    m_dependencyGraphContainer->m_dependencyGraph.addArc(groupStart, currentPassSource);
    m_dependencyGraphContainer->m_dependencyGraph.addArc(currentPassSink, groupEnd);
    if (p->isPassGroup())
    {
        // if current pass is a sub-group, set dependency between source and sink
        m_dependencyGraphContainer->m_dependencyGraph.addArc(currentPassSource, currentPassSink);
    }

    return true;
}

bool PassManager::constructDependencyGraph()
{
    LOG_TRACE(PASS_MANAGER, "Dependency graph construction.");
    bool allPassesAreValid = true;

    // 1. Each pass should be reduced to a node
    for (pPass p : m_registeredPasses)
    {
        if (!p) continue;
        addPassToDependencyGraph(p);
    }

    // 2. Each dependency should be reduced to a directed arc
    LOG_TRACE(PASS_MANAGER, "Setting dependencies.");
    for (pPass p: m_registeredPasses)
    {
        if (!p) continue;

        // add pass dependencies to graph
        allPassesAreValid &= addPassDependenciesToGraph(p);

        // add pass group dependencies (if belongs to a pass group)
        if (allPassesAreValid && m_passGroupMapping[p->getId()] != PASS_ID_MAX_ID)
        {
            allPassesAreValid &= addPassGroupDependenciesToGraph(p);
        }
    }

    if (!allPassesAreValid)
    {
        LOG_ERR(PASS_MANAGER, "Invalid passes detected - cannot construct dependency graph");
        return false;
    }
    // Test post condition. dependency graph must be acyclic
    if (lemon::dag(m_dependencyGraphContainer->m_dependencyGraph))
    {
        LOG_TRACE(PASS_MANAGER, "Successfully constructed directed acyclic dependency graph!");
        return true;
    }
    else
    {
        //TODO: find and print the actual cycle. DFS may be used
        LOG_ERR(PASS_MANAGER, "Detected a cycle!");
        return false;
    }
}

void PassManager::setRerunPassPriorities()
{
    constexpr unsigned maxPriority = PASS_MAX_PRIO + PASS_ID_MAX_ID;

    for (unsigned i = 0; i < m_executionList.size(); i++)
    {
        const pPass& p = m_executionList[i];
        // legacy priority
        if (m_passGroupMapping[p->getId()] == PASS_ID_MAX_ID || m_passGroupMapping[p->getId()] == GROUP_ID_NO_GROUP)
        {
            continue;
        }

        p->setPriority(maxPriority - i);
    }
}

// The execution order is calculated by iteratively adding to a list, the next pass with highest priority with
// all dependency passes already in the list.
// This is done using a temporary array (din) that holds the number of incoming arcs (dependencies) for
// each node (pass). This array gets updated whenever a pass is added to the list, by reducing the value of
// the respective dependents.
// When a pass gets to din == 0, it is pushed to a priority queue. In each iteration, the next pass to add
// to the list is taken from this queue.
bool PassManager::computeExecutionOrder()
{
    auto cmp = [&](const UntypedNode& n1, const UntypedNode& n2) {
        const pPass& p1 = m_dependencyGraphContainer->m_map[n1];
        const pPass& p2 = m_dependencyGraphContainer->m_map[n2];
        return p1->getPriority() < p2->getPriority();
    };
    using PassPrioQueue = std::priority_queue<UntypedNode, std::vector<UntypedNode>, decltype(cmp)>;
    using NodeIt = lemon::ListDigraph::NodeIt;
    using ArcIt = lemon::ListDigraph::OutArcIt;
    using InDegreeMap   = DirectedGraph::NodeMap<unsigned>;

    LOG_TRACE(PASS_MANAGER, "Computing passes execution order");

    PassPrioQueue prioNodes(cmp);
    InDegreeMap   din(m_dependencyGraphContainer->m_dependencyGraph);

    LOG_TRACE(PASS_MANAGER, "Collecting passes without dependencies");
    for (NodeIt n(m_dependencyGraphContainer->m_dependencyGraph); n != lemon::INVALID; ++n)
    {
        din[n] = lemon::countInArcs(m_dependencyGraphContainer->m_dependencyGraph, n);
        if (din[n] == 0)
        {
            prioNodes.push(n);
        }
    }

    HB_ASSERT(!prioNodes.empty(), "No passes without dependencies found. Probably a circular dependency.");

    LOG_TRACE(PASS_MANAGER, "Sorting passes topologically adhering to priority");
    while (!prioNodes.empty())
    {
        UntypedNode nextNode = prioNodes.top();
        pPass       pass     = m_dependencyGraphContainer->m_map[nextNode];
        m_executionList.push_back(pass);
        prioNodes.pop();

        for (ArcIt arc(m_dependencyGraphContainer->m_dependencyGraph, nextNode); arc != lemon::INVALID; ++arc)
        {
            UntypedNode depNode = m_dependencyGraphContainer->m_dependencyGraph.target(arc);
            int         newDin  = --din[depNode];
            if (newDin == 0)
            {
                prioNodes.push(depNode);
            }
        }
    }

    if (m_registeredPassesCounter != m_executionList.size())
    {
        LOG_ERR(PASS_MANAGER, "Pass execution list should contain {} passes, but contains {}.", m_registeredPassesCounter,
                m_executionList.size());
        return false;
    }
    for (pPass registeredPass : m_registeredPasses)
    {
        if (registeredPass != nullptr &&
            std::find(m_executionList.begin(), m_executionList.end(), registeredPass) == m_executionList.end())
        {
            LOG_ERR(PASS_MANAGER, "Pass {} registered but wasn't scheduled.", registeredPass->getName());
            return false;
        }
    }

    return true;
}

pPass PassManager::getNextPass(std::vector<pPass>::iterator& executionListIterator)
{
    pPass reRunPass    = nullptr;
    pPass nextExecPass = nullptr;

    if (!m_reRunPriorityQueue.empty())
    {
        reRunPass = m_reRunPriorityQueue.top();
    }

    if (executionListIterator != m_executionList.end())
    {
        nextExecPass = *executionListIterator;
    }

    if (reRunPass && nextExecPass)
    {
        if (PassCompare::cmp(reRunPass, nextExecPass))
        {
            LOG_TRACE(PASS_MANAGER, "Returning pass from execution list (with higher priority than next re-run pass).");
            updateProgress();
            executionListIterator++;
            return nextExecPass;
        }
        else
        {
            LOG_TRACE(PASS_MANAGER, "Returning pass from rerun execution list (with higher or equal priority to next in"
                            " execution list).");
            m_reRunPriorityQueue.pop();
            return reRunPass;
        }
    }
    else if (reRunPass)
    {
        LOG_TRACE(PASS_MANAGER, "Returning pass from rerun execution list");
        m_reRunPriorityQueue.pop();
        return reRunPass;
    }
    else if (nextExecPass)
    {
        LOG_TRACE(PASS_MANAGER, "Returning pass from execution list");
        updateProgress();
        executionListIterator++;
    }
    else
    {
        LOG_TRACE(PASS_MANAGER, "No passes left to return - Done.");
    }
    return nextExecPass;
}

void PassManager::executeGVDPass(HabanaGraph& graph, const std::string& passName, int passIdx, bool isGraphChanged)
{
    if (m_graphVisualDebug != nullptr)
    {
        int64_t passFilter = GCFG_GVD_PASS_FILTER.value();
        if (passFilter != -1 && passFilter != passIdx && passFilter != passIdx+1) // run before and after pass filter.
        {
            return;
        }
        if (GCFG_ENABLE_PARTIAL_GVD.value() && !isGraphChanged) return;
        std::string prefix = std::to_string(passIdx) + "_" + passName;
        m_graphVisualDebug->setFileName(prefix);
        m_graphVisualDebug->Apply(graph);
    }
}

#define COND_TIMER(COMMAND)                                                                                            \
    if (unlikely(LOG_LEVEL_AT_LEAST_TRACE(PASS_MANAGER)))                                                              \
    {                                                                                                                  \
        COMMAND;                                                                                                       \
    }

void PassManager::advanceState(PassMgrState newState)
{
    switch (state())
    {
        case PassMgrState::INIT:
        {
            HB_ASSERT(PassMgrState::READY == newState, "Only valid transition is INIT->READY");
            m_state = newState;
            break;
        }
        case PassMgrState::READY:
        {
            HB_ASSERT(PassMgrState::RUNNING == newState, "Only valid transition is READY->RUNNING");
            m_state = PassMgrState::RUNNING;
            break;
        }
        case PassMgrState::RUNNING:
        {
            HB_ASSERT(PassMgrState::DONE == newState, "Only valid transition is RUNNING->DONE");
            m_state = PassMgrState::DONE;
            break;
        }
        case PassMgrState::DONE:
        {
            break;  // state is not interesting once done, noop
        }
        default:
        {
            HB_ASSERT(false, "Unknown state {}", state());
        }
    }
}

PassManager::PassMgrState PassManager::state() const
{
    return m_state;
}

template<bool IsPartial>
bool PassManager::executePasses(HabanaGraph& graph, std::optional<PassId> stopBefore)
{
    HB_ASSERT(state() >= PassMgrState::READY, "Expecting static exec order to have already been calculated");
    if (state() == PassMgrState::DONE)
    {
        LOG_INFO(PASS_MANAGER, "Pass manager finished executing passes...");
        return true;
    }

    std::vector<pPass>::iterator executionListIterator;
    if constexpr (!IsPartial)
    {
        executionListIterator = m_executionList.begin();
        if ((GCFG_ENABLE_GVD.value() || GCFG_ENABLE_PARTIAL_GVD.value()) && m_graphVisualDebug == nullptr)
        {
            LOG_INFO(PASS_MANAGER, "GCFG_ENABLE_GVD is set");
            m_graphVisualDebug = std::make_unique<GraphVisualization>("_GVD_", true);
        }
        advanceState(PassMgrState::RUNNING);  // READY-->RUNNING
    }
    else
    {
        executionListIterator = std::find_if(m_executionList.begin(), m_executionList.end(), [this](const auto& pass) {
            return pass->getId() == m_actualExecutionOrder.back()->getId();
        });
        executionListIterator++;  // start execution from after the last executed pass
    }

    HB_ASSERT(executionListIterator < m_executionList.end(), "Expecting valid exec list iterator");
    pPass p = nullptr;

    int currPassIdx = std::distance(m_executionList.begin(), executionListIterator);
    LOG_TRACE(PASS_MANAGER, "Recipe name: {}", graph.getRecipeName());
    LOG_TRACE(PASS_MANAGER, "Executing passes...");
    while ((p = getNextPass(executionListIterator)))
    {
        if constexpr (IsPartial)
        {
            if (stopBefore.has_value() && p->getId() == stopBefore.value())
            {
                // mark the pass as executed for next run to continue after it
                m_actualExecutionOrder.push_back(p);
                // avoid execution of this pass and abort
                break;
            }
        }

        LOG_INFO(PASS_MANAGER, "Executing pass: {}", p->getName());
        try
        {
            COND_TIMER(graph.m_timer.start(p->getName()));
            m_actualExecutionOrder.push_back(p);
            graph.clearGraphChangedInLastPass();

            if (!p->Apply(graph))
            {
                if constexpr (!IsPartial)
                {
                    executeGVDPass(graph, p->getName(), currPassIdx, graph.isGraphChangedInLastPass());
                }
                LOG_ERR(PASS_MANAGER, "Graph optimization failed pass: {}", p->getName());
                return false;
            }

            if (!handleActivePredicates())
            {
                LOG_ERR(PASS_MANAGER, "Predicates handling failed (following pass: {})", p->getName());
                return false;
            }

            if constexpr (!IsPartial)
            {
                executeGVDPass(graph, p->getName(), currPassIdx, graph.isGraphChangedInLastPass());
                graph.dumpGraphToJson(graph_serializer::GraphState::POST_PASS, p->getName());
                printProgress(p->getName());
            }

            COND_TIMER(graph.m_timer.stop(p->getName()));
            COND_TIMER(LOG_TRACE(PASS_MANAGER,
                                 "Total time for {}: {} seconds",
                                 p->getName(),
                                 graph.m_timer.getTotalSeconds(p->getName())));

            currPassIdx++;
            m_passRun[p->getId()] = true;
        }
        catch(SynapseException& e)
        {
            COND_TIMER(graph.m_timer.stop(p->getName()));
            LOG_ERR(PASS_MANAGER, "Graph optimization failed! Got SynapseException: {}", e.what());
            return false;
        }
    }

    if constexpr (!IsPartial)
    {
        if (GCFG_DUMP_QUANT_INFO.value())
        {
            dumpQuantInfoToJson(graph);
        }
    }
    return true;
}

void PassManager::printStaticOrder() const
{
    int                        passIndex = 0;
    std::unordered_set<PassId> groupCount;
    for (pPass p : m_executionList)
    {
        if (p)
        {
            std::string name = p->getName();
            if (p->isPassGroup())
            {
                bool inserted = groupCount.insert(p->getId()).second;
                name += inserted ? "_start" : "_end";
            }
            LOG_TRACE(PASS_MANAGER, "Static pass execution Index: {}. PassName: {}", passIndex++, name);
        }
        else
        {
            LOG_ERR(PASS_MANAGER, "PassExecution Index: {}. nullptr!", passIndex++);
        }
    }
}

void PassManager::printTheRegisteredPassesInSet(const PassIDSet& set) const
{
    for (const auto dep : set)
    {
        if (m_registeredPasses[dep])
        {
            LOG_TRACE(PASS_MANAGER, "{}", m_registeredPasses[dep]->getName());
        }
    }
}

void PassManager::printPassesDependencies() const
{
    // Print passes dependencies to log. The prints are parsed by this tool: "pass_dependencies_graph.py"
    // for the creation of pass dependencies graph.
    if (!LOG_LEVEL_AT_LEAST_TRACE(PASS_MANAGER)) return;
    LOG_TRACE(PASS_MANAGER, "Print passes/groups and their dependencies");
    for (const auto& registeredPass : m_registeredPasses)
    {
        if (registeredPass)
        {
            const auto groupId = m_passGroupMapping[registeredPass->getId()];
            // print pass/group's dependencies
            LOG_TRACE(PASS_MANAGER,
                      "{}: {}{} depends on {} registered passes/groups",
                      registeredPass->isPassGroup() ? "Group" : "Pass",
                      registeredPass->getName(),
                      groupId != PASS_ID_MAX_ID ? " belongs to group: " + m_registeredPasses[groupId]->getName() : "",
                      std::count_if(registeredPass->getDependencySet().begin(),
                                    registeredPass->getDependencySet().end(),
                                    [this](const PassId& dep) { return m_registeredPasses[dep] != nullptr; }));
            printTheRegisteredPassesInSet(registeredPass->getDependencySet());
        }
    }
    LOG_TRACE(PASS_MANAGER, "Print passes/groups and their dependencies finished");
}

bool PassManager::prepareForExecution()
{
    // Verify that if we had replacements, all of them wre used
    for (const auto& v : m_testReplacementPasses)
    {
        const auto& orignalPassId  = v.first;
        const auto& newPass        = v.second.first;
        const bool  replacemntUsed = v.second.second;
        if (!replacemntUsed)
        {
            LOG_ERR(PASS_MANAGER,
                    "Requested replacement for pass id {} with pass {} wasn't applied!",
                    orignalPassId,
                    newPass->getName());
            return false;
        }
    }

    // Bind passes to their predicates
    for (pPass p : m_registeredPasses)
    {
        if (p)
        {
            registerPassToPredicate(p);
        }
    }
    // Print pass-predicate bindings
    for (auto& it : m_predicateTable)
    {
        for (PassId id : it.second)
        {
            LOG_TRACE(PASS_MANAGER, "Pass {} bound to predicate: {}", id, it.first);
        }
    }

    // 1. Reduce registered passes to dependency graph
    if (!constructDependencyGraph())
    {
        LOG_ERR(PASS_MANAGER, "Failed to create dependency graph! Can't compute execution order!");
        return false;
    }

    // 2. Compute the execution order
    if (!computeExecutionOrder())
    {
        LOG_ERR(PASS_MANAGER, "Failed to calculate execution order.");
        return false;
    }

    // 2.1 Print static execution order
    printStaticOrder();

    // 2.2 Set re-run priority
    setRerunPassPriorities();
    return true;
}

bool PassManager::run(HabanaGraph& graph)
{
    LOG_TRACE(PASS_MANAGER, "There are {} registered passes", m_registeredPassesCounter);

    printPassesDependencies();
    if (!prepareForExecution())
    {
        return false;
    }
    advanceState(PassMgrState::READY);  // INIT-->READY
    //3. Execute passes
    bool executionResult = executePasses<false /*IsPartial*/>(graph);
    advanceState(PassMgrState::DONE);  // RUNNING-->DONE
    HB_ASSERT(state() == PassMgrState::DONE, "Expecting state DONE, actual {}", state());

    LOG_TRACE(PASS_MANAGER, "Printing pass execution order.");

    int i = 0;

    for (pPass p: getExecutionOrder())
    {
        LOG_TRACE(PASS_MANAGER, "{}: {}", i++, p->getName());
    }


    return executionResult;
}

// executes the following pass IDs (last executed pass, stopBefore)
bool PassManager::runPartial(HabanaGraph& graph, PassId stopBefore)
{
    if (m_executionList.empty()) return true;  // no passes to run
    // make sure stopBefore exists in executionList
    const auto stopBeforeIt =
        std::find_if(m_executionList.begin(), m_executionList.end(), [&stopBefore](const auto& pass) {
            return pass != nullptr && pass->getId() == stopBefore;
        });
    HB_ASSERT(stopBeforeIt != m_executionList.end(), "Expecting passId {} in pass exec list", stopBefore);

    if (!handleActivePredicates())
    {
        LOG_ERR(PASS_MANAGER, "Failed handling predicates before running passes up to {}", stopBefore);
        return false;
    }

    const bool success = executePasses<true /*IsPartial*/>(graph, stopBefore);
    return success;
}

bool PassCompare::operator()(const pPass hp1, const pPass hp2) const
{
    return cmp(hp1, hp2);
}

bool PassCompare::cmp(const pPass hp1, const pPass hp2)
{
    return hp1->getPriority() < hp2->getPriority();
}

bool PassPriorityQueue::isPassEnqueued(const pPass hp) const
{
    return std::find(c.begin(), c.end(), hp) != c.end();
}

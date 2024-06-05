#pragma once
#include <vector>
#include <string>
#include <functional>
#include "habana_pass.h"
#include "timer.h"
#include <unordered_map>
#include <queue>
#include <optional>

class DependencyGraphContainer;
class GraphVisualization;

// Enum cannot be used as a key of std::unordered_map. This is a defect fixed in c++ 14.
// As a workaround an enum type may be used since the values of the enum constants
// are values of an integral type.
typedef std::unordered_map<int, PassIDSet> PredicateHashMap;
typedef std::unordered_map<int, bool>      PassRunHashMap;
typedef std::array<PassId, PASS_ID_MAX_ID> PassIDMap;

class PassCompare
{
public:
    bool operator()(const pPass hp1, const pPass hp2) const;
    static bool cmp(const pPass hp1, const pPass hp2);
};

class PassPriorityQueue : public std::priority_queue<pPass, std::vector<pPass>, PassCompare>
{
public:
    virtual bool isPassEnqueued(const pPass) const;
};

class PassManager
{
public:

    PassManager();

    ~PassManager();

    PassManager(const PassManager& other);
    PassManager operator=(const PassManager& other) = delete;
    PassManager(PassManager&& other)                = delete;
    PassManager operator=(PassManager&& other) = delete;

    std::unique_ptr<PassManager> clone() const;

    // Register a pass with the manager
    bool registerPass(pPass p);

    void addReplacementPass(pPass newPass);

    // allow legacy features like pass priorities and non group passes
    void setLegacyMode() { m_legacyMode = true; }

    // Compute passes execution order and execute all registered passes
    bool run(HabanaGraph& graph);

    // Assuming pass manager had already initialized execution schedule,
    // proceeds execution from the next pass and up to stopBefore, excluding it.
    // Note: Sets the stopBefore pass as executed, assuming the caller executes it
    bool runPartial(HabanaGraph& graph, PassId stopBefore);

    // Enables to the user of pass manager to affect the actual execution order by requesting pass rerun
    bool reRunPass(PassId id);

    // Returns the execution list
    const std::vector<pPass>& getExecutionList() const;

    // Pass Manager should support multiple execution of passes, the execution list does not contain information about
    // multiple execution.
    // Pre condition: should be called after PassManager::run(HabanaGraph& graph)
    const std::vector<pPass>& getExecutionOrder() const;

    // Enables to the user of pass manager to affect the dynamic execution order
    // Each predicate may trigger passes multiple execution
    bool turnOnPredicate(PredicateId id);

private:
    enum class PassMgrState
    {
        INIT,     // initial state, no exec schedule yet
        READY,    // static exec schedule and rerun priorities calculated
        RUNNING,  // started executing passes
        DONE      // all passes executed
    };
    void         advanceState(PassMgrState newState);
    PassMgrState state() const;

    // Performs the reduction i.e. constructs the dependency graph:
    // 1. Each pass is represented by a node, each pass group is represented by 2 nodes (source and sink)
    // 2. If pass A is in the dependency set of pass B, a directed arc from A to B will be created.
    bool constructDependencyGraph();

    void addPassToDependencyGraph(const pPass& p);

    bool addPassDependenciesToGraph(const pPass& p);

    bool addPassGroupDependenciesToGraph(const pPass& p);

    void setRerunPassPriorities();

    // Performs a topological sort and returns an execution list
    // Pre condition: The dependency graph must be directed and acyclic
    bool computeExecutionOrder();

    // Apply all passes or up to stopAfter passId including it, if exists
    template<bool IsPartial>
    bool executePasses(HabanaGraph& graph, std::optional<PassId> stopBefore = std::nullopt);

    bool registerPassToPredicate(pPass pass);

    bool handleActivePredicates();

    pPass getNextPass(std::vector<pPass>::iterator& executionListIterator);

    void executeGVDPass(HabanaGraph& graph, const std::string& passName, int passIdx, bool isGraphChanged);

    void updateProgress();

    // calc static pass order and priorities
    bool prepareForExecution();

    void printProgress(std::string_view msg) const;

    void printStaticOrder() const;

    void printTheRegisteredPassesInSet(const PassIDSet& set) const;

    void printPassesDependencies() const;

    std::array<pPass, PASS_ID_MAX_ID>  m_registeredPasses;           // Holds all passes that registered to PassManager

    PassIDMap m_passGroupMapping;  // holds the mapping (pass) --> (pass group)

    unsigned int                       m_registeredPassesCounter;    // For debug

    std::unique_ptr<DependencyGraphContainer> m_dependencyGraphContainer;  // Contains the dependency graph
                                                                           // (lemon interface)

    std::vector<pPass>                 m_executionList;              // Passes in execution order

    PassPriorityQueue                  m_reRunPriorityQueue;         // Passes that already ran and should run again

    std::vector<pPass>                 m_actualExecutionOrder;       // Due to passes multiple execution the execution
                                                                     // list not necessarily represent the actual
                                                                     // execution order.

    PredicateHashMap                   m_predicateTable;             // For each predicate holds a set of passes that
                                                                     // predicate is true
                                                                     //Complexity(search,insert): O(1)(average),
                                                                     // worst case - O(n)

    std::array<bool , PREDICATE_ID_MAX_ID> m_predicateState;         // For each predicate holds its state (on/off)

    PassRunHashMap                     m_passRun;                    // Complexity(search,insert): O(1)(average),
                                                                     // worst case - O(n)
    std::unique_ptr<GraphVisualization> m_graphVisualDebug;

    int m_currentProgress = {};
    PassMgrState m_state;
    bool         m_legacyMode;

    // Map from replaced pass ID to the new pass and a bool to track if it's been replaced
    std::map<int, std::pair<pPass, bool>> m_testReplacementPasses;
};

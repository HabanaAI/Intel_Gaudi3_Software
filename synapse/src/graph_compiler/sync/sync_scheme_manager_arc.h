#pragma once

#include <bitset>
#include <functional>
#include "habana_device_types.h"
#include "types.h"
#include "sync/sync_types.h"
#include "node.h"
#include "node_roi.h"

class HabanaGraph;

class SyncSchemeManagerArc
{
public:
    SyncSchemeManagerArc(HabanaGraph* graph) : m_graph(graph) {}
    virtual ~SyncSchemeManagerArc() = default;

    void go();
    void print() const;

protected:
    using NumEngsMap  = std::map<unsigned, unsigned>;  // key = logical id, value = num physical engines
    using ResumePoint = std::pair<NodeList::const_iterator, unsigned>;  // node iterator, pipeline level

    class SobResetManager
    {
    public:
        SobResetManager();
        void bindSafeIncrement(std::function<unsigned(unsigned, unsigned, unsigned)> func) { f_safeIncrement = func; }
        bool registerRois(const std::list<NodeROI*>&      rois,
                          const NodeList::const_iterator& itrNode,
                          unsigned                        logicalId,
                          unsigned                        pipeLevel);

        void setCmeExist() { m_isCmeExist = true; }
        ResumePoint doReset(const HabanaGraph& g, const NumEngsMap& engsMap);

        unsigned getHighestNodeExeIndexAtReset() const { return m_highestNodeExeIndexAtReset; }

    private:
        struct LogicalEngCtx
        {
            unsigned accumSignal   = 0;
            unsigned maxRoiPending = 0;
        };
        struct LogicalEngBalanceInfo
        {
            NodeList::const_iterator itrNode;
            unsigned                 pipeLevel  = -1;
            bool                     isBalanced = false;
        };

        unsigned getMaxRelIdx(const std::list<NodeROI*>& rois) const;
        unsigned getAccumSignals(const std::list<NodeROI*>& rois) const;
        void     makeSureWeCanDoTheReset() const;

        const unsigned                            c_signalLimit;
        std::map<unsigned, LogicalEngCtx>         m_engineCtx;         // key is logical ID
        std::map<unsigned, LogicalEngBalanceInfo> m_currentBalance;    // key is logical ID
        std::map<unsigned, LogicalEngBalanceInfo> m_fullyBalancedPin;  // key is logical ID
        std::bitset<8>                            m_allBalanced;
        std::bitset<8>                            m_logicalIdsInPrevReset;
        unsigned                                  m_highestNodeExeIndexAtReset = 0;
        unsigned                                  m_sobResetId = 0;
        bool                                      m_isCmeExist = false;

        std::function<unsigned(unsigned, unsigned, unsigned)> f_safeIncrement;
    };  // class SobResetManager

    bool          createNodePipelineSyncs(const NodeList::const_iterator& itrNode, unsigned startPipeline);
    DependencyMap nodeSetToDepMap(const NodeSet& nodes) const;
    unsigned      getMaxOverlapSigIdxForNodeToDependOn(const NodePtr& node) const;
    ResumePoint   resetSobs(NodeList::const_iterator itrCurrNode);  // returns the resume point in the graph

    virtual DependencyMap getDependenciesOnControlEdges(const NodePtr& node) const;

    virtual void createRoiPipelineSyncs(const NodePtr&             node,
                                        const std::list<NodeROI*>& rois,
                                        const DependencyMap&       inputDependencies,
                                        DependencyMap&             outputDependencies,
                                        unsigned&                  emittedSigVal) = 0;

    virtual void              resetOverlap()                                                       = 0;
    virtual void              archiveOverlap()                                                     = 0;
    virtual unsigned          getLogicalId(const NodePtr& node) const                              = 0;
    virtual unsigned          convertSigValToOverlapIdx(unsigned logicalId, unsigned sigVal) const = 0;
    virtual const NumEngsMap& getNumEngsPerLogicalId() const                                       = 0;
    virtual std::string       logicalIdToStr(unsigned logicalId) const                             = 0;
    virtual std::string       sigValToStr(unsigned logicalId, Settable<unsigned> sigVal) const     = 0;

    // Reduce the two input dependency maps into a single map with the most restrictive dependencies
    static DependencyMap getReducedDependencies(const DependencyMap& map1, const DependencyMap& map2);

protected:
    HabanaGraph*                m_graph;
    SobResetManager             m_sobResetMngr;
    std::map<unsigned, NodePtr> m_prevNodePerEng;  // key = engine logical ID, value = last node
    NodePtr                     m_prevNode      = nullptr;
    unsigned                    m_breakpointCtr = 0;
    static const uint32_t       c_maxBreakpoint = (1 << 30) - 1;  // we have 30 bits for the breakpoint value
};

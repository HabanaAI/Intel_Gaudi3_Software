#pragma once

#include <memory>
#include <list>
#include <map>
#include <set>
#include <utility>
#include "node_annotation.h"
#include "include/sync/overlap.h"
#include "sync_object_manager.h"
#include "habana_graph.h"
#include "sync_types.h"
#include "sync_conventions.h"

constexpr unsigned MAX_VALUE_FOR_DBG_CTR = std::numeric_limits<int16_t>::max();  // 32767
constexpr uint32_t INVALID_ENGINE_ID = std::numeric_limits<uint32_t>::max();

class Node;

class SyncSchemeManager
{
protected:
    typedef unsigned int PipelineLevel;
    typedef SyncObjectManager::MonitorId MonitorId;
    typedef SyncObjectManager::SyncId SyncId;
    typedef uint32_t queue_id;
    typedef uint32_t logical_engine_id;

    class Sync
    {
    public:
        Sync()                                     :                m_value(-1)  {}
        Sync(SyncId id)                            : m_ids(1, id) , m_value(-1)  {}
        Sync(std::vector<SyncId> ids)              : m_ids(ids)   , m_value(-1)  {}
        Sync(SyncId id, int16_t val)               : m_ids(1, id) , m_value(val) {}
        Sync(std::vector<SyncId> ids, int16_t val) : m_ids(ids)   , m_value(val) {}

        void     setValue(int16_t v)      { m_value = v; }
        int16_t  getValue() const         { return m_value; }
        bool     isValueValid() const     { return m_value >= 0; }
        void     addId(SyncId id)         { m_ids.push_back(id); }
        SyncId   getId(unsigned i) const  { return m_ids.at(i); }
        size_t   getNumIds() const        { return m_ids.size(); }
        bool     hasId(SyncId id) const   { return std::find(m_ids.begin(), m_ids.end(), id) != m_ids.end(); }

        const std::vector<SyncId>&  getAllIds() const { return m_ids; }

        bool operator==(const Sync& rhs) const { return m_ids == rhs.m_ids && m_value == rhs.m_value; }

        bool operator<(const Sync& rhs) const
        {
            if (m_ids < rhs.m_ids) return true;
            if (m_ids > rhs.m_ids) return false;
            return m_value < rhs.m_value;
        }

    private:
        std::vector<SyncId>  m_ids;   // holds group IDs for gaudi/goya2 and sync id for goya1
        int16_t              m_value; // same value for all IDs
    };

    struct IndexAndPipeLevel
    {
        explicit IndexAndPipeLevel(unsigned int index = 0, PipelineLevel pl = 0)
                : index(index), pipeLevel(pl)
        {}

        bool operator<(const IndexAndPipeLevel& rhs) const
        {
            return index != rhs.index ? index < rhs.index : pipeLevel < rhs.pipeLevel;
        }

        unsigned int index = 0;
        PipelineLevel pipeLevel = 0;
    };

public:
    SyncSchemeManager(HabanaGraph* graph, SyncConventions& syncConventions);

    virtual ~SyncSchemeManager();

    virtual void runNodesSyncScheme();

    virtual void printSyncScheme() const;


protected:
    virtual void _fillNodeSyncScheme(NodePtr node);

    virtual void _resetSyncIdsSequentialIO(bool bActivate = false);

    virtual void _monitorPipelineSyncs(NodePtr monitorNode);

    virtual void _monitorRoiPipelineSyncs(NodePtr                        monitorNode,
                                          const std::list<NodeROI*>&     roi,
                                          std::list<MonObject>&          monitors,
                                          std::pair<unsigned, unsigned>* pForcedDependency = nullptr) = 0;

    virtual unsigned numEnginesByLogicalEngine(unsigned engineId) const = 0;

    /**
     * If node support pipeline level (TPC / MME) it will be inserted to pipeline monitors
     * Else it will be inserted to pre exe monitors
     */
    virtual void _addFenceAggMonitors(const NodePtr&            monitorNode,
                                      const IndexAndPipeLevel&  enginePipeLevel,
                                      std::list<MonObject>&     monitors,
                                      bool                      postExec);

    /**
     * Signal for node job finished
     */
    virtual void _addNodeSyncs(NodePtr node);

    virtual void _addPipelinedSync(NodePtr       node,
                                   unsigned int  engineIndex,
                                   Sync&         sync,
                                   PipelineLevel pipeStage,
                                   int16_t       incVal,
                                   int32_t       numSignalsForDbg);

    virtual void
    _addCpPipelinedSync(NodePtr node, unsigned int engineIndex, const SyncObject& sync, PipelineLevel pipeStage);

    virtual void _getNodeInputsProducers(NodePtr node, NodeSet& producers) const;

    /**
     * Get sync ID for device
     * DMA up and DMA down has reserved sync ID
     */
    virtual std::vector<SyncId> getDeviceSyncId(queue_id engineId);

    virtual Sync& _createAndGetSync(logical_engine_id engineId, int16_t incValue);

    virtual void _handleProducers(NodePtr node);

    virtual void _handleConsumers(NodePtr node);

    virtual int16_t _getSyncValueIncrement() const;

    virtual unsigned int _numberOfEngines(NodePtr node) const;

    virtual int16_t _getSyncValue(SyncId syncId, unsigned int syncNodeId, PipelineLevel pipeLevel) const;

    virtual Sync _findSyncForValueByLogicalEngine(queue_id engineId, unsigned overlapValue) const;

    virtual Sync* _findSyncById(SyncId id, logical_engine_id hint = INVALID_ENGINE_ID);

    virtual void _initializeFirstNodes(const NodePtr& node);

    virtual void _addInitialSyncWaits();

    virtual logical_engine_id _getLogicalEngineID(const NodePtr& node, unsigned int engineIdx) const = 0;

    /**
    * Add setup monitors to the reserved monitors ids
    */
    void _initReservedMonitors();

    void _printNodeSyncScheme(NodePtr node) const;

    SyncInteraction&  engineSyncScheme(NodePtr node, unsigned int engineIndex) const;

    void addPipelineMon(NodePtr node, unsigned int engineIndex, const MonObject& monitor, unsigned int pipeStage);

    void addMonitorToNode(NodePtr node, unsigned int engineIndex, const MonObject& monitor, bool pushFront = false, bool postExec = false);

    virtual unsigned _getSyncIdIncrement(unsigned physEngineIdx, unsigned logicalEngine) const;

    void _pushResetAllSyncs(std::list<SyncOrMonitor>& ret) const;

    void _pushResetEngineSemaphors(const std::vector<HabanaDeviceType>& deviceTypes,
                                   std::list<SyncOrMonitor>&            ret) const;

    uint8_t _numEnginesToMask(unsigned numEngines) const;

    virtual unsigned numSignalingEngines(unsigned engineId) const = 0;

    void _generateBlockingNodesMonitors(NodePtr monitorNode, std::list<MonObject>& monitors);

    template<typename T>
    void addForcedDependency(T& dependency, std::pair<unsigned, unsigned>* pForcedDependency) const;

    template <typename T, typename LQ>
    void addMonitorsByOverlap(const std::list<LQ>&                 logicalEngines,
                              MonObject&                           mon,
                              std::list<MonObject>&                monitors,
                              T&                                   dependency) const;

    uint32_t syncIdAndValueToSignalId(queue_id engineId, SyncId syncId, uint16_t value) const;

    template <typename T>
    void removeNodeInternalDependencies(const NodePtr& node,
                                        unsigned       pipelineLevel,
                                        T&             ctx) const;

    uint32_t getMaxSignalIdForEngineToBeDependentOn(const NodePtr& node) const;

    virtual bool      shouldWaitForActiveEngines() const;
    virtual bool      shouldPredicateTheMonSetup() const;
    virtual bool      shouldWaitForLogicalQueue(uint32_t logicalQue) const = 0;
    virtual uint32_t  getCompletionLogicalQueue() const = 0;
    virtual uint32_t  getFinalSyncsQueueId() const = 0;
    virtual unsigned  getOverlapNumEngines() const = 0;

    virtual const std::vector<HabanaDeviceType>& getPlatformDeviceTypes() const = 0;

protected:
    HabanaGraph* m_graph;

    std::shared_ptr<MonitorSetupManager> m_monitorSetupManager;
    std::map<HabanaDeviceType, NodePtr> m_firstNodeByDevice;

    // mapping from logical engine to list of syncs (logical_engine_id = queue_id for goya)
    std::map<logical_engine_id, std::list<Sync>> m_syncByLogicalEngine;

    // map sync id to it's value for every node index & pipe level
    std::map<SyncId, std::map<IndexAndPipeLevel, int16_t>> m_syncAggValueByNodePipeLevel;

    // Map to save the last value each was monitored for each sync, for deleting unused jobs
    std::map<SyncId, std::multiset<int16_t>> m_lastMonValues;

    // Map Sync to relevant node
    std::map<Sync, std::pair<NodePtr, NodeSet>> m_monitorSyncsStatus;

    // For debug
    unsigned m_roiCtr;
    NodePtr m_prevNode;

    // For polymorphism
    SyncConventions& m_syncConventions;
};

/*
 * Support pipelined IO.
 * Enable to run 3 enqueues simultaneously -
 * prev enqueue work on upstream, current enqueue on compute and next enqueue on downstream.
 * Pipeline IO mode "hides" DMA DOWN/UP during the compute time.
 */
class SyncSchemeManagerPipelinedIO : public SyncSchemeManager
{
public:
    explicit SyncSchemeManagerPipelinedIO(HabanaGraph* graph, SyncConventions& syncConventions);
    ~SyncSchemeManagerPipelinedIO() override;

protected:
    virtual void _removeNonMonitoredJobs();
    virtual void _handleFirstComputeNode() = 0;
    virtual void _resetSyncIdsPipelinedIO() = 0;
    virtual bool _removeNodeJob(const Sync& rmSync, const NodePtr& node);
    virtual void _monitorDmaUpInProgress() = 0;
    virtual void _resetComputeReservedSyncs() = 0;
    virtual void _resetEngineSemaphore() = 0;
    virtual void _handleProducers(NodePtr node) override;
    virtual void _markFirstComputeNodeFinish(NodePtr node);
    virtual void addSyncToNode(NodePtr node, unsigned int engineIndex, const SyncObject& sync);

    // DMA Down
    // Aggregated value and number of all compute nodes that are consumers of DMA Down (their producer is the DMA Down)
    unsigned int m_firstComputeFinishAggValue;
    NodeSet m_totalFirstComputeNodes;

    // DMA Up
    NodePtr m_lastDmaUpNode;
    // DMA up in progress indicator - by device and engine index
    std::map<queue_id, SyncId> m_dmaInProgressSyncId;
};

//---------------------------------------------------------
// Templeate function implementation inlined
//---------------------------------------------------------
#include "sync_scheme_manager.inl"

#include "sync_scheme_manager.h"

#include "defs.h"
#include "graph_compiler/habana_nodes/habana_nodes.h"
#include "monitor_setup_manager.h"
#include "sync_conventions.h"
#include "sync_utils.h"
#include "types_exception.h"
#include "types.h"

///////////////////////
// SyncSchemeManager //
///////////////////////

SyncSchemeManager::SyncSchemeManager(HabanaGraph *graph, SyncConventions& syncConventions)
: m_graph(graph)
, m_monitorSetupManager(graph->getCodeGenerator()->getMonitorSetupManager())
, m_roiCtr(0)
, m_prevNode(nullptr)
, m_syncConventions(syncConventions)
{
}

SyncSchemeManager::~SyncSchemeManager()
{
}

// Node sync scheme for Sequential IO mode or when the user is responsible for the graph IO
void SyncSchemeManager::runNodesSyncScheme()
{
    _initReservedMonitors();

    for (auto node : m_graph->getExeSortedNodes())
    {
        if (!node->isLogicalOperation())
        {
            _fillNodeSyncScheme(node);
            m_prevNode = node;
        }
    }

    _resetSyncIdsSequentialIO();
    _addInitialSyncWaits();
}

std::vector<SyncSchemeManager::SyncId> SyncSchemeManager::getDeviceSyncId(queue_id engineId)
{
    std::vector<SyncId> ret;

    int numEngines = numSignalingEngines(engineId);

    do {
        ret.push_back(m_graph->getCodeGenerator()->getSyncObjectManager()->getFreeGroupId());
        numEngines -= (int)m_syncConventions.getGroupSize();
    } while(numEngines > 0);

    return ret;
}

unsigned int SyncSchemeManager::_numberOfEngines(NodePtr node) const
{
    return getNumEnginesForDeviceType(m_graph->getNodeUtility().getNodeDeviceType(node), *m_graph->getHALReader());
}

void SyncSchemeManager::_addInitialSyncWaits()
{
    std::list<SyncOrMonitor> initialSyncs;

    MonObject mon;
    mon.armValue  = 0;
    mon.operation = MONITOR_SO_OP_EQ;
    mon.fenceTargetVal.set(1);

    SyncObject sync;
    sync.value = 1;
    sync.operation = m_syncConventions.getSyncSetOp();
    sync.barrier   = REGISTER_BARRIER;

    SyncOrMonitor syncOrMon;

    for (HabanaDeviceType deviceType : getPlatformDeviceTypes())
    {
        unsigned numberOfEngines = getNumEnginesForDeviceType(deviceType, *m_graph->getHALReader());
        m_graph->getCodeGenerator()->getInitialSyncInstructionsByQueueId()[deviceType].resize(numberOfEngines);
        for (unsigned i = 0 ; i < numberOfEngines ; ++i)
        {
            if (m_graph->isEngineDisabled(deviceType, i)) continue;
            unsigned syncId = m_syncConventions.getSyncObjEngineSem(m_syncConventions.getLowerQueueID(deviceType, i));

            mon.mask.unset();
            if (m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs && deviceType == DEVICE_DMA_HOST_DEVICE)
            {
                // Override fence monitor with SyncObjDmaDownFeedback (use the same monitor for fence and for engine)
                syncId = m_syncConventions.getSyncObjDmaDownFeedback();
                // In case no IO and dma feedback sync Ids are not initialized
                if (syncId == 0)
                {
                    continue;
                }
                //Need to setup mask since this is a group id and num of DMA Down engines is 1
                mon.mask.set(_numEnginesToMask(1));
            }
            if (m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs && deviceType == DEVICE_DMA_DEVICE_HOST)
            {
                // Override fence monitor with SyncObjDmaUpFeedback (use the same monitor for fence and for engine)
                syncId = m_syncConventions.getSyncObjDmaUpFeedback();
                // In case no IO and dma feedback sync Ids are not initialized
                if (syncId == 0)
                {
                    continue;
                }
                //Need to setup mask since this is a group id and num of DMA Up engines is 1
                mon.mask.set(_numEnginesToMask(1));
            }

            mon.id = m_syncConventions.getMonObjEngineSemBase() + m_syncConventions.getLowerQueueID(deviceType, i);
            mon.syncId = syncId;
            syncOrMon.type = SyncOrMonitor::MONITOR_OBJ;
            syncOrMon.monitor = mon;
            initialSyncs.push_back(syncOrMon);

            //If pipeline IO is supported there is no need to signal mutex semaphore
            if (!m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs ||
               (deviceType != DEVICE_DMA_HOST_DEVICE && deviceType != DEVICE_DMA_DEVICE_HOST))
            {
                sync.id = syncId;
                syncOrMon.type = SyncOrMonitor::SYNC_OBJ;
                syncOrMon.sync = sync;
                initialSyncs.push_back(syncOrMon);
            }
            // add monitor + sync
            m_graph->getCodeGenerator()->getInitialSyncInstructionsByQueueId()[deviceType][i].swap(initialSyncs);
        }
    }
}

void SyncSchemeManager::_resetSyncIdsSequentialIO(bool bActivate)
{
    static const uint32_t resetEngine = 0;

    // wait for the last signal of each event
    SyncOrMonitor syncOrMon;
    syncOrMon.type = SyncOrMonitor::MONITOR_OBJ;
    syncOrMon.monitor.operation = MONITOR_SO_OP_EQ;
    syncOrMon.monitor.fenceTargetVal.unset();

    std::list<SyncOrMonitor> finalSyncs;

    if (!bActivate)  // relevant only for execute part
    {
        if (shouldWaitForActiveEngines())
        {
            std::vector<unsigned> activeEngines;
            for (unsigned int engineIdx = 0; engineIdx < getOverlapNumEngines(); ++engineIdx)
            {
                auto syncIt = m_syncByLogicalEngine.find(engineIdx);
                if (syncIt != m_syncByLogicalEngine.end() && (syncIt->second.size() > 0))
                {
                    if (shouldWaitForLogicalQueue(syncIt->first))
                    {
                        size_t    i          = 0;
                        const int groupSize  = (int) m_syncConventions.getGroupSize();
                        int       numEngines = numSignalingEngines(syncIt->first);

                        // There might be more than 8 engines so we might need more than 1 monitor
                        do
                        {
                            unsigned maskable = (numEngines - groupSize <= 0) ? numEngines : groupSize;
                            syncOrMon.monitor.mask.set(_numEnginesToMask(maskable));
                            syncOrMon.monitor.syncId            = syncIt->second.back().getId(i++);
                            syncOrMon.monitor.armValue          = syncIt->second.back().getValue();
                            syncOrMon.monitor.predicateTheSetup = shouldPredicateTheMonSetup();

                            finalSyncs.push_back(syncOrMon);
                            numEngines -= groupSize;
                        } while (numEngines > 0);

                        activeEngines.push_back(engineIdx);
                    }
                    else
                    {
                        HB_ASSERT(0, "don't expect to have working external queues");
                    }
                }
            }
        }
    }

    // Wait for all semaphores to happen before reset them
    // To avoid race between the reset and non-working engines
    for (HabanaDeviceType deviceType : getPlatformDeviceTypes())
    {
        unsigned numberOfEngines = getNumEnginesForDeviceType(deviceType, *m_graph->getHALReader());
        for (unsigned i = 0 ; i < numberOfEngines ; ++i)
        {
            if (m_graph->isEngineDisabled(deviceType, i)) continue;
            syncOrMon.monitor.syncId = m_syncConventions.getSyncObjEngineSem(m_syncConventions.getLowerQueueID(deviceType, i));
            syncOrMon.monitor.mask.unset();
            syncOrMon.monitor.armValue          = 1;
            syncOrMon.monitor.predicateTheSetup = shouldPredicateTheMonSetup();
            finalSyncs.push_back(syncOrMon);
        }
    }

    // Adding monitor for the sig-out sync objects
    if (m_syncConventions.isSignalOutGroupSupported())
    {
        unsigned numOfSignalingTensors;
        unsigned numOfEngineTypes;

        m_graph->getSignalOutInfo(numOfSignalingTensors, numOfEngineTypes);

        if (numOfSignalingTensors)
        {
            auto groupSize = m_graph->getCodeGenerator()->getSyncObjectManager()->getSyncConventions().getGroupSize();
            unsigned mask      = (1 << numOfEngineTypes) - 1;
            // ArmMonitor always multiply the syncId by 8 in case we have mask, so we divide here by 8
            syncOrMon.monitor.syncId = m_syncConventions.getSignalOutGroup() / groupSize;
            syncOrMon.monitor.mask.set(mask);
            // wait until all sig-out tensors are signaled
            syncOrMon.monitor.armValue          = numOfSignalingTensors;
            syncOrMon.monitor.predicateTheSetup = shouldPredicateTheMonSetup();

            LOG_DEBUG(GC,
                      "Creating monitor to the sig-out sync objects: syncId: {}, armValue: {}, mask: {}",
                      syncOrMon.monitor.syncId * groupSize,
                      syncOrMon.monitor.armValue,
                      mask);

            finalSyncs.push_back(syncOrMon);
        }
    }

    // Set monitors ids
    const EngineFenceMonitors& fenceMonitors = m_monitorSetupManager->getFenceMonitorIds(getCompletionLogicalQueue(),
                                                                                         resetEngine,
                                                                                         finalSyncs.size(),
                                                                                         WaitID::ID_0);

    auto fenceMonitorsIter = fenceMonitors.begin();
    for (auto& sm : finalSyncs)
    {
        sm.monitor.id = *fenceMonitorsIter;
        sm.monitor.fenceId = WaitID::ID_0;
        ++fenceMonitorsIter;
    }
    finalSyncs.back().monitor.fenceTargetVal.set(finalSyncs.size());

    // reset the engines sync objects
    _pushResetAllSyncs(finalSyncs);

    // reset the engine sems of all engines.
    _pushResetEngineSemaphors(getPlatformDeviceTypes(), finalSyncs);

    m_graph->getCodeGenerator()->getFinalSyncInstructions(bActivate)[getFinalSyncsQueueId()].swap(finalSyncs);
}

void SyncSchemeManager::_monitorPipelineSyncs(NodePtr monitorNode)
{
    std::map<IndexAndPipeLevel, std::list<MonObject>> monitorsMap;
    MonObject mon;
    mon.operation = MONITOR_SO_OP_GREQ;

    std::pair<unsigned, unsigned>  forcedDependency {0, 0};  // first = logical engine ID, second = sync idx (overlap's)
    std::pair<unsigned, unsigned>* pForcedDependency = nullptr;

    if (GCFG_DISABLE_PARALLELISM.value() && m_prevNode != nullptr)
    {
        // Force waiting on previous node
        auto pipeSyncs = engineSyncScheme(m_prevNode, 0).pipelineSyncs;
        if (!pipeSyncs.empty() && pipeSyncs.back().sync != nullptr)
        {
            unsigned syncId         = pipeSyncs.back().sync->id;
            unsigned value          = _getSyncValue(syncId, m_prevNode->getId(), pipeSyncs.size() - 1);
            unsigned idx            = syncIdAndValueToSignalId(_getLogicalEngineID(m_prevNode, 0), syncId, value);
            forcedDependency.first  = _getLogicalEngineID(m_prevNode, 0);
            forcedDependency.second = idx;
            pForcedDependency       = &forcedDependency;
        }
    }

    std::list<NodeROI>& rois = *m_graph->GetNodeROIs(monitorNode);
    for (auto it = rois.begin(); it != rois.end(); it++)
    {
        std::list<NodeROI*> roi;
        roi.push_back(&*it);
        IndexAndPipeLevel enginePipeLevel(it->engineIndex, it->pipelineLevel);

        // In case breakpoint is enabled - do not skip non-signaling ROIs
        if (!m_graph->getBreakpointEnable() || m_graph->disableBreakpointsForNonSignaling())
        {
            while (it->numSignals == 0)
            {
                it++;
                roi.push_back(&*it);
                HB_ASSERT(it != rois.end(), "Can't end the pipelines with a non signaling roi");
            }
        }

        std::list<MonObject>& monitors = monitorsMap[enginePipeLevel];
        _monitorRoiPipelineSyncs(monitorNode, roi, monitors, pForcedDependency);

        if (m_graph->getBreakpointEnable())
        {
            uint16_t lowBP, highBP;

            highBP = (m_roiCtr + 1) / (MAX_VALUE_FOR_DBG_CTR + 1);
            if (highBP > MAX_VALUE_FOR_DBG_CTR)
            {
                // if high wrapped around, don't stop on BPs anymore
                // with 2 sync objects(15bit each) max m_roiCtr can be 1 073 741 823
                LOG_CRITICAL(GC, "DBG regs reached max value({}), will not stop on BPs anymore",
                             (MAX_VALUE_FOR_DBG_CTR + 1) * (MAX_VALUE_FOR_DBG_CTR + 1) - 1);
            }
            else
            {
                ++m_roiCtr;
                lowBP = m_roiCtr % (MAX_VALUE_FOR_DBG_CTR + 1);
                if (highBP > 0)
                {
                    mon.syncId   = m_syncConventions.getSyncObjDbgCtr() + 1;
                    mon.armValue = highBP;
                    mon.mask.unset();
                    monitors.push_back(mon);
                }

                if (lowBP > 0)
                {
                    mon.syncId   = m_syncConventions.getSyncObjDbgCtr();
                    mon.armValue = lowBP;
                    mon.mask.unset();
                    monitors.push_back(mon);
                }
            }
        }

        // add monitors for blocking nodes
        if (enginePipeLevel.pipeLevel == 0 && shouldBlockOnControlEdges(monitorNode, *m_graph))
        {
            _generateBlockingNodesMonitors(monitorNode, monitors);
        }

        _addFenceAggMonitors(monitorNode, enginePipeLevel, monitors, false);
    }
}

bool SyncSchemeManager::shouldWaitForActiveEngines() const
{
    return true;
}
bool SyncSchemeManager::shouldPredicateTheMonSetup() const
{
    return true;
}

// Get the annotation of specific node in specific engine
SyncInteraction& SyncSchemeManager::engineSyncScheme(NodePtr node, unsigned int engineIndex) const
{
    auto& enginesSyncScheme = node->getNodeAnnotation().syncScheme;
    if (enginesSyncScheme.size() <= engineIndex)
    {
        enginesSyncScheme.resize(engineIndex + 1);
    }
    return enginesSyncScheme[engineIndex];
}

// Add monitor to specific node in specific engine in specific pipe stage
void SyncSchemeManager::addPipelineMon(NodePtr node, unsigned int engineIndex, const MonObject& monitor, unsigned int pipeStage)
{
    std::vector<PipelineSyncScheme>& pipelineSyncs = engineSyncScheme(node, engineIndex).pipelineSyncs;

    if (pipeStage >= pipelineSyncs.size())
    {
        pipelineSyncs.resize(pipeStage + 1);
    }
    pipelineSyncs[pipeStage].monitors.push_back(monitor);
}

// Add monitor to node which is not pipeline monitor (pre sync or post sync)
// used mainly (only?) for sync between different enqueues
void SyncSchemeManager::addMonitorToNode(NodePtr node, unsigned int engineIndex, const MonObject& monitor, bool pushFront, bool postExec)
{
    SyncOrMonitor syncOrMon;
    syncOrMon.type = SyncOrMonitor::MONITOR_OBJ;
    syncOrMon.monitor = monitor;
    auto& syncOrMonCont = postExec ? engineSyncScheme(node, engineIndex).postSyncsAndMons : engineSyncScheme(node, engineIndex).preSyncsAndMon;
    if (pushFront)
    {
        syncOrMonCont.push_front(syncOrMon);
    }
    else
    {
        syncOrMonCont.push_back(syncOrMon);
    }
}

void SyncSchemeManager::_fillNodeSyncScheme(NodePtr node)
{
    // Init for first compute
    _initializeFirstNodes(node);

    // Update the sync id's according to the ROIs
    _addNodeSyncs(node);

    // Add the monitors (every node looks what he is dependant of)
    _monitorPipelineSyncs(node);

    // Mainly used for DMAdown->Compute->DMAup protection
    _handleProducers(node);

    if (m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs)
    {
        // For DMAs
        _handleConsumers(node);
    }
}

void SyncSchemeManager::_initReservedMonitors()
{
    m_monitorSetupManager->initReservedMonitorIds(*m_graph->getHALReader());
}

// Add "signal" logging for each node in SyncScheme annotation + in manager
void SyncSchemeManager::_addNodeSyncs(NodePtr node)
{
    std::list<NodeROI>& nodeRois = *m_graph->GetNodeROIs(node);

    for (auto& roi : nodeRois)
    {
        if (!canSignal(roi)) continue;

        int16_t incValue = _getSyncValueIncrement() * roi.numSignals;

        Sync& sync = _createAndGetSync(_getLogicalEngineID(node, roi.engineIndex), incValue);

        _addPipelinedSync(node, roi.engineIndex, sync, roi.pipelineLevel, incValue, roi.numSignals);
    }
}

// Add sync to pipeline syncs
void SyncSchemeManager::_addPipelinedSync(NodePtr       node,
                                          unsigned      engineIndex,
                                          Sync&         sync,
                                          PipelineLevel pipeStage,
                                          int16_t       incVal,
                                          int32_t       numSignalsForDebg)
{
    SyncObject rollingSync;
    rollingSync.value = _getSyncValueIncrement();
    rollingSync.barrier = REGISTER_BARRIER;
    rollingSync.operation = m_syncConventions.getSyncAddOp();

    unsigned logicalEngine = _getLogicalEngineID(node, engineIndex);
    unsigned numEngines = numEnginesByLogicalEngine(logicalEngine);

    node->getNodeAnnotation().prevSyncId.resize(pipeStage + 1);
    node->getNodeAnnotation().prevSyncVal.resize(pipeStage + 1);
    node->getNodeAnnotation().prevSyncId[pipeStage].resize(numEngines);
    node->getNodeAnnotation().prevSyncVal[pipeStage].resize(numEngines);

    // If this is not the first sync, record the previous sync information
    if (!(m_syncByLogicalEngine[logicalEngine].size() == 1 && sync.getValue() == 0))
    {
        Sync prevSync = sync;
        if (prevSync.getValue() == 0)
        {
            // Get the previous sync id
            prevSync = *std::prev(m_syncByLogicalEngine[logicalEngine].end(), 2);
        }

        for (unsigned i = 0 ; i < numEngines ; ++i)
        {
            unsigned i1 = i / m_syncConventions.getGroupSize();
            unsigned i2 = i % m_syncConventions.getGroupSize();
            node->getNodeAnnotation().prevSyncId[pipeStage][i].set(prevSync.getId(i1) * m_syncConventions.getGroupSize() + i2);
            node->getNodeAnnotation().prevSyncVal[pipeStage][i] = prevSync.getValue();
        }
    }

    sync.setValue(sync.getValue() + incVal); // value += incVal
    HB_ASSERT(sync.isValueValid(), "shouldn't cross the int16 max value");

    for (unsigned i = 0 ; i < numEngines ; ++i)
    {
        // Unrolling group ID to final sync ID
        rollingSync.id = sync.getId(i / m_syncConventions.getGroupSize()) *
                         m_syncConventions.getGroupSize() +
                         _getSyncIdIncrement(i, logicalEngine);

        std::vector<PipelineSyncScheme>& pipelineSyncs = engineSyncScheme(node, i).pipelineSyncs;

        if (pipeStage >= pipelineSyncs.size())
        {
            pipelineSyncs.resize(pipeStage + 1);
        }

        pipelineSyncs[pipeStage].sync = std::make_shared<SyncObject>(rollingSync);
        pipelineSyncs[pipeStage].syncTotalValue = sync.getValue();
        pipelineSyncs[pipeStage].numSignalsForDbg = numSignalsForDebg;
        IndexAndPipeLevel nodePipeLevel(node->getId(), pipeStage);
        m_syncAggValueByNodePipeLevel[rollingSync.id][nodePipeLevel] = sync.getValue(); // Save the base value
    }
}

unsigned SyncSchemeManager::_getSyncIdIncrement(unsigned physEngineIdx, unsigned logicalEngine) const
{
    return physEngineIdx % m_syncConventions.getGroupSize();
}

void SyncSchemeManager::_addFenceAggMonitors(const NodePtr&            monitorNode,
                                             const IndexAndPipeLevel&  enginePipeLevel,
                                             std::list<MonObject>&     monitors,
                                             bool                      postExec)
{
    if (monitors.empty()) return;

    for (auto& mon : monitors)
    {
        mon.fenceTargetVal.unset();
    }
    MonObject& lastMonitor = monitors.back();
    lastMonitor.fenceTargetVal.set(monitors.size());

    unsigned logicalEngine = _getLogicalEngineID(monitorNode, enginePipeLevel.index);
    unsigned numEngines = numEnginesByLogicalEngine(logicalEngine);

    for (unsigned i = 0; i < numEngines; ++i)
    {
        const EngineFenceMonitors& fenceMonitors =
            m_monitorSetupManager->getFenceMonitorIds((HabanaDeviceType)logicalEngine, i, monitors.size());

        auto fenceMonitorsIter = fenceMonitors.begin();
        for (auto& mon : monitors)
        {
            mon.id = *fenceMonitorsIter;
            if (postExec)
            {
                addMonitorToNode(monitorNode, i, mon, false, true /* post exec */);
            }
            else
            {
                addPipelineMon(monitorNode, i, mon, enginePipeLevel.pipeLevel);
            }
            ++fenceMonitorsIter;
        }
    }
}

void SyncSchemeManager::_addCpPipelinedSync(NodePtr node, unsigned int engineIndex, const SyncObject& sync, PipelineLevel pipeStage)
{
    std::vector<PipelineSyncScheme>& pipelineSyncs = engineSyncScheme(node, engineIndex).pipelineSyncs;

    if (pipeStage >= pipelineSyncs.size())
    {
        pipelineSyncs.resize(pipeStage + 1);
    }
    pipelineSyncs[pipeStage].cpSyncs.push_back(sync);
}

// Get all non-logical preceeding nodes (recursion for logical)
// TODO: Move to HabanaGraph?
void SyncSchemeManager::_getNodeInputsProducers(NodePtr node, NodeSet& producers) const
{
    const TensorVector& inputs = node->getInputs();
    for (auto tensor : inputs)
    {
        if (tensor == nullptr) continue;
        if (tensor->isUnitMatrix())
        {
            continue;
        }
        NodePtr producer = m_graph->getTensorProducer(tensor);
        if (producer == nullptr) continue;
        if (!producer->isLogicalOperation())
        {
            producers.insert(producer);
        }
        else
        {
            _getNodeInputsProducers(producer, producers);
        }
    }
}

SyncSchemeManager::Sync& SyncSchemeManager::_createAndGetSync(logical_engine_id engineId, int16_t incValue)
{
    HB_ASSERT(incValue >= 0, "incValue is {} while it should be bigger than 0", incValue);

    auto syncIter = m_syncByLogicalEngine.find(engineId);
    if (syncIter == m_syncByLogicalEngine.end())
    {
        std::list<Sync> syncList(1, Sync(getDeviceSyncId(engineId), 0));
        syncIter = m_syncByLogicalEngine.insert(std::make_pair(engineId, syncList)).first;
    }
    else
    {
        if ((int16_t) (syncIter->second.back().getValue() + incValue) < 0)
        {
            // Value crossed the limit, create new sync
            syncIter->second.push_back(Sync(getDeviceSyncId(engineId), 0));
        }
    }
    return syncIter->second.back();
}

int16_t SyncSchemeManager::_getSyncValueIncrement() const
{
    return 1;
}

int16_t SyncSchemeManager::_getSyncValue(SyncId syncId, unsigned int syncNodeId, PipelineLevel pipeLevel) const
{
    int16_t ret = 0;
    auto syncAggAllValues = m_syncAggValueByNodePipeLevel.find(syncId);
    if (syncAggAllValues != m_syncAggValueByNodePipeLevel.end())
    {
        auto pipelineValue = syncAggAllValues->second.find(IndexAndPipeLevel(syncNodeId, pipeLevel));
        if (pipelineValue != syncAggAllValues->second.end())
        {
            ret = pipelineValue->second;
        }
    }

    return ret;
}

// Find a sync for the given logical engine that can cover the requested overlap-value. Note that:
//   1. overlap values starts from 0 while sync scheme values start from 1
//   2. overlap values don't take into account roll-up
SyncSchemeManager::Sync SyncSchemeManager::_findSyncForValueByLogicalEngine(queue_id engineId,
                                                                            unsigned overlapValue) const
{
    unsigned int signalValue = overlapValue + 1; // convert from overlap realm to sync scheme realm
    auto iter = m_syncByLogicalEngine.find(engineId);
    HB_ASSERT(iter != m_syncByLogicalEngine.end(), "Couldn't find engine id {} in m_syncByLogicalEngine", engineId);
    for (const Sync& sync : iter->second)
    {
        HB_ASSERT(sync.isValueValid(), "invalid sync");
        if (signalValue <= (unsigned int) sync.getValue())
        {
            Sync copy = sync;
            copy.setValue((int16_t)signalValue);
            return copy;
        }
        // roll-up occurred for this overlap-value, so move on to the next sync
        // while subtracting the value covered so far by the current sync
        signalValue -= (unsigned int) sync.getValue();
    }
    HB_ASSERT(false, "Sync for value {} not found", overlapValue);
    throw SynapseException("Sync value not found");
}

SyncSchemeManager::Sync* SyncSchemeManager::_findSyncById(SyncId id, logical_engine_id hint)
{
    if (hint != INVALID_ENGINE_ID)
    {
        auto mapIter = m_syncByLogicalEngine.find(hint);
        if (mapIter != m_syncByLogicalEngine.end())
        {
            for (auto syncItr = mapIter->second.begin(); syncItr != mapIter->second.end(); syncItr++)
            {
                if (syncItr->hasId(id)) return &(*syncItr);
            }
        }
    }

    // we didn't get a hint, or the given hint wasn't helpful, lets do brute-force search
    for (auto engineSyncs = m_syncByLogicalEngine.begin(); engineSyncs != m_syncByLogicalEngine.end(); engineSyncs++)
    {
        if (engineSyncs->first == hint) continue; // no point to check the hint again

        for (auto syncItr = engineSyncs->second.begin(); syncItr != engineSyncs->second.end(); syncItr++)
        {
            if (syncItr->hasId(id)) return &(*syncItr);
        }
    }

    return nullptr;
}

void SyncSchemeManager::_handleProducers(NodePtr node)
{
    return;
}

void SyncSchemeManager::_handleConsumers(NodePtr node)
{
    return;
}

// Initialize lookup for each device type to the first executed node
void SyncSchemeManager::_initializeFirstNodes(const NodePtr& node)
{
    HabanaDeviceType deviceType = m_graph->getNodeUtility().getNodeDeviceType(node);

    if (m_firstNodeByDevice.find(deviceType) == m_firstNodeByDevice.end())
    {
        m_firstNodeByDevice[deviceType] = node;
    }
}

void SyncSchemeManager::_pushResetAllSyncs(std::list<SyncOrMonitor>& ret) const
{
    SyncOrMonitor syncOrMon;
    syncOrMon.type = SyncOrMonitor::SYNC_OBJ;
    syncOrMon.sync.value = 0;
    syncOrMon.sync.operation = m_syncConventions.getSyncSetOp();
    syncOrMon.sync.barrier = REGISTER_BARRIER;

    for (auto engineSyncs : m_syncByLogicalEngine)
    {
        for (const Sync& sync : engineSyncs.second)
        {
            for (SyncId syncId : sync.getAllIds())
            {
                if (syncId == m_syncConventions.getSyncObjDmaDownFeedback() ||
                    syncId == m_syncConventions.getSyncObjDmaUpFeedback())
                {
                    // DMA down/up feedback sync objects are reset in different functions in pipeline IO mode
                    if (m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs)
                    {
                        continue;
                    }
                    syncOrMon.sync.id = syncId * m_syncConventions.getGroupSize();
                    ret.push_back(syncOrMon);
                }
                else
                {
                    for (unsigned i = 0; i < m_syncConventions.getGroupSize(); ++i)
                    {
                        syncOrMon.sync.id = syncId * m_syncConventions.getGroupSize() + i;
                        ret.push_back(syncOrMon);
                        // currently this is not optimized as we are resetting syncs that might never been used (SW-13758)
                    }
                }
            }
        }
    }

    if (m_graph->getBreakpointEnable())
    {
        // Reset debug syncs
        syncOrMon.sync.id = m_syncConventions.getSyncObjDbgCtr();
        ret.push_back(syncOrMon);
        // Reset round robin sync
        ++syncOrMon.sync.id;
        ret.push_back(syncOrMon);
    }

    // Reset sig-out (external-tensors):
    if (m_syncConventions.isSignalOutGroupSupported())
    {
        for (unsigned i = 0; i < m_syncConventions.getNumOfSignalGroups(); ++i)
        {
            syncOrMon.sync.id = m_syncConventions.getSignalOutGroup() + i;
            ret.push_back(syncOrMon);
        }
        // reset the dummy syncId as well
        syncOrMon.sync.id = m_graph->getCodeGenerator()->getSyncObjectManager()->getDummySyncId();
        ret.push_back(syncOrMon);
    }
}

void SyncSchemeManager::_pushResetEngineSemaphors(const std::vector<HabanaDeviceType>& deviceTypes,
                                                  std::list<SyncOrMonitor>&            ret) const
{
    SyncOrMonitor syncOrMon;
    syncOrMon.type = SyncOrMonitor::SYNC_OBJ;
    syncOrMon.sync.value = 0;
    syncOrMon.sync.operation = m_syncConventions.getSyncSetOp();
    syncOrMon.sync.barrier = REGISTER_BARRIER;

    for (HabanaDeviceType deviceType : deviceTypes)
    {
        unsigned numberOfEngines = getNumEnginesForDeviceType(deviceType, *m_graph->getHALReader());
        for (unsigned i = 0 ; i < numberOfEngines ; ++i)
        {
            if (m_graph->isEngineDisabled(deviceType, i)) continue;
            syncOrMon.sync.id = m_syncConventions.getSyncObjEngineSem(m_syncConventions.getLowerQueueID(deviceType, i));
            ret.push_back(syncOrMon);
        }
    }
}

uint8_t SyncSchemeManager::_numEnginesToMask(unsigned numEngines) const
{
    uint8_t currMask = 0;
    for (unsigned i = 0 ; i < numEngines ; ++i)
    {
        currMask |= (1 << i);
    }
    return currMask;
}

//monitor execution of blocking nodes - handle control edges
void SyncSchemeManager::_generateBlockingNodesMonitors(NodePtr monitorNode, std::list<MonObject>& monitors)
{
    std::list<MonObject> blockingNodesMonitors;
    NodeSet              blockingNodes = m_graph->getBlockingNodes(monitorNode);
    for (auto blockingNode : blockingNodes)
    {
        if (blockingNode->isLogicalOperation()) continue;
        bool blockingNodeWasMonitored = false; //verify that each blocking node is monitored
        for (unsigned int engineIdx = 0; engineIdx < _numberOfEngines(blockingNode); engineIdx += m_syncConventions.getGroupSize())
        {
            const auto& pipeSyncs = engineSyncScheme(blockingNode, engineIdx).pipelineSyncs;
            if (pipeSyncs.empty() || pipeSyncs.back().sync == nullptr) continue;
            //create monitor object to monitor the last signal of the blocking node
            MonObject mon;
            queue_id logicalEngineId = _getLogicalEngineID(blockingNode, engineIdx);
            mon.mask.set(_numEnginesToMask(std::min(numSignalingEngines(logicalEngineId) - engineIdx, m_syncConventions.getGroupSize())));
            mon.syncId    = pipeSyncs.back().sync->id / m_syncConventions.getGroupSize();
            mon.armValue  = _getSyncValue(pipeSyncs.back().sync->id, blockingNode->getId(), pipeSyncs.size() - 1);
            mon.operation = MONITOR_SO_OP_GREQ;
            bool monitorAlreadyCovered = false;

            // avoid redundant monitors
            for (MonObject& m : monitors)
            {
                if (m.syncId == mon.syncId &&
                    mon.mask.value() == m.mask.value())
                {
                    m.armValue            = std::max(m.armValue, mon.armValue);
                    monitorAlreadyCovered = true;
                }
            }
            if (!monitorAlreadyCovered)
            {
                monitors.push_back(mon);
            }
            blockingNodeWasMonitored = true;
        }
        if (!blockingNodeWasMonitored)
        {
            //if false, the blocking node isn't covered by any monitor
            LOG_ERR(SYNC_SCHEME, "{}: Didn't find signal for blocking node {} ", HLLOG_FUNC, blockingNode->getNodeName());
            throw PassFailedException();
        }
    }
}


/* -------------------- Prints Related ------------------------- */

void SyncSchemeManager::printSyncScheme() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(SYNC_SCHEME)) return;
    const NodeVector& sortedNodes = m_graph->getExeSortedNodes();

    for (const auto& node : sortedNodes)
    {
        if (!node->isLogicalOperation())
        {
             // in _getLogicalEngineID function all platform except Goya ignore the second argument (engineIndex)
             // so logical queue id may incorrect only in case of TPC node in Goya
            LOG_DEBUG(SYNC_SCHEME, "Node {}, id: {}, type: {} logical queue id: {}",node->getNodeName(), node->getId(),
                                   node->getEngineTypeStr(), _getLogicalEngineID(node, 0));
            _printNodeSyncScheme(node);
        }
    }
}

const std::string& getSyncOpStr(int syncOp)
{
    static const std::string setOp = "SYNC_OP_SET";
    static const std::string addOp = "SYNC_OP_ADD";

    return syncOp == 0 ? setOp : addOp;
}

std::ostream& operator<<(std::ostream& os, const MonitorOp& monOp)
{
    static const std::string operations[] = {"MONITOR_SO_OP_GREQ", "MONITOR_SO_OP_EQ"};
    os << operations[monOp];
    return os;
}

std::ostream& operator<<(std::ostream& os, const SyncObject& sync)
{
    os << " id: "    << sync.id
       << " value: " << sync.value
       << " op: "    << getSyncOpStr(sync.operation);

    return os;
}

std::ostream& operator<<(std::ostream& os, const MonObject& mon)
{
    os << " id: "     << mon.id
       << " SyncId: " << mon.syncId
       << " value: "  << mon.armValue
       << " op: "     << mon.operation;
    if (mon.fenceTargetVal.is_set())
    {
        os << " fenceTargetVal: " << mon.fenceTargetVal.value();
    }

    return os;
}

template<class T_OBJ>
void printObj(const T_OBJ& obj, const std::string& prefix, const std::string& postfix = std::string())
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(SYNC_SCHEME)) return;
    std::ostringstream output;

    output << obj;

    LOG_DEBUG(SYNC_SCHEME, "      {}: {} {}", prefix, output.str(), postfix);
}

std::string getMaskedSignals(unsigned int baseSyncId, uint8_t mask)
{
    std::vector<unsigned int> syncs;
    syncs.reserve(8);
    for (int index = 0; index < 8; ++index)
    {
        if (mask & 1 << index)
        {
            syncs.push_back(baseSyncId * 8 + index);
        }
    }
    return toString(syncs, ',');
}

void printArm(const MonObject &monObj, std::map<int, MonObject> &mapOfMonitors, const SyncConventions& syncConventions)
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(SYNC_SCHEME)) return;
    std::string SyncObjectName = syncConventions.getSyncObjName(monObj.syncId);
    std::string strForSetup = "";

    //check if fence set
    bool isFenceSet = false;

    //check if the setup is to update fence but not wait for it
    bool isUpdateFence = false;

    if (monObj.fenceTargetVal.is_set())
    {
        isFenceSet = true;
    }
    else
    {
        auto search2 = mapOfMonitors.find(monObj.id);
        if (search2 != mapOfMonitors.end())
        {
            if (search2->second.signalSyncId == FENCE_MONITOR_ID)
            {
                isUpdateFence = true;
            }
            else
            {
                if (search2->second.shouldInc)
                {
                    strForSetup += "then increment ";
                }
                else
                {
                    strForSetup += "then write ";
                }

                strForSetup += std::to_string(search2->second.setupValue);
                strForSetup += " to semaphore ";
                strForSetup += std::to_string(search2->second.signalSyncId);
            }
        }
        else
        {
            LOG_ERR(SYNC_SCHEME, "ERROR! Setup not found for monitor {}", monObj.id);
        }

    }

    LOG_DEBUG(SYNC_SCHEME,
              "       Arm monitor {}: wait semaphore {} {} {} {} {} {}",
              monObj.id,
              monObj.mask.is_set() ? getMaskedSignals(monObj.syncId, monObj.mask.value())
                                   : std::to_string(monObj.syncId),
              monObj.mask.is_set() ? "" : SyncObjectName,
              (monObj.operation == MONITOR_SO_OP_GREQ ? "(>=)" : "(==)"),
              monObj.armValue,
              (isFenceSet || isUpdateFence) ? "then write 1 to fence0" : "",
              strForSetup);
    if (isFenceSet)
    {
        LOG_DEBUG(SYNC_SCHEME, "       wait for fence0 == {}",
                  monObj.fenceTargetVal.value());
    }
}

void printSignal(const SyncObject &syncObj,
                 const SyncConventions& syncConventions,
                 const std::string &postfix = std::string(),
                 const std::string &pipelineSyncStr = std::string())
{
    LOG_DEBUG(SYNC_SCHEME,
              "       Signal semaphore {} {}: {} {} {} {}",
              syncObj.id,
              syncConventions.getSyncObjName(syncObj.id),
              (syncObj.operation == syncConventions.getSyncSetOp() ? "Set to" : "Inc by"),
              syncObj.value,
              postfix,
              pipelineSyncStr);
}

void SyncSchemeManager::_printNodeSyncScheme(NodePtr node) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(SYNC_SCHEME)) return;

    std::map<int, MonObject> mapOfMonitors = m_monitorSetupManager->getMapOfMonitors();
    const std::vector<SyncInteraction> &syncScheme = node->getNodeAnnotation().syncScheme;
    for (unsigned int engineIndex = 0; engineIndex < syncScheme.size(); ++engineIndex)
    {
        LOG_DEBUG(SYNC_SCHEME, "     Engine #{}", engineIndex);
        LOG_DEBUG(SYNC_SCHEME, "      Pre execute:");
        const SyncInteraction &engineSyncScheme = syncScheme[engineIndex];
        for (const auto &syncOrMon : engineSyncScheme.preSyncsAndMon)
        {
            if (syncOrMon.type == SyncOrMonitor::MONITOR_OBJ)
            {
                printArm(syncOrMon.monitor, mapOfMonitors, m_syncConventions);
            }
            else if (syncOrMon.type == SyncOrMonitor::SYNC_OBJ)
            {
                printSignal(syncOrMon.sync, m_syncConventions);

            }
        }

        static const std::string pipelineSyncStr = "activation sync #";
        unsigned int pipelineLevel = 0;
        for (const auto& pipelineSync : engineSyncScheme.pipelineSyncs)
        {
            LOG_DEBUG(SYNC_SCHEME, "      Activation number #{}:", pipelineLevel);
            for (const auto& monitor : pipelineSync.monitors)
            {
                printArm(monitor, mapOfMonitors, m_syncConventions);
            }

            for (const auto &sync : pipelineSync.cpSyncs)
            {
                printSignal(sync, m_syncConventions);
            }

            if (pipelineSync.sync != nullptr)
            {
                int16_t aggValue = _getSyncValue((SyncId)pipelineSync.sync->id, node->getId(), (PipelineLevel)pipelineLevel);
                const std::string postfix = ",Aggregated value: " + std::to_string(aggValue);
                printSignal(*pipelineSync.sync, m_syncConventions, postfix, " (descriptor)");
            }
            else
            {
                LOG_DEBUG(SYNC_SCHEME, "      {}: no signal", pipelineSyncStr + std::to_string(pipelineLevel));
            }
            ++pipelineLevel;
        }

        for (const auto &syncOrMon : engineSyncScheme.postSyncsAndMons)
        {
            LOG_DEBUG(SYNC_SCHEME, "      Post execute: #{}:", pipelineLevel);
            if (syncOrMon.type == SyncOrMonitor::SYNC_OBJ)
            {
                static const std::string postSync = " ";
                printSignal(syncOrMon.sync, m_syncConventions);
            }
            else
            {
                static const std::string postMon = " ";
                printArm(syncOrMon.monitor, mapOfMonitors, m_syncConventions);
            }

        }
    }
}

/*
 * Pipelined IO related
 */

SyncSchemeManagerPipelinedIO::SyncSchemeManagerPipelinedIO(HabanaGraph* graph, SyncConventions& syncConventions)
        : SyncSchemeManager(graph, syncConventions),
          m_firstComputeFinishAggValue(0)
{
}

SyncSchemeManagerPipelinedIO::~SyncSchemeManagerPipelinedIO()
{
}

void SyncSchemeManagerPipelinedIO::_removeNonMonitoredJobs()
{
    bool recheck = false;
    auto syncStatusIter = m_monitorSyncsStatus.begin();
    while (syncStatusIter != m_monitorSyncsStatus.end())
    {
        SyncId idNotMonitored = syncStatusIter->first.getId(0);
        int16_t valueNotMonitored = syncStatusIter->first.getValue();
        const std::multiset<int16_t>& lastSyncMonValue = m_lastMonValues[idNotMonitored];
        if ((lastSyncMonValue.empty() || valueNotMonitored > *lastSyncMonValue.rbegin()) && // If a higher value is monitored, then its irrelevant
            syncStatusIter->second.second.empty())
        {
            // No monitors for sync - remove job
            recheck = _removeNodeJob(syncStatusIter->first, syncStatusIter->second.first);
            syncStatusIter = m_monitorSyncsStatus.erase(syncStatusIter);
        }
        else
        {
            ++syncStatusIter;
        }
    }
    if (recheck)
    {
        // If a job were remove, other syncs may now be not monitored
        _removeNonMonitoredJobs();
    }
}

bool SyncSchemeManagerPipelinedIO::_removeNodeJob(const Sync& rmSync, const NodePtr& node)
{
    bool shouldRecheck = false;
    for (unsigned int engineIdx = 0; engineIdx < _numberOfEngines(node); ++engineIdx)
    {
        auto& pipelineSyncs = engineSyncScheme(node, engineIdx).pipelineSyncs;

        if (pipelineSyncs.empty() || pipelineSyncs.back().sync->id != rmSync.getId(0)) continue;
        // Found the sync engine
        PipelineLevel pipeLevel = pipelineSyncs.size() - 1;
        int16_t lastSyncAggValue = m_syncAggValueByNodePipeLevel[rmSync.getId(0)][IndexAndPipeLevel(engineIdx, pipeLevel)];
        if (lastSyncAggValue > rmSync.getValue())
        {
            // An inner activation increment
            break;
        }
        std::list<NodeROI>* nodeRois = m_graph->GetNodeROIs(node);
        for (auto nodeRoiIter = nodeRois->begin(); nodeRoiIter != nodeRois->end(); ++nodeRoiIter)
        {
            if (nodeRoiIter->engineIndex == engineIdx && nodeRoiIter->pipelineLevel == pipeLevel)
            {
                Sync* s = _findSyncById(rmSync.getId(0), _getLogicalEngineID(node, engineIdx)/*hint*/);
                HB_ASSERT_PTR(s);
                s->setValue(s->getValue() - nodeRoiIter->numSignals); // value -= nodeRoiIter->numSignals
                HB_ASSERT(s->isValueValid(), "something went wrong with sync id {}", rmSync.getId(0));
                break;
            }
        }
        // Remove the monitors
        for (const auto& mon : pipelineSyncs.back().monitors)
        {
            if (m_graph->getCodeGenerator()->getSyncObjectManager()->isReservedSyncId(mon.syncId)) continue;
            auto syncStatusIter = m_monitorSyncsStatus.lower_bound(Sync(mon.syncId, mon.armValue));
            HB_ASSERT(syncStatusIter != m_monitorSyncsStatus.end(),
                      "Sync object not in m_monitorSyncsStatus, sync ID [{}], value [{}]",
                      mon.syncId,
                      mon.armValue);
            syncStatusIter->second.second.erase(node);
            shouldRecheck |= syncStatusIter->second.second.empty();
            std::multiset<int16_t>& syncValMonitored = m_lastMonValues[mon.syncId];
            HB_ASSERT(syncValMonitored.find(mon.armValue) != syncValMonitored.end(),
                      "Sync value not in m_lastMonValues");
            syncValMonitored.erase(syncValMonitored.find(mon.armValue));
        }
        // Handle remove of cp-syncs
        for (const auto& cpSync : pipelineSyncs.back().cpSyncs)
        {
            if (cpSync.id == m_syncConventions.getSyncObjFirstComputeFinish())
            {
                m_firstComputeFinishAggValue -= cpSync.value;
            }
        }
        pipelineSyncs.pop_back();

        Sync* s = _findSyncById(rmSync.getId(0), _getLogicalEngineID(node, engineIdx)/*hint*/);
        HB_ASSERT_PTR(s);
        if (s->getValue() == 0)
        {
            // This engine doesn't have activations in this topology
            m_syncByLogicalEngine.erase(_getLogicalEngineID(node, engineIdx));
            m_graph->getCodeGenerator()->getSyncObjectManager()->releaseSyncObject(rmSync.getId(0));
        }
        break;
    }
    return shouldRecheck;
}

/*
 * Pipelined IO - DMA Down
 *
 * For all DMA Down consumers (DMA Down is their producer), fill list of them and handle thr consumers (second nodes)
 */
void SyncSchemeManagerPipelinedIO::_handleProducers(NodePtr node)
{
    NodeSet producers;
    _getNodeInputsProducers(node, producers);

    for (auto producer : producers)
    {
        //If one of the nodes producers is DMA, keep this node in first compute nodes list
        if (producer->isDma() &&
            !(producer->getOutput(0)->getTensorAnnotation().memorySpaceInfo.prefetchInfo.prefetch))
        {
            m_totalFirstComputeNodes.insert(node);
        }

        if (node->isDma())
        {
            m_lastDmaUpNode = node;
        }

        //If pipelined IO is enabled  and nodes producer is in the list of first nodes we filled before, then this is a second node.
        if ((m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs) &&
            (m_totalFirstComputeNodes.find(producer) != m_totalFirstComputeNodes.end()))
        {
            _markFirstComputeNodeFinish(node);
        }
    }
}

/*
 * Pipelined IO - DMA Down
 *
 * For all second nodes (DMA Down id the producer of thier producers)
 */
void SyncSchemeManagerPipelinedIO::_markFirstComputeNodeFinish(NodePtr node)
{
    SyncObject sync;
    sync.id = m_syncConventions.getSyncObjFirstComputeFinish();
    sync.value = 1;
    sync.operation = m_syncConventions.getSyncAddOp();
    sync.barrier = REGISTER_BARRIER;

    std::list<NodeROI>* rois = m_graph->GetNodeROIs(node);
    std::vector<unsigned> engineToMaxPipeLevel;
    engineToMaxPipeLevel.resize(_numberOfEngines(node), 0);
    for (const NodeROI& roi: *rois)
    {
        unsigned engineIndex = roi.engineIndex;
        unsigned pipelineLevel = roi.pipelineLevel;

        if (engineToMaxPipeLevel[engineIndex] < pipelineLevel)
        {
            engineToMaxPipeLevel[engineIndex] = pipelineLevel;
        }
    }

    //Signal when second node startand count the number of signals we need to wait for later on first compute node
    for (unsigned int engineIdx = 0; engineIdx < _numberOfEngines(node); ++engineIdx)
    {
        auto& engineSyncs = engineSyncScheme(node, engineIdx);
        if (!engineSyncs.pipelineSyncs.empty())
        {
            _addCpPipelinedSync(node, engineIdx, sync, engineToMaxPipeLevel[engineIdx]);
            ++m_firstComputeFinishAggValue;
        }
    }
}

// Add sync defense before running a node
void SyncSchemeManagerPipelinedIO::addSyncToNode(NodePtr node, unsigned int engineIndex, const SyncObject& sync)
{
    SyncOrMonitor syncOrMon;
    syncOrMon.type = SyncOrMonitor::SYNC_OBJ;
    syncOrMon.sync = sync;
    engineSyncScheme(node, engineIndex).preSyncsAndMon.push_back(syncOrMon);
}

uint32_t SyncSchemeManager::syncIdAndValueToSignalId(queue_id engineId, SyncId syncId, uint16_t value) const
{
    syncId /= m_syncConventions.getGroupSize();
    uint32_t signalId = 0;
    auto iter = m_syncByLogicalEngine.find(engineId);
    HB_ASSERT(iter != m_syncByLogicalEngine.end(), "Couldn't find engine id {} in m_syncByLogicalEngine", engineId);
    for (const Sync& sync : iter->second)
    {
        HB_ASSERT(sync.isValueValid(), "invalid sync");
        if (sync.hasId(syncId))
        {
            signalId += value;
            signalId -= 1;  // convert from sync scheme realm to overlap realm
            return signalId;
        }
        // roll-up occurred for this overlap-value, so move on to the next sync
        // while adding the value covered so far by the current sync
        signalId += sync.getValue();
    }
    HB_ASSERT(false, "Sync for value {} not found", value);
    throw SynapseException("Sync for value not found");
}

uint32_t SyncSchemeManager::getMaxSignalIdForEngineToBeDependentOn(const NodePtr& node) const
{
    if (!isNodeHandlingInternalDependencies(node))
    {
        return -1;
    }
    auto engineId = _getLogicalEngineID(node, 0);
    auto prevValue0 = node->getNodeAnnotation().prevSyncVal[0][0];
    const Settable<unsigned>& prevSyncId0 = node->getNodeAnnotation().prevSyncId[0][0];
    if (!prevSyncId0.is_set())
    {
        return 0;
    }
    auto value = syncIdAndValueToSignalId(engineId, prevSyncId0.value(), prevValue0);
    auto sync = _findSyncForValueByLogicalEngine(engineId, value);
    HB_ASSERT(sync.getValue() == prevValue0 && prevSyncId0.value() == sync.getId(0) * m_syncConventions.getGroupSize(), "not working: {}, {}", sync.getValue(), value + 1);
    return value + 1; // +1 since we care about new signals
}

#include "sync_scheme_manager_arc.h"

#include "habana_graph.h"
#include "sync_utils.h"
#include "types_exception.h"

#include <algorithm>

void SyncSchemeManagerArc::go()
{
    const NodeVector&        sortedNodesVec = m_graph->getExeSortedNodes();
    const NodeList           sortedNodes(sortedNodesVec.begin(), sortedNodesVec.end());
    NodeList::const_iterator itrNode       = sortedNodes.begin();
    unsigned                 startPipeline = 0;

    while (itrNode != sortedNodes.end())
    {
        if (!(*itrNode)->isLogicalOperation())
        {
            // Attempt to create node syncs (i.e. calculate its dependencies and emitted signal for each pipeline).
            // The only failure that can happen is if a SOB crossed the max limit, in that case we shall do reset.
            if (!createNodePipelineSyncs(itrNode, startPipeline))
            {
                std::tie(itrNode, startPipeline) = resetSobs(itrNode);  // resetSobs returns the resume node + pipeline
                continue;
            }
            m_prevNode                               = *itrNode;
            m_prevNodePerEng[getLogicalId(*itrNode)] = *itrNode;
            startPipeline                            = 0;  // needed for the case we are just after a reset
        }
        itrNode++;
    }
    archiveOverlap();
}

bool SyncSchemeManagerArc::createNodePipelineSyncs(const NodeList::const_iterator& itrNode, unsigned startPipeline)
{
    NodePtr node = *itrNode;

    std::list<NodeROI>& logicalRois = *m_graph->GetNodeROIs(node);

    // Add empty sync interaction for each pipe level, we will populate them later
    node->getNodeAnnotation().arcSyncScheme.resize(logicalRois.size());

    for (auto it = logicalRois.begin(); it != logicalRois.end(); it++)
    {
        if (it->pipelineLevel < startPipeline) continue;

        std::list<NodeROI*> rois;
        rois.push_back(&*it);
        unsigned firstPipeLevel = it->pipelineLevel;

        // Squash non-signaling ROIs if breakpoint is not enabled
        if (!m_graph->getBreakpointEnable() || m_graph->disableBreakpointsForNonSignaling())
        {
            while (it->numSignals == 0)
            {
                it++;  // move on to the next pipe level
                rois.push_back(&*it);
                HB_ASSERT(it != logicalRois.end(), "Can't end the pipelines with a non signaling roi");
            }
        }

        unsigned lastPipeLevel = it->pipelineLevel;

        // Register the ROIs with the reset manager. A failure means that a reset should take place.
        if (!m_sobResetMngr.registerRois(rois, itrNode, getLogicalId(node), lastPipeLevel))
        {
            return false;
        }

        DependencyMap pipelineDeps;
        unsigned      emittedSigVal;

        // Get the pipeline syncs (overlap inside)
        createRoiPipelineSyncs(node,
                               rois,
                               getDependenciesOnControlEdges(node), // provide input control edge dependencies
                               pipelineDeps,                        // output
                               emittedSigVal);                      // output

        // Add dependencies on previous node for debugging purposes (disable parallelism feature)
        if (GCFG_DISABLE_PARALLELISM.value() && m_prevNode != nullptr)
        {
            unsigned logicalId = getLogicalId(m_prevNode);
            unsigned sigValue  = m_prevNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value();
            DependencyMap pipelineDepsNoParallelism;
            pipelineDepsNoParallelism.emplace(logicalId, sigValue);
            pipelineDeps       = getReducedDependencies(pipelineDeps, pipelineDepsNoParallelism);
        }

        // Record the sync interaction for this ROI in the node annotation. In case we squashed several non-signaling
        // ROIs together, the dependencies (i.e. monitors) go on the first pipeline level and the signal on the last.
        node->getNodeAnnotation().arcSyncScheme[firstPipeLevel].dependencies = pipelineDeps;
        node->getNodeAnnotation().arcSyncScheme[lastPipeLevel].emittedSigVal.set(emittedSigVal);

        // Add breakpoint
        if (m_graph->getBreakpointEnable())
        {
            node->getNodeAnnotation().arcSyncScheme[firstPipeLevel].breakpoint.set(++m_breakpointCtr);
            HB_ASSERT(m_breakpointCtr <= c_maxBreakpoint, "breakpoint crossed the limit");
        }
    }
    return true;
}

DependencyMap SyncSchemeManagerArc::getDependenciesOnControlEdges(const NodePtr& node) const
{
    if (shouldBlockOnControlEdges(node, *m_graph))
    {
        return nodeSetToDepMap(m_graph->getBlockingNodes(node));
    }
    else
    {
        return DependencyMap(); // no dependencies
    }
}

DependencyMap SyncSchemeManagerArc::nodeSetToDepMap(const NodeSet& nodes) const
{
    DependencyMap ret;
    for (auto blockingNode : nodes)
    {
        if (!blockingNode) continue;
        if (blockingNode->isLogicalOperation()) continue;
        if (blockingNode->getExecutionOrderedIndex() < m_sobResetMngr.getHighestNodeExeIndexAtReset()) continue;
        if (blockingNode->getNodeAnnotation().arcSyncScheme.empty())
        {
            LOG_ERR(SYNC_SCHEME, "Didn't find signal for blocking node {} ", blockingNode->getNodeName());
            throw PassFailedException();
        }
        unsigned logicalId = getLogicalId(blockingNode);
        unsigned sigValue  = blockingNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value();
        ret[logicalId]     = sigValue;
    }
    return ret;
}

// Reduce the two input dependency maps into a single map with the most restrictive dependencies
DependencyMap SyncSchemeManagerArc::getReducedDependencies(const DependencyMap& map1, const DependencyMap& map2)
{
    DependencyMap ret = map1;
    for (auto const& e : map2)
    {
        ret[e.first] = ret.find(e.first) != ret.end() ? std::max(ret[e.first], e.second) : e.second;
    }
    return ret;
}

unsigned SyncSchemeManagerArc::getMaxOverlapSigIdxForNodeToDependOn(const NodePtr& node) const
{
    unsigned logicalId = getLogicalId(node);

    if (!isNodeHandlingInternalDependencies(node))
    {
        return -1;
    }
    else if (m_prevNodePerEng.find(logicalId) == m_prevNodePerEng.end())
    {
        return 0;
    }
    const NodePtr& prevNode = m_prevNodePerEng.at(logicalId);
    // convert the emitted signal we had on the last ROI of the previous node to overlap's 0-based index-realm
    unsigned lastRoiSigVal = prevNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value();
    unsigned lastRoiSigIdx = convertSigValToOverlapIdx(logicalId, lastRoiSigVal);
    return lastRoiSigIdx + 1;  // +1 since we care about new signals
}

SyncSchemeManagerArc::ResumePoint SyncSchemeManagerArc::resetSobs(NodeList::const_iterator itrCurrNode)
{
    archiveOverlap();
    resetOverlap();
    m_prevNode = nullptr;
    m_prevNodePerEng.clear();
    ResumePoint resumePoint = m_sobResetMngr.doReset(*m_graph, getNumEngsPerLogicalId());
    // The reset rewinds the graph back to the last balanced point so clear the out-of-date sync interactions
    (*resumePoint.first)->getNodeAnnotation().arcSyncScheme.resize(resumePoint.second);
    NodeList::const_iterator tmpItr = resumePoint.first;
    std::for_each(++tmpItr, ++itrCurrNode, [](NodePtr n) { n->getNodeAnnotation().arcSyncScheme.clear(); });
    return resumePoint;
}

void SyncSchemeManagerArc::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(SYNC_SCHEME)) return;

    for (const auto& node : m_graph->getExeSortedNodes())
    {
        if (node->isLogicalOperation()) continue;

        bool hasControlEdges  = m_graph->getBlockingNodes(node).size();
        bool hardControlEdges = shouldBlockOnControlEdges(node, *m_graph);

        std::string controlEdgeStr("");

        if (hasControlEdges && hardControlEdges)
        {
            controlEdgeStr = std::string(", has control-edges that may overrule linear ranges");
        }
        else if (hasControlEdges)
        {
            controlEdgeStr = std::string(", has soft control-edges which are not blocking");
        }

        LOG_DEBUG(SYNC_SCHEME,
                  "Node name: {}, id: {}, engine device type: {}, logical queue id: {} ({}){}",
                  node->getNodeName(),
                  node->getId(),
                  node->getEngineTypeStr(),
                  getLogicalId(node),
                  logicalIdToStr(getLogicalId(node)),
                  controlEdgeStr);

        unsigned pipeLevel = 0;
        for (const ArcSyncInteraction& pipeSync : node->getNodeAnnotation().arcSyncScheme)
        {
            LOG_DEBUG(SYNC_SCHEME, "  Pipeline Level #{}:", pipeLevel++);
            LOG_DEBUG(SYNC_SCHEME, "    Dependencies (monitors):{}", pipeSync.dependencies.empty() ? " none" : "");
            for (const auto& dep : pipeSync.dependencies)
            {
                LOG_DEBUG(SYNC_SCHEME,
                          "      Wait on type: {} to reach (>=) sig-val: {}",
                          logicalIdToStr(dep.first),
                          sigValToStr(dep.first, dep.second));
            }
            LOG_DEBUG(SYNC_SCHEME, "    Emitted sig-val: {}", sigValToStr(getLogicalId(node), pipeSync.emittedSigVal));
            if (pipeSync.breakpoint.is_set())
            {
                LOG_DEBUG(SYNC_SCHEME, "    Has a breakpoint on debug value: {}", pipeSync.breakpoint.value());
            }
            if (pipeSync.sobResetTotalNumEngs > 0)
            {
                LOG_DEBUG(SYNC_SCHEME,
                          "    *** Reset SOBs procedure comes after this activation (reset ID = {}) ***",
                          pipeSync.sobResetId);
            }
        }
    }
}

//----------------------------------------------------------------------------
//                             SobResetManager
//----------------------------------------------------------------------------

SyncSchemeManagerArc::SobResetManager::SobResetManager() : c_signalLimit(GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT.value())
{
    m_allBalanced.set();
}

bool SyncSchemeManagerArc::SobResetManager::registerRois(const std::list<NodeROI*>&      rois,
                                                         const NodeList::const_iterator& itrNode,
                                                         unsigned                        logicalId,
                                                         unsigned                        pipeLevel)
{
    // Don't do anything if the mechanism for resetting SOBs is inactive
    if (GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT.value() == 0) return true;

    SyncSchemeManagerArc::SobResetManager::LogicalEngCtx& engCtx = m_engineCtx[logicalId];

    engCtx.maxRoiPending = std::max(engCtx.maxRoiPending, engCtx.accumSignal + getMaxRelIdx(rois) + 1);
    engCtx.accumSignal = f_safeIncrement(logicalId, engCtx.accumSignal, getAccumSignals(rois));

    bool needReset  = (engCtx.accumSignal > c_signalLimit) || (engCtx.maxRoiPending > c_signalLimit);
    bool isBalanced = engCtx.maxRoiPending == engCtx.accumSignal;  // we won't be balanced in case of partial ROIs

    if (needReset) return false;

    m_currentBalance[logicalId] = {itrNode, pipeLevel, isBalanced};

    // If at the current point in the graph all engines are balanced, then pin this position for future reset needs
    m_allBalanced.set(logicalId, isBalanced);
    if (m_allBalanced.all()) m_fullyBalancedPin = m_currentBalance;
    return true;
}

SyncSchemeManagerArc::ResumePoint SyncSchemeManagerArc::SobResetManager::doReset(const HabanaGraph& g,
                                                                                 const NumEngsMap&  engsMap)
{
    makeSureWeCanDoTheReset();

    // 1. Calc the total number of physical engines across all logical engines that participate in this reset.
    // 2. Find the top node among all the pinned balanced nodes (top node is the one with the highest execution index).
    // 3. Record the logical engine IDs participating in this reset
    unsigned              accumNumInvolvedEngs = m_isCmeExist ? 1 : 0;
    LogicalEngBalanceInfo topNode;
    m_logicalIdsInPrevReset.reset();
    for (const auto& engBalance : m_fullyBalancedPin)
    {
        accumNumInvolvedEngs += engsMap.at(engBalance.first);
        if ((*engBalance.second.itrNode)->getExecutionOrderedIndex() > m_highestNodeExeIndexAtReset)
        {
            m_highestNodeExeIndexAtReset = (*engBalance.second.itrNode)->getExecutionOrderedIndex();
            topNode                      = m_fullyBalancedPin[engBalance.first];
        }
        m_logicalIdsInPrevReset.set(engBalance.first, true);
    }

    // Currently not allowing to reset twice on the same node (may change if we implement SW-89803)
    // this check makes more sense for testing and less for real-life case
    if (!topNode.isBalanced)
    {
        LOG_ERR(SYNC_SCHEME, "Failed to reset SOBs in the middle of the graph - topNode was not set");
        throw PassFailedException();
    }

    // Put the reset indication on the sync scheme of all the pinned balanced nodes
    m_sobResetId++;  // starting from ID 1, ID 0 should not be used
    for (const auto& engBalance : m_fullyBalancedPin)
    {
        NodePtr             node        = *engBalance.second.itrNode;
        unsigned            pipeLevel   = engBalance.second.pipeLevel;
        ArcSyncInteraction& syncScheme  = node->getNodeAnnotation().arcSyncScheme.at(pipeLevel);
        syncScheme.sobResetTotalNumEngs = accumNumInvolvedEngs;
        syncScheme.sobResetId           = m_sobResetId;
    }

    // If CME exist (gaudi3), fill out the SOB reset CME task on the top node
    if (m_isCmeExist)
    {
        std::list<NodeROI>& logicalRois = *(g.GetNodeROIs(*topNode.itrNode));
        unsigned            pipeLevel   = topNode.pipeLevel;
        HB_ASSERT(pipeLevel < logicalRois.size(), "pipeLevel OOB");
        std::list<NodeROI>::iterator itrRoi            = std::next(logicalRois.begin(), pipeLevel);
        itrRoi->cmeTasks.sobReset.sobResetTotalNumEngs = accumNumInvolvedEngs;
        itrRoi->cmeTasks.sobReset.sobResetId           = m_sobResetId;
    }

    // Clean data structures for next reset (keep m_highestNodeExeIndexAtReset intact)
    m_engineCtx.clear();
    m_currentBalance.clear();
    m_fullyBalancedPin.clear();
    m_allBalanced.set();

    // Return the resume-point for graph traversing which is the next pipeline level of the top node. If there is
    // no next pipeline level for the top node, then we should resume from the beginning of the following node.
    if (topNode.pipeLevel + 1 == g.GetNodeROIs(*topNode.itrNode)->size())
    {
        return {++topNode.itrNode, 0};
    }
    else
    {
        return {topNode.itrNode, ++topNode.pipeLevel};
    }
}

unsigned SyncSchemeManagerArc::SobResetManager::getMaxRelIdx(const std::list<NodeROI*>& rois) const
{
    unsigned maxRelSoIdx = 0;

    auto findMaxRelIdx = [&](const TensorROIVector& rois) {
        for (auto roi : rois)
        {
            for (auto sroi : *roi.m_overlapRoi.subRois)
            {
                maxRelSoIdx = std::max(maxRelSoIdx, sroi.relSoIdx);
            }
        }
    };

    for (auto roi : rois)
    {
        findMaxRelIdx(roi->inputRois);
        findMaxRelIdx(roi->outputRois);
    }

    return maxRelSoIdx;
}

unsigned SyncSchemeManagerArc::SobResetManager::getAccumSignals(const std::list<NodeROI*>& rois) const
{
    return std::accumulate(rois.begin(), rois.end(), 0, [](unsigned accum, NodeROI* roi) {
        return accum + roi->numSignals;
    });
}

void SyncSchemeManagerArc::SobResetManager::makeSureWeCanDoTheReset() const
{
    if (m_fullyBalancedPin.empty())
    {
        LOG_ERR(SYNC_SCHEME, "Failed to reset SOBs in the middle of the graph - there is no opportunity to do that");
        throw PassFailedException();
    }

    // We need to enforce the following rule:
    //   Excluding the very first reset, each logical engine (i.e., TPC/MME/DMA/ROT) that participates in the current
    //   reset must also participated in the previous reset.
    if (m_highestNodeExeIndexAtReset > 0)
    {
        for (const auto& engBalance : m_fullyBalancedPin)
        {
            if (!m_logicalIdsInPrevReset.test(engBalance.first))
            {
                // For details and solution see SW-89803
                LOG_ERR(SYNC_SCHEME, "Failed to reset SOBs in the middle of the graph - participants violation");
                throw PassFailedException();
            }
        }
    }
}

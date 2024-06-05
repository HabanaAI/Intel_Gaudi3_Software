#include "generate_cache_maintenance_pass.h"

#include "gaudi3_code_generator.h"
#include "gaudi3_graph.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "types.h"
#include "types_exception.h"

unsigned CacheMaitenanceTasks::getRoiSobResetId(const NodePtr& n, unsigned roiIndex)
{
    unsigned myResetId = 0;

    if (n->getNodeAnnotation().arcSyncScheme.size() > roiIndex)
    {
        myResetId = n->getNodeAnnotation().arcSyncScheme[roiIndex].sobResetId;
    }

    return myResetId;
}

void CacheMaitenanceTasks::updateResetSobIdsMap(const NodePtr& n, unsigned roiIndex)
{
    unsigned myResetId = getRoiSobResetId(n, roiIndex);

    unsigned queueId = gaudi3::deviceTypeToLogicalQueue(m_graph->getNodeUtility().getNodeDeviceType(n), *n);

    if (myResetId > m_resetSobIds[queueId])
    {
        m_resetSobIds[queueId] = myResetId;

        LOG_TRACE(CACHE_MAINT, "Updating Sob ResetId: {} for queueId: {}", myResetId, queueId);
    }
}

void CacheMaitenanceTasks::resolveRoiSobResetId(DependencyMap&                                deps,
                                                const std::unordered_map<unsigned, unsigned>& resetSobIds,
                                                unsigned&                                     maxSobResetId)
{
    maxSobResetId = 0;

    // Find the max sobResetId
    for (auto const& resetSob : resetSobIds)
    {
        if (resetSob.second > maxSobResetId)
        {
            maxSobResetId = resetSob.second;
        }
    }

    LOG_DEBUG(CACHE_MAINT, "Roi Sob ResetId: {}", maxSobResetId);

    // Iterate all resetSobIds: for engine which has a value smaller than max reset id - remove its
    // instance from original DependencyMap since its value represent the pre-reset sob value
    for (auto const& resetSob : resetSobIds)
    {
        if (resetSob.second < maxSobResetId)
        {
            deps.erase(resetSob.first);

            LOG_DEBUG(
                CACHE_MAINT,
                "Removing queue: {} from DependencyMap since its SobResetId: {} is smaller then maxSobResetId: {}",
                resetSob.first,
                resetSob.second,
                maxSobResetId);
        }
    }
}

unsigned CacheMaitenanceTasks::getRoiSobValue(const NodePtr& n, unsigned roiIndex)
{
    // In case an ROI emitted signal is not set, we will extract its SOB value from the following ROIs
    for (unsigned i = roiIndex; i < n->getNodeAnnotation().arcSyncScheme.size(); i++)
    {
        if (n->getNodeAnnotation().arcSyncScheme[i].emittedSigVal.is_set())
        {
            return n->getNodeAnnotation().arcSyncScheme[i].emittedSigVal.value();
        }
    }

    HB_ASSERT(0, "Invalid sync scheme value");
    return 0;
}

unsigned CacheMaitenanceTasks::getSobValue(const NodePtr& n,
                                           NodeROI&       roi,
                                           unsigned       roiIndex,
                                           unsigned       tensorIndex,
                                           unsigned       logicalQueueId,
                                           bool           bIsInput)
{
    unsigned sobVal = 0;

    if (logicalQueueId == gaudi3::DEVICE_MME_LOGICAL_QUEUE && bIsInput)
    {
        // Special handling for MME input operands. We would like to signal as early as possible - once the input TensorROI is read.
        // Note that for MME/output we set the emitted signal of NodeROI since the brain will set to all NodeROI outputRois the same MCID
        if (m_prevMMENode != nullptr)
        {
            sobVal = m_prevMMENode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value();
        }
        if (roi.inputRois[tensorIndex].m_overlapRoi.subRois->size())
        {
            sobVal += roi.inputRois[tensorIndex].m_overlapRoi.subRois->back().relSoIdx + 1;
        }
        else
        {
            // This is a work-around: in rare cases, the m_overlapRoi.subRois is empty so we use the ROI emittedSignal as fallback
            // (till issue is resolved in JIRA #137000)
            sobVal = getRoiSobValue(n, roiIndex);
        }

        LOG_TRACE(CACHE_MAINT, "MME operand: {}, SOB value: {}", tensorIndex, sobVal);
    }
    else
    {
        sobVal = getRoiSobValue(n, roiIndex);
    }

    return sobVal;
}

void CacheMaitenanceTasks::processCacheMetaDataList(const NodePtr&              n,
                                                    NodeROI&                    roi,
                                                    unsigned                    roiIndex,
                                                    std::vector<CacheMetaData>& cacheMetaDataList,
                                                    bool                        bIsInput)
{
    unsigned queueId = gaudi3::deviceTypeToLogicalQueue(m_graph->getNodeUtility().getNodeDeviceType(n), *n);

    for (int i = 0; i < cacheMetaDataList.size(); i++)
    {
        // Skip shape tensors
        // Note: due to a bug in calculateMmeLinearRanges() (SW-139211) we may get transpose node with 1 input/output
        // while its ROI has 2 roiInput/roiOutput and therefore, for the time being, we need to verify the
        // cacheMetaDataList index is within input/output range
        if (bIsInput)
        {
            if (i >= n->getInputs().size() || n->getInput(i) == nullptr || n->getInput(i)->isShapeTensor()) continue;
        }
        else
        {
            if (i >= n->getOutputs().size() || n->getOutput(i) == nullptr || n->getOutput(i)->isShapeTensor()) continue;
        }

        CacheMetaData md = cacheMetaDataList[i];

        if (md.mcid == 0)
        {
            HB_ASSERT(md.cmAction == NOP, "MCID 0 cannot be associated with DEGRADE/DISCARD action");
            continue;
        }
        HB_ASSERT(md.cmAction != NOP, "NOP action cannot be associated with MCID > 0");

        unsigned sobValue = getSobValue(n, roi, roiIndex, i, queueId, bIsInput);

        m_brainMcid2cmROIInfo[md.cmAction][md.mcid].deps[queueId]        = sobValue;
        m_brainMcid2cmROIInfo[md.cmAction][md.mcid].highestExeIdxRoi     = &roi;
        m_brainMcid2cmROIInfo[md.cmAction][md.mcid].resetSobIds[queueId] = m_resetSobIds[queueId];

        LOG_TRACE(CACHE_MAINT,
                  "Found Mcid: {} (Action: {}) on node: {}, orderIndex: {}, sobValue: {}, queueId: {}",
                  md.mcid,
                  md.cmAction,
                  n->getNodeName(),
                  n->getExecutionOrderedIndex(),
                  sobValue,
                  queueId);
    }
}

void CacheMaitenanceTasks::generateCacheMaitenanceTasks()
{
    LOG_DEBUG(CACHE_MAINT, "Generating cme tasks:");
    for (unsigned i = DEGRADE; i <= DISCARD; i++)
    {
        for (auto const& mcidcmROIInfo : m_brainMcid2cmROIInfo[i])
        {
            if (mcidcmROIInfo.second.highestExeIdxRoi == nullptr)
            {
                LOG_ERR(CACHE_MAINT,
                        "No valid ROI to place the cme task on for MCID: {} (Action: {})",
                        mcidcmROIInfo.first,
                        i);
                continue;
            }

            if (!mcidcmROIInfo.second.valid)
            {
                LOG_DEBUG(CACHE_MAINT, "Brain Mcid: {} (Action: {}) is not valid - skipping ", mcidcmROIInfo.first, i);
                continue;
            }

            CmCmd cmCmd;

            cmCmd.op   = (CacheMaintenanceAction)i;
            cmCmd.mcid = m_brainMcid2RealMcid[i][mcidcmROIInfo.first];  // using new real MCID from Map
            cmCmd.deps = mcidcmROIInfo.second.deps;

            mcidcmROIInfo.second.highestExeIdxRoi->cmeTasks.cmCmds.push_back(cmCmd);

            LOG_DEBUG(CACHE_MAINT,
                      "New Mcid: {} (Action: {}) (old Mcid: {}), Depenceny Map (Logical queue Id, SOB value):",
                      cmCmd.mcid,
                      i,
                      mcidcmROIInfo.first);

            for (auto& item : cmCmd.deps)
            {
                LOG_DEBUG(CACHE_MAINT, "queue: {}, value: {}", item.first, item.second);
            }
        }
    }
    LOG_DEBUG(CACHE_MAINT, "Done generating cme tasks");
}

NodeROI* CacheMaitenanceTasks::getNodeRoiWithMaxExeIndex()
{
    NodeROI* roi         = nullptr;
    uint32_t maxExeIndex = 0;

    for (auto deviceRoiPair : m_logicalQueue2LastActiveNodeRoi)
    {
        if (deviceRoiPair.second.nodeExeIndex > maxExeIndex)
        {
            roi         = deviceRoiPair.second.roi;
            maxExeIndex = deviceRoiPair.second.nodeExeIndex;
        }
    }

    return roi;
}

void CacheMaitenanceTasks::handleRollover(unsigned rolloverId)
{
    // Process all active queues. If we already found NodeROI - put the rolloverId on last active NodeROI.
    // If engine is active in the "future" - put the rolloverId on graphAnnotation
    uint8_t rolloverEngineBitmap = 0;

    for (unsigned i = 0; i < LOGICAL_QUEUE_MAX_ID; i++)
    {
        if (m_logicalQueue2LastActiveNodeRoi.find(i) == m_logicalQueue2LastActiveNodeRoi.end())
        {
            if (m_activeLogicalQueues[i])
            {
                // Although the queue is active, we did not encounter with NodeROI yet - update graph annotation
                m_graph->getGraphAnnotation().devicePreNodesRolloverIds[gaudi3::logicalQueueToDeviceType(i)].insert(rolloverId);
            }
        }
        else
        {
            if ((i == gaudi3::DEVICE_XPS_LOGICAL_QUEUE || i == gaudi3::DEVICE_MME_LOGICAL_QUEUE) &&
                m_logicalQueue2LastActiveNodeRoi[i].roi->rolloverIds.size() > 0)
            {
                LOG_ERR(CACHE_MAINT, "Roi contains 2 rollover Ids");
                throw PassFailedException();
            }

            m_logicalQueue2LastActiveNodeRoi[i].roi->rolloverIds.push_back(rolloverId);

            if (i == gaudi3::DEVICE_XPS_LOGICAL_QUEUE || i == gaudi3::DEVICE_MME_LOGICAL_QUEUE)
            {
                rolloverEngineBitmap |= 1;
            }
            else if (i == gaudi3::DEVICE_ROT_LOGICAL_QUEUE)
            {
                rolloverEngineBitmap |= 2;
            }
        }
    }

    // CME Rollover command
    NodeROI* prevNodeROI = getNodeRoiWithMaxExeIndex();

    prevNodeROI->cmeTasks.rollover.doRollover           = true;
    prevNodeROI->cmeTasks.rollover.rolloverId           = rolloverId;
    prevNodeROI->cmeTasks.rollover.rolloverEngineBitmap = rolloverEngineBitmap;

    LOG_DEBUG(CACHE_MAINT, "Generated rollover task for rollover Id: {}, Bitmap:{}", rolloverId, rolloverEngineBitmap);
}

unsigned CacheMaitenanceTasks::detectRollover(const std::vector<CacheMetaData>& cacheMetaDataList)
{
    PhysicalMcid dummy      = 0;
    unsigned     rolloverId = 0;
    unsigned     ret        = 0;

    for (int i = 0; i < cacheMetaDataList.size(); i++)
    {
        if (cacheMetaDataList[i].mcid > 0 && cacheMetaDataList[i].cmAction == DISCARD)
        {
            m_graph->getCodeGenerator()->getMcidConverter().convertDiscard(cacheMetaDataList[i].mcid, dummy, rolloverId);
            ret = std::max(ret, rolloverId);
        }
    }
    return ret;
}

void CacheMaitenanceTasks::detectRollover()
{
    unsigned currentRolloverId = 0;

    for (const NodePtr& n : m_graph->getExeSortedNodes())
    {
        if (n == nullptr || n->isLogicalOperation()) continue;

        unsigned queueId      = gaudi3::deviceTypeToLogicalQueue(m_graph->getNodeUtility().getNodeDeviceType(n), *n);
        uint32_t nodeExeIndex = n->getExecutionOrderedIndex();

        for (auto& roi : (*(m_graph->GetNodeROIs(n))))
        {
            unsigned rolloverId = 0;
            rolloverId          = detectRollover(roi.inputsCacheMetaData);

            if (rolloverId > currentRolloverId)
            {
                LOG_DEBUG(CACHE_MAINT,
                          "Detected new rollover Id: {} on node: {} (input)",
                          rolloverId,
                          n->getNodeName());
                currentRolloverId = rolloverId;
                handleRollover(rolloverId);
            }

            rolloverId = detectRollover(roi.outputsCacheMetaData);

            if (rolloverId > currentRolloverId)
            {
                LOG_DEBUG(CACHE_MAINT,
                          "Detected new rollover Id: {} on node: {} (output)",
                          rolloverId,
                          n->getNodeName());
                currentRolloverId = rolloverId;
                handleRollover(rolloverId);
            }

            // update last active ROI and nodeExeIndex for logical queue
            NodeRoiExeIndex roiExeIndex;

            roiExeIndex.nodeExeIndex                  = nodeExeIndex;
            roiExeIndex.roi                           = &roi;
            m_logicalQueue2LastActiveNodeRoi[queueId] = roiExeIndex;
        }
    }
}

void CacheMaitenanceTasks::buildDependncyMap()
{
    for (const NodePtr& n : m_graph->getExeSortedNodes())
    {
        if (n == nullptr || n->isLogicalOperation()) continue;

        unsigned roiIndex = 0;
        for (auto& roi : (*(m_graph->GetNodeROIs(n))))
        {
            // iterate over inputsCacheMetaData & outputsCacheMetaData
            processCacheMetaDataList(n, roi, roiIndex, roi.inputsCacheMetaData);
            processCacheMetaDataList(n, roi, roiIndex, roi.outputsCacheMetaData, false);
            updateResetSobIdsMap(n, roiIndex);
            roiIndex++;
        }

        if (gaudi3::deviceTypeToLogicalQueue(m_graph->getNodeUtility().getNodeDeviceType(n), *n) == gaudi3::DEVICE_MME_LOGICAL_QUEUE)
        {
            m_prevMMENode = n;
        }
    }
}

void CacheMaitenanceTasks::removeDupliactedMcidFromCacheMetaData(std::vector<CacheMetaData>& cacheMetaDataList,
                                                                 BrainMcid2ExeOrder&         brainMcid2ExeOrder,
                                                                 uint32_t                    nodeExeOrder)
{
    for (int i = 0; i < cacheMetaDataList.size(); i++)
    {
        const CacheMetaData& md = cacheMetaDataList[i];

        if (md.mcid > 0)
        {
            if (brainMcid2ExeOrder[md.cmAction].find(md.mcid) == brainMcid2ExeOrder[md.cmAction].end())
            {
                // first time MCID
                brainMcid2ExeOrder[md.cmAction][md.mcid] = nodeExeOrder;
                LOG_TRACE(CACHE_MAINT,
                          "Mcid: {} (Action: {}) has node exe order index: {}",
                          md.mcid,
                          md.cmAction,
                          nodeExeOrder);
            }
            else
            {
                if (brainMcid2ExeOrder[md.cmAction][md.mcid] != nodeExeOrder)
                {
                    // MCID for DEGRADE/DISCARD was set on higher exeution order node - reset current MCID/Action
                    LOG_TRACE(
                        CACHE_MAINT,
                        "Mcid: {} (Action: {}) with higher node exe order index: {} was already found (current: {})",
                        md.mcid,
                        md.cmAction,
                        brainMcid2ExeOrder[md.cmAction][md.mcid],
                        nodeExeOrder);
                    cacheMetaDataList[i].mcid     = 0;
                    cacheMetaDataList[i].cmAction = NOP;
                }
            }
        }
    }
}

void CacheMaitenanceTasks::removeDupliactedMcidsByExeOrder()
{
    // Iterate over all nodes/ROIs. For duplicate MCID/ACTION - leave only the cache MD for the node with highest
    // exexcution order
    BrainMcid2ExeOrder brainMcid2ExeOrder;

    const NodeVector& allNodes = m_graph->getExeSortedNodes();

    for (NodeVector::const_reverse_iterator n = allNodes.rbegin(); n != allNodes.rend(); ++n)
    {
        if ((*n) == nullptr || (*n)->isLogicalOperation()) continue;

        std::list<NodeROI>* roiList = m_graph->GetNodeROIs(*n);

        for (std::list<NodeROI>::reverse_iterator roi = (*roiList).rbegin(); roi != (*roiList).rend(); ++roi)
        {
            removeDupliactedMcidFromCacheMetaData((*roi).outputsCacheMetaData,
                                                  brainMcid2ExeOrder,
                                                  (*n)->getExecutionOrderedIndex());
            removeDupliactedMcidFromCacheMetaData((*roi).inputsCacheMetaData,
                                                  brainMcid2ExeOrder,
                                                  (*n)->getExecutionOrderedIndex());
        }
    }
}

void CacheMaitenanceTasks::allocateRealMcidForList(std::vector<CacheMetaData>&                   cacheMetaDataList,
                                                   std::array<LogicalMcid, MAX_CM_ACTION_TYPES>& cmOpMcid)
{
    for (int i = 0; i < cacheMetaDataList.size(); i++)
    {
        const CacheMetaData& md = cacheMetaDataList[i];

        if (md.mcid > 0)
        {
            if (m_brainMcid2RealMcid[md.cmAction].find(md.mcid) == m_brainMcid2RealMcid[md.cmAction].end())
            {
                LOG_DEBUG(CACHE_MAINT,
                          "Mcid: {} (Action: {}) was mapped to new Mcid: {}",
                          md.mcid,
                          md.cmAction,
                          cmOpMcid[md.cmAction]);
                m_brainMcid2RealMcid[md.cmAction][md.mcid] = cmOpMcid[md.cmAction]++;
            }
            else
            {
                LOG_DEBUG(CACHE_MAINT,
                          "Mcid: {} (Action: {}) was already mapped to new Mcid: {}",
                          md.mcid,
                          md.cmAction,
                          cmOpMcid[md.cmAction] - 1);
            }
            // updating metadata with real MCID
            cacheMetaDataList[i].mcid = m_brainMcid2RealMcid[md.cmAction][md.mcid];
        }
    }
}

void CacheMaitenanceTasks::allocateRealMcids()
{
    // Allocate real MCIDs based on node/ROI execution order
    std::array<LogicalMcid, MAX_CM_ACTION_TYPES> cmOpMcid;

    cmOpMcid[DEGRADE] = 1;
    cmOpMcid[DISCARD] = 1;

    for (const NodePtr& n : m_graph->getExeSortedNodes())
    {
        if (n == nullptr || n->isLogicalOperation()) continue;

        for (auto& roi : (*(m_graph->GetNodeROIs(n))))
        {
            allocateRealMcidForList(roi.inputsCacheMetaData, cmOpMcid);
            if (LOG_LEVEL_AT_LEAST_DEBUG(CACHE_MAINT))
            {
                for (int i = 0; i < roi.inputsCacheMetaData.size(); i++)
                {
                    if (i >= n->getInputs().size() || n->getInput(i) == nullptr || n->getInput(i)->isShapeTensor()) continue;
                    const CacheMetaData& md = roi.inputsCacheMetaData[i];
                    if (md.cmAction == NOP) continue;
                    LOG_DEBUG(CACHE_MAINT,
                              "Mcid Final Allocation: node: {}, INPUT tensor: {}, mcid: {}, action: {}",
                              n->getNodeName(),
                              n->getInput(i)->getName(),
                              md.mcid,
                              md.cmAction);
                }
            }

            allocateRealMcidForList(roi.outputsCacheMetaData, cmOpMcid);
            if (LOG_LEVEL_AT_LEAST_DEBUG(CACHE_MAINT))
            {
                for (int i = 0; i < roi.outputsCacheMetaData.size(); i++)
                {
                    if (i >= n->getOutputs().size() || n->getOutput(i) == nullptr || n->getOutput(i)->isShapeTensor()) continue;
                    const CacheMetaData& md = roi.outputsCacheMetaData[i];
                    if (md.cmAction == NOP) continue;
                    LOG_DEBUG(CACHE_MAINT,
                              "Mcid Final Allocation: node: {}, OUTPUT tensor: {}, mcid: {}, action: {}",
                              n->getNodeName(),
                              n->getOutput(i)->getName(),
                              md.mcid,
                              md.cmAction);
                }
            }
        }
    }
}

void CacheMaitenanceTasks::generateDependencyMapKey(const DependencyMap& deps, DependencyMapKey& depsKey)
{
    // generate dependency map key from DependencyMap
    for (auto mapItem : deps)
    {
        depsKey.engineSobValue[mapItem.first] = mapItem.second;
    }
}

void CacheMaitenanceTasks::optimizeDuplicatedDependencyMapMcid()
{
    std::array<std::map<DependencyMapKey, LogicalMcid>, MAX_CM_ACTION_TYPES> mDependencyMap2BrainMcid;
    DupBrainMcid2Mcid                                                        mDupBrainMcid2Mcid;

    for (unsigned i = DEGRADE; i <= DISCARD; i++)
    {
        for (auto& mcidcmROIInfo : m_brainMcid2cmROIInfo[i])
        {
            DependencyMapKey depsKey = {};

            generateDependencyMapKey(mcidcmROIInfo.second.deps, depsKey);

            if (mDependencyMap2BrainMcid[i].find(depsKey) == mDependencyMap2BrainMcid[i].end())
            {
                mDependencyMap2BrainMcid[i][depsKey] = mcidcmROIInfo.first;
                LOG_DEBUG(CACHE_MAINT, "Inserting Mcid: {} (Action: {}) key to Dependency Map", mcidcmROIInfo.first, i);
            }
            else
            {
                mDupBrainMcid2Mcid[i][mcidcmROIInfo.first] = mDependencyMap2BrainMcid[i][depsKey];
                mcidcmROIInfo.second.valid                 = false;  // possible to remove the entry entirely
                LOG_DEBUG(CACHE_MAINT,
                          "Mcid: {} (Action: {}) has same Dependency Map key as Mcid: {}",
                          mcidcmROIInfo.first,
                          i,
                          mDependencyMap2BrainMcid[i][depsKey]);
            }
        }
    }

    // Now update the cache metadata for the duplicated MCIDs
    updateDuplicatedMcidCacheMetaData(mDupBrainMcid2Mcid);
}

void CacheMaitenanceTasks::updateDuplicatedMcidCacheMetaDataList(std::vector<CacheMetaData>& cacheMetaDataList,
                                                                 DupBrainMcid2Mcid&          mDupBrainMcid2Mcid)
{
    for (int i = 0; i < cacheMetaDataList.size(); i++)
    {
        const CacheMetaData& md = cacheMetaDataList[i];

        if (md.mcid > 0)
        {
            if (mDupBrainMcid2Mcid[md.cmAction].find(md.mcid) != mDupBrainMcid2Mcid[md.cmAction].end())
            {
                LOG_DEBUG(CACHE_MAINT,
                          "Mcid: {} (Action: {}) has same dependency map as Mcid: {}",
                          cacheMetaDataList[i].mcid,
                          md.cmAction,
                          mDupBrainMcid2Mcid[md.cmAction][md.mcid]);
                cacheMetaDataList[i].mcid = mDupBrainMcid2Mcid[md.cmAction][md.mcid];
            }
        }
    }
}

void CacheMaitenanceTasks::updateDuplicatedMcidCacheMetaData(DupBrainMcid2Mcid& mDupBrainMcid2Mcid)
{
    // Update the cache metadata for the duplicated MCIDs
    LOG_DEBUG(CACHE_MAINT, "Updating duplicated Mcids (same dependency map) in cache meta data");
    for (const NodePtr& n : m_graph->getExeSortedNodes())
    {
        if (n == nullptr || n->isLogicalOperation()) continue;

        for (auto& roi : (*(m_graph->GetNodeROIs(n))))
        {
            updateDuplicatedMcidCacheMetaDataList(roi.inputsCacheMetaData, mDupBrainMcid2Mcid);
            updateDuplicatedMcidCacheMetaDataList(roi.outputsCacheMetaData, mDupBrainMcid2Mcid);
        }
    }
}

void CacheMaitenanceTasks::init()
{
    // 1. Initialize the m_resetSobIds map with only the engines that participate in the topology
    // 2. Collect all active logical queues in topology
    for (auto const& queueToIndex : m_graph->getLogicalQueueToMaxExecutionIndex())
    {
        m_resetSobIds[queueToIndex.first]         = 0;
        m_activeLogicalQueues[queueToIndex.first] = true;
        LOG_TRACE(CACHE_MAINT, "Init active queue: {} in ResetSobIdsMap", queueToIndex.first);
    }
    // TPC considered active for rollover even if not used in topology
    m_activeLogicalQueues[gaudi3::DEVICE_TPC_LOGICAL_QUEUE] = true;
}

void CacheMaitenanceTasks::removeRedundantDependenciesForMcid()
{
    unsigned             sobResetId       = 0;
    bool                 bValidSobResetId = true;
    Gaudi3CodeGenerator& codeGen          = dynamic_cast<Gaudi3CodeGenerator&>(*(m_graph->getCodeGenerator()));

    for (unsigned i = DEGRADE; i <= DISCARD; i++)
    {
        for (auto& mcidcmROIInfo : m_brainMcid2cmROIInfo[i])
        {
            if (mcidcmROIInfo.second.deps.size() > 1)
            {
                LOG_DEBUG(CACHE_MAINT,
                          "Trying to remove dependenies for Mcid: {} (Action: {}) - Depenceny Map Size: {}",
                          mcidcmROIInfo.first,
                          i,
                          mcidcmROIInfo.second.deps.size());

                resolveRoiSobResetId(mcidcmROIInfo.second.deps, mcidcmROIInfo.second.resetSobIds, sobResetId);

                // We may get smallar DependencyMap
                if (mcidcmROIInfo.second.deps.size() > 1)
                {
                    codeGen.removeRedundantDependencies(mcidcmROIInfo.second.deps, sobResetId);
                }

                LOG_DEBUG(CACHE_MAINT,
                          "After dependencies removal for Mcid: {} (Action: {}) - Depenceny Map Size: {}",
                          mcidcmROIInfo.first,
                          i,
                          mcidcmROIInfo.second.deps.size());
            }
        }
    }
}

void CacheMaitenanceTasks::optimizeCacheMaitenanceData()
{
    // for MCIDs with multi dependencies: drop dependencies that are satisfied by other dependencies according to the
    // overlap info
    removeRedundantDependenciesForMcid();

    // different MCIDs that wait for the same dependency shall be replaced by the same MCID
    optimizeDuplicatedDependencyMapMcid();

    // same MCIDs used by different nodes shall be replaced by MCID 0 / NOP except of the last node
    removeDupliactedMcidsByExeOrder();

    // Replace BrainMCIDs with RealMCIDs - just before generating the CME tasks
    allocateRealMcids();
}

void CacheMaitenanceTasks::executePass()
{
    init();

    // Build Dependncy Map based on brain MCIDs
    buildDependncyMap();

    // CacheMaitenanceTasks pass optimizations
    optimizeCacheMaitenanceData();

    // Detect rollover
    detectRollover();

    // generate the CME tasks
    generateCacheMaitenanceTasks();
}

namespace gaudi3
{
// The pass entry function
bool generateCacheMaitenanceTasks(Gaudi3Graph& g)
{
    try
    {
        CacheMaitenanceTasks cm(g);
        cm.executePass();

        return true;
    }
    catch (const SynapseException& e)
    {
        LOG_CRITICAL(GC, "generateCacheMaitenanceTasks pass failed: {}", e.what());
        return false;
    }
}

}  // namespace gaudi3

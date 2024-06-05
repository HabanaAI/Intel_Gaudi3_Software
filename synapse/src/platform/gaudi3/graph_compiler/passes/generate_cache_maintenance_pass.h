#pragma once

#include "habana_device_types.h"
#include "node.h"
#include "gaudi3_graph.h"

using BrainMcid2ExeOrder = std::array<std::unordered_map<LogicalMcid, uint32_t>, MAX_CM_ACTION_TYPES>;
using DupBrainMcid2Mcid  = std::array<std::unordered_map<LogicalMcid, LogicalMcid>, MAX_CM_ACTION_TYPES>;

union DependencyMapKey
{
    struct
    {
        uint16_t engineSobValue[LOGICAL_QUEUE_MAX_ID];  // 0 - MME, 1 - TPC, 2 - ROT, 3 - XPOSE
    };
    uint64_t all = 0;

    bool operator<(const DependencyMapKey& rhs) const { return all < rhs.all; }
};

struct CmROIInfo
{
    DependencyMap                          deps             = {};
    std::unordered_map<unsigned, unsigned> resetSobIds      = {};
    NodeROI*                               highestExeIdxRoi = nullptr;
    bool                                   valid            = true;
};

struct NodeRoiExeIndex
{
    uint32_t nodeExeIndex;
    NodeROI* roi;
};

class CacheMaitenanceTasks
{
public:
    CacheMaitenanceTasks(Gaudi3Graph& g) { m_graph = &g; }
    void executePass();

private:
    void init();
    void buildDependncyMap();
    void processCacheMetaDataList(const NodePtr&              n,
                                  NodeROI&                    roi,
                                  unsigned                    roiIndex,
                                  std::vector<CacheMetaData>& cacheMetaDataList,
                                  bool                        bIsInput = true);

    void optimizeCacheMaitenanceData();
    void removeRedundantDependenciesForMcid();
    void optimizeDuplicatedDependencyMapMcid();
    void updateDuplicatedMcidCacheMetaData(DupBrainMcid2Mcid& mDupBrainMcid2Mcid);
    void updateDuplicatedMcidCacheMetaDataList(std::vector<CacheMetaData>& cacheMetaDataList,
                                               DupBrainMcid2Mcid&          mDupBrainMcid2Mcid);
    void generateDependencyMapKey(const DependencyMap& deps, DependencyMapKey& depsKey);
    void removeDupliactedMcidsByExeOrder();
    void removeDupliactedMcidFromCacheMetaData(std::vector<CacheMetaData>& cacheMetaDataList,
                                               BrainMcid2ExeOrder&         bMcidArray,
                                               uint32_t                    nodeExeOrder);
    void allocateRealMcids();
    void allocateRealMcidForList(std::vector<CacheMetaData>&                   cacheMetaDataList,
                                 std::array<LogicalMcid, MAX_CM_ACTION_TYPES>& cmOpMcid);

    // rollover
    void detectRollover();
    void handleRollover(unsigned rolloverId);

    unsigned detectRollover(const std::vector<CacheMetaData>& cacheMetaDataList);
    NodeROI* getNodeRoiWithMaxExeIndex();

    void generateCacheMaitenanceTasks();

    void updateResetSobIdsMap(const NodePtr& n, unsigned roiIndex);
    void resolveRoiSobResetId(DependencyMap&                                deps,
                              const std::unordered_map<unsigned, unsigned>& resetSobIds,
                              unsigned&                                     maxSobResetId);

    unsigned getRoiSobResetId(const NodePtr& n, unsigned roiIndex);

    unsigned getRoiSobValue(const NodePtr& n, unsigned roiIndex);

    unsigned getSobValue(const NodePtr& n,
                         NodeROI&       roi,
                         unsigned       roiIndex,
                         unsigned       tensorIndex,
                         unsigned       logicalQueueId,
                         bool           bIsInput);

    Gaudi3Graph*  m_graph;
    NodePtr       m_prevMMENode = nullptr;
    DependencyMap m_resetSobIds;

    std::array<std::unordered_map<LogicalMcid, CmROIInfo>, MAX_CM_ACTION_TYPES>   m_brainMcid2cmROIInfo;
    std::array<std::unordered_map<LogicalMcid, LogicalMcid>, MAX_CM_ACTION_TYPES> m_brainMcid2RealMcid;
    // rollover
    std::unordered_map<unsigned, NodeRoiExeIndex> m_logicalQueue2LastActiveNodeRoi;
    bool                                          m_activeLogicalQueues[LOGICAL_QUEUE_MAX_ID] = {0};
};

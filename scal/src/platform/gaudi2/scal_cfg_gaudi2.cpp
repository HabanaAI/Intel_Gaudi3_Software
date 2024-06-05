#include <assert.h>
#include <cstddef>
#include <cstring>
#include <string>
#include <fstream>
#include "scal.h"
#include "scal_utilities.h"
#include "logger.h"
#include "scal_data_gaudi2.h"
#include "scal_gaudi2.h"
#include "common/cfg_parsing_helper.h"
#include "scal_base.h"
#include "infra/sync_mgr.hpp"

// clang-format off



int Scal_Gaudi2::parseConfigFile(const std::string & configFileName)
try
{
    int ret;

    // open the file and get the json handle
    scaljson::json json;
    ret = openConfigFileAndParseJson(configFileName, json);
    if (ret != SCAL_SUCCESS) return ret;

    // get the config version
    ConfigVersion version;
    getConfigVersion(json, version);

    json = getAsicJson(json, "gaudi2");

    // get the path for the fw binaries files
    parseBinPath(json, version);

    // Parse BackwardCompatibleness flags
    parseBackwardCompatibleness(json);

    // memory settings: global hbm size, host shared size, hbm shared size
    MemoryGroups memoryGroups;
    parseMemorySettings(json, version, memoryGroups);

    // define the schedulers and engine clusters (compute_tpc,media_tpc,mme,edma,pdma,nic)
    parseCores(json, version, memoryGroups);

    // define streams_sets (compute,pdma, network_*)
    parseStreams(json, version);

    // define sos pools, monitor pools, completion_queues etc.
    parseSyncInfo(json, version);

    // Parses the jason config file and populates the information to the class members
    // - fw search path (m_fwImageSearchPath)
    // - size of the hbm allocator (m_hbmAllocator)
    // - the vector of pools (m_pools) - The size, type (host/hbm), index and name of each pool
    // - the addresses of the active sync managers  (m_syncManagers)
    // - the list of comp groups (m_completionGroups) - name, dcore, idxInDcore, soBase, soNum, longSO info
    // - The list of stream (m_streams) - name id and type
    // - the list of cores (m_cores) - For each active core: id, name, qman, dccmDevAddr, memory pools, streams and compGroups (and set back pointers)

    // TODO:
    return ret;
}
catch(ExceptionWithCode const & e)
{
    LOG_ERR(SCAL, "{}", e.what());
    return e.code();
}
catch(std::exception const & e)
{
    LOG_ERR(SCAL, "{}", e.what());
    return SCAL_INVALID_CONFIG;
}


void Scal_Gaudi2::parseBackwardCompatibleness(const scaljson::json &json)
try
{
}
CATCH_JSON()

// memory settings: global hbm size, host shared size, hbm shared size
void Scal_Gaudi2::parseMemorySettings(const scaljson::json &json, const ConfigVersion &version, MemoryGroups &groups)
try
{
    // in json file - the "memory" : section
    auto & memElement = json.at(c_config_key_memory);

    memElement.at(c_config_key_memory_tpc_barrier_in_sram).get_to(m_tpcBarrierInSram);

    std::vector<Pool> pools;
    memElement.at(c_config_key_memory_control_cores_memory_pools).get_to(pools);
    if (pools.empty())
    {
        THROW_INVALID_CONFIG(memElement,"error parsing mem element (empty array) {}", c_config_key_memory_control_cores_memory_pools);
    }

    for (unsigned i = 0; i < pools.size(); ++i)
    {
        Pool & pool = pools[i];
        pool.globalIdx = i;
        if (m_pools.count(pool.name) != 0)
        {
            THROW_INVALID_CONFIG(memElement, "error parsing mem element. name duplication {}", c_config_key_memory_control_cores_memory_pools);
        }
        if (pool.type == Pool::Type::HOST)
        {
            if (pool.size % c_host_pool_size_alignment)
            {
                THROW_INVALID_CONFIG(memElement,"error in pool size alignment {}. size: {} MB", c_host_pool_size_alignment, pool.size / 1024 / 1024);
            }
        }
        m_pools[pool.name] = pool;
    }

    for (const scaljson::json & groupJson : memElement[c_config_key_memory_memory_groups])
    {
        parseMemoryGroups(groupJson, version, groups);
    }

    std::string poolName = memElement.at(c_config_key_memory_binary_pool).get<std::string>();
    if (m_pools.count(poolName) == 0)
    {
        THROW_INVALID_CONFIG(memElement.at(c_config_key_memory_binary_pool), "error parsing mem element. pool name not found in m_pools", poolName);
    }
    m_binaryPool = &m_pools[poolName];
    LOG_INFO_F(SCAL, "Binary pool is {}", m_binaryPool->name);

    if (m_binaryPool->coreBase) //the binary pool must not be addressable by the core
    {
        THROW_INVALID_CONFIG(memElement,"error in binaryPool settings");
    }

    if (memElement.find(c_config_key_memory_global_pool) != memElement.end())
    {
        std::string globalPoolName = memElement.at(c_config_key_memory_global_pool).get<std::string>();
        if (m_pools.count(globalPoolName) == 0)
        {
            THROW_INVALID_CONFIG(memElement.at(c_config_key_memory_global_pool), "error parsing mem element. global pool name {} is not in m_pools", globalPoolName);
        }
        m_globalPool = &m_pools[globalPoolName];
    }
    else
    {
        m_globalPool = m_binaryPool;
    }
}
CATCH_JSON()


void Scal_Gaudi2::parseSchedulerCores(const scaljson::json &schedulersJson, const ConfigVersion &version, const MemoryGroups &groups)
try
{
    (void) version; // for future use

    for (const auto & sched : schedulersJson)
    {
        std::unique_ptr<Scheduler> pScheduler = std::make_unique<Scheduler>();
        Scheduler & scheduler = *pScheduler;
        scheduler.scal = this;
        scheduler.streams = std::vector<Stream *>(c_num_max_user_streams, nullptr);
        scheduler.isScheduler = true;

        sched.at(c_config_key_cores_schedulers_core).get_to(scheduler.arcName);
        sched.at(c_config_key_cores_schedulers_name).get_to(scheduler.name);
        sched.at(c_config_key_cores_schedulers_binary_name).get_to(scheduler.imageName);
        sched.at(c_config_key_cores_schedulers_qman).get_to(scheduler.qman);

        if (!arcName2CpuId(scheduler.arcName, scheduler.cpuId))
        {
            THROW_INVALID_CONFIG(sched,"error in arcName2CpuId arName: {} cpuId: {}", scheduler.arcName, scheduler.cpuId);
        }
        if ((scheduler.cpuId >= c_scheduler_nr) || m_cores[scheduler.cpuId])
        {
            THROW_INVALID_CONFIG(sched,"parse scheduler cores core.cpuId: {} c_scheduler_nr: {}", scheduler.cpuId, c_scheduler_nr);
        }

        bool isMmeSlave, isEngine;
        if (!arcName2DccmAddr(scheduler.arcName, scheduler.dccmDevAddress) ||
            !arcName2ArcType(scheduler.arcName, scheduler.arcFarm, isMmeSlave, isEngine) ||
            !schedulerName2DupEngLocalAddress(scheduler.arcName, scheduler.dupEngLocalDevAddress))
        {
            THROW_INVALID_CONFIG(sched, "error arcName2DccmAddr arName: {} cpuId: {}", scheduler.arcName, scheduler.cpuId);
        }
        if (isEngine && !isMmeSlave)
        {
            THROW_INVALID_CONFIG(sched, "error - {} is not a scheduler", scheduler.arcName);
        }

        // find Qman id by Qman name
        if (!arcName2QueueId(scheduler.qman, scheduler.qmanID))
        {
            THROW_INVALID_CONFIG(sched,"error in arcName2QueueId {}", c_config_key_cores_schedulers_qman);
        }

        scheduler.dccmMessageQueueDevAddress =
            scheduler.dccmDevAddress + getCoreAuxOffset(&scheduler) +
            (mmARC_FARM_ARC0_AUX_DCCM_QUEUE_PUSH_REG_0 % c_aux_block_size);

        std::string memGroup = sched.at(c_config_key_cores_schedulers_memory_group).get<std::string>();
        if (groups.find(memGroup) == groups.end())
        {
            THROW_INVALID_CONFIG(sched,"error finding group: {}", memGroup);
        }

        scheduler.pools = groups.at(memGroup).pools;
        scheduler.configPool = groups.at(memGroup).configPool;

        m_cores[scheduler.cpuId] = pScheduler.release();
    }
}
CATCH_JSON()

void Scal_Gaudi2::parseClusterQueues(const scaljson::json& queuesJson, const ConfigVersion& version, Cluster* cluster)
try
{
    (void)version; // for future use

    for (const auto& queueJson : queuesJson)
    {
        Scal::Cluster::Queue queue;

        queueJson.at(c_config_key_cores_engine_clusters_queues_index).get_to(queue.index);
        if (queue.index >= DCCM_QUEUE_COUNT)
        {
            THROW_INVALID_CONFIG(queueJson, "config of cluster {} has queue index {}, max supported is {}", cluster->name,
                    queue.index, DCCM_QUEUE_COUNT);
        }

        VALIDATE_JSON_NODE_IS_OBJECT(queueJson, c_config_key_cores_engine_clusters_queues_scheduler);

        auto queueSchedulerJson = queueJson[c_config_key_cores_engine_clusters_queues_scheduler];

        std::string coreName = queueSchedulerJson.at(c_config_key_cores_engine_clusters_queues_scheduler_name).get<std::string>();
        queue.scheduler = getCoreByName<Scheduler>(coreName);
        if (!queue.scheduler)
        {
            THROW_INVALID_CONFIG(queueJson, "queue scheduler {} is not a scheduler", coreName);
        }

        if (cluster->queues.find(queue.index) != cluster->queues.end())
        {
            // the same queue can be allocated twice for 2 dup triggers only for the SAME scheduler and group
            if (cluster->queues[queue.index].scheduler != queue.scheduler)
            {
                THROW_INVALID_CONFIG(queueJson, "error queue index {} apears twice in cluster {} with different schedulers {} vs {}",
                        queue.index, cluster->name, cluster->queues[queue.index].scheduler->name , queue.scheduler->name);
            }
        }

        std::vector<std::string> secondaryDupTriggers;
        std::string primaryDupTrigger;
        if (queueSchedulerJson.at(c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger).is_array())
        {
            queueSchedulerJson[c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger].get_to(secondaryDupTriggers);

            if (secondaryDupTriggers.empty())
            {
                THROW_INVALID_CONFIG(queueSchedulerJson[c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger],
                                     "error {} array is empty", c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger);
            }
            primaryDupTrigger = secondaryDupTriggers[0];
            secondaryDupTriggers.erase(secondaryDupTriggers.begin());
        }
        else
        {
            queueSchedulerJson[c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger].get_to(primaryDupTrigger);
        }

        Scal_Gaudi2::DupTrigger trigger;
        if (!getDupTriggerByName(primaryDupTrigger, trigger))
        {
            THROW_INVALID_CONFIG(queueJson, "invalid primary dup trigger name {}", primaryDupTrigger);
        }

        queue.dupTrigger = static_cast<Scal_Gaudi2::DupTrigger>(trigger);
        if (!getDupTriggerIndexByName(primaryDupTrigger, queue.dup_trans_data_q_index))
        {
            THROW_INVALID_CONFIG(queueJson, "invalid dup trigger name {}", primaryDupTrigger);
        }

        for (const auto& dupTriggerName : secondaryDupTriggers)
        {
            DupTrigger dupTrigger;
            if (!getDupTriggerByName(dupTriggerName, dupTrigger))
            {
                THROW_INVALID_CONFIG(queueJson, "invalid dup trigger name {}", dupTriggerName);
            }
            queue.secondaryDupTriggers.push_back(dupTrigger);
            unsigned dup_trans_data_q_index;
            if (!getDupTriggerIndexByName(dupTriggerName, dup_trans_data_q_index))
            {
                THROW_INVALID_CONFIG(queueJson, "invalid dup trigger name {}", dupTriggerName);
            }
            if (dup_trans_data_q_index > 0)
            {
                THROW_INVALID_CONFIG(queueJson, "invalid dup trigger offset for secondary dup trigger {} (offset must be 0)", dupTriggerName);
            }
        }

        VALIDATE_JSON_NODE_EXISTS(queueSchedulerJson, c_config_key_cores_engine_clusters_queues_scheduler_group);

        auto clustersQueuesShedulerGroupJson = queueSchedulerJson[c_config_key_cores_engine_clusters_queues_scheduler_group];
        if (clustersQueuesShedulerGroupJson.is_number_unsigned())
        {
            clustersQueuesShedulerGroupJson.get_to(queue.group_index);
        }
        else
        {
            if (clustersQueuesShedulerGroupJson.is_string())
            {
                std::string groupName = clustersQueuesShedulerGroupJson.get<std::string>();
                if (!groupName2GroupIndex(groupName,
                                          queue.group_index))
                {
                    THROW_INVALID_CONFIG(clustersQueuesShedulerGroupJson, "error group name {} is invalid", groupName);
                }
            }
            else
            {
                THROW_INVALID_CONFIG(clustersQueuesShedulerGroupJson, "type mismatch. expected string or unsigned int");
            }
        }
        queue.bit_mask_offset = 0;
        if (queueSchedulerJson.find(c_config_key_cores_engine_clusters_queues_scheduler_bit_mask_offset) != queueSchedulerJson.end())
        {
            queueSchedulerJson.at(c_config_key_cores_engine_clusters_queues_scheduler_bit_mask_offset).get_to(queue.bit_mask_offset);
        }

        cluster->queues[queue.index]             = std::move(queue);
        queue.scheduler->clusters[cluster->name] = cluster;
    }

    if (cluster->queues.empty())
    {
        THROW_INVALID_CONFIG(queuesJson, "error cluster {} has no queues", cluster->name);
    }
}
CATCH_JSON()

void Scal_Gaudi2::parseEngineCores(const scaljson::json &enginesJson, const ConfigVersion &version, const MemoryGroups &groups)
try
{
    (void) version; // for future use
    struct hlthunk_nic_get_ports_masks_out portsMask;
    int ret = hlthunk_nic_get_ports_masks(m_fd, &portsMask);
    if (ret) // if hlthunk_nic_get_ports_masks fail
    {
        THROW_INVALID_CONFIG(enginesJson, "hlthunk_nic_get_ports_masks fail, return:{:#x}, errno:{} - {}", ret, errno, std::strerror(errno));
    }

    std::map<std::string, std::vector<std::string>> nicPortsMap;
    std::bitset<c_ports_count> nicPortsMask(portsMask.ports_mask);
    bool usePortMaskFromDriver = !isSimFD(m_fd) && m_isInternalJson;
    if (usePortMaskFromDriver)
    {
        // When running on ASIC - take scale-up and scale-out clusters from LKD
        if (nicPortsMask.to_ullong() != portsMask.ports_mask)
        {
            THROW_INVALID_CONFIG(enginesJson, "error nic_ports_mask {:#x} has more ports than is supported {}", portsMask.ports_mask, c_ports_count);
        }
        if ((portsMask.ports_mask | portsMask.ext_ports_mask) != portsMask.ports_mask)
        {
            THROW_INVALID_CONFIG(enginesJson, "error nic_ports_external_mask {:#x} has more ports than in nic_ports_mask {:#x}", portsMask.ext_ports_mask, portsMask.ports_mask);
        }

        std::bitset<c_ports_count> nicPortsScaleUpMask(portsMask.ports_mask & (portsMask.ports_mask ^ portsMask.ext_ports_mask));
        std::bitset<c_ports_count> nicPortsScaleOutMask(portsMask.ext_ports_mask);
        LOG_INFO(SCAL, "NIC Ports ScaleUp Mask {}", nicPortsScaleUpMask.to_string());
        LOG_INFO(SCAL, "NIC Ports ScaleOut Mask {}", nicPortsScaleOutMask.to_string());

        for (unsigned nicIdx = 0; nicIdx < c_nics_count; nicIdx++)
        {
            std::string engineName("NIC_" + std::to_string(nicIdx));
            if (nicPortsScaleUpMask.test(nicIdx) && nicPortsScaleOutMask.test(nicIdx))
            {
                // can't logically happen - but lets verify
                THROW_INVALID_CONFIG(enginesJson, "error port {} used for both scaleup and scaleout", nicIdx);
            }
            if (nicPortsScaleUpMask.test(nicIdx))
            {
                nicPortsMap["nic_scaleup"].push_back(engineName);
            }
            if (nicPortsScaleOutMask.test(nicIdx))
            {
                nicPortsMap["nic_scaleout"].push_back(engineName);
            }
        }
        for (auto& [clusterName, nicPorts] : nicPortsMap)
        {
            std::stringstream clusterStringStream;
            for (auto& portName : nicPorts)
            {
                clusterStringStream << portName << " ";
            }
            LOG_INFO(SCAL, "cluster {} overriden by LKD mask: {}", clusterName, clusterStringStream.str());
        }
    }

    for (const auto & clusterObject : enginesJson)
    {
        std::string clusterName;
        clusterObject.at(c_config_key_cores_engine_clusters_name).get_to(clusterName);

        std::string imageName;
        clusterObject.at(c_config_key_cores_engine_clusters_binary_name).get_to(imageName);

        std::string memGroup = clusterObject.at(c_config_key_cores_engine_clusters_memory_group).get<std::string>();
        if (groups.find(memGroup) == groups.end())
        {
            THROW_INVALID_CONFIG(clusterObject,"error memGroup {} for cluster {}", memGroup, clusterName);
        }

        std::string qmanName;
        if (clusterObject.find(c_config_key_cores_engine_clusters_qman) != clusterObject.end())
        {
            clusterObject.at(c_config_key_cores_engine_clusters_qman).get_to(qmanName);
        }

        std::vector<Pool*> pools = groups.at(memGroup).pools;
        std::sort(pools.begin(), pools.end()); // sort for easier validation

        Pool* configPool = groups.at(memGroup).configPool;

        std::vector<EngineWithImage> engines;
        if (clusterName.find("nic") == 0)
        {
            if (usePortMaskFromDriver)
            {
                if (nicPortsMap.find(clusterName) != nicPortsMap.end())
                {
                    for (auto const & engine : nicPortsMap[clusterName])
                    {
                        engines.push_back(EngineWithImage{engine});
                    }
                }
            }
            else
            {
                // If it is a simulator device and the cluster handle nic engines, check the NICs mask and filter out disabled engines
                clusterObject.at(c_config_key_cores_engine_clusters_engines).get_to(engines);
                unsigned Idx = 0;
                while (engines.size() != 0 && Idx < engines.size())
                {
                    // NIC engine decalred as 'NIC_#'
                    if (engines[Idx].engine.length() < 5)
                    {
                        THROW_INVALID_CONFIG(clusterObject, "error parsing NIC engine index {}", engines[Idx].engine);
                    }
                    unsigned nicIdx = stoi(engines[Idx].engine.substr(4));
                    if (!nicPortsMask.test(nicIdx))
                    {
                        engines.erase(engines.begin() + Idx);
                        continue;
                    }
                    Idx++;
                }
            }

            if (engines.empty())
            {
                LOG_INFO(SCAL, "cluster {} has been removed as all engines in the cluster are disabled", clusterName);
                continue;
            }
        }
        else
        {
            clusterObject.at(c_config_key_cores_engine_clusters_engines).get_to(engines);
        }

        if (engines.empty())
        {
            THROW_INVALID_CONFIG(clusterObject.at(c_config_key_cores_engine_clusters_engines),"engines vector is empty for cluster name {}", clusterName);
        }

        if (m_clusters.find(clusterName) != m_clusters.end())
        {
            THROW_INVALID_CONFIG(clusterObject, "error finding {} in m_clusters", clusterName);
        }

        m_clusters[clusterName] = {};
        Cluster* cluster = &m_clusters[clusterName];
        cluster->name = clusterName;

        VALIDATE_JSON_NODE_IS_ARRAY(clusterObject, c_config_key_cores_engine_clusters_queues)
        parseClusterQueues(clusterObject.at(c_config_key_cores_engine_clusters_queues), version, cluster);

        for (unsigned offset = 0; offset < engines.size(); offset++)
        {
            unsigned coreCpuId;
            if (!arcName2CpuId(engines[offset].engine, coreCpuId))
            {
                THROW_INVALID_CONFIG(clusterObject, "error finding CPU ID id of {}", engines[offset].engine);
            }
            if ((coreCpuId < c_scheduler_nr) ||
                (coreCpuId >= c_cores_nr))
            {
                THROW_INVALID_CONFIG(clusterObject, "coreCpuId {} out of range [{}..{}) core.name={}", coreCpuId, c_scheduler_nr, c_cores_nr, engines[offset].engine);
            }
            ArcCore* core = nullptr;
            if (m_cores[coreCpuId] == nullptr)
            {
                // new core
                core = new ArcCore();
                m_cores[coreCpuId] = core;
                core->cpuId = coreCpuId;
                core->arcName = engines[offset].engine;
                core->qman = qmanName.empty() ? core->arcName : qmanName;
                if (!arcName2DccmAddr(core->arcName, core->dccmDevAddress) ||
                    !arcName2QueueId(core->qman,     core->qmanID))
                {
                    THROW_INVALID_CONFIG(clusterObject, "error finding Qman id or type. core->qman={} core->name={}", core->qman, core->name);
                }
                core->indexInGroup = offset;
                core->numEnginesInGroup = engines.size();
                core->name = core->arcName;
                core->pools = pools;
                core->configPool = configPool;
                core->imageName = engines[offset].image.empty() ? imageName : engines[offset].image;
                core->scal = this;
            }
            else
            {
                core = getCore<ArcCore>(coreCpuId);
                // make sure that the queue index is not already in use
                for (const auto& queue : cluster->queues)
                {
                    for (const auto& otherCluster : core->clusters)
                    {
                        if (otherCluster.second->name == cluster->name) continue;
                        for (const auto& otherQueue : otherCluster.second->queues)
                        {
                            if (queue.second.index == otherQueue.second.index)
                            {
                                THROW_INVALID_CONFIG(clusterObject, "error cluster {} and cluster {} are using qman {} with the same index {}", cluster->name, otherCluster.second->name, core->qman, queue.second.index);
                            }
                        }
                    }
                }
                if (configPool != core->configPool)
                {
                    THROW_INVALID_CONFIG(clusterObject, "error core {} in cluster {} config mismatch of config pool", core->arcName, cluster->name);
                }
                if(!std::is_permutation(pools.begin(), pools.end(), core->pools.begin()))
                {
                    THROW_INVALID_CONFIG(clusterObject, "error core {} in cluster {} config mismatch of pools", core->arcName, cluster->name);
                }
                // we use work distribution for network_edma_0/1 - use indexInGroup and groupSize according to this cluster
                if (clusterName == "network_edma_0" || clusterName == "network_edma_1")
                {
                    core->indexInGroup = offset;
                    core->numEnginesInGroup = engines.size();
                }
            }
            if (clusterObject.find(c_config_key_cores_engine_clusters_is_compute) != clusterObject.end())
            {
                unsigned isCompute;
                clusterObject.at(c_config_key_cores_engine_clusters_is_compute).get_to(isCompute);
                if (isCompute)
                {
                    cluster->isCompute = true;
                    m_computeClusters.push_back(cluster);
                }
            }

            CoreType coreType;
            scal_assert(arcName2CoreType(core->arcName, coreType), "arcName2CoreType failed");
            scal_assert((cluster->type == NUM_OF_CORE_TYPES || cluster->type == coreType), "error cluster {} has several core types ({}, {})", clusterName, core->arcName, cluster->engines[0]->name);
            cluster->type = coreType;
            m_clusters[clusterName].engines.emplace_back(core);
            core->clusters[clusterName] = cluster;
        }
    }
}
CATCH_JSON()


void Scal_Gaudi2::parseSyncManagers(const scaljson::json &syncManagersJson, const ConfigVersion &version)
try
{
    // parse "sync_managers" array
    for (const scaljson::json & syncManagerJson : syncManagersJson)
    {
        unsigned smID = syncManagerJson.at(c_config_key_sync_managers_dcore).get<unsigned>();
        if (smID >= c_sync_managers_nr)
        {
            THROW_INVALID_CONFIG(syncManagerJson, "illegal sync manager ID {}", smID);
        }
        // TBD in Gaudi2 - check that each dcore is configured (at most) just one time.
        // "qman": "DCORE0_TPC_0",

        //"sos_pools": []
        VALIDATE_JSON_NODE_IS_ARRAY(syncManagerJson, c_config_key_sync_managers_sos_pools);

        parseSosPools(syncManagerJson[c_config_key_sync_managers_sos_pools], version, smID);
        // "monitors_pools": []
        VALIDATE_JSON_NODE_IS_ARRAY(syncManagerJson, c_config_key_sync_managers_monitors_pools);

        parseMonitorPools(syncManagerJson[c_config_key_sync_managers_monitors_pools], version, smID);

        // map sm to userspace memory
        SyncManager & syncManager = m_syncManagers[smID];
        syncManager.map2userSpace =
            (syncManagerJson.find(c_config_key_sync_managers_map_to_userspace) != syncManagerJson.end()) &&
             syncManagerJson.at(c_config_key_sync_managers_map_to_userspace).get<bool>();
        if (syncManagerJson.find(c_config_key_sync_managers_qman) != syncManagerJson.end())
        {
            syncManagerJson.at(c_config_key_sync_managers_qman).get_to(syncManager.qman);

            if (!arcName2QueueId(syncManager.qman, syncManager.qmanID))
            {
                THROW_INVALID_CONFIG(syncManagerJson, "error in arcName2QueueId. qman \"{}\" not found", syncManager.qman);
            }
        }
    }
}
CATCH_JSON()


void Scal_Gaudi2::parseSosPools(const scaljson::json &sosPoolsJson, const ConfigVersion &version, const unsigned smID)
try
{
    // checks to be done:
    // no overlap between Sos on pools
    // total number of Sos in SM < 8K
    unsigned totalNumOfSos = 0;
    for (const scaljson::json & soJson : sosPoolsJson)
    {
        SyncObjectsPool so;

        soJson.at(c_config_key_sync_managers_sos_pools_name).get_to(so.name);
        if (m_soPools.find(so.name) != m_soPools.end())
        {
            THROW_INVALID_CONFIG(soJson, "duplicate so name {} in smID {}", so.name, smID);
        }

        soJson.at(c_config_key_sync_managers_sos_pools_base_index).get_to(so.baseIdx);
        so.nextAvailableIdx = so.baseIdx;
        soJson.at(c_config_key_sync_managers_sos_pools_size).get_to(so.size);
        totalNumOfSos += so.size;
        if (soJson.find(c_config_key_sync_managers_sos_pools_align) != soJson.end() )
        {
            unsigned align;
            soJson.at(c_config_key_sync_managers_sos_pools_align).get_to(align);
            if (so.baseIdx % align != 0)
            {
                THROW_INVALID_CONFIG(soJson, "base_index {} of {} in smID {} should be align to {}",
                                     so.baseIdx, so.name, smID, align);
            }
        }
        // check for overlap
        for (auto prevSo : m_syncManagers[smID].soPools)
        {
            unsigned int start = so.baseIdx;
            unsigned int end = so.baseIdx + so.size;
            unsigned int otherStart = prevSo->baseIdx;
            unsigned int otherEnd = prevSo->baseIdx + prevSo->size;
            if (isOverlap(start, end, otherStart, otherEnd))
            {
                THROW_INVALID_CONFIG(soJson, "overlap in so indices in smID {} (baseIdx={}) between {} and {}", smID, so.baseIdx, prevSo->name, so.name);
            }
        }
        so.dcoreIndex = smID;
        so.smIndex = smID;
        so.scal = this;
        so.smBaseAddr = SyncMgrG2::getSmBase(smID);
        m_soPools[so.name] = so;
        m_syncManagers[smID].soPools.push_back(&m_soPools[so.name]);
        m_syncManagers[smID].dcoreIndex = smID;
        m_syncManagers[smID].smIndex = smID;
        m_syncManagers[smID].baseAddr = so.smBaseAddr;
    }
    if (totalNumOfSos > c_max_sos_per_sync_manager)
    {
        THROW_INVALID_CONFIG(sosPoolsJson, "total number of so in smID {} > {}  (totalNumOfSos={})", smID, (unsigned)c_max_sos_per_sync_manager, totalNumOfSos);
    }
}
CATCH_JSON()

void Scal_Gaudi2::parseMonitorPools(const scaljson::json &monitorPoolJson, const ConfigVersion &version, const unsigned smID)
try
{
    // checks:
    // no overlap between monitors on pools
    // total number of monitors in SM < 2k

    unsigned totalNumOfMonitors = 0;
    for (const scaljson::json & monitor : monitorPoolJson)
    {
        MonitorsPool mp;
        monitor[c_config_key_sync_managers_monitors_pools_name].get_to(mp.name);

        if (m_monitorPools.find(mp.name) != m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(monitor, "duplicate monitor name {} in smID {}", mp.name, smID);
        }

        monitor.at(c_config_key_sync_managers_monitors_pools_base_index).get_to(mp.baseIdx);
        mp.nextAvailableIdx = mp.baseIdx;
        monitor.at(c_config_key_sync_managers_monitors_pools_size).get_to(mp.size);
        totalNumOfMonitors += mp.size;
        if (monitor.find(c_config_key_sync_managers_monitors_pools_align) != monitor.end() )
        {
            unsigned align;
            monitor.at(c_config_key_sync_managers_monitors_pools_align).get_to(align);
            if (mp.baseIdx % align != 0)
            {
                THROW_INVALID_CONFIG(monitor, "base_index {} of {} smID {} should be align to {}",
                                     mp.baseIdx, mp.name, smID, align);
            }
        }

        // check for overlap
        for (auto prevMp : m_syncManagers[smID].monitorPools)
        {
            unsigned int start = mp.baseIdx;
            unsigned int end = mp.baseIdx + mp.size;
            unsigned int otherStart = prevMp->baseIdx;
            unsigned int otherEnd = prevMp->baseIdx + prevMp->size;
            if (isOverlap(start, end, otherStart, otherEnd))
            {
                THROW_INVALID_CONFIG(monitor, "overlap in monitors indices in smID {} (baseIdx={})", smID, mp.baseIdx);
            }
        }
        mp.dcoreIndex = smID;
        mp.smIndex = smID;
        mp.scal = this;
        mp.smBaseAddr = SyncMgrG2::getSmBase(smID);
        m_monitorPools[mp.name] = mp;
        m_syncManagers[smID].monitorPools.push_back(&m_monitorPools[mp.name]);
    }
    if (totalNumOfMonitors > c_max_monitors_per_sync_manager)
    {
        THROW_INVALID_CONFIG(monitorPoolJson, "total number of monitors in smID {} > {}  (totalNumOfMonitors={})", smID, (unsigned)c_max_monitors_per_sync_manager, totalNumOfMonitors);
    }
}
CATCH_JSON()


void Scal_Gaudi2::parseSyncManagersCompletionQueues(const scaljson::json &syncObjectJson, const ConfigVersion &version)
try
{
    // parse completion queues long so pool
    SyncObjectsPool* longSoPool      = &m_soPools[syncObjectJson.at(c_config_key_completion_queues_long_so_pool).get<std::string>()];

    SyncObjectsPool* sfgSosPool = nullptr;
    MonitorsPool* sfgMonitorsPool = nullptr;
    MonitorsPool* sfgCqMonitorsPool = nullptr;
    parseSfgPool(syncObjectJson, &sfgSosPool, &sfgMonitorsPool, &sfgCqMonitorsPool);

    const scaljson::json &syncManagerJson = syncObjectJson.at(c_config_key_sync_managers);
    for (const scaljson::json & dcoreJson : syncManagerJson)
    {
        unsigned dcoreID = dcoreJson.at(c_config_key_sync_managers_dcore).get<unsigned>();

        // run in a different loop because parseCompletionQueues needs the rest of the sync manager members to be initialized
        VALIDATE_JSON_NODE_IS_ARRAY(dcoreJson, c_config_key_sync_managers_completion_queues);

        parseCompletionQueues(dcoreJson[c_config_key_sync_managers_completion_queues], version, dcoreID,
                                         longSoPool, sfgSosPool, sfgMonitorsPool, sfgCqMonitorsPool);
    }
}
CATCH_JSON()

void Scal_Gaudi2::parseTdrCompletionQueues(const scaljson::json & completionQueueJsonItem, unsigned numberOfInstances, CompletionGroupInterface& cq)
{
    CompQTdr& compQTdr = cq.compQTdr;

    {
        std::string name = cq.name + "_tdr_sos";

        auto it = m_soPools.find(name);
        if (it == m_soPools.end())
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "so pool {} of completion group name {} was not found", name, cq.name);
        }

        if (it->second.size != numberOfInstances)
        {
            THROW_CONFIG_ERROR(SCAL_INVALID_CONFIG, completionQueueJsonItem, "number of tdr sos don't match the cq number {} != {}", it->second.size, numberOfInstances);
        }

        compQTdr.sos = it->second.baseIdx;
        compQTdr.sosPool = &it->second;
        LOG_INFO_F(SCAL, "tdr {} sosSmIdx {} so base is {}", name, compQTdr.sosPool->smIndex, compQTdr.sos);
    }

    {
        std::string name = cq.name + "_tdr_monitors";

        auto it = m_monitorPools.find(name);
        if (it == m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "monitor pool {} of completion group name {} was not found", name, cq.name);
        }

        if (it->second.size != numberOfInstances * CompQTdr::NUM_MON)
        {
            THROW_CONFIG_ERROR(SCAL_INVALID_CONFIG, completionQueueJsonItem, "number of tdr monitors don't match the {} * cq-number {} != {}",
                               CompQTdr::NUM_MON, it->second.size, numberOfInstances);
        }
        compQTdr.monitor = it->second.baseIdx;
        compQTdr.monSmIdx = it->second.smIndex;
        compQTdr.monPool = &it->second;
        LOG_INFO_F(SCAL, "tdr {} monSmIdx {} monitor base is {}", name, compQTdr.monSmIdx, compQTdr.monitor);
    }

    if (compQTdr.monPool->smIndex != compQTdr.sosPool->smIndex)
    {
        THROW_CONFIG_ERROR(SCAL_INVALID_CONFIG, completionQueueJsonItem, "SM-Index mismatch ({}): between Monitor-pool {} and SO-pool {}",
                           cq.name, compQTdr.monPool->smIndex, compQTdr.sosPool->smIndex);
    }
}

void Scal_Gaudi2::parseCompletionQueues(const scaljson::json& completionQueueJson, const ConfigVersion& version,
                                        const unsigned smID, SyncObjectsPool* longSoPool,
                                        SyncObjectsPool* sfgSosPool, MonitorsPool* sfgMonitorsPool,
                                        MonitorsPool* sfgCqMonitorsPool)
try
{
    struct hlthunk_sync_manager_info                syncManagerInfo;
    std::map<std::string, std::vector<std::string>> schedulersCqGroupsMap; // map of: scheduler name -> CQ names

    if (hlthunk_get_sync_manager_info(m_fd, smID, &syncManagerInfo))
    {
        THROW_INVALID_CONFIG(completionQueueJson, "Failed to call hlthunk_get_sync_manager_info() smID={}. errno = {} {}", smID, errno, std::strerror(errno));
    }

    // completion_queues CAN be 0 size
    for (const scaljson::json & completionQueueJsonItem : completionQueueJson)
    {
        CompletionGroup cq(this);

        // name_prefix
        completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_name_prefix).get_to(cq.name);
        if (m_completionGroups.find(cq.name) != m_completionGroups.end())
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "duplicate completion group name {} in smID {}", cq.name, smID);
        }
        // number_of_instances
        unsigned numberOfInstances = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_number_of_instances).get<unsigned>();
        // scheduler(s)
        cq.scheduler = nullptr;
        VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(completionQueueJsonItem, c_config_key_sync_managers_completion_queues_schedulers);

        std::vector<std::string> slaveSchedulersNames;
        std::string masterSchedulerName;
        for (scaljson::json schedJson : completionQueueJsonItem[c_config_key_sync_managers_completion_queues_schedulers])
        {
            std::string schedulerName = schedJson.get<std::string>();
            const Scheduler * scheduler = getCoreByName<Scheduler>(schedulerName);
            if (!scheduler)
            {
                THROW_INVALID_CONFIG(schedJson, "scheduler {} of completion group name {} in smID {} was not found", schedulerName, cq.name, smID);
            }
            auto & schedulerCqGroups = schedulersCqGroupsMap[schedulerName];
            if (std::find(schedulerCqGroups.begin(),schedulerCqGroups.end(),cq.name) != schedulerCqGroups.end())
            {
                THROW_INVALID_CONFIG(schedJson, "duplicate scheduler {} in completion group name {} in smID {}", schedulerName, cq.name, smID);
            }
            schedulerCqGroups.push_back(cq.name);
            if (schedulerCqGroups.size() > COMP_SYNC_GROUP_COUNT)
            {
                THROW_INVALID_CONFIG(schedJson, "scheduler {} of completion group name {} in smID {} have too many CQ Groups ({})", schedulerName, cq.name,
                                     smID, schedulerCqGroups.size() * numberOfInstances);
            }
            if (cq.scheduler == nullptr)
            {
                // 1st scheduler is the master, rest are slaves
                cq.scheduler = scheduler;
                masterSchedulerName = schedulerName;
            }
            else
            {
                SlaveSchedulerInCQ cqSched;
                cqSched.scheduler = scheduler;
                cqSched.idxInScheduler = (unsigned)-1; // for each instance it will be different
                cq.slaveSchedulers.push_back(cqSched);
            }
            cq.qmanID = cq.scheduler->qmanID;
            slaveSchedulersNames.push_back(schedulerName);
        }
        unsigned totalNumSchedulers = slaveSchedulersNames.size();

        // sos_pool
        cq.sosPool = nullptr;
        const std::string sosPoolName = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_sos_pool).get<std::string>();
        if (m_soPools.find(sosPoolName) == m_soPools.end())
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "so pool {} of completion group name {} in smID {} was not found", sosPoolName, cq.name, smID);
        }
        cq.sosPool = &m_soPools[sosPoolName];

        // sos_depth
        completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_sos_depth).get_to(cq.sosNum);
        cq.sosBase = cq.sosPool->nextAvailableIdx;
        unsigned totalSos = cq.sosNum * numberOfInstances;
        if (totalSos + cq.sosPool->nextAvailableIdx - cq.sosPool->baseIdx > cq.sosPool->size)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "too many sos used {} in {} for {} instances x {} schedulers in smID {}",
                                 totalSos, sosPoolName, numberOfInstances, totalNumSchedulers, smID);
        }
        cq.sosPool->nextAvailableIdx += totalSos;

        if (cq.sosPool->smIndex != smID)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "SOs pool {} of SM {} is used by completion group {} in SM {}",
                                 cq.sosPool->name, cq.sosPool->smIndex, cq.name, smID);
        }

        // monitors_pool
        cq.monitorsPool = nullptr;

        const std::string monitorsPoolName = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_monitors_pool).get<std::string>();
        if (m_monitorPools.find(monitorsPoolName) == m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "monitors pool {} of completion group name {} in smID {} was not found", monitorsPoolName, cq.name, smID);
        }

        cq.monitorsPool = &m_monitorPools[monitorsPoolName];
        if (cq.monitorsPool->baseIdx < syncManagerInfo.first_available_monitor)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "illegal monitor index {} used in monitor pool {} in smID {}",
                                 cq.monitorsPool->baseIdx, monitorsPoolName, smID);
        }

        if (cq.monitorsPool->smIndex != smID)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "Monitor pool {} of SM {} is used by completion group {} in SM {}",
                                 cq.monitorsPool->name, cq.monitorsPool->smIndex, cq.name, smID);
        }

        // force order
        bool isForceOrder = true;
        if (completionQueueJsonItem.find(c_config_key_sync_managers_completion_queues_force_order) != completionQueueJsonItem.end())
        {
            isForceOrder = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_force_order).get<bool>();
        }
        cq.force_order = isForceOrder;

        unsigned numberOfUserMonitors = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_monitors_depth).get<unsigned>();
        // monitors_depth
        if (numberOfUserMonitors > COMP_SYNC_GROUP_MAX_MON_GROUP_COUNT)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "error parsing completion group element {} in smID {}",
                                 c_config_key_sync_managers_completion_queues_monitors_depth, smID);
        }

        // credit management
        if (cq.slaveSchedulers.size() != 0)
        {
            int status = getCreditManagmentBaseIndices(cq.creditManagementSobIndex, cq.creditManagementMonIndex, false);

            if (status != SCAL_SUCCESS)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "Invalid Credit-Management configuration");
            }
	    }

        // SFG configuration
        parseSfgCompletionQueueConfig(completionQueueJsonItem, cq, sfgSosPool, sfgMonitorsPool, sfgCqMonitorsPool);

        if (numberOfUserMonitors == 1)
            cq.force_order = false; // if monitor depth is 1, ignore force_order (as there's no need for order between the monitors...)
        /*
            actualNumberOfMonitors scal must config per cq is 3 + (force_order?1:0) + #of slave monitors
            but since the maximum # of messages per monitor is 4, we use a chain reaction
            where we use some monitors to trig other master monitors
            so if #>4 we need extra 1 trigger, if #>8 we need extra 2 trigger monitors
            so maximum is 11  (3+1+5+9/4)  (setSize + forceOrder+NumSlaves+NumTrigerMonitors)
        */
        uint32_t num_extra_monitors =  (unsigned)cq.force_order + totalNumSchedulers - 1; // force_order + # slaves
        cq.actualNumberOfMonitors = c_completion_queue_monitors_set_size + num_extra_monitors;
        uint32_t num_trigger_monitors = (num_extra_monitors+1) / 3; //0  if slaves+force < 2, 1 if 2<= slaves+force < 5,  2  if slaves+force >= 5 - see table just above AddIncNextSyncObjectMonitor in scal_init_###.cpp
        cq.actualNumberOfMonitors += num_trigger_monitors;
        //
        cq.monNum = numberOfUserMonitors * cq.actualNumberOfMonitors;
        cq.monBase = cq.monitorsPool->nextAvailableIdx;
        unsigned totalMon = cq.monNum * numberOfInstances;
        if (totalMon + cq.monitorsPool->nextAvailableIdx - cq.monitorsPool->baseIdx > cq.monitorsPool->size)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "too many monitors used {} in {} in dcore {}",
                    cq.monBase + totalMon, std::string(monitorsPoolName), smID);
        }
        cq.monitorsPool->nextAvailableIdx += totalMon;

        // cq_base_index
        if ((completionQueueJsonItem.find(c_config_key_sync_managers_completion_queues_is_compute_completion_queue) != completionQueueJsonItem.end()))
        {
            bool isCompletionQueue = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_is_compute_completion_queue).get<bool>();
            if (isCompletionQueue)
            {
                if (m_computeCompletionQueuesSos != nullptr) // already set? This shouldn't happen
                {
                    THROW_INVALID_CONFIG(completionQueueJsonItem, "m_computeCompletionQueuesSos already set with {}/{}",
                            m_computeCompletionQueuesSos->baseIdx, m_computeCompletionQueuesSos->size);
                }
                m_computeCompletionQueuesSos = cq.sosPool;
            }
        }

        //TDR
        if (completionQueueJsonItem.find(c_config_key_sync_managers_completion_queues_is_tdr) != completionQueueJsonItem.end())
        {
            bool isTdr = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_is_tdr).get<bool>();
            cq.compQTdr.enabled = isTdr;
            if (isTdr)
            {
                parseTdrCompletionQueues(completionQueueJsonItem, numberOfInstances, cq);
            }
        }

        cq.cqIdx = 0; // initialized per instance
        cq.isrIdx = 0; // initialized per instance
        // disable_isr  SW-82256
        bool enable_isr = true;
        if (completionQueueJsonItem.find(c_config_key_sync_managers_completion_queues_enable_isr) != completionQueueJsonItem.end())
        {
            enable_isr = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_enable_isr).get<bool>();
        }

        if (completionQueueJsonItem.find(c_config_key_sync_managers_completion_queues_is_stub) != completionQueueJsonItem.end())
        {
            cq.isCgStub = completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_is_stub).get<bool>();
        }

        cq.pCounter = 0;    // initialized after cq counters are allocated (in configureCQs)

        cq.longSosPool = longSoPool;
        if (completionQueueJsonItem.find(c_config_key_sync_managers_completion_queues_long_sos) != completionQueueJsonItem.end())
        {
            std::string completionQueueLongSosPoolName;
            completionQueueJsonItem.at(c_config_key_sync_managers_completion_queues_long_sos).get_to(completionQueueLongSosPoolName);
            cq.longSosPool = &m_soPools[completionQueueLongSosPoolName];
        }

        cq.syncManager = &m_syncManagers[smID];

        cq.scal = this;

        // create numberOfInstances new cqs instances according to the json config
        // each has its own name, monitor range, so range and longSO
        for (unsigned instance = 0; instance < numberOfInstances; instance++)
        {
            CompletionGroup cqInstance(cq);
            cqInstance.monBase = cq.monBase + (cq.monNum * instance);
            cqInstance.sosBase = cq.sosBase + (cq.sosNum * instance);
            if (cqInstance.sosBase + cq.sosNum > cq.sosPool->baseIdx + cq.sosPool->size)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "so out of pool range index {} depth {} > so pool base {} size {} at cq {}",
                        cq.sosBase, cq.sosNum, cq.sosPool->baseIdx, cq.sosPool->size, cq.name);
            }

            cqInstance.name     = cq.name + std::to_string(instance);
            cqInstance.compQTdr = cq.compQTdr;

            allocateCqIndex(cqInstance.cqIdx,
                            cqInstance.globalCqIndex,
                            completionQueueJsonItem,
                            smID,
                            m_syncManagers[smID].dcoreIndex,
                            syncManagerInfo.first_available_cq);

            if (cq.compQTdr.enabled)
            {
                unsigned tdrSmID = cqInstance.compQTdr.monSmIdx;

                cqInstance.compQTdr.sos     = cq.compQTdr.sos + instance;
                cqInstance.compQTdr.monitor = cq.compQTdr.monitor + CompQTdr::NUM_MON * instance;

                allocateCqIndex(cqInstance.compQTdr.cqIdx,
                                cqInstance.compQTdr.globalCqIndex,
                                completionQueueJsonItem,
                                tdrSmID,
                                m_syncManagers[tdrSmID].dcoreIndex,
                                syncManagerInfo.first_available_cq);
            }

            // interrupt index
            if (enable_isr)
            {
                cqInstance.isrIdx = m_nextIsr + m_hw_ip.first_available_interrupt_id;
                if (m_nextIsr > m_hw_ip.number_of_user_interrupts)
                {
                    THROW_INVALID_CONFIG(completionQueueJsonItem, "invalid interrupt index {}, max allowed is {}",
                            cqInstance.isrIdx, m_hw_ip.first_available_interrupt_id + m_hw_ip.number_of_user_interrupts - 1);
                }
                m_nextIsr++;
            }
            else
            {
                cqInstance.isrIdx = scal_illegal_index;
            }
            cqInstance.longSoIndex   = cq.longSosPool->nextAvailableIdx;
            cqInstance.longSoSmIndex = cq.longSosPool->smIndex;
            // each long sob (60 bit) consists of 4 regular sob (15bit)
            if (cq.longSosPool->nextAvailableIdx - cq.longSosPool->baseIdx + c_so_group_size > cq.longSosPool->size)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "too many long sos used {} in smID {}", cq.longSosPool->nextAvailableIdx, smID);
            }
            cq.longSosPool->nextAvailableIdx += c_so_group_size;
            cqInstance.idxInScheduler = m_schedulersCqsMap[masterSchedulerName].size();

            if (cqInstance.idxInScheduler >= COMP_SYNC_GROUP_COUNT)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem,"too many CQs in {} scheduler ", masterSchedulerName);
            }
            LOG_DEBUG(SCAL,"cq name {} Idx {} schedulerName {} monBase {} monNum {} sosBase {} sosNum {} longSoIndex {}  longSoSmIndex {} numberOfUserMonitors {} idxInScheduler {} isr {}"
                           " tdr {} sos {}",
                cqInstance.name, cqInstance.cqIdx, masterSchedulerName,
                cqInstance.monBase, cq.monNum , cqInstance.sosBase, cq.sosNum,
                cqInstance.longSoIndex, cqInstance.longSoSmIndex, numberOfUserMonitors, cqInstance.idxInScheduler,
                cqInstance.isrIdx, cqInstance.compQTdr.enabled, cqInstance.compQTdr.sos);

            m_completionGroups.insert({cqInstance.name, cqInstance});
            CompletionGroup* pCQ = &(m_completionGroups.at(cqInstance.name));
            m_cgs.push_back(pCQ);
            m_syncManagers[smID].completionGroups.push_back(pCQ);

            m_schedulersCqsMap[masterSchedulerName].push_back(pCQ);
            // for each slave scheduler, add a reference to this cq instance to its list
            handleSlaveCqs(pCQ, masterSchedulerName);
        }
    }
}
CATCH_JSON()

void Scal_Gaudi2::parseHostFenceCounters(const scaljson::json &syncObjectJson, const ConfigVersion &version)
try
{
    const scaljson::json &syncManagerJson = syncObjectJson.at(c_config_key_sync_managers);
    for (const scaljson::json & smJson : syncManagerJson)
    {
        unsigned smID = smJson.at(c_config_key_sync_managers_dcore).get<unsigned>();
        struct hlthunk_sync_manager_info                syncManagerInfo;

        if (hlthunk_get_sync_manager_info(m_fd, smID, &syncManagerInfo))
        {
            THROW_INVALID_CONFIG(syncManagerJson, "Failed to call hlthunk_get_sync_manager_info() dcore={}. errno = {} {}", smID, errno, std::strerror(errno));
        }

        if (smJson.find(c_config_key_sync_managers_host_fence_counters) != smJson.end())
        {
            VALIDATE_JSON_NODE_IS_ARRAY(smJson, c_config_key_sync_managers_host_fence_counters);
            for (const scaljson::json &fenceCouterJson : smJson.at(c_config_key_sync_managers_host_fence_counters))
            {
                HostFenceCounter ctr;
                ctr.scal = this;
                ctr.syncManager = &m_syncManagers[smID];

                if (ctr.syncManager->map2userSpace == false)
                {
                     THROW_INVALID_CONFIG(fenceCouterJson, "host fence counters must be in sync manager that is mapped to the user space."
                                                           " sync managers of dcore {} are not mapped to the user space", smID);
                }

                // name prefix
                fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_name_prefix).get_to(ctr.name);

                // number_of_instances
                unsigned numberOfInstances = fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_number_of_instances).get<unsigned>();

                // sos_pool
                ctr.sosPool = nullptr;
                const std::string sosPoolName = fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_sos_pool).get<std::string>();
                if (m_soPools.find(sosPoolName) == m_soPools.end())
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "so pool {} of host fence counter {} in dcore {} was not found", sosPoolName, ctr.name, smID);
                }
                ctr.sosPool = &m_soPools[sosPoolName];
                ctr.soIdx = ctr.sosPool->nextAvailableIdx;
                if (numberOfInstances + ctr.sosPool->nextAvailableIdx - ctr.sosPool->baseIdx > ctr.sosPool->size)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "too many sos used in {} for {} fence counter instances in dcore {}",
                                         sosPoolName, numberOfInstances, smID);
                }
                ctr.sosPool->nextAvailableIdx += numberOfInstances;

                if (ctr.sosPool->smIndex != smID)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "SOs pool {} of dcore {} is used by host fence counter {} in dcore {}",
                                         ctr.sosPool->name, ctr.sosPool->smIndex, ctr.name, smID);
                }

                // monitors_pool
                ctr.monitorsPool = nullptr;
                const std::string monitorsPoolName = fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_monitors_pool).get<std::string>();
                if (m_monitorPools.find(monitorsPoolName) == m_monitorPools.end())
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "monitors pool {} of host fence counter name {} in dcore {} was not found", monitorsPoolName, ctr.name, smID);
                }

                ctr.monitorsPool = &m_monitorPools[monitorsPoolName];
                if (ctr.monitorsPool->baseIdx < syncManagerInfo.first_available_monitor)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "illegal monitor index {} used in monitor pool {} in dcore {}",
                                         ctr.monitorsPool->baseIdx, monitorsPoolName, smID);
                }

                ctr.monBase = ctr.monitorsPool->nextAvailableIdx;
                unsigned totalMon = c_host_fence_ctr_mon_nr * numberOfInstances;
                if (totalMon + ctr.monitorsPool->nextAvailableIdx - ctr.monitorsPool->baseIdx > ctr.monitorsPool->size)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "too many monitors used {} in {} in dcore {}",
                            ctr.monBase + totalMon, monitorsPoolName, smID);
                }
                ctr.monitorsPool->nextAvailableIdx += totalMon;

                if (ctr.monitorsPool->smIndex != smID)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "monitors pool {} of dcore {} is used by host fence counter {} in dcore {}",
                                         ctr.monitorsPool->name, ctr.monitorsPool->smIndex, ctr.name, smID);
                }

                ctr.isrEnable = true;
                if (fenceCouterJson.find(c_config_key_sync_managers_host_fence_counters_enable_isr) != fenceCouterJson.end())
                {
                    fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_enable_isr).get_to(ctr.isrEnable);
                }

                if (fenceCouterJson.find(c_config_key_sync_managers_host_fence_counters_is_stub) != fenceCouterJson.end())
                {
                    fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_is_stub).get_to(ctr.isStub);
                }

                for (unsigned instance = 0; instance < numberOfInstances; instance++)
                {
                    HostFenceCounter ctrInstance = ctr;
                    ctrInstance.monBase = ctr.monBase + (c_host_fence_ctr_mon_nr * instance);
                    ctrInstance.soIdx   = ctr.soIdx + instance;
                    ctrInstance.name    = ctr.name + std::to_string(instance);

                    ctrInstance.isrIdx = m_nextIsr + m_hw_ip.first_available_interrupt_id;
                    if (m_nextIsr > m_hw_ip.number_of_user_interrupts)
                    {
                        THROW_INVALID_CONFIG(fenceCouterJson, "invalid interrupt index {}, max allowed is {}",
                                ctrInstance.isrIdx, m_hw_ip.first_available_interrupt_id + m_hw_ip.number_of_user_interrupts - 1);
                    }
                    m_nextIsr++;


                    if (m_hostFenceCounters.find(ctrInstance.name) != m_hostFenceCounters.end())
                    {
                        THROW_INVALID_CONFIG(fenceCouterJson,"duplicate fence counter name", ctrInstance.name);
                    }

                    CompletionGroup completionGroup(this);
                    completionGroup.name                   = "cg_" + ctrInstance.name;
                    completionGroup.isrIdx                 = ctrInstance.isrIdx;
                    completionGroup.syncManager            = ctrInstance.syncManager;
                    completionGroup.monitorsPool           = ctrInstance.monitorsPool;
                    completionGroup.force_order            = false;
                    completionGroup.sosPool                = ctrInstance.sosPool;
                    completionGroup.sosBase                = ctrInstance.soIdx;
                    completionGroup.sosNum                 = 1;
                    completionGroup.monNum                 = c_host_fence_ctr_mon_nr;
                    completionGroup.monBase                = ctrInstance.monBase;
                    completionGroup.qmanID                 = ctrInstance.syncManager->qmanID;
                    completionGroup.longSosPool            = ctrInstance.sosPool;
                    completionGroup.actualNumberOfMonitors = c_host_fence_ctr_mon_nr;
                    completionGroup.fenceCounterName       = ctrInstance.name;

                    allocateCqIndex(completionGroup.cqIdx,
                                    completionGroup.globalCqIndex,
                                    fenceCouterJson,
                                    smID,
                                    m_syncManagers[smID].dcoreIndex,
                                    syncManagerInfo.first_available_cq);

                    assert(completionGroup.qmanID != -1);
                    if (completionGroup.qmanID == -1)
                    {
                        THROW_INVALID_CONFIG(fenceCouterJson,"sync manager configuration must have qman. e.g. \"qman\"=\"DCORE1_TPC_0\"");
                    }

                    m_completionGroups.insert({completionGroup.name, completionGroup});
                    auto pCompGroup = &m_completionGroups.at(completionGroup.name);
                    m_cgs.push_back(pCompGroup);
                    m_syncManagers[smID].completionGroups.push_back(pCompGroup);
                    ctrInstance.completionGroup = pCompGroup;
                    m_hostFenceCounters[ctrInstance.name] = ctrInstance;

                }
            }
        }
    }
}
CATCH_JSON()

void Scal_Gaudi2::parseSoSets(const scaljson::json &soSetJson, const ConfigVersion &version)
try
{
     // parse "sos_sets" array
    for (const scaljson::json & soSet : soSetJson)
    {
        SyncObjectsSetGroup sg;
        soSet.at(c_config_key_sos_sets_name).get_to(sg.name);

        sg.scheduler = nullptr;
        const std::string schedulerName = soSet.at(c_config_key_sos_sets_scheduler).get<std::string>();
        for (unsigned i=0; i<c_scheduler_nr; i++)
        {
            if (m_cores[i] && (m_cores[i]->name == schedulerName))
            {
                sg.scheduler = m_cores[i]->getAs<Scheduler>();
                break;
            }
        }
        if (!sg.scheduler)
        {
            THROW_INVALID_CONFIG(soSet,"could not find  so set scheduler {}", schedulerName);
        }

        soSet.at(c_config_key_sos_sets_set_size).get_to(sg.setSize);
        soSet.at(c_config_key_sos_sets_num_sets).get_to(sg.numSets);

        const std::string sosPoolName = soSet.at(c_config_key_sos_sets_sos_pool).get<std::string>();
        if (m_soPools.find(sosPoolName) == m_soPools.end())
        {
            THROW_INVALID_CONFIG(soSet, "error parsing  so set element {}. could not find {} in so pools",
                c_config_key_sos_sets_sos_pool, sosPoolName);
        }
        sg.sosPool = &m_soPools[sosPoolName];
        if (sg.numSets * sg.setSize > m_soPools[sosPoolName].size)
        {
            THROW_INVALID_CONFIG(soSet, "error parsing so set {}. numSets {} x setSize {} > pool {} size {}",
                sg.name, sg.numSets, sg.setSize,sosPoolName,  m_soPools[sosPoolName].size);
        }

        // GC monitors pool

        // changed to optional TBD
        if (soSet.find(c_config_key_sos_sets_gc_monitors_pool) != soSet.end())
        {
            std::string gcMonitorPoolName = soSet[c_config_key_sos_sets_gc_monitors_pool].get<std::string>();
            if (m_monitorPools.find(gcMonitorPoolName) == m_monitorPools.end())
            {
                THROW_INVALID_CONFIG(soSet, "error parsing so set element {}. could not find gc monitor pool {} in monitor pools",
                    c_config_key_sos_sets_gc_monitors_pool, gcMonitorPoolName);
            }
            // we need SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT monitor per engine - validated at fillEngineConfigs
            sg.gcMonitorsPool = &m_monitorPools[gcMonitorPoolName];
            // verify that both so & monitor are from the same dcore
            if (sg.gcMonitorsPool->dcoreIndex != sg.sosPool->dcoreIndex)
            {
                THROW_INVALID_CONFIG(soSet,"error parsing  so set. so and monitor should belong to the same dcore ");
            }
        }

        // reset monitors pool
        const std::string resetMonitorPoolName = soSet.at(c_config_key_sos_sets_reset_monitors_pool).get<std::string>();
        if (m_monitorPools.find(resetMonitorPoolName) == m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(soSet, "error parsing so set element {}. could not find reset monitor set {} in monitor pools",
                c_config_key_sos_sets_reset_monitors_pool, resetMonitorPoolName);
        }
        // we need 1 monitor per so set
        if (sg.numSets > m_monitorPools[resetMonitorPoolName].size)
        {
            THROW_INVALID_CONFIG(soSet, "error parsing so set element. setSize {} > num monitors in pool {}",
                sg.setSize, m_monitorPools[resetMonitorPoolName].size);
        }
        sg.resetMonitorsPool = &m_monitorPools[resetMonitorPoolName];
        // verify that both so & monitor are from the same dcore
        if ( sg.resetMonitorsPool->dcoreIndex != sg.sosPool->dcoreIndex)
        {
            THROW_INVALID_CONFIG(soSet, "error parsing  so set. so and monitor should belong to the same dcore ");
        }

        // compute back2back monitors pool
        VALIDATE_JSON_NODE_EXISTS(soSet, c_config_key_sos_sets_compute_back2back_monitors);
        const std::string computeBack2BackMonitorsPoolName = soSet.at(c_config_key_sos_sets_compute_back2back_monitors).get<std::string>();
        if (m_monitorPools.find(computeBack2BackMonitorsPoolName) == m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(soSet, "error parsing so set element {}. could not find reset monitor set {} in monitor pools",
                c_config_key_sos_sets_compute_back2back_monitors, computeBack2BackMonitorsPoolName);
        }
        sg.computeBack2BackMonitorsPool = &m_monitorPools[computeBack2BackMonitorsPoolName];
        // verify that both so & monitor are from the same dcore
        if ( sg.computeBack2BackMonitorsPool->smIndex != sg.sosPool->smIndex)
        {
            THROW_INVALID_CONFIG(soSet, "error parsing compute back2back so set. so and monitor should belong to the same sm");
        }

        // topology debugger monitors pool
        VALIDATE_JSON_NODE_EXISTS(soSet, c_config_key_sos_sets_topology_debugger_monitors_pool);
        const std::string topologyDebuggerPoolName = soSet.at(c_config_key_sos_sets_topology_debugger_monitors_pool).get<std::string>();
        if (m_monitorPools.find(topologyDebuggerPoolName) == m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(soSet, "error parsing so set element {}. could not find reset monitor set {} in monitor pools",
                c_config_key_sos_sets_topology_debugger_monitors_pool, topologyDebuggerPoolName);
        }
        sg.topologyDebuggerMonitorsPool = &m_monitorPools[topologyDebuggerPoolName];
        // verify that both so & monitor are from the same dcore
        if ( sg.topologyDebuggerMonitorsPool->dcoreIndex != sg.sosPool->dcoreIndex)
        {
            THROW_INVALID_CONFIG(soSet, "error parsing  so set. so and monitor should belong to the same sm");
        }

        // verify that SOs range to be from the same 1/4 of dcore
        // so that we don't need to touch SID_MSB in MON_CONFIG register during execution,
        if ( (sg.sosPool->baseIdx / c_max_so_range_per_monitor) != ((sg.sosPool->baseIdx + sg.sosPool->size - 1) / c_max_so_range_per_monitor) )
        {
            THROW_INVALID_CONFIG(soSet, "in {} sosPool {} range {} {} sm {} exceeds the max range per monitor {}",
                sg.name, sg.sosPool->name, sg.sosPool->baseIdx, sg.sosPool->baseIdx + sg.sosPool->size - 1, sg.sosPool->smIndex, c_max_so_range_per_monitor);
        }
        m_soSetsGroups[sg.name] = std::make_shared<SyncObjectsSetGroup>(sg);
        m_schedulersSosMap[schedulerName].push_back(m_soSetsGroups[sg.name].get());
        sg.scheduler->m_sosSetGroups.push_back(m_soSetsGroups[sg.name].get());
    }
}
CATCH_JSON()

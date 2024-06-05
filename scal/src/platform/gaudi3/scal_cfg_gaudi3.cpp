#include <cstring>
#include <string>
#include <fstream>
#include <bitset>
#include <unordered_map>
#include "scal.h"
#include "scal_utilities.h"
#include "scal_gaudi3.h"
#include "logger.h"
#include "gaudi3/asic_reg_structs/qman_arc_aux_regs.h"
#include "scal_data_gaudi3.h"
#include "common/cfg_parsing_helper.h"
#include "infra/sync_mgr.hpp"
#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"
#include "gaudi3/asic_reg_structs/pqm_cmn_b_regs.h"

// clang-format off

int Scal_Gaudi3::getSmInfo(int fd, unsigned smId, struct hlthunk_sync_manager_info* info)
{
    int ret = hlthunk_get_sync_manager_info(fd, smId / c_sync_managers_per_hdcores, info);
    if (!ret)
    {
        if (smId & 0x1)
        {
            info->first_available_sync_object = 0;
            info->first_available_monitor = 0;
            info->first_available_cq = 0;
        }
    }
    return ret;
}

int Scal_Gaudi3::parseConfigFile(const std::string & configFileName)
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

    json = getAsicJson(json, "gaudi3");

    // define the pdma channel config
    parsePdmaChannel(json, version);

    // get the path for the fw binaries files
    parseBinPath(json, version);

    // Parse BackwardCompatibleness flags
    parseBackwardCompatibleness(json);

    // memory settings: global hbm size, host shared size, hbm shared size
    MemoryGroups memoryGroups;
    parseMemorySettings(json, version, memoryGroups);

    // define the schedulers and engine clusters (compute_tpc,media_tpc,mme,edma,pdma,nic)
    parseCores(json, version, memoryGroups);

    parseNics(json, version);

    // define streams_sets (compute,pdma, network_*)
    parseStreams(json, version);

    // define sos pools, monitor pools, completion_queues etc.
    parseSyncInfo(json, version);

    // create direct-mode pdma-channels & also suport parse of scheduler mode pdma channels
    parseUserPdmaChannels(json, version);

    // Validate PDMA channels - in case direct-mode there are no scheduler-managed elements
    validatePdmaChannels(json, version);

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

#define ERROR_MEM_SETTINGS "error parsing mem element. "

void Scal_Gaudi3::parseBackwardCompatibleness(const scaljson::json &json)
try
{
    // in json file - the "backward_compatibleness" : section
    auto backwardCompletiblnessElements = json.find(c_config_key_backward_compatibleness);
    if (backwardCompletiblnessElements != json.end())
    {
        auto autoFetcherElement =
                backwardCompletiblnessElements->find(c_config_key_backward_compatibleness_auto_fetcher);
        if (autoFetcherElement != backwardCompletiblnessElements->end())
        {
            m_use_auto_fetcher = autoFetcherElement->get<bool>();
            LOG_DEBUG(SCAL, "m_use_auto_fetcher {}", m_use_auto_fetcher);
        }
        // tmp - to allow tests to set it off
        bool tmp_use_auto_fetcher = (getenv("DISABLE_SCAL_AUTO_FETCHER") != nullptr);
        if (tmp_use_auto_fetcher)
        {
            m_use_auto_fetcher = false;
            LOG_DEBUG(SCAL, "m_use_auto_fetcher {}", m_use_auto_fetcher);
        }
        // end tmp
    }
}
CATCH_JSON()

Scal_Gaudi3::Pool * Scal_Gaudi3::loadPoolByName(const scaljson::json & memElement, const std::string & pool_key, bool failIfNotExists, Pool * defaultValue)
try
{
    Pool * pool = defaultValue;
    std::string poolName;
    if (memElement.find(pool_key) != memElement.end() || failIfNotExists)
    {
        poolName = memElement.at(pool_key).get<std::string>();
        if (m_pools.count(poolName) == 0)
        {
            THROW_INVALID_CONFIG(memElement.at(c_config_key_memory_global_pool), ERROR_MEM_SETTINGS "pool {} not found in {}", poolName, c_config_key_memory_control_cores_memory_pools);
        }
        pool = &m_pools[poolName];
    }
    LOG_INFO_F(SCAL, "pool {} is {}", pool_key, poolName);
    return pool;
}
CATCH_JSON()

// memory settings: global hbm size, host shared size, hbm shared size
void Scal_Gaudi3::parseMemorySettings(const scaljson::json &json, const ConfigVersion &version, MemoryGroups &groups)
try
{
    // in json file - the "memory" : section
    auto & memElement = json.at(c_config_key_memory);

    std::vector<Pool> pools;
    memElement.at(c_config_key_memory_control_cores_memory_pools).get_to(pools);
    if (pools.empty())
    {
        THROW_INVALID_CONFIG(memElement,ERROR_MEM_SETTINGS "(empty array) {}", c_config_key_memory_control_cores_memory_pools);
    }

    for (unsigned i = 0; i < pools.size(); ++i)
    {
        Pool & pool = pools[i];
        pool.globalIdx = i;
        if (m_pools.count(pool.name) != 0)
        {
            THROW_INVALID_CONFIG(memElement, ERROR_MEM_SETTINGS "name duplication {}", c_config_key_memory_control_cores_memory_pools);
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

    m_binaryPool = loadPoolByName(memElement, c_config_key_memory_binary_pool, true, nullptr);
    if (m_binaryPool->coreBase) //the binary pool must not be addressable by the core
    {
        THROW_INVALID_CONFIG(memElement,"error in binaryPool settings");
    }
    m_globalPool = loadPoolByName(memElement, c_config_key_memory_global_pool, false, m_binaryPool);
    m_pdmaPool = loadPoolByName(memElement, c_config_key_memory_pdma_pool, true, nullptr);
}
CATCH_JSON()


void Scal_Gaudi3::parsePdmaChannel(const scaljson::json &json, const ConfigVersion &version)
try
{
    (void)version;

    VALIDATE_JSON_NODE_IS_OBJECT(json, c_config_key_pdma_channel);

    const scaljson::json & pdmaChannel = json[c_config_key_pdma_channel];

    pdmaChannel.at(c_config_key_pdma_channel_syncman_index).get_to(m_configChannel.m_smIdx);

    bool     isOddSmId = m_configChannel.m_smIdx & 0x1;

    struct hlthunk_sync_manager_info syncManagerInfo;
    if (getSmInfo(m_fd, m_configChannel.m_smIdx, &syncManagerInfo))
    {
        THROW_INVALID_CONFIG(pdmaChannel.at(c_config_key_pdma_channel_syncman_index),
                             "Failed to call hlthunk_get_sync_manager_info() dcore={}. errno = {} {}",
                             m_configChannel.m_smIdx,
                             errno, std::strerror(errno));
    }
    pdmaChannel.at(c_config_key_pdma_channel_monitor_index).get_to(m_configChannel.m_monIdx);

    // There will never be reserved MONs for the even SMs
    if ((!isOddSmId) && (m_configChannel.m_monIdx < syncManagerInfo.first_available_monitor))
    {
        THROW_INVALID_CONFIG(pdmaChannel.at(c_config_key_pdma_channel_monitor_index), "illegal monitor (index {}) used for config PDMA channel first_available_monitor={}. SM {}",
                             m_configChannel.m_monIdx, syncManagerInfo.first_available_monitor, m_configChannel.m_smIdx);
    }

    pdmaChannel.at(c_config_key_pdma_channel_completion_queue_index).get_to(m_configChannel.m_cqIdx);

    // There will never be reserved MONs for the even SMs
    if ((!isOddSmId) && (m_configChannel.m_cqIdx < syncManagerInfo.first_available_cq))
    {
        THROW_INVALID_CONFIG(pdmaChannel.at(c_config_key_pdma_channel_completion_queue_index),"illegal cq (index {}) first_available_cq={} used for config PDMA channel. dcore {}",
                             m_configChannel.m_cqIdx, syncManagerInfo.first_available_cq, m_configChannel.m_smIdx);
    }

    if (m_runningIsrIdx >= m_hw_ip.number_of_user_interrupts)
    {
        THROW_INVALID_CONFIG(pdmaChannel.at(c_config_key_pdma_channel_completion_queue_index),
                             "ran out of interrupts. runningIsrIdx {}. first_available_interrupt_id {} number_of_user_interrupts {} m_configChannel",
                             m_runningIsrIdx, m_hw_ip.first_available_interrupt_id,
                             m_hw_ip.number_of_user_interrupts);
    }

    m_configChannel.m_isrIdx = m_hw_ip.first_available_interrupt_id + m_runningIsrIdx;
    m_runningIsrIdx++;

    const std::string pdmaQueueIdName = pdmaChannel.at("pdma_channel").get<std::string>();
    unsigned pdmaQueueId;
    if (!pdmaName2ChannelID(pdmaQueueIdName, pdmaQueueId))
    {
        THROW_INVALID_CONFIG(pdmaChannel.at("pdma_channel"), "error in pdmaName2ChannelID {}", c_config_key_pdma_channel_pdma_channel);
    }
    m_configChannel.m_qid = pdmaQueueId;
}
CATCH_JSON()


void Scal_Gaudi3::parseNics(const scaljson::json &json, const ConfigVersion &version)
try
{
    (void) version; // for future use
    VALIDATE_JSON_NODE_IS_OBJECT(json, c_config_key_cores);

    const scaljson::json & cores = json[c_config_key_cores];

    // temp - optional nic_clusters node before added to arc_fw jsons
    if (cores.find(c_config_key_cores_nic_clusters) == cores.end())
    {
        LOG_WARN(SCAL, "temp workaround - nic_clusters not provided in json");
        return;
    }

    VALIDATE_JSON_NODE_IS_ARRAY(cores, c_config_key_cores_nic_clusters);

    const scaljson::json & nicCluster = cores[c_config_key_cores_nic_clusters];

    std::unordered_map<std::string, std::vector<Nic>> nicPortsMap;
    bool usePortMaskFromDriver = m_isInternalJson; // if default json, use the LKD mask. If not, use the json setting
    if (usePortMaskFromDriver)
    {
        // When running on ASIC - take scale-up and scale-out clusters from LKD
        struct hlthunk_nic_get_ports_masks_out portsMask;
        int ret = hlthunk_nic_get_ports_masks(m_fd, &portsMask);
        if (ret) // if hlthunk_nic_get_ports_masks fail
        {
            THROW_INVALID_CONFIG(nicCluster, "hlthunk_nic_get_ports_masks fail, return:{:#x}, errno:{} - {}", ret, errno, std::strerror(errno));
        }

        std::bitset<c_ports_count> nicPortsMask(portsMask.ports_mask);
        if (nicPortsMask.to_ullong() != portsMask.ports_mask) // check size of ports_mask
        {
            THROW_INVALID_CONFIG(nicCluster, "error nic_ports_mask {:#x} has more ports than is supported {}", portsMask.ports_mask, c_ports_count);
        }
        if ((portsMask.ports_mask | portsMask.ext_ports_mask) != portsMask.ports_mask)
        {
            THROW_INVALID_CONFIG(nicCluster, "error nic_ports_external_mask {:#x} has more ports than in nic_ports_mask {:#x}", portsMask.ext_ports_mask, portsMask.ports_mask);
        }

        std::bitset<c_ports_count> nicPortsScaleUpMask(portsMask.ports_mask & (portsMask.ports_mask ^ portsMask.ext_ports_mask));
        std::bitset<c_ports_count> nicPortsScaleOutMask(portsMask.ext_ports_mask);
        LOG_INFO(SCAL, "NIC Ports ScaleUp Mask {}", nicPortsScaleUpMask.to_string());
        LOG_INFO(SCAL, "NIC Ports ScaleOut Mask {}", nicPortsScaleOutMask.to_string());

        const unsigned portsPerDie = c_nics_count / 2;
        for (unsigned nicIdx = 0; nicIdx < c_nics_count; nicIdx++)
        {
            unsigned die = nicIdx < portsPerDie ? 0 : 1;
            unsigned nicInDie = nicIdx % (c_nics_count / 2);
            unsigned nicPortBase = nicIdx * c_ports_count_per_nic;
            std::string engineName("NIC_" + std::to_string(die) + "_" + std::to_string(nicInDie));
            std::bitset<c_ports_count_per_nic> scaleUpPortsInNicMask;
            std::bitset<c_ports_count_per_nic> scaleOutPortsInNicMask;
            for (unsigned portInNicIdx = 0; portInNicIdx < c_ports_count_per_nic; portInNicIdx++)
            {
                if (nicPortsScaleUpMask.test(nicPortBase + portInNicIdx) && nicPortsScaleOutMask.test(nicPortBase + portInNicIdx))
                {
                    // can't logically happen - but lets verify
                    THROW_INVALID_CONFIG(nicCluster, "error port {} used for both scaleup and scaleout", nicPortBase + portInNicIdx);
                }
                if (nicPortsScaleUpMask.test(nicPortBase + portInNicIdx))
                {
                    scaleUpPortsInNicMask.set(portInNicIdx);
                }
                if (nicPortsScaleOutMask.test(nicPortBase + portInNicIdx))
                {
                    scaleOutPortsInNicMask.set(portInNicIdx);
                }
            }
            if (scaleUpPortsInNicMask.any())
            {
                nicPortsMap["nic_scaleup"].push_back({engineName, scaleUpPortsInNicMask});
            }
            if (scaleOutPortsInNicMask.any())
            {
                nicPortsMap["nic_scaleout"].push_back({engineName, scaleOutPortsInNicMask});
            }
        }
        for (auto& [clusterName, nicPorts] : nicPortsMap)
        {
            std::stringstream clusterStringStream;
            for (auto& [portName, portMask] : nicPorts)
            {
                clusterStringStream << portName << "_" << portMask.to_string() << " ";
            }
            LOG_INFO(SCAL, "cluster {} overriden by LKD mask: {}", clusterName, clusterStringStream.str());
        }
    }

    for (const auto & clusterObject : nicCluster)
    {
        std::string clusterName;
        clusterObject.at(c_config_key_cores_nic_clusters_name).get_to(clusterName);
        if (m_clusters.find(clusterName) != m_clusters.end())
        {
            THROW_INVALID_CONFIG(clusterObject, "error cluster {} already exist in m_clusters", clusterName);
        }
        std::vector<Nic> nicPorts;
        if (usePortMaskFromDriver)
        {
            if (nicPortsMap.find(clusterName) != nicPortsMap.end())
            {
                nicPorts = nicPortsMap[clusterName];
            }
        }
        else
        {
            clusterObject.at(c_config_key_cores_nic_clusters_ports).get_to(nicPorts);
        }
        if (nicPorts.size() == 0)
        {
            LOG_ERR(SCAL, "cluster {} doesn't have any engines", clusterName);
            continue;
        }
        m_clusters[clusterName] = {};
        Cluster* cluster = &m_clusters[clusterName];
        cluster->name = clusterName;
        scal_assert(arcName2CoreType(nicPorts[0].name, cluster->type), "arcName2CoreType failed");

        VALIDATE_JSON_NODE_IS_ARRAY(clusterObject, c_config_key_cores_nic_clusters_queues)
        parseClusterQueues(clusterObject.at(c_config_key_cores_nic_clusters_queues), version, cluster);
        unsigned indexInGroup = 0;
        for (auto& [nicName, portsInNicMask] : nicPorts)
        {
            unsigned coreCpuId;
            if (!arcName2CpuId(nicName, coreCpuId))
            {
                THROW_INVALID_CONFIG(clusterObject, "error finding CPU ID id of {}", nicName);
            }
            G3NicCore *core = new G3NicCore();
            m_nicCores.push_back(core);
            core->cpuId = coreCpuId;
            core->qman = nicName;
            core->portsMask = portsInNicMask;
            core->ports[0] = core->portsMask.test(0) ? (core->cpuId - CPU_ID_NIC_QMAN_ARC0) * 2     : -1;
            core->ports[1] = core->portsMask.test(1) ? (core->cpuId - CPU_ID_NIC_QMAN_ARC0) * 2 + 1 : -1;
            core->arcName = core->qman;
            if (!arcName2DccmAddr(core->qman, core->dccmDevAddress) ||
                !arcName2QueueId(core->qman, core->qmanID))
            {
                THROW_INVALID_CONFIG(clusterObject, "error finding Qman id or type. core->qman={} core->name={}", core->qman, core->name);
            }
            core->indexInGroup = indexInGroup;
            indexInGroup++;
            core->name = core->qman;
            core->scal = this;

            CoreType coreType;
            scal_assert(arcName2CoreType(core->qman, coreType), "arcName2CoreType failed");
            scal_assert(cluster->type == coreType, "error cluster {} has several core types ({}, {})", clusterName, core->qman, cluster->engines[0]->qman);
            cluster->engines.emplace_back(core);
            core->clusters[clusterName] = cluster;
        }
        m_nicClusters.push_back(&m_clusters[clusterName]);
    }
}
CATCH_JSON()

inline bool getQmanBaseAddress(uint64_t& baseAddress, uint32_t qmanID)
{
    static const unsigned AmountOfPdmaChannels =
        GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_5 - GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_0 + 1;
    static const uint64_t QmansBaseAddresses[AmountOfPdmaChannels] =
    {
        mmD0_SPDMA0_CH0_A_BASE,
        mmD0_SPDMA0_CH1_A_BASE,
        mmD0_SPDMA0_CH2_A_BASE,
        mmD0_SPDMA0_CH3_A_BASE,
        mmD0_SPDMA0_CH4_A_BASE,
        mmD0_SPDMA0_CH5_A_BASE,

        mmD0_SPDMA1_CH0_A_BASE,
        mmD0_SPDMA1_CH1_A_BASE,
        mmD0_SPDMA1_CH2_A_BASE,
        mmD0_SPDMA1_CH3_A_BASE,
        mmD0_SPDMA1_CH4_A_BASE,
        mmD0_SPDMA1_CH5_A_BASE,

        mmD1_SPDMA0_CH0_A_BASE,
        mmD1_SPDMA0_CH1_A_BASE,
        mmD1_SPDMA0_CH2_A_BASE,
        mmD1_SPDMA0_CH3_A_BASE,
        mmD1_SPDMA0_CH4_A_BASE,
        mmD1_SPDMA0_CH5_A_BASE,

        mmD1_SPDMA1_CH0_A_BASE,
        mmD1_SPDMA1_CH1_A_BASE,
        mmD1_SPDMA1_CH2_A_BASE,
        mmD1_SPDMA1_CH3_A_BASE,
        mmD1_SPDMA1_CH4_A_BASE,
        mmD1_SPDMA1_CH5_A_BASE
    };

    if ((qmanID < GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_0) ||
        (qmanID > GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_5))
    {
        LOG_ERR(SCAL, "{}: Not supported qmanId {}", __FUNCTION__, qmanID);
        return false;
    }

    baseAddress = QmansBaseAddresses[qmanID - GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_0];

    return true;
}

inline bool getFenceCounterAddress(uint64_t& fenceCounterAddress, uint32_t qmanID)
{
    static const unsigned cpFenceIdx = 0;

    uint64_t qmanBaseAddress = 0;
    if (!getQmanBaseAddress(qmanBaseAddress, qmanID))
    {
        return false;
    }

    uint64_t cpFenceOffset = offsetof(gaudi3::block_pdma_ch_a, pqm_ch) + offsetof(gaudi3::block_pqm_ch_a, fence_inc);
    cpFenceOffset += sizeof(struct gaudi3::block_pqm_ch_a) * cpFenceIdx;

    fenceCounterAddress = qmanBaseAddress + cpFenceOffset;
    return true;
}

int Scal_Gaudi3::createSingleDirectModePdmaChannel(PdmaChannelInfo const* pdmaChannelInfo,
                                                   unsigned               streamIndex,
                                                   const std::string      streamSetName,
                                                   bool                   isIsrEnabled,
                                                   unsigned               isrIdx,
                                                   unsigned               priority,
                                                   unsigned               longSoSmIndex,
                                                   unsigned               longSoIndex,
                                                   unsigned&              cqIndex,
                                                   bool                   isTdrMode,
                                                   const scaljson::json&  json)
{
    unsigned pdmaQueueId = pdmaChannelInfo->engineId;

    if (!pdmaName2ChannelID(pdmaChannelInfo->name, pdmaQueueId))
    {
        LOG_ERR(SCAL,"{}: Failed to get QMAN-ID for pdma-channel {}",
                __FUNCTION__, pdmaChannelInfo->channelId);
        assert(0);

        return SCAL_FAILURE;
    }

    LOG_TRACE(SCAL,"{} channel's name {} qmanId {}", __FUNCTION__, pdmaChannelInfo->name, pdmaQueueId);

    if (m_directModePdmaChannels.find(pdmaQueueId) != m_directModePdmaChannels.end())
    {
        LOG_ERR(SCAL,"{}: QMAN-ID for pdma-channel {} already configured",
                __FUNCTION__, pdmaChannelInfo->channelId);
        assert(0);

        return SCAL_FAILURE;
    }

    std::string streamName(streamSetName + std::to_string(streamIndex));
    std::string completionGroupName(streamSetName + "_completion_queue" + std::to_string(streamIndex));
    std::string pdmaEngineName = pdmaChannelInfo->name;

    // Create and init Channel
    const uint64_t msix_db_reg = mmD0_PCIE_MSIX_BASE;
    //
    uint64_t fenceCounterAddress = 0;
    if (!getFenceCounterAddress(fenceCounterAddress, pdmaQueueId))
    {
        LOG_ERR(SCAL, "{}: QMAN-ID for pdma-channel {} failed to get fence-counter address",
                __FUNCTION__, pdmaChannelInfo->channelId);
        assert(0);
    }
    //

    std::unique_ptr<DirectModePdmaChannel> pDirectModePdmaChannel =
        std::make_unique<DirectModePdmaChannel>(streamName, pdmaEngineName, completionGroupName, this, pdmaQueueId,
                                                msix_db_reg, (isIsrEnabled) ? isrIdx : scal_illegal_index,
                                                longSoSmIndex, longSoIndex, fenceCounterAddress,
                                                cqIndex++, priority,
                                                pdmaChannelInfo->channelId);
    // Store Stream
    if (m_directModePdmaChannelStreams.find(streamName) != m_directModePdmaChannelStreams.end())
    {
        LOG_ERR(SCAL, "stream {} already exists", streamName);
        assert(0);
    }
    //
    m_directModePdmaChannelStreams[streamName] = pDirectModePdmaChannel->getStream();

    // Store CompletionGroup
    if (m_directModeCompletionGroups.find(completionGroupName) != m_directModeCompletionGroups.end())
    {
        LOG_ERR(SCAL, "Completion-Group {} already exists", completionGroupName);
        assert(0);
    }
    //
    DirectModeCompletionGroup* pCompletionGroup = pDirectModePdmaChannel->getCompletionGroup();
    if (pCompletionGroup == nullptr)
    {
        LOG_ERR(SCAL, "Completion-Group {} not created", completionGroupName);
        assert(0);
    }
    m_directModeCompletionGroups[completionGroupName] = pCompletionGroup;
    m_cgs.push_back(pCompletionGroup);
    if (isTdrMode)
    {
        if (pCompletionGroup->longSoSmIndex == -1)
        {
            LOG_ERR(SCAL, "Completion-Group {} longSoSmIndex not set", completionGroupName);
            assert(0);
        }
        else
        {
            parseTdrCompletionQueues(json, 1, *pCompletionGroup);
            unsigned smID = pCompletionGroup->compQTdr.monSmIdx;
            struct hlthunk_sync_manager_info syncManagerInfo;
            if (getSmInfo(m_fd, smID, &syncManagerInfo))
            {
                LOG_ERR(SCAL, "Completion-Group {} getSmInfo {} failed", completionGroupName, smID);
                assert(0);
            }

            allocateCqIndex(pCompletionGroup->compQTdr.cqIdx,
                            pCompletionGroup->compQTdr.globalCqIndex,
                            json,
                            smID,
                            m_syncManagers[smID].dcoreIndex,
                            syncManagerInfo.first_available_cq);

            pCompletionGroup->compQTdr.enabled       = true;
            LOG_INFO(SCAL, "Completion-Group {} {} tdr enabled, sos={}",
                pCompletionGroup->compQTdr.cqIdx, pCompletionGroup->name, pCompletionGroup->compQTdr.sos);
        }
    }

    m_directModePdmaChannels[pdmaChannelInfo->channelId] = std::move(pDirectModePdmaChannel);

    LOG_TRACE(SCAL,"{} Created channel's name {} qmanId {}", __FUNCTION__, pdmaChannelInfo->name, pdmaQueueId);
    return SCAL_SUCCESS;
}

void Scal_Gaudi3::parseUserPdmaChannels(const scaljson::json &json, const ConfigVersion &version)
try
{
    VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(json, c_config_key_pdma_clusters);

    VALIDATE_JSON_NODE_IS_OBJECT(json, c_config_key_sync);
    auto& syncObject = json[c_config_key_sync];

    unsigned directModeCqIndex = getSchedulerModeCQsCount();
    unsigned longSoSmIndex     = -1;
    unsigned longSoPoolSize    = 0;
    unsigned longSoBaseIndex   = -1;
    unsigned longSoPoolIndex   = -1;

    if (syncObject.find(c_config_key_direct_mode_pdma_channels_long_so_pool) != syncObject.end())
    {
        SyncObjectsPool* directModeLongSoPool =
            &m_soPools[syncObject.at(c_config_key_direct_mode_pdma_channels_long_so_pool).get<std::string>()];

        SyncObjectsPool* completionQueueLongSoPool =
            &m_soPools[syncObject.at(c_config_key_completion_queues_long_so_pool).get<std::string>()];

        if (directModeLongSoPool->smIndex != completionQueueLongSoPool->smIndex)
        {
            THROW_INVALID_CONFIG(json,
                                 "Direct-Mode PDMA-Channel's SM ({}) is different from completion-queue's SM ({})",
                                 directModeLongSoPool->smIndex, completionQueueLongSoPool->smIndex);
        }

        longSoSmIndex   = directModeLongSoPool->smIndex;
        longSoPoolSize  = directModeLongSoPool->size;
        longSoBaseIndex = directModeLongSoPool->baseIdx;
        longSoPoolIndex = (c_so_group_size - (longSoBaseIndex % c_so_group_size)) % c_so_group_size;

        scal_sm_base_addr_tuple_t smBaseAddr;
        smBaseAddr.smId = longSoSmIndex;
        smBaseAddr.smBaseAddr = SyncMgrG3::getSmBase(longSoSmIndex);
        smBaseAddr.spdmaMsgBaseIndex = 0;
        m_smBaseAddrDb.push_back(smBaseAddr);
    }

    const scaljson::json & pdma_clusters_json = json[c_config_key_pdma_clusters];
    for (const auto & pdma_cluster : pdma_clusters_json)
    {
        (void) version; // for future use
        std::string pdmaChannelClusterName;
        pdma_cluster.at(c_config_key_pdma_clusters_name).get_to(pdmaChannelClusterName);

        bool isDirectMode = false;
        if (pdma_cluster.find(c_config_key_pdma_clusters_is_direct_mode) != pdma_cluster.end())
        {
            isDirectMode = pdma_cluster.at(c_config_key_pdma_clusters_is_direct_mode).get<bool>();
        }

        bool isTdrMode = false;
        if (pdma_cluster.find(c_config_key_pdma_clusters_is_tdr) != pdma_cluster.end())
        {
            isTdrMode = pdma_cluster.at(c_config_key_pdma_clusters_is_tdr).get<bool>();
        }

        if (isDirectMode)
        {
            // Direct-Mode streams-set has the same name as of the cluster
            std::string& streamSetName = pdmaChannelClusterName;

            VALIDATE_JSON_NODE_EXISTS(pdma_cluster, c_config_key_pdma_clusters_is_isr_enabled);

            if (longSoPoolSize <= 0)
            {
                THROW_INVALID_CONFIG(pdma_cluster,
                                     "Error direct-mode pdma-cluster {} has invalid long-so's pool-size ({})",
                                     pdmaChannelClusterName, longSoPoolSize);
            }

            bool     isIsrEnabled = pdma_cluster.at(c_config_key_pdma_clusters_is_isr_enabled).get<bool>();

            unsigned pdmaClusterPriority = 0;
            pdma_cluster.at(c_config_key_pdma_clusters_priority).get_to(pdmaClusterPriority);
            // validate that given priority value is in range
            if (pdmaClusterPriority < c_min_priority || pdmaClusterPriority > c_max_priority)
            {
                THROW_INVALID_CONFIG(pdma_cluster, "Priority of cluster {} cannot be set set to {}."
                                     " Priorities must be integers in range of [{}..{}].",
                                     pdmaChannelClusterName, pdmaClusterPriority, c_min_priority, c_max_priority);
            }

            unsigned streamIndex = 0;
            VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(pdma_cluster, c_config_key_pdma_clusters_streams);
            const scaljson::json & pdma_streams = pdma_cluster[c_config_key_pdma_clusters_streams];
            for (const scaljson::json & pdma_stream : pdma_streams)
            {
                VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(pdma_stream, c_config_key_pdma_clusters_stream_engines);

                unsigned isrIdx = 0;
                if (isIsrEnabled)
                {
                    // interrupt index
                    if (m_runningIsrIdx >= m_hw_ip.number_of_user_interrupts)
                    {
                        THROW_INVALID_CONFIG(pdma_cluster,
                                            "ran out of interrupts. runningIsrIdx {}. first_available_interrupt_id {}"
                                            " number_of_user_interrupts {} pdma-cluster {}",
                                            m_runningIsrIdx, m_hw_ip.first_available_interrupt_id,
                                            m_hw_ip.number_of_user_interrupts, pdmaChannelClusterName);
                    }
                    isrIdx = m_hw_ip.first_available_interrupt_id + m_runningIsrIdx;
                    m_runningIsrIdx++;
                }

                std::vector<std::string> pdmaEnginesNames;
                pdma_stream.at(c_config_key_pdma_clusters_stream_engines).get_to(pdmaEnginesNames);
                if (pdmaEnginesNames.size() != 1)
                {
                    THROW_INVALID_CONFIG(pdma_stream,
                                        "Error: pdma-cluster {} stream {} does not support multiple engines",
                                        pdmaChannelClusterName, streamIndex);
                }

                std::string pdmaEngineName = pdmaEnginesNames[0];
                const PdmaChannelInfo* pUserPdmaChannelInfo;
                if (!pdmaName2PdmaChannelInfo(pdmaEngineName, pUserPdmaChannelInfo))
                {
                    THROW_INVALID_CONFIG(pdma_stream,
                                         "Error in pdmaName2PdmaChannelInfo cluster {} engine {}",
                                         pdmaChannelClusterName, pdmaEngineName);
                }

                // validate in each DIE that channels 6-11 has priority 7, this is due to a bug in H/W [SW-110596]
                unsigned channelId = pUserPdmaChannelInfo->channelId;
                if (((channelId >= PDMA_DIE0_CH6) && (channelId < PDMA_DIE1_CH0)) ||
                    (channelId >= PDMA_DIE1_CH6))
                {
                    if (pdmaClusterPriority != 7)
                    {
                        // We should THROW_INVALID_CONFIG. But as of now, and due to the PDMA-NW
                        // support (which has channels on both HP and LP PDMAs), we will only warn about it,
                        // and in addition we will update the configuration to match requirement
                        LOG_WARN(SCAL,
                                 "pdma-cluster {} would be re-configured to use priority 7 (instead of {})",
                                 pdmaChannelClusterName,
                                 pdmaClusterPriority);

                        pdmaClusterPriority = 7;
                    }
                }

                // long-so index
                unsigned longSoIndex = longSoBaseIndex + longSoPoolIndex;
                // each long sob (60 bit) consists of 4 regular sob (15bit)
                if (longSoPoolIndex + c_so_group_size > longSoPoolSize)
                {
                    THROW_INVALID_CONFIG(pdma_stream, "too many long sos used {}", longSoPoolIndex + c_so_group_size);
                }
                longSoPoolIndex += c_so_group_size;

                createSingleDirectModePdmaChannel(pUserPdmaChannelInfo, streamIndex, streamSetName, isIsrEnabled, isrIdx,
                                                  pdmaClusterPriority, longSoSmIndex, longSoIndex, directModeCqIndex,
                                                  isTdrMode, pdma_stream);

                LOG_INFO(SCAL,
                         "Created Direct-Mode PDMA-Channel: stream-set {} stream-index {} PDMA channel's engine {}"
                         " isIsrEnabled {} isrIdx {} pdmaClusterPriority {} longSoIndex {} longSoSmIndex {} tdr {}",
                         streamSetName, streamIndex, pdmaEngineName,
                         isIsrEnabled, isrIdx, pdmaClusterPriority, longSoIndex, longSoSmIndex, isTdrMode);

                streamIndex++;
                m_directModePdmaChannelsAmount++;
                m_streamSets[streamSetName].streamsAmount++;

                uint64_t spdmaPqmCmnMsgBaseAddr = pUserPdmaChannelInfo->baseAddrCmnB + offsetof(gaudi3::block_pqm_cmn_b, cp_msg_base_addr);
                if (std::find(m_spdmaMsgBaseAddrDb.begin(), m_spdmaMsgBaseAddrDb.end(), spdmaPqmCmnMsgBaseAddr) == m_spdmaMsgBaseAddrDb.end())
                {
                    m_spdmaMsgBaseAddrDb.push_back(spdmaPqmCmnMsgBaseAddr);
                }
            }
            m_streamSets[streamSetName].name          = streamSetName;
            m_streamSets[streamSetName].isDirectMode  = true;
        }
        else
        { // BWD-compatible
            VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(pdma_cluster, c_config_key_pdma_clusters_engines);
            std::vector<std::string> pdmaEnginesNames;
            pdma_cluster.at(c_config_key_pdma_clusters_engines).get_to(pdmaEnginesNames);

            VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(pdma_cluster, c_config_key_pdma_clusters_channels);
            const scaljson::json & pdma_channels = pdma_cluster[c_config_key_pdma_clusters_channels];
            for (const scaljson::json & pdma_channel : pdma_channels)
            {
                unsigned pdmaChannelIndex, pdmaChannelPriority;
                pdma_channel.at(c_config_key_pdma_clusters_channels_index).get_to(pdmaChannelIndex);
                pdma_channel.at(c_config_key_pdma_clusters_channels_priority).get_to(pdmaChannelPriority);

                // validate that given priority value is in range
                if (pdmaChannelPriority < c_min_priority || pdmaChannelPriority > c_max_priority)
                {
                    THROW_INVALID_CONFIG(pdma_channel, "Priority of channel #{} cannot be set set to {}. Priorities must be integers in range of [{}..{}].",
                                         pdmaChannelIndex, pdmaChannelPriority, c_min_priority, c_max_priority);
                }

                for (const auto& engineName : pdmaEnginesNames)
                {
                    std::string pdmaChannelName = engineName + "_" + std::to_string(pdmaChannelIndex);

                    VALIDATE_JSON_NODE_IS_OBJECT(pdma_channel, c_config_key_pdma_clusters_channels_scheduler);
                    const scaljson::json & scheduler_json = pdma_channel[c_config_key_pdma_clusters_channels_scheduler];

                    VALIDATE_JSON_NODE_EXISTS(scheduler_json, c_config_key_pdma_clusters_channels_scheduler_group);
                    auto pdmaChannelGroup = scheduler_json[c_config_key_pdma_clusters_channels_scheduler_group];
                    unsigned groupIndex;
                    std::string pdmaChannelGroupName;
                    if (pdmaChannelGroup.is_number_unsigned())
                    {
                        pdmaChannelGroup.get_to(groupIndex);
                        pdmaChannelGroupName = std::to_string(groupIndex);
                    }
                    else
                    {
                        if (pdmaChannelGroup.is_string())
                        {
                            pdmaChannelGroupName = pdmaChannelGroup.get<std::string>();
                            if (!groupName2GroupIndex(pdmaChannelGroupName, groupIndex))
                            {
                                THROW_INVALID_CONFIG(pdma_channel, "error group name {} is invalid", pdmaChannelGroupName);
                            }
                        }
                        else
                        {
                            THROW_INVALID_CONFIG(pdma_channel, "type mismatch. expected string or unsigned int");
                        }
                    }
                    std::string pdmaChannelSchedulerName;
                    scheduler_json.at(c_config_key_pdma_clusters_channels_scheduler_name).get_to(pdmaChannelSchedulerName);
                    PdmaChannel pdmaChannel;
                    if (!pdmaName2PdmaChannelInfo(pdmaChannelName, pdmaChannel.pdmaChannelInfo))
                    {
                        THROW_INVALID_CONFIG(pdma_channel,"error in pdmaName2PdmaChannelInfo {}, {}", pdma_channel, pdmaChannelName);
                    }

                    pdmaChannel.engineGroup = groupIndex;
                    pdmaChannel.priority = pdmaChannelPriority;
                    G3Scheduler* scheduler = getCoreByName<G3Scheduler>(pdmaChannelSchedulerName);
                    LOG_INFO(SCAL, "scheduler {} cluster {} group {}/{} is using PDMA channel {} (#{})",
                             scheduler->name, pdmaChannelClusterName, pdmaChannelGroupName, groupIndex, pdmaChannelName, pdmaChannel.pdmaChannelInfo->channelId);
                    scheduler->pdmaChannels.push_back(std::move(pdmaChannel));
                }
                // Each PDMA-Channel represents a single cluster (and has a group of engines)
                m_clusters[pdmaChannelClusterName].name = pdmaChannelClusterName;
            }
        }
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseSchedulerCores(const scaljson::json &schedulersJson, const ConfigVersion &version, const MemoryGroups &groups)
try
{
    (void) version; // for future use

    for (const auto & sched : schedulersJson)
    {
        std::unique_ptr<G3Scheduler> pCore = std::make_unique<G3Scheduler>();
        G3Scheduler & core = *pCore;
        core.scal = this;
        core.streams = std::vector<Stream *>(c_num_max_user_streams, nullptr);
        core.isScheduler = true;

        sched.at(c_config_key_cores_schedulers_core).get_to(core.arcName);
        sched.at(c_config_key_cores_schedulers_name).get_to(core.name);
        sched.at(c_config_key_cores_schedulers_binary_name).get_to(core.imageName);
        sched.at(c_config_key_cores_schedulers_qman).get_to(core.qman);

        if (!arcName2CpuId(core.arcName, core.cpuId))
        {
            THROW_INVALID_CONFIG(sched,"error in arcName2CpuId arName: {} cpuId: {}", core.arcName, core.cpuId);
        }
        if ((core.cpuId >= c_scheduler_nr) || m_cores[core.cpuId])
        {
            THROW_INVALID_CONFIG(sched,"parse scheduler cores core.cpuId: {} c_scheduler_nr: {}", core.cpuId, c_scheduler_nr);
        }

        if (!arcName2DccmAddr(core.arcName, core.dccmDevAddress) ||
            !arcName2ArcType(core.arcName, core.arcFarm)  ||
            !arcName2HdCore(core.arcName, core.hdCore) ||
            !arcName2DCore(core.arcName, core.dCore))
        {
            THROW_INVALID_CONFIG(sched,"error arcName2DccmAddr failed. arcName: {}", core.arcName);
        }
        if (!core.arcFarm)
        {
            THROW_INVALID_CONFIG(sched,"error - {} is not a scheduler", core.arcName);
        }

        core.dupEngLocalDevAddress = core.dccmDevAddress + c_dccm_to_dup_offset;

        // find Qman id by Qman name
        if (!arcName2QueueId(core.qman, core.qmanID))
        {
            THROW_INVALID_CONFIG(sched,"error in arcName2QueueId {}", c_config_key_cores_schedulers_qman);
        }

        core.dccmMessageQueueDevAddress =
            core.dccmDevAddress + getCoreAuxOffset(&core) + offsetof(gaudi3::block_qman_arc_aux, dccm_queue_push_reg[0]);

        std::string memGroup = sched.at(c_config_key_cores_schedulers_memory_group).get<std::string>();
        if (groups.find(memGroup) == groups.end())
        {
            THROW_INVALID_CONFIG(sched,"error finding group: {}", memGroup);
        }

        core.pools = groups.at(memGroup).pools;
        core.configPool = groups.at(memGroup).configPool;
        m_hdcores.insert(core.hdCore);

        m_cores[core.cpuId] = pCore.release();
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseClusterQueues(const scaljson::json& queuesJson, const ConfigVersion& version, Cluster* cluster)
try
{
    (void)version; // for future use

    for (const auto& queueJson : queuesJson)
    {
        Scal::Cluster::Queue queue;

        queueJson.at(c_config_key_cores_engine_clusters_queues_index).get_to(queue.index);
        if (cluster->type != CoreType::NIC && queue.index >= DCCM_QUEUE_COUNT)
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

        if (cluster->type != CoreType::NIC && cluster->queues.find(queue.index) != cluster->queues.end())
        {
            // the same queue can be allocated twice for 2 dup triggers only for the SAME scheduler and group
            if (cluster->queues[queue.index].scheduler != queue.scheduler)
            {
                THROW_INVALID_CONFIG(queueJson, "error queue index {} apears twice in cluster {} with different schedulers {} vs {}",
                        queue.index, cluster->name, cluster->queues[queue.index].scheduler->name , queue.scheduler->name);
            }
        }

        static_assert(sizeof(c_dup_trigger_info) / sizeof(c_dup_trigger_info[0]) <= c_max_push_regs_per_dup, "too many dup triggers defined in c_dup_trigger_info");
        unsigned numOfDupTriggers = sizeof(c_dup_trigger_info) / sizeof(c_dup_trigger_info[0]);
        VALIDATE_JSON_NODE_IS_ARRAY(queueSchedulerJson, c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger);
        for (auto& dup : queueSchedulerJson[c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger])
        {
            queue.dupConfigs.emplace_back();
            Cluster::DupConfig& dupConfig = queue.dupConfigs.back();
            dup.at(c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger_dup).get_to(dupConfig.dupTrigger);
            if (dupConfig.dupTrigger > numOfDupTriggers)
            {
                THROW_INVALID_CONFIG(dup[c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger_dup], "dup trigger out of range");
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
        if (queue.group_index >= QMAN_ENGINE_GROUP_TYPE_COUNT)
        {
            THROW_INVALID_CONFIG(clustersQueuesShedulerGroupJson, "cluster {} queue {} (scheduler {}) group index {} is more then max engine type {}", cluster->name, queue.index, queue.scheduler->name, queue.group_index, QMAN_ENGINE_GROUP_TYPE_COUNT);
        }

        cluster->queues[cluster->queues.size()]  = std::move(queue);
        queue.scheduler->clusters[cluster->name] = cluster;
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseEngineCores(const scaljson::json &enginesJson, const ConfigVersion &version, const MemoryGroups &groups)
try
{
    static const std::string cmeClusterName = "cme";
    (void) version; // for future use
    std::map<std::string, bool> allEngines;
    bool isCmeCoreEnabled = false;
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

        auto& memoryGroup = groups.at(memGroup);
        std::vector<Pool*> pools = memoryGroup.pools;
        Pool* configPool = memoryGroup.configPool;

        std::vector<EngineWithImage> engines;
        clusterObject.at(c_config_key_cores_engine_clusters_engines).get_to(engines);

        std::string qmanName;
        if (clusterObject.find(c_config_key_cores_engine_clusters_qman) != clusterObject.end())
        {
            clusterObject.at(c_config_key_cores_engine_clusters_qman).get_to(qmanName);
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
        scal_assert(arcName2CoreType(engines[0].engine, cluster->type), "arcName2CoreType failed");

        VALIDATE_JSON_NODE_IS_ARRAY(clusterObject, c_config_key_cores_engine_clusters_queues)
        parseClusterQueues(clusterObject.at(c_config_key_cores_engine_clusters_queues), version, cluster);
        cluster->enginesPerHDCore.resize(c_hdcores_nr);
        cluster->enginesPerDCore.resize(c_dcores_nr);
        for (unsigned offset = 0; offset < engines.size(); offset++)
        {
            unsigned coreCpuId;
            if (!arcName2CpuId(engines[offset].engine, coreCpuId))
            {
                THROW_INVALID_CONFIG(clusterObject, "error finding CPU ID id of {}", engines[offset].engine);
            }
            // if ((coreCpuId < c_scheduler_nr) ||
            //     (coreCpuId >= c_cores_nr))
            // {
            //     THROW_INVALID_CONFIG(clusterObject, "coreCpuId {} out of range [{}..{}) core.name={}", coreCpuId, c_scheduler_nr, c_cores_nr, engines[offset]);
            // }
            G3ArcCore* core = nullptr;
            if (m_cores[coreCpuId] == nullptr)
            {
                // new core
                if (clusterName == cmeClusterName)
                {
                    core = new G3CmeCore();
                    isCmeCoreEnabled = true;
                }
                else
                {
                    core = new G3ArcCore();
                }
                m_cores[coreCpuId] = core;
                core->cpuId = coreCpuId;
                core->qman = qmanName.empty() ? engines[offset].engine : qmanName;
                core->arcName = engines[offset].engine;
                CoreType coreType;
                if (!arcName2DccmAddr(core->arcName,  core->dccmDevAddress) ||
                    !arcName2QueueId(core->qman,  core->qmanID)  ||
                    !arcName2HdCore(core->arcName, core->hdCore) ||
                    !arcName2DCore(core->arcName, core->dCore)   ||
                    !arcName2CoreType(core->arcName, coreType))
                {
                    THROW_INVALID_CONFIG(clusterObject, "error finding Qman id or type. core->qman={} core->name={}", core->qman, core->name);
                }
                scal_assert((coreType == CoreType::SCHEDULER || cluster->type == coreType), "error cluster {} has several core types ({}, {})", clusterName, core->qman, cluster->engines[0]->qman);
                core->indexInGroup = offset;
                core->numEnginesInGroup = engines.size();
                core->name = clusterName + std::string("_") + std::to_string(offset);
                core->pools = pools;
                core->configPool = configPool;
                core->imageName = engines[offset].image.empty() ? imageName : engines[offset].image;
                core->scal = this;
                core->indexInGroupInHdCore = cluster->enginesPerHDCore[core->hdCore];
                core->indexInGroupInDCore = cluster->enginesPerDCore[core->dCore];
                core->dccmMessageQueueDevAddress = core->dccmDevAddress + getCoreAuxOffset(core) + offsetof(gaudi3::block_qman_arc_aux, dccm_queue_push_reg[0]);
                cluster->enginesPerHDCore[core->hdCore]++;
                cluster->enginesPerDCore[core->dCore]++;
            }
            else
            {
                core = getCore<G3ArcCore>(coreCpuId);
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
                if (imageName != core->imageName)
                {
                    THROW_INVALID_CONFIG(clusterObject, "error core {} in cluster {} config mismatch of imageName {} vs {}", core->arcName, cluster->name, imageName, core->imageName);
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

            m_clusters[clusterName].engines.emplace_back(core);
            core->clusters[clusterName] = cluster;
            m_hdcores.insert(core->hdCore);
        }

        if (clusterObject.find(c_config_key_cores_engine_clusters_is_local_dup) != clusterObject.end())
        {
            unsigned isLocalDup;
            auto isLocalDupNode = clusterObject.at(c_config_key_cores_engine_clusters_is_local_dup);
            isLocalDupNode.get_to(isLocalDup);
            if (isLocalDup > 0)
            {
                cluster->localDup = true;
            }
        }

        if (clusterObject.find(c_config_key_cores_engine_clusters_is_compute) != clusterObject.end())
        {
            unsigned isCompute;
            auto isComputeNode = clusterObject.at(c_config_key_cores_engine_clusters_is_compute);
            isComputeNode.get_to(isCompute);
            if (isCompute)
            {
                if (clusterName == cmeClusterName)
                {
                    THROW_INVALID_CONFIG(isComputeNode, "CME cluster cannot be compute");
                }
                cluster->isCompute = true;
                m_computeClusters.push_back(cluster);
            }
        }
    }
    if (isCmeCoreEnabled == false)
    {
        m_arc_fw_synapse_config.cme_enable = false;
    }
}
CATCH_JSON()


void Scal_Gaudi3::parseSyncManagers(const scaljson::json &syncManagersJson, const ConfigVersion &version)
try
{
    // parse "sync_managers" array
    for (const scaljson::json & syncManagerJson : syncManagersJson)
    {
        unsigned smID = syncManagerJson.at(c_config_key_sync_managers_index).get<unsigned>();
        if (smID >= c_sync_managers_nr)
        {
            THROW_INVALID_CONFIG(syncManagerJson, "illegal sync manager ID {}", smID);
        }

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


void Scal_Gaudi3::parseSosPools(const scaljson::json &sosPoolsJson, const ConfigVersion &version, const unsigned smID)
try
{
    const bool isOddSmId = smID & 0x1;

    // checks to be done:
    // no overlap between Sos on pools
    // total number of Sos in SM < 8K

    struct hlthunk_sync_manager_info syncManagerInfo;
    if (getSmInfo(m_fd, smID, &syncManagerInfo))
    {
        THROW_INVALID_CONFIG(sosPoolsJson, "Failed to call hlthunk_get_sync_manager_info() sm={}. errno = {} {}", smID, errno, std::strerror(errno));
    }
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
        if (soJson.find(c_config_key_sync_managers_sos_pools_align) != soJson.end() )
        {
            unsigned align;
            soJson.at(c_config_key_sync_managers_sos_pools_align).get_to(align);
            if (so.baseIdx % align != 0)
            {
                THROW_INVALID_CONFIG(soJson,"base_index {} of {} in smID {} should be align to {}",
                   so.baseIdx, so.name, smID, align);
            }
        }
        // check for overlap
        for (auto prevSo : m_syncManagers[smID].soPools)
        {
            unsigned int start = so.baseIdx;
            unsigned int end = so.baseIdx + so.size - 1;
            unsigned int otherStart = prevSo->baseIdx;
            unsigned int otherEnd = prevSo->baseIdx + prevSo->size - 1;
            if (isOverlap(start, end, otherStart, otherEnd))
            {
                THROW_INVALID_CONFIG(soJson, "overlap in so indices in smID {} (baseIdx={}) between {} and {}", smID, so.baseIdx, prevSo->name, so.name);
            }
        }
        const unsigned dcoreIdx = smID / c_sync_managers_per_hdcores;
        so.dcoreIndex = dcoreIdx;
        so.smIndex = smID;
        so.scal = this;
        so.smBaseAddr = SyncMgrG3::getSmBase(smID);
        m_soPools[so.name] = so;
        m_syncManagers[smID].soPools.push_back(&m_soPools[so.name]);
        m_syncManagers[smID].dcoreIndex = dcoreIdx;
        m_syncManagers[smID].smIndex = smID;
        m_syncManagers[smID].baseAddr = so.smBaseAddr;

        // There will never be reserved SOBJs for the even SMs
        if ((!isOddSmId) && (so.baseIdx < syncManagerInfo.first_available_sync_object))
        {
            THROW_INVALID_CONFIG(sosPoolsJson, "pool {} min used so index {} < {} in smID {}", so.name, so.baseIdx, syncManagerInfo.first_available_sync_object, smID);
        }
        if (so.baseIdx + so.size > c_max_sos_per_sync_manager)
        {
            THROW_INVALID_CONFIG(sosPoolsJson, "pool {} max number of sos {} > {} in SM {}", so.name, so.baseIdx + so.size, (unsigned)c_max_sos_per_sync_manager, smID);
        }
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseMonitorPools(const scaljson::json &monitorPoolJson, const ConfigVersion &version, const unsigned smID)
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
        mp.dcoreIndex = smID / 2;
        mp.smIndex = smID;
        mp.scal = this;
        mp.smBaseAddr = SyncMgrG3::getSmBase(smID);
        m_monitorPools[mp.name] = mp;
        m_syncManagers[smID].monitorPools.push_back(&m_monitorPools[mp.name]);
    }
    if (totalNumOfMonitors > c_max_monitors_per_sync_manager)
    {
        THROW_INVALID_CONFIG(monitorPoolJson, "total number of monitors in smID {} > {}  (totalNumOfMonitors={})", smID, (unsigned)c_max_monitors_per_sync_manager, totalNumOfMonitors);
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseSyncManagersCompletionQueues(const scaljson::json &syncObjectJson, const ConfigVersion &version)
try
{
    // parse completion queues long so pool
    SyncObjectsPool* longSoPool      = &m_soPools[syncObjectJson.at(c_config_key_completion_queues_long_so_pool).get<std::string>()];

    SyncObjectsPool* sfgSosPool = nullptr;
    MonitorsPool* sfgMonitorsPool = nullptr;
    MonitorsPool* sfgCqMonitorsPool = nullptr;
    parseSfgPool(syncObjectJson, &sfgSosPool, &sfgMonitorsPool, &sfgCqMonitorsPool);

    const char * cme_engines_monitors_pool = "cme_engines_monitors_pool";
    if (syncObjectJson.find(cme_engines_monitors_pool) != syncObjectJson.end())
    {
        std::string cme_monitors_pool_name = syncObjectJson.at(cme_engines_monitors_pool).get<std::string>();
        if (m_monitorPools.count(cme_monitors_pool_name) == 0)
        {
            THROW_INVALID_CONFIG(syncObjectJson.at(cme_engines_monitors_pool), "cme_engines_monitors_pool \"{}\" not found in so pools", cme_monitors_pool_name);
        }
        m_cmeEnginesMonitorsPool = &m_monitorPools[cme_monitors_pool_name];
    }

    const scaljson::json &syncManagerJson = syncObjectJson.at(c_config_key_sync_managers);
    for (const scaljson::json & smJson : syncManagerJson)
    {
        unsigned smID = smJson.at(c_config_key_sync_managers_index).get<unsigned>();

        // run in a different loop because parseCompletionQueues needs the rest of the sync manager members to be initialized
        if (smJson.find(c_config_key_sync_managers_completion_queues) != smJson.end())
        {
            VALIDATE_JSON_NODE_IS_ARRAY(smJson, c_config_key_sync_managers_completion_queues);
            // only the first SM in each HDCORE has CQs
            if (smID % 2 != 0)
            {
                VALIDATE_JSON_NODE_IS_AN_EMPTY_ARRAY(smJson, c_config_key_sync_managers_completion_queues);
            }
            else
            {
                parseCompletionQueues(smJson[c_config_key_sync_managers_completion_queues], version, smID, longSoPool,
                                      sfgSosPool, sfgMonitorsPool, sfgCqMonitorsPool);
            }
        }
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseTdrCompletionQueues(const scaljson::json & completionQueueJsonItem, unsigned numberOfInstances, CompletionGroupInterface& cq)
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

    if (compQTdr.monSmIdx != compQTdr.sosPool->smIndex)
    {
        THROW_CONFIG_ERROR(SCAL_INVALID_CONFIG, completionQueueJsonItem, "SM-Index mismatch ({}): between Monitor-pool {} and SO-pool {}",
                           cq.name, compQTdr.monPool->smIndex, compQTdr.sosPool->smIndex);
    }
    if (typeid(cq) == typeid(CompletionGroup))// only CompletionGroup has monitorsPool, other classes derived from CompletionGroupInterface do not
    {
        unsigned monPoolsmIndex = ((CompletionGroup&)cq).monitorsPool->smIndex;
        if (compQTdr.monSmIdx != monPoolsmIndex)
        {
            LOG_ERR_F(SCAL, "tdr mon and sos of cq {} SM {} doesn't match CQ SM {}", cq.name, compQTdr.monSmIdx, monPoolsmIndex);
        }
    }
}

void Scal_Gaudi3::parseCompletionQueues(const scaljson::json& completionQueueJson, const ConfigVersion& version,
                                        const unsigned smID, SyncObjectsPool* longSoPool,
                                        SyncObjectsPool* sfgSosPool, MonitorsPool* sfgMonitorsPool, MonitorsPool* sfgCqMonitorsPool)
try
{
    // Remark: Direct-Mode's Completion-Queues are created during scal-init

    struct hlthunk_sync_manager_info                syncManagerInfo;
    std::map<std::string, std::vector<std::string>> schedulersCqGroupsMap; // map of: scheduler name -> CQ names
    if (getSmInfo(m_fd, smID, &syncManagerInfo))
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
            const Scheduler* scheduler = getCoreByName<Scheduler>(schedulerName);
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
        unsigned totalSos = cq.sosNum * numberOfInstances;
        if (totalSos + cq.sosPool->nextAvailableIdx - cq.sosPool->baseIdx > cq.sosPool->size)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "too many sos used {} in {} for {} instances x {} schedulers in smID {}",
                                 totalSos, sosPoolName, numberOfInstances, totalNumSchedulers, smID);
        }
        cq.sosBase = cq.sosPool->nextAvailableIdx;
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

        if (smID == 0 && cq.monitorsPool->baseIdx < syncManagerInfo.first_available_monitor)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "illegal monitor index {} first_available_monitor={} used in monitor pool {} in smID {}",
                                 cq.monitorsPool->baseIdx, syncManagerInfo.first_available_monitor, monitorsPoolName, smID);
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
         uint64_t cmSobjBaseAddr = 0;
        if (cq.slaveSchedulers.size() != 0)
        {
            int status = getCreditManagmentBaseIndices(cq.creditManagementSobIndex, cmSobjBaseAddr, cq.creditManagementMonIndex, false);

            if (status != SCAL_SUCCESS)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "Invalid Credit-Management configuration");
            }
        }

        if (numberOfUserMonitors == 1)
            cq.force_order = false; // if monitor depth is 1, ignore force_order (as there's no need for order between the monitors...)
        /*
            actualNumberOfMonitors scal must config per cq is 3 + (force_order?1:0) + #of slave monitors
            in Gaudi3 - monitors support up to 16 messages
        */
        cq.actualNumberOfMonitors = c_completion_queue_monitors_set_size + (unsigned)cq.force_order + totalNumSchedulers - 1;
        //
        cq.monNum = numberOfUserMonitors * cq.actualNumberOfMonitors;
        cq.monBase = cq.monitorsPool->nextAvailableIdx;
        unsigned totalMon = cq.monNum * numberOfInstances;
        if (totalMon + cq.monitorsPool->nextAvailableIdx - cq.monitorsPool->baseIdx > cq.monitorsPool->size)
        {
            THROW_INVALID_CONFIG(completionQueueJsonItem, "too many monitors used {} in {} in smID {}",
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

        // SFG configuration
        parseSfgCompletionQueueConfig(completionQueueJsonItem, cq, sfgSosPool, sfgMonitorsPool, sfgCqMonitorsPool);

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
        cq.pCounter = 0;    // initialized after cq counters are allocated (in configureCQs)
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
            CompletionGroup cqInstance = cq;
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
                unsigned tdrSmID = cq.compQTdr.monSmIdx;
                if (getSmInfo(m_fd, tdrSmID, &syncManagerInfo))
                {
                    THROW_INVALID_CONFIG(completionQueueJson,
                                         "Failed to call hlthunk_get_sync_manager_info() TDR smID={}. errno = {} {}",
                                         smID,
                                         errno, std::strerror(errno));
                }

                cqInstance.compQTdr.sos           = cq.compQTdr.sos + instance;
                cqInstance.compQTdr.monitor       = cq.compQTdr.monitor + CompQTdr::NUM_MON * instance;

                allocateCqIndex(cqInstance.compQTdr.cqIdx,
                                cqInstance.compQTdr.globalCqIndex,
                                completionQueueJsonItem,
                                tdrSmID,
                                m_syncManagers[tdrSmID].dcoreIndex,
                                syncManagerInfo.first_available_cq);
            }

            // interrupt index
            if (m_runningIsrIdx >= m_hw_ip.number_of_user_interrupts)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "ran out of interrupts. runningIsrIdx {}. first_available_interrupt_id {} number_of_user_interrupts {} cqInstance {}",
                                     m_runningIsrIdx, m_hw_ip.first_available_interrupt_id,
                                     m_hw_ip.number_of_user_interrupts, cqInstance.name);
            }
            // interrupt index
            if (enable_isr)
            {
                cqInstance.isrIdx = m_hw_ip.first_available_interrupt_id + m_runningIsrIdx;
                m_runningIsrIdx++;
            }
            else
            {
                cqInstance.isrIdx = scal_illegal_index;
            }


            cqInstance.longSoIndex   = cq.longSosPool->nextAvailableIdx;
            cqInstance.longSoSmIndex = longSoPool->smIndex;
            // each long sob (60 bit) consists of 4 regular sob (15bit)
            if (cq.longSosPool->nextAvailableIdx - cq.longSosPool->baseIdx + c_so_group_size > longSoPool->size)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "too many long sos used {} in smID {}", cq.longSosPool->nextAvailableIdx, smID);
            }
            cq.longSosPool->nextAvailableIdx += c_so_group_size;
            cqInstance.idxInScheduler = m_schedulersCqsMap[masterSchedulerName].size();

            if (cqInstance.idxInScheduler >= COMP_SYNC_GROUP_COUNT)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem,"too many CQs in {} scheduler ", masterSchedulerName);
            }
            LOG_DEBUG(SCAL,"cq name {} Idx {} schedulerName {} smID {} monBase {} monNum {} sosBase {} sosNum {} longSoIndex"
                      " {} numberOfUserMonitors {} idxInScheduler {} isr {} tdr: {}",
                      cqInstance.name, cqInstance.cqIdx, masterSchedulerName, smID,
                      cqInstance.monBase, cq.monNum , cqInstance.sosBase, cq.sosNum,
                      cqInstance.longSoIndex, numberOfUserMonitors, cqInstance.idxInScheduler,
                      cqInstance.isrIdx, cqInstance.compQTdr.enabled);
            if (cqInstance.compQTdr.enabled)
            {
                 LOG_DEBUG(SCAL,"TDR sos: SM {} index {}, mon: SM {} index {}",
                           cqInstance.compQTdr.sosPool->smIndex, cqInstance.compQTdr.sos,
                           cqInstance.compQTdr.monPool->smIndex, cqInstance.compQTdr.monitor);
            }

            m_completionGroups.insert({cqInstance.name, cqInstance});
            CompletionGroup* pCQ = &m_completionGroups.at(cqInstance.name);
            m_cgs.push_back(pCQ);
            m_syncManagers[smID].completionGroups.push_back(pCQ);

            m_schedulersCqsMap[masterSchedulerName].push_back(pCQ);
            // for each slave scheduler, add a reference to this cq instance to its list
            handleSlaveCqs(pCQ, masterSchedulerName);
        }
    }
    if (m_computeCompletionQueuesSos == nullptr)
    {
        THROW_INVALID_CONFIG(completionQueueJson, "no compute completion queue was defined");
    }
}
CATCH_JSON()

void Scal_Gaudi3::parseHostFenceCounters(const scaljson::json &syncObjectJson, const ConfigVersion &version)
try
{
    const scaljson::json &syncManagerJson = syncObjectJson.at(c_config_key_sync_managers);
    for (const scaljson::json & smJson : syncManagerJson)
    {
        unsigned smID = smJson.at(c_config_key_sync_managers_index).get<unsigned>();
        struct hlthunk_sync_manager_info syncManagerInfo;

        if (getSmInfo(m_fd, smID, &syncManagerInfo))
        {
            THROW_INVALID_CONFIG(syncManagerJson,"Failed to call hlthunk_get_sync_manager_info() dcore={}. errno = {} {}", smID, errno, std::strerror(errno));
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
                                                           " sync manager idx {} is not mapped to the user space", smID);
                }
                // name prefix
                fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_name_prefix).get_to(ctr.name);

                if (smID & 0x1)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson,"fence counter {} is instantiated in SM {} that does not have CQs", ctr.name, smID);
                }

                // number_of_instances
                unsigned numberOfInstances = fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_number_of_instances).get<unsigned>();

                // sos_pool
                ctr.sosPool = nullptr;
                const std::string sosPoolName = fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_sos_pool).get<std::string>();
                if (m_soPools.find(sosPoolName) == m_soPools.end())
                {
                    THROW_INVALID_CONFIG(fenceCouterJson,"so pool {} of host fence counter {} in SM {} was not found", sosPoolName, ctr.name, smID);
                }
                ctr.sosPool = &m_soPools[sosPoolName];
                ctr.soIdx = ctr.sosPool->nextAvailableIdx;
                if (numberOfInstances + ctr.sosPool->nextAvailableIdx - ctr.sosPool->baseIdx > ctr.sosPool->size)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "too many sos used in {} for {} fence counter instances in SM {}",
                                                                    sosPoolName, numberOfInstances, smID);
                }
                ctr.sosPool->nextAvailableIdx += numberOfInstances;

                if (ctr.sosPool->smIndex != smID)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "SOs pool {} of SM {} is used by host fence counter {} in SM {}",
                        ctr.sosPool->name, ctr.sosPool->smIndex, ctr.name, smID);
                }

                // monitors_pool
                ctr.monitorsPool = nullptr;
                const std::string monitorsPoolName = fenceCouterJson.at(c_config_key_sync_managers_host_fence_counters_monitors_pool).get<std::string>();
                if (m_monitorPools.find(monitorsPoolName) == m_monitorPools.end())
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "monitors pool {} of host fence counter name {} in smID {} was not found", monitorsPoolName, ctr.name, smID);
                }

                ctr.monitorsPool = &m_monitorPools[monitorsPoolName];
                if (ctr.monitorsPool->baseIdx < syncManagerInfo.first_available_monitor)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "illegal monitor index {} used in monitor pool {} in SM {}",
                            ctr.monitorsPool->baseIdx, monitorsPoolName, smID);
                }

                ctr.monBase = ctr.monitorsPool->nextAvailableIdx;
                unsigned totalMon = c_host_fence_ctr_mon_nr * numberOfInstances;
                if (totalMon + ctr.monitorsPool->nextAvailableIdx - ctr.monitorsPool->baseIdx > ctr.monitorsPool->size)
                {
                    THROW_INVALID_CONFIG(fenceCouterJson, "too many monitors used {} in {} in SM {}",
                            ctr.monBase + totalMon, std::string(monitorsPoolName), smID);
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
                        THROW_INVALID_CONFIG(fenceCouterJson,"sync manager configuration must have qman. e.g. \"qman\"=\"TPC_1_1\"");
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


void Scal_Gaudi3::parseSoSets(const scaljson::json &soSetJson, const ConfigVersion &version)
try
{
     // parse "sos_sets" array
    for (const scaljson::json & soSet : soSetJson)
    {
        SyncObjectsSetGroupGaudi3 sg;
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

        // central so set
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

        switch (m_arc_fw_synapse_config.sync_scheme_mode)
        {
            case ARC_FW_LEGACY_SYNC_SCHEME:
            case ARC_FW_GAUDI2_SYNC_SCHEME:
            {
                if (sg.setSize < NUM_SOS_PER_SO_SET)
                {
                    THROW_INVALID_CONFIG(soSet, "error parsing so set {} in mode ARC_FW_GAUDI2_SYNC_SCHEME. so set size {} != NUM_SOS_PER_SO_SET {}", sg.name, sg.setSize, NUM_SOS_PER_SO_SET);
                }
                break;
            }
            case ARC_FW_GAUDI3_SYNC_SCHEME:
            {
                if (sg.setSize < NUM_SOS_PER_CENTRAL_SO_SET)
                {
                    THROW_INVALID_CONFIG(soSet, "error parsing so set {} in mode ARC_FW_GAUDI3_SYNC_SCHEME. so set size {} <  NUM_SOS_PER_CENTRAL_SO_SET {}", sg.name, sg.setSize, NUM_SOS_PER_CENTRAL_SO_SET);
                }
                // local so sets
                for (unsigned hdcore : m_hdcores)
                {
                    const std::string localSosPoolName = soSet.at(c_config_key_sos_sets_local_sos_pool).get<std::string>() + "_hdcore" + std::to_string(hdcore);
                    if (m_soPools.find(localSosPoolName) == m_soPools.end())
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing  so set element {}. could not find {} in so pools", c_config_key_sos_sets_local_sos_pool, localSosPoolName);
                    }
                    if (m_soPools[localSosPoolName].size < c_soset_local_sobs_nr)
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing local so set {}. pool size {} < c_soset_local_sobs_nr {}",
                            localSosPoolName, m_soPools[localSosPoolName].size, c_soset_local_sobs_nr);
                    }
                    const std::string localMonPoolName = soSet.at(c_config_key_sos_sets_local_mon_pool).get<std::string>() + "_hdcore" + std::to_string(hdcore);
                    if (m_monitorPools.find(localMonPoolName) == m_monitorPools.end())
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing  so set element {}. could not find {} in so pools", c_config_key_sos_sets_local_mon_pool, localMonPoolName);
                    }
                    if (m_monitorPools[localMonPoolName].size < c_soset_local_monitors_nr)
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing local so set {}. pool size {} < c_soset_local_monitors_nr {}",
                            localMonPoolName, m_monitorPools[localMonPoolName].size, c_soset_local_monitors_nr);
                    }

                    // local barrier
                    const std::string localBarrierSosPoolName = soSet.at(c_config_key_sos_sets_local_barrier_sos_pool).get<std::string>() + "_hdcore" + std::to_string(hdcore);
                    if (m_soPools.find(localBarrierSosPoolName) == m_soPools.end())
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing  so set element {}. could not find {} in so pools", c_config_key_sos_sets_local_barrier_sos_pool, localBarrierSosPoolName);
                    }
                    if (m_soPools[localBarrierSosPoolName].size < c_soset_local_sobs_nr)
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing local so set {}. pool size {} < c_soset_local_sobs_nr {}",
                            localBarrierSosPoolName, m_soPools[localBarrierSosPoolName].size, c_soset_local_sobs_nr);
                    }
                    const std::string localBarrierMonPoolName = soSet.at(c_config_key_sos_sets_local_barrier_mons_pool).get<std::string>() + "_hdcore" + std::to_string(hdcore);
                    if (m_monitorPools.find(localBarrierMonPoolName) == m_monitorPools.end())
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing  so set element {}. could not find {} in so pools", c_config_key_sos_sets_local_barrier_mons_pool, localBarrierMonPoolName);
                    }
                    if (m_monitorPools[localBarrierMonPoolName].size < c_soset_local_monitors_nr)
                    {
                        THROW_INVALID_CONFIG(soSet, "error parsing local so set {}. pool size {} < c_soset_local_monitors_nr {}",
                            localBarrierMonPoolName, m_monitorPools[localBarrierMonPoolName].size, c_soset_local_monitors_nr);
                    }

                    // set the pools
                    sg.localSoSetResources[hdcore] = { .localSosPool = &m_soPools[localSosPoolName],               .localMonitorsPool= &m_monitorPools[localMonPoolName],
                                                       .localBarrierSosPool = &m_soPools[localBarrierSosPoolName], .localBarrierMonitorsPool= &m_monitorPools[localBarrierMonPoolName] };

                }
                break;
            }
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
            // verify that both so & monitor are from the same SM
            if (sg.gcMonitorsPool->smIndex != sg.sosPool->smIndex)
            {
                THROW_INVALID_CONFIG(soSet,"error parsing  so set. so and monitor should belong to the same SM ");
            }
        }

        if (soSet.find(c_config_key_sos_sets_cme_monitors_pool) != soSet.end())
        {
            std::string cmeMonitorPoolName = soSet[c_config_key_sos_sets_cme_monitors_pool].get<std::string>();
            if (m_monitorPools.find(cmeMonitorPoolName) == m_monitorPools.end())
            {
                THROW_INVALID_CONFIG(soSet, "error parsing so set element {}. could not find cme monitor pool {} in monitor pools",
                    c_config_key_sos_sets_cme_monitors_pool, cmeMonitorPoolName);
            }
            // we need SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT monitor per engine - validated at fillEngineConfigs
            m_cmeMonitorsPool = &m_monitorPools[cmeMonitorPoolName];
            // TODO: verification
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
        // verify that both so & monitor are from the same SM
        if ( sg.resetMonitorsPool->smIndex != sg.sosPool->smIndex)
        {
            THROW_INVALID_CONFIG(soSet, "error parsing  so set. so and monitor should belong to the same SM ");
        }

        // compute back2back monitors pool
        if (soSet.find(c_config_key_sos_sets_compute_back2back_monitors) != soSet.end()) // todo: temp apotional pool
        {
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
        }

        // topology debugger monitors pool
        if (soSet.find(c_config_key_sos_sets_topology_debugger_monitors_pool) != soSet.end()) // todo: temp apotional pool
        {
            const std::string topologyDebuggerPoolName = soSet.at(c_config_key_sos_sets_topology_debugger_monitors_pool).get<std::string>();
            if (m_monitorPools.find(topologyDebuggerPoolName) == m_monitorPools.end())
            {
                THROW_INVALID_CONFIG(soSet, "error parsing so set element {}. could not find reset monitor set {} in monitor pools",
                    c_config_key_sos_sets_topology_debugger_monitors_pool, topologyDebuggerPoolName);
            }
            sg.topologyDebuggerMonitorsPool = &m_monitorPools[topologyDebuggerPoolName];
            // verify that both so & monitor are from the same dcore
            if ( sg.topologyDebuggerMonitorsPool->smIndex != sg.sosPool->smIndex)
            {
                THROW_INVALID_CONFIG(soSet, "error parsing topology debugger so set. so and monitor should belong to the same sm");
            }
        }

        // verify that SOs range to be from the same 1/4 of dcore
        // so that we don't need to touch SID_MSB in MON_CONFIG register during execution,
        if ( (sg.sosPool->baseIdx / c_max_so_range_per_monitor) != ((sg.sosPool->baseIdx + sg.sosPool->size - 1) / c_max_so_range_per_monitor) )
        {
            THROW_INVALID_CONFIG(soSet, "in {} sosPool {} range {} {} in sm {} exceeds the max range per monitor {}",
                sg.name, sg.sosPool->name, sg.sosPool->baseIdx, sg.sosPool->baseIdx + sg.sosPool->size - 1, sg.sosPool->smIndex, c_max_so_range_per_monitor);
        }
        m_soSetsGroups[sg.name] = std::make_shared<SyncObjectsSetGroupGaudi3>(sg);
        m_schedulersSosMap[schedulerName].push_back((SyncObjectsSetGroupGaudi3*)(m_soSetsGroups[sg.name].get()));
        sg.scheduler->m_sosSetGroups.push_back((SyncObjectsSetGroupGaudi3*)(m_soSetsGroups[sg.name].get()));
    }
}
CATCH_JSON()

void Scal_Gaudi3::validatePdmaChannels(const scaljson::json &json, const ConfigVersion &version)
try
{
    (void) version; // for future use

    for (auto& channelEntry : m_directModePdmaChannels)
    {
        DirectModePdmaChannel& dmChannel = *(channelEntry.second);
        std::string streamName = dmChannel.getStream()->getName();
        if (m_streams.find(streamName) != m_streams.end())
        {
            THROW_INVALID_CONFIG(json,
                                    "Validation failure: Direct-Mode Stream {} had also been defined"
                                    " as managed by-scheduler",
                                    streamName);
        }

        std::string completionQueueName = dmChannel.getCompletionGroup()->name;
        if (m_completionGroups.find(completionQueueName) != m_completionGroups.end())
        {
            THROW_INVALID_CONFIG(json,
                                    "Validation failure: managed by-scheduler Completion-Queue {}"
                                    " had been defined for Direct-Mode Cluster",
                                    completionQueueName);
        }
    }

    // validate some fields are not defined in the json file for direct mode streamSets
    // since it means they are scheduler managed - error!!
    for (auto& streamSetEntry : m_streamSets)
    {
        if (!(streamSetEntry.second.isDirectMode)) continue;

        std::string streamSetName = streamSetEntry.first;
        std::string sobjsPoolName = streamSetName + "_queues_sos";
        if (m_soPools.find(sobjsPoolName) != m_soPools.end())
        {
            THROW_INVALID_CONFIG(json,
                                    "Validation failure: managed by-scheduler SOBJs {}"
                                    " had been defined for Direct-Mode Cluster {}",
                                    sobjsPoolName,
                                    streamSetName);
        }

        std::string tdrSobjsPoolName = streamSetName + "_completion_queue_tdr_sos";
        if (m_soPools.find(tdrSobjsPoolName) != m_soPools.end())
        {
            THROW_INVALID_CONFIG(json,
                                    "Validation failure: managed by-scheduler Completion-Queue's TDR-SOBJs {}"
                                    " had been defined for Direct-Mode Cluster {}",
                                    tdrSobjsPoolName,
                                    streamSetName);
        }

        std::string monitorsPoolName = streamSetName + "_queue_monitors";
        if (m_monitorPools.find(monitorsPoolName) != m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(json,
                                    "Validation failure: managed by-scheduler MONs {}"
                                    " had been defined for Direct-Mode Cluster {}",
                                    monitorsPoolName,
                                    streamSetName);
        }

        std::string tdrMonitorsPoolName = streamSetName + "_completion_queue_tdr_monitors";
        if (m_monitorPools.find(tdrMonitorsPoolName) != m_monitorPools.end())
        {
            THROW_INVALID_CONFIG(json,
                                    "Validation failure: managed by-scheduler Completion-Queue's TDR-MONs {}"
                                    " had been defined for Direct-Mode Cluster {}",
                                    tdrMonitorsPoolName,
                                    streamSetName);
        }
    }
}
CATCH_JSON()

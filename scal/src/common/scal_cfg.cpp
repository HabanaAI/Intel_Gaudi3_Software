#include <fstream>
#include "scal_base.h"
#include "logger.h"
#include "common/cfg_parsing_helper.h"

extern const char* SCAL_SHA1_VERSION;
extern const char* SCAL_BUILD_TIME;

const unsigned Scal::CompQTdr::NUM_MON;

const scaljson::json Scal::getAsicJson(const scaljson::json& json, const char* asicType)
{
    if (json.find(asicType) != json.end())
    {
        return json[asicType];
    }
    LOG_WARN(SCAL, "{}: asics array no found", asicType);
    return json;
}

void Scal::getConfigVersion(const scaljson::json &json, ConfigVersion &version)
{
    VALIDATE_JSON_NODE_EXISTS(json, c_config_key_version);

    json.at(c_config_key_version).get_to(version);

    if ((version.major != 1U) || (version.minor != 0U) || (version.revision != 0U))
    {
        THROW_CONFIG_ERROR(SCAL_UNSUPPORTED_CONFIG_VERSION, json[c_config_key_version], "unsupported config version major={} minor={} revision={}", version.major, version.minor ,version.revision);
    }
}

Scal::Buffer * Scal::createAllocatedBuffer()
{
    Buffer * newBuffer = new Buffer;
    std::lock_guard guard(m_allocatedBuffersMtx);
    m_allocatedBuffers.insert(newBuffer);
    return newBuffer;
}

void Scal::deleteAllocatedBuffer(const Buffer * buffer)
{
    delete buffer;
    std::lock_guard guard(m_allocatedBuffersMtx);
    m_allocatedBuffers.erase(buffer);
}

void Scal::deleteAllAllocatedBuffers()
{
    std::lock_guard guard(m_allocatedBuffersMtx);
    for (auto buffer : m_allocatedBuffers)
    {
        delete buffer;
    }
    m_allocatedBuffers.clear();
}

void  Scal::printConfigInfo(const std::string &configFileName, const std::string &content)
{
    // to be able to detect when users use old binary with newer json, or new binary with old json
    // we print the scal latest git commit id + hash of the json content.
    std::size_t str_hash = std::hash<std::string>{}(content);
    // we need this in CI logs or bug reports
    // but we need to avoid using LOG_ERR
    auto oldLevel = hl_logger::getLoggingLevel(scal::LoggerTypes::SCAL);
    HLLOG_SET_LOGGING_LEVEL(SCAL, HLLOG_LEVEL_TRACE);
    LOG_INFO(SCAL, "+-------------------------------------------------+");
    LOG_INFO(SCAL, "SCAL Commit SHA1 = {}", SCAL_SHA1_VERSION);
    LOG_INFO(SCAL, "SCAL Build Time = {}", SCAL_BUILD_TIME);
    LOG_INFO(SCAL, "SCAL loading config from {}", configFileName.empty() ? "default.json" : configFileName);
    const char * envVarValue = getenv("SCAL_CFG_OVERRIDE_PATH");
    if (envVarValue)
    {
        LOG_INFO(SCAL, "SCAL config Override = {}", envVarValue);
    }
    LOG_INFO(SCAL, "SCAL config Hash = {:#x}", str_hash);
    LOG_INFO(SCAL, "+-------------------------------------------------+");
    HLLOG_SET_LOGGING_LEVEL(SCAL, oldLevel);
}



// get the path for the fw binaries files
void Scal::parseBinPath(const scaljson::json &json, const ConfigVersion &version)
try
{
    (void) version; // for future use

    const char * envBinPath = getenv(c_bin_path_env_var_name);
    const char * enginesFwBuildPath = getenv(c_engines_fw_build_path_env_var_name);
    const char * packageBinPath = "/opt/habanalabs/engines_fw";
    if (envBinPath)
    {
        // split the environment paths  (separated by ;) to a vectors of strings
        std::string path(envBinPath);
        size_t start = 0;
        size_t end = path.find(";");
        while (end != std::string::npos)
        {
            m_fwImageSearchPath.push_back(path.substr(start, end - start));
            start = end + 1;
            end = path.find(";", start);
        }
        m_fwImageSearchPath.push_back(path.substr(start));
    }
    if (enginesFwBuildPath)
    {
        m_fwImageSearchPath.push_back(enginesFwBuildPath);
    }
    // add the package path
    {
        m_fwImageSearchPath.push_back(packageBinPath);
    }
    if (m_fwImageSearchPath.empty())
    {
        THROW_CONFIG_ERROR(SCAL_EMPTY_PATH, json, "fwImageSearchPath is empty");
    }
}
CATCH_JSON()


void Scal::parseMemoryGroups(const scaljson::json &memoryGroupJson, const ConfigVersion &version, MemoryGroups &groups)
try
{
    (void) version; // for future use

    bool usedRanges[MemoryExtensionRange::RANGES_NR] = {0};
    usedRanges[MemoryExtensionRange::ICCM] = true;
    usedRanges[MemoryExtensionRange::CFG] = true;
    usedRanges[MemoryExtensionRange::HBM0] = true;
    usedRanges[MemoryExtensionRange::DCCM] = true;
    usedRanges[MemoryExtensionRange::LBU] = true;

    const std::string groupName = memoryGroupJson.at(c_config_key_memory_memory_groups_name).get<std::string>();
    if (groups.find(groupName) != groups.end())
    {
        THROW_INVALID_CONFIG(memoryGroupJson[c_config_key_memory_memory_groups_name],"error parsing memoryGroupJson. group name: {} not found in groups ", groupName);
    }

    MemoryGroup & memoryGroup = groups[groupName];

    const std::string configPool = memoryGroupJson.at(c_config_key_memory_memory_groups_config_pool).get<std::string>();

    std::vector<std::string> poolNames;
    memoryGroupJson.at(c_config_key_memory_memory_groups_pools).get_to(poolNames);

    if (poolNames.empty())
    {
        THROW_INVALID_CONFIG(memoryGroupJson.at(c_config_key_memory_memory_groups_pools), "error parsing memoryGroupJson. empty array");
    }

    for (std::string const & poolName : poolNames)
    {
        if (m_pools.find(poolName) == m_pools.end())
        {
            THROW_INVALID_CONFIG(memoryGroupJson.at(c_config_key_memory_memory_groups_pools),"error parsing memoryGroupJson pool name: {} not found in pools", poolName);
        }

        Pool * poolPtr = &m_pools[poolName];
        memoryGroup.pools.push_back(poolPtr);

        const unsigned rangesNr = (poolPtr->size / c_core_memory_extension_range_size) +
                                  ((poolPtr->size % c_core_memory_extension_range_size) ? 1 : 0);

        for (unsigned i=0; i<rangesNr; i++)
        {
            unsigned r = poolPtr->addressExtensionIdx + i;
            if ((r >= MemoryExtensionRange::RANGES_NR) || usedRanges[r])
            {
                THROW_INVALID_CONFIG(memoryGroupJson.at(c_config_key_memory_memory_groups_pools),"error in memoryGroupJson poolName: {} range r={}", poolName, r);
            }

            usedRanges[r] = true;
        }

        if (poolName == configPool)
        {
            memoryGroup.configPool = poolPtr;
        }
    }

    if (!memoryGroup.configPool)
    {
        THROW_INVALID_CONFIG(memoryGroupJson,"error in config pool for group {}", groupName);
    }
}
CATCH_JSON()


void Scal::parseCores(const scaljson::json &json, const ConfigVersion &version, const MemoryGroups &groups)
try
{
    VALIDATE_JSON_NODE_IS_OBJECT(json, c_config_key_cores);

    const scaljson::json & cores = json[c_config_key_cores];

    VALIDATE_JSON_NODE_IS_ARRAY(cores, c_config_key_cores_schedulers);

    parseSchedulerCores(cores[c_config_key_cores_schedulers], version, groups);

    if (cores.find(c_config_key_cores_engine_clusters) != cores.end())
    {
        VALIDATE_JSON_NODE_IS_ARRAY(cores, c_config_key_cores_engine_clusters);

        parseEngineCores(cores[c_config_key_cores_engine_clusters], version, groups);
    }
}
CATCH_JSON()


void Scal::parseStreams(const scaljson::json &json, const ConfigVersion &version)
try
{
    // Remark: StreamsSet-s for Direct-Mode are created during scal-init

    (void) version; // for future use

    VALIDATE_JSON_NODE_IS_ARRAY(json, c_config_key_streams_set)

    for (const auto & streamSetJson : json[c_config_key_streams_set])
    {
        // Set StreamSet
        StreamSet streamSet;

        streamSetJson.at(c_config_key_streams_set_name_prefix).get_to(streamSet.name);
        streamSetJson.at(c_config_key_streams_set_streams_nr).get_to(streamSet.streamsAmount);

        streamSet.isDirectMode = false;

        m_streamSets[streamSet.name] = streamSet;

        // Set Stream (basic)
        Stream stream;

        stream.scheduler = nullptr;
        const std::string schedulerName = streamSetJson.at(c_config_key_streams_set_scheduler).get<std::string>();
        for (unsigned i=0; i < m_schedulerNr; i++)
        {
            if (m_cores[i] && (m_cores[i]->name == schedulerName))
            {
                stream.scheduler = m_cores[i]->getAs<Scheduler>();
            }
        }

        if (!stream.scheduler)
        {
            THROW_INVALID_CONFIG(streamSetJson, "could not find sched name: {} stream-set's name: {}", schedulerName, streamSet.name);
        }

        streamSetJson.at(c_config_key_streams_set_base_idx).get_to(stream.id);
        streamSetJson.at(c_config_key_streams_set_dccm_buffer_size).get_to(stream.dccmBufferSize);

        if (streamSetJson.find(c_config_key_streams_set_is_stub) != streamSetJson.end())
        {
            streamSetJson.at(c_config_key_streams_set_is_stub).get_to(stream.isStub);
        }

        stream.priority = SCAL_LOW_PRIORITY_STREAM;

        for (unsigned i = 0; i < streamSet.streamsAmount; i++)
        {
            std::string newName = streamSet.name + std::to_string(i);
            if (m_streams.find(newName) != m_streams.end())
            {
                THROW_INVALID_CONFIG(streamSetJson, "stream {} already exists in streams. stream-set's name = {}", newName, streamSet.name);
            }

            m_streams[newName]      =  stream;
            m_streams[newName].name =  newName;
            m_streams[newName].id   += i;

            if ((m_streams[newName].id >= c_num_max_user_streams) ||
                m_streams[newName].scheduler->streams[m_streams[newName].id])
            {
                THROW_INVALID_CONFIG(streamSetJson, "illegal id {} at stream {}. stream-set's name = {}", m_streams[newName].id, newName, streamSet.name);
            }

            m_streams[newName].scheduler->streams[m_streams[newName].id] = &m_streams[newName];
        }
    }
}
CATCH_JSON()

void Scal::parseSyncInfo(const scaljson::json &json, const ConfigVersion &version)
try
{
    // parse "sync" object
    VALIDATE_JSON_NODE_IS_OBJECT(json, c_config_key_sync);

    auto & syncObject = json[c_config_key_sync];

    VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(syncObject, c_config_key_sync_managers);

    parseSyncManagers(syncObject[c_config_key_sync_managers], version);

    VALIDATE_JSON_NODE_IS_OBJECT(syncObject, c_config_key_completion_group_credits);
    parseCompletionGroupCredits(syncObject[c_config_key_completion_group_credits], version);

    VALIDATE_JSON_NODE_IS_OBJECT(syncObject, c_config_key_distributed_completion_group_credits);
    parseDistributedCompletionGroupCredits(syncObject[c_config_key_distributed_completion_group_credits], version);

    parseSyncManagersCompletionQueues(syncObject, version);

    parseHostFenceCounters(syncObject, version);

    VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(syncObject, c_config_key_sos_sets);
    parseSoSets(syncObject[c_config_key_sos_sets], version);
}
CATCH_JSON()



void Scal::parseCompletionGroupCredits(const scaljson::json &fwCreditsJson, const ConfigVersion &version)
try
{
    const std::string soPoolName = fwCreditsJson.at(c_config_key_completion_group_credits_sos_pool).get<std::string>();
    if (m_soPools.count(soPoolName) == 0)
    {
        THROW_INVALID_CONFIG(fwCreditsJson, "error parsing completion group credits element {}. could not find {} in sos pools",
            c_config_key_completion_group_credits_sos_pool, soPoolName);
    }
    m_completionGroupCreditsSosPool = &m_soPools[soPoolName];

    const std::string monitorPoolName = fwCreditsJson.at(c_config_key_completion_group_credits_monitors_pool).get<std::string>();
    if (m_monitorPools.count(monitorPoolName) == 0)
    {
        THROW_INVALID_CONFIG(fwCreditsJson, "error parsing completion group credits element {}. could not find {} in monitor pools",
            c_config_key_completion_group_credits_monitors_pool, monitorPoolName);
    }
    m_completionGroupCreditsMonitorsPool = &m_monitorPools[monitorPoolName];
}
CATCH_JSON()


void Scal::parseDistributedCompletionGroupCredits(const scaljson::json &fwCreditsJson, const ConfigVersion &version)
try
{
    const std::string soPoolName = fwCreditsJson.at(c_config_key_distributed_completion_group_credits_sos_pool).get<std::string>();
    if (m_soPools.count(soPoolName) == 0)
    {
        THROW_INVALID_CONFIG(fwCreditsJson, "error parsing distributed completion group credits element {}. could not find {} in sos pools",
            c_config_key_distributed_completion_group_credits_sos_pool, soPoolName);
    }
    m_distributedCompletionGroupCreditsSosPool = &m_soPools[soPoolName];

    const std::string monitorPoolName = fwCreditsJson.at(c_config_key_distributed_completion_group_credits_monitors_pool).get<std::string>();
    if (m_monitorPools.count(monitorPoolName) == 0)
    {
        THROW_INVALID_CONFIG(fwCreditsJson, "error parsing distributed completion group credits element {}. could not find {} in monitor pools",
            c_config_key_distributed_completion_group_credits_monitors_pool, monitorPoolName);
    }
    m_distributedCompletionGroupCreditsMonitorsPool = &m_monitorPools[monitorPoolName];
}
CATCH_JSON()

void Scal::parseSfgPool(const scaljson::json & syncObjectJson, SyncObjectsPool** sfgSosPool, MonitorsPool** sfgMonitorsPool, MonitorsPool** sfgCqMonitorsPool)
try
{
    // sfg_sos_pool
    const char * c_sfg_sos_pool = "sfg_sos_pool";
    const char * c_sfg_monitors_pool = "sfg_monitors_pool";
    const char * c_sfg_cq_monitors_pool = "sfg_cq_monitors_pool";

    *sfgSosPool = nullptr;
    if (syncObjectJson.find(c_sfg_sos_pool) != syncObjectJson.end())                                                               \
    {
        *sfgSosPool = &m_soPools[syncObjectJson.at(c_sfg_sos_pool).get<std::string>()];
    }

    // sfg_monitors_pool
    *sfgMonitorsPool = nullptr;
    if (syncObjectJson.find(c_sfg_monitors_pool) != syncObjectJson.end())
    {
        *sfgMonitorsPool = &m_monitorPools[syncObjectJson.at(c_sfg_monitors_pool).get<std::string>()];
    }

    // sfg_cq_monitors_pool
    *sfgCqMonitorsPool = nullptr;
    if (syncObjectJson.find(c_sfg_cq_monitors_pool) != syncObjectJson.end())
    {
        *sfgCqMonitorsPool = &m_monitorPools[syncObjectJson.at(c_sfg_cq_monitors_pool).get<std::string>()];
    }
}
CATCH_JSON()

void Scal::parseSfgCompletionQueueConfig(const scaljson::json &completionQueueJsonItem, CompletionGroup &cq, SyncObjectsPool* sfgSosPool, MonitorsPool* sfgMonitorsPool, MonitorsPool* sfgCqMonitorsPool)
try
{
    // SFG configuration
    const char * c_sfg_enabled = "sfg_enabled";
    if (completionQueueJsonItem.find(c_sfg_enabled) != completionQueueJsonItem.end())
    {
        completionQueueJsonItem.at(c_sfg_enabled).get_to(cq.sfgInfo.sfgEnabled);
        if (cq.sfgInfo.sfgEnabled)
        {
            if (!sfgSosPool || !sfgMonitorsPool || !sfgCqMonitorsPool)
            {
                THROW_INVALID_CONFIG(completionQueueJsonItem, "Invalid SFG configuration");
            }
            cq.sfgInfo.sfgSosPool         = sfgSosPool;
            cq.sfgInfo.sfgMonitorsPool    = sfgMonitorsPool;
            cq.sfgInfo.sfgCqMonitorsPool  = sfgCqMonitorsPool;
        }
    }
}
CATCH_JSON()

void Scal::handleSlaveCqs(CompletionGroup* pCQ, const std::string& masterSchedulerName)
{
        // sort the slave schedulers list inplace by amount of cqs, the one who has the most cqs will be first
        // this solves the case where a small cq slave scheduler gets the master index, and later the master index must be
        // increased due to a high cq slave scheduler
        sort(pCQ->slaveSchedulers.begin(), pCQ->slaveSchedulers.end(), [&](const SlaveSchedulerInCQ& a, const SlaveSchedulerInCQ& b) -> bool
        {
            return m_schedulersCqsMap[a.scheduler->name].size() > m_schedulersCqsMap[b.scheduler->name].size();
        });
        // for each slave scheduler, add a reference to this cq instance to its list
        for (auto& schedCQ : pCQ->slaveSchedulers)
        {
            unsigned otherCoreCqListSize = m_schedulersCqsMap[schedCQ.scheduler->name].size();
            // HCL credit system depends on all master and slave cqs having
            // the same index number within their respective scheduler cq array

            // if the slave index is lower than the master, increase slave index
            while (otherCoreCqListSize < pCQ->idxInScheduler)
            {
                // add a dummy cq (e.g. null) and create a "hole" in the slave scheduler
                // cq array
                LOG_INFO(SCAL,"adding slave (dummy) cq name {} Idx {} to slave scheduler {} CQIdx {}",
                        pCQ->name, pCQ->idxInScheduler, schedCQ.scheduler->name, otherCoreCqListSize);
                m_schedulersCqsMap[schedCQ.scheduler->name].push_back(nullptr);
                otherCoreCqListSize++;
            }
            // else - if the slave index is larger than the master, increase master index
            while (otherCoreCqListSize > pCQ->idxInScheduler)
            {
                LOG_INFO(SCAL,"adding slave (dummy) cq name {} Idx {} to master scheduler {}",
                    pCQ->name, pCQ->idxInScheduler, masterSchedulerName);
                std::vector<CompletionGroup*>& vec = m_schedulersCqsMap[masterSchedulerName];
                auto it  = std::find(vec.begin(), vec.end(), pCQ);
                if (it != vec.end())
                {
                    m_schedulersCqsMap[masterSchedulerName].insert(it,nullptr);
                    pCQ->idxInScheduler++;
                }
            }
            schedCQ.idxInScheduler = otherCoreCqListSize;
            m_schedulersCqsMap[schedCQ.scheduler->name].push_back(&m_completionGroups.at(pCQ->name));
            LOG_DEBUG(SCAL,"adding slave cq name {} Index in master {} (master scheduler is {}) to slave schedulerName {} idxInScheduler {}",
                    pCQ->name, pCQ->idxInScheduler, masterSchedulerName, schedCQ.scheduler->name, schedCQ.idxInScheduler);
        }
}

int Scal::setTimeouts(const scal_timeouts_t * timeouts)
{
    if (timeouts->timeoutUs != SCAL_TIMEOUT_NOT_SET)
    {
        m_timeoutMicroSec = timeouts->timeoutUs;
    }

    if (timeouts->timeoutNoProgressUs != SCAL_TIMEOUT_NOT_SET)
    {
        m_timeoutUsNoProgress = timeouts->timeoutNoProgressUs;
    }
    if (m_bgWork)
    {
        m_bgWork->setTimeouts(m_timeoutUsNoProgress, m_timeoutDisabled);
    }
    LOG_INFO_F(SCAL, "SCAL timeoutMicroSec was set to {} microseconds by setTimeouts", m_timeoutMicroSec);
    LOG_INFO_F(SCAL, "SCAL timeoutNoProgress was set to {} microseconds by setTimeouts", m_timeoutUsNoProgress);
    return SCAL_SUCCESS;
}

int Scal::getTimeouts(scal_timeouts_t * timeouts)
{
    timeouts->timeoutUs           = m_timeoutMicroSec;
    timeouts->timeoutNoProgressUs = m_timeoutUsNoProgress;
    return SCAL_SUCCESS;
}

int Scal::disableTimeouts(bool disableTimeouts_)
{
    m_timeoutDisabled = disableTimeouts_;
    if (m_bgWork)
    {
        m_bgWork->setTimeouts(m_timeoutUsNoProgress, m_timeoutDisabled);
    }

    LOG_INFO_F(SCAL, "SCAL disableTimeout was set to {} by disableTimeouts", m_timeoutDisabled);
    return SCAL_SUCCESS;
}
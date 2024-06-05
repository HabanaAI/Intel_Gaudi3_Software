#include <cassert>
#include <vector>
#include <mutex>
#include <cstring>
#include <set>
#include <dlfcn.h>
#include "scal.h"
#include "scal_base.h"
#include "scal_shim_if.h"
#include "logger.h"
#include "common/shim_typedefs.h" // from specs/common/shim_typedefs.h
#include "scal_utilities.h"

static std::vector<Scal*> scalInstances;
static std::mutex         scalInstancesMutex;

static std::mutex         scalShimMutex;
static bool               scalShim_Initialized = false;
static void              *scalShim_lib = nullptr;
static PFN_ShimFinish     scalShim_finish = nullptr;
static std::set<int>      scalShim_openedFds; /* keep set of open fds, when the set becomes empty finish the shim */

namespace{
extern const scal_func_table  default_scal_funcs;
}
const scal_func_table *scal_funcs = &default_scal_funcs;


[[maybe_unused]] static void scal_shim_init(int fd)
{
    std::unique_lock<std::mutex> lock(scalShimMutex);

    scalShim_openedFds.insert(fd);

    if (!scalShim_Initialized)
    {
        const char *env_var = getenv("HABANA_SHIM_DISABLE");
        if (env_var == NULL || strcmp(env_var, "1") != 0)
        {
            scalShim_lib = dlopen(SHIM_LIB_NAME, RTLD_LAZY);
            if (scalShim_lib == NULL)
                return;

            PFN_ShimGetFunctions shim_get_functions;
            *(void **) (&shim_get_functions) = dlsym(scalShim_lib, SHIM_GET_FUNCTIONS);

            if (shim_get_functions)
            {
                // Set API version
                PFN_ShimSetApiVersion shim_set_api;
                *(void **) (&shim_set_api) = dlsym(scalShim_lib, SHIM_SET_API_VERSION);

                if (shim_set_api)
                {
                    shim_set_api(SHIM_API_SCAL, SCAL_INTERFACE_VERSION);
                }
                else {
                    LOG_WARN(
                        SCAL,
                        "{} was not found in {}. This may cause unexpected behavior due to interface versions mismatch",
                        SHIM_SET_API_VERSION,
                        SHIM_LIB_NAME);
                }

                *(void **) (&scalShim_finish) = dlsym(scalShim_lib, SHIM_FINISH);

	            /*
	             * TODO: start/stop shim intercept is not supported at the moment.
	             * Currently, we call ShimGetFunctions only once in the initialization
	             * To support interception during execution,
	             * we will have to call it before every (or specific) API calls.
	             */
                scal_funcs = (const scal_func_table *)shim_get_functions(SHIM_API_SCAL, (void *)scal_funcs);
            }
            else {
                LOG_ERR(SCAL, "{}: could not find {} symbol in {}", __FUNCTION__, SHIM_GET_FUNCTIONS, SHIM_LIB_NAME);
                dlclose(scalShim_lib);
                scalShim_lib = NULL;
            }
        }

        scalShim_Initialized = true;
    }
}

[[maybe_unused]] static void scal_shim_fini(int fd)
{
    std::unique_lock<std::mutex> lock(scalShimMutex);

    if (scalShim_openedFds.erase(fd) == 0)
    {
        LOG_ERR(SCAL, "{}: fd {} not initialized or already removed", __FUNCTION__, fd);
        return;
    }

    if (scalShim_openedFds.empty() && scalShim_Initialized && scalShim_lib)
    {
        scal_funcs = &default_scal_funcs;
        if (scalShim_finish)
        {
            (*scalShim_finish)(SHIM_API_SCAL);
        }
        dlclose(scalShim_lib);
        scalShim_Initialized = false;
        scalShim_lib = nullptr;
    }
}

int scal_set_logs_folder(const char * logs_folder)
{
/*
    std::string path = scal::Logger::logsFolder(logs_folder);

    std::string original_path = logs_folder ? logs_folder : "";
    if (!original_path.empty() && original_path.back() != '/')
    {
        original_path.push_back('/');
    }

    return path == original_path ? SCAL_SUCCESS : SCAL_FAILURE;
    */
    return SCAL_SUCCESS;
}

static int scal_init_orig(int fd, const char * config_file_path, scal_handle_t * scal, scal_arc_fw_config_handle_t fwCfg)
{
    if (!config_file_path || !scal || !fd)
    {
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    // using a block to keep lockage to a minimum
    {
        std::unique_lock<std::mutex> lock(scalInstancesMutex);
        for (auto it = scalInstances.begin(); it != scalInstances.end(); ++it)
        {
            if ((*it)->getFD() == fd)
            {
                *scal = (scal_handle_t)(*it);
                return SCAL_SUCCESS;
            }
        }
    }

    Scal *inst;
    // Scal initialization is done here
    int ret = Scal::create(fd, config_file_path, fwCfg, &inst);
    if (ret != SCAL_SUCCESS)
    {
        return ret;
    }
    else if (!inst)
    {
        assert(0);
        return SCAL_FAILURE;
    }

    *scal = (scal_handle_t)inst;
    std::unique_lock<std::mutex> lock(scalInstancesMutex);
    scalInstances.push_back(inst);
    return SCAL_SUCCESS;
}

#define ENABLE_SHIM 1

int scal_init(int fd, const char * config_file_path, scal_handle_t * scal, scal_arc_fw_config_handle_t fwCfg)
{
#if ENABLE_SHIM
    scal_shim_init(fd);
#endif
    return (*scal_funcs->fp_scal_init)(fd, config_file_path, scal, fwCfg);
}

static void scal_destroy_orig(const scal_handle_t scal)
{
    assert(scal);
    std::unique_lock<std::mutex> lock(scalInstancesMutex);
    for (auto it = scalInstances.begin(); it != scalInstances.end(); ++it)
    {
        if (*it == (Scal*)scal)
        {
            scalInstances.erase(it);
            break;
        }
    }
    Scal::destroy((Scal*)scal);
}

void scal_destroy(const scal_handle_t scal)
{
#if ENABLE_SHIM
    int fd = -1;
    if ((Scal *)scal != nullptr)
    {
        // keep fd for scal_shim_fini(). fd is needed to be erased from the set of opended fds.
        fd = ((Scal *)scal)->getFD();
    }
#endif
    (*scal_funcs->fp_scal_destroy)(scal);
#if ENABLE_SHIM
    scal_shim_fini(fd);
#endif
}

static int scal_get_handle_from_fd_orig(int fd, scal_handle_t* scal)
{
    assert(scal);
    std::unique_lock<std::mutex> lock(scalInstancesMutex);
    for (auto it = scalInstances.begin(); it != scalInstances.end(); ++it)
    {
        if ((*it)->getFD() == fd)
        {
            *scal = (scal_handle_t)(*it);
            return SCAL_SUCCESS;
        }
    }
    return SCAL_FAILURE;
}

int scal_get_handle_from_fd(int fd, scal_handle_t* scal)
{
    return (*scal_funcs->fp_scal_get_handle_from_fd)(fd, scal);
}

static int scal_get_fd_orig(const scal_handle_t scal)
{
    assert(scal);
    return  ((Scal*)scal)->getFD();
}

int scal_get_fd(const scal_handle_t scal)
{
    return (*scal_funcs->fp_scal_get_fd)(scal);
}

static uint32_t scal_get_sram_size_orig(const scal_handle_t scal)
{
    assert(scal);
    return  ((Scal*)scal)->getSRAMSize();
}

uint32_t scal_get_sram_size(const scal_handle_t scal)
{
    return (*scal_funcs->fp_scal_get_sram_size)(scal);
}

static int scal_get_pool_handle_by_name_orig(const scal_handle_t scal, const char *pool_name, scal_pool_handle_t *pool)
{
    if (!scal || !pool_name || !pool)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *pool = (scal_pool_handle_t)(((const Scal*)scal)->getPoolByName(pool_name));
    if (!*pool)
    {
        LOG_ERR(SCAL, "{}: pool {} not found", __FUNCTION__, pool_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_get_pool_handle_by_name(const scal_handle_t scal, const char *pool_name, scal_pool_handle_t *pool)
{
    return (*scal_funcs->fp_scal_get_pool_handle_by_name)(scal, pool_name, pool);
}

static int scal_get_pool_handle_by_id_orig(const scal_handle_t scal, const unsigned pool_id, scal_pool_handle_t *pool)
{
    if (!scal || !pool)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *pool = (scal_pool_handle_t) (((const Scal *) scal)->getPoolByID(pool_id));
    if (!*pool)
    {
        LOG_ERR(SCAL, "{}: pool {} not found", __FUNCTION__, pool_id);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_get_pool_handle_by_id(const scal_handle_t scal, const unsigned pool_id, scal_pool_handle_t *pool)
{
    return (*scal_funcs->fp_scal_get_pool_handle_by_id)(scal, pool_id, pool);
}

static int scal_pool_get_info_orig(const scal_pool_handle_t pool, scal_memory_pool_info *info)
{
    if (!pool || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::Pool* pPool = (const Scal::Pool *)pool;
    info->name = pPool->name.c_str();
    info->scal = (scal_handle_t)pPool->scal;
    info->idx = pPool->globalIdx;
    info->device_base_address = pPool->deviceBase;
    info->host_base_address = pPool->hostBase;
    info->core_base_address = pPool->coreBase;
    pPool->allocator->getInfo(info->totalSize, info->freeSize);
    return SCAL_SUCCESS;
}

int scal_pool_get_info(const scal_pool_handle_t pool, scal_memory_pool_info *info)
{
    return (*scal_funcs->fp_scal_pool_get_info)(pool, info);
}

static int scal_pool_get_info_origV2(const scal_pool_handle_t pool, scal_memory_pool_infoV2 *info)
{
    if (!pool || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::Pool* pPool = (const Scal::Pool *)pool;
    info->name = pPool->name.c_str();
    info->scal = (scal_handle_t)pPool->scal;
    info->idx = pPool->globalIdx;
    info->device_base_address = pPool->deviceBase;
    info->device_base_allocated_address = pPool->deviceBaseAllocatedAddress;
    info->host_base_address = pPool->hostBase;
    info->core_base_address = pPool->coreBase;
    pPool->allocator->getInfo(info->totalSize, info->freeSize);
    return SCAL_SUCCESS;
}

int scal_pool_get_infoV2(const scal_pool_handle_t pool, scal_memory_pool_infoV2 *info)
{
    return (*scal_funcs->fp_scal_pool_get_infoV2)(pool, info);
}

static int scal_get_core_handle_by_name_orig(const scal_handle_t scal, const char *core_name, scal_core_handle_t *core)
{
    if (!scal || !core_name || !core)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *core = (scal_core_handle_t)(((const Scal*)scal)->getCoreByName(core_name));
    if (!*core)
    {
        LOG_ERR(SCAL, "{}: core {} not found", __FUNCTION__, core_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_get_core_handle_by_name(const scal_handle_t scal, const char *core_name, scal_core_handle_t *core)
{
    return (*scal_funcs->fp_scal_get_core_handle_by_name)(scal, core_name, core);
}

static int scal_get_core_handle_by_id_orig(const scal_handle_t scal, const unsigned core_id, scal_core_handle_t *core)
{
    if (!scal || !core)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *core = (scal_core_handle_t) (((const Scal *) scal)->getCoreByID(core_id));
    if (!*core)
    {
        LOG_WARN(SCAL, "{}: core {} not found", __FUNCTION__, core_id);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_get_core_handle_by_id(const scal_handle_t scal, const unsigned core_id, scal_core_handle_t *core)
{
    return (*scal_funcs->fp_scal_get_core_handle_by_id)(scal, core_id, core);
}

static int scal_control_core_get_info_orig(const scal_core_handle_t core, scal_control_core_info_t *info)
{
    if (!core || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::Core*      pCore      = (const Scal::Core*)core;
    const Scal::ArcCore*   pArcCore   = pCore->getAs<Scal::ArcCore>();
    const Scal::Scheduler* pScheduler = pCore->getAs<Scal::Scheduler>();

    info->name                       = (pScheduler || pArcCore == nullptr) ? pCore->name.c_str() : pArcCore->arcName.c_str();
    info->scal                       = (scal_handle_t)pCore->scal;
    info->idx                        = pCore->cpuId;
    info->dccm_message_queue_address = pArcCore ? pArcCore->dccmMessageQueueDevAddress : 0;

    return SCAL_SUCCESS;
}

static int scal_control_core_get_infoV2_orig(const scal_core_handle_t core, scal_control_core_infoV2_t *info)
{
    if (!core || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param ({})", __FUNCTION__, (info != nullptr) ? "core" : "info");
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::Core*      pCore      = (const Scal::Core*)core;
    const Scal::ArcCore*   pArcCore   = pCore->getAs<Scal::ArcCore>();
    const Scal::Scheduler* pScheduler = pCore->getAs<Scal::Scheduler>();

    info->name                       = (pScheduler || pArcCore == nullptr) ? pCore->name.c_str() : pArcCore->arcName.c_str();
    info->scal                       = (scal_handle_t)pCore->scal;
    info->idx                        = pCore->cpuId;
    info->dccm_message_queue_address = pArcCore ? pArcCore->dccmMessageQueueDevAddress : 0;
    info->hdCore                     = pCore->getHdCoreIndex();

    return SCAL_SUCCESS;
}

int scal_control_core_get_debug_info(const scal_core_handle_t core, uint32_t *arcRegs,
                                     uint32_t arcRegsSize, scal_control_core_debug_info_t *info)
{
    return (*scal_funcs->fp_scal_control_core_get_debug_info)(core, arcRegs, arcRegsSize, info);
}

int scal_control_core_get_debug_info_orig(const scal_core_handle_t core, uint32_t *arcRegs,
                                          uint32_t arcRegsSize, scal_control_core_debug_info_t *info)
{
    if (arcRegs != nullptr)
    {
        memset(arcRegs, -1, arcRegsSize);
    }

    const Scal::Core* pCore = (const Scal::Core*)core;
    const Scal::ArcCore * pArcCore = pCore->getAs<Scal::ArcCore>();

    uint64_t addr;
    uint64_t regsSize;
    uint32_t heartbeatVal = 1;
    auto     devInfo      = pCore->scal->m_devSpecificInfo.get();

    if (pCore->isScheduler)
    {
        addr     = devInfo->getHeartBeatOffsetInSchedRegs() + (uint64_t)(pArcCore->dccmHostAddress);
        regsSize = std::min(arcRegsSize, devInfo->getSizeOfschedRegs());
    }
    else
    {
        if (pArcCore->name.find("cme") == 0)
        {
            addr     = devInfo->getHeartBeatOffsetInCmeRegs() + (uint64_t)(pArcCore->dccmHostAddress);
            regsSize = std::min(arcRegsSize, devInfo->getSizeOfEngRegs());
        }
        else
        {
            addr     = devInfo->getHeartBeatOffsetInEngRegs() + (uint64_t)(pArcCore->dccmHostAddress);
            regsSize = std::min(arcRegsSize, devInfo->getSizeOfEngRegs());
        }
    }

    readLbwMem( (void*)&heartbeatVal, (volatile void*) addr, sizeof(uint32_t));

    info->heartBeat   = heartbeatVal;
    info->isScheduler = pCore->isScheduler;

    if (arcRegs == nullptr)
    {
        info->returnedSize = 0;
        return SCAL_SUCCESS;
    }

    regsSize = regsSize / sizeof(uint32_t) * sizeof(uint32_t); // multiple of 4 bytes

    info->returnedSize = regsSize;
    readLbwMem((void*)arcRegs, (volatile void*)(pArcCore->dccmHostAddress), regsSize);

    return SCAL_SUCCESS;
}

int scal_control_core_get_info(const scal_core_handle_t core, scal_control_core_info_t *info)
{
    return (*scal_funcs->fp_scal_control_core_get_info)(core, info);
}

int scal_control_core_get_infoV2(const scal_core_handle_t core, scal_control_core_infoV2_t *info)
{
    return (*scal_funcs->fp_scal_control_core_get_infoV2)(core, info);
}

static int scal_get_stream_handle_by_name_orig(const scal_handle_t scal, const char * stream_name, scal_stream_handle_t *stream)
{
    if (!scal || !stream_name || !stream)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *stream = (scal_stream_handle_t)(((const Scal*)scal)->getStreamByName(stream_name));
    if (!*stream)
    {
        LOG_ERR(SCAL, "{}: stream {} not found", __FUNCTION__, stream_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;

}

int scal_get_stream_handle_by_name(const scal_handle_t scal, const char * stream_name, scal_stream_handle_t *stream)
{
    return (*scal_funcs->fp_scal_get_stream_handle_by_name)(scal, stream_name, stream);
}

static int scal_get_used_sm_base_addrs_orig(const scal_handle_t scal, unsigned * num_addrs, const scal_sm_base_addr_tuple_t ** sm_base_addr_db)
{
    if (!scal || !num_addrs || !sm_base_addr_db)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *num_addrs = (unsigned)(((Scal*)scal)->getUsedSmBaseAddrs(sm_base_addr_db));
    if (!*sm_base_addr_db || !*num_addrs)
    {
        LOG_ERR(SCAL, "{}: Failed to get SM base addresses DB, sm_base_addr_db {:x}, num_addrs {}",
                __FUNCTION__,
                TO64(*sm_base_addr_db),
                *num_addrs);

        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int scal_get_used_sm_base_addrs(const scal_handle_t scal, unsigned * num_addrs, const scal_sm_base_addr_tuple_t ** sm_base_addr_db)
{
    return (*scal_funcs->fp_scal_get_used_sm_base_addrs)(scal, num_addrs, sm_base_addr_db);
}

static int scal_get_stream_handle_by_index_orig(const scal_core_handle_t scheduler, const unsigned index, scal_stream_handle_t *stream)
{
    const Scal::Core * core = (const Scal::Core*)scheduler;
    auto pScheduler = core ? core->getAs<Scal::Scheduler>() : nullptr;
    if (!pScheduler || !stream)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    *stream = (scal_stream_handle_t) (core->scal->getStreamByID(pScheduler, index));
    if (!*stream)
    {
        LOG_ERR(SCAL, "{}: stream {} not found", __FUNCTION__, index);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_get_stream_handle_by_index(const scal_core_handle_t scheduler, const unsigned index, scal_stream_handle_t *stream)
{
    return (*scal_funcs->fp_scal_get_stream_handle_by_index)(scheduler, index, stream);
}

static int scal_get_so_pool_handle_by_name_orig(const scal_handle_t scal, const char *pool_name, scal_so_pool_handle_t *so_pool)
{
    if (!scal || !pool_name || !so_pool)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    *so_pool = (scal_so_pool_handle_t)(((const Scal*)scal)->getSoPool(pool_name));
    if (!*so_pool)
    {
        LOG_ERR(SCAL, "{}: so_pool {} not found", __FUNCTION__, pool_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;

}

int scal_get_so_pool_handle_by_name(const scal_handle_t scal, const char *pool_name, scal_so_pool_handle_t *so_pool)
{
    return (*scal_funcs->fp_scal_get_so_pool_handle_by_name)(scal, pool_name, so_pool);
}

static int scal_so_pool_get_info_orig(const scal_so_pool_handle_t so_pool, scal_so_pool_info *info)
{
    if (!so_pool || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::SyncObjectsPool & soPool = *((const Scal::SyncObjectsPool *)so_pool);
    info->name =  soPool.name.c_str();
    info->scal =  (scal_handle_t) soPool.scal;
    info->size =  soPool.size;
    info->smIndex = soPool.smIndex;
    info->baseIdx = soPool.baseIdx;
    info->smBaseAddr = soPool.smBaseAddr;
    info->dcoreIndex = soPool.dcoreIndex; //TODO: remove
    return SCAL_SUCCESS;

}

int scal_so_pool_get_info(const scal_so_pool_handle_t so_pool, scal_so_pool_info *info)
{
    return (*scal_funcs->fp_scal_so_pool_get_info)(so_pool, info);
}

static int scal_get_so_monitor_handle_by_name_orig(const scal_handle_t scal, const char *pool_name, scal_monitor_pool_handle_t *monitor_pool)
{
    if (!scal || !pool_name || !monitor_pool)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    *monitor_pool = (scal_monitor_pool_handle_t)(((const Scal*)scal)->getMonitorPool(pool_name));
    if (!*monitor_pool)
    {
        return SCAL_NOT_FOUND;
    }
    return SCAL_SUCCESS;
}

int scal_get_so_monitor_handle_by_name(const scal_handle_t scal, const char *pool_name, scal_monitor_pool_handle_t *monitor_pool)
{
    return (*scal_funcs->fp_scal_get_so_monitor_handle_by_name)(scal, pool_name, monitor_pool);
}

static int scal_monitor_pool_get_info_orig(const scal_monitor_pool_handle_t mon_pool, scal_monitor_pool_info *info)
{
    if (!mon_pool || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    info->name =  ((const Scal::MonitorsPool *)mon_pool)->name.c_str();
    info->baseIdx = ((const Scal::MonitorsPool *)mon_pool)->baseIdx;
    info->smIndex = ((const Scal::MonitorsPool *)mon_pool)->smIndex;
    info->scal = (scal_handle_t) ((const Scal::MonitorsPool *)mon_pool)->scal;
    info->size = ((const Scal::MonitorsPool *)mon_pool)->size;
    info->dcoreIndex = ((const Scal::MonitorsPool *)mon_pool)->dcoreIndex; // TODO: remove

    return SCAL_SUCCESS;
}

int scal_monitor_pool_get_info(const scal_monitor_pool_handle_t mon_pool, scal_monitor_pool_info *info)
{
    return (*scal_funcs->fp_scal_monitor_pool_get_info)(mon_pool, info);
}


static int scal_get_sm_info_orig(const scal_handle_t scal, unsigned sm_idx, scal_sm_info_t *info)
{
    if (!scal)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    const Scal::SyncManager* sm = (((const Scal*)scal)->getSyncManager(sm_idx));
    if (!sm)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    info->idx = sm_idx;

    if (sm->smIndex != sm_idx)
    {
        info->objs = nullptr;
        info->glbl = nullptr;
    }
    else
    {
        info->objs = sm->objsHostAddress;
        info->glbl = sm->glblHostAddress;
    }

    return SCAL_SUCCESS;
}


int scal_get_sm_info(const scal_handle_t scal, unsigned sm_idx, scal_sm_info_t *info)
{
    return (*scal_funcs->fp_scal_get_sm_info)(scal, sm_idx, info);
}

static int scal_stream_set_commands_buffer_orig(const scal_stream_handle_t stream, const scal_buffer_handle_t buff)
{
    if (!buff || !stream)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    int ret = Scal::streamSetBuffer((Scal::StreamInterface*)stream, (Scal::Buffer*)buff);
    return ret;
}

int scal_stream_set_commands_buffer(const scal_stream_handle_t stream, const scal_buffer_handle_t buff)
{
    return (*scal_funcs->fp_scal_stream_set_commands_buffer)(stream, buff);
}

static int scal_stream_set_priority_orig(const scal_stream_handle_t stream, const unsigned priority)
{
    if (!stream)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    int ret = Scal::streamSetPriority((Scal::StreamInterface*)stream, priority);
    return ret;
}

int scal_stream_set_priority(const scal_stream_handle_t stream, const unsigned priority)
{
    return (*scal_funcs->fp_scal_stream_set_priority)(stream, priority);
}

static int scal_stream_submit_orig(const scal_stream_handle_t stream, const unsigned pi, const unsigned submission_alignment)
{
    if (!stream)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return ((Scal::StreamInterface*)stream)->submit(pi, submission_alignment);;
}

int scal_stream_submit(const scal_stream_handle_t stream, const unsigned pi, const unsigned submission_alignment)
{
    return (*scal_funcs->fp_scal_stream_submit)(stream, pi, submission_alignment);
}

static int scal_stream_get_info_orig(const scal_stream_handle_t stream, scal_stream_info_t *info)
{
    if (!stream || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return ((const Scal::StreamInterface*)stream)->getInfo(*info);
}

int scal_stream_get_info(const scal_stream_handle_t stream, scal_stream_info_t *info)
{
    return (*scal_funcs->fp_scal_stream_get_info)(stream, info);
}

int scal_stream_get_commands_buffer_alignment(const scal_stream_handle_t stream, unsigned* ccb_buffer_alignment)
{
    if (!stream || !ccb_buffer_alignment)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return ((const Scal::StreamInterface*)stream)->getCcbBufferAlignment(*ccb_buffer_alignment);
}

static int scal_get_completion_group_handle_by_name_orig(const scal_handle_t scal, const char * cg_name, scal_comp_group_handle_t *comp_grp)
{
    if (!scal || !cg_name || !comp_grp)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *comp_grp = (scal_comp_group_handle_t)((const Scal*)scal)->getCompletionGroupByName(cg_name);
    if (!*comp_grp)
    {
        LOG_ERR(SCAL, "{}: completion group {} not found", __FUNCTION__, cg_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_get_host_fence_counter_handle_by_name(const scal_handle_t scal, const char * host_fence_counter_name, scal_host_fence_counter_handle_t *host_fence_counter)
{
    return (*scal_funcs->fp_scal_get_host_fence_counter_handle_by_name)(scal, host_fence_counter_name, host_fence_counter);
}

int scal_get_host_fence_counter_handle_by_name_orig(const scal_handle_t scal, const char * host_fence_counter_name, scal_host_fence_counter_handle_t *host_fence_counter)
{
    if (!scal || !host_fence_counter_name || !host_fence_counter)
     {
         LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
         assert(0);
         return SCAL_INVALID_PARAM;
     }

     *host_fence_counter = (scal_host_fence_counter_handle_t)((Scal*)scal)->getHostFenceCounter(host_fence_counter_name);
     if (!*host_fence_counter)
     {
         LOG_WARN(SCAL, "{}: host_fence counter {} not found", __FUNCTION__, host_fence_counter_name);
         return SCAL_NOT_FOUND;
     }

     return SCAL_SUCCESS;
}
int scal_host_fence_counter_get_info(scal_host_fence_counter_handle_t host_fence_counter, scal_host_fence_counter_info_t *info)
{
    return (*scal_funcs->fp_scal_host_fence_counter_get_info)(host_fence_counter, info);
}

int scal_host_fence_counter_get_info_orig(scal_host_fence_counter_handle_t host_fence_counter, scal_host_fence_counter_info_t *info)
{
    if (!host_fence_counter || !info)
     {
         LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
         assert(0);
         return SCAL_INVALID_PARAM;
     }

     const Scal::HostFenceCounter* pHostFenceCounter = (const Scal::HostFenceCounter *)host_fence_counter;
     int ret = Scal::getHostFenceCounterInfo(pHostFenceCounter, info);
     return ret;
}


int scal_host_fence_counter_wait(const scal_host_fence_counter_handle_t host_fence_counter, const uint64_t num_credits, const uint64_t timeoutUs)
{
    return (*scal_funcs->fp_scal_host_fence_counter_wait)(host_fence_counter, num_credits, timeoutUs);
}

int scal_host_fence_counter_wait_orig(const scal_host_fence_counter_handle_t host_fence_counter, const uint64_t num_credits, const uint64_t timeoutUs)
{
    if (!host_fence_counter)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::HostFenceCounter* pHostFenceCounter = (const Scal::HostFenceCounter *)host_fence_counter;
    if (pHostFenceCounter->scal->scalStub || pHostFenceCounter->isStub)
    {
        return SCAL_SUCCESS;
    }
    int ret = Scal::hostFenceCounterWait(pHostFenceCounter, num_credits, timeoutUs);
    return ret;
}

int scal_host_fence_counter_enable_isr(const scal_host_fence_counter_handle_t host_fence_counter, bool enable_isr)
{
    return (*scal_funcs->fp_scal_host_fence_counter_enable_isr)(host_fence_counter, enable_isr);
}

int scal_host_fence_counter_enable_isr_orig(const scal_host_fence_counter_handle_t host_fence_counter, bool enable_isr)
{
    if (!host_fence_counter)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    Scal::HostFenceCounter* pHostFenceCounter = (Scal::HostFenceCounter *)host_fence_counter;
    if (pHostFenceCounter->scal->scalStub || pHostFenceCounter->isStub)
    {
        return SCAL_SUCCESS;
    }
    int ret = Scal::hostFenceCounterEnableIsr(pHostFenceCounter, enable_isr);
    return ret;
}

int scal_get_completion_group_handle_by_name(const scal_handle_t scal, const char * cg_name, scal_comp_group_handle_t *comp_grp)
{
    return (*scal_funcs->fp_scal_get_completion_group_handle_by_name)(scal, cg_name, comp_grp);
}

static int scal_get_completion_group_handle_by_index_orig(const scal_core_handle_t scheduler, const unsigned index, scal_comp_group_handle_t *comp_grp)
{
    const Scal::Core * pCore = (const Scal::Core*)scheduler;
    const Scal::Scheduler* pScheduler = pCore ? pCore->getAs<Scal::Scheduler>() : nullptr;
    if (!pScheduler || !comp_grp)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    if (index >= pScheduler->completionGroups.size() || !pScheduler->completionGroups[index])
    {
        LOG_ERR(SCAL, "{}: completion group {} not found", __FUNCTION__, index);
        return SCAL_NOT_FOUND;
    }

    *comp_grp = (scal_comp_group_handle_t)pScheduler->completionGroups[index];

    return SCAL_SUCCESS;

}

int scal_get_completion_group_handle_by_index(const scal_core_handle_t scheduler, const unsigned index, scal_comp_group_handle_t *comp_grp)
{
    return (*scal_funcs->fp_scal_get_completion_group_handle_by_index)(scheduler, index, comp_grp);
}

static int scal_completion_group_wait_orig(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout)
{
    if (!comp_grp)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::CompletionGroupInterface* pComp_grp = (const Scal::CompletionGroupInterface*)comp_grp;
    if (pComp_grp->isStub())
    {
        return SCAL_SUCCESS;
    }
    Scal* scal = (Scal*)pComp_grp->scal;
    int ret = scal->completionGroupWait(pComp_grp, target, timeout, false);
    return ret;
}

int scal_completion_group_wait(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout)
{
    return (*scal_funcs->fp_scal_completion_group_wait)(comp_grp, target, timeout);
}

static int scal_completion_group_wait_always_interupt_orig(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout)
{
    if (!comp_grp)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::CompletionGroupInterface* pComp_grp = (const Scal::CompletionGroupInterface*)comp_grp;
    if (pComp_grp->isStub())
    {
        return SCAL_SUCCESS;
    }
    Scal* scal = (Scal*)pComp_grp->scal;
    int ret = scal->completionGroupWait(pComp_grp, target, timeout, true);
    return ret;
}

int scal_completion_group_wait_always_interupt(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout)
{
    return (*scal_funcs->fp_scal_completion_group_wait_always_interupt)(comp_grp, target, timeout);
}

static int scal_completion_group_register_timestamp_orig(const scal_comp_group_handle_t comp_grp, const uint64_t target, uint64_t timestamps_handle, uint32_t timestamps_offset)
{
    if (!comp_grp)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    const Scal::CompletionGroupInterface* pComp_grp = (const Scal::CompletionGroupInterface*)comp_grp;
    if (pComp_grp->isStub())
    {
        return SCAL_SUCCESS;
    }

    return Scal::completionGroupRegisterTimestamp(pComp_grp, target, timestamps_handle, timestamps_offset);
}

int scal_completion_group_register_timestamp(const scal_comp_group_handle_t comp_grp, const uint64_t target, uint64_t timestamps_handle, uint32_t timestamps_offset)
{
    return (*scal_funcs->fp_scal_completion_group_register_timestamp)(comp_grp, target, timestamps_handle, timestamps_offset);
}

static int scal_completion_group_get_info_orig(const scal_comp_group_handle_t comp_grp, scal_completion_group_info_t *info)
{
    if (!comp_grp || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    const Scal::CompletionGroupInterface* cq = (const Scal::CompletionGroupInterface*)comp_grp;
    if(!cq->getInfo(*info))
    {
        LOG_ERR(SCAL, "{}: Failed to get info", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return SCAL_SUCCESS;
}

int scal_completion_group_get_infoV2(const scal_comp_group_handle_t comp_grp, scal_completion_group_infoV2_t *info)
{
    return (*scal_funcs->fp_scal_completion_group_get_infoV2)(comp_grp, info);
}

int scal_completion_group_get_infoV2_orig(const scal_comp_group_handle_t comp_grp, scal_completion_group_infoV2_t *info)
{
    if (!comp_grp || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    const Scal::CompletionGroupInterface* cq = (const Scal::CompletionGroupInterface*)comp_grp;
    if(!cq->getInfo(*info))
    {
        LOG_ERR(SCAL, "{}: Failed to get info", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return SCAL_SUCCESS;
}

int scal_completion_group_get_info(const scal_comp_group_handle_t comp_grp, scal_completion_group_info_t *info)
{
    return (*scal_funcs->fp_scal_completion_group_get_info)(comp_grp, info);
}

int scal_completion_group_set_expected_ctr(scal_comp_group_handle_t comp_grp, uint64_t val)
{
    return (*scal_funcs->fp_scal_completion_group_set_expected_ctr)(comp_grp, val);
}

int scal_completion_group_set_expected_ctr_orig(scal_comp_group_handle_t comp_grp, uint64_t val)
{
    if (!comp_grp)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        return SCAL_INVALID_PARAM;
    }

    Scal::CompletionGroupInterface* cq = (Scal::CompletionGroupInterface*)comp_grp;

    cq->compQTdr.expectedCqCtr = val;

    return SCAL_SUCCESS;
}

static int scal_allocate_buffer_orig(const scal_pool_handle_t pool, const uint64_t size, scal_buffer_handle_t *buff)
{
    return scal_allocate_aligned_buffer(pool, size, 128, buff);
}

int scal_allocate_buffer(const scal_pool_handle_t pool, const uint64_t size, scal_buffer_handle_t *buff)
{
    return (*scal_funcs->fp_scal_allocate_buffer)(pool, size, buff);
}

static int scal_allocate_aligned_buffer_orig(const scal_pool_handle_t pool, const uint64_t size, const uint64_t alignment, scal_buffer_handle_t *buff)
{
    if (!pool || !size || !buff || !alignment)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    Scal::Pool* scalPool = (Scal::Pool*)pool;
    uint64_t base = scalPool->allocator->alloc(size, alignment);
    if (base == Scal::Allocator::c_bad_alloc)
    {
        LOG_ERR(SCAL, "{}: out of memory while allocating {} with alignment {}", __FUNCTION__, size, alignment);
        uint64_t totalSize;
        uint64_t freeSize;
        scalPool->allocator->getInfo(totalSize, freeSize);
        LOG_ERR(SCAL, "{}: failed to allocate {} with alignment {} in pool {} allocator total size {} free size {}", __FUNCTION__, size, alignment, scalPool->name, totalSize, freeSize);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    Scal::Buffer *b = scalPool->scal->createAllocatedBuffer();
    b->pool = ((Scal::Pool*)pool);
    b->base = base;
    b->size = size;

    *buff = (scal_buffer_handle_t)b;

    return SCAL_SUCCESS;
}

int scal_allocate_aligned_buffer(const scal_pool_handle_t pool, const uint64_t size, const uint64_t alignment, scal_buffer_handle_t *buff)
{
    return (*scal_funcs->fp_scal_allocate_aligned_buffer)(pool, size, alignment, buff);
}

static int scal_free_buffer_orig(const scal_buffer_handle_t buff)
{
    if (!buff)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    Scal::Buffer * scalBuff = (Scal::Buffer*)buff;
    scalBuff->pool->allocator->free(scalBuff->base);
    scalBuff->pool->scal->deleteAllocatedBuffer(scalBuff);

    return SCAL_SUCCESS;
}

int scal_free_buffer(const scal_buffer_handle_t buff)
{
    return (*scal_funcs->fp_scal_free_buffer)(buff);
}

static int scal_buffer_get_info_orig(const scal_buffer_handle_t buff, scal_buffer_info_t *info)
{
    if (!buff || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    Scal::Buffer * scalBuff = (Scal::Buffer*)buff;
    info->pool           = (scal_pool_handle_t)scalBuff->pool;
    info->core_address   = scalBuff->pool->coreBase + scalBuff->base;
    info->device_address = scalBuff->pool->deviceBase + scalBuff->base;
    info->host_address   = scalBuff->pool->hostBase ? (char*)scalBuff->pool->hostBase + scalBuff->base : nullptr;

    return SCAL_SUCCESS;
}

int scal_buffer_get_info(const scal_buffer_handle_t buff, scal_buffer_info_t *info)
{
    return (*scal_funcs->fp_scal_buffer_get_info)(buff, info);
}

int scal_set_timeouts(const scal_handle_t scal, const scal_timeouts_t * timeouts)
{
    if (!scal || !timeouts)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    return ((Scal*)scal)->setTimeouts(timeouts);
}

int scal_get_timeouts(const scal_handle_t scal, scal_timeouts_t * timeouts)
{
    if (!scal || !timeouts)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    return ((Scal*)scal)->getTimeouts(timeouts);
}

int scal_disable_timeouts(const scal_handle_t scal, bool disableTimeouts)
{
    if (!scal)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    return ((Scal*)scal)->disableTimeouts(disableTimeouts);
}

static uint32_t scal_debug_read_reg_orig(const scal_handle_t scal, uint64_t reg_address)
{
    // TODO
    assert(0); // not implemented yet.
    return SCAL_NOT_IMPLEMENTED;
}

uint32_t scal_debug_read_reg(const scal_handle_t scal, uint64_t reg_address)
{
    return (*scal_funcs->fp_scal_debug_read_reg)(scal, reg_address);
}

static int scal_debug_write_reg_orig(const scal_handle_t scal, uint64_t reg_address, uint32_t reg_value)
{
    // TODO
    assert(0); // not implemented yet.
    return SCAL_NOT_IMPLEMENTED;
}

int scal_debug_write_reg(const scal_handle_t scal, uint64_t reg_address, uint32_t reg_value)
{
    return (*scal_funcs->fp_scal_debug_write_reg)(scal, reg_address, reg_value);
}

static int scal_debug_memcpy_orig(const scal_handle_t scal, uint64_t src, uint64_t dst, uint64_t size)
{
    // TODO
    assert(0); // not implemented yet.
    return SCAL_NOT_IMPLEMENTED;
}

int scal_debug_memcpy(const scal_handle_t scal, uint64_t src, uint64_t dst, uint64_t size)
{
    return (*scal_funcs->fp_scal_debug_memcpy)(scal, src, dst, size);
}

static unsigned scal_debug_stream_get_curr_ci_orig(const scal_stream_handle_t stream)
{
    // TODO
    assert(0); // not implemented yet.
    return SCAL_NOT_IMPLEMENTED;
}

unsigned scal_debug_stream_get_curr_ci(const scal_stream_handle_t stream)
{
    return (*scal_funcs->fp_scal_debug_stream_get_curr_ci)(stream);
}

static int scal_get_cluster_handle_by_name_orig(const scal_handle_t scal, const char *cluster_name, scal_cluster_handle_t *cluster)
{
    if (!scal || !cluster_name || !cluster)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *cluster = (scal_cluster_handle_t)((Scal*)scal)->getClusterByName(cluster_name);
    if (!*cluster)
    {
        LOG_WARN(SCAL, "{}: cluster {} not found", __FUNCTION__, cluster_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;

}

int scal_get_cluster_handle_by_name(const scal_handle_t scal, const char *cluster_name, scal_cluster_handle_t *cluster)
{
    return (*scal_funcs->fp_scal_get_cluster_handle_by_name)(scal, cluster_name, cluster);
}

static int scal_cluster_get_info_orig(const scal_cluster_handle_t cluster, scal_cluster_info_t *info)
{
    if (!cluster || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::Cluster* pCluster = (const Scal::Cluster *)cluster;
    info->name = pCluster->name.c_str();
    info->numEngines = pCluster->engines.size();
    if (pCluster->type == NUM_OF_CORE_TYPES)// dummy cluster
    {
        info->numCompletions = 1;
    }
    else
    {
        assert(info->numEngines > 0);
        info->numCompletions = info->numEngines;
    }
    if (pCluster->type == MME)
    {
        Scal* pScal = pCluster->engines[0]->scal;
        info->numCompletions *= pScal->getNumberOfSignalsPerMme();
    }
    memset(info->engines,0,sizeof(info->engines));
    unsigned index = 0;
    for (auto engine: pCluster->engines)
    {
        info->engines[index++] = Scal::toCoreHandle(engine);
    }

    return SCAL_SUCCESS;
}

int scal_cluster_get_info(const scal_cluster_handle_t cluster, scal_cluster_info_t *info)
{
    return (*scal_funcs->fp_scal_cluster_get_info)(cluster, info);
}

int scal_get_streamset_handle_by_name(const scal_handle_t scal, const char *streamset_name, scal_streamset_handle_t *streamset)
{
    return(*scal_funcs->fp_scal_get_streamset_handle_by_name)(scal, streamset_name, streamset);
}

static int scal_get_streamset_handle_by_name_orig(const scal_handle_t scal, const char *streamset_name, scal_streamset_handle_t *streamset)
{
    if (!scal || !streamset_name || !streamset)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    *streamset = (scal_streamset_handle_t)((Scal*)scal)->getStreamSetByName(streamset_name);
    if (!*streamset)
    {
        LOG_WARN(SCAL, "{}: stream set {} not found", __FUNCTION__, streamset_name);
        return SCAL_NOT_FOUND;
    }

    return SCAL_SUCCESS;
}

int scal_streamset_get_info(const scal_streamset_handle_t streamset_handle, scal_streamset_info_t* info)
{
    return(*scal_funcs->fp_scal_streamset_get_info)(streamset_handle, info);
}

static int scal_streamset_get_info_orig(const scal_streamset_handle_t streamset_handle, scal_streamset_info_t* info)
{
    if (!streamset_handle || !info)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    const Scal::StreamSet* pStreamSet = (const Scal::StreamSet *)streamset_handle;
    info->name = pStreamSet->name.c_str();
    info->isDirectMode  = pStreamSet->isDirectMode;
    info->streamsAmount = pStreamSet->streamsAmount;

    return SCAL_SUCCESS;
}

int scal_bg_work(const scal_handle_t scal, void (*logFunc)(int, const char*))
{
    return (*scal_funcs->fp_scal_bg_work)(scal, logFunc);
}

int scal_bg_workV2(const scal_handle_t scal, void (*logFunc)(int, const char*), char *errMsg, int errMsgSize)
{
    return (*scal_funcs->fp_scal_bg_workV2)(scal, logFunc, errMsg, errMsgSize);
}

static void scal_write_mapped_reg_orig(volatile uint32_t * pointer, uint32_t value)
{
    writeLbwReg(pointer, value);
}

void scal_write_mapped_reg(volatile uint32_t * pointer, uint32_t value)
{
    (*scal_funcs->fp_scal_write_mapped_reg)(pointer, value);
}

static uint32_t scal_read_mapped_reg_orig(volatile uint32_t * pointer)
{
    uint32_t ret;
    readLbwMem(&ret, pointer, sizeof(uint32_t));
    return ret;
}

uint32_t scal_read_mapped_reg(volatile uint32_t * pointer)
{
    return (*scal_funcs->fp_scal_read_mapped_reg)(pointer);
}

int scal_bg_work_orig(const scal_handle_t scal, void (*logFunc)(int, const char*))
{
    return ((Scal*)scal)->bgWork(logFunc, nullptr, 0);
}

int scal_bg_work_origV2(const scal_handle_t scal, void (*logFunc)(int, const char*), char *msg, int msgSize)
{
    return ((Scal*)scal)->bgWork(logFunc, msg, msgSize);
}

static int scal_nics_db_fifos_init_and_alloc_orig(const scal_handle_t scal, ibv_context* ibv_ctxt)
{
    if (!scal)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    auto pScal = (Scal*)scal;
    return pScal->allocAndSetupPortDBFifo(ibv_ctxt);
}

int scal_nics_db_fifos_init_and_alloc(const scal_handle_t scal, ibv_context* ibv_ctxt)
{
    return scal_nics_db_fifos_init_and_alloc_orig(scal, ibv_ctxt);
}

int scal_nics_db_fifos_init_and_allocV2_orig(const scal_handle_t             scal,
                                             const scal_ibverbs_init_params* ibvInitParams,
                                             struct hlibdv_usr_fifo       ** createdFifoBuffers,
                                             uint32_t                      * createdFifoBuffersCount)
{
    bool wrongIbvVerbsParam =  ibvInitParams == nullptr ||
                               ibvInitParams->ibv_ctxt == nullptr;
    if (!scal ||
        !createdFifoBuffersCount ||
        (*createdFifoBuffersCount != 0 && (createdFifoBuffers == nullptr || wrongIbvVerbsParam)))
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    auto pScal = (Scal*)scal;
    return pScal->allocAndSetupPortDBFifoV2(ibvInitParams, createdFifoBuffers, createdFifoBuffersCount);
}

int scal_nics_db_fifos_init_and_allocV2(const scal_handle_t             scal,
                                        const scal_ibverbs_init_params* ibvInitParams,
                                        struct hlibdv_usr_fifo       ** createdFifoBuffers,
                                        uint32_t                      * createdFifoBuffersCount)

{
    return scal_nics_db_fifos_init_and_allocV2_orig(scal, ibvInitParams, createdFifoBuffers, createdFifoBuffersCount);
}

int scal_debug_background_work(const scal_handle_t scal)
{
    return (*scal_funcs->fp_scal_debug_background_work)(scal);
}

int scal_debug_background_work_orig(const scal_handle_t scal)
{
    if (scal == nullptr)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return ((Scal*)scal)->debugBackgroundWork();
}

static int scal_get_nics_db_fifos_params_orig_tmp(const scal_handle_t scal, struct hlibdv_usr_fifo_attr_tmp* nicUserDbFifoParams, unsigned* nicUserDbFifoParamsCount)
{
    if (!scal || !nicUserDbFifoParamsCount)
    {
        LOG_ERR(SCAL, "{}: invalid param", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    auto pScal = (Scal*)scal;
    return pScal->getDbFifoParams_tmp(nicUserDbFifoParams, nicUserDbFifoParamsCount);
}

int scal_get_nics_db_fifos_params_tmp(const scal_handle_t scal, struct hlibdv_usr_fifo_attr_tmp* nicUserDbFifoParams, unsigned* nicUserDbFifoParamsCount)
{
    return scal_get_nics_db_fifos_params_orig_tmp(scal, nicUserDbFifoParams, nicUserDbFifoParamsCount);
}

namespace{
const scal_func_table  default_scal_funcs = {
    .fp_scal_init = scal_init_orig,
    .fp_scal_destroy = scal_destroy_orig,
    .fp_scal_get_fd = scal_get_fd_orig,
    .fp_scal_get_handle_from_fd = scal_get_handle_from_fd_orig,
    .fp_scal_get_sram_size = scal_get_sram_size_orig,
    .fp_scal_get_pool_handle_by_name = scal_get_pool_handle_by_name_orig,
    .fp_scal_get_pool_handle_by_id = scal_get_pool_handle_by_id_orig,
    .fp_scal_pool_get_info = scal_pool_get_info_orig,
    .fp_scal_get_core_handle_by_name = scal_get_core_handle_by_name_orig,
    .fp_scal_get_core_handle_by_id = scal_get_core_handle_by_id_orig,
    .fp_scal_control_core_get_info = scal_control_core_get_info_orig,
    .fp_scal_get_stream_handle_by_name = scal_get_stream_handle_by_name_orig,
    .fp_scal_get_stream_handle_by_index = scal_get_stream_handle_by_index_orig,
    .fp_scal_stream_set_commands_buffer = scal_stream_set_commands_buffer_orig,
    .fp_scal_stream_set_priority = scal_stream_set_priority_orig,
    .fp_scal_stream_submit = scal_stream_submit_orig,
    .fp_scal_stream_get_info = scal_stream_get_info_orig,
    .fp_scal_get_completion_group_handle_by_name = scal_get_completion_group_handle_by_name_orig,
    .fp_scal_get_completion_group_handle_by_index = scal_get_completion_group_handle_by_index_orig,
    .fp_scal_completion_group_wait = scal_completion_group_wait_orig,
    .fp_scal_completion_group_wait_always_interupt = scal_completion_group_wait_always_interupt_orig,
    .fp_scal_completion_group_register_timestamp = scal_completion_group_register_timestamp_orig,
    .fp_scal_completion_group_get_info = scal_completion_group_get_info_orig,
    .fp_scal_get_so_pool_handle_by_name = scal_get_so_pool_handle_by_name_orig,
    .fp_scal_so_pool_get_info = scal_so_pool_get_info_orig,
    .fp_scal_get_so_monitor_handle_by_name = scal_get_so_monitor_handle_by_name_orig,
    .fp_scal_monitor_pool_get_info = scal_monitor_pool_get_info_orig,
    .fp_scal_allocate_buffer = scal_allocate_buffer_orig,
    .fp_scal_allocate_aligned_buffer = scal_allocate_aligned_buffer_orig,
    .fp_scal_free_buffer = scal_free_buffer_orig,
    .fp_scal_buffer_get_info = scal_buffer_get_info_orig,
    .fp_scal_get_cluster_handle_by_name = scal_get_cluster_handle_by_name_orig,
    .fp_scal_cluster_get_info = scal_cluster_get_info_orig,
    .fp_scal_debug_read_reg = scal_debug_read_reg_orig,
    .fp_scal_debug_write_reg = scal_debug_write_reg_orig,
    .fp_scal_debug_memcpy = scal_debug_memcpy_orig,
    .fp_scal_debug_stream_get_curr_ci = scal_debug_stream_get_curr_ci_orig,
    .fp_scal_control_core_get_debug_info = scal_control_core_get_debug_info_orig,
    .fp_scal_completion_group_get_infoV2 = scal_completion_group_get_infoV2_orig,
    .fp_scal_completion_group_inc_expected_ctr = nullptr,
    .fp_scal_completion_group_set_expected_ctr = scal_completion_group_set_expected_ctr_orig,
    .fp_scal_bg_work = scal_bg_work_orig,
    .fp_scal_get_sm_info = scal_get_sm_info_orig,
    .fp_scal_write_mapped_reg = scal_write_mapped_reg_orig,
    .fp_scal_read_mapped_reg = scal_read_mapped_reg_orig,
    .fp_scal_get_host_fence_counter_handle_by_name = scal_get_host_fence_counter_handle_by_name_orig,
    .fp_scal_host_fence_counter_get_info = scal_host_fence_counter_get_info_orig,
    .fp_scal_host_fence_counter_wait = scal_host_fence_counter_wait_orig,
    .fp_scal_host_fence_counter_enable_isr = scal_host_fence_counter_enable_isr_orig,
    .fp_scal_get_streamset_handle_by_name = scal_get_streamset_handle_by_name_orig,
    .fp_scal_streamset_get_info = scal_streamset_get_info_orig,
    .fp_scal_control_core_get_infoV2 = scal_control_core_get_infoV2_orig,
    .fp_scal_get_used_sm_base_addrs = scal_get_used_sm_base_addrs_orig,
    .fp_scal_debug_background_work = scal_debug_background_work_orig,
    .fp_scal_pool_get_infoV2 = scal_pool_get_info_origV2,
    .fp_scal_bg_workV2 = scal_bg_work_origV2,
    .fp_scal_nics_db_fifos_init_and_allocV2 = scal_nics_db_fifos_init_and_allocV2,
};
}

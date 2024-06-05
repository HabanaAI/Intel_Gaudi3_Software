#include "syn_singleton.hpp"

#include "api_calls_counter.hpp"
#include "api.h"
#include "common/shim_typedefs.h"
#include "defenders.h"
#include "define_synapse_common.hpp"
#include "dfa_defines.hpp"
#include "eager/eager_interface.h"
#include "event_triggered_logger.hpp"
#include "graph_compiler/graph_factory.h"
#include "graph_compiler/passes/tpc_fuser.h"
#include "habana_global_conf_runtime.h"
#include "infra/containers/slot_map.hpp"
#include "infra/global_conf_manager.h"
#include "runtime/common/device/device_common.hpp"
#include "runtime/common/osal/osal.hpp"
#include "runtime/common/queues/basic_queue_info.hpp"
#include "runtime/common/queues/queue_interface.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/streams/stream.hpp"
#include "runtime/qman/common/command_buffer.hpp"
#include "runtime/qman/common/command_submission_builder.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/command_submission.hpp"
#include "runtime/qman/gaudi/device_gaudi.hpp"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "timer.h"
#include "types_exception.h"
#include "utils.h"
#include "version.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// TODO [SW-124904]: move out to central pass init
#include "passes/generate_profiler_debug_info.h"

using namespace std;
using namespace common;

using EventSPtr = SlotMapItemSptr<EventInterface>;

synSingletonInterface* synSingleton::m_pInstance         = nullptr;
synSingleton*          synSingleton::m_pInstanceInternal = nullptr;
libHandle              synSingleton::m_profLibHandle     = nullptr;
std::mutex             synSingleton::m_singletonCreationMutex;
bool                   synSingleton::m_isChildProcessWithAcquiredDevice = false;

extern const char* SYNAPSE_SHA1_VERSION;
extern const char* HCL_SHA1_VERSION;
extern const char* MME_SHA1_VERSION;
extern const char* SCAL_SHA1_VERSION;

#define GET_DEV_INTERFACE_RTN_IF_ERR()                                                                         \
    std::shared_ptr<DeviceInterface> deviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);       \
    if (deviceInterface == nullptr)                                                                            \
    {                                                                                                          \
        return synFail;                                                                                        \
    }

#define GET_DEV_QMAN_RTN_IF_ERR()                                                                                      \
    std::shared_ptr<DeviceGaudi> deviceQman = m_deviceManager.getDeviceGaudi(__FUNCTION__);                            \
    if (deviceQman == nullptr)                                                                                         \
    {                                                                                                                  \
        return synFail;                                                                                                \
    }

#define GET_DEV_INTERFACE() std::shared_ptr<DeviceInterface> deviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);


#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch-enum"
inline synStatus statusToString(synStatus status, char* statusDescription, size_t len)
{
#define SYN_STR(x, description)                                                                                        \
    case x:                                                                                                            \
        static_assert(strlen(description) < STATUS_DESCRIPTION_MAX_SIZE);                                              \
        std::strncpy(statusDescription, description, len);                                                             \
        if (strlen(description) >= len)                                                                                \
        {                                                                                                              \
            statusDescription[len - 1] = '\0';                                                                         \
            LOG_WARN(SYN_OSAL, "{}: Given statusDescription's length is shorter than required", HLLOG_FUNC);           \
        }                                                                                                              \
        return synSuccess;

    switch (status)
    {
        SYN_STR(synSuccess,                         "Success");
        SYN_STR(synInvalidArgument,                 "Invalid argument");
        SYN_STR(synCbFull,                          "Command-Buffer is full (OOM)");
        SYN_STR(synOutOfHostMemory,                 "Out of HOST memory (OOM)");
        SYN_STR(synOutOfDeviceMemory,               "Out of Device memory (OOM)");
        SYN_STR(synObjectAlreadyInitialized,        "Unexpected module re-initialization");
        SYN_STR(synObjectNotInitialized,            "Unexpected module not initialized");
        SYN_STR(synCommandSubmissionFailure,        "Failed to submit Command-Submission");
        SYN_STR(synNoDeviceFound,                   "Device not found");
        SYN_STR(synDeviceTypeMismatch,              "Device-type mismatch");
        SYN_STR(synFailedToInitializeCb,            "Failed to initialize Command-Buffer");
        SYN_STR(synFailedToFreeCb,                  "Failed to free Command-Buffer");
        SYN_STR(synFailedToMapCb,                   "Failed to map Command-Buffer");
        SYN_STR(synFailedToUnmapCb,                 "Failed to unmap Command-Buffer");
        SYN_STR(synFailedToAllocateDeviceMemory,    "Failed to allocate Device memory");
        SYN_STR(synFailedToFreeDeviceMemory,        "Failed to free Device memory");
        SYN_STR(synFailedNotEnoughDevicesFound,     "Not enough devices found");
        SYN_STR(synOutOfResources,                  "Out-Of-Resources");
        SYN_STR(synDeviceReset,                     "Device in reset");
        SYN_STR(synUnsupported,                     "Unsupported operation");
        SYN_STR(synWrongParamsFile,                 "Invalid Recipe file");
        SYN_STR(synDeviceAlreadyAcquired,           "Device already acquired");
        SYN_STR(synNameIsAlreadyUsed,               "Re-usage of the same name");
        SYN_STR(synBusy,                            "Device is busy - operation not yet completed");
        SYN_STR(synAllResourcesTaken,               "All resources are taken");
        SYN_STR(synUnavailable,                     "Unavailable resource (was not found)");
        SYN_STR(synInvalidTensorDimensions,         "Invalid tensor's dimensions");
        SYN_STR(synFail,                            "Generic failure");
        SYN_STR(synUninitialized,                   "Synapse was not initialized");
        SYN_STR(synAlreadyInitialized,              "Synapse was re-initialized");
        SYN_STR(synFailedSectionValidation,         "Sections' validation failure");
        SYN_STR(synSynapseTerminated,               "Synapse is being terminated due to fatality");
        SYN_STR(synAssertAsync,                     "Assert-Async event had been recognized");
        SYN_STR(synInvalidEventHandle,              "Invalid event-handle");
        SYN_STR(synMappingNotFound,                 "MMU Mapping for given address is missing");
        SYN_STR(synFailedDynamicPatching,           "Dynamic patching had failed");
        SYN_STR(synFailedStaticPatching,            "Static patching had failed");
        SYN_STR(synFailedToSubmitWorkload,          "Workload submission had failed");
        SYN_STR(synInvalidSectionsDefinition,       "Invalid Tensors' addresses or missing Sections' Address");
        SYN_STR(synInvalidTensorProperties,         "Invalid Tensor properties");
        SYN_STR(synFailHccl,                        "Got failure from HCCL");
        SYN_STR(synFailedToCollectTime,             "Failed to collect time");
        SYN_STR(synTimeout,                         "Got timeout notification from Driver");
        SYN_STR(synResourceBadUsage,                "Resource was not properly used");
    }
#undef SYN_STR
    return synFail;
}
#pragma GCC diagnostic pop


/*
 ***************************************************************************************************
 *   @brief CTOR - private
 ***************************************************************************************************
 */

typedef synSingletonInterface* (*PFN_profilerInit)(synSingletonInterface*);

static PFN_ShimGetFunctions s_pShimGetFunctions = nullptr;
static PFN_ShimFinish       s_pShimFinish       = nullptr;

static bool onCrash(int signal, const char* signalStr, bool isSevere, int stage)
{
    const bool synInitialized = synSingleton::isSynapseInitialized();
    switch (stage)
    {
        case 0:
        {
            synSingleton::printVersionToLog(synapse::LogManager::LogType::SYN_DEV_FAIL,
                                            "Habana Labs exception backtrace");
        }
        break;

        case 1:
        {
            if (GCFG_DFA_ON_SIGNAL.value() == true)
            {
                DfaExtraInfo dfaExtraInfo = {
                    .extraInfo = DfaExtraInfo::DfaExtraInfoSignal {.signal = signal, .signalStr = signalStr, .isSevere = isSevere}};

                if (!synInitialized)
                {
                    LOG_CRITICAL(SYN_DEV_FAIL, "Synapse not initialized, not logging DFA");
                    return synInitialized;
                }

                try // go to DFA flow
                {
                    _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::signal, dfaExtraInfo);
                }
                catch(const std::exception& e)
                {
                    LOG_CRITICAL(SYN_DEV_FAIL,
                                 "Exception while logging DFA info from signal with msg {}", e.what());
                }
                catch(...)
                {
                    LOG_CRITICAL(SYN_DEV_FAIL, "Exception while logging DFA info from signal");
                }
            }
        }
        break;

        default:
            LOG_ERR(SYN_DEV_FAIL, "Unknow crash stage {}", stage);
    }

    return synInitialized;
}

static void setProfilerState()
{
    const char* hpValue  = getenv("HABANA_PROFILE");
    const char* ediValue = getenv("ENABLE_DEBUG_INFO");

    // HABANA_PROFILE - if value not equal to 0x0 - profiler is enabled.
    if ((hpValue && strcmp(hpValue, "0") != 0) || (ediValue && strcmp(ediValue, "1") == 0))
    {
        LOG_DEBUG(SYN_API, "Profiler enabled by env variable");
        GCFG_ENABLE_PROFILER.setValue(true);
    }
}

namespace synapse
{
bool forceSELoggerCompatibilityFunction();
}

void synSingleton::initSingleton()
{
    const char* envValue = getenv("HABANA_SHIM_DISABLE");
    // this function does not do anything
    // we call it just to force the compiler to keep the function inside the binary
    // this function forces the compiler to keep all old spdlog symbols for compatibility with modules that use old spdlog
    synapse::forceSELoggerCompatibilityFunction();
    if (envValue == nullptr || strcmp(envValue, "1"))
    {
        LOG_TRACE(SYN_API, "Engaging shim layer");
        m_profLibHandle = LoadSharedObject(SHIM_LIB_NAME);
        if (m_profLibHandle != nullptr)
        {
            fnHandle fn = GetFunction(m_profLibHandle, SHIM_GET_FUNCTIONS);
            if (fn != nullptr)
            {
                s_pShimGetFunctions = reinterpret_cast<PFN_ShimGetFunctions>(fn);
                fn                  = GetFunction(m_profLibHandle, SHIM_FINISH);
                if (fn != nullptr)
                {
                    s_pShimFinish = reinterpret_cast<PFN_ShimFinish>(fn);
                    LOG_DEBUG(SYN_API, "shim layer initialized successfully");
                }
                else
                {
                    LOG_WARN(SYN_API,
                             "{} was not found in {}. This may cause unexpected behavior in back2back workloads",
                             SHIM_FINISH,
                             SHIM_LIB_NAME);
                }
                fn = GetFunction(m_profLibHandle, SHIM_SET_API_VERSION);
                if (fn != nullptr)
                {
                    PFN_ShimSetApiVersion pShimSetApiVersion = reinterpret_cast<PFN_ShimSetApiVersion>(fn);
                    pShimSetApiVersion(SHIM_API_SYNAPSE, SYNAPSE_SINGLETON_INTERFACE_VERSION);
                }
                else
                {
                    LOG_WARN(
                        SYN_API,
                        "{} was not found in {}. This may cause unexpected behavior due to interface versions mismatch",
                        SHIM_SET_API_VERSION,
                        SHIM_LIB_NAME);
                }
            }
            else
            {
                LOG_ERR(SYN_API, "{} was not found in {}", SHIM_FINISH, SHIM_LIB_NAME);
                UnloadSharedObject(m_profLibHandle);
                m_profLibHandle = nullptr;
            }
        }
        else
        {
            LOG_ERR(SYN_API, "Could not load shim layer binary - {}", SHIM_LIB_NAME);
            LOG_ERR(SYN_API, "dlerror: {} ", dlerror());
            const char* ldLibPath = getenv("LD_LIBRARY_PATH");
            LOG_ERR(SYN_API, "LD_LIBRARY_PATH {}", ldLibPath ? ldLibPath : "none");
        }
    }
    auto                   pNewInternalInstance = new synSingleton;
    synSingletonInterface* pNewInstance         = pNewInternalInstance;

    // s_pShimGetFunctions must be called before any virtual function
    if (s_pShimGetFunctions != nullptr)
    {
        pNewInstance = static_cast<synSingletonInterface*>(s_pShimGetFunctions(SHIM_API_SYNAPSE, pNewInstance));
    }

    // Todo SW-12345 Move HABANA_PROFILE  environment variable query and GCFG_ENABLE_PROFILER  setting to an earlier
    // point in synSingleton initialization
    setProfilerState();

    // initialize() must be called only once
    synStatus status = pNewInstance->initialize();
    if (status != synSuccess)
    {
        throw SynapseStatusException("synInitialize() initialize() failed", status);
    }

    // generate eager recipes
    if (GCFG_EAGER_GENERATE_TEMPLATES.value())
    {
        eager_mode::createEagerTemplates();
    }

    if (GCFG_ENABLE_PROFILER.value())
    {
        // TODO [SW-124904]: move out to central pass init
        initializeProfilerDebugInfo();
    }

    // print huge page initial data
    OSAL::getInstance().printHugePageInfo();

    // make a new instance visible after full initialization
    m_pInstanceInternal = pNewInternalInstance;
    m_pInstance         = pNewInstance;
}

synSingletonInterface* synSingleton::getInstance()
{
    return getInstance(false);
}

synSingletonInterface* synSingleton::getInstance(bool createInstance)
{
    if (unlikely(m_pInstance == nullptr || createInstance))
    {
        std::unique_lock<std::mutex> lock(m_singletonCreationMutex);
        if (m_pInstance == nullptr && !createInstance)
        {
            throw SynapseStatusException("synapse api is called without preceding synInitialize() call.",
                                         synUninitialized);
        }

        if (m_pInstance != nullptr && createInstance)
        {
            if (GCFG_DISABLE_DOUBLE_SYN_INITIALIZE.value())
            {
                LOG_ERR(SYN_API, "synInitialize() was already called. double initialization. ignore.");
                throw SynapseStatusException("synInitialize() was already called. double initialization. ignore",
                                             synAlreadyInitialized);
            }
        }

        if (m_pInstance == nullptr)
        {
            initSingleton();
            return m_pInstance;
        }
    }

    // s_pShimGetFunctions must be called before each virtual function invocation
    // it's here in order to be able to load plugins at runtime
    if (s_pShimGetFunctions != nullptr)
    {
        auto shimInstance = static_cast<synSingletonInterface*>(s_pShimGetFunctions(SHIM_API_SYNAPSE, m_pInstance));
        if (shimInstance != m_pInstance)
        {
            // Most of the time there's no shim layer update, so we'll have the same instance
            // This "if" come to clean helgrind errors of multiple writes from multiple threads
            // By itself, there's no threat of data race as write to 8 bytes is atomic in x86
            m_pInstance = shimInstance;
        }
    }

    return m_pInstance;
}

synStatus synSingleton::initializeInstance()
{
    try
    {
        synSingletonInterface* instance = getInstance(true);
        if (instance == nullptr)
        {
            return synFail;
        }
        synapse::LogManager::instance().enablePeriodicFlush(true);
    }
    catch (...)
    {
        return handleException(__FUNCTION__);
    }

    return synSuccess;
}

synStatus synSingleton::destroyInstance()
{
    std::unique_lock<std::mutex> lock(m_singletonCreationMutex);
    if (m_pInstance)
    {
        auto status = m_pInstance->destroy();
        if (m_profLibHandle != nullptr)
        {
            UnloadSharedObject(m_profLibHandle);
            m_profLibHandle = nullptr;
        }

        synapse::LogManager::instance().enablePeriodicFlush(false);
        synapse::LogManager::instance().flush();
        LOG_INFO(SYN_API, "Synapse API's enter/exit counters:\n{}", ApiCounterRegistry::getInstance().toString());
        ApiCounterRegistry::getInstance().reset();

        return status;
    }
    else
    {
        if (GCFG_DISABLE_DOUBLE_SYN_DESTROY.value())
        {
            LOG_ERR(SYN_API, "destroyInstance: synDestroy called without synInitialize. unable to destroy");
            return synUninitialized;
        }
    }
    return synSuccess;
}

synSingleton::synSingleton()
: synSingletonInterface(nullptr),
  m_graphEntries(m_sectionHndlSlopMap),
  m_deviceManager(),
  m_statApi("synApi", 0, false)  // set as disabled, enable only after GCFG is enabled
{
    std::string logsDir = hl_logger::getLogsFolderPath();
    hl_logger::setLogsFolderPathFromEnv();
    std::string newLogsDir = hl_logger::getLogsFolderPath();
    if (logsDir != newLogsDir)
    {
        // it might be that in the new folder there are dfa logs that should be archived
        archiveDfaLogs();
    }
    synapse::LogManager::instance().setOnCrash(onCrash);
    [[maybe_unused]] static int atForkInitialized =
        pthread_atfork(synSingleton::beforeFork,
                       nullptr,
                       [](){ // child
                         m_isChildProcessWithAcquiredDevice =
                               m_pInstance && getInstanceInternal()->m_deviceManager.getNumDevices() > 0;
                       });

    synSingleton::printVersionToLog(synapse::LogManager::LogType::SYN_API, "HabanaLabs Runtime and GraphCompiler");

    // Initialize global configuration
    std::string iniFileName = getConfigFilename();
    GlobalConfManager::instance().init(iniFileName);

    // We need it here because when we construct the stats the GLOBAL_CNFG are not set yet
    g_globalStat.configurePostGcfgAndLoggerInit();
    m_statApi.setEnableState(true); // Enable only after GCFG is init

    KernelDB::instance().init(tpc_lib_api::DEVICE_ID_MAX);
    // Init ShapeFuncRegistry
    initShapeFuncRegistry();

    m_dfaGlblStatus.dfaPhaseConnection =
        synEventDispatcher.addEventHandler<EventDfaPhase>([&](EventDfaPhase const& eventDfaPhase) {
            LOG_DEBUG_T(SYN_API, "setting dfaPhase to {}", (int)eventDfaPhase.dfaPhase);

            DfaPhase prev = m_dfaGlblStatus.dfaPhase;

            switch (eventDfaPhase.dfaPhase)
            {
                case DfaPhase::NONE:
                    m_dfaGlblStatus.dfaPhase = eventDfaPhase.dfaPhase;
                    break;

                case DfaPhase::ENDED:  // unlcok, dfa ended, can release user thread
                {
                    if (m_dfaGlblStatus.dfaPhase != DfaPhase::STARTED)
                    {
                        LOG_ERR(SYN_DEV_FAIL, "dfa phase expected to be started");
                    }
                    m_dfaGlblStatus.dfaPhase = eventDfaPhase.dfaPhase;  // update the status, then unlock
                    m_dfaGlblStatus.mutex.unlock();
                    break;
                }

                case DfaPhase::STARTED:
                {
                    if (m_dfaGlblStatus.dfaPhase != DfaPhase::NONE)
                    {
                        LOG_ERR(SYN_DEV_FAIL, "dfa phase expected to be none");
                    }
                    m_dfaGlblStatus.mutex.lock();  // lock, this will block new user api-s until dfa is done
                    m_dfaGlblStatus.dfaPhase = eventDfaPhase.dfaPhase;  // update phase, do it only after locking
                    break;
                }
            }
            if (prev != m_dfaGlblStatus.dfaPhase) // HCL does not support re-setting the same phase
            {
                hcclDfaUpdateState(m_dfaGlblStatus.dfaPhase);
            }
        });
}

void synSingleton::beforeFork()
{
    if (m_pInstance)
    {
        if (getInstanceInternal()->m_deviceManager.getNumDevices() > 0)
        {
            LOG_WARN_T(SYN_API, "synapse is forked with acquired device.");
        }
        else
        {
            LOG_WARN_T(SYN_API, "synapse is forked while it is active. synDestroy must be called before fork");
        }
    }
    else
    {
        LOG_DEBUG_T(SYN_API, "synapse is forked. synapse is not active");
    }
}

std::string synSingleton::getConfigFilename()
{
    const char* iniEnvValue = getenv("SYNAPSE_CONFIGURATION_FILE");
    std::string iniFileName(".synapse.ini");
    if (iniEnvValue != nullptr)
    {
        iniFileName = std::string(iniEnvValue);
    }
    else
    {
        static const char* home = getenv("HOME");
        if (home != nullptr)
        {
            iniFileName = std::string(home).append("/").append(iniFileName);
        }
    }
    return iniFileName;
}

/*
 ***************************************************************************************************
 *   @brief DTOR
 ***************************************************************************************************
 */
synSingleton::~synSingleton()
{
    _destroy();
    _destroyAllGraphs();
}

/*
 ***************************************************************************************************
*   @brief Initiate the framework common backend singleton

*   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus synSingleton::initialize()
{
    // TBD - define a macro which will allocate logger and hold it as a "smart-object"
#ifdef ENABLE_EVENT_TRIGGER_LOGGER
    const uint32_t csOrderLoggerSize = 5000;
    EventTriggeredLoggerManager::getInstance()->createLogger(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                                                             EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER,
                                                             csOrderLoggerSize);

    const uint32_t opcodeCheckLoggerSize = 200000;
    EventTriggeredLoggerManager::getInstance()->createLogger(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                                                             EVENT_LOGGER_TRIGGER_TYPE_CHECK_OPCODES,
                                                             opcodeCheckLoggerSize);
#endif

    return synSuccess;
}

synStatus
synSingleton::acquireDevice(uint32_t* pDeviceId, const char* pciBus, synDeviceType deviceType, synModuleId moduleId)
{
    return m_deviceManager.acquireDevice(pDeviceId, pciBus, deviceType, moduleId);
}

synStatus synSingleton::releaseDevice(uint32_t devIdx)
{
    return m_deviceManager.releaseDevice();
}

uint16_t synSingleton::getNumOfAcquiredDevices()
{
    return m_deviceManager.getNumDevices();
}

/*
 ***************************************************************************************************
 *   @brief destroy the backend instance
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus synSingleton::destroy()
{
    synStatus status = _destroy();

    LOG_DEBUG_T(SYN_API, "dfa: setting DfaPhase::NONE");

    bool notifyHcl           = (m_dfaGlblStatus.dfaPhase != DfaPhase::NONE); // check if phase change
    m_dfaGlblStatus.dfaPhase = DfaPhase::NONE;

    if (notifyHcl)
    {
        // notify hcl only if there is a phase change
        hcclDfaUpdateState(DfaPhase::NONE);
    }

    // shim is responsible to delete m_pInstance (in case is different than m_pInstanceInternal)
    if (s_pShimGetFunctions != nullptr)
    {
        delete m_pInstanceInternal;
        m_pInstanceInternal = nullptr;
        m_pInstance         = nullptr;
    }
    else
    {
        // Delete the object - Don't access object members
        if (m_pInstance != m_pInstanceInternal)
        {
            delete m_pInstanceInternal;
        }
        delete m_pInstance;
        m_pInstance = nullptr;
        m_pInstanceInternal = nullptr;
    }
    ShapeFuncRegistry::instance().destroy();

#ifdef ENABLE_EVENT_TRIGGER_LOGGER
    EventTriggeredLoggerManager::getInstance()->releaseLogger(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES);
    EventTriggeredLoggerManager::getInstance()->releaseLogger(EVENT_LOGGER_LOG_TYPE_CS_ORDER);
    EventTriggeredLoggerManager::getInstance()->releaseInstance();
#endif

    g_globalStat.setEnableState(false); // This also flush all statistics until now

    return status;
}

synStatus synSingleton::preApiCallExecution(const char * apiFuncName)
{
    if (m_isChildProcessWithAcquiredDevice)
    {
        LOG_CRITICAL(SYN_API, "api {} call in a forked child process after device acquisition", apiFuncName ? apiFuncName : "");
    }
    switch (m_dfaGlblStatus.dfaPhase)
    {
        case DfaPhase::NONE:
            return synSuccess;

        case DfaPhase::STARTED:
        {
            // hold the thread (by waiting for a lock) until dfa collection is done
            std::unique_lock<std::mutex> lck(m_dfaGlblStatus.mutex);
            return synSynapseTerminated;
        }

        case DfaPhase::ENDED:
            return synSynapseTerminated;
    }
    return synSuccess;
}

synStatus synSingleton::postApiCallExecution()
{
    switch (m_dfaGlblStatus.dfaPhase)
    {
        case DfaPhase::NONE:
            return synSuccess;

        case DfaPhase::STARTED:
        {
            // hold the thread (by waiting for a lock) until dfa collection is done
            std::unique_lock<std::mutex> lck(m_dfaGlblStatus.mutex);
            return synSynapseTerminated;
        }

        case DfaPhase::ENDED:
            return synSynapseTerminated;
    }
    return synSuccess;
}

synStatus synSingleton::_destroy()
{
    synStatus status(synSuccess);

    LOG_SINGLETON_API();

    bool isAcquired = false;

    {
        std::shared_ptr<DeviceInterface> deviceInterface = m_deviceManager.getDeviceInterface(nullptr);
        if (deviceInterface != nullptr)
        {
            isAcquired = true;
        }
    }

    if (isAcquired)
    {
        releaseDevice(0);
    }

    if (_releaseAllRecipes() != synSuccess)
    {
        status = synFail;
    }

    if (m_profLibHandle != nullptr && s_pShimFinish != nullptr)
    {
        s_pShimFinish(SHIM_API_SYNAPSE);
        s_pShimFinish = nullptr;
    }

    // Each synapse should have its own kernelDB library
    KernelDB::instance().clear();

    return status;
}

/*
 ***************************************************************************************************
 *   @brief return the free and total memory on a specific device
 *
 *   @param device_id   [in]  the device id memory info is asked for
 *   @param free        [out] free memory available
 *   @param total       [out] total memory on device
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
synStatus synSingleton::getDeviceDramMemoryInfo(uint32_t devIdx, uint64_t& free, uint64_t& total) const
{
    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->getDramMemInfo(free, total);
}

synStatus synSingleton::allocateDeviceMemory(unsigned  devIdx,
                                             uint64_t  size,
                                             uint32_t  flags,
                                             void**    buffer,
                                             uint64_t  reqVAAddress, /*= 0*/
                                             uint64_t* deviceVA)     /*= nullptr*/
{
    LOG_SINGLETON_API("{}: size 0x{:x}", HLLOG_FUNC, size);

    std::string mappingDesc("User request");

    return m_deviceManager.allocateDeviceMemory(size, flags, buffer, true, reqVAAddress, mappingDesc, deviceVA);
}

synStatus synSingleton::deallocateDeviceMemory(unsigned devIdx, void* pBuffer, uint32_t flags)
{
    LOG_SINGLETON_API();

    return m_deviceManager.deallocateDeviceMemory(pBuffer, flags, true);
}

synStatus synSingleton::mapBufferToDevice(unsigned devIdx, uint64_t size, void* buffer, uint64_t reqVAAddress)
{
    LOG_SINGLETON_API();

    std::string mappingDesc("User mapping");

    return m_deviceManager.mapBufferToDevice(size, buffer, true, reqVAAddress, mappingDesc);
}

synStatus synSingleton::unmapBufferFromDevice(unsigned devIdx, void* buffer)
{
    LOG_SINGLETON_API();

    return m_deviceManager.unmapBufferFromDevice(buffer, true);
}

synStatus synSingleton::tensorRetrieveMetadatasInfosByNameExt(const synRecipeHandle  pRecipeHandle,
                                                              const uint32_t         numOfTensors,
                                                              TensorMetadataInfoExt* tensorsMetadataInfo) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "Pointer to Recipe");
    return pRecipeHandle->basicRecipeHandle.tensorRetrieveMetadatasInfosByName(numOfTensors, tensorsMetadataInfo);
}

synStatus synSingleton::tensorRetrievePersistentAmount(const synRecipeHandle pRecipeHandle,
                                                       uint32_t&             numOfTensors) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "Pointer to Recipe");
    return pRecipeHandle->basicRecipeHandle.tensorRetrievePersistentAmount(numOfTensors);
}

synStatus synSingleton::tensorRetrieveNames(const synRecipeHandle pRecipeHandle,
                                            char                  tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                                            const uint32_t        numOfTensors) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "Pointer to Recipe");
    return pRecipeHandle->basicRecipeHandle.tensorRetrieveNames(tensorsName, numOfTensors);
}

synStatus synSingleton::tensorRetrieveLaunchAmount(const synRecipeHandle pRecipeHandle, uint32_t& numOfTensors) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "pRecipeHandle");

    numOfTensors = pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo.getTensorAmount();
    return synSuccess;
}

synStatus synSingleton::tensorRetrieveLaunchIds(const synRecipeHandle pRecipeHandle,
                                                uint64_t*             tensorsIds,
                                                const uint32_t        numOfTensors) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "pRecipeHandle");

    return pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo.tensorRetrieveIds(tensorsIds, numOfTensors);
}

synStatus synSingleton::tensorRetrieveLaunchInfoByIdExt(const synRecipeHandle            pRecipeHandle,
                                                        const uint32_t                   numOfTensors,
                                                        synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "pRecipeHandle");

    return pRecipeHandle->basicRecipeHandle.tensorRetrieveLaunchInfoById(numOfTensors, tensorsLaunchInfo);
}

synStatus synSingleton::tensorRetrieveIds(const synRecipeHandle pRecipeHandle,
                                          const char**          tensorNames,
                                          uint64_t*             tensorIds,
                                          const uint32_t        numOfTensors)
{
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "Pointer to Recipe");
    return pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo.tensorRetrieveIds(tensorNames,
                                                                                          tensorIds,
                                                                                          numOfTensors);
}

synStatus
synSingleton::kernelsPrintf(synRecipeHandle recipeHandle, uint64_t workspaceAddr, void* hostBuff)
{
    LOG_TRACE(SYN_API, "Print kernels printf commands to log");

    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "Pointer to Recipe");

    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->kernelsPrintf(*recipeHandle, workspaceAddr, hostBuff);
}

synStatus synSingleton::_releaseAllRecipes()
{
    // At this point there is no device which uses any of the recipes. Thus, it is safe to remove them.
    const bool operStatus = m_recipeManager.removeAllRecipeHandle();
    if (!operStatus)
    {
        LOG_ERR(SYN_API, "{}: Failed to remove recipe handle from Recipe-Singleton", HLLOG_FUNC);
    }

    return synSuccess;
}

/*
***************************************************************************************************
*   @brief submits command buffers for execution
*
*   @param cs       [in]  command buffers to submit.
*   @param csHandle [out] Handle for of the command submission in order to get an
*                         indication that the execution is finished.
*
*   @return              Status of the operation.
***************************************************************************************************
*/
synStatus synSingleton::submitCommandBuffers(CommandSubmission* cs, uint64_t* handle)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, handle, "Handle");

    if (cs == nullptr)
    {
        LOG_ERR(SYN_API, "Got nullptr for command submission");
        return synInvalidArgument;
    }

    GET_DEV_INTERFACE_RTN_IF_ERR();

    const uint32_t queueOffset = 0;

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    synStatus status = submitCommandBuffers(*cs, handle, nullptr, queueOffset, nullptr);

    ETL_ADD_LOG_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER, logId, SYN_API, "Submitted CS with handle {}", *handle);

    return status;
}

synStatus synSingleton::submitCommandBuffers(CommandSubmission&   commandSubmission,
                                             uint64_t*            csHandle,
                                             uint64_t*            mappedBuff,
                                             uint32_t             queueOffset,
                                             const StagedInfo*    pStagedInfo,
                                             globalStatPointsEnum point)
{
    GET_DEV_QMAN_RTN_IF_ERR();
    return deviceQman->submitCommandBuffers(commandSubmission, csHandle, nullptr, queueOffset, pStagedInfo, point);
}

synStatus synSingleton::waitAndReleaseCS(uint64_t  handle,
                                         uint64_t  timeout,
                                         bool      returnUponTimeout /* = false */,
                                         bool      collectStats /* = false */,
                                         uint64_t* userEventTime)
{
    LOG_TRACE(SYN_API, "{}: handle {}", HLLOG_FUNC, handle);

    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->waitAndReleaseCS(handle, timeout, returnUponTimeout, collectStats, userEventTime);
}

synStatus synSingleton::recipeDestroy(synRecipeHandle recipeHandle)
{
    LOG_SINGLETON_API();

    if (recipeHandle == nullptr)
    {
        LOG_DEBUG(SYN_API, "Recipe-handle 0x{:x} had already been destroyed", (uint64_t)recipeHandle);
        return synSuccess;
    }

    m_deviceManager.notifyRecipeRemoval(*recipeHandle);

    basicRecipeInfo& basicRecipeHandle = recipeHandle->basicRecipeHandle;
    HB_ASSERT_PTR(basicRecipeHandle.recipe);

    bool operStatus = m_recipeManager.removeRecipeHandle(recipeHandle);

    if (!operStatus)
    {
        LOG_ERR(SYN_API, "{}: Failed to remove recipe handle into Recipe-Singleton", HLLOG_FUNC);
    }

    return synSuccess;
}

synStatus synSingleton::getTopologyWorkspaceSize(uint64_t* pWorkspaceSize, const synRecipeHandle recipeHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_API, pWorkspaceSize, "Workspace size");
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "Recipe handle");
    *pWorkspaceSize = recipeHandle->deviceAgnosticRecipeHandle.m_workspaceSize;
    return synSuccess;
}

synStatus synSingleton::addDeviceAddressToReleaseOnStreamDestroy(uint64_t address)
{
    LOG_TRACE(SYN_API, "{}: address 0x{:x}", HLLOG_FUNC, address);

    GET_DEV_QMAN_RTN_IF_ERR();

    deviceQman->addAddrToReleaseOnStreamDestroy(address);

    return synSuccess;
}

synStatus synSingleton::createStream(synStreamHandle*   pStreamHandle,
                                     const uint32_t     devIdx,
                                     uint32_t           streamType,
                                     const unsigned int flags)
{
    LOG_SINGLETON_API();
    const synStatus status = synUnsupported;
    LOG_ERR(SYN_API,
            "{}: devIdx {} streamType {} flags {} status {} please move from synStreamCreate to synStreamCreateGeneric",
            HLLOG_FUNC,
            devIdx,
            TO64(streamType),
            flags,
            status);
    return status;
}

synStatus synSingleton::createStream(synStreamHandle* pStreamHandle, const uint32_t devIdx, const unsigned int flags)
{
    LOG_SINGLETON_API();
    GET_DEV_INTERFACE_RTN_IF_ERR();
    VERIFY_IS_NULL_POINTER(SYN_DEVICE, pStreamHandle, "pStreamHandle");
    return deviceInterface->createStreamGeneric(flags, *pStreamHandle);
}

synStatus synSingleton::destroyStream(synStreamHandle streamHandle)
{
    LOG_SINGLETON_API();

    if (streamHandle == nullptr)
    {
        LOG_DEBUG(SYN_API, "Stream-handle 0x{:x} is null", (uint64_t)streamHandle);
        return synSuccess;
    }

    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->destroyStream(streamHandle);
}

synStatus synSingleton::createEvent(synEventHandle* pEventHandle, const uint32_t devIdx, const unsigned int flags)
{
    LOG_SINGLETON_API();

    CHECK_POINTER(SYN_API, pEventHandle, "Event handle", synInvalidArgument);
    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->createEvent(pEventHandle, flags);
}

synStatus synSingleton::destroyEvent(synEventHandle eventHandle)
{
    LOG_SINGLETON_API();
    CHECK_POINTER(SYN_API, eventHandle, "eventHandle", synInvalidArgument);

    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->destroyEvent(eventHandle);
}

std::shared_ptr<DeviceInterface> synSingleton::getDevice() const
{
    return m_deviceManager.getDeviceInterface(__FUNCTION__);
}

synStatus synSingleton::synchronizeStream(const synStreamHandle streamHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "Stream handle");

    GET_DEV_INTERFACE();

    if (deviceInterface == nullptr)
    {
        LOG_ERR(SYN_STREAM, "{} verify device failed", HLLOG_FUNC);
        return synFail;
    }

    LOG_TRACE(SYN_STREAM, "{}: Synchronizing stream", HLLOG_FUNC);

    synStatus status = deviceInterface->synchronizeStream(streamHandle);
    if (deviceInterface->isAssertAsyncNoticed()) return synAssertAsync;

    return status;
}

synStatus synSingleton::synchronizeAllStreams(const uint32_t devIdx)
{
    LOG_TRACE(SYN_STREAM, "{}", HLLOG_FUNC);

    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->synchronizeAllStreams();
}

synStatus synSingleton::synchronizeEvent(const synEventHandle eventHandle)
{
    GET_DEV_INTERFACE_RTN_IF_ERR();
    synStatus status = deviceInterface->synchronizeEvent(eventHandle);

    if (deviceInterface->isAssertAsyncNoticed()) return synAssertAsync;

    return status;
}

static synStatus loadDeviceAndEventFromEventHandle(synEventHandle                    eventHandle,
                                                   const char*                       functionName,
                                                   DeviceManager&                    deviceManager,
                                                   std::shared_ptr<DeviceInterface>* deviceInterface,
                                                   EventSPtr&                        eventSptr)
{
    if (deviceInterface == nullptr)
    {
        return synFail;
    }
    *deviceInterface = deviceManager.getDeviceInterface(functionName);
    if (*deviceInterface == nullptr)
    {
        return synFail;
    }

    eventSptr = (*deviceInterface)->getEventSptr(eventHandle);
    if (eventSptr == nullptr)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed on event verification, event handle {:x} is incorrect, probably event was destroyed",
                functionName,
                (SMHandle)eventHandle);
        return synFail;
    }

    return synSuccess;
}

synStatus synSingleton::eventElapsedTime(uint64_t*            pNanoseconds,
                                         const synEventHandle eventHandleStart,
                                         const synEventHandle eventHandleEnd)
{
    std::shared_ptr<DeviceInterface> deviceStreamStart;
    EventSPtr        eventSptrStart;
    EventInterface*  eventStart = nullptr;
    if (loadDeviceAndEventFromEventHandle(eventHandleStart,
                                          __FUNCTION__,
                                          m_deviceManager,
                                          &deviceStreamStart,
                                          eventSptrStart) != synSuccess)
    {
        return synFail;
    }
    eventStart = eventSptrStart.get();

    EventSPtr       eventSptrEnd;
    EventInterface* eventEnd = nullptr;
    if (eventHandleEnd != nullptr)
    {
        std::shared_ptr<DeviceInterface> deviceStreamEnd;
        if (loadDeviceAndEventFromEventHandle(eventHandleEnd,
                                              __FUNCTION__,
                                              m_deviceManager,
                                              &deviceStreamEnd,
                                              eventSptrEnd) != synSuccess)
        {
            return synFail;
        }
        eventEnd = eventSptrEnd.get();
    }
    return QmanEvent::eventElapsedTime(pNanoseconds, eventStart, eventEnd);
}

synStatus synSingleton::eventRecord(synEventHandle eventHandle, const synStreamHandle streamHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "Stream handle");
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->eventRecord(eventHandle, streamHandle);
}

synStatus
synSingleton::streamWaitEvent(synStreamHandle streamHandle, const synEventHandle eventHandle, const unsigned int flags)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "Stream handle");
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->streamWaitEvent(streamHandle, eventHandle, flags);
}

synStatus synSingleton::eventQuery(const synEventHandle eventHandle)
{
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->eventQuery(eventHandle);
}

synStatus synSingleton::waitAndReleaseStreamHandles(const InternalWaitHandlesVector& streamWaitHandles,
                                                    uint64_t                         timeout,
                                                    bool                             returnUponTimeout /* = false */)
{
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->waitAndReleaseStreamHandles(streamWaitHandles, timeout, returnUponTimeout);
}

synStatus synSingleton::streamQuery(const synStreamHandle streamHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "Stream handle");

    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->streamQuery(streamHandle);
}

synStatus synSingleton::getRecipeDebugInfo(synRecipeHandle recipe, const debug_info_t** recipeDebugInfo)
{
    synStatus status = synSuccess;

    if (recipeDebugInfo != nullptr)
    {
        if (recipe != nullptr)
        {
            basicRecipeInfo& basicRecipeHandle = recipe->basicRecipeHandle;
            *recipeDebugInfo                   = &basicRecipeHandle.recipe->debug_profiler_info;
        }
        else
        {
            *recipeDebugInfo = nullptr;
            LOG_ERR(SYN_API, "{} called with recipe as nullptr", HLLOG_FUNC);
            status = synInvalidArgument;
        }
    }
    else
    {
        LOG_ERR(SYN_API, "{} called with recipeDebugInfo as nullptr", HLLOG_FUNC);
        status = synInvalidArgument;
    }

    return status;
}

synStatus synSingleton::getRecipeProgramDataBlobs(synRecipeHandle             recipe,
                                                  const program_data_blob_t** program_data_blobs,
                                                  size_t*                     program_data_blobs_nr)
{
    if (program_data_blobs != nullptr && program_data_blobs_nr != nullptr)
    {
        if (recipe != nullptr)
        {
            basicRecipeInfo& basicRecipeHandle = recipe->basicRecipeHandle;
            *program_data_blobs                = basicRecipeHandle.recipe->program_data_blobs;
            *program_data_blobs_nr             = basicRecipeHandle.recipe->program_data_blobs_nr;
        }
        else
        {
            *program_data_blobs    = nullptr;
            *program_data_blobs_nr = 0x0;
        }
    }
    else
    {
        LOG_TRACE(SYN_API, "{} called with program_data_blobs or program_data_blobs_nr as nullptr", HLLOG_FUNC);
    }

    return synSuccess;
}

synStatus synSingleton::getRecipeSyncScheme(const synRecipeHandle recipe, const debug_sync_scheme_t** recipeSyncScheme)
{
    if (recipeSyncScheme != nullptr)
    {
        if (recipe != nullptr)
        {
            basicRecipeInfo& basicRecipeHandle = recipe->basicRecipeHandle;
            *recipeSyncScheme                  = &basicRecipeHandle.recipe->debug_sync_scheme_info;
        }
        else
        {
            *recipeSyncScheme = nullptr;
        }
    }
    else
    {
        LOG_TRACE(SYN_API, "{} called with recipeSyncScheme as nullptr", HLLOG_FUNC);
    }

    return synSuccess;
}

synStatus synSingleton::recipeSerialize(const synRecipeHandle recipeHandle, const char* recipeFileName)
{
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "recipeHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, recipeFileName, "recipeFileName");

    LOG_TRACE(SYN_API, "{} for {} id {}", HLLOG_FUNC, TO64(recipeHandle), recipeHandle->recipeSeqNum);

    basicRecipeInfo& basicRecipeHandle = recipeHandle->basicRecipeHandle;
    HB_ASSERT_PTR(basicRecipeHandle.recipe);

    const synStatus status = m_recipeManager.recipeSerialize(recipeHandle, recipeFileName);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not serialize file {}", HLLOG_FUNC, recipeFileName);
        return status;
    }

    return synSuccess;
}

synStatus synSingleton::recipeDeSerialize(synRecipeHandle* pRecipeHandle, const char* recipeFileName)
{
    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "pRecipeHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, recipeFileName, "recipeFileName");

    InternalRecipeHandle* pInternalRecipeHandle = nullptr;
    const synStatus       status = m_recipeManager.recipeDeSerialize(pInternalRecipeHandle, recipeFileName);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not deserialize the file {}", HLLOG_FUNC, recipeFileName);
        return status;
    }

    *pRecipeHandle = pInternalRecipeHandle;

    return synSuccess;
}

synStatus synSingleton::recipeGetAttribute(uint64_t*                 retVal,
                                           const synRecipeAttribute* recipeAttr,
                                           const unsigned            querySize,
                                           const synRecipeHandle     recipeHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "recipeHandle");
    LOG_TRACE(SYN_API, "{} for recipe {:x} id {}", HLLOG_FUNC, TO64(recipeHandle), recipeHandle->recipeSeqNum);
    return RecipeManager::recipeGetAttribute(retVal, recipeAttr, querySize, recipeHandle);
}

synStatus synSingleton::enqueue(const synStreamHandle      streamHandle,
                                const synLaunchTensorInfo* enqueueInputTensorsInfo,
                                const uint32_t             inputInfoSize,
                                const synLaunchTensorInfo* enqueueOutputTensorsInfo,
                                const uint32_t             outputInfoSize,
                                uint64_t                   workspaceAddress,
                                const synRecipeHandle      pRecipeHandle,
                                uint32_t                   flags)
{
    synLaunchTensorInfoExt enqueueInputTensorsInfoExt[inputInfoSize];
    elevateSynLaunchTensorInfo(enqueueInputTensorsInfoExt, enqueueInputTensorsInfo, inputInfoSize);
    return enqueue(streamHandle, enqueueInputTensorsInfoExt, inputInfoSize, workspaceAddress, pRecipeHandle, flags);
}

synStatus synSingleton::enqueue(const synStreamHandle         streamHandle,
                                const synLaunchTensorInfoExt* enqueueInputTensorsInfo,
                                const uint32_t                inputInfoSize,
                                const synLaunchTensorInfoExt* enqueueOutputTensorsInfo,
                                const uint32_t                outputInfoSize,
                                uint64_t                      workspaceAddress,
                                const synRecipeHandle         pRecipeHandle,
                                uint32_t                      flags)
{
    return enqueue(streamHandle,
                   enqueueInputTensorsInfo,
                   inputInfoSize,
                   workspaceAddress,
                   pRecipeHandle,
                   nullptr,
                   0,
                   flags);
}

synStatus synSingleton::enqueue(const synStreamHandle         streamHandle,
                                const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                const uint32_t                enqueueTensorsAmount,
                                uint64_t                      workspaceAddress,
                                const synRecipeHandle         pRecipeHandle,
                                uint32_t                      flags)
{
    return enqueue(streamHandle,
                   enqueueTensorsInfo,
                   enqueueTensorsAmount,
                   workspaceAddress,
                   pRecipeHandle,
                   nullptr,  // eventHandleList
                   0,        // numberOfEvents
                   flags);
}

synStatus synSingleton::enqueue(const synStreamHandle         streamHandle,
                                const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                const uint32_t                enqueueTensorsAmount,
                                uint64_t                      workspaceAddress,
                                const synRecipeHandle         pRecipeHandle,
                                synEventHandle*               eventHandleList,
                                uint32_t                      numberOfEvents,
                                uint32_t                      flags)
{
    synLaunchTensorInfoExt launchTensorInfo[enqueueTensorsAmount];
    for (size_t i = 0; i < enqueueTensorsAmount; i++)
    {
        launchTensorInfo[i].tensorName     = enqueueTensorsInfo[i].tensorName;
        launchTensorInfo[i].pTensorAddress = enqueueTensorsInfo[i].pTensorAddress;
        launchTensorInfo[i].tensorType     = enqueueTensorsInfo[i].tensorType;
        launchTensorInfo[i].tensorId       = enqueueTensorsInfo[i].tensorId;
        memcpy(launchTensorInfo[i].tensorSize, enqueueTensorsInfo[i].tensorSize, sizeof(TSize) * HABANA_DIM_MAX);
    }
    return enqueueWithExternalEventsExt(streamHandle,
                                        (const synLaunchTensorInfoExt*)(&launchTensorInfo),
                                        enqueueTensorsAmount,
                                        workspaceAddress,
                                        pRecipeHandle,
                                        eventHandleList,
                                        numberOfEvents,
                                        flags);
}

synStatus synSingleton::enqueueWithExternalEventsExt(const synStreamHandle         streamHandle,
                                                     const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                     const uint32_t                enqueueTensorsAmount,
                                                     uint64_t                      workspaceAddress,
                                                     const synRecipeHandle         pRecipeHandle,
                                                     synEventHandle*               eventHandleList,
                                                     uint32_t                      numberOfEvents,
                                                     uint32_t                      flags)

{
    VERIFY_IS_NULL_POINTER(SYN_API, streamHandle, "Stream handle");
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "Recipe handle");

    GET_DEV_INTERFACE();

    if (deviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Invalid streamHandle", HLLOG_FUNC);
        return synFail;
    }

    synStatus status;
    try
    {
        status = deviceInterface->launchWithExternalEvents(streamHandle,
                                                           enqueueTensorsInfo,
                                                           enqueueTensorsAmount,
                                                           workspaceAddress,
                                                           pRecipeHandle,
                                                           eventHandleList,
                                                           numberOfEvents,
                                                           flags);
    }
    catch (const std::exception& error) // Do we throw???
    {
        RecipeManager::notifyRecipeLaunchFailure(pRecipeHandle, enqueueTensorsInfo, enqueueTensorsAmount, flags);

        throw;
    }

    if (status == synSuccess)
    {
        pRecipeHandle->basicRecipeHandle.recipeStats.numbSuccessfulLaunch++;
    }
    else
    {
        RecipeManager::notifyRecipeLaunchFailure(pRecipeHandle, enqueueTensorsInfo, enqueueTensorsAmount, flags);
    }

    return status;
}

synStatus synSingleton::deviceGetCount(uint32_t* pCount)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, pCount, "pCount");

    return m_deviceManager.deviceGetCount(pCount);
};

synStatus synSingleton::deviceGetModuleIds(uint32_t *pDeviceModuleIds, uint32_t*  size)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, pDeviceModuleIds, "pDeviceModuleIds");
    if (*size == 0)
    {
        LOG_ERR(SYN_API, "{}: given array size is zero", HLLOG_FUNC);
        return  synInvalidArgument;
    }

    return m_deviceManager.deviceGetModuleIds(pDeviceModuleIds, size);
};

synStatus synSingleton::deviceGetCountByDeviceType(uint32_t* pCount, const synDeviceType deviceType)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, pCount, "pCount");

    return m_deviceManager.deviceGetCountByDeviceType(pCount, deviceType);
};

synStatus synSingleton::deviceCount(uint32_t count[synDeviceTypeSize])
{
    LOG_SINGLETON_API();

    return m_deviceManager.deviceCount(count);
}

synStatus synSingleton::deviceGetFd(int* pFd, const synDeviceId devIdx)
{
    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, pFd, "pFd");

    return m_deviceManager.deviceGetFd(pFd);
}

synStatus synSingleton::deviceGetPCIBusId(char* pPciBusId, const int len, const synDeviceId devIdx)
{
    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, pPciBusId, "pPciBusId");

    return m_deviceManager.deviceGetPCIBusId(pPciBusId, len);
}

synStatus synSingleton::getDeviceId(const synStreamHandle streamHandle, synDeviceId& devIdx) const
{
    devIdx = 0;
    return synSuccess;
}

synStatus synSingleton::getDeviceInfo(unsigned devIdx, synDeviceInfo* pDeviceInfo) const
{
    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, pDeviceInfo, "Device-Info");

    return m_deviceManager.getDeviceInfo(pDeviceInfo);
}

synStatus synSingleton::getDeviceInfo(unsigned devIdx, synDeviceInfoV2* pDeviceInfo) const
{
    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, pDeviceInfo, "Device-Info");

    return m_deviceManager.getDeviceInfo(pDeviceInfo);
}

synStatus synSingleton::getDeviceAttribute(const synDeviceId         devIdx,
                                           const synDeviceAttribute* deviceAttr,
                                           const unsigned            querySize,
                                           uint64_t*                 retVal) const
{
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->getDeviceAttribute(deviceAttr, querySize, retVal);
}

synStatus synSingleton::getDeviceTypeAttribute(const synDeviceType       deviceType,
                                               const synDeviceAttribute* deviceAttr,
                                               const unsigned            querySize,
                                               uint64_t*                 retVal) const
{
    return DeviceManager::getDeviceTypeAttribute(deviceType, deviceAttr, querySize, retVal);
}

synStatus synSingleton::eventMapTensorBaseExt(synEventHandle*               eventHandle,
                                              size_t                        numOfEvents,
                                              const synLaunchTensorInfoExt* launchTensorsInfo,
                                              const synRecipeHandle         recipeHandle) const
{
    // mapTensorInEvent
    // this api is stateless - not changing anything here
    for (size_t handleIndex = 0; handleIndex < numOfEvents; handleIndex++)
    {
        uint64_t tensorId = launchTensorsInfo[handleIndex].tensorId;
        uint64_t sequenceOffset =
            recipeHandle->deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getExtTensorExeOrderByExtTensorIdx(tensorId);
        LOG_TRACE(SYN_API,
                 "{}: mapping event 0x{:x} to tensor {} with id {} sequence offset {}",
                 HLLOG_FUNC,
                 TO64(eventHandle),
                 launchTensorsInfo[handleIndex].tensorName,
                 tensorId,
                 sequenceOffset);
        if (sequenceOffset == SignalFromGraphInfo::TENSOR_EXE_ORDER_INVALID)
        {
            LOG_ERR(SYN_API,
                    "{}: Unable to find tensor {} with tensor id {} ",
                    HLLOG_FUNC,
                    launchTensorsInfo[handleIndex].tensorName,
                    launchTensorsInfo[handleIndex].tensorId);
            return synFail;
        }

        GET_DEV_INTERFACE_RTN_IF_ERR();
        auto eventSptr = deviceInterface->getEventSptr(eventHandle[handleIndex]);
        if (!eventSptr)
        {
            LOG_ERR(SYN_API,
                    "{}: event handle {:x} is invalid (index {})",
                    HLLOG_FUNC,
                    (uint64_t)eventHandle[handleIndex],
                    handleIndex);
        }

        EventWithMappedTensor* pEventWithMappedTensor = dynamic_cast<EventWithMappedTensor*>(eventSptr.get());
        if (pEventWithMappedTensor == nullptr)
        {
            LOG_ERR(SYN_API,
                    "{}: Incorrect event type. Event: {:#x} for tensor {} with tensor id {} index {}",
                    HLLOG_FUNC,
                    (uint64_t)eventHandle[handleIndex],
                    launchTensorsInfo[handleIndex].tensorName,
                    launchTensorsInfo[handleIndex].tensorId,
                    handleIndex);
            return synFail;
        }
        pEventWithMappedTensor->clearState();
        pEventWithMappedTensor->setMappedTensor(sequenceOffset,
                                                launchTensorsInfo->tensorId,
                                                launchTensorsInfo->tensorName,
                                                recipeHandle);
    }

    return synSuccess;
}

// this function is for testing only
EventInterface* synSingleton::getEventInterface(synEventHandle eventHandle)
{
    std::shared_ptr<DeviceInterface> deviceInterface;
    EventSPtr        eventSptr;
    if (loadDeviceAndEventFromEventHandle(eventHandle,
                                          __FUNCTION__,
                                          m_deviceManager,
                                          &deviceInterface,
                                          eventSptr) != synSuccess)
    {
        LOG_ERR(SYN_API, "failed to get event from eventHandle");
        return nullptr;
    }
    return eventSptr.get();
}

synStatus synSingleton::externalTensorsExtractExecutionOrder(const synRecipeHandle recipeHandle,
                                                             uint32_t              numOfEvents,
                                                             uint64_t*             tensorIds) const
{
    size_t numOfExternalTensors =
        recipeHandle->deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getNumberOfExternalTensors();

    const std::deque<uint64_t>& tensorExecutionOrder =
        recipeHandle->deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getTensorExecutionOrder();
    if (numOfEvents <= numOfExternalTensors)
    {
        std::copy(tensorExecutionOrder.begin(), tensorExecutionOrder.begin() + numOfEvents, tensorIds);
    }
    else
    {
        std::copy(tensorExecutionOrder.begin(), tensorExecutionOrder.end(), tensorIds);
        for (size_t i = numOfExternalTensors; i < numOfEvents; i++)
        {
            tensorIds[i] = UINT32_MAX;
        }
    }
    return synSuccess;
}

synStatus synSingleton::setCfg(const char* cfgName, const char* cfgValue)
{
    bool ret;

    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, cfgName, "Config Name");
    VERIFY_IS_NULL_POINTER(SYN_API, cfgValue, "Config Value");

    ret = GlobalConfManager::instance().setGlobalConf(cfgName, cfgValue);

    if (!ret)
    {

        LOG_ERR(SYN_API, "config set: {} value: {} failed.", cfgName, cfgValue);
        return synInvalidArgument;
    }

    return synSuccess;
}

synStatus synSingleton::getCfg(const char* cfgName, char* cfgValue, uint64_t size)
{
    bool ret;

    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, cfgName, "Config Name");
    VERIFY_IS_NULL_POINTER(SYN_API, cfgValue, "Config Value");

    ret = GlobalConfManager::instance().getGlobalConf(cfgName, cfgValue, size);

    if (!ret)
    {
        return synInvalidArgument;
    }

    return synSuccess;
}

synStatus
synSingleton::writeI2cReg(uint32_t devIdx, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t value)
{
    LOG_SINGLETON_API();

    return OSAL::getInstance().writeI2cReg(devIdx, i2cBus, i2cAddress, regAddress, value);
}

synStatus
synSingleton::readI2cReg(uint32_t devIdx, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t* pValue)
{
    LOG_SINGLETON_API();

    return OSAL::getInstance().readI2cReg(devIdx, i2cBus, i2cAddress, regAddress, pValue);
}

synStatus synSingleton::setLedState(uint32_t devIdx, uint32_t ledId, bool state)
{
    LOG_SINGLETON_API();

    return OSAL::getInstance().setLedState(devIdx, ledId, state);
}

synStatus synSingleton::setFrequency(uint32_t devIdx, uint32_t pllId, uint32_t frequency)
{
    LOG_SINGLETON_API();

    return OSAL::getInstance().setFrequency(devIdx, pllId, frequency);
}

synStatus synSingleton::getFrequency(uint32_t devIdx, uint32_t pllId, uint32_t* pFrequency)
{
    LOG_SINGLETON_API();

    return OSAL::getInstance().getFrequency(devIdx, pllId, pFrequency);
}

synStatus synSingleton::memcpyAsync(const synStreamHandle streamHandle,
                                    const uint64_t*       pSrc,
                                    const uint64_t*       pSize,
                                    const uint64_t*       pDst,
                                    const synDmaDir       direction,
                                    const uint64_t        numCopies)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, streamHandle, "Stream handle");

    internalDmaDir internalDirection = directionConversion(direction);

    // Temporary solution in order not to break master
    if (direction > DIRECTION_ENUM_MAX)
    {
        internalDirection = static_cast<internalDmaDir>(direction);
    }

    internalMemcopyParams memcpyParams;
    memcpyParams.reserve(numCopies);

    for (int i = 0; i < numCopies; i++)
    {
        if (pSize[i] == 0)
        {
            continue;
        }
        else if (pSrc[i] == 0 || pDst[i] == 0)
        {
            LOG_ERR(SYN_API, "{}: src = {} or dest = {} have 0 address on index {}", HLLOG_FUNC, pSrc[i], pDst[i], i);
            return synInvalidArgument;
        }

        memcpyParams.push_back({.src = pSrc[i], .dst = pDst[i], .size = pSize[i]});
    }

    GET_DEV_INTERFACE();

    if (deviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{} failed to get device", HLLOG_FUNC);
        return synFail;
    }

    return deviceInterface->memcopy(streamHandle, memcpyParams, internalDirection, true);
}

synStatus synSingleton::memsetAsync(const synStreamHandle streamHandle,
                                    uint64_t              pDeviceMem,
                                    const uint32_t        value,
                                    const size_t          numOfElements,
                                    const size_t          elementSize)
{
    LOG_SINGLETON_API();

    VERIFY_IS_NULL_POINTER(SYN_API, streamHandle, "Stream handle");

    GET_DEV_INTERFACE();
    if (deviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{} failed to get device", HLLOG_FUNC);
        return synFail;
    }

    return deviceInterface->memSet(streamHandle, pDeviceMem, value, numOfElements, elementSize);
}

synStatus synSingleton::getTPCLibraryVersionSize(uint32_t* size)
{
    VERIFY_IS_NULL_POINTER(SYN_API, size, "TPC Libs size");
    *size = KernelDB::instance().GetLibraryVersions().size();
    return synSuccess;
}

synStatus synSingleton::getTPCLibraryVersions(const char** libs, uint32_t* versions)
{
    VERIFY_IS_NULL_POINTER(SYN_API, versions, "TPC Libs versions");
    VERIFY_IS_NULL_POINTER(SYN_API, libs, "TPC Libs paths");

    unsigned i = 0;

    const std::unordered_map<std::string, uint32_t>& versionsMap = (KernelDB::instance().GetLibraryVersions());

    for (auto iter = versionsMap.begin(); iter != versionsMap.end(); ++iter, i++)
    {
        libs[i]     = iter->first.c_str();
        versions[i] = iter->second;
    }

    return synSuccess;
}

synStatus synSingleton::getDeviceName(char* pName, const int len, const synDeviceId devIdx)
{
    VERIFY_IS_NULL_POINTER(SYN_API, pName, "pName")

    return m_deviceManager.getDeviceName(pName, len);
}

synStatus synSingleton::profile(unsigned devIdx, hl_debug_args* args)
{
    LOG_SINGLETON_API();

    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->profile(args);
}

synStatus synSingleton::getClockSyncInfo(unsigned devIdx, hlthunk_time_sync_info* infoOut)
{
    LOG_SINGLETON_API();

    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->getClockSyncInfo(infoOut);
}

synStatus synSingleton::getClockSyncPerDieInfo(unsigned devIdx, uint32_t dieIndex, hlthunk_time_sync_info* infoOut)
{
    LOG_SINGLETON_API();

    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->getClockSyncPerDieInfo(dieIndex, infoOut);
}

synStatus synSingleton::getPllFrequency(unsigned devIdx, uint32_t index, struct hlthunk_pll_frequency_info* freqOut)
{
    LOG_SINGLETON_API();

    GET_DEV_INTERFACE_RTN_IF_ERR();

    return deviceInterface->getPllFrequency(index, freqOut);
}

synStatus synSingleton::getModuleId(uint32_t& idOut)
{
    LOG_SINGLETON_API();

    return OSAL::getInstance().getDeviceModuleIdx(idOut);
}

void synSingleton::printVersionToLog(synapse::LogManager::LogType logType, const std::string& description)
{
    static bool bPrinted[(uint32_t)synapse::LogManager::LogType::LOG_MAX] {};

    int width1 = 20, width2 = 50;
    int width3 = width1 + width2;

    if (!bPrinted[(uint32_t)logType])
    {
        bPrinted[(uint32_t)logType] = true;
        std::string version         = fmt::format("{}.{}.{}", HL_DRIVER_MAJOR, HL_DRIVER_MINOR, HL_DRIVER_PATCHLEVEL);

        LOG_INFO_F(logType, "+ {0:-^{1}} +", "", width3);
        LOG_INFO_F(logType, "| {0:{1}}{2:{3}} |", "Version:", width1, version, width2);
        LOG_INFO_F(logType, "| {0:{1}}{2:{3}} |", "Synapse:", width1, SYNAPSE_SHA1_VERSION, width2);
        LOG_INFO_F(logType, "| {0:{1}}{2:{3}} |", "HCL:", width1, HCL_SHA1_VERSION, width2);
        LOG_INFO_F(logType, "| {0:{1}}{2:{3}} |", "MME:", width1, MME_SHA1_VERSION, width2);
        LOG_INFO_F(logType, "| {0:{1}}{2:{3}} |", "SCAL:", width1, SCAL_SHA1_VERSION, width2);
        LOG_INFO_F(logType,
                   "| {0:{1}}{2:{3}} |",
                   "Description:",
                   width1,
                   description,
                   width2);
        LOG_INFO_F(logType, "| {0:{1}}{2:{3}} |", "Time:", width1, TimeTools::timePoint2string(std::chrono::system_clock::now()), width2);
        LOG_INFO_F(logType, "+ {0:-^{1}} +", "", width3);
    }
}

std::vector<std::string> synSingleton::_deviceTypeToStrings(synDeviceType deviceType)
{
    std::vector<std::string> vectorOfDeviceTypesStr;
    switch (deviceType)
    {
        case synDeviceGaudi:
            vectorOfDeviceTypesStr.push_back("GAUDI");
            return vectorOfDeviceTypesStr;
        case synDeviceGaudi2:
            vectorOfDeviceTypesStr.push_back("GAUDI2");
            vectorOfDeviceTypesStr.push_back("GAUDI2B");
            vectorOfDeviceTypesStr.push_back("GAUDI2C");
            vectorOfDeviceTypesStr.push_back("GAUDI2D");
            return vectorOfDeviceTypesStr;
        case synDeviceGaudi3:
            vectorOfDeviceTypesStr.push_back("GAUDI3");
            return vectorOfDeviceTypesStr;
        case synDeviceEmulator:
            vectorOfDeviceTypesStr.push_back("GAUDI Simulator");
            return vectorOfDeviceTypesStr;
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            LOG_ERR(SYN_API, "{} called with unfamiliar device type :{}", HLLOG_FUNC, deviceType);
            return vectorOfDeviceTypesStr;
    }
    return vectorOfDeviceTypesStr;
}

eMappingStatus synSingleton::_getDeviceVirtualAddress(bool      isUserRequest,
                                                      void*     hostAddress,
                                                      uint64_t  bufferSize,
                                                      uint64_t* pDeviceVA,
                                                      bool*     pIsExactKeyFound /*= nullptr*/)
{
    std::shared_ptr<DeviceInterface> deviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);
    if (deviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{}: failed to get device", HLLOG_FUNC);
        return HATVA_MAPPING_STATUS_FAILURE;
    }

    return deviceInterface->getDeviceVirtualAddress(isUserRequest,
                                                    hostAddress,
                                                    bufferSize,
                                                    pDeviceVA,
                                                    pIsExactKeyFound);
}

synStatus synSingleton::submitLinDmaCommand(const internalMemcopyParams& rMemcpyParams,
                                            internalDmaDir               direction,
                                            bool                         isArbitrationRequired,
                                            PhysicalQueuesId             physicalQueueId,
                                            InternalWaitHandle*          waitHandle,
                                            DataChunksDB&                rDataChunks,
                                            CommandSubmissionDataChunks* pCsDataChunks,
                                            bool                         isUserRequest,
                                            bool                         isMemset,
                                            bool                         isInspectCopiedContent,
                                            uint64_t                     maxLinDmaBufferSize,
                                            uint64_t                     arbCommandSize,
                                            uint64_t                     sizeOfLinDmaCommand,
                                            uint64_t                     sizeOfWrappedLinDmaCommand,
                                            uint64_t                     sizeOfSingleCommandBuffer)
{
    GET_DEV_QMAN_RTN_IF_ERR();
    return deviceQman->submitLinDmaCommand(rMemcpyParams,
                                           direction,
                                           isArbitrationRequired,
                                           physicalQueueId,
                                           waitHandle,
                                           rDataChunks,
                                           pCsDataChunks,
                                           isUserRequest,
                                           isMemset,
                                           isInspectCopiedContent,
                                           maxLinDmaBufferSize,
                                           arbCommandSize,
                                           sizeOfLinDmaCommand,
                                           sizeOfWrappedLinDmaCommand,
                                           sizeOfSingleCommandBuffer);
}

synStatus synSingleton::submitTrainingConfigurationCS(synDeviceType      deviceType,
                                                      char*&             pPackets,
                                                      uint64_t           packetsSize,
                                                      const std::string& operationDescription,
                                                      uint32_t           queueId,
                                                      bool               isConfigOnInternal,
                                                      bool               isSyncWithExternalRequired,
                                                      uint32_t           waitQmanId)
{
    GET_DEV_QMAN_RTN_IF_ERR();

    return deviceQman->submitTrainingConfigurationCS(deviceType,
                                                     pPackets,
                                                     packetsSize,
                                                     operationDescription,
                                                     queueId,
                                                     isConfigOnInternal,
                                                     isSyncWithExternalRequired,
                                                     waitQmanId);
}

synStatus synSingleton::supportsComplexGuid()
{
    // allowing ComplexGuid to be disabled only if ALLOW_DISABLED_CG=1
    if (GCFG_ALLOW_DISABLED_CG.value())
    {
        return synSuccess;
    }
    return synUnsupported;
}

synStatus synSingleton::getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress)
{
    GET_DEV_QMAN_RTN_IF_ERR();
    return deviceQman->getCacheDeviceAddressRange(baseAddress, lastAddress);
}

synStatus synSingleton::generateApiId(uint8_t& rApiId)
{
    GET_DEV_INTERFACE_RTN_IF_ERR();
    rApiId = deviceInterface->generateApiId();
    return synSuccess;
}

synStatus synSingleton::syncHCLStreamHandle(synStreamHandle streamHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "streamHandle");

    std::shared_ptr<DeviceInterface> pDeviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{}: failed to get device", HLLOG_FUNC);
        return synInvalidArgument;
    }

    return pDeviceInterface->syncHCLStreamHandle(streamHandle);
}

synStatus synSingleton::isStreamInitialized(synStreamHandle streamHandle, bool& rIsInitialized)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "streamHandle");
    rIsInitialized = true;
    return synSuccess;
}

synStatus synSingleton::flushWaitsOnCollectiveStream(synStreamHandle streamHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "streamHandle");

    std::shared_ptr<DeviceInterface> pDeviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{}: failed to get device", HLLOG_FUNC);
        return synInvalidArgument;
    }

    return pDeviceInterface->flushWaitsOnCollectiveStream(streamHandle);
}

uint32_t synSingleton::getNetworkStreamPhysicalQueueOffset(synStreamHandle streamHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, streamHandle, "streamHandle");

    std::shared_ptr<DeviceInterface> pDeviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{}: failed to get device", HLLOG_FUNC);
        return synInvalidArgument;
    }

    return pDeviceInterface->getNetworkStreamPhysicalQueueOffset(streamHandle);
}

hcl::hclStreamHandle synSingleton::getNetworkStreamHclStreamHandle(synStreamHandle streamHandle)
{
    HB_ASSERT(streamHandle != nullptr, "Got invalid streamHandle {:#x}", streamHandle);

    std::shared_ptr<DeviceInterface> pDeviceInterface = m_deviceManager.getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        LOG_ERR(SYN_API, "{}: failed to get device", HLLOG_FUNC);
        HB_ASSERT(false, "Got invalid streamHandle {:#x}", streamHandle);
    }

    return pDeviceInterface->getNetworkStreamHclStreamHandle(streamHandle);
}

void synSingleton::notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo)
{
    GET_DEV_INTERFACE();
    if (deviceInterface == nullptr)
    {
        LOG_ERR(SYN_DEV_FAIL, "{} device is null", HLLOG_FUNC);
        return;
    }

    return deviceInterface->notifyHlthunkFailure(errCode, dfaExtraInfo);
}

synStatus synSingleton::getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                        synRecipeHandle             recipeHandle,
                                                        std::vector<tensor_info_t>& tensorInfoArray) const
{
    return synSuccess;
}

synStatus synSingleton::getDynamicShapesTensorInfoArrayV2(synStreamHandle             streamHandle,
                                                          synRecipeHandle             recipeHandle,
                                                          std::vector<tensor_info_t>& tensorInfoArray) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, streamHandle, "streamHandle");

    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->getDynamicShapesTensorInfoArray(streamHandle, recipeHandle, tensorInfoArray);
}

synStatus synSingleton::setStreamAffinity(const synDeviceId     deviceId,
                                          const synStreamHandle pStreamHandle,
                                          uint64_t              streamAffinityMask)
{
    LOG_SINGLETON_API();
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->setStreamAffinity(pStreamHandle, streamAffinityMask);
}

synStatus synSingleton::getStreamAffinity(const synDeviceId     deviceId,
                                          const synStreamHandle pStreamHandle,
                                          uint64_t*             streamAffinityMask)
{
    LOG_SINGLETON_API();
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->getStreamAffinity(pStreamHandle, streamAffinityMask);
}

synStatus synSingleton::getDeviceAffinityMaskRange(const synDeviceId deviceId, uint64_t* deviceAffinityMaskRange)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, deviceAffinityMaskRange, "deviceAffinityMaskRange");
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->getDeviceAffinityMaskRange(*deviceAffinityMaskRange);
}

synStatus synSingleton::getDeviceNextStreamAffinity(const synDeviceId deviceId, uint64_t* nextDeviceAffinity)
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, nextDeviceAffinity, "deviceAffinityMaskRange");
    GET_DEV_INTERFACE_RTN_IF_ERR();
    return deviceInterface->getDeviceNextStreamAffinity(*nextDeviceAffinity);
}

void synSingleton::elevateSynLaunchTensorInfo(synLaunchTensorInfoExt*    launchTensorsInfoExt,
                                              const synLaunchTensorInfo* launchTensorsInfo,
                                              uint32_t                   numberTensors)
{
    for (unsigned i = 0; i < numberTensors; i++)
    {
        launchTensorsInfoExt[i].tensorName     = launchTensorsInfo[i].tensorName;
        launchTensorsInfoExt[i].pTensorAddress = launchTensorsInfo[i].pTensorAddress;
        launchTensorsInfoExt[i].tensorType     = launchTensorsInfo[i].tensorType;
        launchTensorsInfoExt[i].tensorId       = launchTensorsInfo[i].tensorId;
        for (unsigned j = 0; j < HABANA_DIM_MAX; j++)
        {
            launchTensorsInfoExt[i].tensorSize[j] = (uint64_t)launchTensorsInfo[i].tensorSize[j];
        }
    }
}

void synSingleton::lowerSynRetrievedLaunchTensorInfoExt(const synRetrievedLaunchTensorInfoExt* launchTensorsInfoExt,
                                                        synRetrievedLaunchTensorInfo*          launchTensorsInfo,
                                                        uint32_t                               numberTensors)
{
    for (unsigned i = 0; i < numberTensors; i++)
    {
        memcpy(launchTensorsInfo[i].tensorName, launchTensorsInfoExt[i].tensorName, ENQUEUE_TENSOR_NAME_MAX_SIZE);
        memcpy(launchTensorsInfo[i].tensorPermutation, launchTensorsInfoExt[i].tensorPermutation, HABANA_DIM_MAX);
        memcpy(launchTensorsInfo[i].reserved, launchTensorsInfoExt[i].reserved, 10);
        launchTensorsInfo[i].tensorId              = launchTensorsInfoExt[i].tensorId;
        launchTensorsInfo[i].tensorType            = launchTensorsInfoExt[i].tensorType;
        launchTensorsInfo[i].tensorDataType        = launchTensorsInfoExt[i].tensorDataType;
        launchTensorsInfo[i].tensorDims            = launchTensorsInfoExt[i].tensorDims;
        launchTensorsInfo[i].tensorSectionId       = launchTensorsInfoExt[i].tensorSectionId;
        launchTensorsInfo[i].tensorOffsetInSection = launchTensorsInfoExt[i].tensorOffsetInSection;
        launchTensorsInfo[i].isInput               = launchTensorsInfoExt[i].isInput;
        for (unsigned j = 0; j < HABANA_DIM_MAX; j++)
        {
            launchTensorsInfo[i].tensorMaxSize[j] = (uint32_t)launchTensorsInfoExt[i].tensorMaxSize[j];
            launchTensorsInfo[i].tensorMinSize[j] = (uint32_t)launchTensorsInfoExt[i].tensorMinSize[j];
        }
    }
}

void synSingleton::lowerTensorMetadataInfoExt(const TensorMetadataInfoExt* tensorsMetadataInfoExt,
                                              TensorMetadataInfo*          tensorsMetadataInfo,
                                              uint32_t                     numberTensors)
{
    for (unsigned i = 0; i < numberTensors; i++)
    {
        memcpy(tensorsMetadataInfo[i].layout, tensorsMetadataInfoExt[i].layout, MAX_LAYOUT_SIZE);
        tensorsMetadataInfo[i].tensorName      = tensorsMetadataInfoExt[i].tensorName;
        tensorsMetadataInfo[i].elementType     = tensorsMetadataInfoExt[i].elementType;
        tensorsMetadataInfo[i].zp              = tensorsMetadataInfoExt[i].zp;
        tensorsMetadataInfo[i].scale           = tensorsMetadataInfoExt[i].scale;
        tensorsMetadataInfo[i].dimensions      = tensorsMetadataInfoExt[i].dimensions;
        tensorsMetadataInfo[i].roiSizeInBytes  = tensorsMetadataInfoExt[i].roiSizeInBytes;
        tensorsMetadataInfo[i].batchSize       = tensorsMetadataInfoExt[i].batchSize;
        tensorsMetadataInfo[i].isInput         = tensorsMetadataInfoExt[i].isInput;
        tensorsMetadataInfo[i].sectionId       = tensorsMetadataInfoExt[i].sectionId;
        tensorsMetadataInfo[i].offsetInSection = tensorsMetadataInfoExt[i].offsetInSection;
        for (unsigned j = 0; j < HABANA_DIM_MAX; j++)
        {
            tensorsMetadataInfo[i].dimensionsSize[j] = (uint32_t)tensorsMetadataInfoExt[i].dimensionsSize[j];
        }
    }
}

void synSingleton::setSynRetrievedLaunchTensorInfoExtIDs(synRetrievedLaunchTensorInfoExt*    launchTensorsInfoExt,
                                                         const synRetrievedLaunchTensorInfo* launchTensorsInfo,
                                                         uint32_t                            numberTensors)
{
    for (unsigned i = 0; i < numberTensors; i++)
    {
        launchTensorsInfoExt[i].tensorId = launchTensorsInfo[i].tensorId;
    }
}

void synSingleton::setTensorsMetadataInfoNamesExt(TensorMetadataInfoExt*    tensorsMetadataInfoExt,
                                                  const TensorMetadataInfo* tensorsMetadataInfo,
                                                  uint32_t                  numberTensors)
{
    for (unsigned i = 0; i < numberTensors; i++)
    {
        tensorsMetadataInfoExt[i].tensorName = tensorsMetadataInfo[i].tensorName;
    }
}

synStatus synSingleton::setSmfCallbacks(smf_callbacks_t* callbacks)
{
    SmfCallbacks::set(callbacks);

    return synSuccess;
}

synStatus synSingleton::convertStatusToString(synStatus status, char* statusDescription, size_t len)
{
    VERIFY_IS_NULL_POINTER(SYN_API, statusDescription, "Status Description");
    return statusToString(status, statusDescription, len);
}

synStatus synSingleton::getDeviceAttributesByModuleId(const synModuleId         moduleId,
                                                      const synDeviceAttribute* deviceAttr,
                                                      const unsigned            querySize,
                                                      uint64_t*                 retVal) const
{
    return m_deviceManager.getDeviceAttributesByModuleId(moduleId, deviceAttr, querySize, retVal);
}

synStatus synSingleton::setHostProfilerArg(const std::vector<synTraceEventArg>& keyValArgs)
{
    return synUnsupported;
}

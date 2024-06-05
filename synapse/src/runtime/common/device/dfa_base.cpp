#if !defined __GNUC__ || __GNUC__ >= 8
#include <filesystem>
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "defs.h"
#include "hlthunk.h"
#include "dfa_base.hpp"

#include "defenders.h"
#include "device_common.hpp"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "drm/habanalabs_accel.h"
#include "syn_singleton.hpp"
#include "synapse_common_types.h"
#include "dfa_observer.hpp"
#include "infra/api_calls_counter.hpp"

#include <iostream>
#include <stdint.h>
#include <string>
#include <sys/klog.h>
#include <sys/sysinfo.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <signal.h>
#include "runtime/common/osal/osal.hpp"
#include "dfa_read_gaudi_regs.hpp"
#include "dfa_read_gaudi2_regs.hpp"
#include "dfa_read_gaudi3_regs.hpp"
#include "syn_event_dispatcher.hpp"
#include "graph_traits.h"
#include "timer.h"

// Sleep for some time between the end of DFA and killing the process. This lets user
// threads return to the user with an error message and lets the user handle (for example, log) before
// killing the process
const unsigned DfaBase::SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US = 5000000;
const uint32_t DfaBase::HBM_READ_CHUNK_SIZE                              = 1024 * 16;   // 16KB - LKD limitation
const uint32_t DfaBase::MAX_KERNEL_SIZE                                  = 1024 * 1024; // 1MB

struct DeviceConstructInfo;

/**
 * returns a string with the error code name
 * @param errCode error code value
 * @return error code name
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch-enum"
static const char* dfaErrorToName(DfaErrorCode errCode)
{
#define DFA_CASE(X)          \
    case DfaErrorCode::X:    \
        return #X;

    switch (errCode)
    {
        DFA_CASE(noError);
        DFA_CASE(tdrFailed);
        DFA_CASE(eventSyncFailed);
        DFA_CASE(streamSyncFailed);
        DFA_CASE(getDeviceInfoFailed);
        DFA_CASE(waitForCsFailed);
        DFA_CASE(hlthunkDebugFailed);
        DFA_CASE(undefinedOpCode);
        DFA_CASE(getTimeSyncInfoFailed);
        DFA_CASE(waitForMultiCsFailed);
        DFA_CASE(memcopyIoctlFailed);
        DFA_CASE(requestCommandBufferFailed);
        DFA_CASE(destroyCommandBufferFailed);
        DFA_CASE(commandSubmissionFailed);
        DFA_CASE(stagedCsFailed);
        DFA_CASE(getDeviceClockRateFailed);
        DFA_CASE(getPciBusIdFailed);
        DFA_CASE(usrEngineErr);
        DFA_CASE(hclFailed);
        DFA_CASE(csTimeout);
        DFA_CASE(razwi);
        DFA_CASE(mmuPageFault);
        DFA_CASE(scalTdrFailed);
        DFA_CASE(assertAsync);
        DFA_CASE(generalHwError);
        DFA_CASE(criticalHwError);
        DFA_CASE(criticalFirmwareError);
        DFA_CASE(signal);

        DFA_CASE(tpcBrespErr);
        DFA_CASE(waitForMultiCsTimedOut);
        DFA_CASE(deviceReset);
        DFA_CASE(deviceUnavailble);
        default:
            return "UNKNOWN_DFA_ERROR_CODE";
    }
#undef DFA_CASE
}
#pragma GCC diagnostic pop

DfaBase::DfaBase(const DeviceConstructInfo& deviceConstructInfo, DeviceCommon* deviceCommon)
: m_devHlIdx(deviceConstructInfo.hlIdx),
  m_fdCompute(deviceConstructInfo.deviceInfo.fd),
  m_fdControl(deviceConstructInfo.fdControl),
  m_devModuleIdx(deviceConstructInfo.devModuleIdx),
  m_devType(deviceConstructInfo.deviceInfo.deviceType),
  m_devIdType(static_cast<hl_pci_ids>(deviceConstructInfo.devIdType)),
  m_devCommon(deviceCommon),
  m_osalInfo(deviceCommon->getDeviceOsalInfo())
{
    int rtn = gethostname(m_serverName, kserverNameSize);
    if (rtn)
    {
        LOG_WARN(SYN_DEV_FAIL, "Failed reading host name with rtn {} errno {}", rtn, errno);
        strcpy(m_serverName, "Failed gethostname");
    }

    int ret = hlthunk_get_hw_ip_info(m_fdCompute, &m_hwIp);

    if (ret < 0)
    {
        LOG_WARN(SYN_DEV_FAIL, "Failed reading hwIp with ret {} errno {}", ret, errno);
        m_hwIp.module_id = -1;
    }
}

void DfaBase::dumpEngStatus()
{
    constexpr uint32_t      size = 128 * 1024;  // should be big enough for the output
    std::unique_ptr<char[]> buff(new char[size] {});

    int actualSize;
    int rtn = hlthunk_get_engine_status(m_fdCompute, buff.get(), size, &actualSize);

    // For now, returned char array might have the value zero in it. Changing them to spaces.
    // LKD should fix it in the future
    for (int i = 0; i < actualSize; i++)
    {
        if (buff[i] == 0)
        {
            buff[i] = ' ';
        }
    }

    if (rtn != 0)
    {
        LOG_TRACE(SYN_DEV_FAIL,
                  "reading engine status failed, can happen on simulator. rtn {} buffer size {}",
                  rtn,
                  size);
    }
    else
    {
        LOG_TRACE(SYN_DEV_FAIL, "actualSize of engine dump {}\n{}", actualSize, buff.get());
    }
}

constexpr int dataSizeBytes = sizeof(uint32_t);

static int getRegNum(uint32_t sizeInBytes)
{
    return sizeInBytes / dataSizeBytes;
}
static uint32_t getSizeBytes(int numRegs)
{
    return numRegs * dataSizeBytes;
}
/*
 ***************************************************************************************************
 *   @brief readRegs() Reads a block of registers (addr, size) and dumps them to SYN_DEV_FAIL log.
 *                     The function calls another function to do the actual read into a given buffer
 *   @param  addr, size - block of registers to be read. Size is in bytes
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::readRegs(uint64_t addr, uint32_t size)
{
    std::unique_ptr<uint32_t[]> buff(new uint32_t[size] {});

    if (GCFG_DFA_READ_REG_MODE.value() == static_cast<uint64_t>(ReadRegMode::lkd))
    {
        int readRes = hlthunk_device_memory_read_block_experimental(m_fdCompute, buff.get(), addr, size, 0);
        if (readRes)
        {
            LOG_ERR_T(SYN_DEV_FAIL,
                      "Failed reading registers fd {} buff {:x} addr {:x} size {:x} readRes {} {}",
                      m_fdCompute,
                      TO64(buff.get()),
                      addr,
                      size,
                      readRes,
                      strerror(-readRes));
            return;
        }
    }
    else if (GCFG_DFA_READ_REG_MODE.value() == static_cast<uint64_t>(ReadRegMode::skip))
    {
        LOG_INFO(SYN_DEV_FAIL, "skip flag set, not reading addr {:x} size {:x}", addr, size);
        return;
    }

    std::string regS;

    regS += fmt::format(FMT_COMPILE("#registers {:x} {:x}\n"), addr, size);

    for (int i = 0; i < getRegNum(size); i++)
    {
        if ((i % REGS_PER_LINE) == 0)
        {
            regS += fmt::format(FMT_COMPILE("{:x}: "), addr + getSizeBytes(i));
        }

        regS += fmt::format(FMT_COMPILE("{:08x} "), buff[i]);
        if (((i % REGS_PER_LINE) == (REGS_PER_LINE - 1)) && (i != (getRegNum(size) - 1)))
        {
            regS += "\n";
        }
    }
    LOG_INFO_T(SYN_DEV_FAIL, "\n{}", regS);
}

bool DfaBase::addObserver(DfaObserver* pDfaObserver)
{
    CHECK_POINTER(SYN_DEV_FAIL, pDfaObserver, "DFA-Observer", false);

    std::unique_lock<std::mutex> guard(m_dfaObserversMutex);

    if (std::find(m_dfaObservers.begin(), m_dfaObservers.end(), pDfaObserver) != m_dfaObservers.end())
    {
        LOG_DEBUG(SYN_DEV_FAIL, "Observer 0x{:x} already exist in DB", (uint64_t)pDfaObserver);
        return false;
    }

    m_dfaObservers.push_back(pDfaObserver);
    return true;
}

bool DfaBase::removeObserver(DfaObserver* pDfaObserver)
{
    CHECK_POINTER(SYN_DEV_FAIL, pDfaObserver, "DFA-Observer", false);

    std::unique_lock<std::mutex> guard(m_dfaObserversMutex);

    auto dfaObserverIter = std::find(m_dfaObservers.begin(), m_dfaObservers.end(), pDfaObserver);
    if (dfaObserverIter == m_dfaObservers.end())
    {
        LOG_DEBUG(SYN_DEV_FAIL, "Observer 0x{:x} does not exist in DB", (uint64_t)pDfaObserver);
        return false;
    }

    m_dfaObservers.erase(dfaObserverIter);

    return true;
}

/*
 ***************************************************************************************************
 *   @brief logDeviceInfo() Logs basic information about the device (fd, etc.)
 *   @param  logType
 *   @param  logLevel
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::logDeviceInfo(synapse::LogManager::LogType logType)
{
    const char* rankId = getenv("OMPI_COMM_WORLD_RANK");
    if (rankId == nullptr)
    {
        rankId = getenv("ID");
    }

    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "Failure occurred on device:");
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "     #device name:     hl{}", m_devHlIdx);
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "     Moudle index:       {}", m_devModuleIdx);
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "     fd compute/control: {}/{}", m_fdCompute, m_fdControl);
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "     #global rank Id:    {}", rankId ? rankId : "---");
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "#device type {}", OSAL::GetDeviceNameByDevType(m_devType));

    bool isSimulator = isSimulatorFunc(m_devIdType);
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "#is simulator  {}", isSimulator ? "Yes" : "No");
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "#acquire time: {}", TimeTools::timePoint2string(m_devCommon->getAcquireTime()));
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "pci addr:      {}", m_devCommon->getPciAddr());
    SYN_LOG(logType, SPDLOG_LEVEL_INFO, "Server name:   {}, ModuleId {}", m_serverName, m_hwIp.module_id);
}

/*
 ***************************************************************************************************
 *   @brief logDevEngines() Logs engine masks)
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::logDevEngines()
{
    GraphTraits graphTraits(m_devType);

    auto halReader = graphTraits.getHalReader();

    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::ENG_MASKS));

    LOG_TRACE(SYN_DEV_FAIL, "TPC_HAL_MASK:   {:x}", halReader->getTpcEnginesMask());
    LOG_TRACE(SYN_DEV_FAIL, "TPC_GCFG_MASK:  {:x}", GCFG_TPC_ENGINES_ENABLED_MASK.value());
    LOG_TRACE(SYN_DEV_FAIL, "#TPC_MASK:      {:x}", halReader->getTpcEnginesMask() & GCFG_TPC_ENGINES_ENABLED_MASK.value());

    LOG_TRACE(SYN_DEV_FAIL, "#MME_NUM:       {:x}", halReader->getNumMmeEngines());
    LOG_TRACE(SYN_DEV_FAIL, "#DMA_MASK:      {:x}", halReader->getInternalDmaEnginesMask());
    LOG_TRACE(SYN_DEV_FAIL, "#ROT_NUM:       {:x}", halReader->getNumRotatorEngines());
}

/*
 ***************************************************************************************************
 *   @brief notifyHlthunkFailure() This function is called from relevant hlthunk calls that fail.
 *          It is called usually from device-common notifyHlthunkFailure
 *   @param  DfaErrorCode - indicates where it failed
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo)
{
    if (errCode == DfaErrorCode::signal)
    {
        LOG_ERR(SYN_DEV_FAIL, "Signal was detected");
    }
    else
    {
        LOG_ERR(SYN_DEV_FAIL, "hl thunk API failed, errCode {}", dfaErrorToName(errCode));

        if ((errCode == DfaErrorCode::scalTdrFailed) ||
            (errCode == DfaErrorCode::tdrFailed))
        {
            ; // in case of a timeout, no need to wait for the LKD to trigger DFA - it is a software decision that
              // LKD is not aware of
        }
        else
        {
            // We prefer the event-fd to start the dfa process because the LKD collects information for us.
            // So sleep for some time to allow the event-fd to trigger first
            std::this_thread::sleep_for(dfaHlthunkTriggerDelay);
        }
    }

    return notifyDeviceFailure(errCode, dfaExtraInfo);
}

/*
 ***************************************************************************************************
 *   @brief notifyDeviceFailure() This function is called when a device failure is detected. It is called from
 *          device-common notifyDeviceFailure or from this class, notifyHlthunkFailure()
 *   1) hlthunk failure (from  notifyHlthunkFailure)
 *   2) Event fd error
 *   3) gaudi2/3 timeout
 *   @param  DfaErrorCode - indicates where it failed
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::notifyDeviceFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo)
{
    DfaStatus dfaStatus = {};
    dfaStatus.addError(errCode);

    checkFailure(dfaStatus, dfaExtraInfo);
}

/*
 ***************************************************************************************************
 *   @brief logUserMsg() This function logs a user-friendly message in the DFA log
 *   @param  DfaStatus - indicates current errors
 *   @return None
 *
 ***************************************************************************************************
*/
void DfaBase::logUserMsg(DfaStatus dfaStatus)
{
    std::string usrMsg = "\n\n\n";
    usrMsg += fmt::format(FMT_COMPILE("{:=^120}"), " DFA triggered on the following errors ");
    usrMsg += "\n\n";

    bool firstErr    = true;
    bool isSimulator = isSimulatorFunc(m_devIdType);

    // special case for simulator
    if (isSimulator &&
        dfaStatus.hasOnlyErrors({ DfaErrorCode::deviceUnavailble, DfaErrorCode::deviceReset, DfaErrorCode::criticalHwError }))
    {
        usrMsg += "Simulator crashed. Please check simulator logs for details.";
        firstErr = false;
    }
    else
    {
        for (int i = 0; i < (sizeof(DfaErrorCode) * 8); i++)
        {
            DfaErrorCode errCode = DfaErrorCode((uint64_t)1 << i);
            if (dfaStatus.hasError(errCode))
            {
                if (!firstErr)
                {
                    usrMsg += "\n\n";
                }
                firstErr = false;
                usrMsg += getDfaTriggerMsg(errCode);
            }
        }
    }

    usrMsg += "\n\n";
    usrMsg += std::string(120, '=');
    usrMsg += "\n\n\n";
    LOG_ERR(SYN_DEV_FAIL, "{}", usrMsg);
}

/**
 * Dumps info about the failure, and update DFA class members.
 * Lets only the first error dump the information
 * @param errCode DFA error code
 */
void DfaBase::checkFailure(DfaStatus dfaStatus, const DfaExtraInfo& dfaExtraInfo)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_isAssertAsyncNoticed)
    {
        return;
    }
    // logging in both critical and error level, in case someone looks only for errors
    LOG_CRITICAL_T(SYN_API, "DFA detected, see separate file for details {}", DEVICE_FAIL_ANALYSIS_FILE);
    LOG_ERR_T(SYN_API, "DFA detected, see separate file for details {}", DEVICE_FAIL_ANALYSIS_FILE);

    raiseIrqUnexpectedInterrupt();

    auto now = std::chrono::system_clock::now();
    if (m_dfaStatus.isSuccess())
    {
        m_dfaStatus.setFirstErrTime(now);  // if first error, set the firstErrTime
    }

    if (!m_isErrorLogged)
    {
        logDfaBegin(synapse::LogManager::LogType::SYN_DEV_FAIL);
        logDfaBegin(synapse::LogManager::LogType::SYN_FAIL_RECIPE);
        logDfaBegin(synapse::LogManager::LogType::DFA_NIC);

        m_dfaExtraInfo = dfaExtraInfo;
        logUserMsg(dfaStatus);
    }
    // log new error/errors
    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::FAIL_INFO));
    LOG_TRACE(SYN_DEV_FAIL, "--- Errors detected ---");
    for (int i = 0; i < (sizeof(DfaErrorCode) * 8); i++)
    {
        DfaErrorCode errCode = DfaErrorCode((uint64_t)1 << i);
        if (dfaStatus.hasError(errCode))
        {
            LOG_ERR(SYN_DEV_FAIL, "#DFA reason: {:20} (code {:#x})", dfaErrorToName(errCode), (uint64_t)errCode);
            m_dfaStatus.addError(errCode);
        }
    }
    LOG_ERR(SYN_DEV_FAIL, "\n");

    bool started = false;
    if (!m_isErrorLogged)
    {
        if ((GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::busyWait) ||
            (GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::synapseTermination))
        {
            // Notify synapse that dfa has started. It will block new api-s until collection is done
            LOG_INFO_T(SYN_API, "dfa: sending DfaPhase::STARTED");
            synEventDispatcher.send(EventDfaPhase {DfaPhase::STARTED});
            started = true;
        }

        synSingleton::printVersionToLog(synapse::LogManager::LogType::SYN_DEV_FAIL, "Habana Labs Device failure analysis");

        logDeviceInfo(synapse::LogManager::LogType::SYN_DEV_FAIL);

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::ENG_STATUS));
        dumpEngStatus();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::LKD_EVENTS));

        getDevErrorsInfoFromLkd(0);  // for timeout, this also fill m_csSeqTimeout

        bool isSimulator = isSimulatorFunc(m_devIdType);

        m_devCommon->checkDevFailure(m_csSeqTimeout, m_dfaStatus, DeviceCommon::ChkDevFailOpt::MAIN, isSimulator);

        // TBD - define parameters specifically to the error information
        uint32_t failureType       = 1;
        uint64_t suspectedCsHandle = 0;
        _notifyFailure(failureType, suspectedCsHandle);

        logDevEngines();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::DMESG));
        logDmesg();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::HW_IP));
        logHwIp();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::GCFG));
        GlobalConfManager::instance().printGlobalConf(true);

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::DEV_MAP));
        SingleEvent singleEvent = getPageFaultEvent(PageFaultEventOpt::MAPPINGS);
        LOG_TRACE(SYN_DEV_FAIL, "{}", singleEvent.eventInfo.description);

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::SW_MAP_MEM));
        m_devCommon->dfaLogMappedMem();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::HL_SMI));
        logHlSmi();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::TPC_KERNEL));
        dumpTpcKernels();

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::SYNAPSE_API_COUNTERS));
        logSynapseApiCounters();

        if (GCFG_DFA_COLLECT_CCB.value())
        {
            m_devCommon->checkDevFailure(m_csSeqTimeout, m_dfaStatus, DeviceCommon::ChkDevFailOpt::CCB, isSimulator);
        }
        else
        {
            LOG_TRACE(SYN_DEV_FAIL, "CCBs are not collected. Use env var {}=true", GCFG_DFA_COLLECT_CCB.primaryName());
        }

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::REGS));
        dumpDevRegs();

        LOG_INFO(SYN_DEV_FAIL, TITLE_STR(DfaMsg::DONE));

        LOG_TRACE(SYN_DEV_FAIL, "#DFA end");

        m_isErrorLogged = true;
    }

    // if not started, don't send ENDED
    if (started)
    {
        LOG_INFO_T(SYN_API, "dfa: sending DfaPhase::ENDED");
        synapse::LogManager::instance().flush();
        synEventDispatcher.send(EventDfaPhase {DfaPhase::ENDED});
    }

    // if we are in dfa from signal flow, do not terminate, just return
    if (dfaStatus.hasError(DfaErrorCode::signal))
    {
        return;
    }

    std::string consoleMsg = getConsoleMsg(dfaStatus);
   _handleSynapseTermination(consoleMsg);
}

/*
 ***************************************************************************************************
 *   @brief getConsoleMsg() This function returns a string to be output to screen on dfa failure
 *   @param  DfaErrorCode - error code
 *   @return string with message
 *   @Note: not all errors are covered, only most frequent ones. The function can't be implemented
 *   @      using switch as we need priority (message on the most critical one only)
 ***************************************************************************************************
 */
std::string DfaBase::getConsoleMsg(DfaStatus dfaStatus)
{
    if (dfaStatus.hasError(DfaErrorCode::tdrFailed))
    {
        return "Compute or dma timeout";
    }

    else if (dfaStatus.hasError(DfaErrorCode::scalTdrFailed))
    {
        return "No progress error";
    }

    else if (dfaStatus.hasError(DfaErrorCode::hclFailed))
    {
        return "HCL error";
    }

    else
    {
        return "please check log files for dfa cause";
    }
}
/*
 ***************************************************************************************************
 *   @brief getDfaTriggerMsg() This function returns a string with basic user-friendly message about the error
 *   @param  DfaErrorCode - error code
 *   @return string with message
 *
 ***************************************************************************************************
 */
std::string DfaBase::getDfaTriggerMsg(DfaErrorCode dfaErrorCode)
{
    const std::string CHK_DMESG_COPY             = std::string("Instead, you can check a copy of the dmesg: ") + DMESG_COPY_FILE;
    const std::string LKD_CALL_FAILED_COMMON_MSG = std::string("Most likely device was reset. Please check dmesg.\n") + CHK_DMESG_COPY;
    const std::string CHK_DMESG_COMMON_MSG       = std::string("Please check dmesg. ") + CHK_DMESG_COPY;

    switch (dfaErrorCode)
    {
        case DfaErrorCode::noError:  // should never happen
            return "";

        case DfaErrorCode::tdrFailed:
        {
            DfaExtraInfo::DfaExtraInfoMsg* dfaExtraInfoMsg = std::get_if<DfaExtraInfo::DfaExtraInfoMsg>(&m_dfaExtraInfo.extraInfo);
            std::string                    streams = (dfaExtraInfoMsg == nullptr) ? "????" : dfaExtraInfoMsg->msg;

            return "Timeout detected on compute/pdma streams: " + streams +
                   "\n"
                   "Check 'Engines Status' for more information on active engines\n"
                   "Check 'Oldest work in each stream' for streams status";
        }

        case DfaErrorCode::eventSyncFailed:
            return "Event synchronization failed\n" + LKD_CALL_FAILED_COMMON_MSG;

        case DfaErrorCode::streamSyncFailed:
            return "Stream synchronization failed\n" + LKD_CALL_FAILED_COMMON_MSG;

        case DfaErrorCode::getDeviceInfoFailed:
        case DfaErrorCode::waitForCsFailed:
        case DfaErrorCode::hlthunkDebugFailed:
        case DfaErrorCode::getTimeSyncInfoFailed:
        case DfaErrorCode::waitForMultiCsFailed:
        case DfaErrorCode::memcopyIoctlFailed:
        case DfaErrorCode::requestCommandBufferFailed:
        case DfaErrorCode::destroyCommandBufferFailed:
        case DfaErrorCode::commandSubmissionFailed:
        case DfaErrorCode::stagedCsFailed:
        case DfaErrorCode::getDeviceClockRateFailed:
        case DfaErrorCode::getPciBusIdFailed:
        case DfaErrorCode::waitForMultiCsTimedOut:
            return "An hl-thunk call failed\n" + LKD_CALL_FAILED_COMMON_MSG;

        case DfaErrorCode::undefinedOpCode:
            return "An undefined Op-Code was detected";

        case DfaErrorCode::csTimeout:
            return "cs timeout detected\n"
                   "Check 'Engines Status' for more information on active engines\n"
                   "Check 'Oldest work in each stream' for stream status and oldest stuck work on each stream";

        case DfaErrorCode::razwi:
        case DfaErrorCode::mmuPageFault:
            return "Access to an illegal address was detected. " + CHK_DMESG_COMMON_MSG;

        case DfaErrorCode::scalTdrFailed:
        {
            DfaExtraInfo::DfaExtraInfoMsg* dfaExtraInfoMsg = std::get_if<DfaExtraInfo::DfaExtraInfoMsg>(&m_dfaExtraInfo.extraInfo);
            std::string                    streams         = (dfaExtraInfoMsg == nullptr) ? "????" : dfaExtraInfoMsg->msg;

            return std::string("No progress on any stream for a long time. Suspected streams " + streams + "\n") +
                   "Check '" + DfaMsg::ENG_STATUS +
                   "' for more information on active engines\n"
                   "Check '" +
                   DfaMsg::NO_PROGRESS_TDR + "' section to see streams with unfinished work";
        }

        case DfaErrorCode::assertAsync:
            return "assert-Async detected";

        case DfaErrorCode::generalHwError:
            return "General device error detected";

        case DfaErrorCode::criticalHwError:
            return "Critical HW error detected";

        case DfaErrorCode::criticalFirmwareError:
            return "Critical Firmware error detected";

        case DfaErrorCode::signal:
        {
            DfaExtraInfo::DfaExtraInfoSignal* dfaExtraInfoSignal = std::get_if<DfaExtraInfo::DfaExtraInfoSignal>(&m_dfaExtraInfo.extraInfo);

            std::string rtn = "A Signal was received.";

            if (dfaExtraInfoSignal)
            {
                return rtn +
                       " Signal " + std::to_string(dfaExtraInfoSignal->signal) +
                       " " + dfaExtraInfoSignal->signalStr + ".  " +
                       "Severity: " + (dfaExtraInfoSignal->isSevere ? "high" : "low");
            }
            else
            {
                return rtn + " Signal information not collected";
            }
        }

        case DfaErrorCode::tpcBrespErr:
            return "TPC bad respose detected";

        case DfaErrorCode::deviceReset:
            return "HW error interrupt - Device must be restarted\n" + CHK_DMESG_COMMON_MSG;

        case DfaErrorCode::deviceUnavailble:
            return "HW error interrupt - Device is not available anymore\n" + CHK_DMESG_COMMON_MSG + "\n\n" +
                   "---> NOTE: REGISTERS ARE NOT AVAILABLE IN THIS DEVICE STATE <---";

        case DfaErrorCode::usrEngineErr:
            return "HW error interrupt - User engine error\n" + CHK_DMESG_COMMON_MSG;

        case DfaErrorCode::hclFailed:
            DfaExtraInfo::DfaExtraInfoMsg* dfaExtraInfoMsg = std::get_if<DfaExtraInfo::DfaExtraInfoMsg>(&m_dfaExtraInfo.extraInfo);

            if (dfaExtraInfoMsg == nullptr || dfaExtraInfoMsg->msg.empty())
            {
                return "HCL triggered DFA, check HCL log for error details.";
            }
            else
            {
                return "HCL triggered DFA, cause: " + dfaExtraInfoMsg->msg;
            }
    }
    return "Missing Description for " + std::to_string(static_cast<uint64_t>(dfaErrorCode)); // we shoild never get here
}

/*
 ***************************************************************************************************
 *   @brief logDfaBegin() Logs a tag that should be common in all dfa files
 *
 *   @param  logType - where to log the tag
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::logDfaBegin(synapse::LogManager::LogType logType)
{
    SYN_LOG(logType,
            SPDLOG_LEVEL_INFO,
            "tid {} #DFA begin {}",
            syscall(__NR_gettid),
            m_dfaStatus.getFirstErrTime().time_since_epoch().count());
}

/*
 ***************************************************************************************************
 *   @brief notifyEventFd() This function is called when an eventFd is triggered.
 *          It sets the relevant error codes and calls the checkFailure() function
 *
 *   @param  events bit-map that were triggered
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::notifyEventFd(uint64_t events)
{
    DfaStatus dfaStatus = {};

    // Special case: TPC assert and asyncBuffer is not 0
    if (events & HL_NOTIFIER_EVENT_TPC_ASSERT)
    {
        LOG_TRACE_T(SYN_DEV_FAIL, "{} events 0x{:x}", HLLOG_FUNC, events);

        uint32_t numberOfBytesToCompare = 256;
        char     zeroArray[numberOfBytesToCompare];
        memset(zeroArray, 0, numberOfBytesToCompare);

        if (memcmp(zeroArray, m_assertAsyncBuffer, numberOfBytesToCompare) != 0)
        {
            m_isAssertAsyncNoticed = true;
            LOG_CRITICAL(SYN_API,
                         "Async assert has been notified with values: node-id:{} msg-id:{}",
                         *((uint32_t*)m_assertAsyncBuffer),
                         *((uint32_t*)m_assertAsyncBuffer + 1));

            memset(m_assertAsyncBuffer, 0, numberOfBytesToCompare);
            _handleSynapseTermination("TPC assert");
            return;
        }
    }

    // Start DFA flow only if HL_NOTIFIER_EVENT_DEVICE_RESET is set
    if (events & HL_NOTIFIER_EVENT_DEVICE_RESET)
    {
        LOG_TRACE_T(SYN_DEV_FAIL, "{} events 0x{:x}", HLLOG_FUNC, events);

        dfaStatus.addError(DfaErrorCode::deviceReset);

#define EVENT_ERR_HANDLING(eventNotification, dfaErrCode)                                                             \
        if (events & eventNotification)                                                                               \
        {                                                                                                             \
            dfaStatus.addError(dfaErrCode);                                                                           \
        }

        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_TPC_ASSERT,         DfaErrorCode::assertAsync);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_UNDEFINED_OPCODE,   DfaErrorCode::undefinedOpCode);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_CS_TIMEOUT,         DfaErrorCode::csTimeout);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_DEVICE_UNAVAILABLE, DfaErrorCode::deviceUnavailble);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_USER_ENGINE_ERR,    DfaErrorCode::usrEngineErr);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_GENERAL_HW_ERR,     DfaErrorCode::generalHwError);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_CRITICL_HW_ERR,     DfaErrorCode::criticalHwError);
        EVENT_ERR_HANDLING(HL_NOTIFIER_EVENT_CRITICL_FW_ERR,     DfaErrorCode::criticalFirmwareError);

        checkFailure(dfaStatus, {});
    }
    else
    // just log the events from lkd
    {
        getDevErrorsInfoFromLkd(events);
    }
}

DfaBase::SingleEvent DfaBase::getOpenEvent()
{
    hlthunk_event_record_open_dev_time openDevEvent {};
    // Get last device power on time
    int res = hlthunk_get_event_record(m_fdControl, hlthunk_event_record_id::HLTHUNK_OPEN_DEV, (void*)&openDevEvent);
    if (res != 0)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed reading open-event res {} errno {}", res, errno);
        return { { DfaEventInfoEnum::OPEN_DEV, "Failed reading"}, 0};
    }

    return { { DfaEventInfoEnum::OPEN_DEV,
             fmt::format(FMT_COMPILE("Device-opened at {}"), openDevEvent.timestamp)},
            openDevEvent.timestamp };
}

DfaBase::SingleEvent DfaBase::getUndefOpCodeEvent()
{
    hlthunk_event_record_undefined_opcode undefinedOpcodeEvent {};

    int res = hlthunk_get_event_record(m_fdCompute, hlthunk_event_record_id::HLTHUNK_UNDEFINED_OPCODE, (void*)&undefinedOpcodeEvent);
    if (res != 0)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed getting undefOpCode event res {} errno {}", res, errno);
        return {{DfaEventInfoEnum::UNDEFINED_OPCODE, "Failed reading"}, 0};
    }

    std::string desc = fmt::format(FMT_COMPILE("undefined opcode: timestamp: {}, engine_id: 0x{:x}, stream_id: 0x{:x},"
                                               " cq_size: 0x{:x}, cq_addr: 0x{:x}\n suspected addresses:\n"),
                                               undefinedOpcodeEvent.timestamp,
                                               undefinedOpcodeEvent.engine_id,
                                               undefinedOpcodeEvent.stream_id,
                                               undefinedOpcodeEvent.cq_size,
                                               undefinedOpcodeEvent.cq_addr);

    uint32_t streamsWithData = undefinedOpcodeEvent.cb_addr_streams_len;
    if (streamsWithData == MAX_QMAN_STREAMS_INFO)
    {
        desc += "undefined opcode occoured on lower cp!\n";
    }
    for (int streamIndex = 0; streamIndex < streamsWithData; streamIndex++)
    {
        for (int i = 0; i < OPCODE_INFO_MAX_ADDR_SIZE; i++)
        {
            if (undefinedOpcodeEvent.cb_addr_streams[streamIndex][i] != 0)
            {
                desc +=  fmt::format(FMT_COMPILE("stream: {}, 0x{:x}\n"),
                                                 streamIndex,
                                                 undefinedOpcodeEvent.cb_addr_streams[streamIndex][i]);
            }
        }
    }

    do
    {
        if (undefinedOpcodeEvent.timestamp == 0)
        {
            break;
        }

        // Print the region where the undefined opcode had been found
        uint64_t bufferSize = (undefinedOpcodeEvent.cq_size + dataSizeBytes - 1) / dataSizeBytes;
        if (bufferSize >= HBM_READ_CHUNK_SIZE)
        {
            LOG_WARN_T(SYN_DEV_FAIL, "Unexpectedly the undefined opcode HBM-range is too big (will not be printed)");
            break;
        }
        if (bufferSize == 0)
        {
            LOG_WARN_T(SYN_DEV_FAIL, "Unexpectedly the undefined opcode HBM-range has zero size (will not be printed)");
            break;
        }

        std::unique_ptr<uint32_t[]> dumpBuffer(new uint32_t[bufferSize] {});
        synStatus status = readHbmMemory(undefinedOpcodeEvent.cq_addr, dumpBuffer.get(), undefinedOpcodeEvent.cq_size);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEV_FAIL, "Failed getting undefined opcode buffer. Status {}", status);
        }
        else
        {
            std::string dump = fmt::format(FMT_COMPILE("#Undefined opcode {:#x} {:#x}\n"),
                                                    undefinedOpcodeEvent.cq_addr,
                                                    undefinedOpcodeEvent.cq_size);

            for (unsigned i = 0; i < bufferSize; i++)
            {
                if ((i % REGS_PER_LINE) == 0)
                {
                    dump += fmt::format(FMT_COMPILE("{:x}: "), undefinedOpcodeEvent.cq_addr + getSizeBytes(i));
                }

                dump += fmt::format(FMT_COMPILE("{:#x} "), dumpBuffer[i]);
                if (((i % REGS_PER_LINE) == (REGS_PER_LINE - 1)) &&
                    (i != (getRegNum(bufferSize) - 1)))
                {
                    dump += "\n";
                }
            }
            LOG_INFO_T(SYN_DEV_FAIL, "{}", dump);
        }
    } while (0); // Do once

    return { { DfaEventInfoEnum::UNDEFINED_OPCODE, desc }, undefinedOpcodeEvent.timestamp };
}

DfaBase::SingleEvent DfaBase::getCsTimeoutEvent(uint64_t& csSeqTimeout)
{
    hlthunk_event_record_cs_timeout csTimeoutEvent {};
    int res =
        hlthunk_get_event_record(m_fdCompute, hlthunk_event_record_id::HLTHUNK_CS_TIMEOUT, (void*)&csTimeoutEvent);
    if (res != 0)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed getting csTimeout event res {} errno {}", res, errno);
        return {{DfaEventInfoEnum::CS_TIMEOUT, "Failed reading"}, 0};
    }

    std::string desc = fmt::format(FMT_COMPILE("cs timeout: timestamp: {}, cs_id: 0x{:x} {}"),
                                               csTimeoutEvent.timestamp,
                                               csTimeoutEvent.seq,
                                               csTimeoutEvent.seq);

    csSeqTimeout = csTimeoutEvent.seq;

    return { { DfaEventInfoEnum::CS_TIMEOUT, desc}, csTimeoutEvent.timestamp };
}

DfaBase::SingleEvent DfaBase::getCriticalHwErrorEvent()
{
    hlthunk_event_record_critical_hw_err csCriticalHwErrorEvent {};
    int res = hlthunk_get_event_record(m_fdCompute,
                                       hlthunk_event_record_id::HLTHUNK_HW_ERR_OPCODE,
                                       (void*)&csCriticalHwErrorEvent);
    if (res != 0)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed getting Critical-HW-Error event res {} errno {}", res, errno);
        return {{DfaEventInfoEnum::CRITICAL_HW_ERR, "Failed reading"}, 0};
    }

    std::string desc = fmt::format(FMT_COMPILE("critical HW error: timestamp: {}, event_id: {:#x}"),
                                               csCriticalHwErrorEvent.timestamp,
                                               csCriticalHwErrorEvent.event_id);
    return { { DfaEventInfoEnum::CRITICAL_HW_ERR, desc}, csCriticalHwErrorEvent.timestamp };
}

DfaBase::SingleEvent DfaBase::getCriticalFirmwareErrorEvent()
{
    hlthunk_event_record_critical_fw_err csCriticalFirmwareErrorEvent {};
    int res = hlthunk_get_event_record(m_fdCompute,
                                       hlthunk_event_record_id::HLTHUNK_FW_ERR_OPCODE,
                                       (void*)&csCriticalFirmwareErrorEvent);
    if (res != 0)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed getting Critical-Firmware-Error event res {} errno {}", res, errno);
        return {{DfaEventInfoEnum::CRITICAL_FIRMWARE_ERR, "Failed reading"}, 0};
    }

    std::string desc =
        fmt::format(FMT_COMPILE("critical Firmware error: timestamp: {}, err_type: {}, reported_err_id: {:#x}"),
                                csCriticalFirmwareErrorEvent.timestamp,
                                csCriticalFirmwareErrorEvent.err_type,
                                csCriticalFirmwareErrorEvent.reported_err_id);
    return { { DfaEventInfoEnum::CRITICAL_FIRMWARE_ERR, desc}, csCriticalFirmwareErrorEvent.timestamp };
}

DfaBase::SingleEvent DfaBase::getRazwiEvent()
{
    hlthunk_event_record_razwi_event razwiEvent {};

    int res = hlthunk_get_event_record(m_fdControl, hlthunk_event_record_id::HLTHUNK_RAZWI_EVENT, (void*)&razwiEvent);
    if (res != 0)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed getting razwi event res {} errno {}", res, errno);
        return {{DfaEventInfoEnum::RAZWI, "Failed reading"}, 0};
    }

    std::string desc = fmt::format(FMT_COMPILE("razwi event: timestamp {} addr {:x} num_of_possible_engines {} flags {:x}\n"),
                                               razwiEvent.timestamp, razwiEvent.addr,
                                               razwiEvent.num_of_possible_engines, razwiEvent.flags);

    auto loopLimit = std::min(razwiEvent.num_of_possible_engines, (uint16_t)HL_RAZWI_MAX_NUM_OF_ENGINES_PER_RTR);

    for (int i = 0; i < loopLimit; i++)
    {
        desc += fmt::format(FMT_COMPILE("{}) {}\n"), i, razwiEvent.engine_id[i]);
    }

    return {{DfaEventInfoEnum::RAZWI, desc}, razwiEvent.timestamp};
}

std::string DfaBase::lkdEvent2str(DfaEventInfoEnum x)
{
    switch (x)
    {
        case DfaEventInfoEnum::OPEN_DEV:              return "OPEN_DEV";
        case DfaEventInfoEnum::CS_TIMEOUT:            return "CS_TIMEOUT";
        case DfaEventInfoEnum::RAZWI:                 return "RAZWI";
        case DfaEventInfoEnum::UNDEFINED_OPCODE:      return "UNDEFINED_OPCODE";
        case DfaEventInfoEnum::PAGE_FAULT:            return "PAGE_FAULT";
        case DfaEventInfoEnum::CRITICAL_HW_ERR:       return "CRITICAL_HW_ERR";
        case DfaEventInfoEnum::CRITICAL_FIRMWARE_ERR: return "CRITICAL_FIRMWARE_ERR";
    }
    return std::to_string((int)x) + "??? ";
}

/*
 ***************************************************************************************************
 *   @brief This function is called from the DFA flow or when an event not causing DFA is detected. In the
 *          second case we want to log only what is needed as events can be streaming and we don't want to
 *          overflow the log. We detect the flow the call is coming from by checking the eventsBitmap, if it
 *          is 0 then this is dfa flow.
 *          getDevErrorsInfoFromLkd() reads error information from LKD based on the events that were
 *          set. Then it orders the events based on time and dumps information about each event
 *          Note: the events were translated to dfaError codes
 *          1. When was the device opened
 *          2. If undefined-opcode, then get undefined-opcode into
 *          3. If cs-timeout, get information about the cs-timeout
 *          4. Always - check HLTHUNK_RAZWI_EVENT
 *
 *          It then orders the events by time and dumps the information
 *   @param  events bit-map that were triggered. Zero is an indication we are from the DFA flow, so read everything
 *   @return None
 *
 ***************************************************************************************************
 */
int DfaBase::getDevErrorsInfoFromLkd(uint64_t eventsBitmap)
{
    // LKD might trigger on some error although the root cause was another one.
    // For example: Page fault is causing ARC_AXI_ERROR_RESPONSE and only then PMMU0_PAGE_FAULT_WR_PERM. So the event-fd
    // is triggered by usrEngineErr+deviceReset and only then page-fault is set. When we read the lkd events we might
    // not have the page-fault info yet. Dani Liberman asked to add a sleep to avoid it. Suggested minimum of 100msec,
    // but 500msec is better

    // sleep only if not signal. If signal, it might be a terminated from MPI, we need to log as fast as possible and
    // the LKD events are not that important.
    if (!m_dfaStatus.hasError(DfaErrorCode::signal))
    {
        usleep(500000);
    }

    int64_t    devOpenTime = 0;
    const bool dfaFlow     = (eventsBitmap == 0);
    // a map between timestamp and event info struct
    std::map<uint64_t, HlThunkEventInfo> eventsInfo;

    if (dfaFlow) // collect only for DFA flow
    {
        SingleEvent singleEvent = getOpenEvent();
        if (singleEvent.timestamp != 0)
        {
            eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
        }
        devOpenTime   = singleEvent.timestamp;
        m_devOpenTime = devOpenTime; // for dfaFlow, keep the value for later use
    }

    if (dfaFlow ||
       (eventsBitmap & HL_NOTIFIER_EVENT_UNDEFINED_OPCODE))
    {
        SingleEvent singleEvent = getUndefOpCodeEvent();
        if (singleEvent.timestamp != 0)
        {
            eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
        }
    }

    if (dfaFlow ||
       (eventsBitmap & HL_NOTIFIER_EVENT_CS_TIMEOUT))
    {
        uint64_t csSeqTimeout = 0;
        // I can set m_csSeqTimeout inside the function, but I think it is more readable this way
        SingleEvent singleEvent = getCsTimeoutEvent(csSeqTimeout);

        if (dfaFlow) // m_csSeqTimeout is used only in dfa flow
        {
            m_csSeqTimeout = csSeqTimeout;
        }

        if ((singleEvent.timestamp != 0) && (singleEvent.timestamp >= devOpenTime))
        {
            eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
        }
    }

    if ((dfaFlow) ||
        (eventsBitmap & HL_NOTIFIER_EVENT_CRITICL_HW_ERR))
    {
        SingleEvent singleEvent = getCriticalHwErrorEvent();
        if (singleEvent.timestamp != 0)
        {
            eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
        }
    }

    if ((dfaFlow) ||
        (eventsBitmap & HL_NOTIFIER_EVENT_CRITICL_FW_ERR))
    {
        SingleEvent singleEvent = getCriticalFirmwareErrorEvent();
        if (singleEvent.timestamp != 0)
        {
            eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
        }
    }

    if (dfaFlow)
    {
        SingleEvent singleEvent = getPageFaultEvent(PageFaultEventOpt::DESCRITPION);
        if (singleEvent.timestamp != 0)
        {
            if (singleEvent.timestamp >= devOpenTime)
            {
                eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
            }
            else
            {
                LOG_TRACE(SYN_DEV_FAIL, "(Ignore: Found a page fault from previous run: {})", singleEvent.eventInfo.description);
            }
        }
    }

    if (dfaFlow)
    {
        SingleEvent singleEvent = getRazwiEvent();

        if (singleEvent.timestamp != 0)
        {
            if (singleEvent.timestamp >= devOpenTime)
            {
                eventsInfo[singleEvent.timestamp] = {singleEvent.eventInfo};
            }
            else
            {
                LOG_TRACE(SYN_DEV_FAIL, "(Ignore: Found a razwi from previous run: {})", singleEvent.eventInfo.description);
            }
        }
    }

    // go thorough the map and print/investigate the events
    if (eventsInfo.empty())
    {
        if (dfaFlow)
        {
            LOG_INFO(SYN_DEV_FAIL, "No LKD events were collected");
        }
        return 0;
    }

    for (const auto& iter : eventsInfo)
    {
        LOG_ERR(SYN_DEV_FAIL, "LKD event Id {} ({}): {}",
                lkdEvent2str(iter.second.eventId), (int)iter.second.eventId, iter.second.description);
    }
    return 0;
}

/*
 ***************************************************************************************************
 *   @brief getReadRegsInfo() Returns a reference to a vector defining the registers blocks to be read
 *
 *   @param  None
 *   @return Reference to a vector of the registers blocks
 *
 ***************************************************************************************************
 */
const std::vector<RegsBlock>& DfaBase::getReadRegsInfo()
{
    switch (m_devType)
    {
        case synDeviceGaudi:
            return regsToReadGaudi;

        case synDeviceGaudi2:
            return regsToReadGaudi2;

        case synDeviceGaudi3:
            return regsToReadGaudi3;

        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            LOG_ERR_T(SYN_DEV_FAIL, "No registers defined for device {}", m_devType);
            return m_emptyRegs;
    }
    return m_emptyRegs;
}

bool DfaBase::shouldReadRegBlock(const RegsBlock& regBlock, bool isSimulator, const EngMasks& engMasks)
{
    if (regBlock.blockType == BlockType::NONE)
    {
        return false;
    }
    if ((regBlock.blockType == BlockType::DEV_ONLY) && isSimulator)
    {
        return false;
    }
    if ((regBlock.blockType == BlockType::SIM_ONLY) && !isSimulator)
    {
        return false;
    }

    uint16_t engId = regBlock.engId;

    switch (regBlock.engGrp)
    {
        case EngGrp::NONE:
            return true;

        case EngGrp::TPC:
            return (1ULL << engId) & engMasks.tpc;

        case EngGrp::MME:
            return (1ULL << engId) & engMasks.mme;

        case EngGrp::ARC:
            return (1ULL << engId) & engMasks.arc;

        case EngGrp::NIC:
            return (1ULL << engId) & engMasks.nic;

        case EngGrp::ARC_FARM: // same as arc
            return (1ULL << engId) & engMasks.arc;

        case EngGrp::PDMA:
            return (1ULL << engId) & engMasks.pdma;

        case EngGrp::ROT:
            return (1ULL << engId) & engMasks.rot;

        case EngGrp::DMA:
            return (1ULL << engId) & engMasks.dma;

        case EngGrp::EDUP:
            return (1ULL << engId) & engMasks.arc;

        case EngGrp::PRT:
            return (1ULL << engId) & engMasks.nic;
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief readInThreads() When reading registers from the simulator we do it in multiple threads, for
 *                          some reason it is faster. Each thread runs the code below. It gets/incs an
                            atomic variable to get what block to read and then reads and dumps it to log
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::readInThreads(bool isSimulator, const EngMasks engMasks)
{
    const std::vector<RegsBlock>& regsVec = getReadRegsInfo();

    while (true)
    {
        uint64_t element = m_readInThreadCnt++;
        if (element >= regsVec.size())
        {
            return;
        }

        auto block = regsVec[element];

        if (!shouldReadRegBlock(block, isSimulator, engMasks))
        {
            LOG_INFO(SYN_DEV_FAIL, "skip reading {:x} {:x}", block.addr, block.size);
            continue;
        }

        readRegs(block.addr, block.size);
    }
}


DfaBase::EngMasks DfaBase::getEngMasks()
{
    EngMasks engMasks{};

    engMasks.tpc     = m_hwIp.tpc_enabled_mask_ext;
    LOG_INFO(SYN_DEV_FAIL, "tpc mask {:x}", engMasks.tpc);

    engMasks.mme     = m_hwIp.mme_enabled_mask;
    LOG_INFO(SYN_DEV_FAIL, "mme mask {:x}", engMasks.mme);

    engMasks.nic     = m_hwIp.nic_ports_mask;
    LOG_INFO(SYN_DEV_FAIL, "nic mask {:x}", engMasks.nic);

    engMasks.arc = m_hwIp.sched_arc_enabled_mask;
    LOG_INFO(SYN_DEV_FAIL, "arc mask {:x}", engMasks.arc);

    engMasks.rot     = m_hwIp.rotator_enabled_mask;
    LOG_INFO(SYN_DEV_FAIL, "rot mask {:x}", engMasks.rot);

    if (m_devType != synDeviceGaudi)
    {
        engMasks.dma = m_hwIp.edma_enabled_mask;
    }
    LOG_INFO(SYN_DEV_FAIL, "dma mask {:x}", engMasks.dma);

    if (m_devType == synDeviceGaudi3)
    {
        engMasks.pdma = m_hwIp.pdma_user_owned_ch_mask;
    }
    LOG_INFO(SYN_DEV_FAIL, "pdma mask {:x}", engMasks.pdma);

    return engMasks;
}


/*
 ***************************************************************************************************
 *   @brief dumpDevRegs() dump device registers to log.
 *                        Register blocks to be dumped to the logs are defined in files created during compilation
 *                        by a script. The files include a vector of registers blocks to be read.
 *                        The function goes over the blocks and calls a function to dump each block
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::dumpDevRegs()
{
    if (m_dfaStatus.hasError(DfaErrorCode::deviceUnavailble))
    {
        LOG_INFO(SYN_DEV_FAIL, "Device not available, can not read registers");
        return;
    }

    bool isSimulator = isSimulatorFunc(m_devIdType);

    if (m_dfaStatus.hasError(DfaErrorCode::signal) && isSimulator)
    {
        LOG_INFO_T(SYN_DEV_FAIL, "Not reading regiseter for an excetion on simulator");
        return;
    }

    EngMasks engMasks = getEngMasks();

    const int NUM_THREADS = 4; // See jira for why 4 threads were chosen

    const std::vector<RegsBlock>& regsVec = getReadRegsInfo();

    LOG_INFO_T(SYN_DEV_FAIL, "Reading registers, has {} blocks. isSimulator {}", regsVec.size(), isSimulator);

    // for simulator read using multiple threads, for some reason it is faster
    if (isSimulator)
    {
        std::thread t[NUM_THREADS];

        m_readInThreadCnt = 0;

        for (int i = 0; i < NUM_THREADS; i++)
        {
            t[i] = std::thread(&DfaBase::readInThreads, this, isSimulator, engMasks);
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            t[i].join();
        }
    }
    else
    {
        for (int i = 0; i < regsVec.size(); i++)
        {
            auto block = regsVec[i];

            if (!shouldReadRegBlock(block, isSimulator, engMasks))
            {
                LOG_INFO(SYN_DEV_FAIL, "skip reading {:x} {:x}", block.addr, block.size);
                continue;
            }

            readRegs(block.addr, block.size);
        }
    }
    LOG_INFO_T(SYN_DEV_FAIL, "Reading registers Done");
}

/*
 ***************************************************************************************************
 *   @brief dumpTpcKernels() Dump TPC kernels to log.
 *                           The function iterates all TPC engines kernel addresses and dump
 *                           all unique kernels after a DFA event occur.
 *                           If group of TPC engines run the same kernel, only one copy will be dumped.
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::dumpTpcKernels()
{
    std::map<uint64_t, unsigned> tpcFoundKernelsMap;
    std::vector<uint64_t> tpcKernelsAddressList;

    switch (m_devType)
    {
        case synDeviceGaudi:
        case synDeviceGaudi2:
        case synDeviceGaudi3:
            tpcKernelsAddressList = m_devCommon->getTpcAddrVector();
            break;

        default:
            LOG_TRACE_T(SYN_API, "Device not supported: {}", m_devType);
            return;
    }

    for (unsigned i = 0; i < tpcKernelsAddressList.size(); i++)
    {
        uint64_t tpcKernelAddress;

        synStatus status = readHbmMemory(tpcKernelsAddressList[i], (uint32_t*)&tpcKernelAddress, sizeof(uint64_t));
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEV_FAIL, "Failed to read TPC kernel address from HBM, addr {} status {}", tpcKernelsAddressList[i], status);
            return;
        }

        if (tpcFoundKernelsMap.find(tpcKernelAddress) != tpcFoundKernelsMap.end())
        {
            LOG_INFO_T(SYN_DEV_FAIL, "#tpc_kernel {} is the same as tpc_kernel {}", i, tpcFoundKernelsMap[tpcKernelAddress]);
            continue;
        }
        else
        {
            tpcFoundKernelsMap[tpcKernelAddress] = i;

            uint64_t hbmBaseAddr;
            uint64_t hbmEndAddr;
            m_devCommon->getDeviceHbmVirtualAddresses(hbmBaseAddr, hbmEndAddr);

            if (tpcKernelAddress < hbmBaseAddr || tpcKernelAddress >= hbmEndAddr)
            {
                LOG_INFO_T(SYN_DEV_FAIL,
                           "#tpc_kernel {} does not exist, kernel address: {:x}, HBM start address: {:x}, HBM end address: {:x}",
                           i,
                           tpcKernelAddress,
                           hbmBaseAddr,
                           hbmEndAddr);
                continue;
            }
            status = readTpcKernelFromHbm(tpcKernelAddress, i);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_DEV_FAIL, "Failed to read TPC kernel from HBM, addr {:x} status {}", tpcKernelAddress, status);
                continue;
            }
        }
    }
}

synStatus DfaBase::readHbmMemory(uint64_t hbmAddress, uint32_t* hostBuffer, unsigned readSize)
{
    if (HBM_READ_CHUNK_SIZE < readSize)
    {
        LOG_ERR_T(SYN_DEV_FAIL,
                  "Invalid HBM read size requested: {} Max allowd size is: {}",
                  readSize,
                  HBM_READ_CHUNK_SIZE);
        return synInvalidArgument;
    }

    int readRes = hlthunk_device_memory_read_block_experimental(m_fdCompute, hostBuffer, hbmAddress, readSize, 0);
    if (readRes)
    {
        LOG_ERR_T(SYN_DEV_FAIL,
                  "Failed reading from HBM memory fd {} buff {:x} addr {:x} size {:x} readRes {} {}",
                  m_fdCompute,
                  TO64(hostBuffer),
                  hbmAddress,
                  readSize,
                  readRes,
                  strerror(-readRes));
        return synFail;
    }

    return synSuccess;
}

synStatus DfaBase::readTpcKernelFromHbm(const uint64_t tpcKernelAddress, unsigned tpcIndex)
{
    std::vector<uint32_t> tpcDumpBuffer;

    uint32_t maxWordsInKernel = MAX_KERNEL_SIZE / sizeof(uint32_t);
    uint32_t maxWordsInChunk  = HBM_READ_CHUNK_SIZE / sizeof(uint32_t);
    synStatus status          = synSuccess;
    uint32_t  haltPosition    = 0;
    uint64_t  haltMask0       = 0;
    uint64_t  haltMask1       = 0;
    uint64_t  haltValue0      = 0;
    uint64_t  haltValue1      = 0;

    switch (m_devType)
    {
        case synDeviceGaudi:
            haltMask0  = 0x0001f8000000003f; // mask for SPU and VPU opcodes
            haltValue0 = 0x0001000000000020; // halt opcode == 32
            break;

        case synDeviceGaudi2:
            haltMask0  = 0x0007e000000000fe; // mask for SPU and VPU opcodes, and compression type
            haltValue0 = 0x0004000000000080; // halt opcode == 32
            haltMask1  = 0x0007e000000000ff; // mask for 2nd part of compressed instruction
            haltValue1 = 0x0004000000000081; // compression type=spu+vpu, spu+cpu opcodes == 32
            break;

        case synDeviceGaudi3:
            haltMask0  = 0x007e0000000000fe; // mask for SPU and VPU opcodes, and compression type
            haltValue0 = 0x0040000000000080; // halt opcode == 32
            haltMask1  = 0x007e0000000000ff; // mask for 2nd part of compressed instruction
            haltValue1 = 0x0040000000000081; // compression type=spu+vpu, spu+cpu opcodes == 32
            break;

        default:
            LOG_ERR(SYN_API, "Device not supported: {}", m_devType);
            return synUnsupported;
    }

    uint32_t sizeWords = 0;

    do {
        uint32_t currSize = tpcDumpBuffer.size();
        tpcDumpBuffer.resize(currSize + maxWordsInChunk);

        status = readHbmMemory(tpcKernelAddress + (currSize * sizeof(uint32_t)), &tpcDumpBuffer[currSize], HBM_READ_CHUNK_SIZE);
        if (status != synSuccess)
        {
            return status;
        }

        // lookup the 'halt' command
        for (uint32_t offset = 0; offset < maxWordsInChunk; offset += 8)
        {
            uint64_t* opCode = (uint64_t*)(&tpcDumpBuffer[currSize + offset]);
            if (((*opCode) & haltMask0) == haltValue0 || (haltMask1 && ((*(opCode + 2)) & haltMask1) == haltValue1))
            {
                haltPosition = currSize + offset;
                if (!sizeWords)
                {
                    // Read extra 64 words (256 bytes) after the halt command
                    sizeWords = haltPosition + 8 + 64;
                }
                break;
            }
        }
    } while ((!haltPosition || tpcDumpBuffer.size() < sizeWords) && tpcDumpBuffer.size() < maxWordsInKernel);

    if (haltPosition)
    {
        tpcDumpBuffer.resize(sizeWords);
    }

    writeTpcKernelDumpToLog(tpcKernelAddress, tpcDumpBuffer, tpcIndex);

    return status;
}

void DfaBase::writeTpcKernelDumpToLog(const uint64_t tpcKernelAddress, std::vector<uint32_t>& tpcDumpBuffer, unsigned tpcIndex) const
{
    std::string tpcDump = fmt::format(FMT_COMPILE("#tpc_kernel {} {:x}\n"), tpcIndex, getSizeBytes(tpcDumpBuffer.size()));
    tpcDump += fmt::format(FMT_COMPILE("#registers {:x} {:x}\n"), tpcKernelAddress, getSizeBytes(tpcDumpBuffer.size()));

    for (int i = 0; i < tpcDumpBuffer.size(); i++)
    {
        if ((i % REGS_PER_LINE) == 0)
        {
            tpcDump += fmt::format(FMT_COMPILE("{:x}: "), tpcKernelAddress + getSizeBytes(i));
        }

        tpcDump += fmt::format(FMT_COMPILE("{:08x} "), tpcDumpBuffer[i]);
        if (((i % REGS_PER_LINE) == (REGS_PER_LINE - 1)) && (i != (tpcDumpBuffer.size() - 1)))
        {
            tpcDump += "\n";
        }
    }
    LOG_INFO_T(SYN_DEV_FAIL, "{}", tpcDump);
}

static void logDmesgRestrict()
{
    std::ifstream  ifs;
    constexpr char file[] = "/proc/sys/kernel/dmesg_restrict";
    ifs.open(file, std::ifstream::in);

    if (!ifs.is_open())
    {
        LOG_ERR(DMESG_LOG, "Failed to open {}", file);
        return;
    }

    LOG_ERR(DMESG_LOG, "--- logging {} ---", file);
    std::string line;
    while (getline(ifs, line)) // This should return one line with 0 or 1
    {
        LOG_ERR(DMESG_LOG, "value of {} is {}", file, line);
    }
}

std::pair<bool, std::string> exec(std::string cmd)
{
    std::array<char, 4096> buffer;
    std::string           result;

    cmd += " 2>&1";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

    if (!pipe)
    {
        LOG_ERR(SYN_DEV_FAIL, "couldn't open pipe for command {} with errno {}", cmd, errno);
        return {false, "couldn't open pipe"};
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return {true, result};
}

static bool moreThanNlines(const std::string& str, int n)
{
    return std::count(str.begin(), str.end(), '\n') > n;
}

static bool logDmesgUsingPipe(const std::string& cmd)
{
    LOG_INFO(DMESG_LOG, "Trying to log dmesg using {}", cmd);

    auto [res, out] = exec(cmd);

    if (!res) return res;

    if (!moreThanNlines(out, 10)) // if less than 10 line, command probably failed
    {
        LOG_ERR(DMESG_LOG, "Could not log dmesg using {}. Not enough lines returned", cmd);
        LOG_ERR(DMESG_LOG, "\n{}", out);
        return false;
    }
    LOG_INFO(SYN_DEV_FAIL, "dmesg using '{}' is copied to {}", cmd, DMESG_COPY_FILE);

    LOG_INFO(DMESG_LOG, TITLE_STR("dmesg copy start"));
    LOG_INFO(DMESG_LOG, "\n{}", out);
    LOG_INFO(DMESG_LOG, TITLE_STR("dmesg copy done"));

    return true;
}

static void logDmesgUsingKkigctl(int size)
{
    LOG_INFO(DMESG_LOG, "Trying to log dmesg using klogctl");
    std::vector<char> buff(size);

    #define SYSLOG_ACTION_READ_ALL 3
    int rtn = klogctl(SYSLOG_ACTION_READ_ALL, buff.data(), size);

    if (rtn == -1)
    {
        LOG_ERR(SYN_DEV_FAIL, "Failed to copy dmesg. rtn code {} errno {} {}", rtn, errno, strerror(errno));
        LOG_ERR(DMESG_LOG, "Failed to copy dmesg. rtn code {} errno {} {}", rtn, errno, strerror(errno));

        // Try to figure out why we failed
        logDmesgRestrict();

        return;
    }

    LOG_INFO(SYN_DEV_FAIL, "dmesg is copied to {}", DMESG_COPY_FILE);

    LOG_INFO(DMESG_LOG, TITLE_STR("dmesg copy start, size {}", size));
    LOG_INFO(DMESG_LOG, "\n{}", buff.data());
    LOG_INFO(DMESG_LOG, TITLE_STR("dmesg copy done, size {}", size));
}

/*
 ***************************************************************************************************
 *   @brief logDmesg() Copies the last X bytes from the dmesg to a log file
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
void DfaBase::logDmesg()
{
    logDfaBegin(synapse::LogManager::LogType::DMESG_LOG);

    // get and log upteime
    struct sysinfo sysInfo;
    int            error = sysinfo(&sysInfo);
    if (error != 0)
    {
        LOG_ERR(DMESG_LOG, "Failed to get uptime. rc {} errno {} {}", error, errno, strerror(errno));
    }
    else
    {
        LOG_INFO(DMESG_LOG, "#uptime: {}", sysInfo.uptime);
    }

    // log device information
    logDeviceInfo(synapse::LogManager::LogType::DMESG_LOG);

    const int  size     = DMESG_COPY_FILE_SIZE - 0x10000;  // keep some room for our messages
    const int  maxLines = size / 1000; // assume less than 1000 chars per line
    static_assert(size > 0, "size to read from dmesg should be bigger than 0");

    bool ok = logDmesgUsingPipe("journalctl --dmesg -n " + std::to_string(maxLines));
    if (!ok)
    {
        ok = logDmesgUsingPipe("dmesg -T 2>&1|tail -c " + std::to_string(size));

        if (!ok)  // fallback to klogctl
        {
            logDmesgUsingKkigctl(size);
        }
    }
}

void DfaBase::logHlSmi()
{
    std::string cmd       = "ENABLE_CONSOLE=true hl-smi -q";
    auto [result, output] = exec(cmd);
    UNUSED(result);
    // If succeeded - prints HL-SMI output. Otherwise, prints the failure output.
    LOG_INFO(SYN_DEV_FAIL, "\n{}", output);
}

void DfaBase::_handleSynapseTermination(const std::string& consoleMsg)
{
    if (GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::busyWait)
    {
        LOG_ERR(SYN_DEV_FAIL, TITLE_STR("Entering busy-wait"));
        synapse::LogManager::instance().flush();

        // This will release user thread for new api, and let synapse return the error code to the user.
        do
        {
            usleep(SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US);
        } while (1);
    }
    else if (GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::synapseTermination)
    {
        LOG_ERR(SYN_DEV_FAIL,
                TITLE_STR("Killing process in {} usec", SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US));

        // get time
        auto timeNow = std::chrono::system_clock::to_time_t(m_dfaStatus.getFirstErrTime());
        char timeStr[100] {};
        auto ok = std::strftime(timeStr, sizeof(timeStr), "%T", std::localtime(&timeNow));
        if (ok == 0)
        {
            strcpy(timeStr, "   N/A  ");
        }

        // output to console
        std::cerr << DFA_KILL_MSG << " " << SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US / 1000 / 1000
        << " seconds (hl: " << m_devHlIdx << ") " << timeStr << " [" << consoleMsg << "]\n";

        usleep(SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US);
        LOG_ERR(SYN_DEV_FAIL, TITLE_STR("Killing process"));
        while (true)
        {
            synapse::LogManager::instance().flush();

            if (0 == kill(getpid(), SIGKILL))
            {
                LOG_ERR(SYN_DEV_FAIL, TITLE_STR("Sent SIGKILL to process"));
            }
            else
            {
                LOG_CRITICAL(SYN_DEV_FAIL, "The sigkill execution failed");
                sleep(1);
            }
        }
    }
    else if (GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::disabledRepeat)
    {
        m_isErrorLogged = false;
    }
}

bool DfaBase::_notifyFailure(uint32_t failureType, uint64_t suspectedCsHandle) const
{
    LOG_ERR(SYN_DEV_FAIL,
            "Notified failure {} (suspected CS {})",
            failureType,
            suspectedCsHandle);

    std::unique_lock<std::mutex> guard(m_dfaObserversMutex);

    for (auto pDfaObserver : m_dfaObservers)
    {
        if (!pDfaObserver->notifyFailureObserved())
        {
            return false;
        }
    }

    return true;
}

DfaBase::SingleEvent DfaBase::getDevMappingMsg(const hlthunk_page_fault_info& pf)
{
    std::string desc;

    desc += fmt::format(FMT_COMPILE("page fault: timestamp {} addr {:x} eng_id {} num_of_mappings {}"),
                                    pf.timestamp, pf.addr, pf.engine_id, pf.num_of_mappings);

    return {{DfaEventInfoEnum::PAGE_FAULT, desc}, pf.timestamp};
}

DfaBase::SingleEvent DfaBase::getPageFaultEvent(PageFaultEventOpt opt)
{
    // First, read the information
    // Assume 1 mappings_buf. If not, the call will return the correct number

    struct hlthunk_page_fault_info pf;
    uint32_t                       numBuffers = 1;

    auto buf        = std::make_unique<hlthunk_user_mapping[]>(numBuffers);
    pf.mappings_buf = buf.get();

    int ret    = hlthunk_get_page_fault_info(m_fdCompute, &pf, numBuffers);

    numBuffers = pf.num_of_mappings;

    if (numBuffers == 0) // no error
    {
        return getDevMappingMsg(pf);
    }

    if (ret != 0)
    {
        if (numBuffers == 0xFFFFFFFF)
        {
            LOG_ERR(SYN_DEV_FAIL, "error getting hlthunk_get_page_fault_info with size 1. Returned num_of_mappings {:x}", numBuffers);
            return getDevMappingMsg(pf);
        }

        // try again with correct number of buffers
        buf             = std::make_unique<hlthunk_user_mapping[]>(numBuffers);
        pf.mappings_buf = buf.get();
        ret             = hlthunk_get_page_fault_info(m_fdCompute, &pf, numBuffers);
        if (ret != 0)
        {
            LOG_ERR(SYN_DEV_FAIL, "error getting hlthunk_get_page_fault_info. Ret {} errorno {} {}", ret, errno, strerror(errno));
            return getDevMappingMsg(pf);
        }
    }

    if (opt == PageFaultEventOpt::DESCRITPION)
    {
        return getDevMappingMsg(pf);
    }

    std::string prefix;
    if (pf.timestamp < m_devOpenTime)
    {
        LOG_TRACE(SYN_DEV_FAIL, "NOTE: page fault time is before open time (it is for previous run) {} < {}",
                  pf.timestamp, m_devOpenTime);
        prefix = "PREV RUN!!! :";
    }

    for (int i = 0; i < pf.num_of_mappings; i++)
    {
        LOG_TRACE(SYN_DEV_FAIL, "   {} {} dev-addr/size {:x}/{:x}", prefix, i, pf.mappings_buf[i].dev_va, pf.mappings_buf[i].size);
    }
    return getDevMappingMsg(pf);
}

bool DfaBase::isSimulatorFunc(hl_pci_ids devIdType)
{
    switch (devIdType)
    {
        case PCI_IDS_INVALID:
        case PCI_IDS_GAUDI:
        case PCI_IDS_GAUDI_HL2000M:
        case PCI_IDS_GAUDI_SEC:
        case PCI_IDS_GAUDI_HL2000M_SEC:
        case PCI_IDS_GAUDI2:
        case PCI_IDS_GAUDI3:
        case PCI_IDS_GAUDI3_DIE1:
        case PCI_IDS_GAUDI3_SINGLE_DIE:
        case PCI_IDS_GAUDI_FPGA:
        case PCI_IDS_GAUDI2_FPGA:
        case PCI_IDS_GAUDI3_FPGA:
            return false;
        case PCI_IDS_GAUDI2_SIMULATOR:
        case PCI_IDS_GAUDI2B_SIMULATOR:
        case PCI_IDS_GAUDI2C_SIMULATOR:
        case PCI_IDS_GAUDI2D_SIMULATOR:
        case PCI_IDS_GAUDI_HL2000M_SIMULATOR:
        case PCI_IDS_GAUDI3_SIMULATOR:
        case PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE:
            return true;

        default:
            return false;
    }
}

void DfaBase::raiseIrqUnexpectedInterrupt()
{
    volatile uint32_t* tdrIrqMonitorArmRegAddr;
    synStatus status = m_devCommon->getTdrIrqMonitorArmRegAddr(tdrIrqMonitorArmRegAddr);

    switch (status)
    {
        case synSuccess:
            // Trigger the TDR monitor to raise interrupt
            scal_write_mapped_reg(tdrIrqMonitorArmRegAddr, 0x1);
            LOG_TRACE(SYN_API,
                      "{}: Write to mapped register address {} value 1",
                      HLLOG_FUNC,
                      TO64(tdrIrqMonitorArmRegAddr));
            break;
        case synUnsupported:
            break;
        default:
            LOG_ERR_T(SYN_API, "DFA failed to arm the TDR monitor, status: {}", status);
            break;
    }
}

void DfaBase::logSynapseApiCounters()
{
    LOG_INFO_T(SYN_DEV_FAIL, "\n{}", ApiCounterRegistry::getInstance().toString());
}

void DfaBase::logHwIp()
{
    LOG_INFO(SYN_DEV_FAIL, "----- hw_ip fd {}-----", m_fdCompute);
    LOG_INFO(SYN_DEV_FAIL, "sram_base_address              {:#x}", m_hwIp.sram_base_address                     );
    LOG_INFO(SYN_DEV_FAIL, "dram_base_address              {:#x}", m_hwIp.dram_base_address                     );
    LOG_INFO(SYN_DEV_FAIL, "dram_size                      {:#x}", m_hwIp.dram_size                             );
    LOG_INFO(SYN_DEV_FAIL, "sram_size                      {:#x}", m_hwIp.sram_size                             );
    LOG_INFO(SYN_DEV_FAIL, "num_of_events                  {:#x}", m_hwIp.num_of_events                         );
    LOG_INFO(SYN_DEV_FAIL, "device_id                      {:#x}", m_hwIp.device_id                             );
    LOG_INFO(SYN_DEV_FAIL, "cpld_version                   {:#x}", m_hwIp.cpld_version                          );
    LOG_INFO(SYN_DEV_FAIL, "psoc_pci_pll_nr                {:#x}", m_hwIp.psoc_pci_pll_nr                       );
    LOG_INFO(SYN_DEV_FAIL, "psoc_pci_pll_nf                {:#x}", m_hwIp.psoc_pci_pll_nf                       );
    LOG_INFO(SYN_DEV_FAIL, "psoc_pci_pll_od                {:#x}", m_hwIp.psoc_pci_pll_od                       );
    LOG_INFO(SYN_DEV_FAIL, "psoc_pci_pll_div_factor        {:#x}", m_hwIp.psoc_pci_pll_div_factor               );
    LOG_INFO(SYN_DEV_FAIL, "tpc_enabled_mask               {:#x}", m_hwIp.tpc_enabled_mask                      );
    LOG_INFO(SYN_DEV_FAIL, "dram_enabled                   {:#x}", m_hwIp.dram_enabled                          );
    LOG_INFO(SYN_DEV_FAIL, "module_id                      {:#x}", m_hwIp.module_id                             );
    LOG_INFO(SYN_DEV_FAIL, "decoder_enabled_mask           {:#x}", m_hwIp.decoder_enabled_mask                  );
    LOG_INFO(SYN_DEV_FAIL, "mme_master_slave_mode          {:#x}", m_hwIp.mme_master_slave_mode                 );
    LOG_INFO(SYN_DEV_FAIL, "tpc_enabled_mask_ext           {:#x}", m_hwIp.tpc_enabled_mask_ext                  );
    LOG_INFO(SYN_DEV_FAIL, "dram_default_page_size         {:#x}", m_hwIp.device_mem_alloc_default_page_size    );
    LOG_INFO(SYN_DEV_FAIL, "dram_page_size                 {:#x}", m_hwIp.dram_page_size                        );
    LOG_INFO(SYN_DEV_FAIL, "first_available_interrupt_id   {:#x}", m_hwIp.first_available_interrupt_id          );
    LOG_INFO(SYN_DEV_FAIL, "edma_enabled_mask              {:#x}", m_hwIp.edma_enabled_mask                     );
    LOG_INFO(SYN_DEV_FAIL, "server_type                    {:#x}", m_hwIp.server_type                           );
    LOG_INFO(SYN_DEV_FAIL, "pdma_user_owned_ch_mask        {:#x}", m_hwIp.pdma_user_owned_ch_mask               );
    LOG_INFO(SYN_DEV_FAIL, "number_of_user_interrupts      {:#x}", m_hwIp.number_of_user_interrupts             );
    LOG_INFO(SYN_DEV_FAIL, "nic_ports_mask                 {:#x}", m_hwIp.nic_ports_mask                        );
    LOG_INFO(SYN_DEV_FAIL, "nic_ports_external_mask        {:#x}", m_hwIp.nic_ports_external_mask               );
    LOG_INFO(SYN_DEV_FAIL, "security_enabled               {:#x}", m_hwIp.security_enabled                      );
    LOG_INFO(SYN_DEV_FAIL, "interposer_version             {:#x}", m_hwIp.interposer_version                    );
    LOG_INFO(SYN_DEV_FAIL, "substrate_version              {:#x}", m_hwIp.substrate_version                     );
    LOG_INFO(SYN_DEV_FAIL, "mme_enabled_mask               {:#x}", m_hwIp.mme_enabled_mask                      );
    LOG_INFO(SYN_DEV_FAIL, "odp_supported                  {:#x}", m_hwIp.odp_supported                         );
    LOG_INFO(SYN_DEV_FAIL, "revision_id                    {:#x}", m_hwIp.revision_id                           );
    LOG_INFO(SYN_DEV_FAIL, "tpc_interrupt_id               {:#x}", m_hwIp.tpc_interrupt_id                      );
    LOG_INFO(SYN_DEV_FAIL, "engine_core_interrupt_reg_addr {:#x}", m_hwIp.engine_core_interrupt_reg_addr        );
    LOG_INFO(SYN_DEV_FAIL, "rotator_enabled_mask           {:#x}", m_hwIp.rotator_enabled_mask                  );
    LOG_INFO(SYN_DEV_FAIL, "sched_arc_enabled_mask         {:#x}", m_hwIp.sched_arc_enabled_mask                );
    LOG_INFO(SYN_DEV_FAIL, "reserved_dram_size             {:#x}", m_hwIp.reserved_dram_size                    );
}

#pragma once
#include <deque>
#include <mutex>
#include <atomic>
#include "common/pci_ids.h"
#include "timer.h"
#include "scal.h"
#include "log_manager.h"
#include "synapse_common_types.h"
#include "dfa_defines.hpp"
#include "hlthunk.h"

// When updating this, so should dfaErrorToName

enum class DfaSynapseTerminationState : uint32_t
{
    disabled,           // DFA does not terminate
    busyWait,           // keep the thread in busy loop (sleep) after logging DFA
    synapseTermination, // default: terminate process
    disabledRepeat,     // Disabled + let us log another DFA (for testing)
};

enum class BlockType
{
    ALL,
    ALL1 = ALL,  // Like all, used in the python script to split a big block if needed (gaudi1 syncManager)
    NONE,
    SIM_ONLY,
    DEV_ONLY
};

enum class EngGrp
{
    NONE,
    TPC,
    MME,
    ARC,
    NIC,
    ARC_FARM,
    PDMA,
    ROT,
    DMA,
    EDUP,
    PRT
};

struct RegsBlock
{
    uint64_t  addr;
    uint32_t  size;
    BlockType blockType;
    EngGrp    engGrp;
    int16_t   engId;
};

class DfaObserver;
class DeviceCommon;

/**
 * 1. archive dfa logs into a dfa_logs.zip file
 * 2. roll dfa_logs.zip (dfa_logs.1.zip .. dfa_logs.4.zip)
 * 3. delete existing dfa logs
 */
void archiveDfaLogs();

/*
 ***************************************************************************************************
 *
 *   Class DfaBase: DFA - Device failure analysis
 *
 ***************************************************************************************************
 */
struct DeviceConstructInfo;

struct hlthunk_event_record_undefined_opcode;

// EventFdController::_mainLoop()->_queryEvent()->DeviceCommon::notifyEventFd()---------------------------------------------->DfaBase::checkFailure()
// EventFdController::_mainLoop()--[timeout]-->DeviceCommon::notifyDeviceFailure()----------->DfaBase::notifyDeviceFailure()--^
// hlthunk-failure->DeviceCommon::notifyHlthunkFailure()->DfaBase::notifyHlthunkFailure()-----^

// checkFailure()->checkDevFailure()->getDevErrorsInfoFromLkd()

class DfaBase
{
public:
    DfaBase(const DeviceConstructInfo& deviceConstructInfo, DeviceCommon* deviceCommon);

    virtual ~DfaBase() = default;

    void dumpEngStatus();
    void logDmesg();
    void logDeviceInfo(synapse::LogManager::LogType logType);
    void logDevEngines();
    void logHlSmi();

    virtual DfaStatus getStatus() const { return m_dfaStatus; }

    virtual bool addObserver(DfaObserver* pDfaObserver);

    virtual bool removeObserver(DfaObserver* pDfaObserver);

    virtual void checkFailure(DfaStatus dfaStatus, const DfaExtraInfo& dfaExtraInfo);

    virtual void notifyEventFd(uint64_t events);

    int getDevErrorsInfoFromLkd(uint64_t eventsBitmap);

    void notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo);

    void notifyDeviceFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo);

    void setAssertAsyncBufferAddress(void* assertAsyncBuffer) { m_assertAsyncBuffer = assertAsyncBuffer; }

    bool isAssertAsyncNoticed() const { return m_isAssertAsyncNoticed; }

    DfaStatus m_dfaStatus {};

    using DeviceDfaObservers = std::deque<DfaObserver*>;

    DeviceDfaObservers m_dfaObservers;
    mutable std::mutex m_dfaObserversMutex;

    // 50ms
    static const unsigned SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US;
    static const uint32_t HBM_READ_CHUNK_SIZE;
    static const uint32_t MAX_KERNEL_SIZE;

protected:
    bool        _notifyFailure(uint32_t failureType, uint64_t suspectedCsHandle) const;

    const unsigned      m_devHlIdx;
    const int           m_fdCompute;
    const int           m_fdControl;
    const uint32_t      m_devModuleIdx;
    const synDeviceType m_devType;
    const hl_pci_ids    m_devIdType;
    uint64_t            m_csSeqTimeout = 0; // The cs we got the timeout on. 0 is not valid

private:
    enum class DfaEventInfoEnum // This enum is a mix of an enum from LKD and additional needed enums
    {
        OPEN_DEV              = HLTHUNK_OPEN_DEV,
        CS_TIMEOUT            = HLTHUNK_CS_TIMEOUT,
        RAZWI                 = HLTHUNK_RAZWI_EVENT,
        UNDEFINED_OPCODE      = HLTHUNK_UNDEFINED_OPCODE,
        CRITICAL_HW_ERR       = HLTHUNK_HW_ERR_OPCODE,
        CRITICAL_FIRMWARE_ERR = HLTHUNK_FW_ERR_OPCODE,

        PAGE_FAULT
    };

    struct HlThunkEventInfo
    {
        DfaEventInfoEnum eventId;
        std::string      description;
    };

    struct SingleEvent
    {
        struct HlThunkEventInfo eventInfo;
        int64_t                 timestamp;
    };

    struct EngMasks
    {
        static uint64_t constexpr MAX64 = std::numeric_limits<uint64_t>::max();

        uint64_t tpc  = MAX64;
        uint64_t mme  = MAX64;
        uint64_t arc  = MAX64;
        uint64_t nic  = MAX64;
        uint64_t pdma = MAX64;
        uint64_t rot  = MAX64;
        uint64_t dma  = MAX64;
    };

    SingleEvent getDevMappingMsg(const hlthunk_page_fault_info& pf);

    SingleEvent getOpenEvent();
    SingleEvent getUndefOpCodeEvent();
    SingleEvent getCsTimeoutEvent(uint64_t& csSeqTimeout);
    SingleEvent getCriticalHwErrorEvent();
    SingleEvent getCriticalFirmwareErrorEvent();
    SingleEvent getRazwiEvent();

    enum class PageFaultEventOpt { DESCRITPION, MAPPINGS};
    SingleEvent getPageFaultEvent(PageFaultEventOpt opt);

    void _handleSynapseTermination(const std::string& consoleMsg);

    const std::vector<RegsBlock>& getReadRegsInfo();
    void                          readRegs(uint64_t addr, uint32_t size);
    void                          dumpDevRegs();
    void                          dumpTpcKernels();
    synStatus                     readHbmMemory(uint64_t hbmAddress, uint32_t* hostBuffer, unsigned readSize);
    synStatus                     readTpcKernelFromHbm(const uint64_t tpcKernelAddress, unsigned tpcIndex);
    void                          writeTpcKernelDumpToLog(const uint64_t tpcKernelAddress, std::vector<uint32_t>& tpcDumpBuffer, unsigned tpcIndex) const;
    void                          logDfaBegin(synapse::LogManager::LogType logType);
    std::string                   getDfaTriggerMsg(DfaErrorCode dfaErrorCode);
    std::string                   getConsoleMsg(DfaStatus dfaStatus);
    void                          logUserMsg(DfaStatus dfaStatus);
    void                          readInThreads(bool isSimulator, const EngMasks engMasks);
    bool                          shouldReadRegBlock(const RegsBlock& regBlock, bool isSimulator, const EngMasks& engMasks);
    EngMasks                      getEngMasks();
    void                          logSynapseApiCounters();
    void                          logHwIp();

    static std::string lkdEvent2str(DfaEventInfoEnum x);

    static bool isSimulatorFunc(hl_pci_ids devIdType);

    void raiseIrqUnexpectedInterrupt();

    std::mutex m_mutex;
    bool       m_isErrorLogged = false;

    const std::vector<RegsBlock> m_emptyRegs;

    bool                 m_isAssertAsyncNoticed = false;
    void*                m_assertAsyncBuffer    = nullptr;
    int64_t              m_devOpenTime          = 0;
    DeviceCommon*        m_devCommon;
    DfaExtraInfo         m_dfaExtraInfo {};
    const synDeviceInfo& m_osalInfo;

    std::atomic<size_t> m_readInThreadCnt;

    static constexpr int kserverNameSize = 255;
    char                 m_serverName[kserverNameSize + 1] {};
    hlthunk_hw_ip_info   m_hwIp {};
};

#pragma once

#include <chrono>
#include <variant>
#include <vector>
#include <string>

#define DMESG_COPY_FILE      "dfa_dmesg.txt"
#define DMESG_COPY_FILE_SIZE (5 * 1024 * 1024)

#define DEVICE_FAIL_ANALYSIS_FILE      "dfa_log.txt"
#define DEVICE_FAIL_ANALYSIS_FILE_SIZE (200 * 1024 * 1024) // Note, if you want to change it, make sure to check the
                                                           // actual size with DFA_COLLECT_CCB enabled

#define DFA_API_FILE      "dfa_api.txt"
#define DFA_API_FILE_SIZE (25 * 1024 * 1024)

#define SUSPECTED_RECIPES           "dfa_recipes.txt"
#define SUSPECTED_RECIPES_FILE_SIZE (100 * 1024 * 1024)

#define DFA_NIC_INFO_FILE "dfa_nic_info.txt"
#define DFA_NIC_INFO_SIZE (25 * 1024 * 1024)

#define REGS_PER_LINE 16

const char* const DFA_KILL_MSG = "Synapse detected a device critical error that requires a restart. Killing process in";

namespace DfaMsg
{
    const char* const LKD_EVENTS            = "Logging LKD events";
    const char* const ENG_STATUS            = "Engines Status";
    const char* const FAIL_INFO             = "Failure Info";
    const char* const DMESG                 = "Copying dmesg to a seperate file";
    const char* const GCFG                  = "Log GCFG values";
    const char* const ENG_MASKS             = "Engines masks / number";
    const char* const DEV_MAP               = "Device Mapping";
    const char* const SW_MAP_MEM            = "SW Mapped memory";
    const char* const HL_SMI                = "hl-smi output";
    const char* const REGS                  = "Dump device registers";
    const char* const DONE                  = "Logging failure info done";
    const char* const QUEUES_DETAILS        = "Queues details";
    const char* const NO_PROGRESS_TDR       = "No-Progress TDR";
    const char* const OLDEST                = "Oldest work in each queue";
    const char* const Q_STATUS              = "Logging Full Queues Status";
    const char* const ARC_HEARTBEAT         = "Arc heart-beat";
    const char* const TPC_KERNEL            = "TPC kernel dump";
    const char* const SYNAPSE_API_COUNTERS  = "Synapse API's enter/exit counters";
    const char* const HW_IP                 = "Device HW_IP";
}  // namespace DfaMsg

enum class DfaErrorCode : uint64_t
{
    noError                    = 0,
    tdrFailed                  = 1ULL << 0,
    eventSyncFailed            = 1ULL << 1,
    streamSyncFailed           = 1ULL << 2,
    getDeviceInfoFailed        = 1ULL << 3,
    waitForCsFailed            = 1ULL << 4,
    hlthunkDebugFailed         = 1ULL << 5,
    undefinedOpCode            = 1ULL << 6,
    getTimeSyncInfoFailed      = 1ULL << 7,
    waitForMultiCsFailed       = 1ULL << 8,
    memcopyIoctlFailed         = 1ULL << 9,
    requestCommandBufferFailed = 1ULL << 10,
    destroyCommandBufferFailed = 1ULL << 11,
    commandSubmissionFailed    = 1ULL << 12,
    stagedCsFailed             = 1ULL << 13,
    getDeviceClockRateFailed   = 1ULL << 14,
    getPciBusIdFailed          = 1ULL << 15,
    usrEngineErr               = 1ULL << 16,
    hclFailed                  = 1ULL << 17,
    csTimeout                  = 1ULL << 18,  // gaudi & greco only
    razwi                      = 1ULL << 19,
    mmuPageFault               = 1ULL << 20,
    scalTdrFailed              = 1ULL << 21,
    assertAsync                = 1ULL << 22,
    generalHwError             = 1ULL << 23,
    criticalHwError            = 1ULL << 24,
    criticalFirmwareError      = 1ULL << 25,
    signal                     = 1ULL << 26,

    tpcBrespErr                = 1ULL << 28,
    waitForMultiCsTimedOut     = 1ULL << 29,
    deviceReset                = 1ULL << 30,
    deviceUnavailble           = 1ULL << 31,
};

struct DfaStatus
{
    using StdTime = std::chrono::time_point<std::chrono::system_clock>;

    DfaStatus() : m_raw(0), m_firstErrChronoTime() {}

    bool hasError(DfaErrorCode errCode) const { return (m_raw & (uint64_t)errCode) == (uint64_t)errCode; }

    void addError(DfaErrorCode errCode) { m_raw |= (uint64_t)errCode; }

    bool hasOnlyErrors(std::vector<DfaErrorCode> errors)
    {
        uint64_t sumErrs = 0;
        for (auto err : errors) sumErrs |= static_cast<uint64_t>(err);
        return m_raw == sumErrs;
    }

    bool isSuccess() const { return m_raw == 0; }

    uint64_t getRawData() const { return m_raw; }
    StdTime  getFirstErrTime() const { return m_firstErrChronoTime; }
    void     setFirstErrTime(StdTime time) { m_firstErrChronoTime = time; }

private:
    uint64_t m_raw;
    StdTime  m_firstErrChronoTime;
};

enum class DfaReq
{
    STREAM_INFO,
    PARSE_CSDC,
    ALL_WORK,
    ERR_WORK,
    SCAL_STREAM,
};

enum class DfaPhase
{
    NONE,
    STARTED,
    ENDED
};

enum class ReadRegMode
{
    lkd  = 0,
    skip = 1,
};

static constexpr std::chrono::duration dfaHlthunkTriggerDelay = std::chrono::milliseconds(500);

struct DfaExtraInfo
{
    struct DfaExtraInfoSignal
    {
        int         signal;
        const char* signalStr;
        bool        isSevere;
    };

    struct DfaExtraInfoMsg
    {
        std::string msg;
    };

    std::variant<std::monostate, DfaExtraInfoSignal, DfaExtraInfoMsg> extraInfo;
};

#define GAUDI_PROGRESS_FMT      "{:40} : {:6} : {:12} : {:8} : {:>8}          : {}/{}"
#define GAUDI_PROGRESS_FMT_WAIT "{:40} : {:6} : {:12} : {:8} : {:>8} {:>8} : {}/{}"
#define SCAL_PROGRESS_FMT       "{:20} : {:>8x} : {:>8x} : {}/{}"
#define SCAL_PROGRESS_HCL_FMT   "network{:<13} : {:>8x} : {:>8x} : "

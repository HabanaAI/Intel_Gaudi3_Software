#pragma once

#include "define.hpp"
#include "fence_state_machine.hpp"

#include <array>
#include <limits>
#include <unordered_map>

using FenceStateMachineDB = std::unordered_map<uint64_t, common::FenceStateMachine>;

namespace common
{
class SyncManagerInfo;
class SyncManagerInfoDatabase;
}  // namespace common

namespace common
{
class QmanCpInfo
{
protected:
    // Monitor-Setup information
    enum eMonitorSetupPhase
    {
        MONITOR_SETUP_PHASE_NOT_SET = 0x0,
        MONITOR_SETUP_PHASE_LOW     = 0x1,
        MONITOR_SETUP_PHASE_HIGH    = 0x2,
        MONITOR_SETUP_PHASE_DATA    = 0x4
    };
#define MONITOR_SETUP_READY (MONITOR_SETUP_PHASE_LOW | MONITOR_SETUP_PHASE_HIGH | MONITOR_SETUP_PHASE_DATA)

#define INVALID_PRED_VALUE (0xFFFF)

    struct MonitorSetupInfo
    {
        void clear()
        {
            m_monitorSetupMonitorId = INVALID_MONITOR_ID;
            m_monitorSetupState     = MONITOR_SETUP_PHASE_NOT_SET;
        };

        bool isMonitorReady() { return (m_monitorSetupState == MONITOR_SETUP_READY); };

        uint32_t m_monitorSetupMonitorId   = INVALID_MONITOR_ID;
        uint32_t m_monitorSetupAddressHigh = 0;
        uint32_t m_monitorSetupAddressLow  = 0;
        uint32_t m_monitorSetupData        = 0;

        eMonitorSetupPhase m_monitorSetupState = MONITOR_SETUP_PHASE_NOT_SET;
    };

public:
    QmanCpInfo() {};
    virtual ~QmanCpInfo() = default;

    bool setNewInfo(uint64_t                 handle,
                    uint64_t                 hostAddress,
                    uint64_t                 bufferSize,
                    uint32_t                 cpIndex,
                    uint16_t                 predValue,
                    SyncManagerInfoDatabase* pSyncManagerInfoDb);

    virtual void reset(bool shouldResetSyncManager, uint64_t cpDmaPacketIndex = std::numeric_limits<uint64_t>::max());

    bool finalize();

    void         printStartParsing(std::string_view qmanDescription);
    virtual void printStartParsing() = 0;

    virtual bool parseSinglePacket(eCpParsingState state) = 0;

    virtual bool parseArbPoint() { return false; };  // Only relevant for the Upper-CP
    virtual bool parseCpDma() { return false; };     // Only relevant for the Upper-CP

    void updateNextBuffer(uint32_t packetSize);

    bool validatePacketSize(uint32_t packetSize);

    // virtual
    virtual bool        isValidPacket(uint64_t packetId) = 0;
    virtual std::string getIndentation()                 = 0;
    virtual std::string getCtrlBlockIndentation()        = 0;

    virtual bool getLowerCpBufferHandleAndSize(uint64_t& handle, uint64_t& size) { return false; };

    bool shouldIgnoreCommand(uint16_t predValue)
    {
        // 0 - is currently set to "always execute"
        return (predValue != INVALID_PRED_VALUE) && (predValue != 0) && (predValue != m_predValue);
    }

    bool isInitialized() const { return m_isInitialized; };

    uint64_t getBufferSize() const { return m_bufferSize; };
    uint64_t getHostAddress() const { return m_hostAddress; };
    uint64_t getCurrentPacketId() const { return m_currentPacketId; };

    // As we increment after each packet, we need to dec it in here
    uint64_t getPacketIndex() { return m_packetIndex - 1; };

    virtual std::string getPacketIndexDesc() const = 0;

    virtual bool checkFenceClearPacket(uint64_t expectedAddress, uint16_t expectedFenceValue) const { return false; };

protected:
    virtual uint64_t getFenceIdsPerCp() = 0;

    static std::string _getMonitorSetupPhaseDescription(eMonitorSetupPhase phaseId);

    // Generic usage In case required, by upper layers parsing-methods
    // Holds MSG_Long content, and used for checking Fence-Clear command
    uint64_t m_currentPacketAddressField = 0;
    uint64_t m_currentPacketValueField   = 0;

    uint64_t m_hostAddress = 0;
    uint64_t m_packetIndex = 0;

    uint64_t m_bufferSize = 0;

    uint64_t m_currentPacketId = std::numeric_limits<uint64_t>::max();
    uint32_t m_cpIndex         = 0;

    SyncManagerInfoDatabase* m_pSyncManagerInfoDb = nullptr;

    // We expect that the monitors will be set one-by-one
    // Hence, only a single instance, and not a DB
    MonitorSetupInfo m_monitorSetupInfo;

    FenceStateMachineDB m_cpFenceSm;

private:
    uint64_t m_handle    = 0;
    uint16_t m_predValue = INVALID_PRED_VALUE;

    bool m_isInitialized = false;

    static const std::unordered_map<eMonitorSetupPhase, std::string> m_monitorSetupPhaseDesc;
};
}  // namespace common
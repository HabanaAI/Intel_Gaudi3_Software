#pragma once

#include "define.hpp"

#include <stdint.h>
#include <unordered_map>

namespace common
{
class FenceStateMachine;
}
using FenceStateMachineDB = std::unordered_map<uint64_t, common::FenceStateMachine>;

// The Monitor-SM & Fence-SM are defined for verifying that a Fence is properly
// handled by the couple (CP does not get stuck)

namespace common
{
class MonitorStateMachine
{
public:
    enum eMonitorState
    {
        MONITOR_STATE_INIT,
        // There was a fence packet that "consumed" this monitor coniguration
        MONITOR_STATE_CONSUMED = MONITOR_STATE_INIT,
        MONITOR_STATE_SETUP,
        MONITOR_STATE_ARM,

        MONITOR_STATE_LAST = MONITOR_STATE_ARM
    };

    MonitorStateMachine()
    : m_monitorId(INVALID_MONITOR_ID),
      m_setupFenceIndex(INVALID_FENCE_INDEX),
      m_armFenceIndex(INVALID_FENCE_INDEX),
      m_rdataValue(0),
      m_isFenceSetup(false),
      m_state(MONITOR_STATE_INIT),
      m_pFenceStateMachineDB(nullptr),
      m_isInitialized(false) {};

    ~MonitorStateMachine() = default;

    void init(uint16_t monitorId, FenceStateMachineDB* pFenceStateMachineDB)
    {
        m_monitorId            = monitorId;
        m_pFenceStateMachineDB = pFenceStateMachineDB;
        m_isInitialized        = true;
    };

    uint32_t getId() { return m_monitorId; };
    bool     isInitialized() { return m_isInitialized; };

    bool isFinalized();

    bool monitorSetupCommandFence(uint16_t fenceIndex, uint16_t rdataValue);

    bool monitorSetupCommandSobj(uint16_t sobjIndex);

    bool monitorArmCommand();

    // Return false if not in ARM state
    bool fenceCommand(uint16_t fenceIndex, uint16_t& fenceCounter);

    static std::string getStateDesc(eMonitorState state);

private:
    uint32_t m_monitorId;

    // For simplicity of SM, we will hold both setup fence-index and arm fence-index
    // Upon monitorSetupCommandFence, we will only update the setup fence-index
    // Upon monitorArmCommand, we will the arm fence-index, and update both QMAN's FenceSM, about the change
    //
    // The fence-index of a given Fence packet, defined by last Monitor-Setup
    uint16_t m_setupFenceIndex;
    // The fence-index of a given Fence packet, that had been set during last Monitor-ARM
    uint16_t m_armFenceIndex;

    uint16_t m_rdataValue;       // Will be zeroed upon consuming, to indicate that the ARM had been consumed
    uint16_t m_setupRdataValue;  // Will always be kept and will not be consumed

    bool m_isFenceSetup;

    eMonitorState m_state;

    FenceStateMachineDB* m_pFenceStateMachineDB;

    bool m_isInitialized;

    static std::string m_stateDescription[MONITOR_STATE_LAST + 1];
};
}  // namespace common
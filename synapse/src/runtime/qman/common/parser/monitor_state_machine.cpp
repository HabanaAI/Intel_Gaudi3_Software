#include "monitor_state_machine.hpp"

#include "defs.h"
#include "fence_state_machine.hpp"
#include "synapse_runtime_logging.h"

using namespace common;

// We use INFO level for the Parser-Print-Log, to be able to print all logs, in case of a failure
// Meaning, there might be a device reset which may lead to process kill,
// and we would like to be able to get the cause, even if we cannot get the full parsing
#define LOG_MSM_VERBOSE(msg, ...) LOG_GCP_VERBOSE("Monitor-ID {}: " msg, m_monitorId, ##__VA_ARGS__)
//
#define LOG_MSM_WARNING(msg, ...) LOG_GCP_WARNING("Monitor-ID {}: " msg, m_monitorId, ##__VA_ARGS__)
#define LOG_MSM_FAILURE(msg, ...) LOG_GCP_FAILURE("Monitor-ID {}: " msg, m_monitorId, ##__VA_ARGS__)

// #define DEBUG_MON_STATE_MACHINE

#ifdef DEBUG_MON_STATE_MACHINE
#define LOG_MSM_DEBUG(msg, ...) LOG_GCP_FAILURE("DBG Monitor-ID {}: " msg, m_monitorId, ##__VA_ARGS__)
#else
#define LOG_MSM_DEBUG(msg, ...)
#endif

std::string MonitorStateMachine::m_stateDescription[MONITOR_STATE_LAST + 1] = {"Consume", "Setup", "Arm"};

std::string MonitorStateMachine::getStateDesc(eMonitorState state)
{
    if (state <= MONITOR_STATE_LAST)
    {
        return m_stateDescription[state];
    }

    HB_ASSERT(false, "Unexpected state");
    return "Invalid";
}

bool MonitorStateMachine::isFinalized()
{
    if ((!m_isFenceSetup) || (m_state == MONITOR_STATE_CONSUMED))
    {
        return true;
    }

    LOG_MSM_FAILURE("Unexpected state {} during finalize", getStateDesc(m_state));

    // returns true anyhow...
    return true;
}

bool MonitorStateMachine::monitorSetupCommandFence(uint16_t fenceIndex, uint16_t rdataValue)
{
    if (fenceIndex >= INVALID_FENCE_INDEX)
    {
        LOG_MSM_FAILURE("Invalid Monitor-setup fenceIndex {}", fenceIndex);

        return false;
    }

    if (rdataValue == 0)
    {
        LOG_MSM_FAILURE("Invalid Monitor-setup RDATA value {}", rdataValue);

        return false;
    }

    // Last Monitor-Configuration had not been consumed by "Fence"
    if (m_state != MONITOR_STATE_CONSUMED)
    {
        // It might be just a waste of packet(s), or indicate a failure
        // Due to that, we will NOT return false
        LOG_MSM_WARNING("Unexpected Monitor-setup (Fence) prior of Monitor usage by Fence (state {})",
                        getStateDesc(m_state));
    }

    m_state           = MONITOR_STATE_SETUP;
    m_setupFenceIndex = fenceIndex;
    m_rdataValue      = rdataValue;
    m_setupRdataValue = rdataValue;
    m_isFenceSetup    = true;

    LOG_MSM_DEBUG("Setup-Fence fenceIndex {} rdata {} setup-rdata {}",
                  m_setupFenceIndex,
                  m_rdataValue,
                  m_setupRdataValue);

    return true;
}

bool MonitorStateMachine::monitorSetupCommandSobj(uint16_t sobjIndex)
{
    if (m_state != MONITOR_STATE_CONSUMED)
    {
        // It might be just a waste of packet(s), or indicate a failure
        // Due to that, we will NOT return false
        LOG_MSM_WARNING("Unexpected Monitor-setup (SOBJ) prior of Monitor usage by Fence (state {})",
                        getStateDesc(m_state));
    }

    m_state           = MONITOR_STATE_INIT;
    m_setupFenceIndex = INVALID_FENCE_INDEX;
    m_rdataValue      = 0;
    m_setupRdataValue = 0;
    m_isFenceSetup    = false;

    LOG_MSM_DEBUG("Setup-SOBJ");

    return true;
}

bool MonitorStateMachine::monitorArmCommand()
{
    if (m_state != MONITOR_STATE_SETUP)
    {
        if (m_setupFenceIndex == INVALID_FENCE_INDEX)
        {
            LOG_MSM_FAILURE("Monitor-Arm prior of any monitor-setup");

            return false;
        }
        else
        {
            // It might be just a waste of packet(s), a failure, or a normal behavior (re-arming with the same setup)
            // Due to that, we will NOT return false
            LOG_MSM_WARNING("Monitor-Arm prior of Monitor setup (might be re-arm) by Fence (state {})",
                            getStateDesc(m_state));
        }
    }

    FenceStateMachine* pFenceSm = nullptr;
    if (m_setupFenceIndex == m_armFenceIndex)
    {
        pFenceSm = &(*m_pFenceStateMachineDB)[m_armFenceIndex];
        if (pFenceSm == nullptr)
        {
            LOG_MSM_FAILURE("Fence SM is invalid for (re-arm) QMAN-ID {}", m_armFenceIndex);
            return false;
        }
        bool status = pFenceSm->rearmMonitor(this);

        m_rdataValue = m_setupRdataValue;
        m_state      = MONITOR_STATE_ARM;

        return status;
    }

    if (m_armFenceIndex != INVALID_FENCE_INDEX)
    {
        pFenceSm = &(*m_pFenceStateMachineDB)[m_armFenceIndex];
        if (pFenceSm == nullptr)
        {
            LOG_MSM_FAILURE("Fence SM is invalid for (arm) QMAN-ID {}", m_armFenceIndex);
            return false;
        }

        if (!pFenceSm->removeMonitor(this))
        {
            return false;
        }
    }

    pFenceSm = &(*m_pFenceStateMachineDB)[m_setupFenceIndex];
    if (pFenceSm == nullptr)
    {
        LOG_MSM_FAILURE("Fence SM is invalid for (setup) QMAN-ID {}", m_setupFenceIndex);
        return false;
    }
    bool status = pFenceSm->addMonitor(this);

    m_armFenceIndex = m_setupFenceIndex;
    m_state         = MONITOR_STATE_ARM;

    LOG_MSM_DEBUG("Arm fenceIndex {} rdata {} setup-rdata {}", m_setupFenceIndex, m_rdataValue, m_setupRdataValue);

    return status;
}

// Return false if not in ARM state
bool MonitorStateMachine::fenceCommand(uint16_t fenceIndex, uint16_t& fenceCounter)
{
    if (m_armFenceIndex != fenceIndex)
    {
        LOG_MSM_FAILURE("Fence-comnnand from Fence-Index {} (while armed Fence-Index {})", fenceIndex, m_armFenceIndex);

        return false;
    }

    if (m_armFenceIndex != m_setupFenceIndex)
    {
        LOG_MSM_FAILURE("Fence-comnnand with different Fence-Index (setup {} arm {})",
                        m_setupFenceIndex,
                        m_armFenceIndex);

        return false;
    }

    if (m_state == MONITOR_STATE_CONSUMED)
    {
        LOG_MSM_DEBUG("Fence-Command Fence-Index {} already consumed", m_armFenceIndex);
        return true;
    }
    else if (m_state == MONITOR_STATE_SETUP)
    {
        LOG_MSM_FAILURE("In Setup state prior of fence packet");

        return false;
    }

    LOG_MSM_DEBUG("Fence-Command fenceIndex {} rdata {}", m_armFenceIndex, m_rdataValue);

    fenceCounter += m_rdataValue;
    m_rdataValue = 0;
    m_state      = MONITOR_STATE_CONSUMED;

    return true;
}
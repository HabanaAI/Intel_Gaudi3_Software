#include "fence_state_machine.hpp"

#include "defs.h"
#include "monitor_state_machine.hpp"

#include "synapse_runtime_logging.h"
#include "utils.h"

using namespace common;

// #define DEBUG_FENCE_STATE_MACHINE

#ifdef DEBUG_FENCE_STATE_MACHINE
#define LOG_FENCE_SM_DEBUG(msg, ...) LOG_CRITICAL(SYN_CS_PARSER, "DBG " msg, ##__VA_ARGS__)
#else
#define LOG_FENCE_SM_DEBUG(msg, ...)
#endif

bool FenceStateMachine::finalize()
{
    if (m_targetValue != 0)
    {
        LOG_GCP_FAILURE("Fence target-value is still set during finalize");

        return false;
    }

    return true;
}

bool FenceStateMachine::addMonitor(MonitorStateMachine* pMonitorSm)
{
    if (pMonitorSm == nullptr)
    {
        LOG_GCP_FAILURE("Unexpectedly nullptr Monitor-SM had been added");

        return false;
    }

    uint64_t fenceMonitorsKey = (uint64_t)pMonitorSm;

    if (m_fenceMonitors.find(fenceMonitorsKey) != m_fenceMonitors.end())
    {
        // Might be just a waste of arming of a (previous configured) monitor
        LOG_GCP_WARNING("Unexpectedly Monitor {} had been added twice", pMonitorSm->getId());
    }

    m_fenceMonitors[fenceMonitorsKey] = pMonitorSm;

    LOG_FENCE_SM_DEBUG("Fence SM {}: add monitor {} (due to arm)", m_fenceIndex, pMonitorSm->getId());

    return true;
}

bool FenceStateMachine::removeMonitor(MonitorStateMachine* pMonitorSm)
{
    if (pMonitorSm == nullptr)
    {
        LOG_GCP_FAILURE("Unexpectedly nullptr Monitor-SM had been removed");

        return false;
    }

    uint64_t fenceMonitorsKey = (uint64_t)pMonitorSm;

    if (m_fenceMonitors.erase(fenceMonitorsKey) == 0)
    {
        // Might be just a waste of arming of a (previous configured) monitor
        LOG_GCP_WARNING("Unexpectedly Monitor {} had been removed but not fenced", pMonitorSm->getId());
    }

    LOG_FENCE_SM_DEBUG("Fence SM {}: remove monitor {}", m_fenceIndex, pMonitorSm->getId());

    return true;
}

bool FenceStateMachine::rearmMonitor(MonitorStateMachine* pMonitorSm)
{
    if (pMonitorSm == nullptr)
    {
        LOG_GCP_FAILURE("Unexpectedly nullptr Monitor-SM had been rearmed");

        return false;
    }

    uint64_t fenceMonitorsKey = (uint64_t)pMonitorSm;

    if (m_fenceMonitors.find(fenceMonitorsKey) == m_fenceMonitors.end())
    {
        m_fenceMonitors[fenceMonitorsKey] = pMonitorSm;
    }
    // else => Already set

    LOG_FENCE_SM_DEBUG("Fence SM {}: re-arm monitor {}", m_fenceIndex, pMonitorSm->getId());

    return true;
}

bool FenceStateMachine::fenceCommand(uint16_t targetValue, uint16_t decrementValue)
{
    if ((m_targetValue != 0) && (m_targetValue != targetValue))
    {
        LOG_GCP_FAILURE("Got Fence packet with invalid target-value (old {} new {})", m_targetValue, targetValue);

        return false;
    }

    m_targetValue    = targetValue;
    m_decrementValue = decrementValue;

    LOG_FENCE_SM_DEBUG("Fence SM {}: fence-command target {} decrement {} counter {}",
                       m_fenceIndex,
                       m_targetValue,
                       m_decrementValue,
                       m_fenceCounter);

    for (auto& [key, pMonitorStateMachine] : m_fenceMonitors)
    {
        UNUSED(key);

        if (!pMonitorStateMachine->fenceCommand(m_fenceIndex, m_fenceCounter))
        {
            return false;
        }

        LOG_FENCE_SM_DEBUG("Fence SM {}: fence-command called monitor {} (new counter {})",
                           m_fenceIndex,
                           pMonitorStateMachine->getId(),
                           m_fenceCounter);
    }

    // We expect that the Fence' samaphore will be cleared, at this point
    if (m_targetValue != m_fenceCounter)
    {
        LOG_GCP_FAILURE("Semaphore of Fence-Index {} was not cleared by monitors (Values: Target {} Counter {})",
                        m_fenceIndex,
                        m_targetValue,
                        m_fenceCounter);

        return false;
    }

    m_targetValue -= m_decrementValue;
    m_fenceCounter -= m_decrementValue;

    return true;
}

void FenceStateMachine::init(uint16_t fenceIndex)
{
    m_fenceIndex = fenceIndex;
}
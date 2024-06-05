#pragma once

#include <unordered_map>
#include <cstdint>
namespace common
{
class MonitorStateMachine;
}

// The Monitor-SM & Fence-SM are defined for verifying that a Fence is properly
// handled by the couple (CP does not get stuck)

namespace common
{
class FenceStateMachine
{
public:
    FenceStateMachine()  = default;
    ~FenceStateMachine() = default;

    void init(uint16_t fenceIndex);

    bool finalize();

    bool addMonitor(MonitorStateMachine* pMonitorSm);
    bool removeMonitor(MonitorStateMachine* pMonitorSm);
    bool rearmMonitor(MonitorStateMachine* pMonitorSm);

    bool fenceCommand(uint16_t targetValue, uint16_t decrementValue);

private:
    uint16_t m_targetValue    = 0;
    uint16_t m_decrementValue = 0;
    uint16_t m_fenceIndex     = 0;
    uint16_t m_fenceCounter   = 0;

    std::unordered_map<uint64_t, MonitorStateMachine*> m_fenceMonitors;
};
}  // namespace common

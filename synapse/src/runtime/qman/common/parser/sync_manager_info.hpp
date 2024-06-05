#pragma once

#include "monitor_state_machine.hpp"

#include <unordered_map>
#include <vector>

namespace common
{
class FenceStateMachine;
}
using FenceStateMachineDB = std::unordered_map<uint64_t, common::FenceStateMachine>;

namespace common
{
class SyncManagerInfo
{
public:
    SyncManagerInfo()  = default;
    ~SyncManagerInfo() = default;

    void init(uint32_t smInstanceId) { m_smInstanceId = smInstanceId; };

    MonitorStateMachine& getMonitorSM(uint32_t monitorId, FenceStateMachineDB* pFenceSmDB);

    bool checkMonitors();

    static constexpr uint32_t SYNC_MNGR_INVALID = std::numeric_limits<uint32_t>::max();

private:
    using MonitorStateMachineDatabase = std::unordered_map<uint64_t, MonitorStateMachine>;

    MonitorStateMachineDatabase m_monitorsStateMachineDB;

    uint32_t m_smInstanceId = SYNC_MNGR_INVALID;
};

class SyncManagerInfoDatabase
{
    typedef std::vector<SyncManagerInfo> SyncManagerInfoDB;

public:
    // Assumes that SM-IDs are in the range of [0, numOfSmId)
    SyncManagerInfoDatabase(uint32_t numOfSmIds);

    SyncManagerInfo& operator[](uint32_t index);

    typename SyncManagerInfoDB::iterator begin() { return m_syncManagerInfoDB.begin(); };
    typename SyncManagerInfoDB::iterator end() { return m_syncManagerInfoDB.end(); };

private:
    SyncManagerInfoDB m_syncManagerInfoDB;
};
}  // namespace common
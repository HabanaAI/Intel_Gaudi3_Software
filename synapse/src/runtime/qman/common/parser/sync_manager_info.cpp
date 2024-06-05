#include "sync_manager_info.hpp"

using namespace common;

// --- SyncManagerInfo ---//
MonitorStateMachine& SyncManagerInfo::getMonitorSM(uint32_t monitorId, FenceStateMachineDB* pFenceSmDB)
{
    MonitorStateMachine& monitorSM = m_monitorsStateMachineDB[monitorId];
    if (!monitorSM.isInitialized())
    {
        monitorSM.init(monitorId, pFenceSmDB);
    }

    return monitorSM;
}

bool SyncManagerInfo::checkMonitors()
{
    for (auto singleMonitorSm : m_monitorsStateMachineDB)
    {
        if (!singleMonitorSm.second.isFinalized())
        {
            return false;
        }
    }

    return true;
}

// --- SyncManagerInfoDatabase ---//
SyncManagerInfoDatabase::SyncManagerInfoDatabase(uint32_t numOfSmIds)
{
    m_syncManagerInfoDB.resize(numOfSmIds);
}

SyncManagerInfo& SyncManagerInfoDatabase::operator[](uint32_t index)
{
    return m_syncManagerInfoDB[index];
}
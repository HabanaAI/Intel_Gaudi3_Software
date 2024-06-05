#include "monitor_setup_manager.h"
#include "sync_types.h"

MonitorSetupManager::MonitorSetupManager(std::shared_ptr<SyncObjectManager>& syncObjectManager):
                                          m_syncObjectManager(syncObjectManager)
{
}

MonitorSetupManager::MonitorSetupManager(const MonitorSetupManager& rhs, std::shared_ptr<SyncObjectManager>& syncObjectManager):
        MonitorSetupManager(rhs)
{
    m_syncObjectManager = syncObjectManager;
}

void MonitorSetupManager::reset()
{
    for (auto engMon : m_deviceToEngineMonitors)
    {
        for (auto vecIter : engMon.second)
        {
            for (auto monObj : vecIter)
            {
                m_syncObjectManager->releaseMonObject(monObj.id);
            }
        }
    }
    m_deviceToEngineMonitors.clear();

    for (auto engFence : m_deviceToEnginesFenceMonitors)
    {
        for (std::map<WaitID, EngineFenceMonitors> vecIter : engFence.second)
        {
            for (const std::pair<const WaitID, EngineFenceMonitors>& monIds : vecIter)
            {
                for (SyncObjectManager::MonitorId monId : monIds.second)
                {
                    m_syncObjectManager->releaseMonObject(monId);
                }
            }
        }
    }
    m_deviceToEnginesFenceMonitors.clear();
}

std::map<int, MonObject> MonitorSetupManager::getMapOfMonitors()
{
    return m_mapOfMonitors;
}

void MonitorSetupManager::addSetupMonitor(HabanaDeviceType             device,
                                          unsigned int                 engineIndex,
                                          SyncObjectManager::MonitorId monId,
                                          SyncObjectManager::SyncId    syncId,
                                          int                          setupValue)
{
    std::vector<EngineSetupMonitors>& engineMonitors = m_deviceToEngineMonitors[device];
    if (engineMonitors.size() <= engineIndex)
    {
        engineMonitors.resize(engineIndex + 1);
    }
    MonObject monitor;
    monitor.id = monId;
    monitor.signalSyncId = syncId;
    monitor.setupValue   = setupValue;
    engineMonitors[engineIndex].push_back(monitor);

    //Add monitor setup to m_mapOfMonitors
    m_mapOfMonitors.insert(std::make_pair(monId, monitor));

}

// Make sure each engine has enough setup monitors
const EngineFenceMonitors&
MonitorSetupManager::getFenceMonitorIds(uint32_t device, unsigned int engineIndex, unsigned int count, WaitID fence)
{
    auto&     enginesMonitors = m_deviceToEnginesFenceMonitors[device];
    MonObject monObjFence;

    if (enginesMonitors.size() <= engineIndex)
    {
        enginesMonitors.resize(engineIndex + 1);
    }
    auto& ret = enginesMonitors[engineIndex][fence];

    // strting from size() in case we already have monitors for this engine that we can use
    for (unsigned int i = ret.size(); i < count; ++i)
    {
        monObjFence.id = m_syncObjectManager->getFreeMonitorId();
        ret.push_back(monObjFence.id);

        monObjFence.signalSyncId = FENCE_MONITOR_ID;
        monObjFence.setupValue   = 1;
        monObjFence.fenceId      = fence;

        //Add monitor setup to m_mapOfMonitors
        m_mapOfMonitors.insert(std::make_pair(monObjFence.id, monObjFence));
    }

    return ret;
}

void MonitorSetupManager::addSetupMonitorForFence(HabanaDeviceType deviceType, unsigned engineId, unsigned monitorId)
{
    std::vector<EngineFenceMonitors>& enginesMonitors = m_deviceTypeToEnginesFenceMonitors[deviceType];

    if (enginesMonitors.size() <= engineId)
    {
        enginesMonitors.resize(engineId + 1);
    }

    enginesMonitors[engineId].push_back(monitorId);
}


void MonitorSetupManager::initSetupMonitorForQueue(CommandQueuePtr queue)
{
    HB_ASSERT(queue != nullptr, "null command queue");

    std::map<unsigned, MonObject> setupMonitors;
    const auto& engineMonitors = m_deviceToEngineMonitors[(HabanaDeviceType)queue->GetDeviceType()];
    if (queue->GetEngineIndex() < engineMonitors.size())
    {
        const auto& monitors = engineMonitors[queue->GetEngineIndex()];
        for (const auto& monitor : monitors)
        {
            setupMonitors[monitor.id] = monitor;
        }
    }

    const auto& engineFenceMonitors = m_deviceToEnginesFenceMonitors[(HabanaDeviceType)queue->GetLogicalQueue()];
    if (queue->GetEngineIndex() < engineFenceMonitors.size())
    {
        MonObject monObjFence;
        monObjFence.signalSyncId = FENCE_MONITOR_ID;
        monObjFence.setupValue   = 1;

        const auto& fenceMonitors = engineFenceMonitors[queue->GetEngineIndex()];
        for (const std::pair<const WaitID, EngineFenceMonitors>& monIdsAndFenceId : fenceMonitors)
        {
            monObjFence.fenceId = monIdsAndFenceId.first;
            for (const auto& monId : monIdsAndFenceId.second)
            {
                monObjFence.id       = monId;
                setupMonitors[monId] = monObjFence;
            }
        }
    }

    const auto& engineFenceMonitorsPerDeviceType = m_deviceTypeToEnginesFenceMonitors[queue->GetDeviceType()];
    if (queue->GetEngineID() < engineFenceMonitorsPerDeviceType.size())
    {
        MonObject monObjFence;
        monObjFence.signalSyncId = FENCE_MONITOR_ID;
        monObjFence.setupValue   = 1;

        // in DEVICE_DMA_DEVICE_HOST and DEVICE_DMA_HOST_DEVICE we just use the default fence0
        monObjFence.fenceId = WaitID::ID_0;

        const auto& fenceMonitors = engineFenceMonitorsPerDeviceType[queue->GetEngineID()];
        for (unsigned int monId : fenceMonitors)
        {
            monObjFence.id = monId;
            setupMonitors[monId] = monObjFence;
        }
    }

    for (const auto& mon : setupMonitors)
    {
        LOG_DEBUG(QMAN,
                  "Setting monitor id {} ({}) to sync {} as setup monitor for queue {}",
                  mon.first,
                  mon.second.id,
                  mon.second.signalSyncId,
                  queue->getName());
    }

    queue->setMonIdToSetupMonitors(setupMonitors);
}

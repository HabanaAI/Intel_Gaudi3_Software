#pragma once

#include "graph_compiler/command_queue.h"
#include "graph_compiler/queue_command_factory.h"
#include "node_annotation.h"
#include "sync_object_manager.h"
#include "sync_types.h"

#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

struct MonObject;

// Since the engine setups a fence, it must contains data=1 and the address
// is a fence so we can just save the monitor's id
typedef std::list<SyncObjectManager::MonitorId> EngineFenceMonitors;
typedef std::list<MonObject> EngineSetupMonitors;

//indication for setup monitor to write for fence
static const int FENCE_MONITOR_ID = 10000;

class MonitorSetupManager
{
public:
    explicit MonitorSetupManager(std::shared_ptr<SyncObjectManager>& syncObjectManager);
    MonitorSetupManager(const MonitorSetupManager& rhs, std::shared_ptr<SyncObjectManager>& syncObjectManager);

    void reset();  // currently unused

    virtual void initReservedMonitorIds(const HalReader& halReader) = 0;

    void addSetupMonitor(HabanaDeviceType             device,
                         unsigned int                 numberOfEngines,
                         SyncObjectManager::MonitorId monId,
                         SyncObjectManager::SyncId    syncId,
                         int                          setupValue);

    // NOTE - the device parameter represents different things for different platforms
    //     For Goya it represents a HabanaDeviceType but for Gaudi it represetns a LogicalQueue
    const EngineFenceMonitors&
    getFenceMonitorIds(uint32_t device, unsigned int engineIndex, unsigned int count, WaitID fence = WaitID::ID_0);

    virtual void initSetupMonitorForQueue(CommandQueuePtr queue);

    std::map<int, MonObject> getMapOfMonitors();

protected:

    void addSetupMonitorForFence(HabanaDeviceType deviceType, unsigned engineId, unsigned monitorId);

    // for gaudi & goya2, key is logical queue; for goya1, key is device type
    std::map<uint32_t, std::vector<std::map<WaitID, EngineFenceMonitors>>> m_deviceToEnginesFenceMonitors;

    // for gaudi & goya2, key is device type; for goya1, not in use
    // Used for DEVICE_DMA_DEVICE_HOST and DEVICE_DMA_HOST_DEVICE devices
    std::map<HabanaDeviceType, std::vector<EngineFenceMonitors> > m_deviceTypeToEnginesFenceMonitors;

    // These monitors aren't signaling the fence. They signal
    // something else: DmaDownFeedbackReset or DmaUpFeedbackReset currently
    // for gaudi not in use; for goya1 & goya2, key is device type
    std::map<HabanaDeviceType, std::vector<EngineSetupMonitors> > m_deviceToEngineMonitors;

    std::shared_ptr<SyncObjectManager> m_syncObjectManager;

    //A map of all monitors object by MonitorId
    std::map<int, MonObject> m_mapOfMonitors;
};

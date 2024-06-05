#include "utils.h"
#include "monitor_setup_manager.h"
#include "platform/gaudi/graph_compiler/hal_conventions.h"
#include "platform/gaudi/graph_compiler/queue_command_factory.h"
#include "sync_scheme_manager.h"
#include "sync_conventions.h"

using namespace gaudi;

MonitorSetupManagerGaudi::MonitorSetupManagerGaudi(std::shared_ptr<SyncObjectManager>& syncObjectManager):
                                                  MonitorSetupManager(syncObjectManager)
{
}

MonitorSetupManagerGaudi::MonitorSetupManagerGaudi(const MonitorSetupManager& rhs, std::shared_ptr<SyncObjectManager>& syncObjectManager):
                                                MonitorSetupManager(rhs, syncObjectManager)
{
}

void MonitorSetupManagerGaudi::initReservedMonitorIds(const HalReader& halReader)
{
    // Add semaphores
    for (auto gaudiType : halReader.getSupportedDeviceTypes())
    {
        for (unsigned i = 0 ; i < getNumEnginesForDeviceType(gaudiType, halReader) ; ++i)
        {
            unsigned lowerQueueId = gaudi::SyncConventions::instance().getLowerQueueID(gaudiType, i);
            addSetupMonitorForFence(gaudiType, i, MON_OBJ_ENGINE_SEM_BASE + lowerQueueId);
        }
    }
}

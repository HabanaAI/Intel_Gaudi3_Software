#pragma once

#include "graph_compiler/sync/monitor_setup_manager.h"
#include "platform/gaudi/graph_compiler/command_queue.h"

class MonitorSetupManagerGaudi: public MonitorSetupManager
{
public:
    explicit MonitorSetupManagerGaudi(std::shared_ptr<SyncObjectManager>& syncObjectManager);
    MonitorSetupManagerGaudi(const MonitorSetupManager& rhs, std::shared_ptr<SyncObjectManager>& syncObjectManager);

    void initReservedMonitorIds(const HalReader& halReader) override;
};

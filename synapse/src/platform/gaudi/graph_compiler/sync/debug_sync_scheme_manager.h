#pragma once

#include "sync_scheme_manager.h"

/**
 * Each node will be dependent on its predecessor
 * NOTE: This sync scheme doesn't support multiple enqueues
 */
class DebugSyncSchemeManagerGaudi : public SyncSchemeManagerGaudi
{
public:
    explicit DebugSyncSchemeManagerGaudi(GaudiGraph* graph);
};

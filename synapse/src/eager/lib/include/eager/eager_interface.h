// This header intends to provide some basic isolation for the Eager library from the GraphCompiler,
// avoiding recursively including a lot of internal impl headers.
#pragma once

// eager includes (relative to src/eager/lib/)
// NOTE: KEEP THIS LIST EMPTY IF POSSIBLE
// #include "eager/lib/eager_brain_base.h"
// #include "eager/lib/utils/general_defs.h"

// synapse api (relative to include/)
#include "synapse_common_types.h"

class HabanaGraph;

namespace eager_mode
{
class EagerMmeBrainBase;

// Generate eager templates. Normally called once at synSingleton::initSingleton().
// If note done, would generate the tempaltes on first use which might be on the hot path.
void createEagerTemplates();

// Check whether an EagerGraph can be created for a given device type
bool isValidForEager(synDeviceType deviceType);

// Create an EagerGraph for the given device type
HabanaGraph* createEagerGraph(synDeviceType deviceType);

const EagerMmeBrainBase& getEagerMmeBrain(const HabanaGraph& eagerGraph);

}  // namespace eager_mode
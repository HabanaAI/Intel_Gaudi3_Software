#include "desc_gen_hal.h"

// eager includes (relative to src/eager/lib/)
#include "utils/general_utils.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "platform/gaudi2/graph_compiler/hal_conventions.h"
#include "platform/gaudi2/graph_compiler/sync/sync_scheme_manager_arc.h"

using namespace gaudi2;

namespace eager_mode::gaudi2_spec_info
{
static_assert(gaudi2::LOGICAL_QUEUE_MAX_ID <= DescGeneratorHal::LOGICAL_QUEUE_MAX_ID,
              "insufficient size for LOGICAL_QUEUE_MAX_ID");

unsigned DescGeneratorHal::deviceTypeToLogicalQueue(EngineType engineType, const Node& node) const
{
    HabanaDeviceType deviceType = engineType2HabanaDeviceType(engineType);
    return gaudi2::deviceTypeToLogicalQueue(deviceType);
}

unsigned DescGeneratorHal::safeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount) const
{
    return OverlapSigToArcSigGaudi2::safeIncrement(logicalId, sigVal, incAmount);
}

}  // namespace eager_mode::gaudi2_spec_info
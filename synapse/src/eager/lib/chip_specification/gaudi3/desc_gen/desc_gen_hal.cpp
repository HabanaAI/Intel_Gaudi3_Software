#include "desc_gen_hal.h"

// eager includes (relative to src/eager/lib/)
#include "utils/general_utils.h"

// synapse-internal gaudi3-specific includes (relative to src/)
#include "platform/gaudi3/graph_compiler/hal_conventions.h"

using namespace gaudi3;

namespace eager_mode::gaudi3_spec_info
{
static_assert(gaudi3::LOGICAL_QUEUE_MAX_ID <= DescGeneratorHal::LOGICAL_QUEUE_MAX_ID,
              "insufficient size for LOGICAL_QUEUE_MAX_ID");

unsigned DescGeneratorHal::deviceTypeToLogicalQueue(EngineType engineType, const Node& node) const
{
    HabanaDeviceType deviceType = engineType2HabanaDeviceType(engineType);
    return gaudi3::deviceTypeToLogicalQueue(deviceType, node);
}

unsigned DescGeneratorHal::safeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount) const
{
    // TODO temporary code to be revisited.
    return sigVal += incAmount;
}

}  // namespace eager_mode::gaudi3_spec_info
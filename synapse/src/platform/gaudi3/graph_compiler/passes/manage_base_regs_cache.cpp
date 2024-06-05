#include "base_regs_cache_manager.h"

#include "gaudi3_graph.h"
#include "habana_pass.h"
#include "hal_reader/gaudi3/hal.h"
#include "platform/gaudi3/graph_compiler/passes.h"

namespace gaudi3
{
static constexpr auto BASE_REGS_CACHE_SIZE = gaudi3::hal().numBaseRegsForAddress;
class BaseRegsCacheManager : public ::BaseRegsCacheManager<LOGICAL_QUEUE_MAX_ID, BASE_REGS_CACHE_SIZE>
{
public:
    BaseRegsCacheManager(Gaudi3Graph& g) : ::BaseRegsCacheManager<LOGICAL_QUEUE_MAX_ID, BASE_REGS_CACHE_SIZE>(g) {}

protected:
    uint64_t getCacheIndex(const NodePtr& node) override
    {
        unsigned logicalId = deviceTypeToLogicalQueue(m_graph.getNodeUtility().getNodeDeviceType(node), *node);
        // MME and Transpose share the same QMAN and base register cache, so return for both the same logical engine ID
        return logicalId == DEVICE_XPS_LOGICAL_QUEUE ? DEVICE_MME_LOGICAL_QUEUE : logicalId;
    }
};

// The pass entry function
bool manageBaseRegsCache(Gaudi3Graph& g)
{
    BaseRegsCacheManager baseRegsCacheManager(g);
    baseRegsCacheManager.go();
    return true;
}

}  // namespace gaudi3

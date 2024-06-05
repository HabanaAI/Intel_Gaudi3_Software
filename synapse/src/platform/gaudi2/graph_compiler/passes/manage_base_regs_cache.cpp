#include "base_regs_cache_manager.h"

#include "gaudi2_graph.h"
#include "habana_pass.h"
#include "hal_reader/gaudi2/hal.h"
#include "platform/gaudi2/graph_compiler/passes.h"

namespace gaudi2
{
static constexpr auto BASE_REGS_CACHE_SIZE = gaudi2::hal().baseRegistersCacheSize;
class BaseRegsCacheManager : public ::BaseRegsCacheManager<LOGICAL_QUEUE_MAX_ID, BASE_REGS_CACHE_SIZE>
{
public:
    BaseRegsCacheManager(Gaudi2Graph& g) : ::BaseRegsCacheManager<LOGICAL_QUEUE_MAX_ID, BASE_REGS_CACHE_SIZE>(g) {}

protected:
    uint64_t getCacheIndex(const NodePtr& node) override
    {
        return deviceTypeToLogicalQueue(m_graph.getNodeUtility().getNodeDeviceType(node));
    }
};

// The pass entry function
bool manageBaseRegsCache(Gaudi2Graph& g)
{
    BaseRegsCacheManager baseRegsCacheManager(g);
    baseRegsCacheManager.go();
    return true;
}


} // namespace gaudi2

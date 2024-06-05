#include "sync_scheme_manager.h"

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "desc_gen/node2desc.h"
#include "eager_graph.h"

// synapse-internal gaudi2-specific includes
#include "platform/gaudi2/graph_compiler/sync/sync_scheme_fw_context.h"

using namespace gaudi2;

namespace eager_mode::gaudi2_spec_info
{

void SyncSchemeManager::generateWorkDistributionContexts(Node2DescContainer& multiNode2Desc) const
{
    SyncSchemeFwContext syncSchemeFwContext(multiNode2Desc.getGraph().getNodeUtility());
    for (SingleNode2Desc& singleNode : multiNode2Desc.getExecSequence())
    {
        singleNode.getDescGen().generateWorkDistributionContexts(&syncSchemeFwContext);
    }
}

}  // namespace eager_mode::gaudi2_spec_info

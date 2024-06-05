#include "tpc_desc.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"
#include "graph_compiler/passes/generate_work_distribution.h"
#include "graph_compiler/types.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "hal_reader/gaudi2/hal.h"
#include "platform/gaudi2/graph_compiler/gaudi2_types.h"
#include "platform/gaudi2/graph_compiler/sync/sync_scheme_fw_context.h"

using namespace gaudi2;

namespace eager_mode::gaudi2_spec_info
{
TpcDescGenerator::TpcDescGenerator(EagerGraph& graph, const EagerNode& node)
: eager_mode::TpcDescGeneratorBase(graph, node)
{
    EAGER_ASSERT(node.getEngineType() == EngineType::TPC, "Invalid engine type");
}

bool TpcDescGenerator::generateTpcDesc()
{
    TPCNode&                             tpcNode = *m_node.get<TPCNode>();
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousNodeLocalityMode = false;
    workDistributionManager::tpcWorkDistribution(tpcNode,
                                                 m_rois,
                                                 m_graph.getNumTpcEng(),
                                                 tpcShuffleIndex,
                                                 previousNodeLocalityMode,
                                                 false,
                                                 true);

    const deviceAddrOffset kernelAddr = m_graph.getProgramDataBlobManager().getKernelAddress(getNode().getUniqueID());
    TpcDescriptorGenerator::generateTpcWdDescriptors(tpcNode, m_rois, kernelAddr, m_descs);
    if (m_descs.size() > 1 || m_descs.size() != m_rois.size()) return false;
    m_descNr = m_descs.size();

    calcTensorsNr();

    return true;
}

// Calculate number of patchable tensors and initialize corresponding base class member variables
void TpcDescGenerator::calcTensorsNr()
{
    EAGER_ASSERT(m_descs.size() == 1, "Unsupported multiple TPC descriptors");
    const TpcDesc& desc = m_descs.begin()->desc;

    EAGER_ASSERT(m_patchableTensorsNr == 0, "Invalid initial state of statistics data");
    for (size_t i = 0; i < desc.c_max_tensors_nr; ++i)
    {
        const block_tpc_tensor& tensor = desc.m_tensors[i];
        // Detect last tensor. It's enough to check size of dim 0, except in ZST (zero size tensors)
        // size at dim 0 can be zero but not tensor_config._raw, hence the check.
        // TODO for ZST support: handle/evaluate a case of n-dim tensor that has zero dims in the second desc.
        const bool missingAddr = (tensor.base_addr_low.v | tensor.base_addr_high.v) == 0;
        if (missingAddr && (tensor.dim_0_size.v == 0)) break;
        ++m_patchableTensorsNr;
    }
    EAGER_ASSERT(calcNumberPatchableTensors(m_node) == m_patchableTensorsNr, "Invalid number of TPC tensors");
}

void TpcDescGenerator::generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant)
{
    auto syncSchemeFwContext = std::get<gaudi2::SyncSchemeFwContext*>(syncSchemeFwContextPtrVariant);
    EAGER_ASSERT_PTR(syncSchemeFwContext);
    EAGER_ASSERT(!getNode().getNodeAnnotation().arcSyncScheme.empty(), "Invalid sync scheme");

    EAGER_ASSERT(m_descs.size() == 1, "Invalid descriptors");
    EAGER_ASSERT(m_descs.size() == m_rois.size(), "Invalid descriptors");
    TpcDescriptorGenerator::DescEntry& descEntry = m_descs.back();
    NodeROI&                           roi       = m_rois.back();
    syncSchemeFwContext->fillArcSyncScheme<tpc_wd_ctxt_t>(m_node, roi.pipelineLevel, descEntry.fwCtx);
    descEntry.fwCtx.switch_bit = 1;
}

deviceAddrOffset TpcDescGenerator::getTensorVirtualAddress(unsigned tensorIdx) const
{
    EAGER_ASSERT(tensorIdx < hal::baseRegistersCacheSize - 1, "Invalid tensor index for TPC node");
    EAGER_ASSERT(m_descs.size() == 1, "Invalid descriptors");
    const TpcDesc& desc = m_descs.back().desc;
    return (static_cast<uint64_t>(desc.m_tensors[tensorIdx].base_addr_high._raw) << 32) +
           desc.m_tensors[tensorIdx].base_addr_low._raw;
}

const Byte* TpcDescGenerator::getDescRaw(unsigned descIdx) const
{
    EAGER_ASSERT(descIdx < m_descs.size(), "getDescRaw descriptor index out of bound");
    EAGER_ASSERT(m_descs.size() == 1, "Invalid descriptors");
    const TpcDesc& desc = m_descs.back().desc;
    return reinterpret_cast<const Byte*>(&desc);
}

const Byte* TpcDescGenerator::getWorkDistributionContextRaw(unsigned descIdx) const
{
    EAGER_ASSERT(descIdx < m_descs.size(), "getDescRaw descriptor index out of bound");
    EAGER_ASSERT(m_descs.size() == 1, "Invalid descriptors");
    const TpcDescriptorGenerator::DescEntry& descEntry = m_descs.back();
    return reinterpret_cast<const Byte*>(&descEntry.fwCtx);
}

}  // namespace eager_mode::gaudi2_spec_info

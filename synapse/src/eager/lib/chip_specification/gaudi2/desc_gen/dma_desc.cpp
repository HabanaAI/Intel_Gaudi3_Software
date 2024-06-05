#include "dma_desc.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "node_info/eager_node.h"
#include "utils/general_defs.h"
#include "utils/numeric_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/dma_node.h"
#include "graph_compiler/passes/project_node_rois_to_tensor_rois.h"
#include "graph_compiler/passes/split_to_physical_rois.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "hal_reader/gaudi2/hal.h"
#include "platform/gaudi2/graph_compiler/command_queue.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_types.h"
#include "platform/gaudi2/graph_compiler/sync/sync_scheme_fw_context.h"

// relative to <specs>/
#include "gaudi2/asic_reg_structs/dma_core_ctx_regs.h"

// std includes
#include <cstdint>
#include <list>

using namespace gaudi2;

namespace eager_mode::gaudi2_spec_info
{

DmaDescGenerator::DmaDescGenerator(EagerGraph& graph, const EagerNode& node) : DescGeneratorBase(graph, node)
{
    EAGER_ASSERT(node.getEngineType() == EngineType::DMA, "Invalid engine type");
    m_patchableTensorsNr = node->getInputs().size() + node->getOutputs().size();
}

static void setRoisAddress(TensorROIVector& tRois)
{
    for (TensorROI& tRoi : tRois)
    {
        EAGER_ASSERT(tRoi.m_parentTensor != nullptr && !tRoi.m_parentTensor->isShapeTensor(), "Invalid ROI tensor");
        tRoi.getLayout().isReduction = tRoi.m_parentTensor->isReductionEnabled();
        tRoi.getLayout().inSram      = false;
        tRoi.getLayout().baseAddress = tRoi.m_parentTensor->getDramOffset();
    }
}

bool DmaDescGenerator::generateDesc()
{
    EAGER_ASSERT(getNode().getDmaType() == DMA_TYPE_INTERNAL, "Wrong node type");

    std::list<NodeROI> rois;
    rois.push_back(getNode().generateRoi());

    QueueDispatcherParams params = m_graph.getEagerDMADispatcherParams();
    getNode().setParallelLevel(params.parallelLevel);
    getNode().setDispatcherIndex(params.index);

    // Logical ROI split for broadcast as it fails when using single ROI
    EAGER_ASSERT(!getNode().isBroadcast(), "Broadcast is not supported by DMA");
    m_logicalRoisNr = rois.size();
    EAGER_ASSERT(m_logicalRoisNr == 1, "Multiple ROIs are not supported");
    {
        NodeROI& roi = rois.back();
        projectDmaRoi(roi, *m_node.get());
        setRoisAddress(roi.inputRois);
        setRoisAddress(roi.outputRois);
    }

    splitToPhysicalROIsForNode(m_graph, m_node, &rois, m_physicalRois);
    EAGER_ASSERT(!m_physicalRois.empty(), "Invalid physical ROIs");
    DescriptorGenerator::generateDmaDescriptors(*m_node.get<DMANode>(), m_physicalRois, m_descs);

    if (m_descs.size() != m_physicalRois.size())
    {
        return false;
    }

    constexpr size_t enginesNr = hal::numInternalDmaEngines;
    // Set signaling for last activation
    {
        size_t lastActivationSize = m_descs.size() % enginesNr;
        if (lastActivationSize == 0)
        {
            lastActivationSize = enginesNr;  // All engines will signal
        }
        auto it = m_descs.rbegin();
        for (size_t i = 0; i < lastActivationSize; ++i)
        {
            it->desc.ctx.wr_comp_wdata.val = gaudi2::DmaDescQueue::calcDescriptorSignaling();
            ++it;
        }
    }

    // Calculate info required for recipe creation
    {
        m_isNopDescNeeded = ((m_descs.size() % enginesNr) != 0);
        // Number of descriptors + optional NOP descriptor for idle engines
        m_descNr          = m_descs.size() + (m_isNopDescNeeded ? 1 : 0);
        m_activationsNr   = div_round_up(m_descs.size(), enginesNr);
        m_requiredWdCtxNr = DescGeneratorBase::calcRequiredWdCtxNr(divRoundUp(m_descs.size(), enginesNr));
    }

    m_sequentialIterTracker.initialize(m_descs.begin(), m_descNr);

    return true;
}

void DmaDescGenerator::generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant)
{
    auto syncSchemeFwContext = std::get<gaudi2::SyncSchemeFwContext*>(syncSchemeFwContextPtrVariant);
    EAGER_ASSERT_PTR(syncSchemeFwContext);
    [[maybe_unused]] std::vector<ArcSyncInteraction>& arcSyncScheme = getNode().getNodeAnnotation().arcSyncScheme;
    EAGER_ASSERT(!arcSyncScheme.empty(), "Invalid sync scheme");

    constexpr size_t enginesNr               = hal::numInternalDmaEngines;
    const size_t     descIdxOfLastActivation = (m_activationsNr - 1) * enginesNr;
    bool             enableSignal            = (descIdxOfLastActivation == 0);

    edma_wd_ctxt_t dmaFwCtx = {};
    syncSchemeFwContext->fillArcSyncScheme<edma_wd_ctxt_t>(m_node, 0, dmaFwCtx);
    // Complete essential values missing in ctxt
    auto commitRegVal =
        gaudi2::DmaDescQueue::getCommitRegVal(*m_node.get<DMANode>(), m_descs.back().desc, enableSignal);
    dmaFwCtx.dma_commit_reg     = commitRegVal._raw;
    dmaFwCtx.switch_bit         = 1;
    dmaFwCtx.dma_op             = EDMA_OP_NO_WD;
    dmaFwCtx.use_alternate_addr = 0; // See SW-138366 for info on this field

    switch (m_activationsNr)
    {
        case 1:
        {
            dmaFwCtx.sig_inc_value = 1;
            m_wdCtxs[0]            = dmaFwCtx;
        }
        break;

        case 2:
        {
            // Handle first activation
            dmaFwCtx.sig_inc_value      = 0;
            dmaFwCtx.use_alternate_addr = m_node->isTranspose() ? 1 : 0;
            m_wdCtxs[0]                 = dmaFwCtx;
            // Handle second activation
            dmaFwCtx.sig_inc_value      = 1;
            dmaFwCtx.virtual_sob_bitmap = 0;
            commitRegVal.wr_comp_en     = 1;
            dmaFwCtx.use_alternate_addr = 0;
            dmaFwCtx.dma_commit_reg     = commitRegVal._raw;
            m_wdCtxs[1]                 = dmaFwCtx;
        }
        break;

        default:
        {
            // Handle first activation
            dmaFwCtx.sig_inc_value      = 0;
            dmaFwCtx.use_alternate_addr = m_node->isTranspose() ? 1 : 0;
            m_wdCtxs[0]                 = dmaFwCtx;
            // Handle middle activations
            dmaFwCtx.virtual_sob_bitmap = 0;
            m_wdCtxs[1]                 = dmaFwCtx;
            // Handle last activation
            dmaFwCtx.sig_inc_value      = 1;
            commitRegVal.wr_comp_en     = 1;
            dmaFwCtx.use_alternate_addr = 0;
            dmaFwCtx.dma_commit_reg     = commitRegVal._raw;
            m_wdCtxs[2]                 = dmaFwCtx;
        }
        break;
    }
}

deviceAddrOffset DmaDescGenerator::getTensorVirtualAddress(unsigned tensorIdx) const
{
    EAGER_ASSERT(tensorIdx < RecipeHalBase::maxDmaTensorsNr, "Invalid tensor index for DMA node");
    const DmaDesc& desc = m_descs.back().desc;
    EAGER_ASSERT(!m_node->getOutputs().empty(), "No DMA outputs");
    switch (tensorIdx)
    {
        case 0:
            if (m_node->getInputs().empty())
            {
                return (static_cast<uint64_t>(desc.ctx.dst_base_hi._raw) << 32) + desc.ctx.dst_base_lo._raw;
            }
            else
            {
                return (static_cast<uint64_t>(desc.ctx.src_base_hi._raw) << 32) + desc.ctx.src_base_lo._raw;
            }
        case 1:
            EAGER_ASSERT(!m_node->getInputs().empty(), "No DMA inputs");
            return (static_cast<uint64_t>(desc.ctx.dst_base_hi._raw) << 32) + desc.ctx.dst_base_lo._raw;
        default:
            EAGER_ASSERT(false, "Invalid tensor index for DMA node");
            return -1;
    }
}

const Byte* DmaDescGenerator::getDescRaw(unsigned descIdx) const
{
    const DmaDesc& desc = m_sequentialIterTracker.getIter(descIdx)->desc;
    return reinterpret_cast<const Byte*>(&desc);
}

const Byte* DmaDescGenerator::getWorkDistributionContextRaw(unsigned descIdx) const
{
    EAGER_ASSERT(descIdx < m_activationsNr * hal::numInternalDmaEngines, "The given DMA desc index is out of bound");
    const unsigned activationId = descIdx / hal::numInternalDmaEngines;
    if (activationId == 0) return reinterpret_cast<const Byte*>(&m_wdCtxs[0]);
    const bool isMidActivation = (m_activationsNr == 2) || (activationId < (m_activationsNr - 1));
    if (isMidActivation) return reinterpret_cast<const Byte*>(&m_wdCtxs[1]);
    return reinterpret_cast<const Byte*>(&m_wdCtxs[2]);
}

}  // namespace eager_mode::gaudi2_spec_info
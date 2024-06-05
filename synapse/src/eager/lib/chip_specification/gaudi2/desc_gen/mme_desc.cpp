#include "mme_desc.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "node_info/eager_node.h"
#include "utils/general_defs.h"
#include "utils/numeric_utils.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "hal_reader/gaudi2/hal.h"
#include "platform/gaudi2/graph_compiler/command_queue.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "platform/gaudi2/graph_compiler/passes/generate_mme_descriptors.h"
#include "platform/gaudi2/graph_compiler/sync/sync_scheme_fw_context.h"

// relative to <mme>/
#include "include/mme_common/mme_brain.h"
#include "include/mme_common/mme_common_enum.h"

// std includes
#include <memory>

using namespace gaudi2;
using namespace Gaudi2;
using namespace MmeCommon;

namespace eager_mode::gaudi2_spec_info
{
// TODO: Wrapped to avoid uncatchable exception, can drop the function if the init is constexpr
static const MmeLayerParams& GetMmeBrainDefaultParams()
{
    // would have been better to have the following as constexpr, but that
    // first requires changes in MME stack.
    static const auto MME_BRAIN_DEFAULT_PARAMS = MmeBrain::getDefaultParams(e_mme_Gaudi2);
    return MME_BRAIN_DEFAULT_PARAMS;
}

static constexpr Mme::MmePerfEvt getPerfRegisterValue(EMmeTraceEngine engine, bool eventOnStart, bool eventOnEnd)
{
    Mme::MmePerfEvt reg = {};
    if (engine == e_mme_trace_input)
    {
        reg.operand = e_mme_operand_0;  // trace should be sent when any port starts, use 0 for simplicity
    }
    else
    {
        reg.operand = e_mme_operand_0 | e_mme_operand_1;  // trace should be sent when both output ports finish
    }
    reg.incMask             = 0;
    reg.rst                 = 1;
    reg.loopMask            = e_mme_outer_loop;
    reg.startEndMask        = (eventOnStart ? 1 : 0) | (eventOnEnd ? 2 : 0);
    reg.slaveSendsPerfEvent = 1;
    return reg;
}

MmeDescGenerator::MmeDescGenerator(EagerGraph& graph, const EagerNode& node)
: MmeDescGeneratorBase(graph, node, GetMmeBrainDefaultParams())
{
}

bool MmeDescGenerator::generateDesc()
{
    EAGER_ASSERT(m_node != nullptr, "Invalid MME node");
    m_node.get()->getNodeAnnotation().splitToLogicalROIs = false;
    // we first query the mme descriptor cache to see if the activations are available.
    // If they are, we just grab them, otherwise we generate them using descriptor generator.
    MMEBrain::generateLayerParamsFromNode(m_graph, m_node, m_params);
    m_cachedActivations = Gaudi2::MmeDescriptorGenerator::getSharedOwnershipForCachedActivations(m_params);
    if (m_cachedActivations != nullptr)
    {
        m_activationsPtr = m_cachedActivations.get();
    }
    else
    {
        m_descGenerator  = MMEBrain::createDescriptorGenerator(m_graph, m_node, m_params);
        m_activationsPtr = &m_descGenerator->getMmeActivations();
        EAGER_ASSERT(m_descGenerator->getParams().controls.signalingMode == e_mme_signaling_once,
                     "Signaling mode is not supported");
    }

    if ((m_activationsPtr == nullptr) || m_activationsPtr->empty())
    {
        return false;
    }

    // calculate patched addresses
    TensorPtr      aTensor, bTensor, cTensor, oTensor;
    AuxTensorArray auxTensors;
    gaudi2::DescriptorGenerator::getInputOutputTensors(*m_node.get<MmeNode>(),
                                                       aTensor,
                                                       bTensor,
                                                       cTensor,
                                                       oTensor,
                                                       auxTensors);
    // Last desc will signal, so output addr is valid
    const MmeDesc& desc        = m_activationsPtr->back().descriptors.back();
    m_operandVirtualAddress[0] = desc.baseAddrA.addr + aTensor->getTensorOffset();
    m_operandVirtualAddress[1] = desc.baseAddrB.addr + bTensor->getTensorOffset();
    m_operandVirtualAddress[2] = desc.baseAddrCOut0.addr + cTensor->getTensorOffset();

    // Calculate info required for recipe creation
    {
        m_activationsNr   = m_activationsPtr->size();
        m_descNr          = m_activationsNr * m_activationsPtr->back().descriptors.size();
        m_logicalRoisNr   = 1;
        m_requiredWdCtxNr = DescGeneratorBase::calcRequiredWdCtxNr(m_activationsNr);
    }

    return true;
}

void MmeDescGenerator::generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant)
{
    auto syncSchemeFwContext = std::get<gaudi2::SyncSchemeFwContext*>(syncSchemeFwContextPtrVariant);
    EAGER_ASSERT_PTR(syncSchemeFwContext);
    EAGER_ASSERT(!getNode().getNodeAnnotation().arcSyncScheme.empty(), "Invalid sync scheme");

    mme_wd_ctxt_t mmeFwCtx = {};
    syncSchemeFwContext->fillArcSyncScheme<mme_wd_ctxt_t>(m_node, 0, mmeFwCtx);
    // Complete essential values missing in ctxt
    static const auto commitRegVal = gaudi2::MmeQueue::getCommitRegVal();
    mmeFwCtx.mme_commit_reg        = commitRegVal;
    mmeFwCtx.switch_bit            = 1;

    const auto sigIncVal = std::max(mmeFwCtx.sig_inc_value, decltype(mme_wd_ctxt_t::sig_inc_value)(1));
    switch (m_activationsNr)
    {
        case 1:
        {
            m_wdCtxs[0] = mmeFwCtx;
        }
        break;

        case 2:
        {
            // Handle first activation
            mmeFwCtx.sig_inc_value = 0;
            m_wdCtxs[0]            = mmeFwCtx;
            // Handle second activation
            mmeFwCtx.sig_inc_value      = sigIncVal;
            mmeFwCtx.virtual_sob_bitmap = 0;
            m_wdCtxs[1]                 = mmeFwCtx;
        }
        break;

        default:
        {
            // Handle first activation
            mmeFwCtx.sig_inc_value = 0;
            m_wdCtxs[0]            = mmeFwCtx;
            // Handle middle activations
            mmeFwCtx.virtual_sob_bitmap = 0;
            m_wdCtxs[1]                 = mmeFwCtx;
            // Handle last activation
            mmeFwCtx.sig_inc_value = sigIncVal;
            m_wdCtxs[2]            = mmeFwCtx;
        }
        break;
    }
}

deviceAddrOffset MmeDescGenerator::getTensorVirtualAddress(unsigned tensorIdx) const
{
    EAGER_ASSERT(tensorIdx < RecipeHalBase::maxMmeTensorsNr, "Invalid tensor index for MME node");
    EAGER_ASSERT((m_activationsPtr != nullptr) && !m_activationsPtr->empty(), "Invalid MME activations");
    return m_operandVirtualAddress[tensorIdx];
}

const Byte* MmeDescGenerator::getDescRaw(unsigned descIdx) const
{
    EAGER_ASSERT_PTR(m_activationsPtr);
    const unsigned                                activationId = descIdx / hal::numMmeEngines;
    const auto&                                   currentActivation = (*m_activationsPtr)[activationId];
    const unsigned                                subDescIdx   = descIdx % hal::numMmeEngines;
    EAGER_ASSERT(subDescIdx < currentActivation.descriptors.size(), "The given MME desc index is out of bound");
    const MmeDesc& desc = currentActivation.descriptors[subDescIdx];
    return reinterpret_cast<const Byte*>(&desc);
}

const Byte* MmeDescGenerator::getWorkDistributionContextRaw(unsigned descIdx) const
{
    EAGER_ASSERT(descIdx < m_activationsNr * hal::numMmeEngines, "The given MME desc index is out of bound");
    const unsigned activationId = descIdx / hal::numMmeEngines;
    if (activationId == 0) return reinterpret_cast<const Byte*>(&m_wdCtxs[0]);
    const bool isMidActivation = (m_activationsNr == 2) || (activationId < (m_activationsNr - 1));
    if (isMidActivation) return reinterpret_cast<const Byte*>(&m_wdCtxs[1]);
    return reinterpret_cast<const Byte*>(&m_wdCtxs[2]);
}

void MmeDescGenerator::copyPerfDescInfoToBlob(Byte*          out,
                                              unsigned       activationIdx,
                                              StructSizeType offsetInDescriptor,
                                              BlobSizeType   sizeToCopy) const
{
    EMmeTraceMode            traceMode        = m_params.tracing.traceMode;
    constexpr StructSizeType perfEvtInOffset  = offsetof(MmeDesc, perfEvtIn);
    constexpr StructSizeType perfEvtOutOffset = offsetof(MmeDesc, perfEvtOut);
    EAGER_ASSERT_PTR(m_activationsPtr);

    bool firstActivation = activationIdx == 0;
    bool lastActivation  = activationIdx == m_activationsPtr->size() - 1;
    bool partialDesc     = ((*m_activationsPtr)[activationIdx].numSignals == 0);

    if (partialDesc && traceMode == MmeCommon::e_mme_trace_mode_desc)
    {
        traceMode = MmeCommon::e_mme_trace_mode_layer_act;
    }

    Mme::MmePerfEvt perfEvtIn  = {};
    Mme::MmePerfEvt perfEvtOut = {};
    switch (traceMode)
    {
        case e_mme_trace_mode_layer_act:
            if (firstActivation)
            {
                perfEvtIn = getPerfRegisterValue(e_mme_trace_input, true, false);
            }
            if (lastActivation)
            {
                perfEvtOut = getPerfRegisterValue(e_mme_trace_output, false, true);
            }
            break;
        case e_mme_trace_mode_desc:
            perfEvtIn  = getPerfRegisterValue(e_mme_trace_input, true, false);
            perfEvtOut = getPerfRegisterValue(e_mme_trace_output, false, true);
            break;
        case e_mme_trace_mode_advanced:
            perfEvtIn  = getPerfRegisterValue(e_mme_trace_input, true, true);
            perfEvtOut = getPerfRegisterValue(e_mme_trace_output, true, true);
            break;
        default:
            EAGER_ASSERT(false, "invalid trace mode");
    }
    // add specific node context
    const auto& mmeNode = *m_node.get<MmeNode>();
    perfEvtIn.value     = mmeNode.getContextId();
    perfEvtOut.value    = mmeNode.getContextId();
    // copy to blob in case offset is in range
    if (isOffsetInRange<size_t>(offsetInDescriptor, sizeToCopy, perfEvtInOffset))
    {
        std::memcpy(out + perfEvtInOffset - offsetInDescriptor, &perfEvtIn, sizeof(Mme::MmePerfEvt));
    }
    if (isOffsetInRange<size_t>(offsetInDescriptor, sizeToCopy, perfEvtOutOffset))
    {
        std::memcpy(out + perfEvtOutOffset - offsetInDescriptor, &perfEvtOut, sizeof(Mme::MmePerfEvt));
    }
}

void MmeDescGenerator::copyDescToBlob(Byte*          out,
                                      unsigned       descIdx,
                                      StructSizeType offsetInDescriptor,
                                      BlobSizeType   sizeToCopy) const
{
    EAGER_ASSERT_PTR(m_activationsPtr);
    const unsigned activationId      = descIdx / hal::numMmeEngines;
    const auto&    currentActivation = (*m_activationsPtr)[activationId];
    const unsigned subDescIdx        = descIdx % hal::numMmeEngines;
    EAGER_ASSERT(subDescIdx < currentActivation.descriptors.size(), "The given MME desc index is out of bound");
    const MmeDesc& desc     = currentActivation.descriptors[subDescIdx];
    auto           descData = reinterpret_cast<const Byte*>(&desc) + offsetInDescriptor;
    std::memcpy(out, descData, sizeToCopy);
    if (unlikely(m_params.tracing.traceMode != e_mme_trace_mode_none))
    {
        copyPerfDescInfoToBlob(out, activationId, offsetInDescriptor, sizeToCopy);
    }
}

}  // namespace eager_mode::gaudi2_spec_info

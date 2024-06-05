#include "platform/gaudi3/graph_compiler/command_queue.h"

#include "defs.h"
#include "gaudi3_types.h"
#include "habana_global_conf.h"
#include "habana_nodes.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "node_annotation.h"
#include "platform/gaudi3/graph_compiler/queue_command_factory.h"
#include "queue_command.h"
#include "sync/sync_conventions.h"

//---------------------------------------------------------
//---------------------- MmeQueue -------------------------
//---------------------------------------------------------

gaudi3::MmeQueue::MmeQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::MmeQueue<gaudi3::MmeDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_MME, engineId))
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = Gaudi3HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;
}

uint32_t gaudi3::MmeQueue::getCommitRegVal(bool isTranspose)
{
    gaudi3::Mme::MmeCmd cmd = {};
    cmd.dw                 = 0;
    cmd.aguIn              = 0b1111;
    cmd.eu                 = 0b1;
    cmd.ap                 = 0b1;
    cmd.aguOut             = 0b1;
    cmd.aguOutDma          = 0b1;
    cmd.copyAndInc         = 0b1;
    cmd.descSel            = 0b00;
    cmd.maskIdleIndication = 0b0;
    cmd.nullDesc           = 0b0;
    cmd.dmaDesc             = isTranspose ? 0b1 : 0b0;
    cmd.nullDmaDesc        = 0b0;
    return cmd.dw;
}

mme_op_type_t gaudi3::MmeQueue::getOpType(bool isTranspose)
{
    return isTranspose ? MME_OP_TRANSPOSE : MME_OP_COMPUTE;
}

QueueCommandPtr gaudi3::MmeQueue::getExeCmd(NodePtr n, const DescriptorWrapper<MmeDesc>& descWrap, bool enableSignal)
{
    HB_ASSERT_PTR(n);
    mme_wd_ctxt_t mmeFwCtx    = descWrap.getFwCtx();
    bool          isTranspose = MmeNode::isDmaOperation(n);
    mmeFwCtx.mme_commit_reg   = getCommitRegVal(isTranspose);
    mmeFwCtx.mme_op_type      = getOpType(isTranspose);
    auto ret = std::make_unique<gaudi3::ArcExeWdMme>(mmeFwCtx);
    ret->SetContainerInfo(descWrap.getBasicFieldsContainerInfoForCtx());
    return ret;
}

void gaudi3::MmeQueue::forceStaticConfig()
{
    // Ensure we have static configuration at the top of the queue
    if (m_queueExe.empty() || m_queueExe.back()->isDynamic())
    {
        QueueCommandPtr nop = std::make_unique<gaudi3::Nop>();
        PushBack(std::move(nop), false);
    }
}

DescriptorShadow& gaudi3::MmeQueue::getDescShadow(const NodePtr& n)
{
    HB_ASSERT_PTR(n);
    return (gaudi3::deviceTypeToLogicalQueue(n->getNodeDeviceType(), *n) == DEVICE_XPS_LOGICAL_QUEUE) ?
        m_descriptorShadowDma : m_descriptorShadowGemm;
}

void gaudi3::MmeQueue::pushAdditionalDynamicCmds4sobReset(const NodePtr& node, unsigned pipeLevel)
{
    // check if current pipeline level requires sync object reset after it, and if so, create ResetSobs command
    // and push it to the queue. Gaudi3's MME block holds both GEMM and Transpose (xps) engines and we might need
    // to join their reset commands.
    bool isResetRequired = pipeLevel < node->getNodeAnnotation().arcSyncScheme.size() &&
                           node->getNodeAnnotation().arcSyncScheme[pipeLevel].sobResetTotalNumEngs > 0;

    if (!isResetRequired) return;

    unsigned totalNumEngs     = node->getNodeAnnotation().arcSyncScheme[pipeLevel].sobResetTotalNumEngs;
    unsigned myResetId        = node->getNodeAnnotation().arcSyncScheme[pipeLevel].sobResetId;
    unsigned sobTargetValGemm = 0;
    unsigned sobTargetValXps  = 0;
    bool     iAmXps           = false;

    if (gaudi3::deviceTypeToLogicalQueue(node->getNodeDeviceType(), *node) == DEVICE_XPS_LOGICAL_QUEUE)
    {
        sobTargetValXps = node->getNodeAnnotation().arcSyncScheme[pipeLevel].emittedSigVal.value();
        iAmXps = true;
    }
    else
    {
        sobTargetValGemm = node->getNodeAnnotation().arcSyncScheme[pipeLevel].emittedSigVal.value();
    }

    // Check if we need to join the reset of both gemm and transpose engines. An equality in reset ID means that
    // the previous node that passed through this code was of the other type (i.e. I am gemm and the previous was
    // transpose, or the vice versa) and we need to join them to a single reset command.
    if (myResetId == m_prevResetId)
    {
        HB_ASSERT_PTR(m_prevResetCmd);
        if (iAmXps)
        {
            sobTargetValGemm = m_prevResetCmd->getTarget(); // bring the target value for gemm from the other
        }
        else
        {
            sobTargetValXps = m_prevResetCmd->getTargetXps(); // bring the target value for transpose from the other
        }

        // Cancel the previous command by setting its targets to zero; later, we will not create ECB Reset SOB
        // command if the targets are zero. We don't remove here the command from the queue entirely since it
        // might carry some important flags, like blob terminator or switch bit.
        m_prevResetCmd->setTarget(0);
        m_prevResetCmd->setTargetXps(0);
        HB_ASSERT(sobTargetValGemm > 0 && sobTargetValXps > 0, "Gemm & Xps joined reset must have both targets set");
    }

    PushBack(std::move(std::make_unique<ResetSobs>(sobTargetValGemm, totalNumEngs, sobTargetValXps)), false);
    m_prevResetId = myResetId;
    m_prevResetCmd = dynamic_cast<ResetSobs*>(m_queueExe.back().get());
}

void gaudi3::MmeQueue::pushAdditionalDynamicCmds4mcidRollover(const NodePtr& node, unsigned pipeLevel)
{
    // check if current pipeline level requires mcid rollover after it, and if so, create McidRollover command
    // and push it to the queue. Gaudi3's MME block holds both GEMM and Transpose (xps) engines and we might need
    // to join their rollover commands.
    const std::list<NodeROI>& logicalRois = *(node->getLogicalRois());
    HB_ASSERT(pipeLevel < logicalRois.size(), "pipeLevel OOB");
    std::list<NodeROI>::const_iterator itrRoi = std::next(logicalRois.begin(), pipeLevel);
    bool isRolloverRequired = itrRoi->rolloverIds.size() > 0;

    if (!isRolloverRequired) return;

    // To avoid complex code for unrealistic situations, we don't expect to get more than 1 rollover
    // indication on MME or XPS node ROI. Compilation will fail by the CME pass if this rule isn't met.
    HB_ASSERT(itrRoi->rolloverIds.size() == 1, "Not supporting more than 1 rollover per ROI for MME");

    unsigned myRolloverId     = itrRoi->rolloverIds.front();
    unsigned sobTargetValGemm = 0;
    unsigned sobTargetValXps  = 0;
    bool     iAmXps           = false;

    if (gaudi3::deviceTypeToLogicalQueue(node->getNodeDeviceType(), *node) == DEVICE_XPS_LOGICAL_QUEUE)
    {
        sobTargetValXps = node->getNodeAnnotation().arcSyncScheme[pipeLevel].emittedSigVal.value();
        iAmXps = true;
    }
    else
    {
        sobTargetValGemm = node->getNodeAnnotation().arcSyncScheme[pipeLevel].emittedSigVal.value();
    }

    // Check if we need to join the rollover of both gemm and transpose engines. An equality in rollover ID means that
    // the previous node that passed through this code was of the other type (i.e. I am gemm and the previous was
    // transpose, or the vice versa) and we need to join them to a single rollover command.
    if (myRolloverId == m_prevRolloverId)
    {
        HB_ASSERT_PTR(m_prevRolloverCmd);
        if (iAmXps)
        {
            sobTargetValGemm = m_prevRolloverCmd->getTarget(); // bring the target value for gemm from the other
        }
        else
        {
            sobTargetValXps = m_prevRolloverCmd->getTargetXps(); // bring the target value for transpose from the other
        }

        // Cancel the previous command by setting its targets to zero; later, we will not create ECB Rollover
        // command if the targets are zero. We don't remove here the command from the queue entirely since it
        // might carry some important flags, like blob terminator or switch bit.
        m_prevRolloverCmd->setTarget(0);
        m_prevRolloverCmd->setTargetXps(0);
        HB_ASSERT(sobTargetValGemm > 0 && sobTargetValXps > 0, "Gemm & Xps joined rollover must have both targets set");
    }

    PushBack(std::move(std::make_unique<McidRollover>(sobTargetValGemm, sobTargetValXps)), false);
    LOG_DEBUG(CACHE_MAINT, "Adding mcid rollover cmd (rolloverId={}) to {}", myRolloverId, getName());
    m_prevRolloverId = myRolloverId;
    m_prevRolloverCmd = dynamic_cast<McidRollover*>(m_queueExe.back().get());
}


//---------------------------------------------------------
//---------------------- TpcQueue -------------------------
//---------------------------------------------------------

gaudi3::TpcQueue::TpcQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::TpcQueue<gaudi3::TpcDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_TPC, engineId))
{
    m_engineId         = engineId;
    m_engineIndex      = engineIndex;
    m_maxStreams       = Gaudi3HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents   = sendSyncEvents;
}

QueueCommandPtr gaudi3::TpcQueue::getExeCmd(pNode n, const DescriptorWrapper<TpcDesc>& descWrap, bool enableSignal)
{
    std::unique_ptr<gaudi3::ArcExeWdTpc> tpcExeWdCmd = std::make_unique<gaudi3::ArcExeWdTpc>();
    for (unsigned i = 0; i < descWrap.getFwCtxCount(); i++)
    {
        tpcExeWdCmd->addCtx(descWrap.getFwCtx(i));
    }
    tpcExeWdCmd->SetContainerInfo(descWrap.getBasicFieldsContainerInfoForCtx());

    return tpcExeWdCmd;
}

DescriptorShadow::AllRegistersProperties
gaudi3::TpcQueue::registersPropertiesForDesc(pNode n, const DescriptorWrapper<TpcDesc>& descWrapper)
{
    std::vector<DescriptorShadow::RegisterProperties> mask(
        MASK_SIZE(TpcDesc),
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    // We must ensure the kernel id is programmed in order to not miss traces in profiler
    if (GCFG_ENABLE_PROFILER.value())
    {
        DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment
            {
                MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_id),
                MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_id_inc)
            },
            DescriptorShadow::RegisterProperties::createFromHandling(
                DescriptorShadow::RegisterDataHandling::AlwaysWrite));
    }

    // Using data/ignore masking from descriptor generator to ignore registers.
    if (descWrapper.getMask().has_value())
    {
        for (size_t i = 0; i < descWrapper.getMask().value().size(); i++)
        {
            if (!descWrapper.getMask().value()[i])
            {
                mask[i].ignore = true;
            }
        }
    }

    return std::make_shared<std::vector<DescriptorShadow::RegisterProperties>>(mask);
}
//---------------------------------------------------------
//---------------------- RotatorQueue -------------------------
//---------------------------------------------------------

gaudi3::RotatorQueue::RotatorQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
        ::RotatorQueue<gaudi3::RotatorDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_ROTATOR, engineId))
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = Gaudi3HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;
}

void gaudi3::RotatorQueue::setDescriptorSignaling(gaudi3::RotatorDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    desc.cpl_msg_data.val = 0;
    desc.cpl_msg_en.val = 1;

    desc.cpl_msg_data.val |= (0xC0000000 & (SyncObjOp::SYNC_OP_ADD << 29)); // increment
    desc.cpl_msg_data.val |= 1;                                             // by 1
}

QueueCommandPtr
gaudi3::RotatorQueue::getExeCmd(NodePtr n, const DescriptorWrapper<RotatorDesc>& descWrap, bool enableSignal)
{
    rot_wd_ctxt_t rotFwCtx = descWrap.getFwCtx();
    rotFwCtx.cpl_msg_en    = enableSignal;
    return std::make_unique<gaudi3::ArcExeWdRot>(rotFwCtx);
}

void gaudi3::RotatorQueue::forceStaticConfig()
{
    // Ensure we have static configuration at the top of the queue
    if (m_queueExe.empty() || m_queueExe.back()->isDynamic())
    {
        QueueCommandPtr nop = std::make_unique<gaudi3::Nop>();
        PushBack(std::move(nop), false);
    }
}

DescriptorShadow::AllRegistersProperties gaudi3::RotatorQueue::registersPropertiesForDesc(pNode n, const DescriptorWrapper<RotatorDesc>& descWrapper)
{
    std::vector<DescriptorShadow::RegisterProperties> mask(
            MASK_SIZE(RotatorDesc),
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, cpl_msg_addr), MASK_OFFSET(block_rot_desc, x_i_start_offset)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned)); // _pad88 included here

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad124), MASK_OFFSET(block_rot_desc, owm_cfg)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad268), MASK_OFFSET(block_rot_desc, rescale_cfg)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad284), MASK_OFFSET(block_rot_desc, grad_stride)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad292), MASK_OFFSET(block_rot_desc, grad_ctrl)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad300), MASK_OFFSET(block_rot_desc, rot_irm_rl)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad312), MASK_OFFSET(block_rot_desc, grad_irm_rl)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(block_rot_desc, _pad348), MASK_OFFSET(block_rot_desc, push_desc)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    // Using data/ignore masking from descriptor generator to ignore registers.
    if (descWrapper.getMask().has_value())
    {
        for (size_t i = 0; i < descWrapper.getMask().value().size(); i++)
        {
            if (!descWrapper.getMask().value()[i])
            {
                mask[i].ignore = true;
            }
        }
    }
    return std::make_shared<std::vector<DescriptorShadow::RegisterProperties>>(mask);
}

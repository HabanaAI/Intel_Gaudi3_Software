#include "platform/gaudi2/graph_compiler/command_queue.h"

#include "block_data.h"
#include "defs.h"
#include "gaudi2_arc_eng_packets.h"
#include "gaudi2_graph.h"
#include "gaudi2_types.h"
#include "habana_global_conf.h"
#include "habana_nodes.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "node_annotation.h"
#include "platform/gaudi2/graph_compiler/hal_conventions.h"
#include "platform/gaudi2/graph_compiler/queue_command_factory.h"
#include "queue_command.h"
#include "sync/sync_conventions.h"

//---------------------------------------------------------
//-------------------- CompletionQueue --------------------
//---------------------------------------------------------

gaudi2::CompletionQueue::CompletionQueue() :
    ::CommandQueue(QueueCommandFactory::instance(),
                   (unsigned)getQueueID(DEVICE_COMPLETION_QUEUE, 0),
                   DEVICE_COMPLETION_QUEUE)
{
    m_maxStreams = Gaudi2HalReader::instance()->getNumEngineStreams();
}


//---------------------------------------------------------
//---------------------- MmeQueue -------------------------
//---------------------------------------------------------

gaudi2::MmeQueue::MmeQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::MmeQueue<gaudi2::MmeDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_MME, engineId))
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = Gaudi2HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;

    createNullDescRegsList();
}

void gaudi2::MmeQueue::setDescriptorSignaling(gaudi2::MmeDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    if (sync != nullptr)  // in ARC mode 3 this sync is always null and we don't need to patch the sync objects
    {
        HB_ASSERT(desc.header.storeColorSet0 == desc.header.storeColorSet1,
                  "Assuming same colorSet for both outputs, different coloring is not supported yet.");
        Gaudi2::MmeDescriptorGenerator::mmePatchSyncObject(
            desc,
            gaudi2::getSyncObjectAddress(sync->id),  // master SO
            0,  // Using the same color for both tensors binds us to use a single sync object for both outputs, primary
                // and secondary. Currently, it is a must.
                // In the future, in the case of using a different color for each output, it is a must to have a
                // separate sync object as well.
            gaudi2::getSyncObjectAddress(sync->id + 1),  // slave SO
            0);                                          // no secondary slave
    }
}

uint32_t gaudi2::MmeQueue::getCommitRegVal()
{
    Gaudi2::Mme::MmeCmd cmd;
    cmd.dw                 = 0;
    cmd.aguIn              = 0b11111;
    cmd.aguOut             = 0b11;
    cmd.eu                 = 1;
    cmd.ap                 = 1;
    cmd.copyAndInc         = 1;
    cmd.descSel            = 0;
    cmd.maskIdleIndication = 0;
    cmd.nullDesc           = 0;
    return cmd.dw;
}

QueueCommandPtr gaudi2::MmeQueue::getExeCmd(pNode n, const DescriptorWrapper<MmeDesc>& descWrap, bool enableSignal)
{
    static const auto commitRegVal = getCommitRegVal();
    mme_wd_ctxt_t mmeFwCtx = descWrap.getFwCtx();
    mmeFwCtx.mme_commit_reg = commitRegVal;
    auto ret = std::make_unique<gaudi2::ArcExeWdMme>(mmeFwCtx);
    ret->SetContainerInfo(descWrap.getBasicFieldsContainerInfoForCtx());
    return ret;
}

void gaudi2::MmeQueue::forceStaticConfig()
{
    // Ensure we have static configuration at the top of the queue
    if (m_queueExe.empty() || m_queueExe.back()->isDynamic())
    {
        QueueCommandPtr nop = std::make_unique<gaudi2::Nop>();
        PushBack(std::move(nop), false);
    }
}

void gaudi2::MmeQueue::createNullDescRegsList()
{
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, cmd) / sizeof(uint32_t));
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, arch_sync_obj_dw0) / sizeof(uint32_t));
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, arch_sync_obj_addr0) / sizeof(uint32_t));
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, arch_sync_obj_val0) / sizeof(uint32_t));
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, arch_sync_obj_addr1) / sizeof(uint32_t));
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, arch_sync_obj_val1) / sizeof(uint32_t));
    m_nullDescRegs.push_back(offsetof(block_mme_ctrl_lo, arch_cout_ss) / sizeof(uint32_t));

    // Invalidating: arch_agu_cout0_master, arch_agu_cout0_slave, arch_agu_cout1_master, arch_agu_cout1_slave.
    // All of type block_mme_agu_core which holds an array of 5 registers. For convinience, just loop from
    // start to end index

    uint32_t startIndex = offsetof(block_mme_ctrl_lo, arch_b_ss) / sizeof(uint32_t) + 1;
    uint32_t endIndex = offsetof(block_mme_ctrl_lo, arch_cout_ss) / sizeof(uint32_t);

    for (uint32_t i = startIndex; i < endIndex; i++)
    {
        m_nullDescRegs.push_back(i);
    }
}

//---------------------------------------------------------
//---------------------- TpcQueue -------------------------
//---------------------------------------------------------

gaudi2::TpcQueue::TpcQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::TpcQueue<gaudi2::TpcDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_TPC, engineId))
{
    m_engineId         = engineId;
    m_engineIndex      = engineIndex;
    m_maxStreams       = Gaudi2HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents   = sendSyncEvents;

    createNullDescRegsList();
}

QueueCommandPtr gaudi2::TpcQueue::getExeCmd(pNode n, const DescriptorWrapper<TpcDesc>& descWrap, bool enableSignal)
{
    auto ret = std::make_unique<gaudi2::ArcExeWdTpc>(descWrap.getFwCtx());
    ret->SetContainerInfo(descWrap.getBasicFieldsContainerInfoForCtx());
    return ret;
}


DescriptorShadow::AllRegistersProperties
gaudi2::TpcQueue::registersPropertiesForDesc(pNode n, const DescriptorWrapper<TpcDesc>& descWrapper)
{
    std::vector<DescriptorShadow::RegisterProperties> mask(
                MASK_SIZE(TpcDesc),
                DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    // We must ensure the kernel address is programmed since the FW may programmed it with the address of NOP kernel
    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment
        {
            MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_base_address_low),
            MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, tid_base_dim_0)
        },
        DescriptorShadow::RegisterProperties::createFromHandling(
            DescriptorShadow::RegisterDataHandling::AlwaysWritePatching));

    // We must ensure the kernel id is programmed in order to not miss traces in profiler
    if (GCFG_ENABLE_PROFILER.value())
    {
        DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment
            {
                MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_id),
                MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, power_loop)
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

void gaudi2::TpcQueue::createNullDescRegsList()
{
    // Kernel base address
    m_nullDescRegs.push_back((offsetof(TpcDesc, m_desc) + offsetof(block_tpc_non_tensor_descriptor, kernel_base_address_low)) / sizeof(uint32_t));
    m_nullDescRegs.push_back((offsetof(TpcDesc, m_desc) + offsetof(block_tpc_non_tensor_descriptor, kernel_base_address_high)) / sizeof(uint32_t));

    // completion
    m_nullDescRegs.push_back((offsetof(TpcDesc, m_so) + offsetof(block_sync_object, message)) / sizeof(uint32_t));
    m_nullDescRegs.push_back((offsetof(TpcDesc, m_so) + offsetof(block_sync_object, addr)) / sizeof(uint32_t));
}

void gaudi2::TpcQueue::validateAllAddressPatchpointsDropped(const NodePtr& n) const
{
    // We expect all legacy patchpoints to get dropped except for the corner case that the node has 16 operands.
    // Because it's possible that the node has also a printf tensor, we relax the check to >= 15 operands.
    // Dynamic shape and rotator nodes have some exceptions so we don't check them.
    if (n == nullptr || n->isDynamicShape() || n->isRotate()) return;
    HB_ASSERT(m_allPatchpointsDropped || n->getOperands().size() >= 15, "found unexpected legacy patchpoints in TPC");
}

//---------------------------------------------------------
//---------------------- RotatorQueue -------------------------
//---------------------------------------------------------

gaudi2::RotatorQueue::RotatorQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
        ::RotatorQueue<gaudi2::RotatorDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_ROTATOR, engineId))
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = Gaudi2HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;

    createNullDescRegsList();
}

void gaudi2::RotatorQueue::setDescriptorSignaling(gaudi2::RotatorDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    desc.cpl_msg_data.val = 0;
    desc.cpl_msg_en.val = 1;

    desc.cpl_msg_data.val |= (0xC0000000 & (SyncObjOp::SYNC_OP_ADD << 29)); // increment
    desc.cpl_msg_data.val |= 1;                                             // by 1

    if (GCFG_DMA_SYNC_TRACE_EN_MASK.value() & GetEngineIndex())
    {
        setSendSyncEvents(desc.cpl_msg_data._raw);
    }

}

QueueCommandPtr
gaudi2::RotatorQueue::getExeCmd(NodePtr n, const DescriptorWrapper<RotatorDesc>& descWrap, bool enableSignal)
{
    rot_wd_ctxt_t rotFwCtx = descWrap.getFwCtx();
    rotFwCtx.cpl_msg_en    = enableSignal;
    return std::make_unique<gaudi2::ArcExeWdRot>(rotFwCtx);
}

void gaudi2::RotatorQueue::forceStaticConfig()
{
    // Ensure we have static configuration at the top of the queue
    if (m_queueExe.empty() || m_queueExe.back()->isDynamic())
    {
        QueueCommandPtr nop = std::make_unique<gaudi2::Nop>();
        PushBack(std::move(nop), false);
    }
}

void gaudi2::RotatorQueue::createNullDescRegsList()
{
    // For gaudi2 rotator, too many registers (roughly half) are being modified. Therefore, we mark all registers.
    // In gaudi3 - rotator IP has NULL descriptor support so we dont need to program any register
    for (uint i=0; i < MASK_SIZE(RotatorDesc); i++)
    {
        m_nullDescRegs.push_back(i);
    }
}

DescriptorShadow::AllRegistersProperties gaudi2::RotatorQueue::registersPropertiesForDesc(pNode n, const DescriptorWrapper<RotatorDesc>& descWrapper)
{
    std::vector<DescriptorShadow::RegisterProperties> mask(
            MASK_SIZE(RotatorDesc),
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {MASK_OFFSET(block_rot_desc, cpl_msg_addr), MASK_OFFSET(block_rot_desc, cpl_msg_awuser)},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {MASK_OFFSET(block_rot_desc, cpl_msg_awuser), MASK_OFFSET(block_rot_desc, x_i_start_offset)},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {MASK_OFFSET(block_rot_desc, hbw_aruser_hi), MASK_OFFSET(block_rot_desc, owm_cfg)},
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

//---------------------------------------------------------
//-------------------- DmaDescQueue -----------------------
//---------------------------------------------------------

gaudi2::DmaDescQueue::DmaDescQueue(unsigned logicalQueue, unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::DmaDescQueue<gaudi2::DmaDesc>(QueueCommandFactory::instance(),
                                    (unsigned)getQueueID(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL, engineId)),
    m_logicalQueue(logicalQueue),
    m_lastNodeOpType(DMA_OP_TYPE::DMA_OP_COPY)
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = Gaudi2HalReader::instance()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;

    createNullDescRegsList();
}

DescriptorShadow::AllRegistersProperties gaudi2::DmaDescQueue::registersPropertiesForDesc(NodePtr n, const DescriptorWrapper<DmaDesc>& descWrapper)
{
    auto dmaNode = std::dynamic_pointer_cast<DMANode>(n);
    HB_ASSERT_PTR(dmaNode);

    std::vector<DescriptorShadow::RegisterProperties> mask(
        MASK_SIZE(DmaDesc),
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(DmaDesc, axuser.hb_wr_reduction), MASK_OFFSET(DmaDesc, axuser.hb_rd_atomic)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {MASK_OFFSET(DmaDesc, ctx.te_numrows), MASK_OFFSET(DmaDesc, ctx.commit)},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

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

gaudi2::dma_core_ctx::reg_commit gaudi2::DmaDescQueue::getCommitRegVal(const DMANode& node, const DmaDesc& desc, bool enableSignal)
{
    dma_core_ctx::reg_commit commit {};

    bool isMemset = node.isMemset();
    bool isLinear = node.isLinearDma();

    // The descriptor generator splits tensor address to base and offset registers, so HW must do the sum
    commit.add_offset_0 = 1;

    // In case of transpose we have sometimes burst of descriptors that expose HW bug (see SW-138366)
    // The WA is to send out a message from the engine upon completion to trigger the back pressure.
    if (enableSignal || node.isTranspose())
    {
        commit.wr_comp_en = 1; // enables write completion message
    }
    if (isMemset)
    {
        commit.mem_set = 1;
    }
    if (isLinear)
    {
        commit.lin = 1;
    }
    return commit;
}

QueueCommandPtr
gaudi2::DmaDescQueue::getExeCmd(NodePtr n, const DescriptorWrapper<DmaDesc>& descWrap, bool enableSignal)
{
    HB_ASSERT(std::dynamic_pointer_cast<DMANode>(n), "invalid node");
    const DMANode& dmaNode = static_cast<DMANode&>(*n);

    dma_core_ctx::reg_commit commit = getCommitRegVal(dmaNode, descWrap.getDescriptor(), enableSignal);

    // DMA has no work distribution
    edma_wd_ctxt_t dmaFwCtx     = descWrap.getFwCtx();
    dmaFwCtx.dma_op             = EDMA_OP_NO_WD;
    dmaFwCtx.dma_commit_reg     = commit._raw;
    dmaFwCtx.use_alternate_addr = dmaNode.isTranspose() && !enableSignal ? 1 : 0; // SW-138366 WA for HW bug
    auto ret = std::make_unique<gaudi2::ArcExeWdDma>(dmaFwCtx);
    ret->SetContainerInfo(descWrap.getBasicFieldsContainerInfoForCtx());
    return ret;
}

// Set seignalling for ARC mode 3
uint32_t gaudi2::DmaDescQueue::calcDescriptorSignaling()
{
    // wr_comp_wdata is the data that is sent to the Sync Manager upon completion
    const uint32_t wrCompWdata = (0xC0000000 & (SyncObjOp::SYNC_OP_ADD << 29)) | 1;  // increment by 1
    return wrCompWdata;
}

void gaudi2::DmaDescQueue::setDescriptorSignaling(gaudi2::DmaDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    desc.ctx.wr_comp_wdata.val = calcDescriptorSignaling();

    if ((desc.ctx.wr_comp_wdata.val != 0) && (GCFG_DMA_SYNC_TRACE_EN_MASK.value() & GetEngineIndex()))
    {
        setSendSyncEvents(desc.ctx.wr_comp_wdata._raw);
    }
}

void gaudi2::DmaDescQueue::forceStaticConfig()
{
    // Ensure we have static configuration at the top of the queue
    if (m_queueExe.empty() || m_queueExe.back()->isDynamic())
    {
        QueueCommandPtr nop = std::make_unique<gaudi2::Nop>();
        PushBack(std::move(nop), false);
    }
}

void gaudi2::DmaDescQueue::createNullDescRegsList()
{
    // Completion
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, wr_comp_wdata)) / sizeof(uint32_t));
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, wr_comp_addr_hi)) / sizeof(uint32_t));
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, wr_comp_addr_lo)) / sizeof(uint32_t));

    // ctrl
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, ctrl)) / sizeof(uint32_t));

    // src
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, src_base_lo)) / sizeof(uint32_t));
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, src_base_hi)) / sizeof(uint32_t));

    // dst
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, dst_base_lo)) / sizeof(uint32_t));
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, dst_base_hi)) / sizeof(uint32_t));

    //size
    m_nullDescRegs.push_back((offsetof(DmaDesc, ctx) + offsetof(block_dma_core_ctx, dst_tsize_0)) / sizeof(uint32_t));
}

void gaudi2::DmaDescQueue::updateQueueStateAfterPush(NodePtr n)
{
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(n);
    HB_ASSERT_PTR(dmaNode);
    m_lastNodeOpType = dmaNode->getOpType();
}

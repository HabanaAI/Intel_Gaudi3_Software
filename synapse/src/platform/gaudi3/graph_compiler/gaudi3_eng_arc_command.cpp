#include <atomic>
#include "infra/defs.h"
#include "log_manager.h"
#include "gaudi3_eng_arc_command.h"
#include "hal_conventions.h"

// --------------------------------------------------------
// -------------- Gaudi3StaticCpDmaEngArcCommand ----------
// --------------------------------------------------------

Gaudi3StaticCpDmaEngArcCommand::Gaudi3StaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                                               EngArcBufferAddrBase srcAddrBaseId,
                                                               uint64_t             dataSize,
                                                               bool                 yield,
                                                               unsigned             engId)
{
    m_binary = {0};

    m_binary.cmd_type    = ECB_CMD_STATIC_DESC_V2;
    m_binary.yield       = yield ? 1 : 0;
    m_binary.size        = dataSize;
    m_binary.addr_index  = srcAddrBaseId;
    m_binary.addr_offset = srcOffset;
    m_binary.cpu_index   = engId;
}

void Gaudi3StaticCpDmaEngArcCommand::print() const
{
    LOG_DEBUG(
        GC_ARC,
        "        StaticCpDmaEngArcCommand cmd_type={}, yield={}, dataSize={}, srcAddrBaseId={}, srcOffset={}, engId={}",
        m_binary.cmd_type,
        m_binary.yield,
        m_binary.size,
        m_binary.addr_index,
        m_binary.addr_offset,
        m_binary.cpu_index);
}

unsigned Gaudi3StaticCpDmaEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_static_desc_v2_t);
}

uint64_t Gaudi3StaticCpDmaEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void Gaudi3StaticCpDmaEngArcCommand::setEngId(unsigned engId)
{
    m_binary.cpu_index = engId;
}

// --------------------------------------------------------
// -------------------Gaudi3NopEngArcCommand --------------
// --------------------------------------------------------

Gaudi3NopEngArcCommand::Gaudi3NopEngArcCommand(bool     switchCQ /* =false */,
                                               bool     yield /* =false */,
                                               unsigned padding /* =0 */)
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_NOP;
    m_binary.yield          = yield ? 1 : 0;
    m_binary.dma_completion = 0;
    m_binary.switch_cq      = switchCQ ? 1 : 0;
    m_binary.padding        = padding;
}

void Gaudi3NopEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        NopEngArcCommand cmd_type={}, yield={}, padding={}, switch_cq={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.padding,
              m_binary.switch_cq);
}

unsigned Gaudi3NopEngArcCommand::sizeInBytes() const
{
    return (sizeof(eng_arc_cmd_nop_t) + m_binary.padding * DWORD_SIZE);
}

uint64_t Gaudi3NopEngArcCommand::serialize(void* dst) const
{
    uint64_t paddingSize = m_binary.padding * DWORD_SIZE;
    memcpy(dst, &m_binary, sizeof(m_binary));
    memset((char*)dst + sizeof(m_binary), 0, paddingSize);
    return sizeof(m_binary) + paddingSize;
}

// --------------------------------------------------------
// ----------- Gaudi3DynamicWorkDistEngArcCommand ---------
// --------------------------------------------------------

Gaudi3DynamicWorkDistEngArcCommand::Gaudi3DynamicWorkDistEngArcCommand(unsigned wdCtxSlot,
                                                                       bool     yield,
                                                                       unsigned numDmaCompletion /* =1 */)
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_WD_FENCE_AND_EXE;
    m_binary.yield          = yield ? 1 : 0;
    m_binary.dma_completion = numDmaCompletion;
    m_binary.wd_ctxt_id     = wdCtxSlot;
}

void Gaudi3DynamicWorkDistEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        DynamicWorkDistEngArcCommand cmd_type={}, yield={}, numDmaCompletion={}, wdCtxSlot={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.dma_completion,
              m_binary.wd_ctxt_id);
}

unsigned Gaudi3DynamicWorkDistEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_wd_fence_and_exec_t);
}

uint64_t Gaudi3DynamicWorkDistEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// ------------- Gaudi3ScheduleDmaEngArcCommand -----------
// --------------------------------------------------------

Gaudi3ScheduleDmaEngArcCommand::Gaudi3ScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                                               EngArcBufferAddrBase srcAddrBaseId,
                                                               uint64_t             dstOffset,
                                                               uint64_t             dataSize,
                                                               bool                 isDmaWithDcoreLocality,
                                                               bool                 yield)
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_SCHED_DMA;
    m_binary.yield          = yield ? 1 : 0;
    m_binary.size           = dataSize;
    m_binary.addr_index     = srcAddrBaseId;
    m_binary.addr_offset    = srcOffset;
    m_binary.gc_ctxt_offset = dstOffset;
    m_binary.wd_type        = isDmaWithDcoreLocality ? TPC_WD_USING_DCORE_INDEX : TPC_WD_USING_GLBL_INDEX;
}

void Gaudi3ScheduleDmaEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        ScheduleDmaEngArcCommand cmd_type={}, yield={}, dataSize={}, srcAddrBaseId={}, srcOffset={}, "
              "dstOffset={}, wd_type={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.size,
              m_binary.addr_index,
              m_binary.addr_offset,
              m_binary.gc_ctxt_offset,
              m_binary.wd_type == 0 ? std::string("global_index") : std::string("dcore_index"));
}

unsigned Gaudi3ScheduleDmaEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_sched_dma_t);
}

uint64_t Gaudi3ScheduleDmaEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// ---------------- Gaudi3ListSizeEngArcCommand -----------
// --------------------------------------------------------

Gaudi3ListSizeEngArcCommand::Gaudi3ListSizeEngArcCommand()
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_LIST_SIZE;
    m_binary.yield          = 0;
    m_binary.topology_start = 0;
    m_binary.list_size      = 0;
}

void Gaudi3ListSizeEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        ListSizeEngArcCommand cmd_type={}, yield={}, topology_start={}, list_size={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.topology_start,
              m_binary.list_size);
}

unsigned Gaudi3ListSizeEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_list_size_t);
}

uint64_t Gaudi3ListSizeEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void Gaudi3ListSizeEngArcCommand::setListSize(unsigned listSize)
{
    m_binary.list_size = listSize;
}

void Gaudi3ListSizeEngArcCommand::setTopologyStart()
{
    m_binary.topology_start = 1;
}

// --------------------------------------------------------
// -------------- Gaudi3SignalOutEngArcCommand ------------
// --------------------------------------------------------

Gaudi3SignalOutEngArcCommand::Gaudi3SignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield)
{
    m_binary = {0};

    m_binary.cmd_type      = ECB_CMD_SFG;
    m_binary.switch_cq     = switchBit ? 1 : 0;
    m_binary.yield         = yield ? 1 : 0;
    m_binary.sob_inc_value = sigValue;
}

void Gaudi3SignalOutEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        SignalOutEngArcCommand cmd_type={}, switch_cq:{}, yield={}, sob_inc_value={}",
              m_binary.cmd_type,
              m_binary.switch_cq,
              m_binary.yield,
              m_binary.sob_inc_value);
}

unsigned Gaudi3SignalOutEngArcCommand::sizeInBytes() const
{
    return sizeof(m_binary);
}

uint64_t Gaudi3SignalOutEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// -------------- Gaudi3ResetSobsArcCommand ---------------
// --------------------------------------------------------

Gaudi3ResetSobsArcCommand::Gaudi3ResetSobsArcCommand(unsigned target,
                                                     unsigned targetXps,
                                                     unsigned totalNumEngs,
                                                     bool     switchBit,
                                                     bool     yield)
{
    m_binary = {0};

    m_binary.cmd_type         = ECB_CMD_RESET_SOSET;
    m_binary.switch_cq        = switchBit ? 1 : 0;
    m_binary.yield            = yield ? 1 : 0;
    m_binary.target           = target;
    m_binary.target_xpose     = targetXps;
    m_binary.num_cmpt_engines = totalNumEngs;
}

void Gaudi3ResetSobsArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        ResetSobsArcCommand cmd_type={}, switch_cq={}, yield={}, target={}, target_xpose={}, totalNumEngs={}",
              m_binary.cmd_type,
              m_binary.switch_cq,
              m_binary.yield,
              m_binary.target,
              m_binary.target_xpose,
              m_binary.num_cmpt_engines);
}

unsigned Gaudi3ResetSobsArcCommand::sizeInBytes() const
{
    return sizeof(m_binary);
}

uint64_t Gaudi3ResetSobsArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// -------------- Gaudi3McidRolloverArcCommand ------------
// --------------------------------------------------------

Gaudi3McidRolloverArcCommand::Gaudi3McidRolloverArcCommand(unsigned target,
                                                           unsigned targetXps,
                                                           bool     switchBit,
                                                           bool     yield)
{
    m_binary = {0};

    m_binary.cmd_type     = ECB_CMD_MCID_ROLLOVER;
    m_binary.switch_cq    = switchBit ? 1 : 0;
    m_binary.yield        = yield ? 1 : 0;
    m_binary.target       = target;
    m_binary.target_xpose = targetXps;
}

void Gaudi3McidRolloverArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        McidRolloverArcCommand cmd_type={}, switch_cq={}, yield={}, target={}, target_xpose={}",
              m_binary.cmd_type,
              m_binary.switch_cq,
              m_binary.yield,
              m_binary.target,
              m_binary.target_xpose);
}

unsigned Gaudi3McidRolloverArcCommand::sizeInBytes() const
{
    return sizeof(m_binary);
}

uint64_t Gaudi3McidRolloverArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// --------------- Gaudi3CmeNopEngArcCommand --------------
// --------------------------------------------------------

Gaudi3CmeNopEngArcCommand::Gaudi3CmeNopEngArcCommand(unsigned padding /* = 0 */)
{
    m_binary = {0};

    m_binary.cmd_type = CME_ECBL_CMD_NOP;
    m_binary.padding  = padding;
}

void Gaudi3CmeNopEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC, "        CmeNopEngArcCommand cmd_type={}, padding={}", m_binary.cmd_type, m_binary.padding);
}

unsigned Gaudi3CmeNopEngArcCommand::sizeInBytes() const
{
    return (sizeof(cme_arc_cmd_nop_t) + m_binary.padding * DWORD_SIZE);
}

uint64_t Gaudi3CmeNopEngArcCommand::serialize(void* dst) const
{
    uint64_t paddingSize = m_binary.padding * DWORD_SIZE;
    memcpy(dst, &m_binary, sizeof(m_binary));
    memset((char*)dst + sizeof(m_binary), 0, paddingSize);
    return sizeof(m_binary) + paddingSize;
}

// --------------------------------------------------------
// --------------- Gaudi3CmeDegradeArcCommand -------------
// --------------------------------------------------------

Gaudi3CmeDegradeArcCommand::Gaudi3CmeDegradeArcCommand(const DependencyMap& deps, PhysicalMcid mcid, bool useDiscardBase)
{
    m_binary = {0};

    m_binary.cmd_type         = CME_ECBL_CMD_DEGRADE_CLS;
    m_binary.mcid_offset      = mcid;
    m_binary.cls              = 0;
    m_binary.use_discard_base = (uint32_t)useDiscardBase;

    uint32_t target_bitmap = 0;

    if (deps.find(gaudi3::DEVICE_TPC_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.tpc.threshold_v2 = deps.at(gaudi3::DEVICE_TPC_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_TPC);
    }

    if (deps.find(gaudi3::DEVICE_MME_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.mme.threshold_v2 = deps.at(gaudi3::DEVICE_MME_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_MME);
    }

    if (deps.find(gaudi3::DEVICE_XPS_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.mme_xpose.threshold_v2 = deps.at(gaudi3::DEVICE_XPS_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_MME_XPOSE);
    }

    if (deps.find(gaudi3::DEVICE_ROT_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.rot.threshold_v2 = deps.at(gaudi3::DEVICE_ROT_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_ROT);
    }
    m_binary.target_bitmap = target_bitmap;
}

void Gaudi3CmeDegradeArcCommand::print() const
{
    std::string dependencyMap("TPC: " + std::to_string(m_binary.tpc.threshold_v2) +
                              ", MME: " + std::to_string(m_binary.mme.threshold_v2) +
                              ", XPOSE: " + std::to_string(m_binary.mme_xpose.threshold_v2) +
                              ", ROT: " + std::to_string(m_binary.rot.threshold_v2));

    LOG_DEBUG(GC_ARC, "        Gaudi3CmeDegradeArcCommand cmd_type={}, mcid={}, target_bitmap={}, DependencyMap={}, use_discard_base={}",
        m_binary.cmd_type, m_binary.mcid_offset, m_binary.target_bitmap, dependencyMap, m_binary.use_discard_base);
}

unsigned Gaudi3CmeDegradeArcCommand::sizeInBytes() const
{
    return (sizeof(cme_arc_cmd_degrade_cls_t));
}

uint64_t Gaudi3CmeDegradeArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// --------------- Gaudi3CmeDiscardArcCommand -------------
// --------------------------------------------------------

Gaudi3CmeDiscardArcCommand::Gaudi3CmeDiscardArcCommand(const DependencyMap& deps, PhysicalMcid mcid)
{
    m_binary = {0};

    m_binary.cmd_type    = CME_ECBL_CMD_DISCARD_CLS;
    m_binary.mcid_offset = mcid;
    m_binary.cls         = 0;

    uint32_t target_bitmap = 0;

    if (deps.find(gaudi3::DEVICE_TPC_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.tpc.threshold_v2 = deps.at(gaudi3::DEVICE_TPC_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_TPC);
    }

    if (deps.find(gaudi3::DEVICE_MME_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.mme.threshold_v2 = deps.at(gaudi3::DEVICE_MME_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_MME);
    }

    if (deps.find(gaudi3::DEVICE_XPS_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.mme_xpose.threshold_v2 = deps.at(gaudi3::DEVICE_XPS_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_MME_XPOSE);

    }

    if (deps.find(gaudi3::DEVICE_ROT_LOGICAL_QUEUE) != deps.end())
    {
        m_binary.rot.threshold_v2 = deps.at(gaudi3::DEVICE_ROT_LOGICAL_QUEUE);
        target_bitmap |= (1 << VIRTUAL_SOB_INDEX_ROT);
    }
    m_binary.target_bitmap = target_bitmap;
}

void Gaudi3CmeDiscardArcCommand::print() const
{
    std::string dependencyMap("TPC: " + std::to_string(m_binary.tpc.threshold_v2) +
                              ", MME: " + std::to_string(m_binary.mme.threshold_v2) +
                              ", XPOSE: " + std::to_string(m_binary.mme_xpose.threshold_v2) +
                              ", ROT: " + std::to_string(m_binary.rot.threshold_v2));

    LOG_DEBUG(GC_ARC, "        Gaudi3CmeDiscardArcCommand cmd_type={}, mcid={}, target_bitmap={}, DependencyMap={}",
        m_binary.cmd_type, m_binary.mcid_offset, m_binary.target_bitmap, dependencyMap);
}

unsigned Gaudi3CmeDiscardArcCommand::sizeInBytes() const
{
    return (sizeof(cme_arc_cmd_discard_cls_t));
}

uint64_t Gaudi3CmeDiscardArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// --------------- Gaudi3CmeMcidRolloverArcCommand -------------
// --------------------------------------------------------

Gaudi3CmeMcidRolloverArcCommand::Gaudi3CmeMcidRolloverArcCommand(bool incMme, bool incRot)
{
    m_binary = {0};

    m_binary.cmd_type   = CME_ECBL_CMD_MCID_ROLLOVER;
    m_binary.signal_mme = (uint32_t)incMme;
    m_binary.signal_rot = (uint32_t)incRot;
}

void Gaudi3CmeMcidRolloverArcCommand::print() const
{
    LOG_DEBUG(GC_ARC, "        Gaudi3CmeMcidRolloverArcCommand cmd_type={}, signal_mme={}, signal_rot={}",
        m_binary.cmd_type, m_binary.signal_mme, m_binary.signal_rot);
}

unsigned Gaudi3CmeMcidRolloverArcCommand::sizeInBytes() const
{
    return (sizeof(cme_arc_cmd_mcid_rollover_t));
}

uint64_t Gaudi3CmeMcidRolloverArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// --------------- Gaudi3CmeResetSobsArcCommand -------------
// --------------------------------------------------------

Gaudi3CmeResetSobsArcCommand::Gaudi3CmeResetSobsArcCommand(unsigned totalNumEngines)
{
    m_binary = {0};

    m_binary.cmd_type    = CME_ECBL_CMD_RESET_SOSET;
    m_binary.num_engines = totalNumEngines;
}

void Gaudi3CmeResetSobsArcCommand::print() const
{
    LOG_DEBUG(GC_ARC, "        Gaudi3CmeResetSobsArcCommand cmd_type={}, num_engines={}",
        m_binary.cmd_type, m_binary.num_engines);
}

unsigned Gaudi3CmeResetSobsArcCommand::sizeInBytes() const
{
    return (sizeof(cme_arc_cmd_reset_soset_t));
}

uint64_t Gaudi3CmeResetSobsArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

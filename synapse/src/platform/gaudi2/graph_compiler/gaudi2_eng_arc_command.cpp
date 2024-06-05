#include <atomic>
#include "infra/defs.h"
#include "log_manager.h"
#include "gaudi2_eng_arc_command.h"

// --------------------------------------------------------
// -------------- Gaudi2StaticCpDmaEngArcCommand ----------
// --------------------------------------------------------

Gaudi2StaticCpDmaEngArcCommand::Gaudi2StaticCpDmaEngArcCommand(uint64_t             srcOffset,
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

void Gaudi2StaticCpDmaEngArcCommand::print() const
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

unsigned Gaudi2StaticCpDmaEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_static_desc_v2_t);
}

uint64_t Gaudi2StaticCpDmaEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void Gaudi2StaticCpDmaEngArcCommand::setEngId(unsigned engId)
{
    m_binary.cpu_index = engId;
}

// --------------------------------------------------------
// -------------------Gaudi2NopEngArcCommand --------------
// --------------------------------------------------------

Gaudi2NopEngArcCommand::Gaudi2NopEngArcCommand(bool     switchCQ /* =false */,
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

void Gaudi2NopEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        NopEngArcCommand cmd_type={}, yield={}, padding={}, switch_cq={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.padding,
              m_binary.switch_cq);
}

unsigned Gaudi2NopEngArcCommand::sizeInBytes() const
{
    return (sizeof(eng_arc_cmd_nop_t) + m_binary.padding * DWORD_SIZE);
}

uint64_t Gaudi2NopEngArcCommand::serialize(void* dst) const
{
    uint64_t paddingSize = m_binary.padding * DWORD_SIZE;
    memcpy(dst, &m_binary, sizeof(m_binary));
    memset((char*)dst + sizeof(m_binary), 0, paddingSize);
    return sizeof(m_binary) + paddingSize;
}

// --------------------------------------------------------
// ----------- Gaudi2DynamicWorkDistEngArcCommand ---------
// --------------------------------------------------------

Gaudi2DynamicWorkDistEngArcCommand::Gaudi2DynamicWorkDistEngArcCommand(unsigned wdCtxSlot,
                                                                       bool     yield,
                                                                       unsigned numDmaCompletion /* =1 */)
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_WD_FENCE_AND_EXE;
    m_binary.yield          = yield ? 1 : 0;
    m_binary.dma_completion = numDmaCompletion;
    m_binary.wd_ctxt_id     = wdCtxSlot;
}

void Gaudi2DynamicWorkDistEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        DynamicWorkDistEngArcCommand cmd_type={}, yield={}, numDmaCompletion={}, wdCtxSlot={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.dma_completion,
              m_binary.wd_ctxt_id);
}

unsigned Gaudi2DynamicWorkDistEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_wd_fence_and_exec_t);
}

uint64_t Gaudi2DynamicWorkDistEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// ------------- Gaudi2ScheduleDmaEngArcCommand -----------
// --------------------------------------------------------

Gaudi2ScheduleDmaEngArcCommand::Gaudi2ScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                                               EngArcBufferAddrBase srcAddrBaseId,
                                                               uint64_t             dstOffset,
                                                               uint64_t             dataSize,
                                                               bool                 yield)
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_SCHED_DMA;
    m_binary.yield          = yield ? 1 : 0;
    m_binary.size           = dataSize;
    m_binary.addr_index     = srcAddrBaseId;
    m_binary.addr_offset    = srcOffset;
    m_binary.gc_ctxt_offset = dstOffset;
}

void Gaudi2ScheduleDmaEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        ScheduleDmaEngArcCommand cmd_type={}, yield={}, dataSize={}, srcAddrBaseId={}, srcOffset={}, "
              "dstOffset={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.size,
              m_binary.addr_index,
              m_binary.addr_offset,
              m_binary.gc_ctxt_offset);
}

unsigned Gaudi2ScheduleDmaEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_sched_dma_t);
}

uint64_t Gaudi2ScheduleDmaEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// ---------------- Gaudi2ListSizeEngArcCommand -----------
// --------------------------------------------------------

Gaudi2ListSizeEngArcCommand::Gaudi2ListSizeEngArcCommand()
{
    m_binary = {0};

    m_binary.cmd_type       = ECB_CMD_LIST_SIZE;
    m_binary.yield          = 0;
    m_binary.topology_start = 0;
    m_binary.list_size      = 0;
}

void Gaudi2ListSizeEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        ListSizeEngArcCommand cmd_type={}, yield={}, topology_start={}, list_size={}",
              m_binary.cmd_type,
              m_binary.yield,
              m_binary.topology_start,
              m_binary.list_size);
}

unsigned Gaudi2ListSizeEngArcCommand::sizeInBytes() const
{
    return sizeof(eng_arc_cmd_list_size_t);
}

uint64_t Gaudi2ListSizeEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void Gaudi2ListSizeEngArcCommand::setListSize(unsigned listSize)
{
    m_binary.list_size = listSize;
}

void Gaudi2ListSizeEngArcCommand::setTopologyStart()
{
    m_binary.topology_start = 1;
}

// --------------------------------------------------------
// -------------- Gaudi2SignalOutEngArcCommand ------------
// --------------------------------------------------------

Gaudi2SignalOutEngArcCommand::Gaudi2SignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield)
{
    m_binary = {0};

    m_binary.cmd_type      = ECB_CMD_SFG;
    m_binary.switch_cq     = switchBit ? 1 : 0;
    m_binary.yield         = yield ? 1 : 0;
    m_binary.sob_inc_value = sigValue;
}

void Gaudi2SignalOutEngArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        SignalOutEngArcCommand cmd_type={}, switch_cq={}, yield={}, sob_inc_value={}",
              m_binary.cmd_type,
              m_binary.switch_cq,
              m_binary.yield,
              m_binary.sob_inc_value);
}

unsigned Gaudi2SignalOutEngArcCommand::sizeInBytes() const
{
    return sizeof(m_binary);
}

uint64_t Gaudi2SignalOutEngArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// -------------- Gaudi2ResetSobsArcCommand ---------------
// --------------------------------------------------------

Gaudi2ResetSobsArcCommand::Gaudi2ResetSobsArcCommand(unsigned target, unsigned totalNumEngs, bool switchBit, bool yield)
{
    m_binary = {0};

    m_binary.cmd_type         = ECB_CMD_RESET_SOSET;
    m_binary.switch_cq        = switchBit ? 1 : 0;
    m_binary.yield            = yield ? 1 : 0;
    m_binary.target           = target;
    m_binary.num_cmpt_engines = totalNumEngs;
}

void Gaudi2ResetSobsArcCommand::print() const
{
    LOG_DEBUG(GC_ARC,
              "        ResetSobsArcCommand cmd_type={}, switch_cq={}, yield={}, target={}, totalNumEngs={}",
              m_binary.cmd_type,
              m_binary.switch_cq,
              m_binary.yield,
              m_binary.target,
              m_binary.num_cmpt_engines);
}

unsigned Gaudi2ResetSobsArcCommand::sizeInBytes() const
{
    return sizeof(m_binary);
}

uint64_t Gaudi2ResetSobsArcCommand::serialize(void* dst) const
{
    memcpy(dst, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

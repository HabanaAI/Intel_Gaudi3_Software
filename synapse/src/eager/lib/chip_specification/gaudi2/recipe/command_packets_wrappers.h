#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/sync/sync_types.h"

// synapse api (relative to include/)
#include "internal/define_synapse_common.hpp"

// relative to <specs>/
#include "gaudi2/gaudi2_packets.h"

// relative to <qman_fw>/engines-arc/include/
#include "gaudi2_arc_eng_packets.h"

// This file contains reusable structure for template creation of various engines.
// Note that none of the classes or structures in this file should define virtual methods or destructors.
// Keep this in mind when using these structures, and avoid dynamically allocating any of them.
// Initializations are not purely generic, they match the specific usage of recipe templates creation.
// In the begging init(...) methods were added for template creation. In case further initializations are
// needed, you may add further init(...) overrides.

namespace eager_mode::gaudi2_spec_info
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// QMAN Packets
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace qman_packets_wrappers
{

struct wreg32 final : public packet_wreg32
{
    void init(AsicRegType regOffset, bool switchBit = false, AsicRegValType val = 0, bool engineBarrier = false)
    {
        opcode      = packet_id::PACKET_WREG_32;
        reg_offset  = regOffset;
        value       = val;
        msg_barrier = 1;
        eng_barrier = engineBarrier ? 1 : 0;
        swtc        = switchBit ? 1 : 0;
    }
};

struct wreg_bulk final : public packet_wreg_bulk
{
    void init(AsicRegType regOffset, TensorsNrType tensorsNr)
    {
        opcode      = packet_id::PACKET_WREG_BULK;
        reg_offset  = regOffset;
        size64      = tensorsNr;
        msg_barrier = 1;
    }
};

struct fence final : public packet_fence
{
    void init()
    {
        opcode     = packet_id::PACKET_FENCE;
        dec_val    = 1;
        target_val = 1;
        id         = ID_2;
    }
};

struct wreg64_long final : public packet_wreg64_long
{
    void init(AsicRegType dregOffset)
    {
        opcode      = packet_id::PACKET_WREG_64_LONG;
        dw_enable   = 0x3;
        rel         = 0x1;
        dreg_offset = dregOffset;
        msg_barrier = 1;
    }
};
}  // namespace qman_packets_wrappers

///////////////////////////////////////////////////////////////////////////////////////////////////
// ECB Packets
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ecb_packets_wrappers
{

struct list_size final : public eng_arc_cmd_list_size_t
{
    void init(EcbCommandSizeType listSize)
    {
        cmd_type                           = eng_arc_cmd_t::ECB_CMD_LIST_SIZE;
        topology_start                     = 1;
        eng_arc_cmd_list_size_t::list_size = listSize;
    }
};

struct nop final : public eng_arc_cmd_nop_t
{
    using PaddingSizeType = decltype(eng_arc_cmd_nop_t::padding);

    void init(PaddingSizeType paddingSize = 0, bool switchBit = false)
    {
        cmd_type  = eng_arc_cmd_t::ECB_CMD_NOP;
        switch_cq = switchBit ? 1 : 0;
        padding   = paddingSize;
    }

    static inline unsigned getCmdSize() { return sizeof(eng_arc_cmd_nop_t); }
};

struct wd_fence_and_exec final : public eng_arc_cmd_wd_fence_and_exec_t
{
    void init()
    {
        cmd_type       = eng_arc_cmd_t::ECB_CMD_WD_FENCE_AND_EXE;
        yield          = 1;
        dma_completion = 1;
        wd_ctxt_id     = 0;
    }
};

struct sched_dma final : public eng_arc_cmd_sched_dma_t
{
    using AddrOffsetType = decltype(eng_arc_cmd_sched_dma_t::addr_offset);

    void init(BlobSizeType bufSize, AddrOffsetType addrOffset)
    {
        cmd_type    = eng_arc_cmd_t::ECB_CMD_SCHED_DMA;
        yield       = 1;
        addr_index  = EngArcBufferAddrBase::DYNAMIC_ADDR_BASE;
        size        = bufSize;
        addr_offset = addrOffset;
    }
};

struct static_desc_v2 final : public eng_arc_cmd_static_desc_v2_t
{
    using AddrOffsetType = decltype(eng_arc_cmd_static_desc_v2_t::addr_offset);

    void init(EngineIdType         cpuIndex,
              bool                 yieldEn,
              BlobSizeType         blobSize,
              EngArcBufferAddrBase addrIndex,
              AddrOffsetType       addrOffset)
    {
        cmd_type    = eng_arc_cmd_t::ECB_CMD_STATIC_DESC_V2;
        cpu_index   = cpuIndex;
        yield       = yieldEn ? 1 : 0;
        size        = blobSize;
        addr_index  = addrIndex;
        addr_offset = addrOffset;
    }
};

}  // namespace ecb_packets_wrappers

}  // namespace eager_mode::gaudi2_spec_info
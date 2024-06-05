#pragma once

#include "address_fields_container_info.h"
#include "defs.h"
#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2/asic_reg_structs/axuser_regs.h"
#include "gaudi2/asic_reg_structs/dma_core_regs.h"
#include "gaudi2/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"
#include "gaudi2/asic_reg_structs/rotator_regs.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi2/asic_reg_structs/tpc_regs.h"
#include "gaudi2/gaudi2_packets.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "platform/gaudi2/graph_compiler/gaudi2_types.h"
#include "utils.h"

namespace gaudi2
{
enum EMonPayloadSelect
{
    MON_PAYLOAD_ADDR_L,
    MON_PAYLOAD_ADDR_H,
    MON_PAYLOAD_DATA
};

#define BLOCK_BASE_MASK   0xF000
#define BLOCK_OFFSET_MASK 0x0FFF

#define TPC_BLOCK_BASE    0xB000
#define MME_BLOCK_BASE    0xB000
#define DMA_BLOCK_BASE    0xB000
#define ROT_BLOCK_BASE    0xB000
#define QMAN_BLOCK_BASE   0xA000

#define GET_ADDR_OF_QMAN_BLOCK_FIELD(field)  (offsetof(block_qman, field) + QMAN_BLOCK_BASE)
#define GET_ADDR_OF_TPC_BLOCK_FIELD(field)   (offsetof(block_tpc, field) + TPC_BLOCK_BASE)
#define GET_ADDR_OF_MME_BLOCK_FIELD(field)   (offsetof(block_mme_ctrl_lo, field) + MME_BLOCK_BASE)
#define GET_ADDR_OF_DMA_BLOCK_FIELD(field)   (offsetof(block_dma_core, ctx_axuser) + offsetof(block_axuser_dma_core_ctx, field) + DMA_BLOCK_BASE)
#define GET_ADDR_OF_ROT_BLOCK_FIELD(field)   (offsetof(block_rotator, desc) + offsetof(block_rot_desc, field) + ROT_BLOCK_BASE)

inline unsigned getHwBlockBase(HabanaDeviceType type)
{
    switch (type)
    {
        case DEVICE_MME:
            return MME_BLOCK_BASE;

        case DEVICE_TPC:
            return TPC_BLOCK_BASE;

        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            return DMA_BLOCK_BASE;

        case DEVICE_ROTATOR:
            return ROT_BLOCK_BASE;

        default:
            HB_ASSERT(0, "Unsupported device type");
            break;
    }
    return 0;
}

inline unsigned getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID)
{
    switch (type)
    {
        case DEVICE_MME:
            return GET_ADDR_OF_MME_BLOCK_FIELD(arch_base_addr);

        case DEVICE_TPC:
            return GET_ADDR_OF_TPC_BLOCK_FIELD(qm_tensor_0);

        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            return GET_ADDR_OF_DMA_BLOCK_FIELD(axuser.hb_asid);

        case DEVICE_ROTATOR:
            return GET_ADDR_OF_ROT_BLOCK_FIELD(context_id);

        default:
            HB_ASSERT(0, "Unsupported device type");
            break;
    }
    return 0;
}

inline unsigned getRegForBaseAddress(unsigned regIndex)
{
    return GET_ADDR_OF_QMAN_BLOCK_FIELD(qman_wr64_base_addr0) + (regIndex * sizeof(struct block_qman_wr64_base_addr));
}

inline uint64_t getSyncObjectAddress(unsigned so)
{
    auto bound = Gaudi2HalReader::instance()->getFirstSyncObjId() + Gaudi2HalReader::instance()->getNumSyncObjects();
    HB_ASSERT(so < bound, "sync-obj out of upper bound");
    HB_ASSERT(so >= Gaudi2HalReader::instance()->getFirstSyncObjId(), "sync-obj below lower bound");
    return mmDCORE0_SYNC_MNGR_OBJS_BASE + varoffsetof(block_sob_objs, sob_obj[so]);
}

inline uint64_t getMonPayloadAddress(unsigned mon, EMonPayloadSelect payloadSelect)
{
    static const uint64_t baseAddr = mmDCORE0_SYNC_MNGR_OBJS_BASE;

    auto bound = Gaudi2HalReader::instance()->getFirstMonObjId() + Gaudi2HalReader::instance()->getNumMonitors();
    HB_ASSERT(mon < bound, "monitor out of upper bound");
    // monitor 0 is used just to get the base address, so it's allowed
    HB_ASSERT(mon == 0 || mon >= Gaudi2HalReader::instance()->getFirstMonObjId(), "monitor below lower bound");

    switch (payloadSelect)
    {
        case MON_PAYLOAD_ADDR_L:
            return baseAddr + varoffsetof(block_sob_objs, mon_pay_addrl[mon]);

        case MON_PAYLOAD_ADDR_H:
            return baseAddr + varoffsetof(block_sob_objs, mon_pay_addrh[mon]);

        case MON_PAYLOAD_DATA:
            return baseAddr + varoffsetof(block_sob_objs, mon_pay_data[mon]);

        default:
            HB_ASSERT(0, "Unsupported monitor payload select");
    }
    return 0;
}

inline unsigned getRegForExecute(HabanaDeviceType type, unsigned deviceID)
{
    switch (type)
    {
        case DEVICE_MME:
            return GET_ADDR_OF_MME_BLOCK_FIELD(cmd);

        case DEVICE_TPC:
            return GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_execute);

        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            return GET_ADDR_OF_DMA_BLOCK_FIELD(ctx.commit);

        case DEVICE_ROTATOR:
            return GET_ADDR_OF_ROT_BLOCK_FIELD(push_desc);

        default:
            HB_ASSERT(0, "Unsupported device type");
            break;
    }
    return 0;
}

inline unsigned getRegForEbPadding()
{
    // using TSB_CFG_MTRR_2 for EB bug Padding
    return offsetof(block_tpc, tsb_cfg_mtrr_2) + TPC_BLOCK_BASE;
}

}  // namespace gaudi2

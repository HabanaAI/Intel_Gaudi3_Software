#pragma once

#include "defs.h"
#include "gaudi3/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi3/asic_reg_structs/qman_regs.h"
#include "gaudi3/asic_reg_structs/rotator_regs.h"
#include "gaudi3/asic_reg_structs/tpc_regs.h"
#include "node.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "types.h"
#include "utils.h"

#include <memory>
#include <string>

namespace gaudi3
{
#define BLOCK_BASE_MASK   0xF000
#define BLOCK_OFFSET_MASK 0x0FFF

#define TPC_BLOCK_BASE    0xA000
#define MME_BLOCK_BASE    0x0000
#define ROT_BLOCK_BASE    0x0000
#define QMAN_BLOCK_BASE   0x9000

#define GET_ADDR_OF_QMAN_BLOCK_FIELD(field)  (offsetof(block_qman, field) + QMAN_BLOCK_BASE)
#define GET_ADDR_OF_TPC_BLOCK_FIELD(field)   (offsetof(block_tpc, field) + TPC_BLOCK_BASE)
#define GET_ADDR_OF_MME_BLOCK_FIELD(field)   (offsetof(block_mme_ctrl_lo, field) + MME_BLOCK_BASE)
#define GET_ADDR_OF_ROT_BLOCK_FIELD(field)   (offsetof(block_rotator, desc) + offsetof(block_rot_desc, field) + ROT_BLOCK_BASE)

inline unsigned getRegForLoadDesc(HabanaDeviceType type, bool isTranspose = false)
{
    switch (type)
    {
        case DEVICE_MME:
            return isTranspose ? GET_ADDR_OF_MME_BLOCK_FIELD(arch_dma_base_addr)
                               : GET_ADDR_OF_MME_BLOCK_FIELD(arch_base_addr);
        case DEVICE_TPC:
            return GET_ADDR_OF_TPC_BLOCK_FIELD(qm_tensor_0);

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

inline unsigned getRegForExecute(HabanaDeviceType type, unsigned deviceID)
{
    switch (type)
    {
        case DEVICE_MME:
            return GET_ADDR_OF_MME_BLOCK_FIELD(cmd);

        case DEVICE_TPC:
            return GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_execute);

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
    // using tsb_cfg_mtrr_2 for EB bug Padding
    return GET_ADDR_OF_TPC_BLOCK_FIELD(tsb_cfg_mtrr_2);
}

}  // namespace gaudi3
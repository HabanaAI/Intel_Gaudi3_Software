#pragma once

#include "runtime/qman/common/parser/define.hpp"

#include "gaudi/asic_reg_structs/qman_regs.h"
#include "gaudi/asic_reg_structs/sob_objs_regs.h"

namespace gaudi
{
// SYNC-Managers (start)
//
enum eSyncManagerInstance
{
    SYNC_MNGR_EAST_NORTH,
    SYNC_MNGR_GC = SYNC_MNGR_EAST_NORTH,
    SYNC_MNGR_EAST_SOUTH,
    SYNC_MNGR_WEST_NORTH,
    SYNC_MNGR_WEST_SOUTH,
    SYNC_MNGR_LAST,
    SYNC_MNGR_NUM = SYNC_MNGR_LAST
};

// SOBJ (start)
//
static const uint16_t SOBJS_AMOUNT            = 2048;
static const uint64_t SYNC_OBJECT_SIZE        = sizeof(sob_objs::reg_sob_obj);
static const uint64_t SYNC_OBJECTS_TOTAL_SIZE = SOBJS_AMOUNT * SYNC_OBJECT_SIZE;
//
// SOBJ (end)

// Monitors (start)
//
static const uint16_t MONITORS_AMOUNT        = 512;
static const uint64_t MONITOR_ARM_SIZE       = sizeof(sob_objs::reg_mon_arm);
static const uint64_t MONITOR_PAYL_LOW_SIZE  = sizeof(sob_objs::reg_mon_pay_addrl);
static const uint64_t MONITOR_PAYL_HIGH_SIZE = sizeof(sob_objs::reg_mon_pay_addrh);
static const uint64_t MONITOR_PAYL_DATA_SIZE = sizeof(sob_objs::reg_mon_pay_data);
static const uint64_t MONITORS_TOTAL_SIZE =
    MONITORS_AMOUNT * (MONITOR_ARM_SIZE + MONITOR_PAYL_LOW_SIZE + MONITOR_PAYL_HIGH_SIZE + MONITOR_PAYL_DATA_SIZE);

static const unsigned MONITOR_CONFIG_BLOCK_BASE  = offsetof(block_sob_objs, mon_pay_addrl);
static const unsigned MONITOR_PAYLOAD_BLOCK_BASE = offsetof(block_sob_objs, mon_pay_addrl);
static const unsigned MONITOR_PAYLOAD_LOW_ADDRESS_BLOCK_BASE =
    offsetof(block_sob_objs, mon_pay_addrl) - MONITOR_CONFIG_BLOCK_BASE;
static const unsigned MONITOR_PAYLOAD_HIGH_ADDRESS_BLOCK_BASE =
    offsetof(block_sob_objs, mon_pay_addrh) - MONITOR_CONFIG_BLOCK_BASE;
static const unsigned MONITOR_PAYLOAD_DATA_BLOCK_BASE =
    offsetof(block_sob_objs, mon_pay_data) - MONITOR_CONFIG_BLOCK_BASE;
static const unsigned MONITOR_ARM_BLOCK_BASE = offsetof(block_sob_objs, mon_arm) - MONITOR_CONFIG_BLOCK_BASE;
//
// Monitors (end)

//      East-North SYNC-Manager
static const uint64_t GC_SOBJ_BASE_ADDRESS    = 0x7FFC4F2000;
static const uint64_t GC_SOBJ_LAST_ADDRESS    = GC_SOBJ_BASE_ADDRESS + SYNC_OBJECTS_TOTAL_SIZE;
static const uint64_t GC_MONITOR_BASE_ADDRESS = 0x7ffc4f4000;
static const uint64_t GC_MONITOR_LAST_ADDRESS = GC_MONITOR_BASE_ADDRESS + MONITORS_TOTAL_SIZE;
//
//      East-South SYNC-Manager
static const uint64_t EAST_SOUTH_SOBJ_BASE_ADDRESS    = 0x7ffc4b2000;
static const uint64_t EAST_SOUTH_SOBJ_LAST_ADDRESS    = EAST_SOUTH_SOBJ_BASE_ADDRESS + SYNC_OBJECTS_TOTAL_SIZE;
static const uint64_t EAST_SOUTH_MONITOR_BASE_ADDRESS = 0x7ffc4b4000;
static const uint64_t EAST_SOUTH_MONITOR_LAST_ADDRESS = EAST_SOUTH_MONITOR_BASE_ADDRESS + MONITORS_TOTAL_SIZE;

//      West-North SYNC-Manager
static const uint64_t WEST_NORTH_SOBJ_BASE_ADDRESS    = 0x7ffc4d2000;
static const uint64_t WEST_NORTH_SOBJ_LAST_ADDRESS    = WEST_NORTH_SOBJ_BASE_ADDRESS + SYNC_OBJECTS_TOTAL_SIZE;
static const uint64_t WEST_NORTH_MONITOR_BASE_ADDRESS = 0x7ffc4d4000;
static const uint64_t WEST_NORTH_MONITOR_LAST_ADDRESS = WEST_NORTH_MONITOR_BASE_ADDRESS + MONITORS_TOTAL_SIZE;

//      West-South SYNC-Manager
static const uint64_t WEST_SOUTH_SOBJ_BASE_ADDRESS    = 0x7ffc492000;
static const uint64_t WEST_SOUTH_SOBJ_LAST_ADDRESS    = WEST_SOUTH_SOBJ_BASE_ADDRESS + SYNC_OBJECTS_TOTAL_SIZE;
static const uint64_t WEST_SOUTH_MONITOR_BASE_ADDRESS = 0x7ffc494000;
static const uint64_t WEST_SOUTH_MONITOR_LAST_ADDRESS = WEST_SOUTH_MONITOR_BASE_ADDRESS + MONITORS_TOTAL_SIZE;
//
// SYNC-Managers (end)

static const uint64_t QMAN_BLOCK_SIZE = 0xd00;  // size of <QMAN-Type>X_QM [Not including the last 4B of last field]
//
//      MME QMAN
static const uint64_t MME_QMAN_SIZE           = 0x80000;  // size of the all QMAN regs(Code, QM, etc.)
static const uint64_t MME_QMANS_AMOUNT        = 4;
static const uint64_t MME_MASTER_QMANS_AMOUNT = 2;
static const uint64_t MME_QMAN_BASE_ADDRESS   = 0x7ffc068000;  // MMEX_QM base address (not the first QMAN address)
static const uint64_t MME_QMAN_END_ADDRESS =
    MME_QMAN_BASE_ADDRESS + MME_QMAN_SIZE * (MME_QMANS_AMOUNT - 1) + QMAN_BLOCK_SIZE;
//
//      DMA QMAN
static const uint64_t DMA_QMAN_SIZE         = 0x20000;
static const uint64_t DMA_QMAN_BLOCK_SIZE   = 0xd00;  // sizeof(MMEX_QM) [Not including the last 4B of last field]
static const uint64_t DMA_QMANS_AMOUNT      = 8;
static const uint64_t DMA_QMAN_BASE_ADDRESS = 0x7ffc508000;
static const uint64_t DMA_QMAN_END_ADDRESS =
    DMA_QMAN_BASE_ADDRESS + DMA_QMAN_SIZE * (DMA_QMANS_AMOUNT - 1) + QMAN_BLOCK_SIZE;
//
//      TPC QMAN
static const uint64_t TPC_QMAN_SIZE         = 0x40000;
static const uint64_t TPC_QMANS_AMOUNT      = 8;
static const uint64_t TPC_QMAN_BASE_ADDRESS = 0x7ffce08000;
static const uint64_t TPC_QMAN_END_ADDRESS =
    TPC_QMAN_BASE_ADDRESS + TPC_QMAN_SIZE * (TPC_QMANS_AMOUNT - 1) + QMAN_BLOCK_SIZE;

//      QMANs amount
static const uint64_t TOTAL_QMANS_AMOUNT = DMA_QMANS_AMOUNT + MME_MASTER_QMANS_AMOUNT + TPC_QMANS_AMOUNT;
static const uint64_t UPPER_CPS_PER_QMAN = 4;
static const uint64_t LOWER_CPS_PER_QMAN = 1;
static const uint64_t CPS_PER_QMAN       = UPPER_CPS_PER_QMAN + LOWER_CPS_PER_QMAN;
static const uint64_t FENCE_IDS_PER_CP   = 4;
// QMANs blocks (end)

// FENCE Block (start)
static const uint64_t FENCE_BLOCK_OFFSET     = offsetof(block_qman, cp_fence0_rdata);
static const uint64_t SINGLE_CP_RDATA_SIZE   = sizeof(qman::reg_cp_fence0_rdata);
static const uint64_t FENCE_INSTANCES_AMOUNT = CPS_PER_QMAN * FENCE_IDS_PER_CP;
static const uint64_t FENCE_BLOCK_TOTAL_SIZE = SINGLE_CP_RDATA_SIZE * FENCE_INSTANCES_AMOUNT;
// FENCE Block (end)
}  // namespace gaudi
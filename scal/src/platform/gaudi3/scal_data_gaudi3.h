#pragma once

#include "common/scal_macros.h"
#include "common/scal_data.h"
#include "platform/gaudi3/scal_gaudi3.h"
#include "gaudi3/asic_reg_structs/arc_dup_eng_regs.h"

static constexpr unsigned c_scal_gaudi3_nic_port_0 = 0b01;
static constexpr unsigned c_scal_gaudi3_nic_port_1 = 0b10;
static constexpr unsigned c_scal_gaudi3_nic_port_0_and_1 = 0b11;
static constexpr unsigned c_scal_gaudi3_num_of_ports_per_nic = 2;


struct EngineInfoG3
{
    const char     * name;
    gaudi3_engine_id queueId;
    uint64_t dccmAddr;
    CoreType coreType;
    unsigned cpuId;
    unsigned hdCore = -1;
};


static const std::array<EngineInfoG3, CPU_ID_MAX> c_engine_info_arr{{
    // schedulers
    {"ARCFARM_0_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD0_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC0,  0},
    {"ARCFARM_0_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD0_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC1,  0},
    {"ARCFARM_1_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD1_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC2,  1},
    {"ARCFARM_1_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD1_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC3,  1},
    {"ARCFARM_2_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD2_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC4,  2},
    {"ARCFARM_2_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD2_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC5,  2},
    {"ARCFARM_3_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD3_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC6,  3},
    {"ARCFARM_3_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD3_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC7,  3},
    {"ARCFARM_4_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD4_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC8,  4},
    {"ARCFARM_4_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD4_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC9,  4},
    {"ARCFARM_5_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD5_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC10, 5},
    {"ARCFARM_5_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD5_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC11, 5},
    {"ARCFARM_6_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD6_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC12, 6},
    {"ARCFARM_6_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD6_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC13, 6},
    {"ARCFARM_7_0",   GAUDI3_ENGINE_ID_SIZE,             mmHD7_ARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC14, 7},
    {"ARCFARM_7_1",   GAUDI3_ENGINE_ID_SIZE,             mmHD7_ARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER,    CPU_ID_SCHED_ARC15, 7},

    {"TPC_0_0",       GAUDI3_HDCORE0_ENGINE_ID_TPC_0,    mmHD0_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC0,  0},
    {"TPC_0_1",       GAUDI3_HDCORE0_ENGINE_ID_TPC_1,    mmHD0_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC1,  0},
    {"TPC_0_2",       GAUDI3_HDCORE0_ENGINE_ID_TPC_2,    mmHD0_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC2,  0},
    {"TPC_0_3",       GAUDI3_HDCORE0_ENGINE_ID_TPC_3,    mmHD0_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC3,  0},
    {"TPC_0_4",       GAUDI3_HDCORE0_ENGINE_ID_TPC_4,    mmHD0_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC4,  0},
    {"TPC_0_5",       GAUDI3_HDCORE0_ENGINE_ID_TPC_5,    mmHD0_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC5,  0},
    {"TPC_0_6",       GAUDI3_HDCORE0_ENGINE_ID_TPC_6,    mmHD0_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC6,  0},
    {"TPC_0_7",       GAUDI3_HDCORE0_ENGINE_ID_TPC_7,    mmHD0_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC7,  0},
    {"TPC_1_0",       GAUDI3_HDCORE1_ENGINE_ID_TPC_0,    mmHD1_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC8,  1},
    {"TPC_1_1",       GAUDI3_HDCORE1_ENGINE_ID_TPC_1,    mmHD1_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC9,  1},
    {"TPC_1_2",       GAUDI3_HDCORE1_ENGINE_ID_TPC_2,    mmHD1_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC10, 1},
    {"TPC_1_3",       GAUDI3_HDCORE1_ENGINE_ID_TPC_3,    mmHD1_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC11, 1},
    {"TPC_1_4",       GAUDI3_HDCORE1_ENGINE_ID_TPC_4,    mmHD1_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC12, 1},
    {"TPC_1_5",       GAUDI3_HDCORE1_ENGINE_ID_TPC_5,    mmHD1_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC13, 1},
    {"TPC_1_6",       GAUDI3_HDCORE1_ENGINE_ID_TPC_6,    mmHD1_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC14, 1},
    {"TPC_1_7",       GAUDI3_HDCORE1_ENGINE_ID_TPC_7,    mmHD1_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC15, 1},
    {"TPC_2_0",       GAUDI3_HDCORE2_ENGINE_ID_TPC_0,    mmHD2_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC16, 2},
    {"TPC_2_1",       GAUDI3_HDCORE2_ENGINE_ID_TPC_1,    mmHD2_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC17, 2},
    {"TPC_2_2",       GAUDI3_HDCORE2_ENGINE_ID_TPC_2,    mmHD2_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC18, 2},
    {"TPC_2_3",       GAUDI3_HDCORE2_ENGINE_ID_TPC_3,    mmHD2_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC19, 2},
    {"TPC_2_4",       GAUDI3_HDCORE2_ENGINE_ID_TPC_4,    mmHD2_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC20, 2},
    {"TPC_2_5",       GAUDI3_HDCORE2_ENGINE_ID_TPC_5,    mmHD2_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC21, 2},
    {"TPC_2_6",       GAUDI3_HDCORE2_ENGINE_ID_TPC_6,    mmHD2_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC22, 2},
    {"TPC_2_7",       GAUDI3_HDCORE2_ENGINE_ID_TPC_7,    mmHD2_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC23, 2},
    {"TPC_3_0",       GAUDI3_HDCORE3_ENGINE_ID_TPC_0,    mmHD3_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC24, 3},
    {"TPC_3_1",       GAUDI3_HDCORE3_ENGINE_ID_TPC_1,    mmHD3_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC25, 3},
    {"TPC_3_2",       GAUDI3_HDCORE3_ENGINE_ID_TPC_2,    mmHD3_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC26, 3},
    {"TPC_3_3",       GAUDI3_HDCORE3_ENGINE_ID_TPC_3,    mmHD3_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC27, 3},
    {"TPC_3_4",       GAUDI3_HDCORE3_ENGINE_ID_TPC_4,    mmHD3_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC28, 3},
    {"TPC_3_5",       GAUDI3_HDCORE3_ENGINE_ID_TPC_5,    mmHD3_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC29, 3},
    {"TPC_3_6",       GAUDI3_HDCORE3_ENGINE_ID_TPC_6,    mmHD3_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC30, 3},
    {"TPC_3_7",       GAUDI3_HDCORE3_ENGINE_ID_TPC_7,    mmHD3_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC31, 3},
    {"TPC_4_0",       GAUDI3_HDCORE4_ENGINE_ID_TPC_0,    mmHD4_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC32, 4},
    {"TPC_4_1",       GAUDI3_HDCORE4_ENGINE_ID_TPC_1,    mmHD4_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC33, 4},
    {"TPC_4_2",       GAUDI3_HDCORE4_ENGINE_ID_TPC_2,    mmHD4_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC34, 4},
    {"TPC_4_3",       GAUDI3_HDCORE4_ENGINE_ID_TPC_3,    mmHD4_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC35, 4},
    {"TPC_4_4",       GAUDI3_HDCORE4_ENGINE_ID_TPC_4,    mmHD4_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC36, 4},
    {"TPC_4_5",       GAUDI3_HDCORE4_ENGINE_ID_TPC_5,    mmHD4_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC37, 4},
    {"TPC_4_6",       GAUDI3_HDCORE4_ENGINE_ID_TPC_6,    mmHD4_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC38, 4},
    {"TPC_4_7",       GAUDI3_HDCORE4_ENGINE_ID_TPC_7,    mmHD4_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC39, 4},
    {"TPC_5_0",       GAUDI3_HDCORE5_ENGINE_ID_TPC_0,    mmHD5_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC40, 5},
    {"TPC_5_1",       GAUDI3_HDCORE5_ENGINE_ID_TPC_1,    mmHD5_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC41, 5},
    {"TPC_5_2",       GAUDI3_HDCORE5_ENGINE_ID_TPC_2,    mmHD5_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC42, 5},
    {"TPC_5_3",       GAUDI3_HDCORE5_ENGINE_ID_TPC_3,    mmHD5_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC43, 5},
    {"TPC_5_4",       GAUDI3_HDCORE5_ENGINE_ID_TPC_4,    mmHD5_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC44, 5},
    {"TPC_5_5",       GAUDI3_HDCORE5_ENGINE_ID_TPC_5,    mmHD5_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC45, 5},
    {"TPC_5_6",       GAUDI3_HDCORE5_ENGINE_ID_TPC_6,    mmHD5_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC46, 5},
    {"TPC_5_7",       GAUDI3_HDCORE5_ENGINE_ID_TPC_7,    mmHD5_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC47, 5},
    {"TPC_6_0",       GAUDI3_HDCORE6_ENGINE_ID_TPC_0,    mmHD6_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC48, 6},
    {"TPC_6_1",       GAUDI3_HDCORE6_ENGINE_ID_TPC_1,    mmHD6_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC49, 6},
    {"TPC_6_2",       GAUDI3_HDCORE6_ENGINE_ID_TPC_2,    mmHD6_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC50, 6},
    {"TPC_6_3",       GAUDI3_HDCORE6_ENGINE_ID_TPC_3,    mmHD6_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC51, 6},
    {"TPC_6_4",       GAUDI3_HDCORE6_ENGINE_ID_TPC_4,    mmHD6_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC52, 6},
    {"TPC_6_5",       GAUDI3_HDCORE6_ENGINE_ID_TPC_5,    mmHD6_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC53, 6},
    {"TPC_6_6",       GAUDI3_HDCORE6_ENGINE_ID_TPC_6,    mmHD6_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC54, 6},
    {"TPC_6_7",       GAUDI3_HDCORE6_ENGINE_ID_TPC_7,    mmHD6_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC55, 6},
    {"TPC_7_0",       GAUDI3_HDCORE7_ENGINE_ID_TPC_0,    mmHD7_TPC0_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC56, 7},
    {"TPC_7_1",       GAUDI3_HDCORE7_ENGINE_ID_TPC_1,    mmHD7_TPC1_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC57, 7},
    {"TPC_7_2",       GAUDI3_HDCORE7_ENGINE_ID_TPC_2,    mmHD7_TPC2_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC58, 7},
    {"TPC_7_3",       GAUDI3_HDCORE7_ENGINE_ID_TPC_3,    mmHD7_TPC3_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC59, 7},
    {"TPC_7_4",       GAUDI3_HDCORE7_ENGINE_ID_TPC_4,    mmHD7_TPC4_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC60, 7},
    {"TPC_7_5",       GAUDI3_HDCORE7_ENGINE_ID_TPC_5,    mmHD7_TPC5_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC61, 7},
    {"TPC_7_6",       GAUDI3_HDCORE7_ENGINE_ID_TPC_6,    mmHD7_TPC6_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC62, 7},
    {"TPC_7_7",       GAUDI3_HDCORE7_ENGINE_ID_TPC_7,    mmHD7_TPC7_QM_DCCM_BASE,           TPC,          CPU_ID_TPC_QMAN_ARC63, 7},

    {"MME_0",         GAUDI3_HDCORE0_ENGINE_ID_MME_0,    mmHD0_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC0,  0},
    {"MME_1",         GAUDI3_HDCORE1_ENGINE_ID_MME_0,    mmHD1_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC1,  1},
    {"MME_2",         GAUDI3_HDCORE2_ENGINE_ID_MME_0,    mmHD2_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC2,  2},
    {"MME_3",         GAUDI3_HDCORE3_ENGINE_ID_MME_0,    mmHD3_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC3,  3},
    {"MME_4",         GAUDI3_HDCORE4_ENGINE_ID_MME_0,    mmHD4_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC4,  4},
    {"MME_5",         GAUDI3_HDCORE5_ENGINE_ID_MME_0,    mmHD5_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC5,  5},
    {"MME_6",         GAUDI3_HDCORE6_ENGINE_ID_MME_0,    mmHD6_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC6,  6},
    {"MME_7",         GAUDI3_HDCORE7_ENGINE_ID_MME_0,    mmHD7_MME_QM_ARC_DCCM_BASE,        MME,          CPU_ID_MME_QMAN_ARC7,  7},

    {"EDMA_1_0",      GAUDI3_HDCORE1_ENGINE_ID_EDMA_0,   mmHD1_SEDMA0_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC0, 1},
    {"EDMA_1_1",      GAUDI3_HDCORE1_ENGINE_ID_EDMA_1,   mmHD1_SEDMA1_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC1, 1},
    {"EDMA_3_0",      GAUDI3_HDCORE3_ENGINE_ID_EDMA_0,   mmHD3_SEDMA0_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC2, 3},
    {"EDMA_3_1",      GAUDI3_HDCORE3_ENGINE_ID_EDMA_1,   mmHD3_SEDMA1_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC3, 3},
    {"EDMA_4_0",      GAUDI3_HDCORE4_ENGINE_ID_EDMA_0,   mmHD4_SEDMA0_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC4, 4},
    {"EDMA_4_1",      GAUDI3_HDCORE4_ENGINE_ID_EDMA_1,   mmHD4_SEDMA1_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC5, 4},
    {"EDMA_6_0",      GAUDI3_HDCORE6_ENGINE_ID_EDMA_0,   mmHD6_SEDMA0_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC6, 6},
    {"EDMA_6_1",      GAUDI3_HDCORE6_ENGINE_ID_EDMA_1,   mmHD6_SEDMA1_QM_DCCM_BASE,         EDMA,         CPU_ID_EDMA_QMAN_ARC7, 6},

    {"ROT_1_0",       GAUDI3_HDCORE1_ENGINE_ID_ROT_0,    mmHD1_ROT0_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC0,  1},
    {"ROT_1_1",       GAUDI3_HDCORE1_ENGINE_ID_ROT_1,    mmHD1_ROT1_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC1,  1},
    {"ROT_3_0",       GAUDI3_HDCORE3_ENGINE_ID_ROT_0,    mmHD3_ROT0_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC2,  3},
    {"ROT_3_1",       GAUDI3_HDCORE3_ENGINE_ID_ROT_1,    mmHD3_ROT1_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC3,  3},
    {"ROT_4_0",       GAUDI3_HDCORE4_ENGINE_ID_ROT_0,    mmHD4_ROT0_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC4,  4},
    {"ROT_4_1",       GAUDI3_HDCORE4_ENGINE_ID_ROT_1,    mmHD4_ROT1_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC5,  4},
    {"ROT_6_0",       GAUDI3_HDCORE6_ENGINE_ID_ROT_0,    mmHD6_ROT0_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC6,  6},
    {"ROT_6_1",       GAUDI3_HDCORE6_ENGINE_ID_ROT_1,    mmHD6_ROT1_QM_ARC_DCCM_BASE,       ROT,          CPU_ID_ROT_QMAN_ARC7,  6},

    {"NIC_0_0",       GAUDI3_DIE0_ENGINE_ID_NIC_0,       mmD0_NIC0_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC0    },
    {"NIC_0_1",       GAUDI3_DIE0_ENGINE_ID_NIC_1,       mmD0_NIC1_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC1    },
    {"NIC_0_2",       GAUDI3_DIE0_ENGINE_ID_NIC_2,       mmD0_NIC2_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC2    },
    {"NIC_0_3",       GAUDI3_DIE0_ENGINE_ID_NIC_3,       mmD0_NIC3_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC3    },
    {"NIC_0_4",       GAUDI3_DIE0_ENGINE_ID_NIC_4,       mmD0_NIC4_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC4    },
    {"NIC_0_5",       GAUDI3_DIE0_ENGINE_ID_NIC_5,       mmD0_NIC5_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC5    },
    {"NIC_1_0",       GAUDI3_DIE1_ENGINE_ID_NIC_0,       mmD1_NIC0_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC6    },
    {"NIC_1_1",       GAUDI3_DIE1_ENGINE_ID_NIC_1,       mmD1_NIC1_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC7    },
    {"NIC_1_2",       GAUDI3_DIE1_ENGINE_ID_NIC_2,       mmD1_NIC2_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC8    },
    {"NIC_1_3",       GAUDI3_DIE1_ENGINE_ID_NIC_3,       mmD1_NIC3_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC9    },
    {"NIC_1_4",       GAUDI3_DIE1_ENGINE_ID_NIC_4,       mmD1_NIC4_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC10   },
    {"NIC_1_5",       GAUDI3_DIE1_ENGINE_ID_NIC_5,       mmD1_NIC5_QM_DCCM_BASE,            NIC,          CPU_ID_NIC_QMAN_ARC11   }
}};

struct PdmaChannelInfo
{
    const char * name;
    gaudi3_engine_id engineId;
    uint64_t baseAddrA;
    uint64_t baseAddrB;
    uint64_t baseAddrCmnB;
    unsigned channelId;
};

static const PdmaChannelInfo c_pdma_channels_info_arr[] = {
    {"PDMA_0_0", GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_0, mmD0_SPDMA0_CH0_A_BASE, mmD0_SPDMA0_CH0_B_BASE, mmD0_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH0 },
    {"PDMA_0_1", GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_1, mmD0_SPDMA0_CH1_A_BASE, mmD0_SPDMA0_CH1_B_BASE, mmD0_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH1 },
    {"PDMA_0_2", GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_2, mmD0_SPDMA0_CH2_A_BASE, mmD0_SPDMA0_CH2_B_BASE, mmD0_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH2 },
    {"PDMA_0_3", GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_3, mmD0_SPDMA0_CH3_A_BASE, mmD0_SPDMA0_CH3_B_BASE, mmD0_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH3 },
    {"PDMA_0_4", GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_4, mmD0_SPDMA0_CH4_A_BASE, mmD0_SPDMA0_CH4_B_BASE, mmD0_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH4 },
    {"PDMA_0_5", GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_5, mmD0_SPDMA0_CH5_A_BASE, mmD0_SPDMA0_CH5_B_BASE, mmD0_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH5 },
    {"PDMA_1_0", GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_0, mmD0_SPDMA1_CH0_A_BASE, mmD0_SPDMA1_CH0_B_BASE, mmD0_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH6 },
    {"PDMA_1_1", GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_1, mmD0_SPDMA1_CH1_A_BASE, mmD0_SPDMA1_CH1_B_BASE, mmD0_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH7 },
    {"PDMA_1_2", GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_2, mmD0_SPDMA1_CH2_A_BASE, mmD0_SPDMA1_CH2_B_BASE, mmD0_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH8 },
    {"PDMA_1_3", GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_3, mmD0_SPDMA1_CH3_A_BASE, mmD0_SPDMA1_CH3_B_BASE, mmD0_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH9 },
    {"PDMA_1_4", GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_4, mmD0_SPDMA1_CH4_A_BASE, mmD0_SPDMA1_CH4_B_BASE, mmD0_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH10},
    {"PDMA_1_5", GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_5, mmD0_SPDMA1_CH5_A_BASE, mmD0_SPDMA1_CH5_B_BASE, mmD0_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE0_CH11},
    {"PDMA_2_0", GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_0, mmD1_SPDMA0_CH0_A_BASE, mmD1_SPDMA0_CH0_B_BASE, mmD1_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH0 },
    {"PDMA_2_1", GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_1, mmD1_SPDMA0_CH1_A_BASE, mmD1_SPDMA0_CH1_B_BASE, mmD1_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH1 },
    {"PDMA_2_2", GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_2, mmD1_SPDMA0_CH2_A_BASE, mmD1_SPDMA0_CH2_B_BASE, mmD1_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH2 },
    {"PDMA_2_3", GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_3, mmD1_SPDMA0_CH3_A_BASE, mmD1_SPDMA0_CH3_B_BASE, mmD1_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH3 },
    {"PDMA_2_4", GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_4, mmD1_SPDMA0_CH4_A_BASE, mmD1_SPDMA0_CH4_B_BASE, mmD1_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH4 },
    {"PDMA_2_5", GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_5, mmD1_SPDMA0_CH5_A_BASE, mmD1_SPDMA0_CH5_B_BASE, mmD1_SPDMA0_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH5 },
    {"PDMA_3_0", GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_0, mmD1_SPDMA1_CH0_A_BASE, mmD1_SPDMA1_CH0_B_BASE, mmD1_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH6 },
    {"PDMA_3_1", GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_1, mmD1_SPDMA1_CH1_A_BASE, mmD1_SPDMA1_CH1_B_BASE, mmD1_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH7 },
    {"PDMA_3_2", GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_2, mmD1_SPDMA1_CH2_A_BASE, mmD1_SPDMA1_CH2_B_BASE, mmD1_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH8 },
    {"PDMA_3_3", GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_3, mmD1_SPDMA1_CH3_A_BASE, mmD1_SPDMA1_CH3_B_BASE, mmD1_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH9 },
    {"PDMA_3_4", GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_4, mmD1_SPDMA1_CH4_A_BASE, mmD1_SPDMA1_CH4_B_BASE, mmD1_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH10},
    {"PDMA_3_5", GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_5, mmD1_SPDMA1_CH5_A_BASE, mmD1_SPDMA1_CH5_B_BASE, mmD1_SPDMA1_CMN_B_PQM_CMN_B_BASE, PDMA_DIE1_CH11}
};


static inline bool queueId2DccmAddr(const unsigned qid, uint64_t& dccmAddr)
{
    return qmanId2DccmAddr(qid, dccmAddr, c_engine_info_arr);
}

static inline bool queueId2OffsetToQman(const unsigned qid, unsigned& offsetToQman)
{
    for (const EngineInfoG3& info : c_engine_info_arr)
    {
        if (qid == info.queueId)
        {
            offsetToQman = info.coreType == CoreType::NIC ? Scal_Gaudi3::c_nic_dccm_to_qm_offset : Scal_Gaudi3::c_dccm_to_qm_offset;
            return true;
        }
    }

    return false;
}


static inline bool arcName2QueueId(const std::string & arcName, unsigned &eid)
{
    return arcName2QueueId(arcName, eid, c_engine_info_arr);
}

static inline bool arcName2DccmAddr(const std::string & arcName, uint64_t &dccmAddr)
{
    return arcName2DccmAddr(arcName, dccmAddr, c_engine_info_arr);
}

static inline bool arcName2ArcType(const std::string & arcName, bool &isScheduler)
{
    for (const EngineInfoG3 & info : c_engine_info_arr)
    {
        if (arcName == info.name)
        {
            isScheduler = (info.coreType == SCHEDULER);
            return true;
        }
    }
    return false;
}

static inline bool arcName2CoreType(const std::string & arcName, CoreType &coreType)
{
    return arcName2CoreType(arcName, coreType, c_engine_info_arr);
}

static inline bool arcName2CpuId(const std::string & arcName, unsigned &cpuId)
{
    return arcName2CpuId(arcName, cpuId, c_engine_info_arr);
}

static inline bool arcName2HdCore(const std::string & arcName, unsigned &hdCore)
{
    for (const EngineInfoG3 & info : c_engine_info_arr)
    {
        if (arcName == info.name)
        {
            hdCore = info.hdCore;
            return true;
        }
    }

    return false;
}

static inline bool arcName2DCore(const std::string & arcName, unsigned &dCore)
{
    for (const EngineInfoG3 & info : c_engine_info_arr)
    {
        if (arcName == info.name)
        {
            dCore = info.hdCore / 2;
            return true;
        }
    }

    return false;
}

struct GroupName2GroupIndex
{
    const char* name;
    unsigned    index;
};

static const GroupName2GroupIndex c_group_name_2_group_index[] = {
    {"MME_COMPUTE_GROUP", ScalComputeGroups::SCAL_MME_COMPUTE_GROUP},
    {"TPC_COMPUTE_GROUP", ScalComputeGroups::SCAL_TPC_COMPUTE_GROUP},
    {"EDMA_COMPUTE_GROUP", ScalComputeGroups::SCAL_EDMA_COMPUTE_GROUP},
    {"CME_GROUP", ScalComputeGroups::SCAL_CME_GROUP},
    {"RTR_COMPUTE_GROUP", ScalComputeGroups::SCAL_RTR_COMPUTE_GROUP},
    {"PDMA_TX_CMD_GROUP", ScalComputeGroups::SCAL_PDMA_TX_CMD_GROUP},
    {"PDMA_TX_DATA_GROUP", ScalComputeGroups::SCAL_PDMA_TX_DATA_GROUP},
    {"PDMA_RX_GROUP", ScalComputeGroups::SCAL_PDMA_RX_GROUP},
    {"PDMA_DEV2DEV_DEBUG_GROUP", ScalComputeGroups::SCAL_PDMA_DEV2DEV_DEBUG_GROUP},
    {"PDMA_RX_DEBUG_GROUP", ScalComputeGroups::SCAL_PDMA_RX_DEBUG_GROUP},
    {"NIC_RECEIVE_SCALE_UP_GROUP", ScalNetworkScaleUpReceiveGroups::SCAL_NIC_RECEIVE_SCALE_UP_GROUP},
    {"NIC_RECEIVE_SCALE_OUT_GROUP", ScalNetworkScaleOutReceiveGroups::SCAL_NIC_RECEIVE_SCALE_OUT_GROUP},
    {"NIC_SEND_SCALE_UP_GROUP", ScalNetworkScaleUpSendGroups::SCAL_NIC_SEND_SCALE_UP_GROUP},
    {"NIC_SEND_SCALE_OUT_GROUP", ScalNetworkScaleOutSendGroups::SCAL_NIC_SEND_SCALE_OUT_GROUP},
    {"EDMA_NETWORK_SCALE_UP_SEND_GROUP0", ScalNetworkScaleUpSendGroups::SCAL_EDMA_NETWORK_SCALE_UP_SEND_GROUP0},
    {"EDMA_NETWORK_GC_REDUCTION_GROUP0", ScalNetworkGarbageCollectorAndReductionGroups::SCAL_EDMA_NETWORK_GC_REDUCTION_GROUP0},
    {"EDMA_NETWORK_SCALE_OUT_SEND_GROUP0", ScalNetworkScaleOutSendGroups::SCAL_EDMA_NETWORK_SCALE_OUT_SEND_GROUP0},
    {"EDMA_NETWORK_SCALE_UP_RECV_GROUP0", ScalNetworkScaleUpReceiveGroups::SCAL_EDMA_NETWORK_SCALE_UP_RECV_GROUP0},
    {"PDMA_NETWORK_SCALE_OUT_SEND_GROUP", ScalNetworkScaleOutSendGroups::SCAL_PDMA_NETWORK_SCALE_OUT_SEND_GROUP},
    {"PDMA_NETWORK_SCALE_OUT_RECV_GROUP", ScalNetworkScaleOutReceiveGroups::SCAL_PDMA_NETWORK_SCALE_OUT_RECV_GROUP}
};

static inline bool groupName2GroupIndex(const std::string & groupName, unsigned& groupIndex)
{
    for (const auto& group : c_group_name_2_group_index)
    {
        if (groupName == group.name)
        {
            groupIndex = group.index;
            return true;
        }
    }

    return false;
};

struct priority_name_to_number
{
    std::string name;
    unsigned    priority;
};

static const priority_name_to_number c_priority_name_to_number[] =
{
    {"PRIORITY_HIGH", SCAL_HIGH_PRIORITY_STREAM},
    {"PRIORITY_LOW", SCAL_LOW_PRIORITY_STREAM}
};

static inline bool getPriorityByName(const std::string& priorityName, unsigned& priority)
{
    for (const auto& pri : c_priority_name_to_number)
    {
        if (priorityName == pri.name)
        {
            priority = pri.priority;
            return true;
        }
    }

    return false;
}

static inline bool pdmaName2ChannelID(const std::string & pdmaName, unsigned &qid)
{
    for (const PdmaChannelInfo& info : c_pdma_channels_info_arr)
    {
        if (pdmaName == info.name)
        {
            qid = (unsigned)info.engineId;
            return true;
        }
    }
    assert(0);
    return false;
}

static inline bool pdmaId2baseAddrA(const unsigned &cid, uint64_t &addr)
{
    for (const PdmaChannelInfo& info : c_pdma_channels_info_arr)
    {
        if (cid == info.engineId)
        {
            addr = info.baseAddrA;
            return true;
        }
    }
    assert(0);
    return false;
}

static inline bool pdmaName2PdmaChannelInfo(const std::string & pdmaName, const PdmaChannelInfo*& pdmaChannelInfo)
{
    for (const PdmaChannelInfo& info : c_pdma_channels_info_arr)
    {
        if (pdmaName == info.name)
        {
            pdmaChannelInfo = &info;
            return true;
        }
    }
    assert(0);
    return false;
}

struct dup_trigger_info
{
    uint64_t   dup_engines_address_base;
    uint64_t   cluster_mask;
    uint64_t   dup_offset;
    unsigned   max_engines;

};

static const dup_trigger_info c_dup_trigger_info[] =
{
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_0 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_0 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_0 ), 64 }, //DUP_ADDR_GR_0,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_1 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_1 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_1 ), 16 }, //DUP_ADDR_GR_1,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_2 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_2 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_2 ), 16 }, //DUP_ADDR_GR_2,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_3 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_3 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_3 ), 16 }, //DUP_ADDR_GR_3,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_4 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_4 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_4 ), 16 }, //DUP_ADDR_GR_4,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_5 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_5 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_5 ), 16 }, //DUP_ADDR_GR_5,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_6 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_6 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_6 ), 16 }, //DUP_ADDR_GR_6,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_7 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_7 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_7 ), 16 }, //DUP_ADDR_GR_7,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_8 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_8 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_8 ), 16 }, //DUP_ADDR_GR_8,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_9 ), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_9 ), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_9 ), 16 }, //DUP_ADDR_GR_9,
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_10), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_10), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_10), 16 }, //DUP_ADDR_GR_10
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_11), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_11), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_11), 16 }, //DUP_ADDR_GR_11
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_12), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_12), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_12), 16 }, //DUP_ADDR_GR_12
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_13), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_13), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_13), 16 }, //DUP_ADDR_GR_13
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_14), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_14), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_14), 16 }, //DUP_ADDR_GR_14
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_15), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_15), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_15), 16 }, //DUP_ADDR_GR_15
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_16), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_16), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_16), 16 }, //DUP_ADDR_GR_16
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_17), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_17), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_17), 16 }, //DUP_ADDR_GR_17
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_18), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_18), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_18), 16 }, //DUP_ADDR_GR_18
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_19), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_19), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_19), 16 }, //DUP_ADDR_GR_19
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_20), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_20), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_20), 16 }, //DUP_ADDR_GR_20
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_21), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_21), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_21), 16 }, //DUP_ADDR_GR_21
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_22), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_22), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_22), 16 }, //DUP_ADDR_GR_22
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_23), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_23), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_23), 16 }, //DUP_ADDR_GR_23
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_24), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_24), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_24), 16 }, //DUP_ADDR_GR_24
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_25), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_25), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_25), 16 }, //DUP_ADDR_GR_25
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_26), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_26), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_26), 16 }, //DUP_ADDR_GR_26
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_27), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_27), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_27), 16 }, //DUP_ADDR_GR_27
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_28), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_28), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_28), 16 }, //DUP_ADDR_GR_28
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_29), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_29), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_29), 16 }, //DUP_ADDR_GR_29
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_30), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_30), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_30), 16 }, //DUP_ADDR_GR_30
    {offsetof(gaudi3::block_arc_dup_eng, dup_addr_gr_31), offsetof(gaudi3::block_arc_dup_eng, dup_mask_gr_31), offsetof(gaudi3::block_arc_dup_eng, dup_offset_gr_31), 16 }, //DUP_ADDR_GR_31
};

static constexpr unsigned c_fence_edup_trigger = 1;
static constexpr unsigned c_commands_edup_trigger_offset = 2;

/* SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2021 HabanaLabs Ltd.
 * All Rights Reserved.
 */

#ifndef __GAUDI3_ARC_COMMON_PACKETS_H__
#define __GAUDI3_ARC_COMMON_PACKETS_H__

/**
 * \file    gaudi3_arc_common_packets.h
 * \brief   CPU IDs for each ARC CPUs
 *          This defines IDs that needs to be programmed into CPU ID
 *          register xx_ARC_AUX_ARC_NUM
 */
enum {
	/* Each HDCORE has 2 scheduler ARCs- Total 16 ARCs */
	CPU_ID_SCHED_ARC0  = 0, /* HD0_FARM_ARC0 */
	CPU_ID_SCHED_ARC1  = 1, /* HD0_FARM_ARC1 */
	CPU_ID_SCHED_ARC2  = 2, /* HD1_FARM_ARC0 */
	CPU_ID_SCHED_ARC3  = 3, /* HD1_FARM_ARC1 */
	CPU_ID_SCHED_ARC4  = 4, /* HD2_FARM_ARC0 */
	CPU_ID_SCHED_ARC5  = 5, /* HD2_FARM_ARC1 */
	CPU_ID_SCHED_ARC6  = 6, /* HD3_FARM_ARC0 */
	CPU_ID_SCHED_ARC7 = 7, /* HD3_FARM_ARC1 */
	CPU_ID_SCHED_ARC8 = 8, /* HD4_FARM_ARC0 */
	CPU_ID_SCHED_ARC9 = 9, /* HD4_FARM_ARC1 */
	CPU_ID_SCHED_ARC10 = 10, /* HD5_FARM_ARC0 */
	CPU_ID_SCHED_ARC11 = 11, /* HD5_FARM_ARC1 */
	CPU_ID_SCHED_ARC12 = 12, /* HD6_FARM_ARC0 */
	CPU_ID_SCHED_ARC13 = 13, /* HD6_FARM_ARC1 */
	CPU_ID_SCHED_ARC14 = 14, /* HD7_FARM_ARC0 */
	CPU_ID_SCHED_ARC15 = 15, /* HD7_FARM_ARC1 */

	/* HD0,2,5,7 has 9 TPCs, HD1,3,4,6 has 8 TPCs, Total 68 ARCs */
	CPU_ID_TPC_QMAN_ARC0 = 16, /* HD0_TPC0 */
	CPU_ID_TPC_QMAN_ARC1 = 17, /* HD0_TPC1 */
	CPU_ID_TPC_QMAN_ARC2 = 18, /* HD0_TPC2 */
	CPU_ID_TPC_QMAN_ARC3 = 19, /* HD0_TPC3 */
	CPU_ID_TPC_QMAN_ARC4 = 20, /* HD0_TPC4 */
	CPU_ID_TPC_QMAN_ARC5 = 21, /* HD0_TPC5 */
	CPU_ID_TPC_QMAN_ARC6 = 22, /* HD0_TPC6 */
	CPU_ID_TPC_QMAN_ARC7 = 23, /* HD0_TPC7 */
	CPU_ID_TPC_QMAN_ARC8 = 24, /* HD1_TPC0 */
	CPU_ID_TPC_QMAN_ARC9 = 25, /* HD1_TPC1 */
	CPU_ID_TPC_QMAN_ARC10 = 26, /* HD1_TPC2 */
	CPU_ID_TPC_QMAN_ARC11 = 27, /* HD1_TPC3 */
	CPU_ID_TPC_QMAN_ARC12 = 28, /* HD1_TPC4 */
	CPU_ID_TPC_QMAN_ARC13 = 29, /* HD1_TPC5 */
	CPU_ID_TPC_QMAN_ARC14 = 30, /* HD1_TPC6 */
	CPU_ID_TPC_QMAN_ARC15 = 31, /* HD1_TPC7 */
	CPU_ID_TPC_QMAN_ARC16 = 32, /* HD2_TPC0 */
	CPU_ID_TPC_QMAN_ARC17 = 33, /* HD2_TPC1 */
	CPU_ID_TPC_QMAN_ARC18 = 34, /* HD2_TPC2 */
	CPU_ID_TPC_QMAN_ARC19 = 35, /* HD2_TPC3 */
	CPU_ID_TPC_QMAN_ARC20 = 36, /* HD2_TPC4 */
	CPU_ID_TPC_QMAN_ARC21 = 37, /* HD2_TPC5 */
	CPU_ID_TPC_QMAN_ARC22 = 38, /* HD2_TPC6 */
	CPU_ID_TPC_QMAN_ARC23 = 39, /* HD2_TPC7 */
	CPU_ID_TPC_QMAN_ARC24 = 40, /* HD3_TPC0 */
	CPU_ID_TPC_QMAN_ARC25 = 41, /* HD3_TPC1 */
	CPU_ID_TPC_QMAN_ARC26 = 42, /* HD3_TPC2 */
	CPU_ID_TPC_QMAN_ARC27 = 43, /* HD3_TPC3 */
	CPU_ID_TPC_QMAN_ARC28 = 44, /* HD3_TPC4 */
	CPU_ID_TPC_QMAN_ARC29 = 45, /* HD3_TPC5 */
	CPU_ID_TPC_QMAN_ARC30 = 46, /* HD3_TPC6 */
	CPU_ID_TPC_QMAN_ARC31 = 47, /* HD3_TPC7 */
	CPU_ID_TPC_QMAN_ARC32 = 48, /* HD4_TPC0 */
	CPU_ID_TPC_QMAN_ARC33 = 49, /* HD4_TPC1 */
	CPU_ID_TPC_QMAN_ARC34 = 50, /* HD4_TPC2 */
	CPU_ID_TPC_QMAN_ARC35 = 51, /* HD4_TPC3 */
	CPU_ID_TPC_QMAN_ARC36 = 52, /* HD4_TPC4 */
	CPU_ID_TPC_QMAN_ARC37 = 53, /* HD4_TPC5 */
	CPU_ID_TPC_QMAN_ARC38 = 54, /* HD4_TPC6 */
	CPU_ID_TPC_QMAN_ARC39 = 55, /* HD4_TPC7 */
	CPU_ID_TPC_QMAN_ARC40 = 56, /* HD5_TPC0 */
	CPU_ID_TPC_QMAN_ARC41 = 57, /* HD5_TPC1 */
	CPU_ID_TPC_QMAN_ARC42 = 58, /* HD5_TPC2 */
	CPU_ID_TPC_QMAN_ARC43 = 59, /* HD5_TPC3 */
	CPU_ID_TPC_QMAN_ARC44 = 60, /* HD5_TPC4 */
	CPU_ID_TPC_QMAN_ARC45 = 61, /* HD5_TPC5 */
	CPU_ID_TPC_QMAN_ARC46 = 62, /* HD5_TPC6 */

	CPU_ID_TPC_QMAN_ARC47 = 63, /* HD5_TPC7 */
	CPU_ID_TPC_QMAN_ARC48 = 64, /* HD6_TPC0 */
	CPU_ID_TPC_QMAN_ARC49 = 65, /* HD6_TPC1 */
	CPU_ID_TPC_QMAN_ARC50 = 66, /* HD6_TPC2 */
	CPU_ID_TPC_QMAN_ARC51 = 67, /* HD6_TPC3 */
	CPU_ID_TPC_QMAN_ARC52 = 68, /* HD6_TPC4 */
	CPU_ID_TPC_QMAN_ARC53 = 69, /* HD6_TPC5 */
	CPU_ID_TPC_QMAN_ARC54 = 70, /* HD6_TPC6 */
	CPU_ID_TPC_QMAN_ARC55 = 71, /* HD6_TPC7 */
	CPU_ID_TPC_QMAN_ARC56 = 72, /* HD7_TPC0 */
	CPU_ID_TPC_QMAN_ARC57 = 73, /* HD7_TPC1 */
	CPU_ID_TPC_QMAN_ARC58 = 74, /* HD7_TPC2 */
	CPU_ID_TPC_QMAN_ARC59 = 75, /* HD7_TPC3 */
	CPU_ID_TPC_QMAN_ARC60 = 76, /* HD7_TPC4 */
	CPU_ID_TPC_QMAN_ARC61 = 77, /* HD7_TPC5 */
	CPU_ID_TPC_QMAN_ARC62 = 78, /* HD7_TPC6 */
	CPU_ID_TPC_QMAN_ARC63 = 79, /* HD7_TPC7 */
	CPU_ID_TPC_QMAN_ARC64 = 80, /* HD0_TPC8 */
	CPU_ID_TPC_QMAN_ARC65 = 81, /* HD2_TPC8 */
	CPU_ID_TPC_QMAN_ARC66 = 82, /* HD5_TPC8 */
	CPU_ID_TPC_QMAN_ARC67 = 83, /* HD7_TPC8 */

	/* Each HDCORE has 1 Master MME Total 8 ARCs */
	CPU_ID_MME_QMAN_ARC0  = 84, /* HD0_MME */
	CPU_ID_MME_QMAN_ARC1  = 85, /* HD1_MME */
	CPU_ID_MME_QMAN_ARC2  = 86, /* HD2_MME */
	CPU_ID_MME_QMAN_ARC3  = 87, /* HD3_MME */
	CPU_ID_MME_QMAN_ARC4  = 88, /* HD4_MME */
	CPU_ID_MME_QMAN_ARC5  = 89, /* HD5_MME */
	CPU_ID_MME_QMAN_ARC6  = 90, /* HD6_MME */
	CPU_ID_MME_QMAN_ARC7  = 91, /* HD7_MME */

	/* HDCORE 1, 3, 4, 6 - each has 2 EDMA ARCs total 8 ARCs */
	CPU_ID_EDMA_QMAN_ARC0 = 92, /* HD1_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC1 = 93, /* HD1_EDMA1 */
	CPU_ID_EDMA_QMAN_ARC2 = 94, /* HD3_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC3 = 95, /* HD3_EDMA1 */
	CPU_ID_EDMA_QMAN_ARC4 = 96, /* HD4_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC5 = 97, /* HD4_EDMA1 */
	CPU_ID_EDMA_QMAN_ARC6 = 98, /* HD6_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC7 = 99, /* HD6_EDMA1 */

	/* HDCORE 1, 3, 4, 6 - each has 2 ROT ARCs Total 8 ARCs */
	CPU_ID_ROT_QMAN_ARC0  = 100,/* HD1_ROT0 */
	CPU_ID_ROT_QMAN_ARC1  = 101,/* HD1_ROT1 */
	CPU_ID_ROT_QMAN_ARC2  = 102,/* HD3_ROT0 */
	CPU_ID_ROT_QMAN_ARC3  = 103,/* HD3_ROT1 */
	CPU_ID_ROT_QMAN_ARC4  = 104,/* HD4_ROT0 */
	CPU_ID_ROT_QMAN_ARC5  = 105,/* HD4_ROT1 */
	CPU_ID_ROT_QMAN_ARC6  = 106,/* HD6_ROT0 */
	CPU_ID_ROT_QMAN_ARC7  = 107,/* HD6_ROT1 */

	/* Each DIE has 6 NIC ARCs Total 12 ARCs */
	CPU_ID_NIC_QMAN_ARC0  = 108, /* D0_NIC0 */
	CPU_ID_NIC_QMAN_ARC1  = 109, /* D0_NIC1 */
	CPU_ID_NIC_QMAN_ARC2  = 110, /* D0_NIC2 */
	CPU_ID_NIC_QMAN_ARC3  = 111, /* D0_NIC3 */
	CPU_ID_NIC_QMAN_ARC4  = 112, /* D0_NIC4 */
	CPU_ID_NIC_QMAN_ARC5  = 113, /* D0_NIC5 */
	CPU_ID_NIC_QMAN_ARC6  = 114, /* D1_NIC0 */
	CPU_ID_NIC_QMAN_ARC7  = 115, /* D1_NIC1 */
	CPU_ID_NIC_QMAN_ARC8  = 116, /* D1_NIC2 */
	CPU_ID_NIC_QMAN_ARC9  = 117, /* D1_NIC3 */
	CPU_ID_NIC_QMAN_ARC10 = 118, /* D1_NIC4 */
	CPU_ID_NIC_QMAN_ARC11 = 119, /* D1_NIC5 */

	CPU_ID_MAX = 120,
	CPU_ID_SCHED_MAX = 16,

	CPU_ID_ALL = 0xFE,
	CPU_ID_INVALID = 0xFF,
};

enum {
	/**
	 * PDMA channel numbers
	 */
	PDMA_DIE0_CH0  = 0,
	PDMA_DIE0_CH1  = 1,
	PDMA_DIE0_CH2  = 2,
	PDMA_DIE0_CH3  = 3,
	PDMA_DIE0_CH4  = 4,
	PDMA_DIE0_CH5  = 5,
	PDMA_DIE0_CH6  = 6,
	PDMA_DIE0_CH7  = 7,
	PDMA_DIE0_CH8  = 8,
	PDMA_DIE0_CH9  = 9,
	PDMA_DIE0_CH10 = 10,
	PDMA_DIE0_CH11 = 11,
	PDMA_DIE1_CH0  = 12,
	PDMA_DIE1_CH1  = 13,
	PDMA_DIE1_CH2  = 14,
	PDMA_DIE1_CH3  = 15,
	PDMA_DIE1_CH4  = 16,
	PDMA_DIE1_CH5  = 17,
	PDMA_DIE1_CH6  = 18,
	PDMA_DIE1_CH7  = 19,
	PDMA_DIE1_CH8  = 20,
	PDMA_DIE1_CH9  = 21,
	PDMA_DIE1_CH10 = 22,
	PDMA_DIE1_CH11 = 23,

	PDMA_MAX_CH = 24
};

/**
 * Message codes of SCAL for firmware usage
 * Updated by SCAL into SCHED_SCAL_STATUS register
 */
enum {
	SCAL_INIT_COMPLETED = 0x00010000
};

/**
 * \enum    qman_engine_type_t
 * \brief   Various engine types
 * \details Engine Types supported by Scheduler ARC
 */
enum qman_engine_type_t {
	QMAN_ENGINE_MME = 0,
	QMAN_ENGINE_TPC = 1,
	QMAN_ENGINE_NIC = 2,
	QMAN_ENGINE_RTR = 3,
	QMAN_ENGINE_EDMA = 4,
	QMAN_ENGINE_PDMA = 5,
	QMAN_ENGINE_TYPE_COUNT = 0x6,
	QMAN_ENGINE_LAST = 0x7,
};

/**
 * \enum    engine_type_t
 * \brief   Various engine types
 * \details Engine Types
 */
enum engine_type_t {
	ENG_TYPE_MME_COMPUTE = 0,
	ENG_TYPE_TPC_COMPUTE = 1,
	ENG_TYPE_EDMA_COMPUTE = 2,
	ENG_TYPE_TPC_MEDIA = 3,
	ENG_TYPE_RTR_MEDIA = 4,
	ENG_TYPE_PDMA_TX = 5,
	ENG_TYPE_PDMA_RX = 6,
	ENG_TYPE_NIC_SCALE_UP = 7,
	ENG_TYPE_NIC_SCALE_OUT = 8,
	ENG_TYPE_EDMA_NIC = 9,
	ENG_TYPE_COUNT = 0xA,
	ENG_TYPE_SIZE = 0xF
};

/**
 * \enum    scheduler_type_t
 * \brief   Various scheduler types
 * \details Scheduler Types
 */
enum scheduler_type_t {
	SCHED_TYPE_COMPUTE = 0,
	SCHED_TYPE_GARBAGE_REDUCTION = 1,
	SCHED_TYPE_SCALE_OUT_RECV = 2,
	SCHED_TYPE_SCALE_OUT_SEND = 3,
	SCHED_TYPE_SCALE_UP_RECV = 4,
	SCHED_TYPE_SCALE_UP_SEND = 5,
	SCHED_TYPE_CME = 6,
	SCHED_TYPE_COUNT = 7,
	SCHED_TYPE_SIZE = 0xF
};

/**
 * \enum    scheduler_type_t
 * \brief   Fence IDs used by Engines FW
 * \details Engine Firmware uses following two fences
 *	    for sync scheme and back to back execution
 */
enum {
	SYNC_SCHEME_FENCE_ID = 0,
	B2B_FENCE_ID = 1,
	QMAN_SYNC_FENCE_ID = 2,
	MCID_ROLLOVER_FENCE_ID = 3
};

/**
 * \enum    mcid_wr64_base_ids_t
 * \brief   QMAN WR64 base registers used by firmware
 * \details Various QMAN WR64 base registers used by firmware for
 * 	    MCID patching
 */
enum mcid_wr64_base_ids_t {
	DEGRADE_MCID_WR64_BASE_ID_0 = 28,
	DEGRADE_MCID_WR64_BASE_ID_1 = 29,
	DISCARD_MCID_WR64_BASE_ID_0 = 30,
	DISCARD_MCID_WR64_BASE_ID_1 = 31
};

/**
 * Total number of engine groups supported by firmware
 */
#define QMAN_ENGINE_GROUP_TYPE_COUNT		16

/**
 * Total number of HDcores
 */
#define SCHED_ARC_MAX_HDCORE			8

/**
 * Total number of Streams supported by a scheduler instance
 */
#define SCHED_ARC_MAX_STREAMS			32

/**
 * Max Priorities
 */
#define SCHED_ARC_MAX_ACP_PRIORITY		4

/**
 * Max number of groups that can be created per engine type
 */
#define SCHED_ARC_MAX_ENGINE_TYPE_GROUPS	2

/**
 * Total number of Fence counters per scheduler instance
 */
#define SCHED_ARC_GLOBAL_FENCE_COUNTERS_COUNT	64

/**
 * Size of the DCCM CCB buffer in bytes. All the commands in the CCB
 * must not cross this boundary.
 */
#define SCHED_ARC_CCB_STREAM_BUFF_SIZE		256

/**<
 * Virtual SOB index used in virtual SOB array
 */
enum sched_cmpt_sync_scheme_bitmap {
	VIRTUAL_SOB_INDEX_TPC = 0,
	VIRTUAL_SOB_INDEX_MME = 1,
	VIRTUAL_SOB_INDEX_MME_XPOSE = 2,
	VIRTUAL_SOB_INDEX_ROT = 3,
	VIRTUAL_SOB_INDEX_DEBUG = 4,
	SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT = 5
};

/**<
 * Sync scheme Monitor count per engine for compute
 */
#define SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT		4

/**<
 * Long DBG SOB Monitor count per engine, needs a long monitor
 */
#define SCHED_CMPT_ENG_SYNC_SCHEME_DBG_MON_COUNT	4

/**<
 * B2B Monitor count per engine
 */
#define SCHED_CMPT_ENG_B2B_MON_COUNT			1

/**<
 * Max number of recipes that can be updated in batch
 * update recipe cmd
 */
#define ARC_CMD_UPDATE_RECIPE_MAX_RECIPES 4

/**
 * \enum    arc_regions_t
 * \brief   ARC address map
 * \details Address map of scheduler as well as engine ARC.
 *          Each region is of size 256MB. Driver programs the extension
 *          registers.
 */
enum arc_regions_t {
	ARC_REGION0_UNSED  = 0,
	/*
	 * Extension registers
	 * None
	 */
	ARC_REGION1_SRAM = 1,
	/*
	 * Extension registers
	 * AUX_SRAM_LSB_ADDR
	 * AUX_SRAM_MSB_ADDR
	 * ARC Address: 0x1000_0000
	 */
	ARC_REGION2_CFG = 2,
	/*
	 * Extension registers
	 * AUX_CFG_LSB_ADDR
	 * AUX_CFG_MSB_ADDR
	 * ARC Address: 0x2000_0000
	 */
	ARC_REGION3_GENERAL = 3,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_0
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_0
	 * ARC Address: 0x3000_0000
	 */
	ARC_REGION4_HBM0_FW = 4,
	/*
	 * Extension registers
	 * AUX_HBM0_LSB_ADDR
	 * AUX_HBM0_MSB_ADDR
	 * AUX_HBM0_OFFSET
	 * ARC Address: 0x4000_0000
	 */
	ARC_REGION5_HBM1_GC_DATA = 5,
	/*
	 * Extension registers
	 * AUX_HBM1_LSB_ADDR
	 * AUX_HBM1_MSB_ADDR
	 * AUX_HBM1_OFFSET
	 * ARC Address: 0x5000_0000
	 */
	ARC_REGION6_HBM2_GC_DATA = 6,
	/*
	 * Extension registers
	 * AUX_HBM2_LSB_ADDR
	 * AUX_HBM2_MSB_ADDR
	 * AUX_HBM2_OFFSET
	 * ARC Address: 0x6000_0000
	 */
	ARC_REGION7_HBM3_GC_DATA = 7,
	/*
	 * Extension registers
	 * AUX_HBM3_LSB_ADDR
	 * AUX_HBM3_MSB_ADDR
	 * AUX_HBM3_OFFSET
	 * ARC Address: 0x7000_0000
	 */
	ARC_REGION8_DCCM = 8,
	/*
	 * Extension registers
	 * None
	 * ARC Address: 0x8000_0000
	 */
	ARC_REGION9_PCIE = 9,
	/*
	 * Extension registers
	 * AUX_PCIE_LSB_ADDR
	 * AUX_PCIE_MSB_ADDR
	 * ARC Address: 0x9000_0000
	 */
	ARC_REGION10_GENERAL = 10,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_1
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_1
	 * ARC Address: 0xA000_0000
	 */
	ARC_REGION11_GENERAL = 11,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_2
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_2
	 * ARC Address: 0xB000_0000
	 */
	ARC_REGION12_GENERAL = 12,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_3
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_3
	 * ARC Address: 0xC000_0000
	 */
	ARC_REGION13_GENERAL = 13,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_4
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_4
	 * ARC Address: 0xD000_0000
	 */
	ARC_REGION14_GENERAL = 14,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_5
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_5
	 * ARC Address: 0xE000_0000
	 */
	ARC_REGION15_LBU = 15
	/*
	 * Extension registers
	 * None
	 * ARC Address: 0xF000_0000
	 */
};

#endif /* __GAUDI3_ARC_COMMON_PACKETS_H__ */

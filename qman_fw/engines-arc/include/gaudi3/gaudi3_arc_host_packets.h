/* SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2020 HabanaLabs Ltd.
 * All Rights Reserved.
 */

#ifndef __GAUDI3_ARC_HOST_PACKETS_H__
#define __GAUDI3_ARC_HOST_PACKETS_H__

#include <stdint.h>
#include "gaudi3/asic_reg/gaudi3_blocks.h"

#include "gaudi3_arc_common_packets.h"

/**
 * \file    host_if.h
 * \brief   Interface file for Host.
 *          This defines data structures to be shared by the Host (LKD)
 *          and QMAN Scheduler ARC.
 */

/**
 * \struct  ccb_ctxt_t
 * \brief   Cyclic Command Buffer Context
 * \details CCB Context contains information that can be used
 *	    during debug
 */
struct ccb_ctxt_t {
	uint32_t dccm_offset;
	/**<
	 * DCCM CCB Buffer address offset
	 * Data from CCB Buffer is copied by firmware using ARC DMAs
	 * Size of the buffer is SCHED_ARC_CCB_STREAM_BUFF_SIZE * 2
	 */
	uint32_t dccm_ci;
	/**<
	 * Updated by Firmware when it completes processing of every
	 * scheduler command
	 * Range: 0 to (SCHED_ARC_CCB_STREAM_BUFF_SIZE * 2 - 1)
	 */
	uint32_t dccm_pi;
	/**<
	 * Updated by Firmware when it completes DMA from Host CCB
	 * buffer into DCCM Buffer
	 * Range: 0 to (SCHED_ARC_CCB_STREAM_BUFF_SIZE * 2 - 1)
	 */
} __attribute__ ((__packed__));

/**
 * \struct  ccb_t
 * \brief   Cyclic Command Buffer (CCB)
 * \details Data structure for Cyclic Command Buffer
 */
struct ccb_t {
	uint32_t ccb_addr;
	/**<
	 * 32 bit address of command buffer, must be 128Byte aligned
	 * This is an offset with respect to the Address Extension register.
	 * User must program the address extension register as well.
	 */
	uint32_t ccb_size;
	/**<
	 * Size of cyclic buffer in bytes
	 */
	uint32_t ccb_ctxt_dccm_offset;
	/**<
	 * CCB Context DCCM Offset
	 * CCB Context is located at this offset from DCCM Base address
	 * for each engine
	 */
#if (defined(USE_FW_ACP) || defined(TARGET_ENV_SIM))
	uint32_t pi;
	uint32_t ci;
#endif
} __attribute__ ((__packed__));

/**
 * \enum
 * \brief   Firmware configuration blob version
 * \details Firmware configuration blob version
 */
enum {
	ARC_FW_INIT_CONFIG_VER = 0x1A
};

/**
 * \struct  engine_resp_sob_set_config_t
 * \brief   Firmware configuration
 * \details Each instance of the firmware running on a particular engine can be
 *	    configured differently. This structure provides configuration
 *	    parameters for each scheduler instance and engine instance.
 */
struct engine_resp_sob_set_config_t {
	uint32_t sob_start_id;
	/**<
	 * SOB start ID, starting from here 2 SOBs are used
	 * SOBs would be used by firmware for implementing flow control
	 * between engine and scheduler for a given queue
	 */
};

/**
 * \enum    arc_fw_recipe_indices_t
 * \brief   Various recipes supported by Firmware
 * \details Various recipes supported by Firmware
 */
enum arc_fw_recipe_indices_t {
	ARC_FW_IDX_PATCHING_ADDR_BASE = 0x0,
	ARC_FW_IDX_EXECUTE_ADDR_BASE = 0x1,
	ARC_FW_IDX_DYNAMIC_ADDR_BASE = 0x2,
	ARC_FW_MAX_RECIPES = 0x8
};

#define ARC_FW_SYNAPSE_CONFIG_SIZE	16

/**
 * \enum    arc_fw_sync_scheme_t
 * \brief   Various sync scheme supported by Firmware
 * \details Various sync scheme supported by Firmware
 */
enum arc_fw_sync_scheme_t {
	ARC_FW_LEGACY_SYNC_SCHEME = 0x0,
	ARC_FW_GAUDI2_SYNC_SCHEME = 0x1,
	ARC_FW_GAUDI3_SYNC_SCHEME = 0x2,
	ARC_FW_COUNT = 0x3
};

/**
 * \struct  arc_fw_synapse_config_t
 * \brief   Synapse parameters for engine configuration
 * \details synapse parameters for a engine instance
 */
struct arc_fw_synapse_config_t {
	union {
		struct {
			uint32_t sync_scheme_mode;
			/**<
			 * Refer to arc_fw_sync_scheme_t
			 */
			uint32_t cme_enable;
			/**<
			 * This indicates whether runtime uses cme
			 * or not.
			 */
		};
		uint32_t synapse_params[ARC_FW_SYNAPSE_CONFIG_SIZE];
	};
};

#define DCCM_QUEUE_COUNT	5

/**
 * \struct  cme_config_t
 * \brief   CME Configuration Parameters
 * \details CME Parameters supplied by SCAL to CME
 */
struct cme_config_t {
	uint32_t version;
	/**<
	 * version of this configuration schema
	 * Firmware checks this against the version supported by it
	 */
	uint32_t engine_count_in_asic[SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT];
	/**<
	 * Total number of engines in ASIC.
	 */
	uint32_t engine_count_in_hdcore[SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT];
	/**<
	 * Total number of engines in HDCore.
	 */
	uint32_t engine_count_in_dcore[SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT];
	/**<
	 * Total number of engines in DCore.
	 */
	uint32_t num_tpc_signals;
	/**<
	 * Total number of TPC signals that are sent to the central
	 * SOB at the end of wd fence and execute command. This depends on
	 * number of HDCores with one or more TPCs are enabled in the SoC.
	 * Each HDCore sends one completion signal if it has at least one TPC.
	 * The HDCores with no TPCs dont send any signals to Central SOBs.
	 */
	uint32_t sync_scheme_mon_id;
	/**<
	 * Monitor that is used by CME Firmware to implement SO Set Reset
	 * and MCID Rollover functionality within the ECB List
	 */
	uint32_t mon_start_id;
	/**<
	 * Monitor start ID
	 */
	uint32_t mon_count;
	/**<
	 * Number of monitors that can be used by CMEF for Engine dependencies
	 * Count should be even number as one pair of monitor is required for
	 * each engine
	 */
	uint32_t dup_tpc_fence0_lbw_addr;
	/**<
	 * DUP address that CME should use to update Fence0 counter of
	 * TPC engines
	 * Using 4 groups of 16 each for 64 TPCs
	 */
	uint32_t dup_tpc_aux_cid_offset0_lbw_addr;
	/**<
	 * DUP address that CME should use to update Fence0 counter of
	 * TPC engines (ARC0_AUX.CID_OFFSET_0)
	 * Using 4 groups of 16 each for 64 TPCs
	 */
	uint32_t dup_mme_fence0_lbw_addr;
	/**<
	 * DUP address that CME should use to update Fence0 counter of
	 * all the MME engines. In some scenarios MME may not
	 * be involved in the graph processing thats why we need a separate
	 * entry to signal only MMEs conditionally.
	 */
	uint32_t dup_rot_fence0_lbw_addr;
	/**<
	 * DUP address that CME should use to update Fence0 counter of
	 * all the Rotator engines. In some scenarios rotators may not
	 * be involved in the graph processing thats why we need a separate
	 * entry to signal only rotators conditionally.
	 */
	uint32_t dup_cs_config_lbw_addr;
	/**<
	 * DUP address that CME should use to program CS entries
	 * This is also known as Payload0
	 */
	uint32_t dup_cme_dccmq_lbw_addr;
	/**<
	 * DUP address that CME should use to get the monitor expiration
	 * message back to CME DCCM queue
	 * This is also known as Payload1
	 */
	uint32_t cme_poll_main_status;
	/**<
	 * This is boolean flag that needs to be set when not all the TPCs are present
	 * in some debug or test coral / pldm images. This informs CMEF to poll the
	 * CS Controller's MAINT_STATUS.NUM_ONGOING register on behalf of TPCs.
	 */
	uint32_t dccm_queue_count;
	/**<
	 * indicates how many entries are valid in the following eng_resp_config array
	 */
	struct engine_resp_sob_set_config_t eng_resp_config[DCCM_QUEUE_COUNT];
	/**<
	 * SOB configuration related parameters for each
	 * DCCM queue. These are used by firmware for flow control.
	 */
	uint32_t soset_pool_start_sob_id;
	/**<
	 * SOB ID of first SOB in the SO Set pool. There are SO_SET_COUNT
	 * SO set count and NUM_SOS_PER_SO_SET number of SOBs in each SO set.
	 * CME arms MONs against a few of these SOBs.
	 */
	uint32_t comp_sched_mon_resp_q_push_reg_aux;
	/**<
	 * dccm q push reg address of scheduler where so_set_reset message from CME should go.
	 */
	struct arc_fw_synapse_config_t synapse_params;
	/**<
	 * Synapse parameters for cme engine instance.
	 * This will be used for backward compatibility until synapse moves to
	 * cme enabled
	 */
};

struct engine_config_t {
	uint32_t version;
	/**<
	 * version of this configuration schema
	 * Firmware checks this against the version supported by it
	 */
	uint32_t soset_pool_start_sob_id;
	/**<
	 * SOB ID of first SOB in the SO Set tool. There are SO_SET_COUNT
	 * SO set count and NUM_SOS_PER_SO_SET number of SOBs in each SO set
	 */
	uint32_t local_soset_pool_start_sob_id;
	/**<
	 * Local SO SET Pool Start SOB ID for this engine (TPC Only)
	 * SOB ID of first SOB in the SO Set tool. There are SO_SET_COUNT
	 * SO set count and NUM_SOS_PER_LOCAL_SO_SET_IN_HDCORE number of SOBs
	 * valid in each HDCore
	 */
	uint32_t mon_start_id;
	/**<
	 * Monitor start ID, starting from here SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT
	 * monitors would be used by engine firmware. This is used in the
	 * arm monitor command in the compute ecb list
	 */
	uint32_t soset_dbg_mon_start_id;
	/**<
	 * Monitor start ID used for Topology debug SOB, Starting from here
	 * 4 Mons are used by the engine (used as long monitor)
	 */
	uint32_t b2b_mon_id;
	/**<
	 * Monitor ID that should be used by this engine for back to back
	 * execution
	 */
	uint32_t tpc_nop_kernel_addr_lo;
	/**<
	 * Lower 32 bit address of TPC NOP kernel
	 */
	uint32_t tpc_nop_kernel_addr_hi;
	/**<
	 * Upper 32 bit address of TPC NOP kernel
	 */
	uint32_t dccm_queue_count;
	/**<
	 * indicates how many entries are valid in the eng_resp_config array
	 */
	struct engine_resp_sob_set_config_t eng_resp_config[DCCM_QUEUE_COUNT];
	/**<
	 * SOB configuration related parameters for each
	 * DCCM queue. These are used by firmware for flow control.
	 */
	uint32_t engine_count_in_asic[SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT];
	/**<
	 * Total number of engines in ASIC.
	 */
	uint32_t engine_count_in_hdcore[SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT];
	/**<
	 * Total number of engines in HDCore.
	 */
	uint32_t engine_count_in_dcore[SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT];
	/**<
	 * Total number of engines in DCore.
	 */
	uint32_t engine_index;
	/**<
	 * Local Engine index
	 * Note: It starts from 0 to one less than number of engines in a
	 * group
	 */
	uint32_t engine_index_in_hdcore;
	/**<
	 * Engine index within HDCore
	 * Note: It starts from 0 to one less than number of engines in a
	 * particular HDCore where this engine is located. Except for TPC
	 * this value is 0 because there is only one engine in HDCore
	 */
	uint32_t engine_index_in_dcore;
	/**<
	 * Engine index within HDCore
	 * Note: It starts from 0 to one less than number of engines in a
	 * particular HDCore where this engine is located. Except for TPC
	 * this value is 0 because there is only one engine in HDCore
	 */
	uint32_t num_engines_in_group;
	/**<
	 * Total number of engines in the group, sharing the work.
	 * For example, this is the number of EDMA Engines doing a shared
	 * memcopy during reproducible reduction.
	 */
	uint32_t cmpt_csg_sob_id_base;
	/**<
	 * SOB Start ID for compute completion groups
	 * This is used for configuring the monitor which is used for
	 * serializing compute workload
	 */
	uint32_t cmpt_csg_sob_id_count;
	/**<
	 * Total SOB count used in all compute completion groups
	 */
	uint32_t sfg_sob_base_id;
	/**<
	 * SOB ID Start/Base ID, compute Stream 0 signals this SOB
	 */
	uint32_t sfg_sob_per_stream;
	/**<
	 * Number of SOBs used by each stream. This field is used by
	 * Firmware to calculate the SOB ID that should be signalled
	 */
	uint32_t barrier_local_sob_id_base;
	/**<
	 * This field is applicable only for TPC engines.
	 * SOB(s) to be used for signaling barrier completion within HDCore
	 * Starting from base 8 SOBs are used. TPC0 signals base + 0,
	 * TPC1 signals base + 1 and so on.
	 */
	uint32_t barrier_local_mon_id_base;
	/**<
	 * chain of monitors used for signaling to Barrier SOB of the
	 * completion group
	 */
	uint32_t edup_sync_scheme_lbw_addr;
	/**<
	 * The LBW LO address to be programmed in the monitors
	 * used for sync scheme to update Fence using EDUP
	 * Note: Valid only for TPC Engines, for the rest the Fence
	 * is updated directly.
	 */
	uint32_t edup_b2b_lbw_addr;
	/**<
	 * The LBW LO address to be programmed in the monitors
	 * used for back to back execution update Fence using EDUP
	 * Note: Valid only for TPC Engines, for the rest the Fence
	 * is updated directly.
	 */
	uint32_t num_tpc_signals;
	/**<
	 * Total number of TPC signals that are sent to the central
	 * SOB at the end of wd fence and execute command. This depends on
	 * number of HDCores with one or more TPCs are enabled in the SoC.
	 * Each HDCore sends one completion signal if it has at least one TPC.
	 * The HDCores with no TPCs dont send any signals to Central SOBs.
	 */
	uint32_t cme_poll_main_status;
	/**<
	 * This is boolean flag that needs to be set when not all the TPCs are present
	 * in some debug or test coral / pldm images. This informs CMEF to poll the
	 * CS Controller's MAINT_STATUS.NUM_ONGOING register on behalf of TPCs.
	 */
	uint32_t addr_lbu_cme_cid_offs_0;
	/**<
	 * This is address of CID_OFFSET_0 of CME. TPCs need to know this address to
	 * assist in polling of CS Status.
	 */
	uint32_t psoc_arc_intr_addr;
	/**<
	 * LBU address of the PSOC ARC scratchpad register which is written
	 * by engine ARCs to generate an interrupt to LKD
	 */
	struct arc_fw_synapse_config_t synapse_params;
	/**<
	 * Synapse parameters for engine instance
	 */
};

#define COMP_SYNC_GROUP_CMAX_TARGET		0x4000

struct comp_sync_group_config_t {
	uint32_t sob_start_id;
	/**<
	 * SOB start ID for this completion group
	 * Number of SOBs is sob_count
	 */
	uint32_t sob_count;
	/**<
	 * Number of SOBs used by this completion group
	 */
	uint32_t credit_sync_sob_id_base;
	/**<
	 * SOB IDs that needs to be updated by the Slaves when half of the
	 * SOBs from sob_count are consumed. Firmware uses two SOBs from
	 * this base in ping/pong manner
	 */
	uint32_t mon_count;
	/**<
	 * Number of monitors per monitor group. SCAL uses a group of
	 * monitors, which all fires together to update various
	 * counters at the end of completion. This variable tells how
	 * many monitors are there in the monitor group.
	 */
	uint32_t mon_group_count;
	/**<
	 * Number of monitor groups used for this completion group
	 * Firmware does not need monitor IDs, it only needs to know the
	 * count to determine the start index in the sob list to arm the SOB
	 * when a monitor expires.
	 */
	uint32_t slave_comp_group;
	/**<
	 * 0 - Master Completion Group
	 * 1 - Slave Completion Group
	 *
	 * Scheduler uses this flag to decide if should re-arm the
	 * monitor or not which is related to completion message
	 * Scheduler re-arms the monitor only for Master Completion
	 * Groups
	 */
	uint32_t in_order_completion;
	/**<
	 * 0 - Out of order completions
	 * 1 - In order completions
	 *
	 * Scheduler uses this flag to make sure the completions
	 * sent on this completion groups are either in order or out
	 * of order based on the flag value
	 */
	uint32_t in_order_monitor_offset;
	/**<
	 * Scheduler uses this offset to pick a monitor for ensuring in order
	 * completion by adding in_order_monitor_offset to the monitor ID in
	 * completion message.
	 */
	uint32_t num_slaves;
	/**<
	 * Number of Slaves which are dependent on this master
	 * applicable only if slave_comp_group = 0
	 */
	uint32_t watch_dog_sob_addr;
	/**<
	 * LBU address of SOB ID to be incremented before dispatching the
	 * Barrier command to engine
	 */
};

#define SO_SET_COUNT				16

/*
 * Number of SOBs per SoSet based on Gaudi2 sync scheme
 */
#define NUM_SOS_PER_SO_SET			128

/*
 * Number of SOBs per SoSet based on Gaudi3 sync scheme
 * Note: later when we remove Gaudi2 related sync scheme
 * we can reduce the NUM_SOS_PER_CENTRAL_SO_SET to Gaudi3
 * related sync scheme (40)
 */
#define NUM_SOS_PER_CENTRAL_SO_SET		128
#define NUM_SOS_PER_LOCAL_SO_SET		64
#define NUM_SOS_PER_LOCAL_SO_SET_IN_HDCORE	8

#define SOB_OFFSET_MME_IN_SO_SET		0
#define SOB_OFFSET_TPC_IN_SO_SET		16
#define SOB_OFFSET_MME_XPOSE_IN_SO_SET		80
#define SOB_OFFSET_ROT_IN_SO_SET		96
#define SOB_OFFSET_DEBUG_IN_SO_SET		112
#define SOB_OFFSET_SEMAPHORE_SO_SET	120

#define SOB_COUNT_MME_IN_SO_SET			16
#define SOB_COUNT_TPC_IN_SO_SET			64
#define SOB_COUNT_MME_XPOSE_IN_SO_SET		16
#define SOB_COUNT_ROT_IN_SO_SET			16
#define SOB_COUNT_DEBUG_IN_SO_SET		8
#define SOB_COUNT_SEMAPHORE_SO_SET		8

struct so_set_config_t {
	uint32_t sob_start_id;
	/**<
	 * SOB start ID for Central SO set
	 * Number of SOBs is NUM_SOS_PER_SO_SET while using
	 * Gaudi2 specific Sync scheme, else it is NUM_SOS_PER_CENTRAL_SO_SET
	 */
	uint32_t mon_id;
	/**<
	 * Monitor ID to be used by this SO Set
	 */
};

/**
 * \struct  mcid_set_config_t
 * \brief   MCID Set Configuration
 * \details MCID Set Parameters supplied by SCAL to scheduler Firmware
 */
struct mcid_set_config_t {
	uint32_t mcid_start_id;
	/**<
	 * Starting from here firmware starts allocation of MCIDs for
	 * Degrade and Discard MCID ranges. First NUM_DEGRADE_MCID many MCIDs
	 * are for degrade operation followed by NUM_DISCARD_MCID for discard operation.
	 */
	uint32_t discard_mcid_count;
	/**<
	 * Number of Discard MCIDs
	 */
	uint32_t degrade_mcid_count;
	/**<
	 * Number of Degrade MCIDs
	 */
};

/*
 * Configuration parameters for each stream in scheduler
 */
#define FENCE_MON_COUNT_PER_STREAM		8
#define FENCE_MON_LONG_COUNT_PER_STREAM		4

/**
 * \struct  sched_streams_config_t
 * \brief   Scheduler streams related configuration
 * \details Configuration parameters related to a streams running on a
 *	    scheduler instance
 */
struct sched_streams_config_t {

};

/**
 * \struct  sched_engine_group_config_t
 * \brief   Scheduler Engine groups related configuration
 * \details Configuration parameters related to a engine groups controlled
 *	    by scheduler instance
 */
struct sched_engine_group_config_t {
	uint32_t engine_count_in_group;
	/**<
	 * Number of engines in this engine group
	 * value of 0 means the group is not valid and rest of the group
	 * specific parameters are ignored
	 */
	uint32_t eng_resp_sob_start_id;
	/**<
	 * SOB start ID, starting from here 2
	 * SOBs would be used by firmware for this DCCM queue
	 */
	uint32_t eng_resp_mon_id;
	/**<
	 * Monitor ID to be used by firmware for this DCCM queue
	 */
	uint32_t dup_trans_data_q_addr;
	/**<
	 * LBU address of the register DUP_TRANS_DATA_Q_x which should be
	 * used by the firmware to push packet to engines
	 * Note: This should be LBU address and not LBW
	 */
	uint32_t nic_db_fifo_size;
	/**<
	 * Size of DB FIFO (in bytes) which is used by NICs
	 * This field is used only for Scaleup and Scaleout schedulers
	 * and ignored for other schedulers
	 */
	uint32_t nic_db_fifo_threshold;
	/**<
	 * DB FIFO threshold (in bytes) after which NIC engines
	 * increment the SOB.
	 * This field is used only for Scaleup and Scaleout schedulers
	 * and ignored for other schedulers
	 */
	uint32_t dup_mask_addr;
	/**<
	 * LBU address of the register DUP_XXX_ENG_MASK where XXX is MME, TPC
	 * NIC, EDMA etc. In some pass through commands for NIC mask register
	 * is programmed by FW but for the rest this field is dont care.
	 */
	uint32_t sec_dup_trans_data_q_addr;
	/**<
	 * LBU address of the register secondary DUP_TRANS_DATA_Q_x which
	 * should be used by the firmware to push packet to engines
	 * Note: This should be LBU address and not LBW
	 */
	uint32_t sec_dup_mask_addr;
	/**<
	 * LBU address of the register secondary DUP_XXX_ENG_MASK where XXX is MME,
	 * TPC, NIC, EDMA etc. In some pass through commands for NIC mask register
	 * is programmed by FW but for the rest this field is dont care.
	 */
};

/**
 * \enum    pdma_ch_direction_t
 * \brief   Directions supported by a PDMA Channel
 * \details Directions supported by a PDMA Channel
 */
enum pdma_ch_direction_t {
	PDMA_CH_DIR_DEV_TO_HOST = 0x0,
	PDMA_CH_DIR_HOST_TO_DEV = 0x1,
	PDMA_CH_DIR_COUNT = 2
};
/**
 * \struct  sched_pdma_ch_cfg_t
 * \brief   PDMA channel related configuration
 * \details Configuration parameters related to PDMA channel
 */
struct sched_pdma_ch_cfg_t {
	uint32_t channel_number;
	/**<
	 * PDMA Channel number between
	 * PDMA_DIE0_CH0 and PDMA_DIE1_CH11
	 */
	uint32_t direction;
	/**<
	 * Refer to pdma_ch_direction_t
	 */
	uint32_t pdma_engine_group_id;
	/**<
	 * Engine Group Id of PDMA TX (CMD/DATA) or
	 * PDMA_RX. This value is same as what is defined in
	 * scal.h
	 */
	struct {
		union {
			struct {
				uint32_t desc_buffer_addr_lo;
				uint32_t desc_buffer_addr_hi;
			};
			uint64_t desc_buffer_addr;
		} __attribute__ ((aligned(4), __packed__));
	};
	/**<
	 * PDMA Descriptor buffer Device address, Typically allocated
	 * in HBM memory
	 * Note: This should be mapped to ARC address space
	 */
	uint32_t desc_buffer_size;
	/**<
	 * PDMA Descriptor buffer size
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * Configuration parameters for each stream in scheduler
 */
#define SCHED_ARC_MAX_PDMA_CHANNEL_COUNT	5

/**
 * \struct  sched_pdma_config_t
 * \brief   PDMA related configuration
 * \details Configuration parameters related to PDMA
 */
struct sched_pdma_config_t {
	struct sched_pdma_ch_cfg_t pdma_ch_cfg[SCHED_ARC_MAX_PDMA_CHANNEL_COUNT];
	uint32_t valid_pdma_ch_count;
	/**<
	 * Index mapping for each PDMA Group type
	 * 0 - SCAL_PDMA_TX_CMD_GROUP
	 * 1 - SCAL_PDMA_TX_DATA_GROUP
	 * 2 - SCAL_PDMA_RX_GROUP
	 * 3 - SCAL_PDMA_DEV2DEV_DEBUG_GROUP
	 * 4 - SCAL_PDMA_RX_DEBUG_GROUP
	 */
} __attribute__ ((aligned(4), __packed__));

enum {
	COMP_SYNC_GROUP_COUNT = 16,
	COMP_SYNC_GROUP_MAX_MON_GROUP_COUNT = 4
};
/**
 * \struct  scheduler_config_t
 * \brief   Scheduler configuration
 * \details Configuration parameters related to a scheduler instance
 */
struct scheduler_config_t {
	uint32_t version;
	/**<
	 * version of this configuration schema
	 * Firmware checks this against the version supported by it
	 */
	struct comp_sync_group_config_t csg_config[COMP_SYNC_GROUP_COUNT];
	/**<
	 * Completion group configuration
	 */
	uint32_t so_sets_count;
	/**<
	 * Valid SO sets configuration, between 0..SO_SET_COUNT
	 */
	struct so_set_config_t so_set_config[SO_SET_COUNT];
	/**<
	 * SO set configuration
	 */
	struct mcid_set_config_t mcid_set_config;
	/**<
	 * MCID related configuration
	 */
	struct sched_engine_group_config_t eng_grp_cfg[QMAN_ENGINE_GROUP_TYPE_COUNT];
	/**<
	 * Engine group related configuration
	 */
	struct sched_pdma_config_t pdma_cfg;
	/**<
	 * PDMA Configuration
	 */
	uint32_t enable_auto_fetcher;
	/**<
	 * Auto fetcher enabled or not
	 */
	uint32_t cme_poll_main_status;
	/**<
	 * This is boolean flag that needs to be set when not all the TPCs are present
	 * in some debug or test coral / pldm images. This informs CMEF to poll the
	 * CS Controller's MAINT_STATUS.NUM_ONGOING register on behalf of TPCs.
	 */
	uint32_t psoc_arc_intr_addr;
	/**<
	 * LBU address of the PSOC ARC scratchpad register which is written
	 * by scheduler ARCs to generate an interrupt to LKD
	 */
	struct arc_fw_synapse_config_t synapse_params;
	/**<
	 * Synapse parameters for scheduler instance
	 */
};

enum {
	/**
	 * Firmware interface registers between Scheduler Firmware and Host
	 */
	SCHED_FENCE_LBU_ADDR_SCRATCHPAD_ID = 0,
	/**<
	 * SCAL writes LBU address of the Fence register which it would
	 * like Firmware to update
	 */
	SCHED_FW_CONFIG_ADDR_SCRATCHPAD_ID = 1,
	/**<
	 * SCAL writes ARC address of the Configuration which should be used
	 * by firmware for initialization
	 */
	SCHED_FW_CONFIG_SIZE_SCRATCHPAD_ID = 2,
	/**<
	 * SCAL writes size in bytes of the Configuration which should be used
	 * by firmware for initialization
	 */

	 SCHED_MAX_CHECKPOINT_COUNT = 256,
};
/**
 * \struct  sched_registers_t
 * \brief   Scheduler Registers
 * \details Various registers exposed by Scheduler ARC
 *          Registers start at the base address of DCCM
 */
struct sched_registers_t {
	struct ccb_t ccb[SCHED_ARC_MAX_STREAMS];
	/**< Registers related to Primary Queues */
	uint32_t canary;
	/**<
	 * SCAL writes SCAL_INIT_COMPLETED into this register to inform
	 * firmware that it has completed the initilization and firmware can
	 * now perform Post Init.
	 * not written by firmware at all during runtime.
	 */
	uint32_t heartbeat;
	/**<
	 * 32bit free running counter, which is incremented
	 * by scheduler ARC periodically
	 */
	uint32_t sched_type;
	/**<
	 * Applications can read this register to know what type
	 * of scheduler firmware is loaded on this scheduler ARC
	 * Look at enum scheduler_type_t for scheduler types
	 */
	uint32_t last_so_set_alloc_start_id;
	/**<
	 * This register is updated only by compute stream,
	 * It contains the SOB Start ID that was last allocated by
	 * the scheduler when SO_SET_ALLOC command is processed.
	 */
	uint32_t assert_code;
	/*
	 * This register is updated in case of asserttion
	 * assert codes are unique to identify various errors
	 */
	uint32_t exception_code;
	/*
	 * This register is updated in case of exception in arc
	 * It has the contents of ECR(exception cause register)
	 * In this register:
	 * Bits 0:7 - Exception Parameter
	 * Bits 8:15 - Exception Cause Code
	 * Bits 16:23 - Exception Vector Number
	 * Please check ARCv2 ISA Programmer's reference manual for details
	 * This register's offset is used as hardcode in reset vector assembly
	 */
	uint32_t checkpoint_index;
	uint32_t checkpoint[SCHED_MAX_CHECKPOINT_COUNT];
};

/**
 * Firmware interface registers between Engine Firmware and Host
 */
enum {
	ENG_FENCE_LBU_ADDR_SCRATCHPAD_ID = 0,
	/**<
	 * SCAL writes LBU address of the Fence register which it would
	 * like Firmware to update
	 */
	ENG_FW_CONFIG_ADDR_SCRATCHPAD_ID = 1,
	/**<
	 * SCAL writes ARC address of the Configuration which should be used
	 * by firmware for initialization
	 */
	ENG_FW_CONFIG_SIZE_SCRATCHPAD_ID = 2
	/**<
	 * SCAL writes size in bytes of the Configuration which should be used
	 * by firmware for initialization
	 */
};

enum {
	ENG_MAX_CHECKPOINT_COUNT = 256
};

/**
 * \struct  engine_arc_reg_t
 * \brief   Engine Registers
 * \details Registers of engine ARC
 *          Registers start at the base address of DCCM
 */
struct engine_arc_reg_t {
	uint32_t canary;
	/**<
	 * SCAL writes SCAL_INIT_COMPLETED into this register to inform
	 * firmware that it has completed the initilization and firmware can
	 * now perform Post Init.
	 * not written by firmware at all during runtime.
	 */
	uint32_t heartbeat;
	/**<
	 * 32bit free running counter, which is incremented
	 * by engine ARC periodically
	 */
	uint32_t eng_type;
	/**<
	 * Applications can read this register to know what type
	 * of engine firmware is loaded on this Engine ARC
	 * Look at enum engine_type_t for engine types
	 */
	uint32_t curr_so_set_start_id;
	/**<
	 * This register is applicable only for compute engines.
	 * It contains the SOB Start ID of the current SO set that
	 * is/was used to process the current/last ECB List. The value
	 * in this register is updated only when Engine receives SO SET ALLOC
	 * command.
	 */
	uint32_t local_so_set_start_id;
	/**<
	 * This register is applicable only for compute engines.
	 * It contains the SOB Start ID of the current local SO set that
	 * is/was used to process the current/last ECB List. The value
	 * in this register is updated only when Engine receives SO SET ALLOC
	 * command.
	 */
	uint32_t assert_code;
	/*
	 * This register is updated in case of assert
	 * assert codes are unique to identify various errors
	 */
	uint32_t exception_code;
	/*
	 * This register is updated in case of exception in arc
	 * It has the contents of ECR(exception cause register)
	 * In this register:
	 * Bits 0:7 - Exception Parameter
	 * Bits 8:15 - Exception Cause Code
	 * Bits 16:23 - Exception Vector Number
	 * Please check ARCv2 ISA Programmer's reference manual for details
	 * This register's offset is used as hardcode in reset vector assembly
	 */
	uint32_t debug_regs_off;
	/*
	 * offset of engine_debug_regs_t structure in DCCM.
	 * zero, if does not exist (compiled with ENABLE_ARC_DEBUG_REGS undefined)
	 */
	uint32_t gc_ctxt_addr_offset;
	/*
	 * offset of tpc_wd_ctxts_t/mme_wd_ctxts_t/edma_wd_ctxts_t/rot_wd_ctxts_t
	 * from the dccm base
	 */
	uint32_t checkpoint_index;
	uint32_t checkpoint[ENG_MAX_CHECKPOINT_COUNT];
};

/**
 * \struct  cme_dbg_info_t
 * \brief   CME Debug Information Registers
 * \details Debug Info of cme
 *          Registers start at a fixed offset of DCCM
 */
struct cme_dbg_info_t {
	uint32_t total_num_cme_cmds_processed;
	/**<
	 * cumulative number of CMDS processed
	 */
	uint32_t num_discard_cls_direct;
	/**<
	 * number of discard commands processed directly without arming mons
	 */
	uint32_t num_discard_cls_mon_fwork;
	/**<
	 * number of discard commands processed using arming mons
	 */
	uint32_t num_degrade_cls_direct;
	/**<
	 * number of degrade commands processed directly without arming mons
	 */
	uint32_t num_degrade_cls_mon_fwork;
	/**<
	 * number of discard commands processed using arming mons
	 */
	uint32_t num_mons_avail;
	/**<
	 * number of MON PAIRs currently available
	 */
	uint32_t num_dbg_mons_avail;
	/**<
	 * number of DBG MON PAIRs currently available (used for Profiler Notifs)
	 */
	uint32_t total_num_mon_pairs_armed;
	/**<
	 * cumulative number of MON PAIRs armed
	 */
	uint32_t total_num_mon_pairs_returned;
	/**<
	 * cumulative number of MON PAIRs fired back, indicating completion
	 */
	uint32_t total_num_mcid_rollover_ecb_cmds;
	/**<
	 * total number of MCID Rollover ECB Commands processed
	 */
};

/**
 * \struct  cme_registers_t
 * \brief   CME Registers
 * \details Registers of cme
 *          Registers start at the base address of DCCM
 */
struct cme_registers_t {
	uint32_t canary;
	/**<
	 * SCAL writes SCAL_INIT_COMPLETED into this register to inform
	 * firmware that it has completed the initilization and firmware can
	 * now perform Post Init.
	 * not written by firmware at all during runtime.
	 */
	uint32_t heartbeat;
	/**<
	 * 32bit free running counter, which is incremented
	 * by engine ARC periodically
	 */
	uint32_t eng_type;
	/**<
	 * Applications can read this register to know what type
	 * of engine firmware is loaded on this Engine ARC
	 * Look at enum engine_type_t for engine types
	 */
	uint32_t curr_so_set_start_id;
	/**<
	 * This register is applicable only for compute engines.
	 * It contains the SOB Start ID of the current SO set that
	 * is/was used to process the current/last ECB List. The value
	 * in this register is updated only when Engine receives SO SET ALLOC
	 * command.
	 */
	uint32_t local_so_set_start_id;
	/**<
	 * This register is applicable only for compute engines.
	 * It contains the SOB Start ID of the current local SO set that
	 * is/was used to process the current/last ECB List. The value
	 * in this register is updated only when Engine receives SO SET ALLOC
	 * command.
	 */
	uint32_t assert_code;
	/*
	 * This register is updated in case of assert
	 * assert codes are unique to identify various errors
	 */
	uint32_t exception_code;
	/*
	 * This register is updated in case of exception in arc
	 * It has the contents of ECR(exception cause register)
	 * In this register:
	 * Bits 0:7 - Exception Parameter
	 * Bits 8:15 - Exception Cause Code
	 * Bits 16:23 - Exception Vector Number
	 * Please check ARCv2 ISA Programmer's reference manual for details
	 * This register's offset is used as hardcode in reset vector assembly
	 */
	uint32_t cme_dbg_info_size;
	/*
	 * Size in bytes of the CME Debug Information
	 */
	uint32_t cme_dbg_info_offset;
	/*
	 * Offset to the struct cme_dbg_info in the DCCM region. This offset
	 * can be added to the base of DCCM, when memory is read. For example,
	 * if CME is running from ARC_FARM0_ARC1, the debuginfo dump command will
	 * be -
	 * readmem <0x300007FFE330000 + cme_dbg_info_offset> <cme_dbg_info_size>
	 */
	uint32_t checkpoint_index;
	uint32_t checkpoint[ENG_MAX_CHECKPOINT_COUNT];
};

#define ARC_FW_METADATA_OFFSET	0x0
/**<
 * This is the offset in the binary file
 * at which the metadata information is located.
 */

#define SCHED_FW_UUID_0	0x326a7af9
#define SCHED_FW_UUID_1	0x68b04180
#define SCHED_FW_UUID_2	0xa7e6fa9c
#define SCHED_FW_UUID_3	0xa83a9e05
/**<
 * UUID for Scheduler Bin:
 * 326a7af9-68b0-4180-a7e6-fa9ca83a9e05
 */

#define ENG_FW_UUID_0	0x8535bf20
#define ENG_FW_UUID_1	0x8e5640a3
#define ENG_FW_UUID_2	0xa4eaa098
#define ENG_FW_UUID_3	0x2c1b6df2
/**<
 * UUID for Engine Bin:
 * 8535bf20-8e56-40a3-a4ea-a0982c1b6df2
 */

/**
 * \struct  arc_fw_metadata_t
 * \brief   Firmware Metadata Information
 * \details ARC FW Binary related Metadata information
 *          like Major/Minor Versions, etc. This is located
 *          at the offset mentioned at ARC_FW_METADATA_OFFSET
 *          from the start of the engine.bin/scheduler.bin files.
 *          This is maintained separately for Scheduler and
 *          Engine ARC FW binaries
 */
struct arc_fw_metadata_t {
	uint32_t uuid[4];
	/**<
	 * This is the Unique identifier for the FW Binary,
	 * one each for Scheduler and Engine FW bins
	 */
	uint32_t specs_version;
	/**<
	 * specs version used during compilation
	 * i.e. ARC_FW_INIT_CONFIG_VER
	 */
	uint32_t major_version;
	/**<
	 * This is the Major version of the FW Binary,
	 * usually incremented when there is an interface change
	 */
	uint32_t minor_version;
	/**<
	 * This is the Minor version of the FW Binary,
	 * usually incremented for bug fix releases
	 */
	uint32_t hbm_section_size;
	/**<
	 * This is the size of HBM loadable section of the FW Binary,
	 * present in the binary file
	 */
	uint32_t hbm_section_offset;
	/**<
	 * This is the offset of HBM loadable section of the FW,
	 * relative to the start of the binary file
	 */
	uint32_t dccm_section_size;
	/**<
	 * This is the size of DCCM loadable section of the FW Binary,
	 * present in the binary file
	 */
	uint32_t dccm_section_offset;
	/**<
	 * This is the offset of DCCM loadable section of the FW,
	 * relative to the start of the binary file
	 */
};

#endif /* __GAUDI3_ARC_HOST_PACKETS_H__ */

/* SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2020 HabanaLabs Ltd.
 * All Rights Reserved.
 */

#ifndef __GAUDI3_ARC_SCHED_PACKETS_H__
#define __GAUDI3_ARC_SCHED_PACKETS_H__

#include <stdint.h>
#include "gaudi3_arc_eng_packets.h"
/**
 * \file    gaudi2_arc_sched_packets.h
 * \brief   QMAN Command definitions for Host.
 *          This defines data structures to be used by Host to submit commands
 */

/**
 * \enum    sched_compute_arc_cmd_opcode_t
 * \brief   commands opcodes for scheduler ARC
 * \details Opcodes processed by the scheduler ARC
 */
enum sched_compute_arc_cmd_opcode_t {
	SCHED_COMPUTE_ARC_CMD_FENCE_WAIT = 0,
	SCHED_COMPUTE_ARC_CMD_LBW_WRITE = 1,
	SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER = 3,
	SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE = 4,
	SCHED_COMPUTE_ARC_CMD_NOP = 5,
	SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2 = 6,
	SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3 = 7,
	SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER = 8,
	SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2 = 9,
	SCHED_COMPUTE_ARC_CMD_ACP_FENCE_WAIT = 10,
	SCHED_COMPUTE_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = 11,
	SCHED_COMPUTE_ARC_CMD_DISPATCH_CME_ECB_LIST = 12,
	SCHED_COMPUTE_ARC_CMD_LBW_READ = 13,			// TODO: Only in Debug
	SCHED_COMPUTE_ARC_CMD_MEM_FENCE = 14,			// TODO: Only in Debug
	SCHED_COMPUTE_ARC_CMD_COUNT = 15,
	SCHED_COMPUTE_ARC_CMD_SIZE = 0x1F
};
/**
 * \enum    sched_scaleup_send_arc_cmd_opcode_t
 * \brief   scaleup send sched commands opcodes
 * \details Opcodes processed by the scaleup send scheduler ARC
 */
enum sched_scaleup_send_arc_cmd_opcode_t {
	SCHED_SCALEUP_SEND_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEUP_SEND_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEUP_SEND_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEUP_SEND_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEUP_SEND_ARC_CMD_NOP = 4,
	SCHED_SCALEUP_SEND_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_SCALEUP_SEND_ARC_CMD_NIC_EDMA_OPS = 6,
	SCHED_SCALEUP_SEND_ARC_CMD_NIC_PASSTHROUGH_V2 = 7,
	SCHED_SCALEUP_SEND_ARC_CMD_ACP_FENCE_WAIT = 8,
	SCHED_SCALEUP_SEND_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = 9,
	SCHED_SCALEUP_SEND_ARC_CMD_LBW_READ = 10,		// TODO: Only in Debug
	SCHED_SCALEUP_SEND_ARC_CMD_MEM_FENCE = 11,		// TODO: Only in Debug
	SCHED_SCALEUP_SEND_ARC_CMD_COUNT = 12,
	SCHED_SCALEUP_SEND_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_scaleup_recv_arc_cmd_opcode_t
 * \brief   scaleup recv sched commands opcodes
 * \details Opcodes processed by the scaleup recv scheduler ARC
 */
enum sched_scaleup_recv_arc_cmd_opcode_t {
	SCHED_SCALEUP_RECV_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEUP_RECV_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEUP_RECV_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEUP_RECV_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEUP_RECV_ARC_CMD_NOP = 4,
	SCHED_SCALEUP_RECV_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_SCALEUP_RECV_ARC_CMD_NIC_PASSTHROUGH_V2 = 6,
	SCHED_SCALEUP_RECV_ARC_CMD_ACP_FENCE_WAIT = 7,
	SCHED_SCALEUP_RECV_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = 8,
	SCHED_SCALEUP_RECV_ARC_CMD_LBW_READ = 9,		// TODO: Only in Debug
	SCHED_SCALEUP_RECV_ARC_CMD_MEM_FENCE = 10,		// TODO: Only in Debug
	SCHED_SCALEUP_RECV_ARC_CMD_COUNT = 11,
	SCHED_SCALEUP_RECV_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_scaleout_send_arc_cmd_opcode_t
 * \brief   Scaleout send sched commands opcodes
 * \details Opcodes processed by the Scaleout send Scheduler ARC
 */
enum sched_scaleout_send_arc_cmd_opcode_t {
	SCHED_SCALEOUT_SEND_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEOUT_SEND_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEOUT_SEND_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEOUT_SEND_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEOUT_SEND_ARC_CMD_NOP = 4,
	SCHED_SCALEOUT_SEND_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_SCALEOUT_SEND_ARC_CMD_NIC_PASSTHROUGH_V2 = 6,
	SCHED_SCALEOUT_SEND_ARC_CMD_ACP_FENCE_WAIT = 7,
	SCHED_SCALEOUT_SEND_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = 8,
	SCHED_SCALEOUT_SEND_ARC_CMD_PDMA_BATCH_TRANSFER = 9,
	SCHED_SCALEOUT_SEND_ARC_CMD_LBW_READ = 10,		// TODO: Only in Debug
	SCHED_SCALEOUT_SEND_ARC_CMD_MEM_FENCE = 11,		// TODO: Only in Debug
	SCHED_SCALEOUT_SEND_ARC_CMD_COUNT = 12,
	SCHED_SCALEOUT_SEND_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_scaleout_recv_arc_cmd_opcode_t
 * \brief   Scaleout Recv sched commands opcodes
 * \details Opcodes processed by the Scaleout Recv Scheduler ARC
 */
enum sched_scaleout_recv_arc_cmd_opcode_t {
	SCHED_SCALEOUT_RECV_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEOUT_RECV_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEOUT_RECV_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEOUT_RECV_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEOUT_RECV_ARC_CMD_NOP = 4,
	SCHED_SCALEOUT_RECV_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_SCALEOUT_RECV_ARC_CMD_NIC_PASSTHROUGH_V2 = 6,
	SCHED_SCALEOUT_RECV_ARC_CMD_ACP_FENCE_WAIT = 7,
	SCHED_SCALEOUT_RECV_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = 8,
	SCHED_SCALEOUT_RECV_ARC_CMD_PDMA_BATCH_TRANSFER = 9,
	SCHED_SCALEOUT_RECV_ARC_CMD_LBW_READ = 10,		// TODO: Only in Debug
	SCHED_SCALEOUT_RECV_ARC_CMD_MEM_FENCE = 11,		// TODO: Only in Debug
	SCHED_SCALEOUT_RECV_ARC_CMD_COUNT = 12,
	SCHED_SCALEOUT_RECV_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_gc_reduction_arc_cmd_opcode_t
 * \brief   garbage collection and reduction sched
 *          commands opcodes
 * \details Opcodes processed by the garbage collection and
 *          reduction Scheduler ARC
 */
enum sched_gc_reduction_arc_cmd_opcode_t {
	SCHED_GC_REDUCTION_ARC_CMD_FENCE_WAIT = 0,
	SCHED_GC_REDUCTION_ARC_CMD_LBW_WRITE = 1,
	SCHED_GC_REDUCTION_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_GC_REDUCTION_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_GC_REDUCTION_ARC_CMD_NOP = 4,
	SCHED_GC_REDUCTION_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_GC_REDUCTION_ARC_CMD_NIC_EDMA_OPS = 6,
	SCHED_GC_REDUCTION_ARC_CMD_ACP_FENCE_WAIT = 7,
	SCHED_GC_REDUCTION_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = 8,
	SCHED_GC_REDUCTION_ARC_CMD_LBW_READ = 9,		// TODO: Only in Debug
	SCHED_GC_REDUCTION_ARC_CMD_MEM_FENCE = 10,		// TODO: Only in Debug
	SCHED_GC_REDUCTION_ARC_CMD_COUNT = 11,
	SCHED_GC_REDUCTION_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_compute_arc_stream_state_t
 * \brief   scheduler stream states
 * \details scheduler stream states, used by profiler to print events
 *	    These enums are for internal usage of scheduler firmware and
 *	    kept here to prepare profiler event table. This should not be
 *	    used by any other components other than ARC firmware.
 */
enum sched_compute_arc_stream_state_t {
	SCHED_COMPUTE_STREAM_IDLE = 0,
	SCHED_COMPUTE_STREAM_DMA_STARTED = 1,
	SCHED_COMPUTE_STREAM_PROCESSING_CMD = 2,
	SCHED_COMPUTE_STREAM_END_OF_CCB = 3,
	SCHED_COMPUTE_STREAM_SUSPENDED = 4,
	SCHED_COMPUTE_STREAM_STATE_COUNT = 5,
	SCHED_COMPUTE_STREAM_STATE_SIZE = 0xFF
};

/**
 * \struct  qman_cmd_nop_t
 * \brief   Stop Message
 * \details Command structure for Stop message
 */
struct sched_arc_cmd_nop_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_NOP */
	uint32_t padding_count:27;
	/**<
	 * number of padded DWORDs(32bits) at the end of the
	 * command for alignment purpose
	 */
	uint32_t padding[0];
	/**<
	 * Padding to align with 256 Bytes command buffer
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  qman_cmd_dispatch_compute_ecb_list_t
 * \brief   Dispatch Compute ECB list command
 * \details Command structure for Dispatch Compute ECB list
 */
struct sched_arc_cmd_dispatch_compute_ecb_list_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST */
	uint32_t reserved:2;
	/**<
	 * Reserved
	 */
	uint32_t cmpt_stream_index:2;
	/**<
	 * Compute stream index that should be used for SFG feature
	 * Firmware receives SOB IDs from SCAL using Engine config
	 * for each of the compute streams that should be used to signal from
	 * the graph
	 */
	uint32_t use_gc_nop:1;
	/**<
	 * use GC provided NOP kernel for TPC
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Various engine types supported by hardware
	 */
	uint32_t single_static_chunk:1;
	/**<
	 * This bit indicates static ecb list contains only one chunk
	 */
	uint32_t single_dynamic_chunk:1;
	/**<
	 * This bit indicates dynamic ecb list contains only one chunk
	 */
	uint32_t reserved1:16;
	/**<
	 * reserved
	 */
	struct {
		uint32_t static_ecb_list_offset:16;
		/**<
		 * Address offset of Static ECB List in memory with respect to
		 * dynamic ecb list
		 * This is in multiple of STATIC_COMPUTE_ECB_LIST_BUFF_SIZE
		 */
		uint32_t static_ecb_list_eng_offset:16;
		/**<
		 * Static ECB List Engine specific offset
		 * This is in multiple of STATIC_COMPUTE_ECB_LIST_BUFF_SIZE
		 */
	} __attribute__ ((aligned(4), __packed__));;
	struct {
		uint32_t reserved2:8;
		/**<
		 * reserved
		 */
		uint32_t dynamic_ecb_list_addr_256:24;
		/**<
		 * Address of ECB List in memory
		 * This is STATIC_COMPUTE_ECB_LIST_BUFF_SIZE bytes aligned
		 */
	} __attribute__ ((aligned(4), __packed__));
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_dispatch_cme_ecb_list_t
 * \brief   Dispatch CME ECB list command
 * \details Command structure for Dispatch CME ECB list
 */
struct sched_arc_cmd_dispatch_cme_ecb_list_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_COMPUTE_ARC_CMD_DISPATCH_CME_ECB_LIST
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Engine Group type of CME
	 */
	uint32_t :7;
	/**<
	 * Reserved
	 */
	uint32_t ecb_list_size:16;
	/**<
	 * ECB List Size
	 */
	struct {
		uint32_t :8;
		/**<
		 * reserved
		 */
		uint32_t ecb_list_addr:24;
		/**<
		 * Address of ECB List in memory
		 * This is DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE bytes aligned
		 */
	} __attribute__ ((aligned(4), __packed__));
} __attribute__ ((aligned(4), __packed__));

/**
 * \enum    sched_mon_exp_opcode_t
 * \brief   Opcodes for monitor expiration messages
 * \details Opcodes used by firmware for monitor expiration messages
 */
enum sched_mon_exp_opcode_t {
	MON_EXP_FENCE_UPDATE = 0,
	/**<
	 * To update firmware fence counter
	 */
	MON_EXP_COMP_FENCE_UPDATE = 1,
	/**<
	 * to indicate completion on the completion group
	 */
	MON_EXP_SO_SET_RESET = 2,
	/**<
	 * used internally by firmware
	 */
	MON_EXP_UPDATE_Q_CREDIT = 3,
	/**<
	 * used internally by firmware
	 */
	MON_EXP_MSG_COUNT = 4
	/**<
	 * Total number of expiration messages
	 */
};

/**
 * \struct  sched_mon_exp_generic_t
 * \brief   Generic message structure
 * \details Structure describes common fields for all the types of messages
 */
struct sched_mon_exp_generic_t {
	uint32_t opcode:2;
	/**<
	 * opcode
	 */
	uint32_t reserved:30;
	/**<
	 * reserved
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_mon_exp_fence_t
 * \brief   Monitor expiration structure to update fence
 * \details structure used for updating global fence counters
 */
struct sched_mon_exp_fence_t {
	union {
		struct {
			uint32_t opcode:2;
			/**<
			 * opcode : MON_EXP_FENCE_UPDATE
			 */
			uint32_t fence_id:7;
			/**<
			 * Index of the fence counter
			 */
			uint32_t reserved:23;
			/**<
			 * reserved
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw;
		/**<
		 * Raw field
		 */
	};
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_mon_exp_comp_fence_t
 * \brief   Monitor expiration structure to update fence counters in completion
 *	    groups
 * \details structure used for updating global fence counters which are part of
 *	    completion groups
 */
struct sched_mon_exp_comp_fence_t {
	union {
		struct {
			uint32_t opcode:2;
			/**<
			 * opcode : MON_EXP_COMP_FENCE_UPDATE
			 */
			uint32_t comp_group_index:4;
			/**<
			 * Index of the completion group
			 */
			uint32_t mon_id:11;
			/**<
			 * Monitor ID
			 */
			uint32_t mon_sm_id:4;
			/**<
			 * Monitor Sync Manager Instance ID
			 */
			uint32_t mon_index:2;
			/**<
			 * local index of the monitor in the completion group
			 * Starts from 0 to max (COMP_SYNC_GROUP_MAX_MON_GROUP_COUNT - 1)
			 */
			uint32_t update_slave_credit:1;
			/**<
			 * Indicates that this message is coming from Slaves
			 * to synchronize the SOB credit with Master
			 */
			uint32_t reserved:8;
			/**<
			 * reserved
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw;
		/**<
		 * Raw field
		 */
	};
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_mon_exp_update_q_credit_t
 * \brief   DCCM Queue Flow control message
 * \details Message that is posted when half of the DCCM Queue is consumed
 *	    by the Engine ARCs.
 */
struct sched_mon_exp_update_q_credit_t {
	union {
		struct {
			uint32_t opcode:2;
			/**<
			 * MON_EXP_UPDATE_Q_CREDIT
			 */
			uint32_t engine_group_type:4;
			/**<
			 * Engine Group type from scal.h file
			 */
			uint32_t :26;
			/**<
			 * Not used
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw;
		/**<
		 * Raw field
		 */
	};
} __attribute__ ((aligned(4), __packed__));

/**
 * \union  sched_mon_exp_msg_t
 * \brief   Various monitor expiration messages
 * \details Various monitor expiration messages shared used by Firmware and
 *	    software stack.
 */
union sched_mon_exp_msg_t {
	struct sched_mon_exp_generic_t generic;
	/**<
	 * generic message
	 */
	struct sched_mon_exp_fence_t fence;
	/**<
	 * monitor expiration message used for global fence counters
	 */
	struct sched_mon_exp_comp_fence_t comp_fence;
	/**<
	 * monitor expiration message used for fence counters of completion
	 * groups
	 */
	uint32_t raw;
	/**<
	 * raw bitfield
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_pdma_commands_params_t
 * \brief   PDMA batch transfer parameters
 * \details This structure provides required PDMA transfer parameters
 *	    for each transfer.
 */
struct sched_arc_pdma_commands_params_t {
	uint32_t transfer_size;
	/**<
	 * Transfer size in bytes
	 */
	uint64_t src_addr;
	/**<
	 * Source address
	 */
	uint64_t dst_addr;
	/**<
	 * Destination address
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * Maximum number of PDMA transfers that can be done
 * using single sched_arc_cmd_pdma_batch_transfer_t
 * command
 */
#define SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT	4

/**
 * \enum    sched_arc_pdma_dir_t
 * \brief   enums for PDMA direction
 * \details enums for PDMA direction
 */
enum sched_arc_pdma_dir_t {
	PDMA_DIR_DEV2HOST = 0,
	PDMA_DIR_HOST2DEV = 1,
	PDMA_DIR_DEV2DEV = 2,
	PDMA_DIR_COUNT = 3
};

/**
 * \struct  sched_arc_cmd_pdma_batch_transfer_t
 * \brief   PDMA commands transfer
 * \details This command allows multiple PDMA transfers to be submitted together.
 */
struct sched_arc_cmd_pdma_batch_transfer_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER
	 */
	uint32_t engine_group_type:4;
	/**<
	 * PDMA Tx command group
	 */
	uint32_t watch_dog_sig_value:1;
	/**<
	 * Value to be used by firmware to increment the watch
	 * dog SOB
	 */
	uint32_t stream_ctxt_id:8;
	/**<
	 * ContextId for the stream
	 * [1:0] : sched_arc_pdma_dir_t
	 * [7:2] : Used only by profiler not used by Firmware
	 */
	uint32_t api_id:5;
	/**<
	 * API ID of the command which is used to tie Host API and H/W events
	 */
	uint32_t workload_type:3;
	/**<
	 * Workload type
	 * 0 - user data
	 * 1 - command
	 */
	uint32_t memset:1;
	/**<
	 * Perform memset operation
	 * Source address field is used as the value for memset
	 */
	uint32_t has_payload:1;
	/**<
	 * 0 - Does not have valid payload to write back on completion
	 * 1 - Has valid payload to write back on completion
	 */
	uint32_t signal_to_cg:1;
	/**<
	 * 0 - depending on the ‘has_payload’ field, signal the payload
	 *     provided by the user to the address provided
	 * 1 - set the PDMA to write its completion message so it increments
	 *     the current SO in the CG completion group by 1
	 * When this bit is set, pay_addr field contains a 4 bit completion
	 * group index.
	 */
	uint32_t batch_count:3;
	/**<
	 * Number of valid transfer parameters in the transfer_params
	 * Maximum number of transfers is limited to
	 * SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT
	 */
	union {
		struct {
			uint32_t :2;
			/**<
			 * Unused
			 */
			uint32_t reduction_ind:1;
			/**<
			  * Reduction indication
			  */
			uint32_t reduction_op:3;
			/**<
			 * Reduction operation to be performed
			 */
			uint32_t reduction_dtype:4;
			/**<
			 * Reduction data type
			 * 0xC- upscaling FP16
			 * 0xD- upscaling BF16
			 */
			uint32_t hbw_axcache:8;
			/**<
			 * HBW AX CACHE Setting for READ/WRITE.
			 * RD[3:0]
			 * WR[7:4]
			 * If no value is supplied, FW will configure to 0x33 (RESET Value)
			 */
			uint32_t reserved1:14;
		};
		uint32_t reduction_dword_raw;
	};
	/**<
	 * new dword for reduction params
	 */
	uint32_t pay_data;
	/**<
	 * payload data, i.e. 32bit data that needs to be written when the
	 * transfer completes
	 */
	uint32_t pay_addr;
	/**<
	 * when signal_to_cg is 0, then this field specifies a
	 * payload address, i.e. address where the data needs to be
	 * written when the transfer completes, higher 32bits of the LBW
	 * address is harcoded in the firmware, so only lower32 bit LBW
	 * address is passed here
	 *
	 * When signal_to_cg is 1, this field contains a
	 * 4 bit completion group index, to which the PDMA completion
	 * is to be signalled. Index ranges between 0 to
	 * (COMP_GROUP_COUNT - 1)
	 */
	struct sched_arc_pdma_commands_params_t batch_params[0];
	/**<
	 * transfer parameter for each batch
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_alloc_barrier_v2_t
 * \brief   allocate barrier
 * \details allocate required barrier resources so that barrier can be sent
 */
struct sched_arc_cmd_alloc_barrier_v2_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_ALLOC_BARRIER_V2 */
	uint32_t comp_group_index:4;
	/**<
	 * Completion Group index. Index bettween 0 to
	 * (COMP_SYNC_GROUP_COUNT - 1)
	 */
	uint32_t target_value:14;
	/**<
	 * Target value of the SOB
	 */
	uint32_t rel_so_set:1;
	/**<
	 * when set, releases and recycles the current SOs set.
	 */
	uint32_t allocate_so_set:1;
	/**<
	 * when set, an SO set is allocated.
	 * Setting this bit can avoid sending an explicit command for
	 * SO SET allocation.
	 * Here are the possible combos:
	 * allocate_so_set | rel_so_set |           comments
	 *          0            0          No SO SET will be alloted/freed
	 *          0            1          user has to send an explicit SO_SET_ALLOC, which will be FREED after Barrier
	 *          1            0          SO SET will be allocated but will never be FREED
	 *          1            1          Single command handles BARRIER + SO SET allocation + FREEing
	 */
	uint32_t b2b_exec_with_no_fence:1;
	/**<
	 * 0: Default: B2B List Execution is FENCED via SM for all-Engine idleness
	 * 1: B2B List Execution will happen with no FENCE
	 */
	uint32_t reserved:2;
	/**<
	 * reserved
	 */
	uint32_t num_engine_group_type:4;
	/**<
	 * Number of engine group types to which the alloc barrier
	 * needs to be sent to.
	 * when set, this command inserts QMAN FENCE_WAIT command
	 * in the QMAN interface to wait for the previous dispatch
	 * to be over.
	 */
	uint8_t engine_group_type[4];
	/**<
	 * Array of engine group types to which the alloc barrier
	 * needs to be sent
	 */
	uint32_t degrade_mcid_count:16;
	/**<
	 * Required number of Degrade MCIDs.
	 * Used only if allocate_so_set is set to 1
	 */
	uint32_t discard_mcid_count:16;
	/**<
	 * Required number of Degrade MCIDs
	 * Used only if allocate_so_set is set to 1
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_alloc_nic_barrier_t
 * \brief   allocate NIC barrier
 * \details allocate required barrier resources so that barrier can be sent
 *          to NIC engines
 */
struct sched_arc_cmd_alloc_nic_barrier_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_ALLOC_NIC_BARRIER */
	uint32_t comp_group_index:8;
	/**<
	 * Completion Group index. Index bettween 0 to
	 * (COMP_GROUP_COUNT - 1)
	 */
	uint32_t required_sobs:14;
	/**<
	 * required number of sobs to be allocated
	 */
	uint32_t reserved:5;
	/**<
	 * reserved
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_dispatch_barrier_t
 * \brief   Dispatch barrier
 * \details Dispatches barrier to an array of engine group types
 */
struct sched_arc_cmd_dispatch_barrier_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_DISPATCH_BARRIER */
	uint32_t watch_dog_sig_value:15;
	/**<
	 * Value to be used by firmware to increment the watch
	 * dog SOB
	 */
	uint32_t reserved:4;
	/**< reserved */
	uint32_t num_engine_group_type:8;
	/**<
	 * Number of engine group types to which the dispatch needs
	 * to be sent
	 */
	uint8_t engine_group_type[4];
	/**<
	 * Array of engine group types to which the dispatch needs
	 * to be sent
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_fence_wait_t
 * \brief   Wait for fence credit
 * \details Checks if the stream fence counter in the specified index has
 *          reached a certain amount of credits. If the stream already has
 *          enough credits it continues, otherwise it blocks until the
 *          required credits are received.
 *          Stream status after the execution:
 *          - The target value is subtracted from to the current value of the
 *          fence counter.
 *          - If the value (after the substruction) is negative the stream is
 *          suspended.
 *          - If the value is non-negative, the stream continues.
 */
struct sched_arc_cmd_fence_wait_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_FENCE_WAIT */
	uint32_t fence_id:7;
	/**< Index of the fence register of this stream */
	uint32_t target:6;
	/**< target value of the fence */
	uint32_t reserved:14;
	/**< reserved */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_acp_fence_wait_t
 * \brief   Wait for ACP fence credit
 * \details This command is used to wait on ACP Counters added in Gaudi3
 *          Stream gets suspended if the counter is negative and resumed
 *          when it becomes 0 or greater.
 */
struct sched_arc_cmd_acp_fence_wait_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_ACP_FENCE_WAIT */
	uint32_t fence_id:6;
	/**< Index of the ACP fence register of this stream */
	uint32_t target:12;
	/**< Signed 12 bit value to be decremented */
	uint32_t reserved:9;
	/**< reserved */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_fence_inc_immediate_t
 * \brief   Increment fence counters by 1
 * \details Increment fence counters by 1 specified in the array.
 */
struct sched_arc_cmd_fence_inc_immediate_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_FENCE_INC_IMMEDIATE */
	uint32_t reserved1:3;
	/**< reserved */
	uint32_t fence_count:8;
	/**< number of valid fence indexes in the array */
	uint32_t fence_index:6;
	/**< fence index value */
	uint32_t reserved2:10;
	/**< reserved */
	uint8_t fence_id[8];
	/**<
	 * array of fence Ids
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_acp_fence_inc_immediate_t
 * \brief   Increment fence counters by specified value
 * \details Increment fence counters by specified value
 */
struct sched_arc_cmd_acp_fence_inc_immediate_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_ACP_FENCE_INC_IMMEDIATE */
	uint32_t fence_id:6;
	/**< number of valid fence indexes in the array */
	uint32_t value:12;
	/**< reserved */
	uint32_t reserved:9;
	/**< reserved */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_lbw_read_t
 * \brief   Reads a mem/register location using LBW interface
 * \details copy a range of registers from the LBW space to the destination
 *          buffer.
 */
struct sched_arc_cmd_lbw_read_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_LBW_READ */
	uint32_t reserved:27;
	/**<
	 * Reserved
	 */
	uint32_t src_addr;
	/**<
	 * source LBU address
	 * Note: its LBU address and not LBW, it should start with
	 * 0xFxxxxxx
	 */
	uint32_t dst_addr;
	/**<
	 * destination address
	 */
	uint32_t size;
	/**<
	 * size in bytes
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_sem_lbw_write_t
 * \brief   Write into mem/register location using LBW interface
 * \details write a range of values into destination LBW address.
 */
struct sched_arc_cmd_lbw_write_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_LBW_WRITE */
	uint32_t reserved:25;
	/**<
	 * Reserved
	 */
	uint32_t block_next:1;
	/**<
	 * Block execution of next command by putting the stream into
	 * suspended state
	 */
	uint32_t wait_for_completion:1;
	/**<
	 * Wait until current write is completed
	 */
	uint32_t dst_addr;
	/**<
	 * destination LBU address
	 * Note: its LBU address and not LBW, it should start with
	 * 0xFxxxxxx
	 */
	uint32_t src_data;
	/**<
	 * source data to be written
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_sem_lbw_burst_write_t
 * \brief   Write into mem/register location using LBW interface
 * \details write 4 DWORDs into destination LBW address.
 */
struct sched_arc_cmd_lbw_burst_write_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_LBW_BURST_WRITE */
	uint32_t reserved:25;
	/**<
	 * Reserved
	 */
	uint32_t block_next:1;
	/**<
	 * Block execution of next command by putting the stream into
	 * suspended state
	 */
	uint32_t wait_for_completion:1;
	/**<
	 * Wait until current write is completed
	 */
	uint32_t dst_addr;
	/**<
	 * destination LBU address
	 * Note: its LBU address and not LBW, it should start with
	 * 0xFxxxxxx
	 */
	uint32_t src_data0;
	uint32_t src_data1;
	uint32_t src_data2;
	uint32_t src_data3;
	/**<
	 * source data to be written, 4 DWORDs are written starting
	 * from destination address
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_mem_fence_t
 * \brief   waits until mem transactions are completed
 * \details waits (in busy mode) until all the ARC memory transactions,
 *          DUP memory transactions and DMA memory transactions are done.
 */
struct sched_arc_cmd_mem_fence_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_MEM_FENCE */
	uint32_t arc:1;
	/**<
	 * wait until all the ARC transactions are completed
	 */
	uint32_t dup_eng:1;
	/**<
	 * wait until DUP engine is idle
	 */
	uint32_t arc_dma:1;
	/**<
	 * wait until ARC DMA engine is idle
	 */
	uint32_t reserved:24;
	/**<
	 * reserved
	 */
} __attribute__ ((aligned(4), __packed__));


/**<
 * Max number of engine groups in update recipe cmd
 */
#define SCHED_CMD_UPDATE_RECIPE_MAX_ENG_GROUPS 4
/**
 * \struct  sched_arc_cmd_update_recipe_base_v2_t
 * \brief   Update receipe base address
 * \details Update a particular recipe base address registers pointed by
 *          base index parameters
 */

struct sched_arc_cmd_update_recipe_base_v2_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_UPDATE_RECIPE_BASE_V2 */
	uint32_t num_recipe_addrs:3;
	/**<
	 * Number of recipe base addresses
	 */
	uint32_t num_engine_group_type:8;
	/**<
	 * Number of engine group types to which the dispatch needs
	 * to be sent
	 */
	union {
		struct {
			uint16_t recipe_base_index0:4;
	 		/**<
			 * Index of the base register-0 which needs to be updated
			 */
			uint16_t recipe_base_index1:4;
	 		/**<
			 * Index of the base register-1 which needs to be updated
			 */
			uint16_t recipe_base_index2:4;
	 		/**<
			 * Index of the base register-2 which needs to be updated
			 */
			uint16_t recipe_base_index3:4;
	 		/**<
			 * Index of the base register-3 which needs to be updated
			 */
		};
		uint16_t recipe_base_indices;
	};
	/**<
	 * Index 0 to 6 are used for Static and Dynamic descriptors
	 * Index 7 is used for storing the GC provided TPC NOP kernel
	 */
	uint8_t engine_group_type[SCHED_CMD_UPDATE_RECIPE_MAX_ENG_GROUPS];
	/**<
	 * Array of engine group types to which the dispatch needs
	 * to be sent
	 */
	uint64_t recipe_base_addr[0];
	/**<
	 * Base address values
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_nic_passthrough_v2_t
 * \brief   Send raw command descriptors to Patcher
 * \details This command is used to send raw command descriptors
 *          to patcher block in direct mode.
 *          These descriptors include update glbl/collective
 *          context, send/recv and NOP commands
 */
struct sched_arc_cmd_nic_passthrough_v2_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_NIC_PASSTHROUGH
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Engine group type to which this command needs to be
	 * send. This can be scaleup send/recv and scaleout send/recv
	 */
	uint32_t num_dwords:4;
	/*
	 * Num of dwords in this command (doesn't include this header)
	 */
	uint32_t dup_mask:12;
	/*
	 * DUP Mask to be programmed before sending the packet
	 * If it is set to 0, then mask register is not programmed and
	 * DUP would work on default value of the mask register
	 */
	uint32_t required_q_credits_inbytes:6;
	/**<
	 * Engine Group queue credits required before pushing configs
	 * to respective engines
	 */
	uint32_t :1;
	/*
	 * Reserved
	 */
	uint32_t passthrough_data[0];
	/**<
	 * passthrough_data to hold the command descriptors. Size = num_dwords * 4
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_nic_edma_ops_t
 * \brief   NIC EDMA related operations
 * \details This command is used to for doing various EDMA operations
 *	    on EDMAs associated with NICs.
 */
struct sched_arc_cmd_nic_edma_ops_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_NIC_EDMA_OPS
	 */
	uint32_t engine_group_type:4;
	/**<
	 * engine group type :
	 * SCAL_EDMA_NETWORK_REDUCTION_GROUP
	 * SCAL_EDMA_NETWORK_LOOPBACK_GROUP
	 * SCAL_EDMA_NETWORK_MEMSET_GROUP
	 */
	uint32_t :7;
	/**<
	 * Unused
	 */
	uint32_t cmd_size:16;
	/**<
	 * Command Size in Bytes, it includes the dword that
	 * contains the size field
	 */
	struct arc_cmd_nic_edma_ops_v3_t edma_ops_v3[0];
	struct arc_cmd_update_edma_nic_ctxt_v3_t edma_ctxt_v3[0];
	struct arc_cmd_nic_edma_ops_cdc_t edma_cdc[0];
	struct arc_cmd_nic_edma_sibo_ops_v3_t sibo_ops_v3[0];
	struct arc_cmd_nic_edma_sibo_memset_v3_t sibo_memset_v3[0];
	struct arc_cmd_nic_edma_sibo_memset_v3_2_t sibo_memset_v3_2[0];
	struct arc_cmd_nic_edma_lin_ops_v3_t lin_ops_v3[0];
	struct arc_cmd_nic_edma_lin_memset_v3_t lin_memset_v3[0];
	/**<
	 * NIC EDMA command payload that needs to be dispatched
	 * to NICs
	 */
} __attribute__ ((aligned(4), __packed__));

#endif /* __GAUDI3_ARC_SCHED_PACKETS_H__ */

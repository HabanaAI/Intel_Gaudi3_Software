/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef __HLIBDV_H__
#define __HLIBDV_H__

#include <stdbool.h>
#include <infiniband/verbs.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of backpressure offsets */
#define HLIBDV_USER_BP_OFFS_MAX	16

/* Number of FnA addresses for SRAM/DCCM completion */
#define HLIBDV_FNA_CMPL_ADDR_NUM 2

#define HL_IB_MTU_8192	6

/* Maximum amount of Collective Scheduler resources */
#define HLIBDV_MAX_NUM_COLL_SCHED_RESOURCES 128

/**
 * struct hlibdv_qp_caps - HL QP capabilities flags.
 * @HLIBDV_QP_CAP_LOOPBACK: Enable QP loopback.
 * @HLIBDV_QP_CAP_CONG_CTRL: Enable congestion control.
 * @HLIBDV_QP_CAP_COMPRESSION: Enable compression.
 * @HLIBDV_QP_CAP_SACK: Enable selective acknowledgment feature.
 * @HLIBDV_QP_CAP_ENCAP: Enable packet encapsulation.
 * @HLIBDV_QP_CAP_COLL: Enable collective operations.
 */
enum hlibdv_qp_caps {
	HLIBDV_QP_CAP_LOOPBACK = 0x1,
	HLIBDV_QP_CAP_CONG_CTRL = 0x2,
	HLIBDV_QP_CAP_COMPRESSION = 0x4,
	HLIBDV_QP_CAP_SACK = 0x8,
	HLIBDV_QP_CAP_ENCAP = 0x10,
	HLIBDV_QP_CAP_COLL = 0x20,
};

/**
 * struct hlibdv_port_ex_caps - HL port extended capabilities flags.
 * @HLIBDV_PORT_CAP_ADVANCED: Enable port advanced features like RDV, QMan, WTD, etc.
 * @HLIBDV_PORT_CAP_ADAPTIVE_TIMEOUT: Enable adaptive timeout feature on this port.
 */
enum hlibdv_port_ex_caps {
	HLIBDV_PORT_CAP_ADVANCED = 0x1,
	HLIBDV_PORT_CAP_ADAPTIVE_TIMEOUT = 0x2,
};

/**
 * enum hlibdv_mem_id - Gaudi2 (or higher) memory allocation methods.
 * @HLIBDV_MEM_HOST: memory allocated on the host memory.
 * @HLIBDV_MEM_DEVICE: memory allocated on the device memory.
 */
enum hlibdv_mem_id {
	HLIBDV_MEM_HOST = 1,
	HLIBDV_MEM_DEVICE
};

/**
 * enum hlibdv_wq_array_type - WQ-array type.
 * @HLIBDV_WQ_ARRAY_TYPE_GENERIC: WQ-array for generic QPs.
 * @HLIBDV_WQ_ARRAY_TYPE_COLLECTIVE: (Gaudi3 and above) WQ-array for collective QPs.
 * @HLIBDV_WQ_ARRAY_TYPE_SCALE_OUT_COLLECTIVE: (Gaudi3 and above) WQ-array for scale-out
 *                                             collective QPs.
 * @HLIBDV_WQ_ARRAY_TYPE_MAX: Max number of values in this enum.
 */
enum hlibdv_wq_array_type {
	HLIBDV_WQ_ARRAY_TYPE_GENERIC,
	HLIBDV_WQ_ARRAY_TYPE_COLLECTIVE,
	HLIBDV_WQ_ARRAY_TYPE_SCALE_OUT_COLLECTIVE,
	HLIBDV_WQ_ARRAY_TYPE_MAX = 5
};

/**
 * enum hlibdv_swq_granularity - send WQE granularity.
 * @HLIBDV_SWQE_GRAN_32B: 32 byte WQE for linear write.
 * @HLIBDV_SWQE_GRAN_64B: 64 byte WQE for multi-stride write.
 */
enum hlibdv_swq_granularity {
	HLIBDV_SWQE_GRAN_32B,
	HLIBDV_SWQE_GRAN_64B
};

/**
 * enum hlibdv_usr_fifo_type - NIC users FIFO modes of operation.
 * @HLIBDV_USR_FIFO_TYPE_DB: (Gaudi2 and above) mode for direct user door-bell submit.
 * @HLIBDV_USR_FIFO_TYPE_CC: (Gaudi2 and above) mode for congestion control.
 * @HLIBDV_USR_FIFO_TYPE_COLL_OPS_SHORT: (Gaudi3 and above) mode for short collective operations.
 * @HLIBDV_USR_FIFO_TYPE_COLL_OPS_LONG: (Gaudi3 and above) mode for long collective operations.
 * @HLIBDV_USR_FIFO_TYPE_DWQ_LIN: (Gaudi3 and above) mode for linear direct WQE submit.
 * @HLIBDV_USR_FIFO_TYPE_DWQ_MS: (Gaudi3 and above) mode for multi-stride WQE submit.
 * @HLIBDV_USR_FIFO_TYPE_COLL_DIR_OPS_SHORT: (Gaudi3 and above) mode for direct short collective
 *                                                              operations.
 * @HLIBDV_USR_FIFO_TYPE_COLL_DIR_OPS_LONG: (Gaudi3 and above) mode for direct long collective
 *                                                             operations.
 */
enum hlibdv_usr_fifo_type {
	HLIBDV_USR_FIFO_TYPE_DB = 0,
	HLIBDV_USR_FIFO_TYPE_CC,
	HLIBDV_USR_FIFO_TYPE_COLL_OPS_SHORT,
	HLIBDV_USR_FIFO_TYPE_COLL_OPS_LONG,
	HLIBDV_USR_FIFO_TYPE_DWQ_LIN,
	HLIBDV_USR_FIFO_TYPE_DWQ_MS,
	HLIBDV_USR_FIFO_TYPE_COLL_DIR_OPS_SHORT,
	HLIBDV_USR_FIFO_TYPE_COLL_DIR_OPS_LONG,
};

/**
 * enum hlibdv_qp_wq_types - QP WQ types.
 * @HLIBDV_WQ_WRITE: WRITE or "native" SEND operations are allowed on this QP.
 *                   NOTE: the latter is currently unsupported.
 * @HLIBDV_WQ_RECV_RDV: RECEIVE-RDV or WRITE operations are allowed on this QP.
 *                      NOTE: posting all operations at the same time is unsupported.
 * @HLIBDV_WQ_READ_RDV: READ-RDV or WRITE operations are allowed on this QP.
 *                      NOTE: posting all operations at the same time is unsupported.
 * @HLIBDV_WQ_SEND_RDV: SEND-RDV operation is allowed on this QP.
 * @HLIBDV_WQ_READ_RDV_ENDP: No operation is allowed on this endpoint QP.
 */
enum hlibdv_qp_wq_types {
	HLIBDV_WQ_WRITE = 0x1,
	HLIBDV_WQ_RECV_RDV = 0x2,
	HLIBDV_WQ_READ_RDV = 0x4,
	HLIBDV_WQ_SEND_RDV = 0x8,
	HLIBDV_WQ_READ_RDV_ENDP = 0x10,
};

/**
 * enum hlibdv_cq_type - CQ types, used during allocation of CQs.
 * @HLIBDV_CQ_TYPE_QP: Standard CQ used for completion of a operation for a QP.
 * @HLIBDV_CQ_TYPE_CC: Congestion control CQ.
 */
enum hlibdv_cq_type {
	HLIBDV_CQ_TYPE_QP = 0,
	HLIBDV_CQ_TYPE_CC,
};

/**
 * enum hlibdv_encap_type - Supported encapsulation types.
 * @HLIBDV_ENCAP_TYPE_NO_ENC: No Tunneling.
 * @HLIBDV_ENCAP_TYPE_ENC_OVER_IPV4: Tunnel RDMA packets through L3 layer.
 * @HLIBDV_ENCAP_TYPE_ENC_OVER_UDP: Tunnel RDMA packets through L4 layer.
 */
enum hlibdv_encap_type {
	HLIBDV_ENCAP_TYPE_NO_ENC = 0,
	HLIBDV_ENCAP_TYPE_ENC_OVER_IPV4,
	HLIBDV_ENCAP_TYPE_ENC_OVER_UDP,
};

/**
 * enum hlibdv_coll_sched_resource_type - FS1 (or higher) scheduler resource type.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_AF_UMR: Resource holding 33 pages mapping the NMS's AF's UMRs.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_AF: Resource holding 2 pages mapping the NMS's AF's configuration
 *                                   registers.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_ARC_AUX: Resource holding 1 page mapping the NMS's AUX Registers.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_ARC_DCCM: Resource holding 16 pages mapping the NMS ARC-DCCM.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_SDUP: Resource holding 8 pages mapping the NMS sDUP.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_SM: Resource holding 8 pages mapping of NMS SM.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_MFC_SLOTS: Resource holding 1 page of NMS MFC.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_EDUP_PUSH: Resource holding 1 page of NMS eDUP PUSH.
 * @HLIBDV_COLL_SCHED_RSRC_T_NMS_EDUP_GRPS: Resource holding 1 page of NMS eDUP groups
 *                                          configuration.
 * @HLIBDV_COLL_SCHED_RSRC_T_PDMA: Resource holding all the pages of PDMA on each of the compute
 *                                          dies.
 * @HLIBDV_COLL_SCHED_RSRC_T_EDMA: Resource holding all the pages of EDMA on each of the compute
 *                                          dies.
 * @HLIBDV_COLL_SCHED_RSRC_T_NIC_USR_FIFO_UMR: 4 NIC user-FIFO UMRs per NIC HW macro.
 * @HLIBDV_COLL_SCHED_RSRC_T_MAX: Number of values in enum.
 */
enum hlibdv_coll_sched_resource_type {
	HLIBDV_COLL_SCHED_RSRC_T_NMS_AF_UMR,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_AF,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_ARC_AUX,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_ARC_DCCM,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_SDUP,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_SM,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_MFC_SLOTS,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_EDUP_PUSH,
	HLIBDV_COLL_SCHED_RSRC_T_NMS_EDUP_GRPS,
	HLIBDV_COLL_SCHED_RSRC_T_PDMA,
	HLIBDV_COLL_SCHED_RSRC_T_EDMA,
	HLIBDV_COLL_SCHED_RSRC_T_NIC_USR_FIFO_UMR,
	HLIBDV_COLL_SCHED_RSRC_T_MAX,
};

/**
 * enum hlibdv_device_attr_cap_type - Device specific attributes.
 * @HLIBDV_DEVICE_ATTR_CAP_CC: Congestion control.
 * @HLIBDV_DEVICE_ATTR_CAP_COLL: Collective QPs.
 */
enum hlibdv_device_attr_cap_type {
	HLIBDV_DEVICE_ATTR_CAP_CC = 1 << 0,
	HLIBDV_DEVICE_ATTR_CAP_COLL = 1 << 1,
};

/**
 * struct hlibdv_ucontext_attr - HL user context attributes.
 * @ports_mask: Mask of the relevant ports for this context (should be 1-based).
 * @core_fd: core device file descriptor.
 */
struct hlibdv_ucontext_attr {
	uint64_t ports_mask;
	uint32_t core_fd;
};

/**
 * struct hlibdv_wq_array_attr - WQ-array attributes.
 * @max_num_of_wqs: Max number of WQs (QPs) to be used.
 * @max_num_of_wqes_in_wq: Max number of WQ elements in each WQ.
 * @mem_id: Memory allocation method.
 * @swq_granularity: Send WQE size.
 */
struct hlibdv_wq_array_attr {
	uint32_t max_num_of_wqs;
	uint32_t max_num_of_wqes_in_wq;
	enum hlibdv_mem_id mem_id;
	enum hlibdv_swq_granularity swq_granularity;
};

/**
 * struct hlibdv_port_ex_attr - HL port extended attributes.
 * @wq_arr_attr: Array of WQ-array attributes for each WQ-array type.
 * @qp_wq_bp_offs: Offsets in NIC memory to signal a back pressure.
 * @atomic_fna_fifo_offs: SRAM/DCCM addresses provided to the HW by the user when FnA completion is
 *                        configured in the SRAM/DCCM.
 * @port_num: Port ID (should be 1-based).
 * @atomic_fna_mask_size: Completion address value mask.
 * @advanced: WQ should support advanced operations such as RDV, QMan, WTD, etc.
 */
struct hlibdv_port_ex_attr {
	struct hlibdv_wq_array_attr wq_arr_attr[HLIBDV_WQ_ARRAY_TYPE_MAX];
	uint32_t qp_wq_bp_offs[HLIBDV_USER_BP_OFFS_MAX];
	uint32_t atomic_fna_fifo_offs[HLIBDV_FNA_CMPL_ADDR_NUM];
	uint32_t port_num;
	uint8_t atomic_fna_mask_size;
	uint8_t advanced;
};

/**
 * struct hlibdv_port_ex_attr_tmp - HL port extended attributes.
 * @wq_arr_attr: Array of WQ-array attributes for each WQ-array type.
 * @caps: Port capabilities bit-mask.
 * @qp_wq_bp_offs: Offsets in NIC memory to signal a back pressure.
 * @atomic_fna_fifo_offs: SRAM/DCCM addresses provided to the HW by the user when FnA completion is
 *                        configured in the SRAM/DCCM.
 * @port_num: Port ID (should be 1-based).
 * @atomic_fna_mask_size: Completion address value mask.
 */
struct hlibdv_port_ex_attr_tmp {
	struct hlibdv_wq_array_attr wq_arr_attr[HLIBDV_WQ_ARRAY_TYPE_MAX];
	uint64_t caps;
	uint32_t qp_wq_bp_offs[HLIBDV_USER_BP_OFFS_MAX];
	uint32_t atomic_fna_fifo_offs[HLIBDV_FNA_CMPL_ADDR_NUM];
	uint32_t port_num;
	uint8_t atomic_fna_mask_size;
};

/**
 * struct hlibdv_query_port_attr - HL query port specific parameters.
 * @max_num_of_qps: Number of QPs that are supported by the driver. User must allocate enough room
 *		    for his work-queues according to this number.
 * @num_allocated_qps: Number of QPs that were already allocated (in use).
 * @max_allocated_qp_num: The highest index of the allocated QPs (i.e. this is where the
 *			  driver may allocate its next QP).
 * @max_cq_size: Maximum size of a CQ buffer.
 * @max_num_of_scale_out_coll_qps: Number of scale-out collective QPs that are supported by the
 *                                 driver. User must allocate enough room for his collective
 *                                 work-queues according to this number.
 * @max_num_of_coll_qps: Number of collective QPs that are supported by the driver. User must
 *                       allocate enough room for his collective work-queues according to this
 *                       number.
 * @base_scale_out_coll_qp_num: The first scale-out collective QP id (common for all ports).
 * @base_coll_qp_num: The first collective QP id (common for all ports).
 * @coll_qps_offset: Specific port collective QPs index offset.
 * @advanced: true if advanced features are supported.
 * @max_num_of_cqs: Maximum number of CQs.
 * @max_num_of_usr_fifos: Maximum number of user FIFOs.
 * @max_num_of_encaps: Maximum number of encapsulations.
 * @nic_macro_idx: macro index of this specific port.
 * @nic_phys_port_idx: physical port index (AKA lane) of this specific port.
 */
struct hlibdv_query_port_attr {
	uint32_t max_num_of_qps;
	uint32_t num_allocated_qps;
	uint32_t max_allocated_qp_num;
	uint32_t max_cq_size;
	uint32_t max_num_of_scale_out_coll_qps;
	uint32_t max_num_of_coll_qps;
	uint32_t base_scale_out_coll_qp_num;
	uint32_t base_coll_qp_num;
	uint32_t coll_qps_offset;
	uint8_t advanced;
	uint8_t max_num_of_cqs;
	uint8_t max_num_of_usr_fifos;
	uint8_t max_num_of_encaps;
	uint8_t nic_macro_idx;
	uint8_t nic_phys_port_idx;
};

/**
 * struct hlibdv_qp_attr - HL QP attributes.
 * @local_key: Unique key for local memory access. Needed for RTR state.
 * @remote_key: Unique key for remote memory access. Needed for RTS state.
 * @congestion_wnd: Congestion-Window size. Needed for RTS state.
 * @qp_num_hint: Explicitly request for specified QP id, valid for collective QPs.
 *               Needed for INIT state.
 * @dest_wq_size: Number of WQEs on the destination. Needed for RDV RTS state.
 * @wq_type: WQ type. e.g. write, rdv etc. Needed for INIT state.
 * @wq_granularity: WQ granularity [0 for 32B or 1 for 64B]. Needed for INIT state.
 * @priority: QoS priority. Needed for RTR and RTS state.
 * @loopback: QP loopback enable/disable. Needed for RTR and RTS state.
 * @congestion_en: Congestion-control enable/disable. Needed for RTS state.
 * @coll_lag_idx: NIC index within LAG. Needed for collective QP RTS state.
 * @coll_last_in_lag: If last NIC in LAG. Needed for collective QP RTS state.
 * @compression_en: Compression enable/disable. Needed for RTS state.
 * @sack_en: Selective acknowledgment enable/disable. Needed for RTR and RTS state.
 * @encap_en: Encapsulation enable flag. Needed for RTS and RTS state.
 * @encap_num: Encapsulation ID. Needed for RTS and RTS state.
 * @is_coll: Is a collective QP. Needed for INIT state.
 */
struct hlibdv_qp_attr {
	uint32_t local_key;
	uint32_t remote_key;
	uint32_t congestion_wnd;
	uint32_t qp_num_hint;
	uint32_t dest_wq_size;
	enum hlibdv_qp_wq_types wq_type;
	enum hlibdv_swq_granularity wq_granularity;
	uint8_t priority;
	uint8_t loopback;
	uint8_t congestion_en;
	uint8_t coll_lag_idx;
	uint8_t coll_last_in_lag;
	uint8_t compression_en;
	uint8_t sack_en;
	uint8_t encap_en;
	uint8_t encap_num;
	uint8_t is_coll;
};

/**
 * struct hlibdv_qp_attr_tmp - HL QP attributes.
 * @caps: QP capabilities bit-mask.
 * @local_key: Unique key for local memory access. Needed for RTR state.
 * @remote_key: Unique key for remote memory access. Needed for RTS state.
 * @congestion_wnd: Congestion-Window size. Needed for RTS state.
 * @qp_num_hint: Explicitly request for specified QP id, valid for collective QPs.
 *               Needed for INIT state.
 * @dest_wq_size: Number of WQEs on the destination. Needed for RDV RTS state.
 * @wq_type: WQ type. e.g. write, rdv etc. Needed for INIT state.
 * @wq_granularity: WQ granularity [0 for 32B or 1 for 64B]. Needed for INIT state.
 * @priority: QoS priority. Needed for RTR and RTS state.
 * @coll_lag_idx: NIC index within LAG. Needed for collective QP RTS state.
 * @coll_last_in_lag: If last NIC in LAG. Needed for collective QP RTS state.
 * @encap_num: Encapsulation ID. Needed for RTS and RTS state.
 */
struct hlibdv_qp_attr_tmp {
	uint64_t caps;
	uint32_t local_key;
	uint32_t remote_key;
	uint32_t congestion_wnd;
	uint32_t qp_num_hint;
	uint32_t dest_wq_size;
	enum hlibdv_qp_wq_types wq_type;
	enum hlibdv_swq_granularity wq_granularity;
	uint8_t priority;
	uint8_t coll_lag_idx;
	uint8_t coll_last_in_lag;
	uint8_t encap_num;
};

/**
 * struct hlibdv_query_qp_attr - Queried HL QP data.
 * @qp_num: HL QP num.
 * @swq_cpu_addr: Send WQ mmap address.
 * @rwq_cpu_addr: Receive WQ mmap address.
 */
struct hlibdv_query_qp_attr {
	uint32_t qp_num;
	void *swq_cpu_addr;
	void *rwq_cpu_addr;
};

/**
 * struct hlibdv_usr_fifo_attr - HL user FIFO attributes.
 * @port_num: Port ID (should be 1-based).
 * @base_sob_addr: Base address of the sync object.
 * @num_sobs: Number of sync objects.
 * @usr_fifo_num_hint: Hint to allocate a specific usr_fifo HW resource.
 * @mode: FIFO Operation mode.
 * @dir_dup_mask: (Gaudi3 and above) Ports for which the HW should duplicate the direct patcher
 *                descriptor.
 */
struct hlibdv_usr_fifo_attr {
	uint32_t port_num;
	uint32_t base_sob_addr;
	uint32_t num_sobs;
	uint32_t usr_fifo_num_hint;
	uint8_t mode;
	uint8_t dir_dup_mask;
};

/**
 * struct hlibdv_usr_fifo_attr_tmp - HL user FIFO attributes.
 * @port_num: Port ID (should be 1-based).
 * @base_sob_addr: Base address of the sync object.
 * @num_sobs: Number of sync objects.
 * @usr_fifo_num_hint: Hint to allocate a specific usr_fifo HW resource.
 * @usr_fifo_type: FIFO Operation mode.
 * @dir_dup_mask: (Gaudi3 and above) Ports for which the HW should duplicate the direct patcher
 *                descriptor.
 */
struct hlibdv_usr_fifo_attr_tmp {
	uint32_t port_num;
	uint32_t base_sob_addr;
	uint32_t num_sobs;
	uint32_t usr_fifo_num_hint;
	enum hlibdv_usr_fifo_type usr_fifo_type;
	uint8_t dir_dup_mask;
};

/**
 * struct hlibdv_usr_fifo - HL user FIFO.
 * @ci_cpu_addr: CI mmap address.
 * @regs_cpu_addr: UMR mmap address.
 * @regs_offset: UMR offset.
 * @usr_fifo_num: DB FIFO ID.
 * @size: Allocated FIFO size.
 * @bp_thresh: Backpressure threshold that was set by the driver.
 */
struct hlibdv_usr_fifo {
	void *ci_cpu_addr;
	void *regs_cpu_addr;
	uint32_t regs_offset;
	uint32_t usr_fifo_num;
	uint32_t size;
	uint32_t bp_thresh;
};

/**
 * struct hlibdv_cq_attr - HL CQ attributes.
 * @port_num: Port number to which CQ is associated (should be 1-based).
 * @cq_type: Type of CQ to be allocated.
 */
struct hlibdv_cq_attr {
	uint8_t port_num;
	enum hlibdv_cq_type cq_type;
};

/**
 * struct hlibdv_cq - HL CQ.
 * @ibvcq: Verbs CQ.
 * @mem_cpu_addr: CQ buffer address.
 * @pi_cpu_addr: CQ PI memory address.
 * @regs_cpu_addr: CQ UMR address.
 * @cq_size: Size of the CQ.
 * @cq_num: CQ number that is allocated.
 * @regs_offset: CQ UMR reg offset.
 */
struct hlibdv_cq {
	struct ibv_cq *ibvcq;
	void *mem_cpu_addr;
	void *pi_cpu_addr;
	void *regs_cpu_addr;
	uint32_t cq_size;
	uint32_t cq_num;
	uint32_t regs_offset;
};

/**
 * struct hlibdv_query_cq_attr - HL CQ.
 * @ibvcq: Verbs CQ.
 * @mem_cpu_addr: CQ buffer address.
 * @pi_cpu_addr: CQ PI memory address.
 * @regs_cpu_addr: CQ UMR address.
 * @cq_size: Size of the CQ.
 * @cq_num: CQ number that is allocated.
 * @regs_offset: CQ UMR reg offset.
 * @cq_type: Type of CQ resource.
 */
struct hlibdv_query_cq_attr {
	struct ibv_cq *ibvcq;
	void *mem_cpu_addr;
	void *pi_cpu_addr;
	void *regs_cpu_addr;
	uint32_t cq_size;
	uint32_t cq_num;
	uint32_t regs_offset;
	enum hlibdv_cq_type cq_type;
};

/**
 * struct hlibdv_coll_qp_attr - HL Collective QP attributes.
 * @is_scale_out: Is this collective connection for scale out.
 */
struct hlibdv_coll_qp_attr {
	uint8_t is_scale_out;
};

/**
 * struct hlibdv_coll_qp - HL Collective QP.
 * @qp_num: collective qp num.
 */
struct hlibdv_coll_qp {
	uint32_t qp_num;
};

/**
 * struct hlibdv_encap_attr - HL encapsulation specific attributes.
 * @tnl_hdr_ptr: Pointer to the tunnel encapsulation header. i.e. specific tunnel header data to be
 *               used in the encapsulation by the HW.
 * @tnl_hdr_size: Tunnel encapsulation header size.
 * @ipv4_addr: Source IP address, set regardless of encapsulation type.
 * @port_num: Port ID (should be 1-based).
 * @udp_dst_port: The UDP destination-port. Valid for L4 tunnel.
 * @ip_proto: IP protocol to use. Valid for L3 tunnel.
 * @encap_type: Encapsulation type. May be either no-encapsulation or encapsulation over L3 or L4.
 */
struct hlibdv_encap_attr {
	uint64_t tnl_hdr_ptr;
	uint32_t tnl_hdr_size;
	uint32_t ipv4_addr;
	uint32_t port_num;
	union {
		uint16_t udp_dst_port;
		uint16_t ip_proto;
	};
	uint8_t encap_type;
};

/**
 * struct hlibdv_encap_attr_tmp - HL encapsulation specific attributes.
 * @tnl_hdr_ptr: Pointer to the tunnel encapsulation header. i.e. specific tunnel header data to be
 *               used in the encapsulation by the HW.
 * @tnl_hdr_size: Tunnel encapsulation header size.
 * @ipv4_addr: Source IP address, set regardless of encapsulation type.
 * @port_num: Port ID (should be 1-based).
 * @udp_dst_port: The UDP destination-port. Valid for L4 tunnel.
 * @ip_proto: IP protocol to use. Valid for L3 tunnel.
 * @encap_type: Encapsulation type. May be either no-encapsulation or encapsulation over L3 or L4.
 */
struct hlibdv_encap_attr_tmp {
	uint64_t tnl_hdr_ptr;
	uint32_t tnl_hdr_size;
	uint32_t ipv4_addr;
	uint32_t port_num;
	union {
		uint16_t udp_dst_port;
		uint16_t ip_proto;
	};
	enum hlibdv_encap_type encap_type;
};

/**
 * struct hlibdv_encap - HL DV encapsulation data.
 * @encap_num: HW encapsulation number.
 */
struct hlibdv_encap {
	uint32_t encap_num;
};

/**
 * struct hlibdv_cc_cq_attr - HL congestion control CQ attributes.
 * @port_num: Port ID (should be 1-based).
 * @num_of_cqes: Number of CQ elements in CQ.
 */
struct hlibdv_cc_cq_attr {
	uint32_t port_num;
	uint32_t num_of_cqes;
};

/**
 * struct hlibdv_cc_cq - HL congestion control CQ.
 * @mem_cpu_addr: CC CQ memory mmap address.
 * @pi_cpu_addr: CC CQ PI mmap address.
 * @cqe_size: CC CQ entry size.
 * @num_of_cqes: Number of CQ elements in CQ.
 */
struct hlibdv_cc_cq {
	void *mem_cpu_addr;
	void *pi_cpu_addr;
	size_t cqe_size;
	uint32_t num_of_cqes;
};

/**
 * struct hlibdv_coll_sched_resource - Collective Scheduler resource.
 * @type: Type of the resource.
 * @id: ID of the NMS to whom the resource belongs.
 * @size: The size of the resource.
 * @pa_offs: LBW address of the resource.
 * @virt_addr: Address of mmapped resource.
 */
struct hlibdv_coll_sched_resource {
	enum hlibdv_coll_sched_resource_type type;
	uint8_t id;
	size_t size;
	uint32_t pa_offs;
	void *virt_addr;
};

/**
 * struct hlibdv_coll_sched_resource - Collective Scheduler resource.
 * @num_resources: How many resources have been allocated.
 * @rsrc: Array of allocated resources.
 */
struct hlibdv_coll_sched_resources {
	int num_resources;
	struct hlibdv_coll_sched_resource rsrc[HLIBDV_MAX_NUM_COLL_SCHED_RESOURCES];
};

/**
 * struct hlibdv_device_attr - Devie specific attributes.
 * @cap_mask: Capabilities mask.
 */
struct hlibdv_device_attr {
	uint64_t cap_mask;
};

bool hlibdv_is_supported(struct ibv_device *device);
struct ibv_context *hlibdv_open_device(struct ibv_device *device,
				       struct hlibdv_ucontext_attr *attr);
int hlibdv_set_port_ex(struct ibv_context *context, struct hlibdv_port_ex_attr *attr);
int hlibdv_set_port_ex_tmp(struct ibv_context *context, struct hlibdv_port_ex_attr_tmp *attr);
/* port_num should be 1-based */
int hlibdv_query_port(struct ibv_context *context, uint32_t port_num,
		      struct hlibdv_query_port_attr *hl_attr);
int hlibdv_modify_qp(struct ibv_qp *ibqp, struct ibv_qp_attr *attr, int attr_mask,
		     struct hlibdv_qp_attr *hl_attr);
int hlibdv_modify_qp_tmp(struct ibv_qp *ibqp, struct ibv_qp_attr *attr, int attr_mask,
		     struct hlibdv_qp_attr_tmp *hl_attr);
struct hlibdv_usr_fifo *hlibdv_create_usr_fifo(struct ibv_context *context,
					       struct hlibdv_usr_fifo_attr *attr);
struct hlibdv_usr_fifo *hlibdv_create_usr_fifo_tmp(struct ibv_context *context,
					       struct hlibdv_usr_fifo_attr_tmp *attr);
int hlibdv_destroy_usr_fifo(struct hlibdv_usr_fifo *usr_fifo);
struct ibv_cq *hlibdv_create_cq(struct ibv_context *context, int cqe,
				struct ibv_comp_channel *channel, int comp_vector,
				struct hlibdv_cq_attr *cq_attr);
int hlibdv_query_cq(struct ibv_cq *ibvcq, struct hlibdv_query_cq_attr *hl_cq);
int hlibdv_query_qp(struct ibv_qp *ibvqp, struct hlibdv_query_qp_attr *qp_attr);
struct hlibdv_encap *hlibdv_create_encap(struct ibv_context *context,
					 struct hlibdv_encap_attr *encap_attr);
struct hlibdv_encap *hlibdv_create_encap_tmp(struct ibv_context *context,
					 struct hlibdv_encap_attr_tmp *encap_attr);
int hlibdv_destroy_encap(struct hlibdv_encap *hl_encap);
int hlibdv_reserve_coll_qps(struct ibv_pd *ibvpd, struct hlibdv_coll_qp_attr *coll_qp_attr,
			    struct hlibdv_coll_qp *coll_qp);
int hlibdv_reserve_coll_sched_resources(struct ibv_context *context,
					struct hlibdv_coll_sched_resources *sched_resrc);
int hlibdv_query_device(struct ibv_context *context, struct hlibdv_device_attr *attr);

#ifdef __cplusplus
}
#endif

#endif /* __HLIBDV_H__ */

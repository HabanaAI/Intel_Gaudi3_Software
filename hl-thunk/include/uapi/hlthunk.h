/* SPDX-License-Identifier: MIT
 *
 * Copyright 2019 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef HLTHUNK_H
#define HLTHUNK_H

#ifdef __cplusplus
extern "C" {
#endif

#include "drm/habanalabs_accel.h"

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define hlthunk_public  __attribute__((visibility("default")))

#define HLTHUNK_MAX_MINOR		256
#define HLTHUNK_DEV_NAME_PRIMARY	"/dev/accel/accel%d"
#define HLTHUNK_DEV_NAME_CONTROL	"/dev/accel/accel_controlD%d"
#define HLTHUNK_DEV_NAME_XE		"/dev/dri/renderD%d"

#define HLTHUNK_BUSY_ENGINES_MASK_SIZE	4

#define PCI_IDS_FS1			0x0b73

enum hlthunk_node_type {
	HLTHUNK_NODE_PRIMARY,
	HLTHUNK_NODE_CONTROL,
	HLTHUNK_NODE_XE,
	HLTHUNK_NODE_MAX
};

enum hlthunk_device_name {
	HLTHUNK_DEVICE_GOYA = 0,
	HLTHUNK_DEVICE_GRECO = 1,
	HLTHUNK_DEVICE_GAUDI = 2,
	HLTHUNK_DEVICE_INVALID = 3,
	HLTHUNK_DEVICE_DONT_CARE = 4,
	HLTHUNK_DEVICE_GAUDI2 = 5,
	HLTHUNK_DEVICE_GAUDI_HL2000M = 6,
	HLTHUNK_DEVICE_GAUDI3 = 7,
	HLTHUNK_DEVICE_GAUDI2B = 8,
	HLTHUNK_DEVICE_FS1 = 9,
	HLTHUNK_DEVICE_GAUDI2C = 10,
	HLTHUNK_DEVICE_GAUDI2D = 11,
	HLTHUNK_DEVICE_MAX
};

enum hlthunk_event_record_id {
	HLTHUNK_OPEN_DEV,
	HLTHUNK_CS_TIMEOUT,
	HLTHUNK_RAZWI_EVENT,
	HLTHUNK_UNDEFINED_OPCODE,
	HLTHUNK_HW_ERR_OPCODE,
	HLTHUNK_FW_ERR_OPCODE,
	HLTHUNK_ENGINE_EVENT,
};

/**
 * struct hlthunk_hw_ip_info - hardware information on various IPs in the ASIC
 * @sram_base_address: The first SRAM physical base address that is free to be
 *                     used by the user.
 * @dram_base_address: The first DRAM virtual or physical base address that is
 *                     free to be used by the user.
 * @dram_size: The DRAM size that is available to the user.
 * @sram_size: The SRAM size that is available to the user.
 * @num_of_events: The number of events that can be received from the f/w. This
 *                 is needed so the user can what is the size of the h/w events
 *                 array he needs to pass to the kernel when he wants to fetch
 *                 the event counters.
 * @device_id: PCI device ID of the ASIC.
 * @cpld_version: CPLD version on the board.
 * @psoc_pci_pll_nr: PCI PLL NR value. Needed by the profiler in some ASICs.
 * @psoc_pci_pll_nf: PCI PLL NF value. Needed by the profiler in some ASICs.
 * @psoc_pci_pll_od: PCI PLL OD value. Needed by the profiler in some ASICs.
 * @psoc_pci_pll_div_factor: PCI PLL DIV factor value. Needed by the profiler
 *                           in some ASICs.
 * @tpc_enabled_mask: Bit-mask that represents which TPCs are enabled. Relevant
 *                    for Goya/Gaudi only.
 * @dram_enabled: Whether the DRAM is enabled.
 * @cpucp_version: The CPUCP f/w version.
 * @module_id: Module ID of the ASIC for mezzanine cards in servers
 *             (From OCP spec).
 * @card_name: The card name as passed by the f/w.
 * @decoder_enabled_mask: Bit-mask that represents which decoders are enabled.
 * @mme_master_slave_mode: Indicate whether the MME is working in master/slave
 *                         configuration. Relevant for Gaudi2 and later.
 * @tpc_enabled_mask_ext: Bit-mask that represents which TPCs are enabled.
 *                        Relevant for Gaudi2 and later.
 * @dram_page_size: The DRAM physical page size.
 * @first_available_interrupt_id: The first available interrupt ID for the user
 *                                to be used when it works with user interrupts.
 *                                Relevant for Gaudi2 and later.
 * @edma_enabled_mask: Bit-mask that represents which EDMAs are enabled.
 *                     Relevant for Gaudi2 and later.
 * @server_type: Server type that the Gaudi ASIC is currently installed in.
 *               The value is according to enum hl_server_type
 * @pdma_user_owned_ch_mask: Bit-mask that represents which PDMA channels are
 *                     enabled and can be used by the user.
 *                     Relevant for Gaudi3 and later.
 * @number_of_user_interrupts: The number of interrupts that are available to the userspace
 *                             application to use. Relevant for Gaudi2 and later.
 * @device_mem_alloc_default_page_size: default page size used in device memory allocation.
 * @nic_ports_mask: Bit-mask that represents the nic ports that are enabled.
 * @interposer_version: Interposer version.
 * @substrate_version: Substrate version.
 * @nic_ports_external_mask: Bit mask that represents the nic ports that are external - used for
 *                           scale-out and are exposed to Linux as network devices.
 * @mme_enabled_mask: Bit-mask that represents which MMEs are enabled.
 * @odp_supported: true if ODP is supported, otherwise false.
 * @security_enabled: true if security is enabled on device.
 * @revision_id: PCI revision ID of the ASIC.
 * @tpc_interrupt_id: interrupt id for TPC to use in order to raise events toward host.
 * @engine_core_interrupt_reg_addr: interrupt register address for engine core to use
 *                                  in order to raise events toward FW.
 * @rotator_enabled_mask: Bit-mask that represents which rotators are enabled.
 * @sched_arc_enabled_mask: Bit-mask that represents which arc sched are enabled.
 * @reserved_dram_size: DRAM size reserved for driver and firmware.
 */
struct hlthunk_hw_ip_info {
	uint64_t sram_base_address;
	uint64_t dram_base_address;
	uint64_t dram_size;
	uint32_t sram_size;
	uint32_t num_of_events;
	uint32_t device_id;
	uint32_t cpld_version;
	uint32_t psoc_pci_pll_nr;
	uint32_t psoc_pci_pll_nf;
	uint32_t psoc_pci_pll_od;
	uint32_t psoc_pci_pll_div_factor;
	uint16_t tpc_enabled_mask;
	uint8_t dram_enabled;
	uint8_t cpucp_version[HL_INFO_VERSION_MAX_LEN];
	uint32_t module_id;
	uint8_t card_name[HL_INFO_CARD_NAME_MAX_LEN];
	uint32_t decoder_enabled_mask;
	uint8_t mme_master_slave_mode;
	uint64_t tpc_enabled_mask_ext;
	uint64_t dram_page_size;
	uint16_t first_available_interrupt_id;
	uint32_t edma_enabled_mask;
	uint16_t server_type;
	uint64_t pdma_user_owned_ch_mask;
	uint16_t number_of_user_interrupts;
	uint64_t device_mem_alloc_default_page_size;
	uint64_t nic_ports_mask;
	uint8_t interposer_version;
	uint8_t substrate_version;
	uint64_t nic_ports_external_mask;
	uint32_t mme_enabled_mask;
	uint8_t odp_supported;
	uint8_t security_enabled;
	uint8_t revision_id;
	uint16_t tpc_interrupt_id;
	uint64_t engine_core_interrupt_reg_addr;
	uint32_t rotator_enabled_mask;
	uint32_t sched_arc_enabled_mask;
	uint64_t reserved_dram_size;
};

struct hlthunk_dram_usage_info {
	uint64_t dram_free_mem;
	uint64_t ctx_dram_mem;
};

struct hlthunk_mac_addr {
	uint8_t addr[ETH_ALEN];
};

struct hlthunk_mac_addr_info {
	struct hlthunk_mac_addr array[HL_INFO_MAC_ADDR_MAX_NUM];
	uint64_t mask[2];
};

struct hlthunk_engines_idle_info {
	uint32_t is_idle;
	uint32_t pad;
	uint64_t mask[HLTHUNK_BUSY_ENGINES_MASK_SIZE];
};

struct hlthunk_pll_frequency_info {
	uint16_t output[HL_PLL_NUM_OUTPUTS];
};

struct hlthunk_reset_count_info {
	uint32_t hard_reset_count;
	uint32_t soft_reset_count;
};

struct hlthunk_time_sync_info {
	uint64_t device_time;
	uint64_t host_time;
	uint64_t tsc_time;
};

struct hlthunk_sync_manager_info {
	uint32_t first_available_sync_object;
	uint32_t first_available_monitor;
	uint32_t first_available_cq;
	uint32_t reserved;
};

struct hlthunk_pci_counters_info {
	uint64_t rx_throughput;
	uint64_t tx_throughput;
	uint32_t replay_cnt;
};

/* clk_throttling_reason masks */
#define HLTHUNK_CLK_THROTTLE_POWER	(1 << HL_CLK_THROTTLE_TYPE_POWER)
#define HLTHUNK_CLK_THROTTLE_THERMAL	(1 << HL_CLK_THROTTLE_TYPE_THERMAL)

struct hlthunk_clk_throttle_info {
	uint32_t clk_throttle_reason_bitmask;
	uint64_t clk_throttle_start_timestamp_us[HL_CLK_THROTTLE_TYPE_MAX];
	uint64_t clk_throttle_duration_ns[HL_CLK_THROTTLE_TYPE_MAX];
};

struct hlthunk_energy_info {
	uint64_t total_energy_consumption;
};

struct hlthunk_power_info {
	uint64_t power;
};

struct hlthunk_cs_in {
	void *chunks_restore;
	void *chunks_execute;
	uint32_t num_chunks_restore;
	uint32_t num_chunks_execute;
	uint32_t flags;
};

struct hlthunk_cs_out {
	uint64_t seq;
	uint32_t status;
	uint16_t sob_count_before_submission;
};

struct hlthunk_signal_in {
	void *chunks_restore;
	uint32_t num_chunks_restore;
	uint32_t queue_index;
	uint32_t flags;
};

struct hlthunk_signal_out {
	uint64_t seq;
	uint32_t status;
	uint32_t sob_base_addr_offset;
	uint16_t sob_count_before_submission;
};

struct hlthunk_sig_res_in {
	uint32_t queue_index;
	uint32_t count;
};

struct reserve_sig_handle {
	uint32_t id;
	uint32_t sob_base_addr_offset;
	uint32_t count;
};

struct hlthunk_sig_res_out {
	struct reserve_sig_handle handle;
	uint32_t status;
};

struct hlthunk_wait_for_signal_data {
	union {
		uint64_t *signal_seq_arr;
		uint64_t encaps_signal_seq;
	};
	uint32_t signal_seq_nr; /* value of 1 is currently supported */
	uint32_t queue_index;
	uint32_t flags; /* currently unused */
	uint32_t collective_engine_id;
	uint32_t encaps_signal_offset;
};

struct hlthunk_wait_in {
	void *chunks_restore;
	uint32_t num_chunks_restore;
	uint64_t *hlthunk_wait_for_signal;
	uint32_t num_wait_for_signal; /* value of 1 is currently supported */
	uint32_t flags;
};

struct hlthunk_wait_out {
	uint64_t seq;
	uint32_t status;
};

enum wq_types {
	WQ_WRITE = 1,
	WQ_RENDEZVOUS_READ = 2,
	WQ_RENDEZVOUS_WRITE = 3
};

/**
 * struct hlthunk_requester_conn_ctx - set up a requester connection context
 * @dst_ip_addr: Destination IP address in native endianness
 * @dst_conn_id: Destination connection ID
 * @last_index: Index of last entry [2..(2^22)-1]. NOTE: relevant for Gaudi1 (only)
 * @dst_mac_addr: Destination MAC address
 * @priority: Connection priority [0..3]
 * @timer_granularity: Timer granularity [0..127]
 * @swq_granularity: SWQ granularity [0 for 32B or 1 for 64B]
 * @wq_type: Work queue type [1..3]
 * @cq_number: Completion queue number
 * @wq_remote_log_size: Remote Work queue log size [2^QPC] Rendezvous
 * @congestion_wnd: Congestion-Window size
 * @mtu: Max Transmit Unit
 * @congestion_en: Enable/disable Congestion-Control
 * @encap_en: used as boolean; indicates if this QP has encapsulation support
 * @encap_id: Encapsulation-id; valid only if 'encap_en' is set
 * @loopback: used as boolean; indicates if this QP used for loopback mode.
 * @wq_size: Max number of elements in the work queue. NOTE: relevant for Gaudi2 (or higher)
 * @coll_lag_idx: (Gaudi3 and above) The index of the specific NIC within the LAG. Note that the
 *                advanced flag must be enabled in case it's being set.
 * @coll_last_in_lag: (Gaudi3 and above) Is the specific NIC the last one within the collective LAG.
 *                    Note that the advanced flag must be enabled in case it's being set.
 * @compression_en: Enable compression
 * @remote_key: Remote-key to be used to generate on outgoing packets
 * @sack_en: (Gaudi3 and above) Enable Selective Acknowlegment (SACK)
 */
struct hlthunk_requester_conn_ctx {
	uint32_t deprecated;
	uint32_t dst_ip_addr;
	uint32_t dst_conn_id;
	uint32_t deprecated1;
	uint32_t last_index;
	uint8_t dst_mac_addr[ETH_ALEN];
	uint8_t deprecated2;
	uint8_t priority;
	uint8_t deprecated3;
	uint8_t timer_granularity;
	uint8_t swq_granularity;
	uint8_t wq_type;
	uint8_t deprecated4;
	uint8_t cq_number;
	uint8_t wq_remote_log_size;
	uint32_t congestion_wnd;
	uint16_t mtu;
	uint8_t congestion_en;
	uint8_t encap_en;
	uint8_t encap_id;
	uint8_t loopback;
	uint32_t wq_size;
	uint8_t coll_lag_idx;
	uint8_t coll_last_in_lag;
	uint8_t compression_en;
	uint32_t remote_key;
	uint8_t sack_en;
};

struct hlthunk_requester_conn_ctx_out {
	uint64_t swq_mem_handle;
	uint64_t rwq_mem_handle;
};

/**
 * struct hlthunk_responder_conn_ctx - set up a responder connection context
 * @dst_ip_addr: Destination IP address in native endianness
 * @dst_conn_id: Destination connection ID
 * @port: NIC port ID
 * @conn_id: Connection ID
 * @dst_mac_addr: Destination MAC address
 * @priority: Connection priority [0..3]
 * @wq_peer_granularity: Work queue granularity
 * @cq_number: Completion queue number
 * @conn_peer: Connection peer
 * @rdv: used as boolean; indicates if this QP is RDV (WRITE or READ)
 * @loopback: used as boolean; indicates if this QP used for loopback mode.
 * @wq_peer_size: size of the peer Work queue
 * @encap_en: used as boolean; indicates if this QP has encapsulation support
 * @encap_id: Encapsulation-id; valid only if 'encap_en' is set
 * @local_key: Local-key to be used to validate against incoming packets
 * @sack_en: (Gaudi3 and above) Enable Selective Acknowlegment (SACK)
 */
struct hlthunk_responder_conn_ctx {
	uint32_t deprecated;
	uint32_t dst_ip_addr;
	uint32_t dst_conn_id;
	uint8_t dst_mac_addr[ETH_ALEN];
	uint8_t priority;
	uint8_t deprecated1;
	uint8_t deprecated2;
	uint8_t wq_peer_granularity;
	uint8_t cq_number;
	uint8_t deprecated3;
	uint32_t conn_peer;
	uint8_t rdv;
	uint8_t loopback;
	uint32_t wq_peer_size;
	uint8_t encap_en;
	uint8_t encap_id;
	uint32_t local_key;
	uint8_t sack_en;
};

struct hlthunk_nic_wq_arr_set_in {
	uint32_t port;
	uint64_t addr;
	uint32_t num_of_wqs;
	uint32_t num_of_wq_entries;
	uint32_t type;
	enum hl_nic_mem_id mem_id;
	uint8_t swq_granularity;
};

struct hlthunk_nic_wq_arr_set_out {
	uint64_t handle;
};

struct hlthunk_nic_user_cq_id_alloc_in {
	uint32_t port;
};

struct hlthunk_nic_user_cq_id_alloc_out {
	uint32_t id;
};

struct hlthunk_nic_user_cq_id_set_in {
	uint32_t port;
	uint32_t num_of_cqes;
	uint32_t id;
};

struct hlthunk_nic_user_cq_id_set_out {
	uint64_t mem_handle;
	uint64_t pi_handle;
	uint64_t regs_handle;
	uint32_t regs_offset;
};

struct hlthunk_nic_user_cq_id_unset_in {
	uint32_t port;
	uint64_t id;
};

struct hlthunk_nic_alloc_coll_conn_in {
	uint8_t is_scale_out;
};

struct hlthunk_encap_cfg {
	uintptr_t tnl_hdr_ptr;
	uint32_t tnl_hdr_size;
	uint32_t port;
	uint32_t id;
	uint32_t src_ipv4_addr;
	enum hl_nic_encap_type encap_type;
	union {
		uint16_t udp_dst_port;
		uint16_t ip_proto;
	};
};

struct hlthunk_open_stats_info {
	uint64_t open_counter;
	uint64_t last_open_period_ms;
	uint8_t is_compute_ctx_active;
	uint8_t compute_ctx_in_release;
};

struct hlthunk_hw_asic_status {
	struct hlthunk_clk_throttle_info throttle;
	struct hlthunk_open_stats_info open_stats;
	struct hlthunk_power_info power;
	enum hl_device_status status;
	uint64_t timestamp_sec;
	uint8_t valid;
};

/**
 * struct hlthunk_nic_user_set_app_params_in -Allow the user application to set general parameters
 *                                        regarding the RDMA nic operation. These parameters stay
 *                                        in effect until the application releases the device.
 * @advanced: A boolean that indicates whether this WQ should support advanced operations, such as
 *            RDV, QMan, WTD, etc.
 * @bp_offs: Offsets in NIC memory to signal a back pressure. Note that the advanced flag
 *              must be enabled in case it's being set.
 * @fna_mask_size: Determines the width of the completion value.
 * @fna_fifo_offs: SRAM/DCCM addresses provided to the HW by the user when FnA completion is
 *              configured in the SRAM/DDCM.
 */
struct hlthunk_nic_user_set_app_params_in {
	__u8 advanced;
	__u32 bp_offs[HL_NIC_USER_BP_OFFS_MAX];
	__u8 deprecated;
	__u8 fna_mask_size;
	__u32 fna_fifo_offs[HL_NIC_FNA_CMPL_ADDR_NUM];
};

struct hlthunk_nic_user_get_app_params_out {
	uint32_t max_num_of_qps;
	uint32_t num_allocated_qps;
	uint32_t max_allocated_qp_idx;
	uint32_t max_cq_size;
	uint8_t advanced;
	uint8_t max_num_of_cqs;
	uint8_t max_num_of_db_fifos;
	uint8_t max_num_of_encaps;
	uint32_t speed;
	uint32_t max_num_of_coll_qps;
	uint32_t coll_qps_offset;
	uint32_t base_coll_qp_idx;
	uint32_t base_scale_out_coll_qp_idx;
	uint32_t max_num_of_scale_out_coll_qps;
};

struct hlthunk_wait_multi_cs_in {
	uint64_t *seq;
	uint64_t timeout_us;
	uint32_t seq_len;
};

struct hlthunk_wait_multi_cs_out {
	uint32_t status;
	uint32_t seq_set;
	uint8_t completed;
};

struct hlthunk_nic_eq_poll_out {
	/* holds HL_NIC_EQ_POLL_STATUS_* */
	uint32_t poll_status;
	uint32_t idx;
	uint32_t ev_data;
	/* holds HL_NIC_EQ_EVENT_TYPE_* */
	uint8_t ev_type;
};

/**
 * struct hlthunk_get_habana_link_state_in - get PCS link state of Habana link internal port.
 * @port: the port to probe.
 */
struct hlthunk_get_habana_link_state_in {
	uint32_t port;
	uint32_t pad;
};

/**
 * struct hlthunk_get_habana_link_state_out - PCS link state of the probed Habana link internal
 * port.
 * @up: state of the PCS link.
 */
struct hlthunk_get_habana_link_state_out {
	uint8_t up;
	uint8_t pad[7];
};

/**
 * struct hlthunk_get_habana_link_stat_in - info for getting a Habana link internal port statistics.
 * @port: the port to probe.
 */
struct hlthunk_get_habana_link_stat_in {
	uint32_t port;
};

/**
 * struct hlthunk_get_habana_link_stat_out - info for result of getting a Habana link internal port
 * statistics.
 * @str_buf: buffer allocated by the user to be filled with the counters names. Recommended size is
 *           HABANA_LINK_STR_LEN * HABANA_LINK_CNT_MAX_NUM.
 * @val_buf: buffer allocated by the user to be filled with the counters values. Recommended size is
 *           sizeof(__u64) * HABANA_LINK_CNT_MAX_NUM.
 * @num_of_stat: the actual number of counters that were retrieved.
 */
struct hlthunk_get_habana_link_stat_out {
	uint8_t *str_buf;
	uint64_t *val_buf;
	uint32_t num_of_stat;
};

/**
 * struct hlthunk_nic_get_ports_masks_out - ports mask info.
 * @ports_mask: Mask of available ports.
 * @ext_ports_mask: Mask of external ports (subset of ports_mask).
 */
struct hlthunk_nic_get_ports_masks_out {
	uint64_t ports_mask;
	uint64_t ext_ports_mask;
};

/* DRAM replaced rows related data structures */
#define DRAM_ROW_REPLACE_MAX	32

enum hlthunk_dram_row_replace_cause {
	HLTHUNK_ROW_REPLACE_CAUSE_DOUBLE_ECC_ERR,
	HLTHUNK_ROW_REPLACE_CAUSE_MULTI_SINGLE_ECC_ERR,
};

struct hlthunk_dram_row_info {
	uint8_t dram_idx;
	uint8_t pc;
	uint8_t sid;
	uint8_t bank_idx;
	uint16_t row_addr;
	uint8_t replaced_row_cause; /* enum hlthunk_dram_row_replace_cause */
	uint8_t pad;
};

/*
 * struct hlthunk_dram_replaced_rows_info -
 * @num_replaced_rows: number of replaced rows.
 * @replaced_rows: replaced rows info.
 */
struct hlthunk_dram_replaced_rows_info {
	uint16_t num_replaced_rows;
	uint8_t pad[6];
	struct hlthunk_dram_row_info replaced_rows[DRAM_ROW_REPLACE_MAX];
};

/**
 * struct hlthunk_event_record_open_dev_time - timestamp of last time device was opened and
 *                                             CS timeout or razwi error occurred.
 * @timestamp: timestamp of device open.
 */
struct hlthunk_event_record_open_dev_time {
	int64_t timestamp;
};

/**
 * struct hlthunk_event_record_cs_timeout - last CS timeout information.
 * @timestamp: timestamp when last CS timeout event occurred.
 * @seq: sequence number of last CS timeout event.
 */
struct hlthunk_event_record_cs_timeout {
	int64_t timestamp;
	uint64_t seq;
};

/**
 * struct hlthunk_nic_user_ccq_set_in - info for cc-q creation
 * @port: the port associated with this ccq
 * @num_of_entries: number of entries in the CCQ buffer
 */
struct hlthunk_nic_user_ccq_set_in {
	uint32_t port;
	uint32_t num_of_entries;
};

/**
 * struct hlthunk_nic_user_ccq_set_out - info for mapping ccq
 * @ccq_handle: handle for mapping NIC CCQ buffer
 * @ccq_pi_handle: handle for mapping NIC CCQ pi
 */
struct hlthunk_nic_user_ccq_set_out {
	uint64_t ccq_handle;
	uint64_t ccq_pi_handle;
};

/**
 * struct hlthunk_nic_user_db_fifo_set_in - ioctl in param
 * @port: nic port id
 * @id: nic db-fifo id
 * @base_sob_addr: base address of Sync Object
 * @num_sobs: number of Sobs should be used by NIC
 * @mode: represents desired mode of operation for provided FIFO, according to hl_nic_db_fifo_type
 * @dir_dup_ports_mask: ports for which the hw should duplicate the direct patcher descriptor
 */
struct hlthunk_nic_user_db_fifo_set_in {
	uint32_t port;
	uint32_t id;
	uint32_t base_sob_addr;
	uint32_t num_sobs;
	uint8_t mode;
	uint8_t dir_dup_ports_mask;
};

/**
 * struct hlthunk_nic_user_db_fifo_set_out - ioctl out param
 * @ci_handle: handle of db-fifo consumer index memory buffer
 * @regs_handle: handle of db-fifo registers base-address
 * @fifo_size: Fifo size allocated by the driver
 * @fifo_bp_thresh: fifo back pressure threshold programmed by the driver
 * @regs_offset: offset to the db-fifo registers
 */
struct hlthunk_nic_user_db_fifo_set_out {
	uint64_t ci_handle;
	uint64_t regs_handle;
	uint32_t fifo_size;
	uint32_t fifo_bp_thresh;
	uint32_t regs_offset;
};

/**
 * struct hlthunk_event_record_razwi_event - razwi information.
 * @timestamp: timestamp of razwi.
 * @addr: address which accessing it caused razwi.
 * @engine_id: engine id of the razwi initiator, if it was initiated by engine that does not
 *             have engine id it will be set to HL_RAZWI_NA_ENG_ID. If there are several possible
 *             engines which caused the razwi, it will hold all of them.
 * @num_of_possible_engines: contains number of possible engine ids. In some asics, razwi indication
 *                           might be common for several engines and there is no way to get the
 *                           exact engine. In this way, engine_id array will be filled with all
 *                           possible engines caused this razwi. Also, there might be possibility
 *                           in gaudi, where we don't indication on specific engine, in that case
 *                           the value of this parameter will be zero.
 * @flags: bitmask for additional data: HL_RAZWI_READ - razwi caused by read operation
 *                                      HL_RAZWI_WRITE - razwi caused by write operation
 *                                      HL_RAZWI_LBW - razwi caused by lbw fabric transaction
 *                                      HL_RAZWI_HBW - razwi caused by hbw fabric transaction
 *                                      HL_RAZWI_RR - razwi caused by range register
 *                                      HL_RAZWI_ADDR_DEC - razwi caused by address decode error
 *         Note: this data is not supported by all asics, in that case the relevant bits will not
 *               be set.
 */
struct hlthunk_event_record_razwi_event {
	int64_t timestamp;
	uint64_t addr;
	uint16_t engine_id[HL_RAZWI_MAX_NUM_OF_ENGINES_PER_RTR];
	uint16_t num_of_possible_engines;
	uint8_t flags;
};

/**
 * struct hlthunk_event_record_undefined_opcode - info about last undefined opcode error
 * @timestamp: timestamp of the undefined opcode error
 * @cb_addr_streams: CB addresses (per stream) that are currently exists in the PQ
 *                   entiers. In case all streams array entries are
 *                   filled with values, it means the execution was in Lower-CP.
 * @cq_addr: the address of the current handled command buffer
 * @cq_size: the size of the current handled command buffer
 * @cb_addr_streams_len: num of streams - actual len of cb_addr_streams array.
 *                       should be equal to 1 incase of undefined opcode
 *                       in Upper-CP (specific stream) and equal to 4 incase
 *                       of undefined opcode in Lower-CP.
 * @engine_id: engine-id that the error occurred on
 * @stream_id: the stream id the error occurred on. In case the stream equals to
 *             MAX_QMAN_STREAMS_INFO it means the error occurred on a Lower-CP.
 */
struct hlthunk_event_record_undefined_opcode {
	int64_t  timestamp;
	uint64_t cb_addr_streams[MAX_QMAN_STREAMS_INFO][OPCODE_INFO_MAX_ADDR_SIZE];
	uint64_t cq_addr;
	uint32_t cq_size;
	uint32_t cb_addr_streams_len;
	uint32_t engine_id;
	uint32_t stream_id;
};

/**
 * struct hlthunk_event_record_critical_hw_err - info about the HW error that occurred
 * @timestamp: The time of the error occurrence.
 * @event_id: The device event generated by this error.
 */
struct hlthunk_event_record_critical_hw_err {
	int64_t timestamp;
	uint16_t event_id;
};

/**
 * struct hlthunk_event_record_critical_fw_err - info about the FW error that occurred
 * @timestamp: timestamp of the undefined opcode error
 * @err_type: The type of error being reported.
 * @reported_err_id: The reported error ID in-case of FW_REPORTED_ERR.
 */
struct hlthunk_event_record_critical_fw_err {
	int64_t timestamp;
	enum hl_info_fw_err_type err_type;
	uint16_t reported_err_id;
};

/**
 * struct hlthunk_event_record_engine_err - info about the engine reported an error
 * @timestamp: timestamp of the engine error.
 * @engine_id: engine id who reported the error.
 * @error_count: Amount of errors reported.
 */
struct  hlthunk_event_record_engine_err {
	int64_t timestamp;
	uint16_t engine_id;
	uint16_t error_count;
};

#define SEC_PCR_DATA_BUF_SZ	256
#define SEC_PCR_QUOTE_BUF_SZ	510	/* (512 - 2) 2 bytes used for size */
#define SEC_SIGNATURE_BUF_SZ	255	/* (256 - 1) 1 byte used for size */
#define SEC_PUB_DATA_BUF_SZ	510	/* (512 - 2) 2 bytes used for size */
#define SEC_CERTIFICATE_BUF_SZ	2046	/* (2048 - 2) 2 bytes used for size */

/*
 * struct hlthunk_sec_attest_info - attestation report of the boot
 * @nonce: number only used once. random number provided by host. this also passed to the quote
 *         command as a qualifying data.
 * @pcr_quote_len: length of the attestation quote data (bytes)
 * @pub_data_len: length of the public data (bytes)
 * @certificate_len: length of the certificate (bytes)
 * @pcr_num_reg: number of PCR registers in the pcr_data array
 * @pcr_reg_len: length of each PCR register in the pcr_data array (bytes)
 * @quote_sig_len: length of the attestation report signature (bytes)
 * @pcr_data: raw values of the PCR registers
 * @pcr_quote: attestation report data structure
 * @quote_sig: signature structure of the attestation report
 * @public_data: public key for the signed attestation
 *		 (outPublic + name + qualifiedName)
 * @certificate: certificate for the attestation signing key
 */
struct hlthunk_sec_attest_info {
	uint32_t nonce;
	uint16_t pcr_quote_len;
	uint16_t pub_data_len;
	uint16_t certificate_len;
	uint8_t pcr_num_reg;
	uint8_t pcr_reg_len;
	uint8_t quote_sig_len;
	uint8_t pcr_data[SEC_PCR_DATA_BUF_SZ];
	uint8_t pcr_quote[SEC_PCR_QUOTE_BUF_SZ];
	uint8_t quote_sig[SEC_SIGNATURE_BUF_SZ];
	uint8_t public_data[SEC_PUB_DATA_BUF_SZ];
	uint8_t certificate[SEC_CERTIFICATE_BUF_SZ];
};

/*
 * struct hlthunk_dev_info_signed - device information signed by a secured device.
 * @nonce: number only used once. random number provided by host. this also passed to the quote
 *         command as a qualifying data.
 * @pub_data_len: length of the public data (bytes)
 * @certificate_len: length of the certificate (bytes)
 * @info_sig_len: length of the attestation signature (bytes)
 * @public_data: public key info signed info data (outPublic + name + qualifiedName)
 * @certificate: certificate for the signing key
 * @info_sig: signature of the info + nonce data.
 * @dev_info_len: length of the device info (bytes)
 * @dev_info: device info (provided as an array of bytes).
 */
struct hlthunk_dev_info_signed {
	uint32_t nonce;
	uint16_t pub_data_len;
	uint16_t certificate_len;
	uint8_t info_sig_len;
	uint8_t public_data[SEC_PUB_DATA_BUF_SZ];
	uint8_t certificate[SEC_CERTIFICATE_BUF_SZ];
	uint8_t info_sig[SEC_SIGNATURE_BUF_SZ];
	uint16_t dev_info_len;
	uint8_t dev_info[SEC_DEV_INFO_BUF_SZ];
};

/**
 * struct hlthunk_user_mapping - user mapping information.
 * @dev_va: device virtual address.
 * @size: virtual address mapping size.
 */
struct hlthunk_user_mapping {
	uint64_t dev_va;
	uint64_t size;
};
/*
 * struct hl_page_fault_info - page fault information.
 * @timestamp: timestamp of page fault.
 * @addr: address which accessing it caused page fault.
 * @engine_id: engine id which caused the page fault, supported only in gaudi3.
 * @num_of_mappings: actual number of user mappings, filled by the driver. Even if error is
 *                   returned, this parameter will be filled by driver, unless driver did not had
 *                   the chance to fill it, in that case it will be equal to 0xFFFFFFFF.
 * @mappings_buf: holds all user mappings
 */
struct hlthunk_page_fault_info {
	int64_t timestamp;
	uint64_t addr;
	uint16_t engine_id;
	uint32_t num_of_mappings;
	struct hlthunk_user_mapping *mappings_buf;

};

struct hlthunk_functions_pointers {
	/*
	 * Functions that will be wrapped with profiler code to enable
	 * profiling
	 */
	int (*fp_hlthunk_command_submission)(int fd, struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out);
	int (*fp_hlthunk_open)(enum hlthunk_device_name device_name,
				const char *busid);
	int (*fp_hlthunk_close)(int fd);
	int (*fp_hlthunk_profiler_start)(int fd);
	int (*fp_hlthunk_profiler_stop)(int fd);
	int (*fp_hlthunk_profiler_get_trace)(int fd, void *buffer,
						uint64_t *size,
						uint64_t *num_entries);

	/* Function for the profiler to use */
	uint64_t (*fp_hlthunk_device_memory_alloc)(int fd, uint64_t size, uint64_t page_size,
						bool contiguous, bool shared);
	int (*fp_hlthunk_device_memory_free)(int fd, uint64_t handle);
	uint64_t (*fp_hlthunk_device_memory_map)(int fd, uint64_t handle,
							uint64_t hint_addr);
	uint64_t (*fp_hlthunk_host_memory_map)(int fd, void *host_virt_addr,
						uint64_t hint_addr,
						uint64_t host_size);
	int (*fp_hlthunk_memory_unmap)(int fd, uint64_t device_virt_addr);
	int (*fp_hlthunk_debug)(int fd, struct hl_debug_args *debug);
	int (*fp_hlthunk_request_command_buffer)(int fd, uint32_t cb_size,
							uint64_t *cb_handle);
	int (*fp_hlthunk_destroy_command_buffer)(int fd, uint64_t cb_handle);
	int (*fp_hlthunk_wait_for_cs)(int fd, uint64_t seq,
					uint64_t timeout_us, uint32_t *status);
	enum hlthunk_device_name (*fp_hlthunk_get_device_name_from_fd)(int fd);
	int (*fp_hlthunk_get_pci_bus_id_from_fd)(int fd, char *pci_bus_id,
							int len);
	int (*fp_hlthunk_get_device_index_from_pci_bus_id)(const char *busid);
	void* (*fp_hlthunk_malloc)(size_t size);
	void (*fp_hlthunk_free)(void *pt);
	int (*fp_hlthunk_signal_submission)(int fd,
					struct hlthunk_signal_in *in,
					struct hlthunk_signal_out *out);
	int (*fp_hlthunk_wait_for_signal)(int fd, struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);
	int (*fp_hlthunk_get_time_sync_info)(int fd,
					struct hlthunk_time_sync_info *info);
	int (*fp_hlthunk_get_cs_counters_info)(int fd,
					struct hl_info_cs_counters *info);
	void (*fp_hlthunk_profiler_destroy)(void);
	int (*fp_hlthunk_request_mapped_command_buffer)(int fd,
					uint32_t cb_size, uint64_t *cb_handle);
	int (*fp_hlthunk_wait_for_collective_sig)(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);
	int (*fp_hlthunk_deprecated_func1)(int fd, uint64_t seq,
					uint64_t timeout_us, uint32_t *status,
					uint64_t *timestamp);
	int (*fp_hlthunk_get_cb_usage_count)(int fd, uint64_t cb_handle,
						uint32_t *usage_cnt);
	int (*fp_hlthunk_staged_command_submission)(int fd, uint64_t sequence,
						struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out);
	int (*fp_hlthunk_get_hw_block)(int fd, uint64_t block_address,
						uint32_t *block_size,
						uint64_t *handle);
	int (*fp_hlthunk_wait_for_interrupt)(int fd, void *addr,
					uint64_t target_value,
					uint32_t interrupt_id,
					uint64_t timeout_us,
					uint32_t *status);
	int (*fp_hlthunk_device_memory_export_dmabuf_fd)(int fd,
							uint64_t handle,
							uint64_t size,
							uint32_t flags);
	int (*fp_hlthunk_command_submission_timeout)(int fd,
						struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out,
						uint32_t timeout);
	int (*fp_hlthunk_signal_submission_timeout)(int fd,
					struct hlthunk_signal_in *in,
					struct hlthunk_signal_out *out,
					uint32_t timeout);
	int (*fp_hlthunk_wait_for_signal_timeout)(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out,
					uint32_t timeout);
	int (*fp_hlthunk_wait_for_collective_sig_timeout)(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out,
					uint32_t timeout);
	int (*fp_hlthunk_staged_cs_timeout)(int fd,
					uint64_t sequence,
					struct hlthunk_cs_in *in,
					struct hlthunk_cs_out *out,
					uint32_t timeout);
	int (*fp_hlthunk_reserve_signals)(int fd,
					struct hlthunk_sig_res_in *in,
					struct hlthunk_sig_res_out *out);
	int (*fp_hlthunk_unreserve_signals)(int fd,
					struct reserve_sig_handle *handle,
					uint32_t *status);
	int (*fp_hlthunk_staged_cs_encaps_signals)(int fd,
					uint64_t sequence,
					struct hlthunk_cs_in *in,
					struct hlthunk_cs_out *out);
	int (*fp_hlthunk_wait_for_reserved_encaps_signals)(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);
	int (*fp_hlthunk_wait_for_collective_reserved_encap_sig)(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);
	uint64_t (*fp_hlthunk_host_memory_map_flags)(int fd, void *host_virt_addr,
						uint64_t hint_addr,
						uint64_t host_size,
						uint32_t flags);
	int (*fp_get_dram_replaced_rows_info)(int fd,
					struct hlthunk_dram_replaced_rows_info *info);
	int (*fp_get_dram_pending_rows_info)(int fd, uint32_t *out);
	int (*fp_DEPRECATED1)(int fd, void *addr, uint64_t target_value, uint32_t engine_id,
				uint64_t timeout_us, uint32_t *status, uint64_t *timestamp);
	int (*fp_hlthunk_wait_for_interrupt_by_handle)(int fd,
					uint64_t cq_counters_handle,
					uint64_t cq_counters_offset,
					uint64_t target_value,
					uint32_t interrupt_id,
					uint64_t timeout_us,
					uint32_t *status);
	int (*fp_hlthunk_get_mapped_cb_device_va_by_handle)(int fd,
						uint64_t cb_handle,
						uint64_t *device_va);
	int (*fp_hlthunk_get_pll_frequency)(int fd, uint32_t index,
					struct hlthunk_pll_frequency_info *frequency);
	int (*fp_hlthunk_deprecated_func2)(int fd, uint32_t interrupt_id,
							uint64_t cq_counters_handle,
							uint64_t cq_counters_offset,
							uint64_t target_value,
							uint64_t timestamp_handle,
							uint64_t timestamp_offset);
	int (*fp_hlthunk_deprecated_func3)(int fd, uint32_t num_elements, uint64_t *handle);
	int (*fp_hlthunk_device_mapped_memory_export_dmabuf_fd)(int fd,
							uint64_t addr,
							uint64_t size,
							uint64_t offset,
							uint32_t flags);
	int (*fp_hlthunk_get_time_sync_per_die_info)(int fd,
					uint32_t die_index,
					struct hlthunk_time_sync_info *info);
	int (*fp_hlthunk_get_hw_ip_info)(int fd,
					struct hlthunk_hw_ip_info *hw_ip);
};

struct hlthunk_debugfs {
	int addr_fd;
	int data_fd;
	int clk_gate_fd;
	char clk_gate_val[32];
};

/**
 * This function opens the habanalabs device according to specified busid, or
 * according to the device name, if busid is NULL. If busid is specified but
 * the device can't be opened, the function fails.
 * @param device_name name of the device that the user wants to open
 * @param busid pci address of the device on the host pci bus
 * @return file descriptor handle or negative value in case of error
 */
hlthunk_public int hlthunk_open(enum hlthunk_device_name device_name,
				const char *busid);

/**
 * This function opens the habanalabs device according to a specified module id.
 * This API is relevant only for ASICs that support this property
 * @param module_id a number representing the module_id in the host machine
 * @return file descriptor handle or negative value in case of error
 */
hlthunk_public int hlthunk_open_by_module_id(uint32_t module_id);

/**
 * This function opens the habanalabs control device according to specified
 * busid, or according to the requested device id number, if busid is NULL. If
 * busid is specified but the device can't be opened, the function fails.
 * @param dev_id the dev_id number of the control device as appears in /dev/.
 * The control devices appear like this:
 * /dev/accel/accel_controlD0
 * /dev/accel/accel_controlD1
 * ...
 * So the dev_id represents the number that appear at the name of the node.
 * Note it is different from the minor number
 * @param busid pci address of the device on the host pci bus
 * @return file descriptor handle or negative value in case of error
 */
hlthunk_public int hlthunk_open_control(int dev_id, const char *busid);

/**
 * This function opens the habanalabs control device according to specified busid,
 * or according to the device name, if busid is NULL. If busid is specified but
 * the device can't be opened, the function fails.
 * @param device_name name of the device that the user wants to open
 * @param busid pci address of the device on the host pci bus
 * @return file descriptor handle or negative value in case of error
 */
hlthunk_public int hlthunk_open_control_by_name(enum hlthunk_device_name device_name,
					const char *busid);

/**
 * This function opens the habanalabs control device according to a specified module id.
 * This API is relevant only for ASICs that support this property
 * @param module_id a number representing the module_id in the host machine
 * @return file descriptor handle or negative value in case of error
 */
hlthunk_public int hlthunk_open_control_by_module_id(uint32_t module_id);

/**
 * This function opens the habanalabs control device according to a specified bus id.
 * @param busid null-terminated string of the device PCI bus ID in the following
 * format: [domain]:[bus]:[device].[function] where domain, bus, device, and
 * function are all hexadecimal values.
 * @return file descriptor handle or negative value in case of error
 */
hlthunk_public int hlthunk_open_control_by_bus_id(const char *busid);

/**
 * This function closes an open file descriptor
 * @param fd file descriptor handle
 * @return the return value of the close() syscall
 */
hlthunk_public int hlthunk_close(int fd);

/**
 * This function retrieves the PCI device ID of a specific device through the
 * INFO IOCTL call
 * @param fd file descriptor handle of habanalabs main or control device
 * @return PCI device ID of the acquired device
 */
hlthunk_public uint32_t hlthunk_get_device_id_from_fd(int fd);

/**
 * This function returns the matching device name (ASIC type) of a specific
 * device through the INFO IOCTL call
 * @param fd file descriptor handle of habanalabs main or control device
 * @return enumeration value that represents the ASIC type of the acquired
 * device
 */
hlthunk_public enum hlthunk_device_name hlthunk_get_device_name_from_fd(int fd);

/**
 * This function retrieves the PCI bus ID of a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param pci_bus_id null-terminated string for the device in the following
 * format: [domain]:[bus]:[device].[function] where domain, bus, device, and
 * function are all hexadecimal values. pci_bus_id should be large enough to
 * store 13 characters including the NULL-terminator.
 * @param len maximum length of string to store in pci_bus_id
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_pci_bus_id_from_fd(int fd, char *pci_bus_id,
							int len);

/**
 * This function retrieves the device index of a specific device by its PCI bus ID
 * @param busid null-terminated string of the device PCI bus ID in the following
 * format: [domain]:[bus]:[device].[function] where domain, bus, device, and
 * function are all hexadecimal values.
 * @return device index for success, negative value for failure
 */
hlthunk_public int hlthunk_get_device_index_from_pci_bus_id(const char *busid);

/**
 * This function retrieves the device index of a specific device by its module ID
 * @param module_id a number representing the module_id in the host machine
 * @return device index for success, negative value for failure
 */
hlthunk_public int hlthunk_get_device_index_from_module_id(uint32_t module_id);

/**
 * This function retrieves H/W IP information for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param hw_ip info pointer to H/W IP information structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_hw_ip_info(int fd,
					struct hlthunk_hw_ip_info *hw_ip);

/**
 * This function retrieves statistics info on device open operations
 * @param fd file descriptor handle of habanalabs main or control device
 * @param open_stats info pointer to open stats structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_open_stats(int fd,
				struct hlthunk_open_stats_info *open_stats);

/**
 * This function retrieves aggregated asic status including general status,
 * open_stas, throttling and power info.
 * @param fd file descriptor handle of habanalabs main or control device
 * @param hw_asic_status info pointer to hw asic status structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_hw_asic_status(int fd,
				struct hlthunk_hw_asic_status *hw_asic_status);

/**
 * This function retrieves DRAM usage information for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param dram_usage pointer to DRAM usage information structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_dram_usage(int fd,
				struct hlthunk_dram_usage_info *dram_usage);

/**
 * This function retrieves the status of a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @return enum value that represents the device's status in case it was acquired,
 * otherwise - returns Linux error code.
 */
hlthunk_public int hlthunk_get_device_status_info(int fd);

/**
 * This function checks whether a specific device is idle
 * @param fd file descriptor handle of habanalabs main device
 * @return true if the acquired device is idle, false otherwise
 */
hlthunk_public bool hlthunk_is_device_idle(int fd);

/**
 * This function retrieves a busy engines bitmask of a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to engines idle information structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_busy_engines_mask(int fd,
					struct hlthunk_engines_idle_info *info);

/**
 * This function retrieves MAC addresses information for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to MAC addresses information structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_mac_addr_info(int fd, struct hlthunk_mac_addr_info *info);

/**
 * This function retrieves the PLL frequency of a specific device.
 * @param fd file descriptor handle of habanalabs device
 * @param index the index of the specified PLL
 * @param frequency pointer to frequency structure to store the frequency in MHz
 * in each of the available outputs. if a certain output is not available a 0
 * value will be set
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_get_pll_frequency(int fd, uint32_t index,
				struct hlthunk_pll_frequency_info *frequency);

/**
 * This function retrieves the device utilization as percentage in the last
 * Xms period.
 * @param fd file descriptor handle of habanalabs main device
 * @param period_ms the period value in ms. Valid values are 100-1000, with
 * resolution of 100.
 * @rate pointer to uint32_t to store the utilization rate as percentage.
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_device_utilization(int fd, uint32_t period_ms,
						uint32_t *rate);

/**
 * This function retrieves the h/w events array
 * @param fd file descriptor handle of habanalabs main device
 * @param aggregate whether to retrieve from last reset or from loading of the
 * driver (aggregate mode)
 * @hw_events_arr_size size of hw_events_arr, in bytes
 * @hw_events_arr pointer to array of uint32_t to store the result.
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_hw_events_arr(int fd, bool aggregate,
			uint32_t hw_events_arr_size, uint32_t *hw_events_arr);

/**
 * This function retrieves the ASIC current and maximum clock rate, in MHz
 * @param fd file descriptor handle of habanalabs main device
 * @param cur_clk_mhz pointer to memory that will be filled by the function
 * with the current clock rate in MHz
 * @param max_clk_mhz pointer to memory that will be filled by the function
 * with the maximum clock rate in MHz
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_clk_rate(int fd, uint32_t *cur_clk_mhz,
					uint32_t *max_clk_mhz);

/**
 * This function retrieves the number of times the device had been hard or soft
 * reset since the last time the driver was loaded
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory that will be filled by the function
 * with the current reset counts.
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_reset_count_info(int fd,
					struct hlthunk_reset_count_info *info);

/**
 * This function retrieves the device's time alongside the host's time
 * for synchronization
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory that will be filled by the function with the
 * device's and host's times
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_time_sync_info(int fd,
					struct hlthunk_time_sync_info *info);

/**
 * This function retrieves the device's sync manager information
 * @param fd file descriptor handle of habanalabs main device
 * @dcore_id dcore id to fetch relevant sm info from
 * @param info pointer to memory that will be filled by the function with the
 * sync manager information
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_sync_manager_info(int fd, int dcore_id,
					struct hlthunk_sync_manager_info *info);

/**
 * This function retrieves the device's cs counters information
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory that will be filled by the function with the
 * cs counters
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_cs_counters_info(int fd,
					struct hl_info_cs_counters *info);

/**
 * This function retrieves the device's pci counters information
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory that will be filled by the function with the
 * pci counters information
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_pci_counters_info(int fd,
					struct hlthunk_pci_counters_info *info);

/**
 * This function retrieves the device's clock throttling information
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory that will be filled by the function with the
 * clock throttling information
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_clk_throttle_info(int fd,
					struct hlthunk_clk_throttle_info *info);

/**
 * This function retrieves the device's total energy consumption
 * in millijoules (mJ), since the driver was loaded.
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory, where to fill the total energy consumption
 * info.
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_total_energy_consumption_info(int fd,
				struct hlthunk_energy_info *info);

/**
 * This function retrieves the current device's power in milliwatts (mW)
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory, where to fill the power info
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_power_info(int fd,
				struct hlthunk_power_info *info);

/**
 * This function retrieves miscellaneous information of a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to device information structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_info(int fd, struct hl_info_args *info);

/**
 * This function creates a command buffer for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param cb_size size of command buffer
 * @param cb_handle pointer to uint64_t to store the command buffer handle
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_request_command_buffer(int fd, uint32_t cb_size,
							uint64_t *cb_handle);
/**
 * This function creates a command buffer for a specific device and maps it to
 * the device's MMU
 * @param fd file descriptor handle of habanalabs main device
 * @param cb_size size of command buffer
 * @param cb_handle pointer to uint64_t to store the command buffer handle
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_request_mapped_command_buffer(int fd,
					uint32_t cb_size, uint64_t *cb_handle);

/**
 * This function destroys a command buffer for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param cb_handle handle of the command buffer
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_destroy_command_buffer(int fd, uint64_t cb_handle);

/**
 * This function retrieves the usage count of a CB for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param cb_handle handle of the CB
 * @param usage_cnt pointer to uint32_t to store the usage count of the CB
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_cb_usage_count(int fd, uint64_t cb_handle,
						uint32_t *usage_cnt);

/**
 * This function submits a set of jobs to a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to command submission input structure
 * @param out pointer to command submission output structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_command_submission(int fd, struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out);

/**
 * This function submits a set of jobs to a specific device with a timeout
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to command submission input structure
 * @param out pointer to command submission output structure
 * @param timeout duration in seconds
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_command_submission_timeout(int fd,
						struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out,
						uint32_t timeout);

/**
 * This function submits a set of jobs to a specific device as part of a
 * staged submission
 * @param fd file descriptor handle of habanalabs main device
 * @sequence sequence number of this staged submission obtained from the
 *           first CS submitted
 * @param in pointer to command submission input structure
 * @param out pointer to command submission output structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_staged_command_submission(int fd,
						uint64_t sequence,
						struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out);

/**
 * This function submits a set of jobs to a specific device as part of a
 * staged submission with a timeout
 * @param fd file descriptor handle of habanalabs main device
 * @sequence number of this staged submission obtained from the
 *           first CS submitted
 * @param in pointer to command submission input structure
 * @param out pointer to command submission output structure
 * @param timeout duration in seconds
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_staged_command_submission_timeout(int fd,
						uint64_t sequence,
						struct hlthunk_cs_in *in,
						struct hlthunk_cs_out *out,
						uint32_t timeout);

/**
 * This function waits until a command submission of a specific device has
 * finished executing
 * @param fd file descriptor handle of habanalabs main device
 * @param seq sequence number of command submission
 * @param timeout_us absolute timeout to wait in microseconds. If the timeout
 * value is 0, the driver won't sleep at all. It will check the status of the
 * CS and return immediately
 * @param status pointer to uint32_t to store the wait status
 * @return negative error code on error, otherwise non-negative CS status
 */
hlthunk_public int hlthunk_wait_for_cs(int fd, uint64_t seq,
					uint64_t timeout_us, uint32_t *status);

/**
 * This function waits until a command submission of a specific device has
 * finished executing
 * @param fd file descriptor handle of habanalabs main device
 * @param seq sequence number of command submission
 * @param timeout_us absolute timeout to wait in microseconds. If the timeout
 * value is 0, the driver won't sleep at all. It will check the status of the
 * CS and return immediately
 * @param status pointer to uint32_t to store the wait status
 * @param timestamp nanoseconds timestamp recorded once cs is completed
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_cs_with_timestamp(int fd, uint64_t seq,
					uint64_t timeout_us, uint32_t *status,
					uint64_t *timestamp);

/**
 * This function waits until at least one command submission from a sequence
 * of command submissions of a specific device has finished executing
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait for multi CS input structure
 * @param out pointer to a wait for multi CS output structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_multi_cs(int fd,
					struct hlthunk_wait_multi_cs_in *in,
					struct hlthunk_wait_multi_cs_out *out);

/**
 * This function waits until at least one command submission from a sequence
 * of command submissions of a specific device has finished executing
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait for multi CS input structure
 * @param out pointer to a wait for multi CS output structure
 * @param timestamp nanoseconds timestamp of the first CS to be completed. set
 * to zero if the timestamp cannot be determined.
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_multi_cs_with_timestamp(int fd,
					struct hlthunk_wait_multi_cs_in *in,
					struct hlthunk_wait_multi_cs_out *out,
					uint64_t *timestamp);

/**
 * This function waits until an interrupt occurs and target value is greater or
 * equal than the content of a given user address
 * @param fd file descriptor handle of habanalabs main device
 * @param addr user address for target value comparison
 * @param target_value target value for comparison
 * @param interrupt_id interrupt id to wait for, set to all 1s in order to
 * register to all user interrupts
 * @param timeout_us absolute timeout to wait in microseconds. If the timeout
 * value is 0, the driver won't sleep at all. It will perform the comparison
 * without waiting for the interrupt to expire and will return immediately
 * @param status pointer to uint32_t to store the wait status
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_interrupt(int fd, void *addr,
					uint64_t target_value,
					uint32_t interrupt_id,
					uint64_t timeout_us,
					uint32_t *status);

/**
 * This function waits until an interrupt occurs and target value is greater or
 * equal than the content of a given user address
 * @param fd file descriptor handle of habanalabs main device
 * @param cq_counters_handle cb handle of the cq counters
 * @param cq_counters_offset offset from the cq_counters_handle
 * @param target_value target value for comparison
 * @param interrupt_id interrupt id to wait for, set to all 1s in order to
 * register to all user interrupts
 * @param timeout_us absolute timeout to wait in microseconds. If the timeout
 * value is 0, the driver won't sleep at all. It will perform the comparison
 * without waiting for the interrupt to expire and will return immediately
 * @param status pointer to uint32_t to store the wait status
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_interrupt_by_handle(int fd,
					uint64_t cq_counters_handle,
					uint64_t cq_counters_offset,
					uint64_t target_value,
					uint32_t interrupt_id,
					uint64_t timeout_us,
					uint32_t *status);

/**
 * This wait registers to get a timestamp when
 * an interrupt occurs and target value is greater or equal
 * than the content of a given user address.
 * timestamp 0 means that the interrupt didn't occur yet (target wasn't reached).
 * @param fd file descriptor handle of habanalabs main device
 * @param addr user address for target value comparison
 * @param target_value target value for comparison
 * @param interrupt_id interrupt id to wait for, set to all 1s in order to
 * register to all user interrupts
 * @param timeout_us absolute timeout to wait in microseconds. If the timeout
 * value is 0, the driver won't sleep at all. It will perform the comparison
 * without waiting for the interrupt to expire and will return immediately
 * @param status pointer to uint32_t to store the wait status
 * @param timestamp system timestamp in nanoseconds at time of interrupt.
 * 0, the interrupt wasn't triggered yet (CQ target wasn't reached).
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_interrupt_with_timestamp(int fd,
					void *addr,
					uint64_t target_value,
					uint32_t interrupt_id,
					uint64_t timeout_us,
					uint32_t *status,
					uint64_t *timestamp);

/**
 * This wait registers to get a timestamp when
 * an interrupt occurs and target value is greater or equal
 * than the content of a given user address.
 * timestamp 0 means that the interrupt didn't occur yet (target wasn't reached).
 * @param fd file descriptor handle of habanalabs main device
 * @param cq_counters_handle cb handle of the cq counters
 * @param cq_counters_offset offset from the cq_counters_handle
 * @param target_value target value for comparison
 * @param interrupt_id interrupt id to wait for, set to all 1s in order to
 * register to all user interrupts
 * @param timeout_us absolute timeout to wait in microseconds. If the timeout
 * value is 0, the driver won't sleep at all. It will perform the comparison
 * without waiting for the interrupt to expire and will return immediately
 * @param status pointer to uint32_t to store the wait status
 * @param timestamp system timestamp in nanoseconds at time of interrupt.
 * 0, the interrupt wasn't triggered yet (CQ target wasn't reached).
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_wait_for_interrupt_by_handle_with_timestamp(int fd,
					uint64_t cq_counters_handle,
					uint64_t cq_counters_offset,
					uint64_t target_value,
					uint32_t interrupt_id,
					uint64_t timeout_us,
					uint32_t *status,
					uint64_t *timestamp);

/**
 * This function submits a job of a signal CS to a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a signal command submission input structure
 * @param out pointer to a  signal command submission output structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_signal_submission(int fd,
					struct hlthunk_signal_in *in,
					struct hlthunk_signal_out *out);

/**
 * This function submits a job of a signal CS to a specific device with timeout
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a signal command submission input structure
 * @param out pointer to a  signal command submission output structure
 * @param timeout duration in seconds
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_signal_submission_timeout(int fd,
					struct hlthunk_signal_in *in,
					struct hlthunk_signal_out *out,
					uint32_t timeout);

/**
 * This function submits a job of a  wait CS to a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait command submission input structure
 * @param out pointer to a wait command submission output structure
 * @return 0 for success, negative value for failure. ULLONG_MAX is returned if
 * the given signal CS was already completed. Undefined behavior if the given
 * seq is not of a  signal CS
 */
hlthunk_public int hlthunk_wait_for_signal(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);

/**
 * This function submits a job of a  wait CS to a specific device with a timeout
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait command submission input structure
 * @param out pointer to a wait command submission output structure
 * @param timeout duration in seconds
 * @return 0 for success, negative value for failure. ULLONG_MAX is returned if
 * the given signal CS was already completed. Undefined behavior if the given
 * seq is not of a  signal CS
 */
hlthunk_public int hlthunk_wait_for_signal_timeout(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out,
					uint32_t timeout);

/**
 * This function submits a job of a  wait CS to a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait command submission input structure
 * @param out pointer to a wait command submission output structure
 * @return 0 for success, negative value for failure. ULLONG_MAX is returned if
 * the given signal CS was already completed. Undefined behavior if the given
 * seq is not of a  signal CS
 */
hlthunk_public int hlthunk_wait_for_collective_signal(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);

/**
 * This function submits a job of a  wait CS to a specific device with a timeout
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait command submission input structure
 * @param out pointer to a wait command submission output structure
 * @param timeout duration in seconds
 * @return 0 for success, negative value for failure. ULLONG_MAX is returned if
 * the given signal CS was already completed. Undefined behavior if the given
 * seq is not of a  signal CS
 */
hlthunk_public int hlthunk_wait_for_collective_signal_timeout(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out,
					uint32_t timeout);

/**
 * This function set supported device memory allocation page orders
 * @param fd file descriptor of the device on which to perform the query
 * @param page_order_bitmask bitmask of supported allocation page orders
 * @return 0 on success, otherwise non 0 error code.
 *
 * note that on ASICs that does not support multiple page sizes of device memory the
 * function will set page_order_bitmask to be 0.
 */
hlthunk_public int hlthunk_get_dev_memalloc_page_orders(int fd, uint64_t *page_order_bitmask);

/**
 * This function allocates DRAM memory on the device
 * @param fd file descriptor of the device on which to allocate the memory
 * @param size how much memory to allocate
 * @param page_size what page size to use in the allocation. 0 means using the default size.
 * @param contiguous whether the memory area will be physically contiguous
 * @param shared whether this memory can be shared with other user processes
 * on the device
 * @return opaque handle representing the memory allocation. 0 is returned
 * upon failure
 */
hlthunk_public uint64_t hlthunk_device_memory_alloc(int fd, uint64_t size, uint64_t page_size,
						bool contiguous, bool shared);

/**
 * This function frees DRAM memory that was allocated on the device using
 * hlthunk_device_memory_alloc
 * @param fd file descriptor of the device that this memory belongs to
 * @param handle the opaque handle that represents this memory
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_device_memory_free(int fd, uint64_t handle);

/**
 * This function asks the driver to map a previously allocated DRAM memory
 * to the device's MMU and to allocate for it a VA in the device address space
 * @param fd file descriptor of the device that this memory belongs to
 * @param handle the opaque handle that represents this memory
 * @param hint_addr the user can request from the driver that the VA will be
 * a specific address. The driver doesn't have to comply to this request but
 * will take it under consideration
 * @return VA in the device address space. 0 is returned upon failure
 */
hlthunk_public uint64_t hlthunk_device_memory_map(int fd, uint64_t handle,
							uint64_t hint_addr);

/**
 * This function asks the driver to map a previously allocated host memory
 * to the device's MMU and to allocate for it a VA in the device address space
 * @param fd file descriptor of the device that this memory will be mapped to
 * @param host_virt_addr the user's VA of memory area on the host
 * @param hint_addr the user can request from the driver that the device VA will
 * be a specific address. The driver doesn't have to comply to this request but
 * will take it under consideration
 * @param host_size the size of the memory area
 * @return VA in the device address space. 0 is returned upon failure
 */
hlthunk_public uint64_t hlthunk_host_memory_map(int fd, void *host_virt_addr,
						uint64_t hint_addr,
						uint64_t host_size);

/**
 * This function asks the driver to map a previously allocated host memory
 * to the device's MMU and to allocate for it a VA in the device address space.
 * In this variation the function allows to specify custom flags for memory map.
 * @param fd file descriptor of the device that this memory will be mapped to
 * @param host_virt_addr the user's VA of memory area on the host
 * @param hint_addr the user can request from the driver that the device VA will
 * be a specific address. The driver doesn't have to comply to this request but
 * will take it under consideration
 * @param host_size the size of the memory area
 * @param flags custom flags for the memory map
 * @return VA in the device address space. 0 is returned upon failure
 */
hlthunk_public uint64_t hlthunk_host_memory_map_flags(int fd, void *host_virt_addr,
						uint64_t hint_addr,
						uint64_t host_size,
						uint32_t flags);

/**
 * This function unmaps a mapping in the device's MMU that was previously done
 * using either hlthunk_device_memory_map or hlthunk_host_memory_map
 * @param fd file descriptor of the device that contains the mapping
 * @param device_virt_addr the VA in the device address space representing
 * the device or host memory area
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_memory_unmap(int fd, uint64_t device_virt_addr);

/**
 * This function asks the driver to create a DMA-BUF object that will represent
 * an existing memory allocation inside the device memory. The function will
 * return a FD that will represent that DMA-BUF object. The application should
 * pass that FD to the importer driver. This function shall be used only for
 * Gaudi1
 * @param fd file descriptor of the device that this memory belongs to
 * @param addr physical address inside the device's DRAM.
 * @param size holds the size of the memory that the user wants to create a
 *             dma-buf that will describe it.
 * @param flags DMA-BUF file/FD flags. For now this parameter is not used.
 * @return file descriptor (positive value). negative value for failure
 */
hlthunk_public int hlthunk_device_memory_export_dmabuf_fd(int fd,
							uint64_t addr,
							uint64_t size,
							uint32_t flags);

/**
 * This function does the same as hlthunk_device_memory_export_dmabuf_fd but
 * with different interface for the application. this function shall be used
 * only for Gaudi2 and later ASICs
 * @param fd file descriptor of the device that this memory belongs to
 * @param addr a device virtual address that represents the start address of
 *             a mapped DRAM memory area inside the device. the address must
 *             be the same as was received from the driver during a previous
 *             HL_MEM_OP_MAP operation.
 * @param size holds the size of the memory that the user wants to create a
 *             dma-buf that will describe it.
 * @param offset an offset inside of the memory area describe by addr. The
 *               offset represents the start address of that the exported
 *               dma-buf object describes.
 * @param flags DMA-BUF file/FD flags. For now this parameter is not used.
 * @return file descriptor (positive value). negative value for failure
 */
hlthunk_public int hlthunk_device_mapped_memory_export_dmabuf_fd(int fd,
							uint64_t addr,
							uint64_t size,
							uint64_t offset,
							uint32_t flags);

/**
 * This function retrieves a HW block handle according to a given address
 * @param fd file descriptor of the device on which to allocate the memory
 * @param block_address HW block address (configuration space only)
 * @param block_size pointer to HW block size. The driver fills this value and
 * the user needs to pass it when calling mmap
 * @param handle pointer to handle that needs to be passed to mmap in order to
 * map the HW block to a VA in the process VA range
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_get_hw_block(int fd, uint64_t block_address,
				uint32_t *block_size, uint64_t *handle);

/**
 * This function enables and retrieves debug traces of a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param debug pointer to debug parameters structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_debug(int fd, struct hl_debug_args *debug);

/**
 * This function allocates connection ID for a specific port of a specific
 * device
 * @param fd file descriptor handle of habanalabs main device
 * @param port port number
 * @param conn_id pointer to uint32_t to store the connection ID
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_alloc_conn(int fd, uint32_t port, uint32_t *conn_id);

/**
 * This function allocates collective connection ID for a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to alloc collective connection input structure
 * @param conn_id pointer to uint32_t to store the collective connection ID
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_alloc_coll_conn(int fd, struct hlthunk_nic_alloc_coll_conn_in *in,
						uint32_t *conn_id);

/**
 * This function sets up a requester connection context for a specific port of a
 * specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param port port number
 * @param conn_id connection ID
 * @param in pointer to requester context input structure
 * @param out pointer to requester context output structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_set_requester_conn_ctx(int fd, uint32_t port,
				uint32_t conn_id,
				const struct hlthunk_requester_conn_ctx *in,
				struct hlthunk_requester_conn_ctx_out *out);

/**
 * This function sets up a responder connection context for a specific port of a
 * specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param port port number
 * @param conn_id connection ID
 * @param ctx pointer to responder context structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_set_responder_conn_ctx(int fd, uint32_t port,
				uint32_t conn_id,
				const struct hlthunk_responder_conn_ctx *ctx);

/**
 * This function destroys connection ID for a specific port of a specific device
 * @param fd file descriptor handle of habanalabs main device
 * @param port port number
 * @param conn_id connection ID
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_destroy_conn(int fd, uint32_t port,
					uint32_t conn_id);

/**
 * This function creates a NIC CQ on a specific device
 * @param fd file descriptor of the device with the NIC CQ to create
 * @param num_of_entries number of entries in the CQ buffer
 * @param handle pointer to the returned handle of NIC CQ buffer
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_cq_create(int fd, uint32_t num_of_entries,
					uint64_t *handle);

/**
 * This function destroys a NIC CQ on a specific device
 * @param fd file descriptor of the device with NIC CQ to destroy
 * @param handle identifier handle of NIC CQ
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_cq_destroy(int fd, uint64_t handle);

/**
 * This function waits on the NIC CQ until there are available CQEs
 * @param fd file descriptor of the device with the NIC CQ to wait on
 * @param handle identifier handle of NIC CQ
 * @param status current status of the NIC CQ
 * @param pi pointer to the returned producer index to return, meaning the first
 * available CQE
 * @param num_of_cqes pointer to the returned number of available CQEs to
 * consume starting from the CQE at pi
 * @param timeout_us timeout for waiting on the CQ in microseconds
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_cq_wait(int fd, uint64_t handle,
					uint32_t *status, uint32_t *pi,
					uint32_t *num_of_cqes,
					uint64_t timeout_us);

/**
 * This function updates the number of consumed CQEs by the user
 * @param fd file descriptor of the device with the NIC CQ to update
 * @param handle identifier handle of NIC CQ
 * @param num_of_consumed_entries number of consumed CQEs by the user
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_cq_update_consumed_entries(int fd,
					uint64_t handle,
					uint32_t num_of_consumed_entries);

/**
 * This function sets a NIC Congestion Control Completion Queue on a specific device
 * @param fd file descriptor of the device with the NIC CCQ to create
 * @param in input info for ccq creation
 * @param out output info for ccq mapping
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_ccq_set(int fd, struct hlthunk_nic_user_ccq_set_in *in,
		struct hlthunk_nic_user_ccq_set_out *out);

/**
 * This function unsets a NIC CCQ on a specific device
 * @param fd file descriptor of the device with NIC CCQ to destroy
 * @param port the port associated with this ccq
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_ccq_unset(int fd, uint32_t port);

/**
 * This function sets a NIC WQ array on a specific device
 * @param fd file descriptor of the device with the NIC WQ array to set
 * @param in pointer to WQ array set input structure
 * @param out pointer to WQ array set output structure
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_wq_arr_set(int fd,
					struct hlthunk_nic_wq_arr_set_in *in,
					struct hlthunk_nic_wq_arr_set_out *out);

/**
 * This function unsets a NIC WQ array on a specific device
 * @param fd file descriptor of the device with the NIC WQ array to unset
 * @param port the NIC port to unset the WQ array on
 * @param type sender/receiver
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_wq_arr_unset(int fd, uint32_t port,
					uint32_t type);

/**
 * This function sets a NIC user CQ on a specific device
 * @param fd file descriptor of the device with the NIC user CQ to set
 * @param port the NIC port to set the user CQ on
 * @param addr the base virtual address of the user CQ to set
 * @param num_of_cqes number of CQ entries for each user CQ
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_cq_set(int fd, uint32_t port, uint64_t addr,
					uint32_t num_of_cqes);

/**
 * This function unsets a NIC user CQ on a specific device
 * @param fd file descriptor of the device with the NIC user CQ to unset
 * @param port the NIC port to unset the user CQ on
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_cq_unset(int fd, uint32_t port);

/**
 * This function updates the user CQ consumer index of the buffer
 * @param fd file descriptor of the device with the NIC user CQ to update
 * @param num_of_consumed_entries number of consumed CQEs by the user
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_cq_update_ci(int fd, uint32_t port, uint32_t ci);

/**
 * This function allocates a NIC user CQ on a specific device
 * @param fd file descriptor of the device with the NIC user CQ to allocate
 * @param in pointer to user CQ alloc input structure
 * @param out pointer to user CQ alloc output structure
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_cq_id_alloc(int fd, struct hlthunk_nic_user_cq_id_alloc_in *in,
						struct hlthunk_nic_user_cq_id_alloc_out *out);

/**
 * This function sets a NIC user CQ on a specific device
 * @param fd file descriptor of the device with the NIC user CQ to set
 * @param in pointer to user CQ set input structure
 * @param out pointer to user CQ set output structure
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_cq_id_set(int fd, struct hlthunk_nic_user_cq_id_set_in *in,
						struct hlthunk_nic_user_cq_id_set_out *out);

/**
 * This function unsets a NIC user CQ on a specific device
 * @param fd file descriptor of the device with the NIC user CQ to unset
 * @param in pointer to user CQ unset input structure
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_nic_user_cq_id_unset(int fd, struct hlthunk_nic_user_cq_id_unset_in *in);

/**
 * This function checks if the port its going to use has advanced features.
 * @param fd file descriptor of the device used by the application
 * @param port the port this application is associated with
 * @return 0 application and port support advanced features
 *  non-zero application and/or port do not support advanced features.
 */
hlthunk_public int hlthunk_nic_user_set_app_params(
		int fd, uint32_t port,
		const struct hlthunk_nic_user_set_app_params_in *in);

/**
 * This function retrieve port resources and configurations.
 * @param fd file descriptor of the device that is used by the application
 * @param port the port for which to get the params
 * @param out holds the nic's retrieved resources and configurations.
 * @return 0 application and port support advanced features
 *  non-zero application and/or port do not support advanced features.
 */
hlthunk_public int hlthunk_nic_user_get_app_params(
		int fd, uint32_t port,
		struct hlthunk_nic_user_get_app_params_out *out);

/**
 * This function reads a single event from the driver-maintained event queue
 * associated with the given fd.
 * @param fd file descriptor of the device that is used by the application
 * @param port the port for which to get the event from
 * @param out holds the nic's retrieved event.
 * @return 0 successfully accessed the associated event-queue and retrieved
 *  a valid entry, negative value in case of an error or if the queue is empty.
 */
hlthunk_public int hlthunk_nic_eq_poll(
				int fd, uint32_t port,
				struct hlthunk_nic_eq_poll_out *out);

/**
 * This function allocates doorbell fifo ID
 * @param fd file descriptor of the device
 * @param port NIC port we are working with
 * @param id allocated db fifo ID
 * @return 0 successfully allocated db fifo ID
 *  non-zero failed to allocate db fifo ID
 */
hlthunk_public int
hlthunk_nic_alloc_user_db_fifo(int fd, uint32_t port, uint32_t *id);

/**
 * This function configures doorbell fifo
 * @param fd file descriptor of the device
 * @param in struct parameter with the needed info for setting db fifo
 * @param out struct parameter with the needed info for mapping db fifo
 * @return 0 successfully configured db fifo
 *  non-zero failed to configure db fifo
 */
hlthunk_public int
hlthunk_nic_user_db_fifo_set(int fd, struct hlthunk_nic_user_db_fifo_set_in *in,
		struct hlthunk_nic_user_db_fifo_set_out *out);

/**
 * This function destroys doorbell fifo.
 * @param fd file descriptor of the device.
 * @param port NIC port we are working with.
 * @param id allocated db fifo ID
 * @return 0 successfully destroyed db fifo
 *  non-zero failed to destroy db fifo
 */
hlthunk_public int
hlthunk_nic_user_db_fifo_unset(int fd, uint32_t port, uint32_t id);

/**
 * This function retrieves the internal port PCS link up/down status
 * @param fd file descriptor of the device with the Habana link port to probe
 * @param in struct parameter with the needed info for getting the link state
 * @param out struct parameter with info of the result of getting the link state
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_get_habana_link_state(int fd,
						struct hlthunk_get_habana_link_state_in *in,
						struct hlthunk_get_habana_link_state_out *out);

/**
 * This function retrieves the internal port statistics
 * @param fd file descriptor of the device with the Habana link port to probe
 * @param in struct parameter with the needed info for getting the statistics
 * @param out struct parameter with info of the result of getting the statistics
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_get_habana_link_statistics(int fd,
						struct hlthunk_get_habana_link_stat_in *in,
						struct hlthunk_get_habana_link_stat_out *out);

/**
 * This function allocates NIC encapsulation ID.
 * @param fd file descriptor of the device that is used by the application.
 * @param port NIC port for which to get the params.
 * @param encap_id holds allocated encapsulation ID.
 * @return 0 if success. Non-zero for any error.
 */
hlthunk_public int hlthunk_nic_user_encap_alloc(
		int fd, uint32_t port, uint32_t *encap_id);

/**
 * This function configures NIC HW to start encapsulation.
 * @param fd file descriptor of the device that is used by the application.
 * @param encap_cfg holds encapsulation parameters.
 * @return 0 if success. Non-zero for any error.
 */
hlthunk_public int hlthunk_nic_user_encap_set(
		int fd, const struct hlthunk_encap_cfg *encap_cfg);

/**
 * This function de-allocates ID and stops encapsulation.
 * @param fd file descriptor of the device that is used by the application.
 * @param port NIC port for which to get the params.
 * @param encap_id holds encapsulation ID to be de-allocated.
 * @return 0 if success. Non-zero for any error.
 */
hlthunk_public int hlthunk_nic_user_encap_unset(
		int fd, uint32_t port, uint32_t encap_id);

/**
 * This function retrieves QPC dump for a given QP of a given NIC port.
 * @param fd file descriptor handle of habanalabs main device.
 * @param port NIC port which holds that QP.
 * @param qpn QP number.
 * @param req true for requester dump, false for responder dump.
 * @param buf buffer to hold the dump.
 * @param buf_size size of the buffer to hold the dump.
 * @return 0 if success. Non-zero for any error.
 */
hlthunk_public int hlthunk_nic_dump_qp(int fd, uint32_t port, uint32_t qpn, uint32_t req,
					char *buf, uint32_t buf_size);

/**
 * This function retrieves the NIC ports and external ports masks. This function shall be used
 * only for Gaudi2 and later ASICs.
 * @param fd file descriptor handle of habanalabs main device.
 * @param out struct parameter with returned masks.
 * @return 0 if success. Non-zero for any error.
 */
hlthunk_public int hlthunk_nic_get_ports_masks(int fd, struct hlthunk_nic_get_ports_masks_out *out);

/**
 * This function retrieves information of recorded events.
 * @param fd file descriptor of the device that is used by the application.
 * @param event_id event id to retrieve its data.
 * @param buf buffer that holds retrieved data of requested event id.
 * @return 0 if success. Non-zero for any error. In case no data was available, function will
 *  return 0 and the timestamp will not be changed. This can happen in the following
 *  events: HLTHUNK_RAZWI_EVENT, HLTHUNK_UNDEFINED_OPCODE, HLTHUNK_HW_ERR_OPCODE and
 *  HLTHUNK_FW_ERR_OPCODE.
 */
hlthunk_public int hlthunk_get_event_record(int fd,
		enum hlthunk_event_record_id event_id, void *buf);

hlthunk_public void *hlthunk_malloc(size_t size);
hlthunk_public void hlthunk_free(void *pt);

/**
 * This function returns a pointer to a char array containing the version. The
 * char array should be freed using hlthunk_free
 * @return valid pointer on success or null in case of error
 */
hlthunk_public char *hlthunk_get_version(void);

/* Functions for hash table implementation */

hlthunk_public void *hlthunk_hash_create(void);
hlthunk_public int hlthunk_hash_destroy(void *t);
hlthunk_public int hlthunk_hash_lookup(void *t, unsigned long key,
					void **value);
hlthunk_public int hlthunk_hash_insert(void *t, unsigned long key, void *value);
hlthunk_public int hlthunk_hash_delete(void *t, unsigned long key);
hlthunk_public int hlthunk_hash_next(void *t, unsigned long *key, void **value);
hlthunk_public int hlthunk_hash_first(void *t, unsigned long *key,
					void **value);


/**
 * This function starts an API controlled profiling session on a device
 * @param fd file descriptor of the device to start profiling on
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_profiler_start(int fd);

/**
 * This function stops an API controlled profiling session on a device
 * @param fd file descriptor of the device to stop profiling on
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_profiler_stop(int fd);

/**
 * This function retrieves the profiler trace created in an API controlled
 * profiling session. This function should be called first with buffer = null
 * in order to get the trace size, allocate the buffer according to this size
 * and call again with the buffer allocated
 * @param fd file descriptor of the device to get the trace from
 * @param buffer a buffer to copy to trace to, if buffer = null then only
 * retrieves the trace size and amount of entries.
 * The ruturned buffer is built in the following format:
 * [synTraceEvent enries][chars][size_t num][size_t version]
 * num: Amount of synTraceEvent entries
 * version: Synprof parser version
 * @param size out param for the amount of bytes copied to buffer (or the trace
 * size if buffer = null)
 * @param num_entries pointer to the returned number of entries in the trace
 * buffer
 * @return 0 on success or negative value in case of error
 */
hlthunk_public int hlthunk_profiler_get_trace(int fd, void *buffer,
					uint64_t *size, uint64_t *num_entries);

/**
 * This function destroys profiler instance if it existed
 * As long as env var HABANA_PROFILE=1
 * or HABANA_PROFILE=<template_name>, the profiler will reinitialize
 * on the next hlthunk_open call
 */
hlthunk_public void hlthunk_profiler_destroy(void);

/**
 * This function opens the debug file system of the habanalabs device already
 * opened via the hlthunk_open routine.
 * @param fd The file descriptor of the device as returned from the call to
 * the hlthunk_open routine.
 * @param debugfs The debug-fs information filled by the open routine.
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_debugfs_open(int fd,
					struct hlthunk_debugfs *debugfs);

/**
 * Using debugfs, this function returns the 32bit value read from the
 * device address space at the specified address.
 * @param debugfs Pointer to the device debug-fs information.
 * @param full_address The 64 bit address in the device address space to read
 * the from
 * @param val The vslue read from the given address
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_debugfs_read(struct hlthunk_debugfs *debugfs,
					uint64_t full_address, uint32_t *val);

/**
 * Using debugfs, this function writes the 32bit value to the specified address
 * in the device address space
 * @param debugfs Pointer to the device debug-fs information.
 * @param full_address The address in the device address space to writ the value
 * to.
 * @param val The vale to write.
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_debugfs_write(struct hlthunk_debugfs *debugfs,
					 uint64_t full_address, uint32_t val);

/**
 * This function closes the open debug file system of the habanalabs device.
 * @param debugfs Pointer to the debug-fs information as filled by the
 * hlthunk_debugfs_open routine
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_debugfs_close(struct hlthunk_debugfs *debugfs);

/**
 * This function reserves a number of signals for a certain stream.
 * @param fd device descriptor
 * @param in operation inputs which contains the queue idx and the signals count
 * @param out output of the operation which is the handle and the status.
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_reserve_encaps_signals(int fd,
					struct hlthunk_sig_res_in *in,
					struct hlthunk_sig_res_out *out);

/**
 * This function releases the signals that were reserved in
 * hlthunk_reserve_encaps_signals
 * @param handle reservation handle obtained from hlthunk_reserve_encaps_signals
 * @param status as output of the operation, which is 0 in case driver succeeded
 * to unreserve, negative value otherwise.
 * @return 0 upon success or negative value in case of error
 */
hlthunk_public int hlthunk_unreserve_encaps_signals(int fd,
					struct reserve_sig_handle *handle,
					uint32_t *status);

/**
 * This function submits a set of jobs to a specific device as the first part
 * of a staged submission, which includes a set of signal cmds to a
 * reserved signals, which were reserved in hlthunk_reserve_encaps_signals.
 * @param fd file descriptor handle of habanalabs main device
 * @handle_id reserved signals handle id, obtained from
 *		hlthunk_reserve_encaps_signals.
 * @param in pointer to command submission input structure
 * @param out pointer to command submission output structure
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_staged_command_submission_encaps_signals(int fd,
					uint64_t handle_id,
					struct hlthunk_cs_in *in,
					struct hlthunk_cs_out *out);

/**
 * This function submits a job of a wait CS to a specific device
 * This will wait for a number of signals which were resereved before calling
 * this function. it can wait for the whole reserved signals or
 * just to a specific offset whithin that range.
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait command submission input structure
 * @param out pointer to a wait command submission output structure
 * @return 0 for success, negative value for failure. ULLONG_MAX is returned if
 * the given signal CS was already completed. Undefined behavior if the given
 * seq is not of a  signal CS
 */
hlthunk_public int hlthunk_wait_for_reserved_encaps_signals(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);

/**
 * This function submits a job of a collective wait CS to a specific device
 * This will wait for a number of signals which were resereved before calling
 * this function. it can wait for the whole reserved signals or
 * just for a specific offset whithin that range.
 * @param fd file descriptor handle of habanalabs main device
 * @param in pointer to a wait command submission input structure
 * @param out pointer to a wait command submission output structure
 * @return 0 for success, negative value for failure. ULLONG_MAX is returned if
 * the given signal CS was already completed. Undefined behavior if the given
 * seq is not of a  signal CS
 */
hlthunk_public int hlthunk_wait_for_reserved_encaps_collective_signals(int fd,
					struct hlthunk_wait_in *in,
					struct hlthunk_wait_out *out);

/**
 * This function retrieves the dram replaced rows info.
 * @param fd file descriptor handle of habanalabs main device
 * @param info pointer to memory, where to fill the replaced rows info
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_dram_replaced_rows_info(int fd,
			struct hlthunk_dram_replaced_rows_info *info);

/**
 * This function retrieves the dram pending rows number.
 * @param fd file descriptor handle of habanalabs main device
 * @param out pointer to memory, where to fill the pending rows number
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_dram_pending_rows_info(int fd, uint32_t *out);

/**
 * This function retrieves attestation report of the boot.
 * @param fd file descriptor handle of habanalabs main device
 * @param nonce number used to generate secured attestation report
 * @param info pointer to attestation report info output buffer
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_sec_attest_info(int fd, uint32_t nonce,
					struct hlthunk_sec_attest_info *info);

/**
 * This function retrieves page fault information.
 * @param fd file descriptor handle of habanalabs main device
 * @param pf holds all page fault info, including user mappings
 * @param num_of_allocated_mappings array size of user buffer holding user mappings
 * @return 0 for success, a negative value for failure. In case no data was available, function will
 * return 0 and the timestamp will not be changed.
 */
hlthunk_public int hlthunk_get_page_fault_info(int fd, struct hlthunk_page_fault_info *pf,
						uint32_t num_of_allocated_mappings);

/**
 * This function retrieve the device va of a command buffer previously allocated
 * in host kernel memory.
 * Note that the buffer should be created with CB_TYPE_KERNEL_MAPPED flag.
 * @param fd file descriptor handle of habanalabs main device
 * @param cb_handle command buffer handle
 * @param device_va device va address of the allocated cb
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_mapped_cb_device_va_by_handle(int fd,
							uint64_t cb_handle,
							uint64_t *device_va);

/**
 * This function flush all PCI HBW writes
 * @param fd file descriptor handle of habanalabs main device
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_cs_flush_pci_hbw_writes(int fd);

/**
 * This function registers a timestamp event of a specific user interrupt id.
 * when interrupt occurred the driver will compare the CQ pi value with the
 * target value, and if CQ reached that value it'll write the timestamp
 * into the specified timestamp offset.
 * - If the timestamp record is already registered for interrupt, that has not
 * been expired yet, the driver will unregister the node and will register it on the new interrupt.
 * - As part of the call, the timestamp handle will be set atomically to TS_NOT_EXP_VAL.
 * The value is then overridden only after the cq counter reaches its target value.
 * users can wait until the timestamp entry is different than TS_NOT_EXP_VAL
 * to conclude that the target value has been reached.
 * - When there are other pending events waiting for the selected interrupt ID,
 * the caller must ensure that the CQ entry (cq_counters_handle + cq_counters_offset)
 * is not registered on other interrupt ID.
 * In other words, each CQ entry can be set at most by a single interrupt ID.
 * Note that if the counter current value has already reach the target value, the call
 * will not wait for the interrupt to set the timestamp, and instead set the
 * timestamp value immediately.
 *
 * @param fd file descriptor handle of habanalabs main device
 * @param interrupt_id interrupt id to register for
 * @param cq_counters_handle a cb which have the CQs counters value
 * @param cq_counters_offset offset in the CQs counters cb
 * @param target_value target value for comparison
 * @param timestamps_handle buffer handle of timestamps
 * @param timestamps_offset offset in the timestamps buffer
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_register_timestamp_interrupt(int fd, uint32_t interrupt_id,
					uint64_t cq_counters_handle,
					uint64_t cq_counters_offset,
					uint64_t target_value,
					uint64_t timestamps_handle,
					uint64_t timestamps_offset);

/**
 * This function allocate buffer in host kernel memory for timestamps events pool.
 * the driver will allocate enough space to store all timestamps data needed
 * by the driver to register/unregister timestamp events
 * This is needed due to a requirement, that driver cannot fail on out-of-memory
 * at event registration phase.
 * Note that each element has the size of uint64_t.
 * The memory will be freed when the user closes the file descriptor(ctx close)
 * @param fd file descriptor handle of habanalabs main device
 * @param elements_num number of timestamps elements, each element is uint64_t size.
 * @param handle buffer handle output
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_allocate_timestamp_elements(int fd,
					uint32_t elements_num,
					uint64_t *handle);

/**
 * This function creates a notifier object that allows the user
 * to receive async notification events, from the Kernel driver.
 * @param fd file descriptor handle of habanalabs main device
 * @return notifier handle for success, a negative value for failure
 */
hlthunk_public int hlthunk_notifier_create(int fd);

/**
 * This function releases a notifier object. it shall be invoked
 * when the user is no longer needs to receive notification events
 * from the Kernel driver.
 * @param fd file descriptor handle of habanalabs main device
 * @param handle of the notifier object
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_notifier_release(int fd, int handle);

/**
 * This function receives a notification event, that raises by the
 * Kernel driver. The function may block until an event is
 * received, or timeout expired. The function returns a bitmap value
 * that indicates, which event has occurred. Each function invocation
 * retrieves a new bitmap value, that indicates the last occurred events.
 * @param fd file descriptor handle of habanalabs main device
 * @param handle of the notifier object
 * @param notifier_events bitmap pointer. Each bit indicates a specific event.
 * @param notifier_cnt pointer to uint64_t, stores the notifier count.
 *  zero - indicates no notification - timeout expired.
 *  number greater from zero - indicates the notifications count since the last read.
 * @param flags of function's operations. Not used for now.
 * @param timeout in milliseconds. If the timeout value is 0, the function
 *  will return immediately, even if no notification was received.
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_notifier_recv(int fd, int handle, uint64_t *notifier_events,
						uint64_t *notifier_cnt, uint32_t flags,
						uint32_t timeout);

/**
 * This function retrieves string containing engines status.
 * @param fd file descriptor handle of habanalabs main device
 * @param status_buf buffer for retrieved string
 * @param status_buf_size buffer size
 * @param actual_size actual size of data
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_get_engine_status(int fd, char *status_buf, uint32_t status_buf_size,
						int *actual_size);

/**
 * This function set a run mode for all the cores that are supplied in core_ids array
 * @param fd file descriptor handle of habanalabs main device
 * @param core_ids a pointer to an array of cores numbers
 * @param num_cores actual number of core numbers in the core_ids array
 * @param mode the mode to be set, HL_ENGINE_CORE_RUN/HALT values.
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_engine_cores_set_mode(int fd, const uint32_t *core_ids,
				uint32_t num_cores, uint32_t mode);

/**
 * This function applies a command for all the engines that are supplied in engine_ids array
 * @param fd file descriptor handle of habanalabs main device
 * @param engine_ids a pointer to an array of engine numbers
 * @param num_engines actual number of engine numbers in the engine_ids array
 * @param command the command to be set
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_engines_command(int fd, const uint32_t *engine_ids,
				uint32_t num_engines, enum hl_engine_command command);

/**
 * This function returns the number of devices for the given device name.
 * @param device_name name of the device to count
 * @return number of devices for success, negative value for failure
 */
hlthunk_public int hlthunk_get_device_count(enum hlthunk_device_name device_name);

/**
 * This function reads a memory block from the device. The function is experimental which
 * means, it can be deprecated without providing backward compatibility and/or alternative
 * solution. Please take that into account when using this function.
 * @param fd file descriptor handle of habanalabs main device
 * @param buff buffer for retrieved the read memory
 * @param 64-bits block_address the device starting address to read from
 * @param block_size the request to read. max size - 16KB
 * @param flags indicates flags option for the read block opeartion. Not in used
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_device_memory_read_block_experimental(int fd, uint32_t *buff,
				uint64_t block_address, uint32_t block_size, uint32_t flags);

hlthunk_public int hlthunk_deprecated_func1(int fd, uint64_t seq,
					uint64_t timeout_us, uint32_t *status,
					uint64_t *timestamp);

hlthunk_public int hlthunk_deprecated_func2(int fd, uint32_t interrupt_id,
						uint64_t cq_counters_handle,
						uint64_t cq_counters_offset,
						uint64_t target_value,
						uint64_t timestamp_handle,
						uint64_t timestamp_offset);

hlthunk_public int hlthunk_deprecated_func3(int fd, uint32_t num_elements, uint64_t *handle);

/**
 * This function used to send a generic request asking FW for some information.
 * The driver will verify the opcode of the command and send it to fw then copy
 * the output to the specified buffer.
 * @param fd file descriptor handle of habanalabs main device
 * @param buff buffer to be sent to the FW.
 *	Note that this buffer may serve for both input/output buffer, for cases
 *	where the command has also input data to be passed to fw.
 * @param buff_size size of the buffer.
 * @sub_opcode generic request sub-opcode.
 * @return 0 for success, a negative value for failure.
 */
hlthunk_public int hlthunk_fw_send_generic_request(int fd, void *buff, uint32_t buff_size,
							uint32_t sub_opcode);

/**
 * This function creates an Xe virtual memory object.
 * @param fd file descriptor handle of the device
 * @param vm pointer to store the created VM ID
 * @param vm_fd pointer to store the VM's exported file descriptor
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_nic_xe_vm_create(int fd, uint32_t *vm, int *vm_fd);

/**
 * This function destroys an Xe virtual memory object.
 * @param fd file descriptor handle of the device
 * @param vm VM ID to destroy
 * @param vm_fd VM's exported file descriptor
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_nic_xe_vm_destroy(int fd, uint32_t vm, int vm_fd);

/**
 * This function allocates Xe device memory.
 * @param fd file descriptor handle of the device
 * @param vm ID of the parent VM
 * @param size requested memory size
 * @return opaque handle representing the allocated device memory, 0 for failure
 */
hlthunk_public uint64_t hlthunk_nic_xe_device_memory_alloc(int fd, uint32_t vm, uint64_t size);

/**
 * This function frees Xe device memory.
 * @param fd file descriptor handle of the device
 * @param handle opaque handle representing the device memory to free
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_nic_xe_device_memory_free(int fd, uint64_t handle);

/**
 * This function maps memory for Xe device. The given device virtual address will be mapped to the
 * given device or host memory. Either the device memory handle or the host memory pointer should
 * be 0 or NULL respectively.
 * @param fd file descriptor handle of the device
 * @param vm ID of the parent VM
 * @param handle opaque handle representing the device memory to map
 * @param hva pointer to host memory to map
 * @param dva device virtual memory to map
 * @param size requested memory size to map
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_nic_xe_memory_map(int fd, uint32_t vm, uint64_t handle, void *hva,
						uint64_t dva, uint64_t size);

/**
 * This function unmaps memory for Xe device.
 * @param fd file descriptor handle of the device
 * @param vm ID of the parent VM
 * @param dva device virtual memory to unmap
 * @param size requested memory size to unmap
 * @return 0 for success, a negative value for failure
 */
hlthunk_public int hlthunk_nic_xe_memory_unmap(int fd, uint32_t vm, uint64_t dva, uint64_t size);

/**
 * This function retrieves the signed device information.
 * @param fd file descriptor handle of habanalabs main device
 * @param nonce number used to generate signed device info report
 * @param info pointer to the signed device info output buffer
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_dev_info_signed(int fd, uint32_t nonce,
						struct hlthunk_dev_info_signed *info);
/**
 * This function retrieves the device's time alongside the host's time
 * on specified die for synchronization
 * @param fd file descriptor handle of habanalabs main device
 * @param die_index index for requested die must be assigned
 * @param info pointer to memory that will be filled by the function with the
 * device's and host's times
 * @return 0 for success, negative value for failure
 */
hlthunk_public int hlthunk_get_time_sync_per_die_info(int fd, uint32_t die_index,
					struct hlthunk_time_sync_info *info);

#ifdef __cplusplus
}   //extern "C"
#endif

#endif /* HLTHUNK_H */

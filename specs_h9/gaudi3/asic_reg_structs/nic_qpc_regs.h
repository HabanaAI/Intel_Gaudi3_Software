/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_NIC_QPC_H_
#define ASIC_REG_STRUCTS_GAUDI3_NIC_QPC_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace nic_qpc {
#else
#	ifndef static_assert
#		if defined( __STDC__ ) && defined( __STDC_VERSION__ ) && __STDC_VERSION__ >= 201112L
#			define static_assert(...) _Static_assert(__VA_ARGS__)
#		else
#			define static_assert(...)
#		endif
#	endif
#endif

/*
 REQ_QPC_CACHE_INVALIDATE 
 b'Requester cache invalidation'
*/
typedef struct reg_req_qpc_cache_invalidate {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_req_qpc_cache_invalidate;
static_assert((sizeof(struct reg_req_qpc_cache_invalidate) == 4), "reg_req_qpc_cache_invalidate size is not 32-bit");
/*
 REQ_QPC_CACHE_INV_STATUS 
*/
typedef struct reg_req_qpc_cache_inv_status {
	union {
		struct {
			uint32_t invalidate_done : 1,
				cache_idle : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_req_qpc_cache_inv_status;
static_assert((sizeof(struct reg_req_qpc_cache_inv_status) == 4), "reg_req_qpc_cache_inv_status size is not 32-bit");
/*
 REQ_STATIC_CONFIG 
*/
typedef struct reg_req_static_config {
	union {
		struct {
			uint32_t plru_eviction : 1,
				release_invalidate : 1,
				link_list_en : 1,
				timer_en : 1,
				resend_wqe_on_rollback : 1,
				cache_stop : 1,
				invalidate_writeback : 1,
				ovr_qp_valid_to_not_valid : 1,
				rem_burst_size_method : 2,
				evict_schedq_base : 4,
				evict_schedq_port_prio : 4,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_req_static_config;
static_assert((sizeof(struct reg_req_static_config) == 4), "reg_req_static_config size is not 32-bit");
/*
 REQ_BASE_ADDRESS_63_32 
 b'Requester cache base address bits 49 to 18'
*/
typedef struct reg_req_base_address_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_req_base_address_63_32;
static_assert((sizeof(struct reg_req_base_address_63_32) == 4), "reg_req_base_address_63_32 size is not 32-bit");
/*
 REQ_BASE_ADDRESS_31_0 
 b'Requester cache base address bits 17_7'
*/
typedef struct reg_req_base_address_31_0 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_req_base_address_31_0;
static_assert((sizeof(struct reg_req_base_address_31_0) == 4), "reg_req_base_address_31_0 size is not 32-bit");
/*
 REQ_CLEAN_LINK_LIST 
 b'Requester cache link list head clean'
*/
typedef struct reg_req_clean_link_list {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_req_clean_link_list;
static_assert((sizeof(struct reg_req_clean_link_list) == 4), "reg_req_clean_link_list size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG1 
 b'tx req update wqe checks error check enable'
*/
typedef struct reg_qp_update_err_cfg1 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg1;
static_assert((sizeof(struct reg_qp_update_err_cfg1) == 4), "reg_qp_update_err_cfg1 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG2 
 b'tx req update wqe checks error check enable'
*/
typedef struct reg_qp_update_err_cfg2 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg2;
static_assert((sizeof(struct reg_qp_update_err_cfg2) == 4), "reg_qp_update_err_cfg2 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG3 
 b'tx req update wqe checks error push to eq'
*/
typedef struct reg_qp_update_err_cfg3 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg3;
static_assert((sizeof(struct reg_qp_update_err_cfg3) == 4), "reg_qp_update_err_cfg3 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG4 
 b'tx req update wqe checks error push to eq'
*/
typedef struct reg_qp_update_err_cfg4 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg4;
static_assert((sizeof(struct reg_qp_update_err_cfg4) == 4), "reg_qp_update_err_cfg4 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG5 
 b'tx req update wqe checks error set error'
*/
typedef struct reg_qp_update_err_cfg5 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg5;
static_assert((sizeof(struct reg_qp_update_err_cfg5) == 4), "reg_qp_update_err_cfg5 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG6 
 b'tx req update wqe checks error set error'
*/
typedef struct reg_qp_update_err_cfg6 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg6;
static_assert((sizeof(struct reg_qp_update_err_cfg6) == 4), "reg_qp_update_err_cfg6 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG7 
 b'UPDATE error configurations'
*/
typedef struct reg_qp_update_err_cfg7 {
	union {
		struct {
			uint32_t db_sec_check_en : 1,
				db_asid_check_en : 1,
				db_pi_ex_wq_size_check_en : 1,
				db_wq_type_rd_check_en : 1,
				db_qpc_valid_check_en : 1,
				db_sec_err_to_eq : 1,
				db_asid_err_to_eq : 1,
				db_pi_ex_wq_size_to_eq : 1,
				db_wq_type_rd_err_to_eq : 1,
				db_qpc_valid_err_to_eq : 1,
				db_sec_err_set_err : 1,
				db_asid_err_set_err : 1,
				db_pi_ex_wq_size_set_err : 1,
				db_wq_type_rd_err_set_err : 1,
				db_qpc_valid_err_set_err : 1,
				patcher_sec_check_en : 1,
				patcher_asid_check_en : 1,
				patcher_qpc_valid_check_e : 1,
				patcher_sec_err_to_eq : 1,
				patcher_asid_err_to_eq : 1,
				patcher_qpc_valid_to_eq : 1,
				patcher_sec_err_set_err : 1,
				patcher_asid_err_set_err : 1,
				patcher_qpc_err_set_err : 1,
				tx_req_wq_type_check_en : 1,
				tx_req_qpc_valid_check_en : 1,
				tx_req_wq_type_err_to_eq : 1,
				tx_req_qpc_vld_err_to_eq : 1,
				tx_req_wq_type_err_set : 1,
				tx_req_qpc_vld_err_set : 1,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg7;
static_assert((sizeof(struct reg_qp_update_err_cfg7) == 4), "reg_qp_update_err_cfg7 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG8 
 b'UPDATE error configurations'
*/
typedef struct reg_qp_update_err_cfg8 {
	union {
		struct {
			uint32_t bbr_sec_check_en : 1,
				bbr_asid_check_en : 1,
				bbr_qpc_valid_check_en : 1,
				bbr_sec_err_to_eq : 1,
				bbr_asid_err_to_eq : 1,
				bbr_qpc_valid_err_to_eq : 1,
				bbr_sec_err_set_err : 1,
				bbr_asid_err_set_err : 1,
				bbr_qpc_valid_err_set_err : 1,
				tx_resp_qpc_check_en : 1,
				tx_resp_qpc_err_to_eq : 1,
				tx_resp_qpc_err_set_err : 1,
				rx_req_qpc_check_en : 1,
				rx_req_pse_excd_check_en : 1,
				rx_req_qpc_err_to_eq : 1,
				rx_req_pse_excd_to_eq : 1,
				rx_req_qpc_err_set_err : 1,
				rx_req_pse_excd_set_err : 1,
				rx_req_rdv_qpc_check_en : 1,
				rx_req_rdv_wq_check_en : 1,
				rx_req_rdv_qpc_err_to_eq : 1,
				rx_req_rdv_wq_err_to_eq : 1,
				rx_req_rdv_qpc_err_set_er : 1,
				rx_req_rdv_wq_err_set_err : 1,
				rx_resp_qpc_check_en : 1,
				rx_resp_qpc_err_to_eq : 1,
				rx_resp_qpc_err_set_err : 1,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg8;
static_assert((sizeof(struct reg_qp_update_err_cfg8) == 4), "reg_qp_update_err_cfg8 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG9 
 b'rx req update wqe checks error check enable'
*/
typedef struct reg_qp_update_err_cfg9 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg9;
static_assert((sizeof(struct reg_qp_update_err_cfg9) == 4), "reg_qp_update_err_cfg9 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG10 
 b'rx req update wqe checks error check enable'
*/
typedef struct reg_qp_update_err_cfg10 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg10;
static_assert((sizeof(struct reg_qp_update_err_cfg10) == 4), "reg_qp_update_err_cfg10 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG11 
 b'rx req update wqe checks error push to eq'
*/
typedef struct reg_qp_update_err_cfg11 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg11;
static_assert((sizeof(struct reg_qp_update_err_cfg11) == 4), "reg_qp_update_err_cfg11 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG12 
 b'rx req update wqe checks error push to eq'
*/
typedef struct reg_qp_update_err_cfg12 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg12;
static_assert((sizeof(struct reg_qp_update_err_cfg12) == 4), "reg_qp_update_err_cfg12 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG13 
 b'rx req update wqe checks error set error'
*/
typedef struct reg_qp_update_err_cfg13 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg13;
static_assert((sizeof(struct reg_qp_update_err_cfg13) == 4), "reg_qp_update_err_cfg13 size is not 32-bit");
/*
 QP_UPDATE_ERR_CFG14 
 b'rx req update wqe checks error set error'
*/
typedef struct reg_qp_update_err_cfg14 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_qp_update_err_cfg14;
static_assert((sizeof(struct reg_qp_update_err_cfg14) == 4), "reg_qp_update_err_cfg14 size is not 32-bit");
/*
 RES_STATIC_CONFIG 
*/
typedef struct reg_res_static_config {
	union {
		struct {
			uint32_t plru_eviction : 1,
				release_invalidate : 1,
				link_list_en : 1,
				rx_push_to_err_fifo_non_v : 1,
				cache_stop : 1,
				invalidate_writeback : 1,
				ovr_qp_valid_to_not_valid : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_res_static_config;
static_assert((sizeof(struct reg_res_static_config) == 4), "reg_res_static_config size is not 32-bit");
/*
 RES_BASE_ADDRESS_63_32 
 b'Responder cache base address bits 49 to 18'
*/
typedef struct reg_res_base_address_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_res_base_address_63_32;
static_assert((sizeof(struct reg_res_base_address_63_32) == 4), "reg_res_base_address_63_32 size is not 32-bit");
/*
 RES_BASE_ADDRESS_31_0 
 b'Responder cache base address bits 17_7'
*/
typedef struct reg_res_base_address_31_0 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_res_base_address_31_0;
static_assert((sizeof(struct reg_res_base_address_31_0) == 4), "reg_res_base_address_31_0 size is not 32-bit");
/*
 RES_CLEAN_LINK_LIST 
 b'Responder cache link list head clean'
*/
typedef struct reg_res_clean_link_list {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_res_clean_link_list;
static_assert((sizeof(struct reg_res_clean_link_list) == 4), "reg_res_clean_link_list size is not 32-bit");
/*
 RETRY_COUNT_MAX 
*/
typedef struct reg_retry_count_max {
	union {
		struct {
			uint32_t timeout : 8,
				sequence_error : 8,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_retry_count_max;
static_assert((sizeof(struct reg_retry_count_max) == 4), "reg_retry_count_max size is not 32-bit");
/*
 AXI_PROT 
*/
typedef struct reg_axi_prot {
	union {
		struct {
			uint32_t req_rd : 3,
				req_wr : 3,
				res_rd : 3,
				res_wr : 3,
				db_wr : 3,
				eq_wr : 3,
				congq_wr : 3,
				_reserved21 : 11;
		};
		uint32_t _raw;
	};
} reg_axi_prot;
static_assert((sizeof(struct reg_axi_prot) == 4), "reg_axi_prot size is not 32-bit");
/*
 RES_QPC_CACHE_INVALIDATE 
 b'Responder cache invalidation'
*/
typedef struct reg_res_qpc_cache_invalidate {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_res_qpc_cache_invalidate;
static_assert((sizeof(struct reg_res_qpc_cache_invalidate) == 4), "reg_res_qpc_cache_invalidate size is not 32-bit");
/*
 RES_QPC_CACHE_INV_STATUS 
*/
typedef struct reg_res_qpc_cache_inv_status {
	union {
		struct {
			uint32_t invalidate_done : 1,
				cache_idle : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_res_qpc_cache_inv_status;
static_assert((sizeof(struct reg_res_qpc_cache_inv_status) == 4), "reg_res_qpc_cache_inv_status size is not 32-bit");
/*
 SWL_BASE_ADDRESS_63_32 
*/
typedef struct reg_swl_base_address_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_swl_base_address_63_32;
static_assert((sizeof(struct reg_swl_base_address_63_32) == 4), "reg_swl_base_address_63_32 size is not 32-bit");
/*
 SWL_BASE_ADDRESS_31_0 
*/
typedef struct reg_swl_base_address_31_0 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_swl_base_address_31_0;
static_assert((sizeof(struct reg_swl_base_address_31_0) == 4), "reg_swl_base_address_31_0 size is not 32-bit");
/*
 MAX_QPN 
*/
typedef struct reg_max_qpn {
	union {
		struct {
			uint32_t max_qpn : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_max_qpn;
static_assert((sizeof(struct reg_max_qpn) == 4), "reg_max_qpn size is not 32-bit");
/*
 DB_FIFO_DUP_EN 
 b'DUP db fifo enable. if disable drop dup requests'
*/
typedef struct reg_db_fifo_dup_en {
	union {
		struct {
			uint32_t en : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_db_fifo_dup_en;
static_assert((sizeof(struct reg_db_fifo_dup_en) == 4), "reg_db_fifo_dup_en size is not 32-bit");
/*
 DUP_DB_FIFO 
 b'DUP db fifo'
*/
typedef struct reg_dup_db_fifo {
	union {
		struct {
			uint32_t dw : 32;
		};
		uint32_t _raw;
	};
} reg_dup_db_fifo;
static_assert((sizeof(struct reg_dup_db_fifo) == 4), "reg_dup_db_fifo size is not 32-bit");
/*
 SWIFT_CFG 
*/
typedef struct reg_swift_cfg {
	union {
		struct {
			uint32_t swift_en : 1,
				rtt_measure_method : 1,
				coalesce_init_val : 7,
				enable_coalesce : 1,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_swift_cfg;
static_assert((sizeof(struct reg_swift_cfg) == 4), "reg_swift_cfg size is not 32-bit");
/*
 MAX_CW 
 b'max congestion window'
*/
typedef struct reg_max_cw {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_max_cw;
static_assert((sizeof(struct reg_max_cw) == 4), "reg_max_cw size is not 32-bit");
/*
 MIN_CW 
 b'min congestion window'
*/
typedef struct reg_min_cw {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_min_cw;
static_assert((sizeof(struct reg_min_cw) == 4), "reg_min_cw size is not 32-bit");
/*
 MAX_OTF_PSN_SACK 
 b'sets limited state if otf psn is bigger'
*/
typedef struct reg_max_otf_psn_sack {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_max_otf_psn_sack;
static_assert((sizeof(struct reg_max_otf_psn_sack) == 4), "reg_max_otf_psn_sack size is not 32-bit");
/*
 LOG_MAX_TX_WQ_SIZE 
 b'log of max size of tx wq'
*/
typedef struct reg_log_max_tx_wq_size {
	union {
		struct {
			uint32_t val : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_log_max_tx_wq_size;
static_assert((sizeof(struct reg_log_max_tx_wq_size) == 4), "reg_log_max_tx_wq_size size is not 32-bit");
/*
 AWUSER_ATTR_TX_WQE 
 b'asid and mmu bypass for tx wqe'
*/
typedef struct reg_awuser_attr_tx_wqe {
	union {
		struct {
			uint32_t asid : 10,
				mmu_bp : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_awuser_attr_tx_wqe;
static_assert((sizeof(struct reg_awuser_attr_tx_wqe) == 4), "reg_awuser_attr_tx_wqe size is not 32-bit");
/*
 LOG_MAX_RX_WQ_SIZE 
 b'log of max size of rx wq'
*/
typedef struct reg_log_max_rx_wq_size {
	union {
		struct {
			uint32_t val : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_log_max_rx_wq_size;
static_assert((sizeof(struct reg_log_max_rx_wq_size) == 4), "reg_log_max_rx_wq_size size is not 32-bit");
/*
 AWUSER_ATTR_RX_WQE 
 b'asid and mmu bypass for rx wqe'
*/
typedef struct reg_awuser_attr_rx_wqe {
	union {
		struct {
			uint32_t asid : 10,
				mmu_bp : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_awuser_attr_rx_wqe;
static_assert((sizeof(struct reg_awuser_attr_rx_wqe) == 4), "reg_awuser_attr_rx_wqe size is not 32-bit");
/*
 LIMITED_STATE_DISABLE 
 b'chicken bits for disabling limited state checks'
*/
typedef struct reg_limited_state_disable {
	union {
		struct {
			uint32_t db_cc : 1,
				db_rdv : 1,
				db_fence : 1,
				bbr_cc : 1,
				bbr_rdv : 1,
				bbr_fence : 1,
				tx_cc_no_sack : 1,
				tx_cc_sack : 1,
				tx_rdv : 1,
				tx_fence : 1,
				rx_cc_no_sack : 1,
				rx_cc_sack : 1,
				rx_rdv : 1,
				rx_fence : 1,
				rdv_cc : 1,
				rdv_rdv : 1,
				rdv_fence : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_limited_state_disable;
static_assert((sizeof(struct reg_limited_state_disable) == 4), "reg_limited_state_disable size is not 32-bit");
/*
 CC_TIMEOUT 
*/
typedef struct reg_cc_timeout {
	union {
		struct {
			uint32_t hw_en : 1,
				sw_en : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cc_timeout;
static_assert((sizeof(struct reg_cc_timeout) == 4), "reg_cc_timeout size is not 32-bit");
/*
 CC_WINDOW_INC_EN 
*/
typedef struct reg_cc_window_inc_en {
	union {
		struct {
			uint32_t thershold : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cc_window_inc_en;
static_assert((sizeof(struct reg_cc_window_inc_en) == 4), "reg_cc_window_inc_en size is not 32-bit");
/*
 CC_TICK_WRAP 
*/
typedef struct reg_cc_tick_wrap {
	union {
		struct {
			uint32_t r : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_cc_tick_wrap;
static_assert((sizeof(struct reg_cc_tick_wrap) == 4), "reg_cc_tick_wrap size is not 32-bit");
/*
 CC_ROLLBACK 
*/
typedef struct reg_cc_rollback {
	union {
		struct {
			uint32_t mantissa : 24,
				exponent : 5,
				hw_en : 1,
				sw_en : 1,
				trigger_hw_en : 1;
		};
		uint32_t _raw;
	};
} reg_cc_rollback;
static_assert((sizeof(struct reg_cc_rollback) == 4), "reg_cc_rollback size is not 32-bit");
/*
 CC_MAX_WINDOW_SIZE 
*/
typedef struct reg_cc_max_window_size {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cc_max_window_size;
static_assert((sizeof(struct reg_cc_max_window_size) == 4), "reg_cc_max_window_size size is not 32-bit");
/*
 CC_MIN_WINDOW_SIZE 
*/
typedef struct reg_cc_min_window_size {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cc_min_window_size;
static_assert((sizeof(struct reg_cc_min_window_size) == 4), "reg_cc_min_window_size size is not 32-bit");
/*
 CC_ALPHA_LINEAR 
*/
typedef struct reg_cc_alpha_linear {
	union {
		struct {
			uint32_t mantissa : 24,
				exponent : 5,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_cc_alpha_linear;
static_assert((sizeof(struct reg_cc_alpha_linear) == 4), "reg_cc_alpha_linear size is not 32-bit");
/*
 CC_ALPHA_LOG 
*/
typedef struct reg_cc_alpha_log {
	union {
		struct {
			uint32_t mantissa : 24,
				exponent : 5,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_cc_alpha_log;
static_assert((sizeof(struct reg_cc_alpha_log) == 4), "reg_cc_alpha_log size is not 32-bit");
/*
 CC_ALPHA_LOG_THRESHOLD 
*/
typedef struct reg_cc_alpha_log_threshold {
	union {
		struct {
			uint32_t r : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cc_alpha_log_threshold;
static_assert((sizeof(struct reg_cc_alpha_log_threshold) == 4), "reg_cc_alpha_log_threshold size is not 32-bit");
/*
 CC_WINDOW_INC 
*/
typedef struct reg_cc_window_inc {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cc_window_inc;
static_assert((sizeof(struct reg_cc_window_inc) == 4), "reg_cc_window_inc size is not 32-bit");
/*
 CC_WINDOW_IN_THRESHOLD 
*/
typedef struct reg_cc_window_in_threshold {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cc_window_in_threshold;
static_assert((sizeof(struct reg_cc_window_in_threshold) == 4), "reg_cc_window_in_threshold size is not 32-bit");
/*
 DB_FIFO_CFG 
 b'CFG DB FIFOs'
*/
typedef struct reg_db_fifo_cfg {
	union {
		struct {
			uint32_t asid : 10,
				mmu_bp : 1,
				db_type : 3,
				update_msg_type : 1,
				update_msg_freq : 10,
				eq_id : 2,
				db_source : 1,
				l2_update_num_of_sob : 3,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_db_fifo_cfg;
static_assert((sizeof(struct reg_db_fifo_cfg) == 4), "reg_db_fifo_cfg size is not 32-bit");
/*
 DB_FIFO_SECURITY 
 b'db fifo security to compare with qpc security'
*/
typedef struct reg_db_fifo_security {
	union {
		struct {
			uint32_t security_level : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_db_fifo_security;
static_assert((sizeof(struct reg_db_fifo_security) == 4), "reg_db_fifo_security size is not 32-bit");
/*
 DB_FIFO_STATUS 
 b'doorbell fifo indices status'
*/
typedef struct reg_db_fifo_status {
	union {
		struct {
			uint32_t read_index : 11,
				write_index : 11,
				full : 1,
				update_sob_count : 8,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_db_fifo_status;
static_assert((sizeof(struct reg_db_fifo_status) == 4), "reg_db_fifo_status size is not 32-bit");
/*
 DB_FIFO_STATUS2 
*/
typedef struct reg_db_fifo_status2 {
	union {
		struct {
			uint32_t dw_left_for_full_entry : 6,
				num_of_full_entries : 11,
				dw_pop_count : 10,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_db_fifo_status2;
static_assert((sizeof(struct reg_db_fifo_status2) == 4), "reg_db_fifo_status2 size is not 32-bit");
/*
 DBG_INDICATION 
 b'indications of the qpc status for debug'
*/
typedef struct reg_dbg_indication {
	union {
		struct {
			uint32_t clock_gate_open : 1,
				requester_all_slices_idle : 1,
				responder_all_slices_idle : 1,
				wtd_all_slices_idle : 1,
				db_fifos_empty : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_dbg_indication;
static_assert((sizeof(struct reg_dbg_indication) == 4), "reg_dbg_indication size is not 32-bit");
/*
 DB_FIFO_AXI_PROT 
 b'awprot for DB FIFO axi writes'
*/
typedef struct reg_db_fifo_axi_prot {
	union {
		struct {
			uint32_t hbw_privileged : 3,
				hbw_secured : 3,
				hbw_unsecured : 3,
				lbw_privileged : 3,
				lbw_secured : 3,
				lbw_unsecured : 3,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_db_fifo_axi_prot;
static_assert((sizeof(struct reg_db_fifo_axi_prot) == 4), "reg_db_fifo_axi_prot size is not 32-bit");
/*
 REQ_TX_EMPTY_CNT 
 b'Indication for TX CG'
*/
typedef struct reg_req_tx_empty_cnt {
	union {
		struct {
			uint32_t cnt : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_req_tx_empty_cnt;
static_assert((sizeof(struct reg_req_tx_empty_cnt) == 4), "reg_req_tx_empty_cnt size is not 32-bit");
/*
 RES_TX_EMPTY_CNT 
 b'Indication for TX CG'
*/
typedef struct reg_res_tx_empty_cnt {
	union {
		struct {
			uint32_t cnt : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_res_tx_empty_cnt;
static_assert((sizeof(struct reg_res_tx_empty_cnt) == 4), "reg_res_tx_empty_cnt size is not 32-bit");
/*
 NUM_ROLLBACKS 
 b'number of ROLLBACK indication for DBG'
*/
typedef struct reg_num_rollbacks {
	union {
		struct {
			uint32_t num : 32;
		};
		uint32_t _raw;
	};
} reg_num_rollbacks;
static_assert((sizeof(struct reg_num_rollbacks) == 4), "reg_num_rollbacks size is not 32-bit");
/*
 LAST_QP_ROLLED_BACK 
 b'last QPN which ROLLED BACK indication for DBG'
*/
typedef struct reg_last_qp_rolled_back {
	union {
		struct {
			uint32_t qpn : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_last_qp_rolled_back;
static_assert((sizeof(struct reg_last_qp_rolled_back) == 4), "reg_last_qp_rolled_back size is not 32-bit");
/*
 NUM_TIMEOUTS 
 b'number of timeouts indication for DBG'
*/
typedef struct reg_num_timeouts {
	union {
		struct {
			uint32_t num : 32;
		};
		uint32_t _raw;
	};
} reg_num_timeouts;
static_assert((sizeof(struct reg_num_timeouts) == 4), "reg_num_timeouts size is not 32-bit");
/*
 LAST_QP_TIMED_OUT 
 b'last QPN which timeed out indication for DBG'
*/
typedef struct reg_last_qp_timed_out {
	union {
		struct {
			uint32_t qpn : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_last_qp_timed_out;
static_assert((sizeof(struct reg_last_qp_timed_out) == 4), "reg_last_qp_timed_out size is not 32-bit");
/*
 INTERRUPT_BASE 
*/
typedef struct reg_interrupt_base {
	union {
		struct {
			uint32_t r : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_interrupt_base;
static_assert((sizeof(struct reg_interrupt_base) == 4), "reg_interrupt_base size is not 32-bit");
/*
 INTERRUPT_DATA 
*/
typedef struct reg_interrupt_data {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_interrupt_data;
static_assert((sizeof(struct reg_interrupt_data) == 4), "reg_interrupt_data size is not 32-bit");
/*
 INTERRUPT_MSI 
*/
typedef struct reg_interrupt_msi {
	union {
		struct {
			uint32_t en : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_interrupt_msi;
static_assert((sizeof(struct reg_interrupt_msi) == 4), "reg_interrupt_msi size is not 32-bit");
/*
 INTERRUPT_WIRE 
*/
typedef struct reg_interrupt_wire {
	union {
		struct {
			uint32_t en : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_interrupt_wire;
static_assert((sizeof(struct reg_interrupt_wire) == 4), "reg_interrupt_wire size is not 32-bit");
/*
 QPC_REQUESTER_AR_ATTR 
 b'class_type,cache,mcid settings'
*/
typedef struct reg_qpc_requester_ar_attr {
	union {
		struct {
			uint32_t mcid : 16,
				cache : 4,
				class_type : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_qpc_requester_ar_attr;
static_assert((sizeof(struct reg_qpc_requester_ar_attr) == 4), "reg_qpc_requester_ar_attr size is not 32-bit");
/*
 QPC_REQUESTER_WORK_AW_ATTR 
 b'class_type,cache,mcid settings'
*/
typedef struct reg_qpc_requester_work_aw_attr {
	union {
		struct {
			uint32_t mcid : 16,
				cache : 4,
				class_type : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_qpc_requester_work_aw_attr;
static_assert((sizeof(struct reg_qpc_requester_work_aw_attr) == 4), "reg_qpc_requester_work_aw_attr size is not 32-bit");
/*
 QPC_REQUESTER_NO_WORK_AW_ATTR 
 b'class_type,cache,mcid settings'
*/
typedef struct reg_qpc_requester_no_work_aw_attr {
	union {
		struct {
			uint32_t mcid : 16,
				cache : 4,
				class_type : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_qpc_requester_no_work_aw_attr;
static_assert((sizeof(struct reg_qpc_requester_no_work_aw_attr) == 4), "reg_qpc_requester_no_work_aw_attr size is not 32-bit");
/*
 QPC_RESPONDER_AR_ATTR 
 b'class_type,cache,mcid settings'
*/
typedef struct reg_qpc_responder_ar_attr {
	union {
		struct {
			uint32_t mcid : 16,
				cache : 4,
				class_type : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_qpc_responder_ar_attr;
static_assert((sizeof(struct reg_qpc_responder_ar_attr) == 4), "reg_qpc_responder_ar_attr size is not 32-bit");
/*
 QPC_RESPONDER_AW_ATTR 
 b'class_type,cache,mcid settings'
*/
typedef struct reg_qpc_responder_aw_attr {
	union {
		struct {
			uint32_t mcid : 16,
				cache : 4,
				class_type : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_qpc_responder_aw_attr;
static_assert((sizeof(struct reg_qpc_responder_aw_attr) == 4), "reg_qpc_responder_aw_attr size is not 32-bit");
/*
 WTD_AWCACHE 
 b'awcache configuration'
*/
typedef struct reg_wtd_awcache {
	union {
		struct {
			uint32_t cache : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_wtd_awcache;
static_assert((sizeof(struct reg_wtd_awcache) == 4), "reg_wtd_awcache size is not 32-bit");
/*
 DB_FIFO_AWCACHE 
*/
typedef struct reg_db_fifo_awcache {
	union {
		struct {
			uint32_t cache : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_db_fifo_awcache;
static_assert((sizeof(struct reg_db_fifo_awcache) == 4), "reg_db_fifo_awcache size is not 32-bit");
/*
 EQ_AWCACHE 
*/
typedef struct reg_eq_awcache {
	union {
		struct {
			uint32_t cache : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_eq_awcache;
static_assert((sizeof(struct reg_eq_awcache) == 4), "reg_eq_awcache size is not 32-bit");
/*
 CONGQ_AWCACHE 
*/
typedef struct reg_congq_awcache {
	union {
		struct {
			uint32_t cache : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_congq_awcache;
static_assert((sizeof(struct reg_congq_awcache) == 4), "reg_congq_awcache size is not 32-bit");
/*
 WQ_BP_ADDR 
 b'WQ BP per prio written to this address'
*/
typedef struct reg_wq_bp_addr {
	union {
		struct {
			uint32_t r : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_wq_bp_addr;
static_assert((sizeof(struct reg_wq_bp_addr) == 4), "reg_wq_bp_addr size is not 32-bit");
/*
 WQ_BP_MSG_EN 
 b'each bit enables a msg to selected address'
*/
typedef struct reg_wq_bp_msg_en {
	union {
		struct {
			uint32_t en : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_wq_bp_msg_en;
static_assert((sizeof(struct reg_wq_bp_msg_en) == 4), "reg_wq_bp_msg_en size is not 32-bit");
/*
 DBG_COUNT_SELECT 
*/
typedef struct reg_dbg_count_select {
	union {
		struct {
			uint32_t r : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_dbg_count_select;
static_assert((sizeof(struct reg_dbg_count_select) == 4), "reg_dbg_count_select size is not 32-bit");
/*
 DBG_CFG 
*/
typedef struct reg_dbg_cfg {
	union {
		struct {
			uint32_t trig : 7,
				req_data_event_every_upd : 1,
				select_wtd_slice : 3,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_dbg_cfg;
static_assert((sizeof(struct reg_dbg_cfg) == 4), "reg_dbg_cfg size is not 32-bit");
/*
 PATCHER_CFG 
*/
typedef struct reg_patcher_cfg {
	union {
		struct {
			uint32_t lag_size : 6,
				direct_qpn_offset : 24,
				rx_wqe_ct_mask : 2;
		};
		uint32_t _raw;
	};
} reg_patcher_cfg;
static_assert((sizeof(struct reg_patcher_cfg) == 4), "reg_patcher_cfg size is not 32-bit");
/*
 PATCHER_CFG2 
*/
typedef struct reg_patcher_cfg2 {
	union {
		struct {
			uint32_t auto_inc_addr : 1,
				auto_inc_remote_sob_val : 5,
				auto_inc_local_sob_val : 5,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_patcher_cfg2;
static_assert((sizeof(struct reg_patcher_cfg2) == 4), "reg_patcher_cfg2 size is not 32-bit");
/*
 PATCHER_CFG3 
*/
typedef struct reg_patcher_cfg3 {
	union {
		struct {
			uint32_t dest_rank_override : 16,
				last_rank_override : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_patcher_cfg3;
static_assert((sizeof(struct reg_patcher_cfg3) == 4), "reg_patcher_cfg3 size is not 32-bit");
/*
 INTERRUPT_CAUSE 
*/
typedef struct reg_interrupt_cause {
	union {
		struct {
			uint32_t r : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_interrupt_cause;
static_assert((sizeof(struct reg_interrupt_cause) == 4), "reg_interrupt_cause size is not 32-bit");
/*
 INTERRUPT_MASK 
*/
typedef struct reg_interrupt_mask {
	union {
		struct {
			uint32_t r : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_interrupt_mask;
static_assert((sizeof(struct reg_interrupt_mask) == 4), "reg_interrupt_mask size is not 32-bit");
/*
 INTERRUPT_CLR 
*/
typedef struct reg_interrupt_clr {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_interrupt_clr;
static_assert((sizeof(struct reg_interrupt_clr) == 4), "reg_interrupt_clr size is not 32-bit");
/*
 INTERRUPT_RESP_ERR_CAUSE 
*/
typedef struct reg_interrupt_resp_err_cause {
	union {
		struct {
			uint32_t r : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_interrupt_resp_err_cause;
static_assert((sizeof(struct reg_interrupt_resp_err_cause) == 4), "reg_interrupt_resp_err_cause size is not 32-bit");
/*
 INTERRUPT_RESP_ERR_MASK 
*/
typedef struct reg_interrupt_resp_err_mask {
	union {
		struct {
			uint32_t r : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_interrupt_resp_err_mask;
static_assert((sizeof(struct reg_interrupt_resp_err_mask) == 4), "reg_interrupt_resp_err_mask size is not 32-bit");
/*
 INTERRUPR_RESP_ERR_CLR 
*/
typedef struct reg_interrupr_resp_err_clr {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_interrupr_resp_err_clr;
static_assert((sizeof(struct reg_interrupr_resp_err_clr) == 4), "reg_interrupr_resp_err_clr size is not 32-bit");
/*
 NIC_ID 
*/
typedef struct reg_nic_id {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_nic_id;
static_assert((sizeof(struct reg_nic_id) == 4), "reg_nic_id size is not 32-bit");
/*
 TMR_GW_VALID 
*/
typedef struct reg_tmr_gw_valid {
	union {
		struct {
			uint32_t r : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_tmr_gw_valid;
static_assert((sizeof(struct reg_tmr_gw_valid) == 4), "reg_tmr_gw_valid size is not 32-bit");
/*
 TMR_GW_DATA0 
*/
typedef struct reg_tmr_gw_data0 {
	union {
		struct {
			uint32_t opcode : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_tmr_gw_data0;
static_assert((sizeof(struct reg_tmr_gw_data0) == 4), "reg_tmr_gw_data0 size is not 32-bit");
/*
 TMR_GW_DATA1 
*/
typedef struct reg_tmr_gw_data1 {
	union {
		struct {
			uint32_t qpn : 24,
				timer_granularity : 7,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_tmr_gw_data1;
static_assert((sizeof(struct reg_tmr_gw_data1) == 4), "reg_tmr_gw_data1 size is not 32-bit");
/*
 RNR_RETRY_COUNT_EN 
*/
typedef struct reg_rnr_retry_count_en {
	union {
		struct {
			uint32_t r : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_rnr_retry_count_en;
static_assert((sizeof(struct reg_rnr_retry_count_en) == 4), "reg_rnr_retry_count_en size is not 32-bit");
/*
 DB_FIFO_CFG2 
 b'db fifo hbw/lbw update cfg'
*/
typedef struct reg_db_fifo_cfg2 {
	union {
		struct {
			uint32_t fifo_offset : 10,
				fifo_l2_size : 4,
				clr : 1,
				direct_port_en : 2,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_db_fifo_cfg2;
static_assert((sizeof(struct reg_db_fifo_cfg2) == 4), "reg_db_fifo_cfg2 size is not 32-bit");
/*
 DB_FIFO_UPD_ADDR_LSB 
 b'db fifo update address lsb'
*/
typedef struct reg_db_fifo_upd_addr_lsb {
	union {
		struct {
			uint32_t lsb : 32;
		};
		uint32_t _raw;
	};
} reg_db_fifo_upd_addr_lsb;
static_assert((sizeof(struct reg_db_fifo_upd_addr_lsb) == 4), "reg_db_fifo_upd_addr_lsb size is not 32-bit");
/*
 DB_FIFO_UPD_ADDR_MSB 
 b'db fifo update address msb/lbw_data'
*/
typedef struct reg_db_fifo_upd_addr_msb {
	union {
		struct {
			uint32_t addr_msb_data : 32;
		};
		uint32_t _raw;
	};
} reg_db_fifo_upd_addr_msb;
static_assert((sizeof(struct reg_db_fifo_upd_addr_msb) == 4), "reg_db_fifo_upd_addr_msb size is not 32-bit");
/*
 EVENT_QUE_BASE_ADDR_63_32 
*/
typedef struct reg_event_que_base_addr_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_event_que_base_addr_63_32;
static_assert((sizeof(struct reg_event_que_base_addr_63_32) == 4), "reg_event_que_base_addr_63_32 size is not 32-bit");
/*
 EVENT_QUE_BASE_ADDR_31_7 
*/
typedef struct reg_event_que_base_addr_31_7 {
	union {
		struct {
			uint32_t r : 25,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_event_que_base_addr_31_7;
static_assert((sizeof(struct reg_event_que_base_addr_31_7) == 4), "reg_event_que_base_addr_31_7 size is not 32-bit");
/*
 EVENT_QUE_LOG_SIZE 
*/
typedef struct reg_event_que_log_size {
	union {
		struct {
			uint32_t r : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_event_que_log_size;
static_assert((sizeof(struct reg_event_que_log_size) == 4), "reg_event_que_log_size size is not 32-bit");
/*
 EVENT_QUE_WRITE_INDEX 
*/
typedef struct reg_event_que_write_index {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_event_que_write_index;
static_assert((sizeof(struct reg_event_que_write_index) == 4), "reg_event_que_write_index size is not 32-bit");
/*
 EVENT_QUE_PRODUCER_INDEX 
*/
typedef struct reg_event_que_producer_index {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_event_que_producer_index;
static_assert((sizeof(struct reg_event_que_producer_index) == 4), "reg_event_que_producer_index size is not 32-bit");
/*
 EVENT_QUE_PI_ADDR_63_32 
*/
typedef struct reg_event_que_pi_addr_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_event_que_pi_addr_63_32;
static_assert((sizeof(struct reg_event_que_pi_addr_63_32) == 4), "reg_event_que_pi_addr_63_32 size is not 32-bit");
/*
 EVENT_QUE_PI_ADDR_31_7 
*/
typedef struct reg_event_que_pi_addr_31_7 {
	union {
		struct {
			uint32_t r : 25,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_event_que_pi_addr_31_7;
static_assert((sizeof(struct reg_event_que_pi_addr_31_7) == 4), "reg_event_que_pi_addr_31_7 size is not 32-bit");
/*
 EVENT_QUE_CONSUMER_INDEX_CB 
*/
typedef struct reg_event_que_consumer_index_cb {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_event_que_consumer_index_cb;
static_assert((sizeof(struct reg_event_que_consumer_index_cb) == 4), "reg_event_que_consumer_index_cb size is not 32-bit");
/*
 EVENT_QUE_CFG 
*/
typedef struct reg_event_que_cfg {
	union {
		struct {
			uint32_t enable : 1,
				overrun_en : 1,
				window_wraparound_en : 1,
				write_pi_en : 1,
				interrupt_per_eqe : 1,
				interrupt_first_eqe : 1,
				interrupt_ci_update : 1,
				eq_id : 2,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_event_que_cfg;
static_assert((sizeof(struct reg_event_que_cfg) == 4), "reg_event_que_cfg size is not 32-bit");
/*
 LBW_PROT 
*/
typedef struct reg_lbw_prot {
	union {
		struct {
			uint32_t interrupt : 3,
				wqe_bp : 3,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_lbw_prot;
static_assert((sizeof(struct reg_lbw_prot) == 4), "reg_lbw_prot size is not 32-bit");
/*
 MEM_WRITE_INIT 
 b'triggers array init flow'
*/
typedef struct reg_mem_write_init {
	union {
		struct {
			uint32_t req_mem_write_init : 1,
				res_mem_write_init : 1,
				db_mem_write_init : 1,
				wtd_mem_write_init : 1,
				patcher_mem_write_init : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_mem_write_init;
static_assert((sizeof(struct reg_mem_write_init) == 4), "reg_mem_write_init size is not 32-bit");
/*
 EVENT_QUE_CONSUMER_INDEX 
 b'SW updates the EQ CI'
*/
typedef struct reg_event_que_consumer_index {
	union {
		struct {
			uint32_t ci : 24,
				eq_num : 8;
		};
		uint32_t _raw;
	};
} reg_event_que_consumer_index;
static_assert((sizeof(struct reg_event_que_consumer_index) == 4), "reg_event_que_consumer_index size is not 32-bit");
/*
 TX_WQ_BASE_ADDR_63_32 
 b'base address of tx wq (4 tables)'
*/
typedef struct reg_tx_wq_base_addr_63_32 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_tx_wq_base_addr_63_32;
static_assert((sizeof(struct reg_tx_wq_base_addr_63_32) == 4), "reg_tx_wq_base_addr_63_32 size is not 32-bit");
/*
 TX_WQ_BASE_ADDR_31_0 
 b'base address of tx wq  (4 tables)'
*/
typedef struct reg_tx_wq_base_addr_31_0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_tx_wq_base_addr_31_0;
static_assert((sizeof(struct reg_tx_wq_base_addr_31_0) == 4), "reg_tx_wq_base_addr_31_0 size is not 32-bit");
/*
 RX_WQ_BASE_ADDR_63_32 
 b'base address of rx wq (4 tables)'
*/
typedef struct reg_rx_wq_base_addr_63_32 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rx_wq_base_addr_63_32;
static_assert((sizeof(struct reg_rx_wq_base_addr_63_32) == 4), "reg_rx_wq_base_addr_63_32 size is not 32-bit");
/*
 RX_WQ_BASE_ADDR_31_0 
 b'base address of rx wq (4 tables)'
*/
typedef struct reg_rx_wq_base_addr_31_0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rx_wq_base_addr_31_0;
static_assert((sizeof(struct reg_rx_wq_base_addr_31_0) == 4), "reg_rx_wq_base_addr_31_0 size is not 32-bit");
/*
 WQE_MEM_WRITE_AXI_PROT 
 b'AXI AWPROT for WQE writes'
*/
typedef struct reg_wqe_mem_write_axi_prot {
	union {
		struct {
			uint32_t tx_privilege : 3,
				tx_secured : 3,
				tx_unsecured : 3,
				rx_privilege : 3,
				rx_secured : 3,
				rx_unsecured : 3,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_wqe_mem_write_axi_prot;
static_assert((sizeof(struct reg_wqe_mem_write_axi_prot) == 4), "reg_wqe_mem_write_axi_prot size is not 32-bit");
/*
 WQ_INC_THRESHOLD 
 b'pi ci distance from wq_size'
*/
typedef struct reg_wq_inc_threshold {
	union {
		struct {
			uint32_t val : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_wq_inc_threshold;
static_assert((sizeof(struct reg_wq_inc_threshold) == 4), "reg_wq_inc_threshold size is not 32-bit");
/*
 WQ_DEC_THRESHOLD 
 b'pi ci distance from wq_size'
*/
typedef struct reg_wq_dec_threshold {
	union {
		struct {
			uint32_t val : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_wq_dec_threshold;
static_assert((sizeof(struct reg_wq_dec_threshold) == 4), "reg_wq_dec_threshold size is not 32-bit");
/*
 WTD_CONFIG 
 b'MISC config of WTD block'
*/
typedef struct reg_wtd_config {
	union {
		struct {
			uint32_t tx_wqe_cache_en : 1,
				ignore_qp_err_send2mem : 1,
				ignore_qp_err_send2cache : 1,
				_reserved6 : 3,
				wq_bp_db_accounted : 1,
				wc_timeout_en : 1,
				wc_timer_wrap : 16,
				wq_backpress_ignore_qperr : 1,
				wtd_wait_qpc_upd : 1,
				wtd_slices_num : 6;
		};
		uint32_t _raw;
	};
} reg_wtd_config;
static_assert((sizeof(struct reg_wtd_config) == 4), "reg_wtd_config size is not 32-bit");
/*
 WTD_CONFIG2 
*/
typedef struct reg_wtd_config2 {
	union {
		struct {
			uint32_t agg_en_rdv : 1,
				agg_en_not_rdv : 1,
				ignore_asid_err : 1,
				ignore_sec_err : 1,
				use_qpn_for_wq_addr : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_wtd_config2;
static_assert((sizeof(struct reg_wtd_config2) == 4), "reg_wtd_config2 size is not 32-bit");
/*
 QPC_CLOCK_GATE 
 b'qpc clock gating cfg'
*/
typedef struct reg_qpc_clock_gate {
	union {
		struct {
			uint32_t stay_open_after_trig : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_qpc_clock_gate;
static_assert((sizeof(struct reg_qpc_clock_gate) == 4), "reg_qpc_clock_gate size is not 32-bit");
/*
 QPC_CLOCK_GATE_DIS 
 b'qpc clock gating chicken bit'
*/
typedef struct reg_qpc_clock_gate_dis {
	union {
		struct {
			uint32_t qpc_cg_dis : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_qpc_clock_gate_dis;
static_assert((sizeof(struct reg_qpc_clock_gate_dis) == 4), "reg_qpc_clock_gate_dis size is not 32-bit");
/*
 CONG_QUE_BASE_ADDR_63_32 
*/
typedef struct reg_cong_que_base_addr_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_cong_que_base_addr_63_32;
static_assert((sizeof(struct reg_cong_que_base_addr_63_32) == 4), "reg_cong_que_base_addr_63_32 size is not 32-bit");
/*
 CONG_QUE_BASE_ADDR_31_7 
*/
typedef struct reg_cong_que_base_addr_31_7 {
	union {
		struct {
			uint32_t r : 25,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_cong_que_base_addr_31_7;
static_assert((sizeof(struct reg_cong_que_base_addr_31_7) == 4), "reg_cong_que_base_addr_31_7 size is not 32-bit");
/*
 CONG_QUE_LOG_SIZE 
*/
typedef struct reg_cong_que_log_size {
	union {
		struct {
			uint32_t r : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cong_que_log_size;
static_assert((sizeof(struct reg_cong_que_log_size) == 4), "reg_cong_que_log_size size is not 32-bit");
/*
 CONG_QUE_WRITE_INDEX 
*/
typedef struct reg_cong_que_write_index {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cong_que_write_index;
static_assert((sizeof(struct reg_cong_que_write_index) == 4), "reg_cong_que_write_index size is not 32-bit");
/*
 CONG_QUE_PRODUCER_INDEX 
*/
typedef struct reg_cong_que_producer_index {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cong_que_producer_index;
static_assert((sizeof(struct reg_cong_que_producer_index) == 4), "reg_cong_que_producer_index size is not 32-bit");
/*
 CONG_QUE_PI_ADDR_63_32 
*/
typedef struct reg_cong_que_pi_addr_63_32 {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_cong_que_pi_addr_63_32;
static_assert((sizeof(struct reg_cong_que_pi_addr_63_32) == 4), "reg_cong_que_pi_addr_63_32 size is not 32-bit");
/*
 CONG_QUE_PI_ADDR_31_7 
*/
typedef struct reg_cong_que_pi_addr_31_7 {
	union {
		struct {
			uint32_t r : 25,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_cong_que_pi_addr_31_7;
static_assert((sizeof(struct reg_cong_que_pi_addr_31_7) == 4), "reg_cong_que_pi_addr_31_7 size is not 32-bit");
/*
 CONG_QUE_CONSUMER_INDEX_CB 
*/
typedef struct reg_cong_que_consumer_index_cb {
	union {
		struct {
			uint32_t r : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cong_que_consumer_index_cb;
static_assert((sizeof(struct reg_cong_que_consumer_index_cb) == 4), "reg_cong_que_consumer_index_cb size is not 32-bit");
/*
 CONG_QUE_CFG 
*/
typedef struct reg_cong_que_cfg {
	union {
		struct {
			uint32_t enable : 1,
				overrun_en : 1,
				window_wraparound_en : 1,
				write_pi_en : 1,
				event_per_eqe : 1,
				event_first_eqe : 1,
				event_ci_update : 1,
				cong_q_id : 2,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_cong_que_cfg;
static_assert((sizeof(struct reg_cong_que_cfg) == 4), "reg_cong_que_cfg size is not 32-bit");
/*
 CONG_QUE_CONSUMER_INDEX 
*/
typedef struct reg_cong_que_consumer_index {
	union {
		struct {
			uint32_t ci : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cong_que_consumer_index;
static_assert((sizeof(struct reg_cong_que_consumer_index) == 4), "reg_cong_que_consumer_index size is not 32-bit");
/*
 GW_BUSY 
*/
typedef struct reg_gw_busy {
	union {
		struct {
			uint32_t r : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_gw_busy;
static_assert((sizeof(struct reg_gw_busy) == 4), "reg_gw_busy size is not 32-bit");
/*
 GW_CTRL 
*/
typedef struct reg_gw_ctrl {
	union {
		struct {
			uint32_t qpn : 24,
				requester : 1,
				doorbell_mask : 1,
				doorbell_force : 1,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_gw_ctrl;
static_assert((sizeof(struct reg_gw_ctrl) == 4), "reg_gw_ctrl size is not 32-bit");
/*
 GW_DATA 
*/
typedef struct reg_gw_data {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_gw_data;
static_assert((sizeof(struct reg_gw_data) == 4), "reg_gw_data size is not 32-bit");
/*
 GW_MASK 
*/
typedef struct reg_gw_mask {
	union {
		struct {
			uint32_t r : 32;
		};
		uint32_t _raw;
	};
} reg_gw_mask;
static_assert((sizeof(struct reg_gw_mask) == 4), "reg_gw_mask size is not 32-bit");

#ifdef __cplusplus
} /* nic_qpc namespace */
#endif

/*
 NIC_QPC block
*/

#ifdef __cplusplus

struct block_nic_qpc {
	struct nic_qpc::reg_req_qpc_cache_invalidate req_qpc_cache_invalidate;
	struct nic_qpc::reg_req_qpc_cache_inv_status req_qpc_cache_inv_status;
	struct nic_qpc::reg_req_static_config req_static_config;
	struct nic_qpc::reg_req_base_address_63_32 req_base_address_63_32;
	struct nic_qpc::reg_req_base_address_31_0 req_base_address_31_0;
	struct nic_qpc::reg_req_clean_link_list req_clean_link_list;
	struct nic_qpc::reg_qp_update_err_cfg1 qp_update_err_cfg1;
	struct nic_qpc::reg_qp_update_err_cfg2 qp_update_err_cfg2;
	struct nic_qpc::reg_qp_update_err_cfg3 qp_update_err_cfg3;
	struct nic_qpc::reg_qp_update_err_cfg4 qp_update_err_cfg4;
	struct nic_qpc::reg_qp_update_err_cfg5 qp_update_err_cfg5;
	struct nic_qpc::reg_qp_update_err_cfg6 qp_update_err_cfg6;
	struct nic_qpc::reg_qp_update_err_cfg7 qp_update_err_cfg7;
	struct nic_qpc::reg_qp_update_err_cfg8 qp_update_err_cfg8;
	struct nic_qpc::reg_qp_update_err_cfg9 qp_update_err_cfg9;
	struct nic_qpc::reg_qp_update_err_cfg10 qp_update_err_cfg10;
	struct nic_qpc::reg_qp_update_err_cfg11 qp_update_err_cfg11;
	struct nic_qpc::reg_qp_update_err_cfg12 qp_update_err_cfg12;
	struct nic_qpc::reg_qp_update_err_cfg13 qp_update_err_cfg13;
	struct nic_qpc::reg_qp_update_err_cfg14 qp_update_err_cfg14;
	struct nic_qpc::reg_res_static_config res_static_config;
	struct nic_qpc::reg_res_base_address_63_32 res_base_address_63_32;
	struct nic_qpc::reg_res_base_address_31_0 res_base_address_31_0;
	struct nic_qpc::reg_res_clean_link_list res_clean_link_list;
	struct nic_qpc::reg_retry_count_max retry_count_max;
	struct nic_qpc::reg_axi_prot axi_prot;
	struct nic_qpc::reg_res_qpc_cache_invalidate res_qpc_cache_invalidate;
	struct nic_qpc::reg_res_qpc_cache_inv_status res_qpc_cache_inv_status;
	struct nic_qpc::reg_swl_base_address_63_32 swl_base_address_63_32;
	struct nic_qpc::reg_swl_base_address_31_0 swl_base_address_31_0;
	struct nic_qpc::reg_max_qpn max_qpn;
	struct nic_qpc::reg_db_fifo_dup_en db_fifo_dup_en;
	struct nic_qpc::reg_dup_db_fifo dup_db_fifo[24];
	struct nic_qpc::reg_swift_cfg swift_cfg;
	struct nic_qpc::reg_max_cw max_cw;
	struct nic_qpc::reg_min_cw min_cw;
	struct nic_qpc::reg_max_otf_psn_sack max_otf_psn_sack;
	struct nic_qpc::reg_log_max_tx_wq_size log_max_tx_wq_size[16];
	struct nic_qpc::reg_awuser_attr_tx_wqe awuser_attr_tx_wqe[16];
	struct nic_qpc::reg_log_max_rx_wq_size log_max_rx_wq_size[16];
	struct nic_qpc::reg_awuser_attr_rx_wqe awuser_attr_rx_wqe[16];
	struct nic_qpc::reg_limited_state_disable limited_state_disable;
	uint32_t _pad500[1];
	struct nic_qpc::reg_cc_timeout cc_timeout;
	struct nic_qpc::reg_cc_window_inc_en cc_window_inc_en;
	struct nic_qpc::reg_cc_tick_wrap cc_tick_wrap;
	struct nic_qpc::reg_cc_rollback cc_rollback;
	struct nic_qpc::reg_cc_max_window_size cc_max_window_size;
	struct nic_qpc::reg_cc_min_window_size cc_min_window_size;
	struct nic_qpc::reg_cc_alpha_linear cc_alpha_linear[16];
	struct nic_qpc::reg_cc_alpha_log cc_alpha_log[16];
	struct nic_qpc::reg_cc_alpha_log_threshold cc_alpha_log_threshold[16];
	struct nic_qpc::reg_cc_window_inc cc_window_inc[16];
	struct nic_qpc::reg_cc_window_in_threshold cc_window_in_threshold[16];
	struct nic_qpc::reg_db_fifo_cfg db_fifo_cfg[24];
	struct nic_qpc::reg_db_fifo_security db_fifo_security[24];
	struct nic_qpc::reg_db_fifo_status db_fifo_status[24];
	struct nic_qpc::reg_db_fifo_status2 db_fifo_status2[24];
	uint32_t _pad1232[12];
	struct nic_qpc::reg_dbg_indication dbg_indication;
	struct nic_qpc::reg_db_fifo_axi_prot db_fifo_axi_prot;
	uint32_t _pad1288[1];
	struct nic_qpc::reg_req_tx_empty_cnt req_tx_empty_cnt;
	struct nic_qpc::reg_res_tx_empty_cnt res_tx_empty_cnt;
	struct nic_qpc::reg_num_rollbacks num_rollbacks;
	struct nic_qpc::reg_last_qp_rolled_back last_qp_rolled_back;
	struct nic_qpc::reg_num_timeouts num_timeouts;
	struct nic_qpc::reg_last_qp_timed_out last_qp_timed_out;
	uint32_t _pad1316[1];
	struct nic_qpc::reg_interrupt_base interrupt_base[8];
	struct nic_qpc::reg_interrupt_data interrupt_data[8];
	struct nic_qpc::reg_interrupt_msi interrupt_msi;
	struct nic_qpc::reg_interrupt_wire interrupt_wire;
	struct nic_qpc::reg_qpc_requester_ar_attr qpc_requester_ar_attr;
	struct nic_qpc::reg_qpc_requester_work_aw_attr qpc_requester_work_aw_attr;
	struct nic_qpc::reg_qpc_requester_no_work_aw_attr qpc_requester_no_work_aw_attr;
	struct nic_qpc::reg_qpc_responder_ar_attr qpc_responder_ar_attr;
	struct nic_qpc::reg_qpc_responder_aw_attr qpc_responder_aw_attr;
	struct nic_qpc::reg_wtd_awcache wtd_awcache;
	struct nic_qpc::reg_db_fifo_awcache db_fifo_awcache;
	struct nic_qpc::reg_eq_awcache eq_awcache;
	struct nic_qpc::reg_congq_awcache congq_awcache;
	uint32_t _pad1428[7];
	struct nic_qpc::reg_wq_bp_addr wq_bp_addr[16];
	struct nic_qpc::reg_wq_bp_msg_en wq_bp_msg_en[4];
	struct nic_qpc::reg_dbg_count_select dbg_count_select[12];
	struct nic_qpc::reg_dbg_cfg dbg_cfg;
	struct nic_qpc::reg_patcher_cfg patcher_cfg;
	struct nic_qpc::reg_patcher_cfg2 patcher_cfg2[4];
	struct nic_qpc::reg_patcher_cfg3 patcher_cfg3;
	uint32_t _pad1612[14];
	struct nic_qpc::reg_interrupt_cause interrupt_cause;
	struct nic_qpc::reg_interrupt_mask interrupt_mask;
	struct nic_qpc::reg_interrupt_clr interrupt_clr;
	struct nic_qpc::reg_interrupt_resp_err_cause interrupt_resp_err_cause;
	struct nic_qpc::reg_interrupt_resp_err_mask interrupt_resp_err_mask;
	struct nic_qpc::reg_interrupr_resp_err_clr interrupr_resp_err_clr;
	struct nic_qpc::reg_nic_id nic_id;
	uint32_t _pad1696[1];
	struct nic_qpc::reg_tmr_gw_valid tmr_gw_valid;
	struct nic_qpc::reg_tmr_gw_data0 tmr_gw_data0;
	struct nic_qpc::reg_tmr_gw_data1 tmr_gw_data1;
	struct nic_qpc::reg_rnr_retry_count_en rnr_retry_count_en;
	struct nic_qpc::reg_db_fifo_cfg2 db_fifo_cfg2[24];
	struct nic_qpc::reg_db_fifo_upd_addr_lsb db_fifo_upd_addr_lsb[24];
	struct nic_qpc::reg_db_fifo_upd_addr_msb db_fifo_upd_addr_msb[24];
	struct nic_qpc::reg_event_que_base_addr_63_32 event_que_base_addr_63_32[4];
	struct nic_qpc::reg_event_que_base_addr_31_7 event_que_base_addr_31_7[4];
	struct nic_qpc::reg_event_que_log_size event_que_log_size[4];
	struct nic_qpc::reg_event_que_write_index event_que_write_index[4];
	struct nic_qpc::reg_event_que_producer_index event_que_producer_index[4];
	struct nic_qpc::reg_event_que_pi_addr_63_32 event_que_pi_addr_63_32[4];
	struct nic_qpc::reg_event_que_pi_addr_31_7 event_que_pi_addr_31_7[4];
	struct nic_qpc::reg_event_que_consumer_index_cb event_que_consumer_index_cb[4];
	struct nic_qpc::reg_event_que_cfg event_que_cfg[4];
	struct nic_qpc::reg_lbw_prot lbw_prot;
	struct nic_qpc::reg_mem_write_init mem_write_init;
	uint32_t _pad2156[1];
	struct nic_qpc::reg_event_que_consumer_index event_que_consumer_index[4];
	struct nic_qpc::reg_tx_wq_base_addr_63_32 tx_wq_base_addr_63_32[16];
	struct nic_qpc::reg_tx_wq_base_addr_31_0 tx_wq_base_addr_31_0[16];
	struct nic_qpc::reg_rx_wq_base_addr_63_32 rx_wq_base_addr_63_32[16];
	struct nic_qpc::reg_rx_wq_base_addr_31_0 rx_wq_base_addr_31_0[16];
	struct nic_qpc::reg_wqe_mem_write_axi_prot wqe_mem_write_axi_prot;
	struct nic_qpc::reg_wq_inc_threshold wq_inc_threshold;
	struct nic_qpc::reg_wq_dec_threshold wq_dec_threshold;
	uint32_t _pad2444[2];
	struct nic_qpc::reg_wtd_config wtd_config;
	uint32_t _pad2456[4];
	struct nic_qpc::reg_wtd_config2 wtd_config2;
	uint32_t _pad2476[2];
	struct nic_qpc::reg_qpc_clock_gate qpc_clock_gate;
	struct nic_qpc::reg_qpc_clock_gate_dis qpc_clock_gate_dis;
	struct nic_qpc::reg_cong_que_base_addr_63_32 cong_que_base_addr_63_32[4];
	struct nic_qpc::reg_cong_que_base_addr_31_7 cong_que_base_addr_31_7[4];
	struct nic_qpc::reg_cong_que_log_size cong_que_log_size[4];
	struct nic_qpc::reg_cong_que_write_index cong_que_write_index[4];
	struct nic_qpc::reg_cong_que_producer_index cong_que_producer_index[4];
	struct nic_qpc::reg_cong_que_pi_addr_63_32 cong_que_pi_addr_63_32[4];
	struct nic_qpc::reg_cong_que_pi_addr_31_7 cong_que_pi_addr_31_7[4];
	struct nic_qpc::reg_cong_que_consumer_index_cb cong_que_consumer_index_cb[4];
	struct nic_qpc::reg_cong_que_cfg cong_que_cfg[4];
	struct nic_qpc::reg_cong_que_consumer_index cong_que_consumer_index[4];
	uint32_t _pad2652[34];
	struct nic_qpc::reg_gw_busy gw_busy;
	struct nic_qpc::reg_gw_ctrl gw_ctrl;
	struct nic_qpc::reg_gw_data gw_data[96];
	struct nic_qpc::reg_gw_mask gw_mask[96];
	uint32_t _pad3564[37];
	struct block_special_regs special;
};
#else

typedef struct block_nic_qpc {
	reg_req_qpc_cache_invalidate req_qpc_cache_invalidate;
	reg_req_qpc_cache_inv_status req_qpc_cache_inv_status;
	reg_req_static_config req_static_config;
	reg_req_base_address_63_32 req_base_address_63_32;
	reg_req_base_address_31_0 req_base_address_31_0;
	reg_req_clean_link_list req_clean_link_list;
	reg_qp_update_err_cfg1 qp_update_err_cfg1;
	reg_qp_update_err_cfg2 qp_update_err_cfg2;
	reg_qp_update_err_cfg3 qp_update_err_cfg3;
	reg_qp_update_err_cfg4 qp_update_err_cfg4;
	reg_qp_update_err_cfg5 qp_update_err_cfg5;
	reg_qp_update_err_cfg6 qp_update_err_cfg6;
	reg_qp_update_err_cfg7 qp_update_err_cfg7;
	reg_qp_update_err_cfg8 qp_update_err_cfg8;
	reg_qp_update_err_cfg9 qp_update_err_cfg9;
	reg_qp_update_err_cfg10 qp_update_err_cfg10;
	reg_qp_update_err_cfg11 qp_update_err_cfg11;
	reg_qp_update_err_cfg12 qp_update_err_cfg12;
	reg_qp_update_err_cfg13 qp_update_err_cfg13;
	reg_qp_update_err_cfg14 qp_update_err_cfg14;
	reg_res_static_config res_static_config;
	reg_res_base_address_63_32 res_base_address_63_32;
	reg_res_base_address_31_0 res_base_address_31_0;
	reg_res_clean_link_list res_clean_link_list;
	reg_retry_count_max retry_count_max;
	reg_axi_prot axi_prot;
	reg_res_qpc_cache_invalidate res_qpc_cache_invalidate;
	reg_res_qpc_cache_inv_status res_qpc_cache_inv_status;
	reg_swl_base_address_63_32 swl_base_address_63_32;
	reg_swl_base_address_31_0 swl_base_address_31_0;
	reg_max_qpn max_qpn;
	reg_db_fifo_dup_en db_fifo_dup_en;
	reg_dup_db_fifo dup_db_fifo[24];
	reg_swift_cfg swift_cfg;
	reg_max_cw max_cw;
	reg_min_cw min_cw;
	reg_max_otf_psn_sack max_otf_psn_sack;
	reg_log_max_tx_wq_size log_max_tx_wq_size[16];
	reg_awuser_attr_tx_wqe awuser_attr_tx_wqe[16];
	reg_log_max_rx_wq_size log_max_rx_wq_size[16];
	reg_awuser_attr_rx_wqe awuser_attr_rx_wqe[16];
	reg_limited_state_disable limited_state_disable;
	uint32_t _pad500[1];
	reg_cc_timeout cc_timeout;
	reg_cc_window_inc_en cc_window_inc_en;
	reg_cc_tick_wrap cc_tick_wrap;
	reg_cc_rollback cc_rollback;
	reg_cc_max_window_size cc_max_window_size;
	reg_cc_min_window_size cc_min_window_size;
	reg_cc_alpha_linear cc_alpha_linear[16];
	reg_cc_alpha_log cc_alpha_log[16];
	reg_cc_alpha_log_threshold cc_alpha_log_threshold[16];
	reg_cc_window_inc cc_window_inc[16];
	reg_cc_window_in_threshold cc_window_in_threshold[16];
	reg_db_fifo_cfg db_fifo_cfg[24];
	reg_db_fifo_security db_fifo_security[24];
	reg_db_fifo_status db_fifo_status[24];
	reg_db_fifo_status2 db_fifo_status2[24];
	uint32_t _pad1232[12];
	reg_dbg_indication dbg_indication;
	reg_db_fifo_axi_prot db_fifo_axi_prot;
	uint32_t _pad1288[1];
	reg_req_tx_empty_cnt req_tx_empty_cnt;
	reg_res_tx_empty_cnt res_tx_empty_cnt;
	reg_num_rollbacks num_rollbacks;
	reg_last_qp_rolled_back last_qp_rolled_back;
	reg_num_timeouts num_timeouts;
	reg_last_qp_timed_out last_qp_timed_out;
	uint32_t _pad1316[1];
	reg_interrupt_base interrupt_base[8];
	reg_interrupt_data interrupt_data[8];
	reg_interrupt_msi interrupt_msi;
	reg_interrupt_wire interrupt_wire;
	reg_qpc_requester_ar_attr qpc_requester_ar_attr;
	reg_qpc_requester_work_aw_attr qpc_requester_work_aw_attr;
	reg_qpc_requester_no_work_aw_attr qpc_requester_no_work_aw_attr;
	reg_qpc_responder_ar_attr qpc_responder_ar_attr;
	reg_qpc_responder_aw_attr qpc_responder_aw_attr;
	reg_wtd_awcache wtd_awcache;
	reg_db_fifo_awcache db_fifo_awcache;
	reg_eq_awcache eq_awcache;
	reg_congq_awcache congq_awcache;
	uint32_t _pad1428[7];
	reg_wq_bp_addr wq_bp_addr[16];
	reg_wq_bp_msg_en wq_bp_msg_en[4];
	reg_dbg_count_select dbg_count_select[12];
	reg_dbg_cfg dbg_cfg;
	reg_patcher_cfg patcher_cfg;
	reg_patcher_cfg2 patcher_cfg2[4];
	reg_patcher_cfg3 patcher_cfg3;
	uint32_t _pad1612[14];
	reg_interrupt_cause interrupt_cause;
	reg_interrupt_mask interrupt_mask;
	reg_interrupt_clr interrupt_clr;
	reg_interrupt_resp_err_cause interrupt_resp_err_cause;
	reg_interrupt_resp_err_mask interrupt_resp_err_mask;
	reg_interrupr_resp_err_clr interrupr_resp_err_clr;
	reg_nic_id nic_id;
	uint32_t _pad1696[1];
	reg_tmr_gw_valid tmr_gw_valid;
	reg_tmr_gw_data0 tmr_gw_data0;
	reg_tmr_gw_data1 tmr_gw_data1;
	reg_rnr_retry_count_en rnr_retry_count_en;
	reg_db_fifo_cfg2 db_fifo_cfg2[24];
	reg_db_fifo_upd_addr_lsb db_fifo_upd_addr_lsb[24];
	reg_db_fifo_upd_addr_msb db_fifo_upd_addr_msb[24];
	reg_event_que_base_addr_63_32 event_que_base_addr_63_32[4];
	reg_event_que_base_addr_31_7 event_que_base_addr_31_7[4];
	reg_event_que_log_size event_que_log_size[4];
	reg_event_que_write_index event_que_write_index[4];
	reg_event_que_producer_index event_que_producer_index[4];
	reg_event_que_pi_addr_63_32 event_que_pi_addr_63_32[4];
	reg_event_que_pi_addr_31_7 event_que_pi_addr_31_7[4];
	reg_event_que_consumer_index_cb event_que_consumer_index_cb[4];
	reg_event_que_cfg event_que_cfg[4];
	reg_lbw_prot lbw_prot;
	reg_mem_write_init mem_write_init;
	uint32_t _pad2156[1];
	reg_event_que_consumer_index event_que_consumer_index[4];
	reg_tx_wq_base_addr_63_32 tx_wq_base_addr_63_32[16];
	reg_tx_wq_base_addr_31_0 tx_wq_base_addr_31_0[16];
	reg_rx_wq_base_addr_63_32 rx_wq_base_addr_63_32[16];
	reg_rx_wq_base_addr_31_0 rx_wq_base_addr_31_0[16];
	reg_wqe_mem_write_axi_prot wqe_mem_write_axi_prot;
	reg_wq_inc_threshold wq_inc_threshold;
	reg_wq_dec_threshold wq_dec_threshold;
	uint32_t _pad2444[2];
	reg_wtd_config wtd_config;
	uint32_t _pad2456[4];
	reg_wtd_config2 wtd_config2;
	uint32_t _pad2476[2];
	reg_qpc_clock_gate qpc_clock_gate;
	reg_qpc_clock_gate_dis qpc_clock_gate_dis;
	reg_cong_que_base_addr_63_32 cong_que_base_addr_63_32[4];
	reg_cong_que_base_addr_31_7 cong_que_base_addr_31_7[4];
	reg_cong_que_log_size cong_que_log_size[4];
	reg_cong_que_write_index cong_que_write_index[4];
	reg_cong_que_producer_index cong_que_producer_index[4];
	reg_cong_que_pi_addr_63_32 cong_que_pi_addr_63_32[4];
	reg_cong_que_pi_addr_31_7 cong_que_pi_addr_31_7[4];
	reg_cong_que_consumer_index_cb cong_que_consumer_index_cb[4];
	reg_cong_que_cfg cong_que_cfg[4];
	reg_cong_que_consumer_index cong_que_consumer_index[4];
	uint32_t _pad2652[34];
	reg_gw_busy gw_busy;
	reg_gw_ctrl gw_ctrl;
	reg_gw_data gw_data[96];
	reg_gw_mask gw_mask[96];
	uint32_t _pad3564[37];
	block_special_regs special;
} block_nic_qpc;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_nic_qpc_defaults[] =
{
	// offset	// value
	{ 0x8   , 0x3001d             , 1 }, // req_static_config
	{ 0x18  , 0xffffffff          , 1 }, // qp_update_err_cfg1
	{ 0x1c  , 0xffffffff          , 1 }, // qp_update_err_cfg2
	{ 0x20  , 0xffffffff          , 1 }, // qp_update_err_cfg3
	{ 0x24  , 0xffffffff          , 1 }, // qp_update_err_cfg4
	{ 0x28  , 0xffffffff          , 1 }, // qp_update_err_cfg5
	{ 0x2c  , 0xffffffff          , 1 }, // qp_update_err_cfg6
	{ 0x30  , 0x3fffffff          , 1 }, // qp_update_err_cfg7
	{ 0x34  , 0x7ffffff           , 1 }, // qp_update_err_cfg8
	{ 0x38  , 0xffffffff          , 1 }, // qp_update_err_cfg9
	{ 0x3c  , 0xffffffff          , 1 }, // qp_update_err_cfg10
	{ 0x40  , 0xffffffff          , 1 }, // qp_update_err_cfg11
	{ 0x44  , 0xffffffff          , 1 }, // qp_update_err_cfg12
	{ 0x48  , 0xffffffff          , 1 }, // qp_update_err_cfg13
	{ 0x4c  , 0xffffffff          , 1 }, // qp_update_err_cfg14
	{ 0x50  , 0xd                 , 1 }, // res_static_config
	{ 0x60  , 0xffff              , 1 }, // retry_count_max
	{ 0x78  , 0xffffff            , 1 }, // max_qpn
	{ 0x7c  , 0xffffff            , 1 }, // db_fifo_dup_en
	{ 0xe0  , 0x201               , 1 }, // swift_cfg
	{ 0xe4  , 0xffffff            , 1 }, // max_cw
	{ 0xec  , 0x400               , 1 }, // max_otf_psn_sack
	{ 0xf0  , 0x14                , 16 }, // log_max_tx_wq_size
	{ 0x170 , 0x14                , 16 }, // log_max_rx_wq_size
	{ 0x1fc , 0x1                 , 1 }, // cc_window_inc_en
	{ 0x200 , 0x200               , 1 }, // cc_tick_wrap
	{ 0x568 , 0xff                , 1 }, // interrupt_msi
	{ 0x56c , 0xff                , 1 }, // interrupt_wire
	{ 0x5f0 , 0xffff              , 4 }, // wq_bp_msg_en
	{ 0x634 , 0x40                , 1 }, // patcher_cfg
	{ 0x638 , 0x109               , 4 }, // patcher_cfg2
	{ 0x688 , 0xff                , 1 }, // interrupt_mask
	{ 0x694 , 0x7f                , 1 }, // interrupt_resp_err_mask
	{ 0x6b4 , 0x18000             , 24 }, // db_fifo_cfg2
	{ 0x7f4 , 0x18                , 4 }, // event_que_log_size
	{ 0x854 , 0x68                , 4 }, // event_que_cfg
	{ 0x864 , 0x9                 , 1 }, // lbw_prot
	{ 0x980 , 0x10281             , 1 }, // wqe_mem_write_axi_prot
	{ 0x984 , 0xfff               , 1 }, // wq_inc_threshold
	{ 0x988 , 0xf                 , 1 }, // wq_dec_threshold
	{ 0x994 , 0xc000ff81          , 1 }, // wtd_config
	{ 0x9a8 , 0x3                 , 1 }, // wtd_config2
	{ 0x9b4 , 0xa                 , 1 }, // qpc_clock_gate
	{ 0x9b8 , 0x1                 , 1 }, // qpc_clock_gate_dis
	{ 0x9dc , 0x18                , 4 }, // cong_que_log_size
	{ 0xa3c , 0x8                 , 4 }, // cong_que_cfg
	{ 0xe80 , 0xffffffff          , 32 }, // glbl_priv
	{ 0xf24 , 0xffff              , 1 }, // mem_ecc_err_addr
	{ 0xf44 , 0xffffffff          , 1 }, // glbl_err_addr
	{ 0xf80 , 0xffffffff          , 32 }, // glbl_sec
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_NIC_QPC_H_ */

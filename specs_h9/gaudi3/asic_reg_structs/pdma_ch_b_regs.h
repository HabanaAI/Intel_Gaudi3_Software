/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_PDMA_CH_B_H_
#define ASIC_REG_STRUCTS_GAUDI3_PDMA_CH_B_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "axuser_hbw_regs.h"
#include "axuser_lbw_regs.h"
#include "pqm_ch_b_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace pdma_ch_b {
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
 HBW_AXCACHE 
 b'hbw axi axcache'
*/
typedef struct reg_hbw_axcache {
	union {
		struct {
			uint32_t rd : 4,
				wr : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_hbw_axcache;
static_assert((sizeof(struct reg_hbw_axcache) == 4), "reg_hbw_axcache size is not 32-bit");
/*
 LBW_AXCACHE 
 b'lbw axi axcache'
*/
typedef struct reg_lbw_axcache {
	union {
		struct {
			uint32_t rd : 4,
				wr : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_lbw_axcache;
static_assert((sizeof(struct reg_lbw_axcache) == 4), "reg_lbw_axcache size is not 32-bit");
/*
 PROT_PRIVLG 
 b"PROT[0] '1' - priv"
*/
typedef struct reg_prot_privlg {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_prot_privlg;
static_assert((sizeof(struct reg_prot_privlg) == 4), "reg_prot_privlg size is not 32-bit");
/*
 PROT_SECURE 
 b"PROT[1] '1' - non secured"
*/
typedef struct reg_prot_secure {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_prot_secure;
static_assert((sizeof(struct reg_prot_secure) == 4), "reg_prot_secure size is not 32-bit");
/*
 ECMPLTN_Q_AXCACHE 
 b'hbw axi axcache for enigine cmplt q wr'
*/
typedef struct reg_ecmpltn_q_axcache {
	union {
		struct {
			uint32_t wr : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_axcache;
static_assert((sizeof(struct reg_ecmpltn_q_axcache) == 4), "reg_ecmpltn_q_axcache size is not 32-bit");
/*
 ECMPLTN_Q_PROT_PRIVLG 
 b"PROT[0] '1' - priv"
*/
typedef struct reg_ecmpltn_q_prot_privlg {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_prot_privlg;
static_assert((sizeof(struct reg_ecmpltn_q_prot_privlg) == 4), "reg_ecmpltn_q_prot_privlg size is not 32-bit");
/*
 ECMPLTN_Q_PROT_SECURE 
 b"PROT[1] '1' - non secured"
*/
typedef struct reg_ecmpltn_q_prot_secure {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_prot_secure;
static_assert((sizeof(struct reg_ecmpltn_q_prot_secure) == 4), "reg_ecmpltn_q_prot_secure size is not 32-bit");
/*
 ERR_ENABLE 
 b'error enable'
*/
typedef struct reg_err_enable {
	union {
		struct {
			uint32_t en_msg_eng_hbw_rd_rsp_err : 1,
				en_stl_eng_hbw_rd_rsp_err : 1,
				en_msg_engine_timeout : 1,
				en_stall_engine_timeout : 1,
				en_msg_eng_lbw_rd_rsp_err : 1,
				en_stl_eng_lbw_rd_rsp_err : 1,
				en_msg_eng_hbw_wr_rsp_err : 1,
				en_stl_eng_hbw_wr_rsp_err : 1,
				en_msg_eng_lbw_wr_rsp_err : 1,
				en_stl_eng_lbw_wr_rsp_err : 1,
				en_msg_eng_msg_wr_err : 1,
				en_stl_eng_msg_wr_err : 1,
				en_msg_eng_nan_detect : 1,
				en_stl_eng_nan_detect : 1,
				en_msg_eng_inf_detect : 1,
				en_stl_eng_inf_detect : 1,
				en_msg_lbw_ch_num_exceed : 1,
				en_stl_lbw_ch_num_exceed : 1,
				en_msg_eng_src_range_err : 1,
				en_stl_eng_src_range_err : 1,
				en_msg_eng_dst_range_err : 1,
				en_stl_eng_dst_range_err : 1,
				en_msg_desc_ovf_err : 1,
				en_stl_desc_ovf_err : 1,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_err_enable;
static_assert((sizeof(struct reg_err_enable) == 4), "reg_err_enable size is not 32-bit");
/*
 ERR_STATUS 
 b'error cause, write 1 clears'
*/
typedef struct reg_err_status {
	union {
		struct {
			uint32_t eng_hbw_rd_rsp_err : 1,
				eng_timeout_err : 1,
				eng_lbw_rd_rsp_err : 1,
				eng_hbw_wr_rsp_err : 1,
				eng_lbw_wr_rsp_err : 1,
				eng_msg_wr_err : 1,
				eng_nan_det : 1,
				eng_inf_det : 1,
				eng_m_inf_det : 1,
				lbw_chn_num_exc : 1,
				src_range_err : 1,
				src_nan_det : 1,
				src_inf_det : 1,
				src_m_inf_det : 1,
				dst_range_err : 1,
				dst_nan_det : 1,
				dst_inf_det : 1,
				dst_m_inf_det : 1,
				desc_ovf : 1,
				_reserved31 : 12,
				valid : 1;
		};
		uint32_t _raw;
	};
} reg_err_status;
static_assert((sizeof(struct reg_err_status) == 4), "reg_err_status size is not 32-bit");
/*
 ERR_CTX_ID 
 b'context ID receiving the error'
*/
typedef struct reg_err_ctx_id {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_err_ctx_id;
static_assert((sizeof(struct reg_err_ctx_id) == 4), "reg_err_ctx_id size is not 32-bit");
/*
 ERR_MSG_ADDR_LO 
 b'error msg base addr lsb'
*/
typedef struct reg_err_msg_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_err_msg_addr_lo;
static_assert((sizeof(struct reg_err_msg_addr_lo) == 4), "reg_err_msg_addr_lo size is not 32-bit");
/*
 ERR_MSG_ADDR_HI 
 b'error msg base addr msb'
*/
typedef struct reg_err_msg_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_err_msg_addr_hi;
static_assert((sizeof(struct reg_err_msg_addr_hi) == 4), "reg_err_msg_addr_hi size is not 32-bit");
/*
 IDLE_IND_MASK 
 b'option to mask idle indications'
*/
typedef struct reg_idle_ind_mask {
	union {
		struct {
			uint32_t desc : 1,
				comp : 1,
				instage : 1,
				core : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_idle_ind_mask;
static_assert((sizeof(struct reg_idle_ind_mask) == 4), "reg_idle_ind_mask size is not 32-bit");
/*
 CH_PRIORITY 
 b'channel priority select'
*/
typedef struct reg_ch_priority {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_ch_priority;
static_assert((sizeof(struct reg_ch_priority) == 4), "reg_ch_priority size is not 32-bit");
/*
 CH_LBW 
 b'channel lbw configuration'
*/
typedef struct reg_ch_lbw {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_ch_lbw;
static_assert((sizeof(struct reg_ch_lbw) == 4), "reg_ch_lbw size is not 32-bit");
/*
 CFG 
 b'cfg per ch'
*/
typedef struct reg_cfg {
	union {
		struct {
			uint32_t halt : 1,
				flush : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg;
static_assert((sizeof(struct reg_cfg) == 4), "reg_cfg size is not 32-bit");
/*
 STS 
 b'sts per ch'
*/
typedef struct reg_sts {
	union {
		struct {
			uint32_t is_busy : 1,
				is_halt : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_sts;
static_assert((sizeof(struct reg_sts) == 4), "reg_sts size is not 32-bit");
/*
 CFG1 
 b'cfg per ch - priviliged'
*/
typedef struct reg_cfg1 {
	union {
		struct {
			uint32_t stop_on_err : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cfg1;
static_assert((sizeof(struct reg_cfg1) == 4), "reg_cfg1 size is not 32-bit");
/*
 ERR_MSG_WDATA 
 b'Error message wdata to send'
*/
typedef struct reg_err_msg_wdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_err_msg_wdata;
static_assert((sizeof(struct reg_err_msg_wdata) == 4), "reg_err_msg_wdata size is not 32-bit");
/*
 DBG_STS 
 b'debug information per ch'
*/
typedef struct reg_dbg_sts {
	union {
		struct {
			uint32_t gskt_empty : 1,
				gskt_full : 1,
				te_ch_busy : 1,
				wr_comp_fifo_empty : 1,
				wr_comp_fifo_full : 1,
				src_desc_fifo_empty : 1,
				src_desc_fifo_full : 1,
				instage_full : 1,
				instage_empty : 1,
				rd_agu_cs : 1,
				wr_agu_cs : 1,
				core_idle_sts : 1,
				desc_cnt_sts : 5,
				comp_cnt_sts : 5,
				wr_fifo_full : 1,
				_reserved23 : 9;
		};
		uint32_t _raw;
	};
} reg_dbg_sts;
static_assert((sizeof(struct reg_dbg_sts) == 4), "reg_dbg_sts size is not 32-bit");
/*
 STS_RD_CTX_SEL 
 b'Current descriptor in process. select dim size sts'
*/
typedef struct reg_sts_rd_ctx_sel {
	union {
		struct {
			uint32_t val : 3,
				_reserved8 : 5,
				stride : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_sts_rd_ctx_sel;
static_assert((sizeof(struct reg_sts_rd_ctx_sel) == 4), "reg_sts_rd_ctx_sel size is not 32-bit");
/*
 STS_RD_CTX_SIZE 
 b'Current desc. in process. dim size selected by sel'
*/
typedef struct reg_sts_rd_ctx_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_rd_ctx_size;
static_assert((sizeof(struct reg_sts_rd_ctx_size) == 4), "reg_sts_rd_ctx_size size is not 32-bit");
/*
 STS_RD_CTX_BASE_LO 
 b'Current descriptor in process. base address 31:0'
*/
typedef struct reg_sts_rd_ctx_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_rd_ctx_base_lo;
static_assert((sizeof(struct reg_sts_rd_ctx_base_lo) == 4), "reg_sts_rd_ctx_base_lo size is not 32-bit");
/*
 STS_RD_CTX_BASE_HI 
 b'Current descriptor in process. base address 63:32'
*/
typedef struct reg_sts_rd_ctx_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_rd_ctx_base_hi;
static_assert((sizeof(struct reg_sts_rd_ctx_base_hi) == 4), "reg_sts_rd_ctx_base_hi size is not 32-bit");
/*
 STS_RD_CTX_ID 
 b'Current descriptor in process. Context ID'
*/
typedef struct reg_sts_rd_ctx_id {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_rd_ctx_id;
static_assert((sizeof(struct reg_sts_rd_ctx_id) == 4), "reg_sts_rd_ctx_id size is not 32-bit");
/*
 STS_WR_CTX_SEL 
 b'Current descriptor in process. select dim size sts'
*/
typedef struct reg_sts_wr_ctx_sel {
	union {
		struct {
			uint32_t val : 3,
				_reserved8 : 5,
				stride : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_sts_wr_ctx_sel;
static_assert((sizeof(struct reg_sts_wr_ctx_sel) == 4), "reg_sts_wr_ctx_sel size is not 32-bit");
/*
 STS_WR_CTX_SIZE 
 b'Current desc. in process. dim size selected by sel'
*/
typedef struct reg_sts_wr_ctx_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_wr_ctx_size;
static_assert((sizeof(struct reg_sts_wr_ctx_size) == 4), "reg_sts_wr_ctx_size size is not 32-bit");
/*
 STS_WR_CTX_BASE_LO 
 b'Current descriptor in process. base address 31:0'
*/
typedef struct reg_sts_wr_ctx_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_wr_ctx_base_lo;
static_assert((sizeof(struct reg_sts_wr_ctx_base_lo) == 4), "reg_sts_wr_ctx_base_lo size is not 32-bit");
/*
 STS_WR_CTX_BASE_HI 
 b'Current descriptor in process. base address 63:32'
*/
typedef struct reg_sts_wr_ctx_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_wr_ctx_base_hi;
static_assert((sizeof(struct reg_sts_wr_ctx_base_hi) == 4), "reg_sts_wr_ctx_base_hi size is not 32-bit");
/*
 STS_WR_CTX_ID 
 b'Current descriptor in process. Context ID'
*/
typedef struct reg_sts_wr_ctx_id {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sts_wr_ctx_id;
static_assert((sizeof(struct reg_sts_wr_ctx_id) == 4), "reg_sts_wr_ctx_id size is not 32-bit");
/*
 PWRLP_STS0 
 b'power loop status - fifos, counters'
*/
typedef struct reg_pwrlp_sts0 {
	union {
		struct {
			uint32_t rlvl : 8,
				wlvl : 8,
				rcnt : 8,
				wcnt : 8;
		};
		uint32_t _raw;
	};
} reg_pwrlp_sts0;
static_assert((sizeof(struct reg_pwrlp_sts0) == 4), "reg_pwrlp_sts0 size is not 32-bit");
/*
 PWRLP_STS1 
 b'power loop status - fifos, counters'
*/
typedef struct reg_pwrlp_sts1 {
	union {
		struct {
			uint32_t rfull : 1,
				wfull : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_pwrlp_sts1;
static_assert((sizeof(struct reg_pwrlp_sts1) == 4), "reg_pwrlp_sts1 size is not 32-bit");
/*
 ERR_INT_MASK 
 b'mask engine error interrupt'
*/
typedef struct reg_err_int_mask {
	union {
		struct {
			uint32_t eng_hbw_rd_rsp_err : 1,
				eng_timeout_err : 1,
				eng_lbw_rd_rsp_err : 1,
				eng_hbw_wr_rsp_err : 1,
				eng_lbw_wr_rsp_err : 1,
				eng_msg_wr_err : 1,
				eng_nan_det : 1,
				eng_inf_det : 1,
				eng_m_inf_det : 1,
				lbw_chn_num_exc : 1,
				src_range_err : 1,
				src_nan_det : 1,
				src_inf_det : 1,
				src_m_inf_det : 1,
				dst_range_err : 1,
				dst_nan_det : 1,
				dst_inf_det : 1,
				dst_m_inf_det : 1,
				desc_ovf : 1,
				_reserved19 : 13;
		};
		uint32_t _raw;
	};
} reg_err_int_mask;
static_assert((sizeof(struct reg_err_int_mask) == 4), "reg_err_int_mask size is not 32-bit");
/*
 CFG_HBW_RSB_DATA_OCCUPANCY 
 b'HBW RSB data occupancy status'
*/
typedef struct reg_cfg_hbw_rsb_data_occupancy {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cfg_hbw_rsb_data_occupancy;
static_assert((sizeof(struct reg_cfg_hbw_rsb_data_occupancy) == 4), "reg_cfg_hbw_rsb_data_occupancy size is not 32-bit");
/*
 CFG_HBW_RSB_MD_OCCUPANCY 
 b'HBW RSB md occupancy status'
*/
typedef struct reg_cfg_hbw_rsb_md_occupancy {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cfg_hbw_rsb_md_occupancy;
static_assert((sizeof(struct reg_cfg_hbw_rsb_md_occupancy) == 4), "reg_cfg_hbw_rsb_md_occupancy size is not 32-bit");
/*
 MAX_WR_COMP_INFLIGHT 
 b'Maximum write completion message inflights'
*/
typedef struct reg_max_wr_comp_inflight {
	union {
		struct {
			uint32_t val : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_max_wr_comp_inflight;
static_assert((sizeof(struct reg_max_wr_comp_inflight) == 4), "reg_max_wr_comp_inflight size is not 32-bit");

#ifdef __cplusplus
} /* pdma_ch_b namespace */
#endif

/*
 PDMA_CH_B block
*/

#ifdef __cplusplus

struct block_pdma_ch_b {
	struct block_pqm_ch_b pqm_ch;
	uint32_t _pad112[36];
	struct block_axuser_hbw pqm_axuser_hbw;
	uint32_t _pad348[41];
	struct block_axuser_lbw pqm_axuser_lbw;
	uint32_t _pad536[59];
	struct pdma_ch_b::reg_hbw_axcache hbw_axcache;
	struct pdma_ch_b::reg_lbw_axcache lbw_axcache;
	struct pdma_ch_b::reg_prot_privlg prot_privlg;
	struct pdma_ch_b::reg_prot_secure prot_secure;
	struct pdma_ch_b::reg_ecmpltn_q_axcache ecmpltn_q_axcache;
	struct pdma_ch_b::reg_ecmpltn_q_prot_privlg ecmpltn_q_prot_privlg;
	struct pdma_ch_b::reg_ecmpltn_q_prot_secure ecmpltn_q_prot_secure;
	struct pdma_ch_b::reg_err_enable err_enable;
	struct pdma_ch_b::reg_err_status err_status;
	struct pdma_ch_b::reg_err_ctx_id err_ctx_id;
	struct pdma_ch_b::reg_err_msg_addr_lo err_msg_addr_lo;
	struct pdma_ch_b::reg_err_msg_addr_hi err_msg_addr_hi;
	struct pdma_ch_b::reg_idle_ind_mask idle_ind_mask;
	struct pdma_ch_b::reg_ch_priority ch_priority;
	struct pdma_ch_b::reg_ch_lbw ch_lbw;
	struct pdma_ch_b::reg_cfg cfg;
	struct pdma_ch_b::reg_sts sts;
	struct pdma_ch_b::reg_cfg1 cfg1;
	struct pdma_ch_b::reg_err_msg_wdata err_msg_wdata;
	struct pdma_ch_b::reg_dbg_sts dbg_sts;
	struct pdma_ch_b::reg_sts_rd_ctx_sel sts_rd_ctx_sel;
	struct pdma_ch_b::reg_sts_rd_ctx_size sts_rd_ctx_size;
	struct pdma_ch_b::reg_sts_rd_ctx_base_lo sts_rd_ctx_base_lo;
	struct pdma_ch_b::reg_sts_rd_ctx_base_hi sts_rd_ctx_base_hi;
	struct pdma_ch_b::reg_sts_rd_ctx_id sts_rd_ctx_id;
	struct pdma_ch_b::reg_sts_wr_ctx_sel sts_wr_ctx_sel;
	struct pdma_ch_b::reg_sts_wr_ctx_size sts_wr_ctx_size;
	struct pdma_ch_b::reg_sts_wr_ctx_base_lo sts_wr_ctx_base_lo;
	struct pdma_ch_b::reg_sts_wr_ctx_base_hi sts_wr_ctx_base_hi;
	struct pdma_ch_b::reg_sts_wr_ctx_id sts_wr_ctx_id;
	struct pdma_ch_b::reg_pwrlp_sts0 pwrlp_sts0;
	struct pdma_ch_b::reg_pwrlp_sts1 pwrlp_sts1;
	struct pdma_ch_b::reg_err_int_mask err_int_mask;
	struct pdma_ch_b::reg_cfg_hbw_rsb_data_occupancy cfg_hbw_rsb_data_occupancy;
	struct pdma_ch_b::reg_cfg_hbw_rsb_md_occupancy cfg_hbw_rsb_md_occupancy;
	struct pdma_ch_b::reg_max_wr_comp_inflight max_wr_comp_inflight;
	uint32_t _pad916[27];
	struct block_axuser_hbw axuser_hbw;
	uint32_t _pad1116[41];
	struct block_axuser_lbw axuser_lbw;
	uint32_t _pad1304[58];
	struct block_axuser_hbw ecmpltn_q_axuser_hbw;
	uint32_t _pad1628[521];
	struct block_special_regs special;
};
#else

typedef struct block_pdma_ch_b {
	block_pqm_ch_b pqm_ch;
	uint32_t _pad112[36];
	block_axuser_hbw pqm_axuser_hbw;
	uint32_t _pad348[41];
	block_axuser_lbw pqm_axuser_lbw;
	uint32_t _pad536[59];
	reg_hbw_axcache hbw_axcache;
	reg_lbw_axcache lbw_axcache;
	reg_prot_privlg prot_privlg;
	reg_prot_secure prot_secure;
	reg_ecmpltn_q_axcache ecmpltn_q_axcache;
	reg_ecmpltn_q_prot_privlg ecmpltn_q_prot_privlg;
	reg_ecmpltn_q_prot_secure ecmpltn_q_prot_secure;
	reg_err_enable err_enable;
	reg_err_status err_status;
	reg_err_ctx_id err_ctx_id;
	reg_err_msg_addr_lo err_msg_addr_lo;
	reg_err_msg_addr_hi err_msg_addr_hi;
	reg_idle_ind_mask idle_ind_mask;
	reg_ch_priority ch_priority;
	reg_ch_lbw ch_lbw;
	reg_cfg cfg;
	reg_sts sts;
	reg_cfg1 cfg1;
	reg_err_msg_wdata err_msg_wdata;
	reg_dbg_sts dbg_sts;
	reg_sts_rd_ctx_sel sts_rd_ctx_sel;
	reg_sts_rd_ctx_size sts_rd_ctx_size;
	reg_sts_rd_ctx_base_lo sts_rd_ctx_base_lo;
	reg_sts_rd_ctx_base_hi sts_rd_ctx_base_hi;
	reg_sts_rd_ctx_id sts_rd_ctx_id;
	reg_sts_wr_ctx_sel sts_wr_ctx_sel;
	reg_sts_wr_ctx_size sts_wr_ctx_size;
	reg_sts_wr_ctx_base_lo sts_wr_ctx_base_lo;
	reg_sts_wr_ctx_base_hi sts_wr_ctx_base_hi;
	reg_sts_wr_ctx_id sts_wr_ctx_id;
	reg_pwrlp_sts0 pwrlp_sts0;
	reg_pwrlp_sts1 pwrlp_sts1;
	reg_err_int_mask err_int_mask;
	reg_cfg_hbw_rsb_data_occupancy cfg_hbw_rsb_data_occupancy;
	reg_cfg_hbw_rsb_md_occupancy cfg_hbw_rsb_md_occupancy;
	reg_max_wr_comp_inflight max_wr_comp_inflight;
	uint32_t _pad916[27];
	block_axuser_hbw axuser_hbw;
	uint32_t _pad1116[41];
	block_axuser_lbw axuser_lbw;
	uint32_t _pad1304[58];
	block_axuser_hbw ecmpltn_q_axuser_hbw;
	uint32_t _pad1628[521];
	block_special_regs special;
} block_pdma_ch_b;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_pdma_ch_b_defaults[] =
{
	// offset	// value
	{ 0x8   , 0x2                 , 1 }, // credit
	{ 0x14  , 0x7fff              , 1 }, // err_msg_enable
	{ 0x60  , 0x7                 , 1 }, // cq_cl
	{ 0x64  , 0x33                , 1 }, // hbw_axcache
	{ 0x304 , 0x33                , 1 }, // hbw_axcache
	{ 0x350 , 0x929               , 1 }, // dbg_sts
	{ 0x390 , 0x40                , 1 }, // max_wr_comp_inflight
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_PDMA_CH_B_H_ */

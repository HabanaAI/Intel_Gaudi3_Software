/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_PQM_CH_B_H_
#define ASIC_REG_STRUCTS_GAUDI3_PQM_CH_B_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace pqm_ch_b {
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
 CREDIT 
 b'channel credits status'
*/
typedef struct reg_credit {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_credit;
static_assert((sizeof(struct reg_credit) == 4), "reg_credit size is not 32-bit");
/*
 ERR_MSG_WDATA 
 b'pqm error message value'
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
 ERR_STOP_ENABLE 
 b'error stop enable per cause'
*/
typedef struct reg_err_stop_enable {
	union {
		struct {
			uint32_t cq_rd_err_en : 1,
				cp_msg_wr_err_en : 1,
				desc_ptr_rls_msg_err_en : 1,
				cp_wreg_err_en : 1,
				cp_stop_op_en : 1,
				cp_undef_cmd_err_en : 1,
				cq_ptr_ovfl_en : 1,
				fence_udf_err_en_0 : 1,
				fence_udf_err_en_1 : 1,
				fence_udf_err_en_2 : 1,
				fence_udf_err_en_3 : 1,
				fence_ovf_err_en_0 : 1,
				fence_ovf_err_en_1 : 1,
				fence_ovf_err_en_2 : 1,
				fence_ovf_err_en_3 : 1,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_err_stop_enable;
static_assert((sizeof(struct reg_err_stop_enable) == 4), "reg_err_stop_enable size is not 32-bit");
/*
 ERR_MSG_ENABLE 
 b'enable error msg per cause'
*/
typedef struct reg_err_msg_enable {
	union {
		struct {
			uint32_t cq_rd_err_en : 1,
				cp_msg_wr_err_en : 1,
				desc_ptr_rls_msg_err_en : 1,
				cp_wreg_err_en : 1,
				cp_stop_op_en : 1,
				cp_undef_cmd_err_en : 1,
				cq_ptr_ovfl_en : 1,
				fence_udf_err_en_0 : 1,
				fence_udf_err_en_1 : 1,
				fence_udf_err_en_2 : 1,
				fence_udf_err_en_3 : 1,
				fence_ovf_err_en_0 : 1,
				fence_ovf_err_en_1 : 1,
				fence_ovf_err_en_2 : 1,
				fence_ovf_err_en_3 : 1,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_err_msg_enable;
static_assert((sizeof(struct reg_err_msg_enable) == 4), "reg_err_msg_enable size is not 32-bit");
/*
 ERR_STATUS 
 b'error cause, write 1 clears'
*/
typedef struct reg_err_status {
	union {
		struct {
			uint32_t cq_rd_err : 1,
				cp_msg_wr_err : 1,
				desc_ptr_rls_msg_err : 1,
				cp_wreg_err : 1,
				cp_stop_op : 1,
				cp_undef_cmd_err : 1,
				cq_ptr_ovfl : 1,
				fence_udf_err_0 : 1,
				fence_udf_err_1 : 1,
				fence_udf_err_2 : 1,
				fence_udf_err_3 : 1,
				fence_ovf_err_0 : 1,
				fence_ovf_err_1 : 1,
				fence_ovf_err_2 : 1,
				fence_ovf_err_3 : 1,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_err_status;
static_assert((sizeof(struct reg_err_status) == 4), "reg_err_status size is not 32-bit");
/*
 ERR_MSG_BASE_ADDR_LO 
 b'error msg base addr lsb'
*/
typedef struct reg_err_msg_base_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_err_msg_base_addr_lo;
static_assert((sizeof(struct reg_err_msg_base_addr_lo) == 4), "reg_err_msg_base_addr_lo size is not 32-bit");
/*
 ERR_MSG_BASE_ADDR_HI 
 b'error msg base addr  msb'
*/
typedef struct reg_err_msg_base_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_err_msg_base_addr_hi;
static_assert((sizeof(struct reg_err_msg_base_addr_hi) == 4), "reg_err_msg_base_addr_hi size is not 32-bit");
/*
 CQ_CP_DATA_HI 
 b'cq_cp_data per ch msb status'
*/
typedef struct reg_cq_cp_data_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_cp_data_hi;
static_assert((sizeof(struct reg_cq_cp_data_hi) == 4), "reg_cq_cp_data_hi size is not 32-bit");
/*
 CQ_CP_DATA_LO 
 b'cq_cp_data per ch lsb status'
*/
typedef struct reg_cq_cp_data_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_cp_data_lo;
static_assert((sizeof(struct reg_cq_cp_data_lo) == 4), "reg_cq_cp_data_lo size is not 32-bit");
/*
 CP_STS0 
 b'ch cp fsm status'
*/
typedef struct reg_cp_sts0 {
	union {
		struct {
			uint32_t cq_cp_channel : 3,
				reserved3 : 1,
				cq_cp_vld : 1,
				cp_cq_rdy : 1,
				cp_cq_en : 1,
				cp_idle : 1,
				ch_cs : 4,
				ch_cs_lpdma_stage : 4,
				bulk_cnt_eq0 : 1,
				msg_fifo_full : 1,
				msg_fifo_empty : 1,
				msg_no_inflight : 1,
				nop_mb_stall : 1,
				credit : 3,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cp_sts0;
static_assert((sizeof(struct reg_cp_sts0) == 4), "reg_cp_sts0 size is not 32-bit");
/*
 CP_STS1 
 b'ch cp fence and wait status'
*/
typedef struct reg_cp_sts1 {
	union {
		struct {
			uint32_t fence_id : 2,
				reserved2 : 2,
				fence_in_preogress : 4,
				fence_tgt_val : 14,
				reserved22 : 2,
				wait_id : 2,
				reserved26 : 2,
				wait_in_progress : 4;
		};
		uint32_t _raw;
	};
} reg_cp_sts1;
static_assert((sizeof(struct reg_cp_sts1) == 4), "reg_cp_sts1 size is not 32-bit");
/*
 CP_HDR_HI 
 b'ch cp current command header msb'
*/
typedef struct reg_cp_hdr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_hdr_hi;
static_assert((sizeof(struct reg_cp_hdr_hi) == 4), "reg_cp_hdr_hi size is not 32-bit");
/*
 CP_HDR_LO 
 b'ch cp current command header lsb'
*/
typedef struct reg_cp_hdr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_hdr_lo;
static_assert((sizeof(struct reg_cp_hdr_lo) == 4), "reg_cp_hdr_lo size is not 32-bit");
/*
 CQ_STS 
 b'ch cq ctl and agu status'
*/
typedef struct reg_cq_sts {
	union {
		struct {
			uint32_t mode : 2,
				wr_mode : 1,
				cq_ch_idle : 1,
				ci : 6,
				reserved10 : 2,
				pi : 6,
				reserved18 : 2,
				cs : 2,
				empty : 1,
				ovfl : 1,
				msq_idle : 1,
				ptr_idle : 1,
				msg_cs : 1,
				msg_full : 1,
				msg_empty : 1,
				msg_no_inflight : 1,
				agu_cs : 1,
				cq_agu_idle : 1;
		};
		uint32_t _raw;
	};
} reg_cq_sts;
static_assert((sizeof(struct reg_cq_sts) == 4), "reg_cq_sts size is not 32-bit");
/*
 CQ_CUR_CTX_MD 
 b'ch cq current pointer to CQE metadata status'
*/
typedef struct reg_cq_cur_ctx_md {
	union {
		struct {
			uint32_t trn_size : 12,
				index : 10,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_cq_cur_ctx_md;
static_assert((sizeof(struct reg_cq_cur_ctx_md) == 4), "reg_cq_cur_ctx_md size is not 32-bit");
/*
 CQ_CUR_CTX_ADDR_HI 
 b'ch cq current pointer to CQE addr msb sts'
*/
typedef struct reg_cq_cur_ctx_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_cur_ctx_addr_hi;
static_assert((sizeof(struct reg_cq_cur_ctx_addr_hi) == 4), "reg_cq_cur_ctx_addr_hi size is not 32-bit");
/*
 CQ_CUR_CTX_ADDR_LO 
 b'ch cq current pointer to CQE addr lsb sts'
*/
typedef struct reg_cq_cur_ctx_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_cur_ctx_addr_lo;
static_assert((sizeof(struct reg_cq_cur_ctx_addr_lo) == 4), "reg_cq_cur_ctx_addr_lo size is not 32-bit");
/*
 CQ_TBL_STS 
 b'ch cq table status'
*/
typedef struct reg_cq_tbl_sts {
	union {
		struct {
			uint32_t tail_id : 6,
				reserved6 : 2,
				head_id : 6,
				reserved14 : 2,
				ch_req_vld : 1,
				cp_cq_en : 1,
				cp_cq_rdy : 1,
				cq_cp_vld : 1,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_cq_tbl_sts;
static_assert((sizeof(struct reg_cq_tbl_sts) == 4), "reg_cq_tbl_sts size is not 32-bit");
/*
 CFG 
 b'ch cfg ctrl options (stop and flush)'
*/
typedef struct reg_cfg {
	union {
		struct {
			uint32_t cp_stop : 1,
				cq_stop : 1,
				cp_flush : 1,
				cq_flush : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cfg;
static_assert((sizeof(struct reg_cfg) == 4), "reg_cfg size is not 32-bit");
/*
 GLBL_STS 
 b'ch global sts'
*/
typedef struct reg_glbl_sts {
	union {
		struct {
			uint32_t is_busy : 1,
				is_halt : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_glbl_sts;
static_assert((sizeof(struct reg_glbl_sts) == 4), "reg_glbl_sts size is not 32-bit");
/*
 CFG1 
 b'ch cfg ctrl (stop on err & en err msg)'
*/
typedef struct reg_cfg1 {
	union {
		struct {
			uint32_t stop_on_err : 1,
				en_err_msg : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg1;
static_assert((sizeof(struct reg_cfg1) == 4), "reg_cfg1 size is not 32-bit");
/*
 CQ_CL 
 b'CQ fetch num of cache line allocations  1..37'
*/
typedef struct reg_cq_cl {
	union {
		struct {
			uint32_t alloc : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_cq_cl;
static_assert((sizeof(struct reg_cq_cl) == 4), "reg_cq_cl size is not 32-bit");
/*
 HBW_AXCACHE 
 b'axcache for pqm hbw master'
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
 b'axcache for pqm lbw master'
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
 CQ_CFG_OPT 
 b'CQ chicken bits'
*/
typedef struct reg_cq_cfg_opt {
	union {
		struct {
			uint32_t ci_view : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cq_cfg_opt;
static_assert((sizeof(struct reg_cq_cfg_opt) == 4), "reg_cq_cfg_opt size is not 32-bit");

#ifdef __cplusplus
} /* pqm_ch_b namespace */
#endif

/*
 PQM_CH_B block
*/

#ifdef __cplusplus

struct block_pqm_ch_b {
	struct pqm_ch_b::reg_prot_privlg prot_privlg;
	struct pqm_ch_b::reg_prot_secure prot_secure;
	struct pqm_ch_b::reg_credit credit;
	struct pqm_ch_b::reg_err_msg_wdata err_msg_wdata;
	struct pqm_ch_b::reg_err_stop_enable err_stop_enable;
	struct pqm_ch_b::reg_err_msg_enable err_msg_enable;
	struct pqm_ch_b::reg_err_status err_status;
	struct pqm_ch_b::reg_err_msg_base_addr_lo err_msg_base_addr_lo;
	struct pqm_ch_b::reg_err_msg_base_addr_hi err_msg_base_addr_hi;
	struct pqm_ch_b::reg_cq_cp_data_hi cq_cp_data_hi;
	struct pqm_ch_b::reg_cq_cp_data_lo cq_cp_data_lo;
	struct pqm_ch_b::reg_cp_sts0 cp_sts0;
	struct pqm_ch_b::reg_cp_sts1 cp_sts1;
	struct pqm_ch_b::reg_cp_hdr_hi cp_hdr_hi;
	struct pqm_ch_b::reg_cp_hdr_lo cp_hdr_lo;
	struct pqm_ch_b::reg_cq_sts cq_sts;
	struct pqm_ch_b::reg_cq_cur_ctx_md cq_cur_ctx_md;
	uint32_t _pad68[1];
	struct pqm_ch_b::reg_cq_cur_ctx_addr_hi cq_cur_ctx_addr_hi;
	struct pqm_ch_b::reg_cq_cur_ctx_addr_lo cq_cur_ctx_addr_lo;
	struct pqm_ch_b::reg_cq_tbl_sts cq_tbl_sts;
	struct pqm_ch_b::reg_cfg cfg;
	struct pqm_ch_b::reg_glbl_sts glbl_sts;
	struct pqm_ch_b::reg_cfg1 cfg1;
	struct pqm_ch_b::reg_cq_cl cq_cl;
	struct pqm_ch_b::reg_hbw_axcache hbw_axcache;
	struct pqm_ch_b::reg_lbw_axcache lbw_axcache;
	struct pqm_ch_b::reg_cq_cfg_opt cq_cfg_opt;
};
#else

typedef struct block_pqm_ch_b {
	reg_prot_privlg prot_privlg;
	reg_prot_secure prot_secure;
	reg_credit credit;
	reg_err_msg_wdata err_msg_wdata;
	reg_err_stop_enable err_stop_enable;
	reg_err_msg_enable err_msg_enable;
	reg_err_status err_status;
	reg_err_msg_base_addr_lo err_msg_base_addr_lo;
	reg_err_msg_base_addr_hi err_msg_base_addr_hi;
	reg_cq_cp_data_hi cq_cp_data_hi;
	reg_cq_cp_data_lo cq_cp_data_lo;
	reg_cp_sts0 cp_sts0;
	reg_cp_sts1 cp_sts1;
	reg_cp_hdr_hi cp_hdr_hi;
	reg_cp_hdr_lo cp_hdr_lo;
	reg_cq_sts cq_sts;
	reg_cq_cur_ctx_md cq_cur_ctx_md;
	uint32_t _pad68[1];
	reg_cq_cur_ctx_addr_hi cq_cur_ctx_addr_hi;
	reg_cq_cur_ctx_addr_lo cq_cur_ctx_addr_lo;
	reg_cq_tbl_sts cq_tbl_sts;
	reg_cfg cfg;
	reg_glbl_sts glbl_sts;
	reg_cfg1 cfg1;
	reg_cq_cl cq_cl;
	reg_hbw_axcache hbw_axcache;
	reg_lbw_axcache lbw_axcache;
	reg_cq_cfg_opt cq_cfg_opt;
} block_pqm_ch_b;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_pqm_ch_b_defaults[] =
{
	// offset	// value
	{ 0x8   , 0x2                 , 1 }, // credit
	{ 0x14  , 0x7fff              , 1 }, // err_msg_enable
	{ 0x60  , 0x7                 , 1 }, // cq_cl
	{ 0x64  , 0x33                , 1 }, // hbw_axcache
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_PQM_CH_B_H_ */

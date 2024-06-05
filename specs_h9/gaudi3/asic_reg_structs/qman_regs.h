/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_QMAN_H_
#define ASIC_REG_STRUCTS_GAUDI3_QMAN_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "axuser_hbw_regs.h"
#include "axuser_lbw_regs.h"
#include "ic_lbw_dbg_cnt_regs.h"
#include "qman_cgm_regs.h"
#include "qman_wr64_base_addr_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace qman {
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
 GLBL_CFG0 
 b'enable cq/arc cq/cp'
*/
typedef struct reg_glbl_cfg0 {
	union {
		struct {
			uint32_t cqf_en : 1,
				arc_cqf_en : 1,
				cp_en : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_glbl_cfg0;
static_assert((sizeof(struct reg_glbl_cfg0) == 4), "reg_glbl_cfg0 size is not 32-bit");
/*
 GLBL_CFG1 
 b'stop and flush'
*/
typedef struct reg_glbl_cfg1 {
	union {
		struct {
			uint32_t cqf_stop : 1,
				arc_cqf_stop : 1,
				cp_stop : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_glbl_cfg1;
static_assert((sizeof(struct reg_glbl_cfg1) == 4), "reg_glbl_cfg1 size is not 32-bit");
/*
 GLBL_CFG2 
 b'qman arc cq stop and flush and other arc cfgs'
*/
typedef struct reg_glbl_cfg2 {
	union {
		struct {
			uint32_t cqf_flush : 1,
				arc_cqf_flush : 1,
				cp_flush : 1,
				disable_flush_cp_rst : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_glbl_cfg2;
static_assert((sizeof(struct reg_glbl_cfg2) == 4), "reg_glbl_cfg2 size is not 32-bit");
/*
 GLBL_ERR_CFG 
 b'error msg enable for cp, cq and arc cq'
*/
typedef struct reg_glbl_err_cfg {
	union {
		struct {
			uint32_t cqf_err_msg_en : 1,
				cp_err_msg_en : 1,
				arc_cqf_err_msg_en : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_glbl_err_cfg;
static_assert((sizeof(struct reg_glbl_err_cfg) == 4), "reg_glbl_err_cfg size is not 32-bit");
/*
 GLBL_ERR_CFG1 
 b'error msg enable and stop on error'
*/
typedef struct reg_glbl_err_cfg1 {
	union {
		struct {
			uint32_t cqf_stop_on_err : 1,
				cp_stop_on_err : 1,
				arc_cqf_stop_on_err : 1,
				arc_stop_on_err : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_glbl_err_cfg1;
static_assert((sizeof(struct reg_glbl_err_cfg1) == 4), "reg_glbl_err_cfg1 size is not 32-bit");
/*
 GLBL_ERR_ARC_HALT_EN 
 b'HALT ARC EN per err indication in all QMANs'
*/
typedef struct reg_glbl_err_arc_halt_en {
	union {
		struct {
			uint32_t err_ind : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_glbl_err_arc_halt_en;
static_assert((sizeof(struct reg_glbl_err_arc_halt_en) == 4), "reg_glbl_err_arc_halt_en size is not 32-bit");
/*
 GLBL_HBW_AXCACHE 
 b'global HBW AXI AXCACHE'
*/
typedef struct reg_glbl_hbw_axcache {
	union {
		struct {
			uint32_t hbw_ar : 4,
				_reserved16 : 12,
				hbw_aw : 4,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_glbl_hbw_axcache;
static_assert((sizeof(struct reg_glbl_hbw_axcache) == 4), "reg_glbl_hbw_axcache size is not 32-bit");
/*
 GLBL_STS0 
 b'Main Cq and CP Status of idle and is_stop'
*/
typedef struct reg_glbl_sts0 {
	union {
		struct {
			uint32_t cqf_idle : 1,
				cp_idle : 1,
				cqf_is_stop : 1,
				cp_is_stop : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_glbl_sts0;
static_assert((sizeof(struct reg_glbl_sts0) == 4), "reg_glbl_sts0 size is not 32-bit");
/*
 GLBL_STS1 
 b'ARC CQ Status of idle and is_stop'
*/
typedef struct reg_glbl_sts1 {
	union {
		struct {
			uint32_t arc_cqf_idle : 1,
				arc_cqf_is_stop : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_glbl_sts1;
static_assert((sizeof(struct reg_glbl_sts1) == 4), "reg_glbl_sts1 size is not 32-bit");
/*
 GLBL_ERR_STS 
 b'Events E\trror cause'
*/
typedef struct reg_glbl_err_sts {
	union {
		struct {
			uint32_t rsvd0 : 1,
				cqf_rd_err : 1,
				cp_rd_err : 1,
				cp_undef_cmd_err : 1,
				cp_stop_op : 1,
				cp_msg_wr_err : 1,
				cp_wreg_err : 1,
				_reserved8 : 1,
				cp_fence0_ovf_err : 1,
				cp_fence1_ovf_err : 1,
				cp_fence2_ovf_err : 1,
				cp_fence3_ovf_err : 1,
				cp_fence0_udf_err : 1,
				cp_fence1_udf_err : 1,
				cp_fence2_udf_err : 1,
				cp_fence3_udf_err : 1,
				rsvd16 : 1,
				rsvd17 : 1,
				cq_wr_ififo_ci_err : 1,
				cq_wr_ctl_ci_err : 1,
				arc_cqf_rd_err : 1,
				arc_cq_wr_ififo_ci_err : 1,
				arc_cq_wr_ctl_ci_err : 1,
				arc_axi_err : 1,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_glbl_err_sts;
static_assert((sizeof(struct reg_glbl_err_sts) == 4), "reg_glbl_err_sts size is not 32-bit");
/*
 GLBL_ERR_MSG_EN 
 b'MSG EN per indication'
*/
typedef struct reg_glbl_err_msg_en {
	union {
		struct {
			uint32_t rsvd0 : 1,
				cqf_rd_err : 1,
				cp_rd_err : 1,
				cp_undef_cmd_err : 1,
				cp_stop_op : 1,
				cp_msg_wr_err : 1,
				cp_wreg_err : 1,
				_reserved8 : 1,
				cp_fence0_ovf_err : 1,
				cp_fence1_ovf_err : 1,
				cp_fence2_ovf_err : 1,
				cp_fence3_ovf_err : 1,
				cp_fence0_udf_err : 1,
				cp_fence1_udf_err : 1,
				cp_fence2_udf_err : 1,
				cp_fence3_udf_err : 1,
				rsvd16 : 1,
				rsvd17 : 1,
				cq_wr_ififo_ci_err : 1,
				cq_wr_ctl_ci_err : 1,
				arc_cqf_rd_err : 1,
				arc_cq_wr_ififo_ci_err : 1,
				arc_cq_wr_ctl_ci_err : 1,
				arc_axi_err : 1,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_glbl_err_msg_en;
static_assert((sizeof(struct reg_glbl_err_msg_en) == 4), "reg_glbl_err_msg_en size is not 32-bit");
/*
 GLBL_PROT 
 b'Protection bit per PQ,CQ,CP. 1 means secure'
*/
typedef struct reg_glbl_prot {
	union {
		struct {
			uint32_t cqf : 1,
				cp : 1,
				err : 1,
				cq_ififo_msg : 1,
				arc_cq_ififo_msg : 1,
				cq_ctl_msg : 1,
				arc_cq_ctl_msg : 1,
				_reserved8 : 1,
				arc_cqf : 1,
				arc_core : 1,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_glbl_prot;
static_assert((sizeof(struct reg_glbl_prot) == 4), "reg_glbl_prot size is not 32-bit");
/*
 CQ_CFG0 
 b'Input FIFO CFG'
*/
typedef struct reg_cq_cfg0 {
	union {
		struct {
			uint32_t _reserved1 : 1,
if_msg_en : 1,
				ctl_msg_en : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_cq_cfg0;
static_assert((sizeof(struct reg_cq_cfg0) == 4), "reg_cq_cfg0 size is not 32-bit");
/*
 CQ_STS0 
 b'Status of CQ buffer'
*/
typedef struct reg_cq_sts0 {
	union {
		struct {
			uint32_t credit_cnt : 8,
				free_cnt : 8,
				inflight_cnt : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cq_sts0;
static_assert((sizeof(struct reg_cq_sts0) == 4), "reg_cq_sts0 size is not 32-bit");
/*
 CQ_CFG1 
 b'buffer and inflight limit'
*/
typedef struct reg_cq_cfg1 {
	union {
		struct {
			uint32_t credit_lim : 8,
				_reserved16 : 8,
				max_inflight : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cq_cfg1;
static_assert((sizeof(struct reg_cq_cfg1) == 4), "reg_cq_cfg1 size is not 32-bit");
/*
 CQ_STS1 
 b'Status of CQ'
*/
typedef struct reg_cq_sts1 {
	union {
		struct {
			uint32_t buf_empty : 1,
				busy : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cq_sts1;
static_assert((sizeof(struct reg_cq_sts1) == 4), "reg_cq_sts1 size is not 32-bit");
/*
 CQ_PTR_LO 
 b'SW config port. Read base address bytes 3-0'
*/
typedef struct reg_cq_ptr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ptr_lo;
static_assert((sizeof(struct reg_cq_ptr_lo) == 4), "reg_cq_ptr_lo size is not 32-bit");
/*
 CQ_PTR_HI 
 b'SW config port. Read base address bytes 7-4'
*/
typedef struct reg_cq_ptr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ptr_hi;
static_assert((sizeof(struct reg_cq_ptr_hi) == 4), "reg_cq_ptr_hi size is not 32-bit");
/*
 CQ_TSIZE 
 b'SW config port. Read transaction size in bytes'
*/
typedef struct reg_cq_tsize {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_tsize;
static_assert((sizeof(struct reg_cq_tsize) == 4), "reg_cq_tsize size is not 32-bit");
/*
 CQ_CTL 
 b'cq ctrl - trigger push to ififo'
*/
typedef struct reg_cq_ctl {
	union {
		struct {
			uint32_t _reserved0 : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ctl;
static_assert((sizeof(struct reg_cq_ctl) == 4), "reg_cq_ctl size is not 32-bit");
/*
 CQ_TSIZE_STS 
 b'Curent transfer size in bytes'
*/
typedef struct reg_cq_tsize_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_tsize_sts;
static_assert((sizeof(struct reg_cq_tsize_sts) == 4), "reg_cq_tsize_sts size is not 32-bit");
/*
 CQ_PTR_LO_STS 
 b'Current transfer base address byte 3-0'
*/
typedef struct reg_cq_ptr_lo_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ptr_lo_sts;
static_assert((sizeof(struct reg_cq_ptr_lo_sts) == 4), "reg_cq_ptr_lo_sts size is not 32-bit");
/*
 CQ_PTR_HI_STS 
 b'Current transfer base address byte 7-4'
*/
typedef struct reg_cq_ptr_hi_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ptr_hi_sts;
static_assert((sizeof(struct reg_cq_ptr_hi_sts) == 4), "reg_cq_ptr_hi_sts size is not 32-bit");
/*
 CQ_IFIFO_STS 
 b'CQ input FIFO status'
*/
typedef struct reg_cq_ififo_sts {
	union {
		struct {
			uint32_t cnt : 3,
				_reserved4 : 1,
				rdy : 1,
				_reserved8 : 3,
				ctl_stall : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_cq_ififo_sts;
static_assert((sizeof(struct reg_cq_ififo_sts) == 4), "reg_cq_ififo_sts size is not 32-bit");
/*
 CP_MSG_BASE_ADDR 
 b'cp msg base addresses - even for lsb, odd for msb'
*/
typedef struct reg_cp_msg_base_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_msg_base_addr;
static_assert((sizeof(struct reg_cp_msg_base_addr) == 4), "reg_cp_msg_base_addr size is not 32-bit");
/*
 CP_FENCE0_RDATA 
 b'Increment value for fence 0'
*/
typedef struct reg_cp_fence0_rdata {
	union {
		struct {
			uint32_t inc_val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cp_fence0_rdata;
static_assert((sizeof(struct reg_cp_fence0_rdata) == 4), "reg_cp_fence0_rdata size is not 32-bit");
/*
 CP_FENCE1_RDATA 
 b'Increment value for fence 1'
*/
typedef struct reg_cp_fence1_rdata {
	union {
		struct {
			uint32_t inc_val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cp_fence1_rdata;
static_assert((sizeof(struct reg_cp_fence1_rdata) == 4), "reg_cp_fence1_rdata size is not 32-bit");
/*
 CP_FENCE2_RDATA 
 b'Increment value for fence 2'
*/
typedef struct reg_cp_fence2_rdata {
	union {
		struct {
			uint32_t inc_val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cp_fence2_rdata;
static_assert((sizeof(struct reg_cp_fence2_rdata) == 4), "reg_cp_fence2_rdata size is not 32-bit");
/*
 CP_FENCE3_RDATA 
 b'Increment value for fence 3'
*/
typedef struct reg_cp_fence3_rdata {
	union {
		struct {
			uint32_t inc_val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cp_fence3_rdata;
static_assert((sizeof(struct reg_cp_fence3_rdata) == 4), "reg_cp_fence3_rdata size is not 32-bit");
/*
 CP_FENCE0_CNT 
 b'Current value of fence 0'
*/
typedef struct reg_cp_fence0_cnt {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_cp_fence0_cnt;
static_assert((sizeof(struct reg_cp_fence0_cnt) == 4), "reg_cp_fence0_cnt size is not 32-bit");
/*
 CP_FENCE1_CNT 
 b'Current value of fence 1'
*/
typedef struct reg_cp_fence1_cnt {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_cp_fence1_cnt;
static_assert((sizeof(struct reg_cp_fence1_cnt) == 4), "reg_cp_fence1_cnt size is not 32-bit");
/*
 CP_FENCE2_CNT 
 b'Current value of fence 2'
*/
typedef struct reg_cp_fence2_cnt {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_cp_fence2_cnt;
static_assert((sizeof(struct reg_cp_fence2_cnt) == 4), "reg_cp_fence2_cnt size is not 32-bit");
/*
 CP_FENCE3_CNT 
 b'Current value of fence 3'
*/
typedef struct reg_cp_fence3_cnt {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_cp_fence3_cnt;
static_assert((sizeof(struct reg_cp_fence3_cnt) == 4), "reg_cp_fence3_cnt size is not 32-bit");
/*
 CP_BARRIER_CFG 
 b'Guard band to allow engine to deassert idle'
*/
typedef struct reg_cp_barrier_cfg {
	union {
		struct {
			uint32_t ebguard : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_cp_barrier_cfg;
static_assert((sizeof(struct reg_cp_barrier_cfg) == 4), "reg_cp_barrier_cfg size is not 32-bit");
/*
 CP_LDMA_BASE_ADDR 
 b'For LDMA. CTX offset relative to DMA QMAN'
*/
typedef struct reg_cp_ldma_base_addr {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_cp_ldma_base_addr;
static_assert((sizeof(struct reg_cp_ldma_base_addr) == 4), "reg_cp_ldma_base_addr size is not 32-bit");
/*
 CP_STS 
 b'status of MSG inflight count, barriers, fence'
*/
typedef struct reg_cp_sts {
	union {
		struct {
			uint32_t msg_inflight_cnt : 8,
				erdy : 1,
				switch_en : 1,
				mrdy : 1,
				sw_stop : 1,
				fence_id : 2,
				fence_in_progress : 1,
				_reserved16 : 1,
				fence_target : 14,
				cur_cq : 1,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_cp_sts;
static_assert((sizeof(struct reg_cp_sts) == 4), "reg_cp_sts size is not 32-bit");
/*
 CP_CURRENT_INST_LO 
 b'Byte 3-0 of current CP instruction'
*/
typedef struct reg_cp_current_inst_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_current_inst_lo;
static_assert((sizeof(struct reg_cp_current_inst_lo) == 4), "reg_cp_current_inst_lo size is not 32-bit");
/*
 CP_CURRENT_INST_HI 
 b'Byte 7-4 of current CP instruction'
*/
typedef struct reg_cp_current_inst_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_current_inst_hi;
static_assert((sizeof(struct reg_cp_current_inst_hi) == 4), "reg_cp_current_inst_hi size is not 32-bit");
/*
 CP_PRED 
 b'Predicates can also be updated by LOADandEXE'
*/
typedef struct reg_cp_pred {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_pred;
static_assert((sizeof(struct reg_cp_pred) == 4), "reg_cp_pred size is not 32-bit");
/*
 CP_PRED_UPEN 
 b'Bit per predicate to allow update by LOADandEXE'
*/
typedef struct reg_cp_pred_upen {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_pred_upen;
static_assert((sizeof(struct reg_cp_pred_upen) == 4), "reg_cp_pred_upen size is not 32-bit");
/*
 CP_DBG_0 
 b'debug reg to reflect CP state'
*/
typedef struct reg_cp_dbg_0 {
	union {
		struct {
			uint32_t cs : 5,
				eb_cnt_not_zero : 1,
				bulk_cnt_not_zero : 1,
				mreb_stall : 1,
				stall : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_cp_dbg_0;
static_assert((sizeof(struct reg_cp_dbg_0) == 4), "reg_cp_dbg_0 size is not 32-bit");
/*
 CP_IN_DATA_LO 
 b'Head of CQ2CP DATA Q (instructions Q) Byte 3-0'
*/
typedef struct reg_cp_in_data_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_in_data_lo;
static_assert((sizeof(struct reg_cp_in_data_lo) == 4), "reg_cp_in_data_lo size is not 32-bit");
/*
 CP_IN_DATA_HI 
 b'Head of CQ2CP DATA Q (instructions Q) Byte 7-4'
*/
typedef struct reg_cp_in_data_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_in_data_hi;
static_assert((sizeof(struct reg_cp_in_data_hi) == 4), "reg_cp_in_data_hi size is not 32-bit");
/*
 ARC_CQ_CFG0 
 b'Input FIFO CFG'
*/
typedef struct reg_arc_cq_cfg0 {
	union {
		struct {
			uint32_t _reserved1 : 1,
if_msg_en : 1,
				ctl_msg_en : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_arc_cq_cfg0;
static_assert((sizeof(struct reg_arc_cq_cfg0) == 4), "reg_arc_cq_cfg0 size is not 32-bit");
/*
 ARC_CQ_CFG1 
 b'buffer and inflight limit'
*/
typedef struct reg_arc_cq_cfg1 {
	union {
		struct {
			uint32_t credit_lim : 8,
				_reserved16 : 8,
				max_inflight : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_arc_cq_cfg1;
static_assert((sizeof(struct reg_arc_cq_cfg1) == 4), "reg_arc_cq_cfg1 size is not 32-bit");
/*
 ARC_CQ_PTR_LO 
 b'SW config port. Read base address bytes 3-0'
*/
typedef struct reg_arc_cq_ptr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ptr_lo;
static_assert((sizeof(struct reg_arc_cq_ptr_lo) == 4), "reg_arc_cq_ptr_lo size is not 32-bit");
/*
 ARC_CQ_PTR_HI 
 b'SW config port. Read base address bytes 7-4'
*/
typedef struct reg_arc_cq_ptr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ptr_hi;
static_assert((sizeof(struct reg_arc_cq_ptr_hi) == 4), "reg_arc_cq_ptr_hi size is not 32-bit");
/*
 ARC_CQ_TSIZE 
 b'SW config port. Read transaction size in bytes'
*/
typedef struct reg_arc_cq_tsize {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_tsize;
static_assert((sizeof(struct reg_arc_cq_tsize) == 4), "reg_arc_cq_tsize size is not 32-bit");
/*
 ARC_CQ_STS0 
 b'Status of ARC CQ buffer'
*/
typedef struct reg_arc_cq_sts0 {
	union {
		struct {
			uint32_t credit_cnt : 8,
				free_cnt : 8,
				inflight_cnt : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_arc_cq_sts0;
static_assert((sizeof(struct reg_arc_cq_sts0) == 4), "reg_arc_cq_sts0 size is not 32-bit");
/*
 ARC_CQ_CTL 
 b'arc cq ctrl -  trigger push to arc cq ififo'
*/
typedef struct reg_arc_cq_ctl {
	union {
		struct {
			uint32_t _reserved0 : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ctl;
static_assert((sizeof(struct reg_arc_cq_ctl) == 4), "reg_arc_cq_ctl size is not 32-bit");
/*
 ARC_CQ_IFIFO_STS 
 b'ARC CQ input FIFO status'
*/
typedef struct reg_arc_cq_ififo_sts {
	union {
		struct {
			uint32_t cnt : 3,
				_reserved4 : 1,
				rdy : 1,
				_reserved8 : 3,
				ctl_stall : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ififo_sts;
static_assert((sizeof(struct reg_arc_cq_ififo_sts) == 4), "reg_arc_cq_ififo_sts size is not 32-bit");
/*
 ARC_CQ_STS1 
 b'Status of ARC CQ'
*/
typedef struct reg_arc_cq_sts1 {
	union {
		struct {
			uint32_t buf_empty : 1,
				busy : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_arc_cq_sts1;
static_assert((sizeof(struct reg_arc_cq_sts1) == 4), "reg_arc_cq_sts1 size is not 32-bit");
/*
 ARC_CQ_TSIZE_STS 
 b'Curent transfer size in bytes'
*/
typedef struct reg_arc_cq_tsize_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_tsize_sts;
static_assert((sizeof(struct reg_arc_cq_tsize_sts) == 4), "reg_arc_cq_tsize_sts size is not 32-bit");
/*
 ARC_CQ_PTR_LO_STS 
 b'Current transfer base address byte 3-0'
*/
typedef struct reg_arc_cq_ptr_lo_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ptr_lo_sts;
static_assert((sizeof(struct reg_arc_cq_ptr_lo_sts) == 4), "reg_arc_cq_ptr_lo_sts size is not 32-bit");
/*
 ARC_CQ_PTR_HI_STS 
 b'Current transfer base address byte 7-4'
*/
typedef struct reg_arc_cq_ptr_hi_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ptr_hi_sts;
static_assert((sizeof(struct reg_arc_cq_ptr_hi_sts) == 4), "reg_arc_cq_ptr_hi_sts size is not 32-bit");
/*
 ARC_CQ_IFIFO_MSG_BASE_HI 
 b'ARC CQ IFIFO shadow CI address Byte 7-4.'
*/
typedef struct reg_arc_cq_ififo_msg_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ififo_msg_base_hi;
static_assert((sizeof(struct reg_arc_cq_ififo_msg_base_hi) == 4), "reg_arc_cq_ififo_msg_base_hi size is not 32-bit");
/*
 ARC_CQ_IFIFO_MSG_BASE_LO 
 b'ARC CQ IFIFO shadow CI address Byte 3-0'
*/
typedef struct reg_arc_cq_ififo_msg_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ififo_msg_base_lo;
static_assert((sizeof(struct reg_arc_cq_ififo_msg_base_lo) == 4), "reg_arc_cq_ififo_msg_base_lo size is not 32-bit");
/*
 ARC_CQ_CTL_MSG_BASE_HI 
 b'ARC CQ CTL shadow CI address Byte 7-4'
*/
typedef struct reg_arc_cq_ctl_msg_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ctl_msg_base_hi;
static_assert((sizeof(struct reg_arc_cq_ctl_msg_base_hi) == 4), "reg_arc_cq_ctl_msg_base_hi size is not 32-bit");
/*
 ARC_CQ_CTL_MSG_BASE_LO 
 b'ARC CQ CTL shadow CI address Byte 3-0'
*/
typedef struct reg_arc_cq_ctl_msg_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ctl_msg_base_lo;
static_assert((sizeof(struct reg_arc_cq_ctl_msg_base_lo) == 4), "reg_arc_cq_ctl_msg_base_lo size is not 32-bit");
/*
 CQ_IFIFO_MSG_BASE_HI 
 b'CQ ififo shadow CI address Byte 7-4.'
*/
typedef struct reg_cq_ififo_msg_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ififo_msg_base_hi;
static_assert((sizeof(struct reg_cq_ififo_msg_base_hi) == 4), "reg_cq_ififo_msg_base_hi size is not 32-bit");
/*
 CQ_IFIFO_MSG_BASE_LO 
 b'CQ ififo shadow CI address Byte 3-0'
*/
typedef struct reg_cq_ififo_msg_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ififo_msg_base_lo;
static_assert((sizeof(struct reg_cq_ififo_msg_base_lo) == 4), "reg_cq_ififo_msg_base_lo size is not 32-bit");
/*
 CQ_CTL_MSG_BASE_HI 
 b'CQ CTL shadow CI address Byte 7-4.'
*/
typedef struct reg_cq_ctl_msg_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ctl_msg_base_hi;
static_assert((sizeof(struct reg_cq_ctl_msg_base_hi) == 4), "reg_cq_ctl_msg_base_hi size is not 32-bit");
/*
 CQ_CTL_MSG_BASE_LO 
 b'CQ CTL shadow CI address Byte 3-0'
*/
typedef struct reg_cq_ctl_msg_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ctl_msg_base_lo;
static_assert((sizeof(struct reg_cq_ctl_msg_base_lo) == 4), "reg_cq_ctl_msg_base_lo size is not 32-bit");
/*
 CQ_IFIFO_CI 
 b'CQ IFIFO CI'
*/
typedef struct reg_cq_ififo_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ififo_ci;
static_assert((sizeof(struct reg_cq_ififo_ci) == 4), "reg_cq_ififo_ci size is not 32-bit");
/*
 ARC_CQ_IFIFO_CI 
 b'ARC CQ IFIFO CI'
*/
typedef struct reg_arc_cq_ififo_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ififo_ci;
static_assert((sizeof(struct reg_arc_cq_ififo_ci) == 4), "reg_arc_cq_ififo_ci size is not 32-bit");
/*
 CQ_CTL_CI 
 b'CQ CTL CI'
*/
typedef struct reg_cq_ctl_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_ctl_ci;
static_assert((sizeof(struct reg_cq_ctl_ci) == 4), "reg_cq_ctl_ci size is not 32-bit");
/*
 ARC_CQ_CTL_CI 
 b'ARC CQ CTL CI'
*/
typedef struct reg_arc_cq_ctl_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_cq_ctl_ci;
static_assert((sizeof(struct reg_arc_cq_ctl_ci) == 4), "reg_arc_cq_ctl_ci size is not 32-bit");
/*
 CP_CFG 
 b'more CP CFG'
*/
typedef struct reg_cp_cfg {
	union {
		struct {
			uint32_t switch_en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cp_cfg;
static_assert((sizeof(struct reg_cp_cfg) == 4), "reg_cp_cfg size is not 32-bit");
/*
 CP_EXT_SWITCH 
 b'overwrite switch state'
*/
typedef struct reg_cp_ext_switch {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cp_ext_switch;
static_assert((sizeof(struct reg_cp_ext_switch) == 4), "reg_cp_ext_switch size is not 32-bit");
/*
 ARC_LB_ADDR_BASE_LO 
 b'qman arc lbw address base bits 31:29'
*/
typedef struct reg_arc_lb_addr_base_lo {
	union {
		struct {
			uint32_t val_31_29 : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_arc_lb_addr_base_lo;
static_assert((sizeof(struct reg_arc_lb_addr_base_lo) == 4), "reg_arc_lb_addr_base_lo size is not 32-bit");
/*
 ARC_LB_ADDR_BASE_HI 
 b'qman arc lbw base address Byte 7-4.'
*/
typedef struct reg_arc_lb_addr_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_lb_addr_base_hi;
static_assert((sizeof(struct reg_arc_lb_addr_base_hi) == 4), "reg_arc_lb_addr_base_hi size is not 32-bit");
/*
 ENGINE_BASE_ADDR_HI 
 b'qman engine base addr byte 7-4'
*/
typedef struct reg_engine_base_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_engine_base_addr_hi;
static_assert((sizeof(struct reg_engine_base_addr_hi) == 4), "reg_engine_base_addr_hi size is not 32-bit");
/*
 ENGINE_BASE_ADDR_LO 
 b'qman engine base address Byte 3-0'
*/
typedef struct reg_engine_base_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_engine_base_addr_lo;
static_assert((sizeof(struct reg_engine_base_addr_lo) == 4), "reg_engine_base_addr_lo size is not 32-bit");
/*
 ENGINE_ADDR_RANGE_SIZE 
 b'qman engine address size'
*/
typedef struct reg_engine_addr_range_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_engine_addr_range_size;
static_assert((sizeof(struct reg_engine_addr_range_size) == 4), "reg_engine_addr_range_size size is not 32-bit");
/*
 QM_BASE_ADDR_HI 
 b'qman cfg base addr byte 7-4'
*/
typedef struct reg_qm_base_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qm_base_addr_hi;
static_assert((sizeof(struct reg_qm_base_addr_hi) == 4), "reg_qm_base_addr_hi size is not 32-bit");
/*
 QM_BASE_ADDR_LO 
 b'qman cfg base addr byte 3-0'
*/
typedef struct reg_qm_base_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qm_base_addr_lo;
static_assert((sizeof(struct reg_qm_base_addr_lo) == 4), "reg_qm_base_addr_lo size is not 32-bit");
/*
 GLBL_ERR_ADDR_LO 
 b'global Error LB address byte 3 to 0'
*/
typedef struct reg_glbl_err_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_err_addr_lo;
static_assert((sizeof(struct reg_glbl_err_addr_lo) == 4), "reg_glbl_err_addr_lo size is not 32-bit");
/*
 GLBL_ERR_ADDR_HI 
 b'global Error LB address byte 7 to 4'
*/
typedef struct reg_glbl_err_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_err_addr_hi;
static_assert((sizeof(struct reg_glbl_err_addr_hi) == 4), "reg_glbl_err_addr_hi size is not 32-bit");
/*
 GLBL_ERR_WDATA 
 b'global Error LB wdata to send'
*/
typedef struct reg_glbl_err_wdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_err_wdata;
static_assert((sizeof(struct reg_glbl_err_wdata) == 4), "reg_glbl_err_wdata size is not 32-bit");
/*
 L2H_MASK_LO 
 b'L2H Addr mask bit 31 to 0 to selecet HBW over LBW'
*/
typedef struct reg_l2h_mask_lo {
	union {
		struct {
			uint32_t _reserved20 : 20,
val : 12;
		};
		uint32_t _raw;
	};
} reg_l2h_mask_lo;
static_assert((sizeof(struct reg_l2h_mask_lo) == 4), "reg_l2h_mask_lo size is not 32-bit");
/*
 L2H_MASK_HI 
 b'L2H Addr mask bit 63 to 32 to selecet HBW over LBW'
*/
typedef struct reg_l2h_mask_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_l2h_mask_hi;
static_assert((sizeof(struct reg_l2h_mask_hi) == 4), "reg_l2h_mask_hi size is not 32-bit");
/*
 L2H_CMPR_LO 
 b'L2H Addr compare bit 31 to 0 to selecet HBW over LBW'
*/
typedef struct reg_l2h_cmpr_lo {
	union {
		struct {
			uint32_t _reserved20 : 20,
val : 12;
		};
		uint32_t _raw;
	};
} reg_l2h_cmpr_lo;
static_assert((sizeof(struct reg_l2h_cmpr_lo) == 4), "reg_l2h_cmpr_lo size is not 32-bit");
/*
 L2H_CMPR_HI 
 b'L2H Addr compare bit 63 to 32 to selecet HBW over LBW'
*/
typedef struct reg_l2h_cmpr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_l2h_cmpr_hi;
static_assert((sizeof(struct reg_l2h_cmpr_hi) == 4), "reg_l2h_cmpr_hi size is not 32-bit");
/*
 LOCAL_RANGE_BASE 
 b'QMAN location relative to 64KB default 0xA000'
*/
typedef struct reg_local_range_base {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_local_range_base;
static_assert((sizeof(struct reg_local_range_base) == 4), "reg_local_range_base size is not 32-bit");
/*
 LOCAL_RANGE_SIZE 
 b'Size of QMAN Address space deafult 0x1000'
*/
typedef struct reg_local_range_size {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_local_range_size;
static_assert((sizeof(struct reg_local_range_size) == 4), "reg_local_range_size size is not 32-bit");
/*
 HBW_RD_RATE_LIM_CFG_1 
 b'rate limiter qman HBW port'
*/
typedef struct reg_hbw_rd_rate_lim_cfg_1 {
	union {
		struct {
			uint32_t tout : 8,
				_reserved31 : 23,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_hbw_rd_rate_lim_cfg_1;
static_assert((sizeof(struct reg_hbw_rd_rate_lim_cfg_1) == 4), "reg_hbw_rd_rate_lim_cfg_1 size is not 32-bit");
/*
 LBW_WR_RATE_LIM_CFG_0 
 b'rate limiter qman LBW port'
*/
typedef struct reg_lbw_wr_rate_lim_cfg_0 {
	union {
		struct {
			uint32_t rst_token : 8,
				_reserved16 : 8,
				sat : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_lbw_wr_rate_lim_cfg_0;
static_assert((sizeof(struct reg_lbw_wr_rate_lim_cfg_0) == 4), "reg_lbw_wr_rate_lim_cfg_0 size is not 32-bit");
/*
 LBW_WR_RATE_LIM_CFG_1 
 b'rate limiter qman LBW port'
*/
typedef struct reg_lbw_wr_rate_lim_cfg_1 {
	union {
		struct {
			uint32_t tout : 8,
				_reserved31 : 23,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_lbw_wr_rate_lim_cfg_1;
static_assert((sizeof(struct reg_lbw_wr_rate_lim_cfg_1) == 4), "reg_lbw_wr_rate_lim_cfg_1 size is not 32-bit");
/*
 HBW_RD_RATE_LIM_CFG_0 
 b'rate limiter qman HBW port'
*/
typedef struct reg_hbw_rd_rate_lim_cfg_0 {
	union {
		struct {
			uint32_t rst_token : 8,
				_reserved16 : 8,
				sat : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_hbw_rd_rate_lim_cfg_0;
static_assert((sizeof(struct reg_hbw_rd_rate_lim_cfg_0) == 4), "reg_hbw_rd_rate_lim_cfg_0 size is not 32-bit");
/*
 IND_GW_APB_CFG 
 b'Addre and Cmd Indirect access to memories gateway'
*/
typedef struct reg_ind_gw_apb_cfg {
	union {
		struct {
			uint32_t addr : 31,
				cmd : 1;
		};
		uint32_t _raw;
	};
} reg_ind_gw_apb_cfg;
static_assert((sizeof(struct reg_ind_gw_apb_cfg) == 4), "reg_ind_gw_apb_cfg size is not 32-bit");
/*
 IND_GW_APB_WDATA 
 b'GW wdata'
*/
typedef struct reg_ind_gw_apb_wdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ind_gw_apb_wdata;
static_assert((sizeof(struct reg_ind_gw_apb_wdata) == 4), "reg_ind_gw_apb_wdata size is not 32-bit");
/*
 IND_GW_APB_RDATA 
 b'Read data Indirect access to memories gateway'
*/
typedef struct reg_ind_gw_apb_rdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ind_gw_apb_rdata;
static_assert((sizeof(struct reg_ind_gw_apb_rdata) == 4), "reg_ind_gw_apb_rdata size is not 32-bit");
/*
 IND_GW_APB_STATUS 
 b'Status Indirect access to memories gateway'
*/
typedef struct reg_ind_gw_apb_status {
	union {
		struct {
			uint32_t rdy : 1,
				err : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_ind_gw_apb_status;
static_assert((sizeof(struct reg_ind_gw_apb_status) == 4), "reg_ind_gw_apb_status size is not 32-bit");
/*
 PERF_CNT_FREE_LO 
 b'free running counter for reference bits 31:0'
*/
typedef struct reg_perf_cnt_free_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_perf_cnt_free_lo;
static_assert((sizeof(struct reg_perf_cnt_free_lo) == 4), "reg_perf_cnt_free_lo size is not 32-bit");
/*
 PERF_CNT_FREE_HI 
 b'free running counter for reference bits 63:32'
*/
typedef struct reg_perf_cnt_free_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_perf_cnt_free_hi;
static_assert((sizeof(struct reg_perf_cnt_free_hi) == 4), "reg_perf_cnt_free_hi size is not 32-bit");
/*
 PERF_CNT_IDLE_LO 
 b'idle cycles counter for monitoring, bits 31:0'
*/
typedef struct reg_perf_cnt_idle_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_perf_cnt_idle_lo;
static_assert((sizeof(struct reg_perf_cnt_idle_lo) == 4), "reg_perf_cnt_idle_lo size is not 32-bit");
/*
 PERF_CNT_IDLE_HI 
 b'idle cycles counter for monitoring, bits 63:32'
*/
typedef struct reg_perf_cnt_idle_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_perf_cnt_idle_hi;
static_assert((sizeof(struct reg_perf_cnt_idle_hi) == 4), "reg_perf_cnt_idle_hi size is not 32-bit");
/*
 PERF_CNT_CFG 
 b'enable perf_cnt and set idle mask'
*/
typedef struct reg_perf_cnt_cfg {
	union {
		struct {
			uint32_t cq_mask : 1,
				cp_mask : 1,
				agent_mask : 1,
				_reserved30 : 27,
				en_free : 1,
				en_idle : 1;
		};
		uint32_t _raw;
	};
} reg_perf_cnt_cfg;
static_assert((sizeof(struct reg_perf_cnt_cfg) == 4), "reg_perf_cnt_cfg size is not 32-bit");
/*
 CP_CUR_CH_PRGM_REG 
 b'set ch for wreg cmds and lindma'
*/
typedef struct reg_cp_cur_ch_prgm_reg {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_cp_cur_ch_prgm_reg;
static_assert((sizeof(struct reg_cp_cur_ch_prgm_reg) == 4), "reg_cp_cur_ch_prgm_reg size is not 32-bit");
/*
 ARC_CTL 
 b'qman arc ctrl options'
*/
typedef struct reg_arc_ctl {
	union {
		struct {
			uint32_t qman_arc_intr_mask : 1,
				lbu_axi2apb_mask_err_rsp : 1,
				lbu_axi2apb_blk_part_strb : 1,
				lbu_axi2apb_spl_err_resp : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_arc_ctl;
static_assert((sizeof(struct reg_arc_ctl) == 4), "reg_arc_ctl size is not 32-bit");
/*
 SEI_STATUS 
 b'sei status cause'
*/
typedef struct reg_sei_status {
	union {
		struct {
			uint32_t qm_int : 1,
				arc_int : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_sei_status;
static_assert((sizeof(struct reg_sei_status) == 4), "reg_sei_status size is not 32-bit");
/*
 SEI_MASK 
 b'sei mask'
*/
typedef struct reg_sei_mask {
	union {
		struct {
			uint32_t qm_int : 1,
				arc_int : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_sei_mask;
static_assert((sizeof(struct reg_sei_mask) == 4), "reg_sei_mask size is not 32-bit");
/*
 ARC_AXI_OVRD 
 b'ARC axi properties override'
*/
typedef struct reg_arc_axi_ovrd {
	union {
		struct {
			uint32_t arc_hbw_awuser_ovrd : 1,
				arc_hbw_aruser_ovrd : 1,
				arc_lbw_awuser_ovrd : 1,
				arc_lbw_aruser_ovrd : 1,
				arc_hbw_awprot_ovrd : 1,
				arc_hbw_arprot_ovrd : 1,
				arc_lbw_awprot_ovrd : 1,
				arc_lbw_arprot_ovrd : 1,
				arc_hbw_awcache_ovrd : 1,
				arc_hbw_arcache_ovrd : 1,
				arc_lbw_awcache_ovrd : 1,
				arc_lbw_arcache_ovrd : 1,
				arc_lbw_buser_ovrd : 1,
				_reserved13 : 19;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ovrd;
static_assert((sizeof(struct reg_arc_axi_ovrd) == 4), "reg_arc_axi_ovrd size is not 32-bit");
/*
 GLBL_LBW_AXCACHE 
 b'global LBW AXI AXCACHE'
*/
typedef struct reg_glbl_lbw_axcache {
	union {
		struct {
			uint32_t lbw_aw : 4,
				lbw_ar : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_glbl_lbw_axcache;
static_assert((sizeof(struct reg_glbl_lbw_axcache) == 4), "reg_glbl_lbw_axcache size is not 32-bit");

#ifdef __cplusplus
} /* qman namespace */
#endif

/*
 QMAN block
*/

#ifdef __cplusplus

struct block_qman {
	struct qman::reg_glbl_cfg0 glbl_cfg0;
	struct qman::reg_glbl_cfg1 glbl_cfg1;
	struct qman::reg_glbl_cfg2 glbl_cfg2;
	struct qman::reg_glbl_err_cfg glbl_err_cfg;
	struct qman::reg_glbl_err_cfg1 glbl_err_cfg1;
	struct qman::reg_glbl_err_arc_halt_en glbl_err_arc_halt_en;
	struct qman::reg_glbl_hbw_axcache glbl_hbw_axcache;
	struct qman::reg_glbl_sts0 glbl_sts0;
	struct qman::reg_glbl_sts1 glbl_sts1;
	struct qman::reg_glbl_err_sts glbl_err_sts;
	struct qman::reg_glbl_err_msg_en glbl_err_msg_en;
	struct qman::reg_glbl_prot glbl_prot;
	struct qman::reg_cq_cfg0 cq_cfg0;
	struct qman::reg_cq_sts0 cq_sts0;
	struct qman::reg_cq_cfg1 cq_cfg1;
	struct qman::reg_cq_sts1 cq_sts1;
	struct qman::reg_cq_ptr_lo cq_ptr_lo;
	struct qman::reg_cq_ptr_hi cq_ptr_hi;
	struct qman::reg_cq_tsize cq_tsize;
	struct qman::reg_cq_ctl cq_ctl;
	struct qman::reg_cq_tsize_sts cq_tsize_sts;
	struct qman::reg_cq_ptr_lo_sts cq_ptr_lo_sts;
	struct qman::reg_cq_ptr_hi_sts cq_ptr_hi_sts;
	struct qman::reg_cq_ififo_sts cq_ififo_sts;
	struct qman::reg_cp_msg_base_addr cp_msg_base_addr[16];
	struct qman::reg_cp_fence0_rdata cp_fence0_rdata;
	struct qman::reg_cp_fence1_rdata cp_fence1_rdata;
	struct qman::reg_cp_fence2_rdata cp_fence2_rdata;
	struct qman::reg_cp_fence3_rdata cp_fence3_rdata;
	struct qman::reg_cp_fence0_cnt cp_fence0_cnt;
	struct qman::reg_cp_fence1_cnt cp_fence1_cnt;
	struct qman::reg_cp_fence2_cnt cp_fence2_cnt;
	struct qman::reg_cp_fence3_cnt cp_fence3_cnt;
	struct qman::reg_cp_barrier_cfg cp_barrier_cfg;
	struct qman::reg_cp_ldma_base_addr cp_ldma_base_addr;
	struct qman::reg_cp_sts cp_sts;
	struct qman::reg_cp_current_inst_lo cp_current_inst_lo;
	struct qman::reg_cp_current_inst_hi cp_current_inst_hi;
	struct qman::reg_cp_pred cp_pred;
	struct qman::reg_cp_pred_upen cp_pred_upen;
	struct qman::reg_cp_dbg_0 cp_dbg_0;
	struct qman::reg_cp_in_data_lo cp_in_data_lo;
	struct qman::reg_cp_in_data_hi cp_in_data_hi;
	struct qman::reg_arc_cq_cfg0 arc_cq_cfg0;
	struct qman::reg_arc_cq_cfg1 arc_cq_cfg1;
	struct qman::reg_arc_cq_ptr_lo arc_cq_ptr_lo;
	struct qman::reg_arc_cq_ptr_hi arc_cq_ptr_hi;
	struct qman::reg_arc_cq_tsize arc_cq_tsize;
	struct qman::reg_arc_cq_sts0 arc_cq_sts0;
	struct qman::reg_arc_cq_ctl arc_cq_ctl;
	struct qman::reg_arc_cq_ififo_sts arc_cq_ififo_sts;
	struct qman::reg_arc_cq_sts1 arc_cq_sts1;
	struct qman::reg_arc_cq_tsize_sts arc_cq_tsize_sts;
	struct qman::reg_arc_cq_ptr_lo_sts arc_cq_ptr_lo_sts;
	struct qman::reg_arc_cq_ptr_hi_sts arc_cq_ptr_hi_sts;
	uint32_t _pad280[2];
	struct qman::reg_arc_cq_ififo_msg_base_hi arc_cq_ififo_msg_base_hi;
	struct qman::reg_arc_cq_ififo_msg_base_lo arc_cq_ififo_msg_base_lo;
	struct qman::reg_arc_cq_ctl_msg_base_hi arc_cq_ctl_msg_base_hi;
	struct qman::reg_arc_cq_ctl_msg_base_lo arc_cq_ctl_msg_base_lo;
	struct qman::reg_cq_ififo_msg_base_hi cq_ififo_msg_base_hi;
	struct qman::reg_cq_ififo_msg_base_lo cq_ififo_msg_base_lo;
	struct qman::reg_cq_ctl_msg_base_hi cq_ctl_msg_base_hi;
	struct qman::reg_cq_ctl_msg_base_lo cq_ctl_msg_base_lo;
	struct qman::reg_cq_ififo_ci cq_ififo_ci;
	struct qman::reg_arc_cq_ififo_ci arc_cq_ififo_ci;
	struct qman::reg_cq_ctl_ci cq_ctl_ci;
	struct qman::reg_arc_cq_ctl_ci arc_cq_ctl_ci;
	struct qman::reg_cp_cfg cp_cfg;
	struct qman::reg_cp_ext_switch cp_ext_switch;
	struct qman::reg_arc_lb_addr_base_lo arc_lb_addr_base_lo;
	struct qman::reg_arc_lb_addr_base_hi arc_lb_addr_base_hi;
	struct qman::reg_engine_base_addr_hi engine_base_addr_hi;
	struct qman::reg_engine_base_addr_lo engine_base_addr_lo;
	struct qman::reg_engine_addr_range_size engine_addr_range_size;
	struct qman::reg_qm_base_addr_hi qm_base_addr_hi;
	struct qman::reg_qm_base_addr_lo qm_base_addr_lo;
	struct qman::reg_glbl_err_addr_lo glbl_err_addr_lo;
	struct qman::reg_glbl_err_addr_hi glbl_err_addr_hi;
	struct qman::reg_glbl_err_wdata glbl_err_wdata;
	struct qman::reg_l2h_mask_lo l2h_mask_lo;
	struct qman::reg_l2h_mask_hi l2h_mask_hi;
	struct qman::reg_l2h_cmpr_lo l2h_cmpr_lo;
	struct qman::reg_l2h_cmpr_hi l2h_cmpr_hi;
	struct qman::reg_local_range_base local_range_base;
	struct qman::reg_local_range_size local_range_size;
	struct qman::reg_hbw_rd_rate_lim_cfg_1 hbw_rd_rate_lim_cfg_1;
	struct qman::reg_lbw_wr_rate_lim_cfg_0 lbw_wr_rate_lim_cfg_0;
	struct qman::reg_lbw_wr_rate_lim_cfg_1 lbw_wr_rate_lim_cfg_1;
	struct qman::reg_hbw_rd_rate_lim_cfg_0 hbw_rd_rate_lim_cfg_0;
	struct qman::reg_ind_gw_apb_cfg ind_gw_apb_cfg;
	struct qman::reg_ind_gw_apb_wdata ind_gw_apb_wdata;
	struct qman::reg_ind_gw_apb_rdata ind_gw_apb_rdata;
	struct qman::reg_ind_gw_apb_status ind_gw_apb_status;
	struct qman::reg_perf_cnt_free_lo perf_cnt_free_lo;
	struct qman::reg_perf_cnt_free_hi perf_cnt_free_hi;
	struct qman::reg_perf_cnt_idle_lo perf_cnt_idle_lo;
	struct qman::reg_perf_cnt_idle_hi perf_cnt_idle_hi;
	struct qman::reg_perf_cnt_cfg perf_cnt_cfg;
	struct qman::reg_cp_cur_ch_prgm_reg cp_cur_ch_prgm_reg;
	struct qman::reg_arc_ctl arc_ctl;
	struct qman::reg_sei_status sei_status;
	struct qman::reg_sei_mask sei_mask;
	struct qman::reg_arc_axi_ovrd arc_axi_ovrd;
	struct qman::reg_glbl_lbw_axcache glbl_lbw_axcache;
	uint32_t _pad484[455];
	struct block_qman_wr64_base_addr qman_wr64_base_addr0;
	struct block_qman_wr64_base_addr qman_wr64_base_addr1;
	struct block_qman_wr64_base_addr qman_wr64_base_addr2;
	struct block_qman_wr64_base_addr qman_wr64_base_addr3;
	struct block_qman_wr64_base_addr qman_wr64_base_addr4;
	struct block_qman_wr64_base_addr qman_wr64_base_addr5;
	struct block_qman_wr64_base_addr qman_wr64_base_addr6;
	struct block_qman_wr64_base_addr qman_wr64_base_addr7;
	struct block_qman_wr64_base_addr qman_wr64_base_addr8;
	struct block_qman_wr64_base_addr qman_wr64_base_addr9;
	struct block_qman_wr64_base_addr qman_wr64_base_addr10;
	struct block_qman_wr64_base_addr qman_wr64_base_addr11;
	struct block_qman_wr64_base_addr qman_wr64_base_addr12;
	struct block_qman_wr64_base_addr qman_wr64_base_addr13;
	struct block_qman_wr64_base_addr qman_wr64_base_addr14;
	struct block_qman_wr64_base_addr qman_wr64_base_addr15;
	struct block_qman_wr64_base_addr qman_wr64_base_addr16;
	struct block_qman_wr64_base_addr qman_wr64_base_addr17;
	struct block_qman_wr64_base_addr qman_wr64_base_addr18;
	struct block_qman_wr64_base_addr qman_wr64_base_addr19;
	struct block_qman_wr64_base_addr qman_wr64_base_addr20;
	struct block_qman_wr64_base_addr qman_wr64_base_addr21;
	struct block_qman_wr64_base_addr qman_wr64_base_addr22;
	struct block_qman_wr64_base_addr qman_wr64_base_addr23;
	struct block_qman_wr64_base_addr qman_wr64_base_addr24;
	struct block_qman_wr64_base_addr qman_wr64_base_addr25;
	struct block_qman_wr64_base_addr qman_wr64_base_addr26;
	struct block_qman_wr64_base_addr qman_wr64_base_addr27;
	struct block_qman_wr64_base_addr qman_wr64_base_addr28;
	struct block_qman_wr64_base_addr qman_wr64_base_addr29;
	struct block_qman_wr64_base_addr qman_wr64_base_addr30;
	struct block_qman_wr64_base_addr qman_wr64_base_addr31;
	uint32_t _pad2560[96];
	struct block_axuser_hbw axuser_hbw;
	uint32_t _pad3036[2];
	struct block_axuser_lbw axuser_lbw;
	uint32_t _pad3068[1];
	struct block_ic_lbw_dbg_cnt dbg_hbw;
	uint32_t _pad3156[11];
	struct block_ic_lbw_dbg_cnt dbg_lbw;
	uint32_t _pad3284[43];
	struct block_qman_cgm cgm;
	uint32_t _pad3468[61];
	struct block_special_regs special;
};
#else

typedef struct block_qman {
	reg_glbl_cfg0 glbl_cfg0;
	reg_glbl_cfg1 glbl_cfg1;
	reg_glbl_cfg2 glbl_cfg2;
	reg_glbl_err_cfg glbl_err_cfg;
	reg_glbl_err_cfg1 glbl_err_cfg1;
	reg_glbl_err_arc_halt_en glbl_err_arc_halt_en;
	reg_glbl_hbw_axcache glbl_hbw_axcache;
	reg_glbl_sts0 glbl_sts0;
	reg_glbl_sts1 glbl_sts1;
	reg_glbl_err_sts glbl_err_sts;
	reg_glbl_err_msg_en glbl_err_msg_en;
	reg_glbl_prot glbl_prot;
	reg_cq_cfg0 cq_cfg0;
	reg_cq_sts0 cq_sts0;
	reg_cq_cfg1 cq_cfg1;
	reg_cq_sts1 cq_sts1;
	reg_cq_ptr_lo cq_ptr_lo;
	reg_cq_ptr_hi cq_ptr_hi;
	reg_cq_tsize cq_tsize;
	reg_cq_ctl cq_ctl;
	reg_cq_tsize_sts cq_tsize_sts;
	reg_cq_ptr_lo_sts cq_ptr_lo_sts;
	reg_cq_ptr_hi_sts cq_ptr_hi_sts;
	reg_cq_ififo_sts cq_ififo_sts;
	reg_cp_msg_base_addr cp_msg_base_addr[16];
	reg_cp_fence0_rdata cp_fence0_rdata;
	reg_cp_fence1_rdata cp_fence1_rdata;
	reg_cp_fence2_rdata cp_fence2_rdata;
	reg_cp_fence3_rdata cp_fence3_rdata;
	reg_cp_fence0_cnt cp_fence0_cnt;
	reg_cp_fence1_cnt cp_fence1_cnt;
	reg_cp_fence2_cnt cp_fence2_cnt;
	reg_cp_fence3_cnt cp_fence3_cnt;
	reg_cp_barrier_cfg cp_barrier_cfg;
	reg_cp_ldma_base_addr cp_ldma_base_addr;
	reg_cp_sts cp_sts;
	reg_cp_current_inst_lo cp_current_inst_lo;
	reg_cp_current_inst_hi cp_current_inst_hi;
	reg_cp_pred cp_pred;
	reg_cp_pred_upen cp_pred_upen;
	reg_cp_dbg_0 cp_dbg_0;
	reg_cp_in_data_lo cp_in_data_lo;
	reg_cp_in_data_hi cp_in_data_hi;
	reg_arc_cq_cfg0 arc_cq_cfg0;
	reg_arc_cq_cfg1 arc_cq_cfg1;
	reg_arc_cq_ptr_lo arc_cq_ptr_lo;
	reg_arc_cq_ptr_hi arc_cq_ptr_hi;
	reg_arc_cq_tsize arc_cq_tsize;
	reg_arc_cq_sts0 arc_cq_sts0;
	reg_arc_cq_ctl arc_cq_ctl;
	reg_arc_cq_ififo_sts arc_cq_ififo_sts;
	reg_arc_cq_sts1 arc_cq_sts1;
	reg_arc_cq_tsize_sts arc_cq_tsize_sts;
	reg_arc_cq_ptr_lo_sts arc_cq_ptr_lo_sts;
	reg_arc_cq_ptr_hi_sts arc_cq_ptr_hi_sts;
	uint32_t _pad280[2];
	reg_arc_cq_ififo_msg_base_hi arc_cq_ififo_msg_base_hi;
	reg_arc_cq_ififo_msg_base_lo arc_cq_ififo_msg_base_lo;
	reg_arc_cq_ctl_msg_base_hi arc_cq_ctl_msg_base_hi;
	reg_arc_cq_ctl_msg_base_lo arc_cq_ctl_msg_base_lo;
	reg_cq_ififo_msg_base_hi cq_ififo_msg_base_hi;
	reg_cq_ififo_msg_base_lo cq_ififo_msg_base_lo;
	reg_cq_ctl_msg_base_hi cq_ctl_msg_base_hi;
	reg_cq_ctl_msg_base_lo cq_ctl_msg_base_lo;
	reg_cq_ififo_ci cq_ififo_ci;
	reg_arc_cq_ififo_ci arc_cq_ififo_ci;
	reg_cq_ctl_ci cq_ctl_ci;
	reg_arc_cq_ctl_ci arc_cq_ctl_ci;
	reg_cp_cfg cp_cfg;
	reg_cp_ext_switch cp_ext_switch;
	reg_arc_lb_addr_base_lo arc_lb_addr_base_lo;
	reg_arc_lb_addr_base_hi arc_lb_addr_base_hi;
	reg_engine_base_addr_hi engine_base_addr_hi;
	reg_engine_base_addr_lo engine_base_addr_lo;
	reg_engine_addr_range_size engine_addr_range_size;
	reg_qm_base_addr_hi qm_base_addr_hi;
	reg_qm_base_addr_lo qm_base_addr_lo;
	reg_glbl_err_addr_lo glbl_err_addr_lo;
	reg_glbl_err_addr_hi glbl_err_addr_hi;
	reg_glbl_err_wdata glbl_err_wdata;
	reg_l2h_mask_lo l2h_mask_lo;
	reg_l2h_mask_hi l2h_mask_hi;
	reg_l2h_cmpr_lo l2h_cmpr_lo;
	reg_l2h_cmpr_hi l2h_cmpr_hi;
	reg_local_range_base local_range_base;
	reg_local_range_size local_range_size;
	reg_hbw_rd_rate_lim_cfg_1 hbw_rd_rate_lim_cfg_1;
	reg_lbw_wr_rate_lim_cfg_0 lbw_wr_rate_lim_cfg_0;
	reg_lbw_wr_rate_lim_cfg_1 lbw_wr_rate_lim_cfg_1;
	reg_hbw_rd_rate_lim_cfg_0 hbw_rd_rate_lim_cfg_0;
	reg_ind_gw_apb_cfg ind_gw_apb_cfg;
	reg_ind_gw_apb_wdata ind_gw_apb_wdata;
	reg_ind_gw_apb_rdata ind_gw_apb_rdata;
	reg_ind_gw_apb_status ind_gw_apb_status;
	reg_perf_cnt_free_lo perf_cnt_free_lo;
	reg_perf_cnt_free_hi perf_cnt_free_hi;
	reg_perf_cnt_idle_lo perf_cnt_idle_lo;
	reg_perf_cnt_idle_hi perf_cnt_idle_hi;
	reg_perf_cnt_cfg perf_cnt_cfg;
	reg_cp_cur_ch_prgm_reg cp_cur_ch_prgm_reg;
	reg_arc_ctl arc_ctl;
	reg_sei_status sei_status;
	reg_sei_mask sei_mask;
	reg_arc_axi_ovrd arc_axi_ovrd;
	reg_glbl_lbw_axcache glbl_lbw_axcache;
	uint32_t _pad484[455];
	block_qman_wr64_base_addr qman_wr64_base_addr0;
	block_qman_wr64_base_addr qman_wr64_base_addr1;
	block_qman_wr64_base_addr qman_wr64_base_addr2;
	block_qman_wr64_base_addr qman_wr64_base_addr3;
	block_qman_wr64_base_addr qman_wr64_base_addr4;
	block_qman_wr64_base_addr qman_wr64_base_addr5;
	block_qman_wr64_base_addr qman_wr64_base_addr6;
	block_qman_wr64_base_addr qman_wr64_base_addr7;
	block_qman_wr64_base_addr qman_wr64_base_addr8;
	block_qman_wr64_base_addr qman_wr64_base_addr9;
	block_qman_wr64_base_addr qman_wr64_base_addr10;
	block_qman_wr64_base_addr qman_wr64_base_addr11;
	block_qman_wr64_base_addr qman_wr64_base_addr12;
	block_qman_wr64_base_addr qman_wr64_base_addr13;
	block_qman_wr64_base_addr qman_wr64_base_addr14;
	block_qman_wr64_base_addr qman_wr64_base_addr15;
	block_qman_wr64_base_addr qman_wr64_base_addr16;
	block_qman_wr64_base_addr qman_wr64_base_addr17;
	block_qman_wr64_base_addr qman_wr64_base_addr18;
	block_qman_wr64_base_addr qman_wr64_base_addr19;
	block_qman_wr64_base_addr qman_wr64_base_addr20;
	block_qman_wr64_base_addr qman_wr64_base_addr21;
	block_qman_wr64_base_addr qman_wr64_base_addr22;
	block_qman_wr64_base_addr qman_wr64_base_addr23;
	block_qman_wr64_base_addr qman_wr64_base_addr24;
	block_qman_wr64_base_addr qman_wr64_base_addr25;
	block_qman_wr64_base_addr qman_wr64_base_addr26;
	block_qman_wr64_base_addr qman_wr64_base_addr27;
	block_qman_wr64_base_addr qman_wr64_base_addr28;
	block_qman_wr64_base_addr qman_wr64_base_addr29;
	block_qman_wr64_base_addr qman_wr64_base_addr30;
	block_qman_wr64_base_addr qman_wr64_base_addr31;
	uint32_t _pad2560[96];
	block_axuser_hbw axuser_hbw;
	uint32_t _pad3036[2];
	block_axuser_lbw axuser_lbw;
	uint32_t _pad3068[1];
	block_ic_lbw_dbg_cnt dbg_hbw;
	uint32_t _pad3156[11];
	block_ic_lbw_dbg_cnt dbg_lbw;
	uint32_t _pad3284[43];
	block_qman_cgm cgm;
	uint32_t _pad3468[61];
	block_special_regs special;
} block_qman;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_qman_defaults[] =
{
	// offset	// value
	{ 0x8   , 0x8                 , 1 }, // glbl_cfg2
	{ 0x1c  , 0x3                 , 1 }, // glbl_sts0
	{ 0x20  , 0x1                 , 1 }, // glbl_sts1
	{ 0x28  , 0xfdff7e            , 1 }, // glbl_err_msg_en
	{ 0x38  , 0x140014            , 1 }, // cq_cfg1
	{ 0x3c  , 0x1                 , 1 }, // cq_sts1
	{ 0x5c  , 0x10                , 1 }, // cq_ififo_sts
	{ 0xc0  , 0x28                , 1 }, // cp_barrier_cfg
	{ 0xc4  , 0xa080              , 1 }, // cp_ldma_base_addr
	{ 0xd4  , 0x1                 , 1 }, // cp_pred
	{ 0xd8  , 0xfffffffe          , 1 }, // cp_pred_upen
	{ 0xdc  , 0x20                , 1 }, // cp_dbg_0
	{ 0xec  , 0x140014            , 1 }, // arc_cq_cfg1
	{ 0x104 , 0x10                , 1 }, // arc_cq_ififo_sts
	{ 0x108 , 0x1                 , 1 }, // arc_cq_sts1
	{ 0x120 , 0x300007f           , 1 }, // arc_cq_ififo_msg_base_hi
	{ 0x124 , 0xfc508654          , 1 }, // arc_cq_ififo_msg_base_lo
	{ 0x128 , 0x300007f           , 1 }, // arc_cq_ctl_msg_base_hi
	{ 0x12c , 0xfc50865c          , 1 }, // arc_cq_ctl_msg_base_lo
	{ 0x130 , 0x300007f           , 1 }, // cq_ififo_msg_base_hi
	{ 0x134 , 0xfc508650          , 1 }, // cq_ififo_msg_base_lo
	{ 0x138 , 0x300007f           , 1 }, // cq_ctl_msg_base_hi
	{ 0x13c , 0xfc508658          , 1 }, // cq_ctl_msg_base_lo
	{ 0x158 , 0x7                 , 1 }, // arc_lb_addr_base_lo
	{ 0x15c , 0x300007f           , 1 }, // arc_lb_addr_base_hi
	{ 0x160 , 0x300007f           , 1 }, // engine_base_addr_hi
	{ 0x164 , 0xfc500000          , 1 }, // engine_base_addr_lo
	{ 0x168 , 0x40000             , 1 }, // engine_addr_range_size
	{ 0x16c , 0x300007f           , 1 }, // qm_base_addr_hi
	{ 0x170 , 0xfc509000          , 1 }, // qm_base_addr_lo
	{ 0x180 , 0xf8000000          , 1 }, // l2h_mask_lo
	{ 0x184 , 0xffffffff          , 1 }, // l2h_mask_hi
	{ 0x188 , 0xf8000000          , 1 }, // l2h_cmpr_lo
	{ 0x18c , 0x300007f           , 1 }, // l2h_cmpr_hi
	{ 0x190 , 0x9000              , 1 }, // local_range_base
	{ 0x194 , 0x1000              , 1 }, // local_range_size
	{ 0x1c8 , 0x3                 , 1 }, // perf_cnt_cfg
	{ 0xc40 , 0x100               , 1 }, // otf_over_th_wr_tot_req_th
	{ 0xc44 , 0x100               , 1 }, // otf_over_th_rd_tot_req_th
	{ 0xc48 , 0x100               , 1 }, // otf_over_th_wr_tot_cyc_th
	{ 0xc4c , 0x100               , 1 }, // otf_over_th_rd_tot_cyc_th
	{ 0xcc0 , 0x100               , 1 }, // otf_over_th_wr_tot_req_th
	{ 0xcc4 , 0x100               , 1 }, // otf_over_th_rd_tot_req_th
	{ 0xcc8 , 0x100               , 1 }, // otf_over_th_wr_tot_cyc_th
	{ 0xccc , 0x100               , 1 }, // otf_over_th_rd_tot_cyc_th
	{ 0xd80 , 0x100080            , 1 }, // cfg
	{ 0xd84 , 0xf00               , 1 }, // sts
	{ 0xd88 , 0x10                , 1 }, // cfg1
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_QMAN_H_ */

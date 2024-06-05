/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_PQM_CMN_B_H_
#define ASIC_REG_STRUCTS_GAUDI3_PQM_CMN_B_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace pqm_cmn_b {
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
 WR64_BASE_ADDR 
 b'wrreg64 base address even for lsb, odd for msb'
*/
typedef struct reg_wr64_base_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr64_base_addr;
static_assert((sizeof(struct reg_wr64_base_addr) == 4), "reg_wr64_base_addr size is not 32-bit");
/*
 CP_MSG_BASE_ADDR 
 b'cp msg base address even for lsb, odd for msb'
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
 PDMA_BASE_ADDR 
 b'dma base address for wreg32 and wreg bulk'
*/
typedef struct reg_pdma_base_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pdma_base_addr;
static_assert((sizeof(struct reg_pdma_base_addr) == 4), "reg_pdma_base_addr size is not 32-bit");
/*
 CH_A_BASE_ADDR 
 b'used in linpdma for pdma channel base addr'
*/
typedef struct reg_ch_a_base_addr {
	union {
		struct {
			uint32_t val : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_ch_a_base_addr;
static_assert((sizeof(struct reg_ch_a_base_addr) == 4), "reg_ch_a_base_addr size is not 32-bit");
/*
 LPDMA_BASE_ADDR 
 b'pdma ch a ctx base addr (descriptor)'
*/
typedef struct reg_lpdma_base_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lpdma_base_addr;
static_assert((sizeof(struct reg_lpdma_base_addr) == 4), "reg_lpdma_base_addr size is not 32-bit");
/*
 CTRL_MAIN_ADDR 
 b'ctx ctrl_main cfg addr'
*/
typedef struct reg_ctrl_main_addr {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_ctrl_main_addr;
static_assert((sizeof(struct reg_ctrl_main_addr) == 4), "reg_ctrl_main_addr size is not 32-bit");
/*
 COMMIT_ADDR 
 b'ctx commit cfg addr'
*/
typedef struct reg_commit_addr {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_commit_addr;
static_assert((sizeof(struct reg_commit_addr) == 4), "reg_commit_addr size is not 32-bit");
/*
 CP_BARRIER_CFG 
 b'PQMCP: Guard band (Allow engine to de-assert idle)'
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
 CGM_CFG 
 b'Clock gate Manager config'
*/
typedef struct reg_cgm_cfg {
	union {
		struct {
			uint32_t idle_th : 12,
				_reserved16 : 4,
				g2f_th : 8,
				_reserved31 : 7,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_cgm_cfg;
static_assert((sizeof(struct reg_cgm_cfg) == 4), "reg_cgm_cfg size is not 32-bit");
/*
 CGM_STS 
 b'Clock Manager Status'
*/
typedef struct reg_cgm_sts {
	union {
		struct {
			uint32_t st : 2,
				_reserved4 : 2,
				cg : 1,
				_reserved8 : 3,
				agent_idle : 1,
				axi_idle : 1,
				cp_idle : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_cgm_sts;
static_assert((sizeof(struct reg_cgm_sts) == 4), "reg_cgm_sts size is not 32-bit");
/*
 CGM_CFG1 
 b'CGM config reg 1 HBW mask Thresh'
*/
typedef struct reg_cgm_cfg1 {
	union {
		struct {
			uint32_t mask_th : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cgm_cfg1;
static_assert((sizeof(struct reg_cgm_cfg1) == 4), "reg_cgm_cfg1 size is not 32-bit");
/*
 GLBL_STS 
 b'PQM global status'
*/
typedef struct reg_glbl_sts {
	union {
		struct {
			uint32_t sts_cp_idle : 1,
				sts_cq_idle : 1,
				engine_busy : 1,
				sts_cp_stop : 1,
				sts_cq_stop : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_glbl_sts;
static_assert((sizeof(struct reg_glbl_sts) == 4), "reg_glbl_sts size is not 32-bit");
/*
 GLBL_CP_STS 
 b'PQM global CP status'
*/
typedef struct reg_glbl_cp_sts {
	union {
		struct {
			uint32_t cq_cp_channel : 3,
				reserved3 : 1,
				cs : 2,
				reserved6 : 2,
				lpdma_cs : 4,
				eb_cs : 2,
				reserved14 : 2,
				eb_cnt_eq0 : 1,
				eb_stall_pre : 1,
				csmr_busy : 1,
				flush : 1,
				predi_skip : 1,
				all_ch_no_msg_in_pipe : 1,
				stop_flag : 1,
				stall : 1,
				ch_cs_idle : 6,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_glbl_cp_sts;
static_assert((sizeof(struct reg_glbl_cp_sts) == 4), "reg_glbl_cp_sts size is not 32-bit");
/*
 GLBL_CQ_STS 
 b'PQM global CQ status'
*/
typedef struct reg_glbl_cq_sts {
	union {
		struct {
			uint32_t arb_cur_ch : 3,
				reserved3 : 1,
				free_list_used : 6,
				reserved10 : 2,
				cq_cp_channel : 3,
				reserved15 : 1,
				cl_ack_sel : 5,
				reserved21 : 3,
				cl_rd_offset : 4,
				cq_tbl_idle : 1,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_sts;
static_assert((sizeof(struct reg_glbl_cq_sts) == 4), "reg_glbl_cq_sts size is not 32-bit");
/*
 GLBL_CQ_CQE_HI 
 b'current CQE at the cq-cp interface msb'
*/
typedef struct reg_glbl_cq_cqe_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cqe_hi;
static_assert((sizeof(struct reg_glbl_cq_cqe_hi) == 4), "reg_glbl_cq_cqe_hi size is not 32-bit");
/*
 GLBL_CQ_CQE_LO 
 b'current CQE at the cq-cp interface lsb'
*/
typedef struct reg_glbl_cq_cqe_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cqe_lo;
static_assert((sizeof(struct reg_glbl_cq_cqe_lo) == 4), "reg_glbl_cq_cqe_lo size is not 32-bit");
/*
 GLBL_CQ_CL_IDLE_0 
 b'CQ table entry 0..31 - 0-not assigned'
*/
typedef struct reg_glbl_cq_cl_idle_0 {
	union {
		struct {
			uint32_t sts : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cl_idle_0;
static_assert((sizeof(struct reg_glbl_cq_cl_idle_0) == 4), "reg_glbl_cq_cl_idle_0 size is not 32-bit");
/*
 GLBL_CQ_CL_IDLE_1 
 b'CQ table entry 32..41 - 0-not assigned'
*/
typedef struct reg_glbl_cq_cl_idle_1 {
	union {
		struct {
			uint32_t sts : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cl_idle_1;
static_assert((sizeof(struct reg_glbl_cq_cl_idle_1) == 4), "reg_glbl_cq_cl_idle_1 size is not 32-bit");
/*
 GLBL_CQ_CL_VLD_0 
 b'CQ table entry 32..41 - 1-valid data in CL'
*/
typedef struct reg_glbl_cq_cl_vld_0 {
	union {
		struct {
			uint32_t sts : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cl_vld_0;
static_assert((sizeof(struct reg_glbl_cq_cl_vld_0) == 4), "reg_glbl_cq_cl_vld_0 size is not 32-bit");
/*
 GLBL_CQ_CL_VLD_1 
 b'CQ table entry 0..31 - 1-valid data in CL'
*/
typedef struct reg_glbl_cq_cl_vld_1 {
	union {
		struct {
			uint32_t sts : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cl_vld_1;
static_assert((sizeof(struct reg_glbl_cq_cl_vld_1) == 4), "reg_glbl_cq_cl_vld_1 size is not 32-bit");
/*
 GLBL_CQ_CL_CH 
 b'the 3 GLBL_CQ_CL_CH contain the 32 cl ch ids'
*/
typedef struct reg_glbl_cq_cl_ch {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_glbl_cq_cl_ch;
static_assert((sizeof(struct reg_glbl_cq_cl_ch) == 4), "reg_glbl_cq_cl_ch size is not 32-bit");
/*
 CFG 
 b'cfg all ch (stop all and flush all)'
*/
typedef struct reg_cfg {
	union {
		struct {
			uint32_t force_stop_all : 1,
				enable_stop_all : 1,
				flush_all : 1,
				en_flush_cp_rst : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cfg;
static_assert((sizeof(struct reg_cfg) == 4), "reg_cfg size is not 32-bit");
/*
 DBG_CHANNEL_FILTER 
 b"filters dbg for a specific ch, '0' enables all"
*/
typedef struct reg_dbg_channel_filter {
	union {
		struct {
			uint32_t trace_ch_sel : 6,
				spmu_ch_sel : 6,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_dbg_channel_filter;
static_assert((sizeof(struct reg_dbg_channel_filter) == 4), "reg_dbg_channel_filter size is not 32-bit");
/*
 SEI_STATUS 
 b'pqm sei intr per ch'
*/
typedef struct reg_sei_status {
	union {
		struct {
			uint32_t ch_int : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_sei_status;
static_assert((sizeof(struct reg_sei_status) == 4), "reg_sei_status size is not 32-bit");
/*
 SEI_MASK 
 b'pqm sei intr mask per ch'
*/
typedef struct reg_sei_mask {
	union {
		struct {
			uint32_t ch_int : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_sei_mask;
static_assert((sizeof(struct reg_sei_mask) == 4), "reg_sei_mask size is not 32-bit");

#ifdef __cplusplus
} /* pqm_cmn_b namespace */
#endif

/*
 PQM_CMN_B block
*/

#ifdef __cplusplus

struct block_pqm_cmn_b {
	struct pqm_cmn_b::reg_wr64_base_addr wr64_base_addr[64];
	struct pqm_cmn_b::reg_cp_msg_base_addr cp_msg_base_addr[16];
	struct pqm_cmn_b::reg_pdma_base_addr pdma_base_addr;
	struct pqm_cmn_b::reg_ch_a_base_addr ch_a_base_addr;
	struct pqm_cmn_b::reg_lpdma_base_addr lpdma_base_addr;
	struct pqm_cmn_b::reg_ctrl_main_addr ctrl_main_addr;
	struct pqm_cmn_b::reg_commit_addr commit_addr;
	struct pqm_cmn_b::reg_cp_barrier_cfg cp_barrier_cfg;
	struct pqm_cmn_b::reg_cgm_cfg cgm_cfg;
	struct pqm_cmn_b::reg_cgm_sts cgm_sts;
	struct pqm_cmn_b::reg_cgm_cfg1 cgm_cfg1;
	struct pqm_cmn_b::reg_glbl_sts glbl_sts;
	struct pqm_cmn_b::reg_glbl_cp_sts glbl_cp_sts;
	struct pqm_cmn_b::reg_glbl_cq_sts glbl_cq_sts;
	struct pqm_cmn_b::reg_glbl_cq_cqe_hi glbl_cq_cqe_hi;
	struct pqm_cmn_b::reg_glbl_cq_cqe_lo glbl_cq_cqe_lo;
	struct pqm_cmn_b::reg_glbl_cq_cl_idle_0 glbl_cq_cl_idle_0;
	struct pqm_cmn_b::reg_glbl_cq_cl_idle_1 glbl_cq_cl_idle_1;
	struct pqm_cmn_b::reg_glbl_cq_cl_vld_0 glbl_cq_cl_vld_0;
	struct pqm_cmn_b::reg_glbl_cq_cl_vld_1 glbl_cq_cl_vld_1;
	struct pqm_cmn_b::reg_glbl_cq_cl_ch glbl_cq_cl_ch[4];
	struct pqm_cmn_b::reg_cfg cfg;
	struct pqm_cmn_b::reg_dbg_channel_filter dbg_channel_filter;
	struct pqm_cmn_b::reg_sei_status sei_status;
	struct pqm_cmn_b::reg_sei_mask sei_mask;
};
#else

typedef struct block_pqm_cmn_b {
	reg_wr64_base_addr wr64_base_addr[64];
	reg_cp_msg_base_addr cp_msg_base_addr[16];
	reg_pdma_base_addr pdma_base_addr;
	reg_ch_a_base_addr ch_a_base_addr;
	reg_lpdma_base_addr lpdma_base_addr;
	reg_ctrl_main_addr ctrl_main_addr;
	reg_commit_addr commit_addr;
	reg_cp_barrier_cfg cp_barrier_cfg;
	reg_cgm_cfg cgm_cfg;
	reg_cgm_sts cgm_sts;
	reg_cgm_cfg1 cgm_cfg1;
	reg_glbl_sts glbl_sts;
	reg_glbl_cp_sts glbl_cp_sts;
	reg_glbl_cq_sts glbl_cq_sts;
	reg_glbl_cq_cqe_hi glbl_cq_cqe_hi;
	reg_glbl_cq_cqe_lo glbl_cq_cqe_lo;
	reg_glbl_cq_cl_idle_0 glbl_cq_cl_idle_0;
	reg_glbl_cq_cl_idle_1 glbl_cq_cl_idle_1;
	reg_glbl_cq_cl_vld_0 glbl_cq_cl_vld_0;
	reg_glbl_cq_cl_vld_1 glbl_cq_cl_vld_1;
	reg_glbl_cq_cl_ch glbl_cq_cl_ch[4];
	reg_cfg cfg;
	reg_dbg_channel_filter dbg_channel_filter;
	reg_sei_status sei_status;
	reg_sei_mask sei_mask;
} block_pqm_cmn_b;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_pqm_cmn_b_defaults[] =
{
	// offset	// value
	{ 0x140 , 0x40000             , 1 }, // pdma_base_addr
	{ 0x144 , 0x40                , 1 }, // ch_a_base_addr
	{ 0x148 , 0x300               , 1 }, // lpdma_base_addr
	{ 0x14c , 0x33c               , 1 }, // ctrl_main_addr
	{ 0x150 , 0x3a4               , 1 }, // commit_addr
	{ 0x154 , 0x28                , 1 }, // cp_barrier_cfg
	{ 0x158 , 0x100080            , 1 }, // cgm_cfg
	{ 0x15c , 0x700               , 1 }, // cgm_sts
	{ 0x160 , 0x10                , 1 }, // cgm_cfg1
	{ 0x178 , 0xffffffff          , 1 }, // glbl_cq_cl_idle_0
	{ 0x17c , 0xffffffff          , 1 }, // glbl_cq_cl_idle_1
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_PQM_CMN_B_H_ */

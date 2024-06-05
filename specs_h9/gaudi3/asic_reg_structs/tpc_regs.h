/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "axuser_hbw_regs.h"
#include "axuser_lbw_regs.h"
#include "special_regs_regs.h"
#include "sync_object_regs.h"
#include "tpc_dcache_regs.h"
#include "tpc_non_tensor_descriptor_regs.h"
#include "tpc_non_tensor_descriptor_qm_regs.h"
#include "tpc_non_tensor_descriptor_smt_regs.h"
#include "tpc_tensor_base_regs.h"
#include "tpc_tensor_shared_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc {
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
 CCL_RANGE_DIS 
 b'disable range check in ccl'
*/
typedef struct reg_ccl_range_dis {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_ccl_range_dis;
static_assert((sizeof(struct reg_ccl_range_dis) == 4), "reg_ccl_range_dis size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_0 
 b'row 0 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_0 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_0;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_0) == 4), "reg_ccl_scram_poly_matrix_0 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_1 
 b'row 1 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_1 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_1;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_1) == 4), "reg_ccl_scram_poly_matrix_1 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_2 
 b'row 2 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_2 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_2;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_2) == 4), "reg_ccl_scram_poly_matrix_2 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_3 
 b'row 3 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_3 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_3;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_3) == 4), "reg_ccl_scram_poly_matrix_3 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_4 
 b'row 4 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_4 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_4;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_4) == 4), "reg_ccl_scram_poly_matrix_4 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_5 
 b'row 5 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_5 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_5;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_5) == 4), "reg_ccl_scram_poly_matrix_5 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_6 
 b'row 6 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_6 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_6;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_6) == 4), "reg_ccl_scram_poly_matrix_6 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_7 
 b'row 7 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_7 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_7;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_7) == 4), "reg_ccl_scram_poly_matrix_7 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_8 
 b'row 8 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_8 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_8;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_8) == 4), "reg_ccl_scram_poly_matrix_8 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_9 
 b'row 9 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_9 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_9;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_9) == 4), "reg_ccl_scram_poly_matrix_9 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_10 
 b'row 10 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_10 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_10;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_10) == 4), "reg_ccl_scram_poly_matrix_10 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_11 
 b'row 11 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_11 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_11;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_11) == 4), "reg_ccl_scram_poly_matrix_11 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_12 
 b'row 12 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_12 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_12;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_12) == 4), "reg_ccl_scram_poly_matrix_12 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_13 
 b'row 13 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_13 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_13;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_13) == 4), "reg_ccl_scram_poly_matrix_13 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_14 
 b'row 14 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_14 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_14;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_14) == 4), "reg_ccl_scram_poly_matrix_14 size is not 32-bit");
/*
 CCL_SCRAM_POLY_MATRIX_15 
 b'row 15 of ccl hash matrix'
*/
typedef struct reg_ccl_scram_poly_matrix_15 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_poly_matrix_15;
static_assert((sizeof(struct reg_ccl_scram_poly_matrix_15) == 4), "reg_ccl_scram_poly_matrix_15 size is not 32-bit");
/*
 CCL_SCRAM_CFG 
 b'ccl scambling configurations'
*/
typedef struct reg_ccl_scram_cfg {
	union {
		struct {
			uint32_t perm_sel : 4,
				scram_en : 1,
				non_lin_func_en : 1,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_ccl_scram_cfg;
static_assert((sizeof(struct reg_ccl_scram_cfg) == 4), "reg_ccl_scram_cfg size is not 32-bit");
/*
 VPU_ONGOING_KERNEL_ID 
 b'the kernel id of the kernel in the vpu'
*/
typedef struct reg_vpu_ongoing_kernel_id {
	union {
		struct {
			uint32_t id : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_vpu_ongoing_kernel_id;
static_assert((sizeof(struct reg_vpu_ongoing_kernel_id) == 4), "reg_vpu_ongoing_kernel_id size is not 32-bit");
/*
 DCACH_MEM_ERR_KERNEL_ID 
 b'dcache memory error kernel id'
*/
typedef struct reg_dcach_mem_err_kernel_id {
	union {
		struct {
			uint32_t id : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_dcach_mem_err_kernel_id;
static_assert((sizeof(struct reg_dcach_mem_err_kernel_id) == 4), "reg_dcach_mem_err_kernel_id size is not 32-bit");
/*
 WQ_MEM_ERR_KERNEL_ID 
 b'wq memory error kernel id'
*/
typedef struct reg_wq_mem_err_kernel_id {
	union {
		struct {
			uint32_t id : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_wq_mem_err_kernel_id;
static_assert((sizeof(struct reg_wq_mem_err_kernel_id) == 4), "reg_wq_mem_err_kernel_id size is not 32-bit");
/*
 TSB_MEM_ERR_KERNEL_ID 
 b'tsb memory error kernel id'
*/
typedef struct reg_tsb_mem_err_kernel_id {
	union {
		struct {
			uint32_t id : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_tsb_mem_err_kernel_id;
static_assert((sizeof(struct reg_tsb_mem_err_kernel_id) == 4), "reg_tsb_mem_err_kernel_id size is not 32-bit");
/*
 SPMU_THRD_MSK 
 b'spmu thread mask'
*/
typedef struct reg_spmu_thrd_msk {
	union {
		struct {
			uint32_t mask : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_spmu_thrd_msk;
static_assert((sizeof(struct reg_spmu_thrd_msk) == 4), "reg_spmu_thrd_msk size is not 32-bit");
/*
 STALL_ON_ERR_MASK_0 
 b'set1 to mask stall on error on corresponding int'
*/
typedef struct reg_stall_on_err_mask_0 {
	union {
		struct {
			uint32_t mask : 32;
		};
		uint32_t _raw;
	};
} reg_stall_on_err_mask_0;
static_assert((sizeof(struct reg_stall_on_err_mask_0) == 4), "reg_stall_on_err_mask_0 size is not 32-bit");
/*
 STALL_ON_ERR_MASK_1 
 b'set1 to mask stall on error on corresponding int'
*/
typedef struct reg_stall_on_err_mask_1 {
	union {
		struct {
			uint32_t mask : 32;
		};
		uint32_t _raw;
	};
} reg_stall_on_err_mask_1;
static_assert((sizeof(struct reg_stall_on_err_mask_1) == 4), "reg_stall_on_err_mask_1 size is not 32-bit");
/*
 TPC_INTR_CAUSE_1 
 b'TPC interrupts cause'
*/
typedef struct reg_tpc_intr_cause_1 {
	union {
		struct {
			uint32_t cause : 32;
		};
		uint32_t _raw;
	};
} reg_tpc_intr_cause_1;
static_assert((sizeof(struct reg_tpc_intr_cause_1) == 4), "reg_tpc_intr_cause_1 size is not 32-bit");
/*
 TPC_INTR_MASK_1 
 b'Set 1 to mask the corresponding interrupt'
*/
typedef struct reg_tpc_intr_mask_1 {
	union {
		struct {
			uint32_t mask : 32;
		};
		uint32_t _raw;
	};
} reg_tpc_intr_mask_1;
static_assert((sizeof(struct reg_tpc_intr_mask_1) == 4), "reg_tpc_intr_mask_1 size is not 32-bit");
/*
 TPC_COUNT 
 b"number of TPC's"
*/
typedef struct reg_tpc_count {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tpc_count;
static_assert((sizeof(struct reg_tpc_count) == 4), "reg_tpc_count size is not 32-bit");
/*
 STALL_ON_ERR 
 b'Stall on error configurations'
*/
typedef struct reg_stall_on_err {
	union {
		struct {
			uint32_t stall_enable : 1,
				stall_on_derr_mask : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_stall_on_err;
static_assert((sizeof(struct reg_stall_on_err) == 4), "reg_stall_on_err size is not 32-bit");
/*
 CLK_EN 
 b'Clock enabler configurations'
*/
typedef struct reg_clk_en {
	union {
		struct {
			uint32_t lbw_cfg_dis : 1,
				_reserved4 : 3,
				dbg_cfg_dis : 1,
				_reserved8 : 3,
				dbg_clk_off : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_clk_en;
static_assert((sizeof(struct reg_clk_en) == 4), "reg_clk_en size is not 32-bit");
/*
 IQ_RL_EN 
 b'Tpc IQ rate limiter enable'
*/
typedef struct reg_iq_rl_en {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_iq_rl_en;
static_assert((sizeof(struct reg_iq_rl_en) == 4), "reg_iq_rl_en size is not 32-bit");
/*
 IQ_RL_SAT 
 b'Tpc IQ rate limiter saturation value'
*/
typedef struct reg_iq_rl_sat {
	union {
		struct {
			uint32_t v : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_iq_rl_sat;
static_assert((sizeof(struct reg_iq_rl_sat) == 4), "reg_iq_rl_sat size is not 32-bit");
/*
 IQ_RL_RST_TOKEN 
 b'Tpc IQ rate limiter reset token'
*/
typedef struct reg_iq_rl_rst_token {
	union {
		struct {
			uint32_t v : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_iq_rl_rst_token;
static_assert((sizeof(struct reg_iq_rl_rst_token) == 4), "reg_iq_rl_rst_token size is not 32-bit");
/*
 IQ_RL_TIMEOUT 
 b'Tpc IQ rate limiter timeout'
*/
typedef struct reg_iq_rl_timeout {
	union {
		struct {
			uint32_t v : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_iq_rl_timeout;
static_assert((sizeof(struct reg_iq_rl_timeout) == 4), "reg_iq_rl_timeout size is not 32-bit");
/*
 TSB_CFG_MTRR_2 
 b'TSB MTRR cfg2'
*/
typedef struct reg_tsb_cfg_mtrr_2 {
	union {
		struct {
			uint32_t phy_base_add_lo : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg_mtrr_2;
static_assert((sizeof(struct reg_tsb_cfg_mtrr_2) == 4), "reg_tsb_cfg_mtrr_2 size is not 32-bit");
/*
 IQ_LBW_CLK_EN 
 b'use lbw_clk bypass for iq'
*/
typedef struct reg_iq_lbw_clk_en {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_iq_lbw_clk_en;
static_assert((sizeof(struct reg_iq_lbw_clk_en) == 4), "reg_iq_lbw_clk_en size is not 32-bit");
/*
 TPC_LOCK_VALUE 
 b'TPC lock value'
*/
typedef struct reg_tpc_lock_value {
	union {
		struct {
			uint32_t value : 32;
		};
		uint32_t _raw;
	};
} reg_tpc_lock_value;
static_assert((sizeof(struct reg_tpc_lock_value) == 4), "reg_tpc_lock_value size is not 32-bit");
/*
 TPC_LOCK 
 b'TPC lock'
*/
typedef struct reg_tpc_lock {
	union {
		struct {
			uint32_t lock : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_tpc_lock;
static_assert((sizeof(struct reg_tpc_lock) == 4), "reg_tpc_lock size is not 32-bit");
/*
 CGU_SB 
 b'CGU SB Configuration Disable'
*/
typedef struct reg_cgu_sb {
	union {
		struct {
			uint32_t tsb_disable : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cgu_sb;
static_assert((sizeof(struct reg_cgu_sb) == 4), "reg_cgu_sb size is not 32-bit");
/*
 CGU_CNT 
 b'CGU CNT Configuration Disable'
*/
typedef struct reg_cgu_cnt {
	union {
		struct {
			uint32_t dcache_disable : 1,
				wq_disable : 1,
				spu_agu_addsub_0_disable : 1,
				spu_agu_addsub_1_disable : 1,
				spu_agu_addsub_2_disable : 1,
				spu_agu_addsub_3_disable : 1,
				spu_agu_addsub_4_disable : 1,
				spu_agu_cmp_0_disable : 1,
				spu_agu_cmp_1_disable : 1,
				spu_agu_cmp_2_disable : 1,
				spu_agu_cmp_3_disable : 1,
				spu_agu_cmp_4_disable : 1,
				msac_disable : 1,
				conv_disable : 1,
				nearbyint_disable : 1,
				cmp_disable : 1,
				fp_mac_disable : 1,
				sops_src_a_d2_disable : 1,
				sops_src_b_d2_disable : 1,
				sops_src_e_d2_disable : 1,
				sops_fma_src_c_e1_disable : 1,
				ld_sops_src_a_d2_disable : 1,
				st_sops_src_a_d2_disable : 1,
				fp_addsub_disable : 1,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cgu_cnt;
static_assert((sizeof(struct reg_cgu_cnt) == 4), "reg_cgu_cnt size is not 32-bit");
/*
 CGU_CPE 
 b'CGU CPE  Configuration Disable'
*/
typedef struct reg_cgu_cpe {
	union {
		struct {
			uint32_t nearbyint_disable : 1,
				sops_src_a_disable : 1,
				sops_src_b_disable : 1,
				sops_src_e_disable : 1,
				sops_src_d_disable : 1,
				sops_src_c_disable : 1,
				ld_sops_src_a_disable : 1,
				msac_disable : 1,
				addsub_disable : 1,
				shift_disable : 1,
				gle_disable : 1,
				cmp_disable : 1,
				conv_disable : 1,
				sb_disable : 1,
				tbuf_disable : 1,
				st_g_disable : 1,
				fp_mac_0_disable : 1,
				fp_mac_1_disable : 1,
				fp_addsub_disable : 1,
				st_sops_src_c_disable : 1,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_cgu_cpe;
static_assert((sizeof(struct reg_cgu_cpe) == 4), "reg_cgu_cpe size is not 32-bit");
/*
 TPC_SB_L0CD 
 b'L0 SB Cache Disable'
*/
typedef struct reg_tpc_sb_l0cd {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_tpc_sb_l0cd;
static_assert((sizeof(struct reg_tpc_sb_l0cd) == 4), "reg_tpc_sb_l0cd size is not 32-bit");
/*
 TSB_OCCUPANCY 
 b'RSB total occupancy status'
*/
typedef struct reg_tsb_occupancy {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tsb_occupancy;
static_assert((sizeof(struct reg_tsb_occupancy) == 4), "reg_tsb_occupancy size is not 32-bit");
/*
 TSB_DATA_OCCUPANCY 
 b'RSB data memory occupancy status'
*/
typedef struct reg_tsb_data_occupancy {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tsb_data_occupancy;
static_assert((sizeof(struct reg_tsb_data_occupancy) == 4), "reg_tsb_data_occupancy size is not 32-bit");
/*
 TSB_MD_OCCUPANCY 
 b'RSB meta data memory occupancy status'
*/
typedef struct reg_tsb_md_occupancy {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tsb_md_occupancy;
static_assert((sizeof(struct reg_tsb_md_occupancy) == 4), "reg_tsb_md_occupancy size is not 32-bit");
/*
 SMT_CAUSE0_ALL_THRD 
 b'cause0 smt OR between all threads'
*/
typedef struct reg_smt_cause0_all_thrd {
	union {
		struct {
			uint32_t cause : 32;
		};
		uint32_t _raw;
	};
} reg_smt_cause0_all_thrd;
static_assert((sizeof(struct reg_smt_cause0_all_thrd) == 4), "reg_smt_cause0_all_thrd size is not 32-bit");
/*
 SMT_CAUSE1_ALL_THRD 
 b'cause1 smt OR between all threads'
*/
typedef struct reg_smt_cause1_all_thrd {
	union {
		struct {
			uint32_t cause : 32;
		};
		uint32_t _raw;
	};
} reg_smt_cause1_all_thrd;
static_assert((sizeof(struct reg_smt_cause1_all_thrd) == 4), "reg_smt_cause1_all_thrd size is not 32-bit");
/*
 SQZ_OOB_PROTECTION 
 b'enable/disable sqeeze out of bound protection'
*/
typedef struct reg_sqz_oob_protection {
	union {
		struct {
			uint32_t enable : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sqz_oob_protection;
static_assert((sizeof(struct reg_sqz_oob_protection) == 4), "reg_sqz_oob_protection size is not 32-bit");
/*
 ARB_QNT_HBW_WEIGHT 
 b'QNT hbw arbitration weight'
*/
typedef struct reg_arb_qnt_hbw_weight {
	union {
		struct {
			uint32_t ar : 12,
				aw : 12,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_arb_qnt_hbw_weight;
static_assert((sizeof(struct reg_arb_qnt_hbw_weight) == 4), "reg_arb_qnt_hbw_weight size is not 32-bit");
/*
 ARB_QNT_LBW_WEIGHT 
 b'QNT lbw arbitration weight'
*/
typedef struct reg_arb_qnt_lbw_weight {
	union {
		struct {
			uint32_t aw : 12,
				ar : 8,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_arb_qnt_lbw_weight;
static_assert((sizeof(struct reg_arb_qnt_lbw_weight) == 4), "reg_arb_qnt_lbw_weight size is not 32-bit");
/*
 ARB_CNT_HBW_WEIGHT 
 b'CNT hbw arbitration weight'
*/
typedef struct reg_arb_cnt_hbw_weight {
	union {
		struct {
			uint32_t ar : 12,
				aw : 12,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_arb_cnt_hbw_weight;
static_assert((sizeof(struct reg_arb_cnt_hbw_weight) == 4), "reg_arb_cnt_hbw_weight size is not 32-bit");
/*
 ARB_CNT_LBW_WEIGHT 
 b'CNT lbw arbitration weight'
*/
typedef struct reg_arb_cnt_lbw_weight {
	union {
		struct {
			uint32_t ar : 8,
				aw : 12,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_arb_cnt_lbw_weight;
static_assert((sizeof(struct reg_arb_cnt_lbw_weight) == 4), "reg_arb_cnt_lbw_weight size is not 32-bit");
/*
 LUT_FUNC32_BASE2_ADDR_LO 
 b'LOOKUP TABLE 32 lines Base2 Address 32 LSB'
*/
typedef struct reg_lut_func32_base2_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func32_base2_addr_lo;
static_assert((sizeof(struct reg_lut_func32_base2_addr_lo) == 4), "reg_lut_func32_base2_addr_lo size is not 32-bit");
/*
 LUT_FUNC32_BASE2_ADDR_HI 
 b'LOOKUP TABLE 32 lines Base2 Address 32 MSB'
*/
typedef struct reg_lut_func32_base2_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func32_base2_addr_hi;
static_assert((sizeof(struct reg_lut_func32_base2_addr_hi) == 4), "reg_lut_func32_base2_addr_hi size is not 32-bit");
/*
 LUT_FUNC64_BASE2_ADDR_LO 
 b'LOOKUP TABLE 64 lines Base2 Address 32 LSB'
*/
typedef struct reg_lut_func64_base2_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func64_base2_addr_lo;
static_assert((sizeof(struct reg_lut_func64_base2_addr_lo) == 4), "reg_lut_func64_base2_addr_lo size is not 32-bit");
/*
 LUT_FUNC64_BASE2_ADDR_HI 
 b'LOOKUP TABLE 64 lines Base2 Address 32 MSB'
*/
typedef struct reg_lut_func64_base2_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func64_base2_addr_hi;
static_assert((sizeof(struct reg_lut_func64_base2_addr_hi) == 4), "reg_lut_func64_base2_addr_hi size is not 32-bit");
/*
 LUT_FUNC128_BASE2_ADDR_LO 
 b'LOOKUP TABLE 128 lines Base2 Address 32 LSB'
*/
typedef struct reg_lut_func128_base2_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func128_base2_addr_lo;
static_assert((sizeof(struct reg_lut_func128_base2_addr_lo) == 4), "reg_lut_func128_base2_addr_lo size is not 32-bit");
/*
 LUT_FUNC128_BASE2_ADDR_HI 
 b'LOOKUP TABLE 128 lines Base2 Address 32 MSB'
*/
typedef struct reg_lut_func128_base2_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func128_base2_addr_hi;
static_assert((sizeof(struct reg_lut_func128_base2_addr_hi) == 4), "reg_lut_func128_base2_addr_hi size is not 32-bit");
/*
 LUT_FUNC256_BASE2_ADDR_LO 
 b'LOOKUP TABLE 256 lines Base2 Address 32 LSB'
*/
typedef struct reg_lut_func256_base2_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func256_base2_addr_lo;
static_assert((sizeof(struct reg_lut_func256_base2_addr_lo) == 4), "reg_lut_func256_base2_addr_lo size is not 32-bit");
/*
 LUT_FUNC256_BASE2_ADDR_HI 
 b'LOOKUP TABLE 256 lines Base2 Address 32 MSB'
*/
typedef struct reg_lut_func256_base2_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func256_base2_addr_hi;
static_assert((sizeof(struct reg_lut_func256_base2_addr_hi) == 4), "reg_lut_func256_base2_addr_hi size is not 32-bit");
/*
 TENSOR_SMT_PRIV 
 b'global priv address space priv bit'
*/
typedef struct reg_tensor_smt_priv {
	union {
		struct {
			uint32_t tensor_smt_priv : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_tensor_smt_priv;
static_assert((sizeof(struct reg_tensor_smt_priv) == 4), "reg_tensor_smt_priv size is not 32-bit");
/*
 TSB_CFG_MTRR_GLBL 
 b'TSB MTRR Global cfg'
*/
typedef struct reg_tsb_cfg_mtrr_glbl {
	union {
		struct {
			uint32_t en : 1,
				_reserved4 : 3,
				default_memory_type : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg_mtrr_glbl;
static_assert((sizeof(struct reg_tsb_cfg_mtrr_glbl) == 4), "reg_tsb_cfg_mtrr_glbl size is not 32-bit");
/*
 TSB_CFG_MTRR 
 b'TSB MTRR cfg'
*/
typedef struct reg_tsb_cfg_mtrr {
	union {
		struct {
			uint32_t valid : 1,
				_reserved4 : 3,
				memory_type : 1,
				_reserved8 : 3,
				phy_base_add : 16,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg_mtrr;
static_assert((sizeof(struct reg_tsb_cfg_mtrr) == 4), "reg_tsb_cfg_mtrr size is not 32-bit");
/*
 TSB_CFG_MTRR_MASK_LO 
 b'TSB MTRR mask cfg lo'
*/
typedef struct reg_tsb_cfg_mtrr_mask_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg_mtrr_mask_lo;
static_assert((sizeof(struct reg_tsb_cfg_mtrr_mask_lo) == 4), "reg_tsb_cfg_mtrr_mask_lo size is not 32-bit");
/*
 TSB_CFG_MTRR_MASK_HI 
 b'TSB MTRR mask cfg hi'
*/
typedef struct reg_tsb_cfg_mtrr_mask_hi {
	union {
		struct {
			uint32_t v : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg_mtrr_mask_hi;
static_assert((sizeof(struct reg_tsb_cfg_mtrr_mask_hi) == 4), "reg_tsb_cfg_mtrr_mask_hi size is not 32-bit");
/*
 HBW_AWLEN_MAX 
 b'maximum HBW AXI AWLEN value'
*/
typedef struct reg_hbw_awlen_max {
	union {
		struct {
			uint32_t hbw_awlen_max : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_hbw_awlen_max;
static_assert((sizeof(struct reg_hbw_awlen_max) == 4), "reg_hbw_awlen_max size is not 32-bit");
/*
 TPC_TIES 
 b'contain some tie values to allow configuration'
*/
typedef struct reg_tpc_ties {
	union {
		struct {
			uint32_t tsb_single_halt_mode : 1,
				pcu_smt_trace_dup_en : 1,
				mask_vpu_halt_trc_th1to3 : 1,
				seq_deactivate_halt_empty : 1,
				vlm_oob_prot_en : 1,
				vlm_oob_prot_type : 1,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_tpc_ties;
static_assert((sizeof(struct reg_tpc_ties) == 4), "reg_tpc_ties size is not 32-bit");
/*
 CCL_ID_TH 
 b'CCL ID backpressure threshold'
*/
typedef struct reg_ccl_id_th {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ccl_id_th;
static_assert((sizeof(struct reg_ccl_id_th) == 4), "reg_ccl_id_th size is not 32-bit");
/*
 ALLOCDH_EXC 
 b'Alloc D,H for exclusive ld/st global access'
*/
typedef struct reg_allocdh_exc {
	union {
		struct {
			uint32_t alloc_h : 1,
				alloc_d : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_allocdh_exc;
static_assert((sizeof(struct reg_allocdh_exc) == 4), "reg_allocdh_exc size is not 32-bit");
/*
 ALLOCKDH_L2_PREF 
 b'Alloc D,H for L2 Dcache prefetch'
*/
typedef struct reg_allockdh_l2_pref {
	union {
		struct {
			uint32_t alloch : 1,
				allocd : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_allockdh_l2_pref;
static_assert((sizeof(struct reg_allockdh_l2_pref) == 4), "reg_allockdh_l2_pref size is not 32-bit");
/*
 CCL_STS 
 b'CCL status'
*/
typedef struct reg_ccl_sts {
	union {
		struct {
			uint32_t ib : 5,
				id : 4,
				fsm_cs : 2,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_ccl_sts;
static_assert((sizeof(struct reg_ccl_sts) == 4), "reg_ccl_sts size is not 32-bit");
/*
 CCL_FL_STS 
 b'CCL free-list status'
*/
typedef struct reg_ccl_fl_sts {
	union {
		struct {
			uint32_t even : 12,
				odd : 12,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_ccl_fl_sts;
static_assert((sizeof(struct reg_ccl_fl_sts) == 4), "reg_ccl_fl_sts size is not 32-bit");
/*
 CCL_STL_ADDR_EVEN_LO_STS 
 b'CCL stall address, low part of even lane'
*/
typedef struct reg_ccl_stl_addr_even_lo_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ccl_stl_addr_even_lo_sts;
static_assert((sizeof(struct reg_ccl_stl_addr_even_lo_sts) == 4), "reg_ccl_stl_addr_even_lo_sts size is not 32-bit");
/*
 CCL_STL_ADDR_EVEN_HI_STS 
 b'CCL stall address, high part of even lane'
*/
typedef struct reg_ccl_stl_addr_even_hi_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ccl_stl_addr_even_hi_sts;
static_assert((sizeof(struct reg_ccl_stl_addr_even_hi_sts) == 4), "reg_ccl_stl_addr_even_hi_sts size is not 32-bit");
/*
 CCL_STL_ADDR_ODD_LO_STS 
 b'CCL stall address, low part of odd lane'
*/
typedef struct reg_ccl_stl_addr_odd_lo_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ccl_stl_addr_odd_lo_sts;
static_assert((sizeof(struct reg_ccl_stl_addr_odd_lo_sts) == 4), "reg_ccl_stl_addr_odd_lo_sts size is not 32-bit");
/*
 CCL_STL_ADDR_ODD_HI_STS 
 b'CCL stall address, high part of odd lane'
*/
typedef struct reg_ccl_stl_addr_odd_hi_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ccl_stl_addr_odd_hi_sts;
static_assert((sizeof(struct reg_ccl_stl_addr_odd_hi_sts) == 4), "reg_ccl_stl_addr_odd_hi_sts size is not 32-bit");
/*
 CCL_STL_CNT_STS 
 b'CCL stall counter value'
*/
typedef struct reg_ccl_stl_cnt_sts {
	union {
		struct {
			uint32_t even : 16,
				odd : 16;
		};
		uint32_t _raw;
	};
} reg_ccl_stl_cnt_sts;
static_assert((sizeof(struct reg_ccl_stl_cnt_sts) == 4), "reg_ccl_stl_cnt_sts size is not 32-bit");
/*
 HB_PROT_BIT0 
 b'AXI HBW AxPROT bit0'
*/
typedef struct reg_hb_prot_bit0 {
	union {
		struct {
			uint32_t awprot : 1,
				arprot : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_hb_prot_bit0;
static_assert((sizeof(struct reg_hb_prot_bit0) == 4), "reg_hb_prot_bit0 size is not 32-bit");
/*
 HB_PROT_BIT1 
 b'AXI HBW AxPROT bit1'
*/
typedef struct reg_hb_prot_bit1 {
	union {
		struct {
			uint32_t awprot : 1,
				arprot : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_hb_prot_bit1;
static_assert((sizeof(struct reg_hb_prot_bit1) == 4), "reg_hb_prot_bit1 size is not 32-bit");
/*
 HB_PROT_BIT2 
 b'AXI HBW AxPROT bit2'
*/
typedef struct reg_hb_prot_bit2 {
	union {
		struct {
			uint32_t awprot : 1,
				arprot : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_hb_prot_bit2;
static_assert((sizeof(struct reg_hb_prot_bit2) == 4), "reg_hb_prot_bit2 size is not 32-bit");
/*
 LB_PROT_BIT0 
 b'AXI LBW AxPROT bit0'
*/
typedef struct reg_lb_prot_bit0 {
	union {
		struct {
			uint32_t awprot : 1,
				arprot : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lb_prot_bit0;
static_assert((sizeof(struct reg_lb_prot_bit0) == 4), "reg_lb_prot_bit0 size is not 32-bit");
/*
 LB_PROT_BIT1 
 b'AXI LBW AxPROT bit1'
*/
typedef struct reg_lb_prot_bit1 {
	union {
		struct {
			uint32_t awprot : 1,
				arprot : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lb_prot_bit1;
static_assert((sizeof(struct reg_lb_prot_bit1) == 4), "reg_lb_prot_bit1 size is not 32-bit");
/*
 LB_PROT_BIT2 
 b'AXI LBW AxPROT bit2'
*/
typedef struct reg_lb_prot_bit2 {
	union {
		struct {
			uint32_t awprot : 1,
				arprot : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lb_prot_bit2;
static_assert((sizeof(struct reg_lb_prot_bit2) == 4), "reg_lb_prot_bit2 size is not 32-bit");
/*
 ICACHE_CFG 
 b'Icache Config'
*/
typedef struct reg_icache_cfg {
	union {
		struct {
			uint32_t plru_methodology_sel : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_icache_cfg;
static_assert((sizeof(struct reg_icache_cfg) == 4), "reg_icache_cfg size is not 32-bit");
/*
 CFG_AXI_128BYTE 
 b'disable SB 256-byte transactions'
*/
typedef struct reg_cfg_axi_128byte {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cfg_axi_128byte;
static_assert((sizeof(struct reg_cfg_axi_128byte) == 4), "reg_cfg_axi_128byte size is not 32-bit");
/*
 TSB_PAD_VAL_64B_HIGH 
 b'high padding value for Out Of Bound read access'
*/
typedef struct reg_tsb_pad_val_64b_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tsb_pad_val_64b_high;
static_assert((sizeof(struct reg_tsb_pad_val_64b_high) == 4), "reg_tsb_pad_val_64b_high size is not 32-bit");
/*
 STATUS 
 b'Used to qeury  the status of the TPC'
*/
typedef struct reg_status {
	union {
		struct {
			uint32_t _reserved1 : 1,
scalar_pipe_empty : 1,
				vector_pipe_empty : 1,
				iq_empty : 1,
				_reserved5 : 1,
				sb_empty : 1,
				qm_idle : 1,
				qm_rdy : 1,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_status;
static_assert((sizeof(struct reg_status) == 4), "reg_status size is not 32-bit");
/*
 CFG_BASE_ADDRESS_HIGH 
 b'higher 32 bits of the CFG base address'
*/
typedef struct reg_cfg_base_address_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cfg_base_address_high;
static_assert((sizeof(struct reg_cfg_base_address_high) == 4), "reg_cfg_base_address_high size is not 32-bit");
/*
 CFG_SUBTRACT_VALUE 
 b'Value to subtract from external MMIO address'
*/
typedef struct reg_cfg_subtract_value {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cfg_subtract_value;
static_assert((sizeof(struct reg_cfg_subtract_value) == 4), "reg_cfg_subtract_value size is not 32-bit");
/*
 SM_BASE_ADDRESS_HIGH 
 b'32 MSBs of SM Base address'
*/
typedef struct reg_sm_base_address_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_sm_base_address_high;
static_assert((sizeof(struct reg_sm_base_address_high) == 4), "reg_sm_base_address_high size is not 32-bit");
/*
 TPC_CMD 
 b'TPC commands'
*/
typedef struct reg_tpc_cmd {
	union {
		struct {
			uint32_t icache_invalidate : 1,
				dcache_invalidate : 1,
				lcache_invalidate : 1,
				tcache_invalidate : 1,
				icache_prefetch_64kb : 1,
				icache_prefetch_32kb : 1,
				qman_stop : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_tpc_cmd;
static_assert((sizeof(struct reg_tpc_cmd) == 4), "reg_tpc_cmd size is not 32-bit");
/*
 TPC_EXECUTE 
 b'TPC kernel execution control'
*/
typedef struct reg_tpc_execute {
	union {
		struct {
			uint32_t v : 1,
				nop_kernel : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_tpc_execute;
static_assert((sizeof(struct reg_tpc_execute) == 4), "reg_tpc_execute size is not 32-bit");
/*
 TPC_STALL 
 b'stalls TPC core'
*/
typedef struct reg_tpc_stall {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_tpc_stall;
static_assert((sizeof(struct reg_tpc_stall) == 4), "reg_tpc_stall size is not 32-bit");
/*
 ICACHE_BASE_ADDERESS_LOW 
 b'32 LSBs of the base address to prefetch in a 64KB'
*/
typedef struct reg_icache_base_adderess_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_icache_base_adderess_low;
static_assert((sizeof(struct reg_icache_base_adderess_low) == 4), "reg_icache_base_adderess_low size is not 32-bit");
/*
 ICACHE_BASE_ADDERESS_HIGH 
 b'32 MSBs of the base address to prefetch in a 64KB'
*/
typedef struct reg_icache_base_adderess_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_icache_base_adderess_high;
static_assert((sizeof(struct reg_icache_base_adderess_high) == 4), "reg_icache_base_adderess_high size is not 32-bit");
/*
 RD_RATE_LIMIT 
 b'AXI Read Port RATE LIMIT Static Config'
*/
typedef struct reg_rd_rate_limit {
	union {
		struct {
			uint32_t enable : 1,
				saturation : 8,
				timeout : 8,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_rd_rate_limit;
static_assert((sizeof(struct reg_rd_rate_limit) == 4), "reg_rd_rate_limit size is not 32-bit");
/*
 WR_RATE_LIMIT 
 b'AXI Write Port RATE LIMIT Static Config'
*/
typedef struct reg_wr_rate_limit {
	union {
		struct {
			uint32_t enable : 1,
				saturation : 8,
				timeout : 8,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_wr_rate_limit;
static_assert((sizeof(struct reg_wr_rate_limit) == 4), "reg_wr_rate_limit size is not 32-bit");
/*
 MSS_CONFIG 
 b'MSS configurations'
*/
typedef struct reg_mss_config {
	union {
		struct {
			uint32_t awcache : 4,
				arcache : 4,
				icache_fetch_line_num : 2,
				exposed_pipe_dis : 1,
				dcache_prefetch_dis : 1,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_mss_config;
static_assert((sizeof(struct reg_mss_config) == 4), "reg_mss_config size is not 32-bit");
/*
 TPC_INTR_CAUSE_0 
 b'TPC interrupts cause'
*/
typedef struct reg_tpc_intr_cause_0 {
	union {
		struct {
			uint32_t cause : 32;
		};
		uint32_t _raw;
	};
} reg_tpc_intr_cause_0;
static_assert((sizeof(struct reg_tpc_intr_cause_0) == 4), "reg_tpc_intr_cause_0 size is not 32-bit");
/*
 TPC_INTR_MASK_0 
 b'Set 1 to mask the corresponding interrupt'
*/
typedef struct reg_tpc_intr_mask_0 {
	union {
		struct {
			uint32_t mask : 32;
		};
		uint32_t _raw;
	};
} reg_tpc_intr_mask_0;
static_assert((sizeof(struct reg_tpc_intr_mask_0) == 4), "reg_tpc_intr_mask_0 size is not 32-bit");
/*
 WQ_CREDITS 
 b'WQ_CREDITS'
*/
typedef struct reg_wq_credits {
	union {
		struct {
			uint32_t st_g : 4,
				kernel_fifo : 3,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_wq_credits;
static_assert((sizeof(struct reg_wq_credits) == 4), "reg_wq_credits size is not 32-bit");
/*
 OPCODE_EXEC 
 b'Opcodes Executed for Counters'
*/
typedef struct reg_opcode_exec {
	union {
		struct {
			uint32_t spu_op : 7,
				spu_en : 1,
				vpu_op : 7,
				vpu_en : 1,
				ld_op : 7,
				ld_en : 1,
				st_op : 7,
				st_en : 1;
		};
		uint32_t _raw;
	};
} reg_opcode_exec;
static_assert((sizeof(struct reg_opcode_exec) == 4), "reg_opcode_exec size is not 32-bit");
/*
 LUT_FUNC32_BASE_ADDR_LO 
 b'LOOKUP TABLE 32 lines Base Address 32 LSB'
*/
typedef struct reg_lut_func32_base_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func32_base_addr_lo;
static_assert((sizeof(struct reg_lut_func32_base_addr_lo) == 4), "reg_lut_func32_base_addr_lo size is not 32-bit");
/*
 LUT_FUNC32_BASE_ADDR_HI 
 b'LOOKUP TABLE 32 lines Base Address 32 MSB'
*/
typedef struct reg_lut_func32_base_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func32_base_addr_hi;
static_assert((sizeof(struct reg_lut_func32_base_addr_hi) == 4), "reg_lut_func32_base_addr_hi size is not 32-bit");
/*
 LUT_FUNC64_BASE_ADDR_LO 
 b'LOOKUP TABLE 64 lines Base Address 32 LSB'
*/
typedef struct reg_lut_func64_base_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func64_base_addr_lo;
static_assert((sizeof(struct reg_lut_func64_base_addr_lo) == 4), "reg_lut_func64_base_addr_lo size is not 32-bit");
/*
 LUT_FUNC64_BASE_ADDR_HI 
 b'LOOKUP TABLE 64 lines Base Address 32 MSB'
*/
typedef struct reg_lut_func64_base_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func64_base_addr_hi;
static_assert((sizeof(struct reg_lut_func64_base_addr_hi) == 4), "reg_lut_func64_base_addr_hi size is not 32-bit");
/*
 LUT_FUNC128_BASE_ADDR_LO 
 b'LOOKUP TABLE 128 lines Base Address 32 LSB'
*/
typedef struct reg_lut_func128_base_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func128_base_addr_lo;
static_assert((sizeof(struct reg_lut_func128_base_addr_lo) == 4), "reg_lut_func128_base_addr_lo size is not 32-bit");
/*
 LUT_FUNC128_BASE_ADDR_HI 
 b'LOOKUP TABLE 128 lines Base Address 32 MSB'
*/
typedef struct reg_lut_func128_base_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func128_base_addr_hi;
static_assert((sizeof(struct reg_lut_func128_base_addr_hi) == 4), "reg_lut_func128_base_addr_hi size is not 32-bit");
/*
 LUT_FUNC256_BASE_ADDR_LO 
 b'LOOKUP TABLE 256 lines Base Address 32 LSB'
*/
typedef struct reg_lut_func256_base_addr_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func256_base_addr_lo;
static_assert((sizeof(struct reg_lut_func256_base_addr_lo) == 4), "reg_lut_func256_base_addr_lo size is not 32-bit");
/*
 LUT_FUNC256_BASE_ADDR_HI 
 b'LOOKUP TABLE 256 lines Base Address 32 MSB'
*/
typedef struct reg_lut_func256_base_addr_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_lut_func256_base_addr_hi;
static_assert((sizeof(struct reg_lut_func256_base_addr_hi) == 4), "reg_lut_func256_base_addr_hi size is not 32-bit");
/*
 TSB_CFG_MAX_SIZE 
 b'TSB Configuration'
*/
typedef struct reg_tsb_cfg_max_size {
	union {
		struct {
			uint32_t data : 16,
				md : 16;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg_max_size;
static_assert((sizeof(struct reg_tsb_cfg_max_size) == 4), "reg_tsb_cfg_max_size size is not 32-bit");
/*
 TSB_CFG 
 b'more TSB configuration'
*/
typedef struct reg_tsb_cfg {
	union {
		struct {
			uint32_t cache_disable : 1,
				max_os : 16,
				enable_cgate : 1,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_tsb_cfg;
static_assert((sizeof(struct reg_tsb_cfg) == 4), "reg_tsb_cfg size is not 32-bit");
/*
 TSB_INFLIGHT_CNTR 
 b'Status of TSB number of read inflights'
*/
typedef struct reg_tsb_inflight_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tsb_inflight_cntr;
static_assert((sizeof(struct reg_tsb_inflight_cntr) == 4), "reg_tsb_inflight_cntr size is not 32-bit");
/*
 WQ_INFLIGHT_CNTR 
 b'Status of WQ number of write inflights'
*/
typedef struct reg_wq_inflight_cntr {
	union {
		struct {
			uint32_t hbw : 16,
				lbw : 9,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_wq_inflight_cntr;
static_assert((sizeof(struct reg_wq_inflight_cntr) == 4), "reg_wq_inflight_cntr size is not 32-bit");
/*
 WQ_LBW_TOTAL_CNTR 
 b'writing reset counter'
*/
typedef struct reg_wq_lbw_total_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_wq_lbw_total_cntr;
static_assert((sizeof(struct reg_wq_lbw_total_cntr) == 4), "reg_wq_lbw_total_cntr size is not 32-bit");
/*
 WQ_HBW_TOTAL_CNTR 
 b'writing reset counter'
*/
typedef struct reg_wq_hbw_total_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_wq_hbw_total_cntr;
static_assert((sizeof(struct reg_wq_hbw_total_cntr) == 4), "reg_wq_hbw_total_cntr size is not 32-bit");
/*
 IRQ_OCCOUPY_CNTR 
 b'IQ memory occupancy status'
*/
typedef struct reg_irq_occoupy_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_irq_occoupy_cntr;
static_assert((sizeof(struct reg_irq_occoupy_cntr) == 4), "reg_irq_occoupy_cntr size is not 32-bit");

#ifdef __cplusplus
} /* tpc namespace */
#endif

/*
 TPC block
*/

#ifdef __cplusplus

struct block_tpc {
	struct block_tpc_non_tensor_descriptor kernel;
	struct block_tpc_non_tensor_descriptor_qm qm_shared_kernel_smt;
	struct block_tpc_tensor_base qm_tensor_0;
	struct block_tpc_tensor_shared qm_tensor_0_shared;
	struct block_tpc_tensor_base qm_tensor_1;
	struct block_tpc_tensor_shared qm_tensor_1_shared;
	struct block_tpc_tensor_base qm_tensor_2;
	struct block_tpc_tensor_shared qm_tensor_2_shared;
	struct block_tpc_tensor_base qm_tensor_3;
	struct block_tpc_tensor_shared qm_tensor_3_shared;
	struct block_tpc_tensor_base qm_tensor_4;
	struct block_tpc_tensor_shared qm_tensor_4_shared;
	struct block_tpc_tensor_base qm_tensor_5;
	struct block_tpc_tensor_shared qm_tensor_5_shared;
	struct block_tpc_tensor_base qm_tensor_6;
	struct block_tpc_tensor_shared qm_tensor_6_shared;
	struct block_tpc_tensor_base qm_tensor_7;
	struct block_tpc_tensor_shared qm_tensor_7_shared;
	struct block_tpc_tensor_base qm_tensor_8;
	struct block_tpc_tensor_shared qm_tensor_8_shared;
	struct block_tpc_tensor_base qm_tensor_9;
	struct block_tpc_tensor_shared qm_tensor_9_shared;
	struct block_tpc_tensor_base qm_tensor_10;
	struct block_tpc_tensor_shared qm_tensor_10_shared;
	struct block_tpc_tensor_base qm_tensor_11;
	struct block_tpc_tensor_shared qm_tensor_11_shared;
	struct block_tpc_tensor_base qm_tensor_12;
	struct block_tpc_tensor_shared qm_tensor_12_shared;
	struct block_tpc_tensor_base qm_tensor_13;
	struct block_tpc_tensor_shared qm_tensor_13_shared;
	struct block_tpc_tensor_base qm_tensor_14;
	struct block_tpc_tensor_shared qm_tensor_14_shared;
	struct block_tpc_tensor_base qm_tensor_15;
	struct block_tpc_tensor_shared qm_tensor_15_shared;
	struct block_sync_object qm_sync_object_th0;
	struct block_tpc_non_tensor_descriptor_smt qm_smt_th0;
	struct block_tpc_non_tensor_descriptor qm;
	struct block_sync_object qm_sync_object_th1;
	struct block_tpc_non_tensor_descriptor_smt qm_smt_th1;
	struct block_sync_object qm_sync_object_th2;
	struct block_tpc_non_tensor_descriptor_smt qm_smt_th2;
	struct block_sync_object qm_sync_object_th3;
	struct block_tpc_non_tensor_descriptor_smt qm_smt_th3;
	struct block_tpc_dcache dcache;
	uint32_t _pad2488[125];
	struct tpc::reg_ccl_range_dis ccl_range_dis;
	struct tpc::reg_ccl_scram_poly_matrix_0 ccl_scram_poly_matrix_0;
	struct tpc::reg_ccl_scram_poly_matrix_1 ccl_scram_poly_matrix_1;
	struct tpc::reg_ccl_scram_poly_matrix_2 ccl_scram_poly_matrix_2;
	struct tpc::reg_ccl_scram_poly_matrix_3 ccl_scram_poly_matrix_3;
	struct tpc::reg_ccl_scram_poly_matrix_4 ccl_scram_poly_matrix_4;
	struct tpc::reg_ccl_scram_poly_matrix_5 ccl_scram_poly_matrix_5;
	struct tpc::reg_ccl_scram_poly_matrix_6 ccl_scram_poly_matrix_6;
	struct tpc::reg_ccl_scram_poly_matrix_7 ccl_scram_poly_matrix_7;
	struct tpc::reg_ccl_scram_poly_matrix_8 ccl_scram_poly_matrix_8;
	struct tpc::reg_ccl_scram_poly_matrix_9 ccl_scram_poly_matrix_9;
	struct tpc::reg_ccl_scram_poly_matrix_10 ccl_scram_poly_matrix_10;
	struct tpc::reg_ccl_scram_poly_matrix_11 ccl_scram_poly_matrix_11;
	struct tpc::reg_ccl_scram_poly_matrix_12 ccl_scram_poly_matrix_12;
	struct tpc::reg_ccl_scram_poly_matrix_13 ccl_scram_poly_matrix_13;
	struct tpc::reg_ccl_scram_poly_matrix_14 ccl_scram_poly_matrix_14;
	struct tpc::reg_ccl_scram_poly_matrix_15 ccl_scram_poly_matrix_15;
	struct tpc::reg_ccl_scram_cfg ccl_scram_cfg;
	struct tpc::reg_vpu_ongoing_kernel_id vpu_ongoing_kernel_id;
	struct tpc::reg_dcach_mem_err_kernel_id dcach_mem_err_kernel_id;
	struct tpc::reg_wq_mem_err_kernel_id wq_mem_err_kernel_id;
	struct tpc::reg_tsb_mem_err_kernel_id tsb_mem_err_kernel_id;
	struct tpc::reg_spmu_thrd_msk spmu_thrd_msk;
	struct tpc::reg_stall_on_err_mask_0 stall_on_err_mask_0;
	struct tpc::reg_stall_on_err_mask_1 stall_on_err_mask_1;
	struct tpc::reg_tpc_intr_cause_1 tpc_intr_cause_1;
	struct tpc::reg_tpc_intr_mask_1 tpc_intr_mask_1;
	struct tpc::reg_tpc_count tpc_count;
	uint32_t _pad3100[1];
	struct tpc::reg_stall_on_err stall_on_err;
	struct tpc::reg_clk_en clk_en;
	struct tpc::reg_iq_rl_en iq_rl_en;
	struct tpc::reg_iq_rl_sat iq_rl_sat;
	struct tpc::reg_iq_rl_rst_token iq_rl_rst_token;
	struct tpc::reg_iq_rl_timeout iq_rl_timeout;
	struct tpc::reg_tsb_cfg_mtrr_2 tsb_cfg_mtrr_2[4];
	struct tpc::reg_iq_lbw_clk_en iq_lbw_clk_en;
	struct tpc::reg_tpc_lock_value tpc_lock_value[4];
	struct tpc::reg_tpc_lock tpc_lock[4];
	struct tpc::reg_cgu_sb cgu_sb;
	struct tpc::reg_cgu_cnt cgu_cnt;
	struct tpc::reg_cgu_cpe cgu_cpe[8];
	struct tpc::reg_tpc_sb_l0cd tpc_sb_l0cd;
	struct tpc::reg_tsb_occupancy tsb_occupancy;
	struct tpc::reg_tsb_data_occupancy tsb_data_occupancy;
	struct tpc::reg_tsb_md_occupancy tsb_md_occupancy;
	struct tpc::reg_smt_cause0_all_thrd smt_cause0_all_thrd;
	struct tpc::reg_smt_cause1_all_thrd smt_cause1_all_thrd;
	struct tpc::reg_sqz_oob_protection sqz_oob_protection;
	struct tpc::reg_arb_qnt_hbw_weight arb_qnt_hbw_weight;
	struct tpc::reg_arb_qnt_lbw_weight arb_qnt_lbw_weight;
	struct tpc::reg_arb_cnt_hbw_weight arb_cnt_hbw_weight;
	struct tpc::reg_arb_cnt_lbw_weight arb_cnt_lbw_weight;
	struct tpc::reg_lut_func32_base2_addr_lo lut_func32_base2_addr_lo;
	struct tpc::reg_lut_func32_base2_addr_hi lut_func32_base2_addr_hi;
	struct tpc::reg_lut_func64_base2_addr_lo lut_func64_base2_addr_lo;
	struct tpc::reg_lut_func64_base2_addr_hi lut_func64_base2_addr_hi;
	struct tpc::reg_lut_func128_base2_addr_lo lut_func128_base2_addr_lo;
	struct tpc::reg_lut_func128_base2_addr_hi lut_func128_base2_addr_hi;
	struct tpc::reg_lut_func256_base2_addr_lo lut_func256_base2_addr_lo;
	struct tpc::reg_lut_func256_base2_addr_hi lut_func256_base2_addr_hi;
	struct tpc::reg_tensor_smt_priv tensor_smt_priv;
	struct tpc::reg_tsb_cfg_mtrr_glbl tsb_cfg_mtrr_glbl;
	struct tpc::reg_tsb_cfg_mtrr tsb_cfg_mtrr[4];
	struct tpc::reg_tsb_cfg_mtrr_mask_lo tsb_cfg_mtrr_mask_lo[4];
	struct tpc::reg_tsb_cfg_mtrr_mask_hi tsb_cfg_mtrr_mask_hi[4];
	uint32_t _pad3352[1];
	struct tpc::reg_hbw_awlen_max hbw_awlen_max;
	struct tpc::reg_tpc_ties tpc_ties;
	struct tpc::reg_ccl_id_th ccl_id_th;
	struct tpc::reg_allocdh_exc allocdh_exc;
	struct tpc::reg_allockdh_l2_pref allockdh_l2_pref;
	struct tpc::reg_ccl_sts ccl_sts;
	struct tpc::reg_ccl_fl_sts ccl_fl_sts;
	struct tpc::reg_ccl_stl_addr_even_lo_sts ccl_stl_addr_even_lo_sts;
	struct tpc::reg_ccl_stl_addr_even_hi_sts ccl_stl_addr_even_hi_sts;
	struct tpc::reg_ccl_stl_addr_odd_lo_sts ccl_stl_addr_odd_lo_sts;
	struct tpc::reg_ccl_stl_addr_odd_hi_sts ccl_stl_addr_odd_hi_sts;
	struct tpc::reg_ccl_stl_cnt_sts ccl_stl_cnt_sts;
	uint32_t _pad3404[4];
	struct tpc::reg_hb_prot_bit0 hb_prot_bit0;
	struct tpc::reg_hb_prot_bit1 hb_prot_bit1;
	struct tpc::reg_hb_prot_bit2 hb_prot_bit2;
	struct tpc::reg_lb_prot_bit0 lb_prot_bit0;
	struct tpc::reg_lb_prot_bit1 lb_prot_bit1;
	struct tpc::reg_lb_prot_bit2 lb_prot_bit2;
	struct tpc::reg_icache_cfg icache_cfg;
	struct tpc::reg_cfg_axi_128byte cfg_axi_128byte;
	struct tpc::reg_tsb_pad_val_64b_high tsb_pad_val_64b_high;
	uint32_t _pad3456[1];
	struct tpc::reg_status status;
	struct tpc::reg_cfg_base_address_high cfg_base_address_high;
	struct tpc::reg_cfg_subtract_value cfg_subtract_value;
	struct tpc::reg_sm_base_address_high sm_base_address_high;
	struct tpc::reg_tpc_cmd tpc_cmd;
	struct tpc::reg_tpc_execute tpc_execute;
	struct tpc::reg_tpc_stall tpc_stall;
	struct tpc::reg_icache_base_adderess_low icache_base_adderess_low;
	struct tpc::reg_icache_base_adderess_high icache_base_adderess_high;
	struct tpc::reg_rd_rate_limit rd_rate_limit;
	struct tpc::reg_wr_rate_limit wr_rate_limit;
	struct tpc::reg_mss_config mss_config;
	struct tpc::reg_tpc_intr_cause_0 tpc_intr_cause_0;
	struct tpc::reg_tpc_intr_mask_0 tpc_intr_mask_0;
	struct tpc::reg_wq_credits wq_credits;
	struct tpc::reg_opcode_exec opcode_exec;
	struct tpc::reg_lut_func32_base_addr_lo lut_func32_base_addr_lo;
	struct tpc::reg_lut_func32_base_addr_hi lut_func32_base_addr_hi;
	struct tpc::reg_lut_func64_base_addr_lo lut_func64_base_addr_lo;
	struct tpc::reg_lut_func64_base_addr_hi lut_func64_base_addr_hi;
	struct tpc::reg_lut_func128_base_addr_lo lut_func128_base_addr_lo;
	struct tpc::reg_lut_func128_base_addr_hi lut_func128_base_addr_hi;
	struct tpc::reg_lut_func256_base_addr_lo lut_func256_base_addr_lo;
	struct tpc::reg_lut_func256_base_addr_hi lut_func256_base_addr_hi;
	struct tpc::reg_tsb_cfg_max_size tsb_cfg_max_size;
	struct tpc::reg_tsb_cfg tsb_cfg;
	struct tpc::reg_tsb_inflight_cntr tsb_inflight_cntr;
	struct tpc::reg_wq_inflight_cntr wq_inflight_cntr;
	struct tpc::reg_wq_lbw_total_cntr wq_lbw_total_cntr;
	struct tpc::reg_wq_hbw_total_cntr wq_hbw_total_cntr;
	struct tpc::reg_irq_occoupy_cntr irq_occoupy_cntr;
	struct block_axuser_hbw axuser_hbw;
	uint32_t _pad3676[2];
	struct block_axuser_lbw axuser_lbw;
	uint32_t _pad3708[1];
	struct block_special_regs special;
};
#else

typedef struct block_tpc {
	block_tpc_non_tensor_descriptor kernel;
	block_tpc_non_tensor_descriptor_qm qm_shared_kernel_smt;
	block_tpc_tensor_base qm_tensor_0;
	block_tpc_tensor_shared qm_tensor_0_shared;
	block_tpc_tensor_base qm_tensor_1;
	block_tpc_tensor_shared qm_tensor_1_shared;
	block_tpc_tensor_base qm_tensor_2;
	block_tpc_tensor_shared qm_tensor_2_shared;
	block_tpc_tensor_base qm_tensor_3;
	block_tpc_tensor_shared qm_tensor_3_shared;
	block_tpc_tensor_base qm_tensor_4;
	block_tpc_tensor_shared qm_tensor_4_shared;
	block_tpc_tensor_base qm_tensor_5;
	block_tpc_tensor_shared qm_tensor_5_shared;
	block_tpc_tensor_base qm_tensor_6;
	block_tpc_tensor_shared qm_tensor_6_shared;
	block_tpc_tensor_base qm_tensor_7;
	block_tpc_tensor_shared qm_tensor_7_shared;
	block_tpc_tensor_base qm_tensor_8;
	block_tpc_tensor_shared qm_tensor_8_shared;
	block_tpc_tensor_base qm_tensor_9;
	block_tpc_tensor_shared qm_tensor_9_shared;
	block_tpc_tensor_base qm_tensor_10;
	block_tpc_tensor_shared qm_tensor_10_shared;
	block_tpc_tensor_base qm_tensor_11;
	block_tpc_tensor_shared qm_tensor_11_shared;
	block_tpc_tensor_base qm_tensor_12;
	block_tpc_tensor_shared qm_tensor_12_shared;
	block_tpc_tensor_base qm_tensor_13;
	block_tpc_tensor_shared qm_tensor_13_shared;
	block_tpc_tensor_base qm_tensor_14;
	block_tpc_tensor_shared qm_tensor_14_shared;
	block_tpc_tensor_base qm_tensor_15;
	block_tpc_tensor_shared qm_tensor_15_shared;
	block_sync_object qm_sync_object_th0;
	block_tpc_non_tensor_descriptor_smt qm_smt_th0;
	block_tpc_non_tensor_descriptor qm;
	block_sync_object qm_sync_object_th1;
	block_tpc_non_tensor_descriptor_smt qm_smt_th1;
	block_sync_object qm_sync_object_th2;
	block_tpc_non_tensor_descriptor_smt qm_smt_th2;
	block_sync_object qm_sync_object_th3;
	block_tpc_non_tensor_descriptor_smt qm_smt_th3;
	block_tpc_dcache dcache;
	uint32_t _pad2488[125];
	reg_ccl_range_dis ccl_range_dis;
	reg_ccl_scram_poly_matrix_0 ccl_scram_poly_matrix_0;
	reg_ccl_scram_poly_matrix_1 ccl_scram_poly_matrix_1;
	reg_ccl_scram_poly_matrix_2 ccl_scram_poly_matrix_2;
	reg_ccl_scram_poly_matrix_3 ccl_scram_poly_matrix_3;
	reg_ccl_scram_poly_matrix_4 ccl_scram_poly_matrix_4;
	reg_ccl_scram_poly_matrix_5 ccl_scram_poly_matrix_5;
	reg_ccl_scram_poly_matrix_6 ccl_scram_poly_matrix_6;
	reg_ccl_scram_poly_matrix_7 ccl_scram_poly_matrix_7;
	reg_ccl_scram_poly_matrix_8 ccl_scram_poly_matrix_8;
	reg_ccl_scram_poly_matrix_9 ccl_scram_poly_matrix_9;
	reg_ccl_scram_poly_matrix_10 ccl_scram_poly_matrix_10;
	reg_ccl_scram_poly_matrix_11 ccl_scram_poly_matrix_11;
	reg_ccl_scram_poly_matrix_12 ccl_scram_poly_matrix_12;
	reg_ccl_scram_poly_matrix_13 ccl_scram_poly_matrix_13;
	reg_ccl_scram_poly_matrix_14 ccl_scram_poly_matrix_14;
	reg_ccl_scram_poly_matrix_15 ccl_scram_poly_matrix_15;
	reg_ccl_scram_cfg ccl_scram_cfg;
	reg_vpu_ongoing_kernel_id vpu_ongoing_kernel_id;
	reg_dcach_mem_err_kernel_id dcach_mem_err_kernel_id;
	reg_wq_mem_err_kernel_id wq_mem_err_kernel_id;
	reg_tsb_mem_err_kernel_id tsb_mem_err_kernel_id;
	reg_spmu_thrd_msk spmu_thrd_msk;
	reg_stall_on_err_mask_0 stall_on_err_mask_0;
	reg_stall_on_err_mask_1 stall_on_err_mask_1;
	reg_tpc_intr_cause_1 tpc_intr_cause_1;
	reg_tpc_intr_mask_1 tpc_intr_mask_1;
	reg_tpc_count tpc_count;
	uint32_t _pad3100[1];
	reg_stall_on_err stall_on_err;
	reg_clk_en clk_en;
	reg_iq_rl_en iq_rl_en;
	reg_iq_rl_sat iq_rl_sat;
	reg_iq_rl_rst_token iq_rl_rst_token;
	reg_iq_rl_timeout iq_rl_timeout;
	reg_tsb_cfg_mtrr_2 tsb_cfg_mtrr_2[4];
	reg_iq_lbw_clk_en iq_lbw_clk_en;
	reg_tpc_lock_value tpc_lock_value[4];
	reg_tpc_lock tpc_lock[4];
	reg_cgu_sb cgu_sb;
	reg_cgu_cnt cgu_cnt;
	reg_cgu_cpe cgu_cpe[8];
	reg_tpc_sb_l0cd tpc_sb_l0cd;
	reg_tsb_occupancy tsb_occupancy;
	reg_tsb_data_occupancy tsb_data_occupancy;
	reg_tsb_md_occupancy tsb_md_occupancy;
	reg_smt_cause0_all_thrd smt_cause0_all_thrd;
	reg_smt_cause1_all_thrd smt_cause1_all_thrd;
	reg_sqz_oob_protection sqz_oob_protection;
	reg_arb_qnt_hbw_weight arb_qnt_hbw_weight;
	reg_arb_qnt_lbw_weight arb_qnt_lbw_weight;
	reg_arb_cnt_hbw_weight arb_cnt_hbw_weight;
	reg_arb_cnt_lbw_weight arb_cnt_lbw_weight;
	reg_lut_func32_base2_addr_lo lut_func32_base2_addr_lo;
	reg_lut_func32_base2_addr_hi lut_func32_base2_addr_hi;
	reg_lut_func64_base2_addr_lo lut_func64_base2_addr_lo;
	reg_lut_func64_base2_addr_hi lut_func64_base2_addr_hi;
	reg_lut_func128_base2_addr_lo lut_func128_base2_addr_lo;
	reg_lut_func128_base2_addr_hi lut_func128_base2_addr_hi;
	reg_lut_func256_base2_addr_lo lut_func256_base2_addr_lo;
	reg_lut_func256_base2_addr_hi lut_func256_base2_addr_hi;
	reg_tensor_smt_priv tensor_smt_priv;
	reg_tsb_cfg_mtrr_glbl tsb_cfg_mtrr_glbl;
	reg_tsb_cfg_mtrr tsb_cfg_mtrr[4];
	reg_tsb_cfg_mtrr_mask_lo tsb_cfg_mtrr_mask_lo[4];
	reg_tsb_cfg_mtrr_mask_hi tsb_cfg_mtrr_mask_hi[4];
	uint32_t _pad3352[1];
	reg_hbw_awlen_max hbw_awlen_max;
	reg_tpc_ties tpc_ties;
	reg_ccl_id_th ccl_id_th;
	reg_allocdh_exc allocdh_exc;
	reg_allockdh_l2_pref allockdh_l2_pref;
	reg_ccl_sts ccl_sts;
	reg_ccl_fl_sts ccl_fl_sts;
	reg_ccl_stl_addr_even_lo_sts ccl_stl_addr_even_lo_sts;
	reg_ccl_stl_addr_even_hi_sts ccl_stl_addr_even_hi_sts;
	reg_ccl_stl_addr_odd_lo_sts ccl_stl_addr_odd_lo_sts;
	reg_ccl_stl_addr_odd_hi_sts ccl_stl_addr_odd_hi_sts;
	reg_ccl_stl_cnt_sts ccl_stl_cnt_sts;
	uint32_t _pad3404[4];
	reg_hb_prot_bit0 hb_prot_bit0;
	reg_hb_prot_bit1 hb_prot_bit1;
	reg_hb_prot_bit2 hb_prot_bit2;
	reg_lb_prot_bit0 lb_prot_bit0;
	reg_lb_prot_bit1 lb_prot_bit1;
	reg_lb_prot_bit2 lb_prot_bit2;
	reg_icache_cfg icache_cfg;
	reg_cfg_axi_128byte cfg_axi_128byte;
	reg_tsb_pad_val_64b_high tsb_pad_val_64b_high;
	uint32_t _pad3456[1];
	reg_status status;
	reg_cfg_base_address_high cfg_base_address_high;
	reg_cfg_subtract_value cfg_subtract_value;
	reg_sm_base_address_high sm_base_address_high;
	reg_tpc_cmd tpc_cmd;
	reg_tpc_execute tpc_execute;
	reg_tpc_stall tpc_stall;
	reg_icache_base_adderess_low icache_base_adderess_low;
	reg_icache_base_adderess_high icache_base_adderess_high;
	reg_rd_rate_limit rd_rate_limit;
	reg_wr_rate_limit wr_rate_limit;
	reg_mss_config mss_config;
	reg_tpc_intr_cause_0 tpc_intr_cause_0;
	reg_tpc_intr_mask_0 tpc_intr_mask_0;
	reg_wq_credits wq_credits;
	reg_opcode_exec opcode_exec;
	reg_lut_func32_base_addr_lo lut_func32_base_addr_lo;
	reg_lut_func32_base_addr_hi lut_func32_base_addr_hi;
	reg_lut_func64_base_addr_lo lut_func64_base_addr_lo;
	reg_lut_func64_base_addr_hi lut_func64_base_addr_hi;
	reg_lut_func128_base_addr_lo lut_func128_base_addr_lo;
	reg_lut_func128_base_addr_hi lut_func128_base_addr_hi;
	reg_lut_func256_base_addr_lo lut_func256_base_addr_lo;
	reg_lut_func256_base_addr_hi lut_func256_base_addr_hi;
	reg_tsb_cfg_max_size tsb_cfg_max_size;
	reg_tsb_cfg tsb_cfg;
	reg_tsb_inflight_cntr tsb_inflight_cntr;
	reg_wq_inflight_cntr wq_inflight_cntr;
	reg_wq_lbw_total_cntr wq_lbw_total_cntr;
	reg_wq_hbw_total_cntr wq_hbw_total_cntr;
	reg_irq_occoupy_cntr irq_occoupy_cntr;
	block_axuser_hbw axuser_hbw;
	uint32_t _pad3676[2];
	block_axuser_lbw axuser_lbw;
	uint32_t _pad3708[1];
	block_special_regs special;
} block_tpc;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_tpc_defaults[] =
{
	// offset	// value
	{ 0x8   , 0x2080902           , 1 }, // kernel_config
	{ 0x28  , 0xf                 , 1 }, // active_thrd
	{ 0x138 , 0x1                 , 1 }, // sync_msg_mode_smt4
	{ 0x13c , 0x280000            , 1 }, // ld_default_hbw_axi_cfg
	{ 0x140 , 0x2c0000            , 1 }, // st_default_hbw_axi_cfg
	{ 0x144 , 0x30202             , 1 }, // dcache_pref_window_init
	{ 0x148 , 0x30202             , 1 }, // dcache_pref_dynamic_window
	{ 0x14c , 0x42080208          , 1 }, // dcache_pref_l1_window_limit
	{ 0x150 , 0x42100210          , 1 }, // dcache_pref_l2_window_limit
	{ 0x154 , 0x2bc0064           , 1 }, // dcache_stall_length_thr
	{ 0x158 , 0x1                 , 1 }, // irf44_sat
	{ 0x160 , 0x800               , 1 }, // irf44_sat_high
	{ 0x164 , 0x3                 , 1 }, // tsb_st_direct_mode
	{ 0x168 , 0x7                 , 1 }, // kernel_fp8_bias
	{ 0x16c , 0x2                 , 1 }, // convert_cfg
	{ 0x170 , 0x8f                , 1 }, // convert_fp32_fp8_cfg
	{ 0x174 , 0xf                 , 1 }, // convert_fp32_fp16_cfg
	{ 0x18c , 0x50000             , 1 }, // tensor_config
	{ 0x1cc , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x1e8 , 0x50000             , 1 }, // tensor_config
	{ 0x228 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x244 , 0x50000             , 1 }, // tensor_config
	{ 0x284 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x2a0 , 0x50000             , 1 }, // tensor_config
	{ 0x2e0 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x2fc , 0x50000             , 1 }, // tensor_config
	{ 0x33c , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x358 , 0x50000             , 1 }, // tensor_config
	{ 0x398 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x3b4 , 0x50000             , 1 }, // tensor_config
	{ 0x3f4 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x410 , 0x50000             , 1 }, // tensor_config
	{ 0x450 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x46c , 0x50000             , 1 }, // tensor_config
	{ 0x4ac , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x4c8 , 0x50000             , 1 }, // tensor_config
	{ 0x508 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x524 , 0x50000             , 1 }, // tensor_config
	{ 0x564 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x580 , 0x50000             , 1 }, // tensor_config
	{ 0x5c0 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x5dc , 0x50000             , 1 }, // tensor_config
	{ 0x61c , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x638 , 0x50000             , 1 }, // tensor_config
	{ 0x678 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x694 , 0x50000             , 1 }, // tensor_config
	{ 0x6d4 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x6f0 , 0x50000             , 1 }, // tensor_config
	{ 0x730 , 0xba0000            , 1 }, // hbw_axi_cfg
	{ 0x784 , 0x2080902           , 1 }, // kernel_config
	{ 0x7a4 , 0xf                 , 1 }, // active_thrd
	{ 0x8b4 , 0x1                 , 1 }, // sync_msg_mode_smt4
	{ 0x8b8 , 0x280000            , 1 }, // ld_default_hbw_axi_cfg
	{ 0x8bc , 0x2c0000            , 1 }, // st_default_hbw_axi_cfg
	{ 0x8c0 , 0x30202             , 1 }, // dcache_pref_window_init
	{ 0x8c4 , 0x30202             , 1 }, // dcache_pref_dynamic_window
	{ 0x8c8 , 0x42080208          , 1 }, // dcache_pref_l1_window_limit
	{ 0x8cc , 0x42100210          , 1 }, // dcache_pref_l2_window_limit
	{ 0x8d0 , 0x2bc0064           , 1 }, // dcache_stall_length_thr
	{ 0x8d4 , 0x1                 , 1 }, // irf44_sat
	{ 0x8dc , 0x800               , 1 }, // irf44_sat_high
	{ 0x8e0 , 0x3                 , 1 }, // tsb_st_direct_mode
	{ 0x9b0 , 0xf54               , 1 }, // dcache_cfg
	{ 0xbb0 , 0x25                , 1 }, // ccl_scram_poly_matrix_0
	{ 0xbb4 , 0x4a                , 1 }, // ccl_scram_poly_matrix_1
	{ 0xbb8 , 0xe5                , 1 }, // ccl_scram_poly_matrix_2
	{ 0xbbc , 0x6e                , 1 }, // ccl_scram_poly_matrix_3
	{ 0xbc0 , 0xde                , 1 }, // ccl_scram_poly_matrix_4
	{ 0xbc4 , 0x2f                , 1 }, // ccl_scram_poly_matrix_5
	{ 0xbc8 , 0xb3                , 1 }, // ccl_scram_poly_matrix_6
	{ 0xbcc , 0x62                , 1 }, // ccl_scram_poly_matrix_7
	{ 0xbd0 , 0x70                , 1 }, // ccl_scram_poly_matrix_8
	{ 0xbd4 , 0xd5                , 1 }, // ccl_scram_poly_matrix_9
	{ 0xbd8 , 0xa7                , 1 }, // ccl_scram_poly_matrix_10
	{ 0xbdc , 0x90                , 1 }, // ccl_scram_poly_matrix_11
	{ 0xbe0 , 0x7f                , 1 }, // ccl_scram_poly_matrix_12
	{ 0xbe4 , 0xe8                , 1 }, // ccl_scram_poly_matrix_13
	{ 0xbe8 , 0xb1                , 1 }, // ccl_scram_poly_matrix_14
	{ 0xbec , 0x80                , 1 }, // ccl_scram_poly_matrix_15
	{ 0xbf0 , 0x31                , 1 }, // ccl_scram_cfg
	{ 0xc04 , 0x1                 , 1 }, // spmu_thrd_msk
	{ 0xc18 , 0x40                , 1 }, // tpc_count
	{ 0xc20 , 0x1                 , 1 }, // stall_on_err
	{ 0xc24 , 0x110               , 1 }, // clk_en
	{ 0xc2c , 0x4                 , 1 }, // iq_rl_sat
	{ 0xc30 , 0x8                 , 1 }, // iq_rl_rst_token
	{ 0xc34 , 0xf                 , 1 }, // iq_rl_timeout
	{ 0xc6c , 0x1                 , 1 }, // cgu_sb
	{ 0xc70 , 0xffffff            , 1 }, // cgu_cnt
	{ 0xc74 , 0xfffff             , 8 }, // cgu_cpe
	{ 0xcac , 0x1                 , 1 }, // sqz_oob_protection
	{ 0xcb0 , 0xf1414f            , 1 }, // arb_qnt_hbw_weight
	{ 0xcb4 , 0xfff14             , 1 }, // arb_qnt_lbw_weight
	{ 0xcb8 , 0x14f14f            , 1 }, // arb_cnt_hbw_weight
	{ 0xcbc , 0x14f14             , 1 }, // arb_cnt_lbw_weight
	{ 0xcc0 , 0x17000             , 1 }, // lut_func32_base2_addr_lo
	{ 0xcc8 , 0x13800             , 1 }, // lut_func64_base2_addr_lo
	{ 0xcd0 , 0xd000              , 1 }, // lut_func128_base2_addr_lo
	{ 0xce0 , 0x1                 , 1 }, // tensor_smt_priv
	{ 0xd20 , 0x34                , 1 }, // tpc_ties
	{ 0xd24 , 0x5                 , 1 }, // ccl_id_th
	{ 0xd28 , 0x1                 , 1 }, // allocdh_exc
	{ 0xd2c , 0x2                 , 1 }, // allockdh_l2_pref
	{ 0xd64 , 0x3                 , 1 }, // hb_prot_bit2
	{ 0xd70 , 0x3                 , 1 }, // lb_prot_bit2
	{ 0xd84 , 0xee                , 1 }, // status
	{ 0xd88 , 0x300007f           , 1 }, // cfg_base_address_high
	{ 0xda8 , 0xe0b               , 1 }, // rd_rate_limit
	{ 0xdac , 0xe0b               , 1 }, // wr_rate_limit
	{ 0xdbc , 0x4a                , 1 }, // wq_credits
	{ 0xdc0 , 0xffffffff          , 1 }, // opcode_exec
	{ 0xdc4 , 0x17000             , 1 }, // lut_func32_base_addr_lo
	{ 0xdcc , 0x13800             , 1 }, // lut_func64_base_addr_lo
	{ 0xdd4 , 0xd000              , 1 }, // lut_func128_base_addr_lo
	{ 0xde8 , 0x20000             , 1 }, // tsb_cfg
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_H_ */

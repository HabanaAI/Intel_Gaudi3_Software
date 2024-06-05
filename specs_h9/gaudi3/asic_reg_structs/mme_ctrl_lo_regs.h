/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_MME_CTRL_LO_H_
#define ASIC_REG_STRUCTS_GAUDI3_MME_CTRL_LO_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "axuser_lbw_regs.h"
#include "mme_address_descriptor_regs.h"
#include "mme_agu_core_regs.h"
#include "mme_non_tensor_descriptor_regs.h"
#include "mme_non_tensor_descriptor_start_regs.h"
#include "mme_tensor_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace mme_ctrl_lo {
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
 ARCH_STATUS_MASTER 
 b'STATUS register (Logical OR on all SHADOW STATUS)'
*/
typedef struct reg_arch_status_master {
	union {
		struct {
			uint32_t agu_in : 4,
				eu : 1,
				ap : 1,
				agu_cout : 1,
				agu_dma_o : 1,
				sb_in_empty : 4,
				agu_cout_sm_idle : 1,
				agu_dma_o_sm_idle : 1,
				wbc_axi_idle : 1,
				_reserved16 : 1,
				sb_in_axi_idle : 4,
				accum_free : 3,
				_reserved30 : 7,
				qm_idle : 1,
				qm_rdy : 1;
		};
		uint32_t _raw;
	};
} reg_arch_status_master;
static_assert((sizeof(struct reg_arch_status_master) == 4), "reg_arch_status_master size is not 32-bit");
/*
 ARCH_STATUS_SLAVE 
 b'STATUS register (Logical OR on all SHADOW STATUS)'
*/
typedef struct reg_arch_status_slave {
	union {
		struct {
			uint32_t agu_in : 4,
				eu : 1,
				ap : 1,
				agu_cout : 1,
				agu_dma_o : 1,
				sb_in_empty : 4,
				agu_cout_sm_idle : 1,
				agu_dma_o_sm_idle : 1,
				wbc_axi_idle : 1,
				_reserved16 : 1,
				sb_in_axi_idle : 4,
				accum_free : 3,
				_reserved30 : 7,
				qm_idle : 1,
				qm_rdy : 1;
		};
		uint32_t _raw;
	};
} reg_arch_status_slave;
static_assert((sizeof(struct reg_arch_status_slave) == 4), "reg_arch_status_slave size is not 32-bit");
/*
 ARCH_A_SS 
 b'SW Descriptor AGU_A Spatial Size'
*/
typedef struct reg_arch_a_ss {
	union {
		struct {
			uint32_t minus_1 : 32;
		};
		uint32_t _raw;
	};
} reg_arch_a_ss;
static_assert((sizeof(struct reg_arch_a_ss) == 4), "reg_arch_a_ss size is not 32-bit");
/*
 ARCH_B_SS 
 b'SW Descriptor MASTER AGU_B Spatial Size'
*/
typedef struct reg_arch_b_ss {
	union {
		struct {
			uint32_t minus_1 : 32;
		};
		uint32_t _raw;
	};
} reg_arch_b_ss;
static_assert((sizeof(struct reg_arch_b_ss) == 4), "reg_arch_b_ss size is not 32-bit");
/*
 ARCH_COUT_SS 
 b'SW Descriptor AGU_COUT Spatial Size'
*/
typedef struct reg_arch_cout_ss {
	union {
		struct {
			uint32_t minus_1 : 32;
		};
		uint32_t _raw;
	};
} reg_arch_cout_ss;
static_assert((sizeof(struct reg_arch_cout_ss) == 4), "reg_arch_cout_ss size is not 32-bit");
/*
 CMD 
 b'COMMAND register'
*/
typedef struct reg_cmd {
	union {
		struct {
			uint32_t agu_in : 4,
				eu : 1,
				ap : 1,
				agu_cout : 1,
				agu_dma_o : 1,
				copy_and_inc : 1,
				desc_sel : 2,
				mask_idle_ind : 1,
				null_desc : 1,
				dma_mode : 1,
				null_dma_desc : 1,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_cmd;
static_assert((sizeof(struct reg_cmd) == 4), "reg_cmd size is not 32-bit");
/*
 QM_STALL 
 b'QMAN STALL'
*/
typedef struct reg_qm_stall {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_qm_stall;
static_assert((sizeof(struct reg_qm_stall) == 4), "reg_qm_stall size is not 32-bit");
/*
 LOG_SHADOW_LO 
 b'MASK SHADOW STATUS CHANGE EVENT'
*/
typedef struct reg_log_shadow_lo {
	union {
		struct {
			uint32_t mask_0 : 16,
				mask_1 : 16;
		};
		uint32_t _raw;
	};
} reg_log_shadow_lo;
static_assert((sizeof(struct reg_log_shadow_lo) == 4), "reg_log_shadow_lo size is not 32-bit");
/*
 LOG_SHADOW_HI 
*/
typedef struct reg_log_shadow_hi {
	union {
		struct {
			uint32_t mask_2 : 16,
				mask_3 : 16;
		};
		uint32_t _raw;
	};
} reg_log_shadow_hi;
static_assert((sizeof(struct reg_log_shadow_hi) == 4), "reg_log_shadow_hi size is not 32-bit");
/*
 SYNC_OBJECT_FIFO_TH 
 b'Sync Object FIFO Theshold'
*/
typedef struct reg_sync_object_fifo_th {
	union {
		struct {
			uint32_t v : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_sync_object_fifo_th;
static_assert((sizeof(struct reg_sync_object_fifo_th) == 4), "reg_sync_object_fifo_th size is not 32-bit");
/*
 EUS_ROLLUP_DLY_DW0 
 b'EUS Rollup additional Delay'
*/
typedef struct reg_eus_rollup_dly_dw0 {
	union {
		struct {
			uint32_t fp : 4,
				fp_pe0 : 4,
				fp_pe1 : 4,
				fp_pe2 : 4,
				fp_pe3 : 4,
				fp_pe4 : 4,
				fp_pe5 : 3,
				fp_pe6 : 3,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_eus_rollup_dly_dw0;
static_assert((sizeof(struct reg_eus_rollup_dly_dw0) == 4), "reg_eus_rollup_dly_dw0 size is not 32-bit");
/*
 EUS_ROLLUP_DLY_DW1 
*/
typedef struct reg_eus_rollup_dly_dw1 {
	union {
		struct {
			uint32_t fp_pe7 : 3,
				fp_pe_hi : 3,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_eus_rollup_dly_dw1;
static_assert((sizeof(struct reg_eus_rollup_dly_dw1) == 4), "reg_eus_rollup_dly_dw1 size is not 32-bit");
/*
 EUS_ROLLUP_CD_PROT_F16 
 b'Common Dimension Protector FP/BF'
*/
typedef struct reg_eus_rollup_cd_prot_f16 {
	union {
		struct {
			uint32_t dly : 12,
				_reserved31 : 19,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_eus_rollup_cd_prot_f16;
static_assert((sizeof(struct reg_eus_rollup_cd_prot_f16) == 4), "reg_eus_rollup_cd_prot_f16 size is not 32-bit");
/*
 EUS_ROLLUP_CD_PROT_F8 
 b'Common Dimension Protector INT8'
*/
typedef struct reg_eus_rollup_cd_prot_f8 {
	union {
		struct {
			uint32_t dly : 12,
				_reserved31 : 19,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_eus_rollup_cd_prot_f8;
static_assert((sizeof(struct reg_eus_rollup_cd_prot_f8) == 4), "reg_eus_rollup_cd_prot_f8 size is not 32-bit");
/*
 EUS_ROLLUP_CD_PROT_FP32 
 b'Common Dimension Protector INT4'
*/
typedef struct reg_eus_rollup_cd_prot_fp32 {
	union {
		struct {
			uint32_t dly : 12,
				_reserved31 : 19,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_eus_rollup_cd_prot_fp32;
static_assert((sizeof(struct reg_eus_rollup_cd_prot_fp32) == 4), "reg_eus_rollup_cd_prot_fp32 size is not 32-bit");
/*
 EUS_ROLLUP_CD_PROT_TF32 
*/
typedef struct reg_eus_rollup_cd_prot_tf32 {
	union {
		struct {
			uint32_t dly : 12,
				_reserved31 : 19,
				en : 1;
		};
		uint32_t _raw;
	};
} reg_eus_rollup_cd_prot_tf32;
static_assert((sizeof(struct reg_eus_rollup_cd_prot_tf32) == 4), "reg_eus_rollup_cd_prot_tf32 size is not 32-bit");
/*
 PCU_RL_DESC0 
 b'PMU RATE LIMITER configs'
*/
typedef struct reg_pcu_rl_desc0 {
	union {
		struct {
			uint32_t rl_rst_token : 16,
				rl_timeout : 8,
				rl_dummy2real_period : 8;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_desc0;
static_assert((sizeof(struct reg_pcu_rl_desc0) == 4), "reg_pcu_rl_desc0 size is not 32-bit");
/*
 PCU_RL_TOKEN_UPDATE 
 b'PMU RATE LIMITER Tokens configs'
*/
typedef struct reg_pcu_rl_token_update {
	union {
		struct {
			uint32_t inc_val : 16,
				dec_val : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_token_update;
static_assert((sizeof(struct reg_pcu_rl_token_update) == 4), "reg_pcu_rl_token_update size is not 32-bit");
/*
 PCU_RL_TH 
 b'PMU RATE LIMITER Thresholds'
*/
typedef struct reg_pcu_rl_th {
	union {
		struct {
			uint32_t pool_th_dec : 16,
				dummy_real_diff_th : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_th;
static_assert((sizeof(struct reg_pcu_rl_th) == 4), "reg_pcu_rl_th size is not 32-bit");
/*
 PCU_RL_MIN 
 b'PMU RATE LIMITER Min values'
*/
typedef struct reg_pcu_rl_min {
	union {
		struct {
			uint32_t avg_min_to_force_dummy : 16,
				token_min_val : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_min;
static_assert((sizeof(struct reg_pcu_rl_min) == 4), "reg_pcu_rl_min size is not 32-bit");
/*
 PCU_RL_CTRL_EN 
 b'PMU RATE LIMITER enable'
*/
typedef struct reg_pcu_rl_ctrl_en {
	union {
		struct {
			uint32_t pcu_disable : 1,
				min_val_prot_en : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_ctrl_en;
static_assert((sizeof(struct reg_pcu_rl_ctrl_en) == 4), "reg_pcu_rl_ctrl_en size is not 32-bit");
/*
 PCU_RL_HISTORY_LOG_SIZE 
 b'PMU History Log Size'
*/
typedef struct reg_pcu_rl_history_log_size {
	union {
		struct {
			uint32_t all_macs : 3,
				real_macs : 2,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_history_log_size;
static_assert((sizeof(struct reg_pcu_rl_history_log_size) == 4), "reg_pcu_rl_history_log_size size is not 32-bit");
/*
 PCU_DUMMY_A_BF16 
 b'PMU Dummy value for A_BF16'
*/
typedef struct reg_pcu_dummy_a_bf16 {
	union {
		struct {
			uint32_t odd : 16,
				even : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_a_bf16;
static_assert((sizeof(struct reg_pcu_dummy_a_bf16) == 4), "reg_pcu_dummy_a_bf16 size is not 32-bit");
/*
 PCU_DUMMY_B_BF16 
 b'PMU Dummy value for B_BF16'
*/
typedef struct reg_pcu_dummy_b_bf16 {
	union {
		struct {
			uint32_t odd : 16,
				even : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_b_bf16;
static_assert((sizeof(struct reg_pcu_dummy_b_bf16) == 4), "reg_pcu_dummy_b_bf16 size is not 32-bit");
/*
 PCU_DUMMY_A_FP16 
 b'PMU Dummy value for A_FP16'
*/
typedef struct reg_pcu_dummy_a_fp16 {
	union {
		struct {
			uint32_t odd : 16,
				even : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_a_fp16;
static_assert((sizeof(struct reg_pcu_dummy_a_fp16) == 4), "reg_pcu_dummy_a_fp16 size is not 32-bit");
/*
 PCU_DUMMY_B_FP16 
 b'PMU Dummy value for B_FP16'
*/
typedef struct reg_pcu_dummy_b_fp16 {
	union {
		struct {
			uint32_t odd : 16,
				even : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_b_fp16;
static_assert((sizeof(struct reg_pcu_dummy_b_fp16) == 4), "reg_pcu_dummy_b_fp16 size is not 32-bit");
/*
 PCU_DUMMY_F8 
 b'PMU Dummy value for INT8'
*/
typedef struct reg_pcu_dummy_f8 {
	union {
		struct {
			uint32_t a_val_odd : 8,
				a_val_even : 8,
				b_val_odd : 8,
				b_val_even : 8;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_f8;
static_assert((sizeof(struct reg_pcu_dummy_f8) == 4), "reg_pcu_dummy_f8 size is not 32-bit");
/*
 PCU_DUMMY_A_FP32_ODD 
 b'PMU Dummy value for INT4'
*/
typedef struct reg_pcu_dummy_a_fp32_odd {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_a_fp32_odd;
static_assert((sizeof(struct reg_pcu_dummy_a_fp32_odd) == 4), "reg_pcu_dummy_a_fp32_odd size is not 32-bit");
/*
 PCU_DUMMY_A_FP32_EVEN 
*/
typedef struct reg_pcu_dummy_a_fp32_even {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_a_fp32_even;
static_assert((sizeof(struct reg_pcu_dummy_a_fp32_even) == 4), "reg_pcu_dummy_a_fp32_even size is not 32-bit");
/*
 PCU_DUMMY_B_FP32_ODD 
*/
typedef struct reg_pcu_dummy_b_fp32_odd {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_b_fp32_odd;
static_assert((sizeof(struct reg_pcu_dummy_b_fp32_odd) == 4), "reg_pcu_dummy_b_fp32_odd size is not 32-bit");
/*
 PCU_DUMMY_B_FP32_EVEN 
*/
typedef struct reg_pcu_dummy_b_fp32_even {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_b_fp32_even;
static_assert((sizeof(struct reg_pcu_dummy_b_fp32_even) == 4), "reg_pcu_dummy_b_fp32_even size is not 32-bit");
/*
 PCU_DUMMY_A_TF32_ODD 
*/
typedef struct reg_pcu_dummy_a_tf32_odd {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_a_tf32_odd;
static_assert((sizeof(struct reg_pcu_dummy_a_tf32_odd) == 4), "reg_pcu_dummy_a_tf32_odd size is not 32-bit");
/*
 PCU_DUMMY_A_TF32_EVEN 
*/
typedef struct reg_pcu_dummy_a_tf32_even {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_a_tf32_even;
static_assert((sizeof(struct reg_pcu_dummy_a_tf32_even) == 4), "reg_pcu_dummy_a_tf32_even size is not 32-bit");
/*
 PCU_DUMMY_B_TF32_ODD 
*/
typedef struct reg_pcu_dummy_b_tf32_odd {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_b_tf32_odd;
static_assert((sizeof(struct reg_pcu_dummy_b_tf32_odd) == 4), "reg_pcu_dummy_b_tf32_odd size is not 32-bit");
/*
 PCU_DUMMY_B_TF32_EVEN 
*/
typedef struct reg_pcu_dummy_b_tf32_even {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dummy_b_tf32_even;
static_assert((sizeof(struct reg_pcu_dummy_b_tf32_even) == 4), "reg_pcu_dummy_b_tf32_even size is not 32-bit");
/*
 EU 
 b'EU configs'
*/
typedef struct reg_eu {
	union {
		struct {
			uint32_t power_save_disable : 1,
				fp_pyr_close_cgate_en : 1,
				fp_cls_close_cgate_en : 1,
				_reserved8 : 5,
				fp_close_cgate_dly : 12,
				fp_close_cgate_on_desc : 1,
				fp_rollup_cdc_stall_dis : 1,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_eu;
static_assert((sizeof(struct reg_eu) == 4), "reg_eu size is not 32-bit");
/*
 COUT_AGU_SM_INFLIGHT_CNTR 
 b'Sync Object Message Outstanding Counter'
*/
typedef struct reg_cout_agu_sm_inflight_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout_agu_sm_inflight_cntr;
static_assert((sizeof(struct reg_cout_agu_sm_inflight_cntr) == 4), "reg_cout_agu_sm_inflight_cntr size is not 32-bit");
/*
 COUT_AGU_SM_TOTAL_CNTR 
 b'Sync Object Message Total Counter'
*/
typedef struct reg_cout_agu_sm_total_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout_agu_sm_total_cntr;
static_assert((sizeof(struct reg_cout_agu_sm_total_cntr) == 4), "reg_cout_agu_sm_total_cntr size is not 32-bit");
/*
 DMA_AGU_SM_INFLIGHT_CNTR 
 b'Sync Object Message Outstanding Counter'
*/
typedef struct reg_dma_agu_sm_inflight_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dma_agu_sm_inflight_cntr;
static_assert((sizeof(struct reg_dma_agu_sm_inflight_cntr) == 4), "reg_dma_agu_sm_inflight_cntr size is not 32-bit");
/*
 DMA_AGU_SM_TOTAL_CNTR 
 b'Sync Object Message Total Counter'
*/
typedef struct reg_dma_agu_sm_total_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dma_agu_sm_total_cntr;
static_assert((sizeof(struct reg_dma_agu_sm_total_cntr) == 4), "reg_dma_agu_sm_total_cntr size is not 32-bit");
/*
 PCU_RL_SAT_SEC 
 b'PMU RATE LIMITER SATURATION'
*/
typedef struct reg_pcu_rl_sat_sec {
	union {
		struct {
			uint32_t val : 20,
				_reserved31 : 11,
				sel : 1;
		};
		uint32_t _raw;
	};
} reg_pcu_rl_sat_sec;
static_assert((sizeof(struct reg_pcu_rl_sat_sec) == 4), "reg_pcu_rl_sat_sec size is not 32-bit");
/*
 EU0_REDUN_CLK_EN 
 b'FP EU0 REDUNDANCY CLK'
*/
typedef struct reg_eu0_redun_clk_en {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_eu0_redun_clk_en;
static_assert((sizeof(struct reg_eu0_redun_clk_en) == 4), "reg_eu0_redun_clk_en size is not 32-bit");
/*
 EU1_REDUN_CLK_EN 
*/
typedef struct reg_eu1_redun_clk_en {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_eu1_redun_clk_en;
static_assert((sizeof(struct reg_eu1_redun_clk_en) == 4), "reg_eu1_redun_clk_en size is not 32-bit");
/*
 REDUN 
 b'EU Redundancy PE'
*/
typedef struct reg_redun {
	union {
		struct {
			uint32_t eu0 : 6,
				_reserved8 : 2,
				eu1 : 6,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_redun;
static_assert((sizeof(struct reg_redun) == 4), "reg_redun size is not 32-bit");
/*
 EU_ISOLATION_DIS 
 b'EU ISOLATION DISABLE - must be set after reset'
*/
typedef struct reg_eu_isolation_dis {
	union {
		struct {
			uint32_t eu0 : 1,
				eu1 : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_eu_isolation_dis;
static_assert((sizeof(struct reg_eu_isolation_dis) == 4), "reg_eu_isolation_dis size is not 32-bit");
/*
 QM 
 b'QMAN chickens'
*/
typedef struct reg_qm {
	union {
		struct {
			uint32_t stop_on_sb_err : 1,
				stop_on_wap_axi_err : 1,
				stop_on_arc_axi_err : 1,
				stop_on_sig_axi_err : 1,
				stop_on_arc_serr : 1,
				stop_on_arc_derr : 1,
				stop_on_acc_serr : 1,
				stop_on_acc_derr : 1,
				stop_on_sb_serr : 1,
				stop_on_sb_derr : 1,
				stop_on_numeric_err : 1,
				dbg_halt_en : 1,
				ext_addr_err_en : 1,
				fence_on_cout_sm_inflight : 1,
				fence_on_dma_sm_inflight : 1,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_qm;
static_assert((sizeof(struct reg_qm) == 4), "reg_qm size is not 32-bit");
/*
 EARLY_RELEASE_STATUS 
 b'EARLY RELEASE STATUS register'
*/
typedef struct reg_early_release_status {
	union {
		struct {
			uint32_t agu_cout_master : 4,
				agu_cout_slave : 4,
				ap_brain_master : 4,
				ap_brain_slave : 4,
				eu_brain_master : 4,
				eu_brain_slave : 4,
				dma_cout_master : 4,
				dma_cout_slave : 4;
		};
		uint32_t _raw;
	};
} reg_early_release_status;
static_assert((sizeof(struct reg_early_release_status) == 4), "reg_early_release_status size is not 32-bit");
/*
 INTR_CAUSE 
 b'CTRL Interrupts Cause Register'
*/
typedef struct reg_intr_cause {
	union {
		struct {
			uint32_t v : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_intr_cause;
static_assert((sizeof(struct reg_intr_cause) == 4), "reg_intr_cause size is not 32-bit");
/*
 INTR_MASK 
 b'CTRL Interrupts Mask Register'
*/
typedef struct reg_intr_mask {
	union {
		struct {
			uint32_t v : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_intr_mask;
static_assert((sizeof(struct reg_intr_mask) == 4), "reg_intr_mask size is not 32-bit");
/*
 INTR_CLEAR 
*/
typedef struct reg_intr_clear {
	union {
		struct {
			uint32_t v : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_intr_clear;
static_assert((sizeof(struct reg_intr_clear) == 4), "reg_intr_clear size is not 32-bit");
/*
 REDUN_PSOC_SEL_SEC 
*/
typedef struct reg_redun_psoc_sel_sec {
	union {
		struct {
			uint32_t eu0 : 1,
				eu1 : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_redun_psoc_sel_sec;
static_assert((sizeof(struct reg_redun_psoc_sel_sec) == 4), "reg_redun_psoc_sel_sec size is not 32-bit");
/*
 BIST 
 b'MME_BIST'
*/
typedef struct reg_bist {
	union {
		struct {
			uint32_t func_mode : 1,
				apb_sw_mode : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_bist;
static_assert((sizeof(struct reg_bist) == 4), "reg_bist size is not 32-bit");
/*
 EU_RL_ENABLE 
*/
typedef struct reg_eu_rl_enable {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_eu_rl_enable;
static_assert((sizeof(struct reg_eu_rl_enable) == 4), "reg_eu_rl_enable size is not 32-bit");
/*
 EU_RL_TOKEN_SEL 
*/
typedef struct reg_eu_rl_token_sel {
	union {
		struct {
			uint32_t stat : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_eu_rl_token_sel;
static_assert((sizeof(struct reg_eu_rl_token_sel) == 4), "reg_eu_rl_token_sel size is not 32-bit");
/*
 EU_RL_CFG 
*/
typedef struct reg_eu_rl_cfg {
	union {
		struct {
			uint32_t rst_token : 8,
				timeout : 8,
				saturation : 8,
				data_size : 8;
		};
		uint32_t _raw;
	};
} reg_eu_rl_cfg;
static_assert((sizeof(struct reg_eu_rl_cfg) == 4), "reg_eu_rl_cfg size is not 32-bit");
/*
 PCU_DBG_MASTER_DW0 
*/
typedef struct reg_pcu_dbg_master_dw0 {
	union {
		struct {
			uint32_t fsm_state : 1,
				_reserved8 : 7,
				real_pool_tokens : 20,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_master_dw0;
static_assert((sizeof(struct reg_pcu_dbg_master_dw0) == 4), "reg_pcu_dbg_master_dw0 size is not 32-bit");
/*
 PCU_DBG_MASTER_DW1 
*/
typedef struct reg_pcu_dbg_master_dw1 {
	union {
		struct {
			uint32_t all_pool_tokens : 20,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_master_dw1;
static_assert((sizeof(struct reg_pcu_dbg_master_dw1) == 4), "reg_pcu_dbg_master_dw1 size is not 32-bit");
/*
 PCU_DBG_MASTER_DW2 
*/
typedef struct reg_pcu_dbg_master_dw2 {
	union {
		struct {
			uint32_t bubble_cyc_cntr : 16,
				dummy_cyc_cntr : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_master_dw2;
static_assert((sizeof(struct reg_pcu_dbg_master_dw2) == 4), "reg_pcu_dbg_master_dw2 size is not 32-bit");
/*
 PCU_DBG_MASTER_DW3 
*/
typedef struct reg_pcu_dbg_master_dw3 {
	union {
		struct {
			uint32_t real_macs_history : 16,
				all_macs_history : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_master_dw3;
static_assert((sizeof(struct reg_pcu_dbg_master_dw3) == 4), "reg_pcu_dbg_master_dw3 size is not 32-bit");
/*
 PCU_DBG_MASTER_WKL_ID 
*/
typedef struct reg_pcu_dbg_master_wkl_id {
	union {
		struct {
			uint32_t b : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_master_wkl_id;
static_assert((sizeof(struct reg_pcu_dbg_master_wkl_id) == 4), "reg_pcu_dbg_master_wkl_id size is not 32-bit");
/*
 PCU_DBG_SLAVE_DW0 
*/
typedef struct reg_pcu_dbg_slave_dw0 {
	union {
		struct {
			uint32_t fsm_state : 1,
				_reserved8 : 7,
				real_pool_tokens : 20,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_slave_dw0;
static_assert((sizeof(struct reg_pcu_dbg_slave_dw0) == 4), "reg_pcu_dbg_slave_dw0 size is not 32-bit");
/*
 PCU_DBG_SLAVE_DW1 
*/
typedef struct reg_pcu_dbg_slave_dw1 {
	union {
		struct {
			uint32_t all_pool_tokens : 20,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_slave_dw1;
static_assert((sizeof(struct reg_pcu_dbg_slave_dw1) == 4), "reg_pcu_dbg_slave_dw1 size is not 32-bit");
/*
 PCU_DBG_SLAVE_DW2 
*/
typedef struct reg_pcu_dbg_slave_dw2 {
	union {
		struct {
			uint32_t bubble_cyc_cntr : 16,
				dummy_cyc_cntr : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_slave_dw2;
static_assert((sizeof(struct reg_pcu_dbg_slave_dw2) == 4), "reg_pcu_dbg_slave_dw2 size is not 32-bit");
/*
 PCU_DBG_SLAVE_DW3 
*/
typedef struct reg_pcu_dbg_slave_dw3 {
	union {
		struct {
			uint32_t real_macs_history : 16,
				all_macs_history : 16;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_slave_dw3;
static_assert((sizeof(struct reg_pcu_dbg_slave_dw3) == 4), "reg_pcu_dbg_slave_dw3 size is not 32-bit");
/*
 PCU_DBG_SLAVE_WKL_ID 
*/
typedef struct reg_pcu_dbg_slave_wkl_id {
	union {
		struct {
			uint32_t b : 32;
		};
		uint32_t _raw;
	};
} reg_pcu_dbg_slave_wkl_id;
static_assert((sizeof(struct reg_pcu_dbg_slave_wkl_id) == 4), "reg_pcu_dbg_slave_wkl_id size is not 32-bit");
/*
 ETF_MEM_WRAP_RM 
*/
typedef struct reg_etf_mem_wrap_rm {
	union {
		struct {
			uint32_t v : 30,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_etf_mem_wrap_rm;
static_assert((sizeof(struct reg_etf_mem_wrap_rm) == 4), "reg_etf_mem_wrap_rm size is not 32-bit");
/*
 SLV_AXI2APB 
*/
typedef struct reg_slv_axi2apb {
	union {
		struct {
			uint32_t msk_error_rsp : 1,
				block_partial_strb : 1,
				spl_error_rsp : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_slv_axi2apb;
static_assert((sizeof(struct reg_slv_axi2apb) == 4), "reg_slv_axi2apb size is not 32-bit");
/*
 SIG_STALL 
 b'force stop on LBW Sync Msg'
*/
typedef struct reg_sig_stall {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sig_stall;
static_assert((sizeof(struct reg_sig_stall) == 4), "reg_sig_stall size is not 32-bit");
/*
 SIG 
*/
typedef struct reg_sig {
	union {
		struct {
			uint32_t stop_on_sb_err : 1,
				stop_on_wap_axi_err : 1,
				stop_on_arc_axi_err : 1,
				stop_on_sig_axi_err : 1,
				stop_on_arc_serr : 1,
				stop_on_arc_derr : 1,
				stop_on_acc_serr : 1,
				stop_on_acc_derr : 1,
				stop_on_sb_serr : 1,
				stop_on_sb_derr : 1,
				stop_on_numeric_err : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_sig;
static_assert((sizeof(struct reg_sig) == 4), "reg_sig size is not 32-bit");
/*
 LOG_ARCH_MASK 
*/
typedef struct reg_log_arch_mask {
	union {
		struct {
			uint32_t master : 8,
				slave : 8,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_log_arch_mask;
static_assert((sizeof(struct reg_log_arch_mask) == 4), "reg_log_arch_mask size is not 32-bit");
/*
 AGU_ERR_WKL_ID 
*/
typedef struct reg_agu_err_wkl_id {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_agu_err_wkl_id;
static_assert((sizeof(struct reg_agu_err_wkl_id) == 4), "reg_agu_err_wkl_id size is not 32-bit");
/*
 SBTE0_HBW_ADDR_SPLIT0 
*/
typedef struct reg_sbte0_hbw_addr_split0 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte0_hbw_addr_split0;
static_assert((sizeof(struct reg_sbte0_hbw_addr_split0) == 4), "reg_sbte0_hbw_addr_split0 size is not 32-bit");
/*
 SBTE0_HBW_ADDR_SPLIT1 
*/
typedef struct reg_sbte0_hbw_addr_split1 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte0_hbw_addr_split1;
static_assert((sizeof(struct reg_sbte0_hbw_addr_split1) == 4), "reg_sbte0_hbw_addr_split1 size is not 32-bit");
/*
 SBTE0_HBW_ADDR_SPLIT2 
*/
typedef struct reg_sbte0_hbw_addr_split2 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte0_hbw_addr_split2;
static_assert((sizeof(struct reg_sbte0_hbw_addr_split2) == 4), "reg_sbte0_hbw_addr_split2 size is not 32-bit");
/*
 SBTE0_HBW_ADDR_SPLIT3 
*/
typedef struct reg_sbte0_hbw_addr_split3 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte0_hbw_addr_split3;
static_assert((sizeof(struct reg_sbte0_hbw_addr_split3) == 4), "reg_sbte0_hbw_addr_split3 size is not 32-bit");
/*
 SBTE1_HBW_ADDR_SPLIT0 
*/
typedef struct reg_sbte1_hbw_addr_split0 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte1_hbw_addr_split0;
static_assert((sizeof(struct reg_sbte1_hbw_addr_split0) == 4), "reg_sbte1_hbw_addr_split0 size is not 32-bit");
/*
 SBTE1_HBW_ADDR_SPLIT1 
*/
typedef struct reg_sbte1_hbw_addr_split1 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte1_hbw_addr_split1;
static_assert((sizeof(struct reg_sbte1_hbw_addr_split1) == 4), "reg_sbte1_hbw_addr_split1 size is not 32-bit");
/*
 SBTE1_HBW_ADDR_SPLIT2 
*/
typedef struct reg_sbte1_hbw_addr_split2 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte1_hbw_addr_split2;
static_assert((sizeof(struct reg_sbte1_hbw_addr_split2) == 4), "reg_sbte1_hbw_addr_split2 size is not 32-bit");
/*
 SBTE1_HBW_ADDR_SPLIT3 
*/
typedef struct reg_sbte1_hbw_addr_split3 {
	union {
		struct {
			uint32_t v : 17,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sbte1_hbw_addr_split3;
static_assert((sizeof(struct reg_sbte1_hbw_addr_split3) == 4), "reg_sbte1_hbw_addr_split3 size is not 32-bit");
/*
 SBTE0_DBG_ADDR_SPLIT0 
*/
typedef struct reg_sbte0_dbg_addr_split0 {
	union {
		struct {
			uint32_t v : 20,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_sbte0_dbg_addr_split0;
static_assert((sizeof(struct reg_sbte0_dbg_addr_split0) == 4), "reg_sbte0_dbg_addr_split0 size is not 32-bit");
/*
 SBTE0_DBG_ADDR_SPLIT1 
*/
typedef struct reg_sbte0_dbg_addr_split1 {
	union {
		struct {
			uint32_t v : 20,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_sbte0_dbg_addr_split1;
static_assert((sizeof(struct reg_sbte0_dbg_addr_split1) == 4), "reg_sbte0_dbg_addr_split1 size is not 32-bit");
/*
 SBTE1_DBG_ADDR_SPLIT0 
*/
typedef struct reg_sbte1_dbg_addr_split0 {
	union {
		struct {
			uint32_t v : 20,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_sbte1_dbg_addr_split0;
static_assert((sizeof(struct reg_sbte1_dbg_addr_split0) == 4), "reg_sbte1_dbg_addr_split0 size is not 32-bit");
/*
 SBTE1_DBG_ADDR_SPLIT1 
*/
typedef struct reg_sbte1_dbg_addr_split1 {
	union {
		struct {
			uint32_t v : 20,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_sbte1_dbg_addr_split1;
static_assert((sizeof(struct reg_sbte1_dbg_addr_split1) == 4), "reg_sbte1_dbg_addr_split1 size is not 32-bit");
/*
 HBW_CLK_ENABLER_DIS 
 b'Disable AXI/APB Clock Enablers'
*/
typedef struct reg_hbw_clk_enabler_dis {
	union {
		struct {
			uint32_t axi : 1,
				apb : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_hbw_clk_enabler_dis;
static_assert((sizeof(struct reg_hbw_clk_enabler_dis) == 4), "reg_hbw_clk_enabler_dis size is not 32-bit");
/*
 MISC 
 b'AGU Configurations'
*/
typedef struct reg_misc {
	union {
		struct {
			uint32_t acc_id_lsb_mask_dis : 1,
				_reserved3 : 2,
				eu_a_clip_inf_input : 1,
				eu_b_clip_inf_input : 1,
				_reserved8 : 3,
				sb_tr_mode_h_min_a : 7,
				_reserved16 : 1,
				sb_tr_mode_h_min_b : 7,
				_reserved24 : 1,
				cmd_trig_soc_sel : 1,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_misc;
static_assert((sizeof(struct reg_misc) == 4), "reg_misc size is not 32-bit");
/*
 SO_AWPROT_BIT0 
 b'CTRL Protection'
*/
typedef struct reg_so_awprot_bit0 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_so_awprot_bit0;
static_assert((sizeof(struct reg_so_awprot_bit0) == 4), "reg_so_awprot_bit0 size is not 32-bit");
/*
 SO_AWPROT_BIT1 
*/
typedef struct reg_so_awprot_bit1 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_so_awprot_bit1;
static_assert((sizeof(struct reg_so_awprot_bit1) == 4), "reg_so_awprot_bit1 size is not 32-bit");
/*
 SO_AWPROT_BIT2 
*/
typedef struct reg_so_awprot_bit2 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_so_awprot_bit2;
static_assert((sizeof(struct reg_so_awprot_bit2) == 4), "reg_so_awprot_bit2 size is not 32-bit");
/*
 DMA_SO_AWPROT_BIT0 
*/
typedef struct reg_dma_so_awprot_bit0 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_dma_so_awprot_bit0;
static_assert((sizeof(struct reg_dma_so_awprot_bit0) == 4), "reg_dma_so_awprot_bit0 size is not 32-bit");
/*
 DMA_SO_AWPROT_BIT1 
*/
typedef struct reg_dma_so_awprot_bit1 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_dma_so_awprot_bit1;
static_assert((sizeof(struct reg_dma_so_awprot_bit1) == 4), "reg_dma_so_awprot_bit1 size is not 32-bit");
/*
 DMA_SO_AWPROT_BIT2 
*/
typedef struct reg_dma_so_awprot_bit2 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_dma_so_awprot_bit2;
static_assert((sizeof(struct reg_dma_so_awprot_bit2) == 4), "reg_dma_so_awprot_bit2 size is not 32-bit");
/*
 ARCH_DMA_A_SS 
 b'SW Descriptor AGU_A Spatial Size'
*/
typedef struct reg_arch_dma_a_ss {
	union {
		struct {
			uint32_t minus_1 : 32;
		};
		uint32_t _raw;
	};
} reg_arch_dma_a_ss;
static_assert((sizeof(struct reg_arch_dma_a_ss) == 4), "reg_arch_dma_a_ss size is not 32-bit");
/*
 ARCH_DMA_B_SS 
 b'SW Descriptor MASTER AGU_B Spatial Size'
*/
typedef struct reg_arch_dma_b_ss {
	union {
		struct {
			uint32_t minus_1 : 32;
		};
		uint32_t _raw;
	};
} reg_arch_dma_b_ss;
static_assert((sizeof(struct reg_arch_dma_b_ss) == 4), "reg_arch_dma_b_ss size is not 32-bit");
/*
 ARCH_DMA_COUT_SS 
 b'SW Descriptor AGU_COUT Spatial Size'
*/
typedef struct reg_arch_dma_cout_ss {
	union {
		struct {
			uint32_t minus_1 : 32;
		};
		uint32_t _raw;
	};
} reg_arch_dma_cout_ss;
static_assert((sizeof(struct reg_arch_dma_cout_ss) == 4), "reg_arch_dma_cout_ss size is not 32-bit");

#ifdef __cplusplus
} /* mme_ctrl_lo namespace */
#endif

/*
 MME_CTRL_LO block
*/

#ifdef __cplusplus

struct block_mme_ctrl_lo {
	struct mme_ctrl_lo::reg_arch_status_master arch_status_master;
	struct mme_ctrl_lo::reg_arch_status_slave arch_status_slave;
	struct block_mme_address_descriptor arch_base_addr;
	struct block_mme_non_tensor_descriptor_start arch_n_ten_st;
	struct block_mme_tensor arch_ten_a;
	struct block_mme_tensor arch_ten_b;
	struct block_mme_tensor arch_ten_cout;
	struct block_mme_agu_core arch_agu_in0_master;
	struct block_mme_agu_core arch_agu_in1_master;
	struct block_mme_agu_core arch_agu_in2_master;
	struct block_mme_agu_core arch_agu_in3_master;
	struct block_mme_agu_core arch_agu_in0_slave;
	struct block_mme_agu_core arch_agu_in1_slave;
	struct block_mme_agu_core arch_agu_in2_slave;
	struct block_mme_agu_core arch_agu_in3_slave;
	struct mme_ctrl_lo::reg_arch_a_ss arch_a_ss;
	struct mme_ctrl_lo::reg_arch_b_ss arch_b_ss;
	struct block_mme_agu_core arch_agu_cout0_master;
	struct block_mme_agu_core arch_agu_cout0_slave;
	struct mme_ctrl_lo::reg_arch_cout_ss arch_cout_ss;
	struct block_mme_non_tensor_descriptor arch_n_ten;
	uint32_t _pad664[26];
	struct mme_ctrl_lo::reg_cmd cmd;
	uint32_t _pad772[63];
	struct mme_ctrl_lo::reg_qm_stall qm_stall;
	struct mme_ctrl_lo::reg_log_shadow_lo log_shadow_lo;
	struct mme_ctrl_lo::reg_log_shadow_hi log_shadow_hi;
	struct mme_ctrl_lo::reg_sync_object_fifo_th sync_object_fifo_th;
	struct mme_ctrl_lo::reg_eus_rollup_dly_dw0 eus_rollup_dly_dw0;
	struct mme_ctrl_lo::reg_eus_rollup_dly_dw1 eus_rollup_dly_dw1;
	struct mme_ctrl_lo::reg_eus_rollup_cd_prot_f16 eus_rollup_cd_prot_f16;
	struct mme_ctrl_lo::reg_eus_rollup_cd_prot_f8 eus_rollup_cd_prot_f8;
	struct mme_ctrl_lo::reg_eus_rollup_cd_prot_fp32 eus_rollup_cd_prot_fp32;
	struct mme_ctrl_lo::reg_eus_rollup_cd_prot_tf32 eus_rollup_cd_prot_tf32;
	struct mme_ctrl_lo::reg_pcu_rl_desc0 pcu_rl_desc0;
	struct mme_ctrl_lo::reg_pcu_rl_token_update pcu_rl_token_update;
	struct mme_ctrl_lo::reg_pcu_rl_th pcu_rl_th;
	struct mme_ctrl_lo::reg_pcu_rl_min pcu_rl_min;
	struct mme_ctrl_lo::reg_pcu_rl_ctrl_en pcu_rl_ctrl_en;
	struct mme_ctrl_lo::reg_pcu_rl_history_log_size pcu_rl_history_log_size;
	struct mme_ctrl_lo::reg_pcu_dummy_a_bf16 pcu_dummy_a_bf16;
	struct mme_ctrl_lo::reg_pcu_dummy_b_bf16 pcu_dummy_b_bf16;
	struct mme_ctrl_lo::reg_pcu_dummy_a_fp16 pcu_dummy_a_fp16;
	struct mme_ctrl_lo::reg_pcu_dummy_b_fp16 pcu_dummy_b_fp16;
	struct mme_ctrl_lo::reg_pcu_dummy_f8 pcu_dummy_f8;
	struct mme_ctrl_lo::reg_pcu_dummy_a_fp32_odd pcu_dummy_a_fp32_odd;
	struct mme_ctrl_lo::reg_pcu_dummy_a_fp32_even pcu_dummy_a_fp32_even;
	struct mme_ctrl_lo::reg_pcu_dummy_b_fp32_odd pcu_dummy_b_fp32_odd;
	struct mme_ctrl_lo::reg_pcu_dummy_b_fp32_even pcu_dummy_b_fp32_even;
	struct mme_ctrl_lo::reg_pcu_dummy_a_tf32_odd pcu_dummy_a_tf32_odd;
	struct mme_ctrl_lo::reg_pcu_dummy_a_tf32_even pcu_dummy_a_tf32_even;
	struct mme_ctrl_lo::reg_pcu_dummy_b_tf32_odd pcu_dummy_b_tf32_odd;
	struct mme_ctrl_lo::reg_pcu_dummy_b_tf32_even pcu_dummy_b_tf32_even;
	uint32_t _pad1140[1];
	struct mme_ctrl_lo::reg_eu eu;
	struct mme_ctrl_lo::reg_cout_agu_sm_inflight_cntr cout_agu_sm_inflight_cntr;
	struct mme_ctrl_lo::reg_cout_agu_sm_total_cntr cout_agu_sm_total_cntr;
	struct mme_ctrl_lo::reg_dma_agu_sm_inflight_cntr dma_agu_sm_inflight_cntr;
	struct mme_ctrl_lo::reg_dma_agu_sm_total_cntr dma_agu_sm_total_cntr;
	struct mme_ctrl_lo::reg_pcu_rl_sat_sec pcu_rl_sat_sec;
	struct mme_ctrl_lo::reg_eu0_redun_clk_en eu0_redun_clk_en;
	struct mme_ctrl_lo::reg_eu1_redun_clk_en eu1_redun_clk_en;
	struct mme_ctrl_lo::reg_redun redun;
	struct mme_ctrl_lo::reg_eu_isolation_dis eu_isolation_dis;
	struct mme_ctrl_lo::reg_qm qm;
	struct mme_ctrl_lo::reg_early_release_status early_release_status;
	struct mme_ctrl_lo::reg_intr_cause intr_cause;
	struct mme_ctrl_lo::reg_intr_mask intr_mask;
	struct mme_ctrl_lo::reg_intr_clear intr_clear;
	struct mme_ctrl_lo::reg_redun_psoc_sel_sec redun_psoc_sel_sec;
	struct mme_ctrl_lo::reg_bist bist;
	struct mme_ctrl_lo::reg_eu_rl_enable eu_rl_enable;
	struct mme_ctrl_lo::reg_eu_rl_token_sel eu_rl_token_sel;
	struct mme_ctrl_lo::reg_eu_rl_cfg eu_rl_cfg;
	struct mme_ctrl_lo::reg_pcu_dbg_master_dw0 pcu_dbg_master_dw0;
	struct mme_ctrl_lo::reg_pcu_dbg_master_dw1 pcu_dbg_master_dw1;
	struct mme_ctrl_lo::reg_pcu_dbg_master_dw2 pcu_dbg_master_dw2;
	struct mme_ctrl_lo::reg_pcu_dbg_master_dw3 pcu_dbg_master_dw3;
	struct mme_ctrl_lo::reg_pcu_dbg_master_wkl_id pcu_dbg_master_wkl_id;
	struct mme_ctrl_lo::reg_pcu_dbg_slave_dw0 pcu_dbg_slave_dw0;
	struct mme_ctrl_lo::reg_pcu_dbg_slave_dw1 pcu_dbg_slave_dw1;
	struct mme_ctrl_lo::reg_pcu_dbg_slave_dw2 pcu_dbg_slave_dw2;
	struct mme_ctrl_lo::reg_pcu_dbg_slave_dw3 pcu_dbg_slave_dw3;
	struct mme_ctrl_lo::reg_pcu_dbg_slave_wkl_id pcu_dbg_slave_wkl_id;
	struct mme_ctrl_lo::reg_etf_mem_wrap_rm etf_mem_wrap_rm;
	struct mme_ctrl_lo::reg_slv_axi2apb slv_axi2apb;
	struct mme_ctrl_lo::reg_sig_stall sig_stall;
	struct mme_ctrl_lo::reg_sig sig;
	struct mme_ctrl_lo::reg_log_arch_mask log_arch_mask;
	struct mme_ctrl_lo::reg_agu_err_wkl_id agu_err_wkl_id;
	struct mme_ctrl_lo::reg_sbte0_hbw_addr_split0 sbte0_hbw_addr_split0;
	struct mme_ctrl_lo::reg_sbte0_hbw_addr_split1 sbte0_hbw_addr_split1;
	struct mme_ctrl_lo::reg_sbte0_hbw_addr_split2 sbte0_hbw_addr_split2;
	struct mme_ctrl_lo::reg_sbte0_hbw_addr_split3 sbte0_hbw_addr_split3;
	struct mme_ctrl_lo::reg_sbte1_hbw_addr_split0 sbte1_hbw_addr_split0;
	struct mme_ctrl_lo::reg_sbte1_hbw_addr_split1 sbte1_hbw_addr_split1;
	struct mme_ctrl_lo::reg_sbte1_hbw_addr_split2 sbte1_hbw_addr_split2;
	struct mme_ctrl_lo::reg_sbte1_hbw_addr_split3 sbte1_hbw_addr_split3;
	struct mme_ctrl_lo::reg_sbte0_dbg_addr_split0 sbte0_dbg_addr_split0;
	struct mme_ctrl_lo::reg_sbte0_dbg_addr_split1 sbte0_dbg_addr_split1;
	struct mme_ctrl_lo::reg_sbte1_dbg_addr_split0 sbte1_dbg_addr_split0;
	struct mme_ctrl_lo::reg_sbte1_dbg_addr_split1 sbte1_dbg_addr_split1;
	struct mme_ctrl_lo::reg_hbw_clk_enabler_dis hbw_clk_enabler_dis;
	struct mme_ctrl_lo::reg_misc misc;
	struct mme_ctrl_lo::reg_so_awprot_bit0 so_awprot_bit0;
	struct mme_ctrl_lo::reg_so_awprot_bit1 so_awprot_bit1;
	struct mme_ctrl_lo::reg_so_awprot_bit2 so_awprot_bit2;
	struct mme_ctrl_lo::reg_dma_so_awprot_bit0 dma_so_awprot_bit0;
	struct mme_ctrl_lo::reg_dma_so_awprot_bit1 dma_so_awprot_bit1;
	struct mme_ctrl_lo::reg_dma_so_awprot_bit2 dma_so_awprot_bit2;
	uint32_t _pad1368[2];
	struct block_axuser_lbw axuser_lbw;
	uint32_t _pad1400[164];
	struct block_mme_address_descriptor arch_dma_base_addr;
	struct block_mme_non_tensor_descriptor_start arch_dma_n_ten_st;
	struct block_mme_tensor arch_dma_ten_a;
	struct block_mme_tensor arch_dma_ten_b;
	struct block_mme_tensor arch_dma_ten_cout;
	struct block_mme_agu_core arch_dma_agu_in0_master;
	struct block_mme_agu_core arch_dma_agu_in1_master;
	struct block_mme_agu_core arch_dma_agu_in2_master;
	struct block_mme_agu_core arch_dma_agu_in3_master;
	struct block_mme_agu_core arch_dma_agu_in0_slave;
	struct block_mme_agu_core arch_dma_agu_in1_slave;
	struct block_mme_agu_core arch_dma_agu_in2_slave;
	struct block_mme_agu_core arch_dma_agu_in3_slave;
	struct mme_ctrl_lo::reg_arch_dma_a_ss arch_dma_a_ss;
	struct mme_ctrl_lo::reg_arch_dma_b_ss arch_dma_b_ss;
	struct block_mme_agu_core arch_dma_agu_cout0_master;
	struct block_mme_agu_core arch_dma_agu_cout0_slave;
	struct mme_ctrl_lo::reg_arch_dma_cout_ss arch_dma_cout_ss;
	struct block_mme_non_tensor_descriptor arch_dma_n_ten;
	uint32_t _pad2712[250];
	struct block_special_regs special;
};
#else

typedef struct block_mme_ctrl_lo {
	reg_arch_status_master arch_status_master;
	reg_arch_status_slave arch_status_slave;
	block_mme_address_descriptor arch_base_addr;
	block_mme_non_tensor_descriptor_start arch_n_ten_st;
	block_mme_tensor arch_ten_a;
	block_mme_tensor arch_ten_b;
	block_mme_tensor arch_ten_cout;
	block_mme_agu_core arch_agu_in0_master;
	block_mme_agu_core arch_agu_in1_master;
	block_mme_agu_core arch_agu_in2_master;
	block_mme_agu_core arch_agu_in3_master;
	block_mme_agu_core arch_agu_in0_slave;
	block_mme_agu_core arch_agu_in1_slave;
	block_mme_agu_core arch_agu_in2_slave;
	block_mme_agu_core arch_agu_in3_slave;
	reg_arch_a_ss arch_a_ss;
	reg_arch_b_ss arch_b_ss;
	block_mme_agu_core arch_agu_cout0_master;
	block_mme_agu_core arch_agu_cout0_slave;
	reg_arch_cout_ss arch_cout_ss;
	block_mme_non_tensor_descriptor arch_n_ten;
	uint32_t _pad664[26];
	reg_cmd cmd;
	uint32_t _pad772[63];
	reg_qm_stall qm_stall;
	reg_log_shadow_lo log_shadow_lo;
	reg_log_shadow_hi log_shadow_hi;
	reg_sync_object_fifo_th sync_object_fifo_th;
	reg_eus_rollup_dly_dw0 eus_rollup_dly_dw0;
	reg_eus_rollup_dly_dw1 eus_rollup_dly_dw1;
	reg_eus_rollup_cd_prot_f16 eus_rollup_cd_prot_f16;
	reg_eus_rollup_cd_prot_f8 eus_rollup_cd_prot_f8;
	reg_eus_rollup_cd_prot_fp32 eus_rollup_cd_prot_fp32;
	reg_eus_rollup_cd_prot_tf32 eus_rollup_cd_prot_tf32;
	reg_pcu_rl_desc0 pcu_rl_desc0;
	reg_pcu_rl_token_update pcu_rl_token_update;
	reg_pcu_rl_th pcu_rl_th;
	reg_pcu_rl_min pcu_rl_min;
	reg_pcu_rl_ctrl_en pcu_rl_ctrl_en;
	reg_pcu_rl_history_log_size pcu_rl_history_log_size;
	reg_pcu_dummy_a_bf16 pcu_dummy_a_bf16;
	reg_pcu_dummy_b_bf16 pcu_dummy_b_bf16;
	reg_pcu_dummy_a_fp16 pcu_dummy_a_fp16;
	reg_pcu_dummy_b_fp16 pcu_dummy_b_fp16;
	reg_pcu_dummy_f8 pcu_dummy_f8;
	reg_pcu_dummy_a_fp32_odd pcu_dummy_a_fp32_odd;
	reg_pcu_dummy_a_fp32_even pcu_dummy_a_fp32_even;
	reg_pcu_dummy_b_fp32_odd pcu_dummy_b_fp32_odd;
	reg_pcu_dummy_b_fp32_even pcu_dummy_b_fp32_even;
	reg_pcu_dummy_a_tf32_odd pcu_dummy_a_tf32_odd;
	reg_pcu_dummy_a_tf32_even pcu_dummy_a_tf32_even;
	reg_pcu_dummy_b_tf32_odd pcu_dummy_b_tf32_odd;
	reg_pcu_dummy_b_tf32_even pcu_dummy_b_tf32_even;
	uint32_t _pad1140[1];
	reg_eu eu;
	reg_cout_agu_sm_inflight_cntr cout_agu_sm_inflight_cntr;
	reg_cout_agu_sm_total_cntr cout_agu_sm_total_cntr;
	reg_dma_agu_sm_inflight_cntr dma_agu_sm_inflight_cntr;
	reg_dma_agu_sm_total_cntr dma_agu_sm_total_cntr;
	reg_pcu_rl_sat_sec pcu_rl_sat_sec;
	reg_eu0_redun_clk_en eu0_redun_clk_en;
	reg_eu1_redun_clk_en eu1_redun_clk_en;
	reg_redun redun;
	reg_eu_isolation_dis eu_isolation_dis;
	reg_qm qm;
	reg_early_release_status early_release_status;
	reg_intr_cause intr_cause;
	reg_intr_mask intr_mask;
	reg_intr_clear intr_clear;
	reg_redun_psoc_sel_sec redun_psoc_sel_sec;
	reg_bist bist;
	reg_eu_rl_enable eu_rl_enable;
	reg_eu_rl_token_sel eu_rl_token_sel;
	reg_eu_rl_cfg eu_rl_cfg;
	reg_pcu_dbg_master_dw0 pcu_dbg_master_dw0;
	reg_pcu_dbg_master_dw1 pcu_dbg_master_dw1;
	reg_pcu_dbg_master_dw2 pcu_dbg_master_dw2;
	reg_pcu_dbg_master_dw3 pcu_dbg_master_dw3;
	reg_pcu_dbg_master_wkl_id pcu_dbg_master_wkl_id;
	reg_pcu_dbg_slave_dw0 pcu_dbg_slave_dw0;
	reg_pcu_dbg_slave_dw1 pcu_dbg_slave_dw1;
	reg_pcu_dbg_slave_dw2 pcu_dbg_slave_dw2;
	reg_pcu_dbg_slave_dw3 pcu_dbg_slave_dw3;
	reg_pcu_dbg_slave_wkl_id pcu_dbg_slave_wkl_id;
	reg_etf_mem_wrap_rm etf_mem_wrap_rm;
	reg_slv_axi2apb slv_axi2apb;
	reg_sig_stall sig_stall;
	reg_sig sig;
	reg_log_arch_mask log_arch_mask;
	reg_agu_err_wkl_id agu_err_wkl_id;
	reg_sbte0_hbw_addr_split0 sbte0_hbw_addr_split0;
	reg_sbte0_hbw_addr_split1 sbte0_hbw_addr_split1;
	reg_sbte0_hbw_addr_split2 sbte0_hbw_addr_split2;
	reg_sbte0_hbw_addr_split3 sbte0_hbw_addr_split3;
	reg_sbte1_hbw_addr_split0 sbte1_hbw_addr_split0;
	reg_sbte1_hbw_addr_split1 sbte1_hbw_addr_split1;
	reg_sbte1_hbw_addr_split2 sbte1_hbw_addr_split2;
	reg_sbte1_hbw_addr_split3 sbte1_hbw_addr_split3;
	reg_sbte0_dbg_addr_split0 sbte0_dbg_addr_split0;
	reg_sbte0_dbg_addr_split1 sbte0_dbg_addr_split1;
	reg_sbte1_dbg_addr_split0 sbte1_dbg_addr_split0;
	reg_sbte1_dbg_addr_split1 sbte1_dbg_addr_split1;
	reg_hbw_clk_enabler_dis hbw_clk_enabler_dis;
	reg_misc misc;
	reg_so_awprot_bit0 so_awprot_bit0;
	reg_so_awprot_bit1 so_awprot_bit1;
	reg_so_awprot_bit2 so_awprot_bit2;
	reg_dma_so_awprot_bit0 dma_so_awprot_bit0;
	reg_dma_so_awprot_bit1 dma_so_awprot_bit1;
	reg_dma_so_awprot_bit2 dma_so_awprot_bit2;
	uint32_t _pad1368[2];
	block_axuser_lbw axuser_lbw;
	uint32_t _pad1400[164];
	block_mme_address_descriptor arch_dma_base_addr;
	block_mme_non_tensor_descriptor_start arch_dma_n_ten_st;
	block_mme_tensor arch_dma_ten_a;
	block_mme_tensor arch_dma_ten_b;
	block_mme_tensor arch_dma_ten_cout;
	block_mme_agu_core arch_dma_agu_in0_master;
	block_mme_agu_core arch_dma_agu_in1_master;
	block_mme_agu_core arch_dma_agu_in2_master;
	block_mme_agu_core arch_dma_agu_in3_master;
	block_mme_agu_core arch_dma_agu_in0_slave;
	block_mme_agu_core arch_dma_agu_in1_slave;
	block_mme_agu_core arch_dma_agu_in2_slave;
	block_mme_agu_core arch_dma_agu_in3_slave;
	reg_arch_dma_a_ss arch_dma_a_ss;
	reg_arch_dma_b_ss arch_dma_b_ss;
	block_mme_agu_core arch_dma_agu_cout0_master;
	block_mme_agu_core arch_dma_agu_cout0_slave;
	reg_arch_dma_cout_ss arch_dma_cout_ss;
	block_mme_non_tensor_descriptor arch_dma_n_ten;
	uint32_t _pad2712[250];
	block_special_regs special;
} block_mme_ctrl_lo;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_mme_ctrl_lo_defaults[] =
{
	// offset	// value
	{ 0x0   , 0xc04f7f00          , 1 }, // arch_status_master
	{ 0x4   , 0xc04f7f00          , 1 }, // arch_status_slave
	{ 0x26c , 0x8040404           , 1 }, // rate_limiter
	{ 0x40c , 0x10                , 1 }, // sync_object_fifo_th
	{ 0x410 , 0x1c567890          , 1 }, // eus_rollup_dly_dw0
	{ 0x414 , 0x2                 , 1 }, // eus_rollup_dly_dw1
	{ 0x418 , 0x80                , 1 }, // eus_rollup_cd_prot_f16
	{ 0x41c , 0x80                , 1 }, // eus_rollup_cd_prot_f8
	{ 0x420 , 0x80                , 1 }, // eus_rollup_cd_prot_fp32
	{ 0x424 , 0x80                , 1 }, // eus_rollup_cd_prot_tf32
	{ 0x428 , 0x4101000           , 1 }, // pcu_rl_desc0
	{ 0x42c , 0x2000200           , 1 }, // pcu_rl_token_update
	{ 0x430 , 0x1990200           , 1 }, // pcu_rl_th
	{ 0x434 , 0x3e800100          , 1 }, // pcu_rl_min
	{ 0x438 , 0x3                 , 1 }, // pcu_rl_ctrl_en
	{ 0x43c , 0x1d                , 1 }, // pcu_rl_history_log_size
	{ 0x440 , 0xaaaa5555          , 1 }, // pcu_dummy_a_bf16
	{ 0x444 , 0x77773333          , 1 }, // pcu_dummy_b_bf16
	{ 0x448 , 0xaaaa5555          , 1 }, // pcu_dummy_a_fp16
	{ 0x44c , 0x77773333          , 1 }, // pcu_dummy_b_fp16
	{ 0x450 , 0xaa55aa55          , 1 }, // pcu_dummy_f8
	{ 0x478 , 0x210000            , 1 }, // eu
	{ 0x48c , 0x80001000          , 1 }, // pcu_rl_sat_sec
	{ 0x4a0 , 0x12af              , 1 }, // qm
	{ 0x4b4 , 0x3                 , 1 }, // redun_psoc_sel_sec
	{ 0x4c0 , 0x1                 , 1 }, // eu_rl_token_sel
	{ 0x4c4 , 0x8040708           , 1 }, // eu_rl_cfg
	{ 0x4f4 , 0x6                 , 1 }, // slv_axi2apb
	{ 0x4fc , 0x2af               , 1 }, // sig
	{ 0x508 , 0x2000              , 1 }, // sbte0_hbw_addr_split0
	{ 0x50c , 0x4000              , 1 }, // sbte0_hbw_addr_split1
	{ 0x510 , 0x6000              , 1 }, // sbte0_hbw_addr_split2
	{ 0x514 , 0x8000              , 1 }, // sbte0_hbw_addr_split3
	{ 0x518 , 0xa000              , 1 }, // sbte1_hbw_addr_split0
	{ 0x51c , 0xc000              , 1 }, // sbte1_hbw_addr_split1
	{ 0x520 , 0xe000              , 1 }, // sbte1_hbw_addr_split2
	{ 0x524 , 0x10000             , 1 }, // sbte1_hbw_addr_split3
	{ 0x528 , 0xa9000             , 1 }, // sbte0_dbg_addr_split0
	{ 0x52c , 0xb0000             , 1 }, // sbte0_dbg_addr_split1
	{ 0x530 , 0xc9000             , 1 }, // sbte1_dbg_addr_split0
	{ 0x534 , 0xd0000             , 1 }, // sbte1_dbg_addr_split1
	{ 0x53c , 0x1f1f18            , 1 }, // misc
	{ 0x544 , 0x1                 , 1 }, // so_awprot_bit1
	{ 0x550 , 0x1                 , 1 }, // dma_so_awprot_bit1
	{ 0xa6c , 0x8040404           , 1 }, // rate_limiter
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_MME_CTRL_LO_H_ */

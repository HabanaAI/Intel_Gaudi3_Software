/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_MME_H_
#define ASIC_REG_STRUCTS_MME_H_

#include <stdint.h>

#pragma pack(push, 1)

namespace mme {
/*
 CMD 
*/
struct reg_cmd {
	union {
		struct {
			uint32_t execute : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_cmd) == 4), "reg_cmd size is not 32-bit");
/*
 STATUS1 
*/
struct reg_status1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_status1) == 4), "reg_status1 size is not 32-bit");
/*
 RESET 
*/
struct reg_reset {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_reset) == 4), "reg_reset size is not 32-bit");
/*
 QM_STALL 
*/
struct reg_qm_stall {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_qm_stall) == 4), "reg_qm_stall size is not 32-bit");
/*
 SYNC_OBJECT_FIFO_TH 
*/
struct reg_sync_object_fifo_th {
	union {
		struct {
			uint32_t v : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_sync_object_fifo_th) == 4), "reg_sync_object_fifo_th size is not 32-bit");
/*
 ARCH_STATUS 
*/
struct reg_arch_status {
	union {
		struct {
			uint32_t agu_s : 1,
				agu_l : 1,
				agu_o : 1,
				ap : 1,
				eu : 1,
				te_s : 1,
				te_l : 1,
				sb_a_empty : 1,
				sb_b_empty : 1,
				_reserved11 : 2,
				sm_idle : 1,
				wbc_axi_idle : 1,
				_reserved18 : 5,
				sb_b_axi_idle : 1,
				_reserved20 : 1,
				sb_a_axi_idle : 1,
				_reserved22 : 1,
				accum_free : 3,
				_reserved30 : 5,
				qm_idle : 1,
				qm_rdy : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_arch_status) == 4), "reg_arch_status size is not 32-bit");
/*
 SHADOW_0_STATUS 
*/
struct reg_shadow_0_status {
	union {
		struct {
			uint32_t agu_s : 1,
				agu_l : 1,
				agu_o : 1,
				ap : 1,
				eu : 1,
				te_s : 1,
				te_l : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_shadow_0_status) == 4), "reg_shadow_0_status size is not 32-bit");
/*
 SHADOW_1_STATUS 
*/
struct reg_shadow_1_status {
	union {
		struct {
			uint32_t agu_s : 1,
				agu_l : 1,
				agu_o : 1,
				ap : 1,
				eu : 1,
				te_s : 1,
				te_l : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_shadow_1_status) == 4), "reg_shadow_1_status size is not 32-bit");
/*
 SHADOW_2_STATUS 
*/
struct reg_shadow_2_status {
	union {
		struct {
			uint32_t agu_s : 1,
				agu_l : 1,
				agu_o : 1,
				ap : 1,
				eu : 1,
				te_s : 1,
				te_l : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_shadow_2_status) == 4), "reg_shadow_2_status size is not 32-bit");
/*
 SHADOW_3_STATUS 
*/
struct reg_shadow_3_status {
	union {
		struct {
			uint32_t agu_s : 1,
				agu_l : 1,
				agu_o : 1,
				ap : 1,
				eu : 1,
				te_s : 1,
				te_l : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_shadow_3_status) == 4), "reg_shadow_3_status size is not 32-bit");
/*
 INTR_CAUSE 
 cause interrupt
*/
struct reg_intr_cause {
	union {
		struct {
			uint32_t v : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_intr_cause) == 4), "reg_intr_cause size is not 32-bit");
/*
 INTR_MASK 
 interrupts cause mask
*/
struct reg_intr_mask {
	union {
		struct {
			uint32_t v : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_intr_mask) == 4), "reg_intr_mask size is not 32-bit");
/*
 LOG_SHADOW 
*/
struct reg_log_shadow {
	union {
		struct {
			uint32_t mask_0 : 7,
				_reserved8 : 1,
				mask_1 : 7,
				_reserved16 : 1,
				mask_2 : 7,
				_reserved24 : 1,
				mask_3 : 7,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_log_shadow) == 4), "reg_log_shadow size is not 32-bit");
/*
 EUS_ROLLUP_CNT_ADD 
*/
struct reg_eus_rollup_cnt_add {
	union {
		struct {
			uint32_t v : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_eus_rollup_cnt_add) == 4), "reg_eus_rollup_cnt_add size is not 32-bit");
/*
 PCU_RL_DESC0 
*/
struct reg_pcu_rl_desc0 {
	union {
		struct {
			uint32_t rl_rst_token : 16,
				rl_timeout : 8,
				rl_dummy2real_period : 8;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_desc0) == 4), "reg_pcu_rl_desc0 size is not 32-bit");
/*
 PCU_RL_TOKEN_UPDATE 
*/
struct reg_pcu_rl_token_update {
	union {
		struct {
			uint32_t inc_val : 16,
				dec_val : 16;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_token_update) == 4), "reg_pcu_rl_token_update size is not 32-bit");
/*
 PCU_RL_TH 
*/
struct reg_pcu_rl_th {
	union {
		struct {
			uint32_t pool_th_dec : 16,
				dummy_real_diff_th : 16;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_th) == 4), "reg_pcu_rl_th size is not 32-bit");
/*
 PCU_RL_MIN 
*/
struct reg_pcu_rl_min {
	union {
		struct {
			uint32_t avg_min_to_force_dummy : 16,
				token_min_val : 16;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_min) == 4), "reg_pcu_rl_min size is not 32-bit");
/*
 PCU_RL_CTRL_EN 
*/
struct reg_pcu_rl_ctrl_en {
	union {
		struct {
			uint32_t pcu_disable : 1,
				min_val_prot_en : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_ctrl_en) == 4), "reg_pcu_rl_ctrl_en size is not 32-bit");
/*
 PCU_DUMMY_A_BF16 
*/
struct reg_pcu_dummy_a_bf16 {
	union {
		struct {
			uint32_t odd : 16,
				even : 16;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_dummy_a_bf16) == 4), "reg_pcu_dummy_a_bf16 size is not 32-bit");
/*
 PCU_DUMMY_B_BF16 
*/
struct reg_pcu_dummy_b_bf16 {
	union {
		struct {
			uint32_t odd : 16,
				even : 16;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_dummy_b_bf16) == 4), "reg_pcu_dummy_b_bf16 size is not 32-bit");
/*
 PCU_DUMMY_A_FP32_ODD 
*/
struct reg_pcu_dummy_a_fp32_odd {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_dummy_a_fp32_odd) == 4), "reg_pcu_dummy_a_fp32_odd size is not 32-bit");
/*
 PCU_DUMMY_A_FP32_EVEN 
*/
struct reg_pcu_dummy_a_fp32_even {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_dummy_a_fp32_even) == 4), "reg_pcu_dummy_a_fp32_even size is not 32-bit");
/*
 PCU_DUMMY_B_FP32_ODD 
*/
struct reg_pcu_dummy_b_fp32_odd {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_dummy_b_fp32_odd) == 4), "reg_pcu_dummy_b_fp32_odd size is not 32-bit");
/*
 PCU_DUMMY_B_FP32_EVEN 
*/
struct reg_pcu_dummy_b_fp32_even {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_dummy_b_fp32_even) == 4), "reg_pcu_dummy_b_fp32_even size is not 32-bit");
/*
 EU_POWER_SAVE_DISABLE 
*/
struct reg_eu_power_save_disable {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_eu_power_save_disable) == 4), "reg_eu_power_save_disable size is not 32-bit");
/*
 CS_DBG_BLOCK_ID 
*/
struct reg_cs_dbg_block_id {
	union {
		struct {
			uint32_t v : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_cs_dbg_block_id) == 4), "reg_cs_dbg_block_id size is not 32-bit");
/*
 CS_DBG_STATUS_DROP_CNT 
*/
struct reg_cs_dbg_status_drop_cnt {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_cs_dbg_status_drop_cnt) == 4), "reg_cs_dbg_status_drop_cnt size is not 32-bit");
/*
 PROT 
*/
struct reg_prot {
	union {
		struct {
			uint32_t v : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_prot) == 4), "reg_prot size is not 32-bit");
/*
 PCU_RL_HISTORY_LOG_SIZE 
*/
struct reg_pcu_rl_history_log_size {
	union {
		struct {
			uint32_t all_macs : 3,
				real_macs : 2,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_history_log_size) == 4), "reg_pcu_rl_history_log_size size is not 32-bit");
/*
 TE_CLOSE_CGATE 
*/
struct reg_te_close_cgate {
	union {
		struct {
			uint32_t tea : 1,
				teb : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_te_close_cgate) == 4), "reg_te_close_cgate size is not 32-bit");
/*
 AGU_SM_INFLIGHT_CNTR 
*/
struct reg_agu_sm_inflight_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_agu_sm_inflight_cntr) == 4), "reg_agu_sm_inflight_cntr size is not 32-bit");
/*
 AGU_SM_TOTAL_CNTR 
 write clears counter
*/
struct reg_agu_sm_total_cntr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_agu_sm_total_cntr) == 4), "reg_agu_sm_total_cntr size is not 32-bit");
/*
 EZSYNC_OUT_CREDIT 
*/
struct reg_ezsync_out_credit {
	union {
		struct {
			uint32_t tea : 4,
				_reserved8 : 4,
				qm_csmr : 4,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_ezsync_out_credit) == 4), "reg_ezsync_out_credit size is not 32-bit");
/*
 PCU_RL_SAT_SEC 
 SECURED RATE LIMITER SATURATION
*/
struct reg_pcu_rl_sat_sec {
	union {
		struct {
			uint32_t val : 20,
				_reserved31 : 11,
				sel : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_sat_sec) == 4), "reg_pcu_rl_sat_sec size is not 32-bit");
/*
 AGU_SYNC_MSG_AXI_USER 
*/
struct reg_agu_sync_msg_axi_user {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_agu_sync_msg_axi_user) == 4), "reg_agu_sync_msg_axi_user size is not 32-bit");
/*
 QM_SLV_LBW_CLK_EN 
*/
struct reg_qm_slv_lbw_clk_en {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_qm_slv_lbw_clk_en) == 4), "reg_qm_slv_lbw_clk_en size is not 32-bit");
} /* mme namespace */

#include "mme_agu_core_regs.h"
#include "mme_non_tensor_descriptor_end_regs.h"
#include "mme_non_tensor_descriptor_start_regs.h"
#include "mme_tensor_regs.h"
/*
 MME block
*/
struct block_mme {
	struct mme::reg_arch_status arch_status;
	uint32_t _pad4[1];
	struct block_mme_non_tensor_descriptor_start arch;
	struct block_mme_tensor arch_tensor_s;
	struct block_mme_agu_core arch_agu_s;
	struct block_mme_tensor arch_tensor_l;
	struct block_mme_agu_core arch_agu_l_local;
	struct block_mme_agu_core arch_agu_l_remote;
	struct block_mme_tensor arch_tensor_o;
	struct block_mme_agu_core arch_agu_o_local;
	struct block_mme_agu_core arch_agu_o_remote;
	struct block_mme_non_tensor_descriptor_end arch_desc;
	uint32_t _pad552[22];
	struct mme::reg_cmd cmd;
	struct mme::reg_status1 status1;
	struct mme::reg_reset reset;
	struct mme::reg_qm_stall qm_stall;
	struct mme::reg_sync_object_fifo_th sync_object_fifo_th;
	struct mme::reg_eus_rollup_cnt_add eus_rollup_cnt_add;
	struct mme::reg_intr_cause intr_cause;
	struct mme::reg_intr_mask intr_mask;
	struct mme::reg_log_shadow log_shadow;
	struct mme::reg_pcu_rl_desc0 pcu_rl_desc0;
	struct mme::reg_pcu_rl_token_update pcu_rl_token_update;
	struct mme::reg_pcu_rl_th pcu_rl_th;
	struct mme::reg_pcu_rl_min pcu_rl_min;
	struct mme::reg_pcu_rl_ctrl_en pcu_rl_ctrl_en;
	struct mme::reg_pcu_rl_history_log_size pcu_rl_history_log_size;
	struct mme::reg_pcu_dummy_a_bf16 pcu_dummy_a_bf16;
	struct mme::reg_pcu_dummy_b_bf16 pcu_dummy_b_bf16;
	struct mme::reg_pcu_dummy_a_fp32_odd pcu_dummy_a_fp32_odd;
	struct mme::reg_pcu_dummy_a_fp32_even pcu_dummy_a_fp32_even;
	struct mme::reg_pcu_dummy_b_fp32_odd pcu_dummy_b_fp32_odd;
	struct mme::reg_pcu_dummy_b_fp32_even pcu_dummy_b_fp32_even;
	struct mme::reg_prot prot;
	struct mme::reg_eu_power_save_disable eu_power_save_disable;
	struct mme::reg_cs_dbg_block_id cs_dbg_block_id;
	struct mme::reg_cs_dbg_status_drop_cnt cs_dbg_status_drop_cnt;
	struct mme::reg_te_close_cgate te_close_cgate;
	struct mme::reg_agu_sm_inflight_cntr agu_sm_inflight_cntr;
	struct mme::reg_agu_sm_total_cntr agu_sm_total_cntr;
	struct mme::reg_ezsync_out_credit ezsync_out_credit;
	struct mme::reg_pcu_rl_sat_sec pcu_rl_sat_sec;
	struct mme::reg_agu_sync_msg_axi_user agu_sync_msg_axi_user;
	struct mme::reg_qm_slv_lbw_clk_en qm_slv_lbw_clk_en;
	uint32_t _pad768[64];
	struct mme::reg_shadow_0_status shadow_0_status;
	uint32_t _pad1028[1];
	struct block_mme_non_tensor_descriptor_start shadow_0;
	struct block_mme_tensor shadow_0_tensor_s;
	struct block_mme_agu_core shadow_0_agu_s;
	struct block_mme_tensor shadow_0_tensor_l;
	struct block_mme_agu_core shadow_0_agu_l_local;
	struct block_mme_agu_core shadow_0_agu_l_remote;
	struct block_mme_tensor shadow_0_tensor_o;
	struct block_mme_agu_core shadow_0_agu_o_local;
	struct block_mme_agu_core shadow_0_agu_o_remote;
	struct block_mme_non_tensor_descriptor_end shadow_0_desc;
	uint32_t _pad1576[22];
	struct mme::reg_shadow_1_status shadow_1_status;
	uint32_t _pad1668[1];
	struct block_mme_non_tensor_descriptor_start shadow_1;
	struct block_mme_tensor shadow_1_tensor_s;
	struct block_mme_agu_core shadow_1_agu_s;
	struct block_mme_tensor shadow_1_tensor_l;
	struct block_mme_agu_core shadow_1_agu_l_local;
	struct block_mme_agu_core shadow_1_agu_l_remote;
	struct block_mme_tensor shadow_1_tensor_o;
	struct block_mme_agu_core shadow_1_agu_o_local;
	struct block_mme_agu_core shadow_1_agu_o_remote;
	struct block_mme_non_tensor_descriptor_end shadow_1_desc;
	uint32_t _pad2216[22];
	struct mme::reg_shadow_2_status shadow_2_status;
	uint32_t _pad2308[1];
	struct block_mme_non_tensor_descriptor_start shadow_2;
	struct block_mme_tensor shadow_2_tensor_s;
	struct block_mme_agu_core shadow_2_agu_s;
	struct block_mme_tensor shadow_2_tensor_l;
	struct block_mme_agu_core shadow_2_agu_l_local;
	struct block_mme_agu_core shadow_2_agu_l_remote;
	struct block_mme_tensor shadow_2_tensor_o;
	struct block_mme_agu_core shadow_2_agu_o_local;
	struct block_mme_agu_core shadow_2_agu_o_remote;
	struct block_mme_non_tensor_descriptor_end shadow_2_desc;
	uint32_t _pad2856[22];
	struct mme::reg_shadow_3_status shadow_3_status;
	uint32_t _pad2948[1];
	struct block_mme_non_tensor_descriptor_start shadow_3;
	struct block_mme_tensor shadow_3_tensor_s;
	struct block_mme_agu_core shadow_3_agu_s;
	struct block_mme_tensor shadow_3_tensor_l;
	struct block_mme_agu_core shadow_3_agu_l_local;
	struct block_mme_agu_core shadow_3_agu_l_remote;
	struct block_mme_tensor shadow_3_tensor_o;
	struct block_mme_agu_core shadow_3_agu_o_local;
	struct block_mme_agu_core shadow_3_agu_o_remote;
	struct block_mme_non_tensor_descriptor_end shadow_3_desc;
};
#include "gaudi_types.h"
const offsetVal block_mme_defaults[]
{
	// offset	// value
	{ 0x0   , 0xc1141980          , 1 }, // arch_status
	{ 0x290 , 0x10                , 1 }, // sync_object_fifo_th
	{ 0x294 , 0x2                 , 1 }, // eus_rollup_cnt_add
	{ 0x2a4 , 0x4101000           , 1 }, // pcu_rl_desc0
	{ 0x2a8 , 0x2000200           , 1 }, // pcu_rl_token_update
	{ 0x2ac , 0x1990200           , 1 }, // pcu_rl_th
	{ 0x2b0 , 0x3e800100          , 1 }, // pcu_rl_min
	{ 0x2b4 , 0x3                 , 1 }, // pcu_rl_ctrl_en
	{ 0x2b8 , 0x1f                , 1 }, // pcu_rl_history_log_size
	{ 0x2bc , 0xaaaa5555          , 1 }, // pcu_dummy_a_bf16
	{ 0x2c0 , 0x77773333          , 1 }, // pcu_dummy_b_bf16
	{ 0x2c4 , 0xaaaaaaaa          , 1 }, // pcu_dummy_a_fp32_odd
	{ 0x2c8 , 0x55555555          , 1 }, // pcu_dummy_a_fp32_even
	{ 0x2cc , 0xdddddddd          , 1 }, // pcu_dummy_b_fp32_odd
	{ 0x2d0 , 0xbbbbbbbb          , 1 }, // pcu_dummy_b_fp32_even
	{ 0x2d4 , 0x2                 , 1 }, // prot
	{ 0x2f0 , 0xc0c               , 1 }, // ezsync_out_credit
	{ 0x2f4 , 0x80001000          , 1 }, // pcu_rl_sat_sec
};

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_MME_H_ */

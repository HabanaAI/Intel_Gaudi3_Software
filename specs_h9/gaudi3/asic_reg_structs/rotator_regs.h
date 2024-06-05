/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_ROTATOR_H_
#define ASIC_REG_STRUCTS_GAUDI3_ROTATOR_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "axuser_hbw_regs.h"
#include "axuser_lbw_regs.h"
#include "rot_desc_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace rotator {
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
 CPL_MSG_AXI 
 b'CPL: AXI attributes for LBW completion message'
*/
typedef struct reg_cpl_msg_axi {
	union {
		struct {
			uint32_t cache : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_axi;
static_assert((sizeof(struct reg_cpl_msg_axi) == 4), "reg_cpl_msg_axi size is not 32-bit");
/*
 AWPROT_0_CPL_MSG 
 b'CPL: CPL MSG AWPROT[0] field'
*/
typedef struct reg_awprot_0_cpl_msg {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_0_cpl_msg;
static_assert((sizeof(struct reg_awprot_0_cpl_msg) == 4), "reg_awprot_0_cpl_msg size is not 32-bit");
/*
 AWPROT_1_CPL_MSG 
 b'CPL: CPL MSG AWPROT[1] field'
*/
typedef struct reg_awprot_1_cpl_msg {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_1_cpl_msg;
static_assert((sizeof(struct reg_awprot_1_cpl_msg) == 4), "reg_awprot_1_cpl_msg size is not 32-bit");
/*
 AWPROT_2_CPL_MSG 
 b'CPL: CPL MSG AWPROT[2] field'
*/
typedef struct reg_awprot_2_cpl_msg {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_2_cpl_msg;
static_assert((sizeof(struct reg_awprot_2_cpl_msg) == 4), "reg_awprot_2_cpl_msg size is not 32-bit");
/*
 CPL_MSG_THRESHOLD 
 b'CPL: Threshold to bound LBW completion messages'
*/
typedef struct reg_cpl_msg_threshold {
	union {
		struct {
			uint32_t val : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_threshold;
static_assert((sizeof(struct reg_cpl_msg_threshold) == 4), "reg_cpl_msg_threshold size is not 32-bit");
/*
 RSB_CFG 
 b'RSB:'
*/
typedef struct reg_rsb_cfg {
	union {
		struct {
			uint32_t cache_disable : 1,
				enable_cgate : 1,
				_reserved4 : 2,
				data_occupancy_sel : 3,
				_reserved8 : 1,
				axi_128byte : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_rsb_cfg;
static_assert((sizeof(struct reg_rsb_cfg) == 4), "reg_rsb_cfg size is not 32-bit");
/*
 ARPROT_0_RSB 
 b'RSB: RSB ARPROT[0] field'
*/
typedef struct reg_arprot_0_rsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_0_rsb;
static_assert((sizeof(struct reg_arprot_0_rsb) == 4), "reg_arprot_0_rsb size is not 32-bit");
/*
 ARPROT_1_RSB 
 b'RSB: RSB ARPROT[1] field'
*/
typedef struct reg_arprot_1_rsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_1_rsb;
static_assert((sizeof(struct reg_arprot_1_rsb) == 4), "reg_arprot_1_rsb size is not 32-bit");
/*
 ARPROT_2_RSB 
 b'RSB: RSB ARPROT[2] field'
*/
typedef struct reg_arprot_2_rsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_2_rsb;
static_assert((sizeof(struct reg_arprot_2_rsb) == 4), "reg_arprot_2_rsb size is not 32-bit");
/*
 ARPROT_0_MRSB 
 b'RSB: MRSB ARPROT[0] field'
*/
typedef struct reg_arprot_0_mrsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_0_mrsb;
static_assert((sizeof(struct reg_arprot_0_mrsb) == 4), "reg_arprot_0_mrsb size is not 32-bit");
/*
 ARPROT_1_MRSB 
 b'RSB: MRSB ARPROT[1] field'
*/
typedef struct reg_arprot_1_mrsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_1_mrsb;
static_assert((sizeof(struct reg_arprot_1_mrsb) == 4), "reg_arprot_1_mrsb size is not 32-bit");
/*
 ARPROT_2_MRSB 
 b'RSB: MRSB ARPROT[2] field'
*/
typedef struct reg_arprot_2_mrsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_2_mrsb;
static_assert((sizeof(struct reg_arprot_2_mrsb) == 4), "reg_arprot_2_mrsb size is not 32-bit");
/*
 ARPROT_0_GRSB 
 b'RSB: GRSB ARPROT[0] field'
*/
typedef struct reg_arprot_0_grsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_0_grsb;
static_assert((sizeof(struct reg_arprot_0_grsb) == 4), "reg_arprot_0_grsb size is not 32-bit");
/*
 ARPROT_1_GRSB 
 b'RSB: GRSB ARPROT[1] field'
*/
typedef struct reg_arprot_1_grsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_1_grsb;
static_assert((sizeof(struct reg_arprot_1_grsb) == 4), "reg_arprot_1_grsb size is not 32-bit");
/*
 ARPROT_2_GRSB 
 b'RSB: GRSB ARPROT[2] field'
*/
typedef struct reg_arprot_2_grsb {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arprot_2_grsb;
static_assert((sizeof(struct reg_arprot_2_grsb) == 4), "reg_arprot_2_grsb size is not 32-bit");
/*
 DISABLE_PAD_CALC 
 b'RSB:'
*/
typedef struct reg_disable_pad_calc {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_disable_pad_calc;
static_assert((sizeof(struct reg_disable_pad_calc) == 4), "reg_disable_pad_calc size is not 32-bit");
/*
 RSB_CAM_MAX_SIZE 
 b'RSB:'
*/
typedef struct reg_rsb_cam_max_size {
	union {
		struct {
			uint32_t data : 16,
				md : 16;
		};
		uint32_t _raw;
	};
} reg_rsb_cam_max_size;
static_assert((sizeof(struct reg_rsb_cam_max_size) == 4), "reg_rsb_cam_max_size size is not 32-bit");
/*
 RSB_MAX_OS 
 b'RSB: max on-flight read transactions'
*/
typedef struct reg_rsb_max_os {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_rsb_max_os;
static_assert((sizeof(struct reg_rsb_max_os) == 4), "reg_rsb_max_os size is not 32-bit");
/*
 RSB_RL 
 b'RSB: RSB rate-limiter'
*/
typedef struct reg_rsb_rl {
	union {
		struct {
			uint32_t saturation : 8,
				timeout : 8,
				rst_token : 8,
				rate_limiter_en : 1,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_rsb_rl;
static_assert((sizeof(struct reg_rsb_rl) == 4), "reg_rsb_rl size is not 32-bit");
/*
 MRSB_RL 
 b'RSB: MSB rate-limiter'
*/
typedef struct reg_mrsb_rl {
	union {
		struct {
			uint32_t rst_token : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_mrsb_rl;
static_assert((sizeof(struct reg_mrsb_rl) == 4), "reg_mrsb_rl size is not 32-bit");
/*
 GRSB_RL 
 b'RSB: GSB rate-limiter'
*/
typedef struct reg_grsb_rl {
	union {
		struct {
			uint32_t rst_token : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_grsb_rl;
static_assert((sizeof(struct reg_grsb_rl) == 4), "reg_grsb_rl size is not 32-bit");
/*
 RSB_INFLIGHTS 
 b'RSB:'
*/
typedef struct reg_rsb_inflights {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rsb_inflights;
static_assert((sizeof(struct reg_rsb_inflights) == 4), "reg_rsb_inflights size is not 32-bit");
/*
 RSB_OCCUPANCY 
 b'RSB:'
*/
typedef struct reg_rsb_occupancy {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rsb_occupancy;
static_assert((sizeof(struct reg_rsb_occupancy) == 4), "reg_rsb_occupancy size is not 32-bit");
/*
 RSB_OCCUPANCY_DATA 
 b'RSB: RSB Occupancy Data'
*/
typedef struct reg_rsb_occupancy_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rsb_occupancy_data;
static_assert((sizeof(struct reg_rsb_occupancy_data) == 4), "reg_rsb_occupancy_data size is not 32-bit");
/*
 MRSB_OCCUPANCY_DATA 
 b'RSB: MRSB Occupancy Data'
*/
typedef struct reg_mrsb_occupancy_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mrsb_occupancy_data;
static_assert((sizeof(struct reg_mrsb_occupancy_data) == 4), "reg_mrsb_occupancy_data size is not 32-bit");
/*
 GRSB_OCCUPANCY_DATA 
 b'RSB: GRSB Occupancy Data'
*/
typedef struct reg_grsb_occupancy_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_grsb_occupancy_data;
static_assert((sizeof(struct reg_grsb_occupancy_data) == 4), "reg_grsb_occupancy_data size is not 32-bit");
/*
 RSB_OCCUPANCY_MD 
 b'RSB: RSB Occupancy MD'
*/
typedef struct reg_rsb_occupancy_md {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rsb_occupancy_md;
static_assert((sizeof(struct reg_rsb_occupancy_md) == 4), "reg_rsb_occupancy_md size is not 32-bit");
/*
 MRSB_OCCUPANCY_MD 
 b'RSB: MRSB Occupancy MD'
*/
typedef struct reg_mrsb_occupancy_md {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mrsb_occupancy_md;
static_assert((sizeof(struct reg_mrsb_occupancy_md) == 4), "reg_mrsb_occupancy_md size is not 32-bit");
/*
 GRSB_OCCUPANCY_MD 
 b'RSB: GRSB Occupancy MD'
*/
typedef struct reg_grsb_occupancy_md {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_grsb_occupancy_md;
static_assert((sizeof(struct reg_grsb_occupancy_md) == 4), "reg_grsb_occupancy_md size is not 32-bit");
/*
 RSB_INFO 
 b'RSB: Status indications'
*/
typedef struct reg_rsb_info {
	union {
		struct {
			uint32_t empty : 1,
				axi_idle : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_rsb_info;
static_assert((sizeof(struct reg_rsb_info) == 4), "reg_rsb_info size is not 32-bit");
/*
 WBC_CTRL 
 b'WCH:'
*/
typedef struct reg_wbc_ctrl {
	union {
		struct {
			uint32_t req_weights_rot : 8,
				req_weights_rsl : 8,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_wbc_ctrl;
static_assert((sizeof(struct reg_wbc_ctrl) == 4), "reg_wbc_ctrl size is not 32-bit");
/*
 WCH_AGG_CFG 
 b'WCH: WCH aggregator config'
*/
typedef struct reg_wch_agg_cfg {
	union {
		struct {
			uint32_t aggr_bypass : 1,
				burst_en : 1,
				idle_timeout_en : 1,
				idle_timeout : 16,
				dis_null_req : 1,
				disable_partial : 1,
				wr_axi_agg_cache_dis_rot : 1,
				wr_axi_agg_cache_dis_rsl : 1,
				dont_agg_on_reduction : 1,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_wch_agg_cfg;
static_assert((sizeof(struct reg_wch_agg_cfg) == 4), "reg_wch_agg_cfg size is not 32-bit");
/*
 AWPROT_0_WCH_ROT 
 b'WCH: WCH ROT AWPROT[0] field'
*/
typedef struct reg_awprot_0_wch_rot {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_0_wch_rot;
static_assert((sizeof(struct reg_awprot_0_wch_rot) == 4), "reg_awprot_0_wch_rot size is not 32-bit");
/*
 AWPROT_1_WCH_ROT 
 b'WCH: WCH ROT AWPROT[1] field'
*/
typedef struct reg_awprot_1_wch_rot {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_1_wch_rot;
static_assert((sizeof(struct reg_awprot_1_wch_rot) == 4), "reg_awprot_1_wch_rot size is not 32-bit");
/*
 AWPROT_2_WCH_ROT 
 b'WCH: WCH ROT AWPROT[2] field'
*/
typedef struct reg_awprot_2_wch_rot {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_2_wch_rot;
static_assert((sizeof(struct reg_awprot_2_wch_rot) == 4), "reg_awprot_2_wch_rot size is not 32-bit");
/*
 AWPROT_0_WCH_RSL 
 b'WCH: WCH RSL AWPROT[0] field'
*/
typedef struct reg_awprot_0_wch_rsl {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_0_wch_rsl;
static_assert((sizeof(struct reg_awprot_0_wch_rsl) == 4), "reg_awprot_0_wch_rsl size is not 32-bit");
/*
 AWPROT_1_WCH_RSL 
 b'WCH: WCH RSL AWPROT[1] field'
*/
typedef struct reg_awprot_1_wch_rsl {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_1_wch_rsl;
static_assert((sizeof(struct reg_awprot_1_wch_rsl) == 4), "reg_awprot_1_wch_rsl size is not 32-bit");
/*
 AWPROT_2_WCH_RSL 
 b'WCH: WCH RSL AWPROT[2] field'
*/
typedef struct reg_awprot_2_wch_rsl {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_awprot_2_wch_rsl;
static_assert((sizeof(struct reg_awprot_2_wch_rsl) == 4), "reg_awprot_2_wch_rsl size is not 32-bit");
/*
 WBC_MAX_OUTSTANDING 
 b'WCH: max on-flight write transactions'
*/
typedef struct reg_wbc_max_outstanding {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_wbc_max_outstanding;
static_assert((sizeof(struct reg_wbc_max_outstanding) == 4), "reg_wbc_max_outstanding size is not 32-bit");
/*
 WBC_RL 
 b'WCH: WBC rate-limiter'
*/
typedef struct reg_wbc_rl {
	union {
		struct {
			uint32_t saturation : 8,
				timeout : 8,
				rate_limiter_en : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_wbc_rl;
static_assert((sizeof(struct reg_wbc_rl) == 4), "reg_wbc_rl size is not 32-bit");
/*
 WBC_RL_RST_TOKEN 
 b'WCH:'
*/
typedef struct reg_wbc_rl_rst_token {
	union {
		struct {
			uint32_t rst_token_rot : 8,
				_reserved16 : 8,
				rst_token_rsl : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_wbc_rl_rst_token;
static_assert((sizeof(struct reg_wbc_rl_rst_token) == 4), "reg_wbc_rl_rst_token size is not 32-bit");
/*
 WBC_INFLIGHTS 
 b'WCH: in-flight transaction counter'
*/
typedef struct reg_wbc_inflights {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_wbc_inflights;
static_assert((sizeof(struct reg_wbc_inflights) == 4), "reg_wbc_inflights size is not 32-bit");
/*
 WBC_INFO 
 b'WCH: Status indications'
*/
typedef struct reg_wbc_info {
	union {
		struct {
			uint32_t empty : 1,
				axi_idle : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_wbc_info;
static_assert((sizeof(struct reg_wbc_info) == 4), "reg_wbc_info size is not 32-bit");
/*
 WBC_MON 
 b'WCH: Performance monitor results'
*/
typedef struct reg_wbc_mon {
	union {
		struct {
			uint32_t cnt : 1,
				_reserved8 : 7,
				ts : 2,
				_reserved16 : 6,
				context_id : 16;
		};
		uint32_t _raw;
	};
} reg_wbc_mon;
static_assert((sizeof(struct reg_wbc_mon) == 4), "reg_wbc_mon size is not 32-bit");
/*
 CLK_EN 
 b'MSS:'
*/
typedef struct reg_clk_en {
	union {
		struct {
			uint32_t lbw_cfg_dis : 1,
				_reserved4 : 3,
				dbg_cfg_dis : 1,
				sb_empty_mask : 1,
				dbg_trigout_req_dis : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_clk_en;
static_assert((sizeof(struct reg_clk_en) == 4), "reg_clk_en size is not 32-bit");
/*
 QMAN_CFG 
 b'MSS: Qman Configuration'
*/
typedef struct reg_qman_cfg {
	union {
		struct {
			uint32_t force_stop : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_qman_cfg;
static_assert((sizeof(struct reg_qman_cfg) == 4), "reg_qman_cfg size is not 32-bit");
/*
 ERR_CFG 
 b'MSS:'
*/
typedef struct reg_err_cfg {
	union {
		struct {
			uint32_t stop_on_err : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_err_cfg;
static_assert((sizeof(struct reg_err_cfg) == 4), "reg_err_cfg size is not 32-bit");
/*
 MSS_HALT 
 b'MSS: halt for sb, wbc and mesh sb'
*/
typedef struct reg_mss_halt {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_mss_halt;
static_assert((sizeof(struct reg_mss_halt) == 4), "reg_mss_halt size is not 32-bit");
/*
 MSS_STS 
 b'MSS: MSS halt status'
*/
typedef struct reg_mss_sts {
	union {
		struct {
			uint32_t is_halt : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_mss_sts;
static_assert((sizeof(struct reg_mss_sts) == 4), "reg_mss_sts size is not 32-bit");
/*
 MSS_SEI_CLEAR 
 b'MSS: Clear SEI Interrupts Status'
*/
typedef struct reg_mss_sei_clear {
	union {
		struct {
			uint32_t i0 : 1,
				i1 : 1,
				i2 : 1,
				i3 : 1,
				i4 : 1,
				i5 : 1,
				i6 : 1,
				i7 : 1,
				i8 : 1,
				i9 : 1,
				i10 : 1,
				i11 : 1,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_mss_sei_clear;
static_assert((sizeof(struct reg_mss_sei_clear) == 4), "reg_mss_sei_clear size is not 32-bit");
/*
 MSS_SEI_MASK 
 b'MSS: SEI INTERRUPTS mask'
*/
typedef struct reg_mss_sei_mask {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_mss_sei_mask;
static_assert((sizeof(struct reg_mss_sei_mask) == 4), "reg_mss_sei_mask size is not 32-bit");
/*
 MSS_SEI_CAUSE 
 b'MSS: SEI INTERRUPTS status'
*/
typedef struct reg_mss_sei_cause {
	union {
		struct {
			uint32_t i0 : 1,
				i1 : 1,
				i2 : 1,
				i3 : 1,
				i4 : 1,
				i5 : 1,
				i6 : 1,
				i7 : 1,
				i8 : 1,
				i9 : 1,
				i10 : 1,
				i11 : 1,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_mss_sei_cause;
static_assert((sizeof(struct reg_mss_sei_cause) == 4), "reg_mss_sei_cause size is not 32-bit");
/*
 MSS_SPI_CLEAR 
 b'MSS: Clear SPI Interrupt Status'
*/
typedef struct reg_mss_spi_clear {
	union {
		struct {
			uint32_t i0 : 1,
				i1 : 1,
				i2 : 1,
				i3 : 1,
				i4 : 1,
				i5 : 1,
				i6 : 1,
				i7 : 1,
				i8 : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_mss_spi_clear;
static_assert((sizeof(struct reg_mss_spi_clear) == 4), "reg_mss_spi_clear size is not 32-bit");
/*
 MSS_SPI_MASK 
 b'MSS: SPI INTERRUPTS mask'
*/
typedef struct reg_mss_spi_mask {
	union {
		struct {
			uint32_t val : 9,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_mss_spi_mask;
static_assert((sizeof(struct reg_mss_spi_mask) == 4), "reg_mss_spi_mask size is not 32-bit");
/*
 MSS_SPI_CAUSE 
 b'MSS: SPI INTERRUPTS status'
*/
typedef struct reg_mss_spi_cause {
	union {
		struct {
			uint32_t i0 : 1,
				i1 : 1,
				i2 : 1,
				i3 : 1,
				i4 : 1,
				i5 : 1,
				i6 : 1,
				i7 : 1,
				i8 : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_mss_spi_cause;
static_assert((sizeof(struct reg_mss_spi_cause) == 4), "reg_mss_spi_cause size is not 32-bit");
/*
 DBG_CONTEXT_ID_CONTROL 
 b'MSS: 1Hot Error type selection crtl for Context ID'
*/
typedef struct reg_dbg_context_id_control {
	union {
		struct {
			uint32_t rsb_rr_error_sel : 1,
				rsb_num_error_sel : 1,
				rsb_slv_error_sel : 1,
				mrsb_rr_error_sel : 1,
				mrsb_num_error_sel : 1,
				mrsb_slv_error_sel : 1,
				grsb_rr_error_sel : 1,
				grsb_num_error_sel : 1,
				grsb_slv_error_sel : 1,
				wch_ch0_rr_error_sel : 1,
				wch_ch0_pinf_error_sel : 1,
				wch_ch0_ninf_error_sel : 1,
				wch_ch0_nan_error_sel : 1,
				wch_ch0_slv_error_sel : 1,
				wch_ch1_rr_error_sel : 1,
				wch_ch1_pinf_error_sel : 1,
				wch_ch1_ninf_error_sel : 1,
				wch_ch1_nan_error_sel : 1,
				wch_ch1_slv_error_sel : 1,
				rinterp_pinf_error_sel : 1,
				rinterp_ninf_error_sel : 1,
				rinterp_nan_error_sel : 1,
				minterp_pinf_error_sel : 1,
				minterp_ninf_error_sel : 1,
				minterp_nan_error_sel : 1,
				coord_pinf_error_sel : 1,
				coord_ninf_error_sel : 1,
				coord_nan_error_sel : 1,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_dbg_context_id_control;
static_assert((sizeof(struct reg_dbg_context_id_control) == 4), "reg_dbg_context_id_control size is not 32-bit");
/*
 RSB_ERR_STATUS 
 b'MSS: RSB Error Status'
*/
typedef struct reg_rsb_err_status {
	union {
		struct {
			uint32_t rsb_rr_error : 1,
				rsb_num_error : 1,
				rsb_slv_error : 1,
				mrsb_rr_error : 1,
				mrsb_num_error : 1,
				mrsb_slv_error : 1,
				grsb_rr_error : 1,
				grsb_num_error : 1,
				grsb_slv_error : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_rsb_err_status;
static_assert((sizeof(struct reg_rsb_err_status) == 4), "reg_rsb_err_status size is not 32-bit");
/*
 GRSB_MRSB_ERR_CONTEXT_ID 
 b'MSS: last context id of GRSB and MRSB Error'
*/
typedef struct reg_grsb_mrsb_err_context_id {
	union {
		struct {
			uint32_t grsb : 16,
				mrsb : 16;
		};
		uint32_t _raw;
	};
} reg_grsb_mrsb_err_context_id;
static_assert((sizeof(struct reg_grsb_mrsb_err_context_id) == 4), "reg_grsb_mrsb_err_context_id size is not 32-bit");
/*
 RSB_ERR_CONTEXT_ID 
 b'MSS: Captures last context id of RSB Error'
*/
typedef struct reg_rsb_err_context_id {
	union {
		struct {
			uint32_t rsb : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_rsb_err_context_id;
static_assert((sizeof(struct reg_rsb_err_context_id) == 4), "reg_rsb_err_context_id size is not 32-bit");
/*
 WCH_ERR_STATUS 
 b'MSS: WCH Error Status'
*/
typedef struct reg_wch_err_status {
	union {
		struct {
			uint32_t wch_ch0_rr_error : 1,
				wch_ch0_pinf_error : 1,
				wch_ch0_ninf_error : 1,
				wch_ch0_nan_error : 1,
				wch_ch0_slv_error : 1,
				wch_ch1_rr_error : 1,
				wch_ch1_pinf_error : 1,
				wch_ch1_ninf_error : 1,
				wch_ch1_nan_error : 1,
				wch_ch1_slv_error : 1,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_wch_err_status;
static_assert((sizeof(struct reg_wch_err_status) == 4), "reg_wch_err_status size is not 32-bit");
/*
 WCH_ERR_CONTEXT_ID 
 b'MSS: Last context id of Channel WCH Error'
*/
typedef struct reg_wch_err_context_id {
	union {
		struct {
			uint32_t channel0 : 16,
				channel1 : 16;
		};
		uint32_t _raw;
	};
} reg_wch_err_context_id;
static_assert((sizeof(struct reg_wch_err_context_id) == 4), "reg_wch_err_context_id size is not 32-bit");
/*
 IP_NUM_ERR_STATUS 
 b'MSS: IP Numerical Error Status'
*/
typedef struct reg_ip_num_err_status {
	union {
		struct {
			uint32_t rinterp_pinf_error : 1,
				rinterp_ninf_error : 1,
				rinterp_nan_error : 1,
				minterp_pinf_error : 1,
				minterp_ninf_error : 1,
				minterp_nan_error : 1,
				coord_pinf_error : 1,
				coord_ninf_error : 1,
				coord_nan_error : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_ip_num_err_status;
static_assert((sizeof(struct reg_ip_num_err_status) == 4), "reg_ip_num_err_status size is not 32-bit");
/*
 RINTERP_MINTERP_NUM_ERR_CONTEXT_ID 
 b'MSS: Last context id of ROT & Mesh Num Error'
*/
typedef struct reg_rinterp_minterp_num_err_context_id {
	union {
		struct {
			uint32_t rinterp : 16,
				minterp : 16;
		};
		uint32_t _raw;
	};
} reg_rinterp_minterp_num_err_context_id;
static_assert((sizeof(struct reg_rinterp_minterp_num_err_context_id) == 4), "reg_rinterp_minterp_num_err_context_id size is not 32-bit");
/*
 COORD_NUM_ERR_CONTEXT_ID 
 b'MSS: Captures last context id of Co-ord Num Error'
*/
typedef struct reg_coord_num_err_context_id {
	union {
		struct {
			uint32_t coord : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_coord_num_err_context_id;
static_assert((sizeof(struct reg_coord_num_err_context_id) == 4), "reg_coord_num_err_context_id size is not 32-bit");
/*
 MSS_ROT_APB_AXI2APB_CFG 
 b'MSS : Response control knobs of ROT_APB AXI2APB'
*/
typedef struct reg_mss_rot_apb_axi2apb_cfg {
	union {
		struct {
			uint32_t msk_error_rsp : 1,
				block_partial_strb : 1,
				spl_error_rsp : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_mss_rot_apb_axi2apb_cfg;
static_assert((sizeof(struct reg_mss_rot_apb_axi2apb_cfg) == 4), "reg_mss_rot_apb_axi2apb_cfg size is not 32-bit");
/*
 CORE_CFG 
 b'CORE : Knobs  for ROT_CORE configurations'
*/
typedef struct reg_core_cfg {
	union {
		struct {
			uint32_t nm1_en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_core_cfg;
static_assert((sizeof(struct reg_core_cfg) == 4), "reg_core_cfg size is not 32-bit");
/*
 ROT_MAX_OS 
 b'CORE: ROT IRM/RSB max outstanding'
*/
typedef struct reg_rot_max_os {
	union {
		struct {
			uint32_t val : 11,
				_reserved16 : 5,
				throttle_val : 11,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_rot_max_os;
static_assert((sizeof(struct reg_rot_max_os) == 4), "reg_rot_max_os size is not 32-bit");
/*
 MESH_MAX_OS 
 b'CORE: Mesh IRM/RSB max outstanding'
*/
typedef struct reg_mesh_max_os {
	union {
		struct {
			uint32_t val : 11,
				_reserved16 : 5,
				throttle_val : 11,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_mesh_max_os;
static_assert((sizeof(struct reg_mesh_max_os) == 4), "reg_mesh_max_os size is not 32-bit");
/*
 GRAD_MAX_OS 
 b'CORE: Grad IRM/RSB max outstanding'
*/
typedef struct reg_grad_max_os {
	union {
		struct {
			uint32_t val : 11,
				_reserved16 : 5,
				throttle_val : 11,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_grad_max_os;
static_assert((sizeof(struct reg_grad_max_os) == 4), "reg_grad_max_os size is not 32-bit");
/*
 CFG_RS_TRC 
 b'CORE: RTA, RSRA trace data events enable and freq'
*/
typedef struct reg_cfg_rs_trc {
	union {
		struct {
			uint32_t freq : 16,
				en : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_cfg_rs_trc;
static_assert((sizeof(struct reg_cfg_rs_trc) == 4), "reg_cfg_rs_trc size is not 32-bit");
/*
 RSB_LL_THRESHOLD 
 b'CORE: RSB LL memory data occupancy threshold'
*/
typedef struct reg_rsb_ll_threshold {
	union {
		struct {
			uint32_t val : 10,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_rsb_ll_threshold;
static_assert((sizeof(struct reg_rsb_ll_threshold) == 4), "reg_rsb_ll_threshold size is not 32-bit");
/*
 NUM_CONV_CFG 
 b'CORE: Numeric Conversion Cfg'
*/
typedef struct reg_num_conv_cfg {
	union {
		struct {
			uint32_t clip_fp : 1,
				clip_input_inf : 1,
				ftz_en : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_num_conv_cfg;
static_assert((sizeof(struct reg_num_conv_cfg) == 4), "reg_num_conv_cfg size is not 32-bit");
/*
 PLRU 
 b'CORE: PLRU eviction mode'
*/
typedef struct reg_plru {
	union {
		struct {
			uint32_t mode : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_plru;
static_assert((sizeof(struct reg_plru) == 4), "reg_plru size is not 32-bit");

#ifdef __cplusplus
} /* rotator namespace */
#endif

/*
 ROTATOR block
*/

#ifdef __cplusplus

struct block_rotator {
	struct rotator::reg_cpl_msg_axi cpl_msg_axi;
	struct rotator::reg_awprot_0_cpl_msg awprot_0_cpl_msg;
	struct rotator::reg_awprot_1_cpl_msg awprot_1_cpl_msg;
	struct rotator::reg_awprot_2_cpl_msg awprot_2_cpl_msg;
	struct rotator::reg_cpl_msg_threshold cpl_msg_threshold;
	uint32_t _pad20[11];
	struct rotator::reg_rsb_cfg rsb_cfg;
	struct rotator::reg_arprot_0_rsb arprot_0_rsb;
	struct rotator::reg_arprot_1_rsb arprot_1_rsb;
	struct rotator::reg_arprot_2_rsb arprot_2_rsb;
	struct rotator::reg_arprot_0_mrsb arprot_0_mrsb;
	struct rotator::reg_arprot_1_mrsb arprot_1_mrsb;
	struct rotator::reg_arprot_2_mrsb arprot_2_mrsb;
	struct rotator::reg_arprot_0_grsb arprot_0_grsb;
	struct rotator::reg_arprot_1_grsb arprot_1_grsb;
	struct rotator::reg_arprot_2_grsb arprot_2_grsb;
	struct rotator::reg_disable_pad_calc disable_pad_calc;
	struct rotator::reg_rsb_cam_max_size rsb_cam_max_size;
	struct rotator::reg_rsb_max_os rsb_max_os;
	struct rotator::reg_rsb_rl rsb_rl;
	struct rotator::reg_mrsb_rl mrsb_rl;
	struct rotator::reg_grsb_rl grsb_rl;
	struct rotator::reg_rsb_inflights rsb_inflights;
	struct rotator::reg_rsb_occupancy rsb_occupancy;
	struct rotator::reg_rsb_occupancy_data rsb_occupancy_data;
	struct rotator::reg_mrsb_occupancy_data mrsb_occupancy_data;
	struct rotator::reg_grsb_occupancy_data grsb_occupancy_data;
	struct rotator::reg_rsb_occupancy_md rsb_occupancy_md;
	struct rotator::reg_mrsb_occupancy_md mrsb_occupancy_md;
	struct rotator::reg_grsb_occupancy_md grsb_occupancy_md;
	struct rotator::reg_rsb_info rsb_info;
	uint32_t _pad164[7];
	struct rotator::reg_wbc_ctrl wbc_ctrl;
	struct rotator::reg_wch_agg_cfg wch_agg_cfg;
	struct rotator::reg_awprot_0_wch_rot awprot_0_wch_rot;
	struct rotator::reg_awprot_1_wch_rot awprot_1_wch_rot;
	struct rotator::reg_awprot_2_wch_rot awprot_2_wch_rot;
	struct rotator::reg_awprot_0_wch_rsl awprot_0_wch_rsl;
	struct rotator::reg_awprot_1_wch_rsl awprot_1_wch_rsl;
	struct rotator::reg_awprot_2_wch_rsl awprot_2_wch_rsl;
	struct rotator::reg_wbc_max_outstanding wbc_max_outstanding;
	struct rotator::reg_wbc_rl wbc_rl;
	struct rotator::reg_wbc_rl_rst_token wbc_rl_rst_token;
	struct rotator::reg_wbc_inflights wbc_inflights;
	struct rotator::reg_wbc_info wbc_info;
	struct rotator::reg_wbc_mon wbc_mon;
	uint32_t _pad248[18];
	struct rotator::reg_clk_en clk_en;
	struct rotator::reg_qman_cfg qman_cfg;
	struct rotator::reg_err_cfg err_cfg;
	struct rotator::reg_mss_halt mss_halt;
	struct rotator::reg_mss_sts mss_sts;
	struct rotator::reg_mss_sei_clear mss_sei_clear;
	struct rotator::reg_mss_sei_mask mss_sei_mask;
	struct rotator::reg_mss_sei_cause mss_sei_cause;
	struct rotator::reg_mss_spi_clear mss_spi_clear;
	struct rotator::reg_mss_spi_mask mss_spi_mask;
	struct rotator::reg_mss_spi_cause mss_spi_cause;
	struct rotator::reg_dbg_context_id_control dbg_context_id_control;
	struct rotator::reg_rsb_err_status rsb_err_status;
	struct rotator::reg_grsb_mrsb_err_context_id grsb_mrsb_err_context_id;
	struct rotator::reg_rsb_err_context_id rsb_err_context_id;
	struct rotator::reg_wch_err_status wch_err_status;
	struct rotator::reg_wch_err_context_id wch_err_context_id;
	struct rotator::reg_ip_num_err_status ip_num_err_status;
	struct rotator::reg_rinterp_minterp_num_err_context_id rinterp_minterp_num_err_context_id;
	struct rotator::reg_coord_num_err_context_id coord_num_err_context_id;
	struct rotator::reg_mss_rot_apb_axi2apb_cfg mss_rot_apb_axi2apb_cfg;
	uint32_t _pad404[11];
	struct rotator::reg_core_cfg core_cfg;
	struct rotator::reg_rot_max_os rot_max_os;
	struct rotator::reg_mesh_max_os mesh_max_os;
	struct rotator::reg_grad_max_os grad_max_os;
	struct rotator::reg_cfg_rs_trc cfg_rs_trc;
	struct rotator::reg_rsb_ll_threshold rsb_ll_threshold;
	struct rotator::reg_num_conv_cfg num_conv_cfg;
	struct rotator::reg_plru plru;
	uint32_t _pad480[8];
	struct block_rot_desc desc;
	uint32_t _pad928[88];
	struct block_axuser_hbw rot_axuser_hbw;
	uint32_t _pad1372[41];
	struct block_axuser_lbw rot_axuser_lbw;
	uint32_t _pad1560[538];
	struct block_special_regs special;
};
#else

typedef struct block_rotator {
	reg_cpl_msg_axi cpl_msg_axi;
	reg_awprot_0_cpl_msg awprot_0_cpl_msg;
	reg_awprot_1_cpl_msg awprot_1_cpl_msg;
	reg_awprot_2_cpl_msg awprot_2_cpl_msg;
	reg_cpl_msg_threshold cpl_msg_threshold;
	uint32_t _pad20[11];
	reg_rsb_cfg rsb_cfg;
	reg_arprot_0_rsb arprot_0_rsb;
	reg_arprot_1_rsb arprot_1_rsb;
	reg_arprot_2_rsb arprot_2_rsb;
	reg_arprot_0_mrsb arprot_0_mrsb;
	reg_arprot_1_mrsb arprot_1_mrsb;
	reg_arprot_2_mrsb arprot_2_mrsb;
	reg_arprot_0_grsb arprot_0_grsb;
	reg_arprot_1_grsb arprot_1_grsb;
	reg_arprot_2_grsb arprot_2_grsb;
	reg_disable_pad_calc disable_pad_calc;
	reg_rsb_cam_max_size rsb_cam_max_size;
	reg_rsb_max_os rsb_max_os;
	reg_rsb_rl rsb_rl;
	reg_mrsb_rl mrsb_rl;
	reg_grsb_rl grsb_rl;
	reg_rsb_inflights rsb_inflights;
	reg_rsb_occupancy rsb_occupancy;
	reg_rsb_occupancy_data rsb_occupancy_data;
	reg_mrsb_occupancy_data mrsb_occupancy_data;
	reg_grsb_occupancy_data grsb_occupancy_data;
	reg_rsb_occupancy_md rsb_occupancy_md;
	reg_mrsb_occupancy_md mrsb_occupancy_md;
	reg_grsb_occupancy_md grsb_occupancy_md;
	reg_rsb_info rsb_info;
	uint32_t _pad164[7];
	reg_wbc_ctrl wbc_ctrl;
	reg_wch_agg_cfg wch_agg_cfg;
	reg_awprot_0_wch_rot awprot_0_wch_rot;
	reg_awprot_1_wch_rot awprot_1_wch_rot;
	reg_awprot_2_wch_rot awprot_2_wch_rot;
	reg_awprot_0_wch_rsl awprot_0_wch_rsl;
	reg_awprot_1_wch_rsl awprot_1_wch_rsl;
	reg_awprot_2_wch_rsl awprot_2_wch_rsl;
	reg_wbc_max_outstanding wbc_max_outstanding;
	reg_wbc_rl wbc_rl;
	reg_wbc_rl_rst_token wbc_rl_rst_token;
	reg_wbc_inflights wbc_inflights;
	reg_wbc_info wbc_info;
	reg_wbc_mon wbc_mon;
	uint32_t _pad248[18];
	reg_clk_en clk_en;
	reg_qman_cfg qman_cfg;
	reg_err_cfg err_cfg;
	reg_mss_halt mss_halt;
	reg_mss_sts mss_sts;
	reg_mss_sei_clear mss_sei_clear;
	reg_mss_sei_mask mss_sei_mask;
	reg_mss_sei_cause mss_sei_cause;
	reg_mss_spi_clear mss_spi_clear;
	reg_mss_spi_mask mss_spi_mask;
	reg_mss_spi_cause mss_spi_cause;
	reg_dbg_context_id_control dbg_context_id_control;
	reg_rsb_err_status rsb_err_status;
	reg_grsb_mrsb_err_context_id grsb_mrsb_err_context_id;
	reg_rsb_err_context_id rsb_err_context_id;
	reg_wch_err_status wch_err_status;
	reg_wch_err_context_id wch_err_context_id;
	reg_ip_num_err_status ip_num_err_status;
	reg_rinterp_minterp_num_err_context_id rinterp_minterp_num_err_context_id;
	reg_coord_num_err_context_id coord_num_err_context_id;
	reg_mss_rot_apb_axi2apb_cfg mss_rot_apb_axi2apb_cfg;
	uint32_t _pad404[11];
	reg_core_cfg core_cfg;
	reg_rot_max_os rot_max_os;
	reg_mesh_max_os mesh_max_os;
	reg_grad_max_os grad_max_os;
	reg_cfg_rs_trc cfg_rs_trc;
	reg_rsb_ll_threshold rsb_ll_threshold;
	reg_num_conv_cfg num_conv_cfg;
	reg_plru plru;
	uint32_t _pad480[8];
	block_rot_desc desc;
	uint32_t _pad928[88];
	block_axuser_hbw rot_axuser_hbw;
	uint32_t _pad1372[41];
	block_axuser_lbw rot_axuser_lbw;
	uint32_t _pad1560[538];
	block_special_regs special;
} block_rotator;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_rotator_defaults[] =
{
	// offset	// value
	{ 0x10  , 0x10                , 1 }, // cpl_msg_threshold
	{ 0x40  , 0x10                , 1 }, // rsb_cfg
	{ 0xc0  , 0x101               , 1 }, // wbc_ctrl
	{ 0xc4  , 0x8ffffa            , 1 }, // wch_agg_cfg
	{ 0x16c , 0xfffffff           , 1 }, // dbg_context_id_control
	{ 0x1c0 , 0x1                 , 1 }, // core_cfg
	{ 0x1c4 , 0x50320             , 1 }, // rot_max_os
	{ 0x1c8 , 0x50320             , 1 }, // mesh_max_os
	{ 0x1cc , 0x50100             , 1 }, // grad_max_os
	{ 0x1d0 , 0x400               , 1 }, // cfg_rs_trc
	{ 0x1d4 , 0x1f0               , 1 }, // rsb_ll_threshold
	{ 0x290 , 0x8030000           , 1 }, // ctrl_cfg
	{ 0x294 , 0xffff              , 1 }, // pixel_pad
	{ 0x298 , 0xf00               , 1 }, // prec_shift
	{ 0x29c , 0xffff              , 1 }, // max_val
	{ 0x2fc , 0x30808             , 1 }, // buf_cfg
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_ROTATOR_H_ */

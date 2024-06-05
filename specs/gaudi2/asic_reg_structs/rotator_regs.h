/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI2_ROTATOR_H_
#define ASIC_REG_STRUCTS_GAUDI2_ROTATOR_H_

#include <stdint.h>
#include "gaudi2_types.h"
#include "rot_desc_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi2 {
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
 KMD_MODE 
 b'Rotator being operate by KMD'
*/
typedef struct reg_kmd_mode {
	union {
		struct {
			uint32_t en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_kmd_mode;
static_assert((sizeof(struct reg_kmd_mode) == 4), "reg_kmd_mode size is not 32-bit");
/*
 CPL_QUEUE_EN 
 b'HBW completion message enable'
*/
typedef struct reg_cpl_queue_en {
	union {
		struct {
			uint32_t q_en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cpl_queue_en;
static_assert((sizeof(struct reg_cpl_queue_en) == 4), "reg_cpl_queue_en size is not 32-bit");
/*
 CPL_QUEUE_ADDR_L 
 b'HBW completion message address'
*/
typedef struct reg_cpl_queue_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cpl_queue_addr_l;
static_assert((sizeof(struct reg_cpl_queue_addr_l) == 4), "reg_cpl_queue_addr_l size is not 32-bit");
/*
 CPL_QUEUE_ADDR_H 
 b'HBW completion message address'
*/
typedef struct reg_cpl_queue_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cpl_queue_addr_h;
static_assert((sizeof(struct reg_cpl_queue_addr_h) == 4), "reg_cpl_queue_addr_h size is not 32-bit");
/*
 CPL_QUEUE_DATA 
 b'HBW completion message data'
*/
typedef struct reg_cpl_queue_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cpl_queue_data;
static_assert((sizeof(struct reg_cpl_queue_data) == 4), "reg_cpl_queue_data size is not 32-bit");
/*
 CPL_QUEUE_AWUSER 
 b'HBW CPL QUEUE AWUSER value'
*/
typedef struct reg_cpl_queue_awuser {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cpl_queue_awuser;
static_assert((sizeof(struct reg_cpl_queue_awuser) == 4), "reg_cpl_queue_awuser size is not 32-bit");
/*
 CPL_QUEUE_AXI 
 b'AXI attributes for HBW completion message'
*/
typedef struct reg_cpl_queue_axi {
	union {
		struct {
			uint32_t cache : 4,
				prot : 3,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_cpl_queue_axi;
static_assert((sizeof(struct reg_cpl_queue_axi) == 4), "reg_cpl_queue_axi size is not 32-bit");
/*
 CPL_MSG_THRESHOLD 
 b'Threshold to bound LBW completion messages'
*/
typedef struct reg_cpl_msg_threshold {
	union {
		struct {
			uint32_t val : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_threshold;
static_assert((sizeof(struct reg_cpl_msg_threshold) == 4), "reg_cpl_msg_threshold size is not 32-bit");
/*
 CPL_MSG_AXI 
 b'AXI attributes for LBW completion message'
*/
typedef struct reg_cpl_msg_axi {
	union {
		struct {
			uint32_t cache : 4,
				prot : 3,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_axi;
static_assert((sizeof(struct reg_cpl_msg_axi) == 4), "reg_cpl_msg_axi size is not 32-bit");
/*
 AXI_WB 
 b'AXI attributes for WBC and SB'
*/
typedef struct reg_axi_wb {
	union {
		struct {
			uint32_t cache : 4,
				prot : 3,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_axi_wb;
static_assert((sizeof(struct reg_axi_wb) == 4), "reg_axi_wb size is not 32-bit");
/*
 ERR_CFG 
 b'Enable Stop on Error'
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
 ERR_STATUS 
 b'Not implemented; Use SEI INTR STATUS'
*/
typedef struct reg_err_status {
	union {
		struct {
			uint32_t rot_hbw_rd : 1,
				rot_hbw_wr : 1,
				qman_hbw_rd : 1,
				qman_hbw_wr : 1,
				rot_lbw_wr : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_err_status;
static_assert((sizeof(struct reg_err_status) == 4), "reg_err_status size is not 32-bit");
/*
 WBC_MAX_OUTSTANDING 
 b'max on-flight write transactions'
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
 b'WBC rate-limiter'
*/
typedef struct reg_wbc_rl {
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
} reg_wbc_rl;
static_assert((sizeof(struct reg_wbc_rl) == 4), "reg_wbc_rl size is not 32-bit");
/*
 WBC_INFLIGHTS 
 b'in-flight transaction counter'
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
 b'Status indications'
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
 b'Performance monitor results'
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
 RSB_CAM_MAX_SIZE 
 b'Defines the effective size of suspension buffer'
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
 RSB_CFG 
 b'CFG for Input Image Read Suspension Buffer'
*/
typedef struct reg_rsb_cfg {
	union {
		struct {
			uint32_t cache_disable : 1,
				enable_cgate : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_rsb_cfg;
static_assert((sizeof(struct reg_rsb_cfg) == 4), "reg_rsb_cfg size is not 32-bit");
/*
 RSB_MAX_OS 
 b'max on-flight read transactions'
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
 b'RSB rate-limiter'
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
 RSB_INFLIGHTS 
 b'Number of Inflight trans from Inp Image SuspBuffer'
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
 b'Input Image Susp Buffer Occupancy'
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
 RSB_INFO 
 b'Status indications'
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
 RSB_MON 
 b'Performance monitor results'
*/
typedef struct reg_rsb_mon {
	union {
		struct {
			uint32_t cnt : 13,
				_reserved16 : 3,
				ts : 2,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_rsb_mon;
static_assert((sizeof(struct reg_rsb_mon) == 4), "reg_rsb_mon size is not 32-bit");
/*
 RSB_MON_CONTEXT_ID 
 b'Performance monitor results'
*/
typedef struct reg_rsb_mon_context_id {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rsb_mon_context_id;
static_assert((sizeof(struct reg_rsb_mon_context_id) == 4), "reg_rsb_mon_context_id size is not 32-bit");
/*
 MSS_HALT 
 b'halt for sb, wbc and mesh sb'
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
 MSS_SEI_STATUS 
 b'SEI INTERRUPTS status'
*/
typedef struct reg_mss_sei_status {
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
				i12 : 1,
				i13 : 1,
				i14 : 1,
				i15 : 1,
				i16 : 1,
				i17 : 1,
				i18 : 1,
				i19 : 1,
				i20 : 1,
				i21 : 1,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_mss_sei_status;
static_assert((sizeof(struct reg_mss_sei_status) == 4), "reg_mss_sei_status size is not 32-bit");
/*
 MSS_SEI_MASK 
 b'SEI INTERRUPTS mask'
*/
typedef struct reg_mss_sei_mask {
	union {
		struct {
			uint32_t val : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_mss_sei_mask;
static_assert((sizeof(struct reg_mss_sei_mask) == 4), "reg_mss_sei_mask size is not 32-bit");
/*
 MSS_SPI_STATUS 
 b'SPI INTERRUPTS status'
*/
typedef struct reg_mss_spi_status {
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
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_mss_spi_status;
static_assert((sizeof(struct reg_mss_spi_status) == 4), "reg_mss_spi_status size is not 32-bit");
/*
 MSS_SPI_MASK 
 b'SPI INTERRUPTS mask'
*/
typedef struct reg_mss_spi_mask {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_mss_spi_mask;
static_assert((sizeof(struct reg_mss_spi_mask) == 4), "reg_mss_spi_mask size is not 32-bit");
/*
 DISABLE_PAD_CALC 
 b'Disable pad calc for Inp Image & Mesh Susp Buffer'
*/
typedef struct reg_disable_pad_calc {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_disable_pad_calc;
static_assert((sizeof(struct reg_disable_pad_calc) == 4), "reg_disable_pad_calc size is not 32-bit");
/*
 QMAN_CFG 
 b'Force Stop'
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
 CLK_EN 
 b'static cfg or clk ungating & SB status mask'
*/
typedef struct reg_clk_en {
	union {
		struct {
			uint32_t lbw_cfg_dis : 1,
				_reserved4 : 3,
				dbg_cfg_dis : 1,
				sb_empty_mask : 1,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_clk_en;
static_assert((sizeof(struct reg_clk_en) == 4), "reg_clk_en size is not 32-bit");
/*
 MRSB_CAM_MAX_SIZE 
 b'Defines the effective size of Mesh SuspBuffer'
*/
typedef struct reg_mrsb_cam_max_size {
	union {
		struct {
			uint32_t data : 16,
				md : 16;
		};
		uint32_t _raw;
	};
} reg_mrsb_cam_max_size;
static_assert((sizeof(struct reg_mrsb_cam_max_size) == 4), "reg_mrsb_cam_max_size size is not 32-bit");
/*
 MRSB_CFG 
 b'Static CFG for Mesh Image Read Suspension Buffer'
*/
typedef struct reg_mrsb_cfg {
	union {
		struct {
			uint32_t cache_disable : 1,
				enable_cgate : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_mrsb_cfg;
static_assert((sizeof(struct reg_mrsb_cfg) == 4), "reg_mrsb_cfg size is not 32-bit");
/*
 MRSB_MAX_OS 
 b'max on-flight read transactions'
*/
typedef struct reg_mrsb_max_os {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_mrsb_max_os;
static_assert((sizeof(struct reg_mrsb_max_os) == 4), "reg_mrsb_max_os size is not 32-bit");
/*
 MRSB_RL 
 b'RSB rate-limiter'
*/
typedef struct reg_mrsb_rl {
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
} reg_mrsb_rl;
static_assert((sizeof(struct reg_mrsb_rl) == 4), "reg_mrsb_rl size is not 32-bit");
/*
 MRSB_INFLIGHTS 
 b'Number of Inflight trans from Mesh SuspBuffer'
*/
typedef struct reg_mrsb_inflights {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mrsb_inflights;
static_assert((sizeof(struct reg_mrsb_inflights) == 4), "reg_mrsb_inflights size is not 32-bit");
/*
 MRSB_OCCUPANCY 
 b'Mesh Susp Buffer Occupancy'
*/
typedef struct reg_mrsb_occupancy {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mrsb_occupancy;
static_assert((sizeof(struct reg_mrsb_occupancy) == 4), "reg_mrsb_occupancy size is not 32-bit");
/*
 MRSB_INFO 
 b'Status indications'
*/
typedef struct reg_mrsb_info {
	union {
		struct {
			uint32_t empty : 1,
				axi_idle : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_mrsb_info;
static_assert((sizeof(struct reg_mrsb_info) == 4), "reg_mrsb_info size is not 32-bit");
/*
 MRSB_MON 
 b'Performance monitor results'
*/
typedef struct reg_mrsb_mon {
	union {
		struct {
			uint32_t cnt : 13,
				_reserved16 : 3,
				ts : 2,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_mrsb_mon;
static_assert((sizeof(struct reg_mrsb_mon) == 4), "reg_mrsb_mon size is not 32-bit");
/*
 MRSB_MON_CONTEXT_ID 
 b'Performance monitor results'
*/
typedef struct reg_mrsb_mon_context_id {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mrsb_mon_context_id;
static_assert((sizeof(struct reg_mrsb_mon_context_id) == 4), "reg_mrsb_mon_context_id size is not 32-bit");
/*
 MSS_STS 
 b'MSS halt status'
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

#ifdef __cplusplus
} /* rotator namespace */
#endif

/*
 ROTATOR block
*/

#ifdef __cplusplus

struct block_rotator {
	struct rotator::reg_kmd_mode kmd_mode;
	struct rotator::reg_cpl_queue_en cpl_queue_en;
	struct rotator::reg_cpl_queue_addr_l cpl_queue_addr_l;
	struct rotator::reg_cpl_queue_addr_h cpl_queue_addr_h;
	struct rotator::reg_cpl_queue_data cpl_queue_data;
	struct rotator::reg_cpl_queue_awuser cpl_queue_awuser;
	struct rotator::reg_cpl_queue_axi cpl_queue_axi;
	uint32_t _pad28[1];
	struct rotator::reg_cpl_msg_threshold cpl_msg_threshold;
	struct rotator::reg_cpl_msg_axi cpl_msg_axi;
	struct rotator::reg_axi_wb axi_wb;
	struct rotator::reg_err_cfg err_cfg;
	struct rotator::reg_err_status err_status;
	uint32_t _pad52[1];
	struct rotator::reg_wbc_max_outstanding wbc_max_outstanding;
	struct rotator::reg_wbc_rl wbc_rl;
	struct rotator::reg_wbc_inflights wbc_inflights;
	struct rotator::reg_wbc_info wbc_info;
	struct rotator::reg_wbc_mon wbc_mon;
	struct rotator::reg_rsb_cam_max_size rsb_cam_max_size;
	struct rotator::reg_rsb_cfg rsb_cfg;
	struct rotator::reg_rsb_max_os rsb_max_os;
	struct rotator::reg_rsb_rl rsb_rl;
	struct rotator::reg_rsb_inflights rsb_inflights;
	struct rotator::reg_rsb_occupancy rsb_occupancy;
	struct rotator::reg_rsb_info rsb_info;
	struct rotator::reg_rsb_mon rsb_mon;
	struct rotator::reg_rsb_mon_context_id rsb_mon_context_id;
	struct rotator::reg_mss_halt mss_halt;
	struct rotator::reg_mss_sei_status mss_sei_status;
	struct rotator::reg_mss_sei_mask mss_sei_mask;
	struct rotator::reg_mss_spi_status mss_spi_status;
	struct rotator::reg_mss_spi_mask mss_spi_mask;
	struct rotator::reg_disable_pad_calc disable_pad_calc;
	struct rotator::reg_qman_cfg qman_cfg;
	struct rotator::reg_clk_en clk_en;
	struct rotator::reg_mrsb_cam_max_size mrsb_cam_max_size;
	struct rotator::reg_mrsb_cfg mrsb_cfg;
	struct rotator::reg_mrsb_max_os mrsb_max_os;
	struct rotator::reg_mrsb_rl mrsb_rl;
	struct rotator::reg_mrsb_inflights mrsb_inflights;
	struct rotator::reg_mrsb_occupancy mrsb_occupancy;
	struct rotator::reg_mrsb_info mrsb_info;
	struct rotator::reg_mrsb_mon mrsb_mon;
	struct rotator::reg_mrsb_mon_context_id mrsb_mon_context_id;
	struct rotator::reg_mss_sts mss_sts;
	uint32_t _pad184[18];
	struct block_rot_desc desc;
	uint32_t _pad520[798];
	struct block_special_regs special;
};
#else

typedef struct block_rotator {
	reg_kmd_mode kmd_mode;
	reg_cpl_queue_en cpl_queue_en;
	reg_cpl_queue_addr_l cpl_queue_addr_l;
	reg_cpl_queue_addr_h cpl_queue_addr_h;
	reg_cpl_queue_data cpl_queue_data;
	reg_cpl_queue_awuser cpl_queue_awuser;
	reg_cpl_queue_axi cpl_queue_axi;
	uint32_t _pad28[1];
	reg_cpl_msg_threshold cpl_msg_threshold;
	reg_cpl_msg_axi cpl_msg_axi;
	reg_axi_wb axi_wb;
	reg_err_cfg err_cfg;
	reg_err_status err_status;
	uint32_t _pad52[1];
	reg_wbc_max_outstanding wbc_max_outstanding;
	reg_wbc_rl wbc_rl;
	reg_wbc_inflights wbc_inflights;
	reg_wbc_info wbc_info;
	reg_wbc_mon wbc_mon;
	reg_rsb_cam_max_size rsb_cam_max_size;
	reg_rsb_cfg rsb_cfg;
	reg_rsb_max_os rsb_max_os;
	reg_rsb_rl rsb_rl;
	reg_rsb_inflights rsb_inflights;
	reg_rsb_occupancy rsb_occupancy;
	reg_rsb_info rsb_info;
	reg_rsb_mon rsb_mon;
	reg_rsb_mon_context_id rsb_mon_context_id;
	reg_mss_halt mss_halt;
	reg_mss_sei_status mss_sei_status;
	reg_mss_sei_mask mss_sei_mask;
	reg_mss_spi_status mss_spi_status;
	reg_mss_spi_mask mss_spi_mask;
	reg_disable_pad_calc disable_pad_calc;
	reg_qman_cfg qman_cfg;
	reg_clk_en clk_en;
	reg_mrsb_cam_max_size mrsb_cam_max_size;
	reg_mrsb_cfg mrsb_cfg;
	reg_mrsb_max_os mrsb_max_os;
	reg_mrsb_rl mrsb_rl;
	reg_mrsb_inflights mrsb_inflights;
	reg_mrsb_occupancy mrsb_occupancy;
	reg_mrsb_info mrsb_info;
	reg_mrsb_mon mrsb_mon;
	reg_mrsb_mon_context_id mrsb_mon_context_id;
	reg_mss_sts mss_sts;
	uint32_t _pad184[18];
	block_rot_desc desc;
	uint32_t _pad520[798];
	block_special_regs special;
} block_rotator;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_rotator_defaults[] =
{
	// offset	// value
	{ 0x20  , 0x10                , 1 }, // cpl_msg_threshold
	{ 0x4c  , 0x6b0400            , 1 }, // rsb_cam_max_size
	{ 0x78  , 0x1fe000            , 1 }, // mss_sei_mask
	{ 0x90  , 0x6b0400            , 1 }, // mrsb_cam_max_size
	{ 0x190 , 0x8030000           , 1 }, // ctrl_cfg
	{ 0x194 , 0xffff              , 1 }, // pixel_pad
	{ 0x198 , 0xf00               , 1 }, // prec_shift
	{ 0x19c , 0xffff              , 1 }, // max_val
	{ 0x1fc , 0x30808             , 1 }, // buf_cfg
	{ 0xe80 , 0xffffffff          , 32 }, // glbl_priv
	{ 0xf24 , 0xffff              , 1 }, // mem_ecc_err_addr
	{ 0xf44 , 0xffffffff          , 1 }, // glbl_err_addr
	{ 0xf80 , 0xffffffff          , 32 }, // glbl_sec
};
#endif

#ifdef __cplusplus
} /* gaudi2 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI2_ROTATOR_H_ */

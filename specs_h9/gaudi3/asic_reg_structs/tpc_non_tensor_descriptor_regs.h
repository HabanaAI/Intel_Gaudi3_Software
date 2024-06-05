/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc_non_tensor_descriptor {
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
 KERNEL_BASE_ADDRESS_LOW 
 b'lower 32 bits of the kernel base address'
*/
typedef struct reg_kernel_base_address_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_kernel_base_address_low;
static_assert((sizeof(struct reg_kernel_base_address_low) == 4), "reg_kernel_base_address_low size is not 32-bit");
/*
 KERNEL_BASE_ADDRESS_HIGH 
 b'higher 32 bits of the kernel base address'
*/
typedef struct reg_kernel_base_address_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_kernel_base_address_high;
static_assert((sizeof(struct reg_kernel_base_address_high) == 4), "reg_kernel_base_address_high size is not 32-bit");
/*
 KERNEL_CONFIG 
 b'Kernel configurations'
*/
typedef struct reg_kernel_config {
	union {
		struct {
			uint32_t small_vlm : 1,
				aso_evict_l0 : 1,
				num_valid_srfs : 7,
				rd_rate_limit_rst_token : 8,
				wr_rate_limit_rst_token : 8,
				irf_32bit_compatibilty : 1,
				ccl_dis : 1,
				irf_dim0_last_48bit_mode : 1,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_kernel_config;
static_assert((sizeof(struct reg_kernel_config) == 4), "reg_kernel_config size is not 32-bit");
/*
 POWER_LOOP 
 b'Power loop message configurations'
*/
typedef struct reg_power_loop {
	union {
		struct {
			uint32_t start_en : 1,
				end_en : 1,
				_reserved4 : 2,
				payload : 8,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_power_loop;
static_assert((sizeof(struct reg_power_loop) == 4), "reg_power_loop size is not 32-bit");
/*
 SMT_EN 
 b'Enable SMT4'
*/
typedef struct reg_smt_en {
	union {
		struct {
			uint32_t smt_en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_smt_en;
static_assert((sizeof(struct reg_smt_en) == 4), "reg_smt_en size is not 32-bit");
/*
 QOS 
 b'QOS'
*/
typedef struct reg_qos {
	union {
		struct {
			uint32_t qos : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_qos;
static_assert((sizeof(struct reg_qos) == 4), "reg_qos size is not 32-bit");
/*
 MCID_FAST_CFG 
 b'MCID fast configuration register'
*/
typedef struct reg_mcid_fast_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				mask : 16;
		};
		uint32_t _raw;
	};
} reg_mcid_fast_cfg;
static_assert((sizeof(struct reg_mcid_fast_cfg) == 4), "reg_mcid_fast_cfg size is not 32-bit");
/*
 CLASS_FAST_CFG 
 b'CLASS fast configuration register'
*/
typedef struct reg_class_fast_cfg {
	union {
		struct {
			uint32_t clas : 2,
				mask : 16,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_class_fast_cfg;
static_assert((sizeof(struct reg_class_fast_cfg) == 4), "reg_class_fast_cfg size is not 32-bit");
/*
 KERNEL_ID 
 b'Kernel uniq ID'
*/
typedef struct reg_kernel_id {
	union {
		struct {
			uint32_t v : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_kernel_id;
static_assert((sizeof(struct reg_kernel_id) == 4), "reg_kernel_id size is not 32-bit");
/*
 KERNEL_ID_INC 
 b'kernel id increment value'
*/
typedef struct reg_kernel_id_inc {
	union {
		struct {
			uint32_t v : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_kernel_id_inc;
static_assert((sizeof(struct reg_kernel_id_inc) == 4), "reg_kernel_id_inc size is not 32-bit");
/*
 ACTIVE_THRD 
 b'Active threads'
*/
typedef struct reg_active_thrd {
	union {
		struct {
			uint32_t active_thrd : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_active_thrd;
static_assert((sizeof(struct reg_active_thrd) == 4), "reg_active_thrd size is not 32-bit");
/*
 SRF 
 b'64 SRF registers which should be preloaded'
*/
typedef struct reg_srf {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_srf;
static_assert((sizeof(struct reg_srf) == 4), "reg_srf size is not 32-bit");
/*
 ICACHE_AXI_CFG 
 b'ARCACH/MCID/CLASS for instruction cache access'
*/
typedef struct reg_icache_axi_cfg {
	union {
		struct {
			uint32_t arcach : 4,
				mcid : 16,
				clas : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_icache_axi_cfg;
static_assert((sizeof(struct reg_icache_axi_cfg) == 4), "reg_icache_axi_cfg size is not 32-bit");
/*
 LKUP_AXI_CFG 
 b'ARCACH/MCID/CLASS for instruction lookup access'
*/
typedef struct reg_lkup_axi_cfg {
	union {
		struct {
			uint32_t arcache : 4,
				mcid : 16,
				clas : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_lkup_axi_cfg;
static_assert((sizeof(struct reg_lkup_axi_cfg) == 4), "reg_lkup_axi_cfg size is not 32-bit");
/*
 DCACHE_AXI_CFG 
 b'AWCACH/MCID/CLASS for dcache flush'
*/
typedef struct reg_dcache_axi_cfg {
	union {
		struct {
			uint32_t awcache : 4,
				mcid : 16,
				clas : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_dcache_axi_cfg;
static_assert((sizeof(struct reg_dcache_axi_cfg) == 4), "reg_dcache_axi_cfg size is not 32-bit");
/*
 CLASS_L2_PREF 
 b'L2 prefetch request axuser class bits'
*/
typedef struct reg_class_l2_pref {
	union {
		struct {
			uint32_t clas : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_class_l2_pref;
static_assert((sizeof(struct reg_class_l2_pref) == 4), "reg_class_l2_pref size is not 32-bit");
/*
 CLASS_CTRL_L1_PREF_LD 
 b'L1 prefetch/load request axuser class bits'
*/
typedef struct reg_class_ctrl_l1_pref_ld {
	union {
		struct {
			uint32_t clas : 2,
				sel_class_src : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_class_ctrl_l1_pref_ld;
static_assert((sizeof(struct reg_class_ctrl_l1_pref_ld) == 4), "reg_class_ctrl_l1_pref_ld size is not 32-bit");
/*
 HALT_ZERO_SQZ 
 b'clear sqz counters on halt'
*/
typedef struct reg_halt_zero_sqz {
	union {
		struct {
			uint32_t halt_zero_sqz : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_halt_zero_sqz;
static_assert((sizeof(struct reg_halt_zero_sqz) == 4), "reg_halt_zero_sqz size is not 32-bit");
/*
 RMW_CLIP_FP 
 b'clip fp for st_tnsr_with RMW'
*/
typedef struct reg_rmw_clip_fp {
	union {
		struct {
			uint32_t rmw_clip_fp : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_rmw_clip_fp;
static_assert((sizeof(struct reg_rmw_clip_fp) == 4), "reg_rmw_clip_fp size is not 32-bit");
/*
 SYNC_MSG_MODE_SMT4 
 b'sync-message scheme - 1 : single sync message'
*/
typedef struct reg_sync_msg_mode_smt4 {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sync_msg_mode_smt4;
static_assert((sizeof(struct reg_sync_msg_mode_smt4) == 4), "reg_sync_msg_mode_smt4 size is not 32-bit");
/*
 LD_DEFAULT_HBW_AXI_CFG 
 b'defaullt axi configuration for ld_tnsr direct'
*/
typedef struct reg_ld_default_hbw_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				clas : 2,
				arcache : 4,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_ld_default_hbw_axi_cfg;
static_assert((sizeof(struct reg_ld_default_hbw_axi_cfg) == 4), "reg_ld_default_hbw_axi_cfg size is not 32-bit");
/*
 ST_DEFAULT_HBW_AXI_CFG 
 b'defaullt axi configuration for st_tnsr direct'
*/
typedef struct reg_st_default_hbw_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				clas : 2,
				awcache : 4,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_st_default_hbw_axi_cfg;
static_assert((sizeof(struct reg_st_default_hbw_axi_cfg) == 4), "reg_st_default_hbw_axi_cfg size is not 32-bit");
/*
 DCACHE_PREF_WINDOW_INIT 
 b'prefetch window init cfg'
*/
typedef struct reg_dcache_pref_window_init {
	union {
		struct {
			uint32_t l1_pref_window_init_val : 8,
				l2_pref_window_init_val : 8,
				l1_pref_window_init_en : 1,
				l2_pref_window_init_en : 1,
				_reserved18 : 14;
		};
		uint32_t _raw;
	};
} reg_dcache_pref_window_init;
static_assert((sizeof(struct reg_dcache_pref_window_init) == 4), "reg_dcache_pref_window_init size is not 32-bit");
/*
 DCACHE_PREF_DYNAMIC_WINDOW 
 b'prefetch dynamic window cfg'
*/
typedef struct reg_dcache_pref_dynamic_window {
	union {
		struct {
			uint32_t l1_pref_window_inc_val : 8,
				l2_pref_window_inc_val : 8,
				l1_pref_window_inc_en : 1,
				l2_pref_window_inc_en : 1,
				l1_pref_window_dec_en : 1,
				l2_pref_window_dec_en : 1,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_dcache_pref_dynamic_window;
static_assert((sizeof(struct reg_dcache_pref_dynamic_window) == 4), "reg_dcache_pref_dynamic_window size is not 32-bit");
/*
 DCACHE_PREF_L1_WINDOW_LIMIT 
 b'prefetch l1 window limit cfg'
*/
typedef struct reg_dcache_pref_l1_window_limit {
	union {
		struct {
			uint32_t l1_pref_window_max : 8,
				l1_pref_window_min : 8,
				l1_sum_window_limit : 8,
				l1_inc_low_th : 4,
				l1_inc_upper_th : 4;
		};
		uint32_t _raw;
	};
} reg_dcache_pref_l1_window_limit;
static_assert((sizeof(struct reg_dcache_pref_l1_window_limit) == 4), "reg_dcache_pref_l1_window_limit size is not 32-bit");
/*
 DCACHE_PREF_L2_WINDOW_LIMIT 
 b'prefetch l2 window limit cfg'
*/
typedef struct reg_dcache_pref_l2_window_limit {
	union {
		struct {
			uint32_t l2_pref_window_max : 8,
				l2_pref_window_min : 8,
				l2_sum_window_limit : 8,
				l2_inc_low_th : 4,
				l2_inc_upper_th : 4;
		};
		uint32_t _raw;
	};
} reg_dcache_pref_l2_window_limit;
static_assert((sizeof(struct reg_dcache_pref_l2_window_limit) == 4), "reg_dcache_pref_l2_window_limit size is not 32-bit");
/*
 DCACHE_STALL_LENGTH_THR 
 b'dynamic window th'
*/
typedef struct reg_dcache_stall_length_thr {
	union {
		struct {
			uint32_t pref_l1_th : 16,
				pref_l2_th : 16;
		};
		uint32_t _raw;
	};
} reg_dcache_stall_length_thr;
static_assert((sizeof(struct reg_dcache_stall_length_thr) == 4), "reg_dcache_stall_length_thr size is not 32-bit");
/*
 IRF44_SAT 
 b'IRF44 saturation enable'
*/
typedef struct reg_irf44_sat {
	union {
		struct {
			uint32_t en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_irf44_sat;
static_assert((sizeof(struct reg_irf44_sat) == 4), "reg_irf44_sat size is not 32-bit");
/*
 IRF44_SAT_LOW 
 b'IRF44 saturation value low part'
*/
typedef struct reg_irf44_sat_low {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_irf44_sat_low;
static_assert((sizeof(struct reg_irf44_sat_low) == 4), "reg_irf44_sat_low size is not 32-bit");
/*
 IRF44_SAT_HIGH 
 b'IRF44 saturation value high part'
*/
typedef struct reg_irf44_sat_high {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_irf44_sat_high;
static_assert((sizeof(struct reg_irf44_sat_high) == 4), "reg_irf44_sat_high size is not 32-bit");
/*
 TSB_ST_DIRECT_MODE 
 b'ST_TNSR*.direct mode for SB cache invalidate'
*/
typedef struct reg_tsb_st_direct_mode {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_tsb_st_direct_mode;
static_assert((sizeof(struct reg_tsb_st_direct_mode) == 4), "reg_tsb_st_direct_mode size is not 32-bit");

#ifdef __cplusplus
} /* tpc_non_tensor_descriptor namespace */
#endif

/*
 TPC_NON_TENSOR_DESCRIPTOR block
*/

#ifdef __cplusplus

struct block_tpc_non_tensor_descriptor {
	struct tpc_non_tensor_descriptor::reg_kernel_base_address_low kernel_base_address_low;
	struct tpc_non_tensor_descriptor::reg_kernel_base_address_high kernel_base_address_high;
	struct tpc_non_tensor_descriptor::reg_kernel_config kernel_config;
	struct tpc_non_tensor_descriptor::reg_power_loop power_loop;
	struct tpc_non_tensor_descriptor::reg_smt_en smt_en;
	struct tpc_non_tensor_descriptor::reg_qos qos;
	struct tpc_non_tensor_descriptor::reg_mcid_fast_cfg mcid_fast_cfg;
	struct tpc_non_tensor_descriptor::reg_class_fast_cfg class_fast_cfg;
	struct tpc_non_tensor_descriptor::reg_kernel_id kernel_id;
	struct tpc_non_tensor_descriptor::reg_kernel_id_inc kernel_id_inc;
	struct tpc_non_tensor_descriptor::reg_active_thrd active_thrd;
	struct tpc_non_tensor_descriptor::reg_srf srf[60];
	struct tpc_non_tensor_descriptor::reg_icache_axi_cfg icache_axi_cfg;
	struct tpc_non_tensor_descriptor::reg_lkup_axi_cfg lkup_axi_cfg;
	struct tpc_non_tensor_descriptor::reg_dcache_axi_cfg dcache_axi_cfg;
	struct tpc_non_tensor_descriptor::reg_class_l2_pref class_l2_pref;
	struct tpc_non_tensor_descriptor::reg_class_ctrl_l1_pref_ld class_ctrl_l1_pref_ld;
	struct tpc_non_tensor_descriptor::reg_halt_zero_sqz halt_zero_sqz;
	struct tpc_non_tensor_descriptor::reg_rmw_clip_fp rmw_clip_fp;
	struct tpc_non_tensor_descriptor::reg_sync_msg_mode_smt4 sync_msg_mode_smt4;
	struct tpc_non_tensor_descriptor::reg_ld_default_hbw_axi_cfg ld_default_hbw_axi_cfg;
	struct tpc_non_tensor_descriptor::reg_st_default_hbw_axi_cfg st_default_hbw_axi_cfg;
	struct tpc_non_tensor_descriptor::reg_dcache_pref_window_init dcache_pref_window_init;
	struct tpc_non_tensor_descriptor::reg_dcache_pref_dynamic_window dcache_pref_dynamic_window;
	struct tpc_non_tensor_descriptor::reg_dcache_pref_l1_window_limit dcache_pref_l1_window_limit;
	struct tpc_non_tensor_descriptor::reg_dcache_pref_l2_window_limit dcache_pref_l2_window_limit;
	struct tpc_non_tensor_descriptor::reg_dcache_stall_length_thr dcache_stall_length_thr;
	struct tpc_non_tensor_descriptor::reg_irf44_sat irf44_sat;
	struct tpc_non_tensor_descriptor::reg_irf44_sat_low irf44_sat_low;
	struct tpc_non_tensor_descriptor::reg_irf44_sat_high irf44_sat_high;
	struct tpc_non_tensor_descriptor::reg_tsb_st_direct_mode tsb_st_direct_mode;
};
#else

typedef struct block_tpc_non_tensor_descriptor {
	reg_kernel_base_address_low kernel_base_address_low;
	reg_kernel_base_address_high kernel_base_address_high;
	reg_kernel_config kernel_config;
	reg_power_loop power_loop;
	reg_smt_en smt_en;
	reg_qos qos;
	reg_mcid_fast_cfg mcid_fast_cfg;
	reg_class_fast_cfg class_fast_cfg;
	reg_kernel_id kernel_id;
	reg_kernel_id_inc kernel_id_inc;
	reg_active_thrd active_thrd;
	reg_srf srf[60];
	reg_icache_axi_cfg icache_axi_cfg;
	reg_lkup_axi_cfg lkup_axi_cfg;
	reg_dcache_axi_cfg dcache_axi_cfg;
	reg_class_l2_pref class_l2_pref;
	reg_class_ctrl_l1_pref_ld class_ctrl_l1_pref_ld;
	reg_halt_zero_sqz halt_zero_sqz;
	reg_rmw_clip_fp rmw_clip_fp;
	reg_sync_msg_mode_smt4 sync_msg_mode_smt4;
	reg_ld_default_hbw_axi_cfg ld_default_hbw_axi_cfg;
	reg_st_default_hbw_axi_cfg st_default_hbw_axi_cfg;
	reg_dcache_pref_window_init dcache_pref_window_init;
	reg_dcache_pref_dynamic_window dcache_pref_dynamic_window;
	reg_dcache_pref_l1_window_limit dcache_pref_l1_window_limit;
	reg_dcache_pref_l2_window_limit dcache_pref_l2_window_limit;
	reg_dcache_stall_length_thr dcache_stall_length_thr;
	reg_irf44_sat irf44_sat;
	reg_irf44_sat_low irf44_sat_low;
	reg_irf44_sat_high irf44_sat_high;
	reg_tsb_st_direct_mode tsb_st_direct_mode;
} block_tpc_non_tensor_descriptor;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_tpc_non_tensor_descriptor_defaults[] =
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
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_H_ */

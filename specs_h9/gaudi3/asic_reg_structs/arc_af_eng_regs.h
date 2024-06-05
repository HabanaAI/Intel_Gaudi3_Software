/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_ARC_AF_ENG_H_
#define ASIC_REG_STRUCTS_GAUDI3_ARC_AF_ENG_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace arc_af_eng {
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
 SB_MAX_SIZE 
*/
typedef struct reg_sb_max_size {
	union {
		struct {
			uint32_t ms_data : 16,
				ms_md : 16;
		};
		uint32_t _raw;
	};
} reg_sb_max_size;
static_assert((sizeof(struct reg_sb_max_size) == 4), "reg_sb_max_size size is not 32-bit");
/*
 SB_FORCE_MISS 
*/
typedef struct reg_sb_force_miss {
	union {
		struct {
			uint32_t fmiss : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sb_force_miss;
static_assert((sizeof(struct reg_sb_force_miss) == 4), "reg_sb_force_miss size is not 32-bit");
/*
 SB_MAX 
*/
typedef struct reg_sb_max {
	union {
		struct {
			uint32_t os : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_sb_max;
static_assert((sizeof(struct reg_sb_max) == 4), "reg_sb_max size is not 32-bit");
/*
 SB_RL 
*/
typedef struct reg_sb_rl {
	union {
		struct {
			uint32_t rate_limiter_en : 1,
				rl_saturation : 8,
				rl_timeout : 8,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_sb_rl;
static_assert((sizeof(struct reg_sb_rl) == 4), "reg_sb_rl size is not 32-bit");
/*
 SB_HALT 
 b'SB HALT'
*/
typedef struct reg_sb_halt {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sb_halt;
static_assert((sizeof(struct reg_sb_halt) == 4), "reg_sb_halt size is not 32-bit");
/*
 DCCM_HALT 
*/
typedef struct reg_dccm_halt {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_dccm_halt;
static_assert((sizeof(struct reg_dccm_halt) == 4), "reg_dccm_halt size is not 32-bit");
/*
 SB_ARCACHE 
*/
typedef struct reg_sb_arcache {
	union {
		struct {
			uint32_t arcache : 7,
				_reserved8 : 1,
				arprot : 16,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_sb_arcache;
static_assert((sizeof(struct reg_sb_arcache) == 4), "reg_sb_arcache size is not 32-bit");
/*
 SB_OCCUPANCY 
*/
typedef struct reg_sb_occupancy {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sb_occupancy;
static_assert((sizeof(struct reg_sb_occupancy) == 4), "reg_sb_occupancy size is not 32-bit");
/*
 SB_INFLIGHTS 
*/
typedef struct reg_sb_inflights {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sb_inflights;
static_assert((sizeof(struct reg_sb_inflights) == 4), "reg_sb_inflights size is not 32-bit");
/*
 SB_PROT 
*/
typedef struct reg_sb_prot {
	union {
		struct {
			uint32_t priv : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sb_prot;
static_assert((sizeof(struct reg_sb_prot) == 4), "reg_sb_prot size is not 32-bit");
/*
 SB_AXI_128BYTE 
*/
typedef struct reg_sb_axi_128byte {
	union {
		struct {
			uint32_t en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sb_axi_128byte;
static_assert((sizeof(struct reg_sb_axi_128byte) == 4), "reg_sb_axi_128byte size is not 32-bit");
/*
 AF_STATUS_REG 
 b'Auto-fetcher: Status information'
*/
typedef struct reg_af_status_reg {
	union {
		struct {
			uint32_t state : 3,
				_reserved4 : 1,
				fullness : 16,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_af_status_reg;
static_assert((sizeof(struct reg_af_status_reg) == 4), "reg_af_status_reg size is not 32-bit");
/*
 AF_INTR_CAUSE 
*/
typedef struct reg_af_intr_cause {
	union {
		struct {
			uint32_t slv_err_legacy : 1,
				slv_err : 1,
				rr_dbg : 1,
				num_err : 1,
				dccm_err : 1,
				lbw_stall : 1,
				_reserved16 : 10,
				cntx : 16;
		};
		uint32_t _raw;
	};
} reg_af_intr_cause;
static_assert((sizeof(struct reg_af_intr_cause) == 4), "reg_af_intr_cause size is not 32-bit");
/*
 AF_INTR_MASK 
*/
typedef struct reg_af_intr_mask {
	union {
		struct {
			uint32_t slv_err_legacy : 1,
				slv_err : 1,
				rr_dbg : 1,
				num_err : 1,
				dccm_err : 1,
				lbw_stall_err : 1,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_af_intr_mask;
static_assert((sizeof(struct reg_af_intr_mask) == 4), "reg_af_intr_mask size is not 32-bit");
/*
 AF_INTR_CLEAR 
*/
typedef struct reg_af_intr_clear {
	union {
		struct {
			uint32_t slv_err_legacy : 1,
				slv_err : 1,
				rr_dbg : 1,
				num_err : 1,
				dccm_err : 1,
				lbw_stall_err : 1,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_af_intr_clear;
static_assert((sizeof(struct reg_af_intr_clear) == 4), "reg_af_intr_clear size is not 32-bit");
/*
 SB_CFG_MISC 
 b'Misc SB configurations'
*/
typedef struct reg_sb_cfg_misc {
	union {
		struct {
			uint32_t enable_cgate : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_sb_cfg_misc;
static_assert((sizeof(struct reg_sb_cfg_misc) == 4), "reg_sb_cfg_misc size is not 32-bit");
/*
 SB_DBG_STATUS 
*/
typedef struct reg_sb_dbg_status {
	union {
		struct {
			uint32_t dbg_noc_2_sb_bp : 1,
				almost_full0 : 1,
				almost_full1 : 1,
				almost_full2 : 1,
				almost_full3 : 1,
				initiator_bp : 1,
				initiator_bp_sb_full : 1,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_sb_dbg_status;
static_assert((sizeof(struct reg_sb_dbg_status) == 4), "reg_sb_dbg_status size is not 32-bit");
/*
 SB_MON_FIRST 
*/
typedef struct reg_sb_mon_first {
	union {
		struct {
			uint32_t cntx : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_sb_mon_first;
static_assert((sizeof(struct reg_sb_mon_first) == 4), "reg_sb_mon_first size is not 32-bit");
/*
 SB_MON_LAST 
*/
typedef struct reg_sb_mon_last {
	union {
		struct {
			uint32_t cntx : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_sb_mon_last;
static_assert((sizeof(struct reg_sb_mon_last) == 4), "reg_sb_mon_last size is not 32-bit");
/*
 AF_HOST_STREAM_BASE_L 
 b'Host stream: CB queue Base-Address (L)'
*/
typedef struct reg_af_host_stream_base_l {
	union {
		struct {
			uint32_t lsb : 32;
		};
		uint32_t _raw;
	};
} reg_af_host_stream_base_l;
static_assert((sizeof(struct reg_af_host_stream_base_l) == 4), "reg_af_host_stream_base_l size is not 32-bit");
/*
 AF_HOST_STREAM_BASE_H 
 b'Host stream: CB queue Base-Address (H)'
*/
typedef struct reg_af_host_stream_base_h {
	union {
		struct {
			uint32_t msb : 32;
		};
		uint32_t _raw;
	};
} reg_af_host_stream_base_h;
static_assert((sizeof(struct reg_af_host_stream_base_h) == 4), "reg_af_host_stream_base_h size is not 32-bit");
/*
 AF_HOST_STREAM_SIZE 
 b'Host-Stream: CB_Queue-Size Log'
*/
typedef struct reg_af_host_stream_size {
	union {
		struct {
			uint32_t log2_size : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_af_host_stream_size;
static_assert((sizeof(struct reg_af_host_stream_size) == 4), "reg_af_host_stream_size size is not 32-bit");
/*
 AF_DCCM_STREAM_ADDR 
 b'DCCM base/size'
*/
typedef struct reg_af_dccm_stream_addr {
	union {
		struct {
			uint32_t abase : 16,
				asize : 16;
		};
		uint32_t _raw;
	};
} reg_af_dccm_stream_addr;
static_assert((sizeof(struct reg_af_dccm_stream_addr) == 4), "reg_af_dccm_stream_addr size is not 32-bit");
/*
 AF_HOST_STREAM_PI 
 b'PI of Host-Stream  [1B Granularity]'
*/
typedef struct reg_af_host_stream_pi {
	union {
		struct {
			uint32_t offset : 32;
		};
		uint32_t _raw;
	};
} reg_af_host_stream_pi;
static_assert((sizeof(struct reg_af_host_stream_pi) == 4), "reg_af_host_stream_pi size is not 32-bit");
/*
 AF_HOST_STREAM_CI 
 b'CI of Host-Stream [1B Granularity]'
*/
typedef struct reg_af_host_stream_ci {
	union {
		struct {
			uint32_t offset : 32;
		};
		uint32_t _raw;
	};
} reg_af_host_stream_ci;
static_assert((sizeof(struct reg_af_host_stream_ci) == 4), "reg_af_host_stream_ci size is not 32-bit");
/*
 AF_DCCM_STREAM_PI 
 b'PI for DCCM stream [1B Granularity]'
*/
typedef struct reg_af_dccm_stream_pi {
	union {
		struct {
			uint32_t pi : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_af_dccm_stream_pi;
static_assert((sizeof(struct reg_af_dccm_stream_pi) == 4), "reg_af_dccm_stream_pi size is not 32-bit");
/*
 AF_DCCM_STREAM_CI 
 b'CI for DCCM stream [1B Granularity]'
*/
typedef struct reg_af_dccm_stream_ci {
	union {
		struct {
			uint32_t ci : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_af_dccm_stream_ci;
static_assert((sizeof(struct reg_af_dccm_stream_ci) == 4), "reg_af_dccm_stream_ci size is not 32-bit");
/*
 AF_DCCM_MIN_THRESHOLD 
 b'DCCM min DCCM space for write'
*/
typedef struct reg_af_dccm_min_threshold {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_af_dccm_min_threshold;
static_assert((sizeof(struct reg_af_dccm_min_threshold) == 4), "reg_af_dccm_min_threshold size is not 32-bit");
/*
 AF_STREAM_GLOBAL 
 b'Global stream CFG'
*/
typedef struct reg_af_stream_global {
	union {
		struct {
			uint32_t spriority : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_af_stream_global;
static_assert((sizeof(struct reg_af_stream_global) == 4), "reg_af_stream_global size is not 32-bit");
/*
 AF_STREAM_MASK 
*/
typedef struct reg_af_stream_mask {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_af_stream_mask;
static_assert((sizeof(struct reg_af_stream_mask) == 4), "reg_af_stream_mask size is not 32-bit");
/*
 AF_PRIO_WEIGHT 
*/
typedef struct reg_af_prio_weight {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_af_prio_weight;
static_assert((sizeof(struct reg_af_prio_weight) == 4), "reg_af_prio_weight size is not 32-bit");
/*
 AF_WR_DCCM 
 b'DCCM AXI general CFG'
*/
typedef struct reg_af_wr_dccm {
	union {
		struct {
			uint32_t os_limit : 5,
				fixed_id : 1,
				reorder_en : 1,
				nicag_dis : 1,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_af_wr_dccm;
static_assert((sizeof(struct reg_af_wr_dccm) == 4), "reg_af_wr_dccm size is not 32-bit");
/*
 AF_LBW_STALL 
*/
typedef struct reg_af_lbw_stall {
	union {
		struct {
			uint32_t stall_cntr : 24,
				fifo_thr : 5,
				_reserved31 : 2,
				stall_en : 1;
		};
		uint32_t _raw;
	};
} reg_af_lbw_stall;
static_assert((sizeof(struct reg_af_lbw_stall) == 4), "reg_af_lbw_stall size is not 32-bit");
/*
 AF_DBG_PNC 
 b'AF debug info - Pi not eq Ci'
*/
typedef struct reg_af_dbg_pnc {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_af_dbg_pnc;
static_assert((sizeof(struct reg_af_dbg_pnc) == 4), "reg_af_dbg_pnc size is not 32-bit");
/*
 AF_DBG_STATE 
*/
typedef struct reg_af_dbg_state {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_af_dbg_state;
static_assert((sizeof(struct reg_af_dbg_state) == 4), "reg_af_dbg_state size is not 32-bit");

#ifdef __cplusplus
} /* arc_af_eng namespace */
#endif

/*
 ARC_AF_ENG block
*/

#ifdef __cplusplus

struct block_arc_af_eng {
	uint32_t _pad0[5];
	struct arc_af_eng::reg_sb_max_size sb_max_size;
	struct arc_af_eng::reg_sb_force_miss sb_force_miss;
	struct arc_af_eng::reg_sb_max sb_max;
	struct arc_af_eng::reg_sb_rl sb_rl;
	struct arc_af_eng::reg_sb_halt sb_halt;
	struct arc_af_eng::reg_dccm_halt dccm_halt;
	struct arc_af_eng::reg_sb_arcache sb_arcache;
	struct arc_af_eng::reg_sb_occupancy sb_occupancy;
	uint32_t _pad52[1];
	struct arc_af_eng::reg_sb_inflights sb_inflights;
	struct arc_af_eng::reg_sb_prot sb_prot;
	uint32_t _pad64[1];
	struct arc_af_eng::reg_sb_axi_128byte sb_axi_128byte;
	uint32_t _pad72[47];
	struct arc_af_eng::reg_af_status_reg af_status_reg;
	uint32_t _pad264[2];
	struct arc_af_eng::reg_af_intr_cause af_intr_cause;
	uint32_t _pad276[1];
	struct arc_af_eng::reg_af_intr_mask af_intr_mask;
	struct arc_af_eng::reg_af_intr_clear af_intr_clear;
	uint32_t _pad288[12];
	struct arc_af_eng::reg_sb_cfg_misc sb_cfg_misc;
	uint32_t _pad340[3];
	struct arc_af_eng::reg_sb_dbg_status sb_dbg_status;
	uint32_t _pad356[3];
	struct arc_af_eng::reg_sb_mon_first sb_mon_first;
	struct arc_af_eng::reg_sb_mon_last sb_mon_last;
	uint32_t _pad376[35];
	struct arc_af_eng::reg_af_host_stream_base_l af_host_stream_base_l[32];
	uint32_t _pad644[32];
	struct arc_af_eng::reg_af_host_stream_base_h af_host_stream_base_h[32];
	uint32_t _pad900[32];
	struct arc_af_eng::reg_af_host_stream_size af_host_stream_size[32];
	uint32_t _pad1156[32];
	struct arc_af_eng::reg_af_dccm_stream_addr af_dccm_stream_addr[32];
	uint32_t _pad1412[32];
	struct arc_af_eng::reg_af_host_stream_pi af_host_stream_pi[32];
	uint32_t _pad1668[32];
	struct arc_af_eng::reg_af_host_stream_ci af_host_stream_ci[32];
	uint32_t _pad1924[32];
	struct arc_af_eng::reg_af_dccm_stream_pi af_dccm_stream_pi[32];
	uint32_t _pad2180[32];
	struct arc_af_eng::reg_af_dccm_stream_ci af_dccm_stream_ci[32];
	uint32_t _pad2436[3];
	struct arc_af_eng::reg_af_dccm_min_threshold af_dccm_min_threshold;
	uint32_t _pad2452[28];
	struct arc_af_eng::reg_af_stream_global af_stream_global[32];
	uint32_t _pad2692[32];
	struct arc_af_eng::reg_af_stream_mask af_stream_mask[32];
	uint32_t _pad2948[32];
	struct arc_af_eng::reg_af_prio_weight af_prio_weight[4];
	uint32_t _pad3092[7];
	struct arc_af_eng::reg_af_wr_dccm af_wr_dccm;
	struct arc_af_eng::reg_af_lbw_stall af_lbw_stall;
	uint32_t _pad3128[51];
	struct arc_af_eng::reg_af_dbg_pnc af_dbg_pnc;
	struct arc_af_eng::reg_af_dbg_state af_dbg_state[2];
	uint32_t _pad3344[92];
	struct block_special_regs special;
};
#else

typedef struct block_arc_af_eng {
	uint32_t _pad0[5];
	reg_sb_max_size sb_max_size;
	reg_sb_force_miss sb_force_miss;
	reg_sb_max sb_max;
	reg_sb_rl sb_rl;
	reg_sb_halt sb_halt;
	reg_dccm_halt dccm_halt;
	reg_sb_arcache sb_arcache;
	reg_sb_occupancy sb_occupancy;
	uint32_t _pad52[1];
	reg_sb_inflights sb_inflights;
	reg_sb_prot sb_prot;
	uint32_t _pad64[1];
	reg_sb_axi_128byte sb_axi_128byte;
	uint32_t _pad72[47];
	reg_af_status_reg af_status_reg;
	uint32_t _pad264[2];
	reg_af_intr_cause af_intr_cause;
	uint32_t _pad276[1];
	reg_af_intr_mask af_intr_mask;
	reg_af_intr_clear af_intr_clear;
	uint32_t _pad288[12];
	reg_sb_cfg_misc sb_cfg_misc;
	uint32_t _pad340[3];
	reg_sb_dbg_status sb_dbg_status;
	uint32_t _pad356[3];
	reg_sb_mon_first sb_mon_first;
	reg_sb_mon_last sb_mon_last;
	uint32_t _pad376[35];
	reg_af_host_stream_base_l af_host_stream_base_l[32];
	uint32_t _pad644[32];
	reg_af_host_stream_base_h af_host_stream_base_h[32];
	uint32_t _pad900[32];
	reg_af_host_stream_size af_host_stream_size[32];
	uint32_t _pad1156[32];
	reg_af_dccm_stream_addr af_dccm_stream_addr[32];
	uint32_t _pad1412[32];
	reg_af_host_stream_pi af_host_stream_pi[32];
	uint32_t _pad1668[32];
	reg_af_host_stream_ci af_host_stream_ci[32];
	uint32_t _pad1924[32];
	reg_af_dccm_stream_pi af_dccm_stream_pi[32];
	uint32_t _pad2180[32];
	reg_af_dccm_stream_ci af_dccm_stream_ci[32];
	uint32_t _pad2436[3];
	reg_af_dccm_min_threshold af_dccm_min_threshold;
	uint32_t _pad2452[28];
	reg_af_stream_global af_stream_global[32];
	uint32_t _pad2692[32];
	reg_af_stream_mask af_stream_mask[32];
	uint32_t _pad2948[32];
	reg_af_prio_weight af_prio_weight[4];
	uint32_t _pad3092[7];
	reg_af_wr_dccm af_wr_dccm;
	reg_af_lbw_stall af_lbw_stall;
	uint32_t _pad3128[51];
	reg_af_dbg_pnc af_dbg_pnc;
	reg_af_dbg_state af_dbg_state[2];
	uint32_t _pad3344[92];
	block_special_regs special;
} block_arc_af_eng;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_arc_af_eng_defaults[] =
{
	// offset	// value
	{ 0xc04 , 0x1                 , 4 }, // af_prio_weight
	{ 0xc30 , 0x16                , 1 }, // af_wr_dccm
	{ 0xc34 , 0x8b100000          , 1 }, // af_lbw_stall
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_ARC_AF_ENG_H_ */

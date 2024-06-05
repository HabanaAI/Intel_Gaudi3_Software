/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_ARC_ACP_ENG_H_
#define ASIC_REG_STRUCTS_GAUDI3_ARC_ACP_ENG_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace arc_acp_eng {
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
 QSEL_NIC_AGG_RES 
*/
typedef struct reg_qsel_nic_agg_res {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qsel_nic_agg_res;
static_assert((sizeof(struct reg_qsel_nic_agg_res) == 4), "reg_qsel_nic_agg_res size is not 32-bit");
/*
 QSEL_DBG_MASK_INT 
*/
typedef struct reg_qsel_dbg_mask_int {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qsel_dbg_mask_int;
static_assert((sizeof(struct reg_qsel_dbg_mask_int) == 4), "reg_qsel_dbg_mask_int size is not 32-bit");
/*
 QSEL_DBG_MASK_CNTR 
*/
typedef struct reg_qsel_dbg_mask_cntr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qsel_dbg_mask_cntr;
static_assert((sizeof(struct reg_qsel_dbg_mask_cntr) == 4), "reg_qsel_dbg_mask_cntr size is not 32-bit");
/*
 QSEL_PRIO 
 b'Priority Registers'
*/
typedef struct reg_qsel_prio {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_qsel_prio;
static_assert((sizeof(struct reg_qsel_prio) == 4), "reg_qsel_prio size is not 32-bit");
/*
 QSEL_MASK 
 b'Selector mask bit'
*/
typedef struct reg_qsel_mask {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_qsel_mask;
static_assert((sizeof(struct reg_qsel_mask) == 4), "reg_qsel_mask size is not 32-bit");
/*
 QSEL_INT_MASK 
 b'Internal mask array'
*/
typedef struct reg_qsel_int_mask {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_qsel_int_mask;
static_assert((sizeof(struct reg_qsel_int_mask) == 4), "reg_qsel_int_mask size is not 32-bit");
/*
 QSEL_SELECTED_ID 
 b'ACP Selected queue ID'
*/
typedef struct reg_qsel_selected_id {
	union {
		struct {
			uint32_t stream : 6,
				prio : 2,
				vld : 1,
				_reserved16 : 7,
				pi : 16;
		};
		uint32_t _raw;
	};
} reg_qsel_selected_id;
static_assert((sizeof(struct reg_qsel_selected_id) == 4), "reg_qsel_selected_id size is not 32-bit");
/*
 ACP_GRANTS_WEIGHT_PRIO 
 b'For starvation prevention - each trans weight'
*/
typedef struct reg_acp_grants_weight_prio {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_acp_grants_weight_prio;
static_assert((sizeof(struct reg_acp_grants_weight_prio) == 4), "reg_acp_grants_weight_prio size is not 32-bit");
/*
 ACP_GRANTS_COUNTER_PRIO 
 b'For starvation prevention - each priority TH'
*/
typedef struct reg_acp_grants_counter_prio {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_acp_grants_counter_prio;
static_assert((sizeof(struct reg_acp_grants_counter_prio) == 4), "reg_acp_grants_counter_prio size is not 32-bit");
/*
 ACP_DBG_PRIO_OUT_CNT 
 b'times each priority outputs'
*/
typedef struct reg_acp_dbg_prio_out_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_acp_dbg_prio_out_cnt;
static_assert((sizeof(struct reg_acp_dbg_prio_out_cnt) == 4), "reg_acp_dbg_prio_out_cnt size is not 32-bit");
/*
 ACP_DBG_PRIO_RD_CNT 
 b'times each priority read'
*/
typedef struct reg_acp_dbg_prio_rd_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_acp_dbg_prio_rd_cnt;
static_assert((sizeof(struct reg_acp_dbg_prio_rd_cnt) == 4), "reg_acp_dbg_prio_rd_cnt size is not 32-bit");
/*
 ACP_DBG_REG 
 b'General DBG data and CTL'
*/
typedef struct reg_acp_dbg_reg {
	union {
		struct {
			uint32_t last_sel_q : 6,
				prev_out_q : 6,
				masked_vld_qs : 7,
				reset_prio_out_cnt : 4,
				reset_prio_rd_cnt : 4,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_acp_dbg_reg;
static_assert((sizeof(struct reg_acp_dbg_reg) == 4), "reg_acp_dbg_reg size is not 32-bit");
/*
 QSEL_CTR_IDX 
 b'Defines which counter to use for mask'
*/
typedef struct reg_qsel_ctr_idx {
	union {
		struct {
			uint32_t idx : 8,
				_reserved31 : 23,
				disable : 1;
		};
		uint32_t _raw;
	};
} reg_qsel_ctr_idx;
static_assert((sizeof(struct reg_qsel_ctr_idx) == 4), "reg_qsel_ctr_idx size is not 32-bit");
/*
 QSEL_MASK_MISC 
 b'Misc mask CFG'
*/
typedef struct reg_qsel_mask_misc {
	union {
		struct {
			uint32_t mask_reg_after_sel : 1,
				int_mask_disable : 1,
				cb_int_mask_wr_mask : 1,
				cb_int_mask_wr_cntr_value : 1,
				cb_int_mask_wr_cntr_idx : 1,
				cb_pi_update_mode : 1,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_qsel_mask_misc;
static_assert((sizeof(struct reg_qsel_mask_misc) == 4), "reg_qsel_mask_misc size is not 32-bit");
/*
 QSEL_MASK_COUNTER 
 b'Selector Mask Counter array'
*/
typedef struct reg_qsel_mask_counter {
	union {
		struct {
			uint32_t value : 12,
				_reserved31 : 19,
				op : 1;
		};
		uint32_t _raw;
	};
} reg_qsel_mask_counter;
static_assert((sizeof(struct reg_qsel_mask_counter) == 4), "reg_qsel_mask_counter size is not 32-bit");
/*
 NIC_MASK_AGGREGATOR_MISC 
 b'Mask aggregator common registers'
*/
typedef struct reg_nic_mask_aggregator_misc {
	union {
		struct {
			uint32_t en : 1,
				dccm_ptr : 16,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_nic_mask_aggregator_misc;
static_assert((sizeof(struct reg_nic_mask_aggregator_misc) == 4), "reg_nic_mask_aggregator_misc size is not 32-bit");
/*
 NIC_AGGREGATOR_PORT 
 b'AGGREGATOR PORT'
*/
typedef struct reg_nic_aggregator_port {
	union {
		struct {
			uint32_t state_p0 : 1,
				state_p1 : 1,
				state_p2 : 1,
				state_p3 : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_nic_aggregator_port;
static_assert((sizeof(struct reg_nic_aggregator_port) == 4), "reg_nic_aggregator_port size is not 32-bit");
/*
 MASK_AGGREGATOR_P0_BITS_LSB 
 b'Priority0 information (Ports 0..31)'
*/
typedef struct reg_mask_aggregator_p0_bits_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p0_bits_lsb;
static_assert((sizeof(struct reg_mask_aggregator_p0_bits_lsb) == 4), "reg_mask_aggregator_p0_bits_lsb size is not 32-bit");
/*
 MASK_AGGREGATOR_P0_BITS_MSB 
 b'Priority0 information (Ports 32..47)'
*/
typedef struct reg_mask_aggregator_p0_bits_msb {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p0_bits_msb;
static_assert((sizeof(struct reg_mask_aggregator_p0_bits_msb) == 4), "reg_mask_aggregator_p0_bits_msb size is not 32-bit");
/*
 MASK_AGGREGATOR_P1_BITS_LSB 
 b'Priority1 information (Ports 0..31)'
*/
typedef struct reg_mask_aggregator_p1_bits_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p1_bits_lsb;
static_assert((sizeof(struct reg_mask_aggregator_p1_bits_lsb) == 4), "reg_mask_aggregator_p1_bits_lsb size is not 32-bit");
/*
 MASK_AGGREGATOR_P1_BITS_MSB 
 b'Priority1 information (Ports 32..47)'
*/
typedef struct reg_mask_aggregator_p1_bits_msb {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p1_bits_msb;
static_assert((sizeof(struct reg_mask_aggregator_p1_bits_msb) == 4), "reg_mask_aggregator_p1_bits_msb size is not 32-bit");
/*
 MASK_AGGREGATOR_P2_BITS_LSB 
 b'Priority2 information (Ports 0..31)'
*/
typedef struct reg_mask_aggregator_p2_bits_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p2_bits_lsb;
static_assert((sizeof(struct reg_mask_aggregator_p2_bits_lsb) == 4), "reg_mask_aggregator_p2_bits_lsb size is not 32-bit");
/*
 MASK_AGGREGATOR_P2_BITS_MSB 
 b'Priority2 information (Ports 32..47)'
*/
typedef struct reg_mask_aggregator_p2_bits_msb {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p2_bits_msb;
static_assert((sizeof(struct reg_mask_aggregator_p2_bits_msb) == 4), "reg_mask_aggregator_p2_bits_msb size is not 32-bit");
/*
 MASK_AGGREGATOR_P3_BITS_LSB 
 b'Priority3 information (Ports 0..31)'
*/
typedef struct reg_mask_aggregator_p3_bits_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p3_bits_lsb;
static_assert((sizeof(struct reg_mask_aggregator_p3_bits_lsb) == 4), "reg_mask_aggregator_p3_bits_lsb size is not 32-bit");
/*
 MASK_AGGREGATOR_P3_BITS_MSB 
 b'Priority3 information (Ports 32..47)'
*/
typedef struct reg_mask_aggregator_p3_bits_msb {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_mask_aggregator_p3_bits_msb;
static_assert((sizeof(struct reg_mask_aggregator_p3_bits_msb) == 4), "reg_mask_aggregator_p3_bits_msb size is not 32-bit");
/*
 QSEL_CNTR 
*/
typedef struct reg_qsel_cntr {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_qsel_cntr;
static_assert((sizeof(struct reg_qsel_cntr) == 4), "reg_qsel_cntr size is not 32-bit");

#ifdef __cplusplus
} /* arc_acp_eng namespace */
#endif

/*
 ARC_ACP_ENG block
*/

#ifdef __cplusplus

struct block_arc_acp_eng {
	uint32_t _pad0[64];
	struct arc_acp_eng::reg_qsel_nic_agg_res qsel_nic_agg_res;
	struct arc_acp_eng::reg_qsel_dbg_mask_int qsel_dbg_mask_int;
	struct arc_acp_eng::reg_qsel_dbg_mask_cntr qsel_dbg_mask_cntr;
	uint32_t _pad268[61];
	struct arc_acp_eng::reg_qsel_prio qsel_prio[32];
	uint32_t _pad640[32];
	struct arc_acp_eng::reg_qsel_mask qsel_mask[32];
	struct arc_acp_eng::reg_qsel_int_mask qsel_int_mask[32];
	struct arc_acp_eng::reg_qsel_selected_id qsel_selected_id;
	struct arc_acp_eng::reg_acp_grants_weight_prio acp_grants_weight_prio[3];
	struct arc_acp_eng::reg_acp_grants_counter_prio acp_grants_counter_prio[3];
	struct arc_acp_eng::reg_acp_dbg_prio_out_cnt acp_dbg_prio_out_cnt[4];
	struct arc_acp_eng::reg_acp_dbg_prio_rd_cnt acp_dbg_prio_rd_cnt[4];
	struct arc_acp_eng::reg_acp_dbg_reg acp_dbg_reg;
	uint32_t _pad1088[48];
	struct arc_acp_eng::reg_qsel_ctr_idx qsel_ctr_idx[32];
	uint32_t _pad1408[1];
	struct arc_acp_eng::reg_qsel_mask_misc qsel_mask_misc;
	uint32_t _pad1416[30];
	struct arc_acp_eng::reg_qsel_mask_counter qsel_mask_counter[64];
	struct arc_acp_eng::reg_nic_mask_aggregator_misc nic_mask_aggregator_misc;
	struct arc_acp_eng::reg_nic_aggregator_port nic_aggregator_port[48];
	uint32_t _pad1988[15];
	struct arc_acp_eng::reg_mask_aggregator_p0_bits_lsb mask_aggregator_p0_bits_lsb;
	struct arc_acp_eng::reg_mask_aggregator_p0_bits_msb mask_aggregator_p0_bits_msb;
	struct arc_acp_eng::reg_mask_aggregator_p1_bits_lsb mask_aggregator_p1_bits_lsb;
	struct arc_acp_eng::reg_mask_aggregator_p1_bits_msb mask_aggregator_p1_bits_msb;
	struct arc_acp_eng::reg_mask_aggregator_p2_bits_lsb mask_aggregator_p2_bits_lsb;
	struct arc_acp_eng::reg_mask_aggregator_p2_bits_msb mask_aggregator_p2_bits_msb;
	struct arc_acp_eng::reg_mask_aggregator_p3_bits_lsb mask_aggregator_p3_bits_lsb;
	struct arc_acp_eng::reg_mask_aggregator_p3_bits_msb mask_aggregator_p3_bits_msb;
	uint32_t _pad2080[56];
	struct arc_acp_eng::reg_qsel_cntr qsel_cntr[64];
	uint32_t _pad2560[288];
	struct block_special_regs special;
};
#else

typedef struct block_arc_acp_eng {
	uint32_t _pad0[64];
	reg_qsel_nic_agg_res qsel_nic_agg_res;
	reg_qsel_dbg_mask_int qsel_dbg_mask_int;
	reg_qsel_dbg_mask_cntr qsel_dbg_mask_cntr;
	uint32_t _pad268[61];
	reg_qsel_prio qsel_prio[32];
	uint32_t _pad640[32];
	reg_qsel_mask qsel_mask[32];
	reg_qsel_int_mask qsel_int_mask[32];
	reg_qsel_selected_id qsel_selected_id;
	reg_acp_grants_weight_prio acp_grants_weight_prio[3];
	reg_acp_grants_counter_prio acp_grants_counter_prio[3];
	reg_acp_dbg_prio_out_cnt acp_dbg_prio_out_cnt[4];
	reg_acp_dbg_prio_rd_cnt acp_dbg_prio_rd_cnt[4];
	reg_acp_dbg_reg acp_dbg_reg;
	uint32_t _pad1088[48];
	reg_qsel_ctr_idx qsel_ctr_idx[32];
	uint32_t _pad1408[1];
	reg_qsel_mask_misc qsel_mask_misc;
	uint32_t _pad1416[30];
	reg_qsel_mask_counter qsel_mask_counter[64];
	reg_nic_mask_aggregator_misc nic_mask_aggregator_misc;
	reg_nic_aggregator_port nic_aggregator_port[48];
	uint32_t _pad1988[15];
	reg_mask_aggregator_p0_bits_lsb mask_aggregator_p0_bits_lsb;
	reg_mask_aggregator_p0_bits_msb mask_aggregator_p0_bits_msb;
	reg_mask_aggregator_p1_bits_lsb mask_aggregator_p1_bits_lsb;
	reg_mask_aggregator_p1_bits_msb mask_aggregator_p1_bits_msb;
	reg_mask_aggregator_p2_bits_lsb mask_aggregator_p2_bits_lsb;
	reg_mask_aggregator_p2_bits_msb mask_aggregator_p2_bits_msb;
	reg_mask_aggregator_p3_bits_lsb mask_aggregator_p3_bits_lsb;
	reg_mask_aggregator_p3_bits_msb mask_aggregator_p3_bits_msb;
	uint32_t _pad2080[56];
	reg_qsel_cntr qsel_cntr[64];
	uint32_t _pad2560[288];
	block_special_regs special;
} block_arc_acp_eng;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_arc_acp_eng_defaults[] =
{
	// offset	// value
	{ 0x500 , 0x80000000          , 32 }, // qsel_ctr_idx
	{ 0x584 , 0x1                 , 1 }, // qsel_mask_misc
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_ARC_ACP_ENG_H_ */

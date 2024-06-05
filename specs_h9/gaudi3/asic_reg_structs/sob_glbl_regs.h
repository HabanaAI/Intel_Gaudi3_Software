/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_SOB_GLBL_H_
#define ASIC_REG_STRUCTS_GAUDI3_SOB_GLBL_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "axuser_hbw_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace sob_glbl {
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
 SM_SEI_MASK 
 b'SM System Error Indication Mask'
*/
typedef struct reg_sm_sei_mask {
	union {
		struct {
			uint32_t so_overflow : 1,
				mst_unalign4b : 1,
				mst_rsp_err : 1,
				direct_cq_ovrflow : 1,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_sm_sei_mask;
static_assert((sizeof(struct reg_sm_sei_mask) == 4), "reg_sm_sei_mask size is not 32-bit");
/*
 SM_SEI_CAUSE 
 b'SM System Error Indication Cause'
*/
typedef struct reg_sm_sei_cause {
	union {
		struct {
			uint32_t cause : 4,
				log : 16,
				_reserved20 : 12;
		};
		uint32_t _raw;
	};
} reg_sm_sei_cause;
static_assert((sizeof(struct reg_sm_sei_cause) == 4), "reg_sm_sei_cause size is not 32-bit");
/*
 L2H_CPMR_L 
 b'Base address for LBW access from SOB (units 1MB)'
*/
typedef struct reg_l2h_cpmr_l {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_l2h_cpmr_l;
static_assert((sizeof(struct reg_l2h_cpmr_l) == 4), "reg_l2h_cpmr_l size is not 32-bit");
/*
 L2H_CPMR_H 
 b'Base address for LBW access (MSB)'
*/
typedef struct reg_l2h_cpmr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_l2h_cpmr_h;
static_assert((sizeof(struct reg_l2h_cpmr_h) == 4), "reg_l2h_cpmr_h size is not 32-bit");
/*
 L2H_MASK_L 
 b'Address mask for LBW access from SOB,units 1MB'
*/
typedef struct reg_l2h_mask_l {
	union {
		struct {
			uint32_t val : 12,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_l2h_mask_l;
static_assert((sizeof(struct reg_l2h_mask_l) == 4), "reg_l2h_mask_l size is not 32-bit");
/*
 L2H_MASK_H 
 b'Address mask for LBW access (MSB)'
*/
typedef struct reg_l2h_mask_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_l2h_mask_h;
static_assert((sizeof(struct reg_l2h_mask_h) == 4), "reg_l2h_mask_h size is not 32-bit");
/*
 LBW_DELAY 
 b'LBW Delay'
*/
typedef struct reg_lbw_delay {
	union {
		struct {
			uint32_t val : 16,
				en : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_lbw_delay;
static_assert((sizeof(struct reg_lbw_delay) == 4), "reg_lbw_delay size is not 32-bit");
/*
 PI_SIZE 
 b'PI increment size (Default is 4)'
*/
typedef struct reg_pi_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pi_size;
static_assert((sizeof(struct reg_pi_size) == 4), "reg_pi_size size is not 32-bit");
/*
 CQ_INTR 
 b'CQ Interrupt (Mask & Queue ID)'
*/
typedef struct reg_cq_intr {
	union {
		struct {
			uint32_t cq_sec_intr : 1,
				_reserved8 : 7,
				cq_sec_intr_mask : 1,
				_reserved16 : 7,
				cq_intr_queue_index : 6,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_cq_intr;
static_assert((sizeof(struct reg_cq_intr) == 4), "reg_cq_intr size is not 32-bit");
/*
 PI_INC_MODE_SIZE 
 b'PI INC mode size'
*/
typedef struct reg_pi_inc_mode_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pi_inc_mode_size;
static_assert((sizeof(struct reg_pi_inc_mode_size) == 4), "reg_pi_inc_mode_size size is not 32-bit");
/*
 CQ_SEC_PROT 
 b'CQ PI  security level (1-> PROT[1]==0)'
*/
typedef struct reg_cq_sec_prot {
	union {
		struct {
			uint32_t sec : 32;
		};
		uint32_t _raw;
	};
} reg_cq_sec_prot;
static_assert((sizeof(struct reg_cq_sec_prot) == 4), "reg_cq_sec_prot size is not 32-bit");
/*
 CQ_SEC_OVRD 
 b'CQ Security override (1 access is secure)'
*/
typedef struct reg_cq_sec_ovrd {
	union {
		struct {
			uint32_t ovrd : 32;
		};
		uint32_t _raw;
	};
} reg_cq_sec_ovrd;
static_assert((sizeof(struct reg_cq_sec_ovrd) == 4), "reg_cq_sec_ovrd size is not 32-bit");
/*
 CQ_SEC_PRV 
 b'CQ PI  privilege level  (1-> PROT[0]==1)'
*/
typedef struct reg_cq_sec_prv {
	union {
		struct {
			uint32_t priv : 32;
		};
		uint32_t _raw;
	};
} reg_cq_sec_prv;
static_assert((sizeof(struct reg_cq_sec_prv) == 4), "reg_cq_sec_prv size is not 32-bit");
/*
 CQ_BASE_ADDR_L 
 b'CQ Base Address (LSB)'
*/
typedef struct reg_cq_base_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_base_addr_l;
static_assert((sizeof(struct reg_cq_base_addr_l) == 4), "reg_cq_base_addr_l size is not 32-bit");
/*
 CQ_BASE_ADDR_H 
 b'CQ Base Address (MSB)'
*/
typedef struct reg_cq_base_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_base_addr_h;
static_assert((sizeof(struct reg_cq_base_addr_h) == 4), "reg_cq_base_addr_h size is not 32-bit");
/*
 CQ_SIZE_LOG2 
 b'CQ Base Address (Size)'
*/
typedef struct reg_cq_size_log2 {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cq_size_log2;
static_assert((sizeof(struct reg_cq_size_log2) == 4), "reg_cq_size_log2 size is not 32-bit");
/*
 CQ_PI 
 b'CQ PI (Producer Index)'
*/
typedef struct reg_cq_pi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_pi;
static_assert((sizeof(struct reg_cq_pi) == 4), "reg_cq_pi size is not 32-bit");
/*
 LBW_ADDR_L 
 b'LBW Address (LSB) sent once CQ written (HBW)'
*/
typedef struct reg_lbw_addr_l {
	union {
		struct {
			uint32_t addrl : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_addr_l;
static_assert((sizeof(struct reg_lbw_addr_l) == 4), "reg_lbw_addr_l size is not 32-bit");
/*
 LBW_ADDR_H 
 b'LBW Address (MSB) sent once CQ written (HBW)'
*/
typedef struct reg_lbw_addr_h {
	union {
		struct {
			uint32_t addrh : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_addr_h;
static_assert((sizeof(struct reg_lbw_addr_h) == 4), "reg_lbw_addr_h size is not 32-bit");
/*
 LBW_DATA 
 b'LBW data sent once CQ was written (HBW)'
*/
typedef struct reg_lbw_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_data;
static_assert((sizeof(struct reg_lbw_data) == 4), "reg_lbw_data size is not 32-bit");
/*
 CQ_INC_MODE 
 b'CQ Mode (32b/64b-Incremental)'
*/
typedef struct reg_cq_inc_mode {
	union {
		struct {
			uint32_t mode : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cq_inc_mode;
static_assert((sizeof(struct reg_cq_inc_mode) == 4), "reg_cq_inc_mode size is not 32-bit");
/*
 CQ_DIR_LBW_EN 
 b'Enable LBW MSIX for CQ DIRECT WR'
*/
typedef struct reg_cq_dir_lbw_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cq_dir_lbw_en;
static_assert((sizeof(struct reg_cq_dir_lbw_en) == 4), "reg_cq_dir_lbw_en size is not 32-bit");

#ifdef __cplusplus
} /* sob_glbl namespace */
#endif

/*
 SOB_GLBL block
*/

#ifdef __cplusplus

struct block_sob_glbl {
	struct sob_glbl::reg_sm_sei_mask sm_sei_mask;
	struct sob_glbl::reg_sm_sei_cause sm_sei_cause;
	struct sob_glbl::reg_l2h_cpmr_l l2h_cpmr_l;
	struct sob_glbl::reg_l2h_cpmr_h l2h_cpmr_h;
	struct sob_glbl::reg_l2h_mask_l l2h_mask_l;
	struct sob_glbl::reg_l2h_mask_h l2h_mask_h;
	uint32_t _pad24[3];
	struct sob_glbl::reg_lbw_delay lbw_delay;
	struct sob_glbl::reg_pi_size pi_size;
	struct sob_glbl::reg_cq_intr cq_intr;
	struct sob_glbl::reg_pi_inc_mode_size pi_inc_mode_size;
	struct sob_glbl::reg_cq_sec_prot cq_sec_prot[2];
	struct sob_glbl::reg_cq_sec_ovrd cq_sec_ovrd[2];
	struct sob_glbl::reg_cq_sec_prv cq_sec_prv[2];
	uint32_t _pad76[1];
	struct sob_glbl::reg_cq_base_addr_l cq_base_addr_l[64];
	struct sob_glbl::reg_cq_base_addr_h cq_base_addr_h[64];
	struct sob_glbl::reg_cq_size_log2 cq_size_log2[64];
	struct sob_glbl::reg_cq_pi cq_pi[64];
	struct sob_glbl::reg_lbw_addr_l lbw_addr_l[64];
	struct sob_glbl::reg_lbw_addr_h lbw_addr_h[64];
	struct sob_glbl::reg_lbw_data lbw_data[64];
	struct sob_glbl::reg_cq_inc_mode cq_inc_mode[64];
	struct sob_glbl::reg_cq_dir_lbw_en cq_dir_lbw_en[64];
	struct block_axuser_hbw usr_hbw_user;
	uint32_t _pad2476[41];
	struct block_axuser_hbw priv_hbw_user;
	uint32_t _pad2732[41];
	struct block_axuser_hbw sec_hbw_user;
	uint32_t _pad2988[181];
	struct block_special_regs special;
};
#else

typedef struct block_sob_glbl {
	reg_sm_sei_mask sm_sei_mask;
	reg_sm_sei_cause sm_sei_cause;
	reg_l2h_cpmr_l l2h_cpmr_l;
	reg_l2h_cpmr_h l2h_cpmr_h;
	reg_l2h_mask_l l2h_mask_l;
	reg_l2h_mask_h l2h_mask_h;
	uint32_t _pad24[3];
	reg_lbw_delay lbw_delay;
	reg_pi_size pi_size;
	reg_cq_intr cq_intr;
	reg_pi_inc_mode_size pi_inc_mode_size;
	reg_cq_sec_prot cq_sec_prot[2];
	reg_cq_sec_ovrd cq_sec_ovrd[2];
	reg_cq_sec_prv cq_sec_prv[2];
	uint32_t _pad76[1];
	reg_cq_base_addr_l cq_base_addr_l[64];
	reg_cq_base_addr_h cq_base_addr_h[64];
	reg_cq_size_log2 cq_size_log2[64];
	reg_cq_pi cq_pi[64];
	reg_lbw_addr_l lbw_addr_l[64];
	reg_lbw_addr_h lbw_addr_h[64];
	reg_lbw_data lbw_data[64];
	reg_cq_inc_mode cq_inc_mode[64];
	reg_cq_dir_lbw_en cq_dir_lbw_en[64];
	block_axuser_hbw usr_hbw_user;
	uint32_t _pad2476[41];
	block_axuser_hbw priv_hbw_user;
	uint32_t _pad2732[41];
	block_axuser_hbw sec_hbw_user;
	uint32_t _pad2988[181];
	block_special_regs special;
} block_sob_glbl;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_sob_glbl_defaults[] =
{
	// offset	// value
	{ 0x8   , 0xe00               , 1 }, // l2h_cpmr_l
	{ 0xc   , 0x300007f           , 1 }, // l2h_cpmr_h
	{ 0x10  , 0xe00               , 1 }, // l2h_mask_l
	{ 0x14  , 0xffffffff          , 1 }, // l2h_mask_h
	{ 0x24  , 0xff                , 1 }, // lbw_delay
	{ 0x28  , 0x4                 , 1 }, // pi_size
	{ 0x30  , 0x8                 , 1 }, // pi_inc_mode_size
	{ 0x250 , 0x8                 , 64 }, // cq_size_log2
	{ 0x850 , 0x1                 , 64 }, // cq_dir_lbw_en
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_SOB_GLBL_H_ */

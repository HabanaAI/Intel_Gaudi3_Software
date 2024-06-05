/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_PDMA_CH_A_H_
#define ASIC_REG_STRUCTS_GAUDI3_PDMA_CH_A_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "pdma_ctx_a_regs.h"
#include "pqm_ch_a_regs.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace pdma_ch_a {
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
 ECMPLTN_Q_CH_CFG 
 b'cfg HBW completion feature'
*/
typedef struct reg_ecmpltn_q_ch_cfg {
	union {
		struct {
			uint32_t enable : 1,
				compltn_mode : 1,
				lbw_wr_en : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_ch_cfg;
static_assert((sizeof(struct reg_ecmpltn_q_ch_cfg) == 4), "reg_ecmpltn_q_ch_cfg size is not 32-bit");
/*
 ECMPLTN_Q_BASE_LO 
 b'Base-Address of completion queue LSB'
*/
typedef struct reg_ecmpltn_q_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_base_lo;
static_assert((sizeof(struct reg_ecmpltn_q_base_lo) == 4), "reg_ecmpltn_q_base_lo size is not 32-bit");
/*
 ECMPLTN_Q_BASE_HI 
 b'Base-Address of completion queue MSB'
*/
typedef struct reg_ecmpltn_q_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_base_hi;
static_assert((sizeof(struct reg_ecmpltn_q_base_hi) == 4), "reg_ecmpltn_q_base_hi size is not 32-bit");
/*
 ECMPLTN_Q_BASE_SIZE 
 b'Size of completion queue in Bytes'
*/
typedef struct reg_ecmpltn_q_base_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_base_size;
static_assert((sizeof(struct reg_ecmpltn_q_base_size) == 4), "reg_ecmpltn_q_base_size size is not 32-bit");
/*
 ECMPLTN_Q_BASE_FREQ 
 b'completion per num of descriptors'
*/
typedef struct reg_ecmpltn_q_base_freq {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_base_freq;
static_assert((sizeof(struct reg_ecmpltn_q_base_freq) == 4), "reg_ecmpltn_q_base_freq size is not 32-bit");
/*
 ECMPLTN_Q_BASE_PI 
 b'incremented by HW after we write to the CQ'
*/
typedef struct reg_ecmpltn_q_base_pi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_base_pi;
static_assert((sizeof(struct reg_ecmpltn_q_base_pi) == 4), "reg_ecmpltn_q_base_pi size is not 32-bit");
/*
 ECMPLTN_Q_LBW_ADDR_L 
 b'engine cmpltn lbw msg addr lsb'
*/
typedef struct reg_ecmpltn_q_lbw_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_lbw_addr_l;
static_assert((sizeof(struct reg_ecmpltn_q_lbw_addr_l) == 4), "reg_ecmpltn_q_lbw_addr_l size is not 32-bit");
/*
 ECMPLTN_Q_LBW_ADDR_H 
 b'engine cmpltn lbw msg addr msb'
*/
typedef struct reg_ecmpltn_q_lbw_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_lbw_addr_h;
static_assert((sizeof(struct reg_ecmpltn_q_lbw_addr_h) == 4), "reg_ecmpltn_q_lbw_addr_h size is not 32-bit");
/*
 ECMPLTN_Q_LBW_PAYLD 
 b'engine cmpltn lbw msg payload'
*/
typedef struct reg_ecmpltn_q_lbw_payld {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ecmpltn_q_lbw_payld;
static_assert((sizeof(struct reg_ecmpltn_q_lbw_payld) == 4), "reg_ecmpltn_q_lbw_payld size is not 32-bit");

#ifdef __cplusplus
} /* pdma_ch_a namespace */
#endif

/*
 PDMA_CH_A block
*/

#ifdef __cplusplus

struct block_pdma_ch_a {
	struct block_pqm_ch_a pqm_ch;
	uint32_t _pad528[60];
	struct block_pdma_ctx_a ctx;
	uint32_t _pad936[22];
	struct pdma_ch_a::reg_ecmpltn_q_ch_cfg ecmpltn_q_ch_cfg;
	struct pdma_ch_a::reg_ecmpltn_q_base_lo ecmpltn_q_base_lo;
	struct pdma_ch_a::reg_ecmpltn_q_base_hi ecmpltn_q_base_hi;
	struct pdma_ch_a::reg_ecmpltn_q_base_size ecmpltn_q_base_size;
	struct pdma_ch_a::reg_ecmpltn_q_base_freq ecmpltn_q_base_freq;
	struct pdma_ch_a::reg_ecmpltn_q_base_pi ecmpltn_q_base_pi;
	struct pdma_ch_a::reg_ecmpltn_q_lbw_addr_l ecmpltn_q_lbw_addr_l;
	struct pdma_ch_a::reg_ecmpltn_q_lbw_addr_h ecmpltn_q_lbw_addr_h;
	struct pdma_ch_a::reg_ecmpltn_q_lbw_payld ecmpltn_q_lbw_payld;
	uint32_t _pad1060[663];
	struct block_special_regs special;
};
#else

typedef struct block_pdma_ch_a {
	block_pqm_ch_a pqm_ch;
	uint32_t _pad528[60];
	block_pdma_ctx_a ctx;
	uint32_t _pad936[22];
	reg_ecmpltn_q_ch_cfg ecmpltn_q_ch_cfg;
	reg_ecmpltn_q_base_lo ecmpltn_q_base_lo;
	reg_ecmpltn_q_base_hi ecmpltn_q_base_hi;
	reg_ecmpltn_q_base_size ecmpltn_q_base_size;
	reg_ecmpltn_q_base_freq ecmpltn_q_base_freq;
	reg_ecmpltn_q_base_pi ecmpltn_q_base_pi;
	reg_ecmpltn_q_lbw_addr_l ecmpltn_q_lbw_addr_l;
	reg_ecmpltn_q_lbw_addr_h ecmpltn_q_lbw_addr_h;
	reg_ecmpltn_q_lbw_payld ecmpltn_q_lbw_payld;
	uint32_t _pad1060[663];
	block_special_regs special;
} block_pdma_ch_a;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_pdma_ch_a_defaults[] =
{
	// offset	// value
	{ 0x1b0 , 0x1                 , 1 }, // cp_pred
	{ 0x1b8 , 0x20                , 1 }, // desc_ptr_rls_cfg
	{ 0x318 , 0x3000007f          , 1 }, // wr_comp0_addr_hi
	{ 0x330 , 0x3000007f          , 1 }, // wr_comp1_addr_hi
	{ 0x334 , 0x3000007f          , 1 }, // wr_comp2_addr_hi
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_PDMA_CH_A_H_ */

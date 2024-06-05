/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_ARC_AUX_H_
#define ASIC_REG_STRUCTS_GAUDI3_ARC_AUX_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace arc_aux {
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
 ADC 
 b'ADC DATA Register'
*/
typedef struct reg_adc {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_adc;
static_assert((sizeof(struct reg_adc) == 4), "reg_adc size is not 32-bit");
/*
 ADC_AVG 
 b'ADC AVG DATA Register'
*/
typedef struct reg_adc_avg {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_adc_avg;
static_assert((sizeof(struct reg_adc_avg) == 4), "reg_adc_avg size is not 32-bit");
/*
 SCRATCHPAD 
 b'General Purpose Scartchpad'
*/
typedef struct reg_scratchpad {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_scratchpad;
static_assert((sizeof(struct reg_scratchpad) == 4), "reg_scratchpad size is not 32-bit");
/*
 HW_FIFO_CONTROL 
 b'HW FIFO Control Register'
*/
typedef struct reg_hw_fifo_control {
	union {
		struct {
			uint32_t entries_num_cfg : 8,
				occupied_entries : 8,
				full_ind : 1,
				empty_ind : 1,
				ovrd_en : 1,
				intr_en : 1,
				drop_ind : 1,
				_reserved21 : 11;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_control;
static_assert((sizeof(struct reg_hw_fifo_control) == 4), "reg_hw_fifo_control size is not 32-bit");
/*
 HW_FIFO_CMD 
 b'HW FIFO Command Register'
*/
typedef struct reg_hw_fifo_cmd {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_cmd;
static_assert((sizeof(struct reg_hw_fifo_cmd) == 4), "reg_hw_fifo_cmd size is not 32-bit");
/*
 HW_FIFO_RD_PTR 
 b'HW FIFO Read Pointer Register'
*/
typedef struct reg_hw_fifo_rd_ptr {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_rd_ptr;
static_assert((sizeof(struct reg_hw_fifo_rd_ptr) == 4), "reg_hw_fifo_rd_ptr size is not 32-bit");
/*
 HW_FIFO_WR_PTR 
 b'HW FIFO Write Pointer Register'
*/
typedef struct reg_hw_fifo_wr_ptr {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_wr_ptr;
static_assert((sizeof(struct reg_hw_fifo_wr_ptr) == 4), "reg_hw_fifo_wr_ptr size is not 32-bit");
/*
 HW_FIFO_EMPTY_FULL_OVR 
 b'HW FIFO Empty Full Override'
*/
typedef struct reg_hw_fifo_empty_full_ovr {
	union {
		struct {
			uint32_t empty_ovr : 1,
				full_ovr : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_empty_full_ovr;
static_assert((sizeof(struct reg_hw_fifo_empty_full_ovr) == 4), "reg_hw_fifo_empty_full_ovr size is not 32-bit");
/*
 HW_FIFO_INTR_CLR 
 b'HW_FIFO Interrupt Clear'
*/
typedef struct reg_hw_fifo_intr_clr {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_intr_clr;
static_assert((sizeof(struct reg_hw_fifo_intr_clr) == 4), "reg_hw_fifo_intr_clr size is not 32-bit");
/*
 HW_FIFO_DATA 
 b'HW FIFO Data Register'
*/
typedef struct reg_hw_fifo_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_hw_fifo_data;
static_assert((sizeof(struct reg_hw_fifo_data) == 4), "reg_hw_fifo_data size is not 32-bit");
/*
 ENG_STS_CNTR 
 b'Engine Status Counter and Data'
*/
typedef struct reg_eng_sts_cntr {
	union {
		struct {
			uint32_t data : 8,
				cntr : 20,
				_reserved31 : 3,
				start_bit : 1;
		};
		uint32_t _raw;
	};
} reg_eng_sts_cntr;
static_assert((sizeof(struct reg_eng_sts_cntr) == 4), "reg_eng_sts_cntr size is not 32-bit");
/*
 ENG_STS_CNTR_EN 
 b'Engine Status Enable'
*/
typedef struct reg_eng_sts_cntr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_eng_sts_cntr_en;
static_assert((sizeof(struct reg_eng_sts_cntr_en) == 4), "reg_eng_sts_cntr_en size is not 32-bit");
/*
 ENG_STS 
 b'Engine Status Start Bit'
*/
typedef struct reg_eng_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_eng_sts;
static_assert((sizeof(struct reg_eng_sts) == 4), "reg_eng_sts size is not 32-bit");
/*
 ENG_STS_CLR 
 b'Engine Status CLR'
*/
typedef struct reg_eng_sts_clr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_eng_sts_clr;
static_assert((sizeof(struct reg_eng_sts_clr) == 4), "reg_eng_sts_clr size is not 32-bit");

#ifdef __cplusplus
} /* arc_aux namespace */
#endif

/*
 ARC_AUX block
*/

#ifdef __cplusplus

struct block_arc_aux {
	uint32_t _pad0[64];
	struct arc_aux::reg_adc adc;
	struct arc_aux::reg_adc_avg adc_avg;
	uint32_t _pad264[14];
	struct arc_aux::reg_scratchpad scratchpad[8];
	uint32_t _pad352[8];
	struct arc_aux::reg_hw_fifo_control hw_fifo_control[32];
	struct arc_aux::reg_hw_fifo_cmd hw_fifo_cmd[32];
	struct arc_aux::reg_hw_fifo_rd_ptr hw_fifo_rd_ptr[32];
	struct arc_aux::reg_hw_fifo_wr_ptr hw_fifo_wr_ptr[32];
	struct arc_aux::reg_hw_fifo_empty_full_ovr hw_fifo_empty_full_ovr[32];
	struct arc_aux::reg_hw_fifo_intr_clr hw_fifo_intr_clr[32];
	uint32_t _pad1152[32];
	struct arc_aux::reg_hw_fifo_data hw_fifo_data[256];
	struct arc_aux::reg_eng_sts_cntr eng_sts_cntr[128];
	struct arc_aux::reg_eng_sts_cntr_en eng_sts_cntr_en[4];
	uint32_t _pad2832[4];
	struct arc_aux::reg_eng_sts eng_sts[4];
	uint32_t _pad2864[4];
	struct arc_aux::reg_eng_sts_clr eng_sts_clr[4];
	uint32_t _pad2896[204];
	struct block_special_regs special;
};
#else

typedef struct block_arc_aux {
	uint32_t _pad0[64];
	reg_adc adc;
	reg_adc_avg adc_avg;
	uint32_t _pad264[14];
	reg_scratchpad scratchpad[8];
	uint32_t _pad352[8];
	reg_hw_fifo_control hw_fifo_control[32];
	reg_hw_fifo_cmd hw_fifo_cmd[32];
	reg_hw_fifo_rd_ptr hw_fifo_rd_ptr[32];
	reg_hw_fifo_wr_ptr hw_fifo_wr_ptr[32];
	reg_hw_fifo_empty_full_ovr hw_fifo_empty_full_ovr[32];
	reg_hw_fifo_intr_clr hw_fifo_intr_clr[32];
	uint32_t _pad1152[32];
	reg_hw_fifo_data hw_fifo_data[256];
	reg_eng_sts_cntr eng_sts_cntr[128];
	reg_eng_sts_cntr_en eng_sts_cntr_en[4];
	uint32_t _pad2832[4];
	reg_eng_sts eng_sts[4];
	uint32_t _pad2864[4];
	reg_eng_sts_clr eng_sts_clr[4];
	uint32_t _pad2896[204];
	block_special_regs special;
} block_arc_aux;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_arc_aux_defaults[] =
{
	// offset	// value
	{ 0x180 , 0x8                 , 32 }, // hw_fifo_control
	{ 0x380 , 0x1                 , 32 }, // hw_fifo_empty_full_ovr
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
#endif /* ASIC_REG_STRUCTS_GAUDI3_ARC_AUX_H_ */

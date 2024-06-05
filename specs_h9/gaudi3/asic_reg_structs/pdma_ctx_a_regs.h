/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_PDMA_CTX_A_H_
#define ASIC_REG_STRUCTS_GAUDI3_PDMA_CTX_A_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace pdma_ctx_a {
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
 SRC_BASE_LO 
 b'DMA source base address bytes 3-0'
*/
typedef struct reg_src_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_base_lo;
static_assert((sizeof(struct reg_src_base_lo) == 4), "reg_src_base_lo size is not 32-bit");
/*
 DST_BASE_LO 
 b'destination base address bytes 3-0'
*/
typedef struct reg_dst_base_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_base_lo;
static_assert((sizeof(struct reg_dst_base_lo) == 4), "reg_dst_base_lo size is not 32-bit");
/*
 SRC_BASE_HI 
 b'source base address bytes 7-4'
*/
typedef struct reg_src_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_base_hi;
static_assert((sizeof(struct reg_src_base_hi) == 4), "reg_src_base_hi size is not 32-bit");
/*
 DST_BASE_HI 
 b'destination base address bytes 7-4'
*/
typedef struct reg_dst_base_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_base_hi;
static_assert((sizeof(struct reg_dst_base_hi) == 4), "reg_dst_base_hi size is not 32-bit");
/*
 WR_COMP0_ADDR_LO 
 b'Wr completion0 msg address byte 3-0'
*/
typedef struct reg_wr_comp0_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp0_addr_lo;
static_assert((sizeof(struct reg_wr_comp0_addr_lo) == 4), "reg_wr_comp0_addr_lo size is not 32-bit");
/*
 WR_COMP0_WDATA 
 b'wr completion 0 msg wdata to send'
*/
typedef struct reg_wr_comp0_wdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp0_wdata;
static_assert((sizeof(struct reg_wr_comp0_wdata) == 4), "reg_wr_comp0_wdata size is not 32-bit");
/*
 WR_COMP0_ADDR_HI 
 b'Wr completion0 msg address byte 7-4'
*/
typedef struct reg_wr_comp0_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp0_addr_hi;
static_assert((sizeof(struct reg_wr_comp0_addr_hi) == 4), "reg_wr_comp0_addr_hi size is not 32-bit");
/*
 PLACE_HOLDER_NOT_USED 
 b'not used, keep space for linpdma sequence'
*/
typedef struct reg_place_holder_not_used {
	union {
		struct {
			uint32_t _reserved0 : 32;
		};
		uint32_t _raw;
	};
} reg_place_holder_not_used;
static_assert((sizeof(struct reg_place_holder_not_used) == 4), "reg_place_holder_not_used size is not 32-bit");
/*
 WR_COMP1_ADDR_LO 
 b'Wr completion1 msg address byte 3-0'
*/
typedef struct reg_wr_comp1_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp1_addr_lo;
static_assert((sizeof(struct reg_wr_comp1_addr_lo) == 4), "reg_wr_comp1_addr_lo size is not 32-bit");
/*
 WR_COMP1_WDATA 
 b'wr completion 1 msg wdata to send'
*/
typedef struct reg_wr_comp1_wdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp1_wdata;
static_assert((sizeof(struct reg_wr_comp1_wdata) == 4), "reg_wr_comp1_wdata size is not 32-bit");
/*
 WR_COMP2_ADDR_LO 
 b'Wr completion2 msg address byte 3-0'
*/
typedef struct reg_wr_comp2_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp2_addr_lo;
static_assert((sizeof(struct reg_wr_comp2_addr_lo) == 4), "reg_wr_comp2_addr_lo size is not 32-bit");
/*
 WR_COMP2_WDATA 
 b'wr completion 2 msg wdata to send'
*/
typedef struct reg_wr_comp2_wdata {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp2_wdata;
static_assert((sizeof(struct reg_wr_comp2_wdata) == 4), "reg_wr_comp2_wdata size is not 32-bit");
/*
 WR_COMP1_ADDR_HI 
 b'Wr completion 1 msg address byte 7-4'
*/
typedef struct reg_wr_comp1_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp1_addr_hi;
static_assert((sizeof(struct reg_wr_comp1_addr_hi) == 4), "reg_wr_comp1_addr_hi size is not 32-bit");
/*
 WR_COMP2_ADDR_HI 
 b'Wr completion 2 msg address byte 7-4'
*/
typedef struct reg_wr_comp2_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_wr_comp2_addr_hi;
static_assert((sizeof(struct reg_wr_comp2_addr_hi) == 4), "reg_wr_comp2_addr_hi size is not 32-bit");
/*
 CTRL_MISC 
 b'DMA secondary control register'
*/
typedef struct reg_ctrl_misc {
	union {
		struct {
			uint32_t sb_uncache : 1,
				consec_comp1_2_addr_l : 1,
				advance_scheduler_cnt : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_ctrl_misc;
static_assert((sizeof(struct reg_ctrl_misc) == 4), "reg_ctrl_misc size is not 32-bit");
/*
 CTRL_MAIN 
 b'DMA main control register'
*/
typedef struct reg_ctrl_main {
	union {
		struct {
			uint32_t endian_swap : 2,
				dma_modes : 5,
				ctx_id_inc : 1,
				add_offset_0 : 1,
				reserved_9 : 1,
				src_size0_from_dst_size0 : 1,
				src_ofst_from_dst_ofst : 1,
				converted_data_type : 2,
				converted_data_clipping : 1,
				converted_data_rounding : 4,
				dim1_mode : 2,
				dim2_mode : 2,
				dim3_mode : 2,
				dim4_mode : 2,
				reserved_28_27 : 2,
				wr_comp_en : 2,
				commit : 1;
		};
		uint32_t _raw;
	};
} reg_ctrl_main;
static_assert((sizeof(struct reg_ctrl_main) == 4), "reg_ctrl_main size is not 32-bit");
/*
 CH_DIRECTION 
 b'channel direction'
*/
typedef struct reg_ch_direction {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_ch_direction;
static_assert((sizeof(struct reg_ch_direction) == 4), "reg_ch_direction size is not 32-bit");
/*
 DST_TSIZE_0 
 b'size in bytes of destination dim 0 write'
*/
typedef struct reg_dst_tsize_0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_tsize_0;
static_assert((sizeof(struct reg_dst_tsize_0) == 4), "reg_dst_tsize_0 size is not 32-bit");
/*
 DST_TSIZE_1 
 b'size in elements of dim 1 write'
*/
typedef struct reg_dst_tsize_1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_tsize_1;
static_assert((sizeof(struct reg_dst_tsize_1) == 4), "reg_dst_tsize_1 size is not 32-bit");
/*
 DST_TSIZE_2 
 b'size in elements of dim 2 write'
*/
typedef struct reg_dst_tsize_2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_tsize_2;
static_assert((sizeof(struct reg_dst_tsize_2) == 4), "reg_dst_tsize_2 size is not 32-bit");
/*
 DST_TSIZE_3 
 b'size in elements of dim 3 write'
*/
typedef struct reg_dst_tsize_3 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_tsize_3;
static_assert((sizeof(struct reg_dst_tsize_3) == 4), "reg_dst_tsize_3 size is not 32-bit");
/*
 DST_TSIZE_4 
 b'size in elements of dim 4 write'
*/
typedef struct reg_dst_tsize_4 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_tsize_4;
static_assert((sizeof(struct reg_dst_tsize_4) == 4), "reg_dst_tsize_4 size is not 32-bit");
/*
 DST_STRIDE_1 
 b'size in elements of dim 1 destination stride'
*/
typedef struct reg_dst_stride_1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_stride_1;
static_assert((sizeof(struct reg_dst_stride_1) == 4), "reg_dst_stride_1 size is not 32-bit");
/*
 DST_STRIDE_2 
 b'size in elements of dim 2 destination stride'
*/
typedef struct reg_dst_stride_2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_stride_2;
static_assert((sizeof(struct reg_dst_stride_2) == 4), "reg_dst_stride_2 size is not 32-bit");
/*
 DST_STRIDE_3 
 b'size in elements of dim 4 destination stride'
*/
typedef struct reg_dst_stride_3 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_stride_3;
static_assert((sizeof(struct reg_dst_stride_3) == 4), "reg_dst_stride_3 size is not 32-bit");
/*
 DST_STRIDE_4 
 b'size in elements of dim 3 destination stride'
*/
typedef struct reg_dst_stride_4 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_stride_4;
static_assert((sizeof(struct reg_dst_stride_4) == 4), "reg_dst_stride_4 size is not 32-bit");
/*
 SRC_TSIZE_0 
 b'size in bytes of source dim 0 write'
*/
typedef struct reg_src_tsize_0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_tsize_0;
static_assert((sizeof(struct reg_src_tsize_0) == 4), "reg_src_tsize_0 size is not 32-bit");
/*
 SRC_TSIZE_1 
 b'size in elements of dim 1 write'
*/
typedef struct reg_src_tsize_1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_tsize_1;
static_assert((sizeof(struct reg_src_tsize_1) == 4), "reg_src_tsize_1 size is not 32-bit");
/*
 SRC_TSIZE_2 
 b'size in elements of dim 2 write'
*/
typedef struct reg_src_tsize_2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_tsize_2;
static_assert((sizeof(struct reg_src_tsize_2) == 4), "reg_src_tsize_2 size is not 32-bit");
/*
 SRC_TSIZE_3 
 b'size in elements of dim 3 write'
*/
typedef struct reg_src_tsize_3 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_tsize_3;
static_assert((sizeof(struct reg_src_tsize_3) == 4), "reg_src_tsize_3 size is not 32-bit");
/*
 SRC_TSIZE_4 
 b'size in elements of dim 4 write'
*/
typedef struct reg_src_tsize_4 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_tsize_4;
static_assert((sizeof(struct reg_src_tsize_4) == 4), "reg_src_tsize_4 size is not 32-bit");
/*
 SRC_STRIDE_1 
 b'size in elements of dim 1 destination stride'
*/
typedef struct reg_src_stride_1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_stride_1;
static_assert((sizeof(struct reg_src_stride_1) == 4), "reg_src_stride_1 size is not 32-bit");
/*
 SRC_STRIDE_2 
 b'size in elements of dim 2 destination stride'
*/
typedef struct reg_src_stride_2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_stride_2;
static_assert((sizeof(struct reg_src_stride_2) == 4), "reg_src_stride_2 size is not 32-bit");
/*
 SRC_STRIDE_3 
 b'size in elements of dim 3 destination stride'
*/
typedef struct reg_src_stride_3 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_stride_3;
static_assert((sizeof(struct reg_src_stride_3) == 4), "reg_src_stride_3 size is not 32-bit");
/*
 SRC_STRIDE_4 
 b'size in elements of dim 4 destination stride'
*/
typedef struct reg_src_stride_4 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_stride_4;
static_assert((sizeof(struct reg_src_stride_4) == 4), "reg_src_stride_4 size is not 32-bit");
/*
 SRC_OFFSET_LO 
 b'source offset bytes 3-0'
*/
typedef struct reg_src_offset_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_offset_lo;
static_assert((sizeof(struct reg_src_offset_lo) == 4), "reg_src_offset_lo size is not 32-bit");
/*
 SRC_OFFSET_HI 
 b'source offset bytes 7-4'
*/
typedef struct reg_src_offset_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_src_offset_hi;
static_assert((sizeof(struct reg_src_offset_hi) == 4), "reg_src_offset_hi size is not 32-bit");
/*
 DST_OFFSET_LO 
 b'destination offset bytes 3-0'
*/
typedef struct reg_dst_offset_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_offset_lo;
static_assert((sizeof(struct reg_dst_offset_lo) == 4), "reg_dst_offset_lo size is not 32-bit");
/*
 DST_OFFSET_HI 
 b'destination offset bytes 7-4'
*/
typedef struct reg_dst_offset_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dst_offset_hi;
static_assert((sizeof(struct reg_dst_offset_hi) == 4), "reg_dst_offset_hi size is not 32-bit");
/*
 IDX 
 b'context ID'
*/
typedef struct reg_idx {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_idx;
static_assert((sizeof(struct reg_idx) == 4), "reg_idx size is not 32-bit");
/*
 IDX_INC 
 b'context Index INC value'
*/
typedef struct reg_idx_inc {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_idx_inc;
static_assert((sizeof(struct reg_idx_inc) == 4), "reg_idx_inc size is not 32-bit");
/*
 COMMIT 
 b'writing to this reg initiates transfer'
*/
typedef struct reg_commit {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_commit;
static_assert((sizeof(struct reg_commit) == 4), "reg_commit size is not 32-bit");

#ifdef __cplusplus
} /* pdma_ctx_a namespace */
#endif

/*
 PDMA_CTX_A block
*/

#ifdef __cplusplus

struct block_pdma_ctx_a {
	struct pdma_ctx_a::reg_src_base_lo src_base_lo;
	struct pdma_ctx_a::reg_dst_base_lo dst_base_lo;
	struct pdma_ctx_a::reg_src_base_hi src_base_hi;
	struct pdma_ctx_a::reg_dst_base_hi dst_base_hi;
	struct pdma_ctx_a::reg_wr_comp0_addr_lo wr_comp0_addr_lo;
	struct pdma_ctx_a::reg_wr_comp0_wdata wr_comp0_wdata;
	struct pdma_ctx_a::reg_wr_comp0_addr_hi wr_comp0_addr_hi;
	struct pdma_ctx_a::reg_place_holder_not_used place_holder_not_used;
	struct pdma_ctx_a::reg_wr_comp1_addr_lo wr_comp1_addr_lo;
	struct pdma_ctx_a::reg_wr_comp1_wdata wr_comp1_wdata;
	struct pdma_ctx_a::reg_wr_comp2_addr_lo wr_comp2_addr_lo;
	struct pdma_ctx_a::reg_wr_comp2_wdata wr_comp2_wdata;
	struct pdma_ctx_a::reg_wr_comp1_addr_hi wr_comp1_addr_hi;
	struct pdma_ctx_a::reg_wr_comp2_addr_hi wr_comp2_addr_hi;
	struct pdma_ctx_a::reg_ctrl_misc ctrl_misc;
	struct pdma_ctx_a::reg_ctrl_main ctrl_main;
	struct pdma_ctx_a::reg_ch_direction ch_direction;
	struct pdma_ctx_a::reg_dst_tsize_0 dst_tsize_0;
	struct pdma_ctx_a::reg_dst_tsize_1 dst_tsize_1;
	struct pdma_ctx_a::reg_dst_tsize_2 dst_tsize_2;
	struct pdma_ctx_a::reg_dst_tsize_3 dst_tsize_3;
	struct pdma_ctx_a::reg_dst_tsize_4 dst_tsize_4;
	struct pdma_ctx_a::reg_dst_stride_1 dst_stride_1;
	struct pdma_ctx_a::reg_dst_stride_2 dst_stride_2;
	struct pdma_ctx_a::reg_dst_stride_3 dst_stride_3;
	struct pdma_ctx_a::reg_dst_stride_4 dst_stride_4;
	struct pdma_ctx_a::reg_src_tsize_0 src_tsize_0;
	struct pdma_ctx_a::reg_src_tsize_1 src_tsize_1;
	struct pdma_ctx_a::reg_src_tsize_2 src_tsize_2;
	struct pdma_ctx_a::reg_src_tsize_3 src_tsize_3;
	struct pdma_ctx_a::reg_src_tsize_4 src_tsize_4;
	struct pdma_ctx_a::reg_src_stride_1 src_stride_1;
	struct pdma_ctx_a::reg_src_stride_2 src_stride_2;
	struct pdma_ctx_a::reg_src_stride_3 src_stride_3;
	struct pdma_ctx_a::reg_src_stride_4 src_stride_4;
	struct pdma_ctx_a::reg_src_offset_lo src_offset_lo;
	struct pdma_ctx_a::reg_src_offset_hi src_offset_hi;
	struct pdma_ctx_a::reg_dst_offset_lo dst_offset_lo;
	struct pdma_ctx_a::reg_dst_offset_hi dst_offset_hi;
	struct pdma_ctx_a::reg_idx idx;
	struct pdma_ctx_a::reg_idx_inc idx_inc;
	struct pdma_ctx_a::reg_commit commit;
};
#else

typedef struct block_pdma_ctx_a {
	reg_src_base_lo src_base_lo;
	reg_dst_base_lo dst_base_lo;
	reg_src_base_hi src_base_hi;
	reg_dst_base_hi dst_base_hi;
	reg_wr_comp0_addr_lo wr_comp0_addr_lo;
	reg_wr_comp0_wdata wr_comp0_wdata;
	reg_wr_comp0_addr_hi wr_comp0_addr_hi;
	reg_place_holder_not_used place_holder_not_used;
	reg_wr_comp1_addr_lo wr_comp1_addr_lo;
	reg_wr_comp1_wdata wr_comp1_wdata;
	reg_wr_comp2_addr_lo wr_comp2_addr_lo;
	reg_wr_comp2_wdata wr_comp2_wdata;
	reg_wr_comp1_addr_hi wr_comp1_addr_hi;
	reg_wr_comp2_addr_hi wr_comp2_addr_hi;
	reg_ctrl_misc ctrl_misc;
	reg_ctrl_main ctrl_main;
	reg_ch_direction ch_direction;
	reg_dst_tsize_0 dst_tsize_0;
	reg_dst_tsize_1 dst_tsize_1;
	reg_dst_tsize_2 dst_tsize_2;
	reg_dst_tsize_3 dst_tsize_3;
	reg_dst_tsize_4 dst_tsize_4;
	reg_dst_stride_1 dst_stride_1;
	reg_dst_stride_2 dst_stride_2;
	reg_dst_stride_3 dst_stride_3;
	reg_dst_stride_4 dst_stride_4;
	reg_src_tsize_0 src_tsize_0;
	reg_src_tsize_1 src_tsize_1;
	reg_src_tsize_2 src_tsize_2;
	reg_src_tsize_3 src_tsize_3;
	reg_src_tsize_4 src_tsize_4;
	reg_src_stride_1 src_stride_1;
	reg_src_stride_2 src_stride_2;
	reg_src_stride_3 src_stride_3;
	reg_src_stride_4 src_stride_4;
	reg_src_offset_lo src_offset_lo;
	reg_src_offset_hi src_offset_hi;
	reg_dst_offset_lo dst_offset_lo;
	reg_dst_offset_hi dst_offset_hi;
	reg_idx idx;
	reg_idx_inc idx_inc;
	reg_commit commit;
} block_pdma_ctx_a;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_pdma_ctx_a_defaults[] =
{
	// offset	// value
	{ 0x18  , 0x3000007f          , 1 }, // wr_comp0_addr_hi
	{ 0x30  , 0x3000007f          , 1 }, // wr_comp1_addr_hi
	{ 0x34  , 0x3000007f          , 1 }, // wr_comp2_addr_hi
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_PDMA_CTX_A_H_ */

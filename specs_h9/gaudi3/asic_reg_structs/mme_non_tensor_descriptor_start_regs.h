/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_MME_NON_TENSOR_DESCRIPTOR_START_H_
#define ASIC_REG_STRUCTS_GAUDI3_MME_NON_TENSOR_DESCRIPTOR_START_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace mme_non_tensor_descriptor_start {
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
 BRAINS_LO 
 b'MME AGU Loop mask and Enables'
*/
typedef struct reg_brains_lo {
	union {
		struct {
			uint32_t a_loop_mask : 6,
				a_master_en : 1,
				a_slave_en : 1,
				b_loop_mask : 6,
				b_master_en : 1,
				b_slave_en : 1,
				agu_cout0_loop_mask : 6,
				agu_cout0_master_en : 1,
				agu_cout0_slave_en : 1,
				eu_loop_mask : 6,
				eu_master_en : 1,
				eu_slave_en : 1;
		};
		uint32_t _raw;
	};
} reg_brains_lo;
static_assert((sizeof(struct reg_brains_lo) == 4), "reg_brains_lo size is not 32-bit");
/*
 BRAINS_HI 
 b'MME AGU/Brains Loop mask and Enables'
*/
typedef struct reg_brains_hi {
	union {
		struct {
			uint32_t ap_loop_mask : 6,
				ap_master_en : 1,
				ap_slave_en : 1,
				agu_dma_o_loop_mask : 6,
				agu_dma_o_master_en : 1,
				agu_dma_o_slave_en : 1,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_brains_hi;
static_assert((sizeof(struct reg_brains_hi) == 4), "reg_brains_hi size is not 32-bit");
/*
 HEADER_LO 
 b'MME HEADER Configurations Part0'
*/
typedef struct reg_header_lo {
	union {
		struct {
			uint32_t trans_a : 1,
				trans_b : 1,
				sb_trans_mode_a : 1,
				sb_trans_mode_b : 1,
				advance_a : 1,
				advance_b : 1,
				advance_c : 1,
				accum_en : 1,
				lower_a : 1,
				lower_b : 1,
				roll_accums : 3,
				store_en0 : 1,
				store_en1 : 1,
				relu_en : 1,
				double_accums : 1,
				bgemm : 1,
				shuffle_a : 1,
				rounding_mode : 3,
				no_rollup : 1,
				null_desc : 1,
				data_type_in : 4,
				data_type_cout : 4;
		};
		uint32_t _raw;
	};
} reg_header_lo;
static_assert((sizeof(struct reg_header_lo) == 4), "reg_header_lo size is not 32-bit");
/*
 HEADER_HI 
 b'MME HEADER Configurations Part1'
*/
typedef struct reg_header_hi {
	union {
		struct {
			uint32_t swap_base_w_offset_a : 1,
				swap_base_w_offset_b : 1,
				swap_base_w_offset_cout : 1,
				a_non_shared : 1,
				clip_fp_eu : 1,
				clip_fp_ap : 1,
				sba_cache_en : 1,
				sbb_cache_en : 1,
				partial_height_loop_a : 6,
				store_color_set0 : 1,
				store_color_set1 : 1,
				partial_height_loop_b : 6,
				te_bypass_a : 1,
				te_bypass_b : 1,
				te_accel_a : 3,
				sto_ftz_fp32_to_fp8 : 1,
				wsb_cache_en : 1,
				dual_gemm : 1,
				dma_mode : 1,
				ftz : 1;
		};
		uint32_t _raw;
	};
} reg_header_hi;
static_assert((sizeof(struct reg_header_hi) == 4), "reg_header_hi size is not 32-bit");

#ifdef __cplusplus
} /* mme_non_tensor_descriptor_start namespace */
#endif

/*
 MME_NON_TENSOR_DESCRIPTOR_START block
*/

#ifdef __cplusplus

struct block_mme_non_tensor_descriptor_start {
	struct mme_non_tensor_descriptor_start::reg_brains_lo brains_lo;
	struct mme_non_tensor_descriptor_start::reg_brains_hi brains_hi;
	struct mme_non_tensor_descriptor_start::reg_header_lo header_lo;
	struct mme_non_tensor_descriptor_start::reg_header_hi header_hi;
};
#else

typedef struct block_mme_non_tensor_descriptor_start {
	reg_brains_lo brains_lo;
	reg_brains_hi brains_hi;
	reg_header_lo header_lo;
	reg_header_hi header_hi;
} block_mme_non_tensor_descriptor_start;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_MME_NON_TENSOR_DESCRIPTOR_START_H_ */

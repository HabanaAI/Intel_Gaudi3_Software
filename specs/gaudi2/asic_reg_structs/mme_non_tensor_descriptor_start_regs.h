/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI2_MME_NON_TENSOR_DESCRIPTOR_START_H_
#define ASIC_REG_STRUCTS_GAUDI2_MME_NON_TENSOR_DESCRIPTOR_START_H_

#include <stdint.h>
#include "gaudi2_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi2 {
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
 BRAINS_LOW 
 b'MME AGU Loop mask and Enables'
*/
typedef struct reg_brains_low {
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
				agu_cout1_loop_mask : 6,
				agu_cout1_master_en : 1,
				agu_cout1_slave_en : 1;
		};
		uint32_t _raw;
	};
} reg_brains_low;
static_assert((sizeof(struct reg_brains_low) == 4), "reg_brains_low size is not 32-bit");
/*
 BRAINS_HIGH 
 b'MME AGU/Brains Loop mask and Enables'
*/
typedef struct reg_brains_high {
	union {
		struct {
			uint32_t eu_loop_mask : 6,
				eu_master_en : 1,
				eu_slave_en : 1,
				ap_loop_mask : 6,
				ap_master_en : 1,
				ap_slave_en : 1,
				dec_en : 1,
				shuffle_a : 2,
				bgemm : 1,
				clip_fp_eu : 1,
				clip_fp_ap : 1,
				sba_cache_en : 1,
				sbb_cache_en : 1,
				rounding_mode : 3,
				relu_en : 1,
				no_rollup : 1,
				null_desc : 1,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_brains_high;
static_assert((sizeof(struct reg_brains_high) == 4), "reg_brains_high size is not 32-bit");
/*
 HEADER_LOW 
 b'MME HEADER Configurations Part0'
*/
typedef struct reg_header_low {
	union {
		struct {
			uint32_t trans_a : 1,
				trans_b : 1,
				advance_a : 1,
				advance_b : 1,
				advance_c : 1,
				lower_a : 1,
				lower_b : 1,
				accum_en : 1,
				roll_accums : 3,
				agu_reads_a : 5,
				agu_reads_b : 5,
				double_accums : 1,
				store_en0 : 1,
				store_en1 : 1,
				data_type_in : 4,
				data_type_cout : 4;
		};
		uint32_t _raw;
	};
} reg_header_low;
static_assert((sizeof(struct reg_header_low) == 4), "reg_header_low size is not 32-bit");
/*
 HEADER_HIGH 
 b'MME HEADER Configurations Part1'
*/
typedef struct reg_header_high {
	union {
		struct {
			uint32_t swap_base_and_offset_a : 1,
				swap_base_and_offset_b : 1,
				swap_base_and_offset_cout : 1,
				_reserved8 : 5,
				store_color_set0 : 1,
				store_color_set1 : 1,
				hx2 : 1,
				_reserved16 : 5,
				partial_height_loop_a : 6,
				_reserved24 : 2,
				partial_height_loop_b : 6,
				te_bypass_a : 1,
				te_bypass_b : 1;
		};
		uint32_t _raw;
	};
} reg_header_high;
static_assert((sizeof(struct reg_header_high) == 4), "reg_header_high size is not 32-bit");
/*
 EUS_MASTER 
 b'EUS MASTER'
*/
typedef struct reg_eus_master {
	union {
		struct {
			uint32_t sb0_en : 1,
				sb1_en : 1,
				sb2_en : 1,
				sb3_en : 1,
				sb4_en : 1,
				remote_in0_en : 1,
				remote_in1_en : 1,
				sb0_sel : 3,
				sb1_sel : 3,
				sb2_sel : 3,
				sb3_sel : 3,
				sb4_sel : 3,
				remote_in0_sel : 3,
				remote_in1_sel : 3,
				sb0_remote_out_en : 1,
				sb2_remote_out_en : 1,
				sb3_remote_out_en : 1,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_eus_master;
static_assert((sizeof(struct reg_eus_master) == 4), "reg_eus_master size is not 32-bit");
/*
 EUS_SLAVE 
 b'EUS Slave'
*/
typedef struct reg_eus_slave {
	union {
		struct {
			uint32_t sb0_en : 1,
				sb1_en : 1,
				sb2_en : 1,
				sb3_en : 1,
				sb4_en : 1,
				remote_in0_en : 1,
				remote_in1_en : 1,
				sb0_sel : 3,
				sb1_sel : 3,
				sb2_sel : 3,
				sb3_sel : 3,
				sb4_sel : 3,
				remote_in0_sel : 3,
				remote_in1_sel : 3,
				sb0_remote_out_en : 1,
				sb2_remote_out_en : 1,
				sb3_remote_out_en : 1,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_eus_slave;
static_assert((sizeof(struct reg_eus_slave) == 4), "reg_eus_slave size is not 32-bit");

#ifdef __cplusplus
} /* mme_non_tensor_descriptor_start namespace */
#endif

/*
 MME_NON_TENSOR_DESCRIPTOR_START block
*/

#ifdef __cplusplus

struct block_mme_non_tensor_descriptor_start {
	struct mme_non_tensor_descriptor_start::reg_brains_low brains_low;
	struct mme_non_tensor_descriptor_start::reg_brains_high brains_high;
	struct mme_non_tensor_descriptor_start::reg_header_low header_low;
	struct mme_non_tensor_descriptor_start::reg_header_high header_high;
	struct mme_non_tensor_descriptor_start::reg_eus_master eus_master;
	struct mme_non_tensor_descriptor_start::reg_eus_slave eus_slave;
};
#else

typedef struct block_mme_non_tensor_descriptor_start {
	reg_brains_low brains_low;
	reg_brains_high brains_high;
	reg_header_low header_low;
	reg_header_high header_high;
	reg_eus_master eus_master;
	reg_eus_slave eus_slave;
} block_mme_non_tensor_descriptor_start;
#endif


#ifdef __cplusplus
} /* gaudi2 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI2_MME_NON_TENSOR_DESCRIPTOR_START_H_ */

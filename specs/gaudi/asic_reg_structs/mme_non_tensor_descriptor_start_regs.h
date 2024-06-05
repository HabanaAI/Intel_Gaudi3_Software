/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_MME_NON_TENSOR_DESCRIPTOR_START_H_
#define ASIC_REG_STRUCTS_MME_NON_TENSOR_DESCRIPTOR_START_H_

#include <stdint.h>

#pragma pack(push, 1)

namespace mme_non_tensor_descriptor_start {
/*
 BASE_ADDR_HIGH_S 
*/
struct reg_base_addr_high_s {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_base_addr_high_s) == 4), "reg_base_addr_high_s size is not 32-bit");
/*
 BASE_ADDR_HIGH_L 
*/
struct reg_base_addr_high_l {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_base_addr_high_l) == 4), "reg_base_addr_high_l size is not 32-bit");
/*
 BASE_ADDR_HIGH_O 
*/
struct reg_base_addr_high_o {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_base_addr_high_o) == 4), "reg_base_addr_high_o size is not 32-bit");
/*
 BASE_ADDR_LOW_S 
*/
struct reg_base_addr_low_s {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_base_addr_low_s) == 4), "reg_base_addr_low_s size is not 32-bit");
/*
 BASE_ADDR_LOW_L 
*/
struct reg_base_addr_low_l {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_base_addr_low_l) == 4), "reg_base_addr_low_l size is not 32-bit");
/*
 BASE_ADDR_LOW_O 
*/
struct reg_base_addr_low_o {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_base_addr_low_o) == 4), "reg_base_addr_low_o size is not 32-bit");
/*
 HEADER_LOW 
*/
struct reg_header_low {
	union {
		struct {
			uint32_t trans_s : 1,
				trans_l : 1,
				trans_o : 1,
				advance_s : 1,
				advance_l : 1,
				advance_o : 1,
				lower_l : 1,
				lower_s : 1,
				accum_mask : 4,
				acc_store_inc_disable : 1,
				rounding_mode : 3,
				data_type_in : 1,
				data_type_out : 1,
				accum : 1,
				store_en : 1,
				roll_accums : 4,
				signal_mask : 6,
				signal_en : 1,
				relu_en : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_header_low) == 4), "reg_header_low size is not 32-bit");
/*
 HEADER_HIGH 
*/
struct reg_header_high {
	union {
		struct {
			uint32_t partial_height_loop_s : 6,
				partial_height_loop_l_loc : 6,
				partial_height_loop_l_rem : 6,
				partial_height_loop_o_loc : 6,
				partial_height_loop_o_rem : 6,
				reserv_sw : 1,
				eub_en : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_header_high) == 4), "reg_header_high size is not 32-bit");
/*
 CONV_KERNEL_SIZE_MINUS_1 
*/
struct reg_conv_kernel_size_minus_1 {
	union {
		struct {
			uint32_t dim0 : 8,
				dim1 : 8,
				dim2 : 8,
				dim3 : 8;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_conv_kernel_size_minus_1) == 4), "reg_conv_kernel_size_minus_1 size is not 32-bit");
/*
 CONV_ASSOCIATED_DIMS_LOW 
*/
struct reg_conv_associated_dims_low {
	union {
		struct {
			uint32_t dim_s_dim0 : 3,
				dim_l_dim0 : 3,
				dim_o_dim0 : 3,
				_reserved16 : 7,
				dim_s_dim1 : 3,
				dim_l_dim1 : 3,
				dim_o_dim1 : 3,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_conv_associated_dims_low) == 4), "reg_conv_associated_dims_low size is not 32-bit");
/*
 CONV_ASSOCIATED_DIMS_HIGH 
*/
struct reg_conv_associated_dims_high {
	union {
		struct {
			uint32_t dim_s_dim2 : 3,
				dim_l_dim2 : 3,
				dim_o_dim2 : 3,
				_reserved16 : 7,
				dim_s_dim3 : 3,
				dim_l_dim3 : 3,
				dim_o_dim3 : 3,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_conv_associated_dims_high) == 4), "reg_conv_associated_dims_high size is not 32-bit");
/*
 NUM_ITERATIONS_MINUS_1 
*/
struct reg_num_iterations_minus_1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_num_iterations_minus_1) == 4), "reg_num_iterations_minus_1 size is not 32-bit");
/*
 OUTER_LOOP 
*/
struct reg_outer_loop {
	union {
		struct {
			uint32_t dim_s : 3,
				dim_l : 3,
				dim_o : 3,
				_reserved16 : 7,
				size_minus_1 : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_outer_loop) == 4), "reg_outer_loop size is not 32-bit");
} /* mme_non_tensor_descriptor_start namespace */

/*
 MME_NON_TENSOR_DESCRIPTOR_START block
*/
struct block_mme_non_tensor_descriptor_start {
	struct mme_non_tensor_descriptor_start::reg_base_addr_high_s base_addr_high_s;
	struct mme_non_tensor_descriptor_start::reg_base_addr_high_l base_addr_high_l;
	struct mme_non_tensor_descriptor_start::reg_base_addr_high_o base_addr_high_o;
	struct mme_non_tensor_descriptor_start::reg_base_addr_low_s base_addr_low_s;
	struct mme_non_tensor_descriptor_start::reg_base_addr_low_l base_addr_low_l;
	struct mme_non_tensor_descriptor_start::reg_base_addr_low_o base_addr_low_o;
	struct mme_non_tensor_descriptor_start::reg_header_low header_low;
	struct mme_non_tensor_descriptor_start::reg_header_high header_high;
	struct mme_non_tensor_descriptor_start::reg_conv_kernel_size_minus_1 conv_kernel_size_minus_1;
	struct mme_non_tensor_descriptor_start::reg_conv_associated_dims_low conv_associated_dims_low;
	struct mme_non_tensor_descriptor_start::reg_conv_associated_dims_high conv_associated_dims_high;
	struct mme_non_tensor_descriptor_start::reg_num_iterations_minus_1 num_iterations_minus_1;
	struct mme_non_tensor_descriptor_start::reg_outer_loop outer_loop;
};

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_MME_NON_TENSOR_DESCRIPTOR_START_H_ */

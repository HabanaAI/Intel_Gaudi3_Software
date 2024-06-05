/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_MME_TENSOR_H_
#define ASIC_REG_STRUCTS_MME_TENSOR_H_

#include <stdint.h>

#pragma pack(push, 1)

namespace mme_tensor {
/*
 VALID_ELEMENTS 
*/
struct reg_valid_elements {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_valid_elements) == 4), "reg_valid_elements size is not 32-bit");
/*
 LOOP_STRIDE 
*/
struct reg_loop_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_loop_stride) == 4), "reg_loop_stride size is not 32-bit");
/*
 ROI_SIZE 
*/
struct reg_roi_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_roi_size) == 4), "reg_roi_size size is not 32-bit");
/*
 SPATIAL_STRIDES 
*/
struct reg_spatial_strides {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_spatial_strides) == 4), "reg_spatial_strides size is not 32-bit");
/*
 SPATIAL_SIZE_MINUS_1 
*/
struct reg_spatial_size_minus_1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_spatial_size_minus_1) == 4), "reg_spatial_size_minus_1 size is not 32-bit");
} /* mme_tensor namespace */

/*
 MME_TENSOR block
*/
struct block_mme_tensor {
	struct mme_tensor::reg_valid_elements valid_elements[5];
	struct mme_tensor::reg_loop_stride loop_stride[5];
	struct mme_tensor::reg_roi_size roi_size[4];
	struct mme_tensor::reg_spatial_strides spatial_strides[4];
	struct mme_tensor::reg_spatial_size_minus_1 spatial_size_minus_1;
};

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_MME_TENSOR_H_ */

/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI2_MME_TENSOR_H_
#define ASIC_REG_STRUCTS_GAUDI2_MME_TENSOR_H_

#include <stdint.h>
#include "gaudi2_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi2 {
namespace mme_tensor {
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
 VALID_ELEMENTS 
 b'TENSOR VALID ELEMENTS'
*/
typedef struct reg_valid_elements {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_valid_elements;
static_assert((sizeof(struct reg_valid_elements) == 4), "reg_valid_elements size is not 32-bit");
/*
 LOOP_STRIDE 
 b'TENSOR LOOP STRIDE'
*/
typedef struct reg_loop_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_loop_stride;
static_assert((sizeof(struct reg_loop_stride) == 4), "reg_loop_stride size is not 32-bit");
/*
 ROI_SIZE 
 b'TENSOR ROI SIZE'
*/
typedef struct reg_roi_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_roi_size;
static_assert((sizeof(struct reg_roi_size) == 4), "reg_roi_size size is not 32-bit");
/*
 SPATIAL_STRIDES 
 b'TENSOR SPATIAL STRIDES'
*/
typedef struct reg_spatial_strides {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_spatial_strides;
static_assert((sizeof(struct reg_spatial_strides) == 4), "reg_spatial_strides size is not 32-bit");
/*
 START_OFFSET 
 b'TENSOR START OFFSET'
*/
typedef struct reg_start_offset {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_start_offset;
static_assert((sizeof(struct reg_start_offset) == 4), "reg_start_offset size is not 32-bit");

#ifdef __cplusplus
} /* mme_tensor namespace */
#endif

/*
 MME_TENSOR block
*/

#ifdef __cplusplus

struct block_mme_tensor {
	struct mme_tensor::reg_valid_elements valid_elements[5];
	struct mme_tensor::reg_loop_stride loop_stride[5];
	struct mme_tensor::reg_roi_size roi_size[4];
	struct mme_tensor::reg_spatial_strides spatial_strides[4];
	struct mme_tensor::reg_start_offset start_offset[4];
};
#else

typedef struct block_mme_tensor {
	reg_valid_elements valid_elements[5];
	reg_loop_stride loop_stride[5];
	reg_roi_size roi_size[4];
	reg_spatial_strides spatial_strides[4];
	reg_start_offset start_offset[4];
} block_mme_tensor;
#endif


#ifdef __cplusplus
} /* gaudi2 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI2_MME_TENSOR_H_ */

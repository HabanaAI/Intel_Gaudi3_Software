/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_TENSOR_SHARED_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_TENSOR_SHARED_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc_tensor_shared {
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
 PADDING_VALUE 
 b'padding value when tensor access is out of bound'
*/
typedef struct reg_padding_value {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_padding_value;
static_assert((sizeof(struct reg_padding_value) == 4), "reg_padding_value size is not 32-bit");
/*
 TENSOR_CONFIG 
 b'general tensor configuration'
*/
typedef struct reg_tensor_config {
	union {
		struct {
			uint32_t data_type : 4,
				_reserved8 : 4,
				valid_dim_mask : 5,
				_reserved16 : 3,
				last_dim : 3,
				rmw_set : 1,
				_reserved21 : 1,
				rmw_op : 3,
				dup_oob : 1,
				l0cd : 1,
				t_pref_dis : 1,
				l2_pref_dis : 1,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_tensor_config;
static_assert((sizeof(struct reg_tensor_config) == 4), "reg_tensor_config size is not 32-bit");
/*
 DIM_0_SIZE 
 b'number of elements in dimension 0'
*/
typedef struct reg_dim_0_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_0_size;
static_assert((sizeof(struct reg_dim_0_size) == 4), "reg_dim_0_size size is not 32-bit");
/*
 DIM_0_STRIDE 
 b'dimension 0 stride'
*/
typedef struct reg_dim_0_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_0_stride;
static_assert((sizeof(struct reg_dim_0_stride) == 4), "reg_dim_0_stride size is not 32-bit");
/*
 DIM_1_SIZE 
 b'number of elements in dimension 1'
*/
typedef struct reg_dim_1_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_1_size;
static_assert((sizeof(struct reg_dim_1_size) == 4), "reg_dim_1_size size is not 32-bit");
/*
 DIM_1_STRIDE 
 b'dimension 1 stride'
*/
typedef struct reg_dim_1_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_1_stride;
static_assert((sizeof(struct reg_dim_1_stride) == 4), "reg_dim_1_stride size is not 32-bit");
/*
 DIM_2_SIZE 
 b'number of elements in dimension 2'
*/
typedef struct reg_dim_2_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_2_size;
static_assert((sizeof(struct reg_dim_2_size) == 4), "reg_dim_2_size size is not 32-bit");
/*
 DIM_2_STRIDE 
 b'dimension 2 stride'
*/
typedef struct reg_dim_2_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_2_stride;
static_assert((sizeof(struct reg_dim_2_stride) == 4), "reg_dim_2_stride size is not 32-bit");
/*
 DIM_3_SIZE 
 b'number of elements in dimension 3'
*/
typedef struct reg_dim_3_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_3_size;
static_assert((sizeof(struct reg_dim_3_size) == 4), "reg_dim_3_size size is not 32-bit");
/*
 DIM_3_STRIDE 
 b'dimension 3 stride'
*/
typedef struct reg_dim_3_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_3_stride;
static_assert((sizeof(struct reg_dim_3_stride) == 4), "reg_dim_3_stride size is not 32-bit");
/*
 DIM_4_SIZE 
 b'number of elements in dimension4'
*/
typedef struct reg_dim_4_size {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_4_size;
static_assert((sizeof(struct reg_dim_4_size) == 4), "reg_dim_4_size size is not 32-bit");
/*
 DIM_4_STRIDE 
 b'dimension 4 stride'
*/
typedef struct reg_dim_4_stride {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_dim_4_stride;
static_assert((sizeof(struct reg_dim_4_stride) == 4), "reg_dim_4_stride size is not 32-bit");
/*
 DIM_0_SIZE_STRIDE_H 
 b'Size and Stride high part dim 0'
*/
typedef struct reg_dim_0_size_stride_h {
	union {
		struct {
			uint32_t stride_h : 12,
				_reserved16 : 4,
				size_h : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_dim_0_size_stride_h;
static_assert((sizeof(struct reg_dim_0_size_stride_h) == 4), "reg_dim_0_size_stride_h size is not 32-bit");
/*
 DIM_1_SIZE_STRIDE_H 
 b'Size and Stride high part dim 1'
*/
typedef struct reg_dim_1_size_stride_h {
	union {
		struct {
			uint32_t stride_h : 12,
				_reserved16 : 4,
				size_h : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_dim_1_size_stride_h;
static_assert((sizeof(struct reg_dim_1_size_stride_h) == 4), "reg_dim_1_size_stride_h size is not 32-bit");
/*
 DIM_2_SIZE_STRIDE_H 
 b'Size and Stride high part dim 2'
*/
typedef struct reg_dim_2_size_stride_h {
	union {
		struct {
			uint32_t stride_h : 12,
				_reserved16 : 4,
				size_h : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_dim_2_size_stride_h;
static_assert((sizeof(struct reg_dim_2_size_stride_h) == 4), "reg_dim_2_size_stride_h size is not 32-bit");
/*
 DIM_3_SIZE_STRIDE_H 
 b'Size and Stride high part dim 3'
*/
typedef struct reg_dim_3_size_stride_h {
	union {
		struct {
			uint32_t stride_h : 12,
				_reserved16 : 4,
				size_h : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_dim_3_size_stride_h;
static_assert((sizeof(struct reg_dim_3_size_stride_h) == 4), "reg_dim_3_size_stride_h size is not 32-bit");
/*
 DIM_4_SIZE_STRIDE_H 
 b'Size and Stride high part dim 4'
*/
typedef struct reg_dim_4_size_stride_h {
	union {
		struct {
			uint32_t stride_h : 12,
				_reserved16 : 4,
				size_h : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_dim_4_size_stride_h;
static_assert((sizeof(struct reg_dim_4_size_stride_h) == 4), "reg_dim_4_size_stride_h size is not 32-bit");
/*
 HBW_AXI_CFG 
 b'hbw axi configuration (MCID,ARCACH,AWCACHE,CLASS)'
*/
typedef struct reg_hbw_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				arcache : 4,
				awcache : 4,
				clas : 2,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
} reg_hbw_axi_cfg;
static_assert((sizeof(struct reg_hbw_axi_cfg) == 4), "reg_hbw_axi_cfg size is not 32-bit");
/*
 TSB_ST_MODE 
 b'ST_TNSR mode for SB cache invalidate'
*/
typedef struct reg_tsb_st_mode {
	union {
		struct {
			uint32_t sel : 1,
				cfg : 2,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_tsb_st_mode;
static_assert((sizeof(struct reg_tsb_st_mode) == 4), "reg_tsb_st_mode size is not 32-bit");

#ifdef __cplusplus
} /* tpc_tensor_shared namespace */
#endif

/*
 TPC_TENSOR_SHARED block
*/

#ifdef __cplusplus

struct block_tpc_tensor_shared {
	struct tpc_tensor_shared::reg_padding_value padding_value;
	struct tpc_tensor_shared::reg_tensor_config tensor_config;
	struct tpc_tensor_shared::reg_dim_0_size dim_0_size;
	struct tpc_tensor_shared::reg_dim_0_stride dim_0_stride;
	struct tpc_tensor_shared::reg_dim_1_size dim_1_size;
	struct tpc_tensor_shared::reg_dim_1_stride dim_1_stride;
	struct tpc_tensor_shared::reg_dim_2_size dim_2_size;
	struct tpc_tensor_shared::reg_dim_2_stride dim_2_stride;
	struct tpc_tensor_shared::reg_dim_3_size dim_3_size;
	struct tpc_tensor_shared::reg_dim_3_stride dim_3_stride;
	struct tpc_tensor_shared::reg_dim_4_size dim_4_size;
	struct tpc_tensor_shared::reg_dim_4_stride dim_4_stride;
	struct tpc_tensor_shared::reg_dim_0_size_stride_h dim_0_size_stride_h;
	struct tpc_tensor_shared::reg_dim_1_size_stride_h dim_1_size_stride_h;
	struct tpc_tensor_shared::reg_dim_2_size_stride_h dim_2_size_stride_h;
	struct tpc_tensor_shared::reg_dim_3_size_stride_h dim_3_size_stride_h;
	struct tpc_tensor_shared::reg_dim_4_size_stride_h dim_4_size_stride_h;
	struct tpc_tensor_shared::reg_hbw_axi_cfg hbw_axi_cfg;
	struct tpc_tensor_shared::reg_tsb_st_mode tsb_st_mode;
};
#else

typedef struct block_tpc_tensor_shared {
	reg_padding_value padding_value;
	reg_tensor_config tensor_config;
	reg_dim_0_size dim_0_size;
	reg_dim_0_stride dim_0_stride;
	reg_dim_1_size dim_1_size;
	reg_dim_1_stride dim_1_stride;
	reg_dim_2_size dim_2_size;
	reg_dim_2_stride dim_2_stride;
	reg_dim_3_size dim_3_size;
	reg_dim_3_stride dim_3_stride;
	reg_dim_4_size dim_4_size;
	reg_dim_4_stride dim_4_stride;
	reg_dim_0_size_stride_h dim_0_size_stride_h;
	reg_dim_1_size_stride_h dim_1_size_stride_h;
	reg_dim_2_size_stride_h dim_2_size_stride_h;
	reg_dim_3_size_stride_h dim_3_size_stride_h;
	reg_dim_4_size_stride_h dim_4_size_stride_h;
	reg_hbw_axi_cfg hbw_axi_cfg;
	reg_tsb_st_mode tsb_st_mode;
} block_tpc_tensor_shared;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_tpc_tensor_shared_defaults[] =
{
	// offset	// value
	{ 0x4   , 0x50000             , 1 }, // tensor_config
	{ 0x44  , 0xba0000            , 1 }, // hbw_axi_cfg
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_TENSOR_SHARED_H_ */

/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_QM_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_QM_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc_non_tensor_descriptor_qm {
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
 KERNEL_FP8_BIAS 
 b'FP8 143 Bias Value'
*/
typedef struct reg_kernel_fp8_bias {
	union {
		struct {
			uint32_t bias_143 : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_kernel_fp8_bias;
static_assert((sizeof(struct reg_kernel_fp8_bias) == 4), "reg_kernel_fp8_bias size is not 32-bit");
/*
 CONVERT_CFG 
 b'Convert configuration register'
*/
typedef struct reg_convert_cfg {
	union {
		struct {
			uint32_t dnorm_ftz : 1,
				no_clip_fp_inf_input : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_convert_cfg;
static_assert((sizeof(struct reg_convert_cfg) == 4), "reg_convert_cfg size is not 32-bit");
/*
 CONVERT_FP32_FP8_CFG 
 b'convert FP32 to FP8 configuration register'
*/
typedef struct reg_convert_fp32_fp8_cfg {
	union {
		struct {
			uint32_t bias : 6,
				fp8_143_no_inf_nan : 2,
				fp8_152_no_inf_nan : 2,
				stoch_ftz_fp8 : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_convert_fp32_fp8_cfg;
static_assert((sizeof(struct reg_convert_fp32_fp8_cfg) == 4), "reg_convert_fp32_fp8_cfg size is not 32-bit");
/*
 CONVERT_FP32_FP16_CFG 
 b'convert FP32 to FP16 configuration register'
*/
typedef struct reg_convert_fp32_fp16_cfg {
	union {
		struct {
			uint32_t bias : 6,
				fp16_no_inf_nan : 1,
				uns : 1,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_convert_fp32_fp16_cfg;
static_assert((sizeof(struct reg_convert_fp32_fp16_cfg) == 4), "reg_convert_fp32_fp16_cfg size is not 32-bit");

#ifdef __cplusplus
} /* tpc_non_tensor_descriptor_qm namespace */
#endif

/*
 TPC_NON_TENSOR_DESCRIPTOR_QM block
*/

#ifdef __cplusplus

struct block_tpc_non_tensor_descriptor_qm {
	struct tpc_non_tensor_descriptor_qm::reg_kernel_fp8_bias kernel_fp8_bias;
	struct tpc_non_tensor_descriptor_qm::reg_convert_cfg convert_cfg;
	struct tpc_non_tensor_descriptor_qm::reg_convert_fp32_fp8_cfg convert_fp32_fp8_cfg;
	struct tpc_non_tensor_descriptor_qm::reg_convert_fp32_fp16_cfg convert_fp32_fp16_cfg;
};
#else

typedef struct block_tpc_non_tensor_descriptor_qm {
	reg_kernel_fp8_bias kernel_fp8_bias;
	reg_convert_cfg convert_cfg;
	reg_convert_fp32_fp8_cfg convert_fp32_fp8_cfg;
	reg_convert_fp32_fp16_cfg convert_fp32_fp16_cfg;
} block_tpc_non_tensor_descriptor_qm;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_tpc_non_tensor_descriptor_qm_defaults[] =
{
	// offset	// value
	{ 0x0   , 0x7                 , 1 }, // kernel_fp8_bias
	{ 0x4   , 0x2                 , 1 }, // convert_cfg
	{ 0x8   , 0x8f                , 1 }, // convert_fp32_fp8_cfg
	{ 0xc   , 0xf                 , 1 }, // convert_fp32_fp16_cfg
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_QM_H_ */

/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI2_MME_NON_TENSOR_DESCRIPTOR_H_
#define ASIC_REG_STRUCTS_GAUDI2_MME_NON_TENSOR_DESCRIPTOR_H_

#include <stdint.h>
#include "gaudi2_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi2 {
namespace mme_non_tensor_descriptor {
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
 CONV_KERNEL_SIZE_MINUS_1 
 b'CMD Configurations Part0'
*/
typedef struct reg_conv_kernel_size_minus_1 {
	union {
		struct {
			uint32_t dim0 : 8,
				dim1 : 8,
				dim2 : 8,
				dim3 : 8;
		};
		uint32_t _raw;
	};
} reg_conv_kernel_size_minus_1;
static_assert((sizeof(struct reg_conv_kernel_size_minus_1) == 4), "reg_conv_kernel_size_minus_1 size is not 32-bit");
/*
 CONV_LOW 
 b'CMD Configurations Part1'
*/
typedef struct reg_conv_low {
	union {
		struct {
			uint32_t dim0_associated_dims_a : 3,
				dim0_associated_dims_b : 3,
				dim0_associated_dims_cout : 3,
				_reserved16 : 7,
				dim1_associated_dims_a : 3,
				dim1_associated_dims_b : 3,
				dim1_associated_dims_cout : 3,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_conv_low;
static_assert((sizeof(struct reg_conv_low) == 4), "reg_conv_low size is not 32-bit");
/*
 CONV_HIGH 
 b'Command configuration Part 2'
*/
typedef struct reg_conv_high {
	union {
		struct {
			uint32_t dim2_associated_dims_a : 3,
				dim2_associated_dims_b : 3,
				dim2_associated_dims_cout : 3,
				_reserved16 : 7,
				dim3_associated_dims_a : 3,
				dim3_associated_dims_b : 3,
				dim3_associated_dims_cout : 3,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_conv_high;
static_assert((sizeof(struct reg_conv_high) == 4), "reg_conv_high size is not 32-bit");
/*
 OUTER_LOOP 
 b'Outer loop configuration'
*/
typedef struct reg_outer_loop {
	union {
		struct {
			uint32_t associated_dims_a : 3,
				associated_dims_b : 3,
				associated_dims_cout : 3,
				_reserved16 : 7,
				size_minus_1 : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_outer_loop;
static_assert((sizeof(struct reg_outer_loop) == 4), "reg_outer_loop size is not 32-bit");
/*
 NUM_ITERATIONS_MINUS_1 
 b'Num iterations minus 1'
*/
typedef struct reg_num_iterations_minus_1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_num_iterations_minus_1;
static_assert((sizeof(struct reg_num_iterations_minus_1) == 4), "reg_num_iterations_minus_1 size is not 32-bit");
/*
 SB_REPEAT 
 b'LOOP REPEAT configs'
*/
typedef struct reg_sb_repeat {
	union {
		struct {
			uint32_t repeat_a_minus_1 : 8,
				repeat_b_minus_1 : 8,
				repeat_a_mask : 6,
				_reserved24 : 2,
				repeat_b_mask : 6,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_sb_repeat;
static_assert((sizeof(struct reg_sb_repeat) == 4), "reg_sb_repeat size is not 32-bit");
/*
 FP8_BIAS 
 b'FP8 Bias'
*/
typedef struct reg_fp8_bias {
	union {
		struct {
			uint32_t a : 4,
				b : 4,
				out : 5,
				_reserved13 : 19;
		};
		uint32_t _raw;
	};
} reg_fp8_bias;
static_assert((sizeof(struct reg_fp8_bias) == 4), "reg_fp8_bias size is not 32-bit");
/*
 RATE_LIMITER 
 b'SB Rate Limiter Reset Tokens'
*/
typedef struct reg_rate_limiter {
	union {
		struct {
			uint32_t agu_a : 8,
				agu_b : 8,
				agu_cout : 8,
				eu : 8;
		};
		uint32_t _raw;
	};
} reg_rate_limiter;
static_assert((sizeof(struct reg_rate_limiter) == 4), "reg_rate_limiter size is not 32-bit");
/*
 USER_DATA 
 b'AxUSER Static config'
*/
typedef struct reg_user_data {
	union {
		struct {
			uint32_t first : 10,
				steady : 10,
				mask : 6,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
} reg_user_data;
static_assert((sizeof(struct reg_user_data) == 4), "reg_user_data size is not 32-bit");
/*
 PERF_EVT_IN 
 b'SB Perf Events Input Operands Config'
*/
typedef struct reg_perf_evt_in {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				start_end_mask : 2,
				loop_mask : 6,
				operand : 5,
				slave_send_perf_evt : 1;
		};
		uint32_t _raw;
	};
} reg_perf_evt_in;
static_assert((sizeof(struct reg_perf_evt_in) == 4), "reg_perf_evt_in size is not 32-bit");
/*
 PERF_EVT_OUT 
 b'WBC Perf Events Config'
*/
typedef struct reg_perf_evt_out {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				start_end_mask : 2,
				loop_mask : 6,
				operand : 5,
				slave_send_perf_evt : 1;
		};
		uint32_t _raw;
	};
} reg_perf_evt_out;
static_assert((sizeof(struct reg_perf_evt_out) == 4), "reg_perf_evt_out size is not 32-bit");
/*
 PCU 
 b'PMU Rate Limiter Saturation'
*/
typedef struct reg_pcu {
	union {
		struct {
			uint32_t rl_saturation : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_pcu;
static_assert((sizeof(struct reg_pcu) == 4), "reg_pcu size is not 32-bit");
/*
 SLAVE_SYNC_OBJ0_ADDR 
 b'Sync Object0 Addr'
*/
typedef struct reg_slave_sync_obj0_addr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_slave_sync_obj0_addr;
static_assert((sizeof(struct reg_slave_sync_obj0_addr) == 4), "reg_slave_sync_obj0_addr size is not 32-bit");
/*
 SLAVE_SYNC_OBJ1_ADDR 
 b'Sync Object1 Addr'
*/
typedef struct reg_slave_sync_obj1_addr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_slave_sync_obj1_addr;
static_assert((sizeof(struct reg_slave_sync_obj1_addr) == 4), "reg_slave_sync_obj1_addr size is not 32-bit");
/*
 POWER_LOOP 
 b'Power Loop Configs'
*/
typedef struct reg_power_loop {
	union {
		struct {
			uint32_t ctrl_master : 2,
				ctrl_slave : 2,
				md : 8,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_power_loop;
static_assert((sizeof(struct reg_power_loop) == 4), "reg_power_loop size is not 32-bit");
/*
 SPARE0_MASTER 
 b'SPARE Configs'
*/
typedef struct reg_spare0_master {
	union {
		struct {
			uint32_t sb0 : 16,
				sb1 : 16;
		};
		uint32_t _raw;
	};
} reg_spare0_master;
static_assert((sizeof(struct reg_spare0_master) == 4), "reg_spare0_master size is not 32-bit");
/*
 SPARE1_MASTER 
 b'Spare configs'
*/
typedef struct reg_spare1_master {
	union {
		struct {
			uint32_t sb2 : 16,
				sb3 : 16;
		};
		uint32_t _raw;
	};
} reg_spare1_master;
static_assert((sizeof(struct reg_spare1_master) == 4), "reg_spare1_master size is not 32-bit");
/*
 SPARE2_MASTER 
 b'Spare configs'
*/
typedef struct reg_spare2_master {
	union {
		struct {
			uint32_t sb4 : 16,
				out : 16;
		};
		uint32_t _raw;
	};
} reg_spare2_master;
static_assert((sizeof(struct reg_spare2_master) == 4), "reg_spare2_master size is not 32-bit");
/*
 SPARE3_MASTER 
 b'Spare configs'
*/
typedef struct reg_spare3_master {
	union {
		struct {
			uint32_t eu : 16,
				ap : 16;
		};
		uint32_t _raw;
	};
} reg_spare3_master;
static_assert((sizeof(struct reg_spare3_master) == 4), "reg_spare3_master size is not 32-bit");
/*
 SPARE0_SLAVE 
 b'SPARE Configs'
*/
typedef struct reg_spare0_slave {
	union {
		struct {
			uint32_t sb0 : 16,
				sb1 : 16;
		};
		uint32_t _raw;
	};
} reg_spare0_slave;
static_assert((sizeof(struct reg_spare0_slave) == 4), "reg_spare0_slave size is not 32-bit");
/*
 SPARE1_SLAVE 
 b'Spare configs'
*/
typedef struct reg_spare1_slave {
	union {
		struct {
			uint32_t sb2 : 16,
				sb3 : 16;
		};
		uint32_t _raw;
	};
} reg_spare1_slave;
static_assert((sizeof(struct reg_spare1_slave) == 4), "reg_spare1_slave size is not 32-bit");
/*
 SPARE2_SLAVE 
 b'Spare configs'
*/
typedef struct reg_spare2_slave {
	union {
		struct {
			uint32_t sb4 : 16,
				out : 16;
		};
		uint32_t _raw;
	};
} reg_spare2_slave;
static_assert((sizeof(struct reg_spare2_slave) == 4), "reg_spare2_slave size is not 32-bit");
/*
 SPARE3_SLAVE 
 b'Spare configs'
*/
typedef struct reg_spare3_slave {
	union {
		struct {
			uint32_t eu : 16,
				ap : 16;
		};
		uint32_t _raw;
	};
} reg_spare3_slave;
static_assert((sizeof(struct reg_spare3_slave) == 4), "reg_spare3_slave size is not 32-bit");
/*
 WKL_ID 
 b'WorkLoad ID and CMD STEP OFFSET'
*/
typedef struct reg_wkl_id {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_wkl_id;
static_assert((sizeof(struct reg_wkl_id) == 4), "reg_wkl_id size is not 32-bit");

#ifdef __cplusplus
} /* mme_non_tensor_descriptor namespace */
#endif

/*
 MME_NON_TENSOR_DESCRIPTOR block
*/

#ifdef __cplusplus

struct block_mme_non_tensor_descriptor {
	struct mme_non_tensor_descriptor::reg_conv_kernel_size_minus_1 conv_kernel_size_minus_1;
	struct mme_non_tensor_descriptor::reg_conv_low conv_low;
	struct mme_non_tensor_descriptor::reg_conv_high conv_high;
	struct mme_non_tensor_descriptor::reg_outer_loop outer_loop;
	struct mme_non_tensor_descriptor::reg_num_iterations_minus_1 num_iterations_minus_1;
	struct mme_non_tensor_descriptor::reg_sb_repeat sb_repeat;
	struct mme_non_tensor_descriptor::reg_fp8_bias fp8_bias;
	struct mme_non_tensor_descriptor::reg_rate_limiter rate_limiter;
	struct mme_non_tensor_descriptor::reg_user_data user_data;
	struct mme_non_tensor_descriptor::reg_perf_evt_in perf_evt_in;
	struct mme_non_tensor_descriptor::reg_perf_evt_out perf_evt_out;
	struct mme_non_tensor_descriptor::reg_pcu pcu;
	struct mme_non_tensor_descriptor::reg_slave_sync_obj0_addr slave_sync_obj0_addr;
	struct mme_non_tensor_descriptor::reg_slave_sync_obj1_addr slave_sync_obj1_addr;
	struct mme_non_tensor_descriptor::reg_power_loop power_loop;
	struct mme_non_tensor_descriptor::reg_spare0_master spare0_master;
	struct mme_non_tensor_descriptor::reg_spare1_master spare1_master;
	struct mme_non_tensor_descriptor::reg_spare2_master spare2_master;
	struct mme_non_tensor_descriptor::reg_spare3_master spare3_master;
	struct mme_non_tensor_descriptor::reg_spare0_slave spare0_slave;
	struct mme_non_tensor_descriptor::reg_spare1_slave spare1_slave;
	struct mme_non_tensor_descriptor::reg_spare2_slave spare2_slave;
	struct mme_non_tensor_descriptor::reg_spare3_slave spare3_slave;
	struct mme_non_tensor_descriptor::reg_wkl_id wkl_id;
};
#else

typedef struct block_mme_non_tensor_descriptor {
	reg_conv_kernel_size_minus_1 conv_kernel_size_minus_1;
	reg_conv_low conv_low;
	reg_conv_high conv_high;
	reg_outer_loop outer_loop;
	reg_num_iterations_minus_1 num_iterations_minus_1;
	reg_sb_repeat sb_repeat;
	reg_fp8_bias fp8_bias;
	reg_rate_limiter rate_limiter;
	reg_user_data user_data;
	reg_perf_evt_in perf_evt_in;
	reg_perf_evt_out perf_evt_out;
	reg_pcu pcu;
	reg_slave_sync_obj0_addr slave_sync_obj0_addr;
	reg_slave_sync_obj1_addr slave_sync_obj1_addr;
	reg_power_loop power_loop;
	reg_spare0_master spare0_master;
	reg_spare1_master spare1_master;
	reg_spare2_master spare2_master;
	reg_spare3_master spare3_master;
	reg_spare0_slave spare0_slave;
	reg_spare1_slave spare1_slave;
	reg_spare2_slave spare2_slave;
	reg_spare3_slave spare3_slave;
	reg_wkl_id wkl_id;
} block_mme_non_tensor_descriptor;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_mme_non_tensor_descriptor_defaults[] =
{
	// offset	// value
	{ 0x1c  , 0x8040404           , 1 }, // rate_limiter
};
#endif

#ifdef __cplusplus
} /* gaudi2 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI2_MME_NON_TENSOR_DESCRIPTOR_H_ */

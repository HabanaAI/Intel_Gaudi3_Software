/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_MME_NON_TENSOR_DESCRIPTOR_H_
#define ASIC_REG_STRUCTS_GAUDI3_MME_NON_TENSOR_DESCRIPTOR_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
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
 CONV_LO 
 b'CMD Configurations Part1'
*/
typedef struct reg_conv_lo {
	union {
		struct {
			uint32_t dim0_assoc_dims_a : 3,
				dim0_assoc_dims_b : 3,
				dim0_assoc_dims_cout : 3,
				_reserved16 : 7,
				dim1_assoc_dims_a : 3,
				dim1_assoc_dims_b : 3,
				dim1_assoc_dims_cout : 3,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_conv_lo;
static_assert((sizeof(struct reg_conv_lo) == 4), "reg_conv_lo size is not 32-bit");
/*
 CONV_HI 
*/
typedef struct reg_conv_hi {
	union {
		struct {
			uint32_t dim2_assoc_dims_a : 3,
				dim2_assoc_dims_b : 3,
				dim2_assoc_dims_cout : 3,
				_reserved16 : 7,
				dim3_assoc_dims_a : 3,
				dim3_assoc_dims_b : 3,
				dim3_assoc_dims_cout : 3,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_conv_hi;
static_assert((sizeof(struct reg_conv_hi) == 4), "reg_conv_hi size is not 32-bit");
/*
 OUTER_LOOP 
*/
typedef struct reg_outer_loop {
	union {
		struct {
			uint32_t assoc_dims_a : 3,
				assoc_dims_b : 3,
				assoc_dims_cout : 3,
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
 SO_CTRL 
*/
typedef struct reg_so_ctrl {
	union {
		struct {
			uint32_t signal_mask0 : 6,
				signal_en0 : 1,
				_reserved8 : 1,
				signal_mask1 : 6,
				signal_en1 : 1,
				master_wait_slave_fence : 1,
				slave_send_fence2master : 1,
				slave_signal_en : 1,
				slave0_use_slv_adr : 1,
				slave1_use_slv_adr : 1,
				slave0_use_mstr_adr_plus4 : 1,
				slave1_use_mstr_adr_plus4 : 1,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_so_ctrl;
static_assert((sizeof(struct reg_so_ctrl) == 4), "reg_so_ctrl size is not 32-bit");
/*
 SO_ADDR0 
*/
typedef struct reg_so_addr0 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_so_addr0;
static_assert((sizeof(struct reg_so_addr0) == 4), "reg_so_addr0 size is not 32-bit");
/*
 SO_VAL0 
*/
typedef struct reg_so_val0 {
	union {
		struct {
			uint32_t so_value : 15,
				so_reserved : 15,
				so_perf_en : 1,
				so_op : 1;
		};
		uint32_t _raw;
	};
} reg_so_val0;
static_assert((sizeof(struct reg_so_val0) == 4), "reg_so_val0 size is not 32-bit");
/*
 SO_ADDR1 
*/
typedef struct reg_so_addr1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_so_addr1;
static_assert((sizeof(struct reg_so_addr1) == 4), "reg_so_addr1 size is not 32-bit");
/*
 SO_VAL1 
*/
typedef struct reg_so_val1 {
	union {
		struct {
			uint32_t so_value : 15,
				so_reserved : 15,
				so_perf_en : 1,
				so_op : 1;
		};
		uint32_t _raw;
	};
} reg_so_val1;
static_assert((sizeof(struct reg_so_val1) == 4), "reg_so_val1 size is not 32-bit");
/*
 SLAVE_SO_0_ADDR 
*/
typedef struct reg_slave_so_0_addr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_slave_so_0_addr;
static_assert((sizeof(struct reg_slave_so_0_addr) == 4), "reg_slave_so_0_addr size is not 32-bit");
/*
 SLAVE_SO_1_ADDR 
*/
typedef struct reg_slave_so_1_addr {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_slave_so_1_addr;
static_assert((sizeof(struct reg_slave_so_1_addr) == 4), "reg_slave_so_1_addr size is not 32-bit");
/*
 NUMERIC 
*/
typedef struct reg_numeric {
	union {
		struct {
			uint32_t bias_a : 6,
				bias_b : 6,
				bias_out : 6,
				acc_round_mode : 2,
				fp8_flav_a : 1,
				fp8_flav_b : 1,
				fp8_flav_out : 1,
				fp16_flav_a : 1,
				fp16_flav_b : 1,
				fp16_flav_out : 1,
				no_inf_nan_a : 2,
				no_inf_nan_b : 2,
				no_inf_nan_out : 2;
		};
		uint32_t _raw;
	};
} reg_numeric;
static_assert((sizeof(struct reg_numeric) == 4), "reg_numeric size is not 32-bit");
/*
 AXI_AWUSER_DATA 
 b'AWUSER reduction config'
*/
typedef struct reg_axi_awuser_data {
	union {
		struct {
			uint32_t first : 11,
				steady : 11,
				mask : 6,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_axi_awuser_data;
static_assert((sizeof(struct reg_axi_awuser_data) == 4), "reg_axi_awuser_data size is not 32-bit");
/*
 AXI_USER_DATA_A 
*/
typedef struct reg_axi_user_data_a {
	union {
		struct {
			uint32_t qos_first : 4,
				qos_steady : 4,
				qos_mask : 6,
				mcid : 16,
				class_ : 2;
		};
		uint32_t _raw;
	};
} reg_axi_user_data_a;
static_assert((sizeof(struct reg_axi_user_data_a) == 4), "reg_axi_user_data_a size is not 32-bit");
/*
 AXI_USER_DATA_B 
*/
typedef struct reg_axi_user_data_b {
	union {
		struct {
			uint32_t qos_first : 4,
				qos_steady : 4,
				qos_mask : 6,
				mcid : 16,
				class_ : 2;
		};
		uint32_t _raw;
	};
} reg_axi_user_data_b;
static_assert((sizeof(struct reg_axi_user_data_b) == 4), "reg_axi_user_data_b size is not 32-bit");
/*
 AXI_USER_DATA_COUT 
*/
typedef struct reg_axi_user_data_cout {
	union {
		struct {
			uint32_t qos_first : 4,
				qos_steady : 4,
				qos_mask : 6,
				mcid : 16,
				class_ : 2;
		};
		uint32_t _raw;
	};
} reg_axi_user_data_cout;
static_assert((sizeof(struct reg_axi_user_data_cout) == 4), "reg_axi_user_data_cout size is not 32-bit");
/*
 AXI_CACHE_DATA 
*/
typedef struct reg_axi_cache_data {
	union {
		struct {
			uint32_t agu_a : 4,
				agu_b : 4,
				agu_out : 4,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_axi_cache_data;
static_assert((sizeof(struct reg_axi_cache_data) == 4), "reg_axi_cache_data size is not 32-bit");
/*
 PERF_EVT_IN 
 b'SB Perf Events Input Operands Config'
*/
typedef struct reg_perf_evt_in {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_en : 1,
				start_end_en : 2,
				loop_mask : 6,
				operand : 4,
				slave_send_perf_evt : 1,
				_reserved31 : 1;
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
				inc_en : 1,
				start_end_en : 2,
				loop_mask : 6,
				operand : 4,
				slave_send_perf_evt : 1,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_perf_evt_out;
static_assert((sizeof(struct reg_perf_evt_out) == 4), "reg_perf_evt_out size is not 32-bit");
/*
 PERF_EVT_EU 
*/
typedef struct reg_perf_evt_eu {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_en : 1,
				start_end_en : 2,
				loop_mask : 6,
				operand : 4,
				slave_send_perf_evt : 1,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_perf_evt_eu;
static_assert((sizeof(struct reg_perf_evt_eu) == 4), "reg_perf_evt_eu size is not 32-bit");
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
 POWER 
 b'Power Loop Configs'
*/
typedef struct reg_power {
	union {
		struct {
			uint32_t loop_ctrl : 2,
				loop_md : 8,
				pmu_rl_saturation : 20,
				sb_opp_dis_a : 1,
				sb_opp_dis_b : 1;
		};
		uint32_t _raw;
	};
} reg_power;
static_assert((sizeof(struct reg_power) == 4), "reg_power size is not 32-bit");
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
*/
typedef struct reg_spare2_master {
	union {
		struct {
			uint32_t out : 16,
				eu : 16;
		};
		uint32_t _raw;
	};
} reg_spare2_master;
static_assert((sizeof(struct reg_spare2_master) == 4), "reg_spare2_master size is not 32-bit");
/*
 SPARE3_MASTER 
*/
typedef struct reg_spare3_master {
	union {
		struct {
			uint32_t ap : 16,
				_reserved16 : 16;
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
*/
typedef struct reg_spare2_slave {
	union {
		struct {
			uint32_t out : 16,
				eu : 16;
		};
		uint32_t _raw;
	};
} reg_spare2_slave;
static_assert((sizeof(struct reg_spare2_slave) == 4), "reg_spare2_slave size is not 32-bit");
/*
 SPARE3_SLAVE 
*/
typedef struct reg_spare3_slave {
	union {
		struct {
			uint32_t ap : 16,
				_reserved16 : 16;
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
	struct mme_non_tensor_descriptor::reg_conv_lo conv_lo;
	struct mme_non_tensor_descriptor::reg_conv_hi conv_hi;
	struct mme_non_tensor_descriptor::reg_outer_loop outer_loop;
	struct mme_non_tensor_descriptor::reg_num_iterations_minus_1 num_iterations_minus_1;
	struct mme_non_tensor_descriptor::reg_sb_repeat sb_repeat;
	struct mme_non_tensor_descriptor::reg_so_ctrl so_ctrl;
	struct mme_non_tensor_descriptor::reg_so_addr0 so_addr0;
	struct mme_non_tensor_descriptor::reg_so_val0 so_val0;
	struct mme_non_tensor_descriptor::reg_so_addr1 so_addr1;
	struct mme_non_tensor_descriptor::reg_so_val1 so_val1;
	struct mme_non_tensor_descriptor::reg_slave_so_0_addr slave_so_0_addr;
	struct mme_non_tensor_descriptor::reg_slave_so_1_addr slave_so_1_addr;
	struct mme_non_tensor_descriptor::reg_numeric numeric;
	struct mme_non_tensor_descriptor::reg_axi_awuser_data axi_awuser_data;
	struct mme_non_tensor_descriptor::reg_axi_user_data_a axi_user_data_a;
	struct mme_non_tensor_descriptor::reg_axi_user_data_b axi_user_data_b;
	struct mme_non_tensor_descriptor::reg_axi_user_data_cout axi_user_data_cout;
	struct mme_non_tensor_descriptor::reg_axi_cache_data axi_cache_data;
	struct mme_non_tensor_descriptor::reg_perf_evt_in perf_evt_in;
	struct mme_non_tensor_descriptor::reg_perf_evt_out perf_evt_out;
	struct mme_non_tensor_descriptor::reg_perf_evt_eu perf_evt_eu;
	struct mme_non_tensor_descriptor::reg_rate_limiter rate_limiter;
	struct mme_non_tensor_descriptor::reg_power power;
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
	reg_conv_lo conv_lo;
	reg_conv_hi conv_hi;
	reg_outer_loop outer_loop;
	reg_num_iterations_minus_1 num_iterations_minus_1;
	reg_sb_repeat sb_repeat;
	reg_so_ctrl so_ctrl;
	reg_so_addr0 so_addr0;
	reg_so_val0 so_val0;
	reg_so_addr1 so_addr1;
	reg_so_val1 so_val1;
	reg_slave_so_0_addr slave_so_0_addr;
	reg_slave_so_1_addr slave_so_1_addr;
	reg_numeric numeric;
	reg_axi_awuser_data axi_awuser_data;
	reg_axi_user_data_a axi_user_data_a;
	reg_axi_user_data_b axi_user_data_b;
	reg_axi_user_data_cout axi_user_data_cout;
	reg_axi_cache_data axi_cache_data;
	reg_perf_evt_in perf_evt_in;
	reg_perf_evt_out perf_evt_out;
	reg_perf_evt_eu perf_evt_eu;
	reg_rate_limiter rate_limiter;
	reg_power power;
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
	{ 0x58  , 0x8040404           , 1 }, // rate_limiter
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_MME_NON_TENSOR_DESCRIPTOR_H_ */

/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_MME_NON_TENSOR_DESCRIPTOR_END_H_
#define ASIC_REG_STRUCTS_MME_NON_TENSOR_DESCRIPTOR_END_H_

#include <stdint.h>

#pragma pack(push, 1)

namespace mme_non_tensor_descriptor_end {
/*
 SB_REPEAT 
*/
struct reg_sb_repeat {
	union {
		struct {
			uint32_t repeat_s_minus_1 : 8,
				agu_s_loop_mask : 6,
				load_s : 1,
				teb_s_en : 1,
				repeat_l_minus_1 : 8,
				agu_l_loop_mask : 6,
				load_l : 1,
				teb_l_en : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_sb_repeat) == 4), "reg_sb_repeat size is not 32-bit");
/*
 RATE_LIMITER 
*/
struct reg_rate_limiter {
	union {
		struct {
			uint32_t agu_s : 8,
				agu_l : 8,
				agu_o : 8,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_rate_limiter) == 4), "reg_rate_limiter size is not 32-bit");
/*
 SYNC_OBJECT_ADDR_LOW_LOCAL 
*/
struct reg_sync_object_addr_low_local {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_sync_object_addr_low_local) == 4), "reg_sync_object_addr_low_local size is not 32-bit");
/*
 SYNC_OBJECT_ADDR_LOW_REMOTE 
*/
struct reg_sync_object_addr_low_remote {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_sync_object_addr_low_remote) == 4), "reg_sync_object_addr_low_remote size is not 32-bit");
/*
 SYNC_OBJECT_ADDR_HIGH 
*/
struct reg_sync_object_addr_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_sync_object_addr_high) == 4), "reg_sync_object_addr_high size is not 32-bit");
/*
 SYNC_OBJECT_DATA 
*/
struct reg_sync_object_data {
	union {
		struct {
			uint32_t value : 15,
				_reserved30 : 15,
				perf_en : 1,
				operation : 1;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_sync_object_data) == 4), "reg_sync_object_data size is not 32-bit");
/*
 AXI_USER_DATA 
*/
struct reg_axi_user_data {
	union {
		struct {
			uint32_t first : 9,
				steady : 9,
				mask : 6,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_axi_user_data) == 4), "reg_axi_user_data size is not 32-bit");
/*
 PERF_EVT_S 
*/
struct reg_perf_evt_s {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				end_mask : 2,
				loop_mask : 6,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_perf_evt_s) == 4), "reg_perf_evt_s size is not 32-bit");
/*
 PERF_EVT_L_LOCAL 
*/
struct reg_perf_evt_l_local {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				end_mask : 2,
				loop_mask : 6,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_perf_evt_l_local) == 4), "reg_perf_evt_l_local size is not 32-bit");
/*
 PERF_EVT_L_REMOTE 
*/
struct reg_perf_evt_l_remote {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				end_mask : 2,
				loop_mask : 6,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_perf_evt_l_remote) == 4), "reg_perf_evt_l_remote size is not 32-bit");
/*
 PERF_EVT_O_LOCAL 
*/
struct reg_perf_evt_o_local {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				end_mask : 2,
				loop_mask : 6,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_perf_evt_o_local) == 4), "reg_perf_evt_o_local size is not 32-bit");
/*
 PERF_EVT_O_REMOTE 
*/
struct reg_perf_evt_o_remote {
	union {
		struct {
			uint32_t value : 16,
				rst : 1,
				inc_mask : 1,
				end_mask : 2,
				loop_mask : 6,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_perf_evt_o_remote) == 4), "reg_perf_evt_o_remote size is not 32-bit");
/*
 PADDING_VALUE_S 
*/
struct reg_padding_value_s {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_padding_value_s) == 4), "reg_padding_value_s size is not 32-bit");
/*
 PADDING_VALUE_L 
*/
struct reg_padding_value_l {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_padding_value_l) == 4), "reg_padding_value_l size is not 32-bit");
/*
 META_DATA_AGU_S 
*/
struct reg_meta_data_agu_s {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_meta_data_agu_s) == 4), "reg_meta_data_agu_s size is not 32-bit");
/*
 META_DATA_AGU_L_LOCAL 
*/
struct reg_meta_data_agu_l_local {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_meta_data_agu_l_local) == 4), "reg_meta_data_agu_l_local size is not 32-bit");
/*
 META_DATA_AGU_L_REMOTE 
*/
struct reg_meta_data_agu_l_remote {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_meta_data_agu_l_remote) == 4), "reg_meta_data_agu_l_remote size is not 32-bit");
/*
 META_DATA_AGU_O_LOCAL 
*/
struct reg_meta_data_agu_o_local {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_meta_data_agu_o_local) == 4), "reg_meta_data_agu_o_local size is not 32-bit");
/*
 META_DATA_AGU_O_REMOTE 
*/
struct reg_meta_data_agu_o_remote {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_meta_data_agu_o_remote) == 4), "reg_meta_data_agu_o_remote size is not 32-bit");
/*
 PCU_RL_SATURATION 
*/
struct reg_pcu_rl_saturation {
	union {
		struct {
			uint32_t v : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_pcu_rl_saturation) == 4), "reg_pcu_rl_saturation size is not 32-bit");
/*
 DUMMY 
*/
struct reg_dummy {
	union {
		struct {
			uint32_t v : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_dummy) == 4), "reg_dummy size is not 32-bit");
} /* mme_non_tensor_descriptor_end namespace */

/*
 MME_NON_TENSOR_DESCRIPTOR_END block
*/
struct block_mme_non_tensor_descriptor_end {
	struct mme_non_tensor_descriptor_end::reg_sb_repeat sb_repeat;
	struct mme_non_tensor_descriptor_end::reg_rate_limiter rate_limiter;
	struct mme_non_tensor_descriptor_end::reg_sync_object_addr_low_local sync_object_addr_low_local;
	struct mme_non_tensor_descriptor_end::reg_sync_object_addr_low_remote sync_object_addr_low_remote;
	struct mme_non_tensor_descriptor_end::reg_sync_object_addr_high sync_object_addr_high;
	struct mme_non_tensor_descriptor_end::reg_sync_object_data sync_object_data;
	struct mme_non_tensor_descriptor_end::reg_axi_user_data axi_user_data;
	struct mme_non_tensor_descriptor_end::reg_perf_evt_s perf_evt_s;
	struct mme_non_tensor_descriptor_end::reg_perf_evt_l_local perf_evt_l_local;
	struct mme_non_tensor_descriptor_end::reg_perf_evt_l_remote perf_evt_l_remote;
	struct mme_non_tensor_descriptor_end::reg_perf_evt_o_local perf_evt_o_local;
	struct mme_non_tensor_descriptor_end::reg_perf_evt_o_remote perf_evt_o_remote;
	struct mme_non_tensor_descriptor_end::reg_padding_value_s padding_value_s;
	struct mme_non_tensor_descriptor_end::reg_padding_value_l padding_value_l;
	struct mme_non_tensor_descriptor_end::reg_meta_data_agu_s meta_data_agu_s;
	struct mme_non_tensor_descriptor_end::reg_meta_data_agu_l_local meta_data_agu_l_local;
	struct mme_non_tensor_descriptor_end::reg_meta_data_agu_l_remote meta_data_agu_l_remote;
	struct mme_non_tensor_descriptor_end::reg_meta_data_agu_o_local meta_data_agu_o_local;
	struct mme_non_tensor_descriptor_end::reg_meta_data_agu_o_remote meta_data_agu_o_remote;
	struct mme_non_tensor_descriptor_end::reg_pcu_rl_saturation pcu_rl_saturation;
	struct mme_non_tensor_descriptor_end::reg_dummy dummy;
};

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_MME_NON_TENSOR_DESCRIPTOR_END_H_ */

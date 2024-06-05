/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_SOB_OBJS_H_
#define ASIC_REG_STRUCTS_GAUDI3_SOB_OBJS_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace sob_objs {
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
 SOB_OBJ_0 
 b'Sync object counters (0)'
*/
typedef struct reg_sob_obj_0 {
	union {
		struct {
			uint32_t val : 15,
				_reserved24 : 9,
				long_sob : 1,
				_reserved28 : 3,
				zero_sob_cnt : 1,
				_reserved30 : 1,
				trace_evict : 1,
				inc : 1;
		};
		uint32_t _raw;
	};
} reg_sob_obj_0;
static_assert((sizeof(struct reg_sob_obj_0) == 4), "reg_sob_obj_0 size is not 32-bit");
/*
 MON_PAY_ADDRL_0 
 b'Monitor Address (LSB)'
*/
typedef struct reg_mon_pay_addrl_0 {
	union {
		struct {
			uint32_t addrl : 32;
		};
		uint32_t _raw;
	};
} reg_mon_pay_addrl_0;
static_assert((sizeof(struct reg_mon_pay_addrl_0) == 4), "reg_mon_pay_addrl_0 size is not 32-bit");
/*
 MON_PAY_ADDRH_0 
 b'Monitor Address (MSB)'
*/
typedef struct reg_mon_pay_addrh_0 {
	union {
		struct {
			uint32_t addrh : 32;
		};
		uint32_t _raw;
	};
} reg_mon_pay_addrh_0;
static_assert((sizeof(struct reg_mon_pay_addrh_0) == 4), "reg_mon_pay_addrh_0 size is not 32-bit");
/*
 MON_PAY_DATA_0 
 b'Monitor Data'
*/
typedef struct reg_mon_pay_data_0 {
	union {
		struct {
			uint32_t data : 32;
		};
		uint32_t _raw;
	};
} reg_mon_pay_data_0;
static_assert((sizeof(struct reg_mon_pay_data_0) == 4), "reg_mon_pay_data_0 size is not 32-bit");
/*
 MON_ARM_0 
 b'Monitor ARM configuration'
*/
typedef struct reg_mon_arm_0 {
	union {
		struct {
			uint32_t sid : 8,
				mask : 8,
				sop : 1,
				sod : 15;
		};
		uint32_t _raw;
	};
} reg_mon_arm_0;
static_assert((sizeof(struct reg_mon_arm_0) == 4), "reg_mon_arm_0 size is not 32-bit");
/*
 MON_CONFIG_0 
 b'Monitor ARM configuration (Additional..)'
*/
typedef struct reg_mon_config_0 {
	union {
		struct {
			uint32_t long_sob : 1,
				_reserved4 : 3,
				cq_en : 1,
				_reserved8 : 3,
				lbw_en : 1,
				_reserved15 : 6,
				sm_data_config : 1,
				msb_sid : 4,
				auto_zero : 1,
				_reserved24 : 3,
				wr_num : 4,
				_reserved31 : 3,
				long_high_group : 1;
		};
		uint32_t _raw;
	};
} reg_mon_config_0;
static_assert((sizeof(struct reg_mon_config_0) == 4), "reg_mon_config_0 size is not 32-bit");
/*
 MON_STATUS_0 
 b'Monitor status'
*/
typedef struct reg_mon_status_0 {
	union {
		struct {
			uint32_t valid : 1,
				pending : 8,
				prot : 1,
				priv : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_mon_status_0;
static_assert((sizeof(struct reg_mon_status_0) == 4), "reg_mon_status_0 size is not 32-bit");
/*
 CQ_DIRECT 
 b'CQ Direct write'
*/
typedef struct reg_cq_direct {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cq_direct;
static_assert((sizeof(struct reg_cq_direct) == 4), "reg_cq_direct size is not 32-bit");
/*
 SM_SEC_0 
 b'SM security setting (Secured)'
*/
typedef struct reg_sm_sec_0 {
	union {
		struct {
			uint32_t sec_vec : 32;
		};
		uint32_t _raw;
	};
} reg_sm_sec_0;
static_assert((sizeof(struct reg_sm_sec_0) == 4), "reg_sm_sec_0 size is not 32-bit");
/*
 SM_PRIV_0 
 b'SM security setting (Privileged)'
*/
typedef struct reg_sm_priv_0 {
	union {
		struct {
			uint32_t priv : 32;
		};
		uint32_t _raw;
	};
} reg_sm_priv_0;
static_assert((sizeof(struct reg_sm_priv_0) == 4), "reg_sm_priv_0 size is not 32-bit");
/*
 SOB_OBJ_1 
 b'Sync object counters (1)'
*/
typedef struct reg_sob_obj_1 {
	union {
		struct {
			uint32_t val : 15,
				_reserved24 : 9,
				long_sob : 1,
				_reserved28 : 3,
				zero_sob_cnt : 1,
				_reserved30 : 1,
				trace_evict : 1,
				inc : 1;
		};
		uint32_t _raw;
	};
} reg_sob_obj_1;
static_assert((sizeof(struct reg_sob_obj_1) == 4), "reg_sob_obj_1 size is not 32-bit");
/*
 MON_PAY_ADDRL_1 
 b'Monitor Address (LSB)'
*/
typedef struct reg_mon_pay_addrl_1 {
	union {
		struct {
			uint32_t addrl : 32;
		};
		uint32_t _raw;
	};
} reg_mon_pay_addrl_1;
static_assert((sizeof(struct reg_mon_pay_addrl_1) == 4), "reg_mon_pay_addrl_1 size is not 32-bit");
/*
 MON_PAY_ADDRH_1 
 b'Monitor Address (MSB)'
*/
typedef struct reg_mon_pay_addrh_1 {
	union {
		struct {
			uint32_t addrh : 32;
		};
		uint32_t _raw;
	};
} reg_mon_pay_addrh_1;
static_assert((sizeof(struct reg_mon_pay_addrh_1) == 4), "reg_mon_pay_addrh_1 size is not 32-bit");
/*
 MON_PAY_DATA_1 
 b'Monitor Data'
*/
typedef struct reg_mon_pay_data_1 {
	union {
		struct {
			uint32_t data : 32;
		};
		uint32_t _raw;
	};
} reg_mon_pay_data_1;
static_assert((sizeof(struct reg_mon_pay_data_1) == 4), "reg_mon_pay_data_1 size is not 32-bit");
/*
 MON_ARM_1 
 b'Monitor ARM configuration'
*/
typedef struct reg_mon_arm_1 {
	union {
		struct {
			uint32_t sid : 8,
				mask : 8,
				sop : 1,
				sod : 15;
		};
		uint32_t _raw;
	};
} reg_mon_arm_1;
static_assert((sizeof(struct reg_mon_arm_1) == 4), "reg_mon_arm_1 size is not 32-bit");
/*
 MON_CONFIG_1 
 b'Monitor ARM configuration (Additional..)'
*/
typedef struct reg_mon_config_1 {
	union {
		struct {
			uint32_t long_sob : 1,
				_reserved4 : 3,
				cq_en : 1,
				_reserved8 : 3,
				lbw_en : 1,
				_reserved15 : 6,
				sm_data_config : 1,
				msb_sid : 4,
				auto_zero : 1,
				_reserved24 : 3,
				wr_num : 4,
				_reserved31 : 3,
				long_high_group : 1;
		};
		uint32_t _raw;
	};
} reg_mon_config_1;
static_assert((sizeof(struct reg_mon_config_1) == 4), "reg_mon_config_1 size is not 32-bit");
/*
 MON_STATUS_1 
 b'Monitor status'
*/
typedef struct reg_mon_status_1 {
	union {
		struct {
			uint32_t valid : 1,
				pending : 8,
				prot : 1,
				priv : 1,
				_reserved11 : 21;
		};
		uint32_t _raw;
	};
} reg_mon_status_1;
static_assert((sizeof(struct reg_mon_status_1) == 4), "reg_mon_status_1 size is not 32-bit");
/*
 SM_SEC_1 
 b'SM security setting (Secured)'
*/
typedef struct reg_sm_sec_1 {
	union {
		struct {
			uint32_t sec_vec : 32;
		};
		uint32_t _raw;
	};
} reg_sm_sec_1;
static_assert((sizeof(struct reg_sm_sec_1) == 4), "reg_sm_sec_1 size is not 32-bit");
/*
 SM_PRIV_1 
 b'SM security setting (Privileged)'
*/
typedef struct reg_sm_priv_1 {
	union {
		struct {
			uint32_t priv : 32;
		};
		uint32_t _raw;
	};
} reg_sm_priv_1;
static_assert((sizeof(struct reg_sm_priv_1) == 4), "reg_sm_priv_1 size is not 32-bit");

#ifdef __cplusplus
} /* sob_objs namespace */
#endif

/*
 SOB_OBJS block
*/

#ifdef __cplusplus

struct block_sob_objs {
	struct sob_objs::reg_sob_obj_0 sob_obj_0[8192];
	struct sob_objs::reg_mon_pay_addrl_0 mon_pay_addrl_0[1024];
	struct sob_objs::reg_mon_pay_addrh_0 mon_pay_addrh_0[1024];
	struct sob_objs::reg_mon_pay_data_0 mon_pay_data_0[1024];
	struct sob_objs::reg_mon_arm_0 mon_arm_0[1024];
	struct sob_objs::reg_mon_config_0 mon_config_0[1024];
	struct sob_objs::reg_mon_status_0 mon_status_0[1024];
	struct sob_objs::reg_cq_direct cq_direct[64];
	uint32_t _pad57600[960];
	struct sob_objs::reg_sm_sec_0 sm_sec_0[512];
	struct sob_objs::reg_sm_priv_0 sm_priv_0[512];
	struct sob_objs::reg_sob_obj_1 sob_obj_1[8192];
	struct sob_objs::reg_mon_pay_addrl_1 mon_pay_addrl_1[1024];
	struct sob_objs::reg_mon_pay_addrh_1 mon_pay_addrh_1[1024];
	struct sob_objs::reg_mon_pay_data_1 mon_pay_data_1[1024];
	struct sob_objs::reg_mon_arm_1 mon_arm_1[1024];
	struct sob_objs::reg_mon_config_1 mon_config_1[1024];
	struct sob_objs::reg_mon_status_1 mon_status_1[1024];
	uint32_t _pad122880[1024];
	struct sob_objs::reg_sm_sec_1 sm_sec_1[512];
	struct sob_objs::reg_sm_priv_1 sm_priv_1[512];
};
#else

typedef struct block_sob_objs {
	reg_sob_obj_0 sob_obj_0[8192];
	reg_mon_pay_addrl_0 mon_pay_addrl_0[1024];
	reg_mon_pay_addrh_0 mon_pay_addrh_0[1024];
	reg_mon_pay_data_0 mon_pay_data_0[1024];
	reg_mon_arm_0 mon_arm_0[1024];
	reg_mon_config_0 mon_config_0[1024];
	reg_mon_status_0 mon_status_0[1024];
	reg_cq_direct cq_direct[64];
	uint32_t _pad57600[960];
	reg_sm_sec_0 sm_sec_0[512];
	reg_sm_priv_0 sm_priv_0[512];
	reg_sob_obj_1 sob_obj_1[8192];
	reg_mon_pay_addrl_1 mon_pay_addrl_1[1024];
	reg_mon_pay_addrh_1 mon_pay_addrh_1[1024];
	reg_mon_pay_data_1 mon_pay_data_1[1024];
	reg_mon_arm_1 mon_arm_1[1024];
	reg_mon_config_1 mon_config_1[1024];
	reg_mon_status_1 mon_status_1[1024];
	uint32_t _pad122880[1024];
	reg_sm_sec_1 sm_sec_1[512];
	reg_sm_priv_1 sm_priv_1[512];
} block_sob_objs;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_sob_objs_defaults[] =
{
	// offset	// value
	{ 0xd000, 0x200               , 1024 }, // mon_status_0
	{ 0xf000, 0xffffffff          , 512 }, // sm_sec_0
	{ 0xf800, 0xffffffff          , 512 }, // sm_priv_0
	{ 0x1d000, 0x200               , 1024 }, // mon_status_1
	{ 0x1f000, 0xffffffff          , 512 }, // sm_sec_1
	{ 0x1f800, 0xffffffff          , 512 }, // sm_priv_1
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_SOB_OBJS_H_ */

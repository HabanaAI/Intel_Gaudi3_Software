/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_PQM_CH_A_H_
#define ASIC_REG_STRUCTS_GAUDI3_PQM_CH_A_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace pqm_ch_a {
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
 ADDR_HI 
 b'msb bits of the ptr addr'
*/
typedef struct reg_addr_hi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_addr_hi;
static_assert((sizeof(struct reg_addr_hi) == 4), "reg_addr_hi size is not 32-bit");
/*
 ADDR_LO 
 b'lsb bits of the ptr addr'
*/
typedef struct reg_addr_lo {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_addr_lo;
static_assert((sizeof(struct reg_addr_lo) == 4), "reg_addr_lo size is not 32-bit");
/*
 CTRL 
 b'cq pointer mode entry ctrl (ptr size, index)'
*/
typedef struct reg_ctrl {
	union {
		struct {
			uint32_t ptr_size : 16,
				index : 10,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
} reg_ctrl;
static_assert((sizeof(struct reg_ctrl) == 4), "reg_ctrl size is not 32-bit");
/*
 PI 
 b'wr pointer. SW managed.wraps at 2xPOW2(32)'
*/
typedef struct reg_pi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pi;
static_assert((sizeof(struct reg_pi) == 4), "reg_pi size is not 32-bit");
/*
 CI 
 b'rd pointer. HW managed.wraps at 2xPOW2(32)'
*/
typedef struct reg_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_ci;
static_assert((sizeof(struct reg_ci) == 4), "reg_ci size is not 32-bit");
/*
 CP_PRED 
 b'32 predicates'
*/
typedef struct reg_cp_pred {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cp_pred;
static_assert((sizeof(struct reg_cp_pred) == 4), "reg_cp_pred size is not 32-bit");
/*
 DESC_SUBMIT_FIFO_CFG 
 b'CQ Descriptor submission fifo configuration'
*/
typedef struct reg_desc_submit_fifo_cfg {
	union {
		struct {
			uint32_t mode : 2,
				write_mode : 1,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_desc_submit_fifo_cfg;
static_assert((sizeof(struct reg_desc_submit_fifo_cfg) == 4), "reg_desc_submit_fifo_cfg size is not 32-bit");
/*
 DESC_PTR_RLS_CFG 
 b'CQ descriptor pointer release configuration'
*/
typedef struct reg_desc_ptr_rls_cfg {
	union {
		struct {
			uint32_t msg_enable : 1,
				msg_type : 1,
				max_inflight : 4,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_desc_ptr_rls_cfg;
static_assert((sizeof(struct reg_desc_ptr_rls_cfg) == 4), "reg_desc_ptr_rls_cfg size is not 32-bit");
/*
 DESC_PTR_RLS_MSG_ADDR_L 
 b'descriptor pointer release message addr lbw'
*/
typedef struct reg_desc_ptr_rls_msg_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_desc_ptr_rls_msg_addr_l;
static_assert((sizeof(struct reg_desc_ptr_rls_msg_addr_l) == 4), "reg_desc_ptr_rls_msg_addr_l size is not 32-bit");
/*
 DESC_PTR_RLS_MSG_ADDR_H 
 b'descriptor pointer release message addr hbw'
*/
typedef struct reg_desc_ptr_rls_msg_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_desc_ptr_rls_msg_addr_h;
static_assert((sizeof(struct reg_desc_ptr_rls_msg_addr_h) == 4), "reg_desc_ptr_rls_msg_addr_h size is not 32-bit");
/*
 DESC_PTR_RLS_MSG_FREQ 
 b'Send message only after this number of pointer rls'
*/
typedef struct reg_desc_ptr_rls_msg_freq {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_desc_ptr_rls_msg_freq;
static_assert((sizeof(struct reg_desc_ptr_rls_msg_freq) == 4), "reg_desc_ptr_rls_msg_freq size is not 32-bit");
/*
 DESC_PTR_RLS_MSG_PAYLOAD 
 b'Data to send on ptr release message'
*/
typedef struct reg_desc_ptr_rls_msg_payload {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_desc_ptr_rls_msg_payload;
static_assert((sizeof(struct reg_desc_ptr_rls_msg_payload) == 4), "reg_desc_ptr_rls_msg_payload size is not 32-bit");
/*
 FENCE_INC 
 b'increment value for fence 0..3'
*/
typedef struct reg_fence_inc {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_fence_inc;
static_assert((sizeof(struct reg_fence_inc) == 4), "reg_fence_inc size is not 32-bit");
/*
 FENCE_CNT 
 b'current value for fence 0..3'
*/
typedef struct reg_fence_cnt {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_fence_cnt;
static_assert((sizeof(struct reg_fence_cnt) == 4), "reg_fence_cnt size is not 32-bit");
/*
 MSQ_BASE_H 
 b'MSB of the pointer to the base of submission queue'
*/
typedef struct reg_msq_base_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_msq_base_h;
static_assert((sizeof(struct reg_msq_base_h) == 4), "reg_msq_base_h size is not 32-bit");
/*
 MSQ_BASE_L 
 b'LSB of the pointer to the base of submission queue'
*/
typedef struct reg_msq_base_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_msq_base_l;
static_assert((sizeof(struct reg_msq_base_l) == 4), "reg_msq_base_l size is not 32-bit");
/*
 MSQ_SIZE 
 b'size (in bytes) of the submission queue'
*/
typedef struct reg_msq_size {
	union {
		struct {
			uint32_t val : 30,
				_reserved30 : 2;
		};
		uint32_t _raw;
	};
} reg_msq_size;
static_assert((sizeof(struct reg_msq_size) == 4), "reg_msq_size size is not 32-bit");
/*
 MSQ_PI 
 b'pointer to the current producer index'
*/
typedef struct reg_msq_pi {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_msq_pi;
static_assert((sizeof(struct reg_msq_pi) == 4), "reg_msq_pi size is not 32-bit");
/*
 MSQ_CI 
 b'pointer to the current consumer index'
*/
typedef struct reg_msq_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_msq_ci;
static_assert((sizeof(struct reg_msq_ci) == 4), "reg_msq_ci size is not 32-bit");
/*
 DESC_SUBMIT_PUSH_REG2 
 b'pushed to CTRL (size & Index)'
*/
typedef struct reg_desc_submit_push_reg2 {
	union {
		struct {
			uint32_t ptr_size : 16,
				index : 10,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
} reg_desc_submit_push_reg2;
static_assert((sizeof(struct reg_desc_submit_push_reg2) == 4), "reg_desc_submit_push_reg2 size is not 32-bit");
/*
 DESC_SUBMIT_PUSH_REG1 
 b'pushed to ADDR_HI'
*/
typedef struct reg_desc_submit_push_reg1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_desc_submit_push_reg1;
static_assert((sizeof(struct reg_desc_submit_push_reg1) == 4), "reg_desc_submit_push_reg1 size is not 32-bit");
/*
 DESC_SUBMIT_PUSH_REG0 
 b'pushed to ADDR_LO'
*/
typedef struct reg_desc_submit_push_reg0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_desc_submit_push_reg0;
static_assert((sizeof(struct reg_desc_submit_push_reg0) == 4), "reg_desc_submit_push_reg0 size is not 32-bit");
/*
 CQ_RECONFIG 
 b'stop cq to enable reconfig of pi/ci and descriptor'
*/
typedef struct reg_cq_reconfig {
	union {
		struct {
			uint32_t en : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cq_reconfig;
static_assert((sizeof(struct reg_cq_reconfig) == 4), "reg_cq_reconfig size is not 32-bit");

#ifdef __cplusplus
} /* pqm_ch_a namespace */
#endif

/*
 PQM_CH_A block
*/

#ifdef __cplusplus

struct block_pqm_ch_a {
	struct pqm_ch_a::reg_addr_hi addr_hi[32];
	struct pqm_ch_a::reg_addr_lo addr_lo[32];
	struct pqm_ch_a::reg_ctrl ctrl[32];
	struct pqm_ch_a::reg_pi pi;
	struct pqm_ch_a::reg_ci ci;
	uint32_t _pad392[10];
	struct pqm_ch_a::reg_cp_pred cp_pred;
	struct pqm_ch_a::reg_desc_submit_fifo_cfg desc_submit_fifo_cfg;
	struct pqm_ch_a::reg_desc_ptr_rls_cfg desc_ptr_rls_cfg;
	struct pqm_ch_a::reg_desc_ptr_rls_msg_addr_l desc_ptr_rls_msg_addr_l;
	struct pqm_ch_a::reg_desc_ptr_rls_msg_addr_h desc_ptr_rls_msg_addr_h;
	struct pqm_ch_a::reg_desc_ptr_rls_msg_freq desc_ptr_rls_msg_freq;
	struct pqm_ch_a::reg_desc_ptr_rls_msg_payload desc_ptr_rls_msg_payload;
	struct pqm_ch_a::reg_fence_inc fence_inc[4];
	struct pqm_ch_a::reg_fence_cnt fence_cnt[4];
	struct pqm_ch_a::reg_msq_base_h msq_base_h;
	struct pqm_ch_a::reg_msq_base_l msq_base_l;
	struct pqm_ch_a::reg_msq_size msq_size;
	struct pqm_ch_a::reg_msq_pi msq_pi;
	struct pqm_ch_a::reg_msq_ci msq_ci;
	struct pqm_ch_a::reg_desc_submit_push_reg2 desc_submit_push_reg2;
	struct pqm_ch_a::reg_desc_submit_push_reg1 desc_submit_push_reg1;
	struct pqm_ch_a::reg_desc_submit_push_reg0 desc_submit_push_reg0;
	struct pqm_ch_a::reg_cq_reconfig cq_reconfig;
};
#else

typedef struct block_pqm_ch_a {
	reg_addr_hi addr_hi[32];
	reg_addr_lo addr_lo[32];
	reg_ctrl ctrl[32];
	reg_pi pi;
	reg_ci ci;
	uint32_t _pad392[10];
	reg_cp_pred cp_pred;
	reg_desc_submit_fifo_cfg desc_submit_fifo_cfg;
	reg_desc_ptr_rls_cfg desc_ptr_rls_cfg;
	reg_desc_ptr_rls_msg_addr_l desc_ptr_rls_msg_addr_l;
	reg_desc_ptr_rls_msg_addr_h desc_ptr_rls_msg_addr_h;
	reg_desc_ptr_rls_msg_freq desc_ptr_rls_msg_freq;
	reg_desc_ptr_rls_msg_payload desc_ptr_rls_msg_payload;
	reg_fence_inc fence_inc[4];
	reg_fence_cnt fence_cnt[4];
	reg_msq_base_h msq_base_h;
	reg_msq_base_l msq_base_l;
	reg_msq_size msq_size;
	reg_msq_pi msq_pi;
	reg_msq_ci msq_ci;
	reg_desc_submit_push_reg2 desc_submit_push_reg2;
	reg_desc_submit_push_reg1 desc_submit_push_reg1;
	reg_desc_submit_push_reg0 desc_submit_push_reg0;
	reg_cq_reconfig cq_reconfig;
} block_pqm_ch_a;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_pqm_ch_a_defaults[] =
{
	// offset	// value
	{ 0x1b0 , 0x1                 , 1 }, // cp_pred
	{ 0x1b8 , 0x20                , 1 }, // desc_ptr_rls_cfg
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_PQM_CH_A_H_ */

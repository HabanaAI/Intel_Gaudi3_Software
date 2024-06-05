/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_SMT_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_SMT_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc_non_tensor_descriptor_smt {
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
 TID_BASE_DIM_0 
 b'read/write initial value of IRF0 dim 0'
*/
typedef struct reg_tid_base_dim_0 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_base_dim_0;
static_assert((sizeof(struct reg_tid_base_dim_0) == 4), "reg_tid_base_dim_0 size is not 32-bit");
/*
 TID_SIZE_DIM_0 
 b'read/write initial value of IRF1 dim 0'
*/
typedef struct reg_tid_size_dim_0 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_size_dim_0;
static_assert((sizeof(struct reg_tid_size_dim_0) == 4), "reg_tid_size_dim_0 size is not 32-bit");
/*
 TID_BASE_DIM_1 
 b'read/write initial value of IRF0 dim 1'
*/
typedef struct reg_tid_base_dim_1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_base_dim_1;
static_assert((sizeof(struct reg_tid_base_dim_1) == 4), "reg_tid_base_dim_1 size is not 32-bit");
/*
 TID_SIZE_DIM_1 
 b'read/write initial value of IRF1 dim 1'
*/
typedef struct reg_tid_size_dim_1 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_size_dim_1;
static_assert((sizeof(struct reg_tid_size_dim_1) == 4), "reg_tid_size_dim_1 size is not 32-bit");
/*
 TID_BASE_DIM_2 
 b'read/write initial value of IRF0 dim 2'
*/
typedef struct reg_tid_base_dim_2 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_base_dim_2;
static_assert((sizeof(struct reg_tid_base_dim_2) == 4), "reg_tid_base_dim_2 size is not 32-bit");
/*
 TID_SIZE_DIM_2 
 b'read/write initial value of IRF1 dim 2'
*/
typedef struct reg_tid_size_dim_2 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_size_dim_2;
static_assert((sizeof(struct reg_tid_size_dim_2) == 4), "reg_tid_size_dim_2 size is not 32-bit");
/*
 TID_BASE_DIM_3 
 b'read/write initial value of IRF0 dim 3'
*/
typedef struct reg_tid_base_dim_3 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_base_dim_3;
static_assert((sizeof(struct reg_tid_base_dim_3) == 4), "reg_tid_base_dim_3 size is not 32-bit");
/*
 TID_SIZE_DIM_3 
 b'read/write initial value of IRF1 dim 3'
*/
typedef struct reg_tid_size_dim_3 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_size_dim_3;
static_assert((sizeof(struct reg_tid_size_dim_3) == 4), "reg_tid_size_dim_3 size is not 32-bit");
/*
 TID_BASE_DIM_4 
 b'read/write initial value of IRF0 dim 4'
*/
typedef struct reg_tid_base_dim_4 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_base_dim_4;
static_assert((sizeof(struct reg_tid_base_dim_4) == 4), "reg_tid_base_dim_4 size is not 32-bit");
/*
 TID_SIZE_DIM_4 
 b'read/write initial value of IRF1 dim 4'
*/
typedef struct reg_tid_size_dim_4 {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_tid_size_dim_4;
static_assert((sizeof(struct reg_tid_size_dim_4) == 4), "reg_tid_size_dim_4 size is not 32-bit");
/*
 TID_BASE_SIZE_HIGH_DIM_0 
 b'read/write initial value of IRF01 dim 0 high bits'
*/
typedef struct reg_tid_base_size_high_dim_0 {
	union {
		struct {
			uint32_t base_high : 12,
				_reserved16 : 4,
				size_high : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_tid_base_size_high_dim_0;
static_assert((sizeof(struct reg_tid_base_size_high_dim_0) == 4), "reg_tid_base_size_high_dim_0 size is not 32-bit");
/*
 TID_BASE_SIZE_HIGH_DIM_1 
 b'read/write initial value of IRF01 dim 1 high bits'
*/
typedef struct reg_tid_base_size_high_dim_1 {
	union {
		struct {
			uint32_t base_high : 12,
				_reserved16 : 4,
				size_high : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_tid_base_size_high_dim_1;
static_assert((sizeof(struct reg_tid_base_size_high_dim_1) == 4), "reg_tid_base_size_high_dim_1 size is not 32-bit");
/*
 TID_BASE_SIZE_HIGH_DIM_2 
 b'read/write initial value of IRF01 dim 2 high bits'
*/
typedef struct reg_tid_base_size_high_dim_2 {
	union {
		struct {
			uint32_t base_high : 12,
				_reserved16 : 4,
				size_high : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_tid_base_size_high_dim_2;
static_assert((sizeof(struct reg_tid_base_size_high_dim_2) == 4), "reg_tid_base_size_high_dim_2 size is not 32-bit");
/*
 TID_BASE_SIZE_HIGH_DIM_3 
 b'read/write initial value of IRF01 dim 3 high bits'
*/
typedef struct reg_tid_base_size_high_dim_3 {
	union {
		struct {
			uint32_t base_high : 12,
				_reserved16 : 4,
				size_high : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_tid_base_size_high_dim_3;
static_assert((sizeof(struct reg_tid_base_size_high_dim_3) == 4), "reg_tid_base_size_high_dim_3 size is not 32-bit");
/*
 TID_BASE_SIZE_HIGH_DIM_4 
 b'read/write initial value of IRF01 dim 4 high bits'
*/
typedef struct reg_tid_base_size_high_dim_4 {
	union {
		struct {
			uint32_t base_high : 12,
				_reserved16 : 4,
				size_high : 12,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_tid_base_size_high_dim_4;
static_assert((sizeof(struct reg_tid_base_size_high_dim_4) == 4), "reg_tid_base_size_high_dim_4 size is not 32-bit");

#ifdef __cplusplus
} /* tpc_non_tensor_descriptor_smt namespace */
#endif

/*
 TPC_NON_TENSOR_DESCRIPTOR_SMT block
*/

#ifdef __cplusplus

struct block_tpc_non_tensor_descriptor_smt {
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_dim_0 tid_base_dim_0;
	struct tpc_non_tensor_descriptor_smt::reg_tid_size_dim_0 tid_size_dim_0;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_dim_1 tid_base_dim_1;
	struct tpc_non_tensor_descriptor_smt::reg_tid_size_dim_1 tid_size_dim_1;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_dim_2 tid_base_dim_2;
	struct tpc_non_tensor_descriptor_smt::reg_tid_size_dim_2 tid_size_dim_2;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_dim_3 tid_base_dim_3;
	struct tpc_non_tensor_descriptor_smt::reg_tid_size_dim_3 tid_size_dim_3;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_dim_4 tid_base_dim_4;
	struct tpc_non_tensor_descriptor_smt::reg_tid_size_dim_4 tid_size_dim_4;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_size_high_dim_0 tid_base_size_high_dim_0;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_size_high_dim_1 tid_base_size_high_dim_1;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_size_high_dim_2 tid_base_size_high_dim_2;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_size_high_dim_3 tid_base_size_high_dim_3;
	struct tpc_non_tensor_descriptor_smt::reg_tid_base_size_high_dim_4 tid_base_size_high_dim_4;
};
#else

typedef struct block_tpc_non_tensor_descriptor_smt {
	reg_tid_base_dim_0 tid_base_dim_0;
	reg_tid_size_dim_0 tid_size_dim_0;
	reg_tid_base_dim_1 tid_base_dim_1;
	reg_tid_size_dim_1 tid_size_dim_1;
	reg_tid_base_dim_2 tid_base_dim_2;
	reg_tid_size_dim_2 tid_size_dim_2;
	reg_tid_base_dim_3 tid_base_dim_3;
	reg_tid_size_dim_3 tid_size_dim_3;
	reg_tid_base_dim_4 tid_base_dim_4;
	reg_tid_size_dim_4 tid_size_dim_4;
	reg_tid_base_size_high_dim_0 tid_base_size_high_dim_0;
	reg_tid_base_size_high_dim_1 tid_base_size_high_dim_1;
	reg_tid_base_size_high_dim_2 tid_base_size_high_dim_2;
	reg_tid_base_size_high_dim_3 tid_base_size_high_dim_3;
	reg_tid_base_size_high_dim_4 tid_base_size_high_dim_4;
} block_tpc_non_tensor_descriptor_smt;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_NON_TENSOR_DESCRIPTOR_SMT_H_ */

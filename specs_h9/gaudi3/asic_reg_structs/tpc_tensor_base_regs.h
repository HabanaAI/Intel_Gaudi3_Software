/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_TENSOR_BASE_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_TENSOR_BASE_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc_tensor_base {
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
 BASE_ADDR_LOW 
 b'bits 31 to 0 of the base address'
*/
typedef struct reg_base_addr_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_base_addr_low;
static_assert((sizeof(struct reg_base_addr_low) == 4), "reg_base_addr_low size is not 32-bit");
/*
 BASE_ADDR_HIGH 
 b'bits 63 to 32 of the base address'
*/
typedef struct reg_base_addr_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_base_addr_high;
static_assert((sizeof(struct reg_base_addr_high) == 4), "reg_base_addr_high size is not 32-bit");
/*
 PREF_STRIDE 
 b'prefetcher strides'
*/
typedef struct reg_pref_stride {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_pref_stride;
static_assert((sizeof(struct reg_pref_stride) == 4), "reg_pref_stride size is not 32-bit");
/*
 L2_WINDOW_SIZE 
 b'dcache l2 prefetcher window size'
*/
typedef struct reg_l2_window_size {
	union {
		struct {
			uint32_t val : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_l2_window_size;
static_assert((sizeof(struct reg_l2_window_size) == 4), "reg_l2_window_size size is not 32-bit");

#ifdef __cplusplus
} /* tpc_tensor_base namespace */
#endif

/*
 TPC_TENSOR_BASE block
*/

#ifdef __cplusplus

struct block_tpc_tensor_base {
	struct tpc_tensor_base::reg_base_addr_low base_addr_low;
	struct tpc_tensor_base::reg_base_addr_high base_addr_high;
	struct tpc_tensor_base::reg_pref_stride pref_stride;
	struct tpc_tensor_base::reg_l2_window_size l2_window_size;
};
#else

typedef struct block_tpc_tensor_base {
	reg_base_addr_low base_addr_low;
	reg_base_addr_high base_addr_high;
	reg_pref_stride pref_stride;
	reg_l2_window_size l2_window_size;
} block_tpc_tensor_base;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_TENSOR_BASE_H_ */

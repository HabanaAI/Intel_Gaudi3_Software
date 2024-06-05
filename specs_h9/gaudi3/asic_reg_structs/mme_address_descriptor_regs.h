/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_MME_ADDRESS_DESCRIPTOR_H_
#define ASIC_REG_STRUCTS_GAUDI3_MME_ADDRESS_DESCRIPTOR_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace mme_address_descriptor {
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
 COUT1_LO 
 b'COUT COLOR1 TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_cout1_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout1_lo;
static_assert((sizeof(struct reg_cout1_lo) == 4), "reg_cout1_lo size is not 32-bit");
/*
 COUT1_HI 
 b'COUT COLOR1 TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_cout1_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout1_hi;
static_assert((sizeof(struct reg_cout1_hi) == 4), "reg_cout1_hi size is not 32-bit");
/*
 COUT0_LO 
 b'COUT COLOR0 TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_cout0_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout0_lo;
static_assert((sizeof(struct reg_cout0_lo) == 4), "reg_cout0_lo size is not 32-bit");
/*
 COUT0_HI 
 b'COUT COLOR0 TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_cout0_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout0_hi;
static_assert((sizeof(struct reg_cout0_hi) == 4), "reg_cout0_hi size is not 32-bit");
/*
 A_LO 
 b'OPERAND A TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_a_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_a_lo;
static_assert((sizeof(struct reg_a_lo) == 4), "reg_a_lo size is not 32-bit");
/*
 A_HI 
 b'OPERAND A TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_a_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_a_hi;
static_assert((sizeof(struct reg_a_hi) == 4), "reg_a_hi size is not 32-bit");
/*
 B_LO 
 b'OPERAND B TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_b_lo {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_b_lo;
static_assert((sizeof(struct reg_b_lo) == 4), "reg_b_lo size is not 32-bit");
/*
 B_HI 
 b'OPERAND B TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_b_hi {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_b_hi;
static_assert((sizeof(struct reg_b_hi) == 4), "reg_b_hi size is not 32-bit");

#ifdef __cplusplus
} /* mme_address_descriptor namespace */
#endif

/*
 MME_ADDRESS_DESCRIPTOR block
*/

#ifdef __cplusplus

struct block_mme_address_descriptor {
	struct mme_address_descriptor::reg_cout1_lo cout1_lo;
	struct mme_address_descriptor::reg_cout1_hi cout1_hi;
	struct mme_address_descriptor::reg_cout0_lo cout0_lo;
	struct mme_address_descriptor::reg_cout0_hi cout0_hi;
	struct mme_address_descriptor::reg_a_lo a_lo;
	struct mme_address_descriptor::reg_a_hi a_hi;
	struct mme_address_descriptor::reg_b_lo b_lo;
	struct mme_address_descriptor::reg_b_hi b_hi;
};
#else

typedef struct block_mme_address_descriptor {
	reg_cout1_lo cout1_lo;
	reg_cout1_hi cout1_hi;
	reg_cout0_lo cout0_lo;
	reg_cout0_hi cout0_hi;
	reg_a_lo a_lo;
	reg_a_hi a_hi;
	reg_b_lo b_lo;
	reg_b_hi b_hi;
} block_mme_address_descriptor;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_MME_ADDRESS_DESCRIPTOR_H_ */

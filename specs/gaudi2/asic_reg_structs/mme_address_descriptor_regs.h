/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI2_MME_ADDRESS_DESCRIPTOR_H_
#define ASIC_REG_STRUCTS_GAUDI2_MME_ADDRESS_DESCRIPTOR_H_

#include <stdint.h>
#include "gaudi2_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi2 {
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
 COUT1_LOW 
 b'COUT COLOR1 TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_cout1_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout1_low;
static_assert((sizeof(struct reg_cout1_low) == 4), "reg_cout1_low size is not 32-bit");
/*
 COUT1_HIGH 
 b'COUT COLOR1 TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_cout1_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout1_high;
static_assert((sizeof(struct reg_cout1_high) == 4), "reg_cout1_high size is not 32-bit");
/*
 COUT0_LOW 
 b'COUT COLOR0 TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_cout0_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout0_low;
static_assert((sizeof(struct reg_cout0_low) == 4), "reg_cout0_low size is not 32-bit");
/*
 COUT0_HIGH 
 b'COUT COLOR0 TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_cout0_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_cout0_high;
static_assert((sizeof(struct reg_cout0_high) == 4), "reg_cout0_high size is not 32-bit");
/*
 A_LOW 
 b'OPERAND A TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_a_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_a_low;
static_assert((sizeof(struct reg_a_low) == 4), "reg_a_low size is not 32-bit");
/*
 A_HIGH 
 b'OPERAND A TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_a_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_a_high;
static_assert((sizeof(struct reg_a_high) == 4), "reg_a_high size is not 32-bit");
/*
 B_LOW 
 b'OPERAND B TENSOR BASE ADDRESS LOW'
*/
typedef struct reg_b_low {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_b_low;
static_assert((sizeof(struct reg_b_low) == 4), "reg_b_low size is not 32-bit");
/*
 B_HIGH 
 b'OPERAND B TENSOR BASE ADDRESS HIGH'
*/
typedef struct reg_b_high {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_b_high;
static_assert((sizeof(struct reg_b_high) == 4), "reg_b_high size is not 32-bit");

#ifdef __cplusplus
} /* mme_address_descriptor namespace */
#endif

/*
 MME_ADDRESS_DESCRIPTOR block
*/

#ifdef __cplusplus

struct block_mme_address_descriptor {
	struct mme_address_descriptor::reg_cout1_low cout1_low;
	struct mme_address_descriptor::reg_cout1_high cout1_high;
	struct mme_address_descriptor::reg_cout0_low cout0_low;
	struct mme_address_descriptor::reg_cout0_high cout0_high;
	struct mme_address_descriptor::reg_a_low a_low;
	struct mme_address_descriptor::reg_a_high a_high;
	struct mme_address_descriptor::reg_b_low b_low;
	struct mme_address_descriptor::reg_b_high b_high;
};
#else

typedef struct block_mme_address_descriptor {
	reg_cout1_low cout1_low;
	reg_cout1_high cout1_high;
	reg_cout0_low cout0_low;
	reg_cout0_high cout0_high;
	reg_a_low a_low;
	reg_a_high a_high;
	reg_b_low b_low;
	reg_b_high b_high;
} block_mme_address_descriptor;
#endif


#ifdef __cplusplus
} /* gaudi2 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI2_MME_ADDRESS_DESCRIPTOR_H_ */

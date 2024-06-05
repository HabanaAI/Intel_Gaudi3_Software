/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_MME_AGU_CORE_H_
#define ASIC_REG_STRUCTS_GAUDI3_MME_AGU_CORE_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace mme_agu_core {
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
 ROI_BASE_OFFSET 
 b'AGU ROI BASE OFFSET'
*/
typedef struct reg_roi_base_offset {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
} reg_roi_base_offset;
static_assert((sizeof(struct reg_roi_base_offset) == 4), "reg_roi_base_offset size is not 32-bit");

#ifdef __cplusplus
} /* mme_agu_core namespace */
#endif

/*
 MME_AGU_CORE block
*/

#ifdef __cplusplus

struct block_mme_agu_core {
	struct mme_agu_core::reg_roi_base_offset roi_base_offset[5];
};
#else

typedef struct block_mme_agu_core {
	reg_roi_base_offset roi_base_offset[5];
} block_mme_agu_core;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_MME_AGU_CORE_H_ */

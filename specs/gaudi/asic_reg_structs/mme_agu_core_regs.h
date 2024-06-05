/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_MME_AGU_CORE_H_
#define ASIC_REG_STRUCTS_MME_AGU_CORE_H_

#include <stdint.h>

#pragma pack(push, 1)

namespace mme_agu_core {
/*
 ROI_BASE_OFFSET 
*/
struct reg_roi_base_offset {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_roi_base_offset) == 4), "reg_roi_base_offset size is not 32-bit");
/*
 START_OFFSET 
*/
struct reg_start_offset {
	union {
		struct {
			uint32_t v : 32;
		};
		uint32_t _raw;
	};
};
static_assert((sizeof(struct reg_start_offset) == 4), "reg_start_offset size is not 32-bit");
} /* mme_agu_core namespace */

/*
 MME_AGU_CORE block
*/
struct block_mme_agu_core {
	struct mme_agu_core::reg_roi_base_offset roi_base_offset[5];
	struct mme_agu_core::reg_start_offset start_offset[4];
};

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_MME_AGU_CORE_H_ */

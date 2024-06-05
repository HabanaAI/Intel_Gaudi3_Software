/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_AXUSER_LBW_H_
#define ASIC_REG_STRUCTS_GAUDI3_AXUSER_LBW_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace axuser_lbw {
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
 LB_LOCK_IND 
 b'TPC Lock indication'
*/
typedef struct reg_lb_lock_ind {
	union {
		struct {
			uint32_t rd : 1,
				_reserved4 : 3,
				wr : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lb_lock_ind;
static_assert((sizeof(struct reg_lb_lock_ind) == 4), "reg_lb_lock_ind size is not 32-bit");
/*
 LB_RESERVED 
 b'Reserved'
*/
typedef struct reg_lb_reserved {
	union {
		struct {
			uint32_t rd_3_1 : 3,
				_reserved4 : 1,
				wr_3_1 : 3,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_lb_reserved;
static_assert((sizeof(struct reg_lb_reserved) == 4), "reg_lb_reserved size is not 32-bit");
/*
 LB_LOCK_IND_OVRD 
 b'Override enable for TPC Lock indication'
*/
typedef struct reg_lb_lock_ind_ovrd {
	union {
		struct {
			uint32_t rd : 1,
				_reserved4 : 3,
				wr : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lb_lock_ind_ovrd;
static_assert((sizeof(struct reg_lb_lock_ind_ovrd) == 4), "reg_lb_lock_ind_ovrd size is not 32-bit");
/*
 LB_RESERVED_OVRD 
 b'Override enable for Reserved'
*/
typedef struct reg_lb_reserved_ovrd {
	union {
		struct {
			uint32_t rd_3_1 : 3,
				_reserved4 : 1,
				wr_3_1 : 3,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_lb_reserved_ovrd;
static_assert((sizeof(struct reg_lb_reserved_ovrd) == 4), "reg_lb_reserved_ovrd size is not 32-bit");

#ifdef __cplusplus
} /* axuser_lbw namespace */
#endif

/*
 AXUSER_LBW block
*/

#ifdef __cplusplus

struct block_axuser_lbw {
	struct axuser_lbw::reg_lb_lock_ind lb_lock_ind;
	struct axuser_lbw::reg_lb_reserved lb_reserved;
	uint32_t _pad8[2];
	struct axuser_lbw::reg_lb_lock_ind_ovrd lb_lock_ind_ovrd;
	struct axuser_lbw::reg_lb_reserved_ovrd lb_reserved_ovrd;
};
#else

typedef struct block_axuser_lbw {
	reg_lb_lock_ind lb_lock_ind;
	reg_lb_reserved lb_reserved;
	uint32_t _pad8[2];
	reg_lb_lock_ind_ovrd lb_lock_ind_ovrd;
	reg_lb_reserved_ovrd lb_reserved_ovrd;
} block_axuser_lbw;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_AXUSER_LBW_H_ */

/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_AXUSER_HBW_H_
#define ASIC_REG_STRUCTS_GAUDI3_AXUSER_HBW_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace axuser_hbw {
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
 HB_MMU_BYPASS 
 b'MMU bypass'
*/
typedef struct reg_hb_mmu_bypass {
	union {
		struct {
			uint32_t wr : 1,
				_reserved4 : 3,
				rd : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_hb_mmu_bypass;
static_assert((sizeof(struct reg_hb_mmu_bypass) == 4), "reg_hb_mmu_bypass size is not 32-bit");
/*
 HB_ASID 
 b'HBW ASID'
*/
typedef struct reg_hb_asid {
	union {
		struct {
			uint32_t wr : 10,
				_reserved16 : 6,
				rd : 10,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
} reg_hb_asid;
static_assert((sizeof(struct reg_hb_asid) == 4), "reg_hb_asid size is not 32-bit");
/*
 HB_QOS 
 b'QOS'
*/
typedef struct reg_hb_qos {
	union {
		struct {
			uint32_t wr : 4,
				rd : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_hb_qos;
static_assert((sizeof(struct reg_hb_qos) == 4), "reg_hb_qos size is not 32-bit");
/*
 HB_MCID 
 b'MCID'
*/
typedef struct reg_hb_mcid {
	union {
		struct {
			uint32_t wr : 16,
				rd : 16;
		};
		uint32_t _raw;
	};
} reg_hb_mcid;
static_assert((sizeof(struct reg_hb_mcid) == 4), "reg_hb_mcid size is not 32-bit");
/*
 HB_CLASS_TYPE 
 b'CLASS'
*/
typedef struct reg_hb_class_type {
	union {
		struct {
			uint32_t wr : 2,
				_reserved4 : 2,
				rd : 2,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_hb_class_type;
static_assert((sizeof(struct reg_hb_class_type) == 4), "reg_hb_class_type size is not 32-bit");
/*
 HB_REDUCTION 
 b'write Reduction operators'
*/
typedef struct reg_hb_reduction {
	union {
		struct {
			uint32_t ind_wr : 1,
				_reserved4 : 3,
				operation_wr : 3,
				_reserved8 : 1,
				round_mode_wr : 2,
				_reserved12 : 2,
				data_type_wr : 4,
				clip_wr : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_hb_reduction;
static_assert((sizeof(struct reg_hb_reduction) == 4), "reg_hb_reduction size is not 32-bit");
/*
 HB_ATOMIC 
 b'Read atomic fetch'
*/
typedef struct reg_hb_atomic {
	union {
		struct {
			uint32_t fetch_add_rd : 1,
				_reserved4 : 3,
				fetch_clr_rd : 1,
				_reserved8 : 3,
				add_val_rd : 8,
				add_mask_rd : 6,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_hb_atomic;
static_assert((sizeof(struct reg_hb_atomic) == 4), "reg_hb_atomic size is not 32-bit");
/*
 HB_DOWN_CONV 
 b'down convert'
*/
typedef struct reg_hb_down_conv {
	union {
		struct {
			uint32_t rd : 2,
				_reserved4 : 2,
				round_rd : 3,
				_reserved8 : 1,
				clip_rd : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_hb_down_conv;
static_assert((sizeof(struct reg_hb_down_conv) == 4), "reg_hb_down_conv size is not 32-bit");
/*
 HB_MMU_PF_EN 
 b'Prefetch to Cache'
*/
typedef struct reg_hb_mmu_pf_en {
	union {
		struct {
			uint32_t rd : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_hb_mmu_pf_en;
static_assert((sizeof(struct reg_hb_mmu_pf_en) == 4), "reg_hb_mmu_pf_en size is not 32-bit");
/*
 HB_STRONG_ORDER 
 b'STRONG_ORDER'
*/
typedef struct reg_hb_strong_order {
	union {
		struct {
			uint32_t wr : 1,
				_reserved4 : 3,
				rd : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_hb_strong_order;
static_assert((sizeof(struct reg_hb_strong_order) == 4), "reg_hb_strong_order size is not 32-bit");
/*
 HB_RESERVED 
 b'For reserved fields of user bus'
*/
typedef struct reg_hb_reserved {
	union {
		struct {
			uint32_t rd_58 : 1,
				_reserved4 : 3,
				wr_56_45 : 12,
				wr_58 : 1,
				_reserved20 : 3,
				rd_1 : 1,
				_reserved24 : 3,
				wr_1 : 1,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_hb_reserved;
static_assert((sizeof(struct reg_hb_reserved) == 4), "reg_hb_reserved size is not 32-bit");
/*
 HB_MMU_BYPASS_OVRD 
 b'Override enable for MMU bypass'
*/
typedef struct reg_hb_mmu_bypass_ovrd {
	union {
		struct {
			uint32_t wr : 1,
				_reserved4 : 3,
				rd : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_hb_mmu_bypass_ovrd;
static_assert((sizeof(struct reg_hb_mmu_bypass_ovrd) == 4), "reg_hb_mmu_bypass_ovrd size is not 32-bit");
/*
 HB_ASID_OVRD 
 b'Override enable for HBW ASID'
*/
typedef struct reg_hb_asid_ovrd {
	union {
		struct {
			uint32_t wr : 10,
				_reserved16 : 6,
				rd : 10,
				_reserved26 : 6;
		};
		uint32_t _raw;
	};
} reg_hb_asid_ovrd;
static_assert((sizeof(struct reg_hb_asid_ovrd) == 4), "reg_hb_asid_ovrd size is not 32-bit");
/*
 HB_QOS_OVRD 
 b'Override enable for QOS'
*/
typedef struct reg_hb_qos_ovrd {
	union {
		struct {
			uint32_t wr : 4,
				rd : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_hb_qos_ovrd;
static_assert((sizeof(struct reg_hb_qos_ovrd) == 4), "reg_hb_qos_ovrd size is not 32-bit");
/*
 HB_MCID_OVRD 
 b'Override enable for MCID'
*/
typedef struct reg_hb_mcid_ovrd {
	union {
		struct {
			uint32_t wr : 16,
				rd : 16;
		};
		uint32_t _raw;
	};
} reg_hb_mcid_ovrd;
static_assert((sizeof(struct reg_hb_mcid_ovrd) == 4), "reg_hb_mcid_ovrd size is not 32-bit");
/*
 HB_CLASS_TYPE_OVRD 
 b'Override enable for CLASS'
*/
typedef struct reg_hb_class_type_ovrd {
	union {
		struct {
			uint32_t wr : 2,
				_reserved4 : 2,
				rd : 2,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_hb_class_type_ovrd;
static_assert((sizeof(struct reg_hb_class_type_ovrd) == 4), "reg_hb_class_type_ovrd size is not 32-bit");
/*
 HB_REDUCTION_OVRD 
 b'Override enable for write Reduction operators'
*/
typedef struct reg_hb_reduction_ovrd {
	union {
		struct {
			uint32_t ind_wr : 1,
				_reserved4 : 3,
				operation_wr : 3,
				_reserved8 : 1,
				round_mode_wr : 2,
				_reserved12 : 2,
				data_type_wr : 4,
				clip_wr : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_hb_reduction_ovrd;
static_assert((sizeof(struct reg_hb_reduction_ovrd) == 4), "reg_hb_reduction_ovrd size is not 32-bit");
/*
 HB_ATOMIC_OVRD 
 b'Override enable for Read atomic fetch'
*/
typedef struct reg_hb_atomic_ovrd {
	union {
		struct {
			uint32_t fetch_add_rd : 1,
				_reserved4 : 3,
				fetch_clr_rd : 1,
				_reserved8 : 3,
				add_val_rd : 8,
				add_mask_rd : 6,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_hb_atomic_ovrd;
static_assert((sizeof(struct reg_hb_atomic_ovrd) == 4), "reg_hb_atomic_ovrd size is not 32-bit");
/*
 HB_DOWN_CONV_OVRD 
 b'Override enable for down convert'
*/
typedef struct reg_hb_down_conv_ovrd {
	union {
		struct {
			uint32_t rd : 2,
				_reserved4 : 2,
				round_rd : 3,
				_reserved8 : 1,
				clip_rd : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_hb_down_conv_ovrd;
static_assert((sizeof(struct reg_hb_down_conv_ovrd) == 4), "reg_hb_down_conv_ovrd size is not 32-bit");
/*
 HB_MMU_PF_EN_OVRD 
 b'Override enable for Prefetch to Cache'
*/
typedef struct reg_hb_mmu_pf_en_ovrd {
	union {
		struct {
			uint32_t rd : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_hb_mmu_pf_en_ovrd;
static_assert((sizeof(struct reg_hb_mmu_pf_en_ovrd) == 4), "reg_hb_mmu_pf_en_ovrd size is not 32-bit");
/*
 HB_STRONG_ORDER_OVRD 
 b'Override enable for STRONG_ORDER'
*/
typedef struct reg_hb_strong_order_ovrd {
	union {
		struct {
			uint32_t wr : 1,
				_reserved4 : 3,
				rd : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_hb_strong_order_ovrd;
static_assert((sizeof(struct reg_hb_strong_order_ovrd) == 4), "reg_hb_strong_order_ovrd size is not 32-bit");
/*
 HB_RESERVED_OVRD 
 b'Override enable for reserved fields of user bus'
*/
typedef struct reg_hb_reserved_ovrd {
	union {
		struct {
			uint32_t rd_58 : 1,
				_reserved4 : 3,
				wr_56_45 : 12,
				wr_58 : 1,
				_reserved20 : 3,
				rd_1 : 1,
				_reserved24 : 3,
				wr_1 : 1,
				_reserved25 : 7;
		};
		uint32_t _raw;
	};
} reg_hb_reserved_ovrd;
static_assert((sizeof(struct reg_hb_reserved_ovrd) == 4), "reg_hb_reserved_ovrd size is not 32-bit");

#ifdef __cplusplus
} /* axuser_hbw namespace */
#endif

/*
 AXUSER_HBW block
*/

#ifdef __cplusplus

struct block_axuser_hbw {
	struct axuser_hbw::reg_hb_mmu_bypass hb_mmu_bypass;
	struct axuser_hbw::reg_hb_asid hb_asid;
	struct axuser_hbw::reg_hb_qos hb_qos;
	struct axuser_hbw::reg_hb_mcid hb_mcid;
	struct axuser_hbw::reg_hb_class_type hb_class_type;
	struct axuser_hbw::reg_hb_reduction hb_reduction;
	struct axuser_hbw::reg_hb_atomic hb_atomic;
	struct axuser_hbw::reg_hb_down_conv hb_down_conv;
	struct axuser_hbw::reg_hb_mmu_pf_en hb_mmu_pf_en;
	struct axuser_hbw::reg_hb_strong_order hb_strong_order;
	struct axuser_hbw::reg_hb_reserved hb_reserved;
	uint32_t _pad44[1];
	struct axuser_hbw::reg_hb_mmu_bypass_ovrd hb_mmu_bypass_ovrd;
	struct axuser_hbw::reg_hb_asid_ovrd hb_asid_ovrd;
	struct axuser_hbw::reg_hb_qos_ovrd hb_qos_ovrd;
	struct axuser_hbw::reg_hb_mcid_ovrd hb_mcid_ovrd;
	struct axuser_hbw::reg_hb_class_type_ovrd hb_class_type_ovrd;
	struct axuser_hbw::reg_hb_reduction_ovrd hb_reduction_ovrd;
	struct axuser_hbw::reg_hb_atomic_ovrd hb_atomic_ovrd;
	struct axuser_hbw::reg_hb_down_conv_ovrd hb_down_conv_ovrd;
	struct axuser_hbw::reg_hb_mmu_pf_en_ovrd hb_mmu_pf_en_ovrd;
	struct axuser_hbw::reg_hb_strong_order_ovrd hb_strong_order_ovrd;
	struct axuser_hbw::reg_hb_reserved_ovrd hb_reserved_ovrd;
};
#else

typedef struct block_axuser_hbw {
	reg_hb_mmu_bypass hb_mmu_bypass;
	reg_hb_asid hb_asid;
	reg_hb_qos hb_qos;
	reg_hb_mcid hb_mcid;
	reg_hb_class_type hb_class_type;
	reg_hb_reduction hb_reduction;
	reg_hb_atomic hb_atomic;
	reg_hb_down_conv hb_down_conv;
	reg_hb_mmu_pf_en hb_mmu_pf_en;
	reg_hb_strong_order hb_strong_order;
	reg_hb_reserved hb_reserved;
	uint32_t _pad44[1];
	reg_hb_mmu_bypass_ovrd hb_mmu_bypass_ovrd;
	reg_hb_asid_ovrd hb_asid_ovrd;
	reg_hb_qos_ovrd hb_qos_ovrd;
	reg_hb_mcid_ovrd hb_mcid_ovrd;
	reg_hb_class_type_ovrd hb_class_type_ovrd;
	reg_hb_reduction_ovrd hb_reduction_ovrd;
	reg_hb_atomic_ovrd hb_atomic_ovrd;
	reg_hb_down_conv_ovrd hb_down_conv_ovrd;
	reg_hb_mmu_pf_en_ovrd hb_mmu_pf_en_ovrd;
	reg_hb_strong_order_ovrd hb_strong_order_ovrd;
	reg_hb_reserved_ovrd hb_reserved_ovrd;
} block_axuser_hbw;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_AXUSER_HBW_H_ */

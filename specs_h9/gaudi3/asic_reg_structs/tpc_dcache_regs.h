/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_TPC_DCACHE_H_
#define ASIC_REG_STRUCTS_GAUDI3_TPC_DCACHE_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace tpc_dcache {
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
 DCACHE_CFG 
 b'Dcache Config'
*/
typedef struct reg_dcache_cfg {
	union {
		struct {
			uint32_t g_pref_dis : 1,
				g_pref_vld_clr : 1,
				halt_flush : 1,
				dealign_dis : 1,
				l2_hw_window_pref_en : 1,
				plru_methodology_sel : 1,
				l1_hw_window_pref_en : 1,
				rst_pref_when_limit_to_ze : 1,
				halt_hw_prefetch_r : 1,
				halt_l2_clean : 1,
				flush_l2_clean : 1,
				inv_l2_clean : 1,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_dcache_cfg;
static_assert((sizeof(struct reg_dcache_cfg) == 4), "reg_dcache_cfg size is not 32-bit");
/*
 TPC_DCACHE_L0CD 
 b'L0 DCACHE Cache Disable'
*/
typedef struct reg_tpc_dcache_l0cd {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_tpc_dcache_l0cd;
static_assert((sizeof(struct reg_tpc_dcache_l0cd) == 4), "reg_tpc_dcache_l0cd size is not 32-bit");

#ifdef __cplusplus
} /* tpc_dcache namespace */
#endif

/*
 TPC_DCACHE block
*/

#ifdef __cplusplus

struct block_tpc_dcache {
	struct tpc_dcache::reg_dcache_cfg dcache_cfg;
	struct tpc_dcache::reg_tpc_dcache_l0cd tpc_dcache_l0cd;
};
#else

typedef struct block_tpc_dcache {
	reg_dcache_cfg dcache_cfg;
	reg_tpc_dcache_l0cd tpc_dcache_l0cd;
} block_tpc_dcache;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_tpc_dcache_defaults[] =
{
	// offset	// value
	{ 0x0   , 0xf54               , 1 }, // dcache_cfg
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_TPC_DCACHE_H_ */

/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_CACHE_MAINTENANCE_H_
#define ASIC_REG_STRUCTS_GAUDI3_CACHE_MAINTENANCE_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace cache_maintenance {
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
 STATUS 
 b'Status based on activity counters'
*/
typedef struct reg_status {
	union {
		struct {
			uint32_t num_ongoing : 8,
				num_ongoing_priv : 8,
				num_dropped : 16;
		};
		uint32_t _raw;
	};
} reg_status;
static_assert((sizeof(struct reg_status) == 4), "reg_status size is not 32-bit");
/*
 ATTR_MCID 
 b'set LTA class and/or clear m-bit for for same MCID'
*/
typedef struct reg_attr_mcid {
	union {
		struct {
			uint32_t vc : 1,
				_reserved4 : 3,
				vm : 1,
				_reserved8 : 3,
				cls : 2,
				_reserved16 : 6,
				mcid : 16;
		};
		uint32_t _raw;
	};
} reg_attr_mcid;
static_assert((sizeof(struct reg_attr_mcid) == 4), "reg_attr_mcid size is not 32-bit");
/*
 INV_ENTRY 
 b'invalidate specific way'
*/
typedef struct reg_inv_entry {
	union {
		struct {
			uint32_t bank : 2,
				_reserved4 : 2,
				set : 7,
				_reserved12 : 1,
				way : 5,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_inv_entry;
static_assert((sizeof(struct reg_inv_entry) == 4), "reg_inv_entry size is not 32-bit");
/*
 FLUSH_ENTRY 
 b'flush specific way (inc. evict in case modified)'
*/
typedef struct reg_flush_entry {
	union {
		struct {
			uint32_t bank : 2,
				_reserved4 : 2,
				set : 7,
				_reserved12 : 1,
				way : 5,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_flush_entry;
static_assert((sizeof(struct reg_flush_entry) == 4), "reg_flush_entry size is not 32-bit");
/*
 INV_ALL 
 b'invalidate the entire cache'
*/
typedef struct reg_inv_all {
	union {
		struct {
			uint32_t value_dont_care : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_inv_all;
static_assert((sizeof(struct reg_inv_all) == 4), "reg_inv_all size is not 32-bit");
/*
 FLUSH_ALL 
 b'flush the entire cache (inc. evict if modified)'
*/
typedef struct reg_flush_all {
	union {
		struct {
			uint32_t value_dont_care : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_flush_all;
static_assert((sizeof(struct reg_flush_all) == 4), "reg_flush_all size is not 32-bit");
/*
 READ_ENTRY 
 b'read request for spefic way in specif structure'
*/
typedef struct reg_read_entry {
	union {
		struct {
			uint32_t bank : 2,
				_reserved4 : 2,
				set : 7,
				_reserved12 : 1,
				way : 5,
				_reserved20 : 3,
				structure : 3,
				_reserved23 : 9;
		};
		uint32_t _raw;
	};
} reg_read_entry;
static_assert((sizeof(struct reg_read_entry) == 4), "reg_read_entry size is not 32-bit");
/*
 INFO_READY 
 b's/w should read INFO* registers when 1'
*/
typedef struct reg_info_ready {
	union {
		struct {
			uint32_t value : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_info_ready;
static_assert((sizeof(struct reg_info_ready) == 4), "reg_info_ready size is not 32-bit");
/*
 INFO0 
 b'LSB infromation captured upon READ_ENTRY'
*/
typedef struct reg_info0 {
	union {
		struct {
			uint32_t value : 32;
		};
		uint32_t _raw;
	};
} reg_info0;
static_assert((sizeof(struct reg_info0) == 4), "reg_info0 size is not 32-bit");
/*
 INFO1 
 b'MSB infromation captured upon READ_ENTRY'
*/
typedef struct reg_info1 {
	union {
		struct {
			uint32_t value : 32;
		};
		uint32_t _raw;
	};
} reg_info1;
static_assert((sizeof(struct reg_info1) == 4), "reg_info1 size is not 32-bit");

#ifdef __cplusplus
} /* cache_maintenance namespace */
#endif

/*
 CACHE_MAINTENANCE block
*/

#ifdef __cplusplus

struct block_cache_maintenance {
	struct cache_maintenance::reg_status status;
	struct cache_maintenance::reg_attr_mcid attr_mcid;
	struct cache_maintenance::reg_inv_entry inv_entry;
	struct cache_maintenance::reg_flush_entry flush_entry;
	struct cache_maintenance::reg_inv_all inv_all;
	struct cache_maintenance::reg_flush_all flush_all;
	struct cache_maintenance::reg_read_entry read_entry;
	uint32_t _pad28[1];
	struct cache_maintenance::reg_info_ready info_ready;
	struct cache_maintenance::reg_info0 info0;
	struct cache_maintenance::reg_info1 info1;
};
#else

typedef struct block_cache_maintenance {
	reg_status status;
	reg_attr_mcid attr_mcid;
	reg_inv_entry inv_entry;
	reg_flush_entry flush_entry;
	reg_inv_all inv_all;
	reg_flush_all flush_all;
	reg_read_entry read_entry;
	uint32_t _pad28[1];
	reg_info_ready info_ready;
	reg_info0 info0;
	reg_info1 info1;
} block_cache_maintenance;
#endif


#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_CACHE_MAINTENANCE_H_ */

#pragma once
#include <hl_gcfg/hlgcfg_item.hpp>
using GlobalConfUint64 = hl_gcfg::GcfgItemUint64;
using GlobalConfBool = hl_gcfg::GcfgItemBool;

extern GlobalConfUint64 GCFG_TEST_VALUE_U64;
extern GlobalConfUint64 GCFG_TEST_VALUE_U64_2;
extern GlobalConfUint64 GCFG_TEST_PUBLIC_UINT64;
extern GlobalConfUint64 GCFG_TEST_PUBLIC_UINT64_2;
extern GlobalConfBool GCFG_TEST_VALUE_BOOL;

extern GlobalConfBool GCFG_MAKE_CTRL_DEP_SOFT;
extern GlobalConfBool GCFG_CHECK_SECTION_OVERLAP;
extern GlobalConfBool GCFG_TWO_NAMES;


extern GlobalConfBool GCFG_DEVICE_DEPENDENT_BOOL;
extern GlobalConfUint64 GCFG_DEVICE_DEPENDENT_UINT64;

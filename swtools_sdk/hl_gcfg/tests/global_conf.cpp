#include "global_conf.hpp"
using hl_gcfg::MakePrivate;
using hl_gcfg::MakePublic;
using GlobalConfObserver = hl_gcfg::GcfgItemObserver;
GlobalConfUint64 GCFG_TEST_VALUE_U64(
        "TEST_VALUE_U64",
        {"TEST_VALUE_U64_1"},
        "comment (0 == default)",
        999);
GlobalConfUint64 GCFG_TEST_VALUE_U64_2(
        "TEST_VALUE_U64",
        {"TEST_VALUE_U64_1"},
        "comment (0 == default)",
        999);

GlobalConfUint64 GCFG_TEST_PUBLIC_UINT64(
        "TEST_PUBLIC_VALUE_U64",
        {},
        "comment (0 == default)",
        999,
        MakePublic);

GlobalConfUint64 GCFG_TEST_PUBLIC_UINT64_2(
        "TEST_PUBLIC_VALUE_U64",
        {},
        "comment (0 == default)",
        999,
        MakePublic);

GlobalConfBool GCFG_TEST_VALUE_BOOL(
        "TEST_VALUE_BOOL",
        {"TEST_VALUE_BOOL_1"},
        "set bool value",
        false,
        MakePrivate);

GlobalConfBool GCFG_MAKE_CTRL_DEP_SOFT(
        "MAKE_CTRL_DEP_SOFT",
        "Make control edges as soft control edges (enable pipeline, do not guarantee execution order on different engines)",
        true,
        MakePrivate);

GlobalConfObserver makeCtrlDepSoftObserver(&GCFG_MAKE_CTRL_DEP_SOFT, {{"false", "false"}, {"0", "false"}, {"true", "true"}, {"1", "true"}}, true);

GlobalConfBool GCFG_CHECK_SECTION_OVERLAP(
        "CHECK_SECTION_OVERLAP_CHECK",
        "check section overlap , Warning: if this flag disabled MAKE_CTRL_DEP_SOFT functionality also disabled",
        true,
        MakePrivate,
        {&makeCtrlDepSoftObserver});

GlobalConfBool GCFG_TWO_NAMES(
        "TWO_NAMES",
        {"TWO_NAMES_1"},
        "check 2 names of a gcfg",
        false,
        MakePublic);

GlobalConfBool GCFG_DEVICE_DEPENDENT_BOOL(
        "DEVICE_DEPENDENT_BOOL",
        "device dependent value",
        hl_gcfg::DfltBool(true) <<  hl_gcfg::deviceValue(1, false),
        MakePrivate);

GlobalConfUint64 GCFG_DEVICE_DEPENDENT_UINT64(
        "DEVICE_DEPENDENT_UINT64",
        "device dependent value",
        hl_gcfg::DfltUint64 (100)  <<  hl_gcfg::deviceValue(1, 10) << hl_gcfg::deviceValue(2, 20),
        MakePrivate);
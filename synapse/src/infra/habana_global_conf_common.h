#pragma once

#include <cstdint>
#include <hl_gcfg/hlgcfg_item.hpp>

using GlobalConfUint64 = hl_gcfg::GcfgItemUint64;
using GlobalConfInt64  = hl_gcfg::GcfgItemInt64;
using GlobalConfSize   = hl_gcfg::GcfgItemSize;
using GlobalConfBool   = hl_gcfg::GcfgItemBool;
using GlobalConfFloat  = hl_gcfg::GcfgItemFloat;
using GlobalConfString = hl_gcfg::GcfgItemString;

// Common:
extern GlobalConfUint64    GCFG_TPC_ENGINES_ENABLED_MASK;
extern GlobalConfUint64    GCFG_TPC_PRINTF_MAX_BUFFER_SIZE;
extern GlobalConfUint64    GCFG_TPC_PRINTF_TENSOR_SIZE;
extern GlobalConfBool      GCFG_ENABLE_STAGED_SUBMISSION;
extern GlobalConfFloat     GCFG_MIN_TPC_PIPELINE_FACTOR;
extern GlobalConfBool      GCFG_ARC_ARCHITECTURE;

// Gaudi:
extern GlobalConfBool      GCFG_GAUDI_DEMO;
extern GlobalConfBool      GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS;

#pragma once

#include <hl_gcfg/hlgcfg_item.hpp>
#include <synapse_common_types.h>

using GlobalConfUint64 = hl_gcfg::GcfgItemUint64;
using GlobalConfFloat = hl_gcfg::GcfgItemFloat;
using GlobalConfBool = hl_gcfg::GcfgItemBool;


extern GlobalConfUint64 GCFG_ALL_REQUIRED_INPUT_CACHING_MAX_SIZE_BYTES;
extern GlobalConfUint64 GCFG_SMALL_INPUT_FORCE_CACHING_MAX_SIZE_BYTES;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_BUNDLE_MIN_NOF_SLICES;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_BUNDLE_MAX_NOF_SLICES;
extern GlobalConfFloat  GCFG_LAYERED_BRAIN_MIN_VALID_MME_UTIL_RATIO;
extern GlobalConfFloat  GCFG_LAYERED_BRAIN_MAX_MME_UTIL_RATIO_FOR_INFLATION;
extern GlobalConfFloat  GCFG_LAYERED_BRAIN_MAX_VALID_MME_BW;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_SCHEDULE_ROUTES_CREATION_ALGO;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_BRAIN_SCHEDULER_OPTIMIZE_MEMSET;
extern GlobalConfBool   GCFG_LAYERED_BRAIN_PREFER_SCHEDULING_SAME_THREAD;
extern GlobalConfBool   GCFG_LAYERED_BRAIN_SCHEDULER_PREFETCH_NEXT_THREAD;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_SCHEDULER_MIN_PIPELINE_DEPTH;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_SCHEDULER_MAX_PIPELINE_DEPTH;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_FLOW_MODE;
extern GlobalConfBool   GCFG_ENABLE_LB_DUPLICATE_SHARED_BUNDLE_INPUTS;
extern GlobalConfBool   GCFG_ENABLE_ADD_CACHE_WARMUP;
extern GlobalConfBool   GCFG_ENABLE_CACHE_WARMUP_ON_SINGLE_DCORE;
extern GlobalConfUint64 GCFG_PARTIALS_CACHE_WARMUP_KERNEL;
extern GlobalConfBool   GCFG_ENABLE_REPLACE_MEMSET_BY_CACHE_WARMUP;
extern GlobalConfUint64 GCFG_LAYERED_BRAIN_CACHE_THRESHING_PREVENTION_MODE;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_BRAIN_FULL_LOGGING;
extern GlobalConfBool   GCFG_ENABLE_MME_BRAIN;
extern GlobalConfUint64 GCFG_HARD_TRIM_MME_BRAIN_STRATEGIES;
extern GlobalConfBool   GCFG_ENABLE_CONFLICT_BASED_STRATEGY_TRIMMING;
extern GlobalConfBool   GCFG_ENABLE_BRAIN_LOCALITY_HINTS_ANNOTATION;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_BRAIN_PERFORATION;
extern GlobalConfFloat  GCFG_PERFORATION_UTILIZATION_THRESHOLD;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_PIPELINE_BRAIN;
extern GlobalConfBool   GCFG_ENABLE_BUNDLE_MEMORY_MANAGEMENT;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_BRAIN_CONV_BUNDLING;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS;
extern GlobalConfBool   GCFG_ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS;
extern GlobalConfBool   GCFG_ENABLE_LB_HYBRID_MODE;
extern GlobalConfBool   GCFG_ENABLE_LB_PARTIALS_WRITE_TPC_HANDLING;
extern GlobalConfBool   GCFG_ENABLE_LB_PARTIALS_WRITE_MME_HANDLING;
extern GlobalConfBool   GCFG_ENABLE_LB_PARTIALS_DETECTION_HALF_CL_REFINEMENT;
extern GlobalConfBool   GCFG_ENABLE_LB_MME_CONCURRENCY_OPT;
extern GlobalConfBool   GCFG_ENABLE_EVALUATE_PERFORATION_UTIL;
extern GlobalConfBool   GCFG_ENABLE_LB_PREFER_CONSUMERS;
extern GlobalConfBool   GCFG_LIMIT_GEMM_BUNDLES_EXPANSION;
extern GlobalConfBool   GCFG_LIMIT_CONV_BUNDLES_EXPANSION;
extern GlobalConfBool   GCFG_ENABLE_LB_CACHE_YIELDING;
extern GlobalConfBool   GCFG_ENABLE_LB_CACHE_REUSED_SLICES;
extern GlobalConfBool   GCFG_ENABLE_LB_NON_BPT_SLICES_DISCARDING;
extern GlobalConfUint64 GCFG_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE;
extern GlobalConfFloat  GCFG_MAX_RELATIVE_ALIGNMENT_INCREASE_RATIO;
extern GlobalConfBool   GCFG_ENABLE_LB_SAMPLE_MODE;
extern GlobalConfBool   GCFG_ENABLE_LB_PARTIALS_WRITE_UNBUNDELED_NODES_HANDLING;
extern GlobalConfUint64 GCFG_SINGLE_DCORE_PREFORATION_WORK_DCORE;
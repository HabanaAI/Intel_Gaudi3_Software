#include "brain_conf.h"

using hl_gcfg::MakePrivate;
using hl_gcfg::MakePublic;

using hl_gcfg::DfltInt64;
using hl_gcfg::DfltUint64;
using hl_gcfg::DfltBool;
using hl_gcfg::DfltFloat;
using hl_gcfg::DfltString;
using hl_gcfg::DfltSize;
using hl_gcfg::deviceValue;

GlobalConfUint64
    GCFG_ALL_REQUIRED_INPUT_CACHING_MAX_SIZE_BYTES("ALL_REQUIRED_INPUT_CACHING_MAX_SIZE_BYTES",
                                                   "Maximal size of an allRequired bundle input to be cached",
                                                   1024ull * 1024,
                                                   MakePrivate);

GlobalConfUint64 GCFG_SMALL_INPUT_FORCE_CACHING_MAX_SIZE_BYTES("SMALL_INPUT_FORCE_CACHING_MAX_SIZE_BYTES",
                                                               "Maximal size of a bundle input to be forced cached",
                                                               4096ull,
                                                               MakePrivate);

GlobalConfUint64
    GCFG_LAYERED_BRAIN_FLOW_MODE("GC_BRAIN_MODE",
                                 "0 - Forward Progress (single pass), 1 - Iterative (gradual bundle expansion)",
                                 DfltUint64(0) << deviceValue(synDeviceGaudi3, 1),
                                 MakePrivate);

GlobalConfBool
    GCFG_ENABLE_LB_DUPLICATE_SHARED_BUNDLE_INPUTS("ENABLE_LB_DUPLICATE_SHARED_BUNDLE_INPUTS",
                                                  "Avoid undirected cycles in bundle by duplicating shared input BPTs",
                                                  DfltBool(true),
                                                  MakePrivate);

GlobalConfBool GCFG_ENABLE_ADD_CACHE_WARMUP(
    "ENABLE_ADD_CACHE_WARMUP",
    "Enable adding cl aware memset/memget to warmup cache, when partials writes detected for a tensor",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CACHE_WARMUP_ON_SINGLE_DCORE(
    "ENABLE_CACHE_WARMUP_ON_SINGLE_DCORE",
    "Enable adding cache warmup node when the warmup is done on a single DCore",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfUint64 GCFG_PARTIALS_CACHE_WARMUP_KERNEL(
    "PARTIALS_CACHE_WARMUP_KERNEL",
    "Select the kernel type for cache warmup: 0 - auto, 1 - memset, 2 - memget, 3 - hybrid ",
    0,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_REPLACE_MEMSET_BY_CACHE_WARMUP(
    "ENABLE_REPLACE_MEMSET_BY_CACHE_WARMUP",
    "Enable replacing existing memset by cl aware memset when cache warmup is required",
    DfltBool(false),
    MakePrivate);

GlobalConfUint64
    GCFG_LAYERED_BRAIN_CACHE_THRESHING_PREVENTION_MODE("LAYERED_BRAIN_CACHE_THRESHING_PREVENTION_MODE",
                                                       "Select the cache threshing prevention mode: 0 - Sync all, 1 - "
                                                       "Skip syncing same engine, 2 - reserved, 3 - Sync none",
                                                       DfltUint64(3),
                                                       MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_BRAIN_FULL_LOGGING("ENABLE_LAYERED_BRAIN_FULL_LOGGING",
                                                      "Enable logging when running layered brain generic passes,"
                                                      "WARNING: enabling thie cfg may bloat the log files",
                                                      DfltBool(false),
                                                      MakePrivate);

GlobalConfBool GCFG_ENABLE_MME_BRAIN("ENABLE_MME_BRAIN",
                                     "Use MME brain to get slicing strategies for MME nodes in the bundle",
                                     DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                                     MakePrivate);

GlobalConfUint64 GCFG_HARD_TRIM_MME_BRAIN_STRATEGIES(  // TODO [SW-149627] - get rid of this. Should be temporary.
    "HARD_TRIM_MME_BRAIN_STRATEGIES",
    "Setting this to N>0 will keep only the first N strategies that MME brain devises. N=0 - keep all of them",
    DfltUint64(0),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONFLICT_BASED_STRATEGY_TRIMMING(
    "ENABLE_CONFLICT_BASED_STRATEGY_TRIMMING",
    "Enable trimming strategies with conflicting requirements for the key nodes.",
    DfltBool(true),
    MakePrivate);

GlobalConfUint64 GCFG_LAYERED_BRAIN_BUNDLE_MIN_NOF_SLICES("LAYERED_BRAIN_BUNDLE_MIN_NOF_SLICES",
                                                          "Minimal number of slices in a layered brain bundle",
                                                          DfltUint64(4),
                                                          MakePrivate);

GlobalConfUint64 GCFG_LAYERED_BRAIN_BUNDLE_MAX_NOF_SLICES("LAYERED_BRAIN_BUNDLE_MAX_NOF_SLICES",
                                                          "Maximum number of slices in a layered brain bundle",
                                                          DfltUint64(128),
                                                          MakePrivate);

GlobalConfFloat
    GCFG_LAYERED_BRAIN_MIN_VALID_MME_UTIL_RATIO("LAYERED_BRAIN_MIN_VALID_MME_UTIL_RATIO",
                                                "Minimum valid ratio of MME current/max MME node utilization",
                                                DfltFloat(0.85),
                                                MakePrivate);

GlobalConfFloat GCFG_LAYERED_BRAIN_MAX_MME_UTIL_RATIO_FOR_INFLATION(
    "LAYERED_BRAIN_MAX_MME_UTIL_RATIO_FOR_INFLATION",
    "Avoid inflation for utilization if current to max utilization ratio is bigger than this threshold",
    DfltFloat(0.99),
    MakePrivate);

GlobalConfFloat GCFG_LAYERED_BRAIN_MAX_VALID_MME_BW("LAYERED_BRAIN_MAX_VALID_MME_BW",
                                                    "Maximum valid MME bandwidth in GB/s",
                                                    DfltFloat(20000.0),
                                                    MakePrivate);

GlobalConfUint64 GCFG_LAYERED_BRAIN_SCHEDULE_ROUTES_CREATION_ALGO(
    "LAYERED_BRAIN_SCHEDULE_ROUTES_CREATION_ALGO",
    "Select layered brain scheduler's algorithm to create nodes route to slice: 0 - DFS, 1 - BFS",
    DfltUint64(1),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_BRAIN_SCHEDULER_OPTIMIZE_MEMSET(
    "ENABLE_LAYERED_BRAIN_SCHEDULER_OPTIMIZE_MEMSET",
    "Enable layered brain scheduler to be in charge of bundled memsets optimization, instead of graph scheduler",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_LAYERED_BRAIN_PREFER_SCHEDULING_SAME_THREAD(
    "LAYERED_BRAIN_PREFER_SCHEDULING_SAME_THREAD",
    "Let the threads scheduler prefer scheduling nodes from the same thread, when no data dependency detected",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_LAYERED_BRAIN_SCHEDULER_PREFETCH_NEXT_THREAD(
    "LAYERED_BRAIN_SCHEDULER_PREFETCH_NEXT_THREAD",
    "Let the threads scheduler prefetch a node from the next pending thread immediately upon thread ending",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_BRAIN_PERFORATION("ENABLE_LAYERED_BRAIN_PERFORATION",
                                                     "Enable locality optimizations for layered-brain bundles",
                                                     DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                                                     MakePrivate);

GlobalConfFloat
    GCFG_PERFORATION_UTILIZATION_THRESHOLD("PERFORATION_UTILIZATION_THRESHOLD",
                                           "Minimum percentage of accepted utilization as a result of perforation",
                                           0.87f,
                                           MakePrivate);

GlobalConfUint64 GCFG_LAYERED_BRAIN_SCHEDULER_MIN_PIPELINE_DEPTH(
    "LAYERED_BRAIN_SCHEDULER_MIN_PIPELINE_DEPTH",
    "Min number of concurrent threads scheduled by the layered brain scheduler",
    DfltUint64(2),
    MakePrivate);

GlobalConfUint64 GCFG_LAYERED_BRAIN_SCHEDULER_MAX_PIPELINE_DEPTH(
    "LAYERED_BRAIN_SCHEDULER_MAX_PIPELINE_DEPTH",
    "Max number of concurrent threads scheduled by the layered brain scheduler",
    DfltUint64(3),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_PIPELINE_BRAIN(
    "ENABLE_LAYERED_PIPELINE_BRAIN",
    "Optimize graph pipelining and memory hierarchy using brain with separated layers for subclustering,"
    " work distribution and memory hierarchy management",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool
    GCFG_ENABLE_BRAIN_LOCALITY_HINTS_ANNOTATION("ENABLE_BRAIN_LOCALITY_HINTS_ANNOTATION",
                                                "Annotate slicing and caching decisions for locality optimizations",
                                                DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                                                MakePrivate);

GlobalConfFloat GCFG_FRAGMENTATION_COMPENSATION_FACTOR(
    "FRAGMENTATION_COMPENSATION_FACTOR",
    "Consider only part of the SRAM capacity for the Layered Brain SRAM budget, so that if fragmentation occur, there "
    "will be some wiggle room to compensate.",
    DfltFloat(0.8) << deviceValue(synDeviceGaudi3, 1.0),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_BUNDLE_MEMORY_MANAGEMENT(
    "ENABLE_BUNDLE_MEMORY_MANAGEMENT",
    "When slicing in forward progress (layered) brain, this flag enables placement of slices in SRAM after scheduling",
    DfltBool(true) << deviceValue(synDeviceGaudi2,  false),  // disable in Gaudi2 to make it more similar to Gaudi3 when using LB.
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_BRAIN_CONV_BUNDLING("ENABLE_LAYERED_BRAIN_CONV_BUNDLING",
                                                       "Enable collecting conv (fwd/bwd) seeds in the layered brain",
                                                       DfltBool(true),
                                                       MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS",
                                                         "Enable collecting multi-mme seeds in the layered brain",
                                                         DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                                                         MakePrivate);

GlobalConfBool GCFG_ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS("ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS",
                                                         "Enable MME->TPC->MME pattern in the layered brain bundler",
                                                         DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                                                         MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_HYBRID_MODE("ENABLE_LB_HYBRID_MODE",
                                          "Run pipeline management as a fallback after layered brain",
                                          DfltBool(false),
                                          MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_PARTIALS_WRITE_TPC_HANDLING(
    "ENABLE_LB_PARTIALS_WRITE_TPC_HANDLING",
    "Detect partials writes for TPC nodes outputs and prevent multiple cores access",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_PARTIALS_WRITE_MME_HANDLING(
    "ENABLE_LB_PARTIALS_WRITE_MME_HANDLING",
    "Detect partials writes for MME nodes outputs and prevent multiple cores access",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_PARTIALS_DETECTION_HALF_CL_REFINEMENT(
    "ENABLE_LB_PARTIALS_DETECTION_HALF_CL_REFINEMENT",
    "Enable a refinement to the partials writes detection to avoid handling outputs aligned to half cache line",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_MME_CONCURRENCY_OPT(
    "ENABLE_LB_MME_CONCURRENCY_OPT",
    "enable MME to generate solutions with concurrency optimizations - batch,cd,hybrid.",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_EVALUATE_PERFORATION_UTIL(
    "ENABLE_EVALUATE_PERFORATION_UTIL",
    "Evaluate bundle perforation using perforation util metric instead of the perforation bvd multiplier",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_PREFER_CONSUMERS("ENABLE_LB_PREFER_CONSUMERS",
                                               "When enabled LB attempts expanding from seed consumers before "
                                               "producers. When disabled producers are expanded first.",
                                               DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                                               MakePrivate);

GlobalConfBool
    GCFG_LIMIT_GEMM_BUNDLES_EXPANSION("LIMIT_GEMM_BUNDLES_EXPANSION",
                                      "Block compositions with both producers and consumers for GEMM bundles.",
                                      DfltBool(false) << deviceValue(synDeviceGaudi3, false),
                                      MakePrivate);

GlobalConfBool
    GCFG_LIMIT_CONV_BUNDLES_EXPANSION("LIMIT_CONV_BUNDLES_EXPANSION",
                                      "Block compositions with both producers and consumers for CONV bundles.",
                                      DfltBool(false) << deviceValue(synDeviceGaudi3, false),
                                      MakePrivate);

GlobalConfBool
    GCFG_ENABLE_LB_CACHE_YIELDING("ENABLE_LB_CACHE_YIELDING",
                                  "Enable yielding the cache budget of infrequently used buffers for other users",
                                  DfltBool(true),
                                  MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_CACHE_REUSED_SLICES("ENABLE_LB_CACHE_REUSED_SLICES",
                                                  "Enable caching slices that are reused multiple times in the bundle "
                                                  "schedule, even if each operation reads them once.",
                                                  DfltBool(true),
                                                  MakePrivate);

// TODO: SW-174721 Enable CME discard when accuracy issue is fixed (SW-170068)
GlobalConfBool
    GCFG_ENABLE_LB_NON_BPT_SLICES_DISCARDING("ENABLE_LB_NON_BPT_SLICES_DISCARDING",
                                             "Enable adding discard CME events for non-BPT slices after they are no "
                                             "longer used. This saves the HBM BW required for their write-back.",
                                             DfltBool(false),
                                             MakePrivate);

GlobalConfUint64 GCFG_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE(
    "ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE",
    "0: disabled, 1: align to full CL, 2: align fcd stride < CL/2 to CL/2 else align to full CL",
    DfltUint64(0) << deviceValue(synDeviceGaudi3, 1),
    MakePrivate);

GlobalConfFloat GCFG_MAX_RELATIVE_ALIGNMENT_INCREASE_RATIO(
    "MAX_RELATIVE_ALIGNMENT_INCREASE_RATIO",
    "Depicts the max allowed increase in size due to alignment of BPT fcd stride to cache line size"
    "relative to the unaligned size. To align all eligible bpts, set to a value >> 1.0."
    "To prevent alignment, set to 0.",
    DfltFloat(100.0),
    MakePrivate);

GlobalConfBool
    GCFG_ENABLE_LB_SAMPLE_MODE("ENABLE_LB_SAMPLE_MODE",
                               "Enable partial sliced graph generation in dry-runs to improve compilation time",
                               DfltBool(false) << deviceValue(synDeviceGaudi3, true),
                               MakePrivate);

GlobalConfBool GCFG_ENABLE_LB_PARTIALS_WRITE_UNBUNDELED_NODES_HANDLING(
    "ENABLE_LB_PARTIALS_WRITE_UNBUNDELED_NODES_HANDLING",
    "Enable layered brain partials write handling of unbundeled nodes",
    DfltBool(false),
    MakePrivate);

GlobalConfUint64 GCFG_SINGLE_DCORE_PREFORATION_WORK_DCORE(
    "SINGLE_DCORE_PREFORATION_WORK_DCORE",
    "set work dcore to perforate on when perforating on single dcore",
    DfltUint64(0),
    MakePrivate);
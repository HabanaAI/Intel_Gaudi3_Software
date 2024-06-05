#include <bitset>
#include <string.h>
#include <limits>
#include <hl_gcfg/hlgcfg_default_item.hpp>
#include <hl_gcfg/hlgcfg_item.hpp>
#include <hl_gcfg/hlgcfg_item_observer.hpp>

#include "synapse_common_types.h"
#include "habana_global_conf.h"
using GlobalConfObserver = hl_gcfg::GcfgItemObserver;
using hl_gcfg::DfltInt64;
using hl_gcfg::DfltUint64;
using hl_gcfg::DfltBool;
using hl_gcfg::DfltFloat;
using hl_gcfg::DfltString;
using hl_gcfg::DfltSize;
using hl_gcfg::deviceValue;

using hl_gcfg::MakePrivate;
using hl_gcfg::MakePublic;
/* Habana (common) Global Configurations definition */
/*--------------------------------------------------*/

// clang-format off

GlobalConfUint64 GCFG_SET_SRAM_SIZE(
        "SET_SRAM_SIZE",
        "Set SRAM size in bytes (0 == default)",
        0);

GlobalConfBool GCFG_INTERNAL_SIMULATION(
    "INTERNAL_SIMULATION",
    "Set internal simulation mode",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_RAGGED_SOFTMAX_OPT(
    "ENABLE_RAGGED_SOFTMAX_OPT",
    "Try to fuse ragged softmax pattern from BERT",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_GC_NODES_VALIDATION_BY_OPS_DB(
    "ENABLE_GC_NODES_VALIDATION_BY_OPS_DB",
    "Enable gc validation of synapse API nodes",
    true,
    MakePrivate);

GlobalConfFloat GCFG_RAGGED_SOFTMAX_OPT_AMP_VAL(
    "RAGGED_SOFTMAX_OPT_AMP_VAL",
    "The max scalar value for the valid indices amplifying mult node",
    -10000.0,
    MakePublic);

GlobalConfUint64 GCFG_TPC_ENGINES_ENABLED_MASK(
    "TPC_ENGINES_ENABLED_MASK",
    "TPC engines enabled mask (bitmask)",
    0xFFFFFFFFFFFFFFFF,
    MakePrivate); // setting the mask to cover the platform with the largest number of engines (gaudi3 has 64 TPC engines)

GlobalConfBool GCFG_GRAPH_VISUALIZATION(
    "GRAPH_VISUALIZATION",
    "Enable creation of graph visualization files",
    false,
    MakePublic);

GlobalConfBool GCFG_GRAPH_VISUALIZATION_COLLAPSE_BUNDLES(
    "GRAPH_VISUALIZATION_COLLAPSE_BUNDLES",
    "Create graph visualization with bundles collapsed into one node for each",
    false,
    MakePrivate);

GlobalConfString GCFG_GRAPH_VISUALIZATION_START_TENSOR(
    "GRAPH_VISUALIZATION_START_TENSOR",
    "Set graph visualization start tensor",
    std::string(),
    MakePrivate);

GlobalConfString GCFG_GRAPH_VISUALIZATION_END_TENSOR(
    "GRAPH_VISUALIZATION_END_TENSOR",
    "Set graph visualization end tensor",
    std::string(),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_GVD(
    "ENABLE_GVD",
    "Enable creation of full graph visualization directory",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_PARTIAL_GVD(
    "ENABLE_PARTIAL_GVD",
    "Enable creation of full graph visualization directory, dump pass graph only when graph changed",
    false,
    MakePrivate);

GlobalConfInt64 GCFG_VISUALIZATION_MODE(
    "VISUALIZATION_MODE",
    "Visualization mode for GVD & Graph Visualization. 0 = MXNet style json, 1 = TF style protobuf txt (default)",
    1,
    MakePrivate);

GlobalConfInt64 GCFG_GVD_PASS_FILTER(
    "GVD_PASS_FILTER",
    "Add visualization for specific pass, GCFG_ENABLE_GVD should be enabled",
    -1,
    MakePrivate);

GlobalConfUint64 GCFG_DEBUG_MODE(
    "DEBUG_MODE",
    "Generic flag for turning on/off debugging code. 0 - no debugging, 1 - Basic Resnet debug,\
    2 - Resnet debug with persistent tensors dump, 3 - Resnet debug with persistent and intermediate tensors dump",
    0,
    MakePrivate);


GlobalConfBool GCFG_DISABLE_LOAD_DIFF_DESC(
    "DISABLE_LOAD_DIFF_DESC",
    "Disable load diff descriptor",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_DEFAULT_PIPELINE_DEPTH(
    "DEFAULT_PIPELINE_DEPTH",
    "Set default pipeline depth",
    3,
    MakePrivate);

GlobalConfUint64 GCFG_MAX_DYNAMIC_PIPELINE_DEPTH(
    "MAX_DYNAMIC_PIPELINE_DEPTH",
    "Limit max pipeline depth when decide dynamically",
    8,
    MakePrivate);

GlobalConfUint64 GCFG_MAX_NUM_DMA_CHUNKS(
    "MAX_NUM_DMA_CHUNKS",
    "Set max number of DMA chunks",
    256,
    MakePrivate);

GlobalConfUint64 GCFG_NUM_OF_THREADS_CONF(
    "NUM_OF_THREADS_CONF",
    "Number of threads to be open when using thread pool. 0/1 to disable",
    1,
    MakePrivate);

GlobalConfBool GCFG_COMPRESS_BLOBS(
    "COMPRESS_BLOBS",
    "Whether to compress the blobs",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS(
    "ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS",
    "Enable inplace reuse for TPC suggestion requests",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_MAX_PATH_SCHEDULE(
    "ENABLE_MAX_PATH_SCHEDULE",
    "enable max-path scheduling for TPC-only graphs",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_PARENT_ID_SCHEDULE(
    "ENABLE_PARENT_ID_SCHEDULE",
    "use parent-id as the tie-breaker when scheduling graphs",
    true,
    MakePrivate);

GlobalConfBool GCFG_PRINT_FILE_AND_LINE(
    "PRINT_FILE_AND_LINE",
    "Whether to print the file name and the line",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_PROFILER(
    "ENABLE_PROFILER",
    "Enable Habana profiler",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY(
    "ENABLE_REMOVE_REDUNDANT_MEMCPY",
    "Enable remove redundant memcopy nodes from pre-graph",
    true,
    MakePrivate);

GlobalConfBool GCFG_PRINT_TIME(
    "PRINT_TIME",
    "Whether to print the time",
    true,
    MakePrivate);

GlobalConfBool GCFG_PRINT_DATE(
    "PRINT_DATE",
    "Whether to print the date",
    false,
    MakePrivate);

GlobalConfBool GCFG_LOG_THREAD_CONTEXT(
    "LOG_THREAD_CONTEXT",
    "Add thread context to the beginning of each log line",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_TPC_PRINTF_MAX_BUFFER_SIZE(
    "TPC_PRINTF_MAX_BUFFER_SIZE",
    "Max total size of TPC printf tensors allowed",
    20*1024*1024,  // 20MB
    MakePrivate);

GlobalConfUint64 GCFG_TPC_PRINTF_TENSOR_SIZE(
    "TPC_PRINTF_TENSOR_SIZE",
    "TPC printf tensor buffer size",
    DfltUint64(512*1024)
    << deviceValue(synDeviceGaudi2, 2*1024*1024)  // 2MB for Gaudi2
    << deviceValue(synDeviceGaudi3, 8*1024*1024), // 8MB for Gaudi3
    MakePrivate);

GlobalConfBool GCFG_CREATE_USED_CONFIGS_FILE(
    "CREATE_USED_CONFIGS_FILE",
    "Dump of used configurations",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_STATS(
    "ENABLE_STATS",
    "Enable statistics collection",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_STATS_TBL(
    "ENABLE_STATS_TBL",
    "Enable statistics collection in table format",
    false,
    MakePrivate);

GlobalConfBool GCFG_STATS_PER_SYNLAUNCH(
    "STATS_PER_SYNLAUNCH",
    "Collect Global statistics after every synLaunch",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_STATS_FREQ(
    "STATS_FREQ",
    "Collect Global statistics frequency",
    0,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SEP_GC_LOG_PER_THREAD(
    "ENABLE_SEP_GC_LOG_PER_THREAD",
    "Create separate graph_compiler log per graph (therefore per thread)",
    false,
    MakePrivate);

GlobalConfBool GCFG_EMBED_BY_WEIGHTS(
    "EMBED_BY_WEIGHTS",
    "Quantize Embedding node by weights scale",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_BUNDLES(
    "ENABLE_TPC_BUNDLES",
    "Create TPC bundles for more than 3 TPC in a sequence, between two MME",
    DfltBool(false)
    << deviceValue(synDeviceGaudi, true)
    << deviceValue(synDeviceGaudi2, true)
    << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_HBM_SLICES_ALLOCATION_OPTIMIZATION(
    "ENABLE_HBM_SLICES_ALLOCATION_OPTIMIZATION",
    "Optimize slices allocation in HBM by allocating all of them at the same time as one big tensor",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_DETERMINISTIC_MODE(
    "DETERMINISTIC_MODE",
    "Controls node deterministics computation mode \
    (0=force off, 1=force on, 2=user controlled - default) ",
    DfltUint64(2),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SYNAPSE_QUANTIZATION(
    "ENABLE_SYNAPSE_QUANTIZATION",
    "Enable quantization in synapse",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_ELIMINATE_DYNAMIC_RANGE_OP(
    "ENABLE_ELIMINATE_DYNAMIC_RANGE_OP",
    "Enable removal of dynamic range nodes",
    false,
    MakePrivate);

GlobalConfBool GCFG_SYNAPSE_DATA_TYPE_SELECTION(
    "SYNAPSE_DATA_TYPE_SELECTION",
    "Enable data type selection in synapse",
    DfltBool(true),
    MakePublic);

GlobalConfBool GCFG_IGNORE_USER_DATA_TYPE(
    "IGNORE_USER_DATA_TYPE",
    "Ignore user given data types during data type selection",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE(
        "ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE",
        "Enable memory oriented schedule for flash attention subgraphs",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_FLASH_ATTENTION_SLICING(
        "ENABLE_FLASH_ATTENTION_SLICING",
        "Slice SDPA nodes on external batch dim before complex-guid pass",
        DfltBool(false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_LONG_UNSLICED_NON_SHARED_PROD_CHAIN(
    "ENABLE_LONG_UNSLICED_NON_SHARED_PROD_CHAIN",
    "Enable to bundle more than one tpc producer in unsliced non shared producer chain",
    false,
    MakePrivate);

GlobalConfBool GCFG_UPDATE_MME_PRECISION(
    "UPDATE_MME_PRECISION",
    "Update mme nodes input precision to PROFILE_PRECISION",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_UPDATE_GRAPH_OUTPUT_MME(
    "UPDATE_GRAPH_OUTPUT_MME",
    "Update graph output mme nodes input precision to PROFILE_PRECISION",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CALC_DYNAMIC_RANGE(
    "ENABLE_CALC_DYNAMIC_RANGE",
    "Enable calc dynamic range pass",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ALLOW_DEFAULT_QUANT_PARAMS(
    "ALLOW_DEFAULT_QUANT_PARAMS",
    "Use default quantization params, if was not provided by user and dynamic range was not provided",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_PROPAGATE_CASTS(
    "PROPAGATE_CASTS",
    "Propagate casts to lower precision data type, above logical operations for better performance",
    DfltBool(true),
    MakePrivate);
GlobalConfString GCFG_UPDATE_MME_OUTPUT_PRECISION_FILTER(
    "UPDATE_MME_OUTPUT_PRECISION_FILTER",
    "Set filterd MME nodes output precision to PROFILE_PRECISION seprated by ',' delimiter",
    std::string(""),
    MakePrivate);

GlobalConfString GCFG_PROFILE_PRECISION(
    "PROFILE_PRECISION",
    "The minimum precision that is allowed for the nodes",
    std::string("hf8"),
    MakePublic);

GlobalConfString GCFG_PRECISION_TO_RAISE(
    "PRECISION_TO_RAISE",
    "The precision to raise to in case a layer (node) is chosen to be raised",
    std::string(""),
    MakePrivate);

GlobalConfUint64 GCFG_NUM_OF_LAYERS_TO_RAISE(
    "NUM_OF_LAYERS_TO_RAISE",
    "The Number of layers (nodes) to raise their precision",
    0,
    MakePublic);

GlobalConfBool GCFG_DISABLE_ADJUST_RESTRICTIONS(
    "DISABLE_ADJUST_RESTRICTIONS",
    "Disable adjust restrictions pass",
    false,
    MakePrivate);

GlobalConfBool GCFG_MEDIA_RECIPES(
    "MEDIA_RECIPES",
    "Enable dumping recipes for media layer use",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_SPARSITY_WEIGHTS(
    "ENABLE_SPARSITY_WEIGHTS",
    "Enable sparsity weights quantization for MME weights",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_MAPPING_IN_STREAM_COPY(
    "ENABLE_MAPPING_IN_STREAM_COPY",
    "Enable mapping the host buffer in the stream copy",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_MAX_WAIT_TIME_FOR_MAPPING_IN_STREAM_COPY(
    "MAX_WAIT_TIME_FOR_MAPPING_IN_STREAM_COPY",
    "In case mapping exists, the max time in ms to wait for the memory to unmap",
    10000,
    MakePrivate);

GlobalConfUint64 GCFG_POOL_MAPPING_SIZE_IN_STREAM_COPY(
    "POOL_MAPPING_SIZE_IN_STREAM_COPY",
    "use a pool when mapping the host buffer in the stream copy",
    200,// in MB, 0 to disable
    MakePrivate);

GlobalConfBool GCFG_ENABLE_POOL_MAPPING_WAIT_IN_STREAM_COPY(
    "ENABLE_POOL_MAPPING_WAIT_IN_STREAM_COPY",
    "when using a pool for mapping, wait if pool is full instead of using the LKD for mapping",
    true,
    MakePrivate);

GlobalConfBool GCFG_DFA_ON_SIGNAL(
    "DFA_ON_SIGNAL",
    "Start DFA flow on an exception signal",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_STAGED_SUBMISSION(
    "ENABLE_STAGED_SUBMISSION",
    "Enable stage submission by patch points ordering",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_STAGED_SUBMISSION_NODES_PER_STAGE(
    "STAGED_SUBMISSION_NODES_PER_STAGE",
    "Number of nodes per stage submission (must be > 1)",
    200,
    MakePrivate);

GlobalConfFloat GCFG_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR(
    "STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR",
    "Factor to apply to STAGED_SUBMISSION_NODES_PER_STAGE after every stage (must be >= 1.0)",
    4.0,
    MakePrivate);

GlobalConfUint64 GCFG_DSD_STAGED_SUBMISSION_NODES_PER_STAGE(
    "DSD_STAGED_SUBMISSION_NODES_PER_STAGE",
    "In case of dynamic shapes - number of nodes per stage submission (must be > 1)",
    10,
    MakePrivate);

GlobalConfFloat GCFG_DSD_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR(
    "DSD_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR",
    "In case of dynamic shapes - factor to apply to STAGED_SUBMISSION_NODES_PER_STAGE after every stage (must be >= 1.0)",
    3,
    MakePrivate);

GlobalConfBool GCFG_STAGED_SUBMISSION_NODE_EXE_VALIDATION(
    "STAGED_SUBMISSION_NODE_EXE_VALIDATION",
    "Enable stage submission node exe validation (will have performance impact)",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS(
    "ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS",
    "Enable profiler annotations specific for synLaunch",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_RMW_SECTION_MAX_SIZE_BYTES(
    "SYN_RMW_SECTION_MAX_SIZE_BYTES",
    "Maximal size of an RMW section (per device)",
    DfltUint64(16ull * 1024 * 1024),
    MakePrivate);

GlobalConfBool GCFG_MAKE_LOGICAL_CRTL_DEP_PHYSICAL(
    "MAKE_LOGICAL_CRTL_DEP_PHYSICAL",
    "Make logical operation physical when enforcing control dependency on the node.",
    false,
    MakePrivate);

GlobalConfBool GCFG_MAKE_CTRL_DEP_SOFT(
    "MAKE_CTRL_DEP_SOFT",
    "Make control edges as soft control edges (enable pipeline, do not guarantee execution order on different engines)",
    true,
    MakePrivate);

GlobalConfBool GCFG_DUMP_QUANT_INFO(
    "DUMP_QUANT_INFO",
    "dump quantization info per tensor to json file",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_DEBUG_NUM_OF_DOUBLE_WBULK_REGS_TO_DUMP(
    "DEBUG_NUM_OF_DOUBLE_WBULK_REGS_TO_DUMP",
    "Number of double-regs (u64) to print in WBulk dump",
    10,
    MakePrivate);

GlobalConfString GCFG_TPC_FUSER_LIB_NAME(
    "TPC_FUSER_LIB_NAME",
    "TPC Fuser shared object name (Default=libTPCFuser.so)",
    DfltString("libTPCFuser.so"),
    MakePrivate);

GlobalConfString GCFG_COMPLEX_GUID_LIB_NAME(
    "COMPLEX_GUID_LIB_NAME",
    "Complex GUID shared object name (Default=libComplexGuid.so)",
    DfltString("libComplexGuid.so"),
    MakePrivate);

GlobalConfUint64 GCFG_COMPLEX_GUID_EXTRACTOR_MODE(
     "COMPLEX_GUID_EXTRACTOR_MODE",
     "Possible modes for complex guid extractor: 0 - disabled, 1 - external lib, 2 - dummy lib",
     DfltUint64(1),
     MakePrivate);

GlobalConfBool GCFG_COMPLEX_GUID_CLUSTERING(
    "COMPLEX_GUID_CLUSTERING",
    {"NORM_MOMENTS_CLUSTERING"},
    "Enable optimization for TPC Fuser clustering of nodes extracted from specific complex guids",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_COMPLEX_GUID_SKIP_PERF_PASS(
    "COMPLEX_GUID_SKIP_PERF_PASS",
    "Skip performance CGUID extraction pass",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_COMPLEX_GUID_VALIDATE_EXTRACTED_GRAPH(
    "COMPLEX_GUID_VALIDATE_EXTRACTED_GRAPH",
    "Skip performance CGUID extraction pass",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_STITCHING_TO_BGEMM(
    "ENABLE_TPC_STITCHING_TO_BGEMM",
    "Enable stitching of TPC nodes to batch gemm nodes in Gaudi1",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONV_PACKING_TRAINING(
    "ENABLE_CONV_PACKING_TRAINING",
    "Set packing enabled mode",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_PACKING_FACTOR_COST_FUNCTION(
    "ENABLE_PACKING_FACTOR_COST_FUNCTION",
    "Use MME utilization coast function to select the optimal packing factor",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING(
    "ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING",
    "Enable constant folding of gconv_fwd kernel",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONSTANT_FOLDING(
    "ENABLE_CONSTANT_FOLDING",
    "Enable constant folding of all nodes",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_WEIGHT_PACKING_CONSTANT_FOLDING(
    "ENABLE_WEIGHT_PACKING_CONSTANT_FOLDING",
    "Enable constant folding weight packing",
    false,
    MakePrivate);

// Delay the return of free tensors to reduce the chance for
// memory dependency bubble
GlobalConfUint64 GCFG_TENSORS_KEEP_ALLOCATED(
    "TENSORS_KEEP_ALLOCATED",
    "Number of tensors to keep allocated. 0 - free immediately",
    5,
    MakePrivate);

GlobalConfUint64 GCFG_SYNAPSE_MLIR_MODE(
    "SYNAPSE_MLIR_MODE",
    "Modes of optimizations in MLIR: "
    "0 - disabled, 1 - enabled, 2 - enabled with graph validations (testing and debug mode)",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_MAX_CONST_TENSOR_SIZE_BYTES(
    "MAX_CONST_TENSOR_SIZE_BYTES",
    "Maximal size of const tensor in bytes",
    DfltUint64(1ull * 1024 * 1024),
    MakePrivate);

GlobalConfUint64 GCFG_HBM_GLOBAL_MEM_SIZE_MEGAS(
    "HBM_GLOBAL_MEM_SIZE_MEGAS",
    "HBM global size for recipe in megas",
    DfltUint64(64ull),
    MakePrivate);

/* Goya Global Configurations definition */
/*---------------------------------------*/

GlobalConfBool GCFG_PACKING_ENABLED(
    "PACKING_ENABLED",
    "Set packing enabled mode (GOYA)",
    false,
    MakePrivate);

GlobalConfBool GCFG_COMPATIBILITY_MODE(
    "COMPATIBILITY_MODE",
    "Set compatibility mode (GOYA)",
    false,
    MakePrivate);

GlobalConfBool GCFG_ELIMINATE_FIRST_TRANSPOSE(
    "ELIMINATE_FIRST_TRANSPOSE",
    "Eliminate graph first transpose operation (GOYA)",
    false,
    MakePublic);

GlobalConfBool GCFG_ELIMINATE_LAST_TRANSPOSE(
    "ELIMINATE_LAST_TRANSPOSE",
    "Eliminate graph last transpose operation (GOYA)",
    false,
    MakePublic);

GlobalConfBool GCFG_DISABLE_TENSORS_PINNING(
    "DISABLE_TENSORS_PINNING",
    "Disable tensors pinning (GOYA)",
    false,
    MakePublic);

GlobalConfBool GCFG_ENABLE_BREAKPOINT_MODE(
    "ENABLE_BREAKPOINT_MODE",
    "Enable breakpoint mode (GOYA)",
    false,
    MakePrivate);

GlobalConfBool GCFG_DISABLE_NON_SIGNALING_ROI_BREAKPOINT(
    "DISABLE_NON_SIGNALING_ROI_BREAKPOINT",
    "Do not generate breakpoint for non signaling ROIs (GOYA)",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_PINNING_BUFFER_SIZE(
    "PINNING_BUFFER_SIZE",
    "Set pinning buffer size (GOYA)",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_PREFETCH_BUFFER_SIZE(
    "PREFETCH_BUFFER_SIZE",
    "Set prefetch buffer size (GOYA)",
    0,
    MakePrivate);


GlobalConfBool GCFG_ALLOCATE_ALL_IN_DRAM(
    "ALLOCATE_ALL_IN_DRAM",
    "Allocate all in DRAM (GOYA)",
    false,
    MakePrivate);

/* Gaudi Global Configurations definition */
/*----------------------------------------*/

GlobalConfBool GCFG_GAUDI_DEMO(
    "GAUDI_DEMO",
    "Run specific changes for gaudi demo (GAUDI)",
    false,
    MakePrivate);

GlobalConfBool GCFG_RUN_TPC_FUSER(
    "RUN_TPC_FUSER",
    {"GAUDI_RUN_TPC_FUSER"},
    "Try to fuse TPC nodes",
    true,
    MakePrivate);

GlobalConfBool GCFG_DISABLE_PARALLELISM(
    "DISABLE_PARALLELISM",
    "Disable all parallelism between adjacent nodes in execution order",
    false,
    MakePrivate);

GlobalConfBool GCFG_GAUDI_ENABLE_GROUP_CONV_TO_FILTER_2D(
    "GAUDI_ENABLE_GROUP_CONV_TO_FILTER_2D",
    "Enable the pass that replaces group conv with filter 2D. (GAUDI)",
    false,
    MakePrivate);


GlobalConfBool GCFG_ENABLE_BROADCAST_TPC_FUSION(
    "ENABLE_BROADCAST_TPC_FUSION",
    "Enable fusion of broadcast and tpc kernels that support broadcasting",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE(
    "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE",
    "perform strided copy with small fcd with transpose engine if it possible",
    DfltBool(false) << deviceValue(synDeviceGaudi, true)  << deviceValue(synDeviceGaudi2, true),
    MakePrivate);

GlobalConfUint64 GCFG_DMA_TRANSPOSE_SOLVER_MIN_FCD_SIZE(
    "DMA_TRANSPOSE_SOLVER_MIN_FCD_SIZE",
    "Minimal size of input 2d transpose fcd for output to be in sram.",
    15000,
    MakePrivate);

GlobalConfUint64 GCFG_TRANSPOSE_DONT_CARE_AGGREGATION_NODE_LOSS_FACTOR(
    "TRANSPOSE_DONT_CARE_AGGREGATION_NODE_LOSS_FACTOR",
    "A factor to decide when an aggregation node will cause too much performance degradation when running on the user"
    "layout without additional transposes in comparison to aggregation node being wrapped in transposes.",
    16,
    MakePrivate);

GlobalConfFloat GCFG_TRANSPOSE_DONT_CARE_MIN_FCD_UTILIZATION_THRESHOLD(
    "TRANSPOSE_DONT_CARE_MIN_FCD_UTILIZATION_THRESHOLD",
    "The minimum threshold for the FCD utilization percentage of a TPC node after it’s been wrapped with transposes,"
    "which, if exceeded, the wrapping process is skipped to prevent potential significant performance degradation.",
    0.2,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE(
    "ENABLE_RESTRICTED_LAYOUTS_MODE",
    "Enable marking layouts as restricted by perfLib/CGUID",
    false,
    MakePrivate);

GlobalConfBool GCFG_TRANSPOSE_DONT_CARE_USE_BFS(
    "TRANSPOSE_DONT_CARE_USE_BFS",
    "Use a BFS based algorithm to traverse the graph in the pass",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_DMA_TRANSPOSE_SOLVER_MAX_SCD_SIZE(
    "DMA_TRANSPOSE_SOLVER_MAX_SCD_SIZE",
    "Maximal size of input 2d transpose scd for output to be in sram.",
    127,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_DMA_TRANSPOSE_SOLVER(
    "ENABLE_DMA_TRANSPOSE_SOLVER",
    "Enable DMA transpose solver for Gaudi1",
    true,
    MakePrivate);

// For more details, see comment regarding the composition of virtual address in utils.cpp
GlobalConfUint64 GCFG_MEMORY_ID_BITFIELD_WIDTH_CONF(
    "MEMORY_ID_BITFIELD_WIDTH_CONF",
    "Number of bits (MSB) assigned to the memory ID part within tensor virtual address. (GAUDI)",
    24,
    MakePrivate);

GlobalConfUint64 GCFG_TPC_SYNC_TRACE_EN_MASK(
    "TPC_SYNC_TRACE_EN_MASK",
    "Set TPC trace enabled mask (GAUDI)",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_MME_SYNC_TRACE_EN_MASK(
    "MME_SYNC_TRACE_EN_MASK",
    "Set MME trace enabled mask (GAUDI)",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_DMA_SYNC_TRACE_EN_MASK(
    "DMA_SYNC_TRACE_EN_MASK",
    "Set DMA trace enabled mask (GAUDI)",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_ROTATOR_SYNC_TRACE_EN_MASK(
    "ROTATOR_SYNC_TRACE_EN_MASK",
    "Set rotator trace enabled mask",
    0,
    MakePrivate);

GlobalConfBool GCFG_LIN_DMA_OPTIMIZATION_ENABLED(
    "LIN_DMA_OPTIMIZATION_ENABLED",
    "In linear DMA transaction, use LinDMA packet instead of Tensor DMA descriptor",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SPILL_FILL_FUSION(
    "ENABLE_SPILL_FILL_FUSION",
    "This flag enables the optimization of fusion of spill and fill directives to their producer/consumer",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_MME_INDEX_SPACE(
    "ENABLE_MME_INDEX_SPACE",
    "set ROIs according the mme index space",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_PIPELINE_MANAGEMENT_SRAM_OVERRIDE(
    "ENABLE_PIPELINE_MANAGEMENT_SRAM_OVERRIDE",
    "Discard SRAM decisions as  taken by pipeline management",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

GlobalConfBool GCFG_LITE_PERFORATION_SKIP_BUNDLE_CHECK(
    "LITE_PERFORATION_SKIP_BUNDLE_CHECK",
    "Skip budle check to allow cache maintenance commands for unbundled nodes",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_DCORE_LOCALITY_SPLIT(
    "ENABLE_DCORE_LOCALITY_SPLIT",
    "Enable the locality dcore split",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_UNEVEN_PERFORATION_IN_MME(
    "ENABLE_UNEVEN_PERFORATION_IN_MME",
    "Enable uneven dcore split for MME nodes",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_COST_MODEL_ENABLED(
    "SRAM_SLICER_COST_MODEL_ENABLED",
    "Enable cost model for strategies comparison (instead of metrics)",
    DfltBool(false) << deviceValue(synDeviceGaudi, true),
    MakePrivate);

GlobalConfBool GCFG_FAST_COMPILATION(
    "FAST_COMPILATION",
    "Enable fast compilation (will remove some optimizations)",
    DfltBool(false),
    MakePrivate);

GlobalConfObserver makeFastCompilationObserver(&GCFG_FAST_COMPILATION, {{"true", "false"}, {"1", "false"}}, true);

GlobalConfBool GCFG_SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_ENABLED(
    "SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_ENABLED",
    "Enable creating additional strategies which have larger slices than the originally created slices",
    true,
    MakePrivate,
    {&makeFastCompilationObserver});

GlobalConfBool GCFG_SRAM_SLICER_ALIGN_TO_CACHE_LINE(
    "SYN_ENABLE_SRAM_SLICER_ALIGN_TO_CACHE_LINE",
    "Align tensor sizes to cache line (MME optimization) during SRAM slicing pass (GAUDI)",
    true,
    MakePrivate);

GlobalConfBool GCFG_ALIGN_ALL_MME_INPUTS(
    "ALIGN_ALL_MME_INPUTS",
    "Try to align all mme inputs in a bundle",
    true,
    MakePrivate);

GlobalConfBool GCFG_PREFER_SLICING_UNALIGNED_MME_INPUT(
    "PREFER_SLICING_UNALIGNED_MME_INPUT",
    "Give priority to unaligned MME input in slicing and fetching to SRAM for single node bundles",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS(
    "ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS",
    "Enable adding consumer chain with unlimited granularity to language bundle with producers",
    true,
    MakePrivate);

// note: effective only if GCFG_SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_ENABLED == true
GlobalConfFloat GCFG_SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_FACTOR(
    "SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_FACTOR",
    "Control the multiplication factor of the graph size optimization solver",
    2.0f,
    MakePrivate);

GlobalConfUint64 GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES(
    "SRAM_SLICER_MAX_CAPACITY_BYTES",
    "Maximal SRAM capacity to use when slicing operation to fit in SRAM "
    "(0: Slicing disabled; -1: Use platform default)",
    0xFFFFFFFFFFFFFFFF,
    MakePrivate);

GlobalConfUint64 GCFG_SRAM_SLICER_BATCHGEMM_MAX_BATCH_SIZE(
    "SRAM_SLICER_BATCHGEMM_MAX_BATCH_SIZE",
    "Maximal BatchGemm batch size of each slice",
    256,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_FORCE_BATCHGEMM_MAX_BATCH_SIZE(
    "SRAM_SLICER_FORCE_BATCHGEMM_MAX_BATCH_SIZE",
    "Force maximal BatchGemm batch size of each slice (ignore optimal batch sizes)",
    false,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_BUNDLE_EXPANSION_ENABLED(
    "SRAM_SLICER_BUNDLE_EXPANSION_ENABLED",
    "Enable more SRAM utilization if possible",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_MME_TPC_EXPANSION_ENABLED(
    "SRAM_SLICER_MME_TPC_EXPANSION_ENABLED",
    "Enable MME -> TPC data sharing through SRAM when possible",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED(
    "SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED",
    "Enable MME shared input slice sharing through SRAM when possible",
    true,
    MakePrivate,
    {&makeFastCompilationObserver});

GlobalConfBool GCFG_SRAM_SLICER_SHARED_MME_INPUT_PRODUCER_EXPANSION_ENABLED(
    "SRAM_SLICER_SHARED_MME_INPUT_PRODUCER_EXPANSION_ENABLED",
    "Enable TPC -> MME with shared input data sharing through SRAM when possible",
    true,
    MakePrivate,
    {&makeFastCompilationObserver});

GlobalConfBool GCFG_SRAM_SLICER_SHARED_MME_INPUT_CONSUMER_EXPANSION_ENABLED(
    "SRAM_SLICER_SHARED_MME_INPUT_CONSUMER_EXPANSION_ENABLED",
    "Enable MME with shared input -> TPC data sharing through SRAM when possible",
    true,
    MakePrivate,
    {&makeFastCompilationObserver});

GlobalConfUint64 GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT(
    "MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT",
    "Ignore the request of scalar pipe for SRAM below this threshold (no performance gain, since after the first load the data is cached)",
    0,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_COMPLEX_GUID_BUNDLES(
    "ENABLE_COMPLEX_GUID_BUNDLES",
    "Create a bundle for each complex-guid, to prevent RMW users from mixing with SRAM management bundles",
    DfltBool(true),
    MakePrivate);

// instead of choosing the first solver that is effective for the bundle, allow all the effective solvers to try and
// add strategies
GlobalConfBool GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED(
    "SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED",
    "Allow multiple solvers to solve the bundle",
    true,
    MakePrivate,
    {&makeFastCompilationObserver});

GlobalConfBool GCFG_PIPELINE_BUNDLE_EDGES_ENABLED(
    "PIPELINE_BUNDLE_EDGES_ENABLED",
    "Enable Pipelining bundle edges between different physical engines",
    DfltBool(false) << deviceValue(synDeviceGaudi2, true) << deviceValue(synDeviceGaudi3, true),
    MakePrivate); // TODO SW-10788 Currently causes long compilation times in some tests and perf drop.

GlobalConfBool GCFG_ENABLE_BUNDLE_EVICTION_FUSING(
    "ENABLE_BUNDLE_EVICTION_FUSING",
    "Enable fusing of sram eviction operation with bundle node's optional output or through TPC fuser",
    DfltBool(true) << deviceValue(synDeviceGaudi3, false),
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_GRAPH_VISUALIZATION(
    "SRAM_SLICER_GRAPH_VISUALIZATION",
    "Enable creation of graph visualization per bundle during SRAM slicing pass",
    false,
    MakePrivate);

GlobalConfFloat GCFG_SRAM_SLICER_REUSE_LIMIT_FACTOR(
    "SRAM_SLICER_REUSE_LIMIT_FACTOR",
    "Minimal ratio between slice processing and data time. less than 1.0: use fixed limit (disabled).",
    1.1,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED(
    "SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED",
    "Allow 4D slicing of big images on an image dim, for convolution that is not convertible to GEMM",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED(
    "SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED",
    "Allow 4D slicing of big images on an image dim, for dedw that is not convertible to GEMM",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED(
    "SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED",
    "Allow 4D slicing of big images on an image dim, for dedx that is not convertible to GEMM",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_CONV_MULTI_SPATIAL_DIMS_SLICE_ENABLED(
    "SRAM_SLICER_CONV_MULTI_SPATIAL_DIMS_SLICE_ENABLED",
    "Allow spatial slicing of big images on multiple image dimensions",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_DYNAMIC_4D_CONV_SPATIAL_SLICE_ENABLED(
    "SRAM_SLICER_DYNAMIC_4D_CONV_SPATIAL_SLICE_ENABLED",
    "Allow 4D slicing of big images on an image dim, for dynamic convolution that is not convertible to GEMM",
    true);

GlobalConfBool GCFG_SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE(
    "SYN_SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE",
    {"SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE"},
    "Slice BGEMM to multiple GEMMs per slice",
    true,
    MakePrivate);

GlobalConfBool GCFG_SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED(
    "SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED",
    "Allow common dim slicing for GEMM, which has no other slicing solution",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_SRAM_SLICER_COST_MODEL_OVERHEAD_PER_SLICE(
    "SRAM_SLICER_COST_MODEL_OVERHEAD_PER_SLICE",
    "Overhead per slice (in cycles) for strategy cost estimation",
    1500,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING(
    "ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING",
    "Allow in graph im2col (flatten 1x1 4-5D convs to GEMMs)",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING(
    "ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING",
    "Allow in graph im2col (flatten full broadcast BGEMMs to GEMMs)",
    DfltBool(true) << deviceValue(synDeviceGaudi2, false) << deviceValue(synDeviceGaudi3, false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT(
    "ENABLE_SLICER_RESHAPE_ALIGNMENT",
    "Allow reshape to be moved out of a sliced bundle by reshaping TPC operands",
    false,
    MakePrivate);

GlobalConfBool GCFG_IGNORE_INDEX_SPACE_FOR_SLICING(
    "IGNORE_INDEX_SPACE_FOR_SLICING",
    "Allow stitch and slice TPC nodes in a bundle not according to index-space (using whitelist/blacklist) (Gaudi1)",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING(
    "NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING",
    "Recommended number of slices in case of small operands when slicing MME nodes on non common dimensions",
    4,
    MakePrivate);

GlobalConfUint64 GCFG_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING(
    "COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING",
    "Recommended number of slices in case of small operands when slicing MME nodes on common dimensions",
    8,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING(
    "ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING",
    "Match grad A/B pattern where gradB is reshaped and pair them together (by reshaping gradA too) for better pipelining",
    DfltBool(true),
    MakePrivate);

GlobalConfSize GCFG_MIN_SLICE_SIZE_FOR_SRAM_FILLING(
    "MIN_SLICE_SIZE_FOR_SRAM_FILLING",
    "Minimal slice size eligible for fill directive (fetch) from HBM to SRAM. Smaller slices will be read from HBM"
    " regardless of the solver directives.",
    hl_gcfg::SizeParam("256kb"),
    MakePrivate
);

GlobalConfBool GCFG_DISABLE_KERNEL_V2_PRIORITY(
    "DISABLE_KERNEL_V2_PRIORITY",
    "Disable kernel V2 priority over V1",
    false,
    MakePrivate
);

GlobalConfBool GCFG_CODE_GEN_ARM_MON_BEFORE_DESC(
    "CODE_GEN_ARM_MON_BEFORE_DESC",
    "Separate the monitor arm command form the fence and put it before descriptor configuration",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_MME_DESCRIPTORS_CACHE_SIZE(
    "MME_DESCRIPTORS_CACHE_SIZE",
    "Set max size of descriptors cache size. Saving last recently used descriptors in oder to reduce the descriptors generation time",
    128,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_PREDICATED_CMD(
    "ENABLE_TPC_PREDICATED_CMD",
    "Enable TPC predicated commands",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_LAST_DIM_OPT(
    "ENABLE_TPC_LAST_DIM_OPT",
    "Enable TPC last_dim optimization; extends addressing range",
    true,
    MakePrivate);

GlobalConfObserver makeCtrlDepSoftObserver(&GCFG_MAKE_CTRL_DEP_SOFT, {{"false", "false"}, {"0", "false"}}, true);

GlobalConfBool GCFG_CHECK_SECTION_OVERLAP(
    "CHECK_SECTION_OVERLAP_CHECK",
    "check section overlap , Warning: if this flag disabled MAKE_CTRL_DEP_SOFT functionality also disabled",
    true,
    MakePrivate,
    {&makeCtrlDepSoftObserver});

GlobalConfFloat GCFG_NON_BUNDLE_SRAM_ALLOCATION_FACTOR(
    "NON_BUNDLE_SRAM_ALLOCATION_FACTOR",
    "Factor to allocate non bundle tensors in SRAM - factor from max live capacity. <1 for disable ",
    2.0f,
    MakePrivate);

GlobalConfSize GCFG_MIN_SRAM_SIZE_FOR_NON_BUNDLE_TENSORS(
    "MIN_SRAM_SIZE_FOR_NON_BUNDLE_TENSORS",
    "Minimal SRAM size to allocate the non-bundle tensors (to reduce memory reuse and data dependencies)",
    hl_gcfg::SizeParam("128KB"),
    MakePrivate);

GlobalConfUint64 GCFG_RECIPE_CACHE_SIZE{
    "RECIPE_CACHE_SIZE",
    "Size of device HBM allocated for recipe caching in KB",
    500 * 1024,
    MakePrivate};

GlobalConfUint64 GCFG_RECIPE_CACHE_BLOCK_SIZE {
    "RECIPE_CACHE_BLOCK_SIZE",
    "Device recipe cache block size in KB, needs to be "
    "multiplication of patchable-blobs' static chunks-size (128)",
    128,
    MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_REDUCE_FACTOR{
    "STREAM_COMPUTE_REDUCE_FACTOR",
    "Compute stream reduce factor. Value of 0 is illegal value of 1 is no reduction",
    4,
    MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP{
    "STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP",
    "Size of a single upper CP data chunk in cache, in KB",
    128,
    MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP{
    "STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP",
    "Stream compute upper CP maximum number of free data chunks in cache",
    8192,
    MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP{
    "STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP",
    "Size of a single lower CP data chunk in cache, in KB",
    256,
    MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP{
    "STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP",
    "Stream compute lower CP maximum number of free data chunks in cache",
    1024,
    MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_ARC_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP{
        "STREAM_COMPUTE_ARC_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP",
        "ARC size of a single lower CP data chunk in cache, in KB",
        256,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP{
        "STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP",
        "ARC stream compute lower CP minimum number of free data chunks in cache",
        512,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_SMALL{
        "STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_SMALL",
        "Stream copy up small maximum number of free data chunks in cache",
        2048,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM{
        "STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM",
        "Stream copy up medium maximum number of free data chunks in cache",
        1536,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_LARGE{
        "STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_LARGE",
        "Stream copy up large maximum number of free data chunks in cache",
        8,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_SMALL{
        "STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_SMALL",
        "Stream copy down synapse small maximum number of free data chunks in cache",
        2048,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM{
        "STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM",
        "Stream copy down synapse medium maximum number of free data chunks in cache",
        1536,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_SMALL{
        "STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_SMALL",
        "Stream copy down small maximum number of free data chunks in cache",
        2048,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM{
        "STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM",
        "Stream copy down medium maximum number of free data chunks in cache",
        1536,
        MakePrivate};

GlobalConfUint64 GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_LARGE{
        "STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_LARGE",
        "Stream copy down large maximum number of free data chunks in cache",
        8,
        MakePrivate};

GlobalConfUint64 GCFG_DATACHUNK_LOOP_NUM_RETRIES{
    "DATACHUNK_LOOP_NUM_RETRIES",
    "Maximum number of retries of acquireDataChunksAmount",
    75,
    MakePrivate};

GlobalConfUint64 GCFG_MAX_RMW_TENSOR_BYTES(
    "MAX_RMW_TENSOR_BYTES",
    "Gaudi, max tensor capacity when using it for read modify write operation on SRAM",
    DfltUint64(16ul * 1024) << deviceValue(synDeviceGaudi2, 32ul * 1024),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_ICACHE_PREFETCH(
    "ENABLE_TPC_ICACHE_PREFETCH",
    "Enable TPC I-cache prefetch | Gaudi",
    DfltBool(true) << deviceValue(synDeviceGaudi, false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_DCACHE_PREFETCH(
    "ENABLE_TPC_DCACHE_PREFETCH",
    "Enable TPC D-cache prefetch",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION(
    "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION",
    "Enable TPC tensors shape manipulation to improve performance",
    DfltBool(true),
    MakePrivate);

GlobalConfFloat GCFG_TRANSPOSE_SPLITTING_THRESHOLD(
    "TRANSPOSE_SPLITTING_THRESHOLD",
    "The improvement threshold that needed to replace single transpose "
    "with the sequence, (0.0 - always replace, 1.0 - disable splitter)",
    DfltFloat(1.0) << deviceValue(synDeviceGaudi, 0.1) << deviceValue(synDeviceGaudi2, 0.1),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TWO_DIM_TRANSPOSE_RESHAPER(
    "ENABLE_TWO_DIM_TRANSPOSE_RESHAPER",
    "Enable the transpose reshaper that try to reshape 2d transpose into 3d transpose to improve utilizaztion",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_STRIDED_TENSOR_SHAPE_MANIPULATION(
    "ENABLE_TPC_STRIDED_TENSOR_SHAPE_MANIPULATION",
    "Enable TPC shape manipulation on strided tensors | Gaudi",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SLICE_NORM_BUNDLING(
    "ENABLE_SLICE_NORM_BUNDLING",
    "Enable bundling of slice and norm nodes",
    true,
    MakePrivate);

GlobalConfFloat GCFG_MAX_TPC_VEC_UTIL_FOR_STRIDED_RESHAPE_MANIPULATION(
    "MAX_TPC_VEC_UTIL_FOR_STRIDED_RESHAPE_MANIPULATION",
    "Reshape manipulation on strided tensors is applied only when the tpc vector utilization is lower then this threshold. values should be between 0-1 | Gaudi",
    true,
    MakePrivate);

GlobalConfFloat GCFG_MAX_TPC_VEC_UTIL_FOR_BROADCAST_TPC_FUSION(
    "MAX_TPC_VEC_UTIL_FOR_BROADCAST_TPC_FUSION",
    "Fusion of broadcast producer and tpc broadcastable consumer is applied only when the tpc vector utilization is lower then this threshold. values should be between 0-1",
    0.9f,
    MakePrivate);

GlobalConfUint64 GCFG_NUM_OF_CSDC_TO_CHK(
    "GCFG_NUM_OF_CSDC_TO_CHK",
    "Number of csDataChunks to check (over 1) during first loop",
    9,
    MakePrivate);

GlobalConfUint64 GCFG_WCM_REPORT_AMOUNT(
        "WCM_REPORT_AMOUNT",
        "WCM report threshold based on amount of queries per physical stream (0 for no report)",
        10000,
        MakePrivate);

GlobalConfUint64 GCFG_WCM_QUERIER_REPORT_AMOUNT(
        "WCM_QUERIER_REPORT_AMOUNT",
        "WCM querier report threshold based on amount of queries per device (0 for no report)",
        10000,
        MakePrivate);

GlobalConfBool GCFG_PROTECT_UNSAFE_DMA_TRANSPOSE(
    "PROTECT_DMA_TRANSPOSE",
    "Set EB on every commit of DMA transpose operations. Temporary bug work-around",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_SRAM_SIZE_RESERVED_FOR_HCL(
    "HCL_SRAM_SIZE",
    "The size of SRAM reserved for HCL reduction use",
    320 * 1024, // Gaudi1 and Gaudi2 (not relevant for G3)
    MakePrivate);

GlobalConfBool GCFG_GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS(
    "GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS",
    "Adds Serialize/Deserialize pass for dynamic shapes graphs",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION(
    "ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION",
    "Enable AggregateFcdWithStaticReshape optimization",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_GCONV_PACKING(
    "ENABLE_GCONV_PACKING",
    "Enable packing of weights for grouped convolution optimization",
    true);

GlobalConfBool GCFG_ENABLE_GCONV_SPLIT_NODES_REUSE(
    "ENABLE_GCONV_SPLIT_NODES_REUSE",
    "This flag indicates whether several grouped convolution nodes can share the same split node (for example: dedx + dedw)",
    true,
    MakePrivate);

GlobalConfBool GCFG_USE_DMA_IN_PHYSICAL_RESHAPE(
    "USE_DMA_IN_PHYSICAL_RESHAPE",
    "This flag indicates whether to use DMA or TPC for physical reshape. Default to TPC.",
    false,
    MakePrivate);

GlobalConfBool GCFG_VALIDATE_MEMORY_SECTION_TENSORS(
    "VALIDATE_MEMORY_SECTION_TENSORS",
    "This flag controls validation pass to make sure node order is well defined if tensors memory overlaps",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CSE_OPTIMIZATION(
    "ENABLE_CSE_OPTIMIZATION",
    "This flag enables the common sub-expresseion elimination pass",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_HANDLE_MEMORY_COHERENCE(
    "HANDLE_MEMORY_COHERENCE",
    "This flag controls the enforcment of memory coherence - restricting the order of writing of overlapping persistent tensors",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_SCRATCHPAD_SRAM_MAX_SIZE(
    "SCRATCHPAD_SRAM_MAX_SIZE",
    "This flag controls the max size for auxTensor without data to be in SRAM",
    10 * 1024 * 1024,
    MakePrivate);

// In case equals zero => Disabled
// Otherwise, the event-fd thead will be the smaller of the following two GCFGs
GlobalConfUint64 GCFG_EVENT_FD_THREAD_DEBUG_INTERVAL(
    "EVENT_FD_THREAD_DEBUG_INTERVAL",
    "Debug Event-FD operation time-interval (in milliseconds)",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_EVENT_FD_THREAD_SLEEP_TIME(
    "EVENT_FD_THREAD_SLEEP_TIME",
    "EVENT_FD single thread time to sleep between iterations in milliseconds",
    200,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_INTERNAL_NODES(
    "ENABLE_INTERNAL_NODES",
    "Allows creation of internal nodes through synapse api",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_DRAM_MULTI_BUFFERING(
    "ENABLE_DRAM_MULTI_BUFFERING",
    "Use multi buffering for DRAM allocation",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SRAM_MULTI_BUFFERING(
    "ENABLE_SRAM_MULTI_BUFFERING",
    "Use multi buffering for SRAM allocation",
    true,
    MakePrivate);
GlobalConfBool GCFG_GAUDI_MEMSET_HW_WA(
        "GAUDI_MEMSET_HW_WA",
        "Operate memset as a memcpy operation from zero cache line",
        DfltBool(false) << deviceValue(synDeviceGaudi, true),
        MakePrivate);

GlobalConfBool GCFG_SKIP_BN_SPLIT(
    "SKIP_BN_SPLIT",
    "Whether to skip the optimization of splitting batch normalization kernel into 2 step kernels",
    false,
    MakePrivate);

GlobalConfBool GCFG_SKIP_BN_SPLIT_FOR_IN_REDUCED_TO_BN(
    "SKIP_BN_SPLIT_FOR_IN_REDUCED_TO_BN",
    "Whether to skip batch norm split into 2 stage kernels for instance-norm reduced to multiple batch-norms with B=1",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_BATCH_NORM_MEMCPY_FUSION(
    "ENABLE_BATCH_NORM_MEMCPY_FUSION",
    "Whether to enable using batch norm stageX kernel's optional output for SRAM eviction",
    false,
    MakePrivate);

GlobalConfBool GCFG_SKIP_LAYER_NORM_BWD_SPLIT(
    "GCFG_SKIP_LAYER_NORM_BWD_SPLIT",
    "Whether to skip the optimization of splitting layer normalization bwd kernel into 2 step kernels",
     DfltBool(false)  << deviceValue(synDeviceGaudi2, true),
     MakePrivate);


GlobalConfBool GCFG_ENABLE_LAYER_NORM_BWD_EXPERIMENTAL_SPLIT(
    "ENABLE_LAYER_NORM_BWD_EXPERIMENTAL_SPLIT",
    "Experimental split method for layer_norm_bwd. Should provide better pipelining.",
    true,
    MakePrivate);

GlobalConfBool GCFG_SB_REUSE(
    "SB_REUSE",
    "Is MME SB reuse allowed",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_MME_PARTIALS_TO_MEMORY(
    "MME_PARTIALS_TO_MEMORY",
    "Enable MME stack to perform reduction to memory instead of accumulating internally when beneficial",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_DEDW_UNROLL(
    "DEDW_UNROLL",
    "Is MME batch concurrency allowed (prevoiusly called dedw unroll)",
    DfltBool(true),
    MakePrivate);

GlobalConfUint64 GCFG_ENABLED_COMPRESSION_MODES(
    "ENABLED_COMPRESSION_MODES",
    "The compression modes available: 0 = None/Serialization only, 1 = Huffman only, 2 = Sparsity only, 3 = Huffman +sparsity",
    0,
    MakePrivate);
GlobalConfBool GCFG_MME_PROFILE_PER_DESC(
    "MME_PROFILE_PER_DESC",
    "improve the profiler resolution to per desc instead of per layer",
    false,
    MakePrivate);

GlobalConfBool GCFG_MME_ADVANCED_PROFILE(
    "MME_ADVANCED_PROFILE",
    "further improve the profiler resolution to display MME events per engine",
    false,
    MakePrivate);

GlobalConfBool GCFG_PARSE_EACH_COMPUTE_CS(
    "PARSE_EACH_COMPUTE_CS",
    "Enable parsing of each Compute CS",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_INT16_LIMITED_BITS(
    "INT16_LIMITED_BITS",
    "Set number of bits for int16ltd type: 1-16. default: 12",
    12,
    MakePublic);

GlobalConfBool
    GCFG_MME_STRATEGY_ALIGNED_ADDRESSES_ENABLED("MME_STRATEGY_ALIGNED_ADDRESSES_ENABLED",
                                                "MME strategy aligned address has a small performance impact and an "
                                                "unknown power improvemnt, disable it for now (see SW-22388).",
                                                DfltBool(true) << deviceValue(synDeviceGaudi, false),
                                                MakePrivate);

GlobalConfBool GCFG_MME_PARTIAL_STRATEGY_SETTING_ENABLED(
    "MME_PARTIAL_STRATEGY_SETTING_ENABLED",
    "When enabled, when MME params are created for the pipeline manager, set only "
    "geometry and pattern in the strategy. When disabled, all fields are set.",
    true,
    MakePrivate);

GlobalConfBool GCFG_GAUDI_DYNAMIC_SHAPE_VALIDATION_PASS_ENABLED(
    "GAUDI_DYNAMIC_SHAPE_VALIDATION_PASS_ENABLED",
    "Enable dynamic shape validation pass. Should be disabled for GC tests that create currently-unsupported by run-time 3D reshape node",
    true,
    MakePrivate);

GlobalConfBool GCFG_DISABLE_BASE_REGISTERS_CACHE(
    "DISABLE_BASE_REGISTERS_CACHE",
    "For platforms that support base registers cache, disable the use of the cache",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_MEMSET_PARALLEL_LEVEL(
    "MEMSET_PARALLEL_LEVEL",
    "Whether to run memset and memcpy on different engines and set memset parallel level (0 - disabled)",
    DfltUint64(0)
    << deviceValue(synDeviceGaudi, 1)
    << deviceValue(synDeviceGaudi2, 0)
    << deviceValue(synDeviceGaudi3, 0),
    MakePrivate);

GlobalConfBool GCFG_RUNTIME_DUMP_RECIPE(
        "RUNTIME_DUMP_RECIPE",
        "Dump the recipe to log file before parsing it",
        false,
        MakePrivate);

GlobalConfBool GCFG_RUNTIME_SKIP_RECIPE_VALIDATION(
        "RUNTIME_SKIP_RECIPE_VALIDATION",
        "skip runtime recipe valikdation",
        false,
        MakePrivate);

GlobalConfUint64 GCFG_DMA_CHUNK_SIZE(
    "DMA_CHUNK_SIZE",
    "The maximal size per DMA descriptor",
    DfltUint64(16 * 1024)
    << deviceValue(synDeviceGaudi2, 128ull * 1024)
    << deviceValue(synDeviceGaudi3, 128ull * 1024),
    MakePrivate);

GlobalConfString GCFG_DUMP_PRE_GRAPHS(
    "DUMP_PRE_GRAPHS",
    "File/Folder path to dump pre compilation graph/s to. If the path points to an existing folder, each graph is written to different file by"
    "the recipe name. If the path points to a file it is created and all the graphs are written into it",
    std::string(),
    MakePublic);

GlobalConfString GCFG_DUMP_POST_GRAPHS(
    "DUMP_POST_GRAPHS",
    "File/Folder path to dump post compilation graph/s to. If the path points to an existing folder, each graph is written to different file by"
    "the recipe name. If the path points to a file it is created and all the graphs are written into it",
    std::string(),
    MakePublic);

GlobalConfString GCFG_DUMP_PASSES_GRAPHS(
    "DUMP_PASSES_GRAPHS",
    "Folder path to dump intermidiate graphs to. Each graph is written to different file by the recipe name following the last executed compilation pass.",
    std::string(),
    MakePrivate);

GlobalConfString GCFG_DUMP_PASSES_FILTER(
    "DUMP_PASSES_FILTER",
    "Dump only the specified pass.",
    std::string(),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_IN_GRAPH_BROADCAST_FOR_BGEMM(
    "ENABLE_IN_GRAPH_BROADCAST_FOR_BGEMM",
    "Enable planting broadcast node when broadcast bgemm is found and not supported natively",
    DfltBool(true) << deviceValue(synDeviceGaudi2, false) << deviceValue(synDeviceGaudi3, false),
    MakePrivate);

GlobalConfBool GCFG_ADD_EXPLICIT_BROADCAST_FOR_BGEMM(
    "ADD_EXPLICIT_BROADCAST_FOR_BGEMM",
    "Always turn a batch gemm into a symmetric one, even when it is Full broadcast.",
    DfltBool(false),
    MakePrivate);

GlobalConfUint64 GCFG_RUN_MEMSET_ON_DMA_THRESHOLD(
        "RUN_MEMSET_ON_DMA_THRESHOLD",
        "Above this value memset run on TPC.",
        DfltUint64 (std::numeric_limits<uint64_t>::max())
                    << deviceValue(synDeviceGaudi, std::numeric_limits<uint64_t>::max())
                    << deviceValue(synDeviceGaudi2, 0x6400000), //100MB
                    MakePrivate);

GlobalConfBool GCFG_ENABLE_PIPELINE_MANAGEMENT(
        "ENABLE_PIPELINE_MANAGEMENT",
        "Enable pipeline management algorithm for graph bundling, slicing and SRAM placing",
        DfltBool(true) << deviceValue(synDeviceGaudi, false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_SLICER_SILENT_FAILURE(
        "ENABLE_SLICER_SILENT_FAILURE",
        "Avoid failing compilation on slicer error",
        DfltBool(false),
        MakePrivate);

GlobalConfBool GCFG_ALIGN_BATCH_GEMM_DIMS(
        "ALIGN_BATCH_GEMM_DIMS",
        "Align batch-gemm batch ranks",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_DIRECT_CYCLIC_RANGE_CALC(
        "ENABLE_DIRECT_CYCLIC_RANGE_CALC",
        "use new direct method for calculating cyclic ranges",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_ATOMIC_NODES_VALIDATION("ENABLE_ATOMIC_NODES_VALIDATION",
                                                   "when this is enabled, atomic nodes validation is activated",
                                                   true,
                                                   MakePrivate);

GlobalConfBool GCFG_ENABLE_OPTIMIZE_STRIDED_INSERT(
        "ENABLE_OPTIMIZE_STRIDED_INSERT",
        "when this is enabled, the optimization pass 'optimizeStridedInsert will run'",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_INTERMEDIATE_TENSOR_PERMUTATION(
        "ENABLE_INTERMEDIATE_TENSOR_PERMUTATION",
        "when this is enabled, intermediate persistent tensors with allowPermutation may be permuted",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_LFSR_KERNEL_SHAPE_MANIPULATION(
        "ENABLE_LFSR_KERNEL_SHAPE_MANIPULATION",
        "when this is disabled, shape manipulation will not be allowed on kernels using Linear-feedback shift registers as random generators",
        true,
        MakePrivate);

GlobalConfBool GCFG_INTERNAL_TEST(
        "INTERNAL_TEST",
        "signal to GC that we are not running above API, refrain from certain assumptions on the state",
        false,
        MakePrivate);

GlobalConfBool GCFG_DISABLE_DOUBLE_SYN_INITIALIZE(
        "HABANA_DISABLE_DOUBLE_SYN_INITIALIZE",
        "disable double synInitialize to succeed. if ON the second synInitialize returns synAlreadyInitialized",
        false,
        MakePublic);

GlobalConfBool GCFG_DISABLE_DOUBLE_SYN_DESTROY(
        "HABANA_DISABLE_DOUBLE_SYN_DESTROY",
        "disable double synDestroy to succeed. if ON the second synDestroy returns synUninitialized",
        true,
        MakePublic);

GlobalConfBool GCFG_FUSE_CAST_TO_MME(
        "FUSE_CAST_TO_MME",
        "Fuse cast nodes to mme if it's supported",
        true,
        MakePrivate);

GlobalConfBool GCFG_FUSE_CONVERT_TO_MME(
        "FUSE_CONVERT_TO_MME",
        "Fuse convert nodes to mme if it's supported",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_CONTIGUOUS_CAST_REMOVAL(
        "ENABLE_CONTIGUOUS_CAST_REMOVAL",
        "Remove redundant cast nodes to improve performance",
        true,
        MakePrivate);

GlobalConfObserver fuseCastToMMEObserver(&GCFG_FUSE_CAST_TO_MME, {{"true", "false"}, {"1", "false"}}, true);
GlobalConfObserver contiguousCastRemovalObserver(&GCFG_ENABLE_CONTIGUOUS_CAST_REMOVAL, {{"true", "false"}, {"1", "false"}}, true);

GlobalConfBool GCFG_KEEP_NUMERICS(
        "KEEP_NUMERICS",
        "Disable cast elimination passes to keep the numerics identical to the original graph",
        false,
        MakePrivate,
        {&fuseCastToMMEObserver, &contiguousCastRemovalObserver});

GlobalConfBool GCFG_RECOVER_INCOMPATIBLE_DATA_TYPE(
        "RECOVER_INCOMPATIBLE_DATA_TYPE",
        "Allow adding cast for TPC incompatible type error",
        true,
        MakePrivate);

GlobalConfSize GCFG_ELF_BUFFER_INITIAL_SIZE(
        "ELF_BUFFER_INITIAL_SIZE",
        "Set TPC kernel elf buffer initial size",
        hl_gcfg::SizeParam("64KB"),
        MakePrivate);

GlobalConfBool GCFG_ETL_DISABLE(
        "ETL_DISABLE",
        "Disable ETL feature (only SPDL will be used)",
        false,
        MakePrivate);

GlobalConfBool GCFG_ARC_ARCHITECTURE(
        "ARC_ARCHITECTURE",
        "Indicating if the platform is ARC-based architecture",
        DfltBool(true) << deviceValue(synDeviceGaudi, false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_MULTI_SIF(
    "ENABLE_MULTI_SIF",
    "Enable fused nodes with multiple SIFs taken from the pre-fusion nodes",
    true,
    MakePrivate);

GlobalConfUint64 GCFG_TERMINATE_SYNAPSE_UPON_DFA(
    "TERMINATE_SYNAPSE_UPON_DFA",
    "Synapse termination upon EDF's device-reset event [0-Disable, 1-Terminate (busy-wait and block APIs), 2-Kill (use SIGKILL)]",
    2,
    MakePrivate);

GlobalConfUint64 GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE(
    "RT_SCAL_DEBUG_MODE",
    "RT's SCAL debug mode",
    0,      // Bitwise value
    MakePrivate);

GlobalConfBool GCFG_PRESERVE_TESTS_RECIPE(
    "PRESERVE_TESTS_RECIPE",
    "Preserves the test's recipe file in a file with a similar name",
    false,
    MakePrivate);

GlobalConfBool GCFG_ALLOW_DISABLED_CG(
    "ALLOW_DISABLED_CG",
    "Allow running without complex guid",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CYCLIC_RANGES(
    "ENABLE_CYCLIC_RANGES",
    "enables usage of cyclic ranges in overlap mechanism",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_NAN_INF_PROBE(
        "ENABLE_NAN_INF_PROBE",
        "insert a kernel after each op to be stuck if there are infs or nans",
        false,
        MakePrivate);

GlobalConfBool GCFG_DISABLE_ZST_SUPPORT(
    "DISABLE_ZST_SUPPORT",
    "Disables the zero-sized tensors removal mechanism",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_DOUBLE_TRANSPOSE(
    "ENABLE_DOUBLE_TRANSPOSE",
    "Enable double transpose -> so that one physical transpose can be configured as two",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SINGLE_TRANSPOSE_DYNAMIC_PRIORITY(
    "ENABLE_SINGLE_TRANSPOSE_DYNAMIC_PRIORITY",
    "prefer single physical transpose over dual when benificial",
    DfltBool(false) << deviceValue(synDeviceGaudi, true) << deviceValue(synDeviceGaudi2, false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_PREFER_GENERIC_DMA_OPT(
    "ENABLE_PREFER_GENERIC_DMA_OPT",
    "prefer generic dma transpose strategy over double and fully utilized strategies",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_MAKE_BROADCAST_PHYSICAL(
    "MAKE_BROADCAST_PHYSICAL",
    "Make concat as physical concat",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SLICE_FCD_OPTIMIZATION(
    "ENABLE_SLICE_FCD_OPTIMIZATION",
    "Enable optimization slice fcd with opposite shift transpose optimization",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_OPTIMIZE_SPLIT_CONCAT_ON_FCD(
    "OPTIMIZE_SPLIT_CONCAT_ON_FCD",
    "Add high performance transpose sequence before and after the operation to avoid low utilization DMAs",
    DfltBool(false) << deviceValue(synDeviceGaudi, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONTIGUOUS_TRANSPOSE_REMOVAL(
    "ENABLE_CONTIGUOUS_TRANSPOSE_REMOVAL",
    "Remove redundant transpose nodes to improve performance",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_OPPOSITE_CONCAT_SPLIT_REMOVAL(
    "ENABLE_OPPOSITE_CONCAT_SPLIT_REMOVAL",
    "Remove opposite concat and split nodes to improve performance",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_FUSING_CONTIGUOUS_TRANSPOSE_NODES(
    "ENABLE_FUSING_CONTIGUOUS_TRANSPOSE_NODES",
    "Fuse contiguous transpose nodes",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_HUGE_TENSOR_SLICING(
    "ENABLE_HUGE_TENSOR_SLICING",
    "Handling of huge tensors, and slicing them at pre compilation stage",
    DfltBool(true) << deviceValue(synDeviceGaudi, false),
    MakePrivate);

GlobalConfString GCFG_SCAL_CONFIG_FILE_PATH(
    "SCAL_CONFIG_FILE_PATH",
    "File path of scal configuration json",
    std::string(),
    MakePublic);

GlobalConfBool GCFG_OLD_TENSOR_CREATION_API(
    "OLD_TENSOR_CREATION_API",
    "Still using the old tensor creation API, since, old tests, or something. Please update your tests on your free time",
    false,
    MakePrivate);

GlobalConfBool GCFG_RUNNING_ON_PLDM(
        "RUNNING_ON_PLDM",
        "Relevant for gaudi2/gaudi3 to enable/disable the rotator engine accordingly",
        false,
        MakePrivate);

GlobalConfBool GCFG_GAUDI2_FORCE_MME_FP32_IEEE(
        "GAUDI2_FORCE_MME_FP32_IEEE",
        "(gaudi2 only) - forcing MME FP32 computation to be IEEE",
        false,
        MakePrivate);

GlobalConfBool GCFG_MME_ENABLE_COMPARE_NEW_VS_OLD_DESCRIPTORS(
        "MME_ENABLE_COMPARE_NEW_VS_OLD_DESCRIPTORS",
        "Compare MME new to legacy descriptors",
        false,
        MakePrivate);

GlobalConfBool GCFG_MME_ENABLE_USE_OLD_DESCRIPTORS(
        "MME_ENABLE_USE_OLD_DESCRIPTORS",
        "Use MME old descriptors",
        false,
        MakePrivate);

GlobalConfBool GCFG_PRINT_MME_DESCRIPTORS(
        "PRINT_MME_DESCRIPTORS",
        "print MME descriptors under MME STACK",
        false,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_MME_CD_CONCURRENCY(
        "ENABLE_MME_CD_CONCURRENCY",
        "Enable common dim concurrency",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_DISABLE_MME_CD_CONCURRENCY_FOR_DSD(
        "DISABLE_MME_CD_CONCURRENCY_FOR_DSD",
        "Disable common dim concurrency in dynamic graphs",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_MME_CD_CONCURRENCY_REDUCTION_BF16_FP16(
        "ENABLE_MME_CD_CONCURRENCY_REDUCTION_BF16_FP16",
        "Enable common dim concurrency in case of adding reduction and memset when the output is bf16 or fp16",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_RELAXED_IGNORE_IN_SRAM_CAP_CALC(
        "ENABLE_RELAXED_IGNORE_IN_SRAM_CAP_CALC",
        "Enable relaxed heuristic on whether to ignore a node in sram capacity calculation",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_FORCE_MME_CD_CONCURRENCY_NON_DETERMINISTIC(
        "FORCE_MME_CD_CONCURRENCY_NON_DETERMINISTIC",
        "override determinism requirement (to enable testing cd concurrency)",
        false,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_BUNDLE_TRANSPOSE(
        "ENABLE_BUNDLE_TRANSPOSE",
        "Toggle pipeline mangement bundlizing of logical transpose nodes",
        true,
        MakePrivate);

GlobalConfUint64 GCFG_BGEMM_PRODUCER_CHAIN_MAX_SIZE_FOR_FUSING_WITH_BROADCAST(
        "BGEMM_PRODUCER_CHAIN_MAX_SIZE_FOR_FUSING_WITH_BROADCAST",
        "Max distance between broadcast and bgemm that the broadcast is still considered fuseable",
        5,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_FUSE_BROADCAST_BGEMM(
        "ENABLE_FUSE_BROADCAST_BGEMM",
        "Enable fusion of broadcast and bgemm",
        DfltBool(true) << deviceValue(synDeviceGaudi, false) << deviceValue(synDeviceGaudi3, false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_FUSE_BATCH_NORM(
        "ENABLE_FUSE_BATCH_NORM",
        "enable GC fusion pass of batch norm patterns",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN(
        "ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN",
        "Enable pipeline mangement placing the shared operand producers chain in a shared multi buffer",
        false,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_SLICING_BOTH_PRODUCER_CHAINS(
        "ENABLE_SLICING_BOTH_PRODUCER_CHAINS",
        "Enable slicing of the non master producer chain if all slices can be placed concurrently in SRAM",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_SLICING_ON_MORE_THAN_ONE_DIM(
        "ENABLE_SLICING_ON_MORE_THAN_ONE_DIM",
        "Enable slicing the master operand on more than one dim.",
        true,
        MakePrivate);

GlobalConfUint64 GCFG_PIPELINE_MANAGEMENT_NUM_DIMS_TO_SLICE(
        "NUM_DIMS_TO_SLICE",
        "Number of dimentions to slice when slicing on more than one dim",
        2,
        MakePrivate);

GlobalConfUint64 GCFG_MIN_CYCLES_FOR_MME_SLICING(
        "MIN_CYCLES_FOR_MME_SLICING",
        "Minimal number of compute cycles required for slicing an mme node",
        0,
        MakePrivate);

GlobalConfBool
    GCFG_ENABLE_MME_CONV_LOWERING("ENABLE_MME_CONV_LOWERING", "enable mme convolution lowering", true, MakePrivate);

GlobalConfBool GCFG_ENABLE_MME_ALIGN_OPT(
        "ENABLE_MME_ALIGN_OPT",
        "enable mme alignment optimization",
        DfltBool(true),
        MakePrivate);

GlobalConfUint64 GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER(
        "GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER",
        "Set max allowed multi consumers tensors in TPC FUSER cluster",
        0,
        MakePrivate);

GlobalConfBool GCFG_ALLOW_DUPLICATE_KERNELS(
        "ALLOW_DUPLICATE_KERNELS",
        "Allow multiple kernels with the same guid (for GC test)",
        true,
        MakePrivate);

GlobalConfUint64 GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER(
        "NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER",
        "Enable canary protection (header and footer) for data chunks, means number of DCs per canary header, 0 for none",
        0,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_SIF_FOR_STATIC_NODES(
        "ENABLE_SIF_FOR_STATIC_NODES",
        "Enable running SIF for static nodes in addition to dynamic nodes",
        false,
        MakePrivate);

GlobalConfBool GCFG_DUMP_TPC_NODES_DATA_TO_JSON(
    "DUMP_TPC_NODES_DATA_TO_JSON",
    "Dump TPC nodes data into json file specified in TPC_NODES_JSON_FILE",
    false,
    MakePrivate);

GlobalConfBool GCFG_DUMP_TPC_COST_MODEL_DATA(
    "DUMP_TPC_COST_MODEL_DATA",
    "Dump TPC cost model data into post graph json file (requires tpc kernels to be compiled with --kernel-cost)",
    false,
    MakePrivate);

GlobalConfBool GCFG_SPILL_PERSISTENT_TENSORS(
    "SPILL_PERSISTENT_TENSORS",
    "Spill persistent tensors from graph execution",
    true,
    MakePrivate);

GlobalConfString GCFG_TPC_NODES_JSON_FILE(
    "TPC_NODES_JSON_FILE",
    "Json file name with full path for TPC nodes info dump",
    std::string("tpc_nodes_dump.json"),
    MakePrivate);

GlobalConfBool GCFG_GAUDI3_SINGLE_DIE_CHIP(
        "GAUDI3_SINGLE_DIE_CHIP",
        "This flag indicates if Gaudi3 operates in a single die mode",
        false,
        MakePrivate);

GlobalConfBool GCFG_GAUDI3_PLDM_FULL_CACHE_CHIP(
        "GAUDI3_PLDM_FULL_CACHE_CHIP",
        "This flag indicates if Gaudi3 operates in PLDM single die full cache mode",
        false,
        MakePrivate);

GlobalConfBool GCFG_INIT_HCCL_ON_ACQUIRE(
        "INIT_HCCL_ON_ACQUIRE",
        "run HCCL device init during device acquire",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_MME_ENABLE_STOCHASTIC_ROUNDING(
        "MME_ENABLE_STOCHASTIC_ROUNDING",
        "enable stochastic rounding in the mme",
        false,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_PARALLEL_COMPILATION(
        "ENABLE_PARALLEL_COMPILATION",
        "Enable parallel compilation in multi threaded environemnt",
        true,
        MakePrivate);

GlobalConfBool GCFG_LOG_LAUNCH_INFO_UPON_FAILURE(
        "LOG_LAUNCH_INFO_UPON_FAILURE",
        "Log launch' tensors-info and serialize recipe upon launch-failure",
        true,
        MakePrivate);

GlobalConfUint64 GCFG_DFA_READ_REG_MODE(
    "DFA_READ_REG_MODE",
    "DFA Read registers mode: 0=LKD (if supported) 1=Skip",
    0,
    MakePrivate);

GlobalConfBool GCFG_DFA_COLLECT_CCB(
    "DFA_COLLECT_CCB",
    "DFA: Enalbe collection of CCBs from all streams",
    DfltBool(false) << deviceValue(synDeviceGaudi3, true),
    MakePrivate);

// TODO [SW-89250] remove global configuration
GlobalConfBool GCFG_ALLOW_PERMUTATION_ON_USER_TRANSPOSE(
        "ALLOW_PERMUTATION_ON_USER_TRANSPOSE",
        "Allow convert user transpose to logical transpose and use the allow permutation flag",
        DfltBool(false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_DYNAMIC_SHAPE_IN_HIGH_DIMENSION(
        "ENABLE_DYNAMIC_SHAPE_IN_HIGH_DIMENSION",
        "Enable a dynamic shaped tensor in a higher dimension than 5",
        DfltBool(false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_TPC_SHUFFLE_INDEX(
        "ENABLE_TPC_SHUFFLE_INDEX",
        "enable tpc distribution work distribution shuffle index",
        true,
        MakePrivate);

GlobalConfUint64 GCFG_EDMA_NUM_BINNED(
        "EDMA_NUM_BINNED",
        "Number of EDMA engines to bin off (currently affecting only Gaudi2)",
        0,
        MakePrivate);

GlobalConfUint64 GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT(
        "ARC_SYNC_SCHEME_SIGNAL_LIMIT",
        "The max signal value for an engine before going into the SOB Reset Procedure in the middle of the graph.",
        // In gaudi2 we use 2 SOB sets so we are allowed to reach twice the number 0x7fff (SOB has 15 bits)
        // In gaudi3 we use 1 SOB set so we are allowed to reach only once the number 0x7fff
        // In both cases we reduce 5 additional signals just to stay away from the border
        DfltUint64(0) << deviceValue(synDeviceGaudi2, ((2 * 0x7fff) - 5)) << deviceValue(synDeviceGaudi3, 0x7fff - 5),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_ARBITRARY_SCALE(
    "ENABLE_ARBITRARY_SCALE",
    "set mode for mode for arbitrary per-tensor scale for fp8 quantization, this mode doesn't calculate an exponent bias.",
    false,
    MakePrivate);

GlobalConfUint64 GCFG_PIPELINE_MANAGEMENT_FORCE_BUNDLIZER(
    "PIPELINE_MANAGEMENT_FORCE_BUNDLIZER",
    "Force bundling policy: 0 - default behaviour, 1 - Force BUNDLE_BY_VISION_PATTERNS, 2 - Force BUNDLE_BY_TRANSFORMER_PATTERNS",
    0,
    MakePrivate);

GlobalConfUint64 GCFG_CYCLE_PRINTING_LEVEL(
        "CYCLE_PRINTING_LEVEL",
        "0 - Don't print cycles, 1 - Print if lemon finds a cycle (getExeSortedNodes), 2 - Check and print cycles every time a node is added to the graph (increase compilation time)",
        0,
        MakePrivate);

///////////////////////////////////////////////////////////////////////////////////////////////////
// Eager specific configs - BEGIN
///////////////////////////////////////////////////////////////////////////////////////////////////

//
// Eager - Functional features
//

GlobalConfInt64 GCFG_FORCE_EAGER(
        "FORCE_EAGER",
        "Control compilation mode:"
                                   "0 - Force graph mode,"
                                   "1 - Force eager mode,"
                                   "2 - Go by the graph's configuration and allow fallback to graph mode (default),"
                                   "3 - Go by the graph's configuration and disallow fallback to graph mode",
        2,
        MakePrivate);

GlobalConfBool GCFG_EAGER_GENERATE_TEMPLATES(
        "EAGER_GENERATE_TEMPLATES",
        "Generate recipes templates upon synSingleton::initSingleton()",
        true,
        MakePrivate);

GlobalConfInt64 GCFG_ENABLE_COMPLEX_GUID_LIB_IN_EAGER(
        "ENABLE_COMPLEX_GUID_LIB_IN_EAGER",
        "Use complex GUID library to apply node extraction: 0 - Disable, 1 - Enable, 2 - Enable for specific GUIDs only",
        1,
        MakePrivate
);

//
// Eager - Runtime and compilation optimizations of arch features
//

GlobalConfBool GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS(
        "ENABLE_EAGER_ARCH_OPTIMIZATIONS",
        "Master enabler for all arch optimizations in eager mode",
        true,
        MakePrivate
);
GlobalConfBool GCFG_ENABLE_EAGER_NOP_IN_RECIPE(
        "ENABLE_EAGER_NOP_IN_RECIPE",
        "Try to plant nop kernel in recipe to prevent fetching from HBM",
        true,
        MakePrivate);

GlobalConfInt64 GCFG_ENABLE_EAGER_SB_REUSE_G2(
        "ENABLE_EAGER_SB_REUSE_G2",
        "Enable MME suspension buffer reuse in eager mode for gaudi2: 0-OFF, 1-FORCE-ON, 2-DEFAULT-HUERISTIC",
        2,
        MakePrivate
);

GlobalConfInt64 GCFG_ENABLE_EAGER_SB_REUSE_G3(
        "ENABLE_EAGER_SB_REUSE_G3",
        "Enable MME suspension buffer reuse in eager mode for gaudi3: 0-OFF, 1-FORCE_ON, 2-DEFAULT-HUERISTIC",
        0,
        MakePrivate
);

GlobalConfBool GCFG_ENABLE_EAGER_BATCH_CONCURRENCY(
        "ENABLE_EAGER_BATCH_CONCURRENCY",
        "Enable MME batch concurrency in eager mode",
        true,
        MakePrivate
);

GlobalConfBool GCFG_ENABLE_EAGER_MME_CONCURRENCY(
        "ENABLE_EAGER_MME_CONCURRENCY",
        "Enable common dim concurrency in Eager mode",
        true,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_CONV_PACKING_EAGER(
    "ENABLE_CONV_PACKING_EAGER",
    "Set packing enabled mode for Eager",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SUGGESTED_MANIPULATION_IN_EAGER(
    "ENABLE_SUGGESTED_MANIPULATION_IN_EAGER",
    "Allow checking for TPC suggested manipulation in Eager",
    true,
    MakePrivate);

GlobalConfString GCFG_ENABLE_EAGER_PARALLEL_EXECUTION(
    "ENABLE_EAGER_PARALLEL_EXECUTION",
    "Three possible values:"
    " 'enable': enable parallel execution attempts for any graph."
    " 'disable': force serial execution for all nodes, ignoring any possibility to run in parallel."
    " 'auto': depends on the graph decide if it's worth to attempt parallel execution or not.",
    std::string("disable"),  // TODO: enable once supporting control dependencies (SW-159454)
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CONSTANT_OPTIMIZATION_IN_EAGER(
    "ENABLE_CONSTANT_OPTIMIZATION_IN_EAGER",
    "Replace constant when possible by const tensors",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CAST_OPTIMIZATION_IN_EAGER(
    "ENABLE_CAST_OPTIMIZATION_IN_EAGER",
    "Replace cast of const tensor when possible by const tensors",
    true,
    MakePrivate);

//
// Eager - Node displacement optimizations
//

GlobalConfBool GCFG_ENABLE_EAGER_NODE_DISPLACEMENT_OPTIMIZATIONS(
        "ENABLE_EAGER_NODE_DISPLACEMENT_OPTIMIZATIONS",
        "Master enabler for all node displacement optimizations in eager mode",
        true,
        MakePrivate
);

GlobalConfBool GCFG_ENABLE_BATCH_NORM_SPLIT_IN_EAGER(
    "ENABLE_BATCH_NORM_SPLIT_IN_EAGER",
    "Split batch norm into two stages internally without complex GUID intervention in Eager",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TRANSPOSE_FUSION_IN_EAGER(
    "ENABLE_TRANSPOSE_FUSION_IN_EAGER",
    "Fuse transpose to gemm or batch gemm when possible",
    true,
    MakePrivate);

///////////////////////////////////////////////////////////////////////////////////////////////////
// Eager specific configs - END
///////////////////////////////////////////////////////////////////////////////////////////////////

GlobalConfBool GCFG_DISABLE_DS_MME_ROI_PATCHING (
    "DISABLE_DS_MME_ROI_PATCHING",
    "Disable creation of MME patch points for dynamic shapes",
    false,
    MakePrivate);

GlobalConfBool GCFG_DISABLE_DS_TPC_ROI_PATCHING(
    "DISABLE_DS_TPC_ROI_PATCHING",
    "Disable creation of TPC patch points for dynamic shapes",
    false,
    MakePrivate);

GlobalConfBool GCFG_DISABLE_DS_DMA_ROI_PATCHING(
    "DISABLE_DS_DMA_ROI_PATCHING",
    "Disable creation of DMA patch points for dynamic shapes",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TRANSPOSE_LOGICAL_ROIS_SPLIT(
    "ENABLE_TRANSPOSE_LOGICAL_ROIS_SPLIT",
    "Enable dma transpose split to logical rois",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_PER_CHANNEL_SCALING(
    "PER_CHANNEL_SCALING",
    "Enable per channel scaling for static tensors for MME nodes",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_OPTIMIZE_MEMCPY_NODES(
    "ENABLE_OPTIMIZE_MEMCPY_NODES",
    "Reshape memcopy input and output to be dense as possible",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_STATIC_TRANSPOSE_STRATEGY("ENABLE_STATIC_TRANSPOSE_STRATEGY",
                                                     "Allow performing transpose with a static shape",
                                                     DfltBool(true),
                                                     MakePrivate);

GlobalConfFloat GCFG_WORKSPACE_EPOCH_SIZE_PERCISION(
    "WORKSPACE_EPOCH_SIZE_PERCISION",
    "Defines when we need to stop the learning of the ideal epoch size for the workspace",
    0.001f,
    MakePrivate);

GlobalConfSize GCFG_WORKSPACE_MIN_SIZE_LIMIT(
    "WORKSPACE_MIN_SIZE_LIMIT",
    "Epoch DRAM allcoator is not utilized for workspace sizes smaller than this value",
    hl_gcfg::SizeParam("512mb"),
    MakePrivate
);

GlobalConfBool GCFG_ENABLE_PERSISTENT_OUTPUT_REUSE(
    "ENABLE_PERSISTENT_OUTPUT_REUSE",
    "Enable persistent output reuse for intermediate tensors",
    DfltBool(true),
    MakePrivate
);

GlobalConfBool GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME(
        "ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME",
        "Enable fusing transpose into mme when identity node is in between",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_FUSE_TRANSPOSE_TO_GEMM_OUTPUT(
        "ENABLE_FUSE_TRANSPOSE_TO_GEMM_OUTPUT",
        "Enable fusion of GEMM -> Transpose",
        DfltBool(true),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_WIDE_BUCKET(
        "ENABLE_WIDE_BUCKET",
        "Enable compiling with all dynamic minimal sizes are forced to 0",
        DfltBool(false),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_ROUNDING_MODE_PLAYBACK(
        "ENABLE_ROUNDING_MODE_PLAYBACK",
        "Set rounding mode on json file playback",
        DfltBool(false),
        MakePrivate);

GlobalConfInt64 GCFG_DEFAULT_CACHE_DIRECTIVE(
        "DEFAULT_CACHE_DIRECTIVE",
        "Set the default cache directive for gaudi3 cache. 0 - compiler decision, 1 - SkipCache, 2 - NoAllocate, 3 - HomeAllocate, 4 - DcoreAllocate, 5 - SharedAllocate",
        (3),
        MakePrivate);

GlobalConfInt64 GCFG_LITE_CME_MODE(
        "LITE_CME_MODE",
        "Set the lite cme mode for gaudi3. 0 - disabled, 1 - DISCARD only, 2 - DEGRADE only, 3 - Both (DEGRADE/DISCARD) - default",
        (3),
        MakePrivate);

GlobalConfUint64 GCFG_SFG_MAX_NUM_OF_MONITORS(
        "SFG_MAX_NUM_OF_MONITORS",
        "Maximal number of monitors to be used by SFG",
        400,
        MakePrivate);

GlobalConfBool GCFG_DBG_ENFORCE_NEW_SCRATCHPAD_SECTION_ADDRESS(
        "DBG_ENFORCE_NEW_SCRATCHPAD_SECTION_ADDRESS",
        "Enforce that the workspace section-address is new, even if it is not",
        false,
        MakePrivate);

GlobalConfInt64 GCFG_DBG_ENFORCE_NUM_OF_NEW_SECTIONS_GROUP_ADDRESSES(
        "DBG_ENFORCE_NUM_OF_NEW_PTS_SECTIONS_GROUP_ADDRESSES",
        "Enforce the amount of persistent-tensors' sections-groups that one of their sections' addresses had changed, even if they are not",
        0,
        MakePrivate);

GlobalConfFloat GCFG_MIN_TPC_PIPELINE_FACTOR(
        "MIN_TPC_PIPELINE_FACTOR",
        "minimum tpc pipeline level for small index space is (indexSpaceDepth/(num_tpc_engines*MIN_TPC_PIPELINE_FACTOR)",
        DfltFloat(1)
        << deviceValue(synDeviceGaudi2, 1.1),
        MakePrivate);

GlobalConfBool GCFG_ENABLE_BIG_TENSOR_PP_PRUNE(
        "ENABLE_BIG_TENSOR_PP_PRUNE",
        "Enable pruning of patchpoits",
        DfltBool(true),
        MakePrivate);

GlobalConfInt64 GCFG_TPC_MCID_CONFIG_MASK(
    "TPC_MCID_CONFIG_MASK",
    "TPC MCID configuration options mask (bitmask). bit:0 - FAST_CFG, bit:1 - SRF",
    0x1,
    MakePrivate);

GlobalConfInt64 GCFG_TPC_MCID_NUM_SRF(
    "TPC_MCID_NUM_SRF",
    "Number of SRFs registers used for TPC MCID configuration",
    0,
    MakePrivate);


GlobalConfBool GCFG_DISABLE_TENSOR_SIZE_VALIDATION(
    "DISABLE_TENSOR_SIZE_VALIDATION",
    "Disable tensor size validation - disabling this validation may overflow the engines on large tensors",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_DISABLE_SYNAPSE_HUGE_PAGES(
    "DISABLE_SYNAPSE_HUGE_PAGES",
    "Disable Runtime usage of huge pages when allocating/mapping memory",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_LOWER_DEDX(
    "ENABLE_LOWER_DEDX",
    "Enable lower dedx pass - replaces dedx nodes by transpose + transposedDedx",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_LOWER_DEDX_REVERSE_WEIGHTS(
    "LOWER_DEDX_REVERSE_WEIGHTS",
    "Reverse weights in lower dedx pass",
    DfltBool(true),
    MakePrivate);

GlobalConfBool GCFG_DISABLE_DS_TPC_INDEX_SPACE_PATCHING(
    "DISABLE_DS_TPC_INDEX_SPACE_PATCHING",
    "Disable creation of TPC index space patch points for dynamic shapes",
    false,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_TRANSPOSE_VIA_GEMM(
    "ENABLE_TRANSPOSE_VIA_GEMM",
    "Perform Gaudi3 transpose in mme via gemm pipe (and not via dma pipe)",
    DfltBool(false) << deviceValue(synDeviceGaudi3, false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CD_PARALLEL(
    "ENABLE_CD_PARALLEL",
    "Enable CD parallel (between dcores) optimization in gaudi3",
    DfltBool(false) << deviceValue(synDeviceGaudi3, false), // TODO [SW-156982]: enable cd parallel opt on gaudi3
    MakePrivate);

GlobalConfBool GCFG_ENABLE_CHECK_EVENT_REUSE(
    "ENABLE_CHECK_EVENT_REUSE",
    "Gaudi3 check if event was reused before querried or synchronized or time elapsed",
    DfltBool(false) << deviceValue(synDeviceGaudi3, false), //[SW-95964] Prevent reuse of events before wait was triggered
    MakePrivate);

GlobalConfString GCFG_QUANTIZATION_PARAMS_PATH(
    "QUANTIZATION_PARAMS_PATH",
    "A path to a JSON file containing per-tensor scales/dynamic range with the desired mode to use for calculations",
    std::string(),
    MakePrivate);


GlobalConfString GCFG_PATH_TO_IMPORT_STRATEGIES(
    "PATH_TO_IMPORT_STRATEGIES",
    "Set a path to a JSON file to import MME strategy or another",
    std::string(),
    MakePrivate
);

GlobalConfString GCFG_PATH_TO_EXPORT_STRATEGIES(
    "PATH_TO_EXPORT_STRATEGIES",
    "Set a path to a JSON file export MME strategy or another",
    std::string(),
    MakePrivate
);


GlobalConfUint64 GCFG_NUM_OF_USER_STREAM_EVENTS(
    "GCFG_NUM_OF_USER_STREAM_EVENTS",
    "Number of event handles user can create",
    1024*1024*10,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_STRIDED_OP_DECODING(
    "ENABLE_STRIDED_OP_DECODING",
    "Decode strided op into it's logical counterparts when possible",
    true,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_MME_DESCRIPTOR_CACHE(
    "GCFG_ENABLE_MME_DESCRIPTOR_CACHE",
    "Use MME descriptor cache",
    DfltBool(true),
    MakePrivate);

GlobalConfUint64 GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING(
    "CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING",
    "The max discard mcid value before going into rollover. Use it for testing purposes only. 0 means not in use.",
    DfltUint64(0),
    MakePrivate);

GlobalConfUint64 GCFG_MAX_MME_ACTIVATIONS(
        "MAX_MME_ACTIVATIONS",
        "limit number of Activations to not overload memory",
        100000,
        MakePrivate);

GlobalConfBool GCFG_ENABLE_OPTIMIZED_LOGICAL_ROI_SPLIT(
    "ENABLE_OPTIMIZED_LOGICAL_ROI_SPLIT",
    "Enable logical ROI split optimization according to number of physical engines",
    DfltBool(false) << deviceValue(synDeviceGaudi2, true),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_INPUT_REUSE_AS_LOGICAL_NODE(
    "ENABLE_INPUT_REUSE_AS_LOGICAL_NODE",
    "Enable handling input reuse binding nodes with the logical nodes",
    true,
    MakePrivate);

GlobalConfBool GCFG_ADD_ALIGNMENT_PENALTY_MME_BRAIN(
    "ADD_ALIGNMENT_PENALTY_MME_BRAIN",
    "Performance calc takes unalignment into account in MME brain",
    DfltBool(false),
    MakePublic);

GlobalConfBool GCFG_TIE_BRAEKER_PREFERRED_REUSE_OPERAND_MME_BRAIN(
    "TIE_BRAEKER_PREFERRED_REUSE_OPERAND_MME_BRAIN",
    "Allow MME brain to prefer the shortest geometry on the operand with SRAM hint, in case of performance tie between geometries.",
    DfltBool(false),
    MakePublic);

GlobalConfBool GCFG_ENABLE_BN_FCD_PACKING(
    "ENABLE_BN_FCD_PACKING",
    "allow BN low fcd packing optimization",
    DfltBool(true),
    MakePublic);

GlobalConfBool GCFG_ENABLE_RUN_ON_CPU_DUMMY_MODE(
    "ENABLE_RUN_ON_CPU_DUMMY_MODE",
    "Enable run on cpu dummy mode",
    DfltBool(false),
    MakePrivate);

GlobalConfUint64 GCFG_HOST_CYCLIC_BUFFER_SIZE(
    "SET_HOST_CYCLIC_BUFFER_SIZE",
    "Set host command cyclic buffer size (KB)",
    128,
    MakePrivate);

GlobalConfUint64 GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT(
    "SET_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT",
    "Set host command cyclic buffer amount of chunks",
    16,
    MakePrivate);

GlobalConfBool GCFG_ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK(
    "ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK",
    "Enable sampling mechanism watermark for host command cyclic buffer",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_DFA_SAVE_RECIPE(
    "DFA_SAVE_RECIPE",
    "Serialize suspected recipe to disk",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENFORCE_POST_SLICING_SHAPE_CHECK(
    "ENFORCE_POST_SLICING_SHAPE_CHECK",
    "Force compilation failure on post-slicing shape mismatch",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_DISABLE_ALL_CME_COMMANDS(
    "DISABLE_ALL_CME_COMMANDS",
    "Disable all CME commands (for testing and debugging)",
    DfltBool(false),
    MakePrivate);

GlobalConfBool GCFG_ENABLE_DSD_WITH_LB(
    "ENABLE_DSD_WITH_LB",
    "Enable dynamic shapes with layered brain",
     DfltBool(true),
     MakePrivate);

// clang-format on

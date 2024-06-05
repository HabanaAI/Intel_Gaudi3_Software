#pragma once

#include <string>
#include <set>
#include <memory>

class HabanaGraph;

typedef enum PassId
{
    PASS_ID_INVALID_ID,
    PASS_ID_GRAPH_VISUALIZATION_PRE,
    PASS_ID_CONTROL_DEP_RELAXATION,
    PASS_ID_SPILL_PERSISTENT_TENSORS,
    PASS_ID_EXTRACT_RECURRENT_NODES,
    PASS_ID_HANDLE_IDENTITY_CAST_NODES,
    PASS_ID_FUSE_BN_CONV,
    PASS_ID_REPLACE_GROUP_CONV_FILTER2D,
    PASS_ID_FUSE_CONV_ADD_BIAS,
    PASS_ID_FUSE_CONV_ADD_CIN,
    PASS_ID_FUSE_PAD_INTO_CONV_POOL,
    PASS_ID_FUSE_LEAKY_RELU_ADD,
    PASS_ID_ADD_AS_BIAS_TO_FC,
    PASS_ID_FUSE_RELU,
    PASS_ID_ELU_MUL_SCALAR_FUSION,
    PASS_ID_LOG_SIGMA_EXP_FUSE,
    PASS_ID_FUSE_INTO_MASK_INVALID_SOFTMAX,
    PASS_ID_ELIMINATE_NODES_WITH_STATIC_INPUTS,
    PASS_ID_ELIMINATE_DYNAMIC_RANGE_OPS,
    PASS_ID_REMOVE_ZERO_SIZED_PAD,
    PASS_ID_UPGRADE_TPC_NODES_PRECISION,
    PASS_ID_CALC_DYNAMIC_RANGE,
    PASS_ID_UPDATE_MME_NODES_PRECISION,
    PASS_ID_SET_SUPPORTED_LAYOUTS,
    PASS_ID_OPTIMIZE_STRIDED_INSERT,
    PASS_ID_ADJUST_DATA_LAYOUT,
    PASS_ID_TRANSPOSE_DONT_CARE_NODES,
    PASS_ID_HANDLE_PERMUTED_TENSORS,
    PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES,
    PASS_ID_ELIMINATE_TRANSPOSE_BEFORE_FC,
    PASS_ID_ELIMINATE_FIRST_LAST_TRANSPOSE,
    PASS_ID_EXTRACT_MULTI_NODES,
    PASS_ID_EXTRACT_DATA_MOVEMENT_MULTI_NODES,
    PASS_ID_TRANSPOSE_REDUCE_DIMENSIONS,
    PASS_ID_FLATTEN_PHYSICAL_TRANSPOSE,
    PASS_ID_SET_COMPATABILITY_MODE,
    PASS_ID_TRANSFORM_CONVOLUTIONS_TO_GEMM,
    PASS_ID_FUSE_NODES,
    PASS_ID_INSERT_TPC_LOWERING_NODES,
    PASS_ID_CAST_FOR_NON_TPC_NODES,
    PASS_ID_FIX_STRIDED_TRANSPOSES,
    PASS_ID_HANDLE_GROUPED_CONVOLUTIONS,
    PASS_ID_LOAD_TPC_KERNELS,
    PASS_ID_IN_PLACE_INPUT_REUSE_BINDING,
    PASS_ID_IN_PLACE_INPUT_REUSE_SUGGESTION,
    PASS_ID_REMOVE_CONTIGUOUS_RESHAPES,
    PASS_ID_MEMSET_NODE_OUTPUT,
    PASS_ID_OPTIMIZE_TPC_KERNELS,
    PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES,
    PASS_ID_REMOVE_CONTIGUOUS_CONVERTS,
    PASS_ID_FUSE_CAST_MME,
    PASS_ID_FUSE_CONVERT_MME,
    PASS_ID_REMOVE_REDUNDANT_MEMCPY_NODES,
    PASS_ID_VALIDATE_NODES_LAYOUT,
    PASS_ID_PLACE_IO_IN_DRAM,
    PASS_ID_DUPLICATE_LOGICAL_OPS,
    PASS_ID_SPILL_LONG_RESIDUALS,
    PASS_ID_MAKE_CONCATS_PHYSICAL,
    PASS_ID_SPLIT_INTO_RANGES,
    PASS_ID_SLICE_GRAPH_RANGES,
    PASS_ID_SET_ADVANCEMENT_ORDER,
    PASS_ID_GENERATE_ROIS,
    PASS_ID_SPLIT_TO_DCORE_ROIS,
    PASS_ID_DALI_CONVOLUTION_STRATEGY,
    PASS_ID_COMPRESS_STRATEGY,
    PASS_ID_INTERLEAVE_LSTM_WEIGHTS,
    PASS_ID_PACK_CONVOLUTIONS,
    PASS_ID_LINEARIZE_WEIGHTS,
    PASS_ID_COMPRESS_WEIGHTS,
    PASS_ID_SET_COMPRESSED_TENSOR_ALIGNMENT,
    PASS_ID_INSERT_DMA_NODES,
    PASS_ID_STATIC_TENSORS_INVALIDATOR,
    PASS_ID_SLICE_GRAPH_WEIGHT_TENSORS,
    PASS_ID_IN_PLACE_SRAM_REUSE,
    PASS_ID_OPTIMIZE_MEMORY_STRATEGY,
    PASS_ID_ALLOCATE_TENSORS,
    PASS_ID_INSERT_PREFETCH_DMA_NODES,
    PASS_ID_SPLIT_TPC_DIMS,
    PASS_ID_SPLIT_ROIS,
    PASS_ID_PROJECT_NODE_ROIS,
    PASS_ID_SET_ROI_ENGINE,
    PASS_ID_ASSIGN_ADDRESSES_TO_TENSOR_ROIS,
    PASS_ID_CALCULATE_TENSOR_ROIS_LINEAR_RANGES,
    PASS_ID_GENERATE_CACHE_MAINTENANCE_TASKS,
    PASS_ID_ALLOCATE_SYNCS,
    PASS_ID_VALIDATE_SYNC_SCHEME,
    PASS_ID_GRAPH_VISUALIZATION_POST,
    PASS_ID_UPDATE_NODES_WITH_ALIAS_TENSORS,
    PASS_ID_SPLIT_TF_BATCH_NORM,
    PASS_ID_SPLIT_MOMENTS,
    PASS_ID_SPLIT_BATCH_NORM,
    PASS_ID_SPLIT_TO_LOGICAL_ROIS,
    PASS_ID_SPLIT_TO_PHYSICAL_ROIS,
    PASS_ID_DISABLE_BUNDLE_ROIS,
    PASS_ID_ADD_MME_BIAS,
    PASS_ID_CREATE_DMA_DISPATCHERS,
    PASS_ID_SELECT_MEMCPY_ENGINE,
    PASS_ID_REMOVE_UNREQUIRED_REQUANTS,
    PASS_ID_SET_REDUCTION_MEMSET,
    PASS_ID_HANDLE_LOGICAL_OPERATIONS,
    PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS,
    PASS_ID_HANDLE_LOGICAL_OPERATIONS_POST_PROCESS,
    PASS_ID_OPTIMIZE_MEMCPY_NODES,
    PASS_ID_CALC_QUANTIZATION_INFO,
    PASS_ID_FUSE_GELU,
    PASS_ID_TPC_FUSER,
    PASS_ID_VALIDATE_MEMORY_ALLOCATION,
    PASS_ID_SET_HABANA_LAYOUTS,
    PASS_ID_MARK_REDUCTION_INPUTS,
    PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,
    PASS_ID_FUSE_SPILL_FILL,
    PASS_ID_BUNDLE_NODES_SCHEDULE,
    PASS_ID_HANDLE_PARTIALS_WRITES,
    PASS_ID_BUNDLE_MEMORY_MANAGEMENT,
    PASS_ID_MME_CONCURRENCY_IDENTIFIER,
    PASS_ID_MME_CONCURRENCY_MEMSET,
    PASS_ID_MME_CONCURRENCY,
    PASS_ID_CHECK_INPUT_PERSISTENCE,
    PASS_ID_CONVERT_1X1BATCH_GEMM_TO_GEMM,
    PASS_ID_VALIDATE_MME_NODES,
    PASS_ID_INIT_MME_BRAIN_IFC,
    PASS_ID_VALIDATE_ATOMIC_NODES,
    PASS_ID_VALIDATE_DMA_NODES,
    PASS_ID_REMOVE_REDUNDANT_LOGICAL_NODES,
    PASS_ID_CONVERT_BATCH_GEMM_TO_GEMM,
    PASS_ID_RESHAPE_ELEMENTWISE_5D_TPC_NODES,
    PASS_ID_HANDLE_HUGE_TENSORS,
    PASS_ID_FUSE_BATCH_NORM,
    PASS_ID_FUSE_BATCH_NORM_MEMCPY,
    PASS_ID_FUSE_TRANSPOSE_MME,
    PASS_ID_VALIDATE_EXECUTION_SCHEDULE_BUNDLES,
    PASS_ID_SET_NON_PERSISTENT_SECTION_INFO,
    PASS_ID_GENERATE_MME_DESCRIPTORS,
    PASS_ID_PACKING_MME_NODES,
    PASS_ID_COMPRESS_WEIGHT_TENSORS,
    PASS_ID_ALIGN_MME_TENSORS,
    PASS_ID_UPDATE_META_DATA_BUFFER,
    PASS_ID_INTERNAL_TENSORS_DYNAMIC_SHAPE,
    PASS_ID_LINK_REDUCTION_MEMSET_SHAPES,
    PASS_ID_VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE,
    PASS_ID_PATCH_MME_DESCRIPTORS,
    PASS_ID_PATCH_MME_MCIDS,
    PASS_ID_HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES,
    PASS_ID_REPLACE_OPS_WITH_LOGICAL_OPS,
    PASS_ID_STATIC_TENSORS_FLOAT_CONVERSION,
    PASS_ID_FUSE_LAYER_NORM,
    PASS_ID_FUSE_WAITS,
    PASS_ID_FUSE_RAGGED_SOFTMAX,
    PASS_ID_FUSE_MULT_INTO_FC,
    PASS_ID_HANDLE_PIPELINING_POWER_LIMITATIONS,
    PASS_ID_SPLIT_LAYER_NORM_BWD,
    PASS_ID_HANDLE_RMW_TPC_KERNELS,
    PASS_ID_ALLOCATE_TPC_KERNELS,
    PASS_ID_SET_ROI_SHAPE_TYPE,
    PASS_ID_ADJUST_RESTRICTIONS,
    PASS_ID_LOCK_ANCESTORS_FOR_REQUANT,
    PASS_ID_UPDATE_PAD_QUANTIZER,
    PASS_ID_ADJUST_SCALES,
    PASS_ID_ENFORCE_NODE_PRECISION,
    PASS_ID_VALIDATE_QUANTIZATION,
    PASS_ID_REQUANT_CONFLICTS,
    PASS_ID_GENERATE_PROFILER_DEBUG_INFO,
    PASS_ID_INSERT_SERIALIZE_DESERIALIZE,
    PASS_ID_VALIDATE_DYNAMIC_SHAPES,
    PASS_ID_MANAGE_BASE_REGS_CACHE,
    PASS_ID_FUSE_TRANSPOSE_BATCH_GEMM,
    PASS_ID_VALIDATE_USER_MEMORY_SECTIONS,
    PASS_ID_VALIDATE_MEMORY_SECTION_TENSORS,
    PASS_ID_SET_DMA_PARALLEL_LEVEL,
    PASS_ID_SPLIT_FROBENIUS_LAYER_NORM,
    PASS_ID_NODES_PRECISION_SELECTION,
    PASS_ID_TENSORS_DATA_TYPE_SELECTION,
    PASS_ID_TRANSPOSE_FCD_BROADCAST,
    PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES,
    PASS_ID_EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES,
    PASS_ID_HANDLE_PARTIAL_BROADCAST_BGEMM,
    PASS_ID_INSERT_NAN_INF_PROBE,
    PASS_ID_REMOVE_ZERO_SIZED_TENSORS,
    PASS_ID_GENERATE_WORK_DISTRIBUTION,
    PASS_ID_HANDLE_MEMORY_REUSE,
    PASS_ID_ADD_H2D_OP,
    PASS_ID_SIGNAL_OUT_FROM_GRAPH,
    PASS_ID_FUSE_BROADCAST_TPC,
    PASS_ID_REGISTER_MEM_COHERENCE,
    PASS_ID_CSE_OPTIMIZATION,
    PASS_ID_ALIGN_ASYMMETRIC_BGEMM,
    PASS_ID_FORCE_EXPLICIT_PADDING,
    PASS_ID_FUSE_CONST_TRANSPOSE,
    PASS_ID_SET_TENSOR_IN_HBM,
    PASS_ID_ALIGN_TRANSPOSE_VIA_GEMM_OUTPUT,
    PASS_ID_ELIMINATE_REDUNDANT_NODES,
    PASS_ID_LOWER_DEDX,
    PASS_ID_GC_PERF_CHECKS,
    PASS_ID_REMOVE_OPPOSITE_SPLIT_CONCAT,
    PASS_ID_STATIC_TENSOR_CAST_INSERTION,
    PASS_ID_CAST_FOR_TPC_NODES,
    PASS_ID_INJECT_SCALE_FOR_MME_NODES,
    PASS_ID_PROPAGATE_CAST_NODES,
    PASS_ID_CHECK_MAX_DIMS_PRE,
    PASS_ID_CHECK_MAX_DIMS_POST,
    PASS_ID_ADD_STATIC_SHAPE_TENSORS,
    PASS_ID_VALIDATE_PRE_SLICING_SIZES,
    PASS_ID_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE,
    PASS_ID_RUN_LAYERED_BRAIN,
    PASS_ID_SET_FLASH_ATTN_SCHEDULE,
    PASS_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE,

    PASS_ID_GRAPH_MUTATIONS_GROUP,  // Group to separate the passes that change the graph from passes that rely on the
                                    // graph being stable
    /* Groups */
    GROUP_ID_FIRST_GROUP,
    GROUP_ID_NO_GROUP = GROUP_ID_FIRST_GROUP,  // temporary until [SW-63782] is done
    GROUP_ID_PRE_COMPILE_GROUP,
    GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,
    GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL,
    GROUP_ID_CORRECT_USER_GRAPH_DATA_LAYOUT,
    GROUP_ID_PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH,
    GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,
    GROUP_ID_PRE_COMPILE_PRE_COMPILE_OPTIMIZE_DATA_LAYOUT,
    GROUP_ID_LOWERING_GROUP,
    GROUP_ID_BE_GROUP,
    GROUP_ID_BE_GRAPH_OPT_GROUP,
    GROUP_ID_BE_MAIN_GROUP,
    GROUP_ID_BE_TPC_GROUP,
    GROUP_ID_BE_MME_OPTIMIZATION_GROUP,
    GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,
    GROUP_ID_BE_GC_BRAIN_GROUP,
    GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,
    GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,
    GROUP_ID_BE_GC_BRAIN_MEMORY_GROUP,
    GROUP_ID_BE_TPC_FUSER_GROUP,
    GROUP_ID_BE_POST_GRAPH_GROUP,
    GROUP_ID_GCONV_HANDLING_GROUP,
    GROUP_ID_LOWERING_NORM_OPS,
    GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO,
    GROUP_ID_PRE_COMPILE_DATA_TYPE_SELECTION,
    GROUP_ID_PRE_COMPILE_CASTS_INJECTION,
    GROUP_ID_HUGE_TENSORS_HANDLING_GROUP,
    GROUP_ID_POST_GRAPH_CODE_GEN_LINEAR_RANGES,
    GROUP_ID_POST_GRAPH_CODE_GEN_LOGICAL_ROI,
    GROUP_ID_POST_GRAPH_CODE_GEN_MAIN_GROUP,
    GROUP_ID_POST_GRAPH_CODE_GEN,
    GROUP_ID_POST_GRAPH_GROUP,
    GROUP_ID_POST_GRAPH_MEMORY,
    GROUP_ID_POST_GRAPH_VALIDATIONS,
    PASS_ID_MAX_ID

} PassId;

enum PredicateId
{
    PREDICATE_ID_NODE_CREATED,
    PREDICATE_ID_NODE_CREATED_CONST_INPUT,
    PREDICATE_ID_TPC_NODE_INITIALIZED,
    PREDICATE_ID_TPC_NODE_CREATED,
    PREDICATE_ID_MEMCPY_NODE_CREATED,
    PREDICATE_ID_LOGICAL_NODE_CREATED,
    PREDICATE_ID_RESHAPE_NODE_CREATED,
    PREDICATE_ID_DMA_NODE_CREATED,
    PREDICATE_ID_CAST_NODE_CREATED,
    PREDICATE_ID_TPC_MEMCPY_NODE_CREATED,
    PREDICATE_ID_MME_NODE_CREATED,
    PREDICATE_ID_TRANSPOSE_NODE_CREATED,
    PREDICATE_ID_REDUCTION_NODE_CREATED,
    PREDICATE_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE,
    PREDICATE_ID_LOGICAL_NODE_RAN,
    PREDICATE_ID_ELIMINATE_REDUNDANT_NODES,
    PREDICATE_ID_MEMORY_SECTION_TENSOR_CREATED,
    PREDICATE_ID_BROADCAST_NODE_CREATED,
    PREDICATE_ID_FUSED_NODE_TO_MME,
    PREDICATE_ID_PHYSICAL_TRANSPOSE_NODE_CREATED,
    PREDICATE_ID_EXTERNAL_MME_NODE_CREATED,

    // must be last
    PREDICATE_ID_MAX_ID
};

typedef std::set<PassId> PassIDSet;
typedef std::set<PredicateId> PredicateIDSet;

// TODO [SW-5774] These macros should not be necessary, remove when they become unused.
#define DEP_SET(set) {set}
#define NO_DEP()     (PassIDSet())

typedef unsigned PassPriority;

// Passes are defined by default with priority 1000, in order to easily allow passes with low priority.
constexpr PassPriority PASS_DEF_PRIO        = 1000;
constexpr PassPriority PASS_MAX_PRIO        = 9999;
constexpr PassPriority PASS_MIN_PRIO        = 0;
constexpr PassPriority PASS_VALIDATION_PRIO = PASS_MAX_PRIO-1; // Validation passes priority (should happen as soon as possible)
constexpr PassPriority PASS_HIGH_PRIO       = PASS_DEF_PRIO + 1;
// Groups are defined by default with priority 0, so the re-run queue would be executed before moving to the next group.
constexpr PassPriority GROUP_DEF_PRIO = 0;

// Passes that should run as soon as possible with some order may use the following priorities
constexpr PassPriority PASS_URGENT_0 = PASS_MAX_PRIO - 0;
constexpr PassPriority PASS_URGENT_1 = PASS_MAX_PRIO - 1;
constexpr PassPriority PASS_URGENT_2 = PASS_MAX_PRIO - 2;
constexpr PassPriority PASS_URGENT_3 = PASS_MAX_PRIO - 3;
constexpr PassPriority PASS_URGENT_4 = PASS_MAX_PRIO - 4;

// INTERNAL_TENSORS_DYNAMIC_SHAPE should run after each node creation, but
// in case a reduction node was created, need to run LINK_REDUCTION_MEMSET_SHAPES first, so
constexpr PassPriority LINK_REDUCTION_MEMSET_SHAPES_PRIO = PASS_URGENT_0;
constexpr PassPriority INTERNAL_TENSORS_DYNAMIC_SHAPE_PRIO = PASS_URGENT_1;
// in case a new memory section tensors is created, we must register it before any graph modification
constexpr PassPriority REGISTER_MEMORY_COHERENCE_PRIO = PASS_URGENT_2;
// handle logical ops pass should be low priority, but not too low, since there are passes that must run after it
constexpr PassPriority HANDLE_LOGICAL_OPS_PRIO = (PASS_MIN_PRIO + PASS_DEF_PRIO) / 2;

class Pass;
typedef std::shared_ptr<Pass> pPass;

class Pass
{
public:
    Pass(std::string_view name, PassId id, PassPriority priority);
    Pass(std::string_view name, PassId id, PassPriority priority, PredicateIDSet predicateSet, PassIDSet dependencySet);
    virtual ~Pass();

    const std::string&    getName() const;
    PassId                getId() const;
    virtual bool          isPassGroup() const { return false; }
    const PassIDSet&      getDependencySet() const;
    const PredicateIDSet& getPredicateSet() const;
    virtual PassPriority  getPriority() const { return m_priority; }
    virtual bool          Apply(HabanaGraph& g) const = 0;
    virtual pPass         create() const = 0;
    virtual Pass*         setPriority(PassPriority priority) { m_priority = priority; return this; }
    virtual Pass*         addPredicate(PredicateId predId);

    // Subclasses that allow predicates need to override this.
    virtual inline bool   canRunMultipleTimes() { return false; }

protected:
    std::string    m_name;
    PassId         m_id;
    PredicateIDSet m_predicateSet;
    PassIDSet      m_dependencySet;
    PassPriority   m_priority;
};

template<bool (*applyFunc)(HabanaGraph&)>
class HabanaPass : public Pass
{
public:
    HabanaPass(std::string_view name, PassId id, PassIDSet depSet = {})
    : Pass(name, id, PASS_DEF_PRIO, {}, std::move(depSet))
    {}

    bool Apply(HabanaGraph& graph) const override { return applyFunc(graph); }
    pPass create() const override { return pPass(new HabanaPass<applyFunc>(*this)); }

    // Passes that allow predicates should define specialization to override the default.
    inline bool canRunMultipleTimes() override { return Pass::canRunMultipleTimes(); }
};

class PassGroup : public Pass
{
public:
    PassGroup(std::string_view name, PassId id, PassIDSet groupMembers, PassIDSet depSet)
    : Pass(name, id, PASS_DEF_PRIO, {}, std::move(depSet)), m_groupMembers(std::move(groupMembers))
    {}

    const PassIDSet& getGroupPasses() const { return m_groupMembers; }
    virtual bool isPassGroup() const override { return true; }
    bool Apply(HabanaGraph&) const override { return true; }
    pPass        create() const override { return pPass(new PassGroup(*this)); }

protected:
    PassIDSet m_groupMembers;
};

#define REGISTER_HABANA_PASS(func_, id_, depSet_) addPass(pPass(new HabanaPass<func_>(#func_, (id_), depSet_)))

#define SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(pass_) \
    template<> inline bool HabanaPass<pass_>::canRunMultipleTimes() { return true; }

#define REGISTER_GROUP(name_, id_, groupMembers_, depSet_)                                                             \
    addPass(pPass(new PassGroup(#name_, (id_), groupMembers_, depSet_)))


bool fuseBroadcast(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(fuseBroadcast);

bool generateROIs(HabanaGraph& g);
bool splitToDcoreROIs(HabanaGraph& g);
bool validateNodesLayout(HabanaGraph& g);
bool assignAddressesToTensorROIs(HabanaGraph& g);
bool projectNodeROIs(HabanaGraph& g);
bool splitTPCDims(HabanaGraph& g);
bool fuseBatchNorm(HabanaGraph&);
bool fuseBatchNormMemCpy(HabanaGraph&);
bool extractMultiNodes(HabanaGraph& g);
bool extractDataMovementMultiNodes(HabanaGraph& g);
bool graphVisualizationPre(HabanaGraph&);
bool graphVisualizationPost(HabanaGraph&);
bool convertBatchGemmToGemm(HabanaGraph& g);
bool tpcFuser(HabanaGraph &g);
bool splitToPhysicalROIs(HabanaGraph& g);
bool setRoiShapeType(HabanaGraph& g);
bool setNonPersistentSectionInfo(HabanaGraph& g);
bool insertPrefetchDmaNodes(HabanaGraph&);
bool fuseWaits(HabanaGraph& g);
bool setContextId(HabanaGraph& g);
bool setSupportedLayouts(HabanaGraph& g);
bool optimizeStridedInsert(HabanaGraph& g);
bool adjustDataLayout(HabanaGraph& g);
bool transposeDontCareNodes(HabanaGraph& g);
bool handlePermutedTensors(HabanaGraph& g);
bool eliminateTransposeBeforeFC(HabanaGraph& g);
bool eliminateFirstLastTranspose(HabanaGraph& g);
bool castForNonTPCNodes(HabanaGraph& g);
bool eliminateDynamicRangeOps(HabanaGraph& g);
bool transposeRemoveRedundantDimensions(HabanaGraph& g);
bool upgradeTPCNodesPrecision(HabanaGraph& g);
bool calcDynamicRange(HabanaGraph& g);
bool calcQuantizationInfo(HabanaGraph& g);
bool adjustRestrictions(HabanaGraph& g);
bool adjustScales(HabanaGraph& g);
bool fuseGelu(HabanaGraph& g);
bool lockAncestorsForRequant(HabanaGraph& g);
bool handleHugeTensors(HabanaGraph& g);
bool validateQuantization(HabanaGraph& g);
bool updatePadQuantizer(HabanaGraph& g);
bool enforceNodePrecision(HabanaGraph& g);
bool requantConflicts(HabanaGraph& g);
bool disablePipeliningCompletely(HabanaGraph& g);
bool checkInputPersistence(HabanaGraph& g);
bool allocateTensors(HabanaGraph& g);
bool validateExecutionScheduleBundles(HabanaGraph& g);
bool validateMMENodes(HabanaGraph& g);
bool InitMmeBrainIfc(HabanaGraph& g);                   SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(InitMmeBrainIfc)
bool validateAtomicNodes(HabanaGraph& g);
bool validateMemoryAllocation(HabanaGraph& g);
bool updateNodesWithAliasTensors(HabanaGraph& g);
bool handleTpcRmwKernels(HabanaGraph& g);
bool relaxCtrlDeps(HabanaGraph& g);
bool memsetNodeOutput(HabanaGraph& g);
bool flattenPhysicalTranspose(HabanaGraph& g);
bool setReductionMemset(HabanaGraph& g);
bool generateProfilerDebugInfo(HabanaGraph& g);
bool replaceGroupConvFilter2d(HabanaGraph& g);
bool sliceGraphToSRAMCapacity(HabanaGraph& g);
bool identifyMmeConcurrency(HabanaGraph& g);
bool applyMmeConcurrencyMemset(HabanaGraph& g);
bool enableMmeNodeCommonDimConcurrency(HabanaGraph& g);
bool sliceGraphForPipeline(HabanaGraph& g);
bool bundleNodesSchedule(HabanaGraph& g);
bool handlePartialsWrites(HabanaGraph& g);
bool bundleMemoryManagement(HabanaGraph& g);
bool disableBundleRois(HabanaGraph& g);
bool removeUnrequiredRequants(HabanaGraph& g);
bool convert1x1BatchGemmToGemm(HabanaGraph& g);
bool addMmeBias(HabanaGraph& g);
bool setHabanaLayouts(HabanaGraph& g);
bool splitTfBatchNorm(HabanaGraph& g);
bool splitMoments(HabanaGraph& g);
bool splitBatchNorm(HabanaGraph& g);
bool splitLayerNormBwd(HabanaGraph& g);
bool handleGroupedConvolutions(HabanaGraph& g);
bool validateUserMemorySections(HabanaGraph& g);
bool validateMemorySectionTensors(HabanaGraph& g);
bool setDmaParallelLevel(HabanaGraph& g);
bool eluMulScalarFusion(HabanaGraph&);
bool fuseIntoMaskInvalidSoftmax(HabanaGraph&);
bool fuseBNConv(HabanaGraph& g);
bool removeZeroSizedPad(HabanaGraph& g);
bool fusePadIntoConvPool(HabanaGraph& g);
bool removeRedundantMemcpyNodes(HabanaGraph& g);
bool replaceOpsWithLogicalOps(HabanaGraph& g);
bool splitFrobeniusLayerNorm(HabanaGraph& g);
bool setGraphTensorsDataType(HabanaGraph& g);
bool handleIdentityCastNodes(HabanaGraph& g);
bool transposeFcdBroadcast(HabanaGraph& g);
bool extractFunctionalComplexGuidNodes(HabanaGraph& g);
bool extractPerformanceComplexGuidNodes(HabanaGraph& g);
bool handleBroadcastBatchGemm(HabanaGraph& g);
bool alignAsymmetricBgemm(HabanaGraph& g);
bool insertSerializeDeserialize(HabanaGraph& g);
bool validateDynamicShapes(HabanaGraph& g);
bool insertProbeNanNodes(HabanaGraph& g);
bool setLogicalBeforePhysicalTranspose(HabanaGraph& g);
bool generateWorkDistribution(HabanaGraph& g);
bool packingMmeNodes(HabanaGraph&);
bool inPlaceInputReuseSuggestion(HabanaGraph& g);
bool scheduleFlashAttentionNodes(HabanaGraph& g);
bool commonSubExpressionElimination(HabanaGraph&);
bool forceExplicitPadding(HabanaGraph&);
bool synapseMLIROptimizer(HabanaGraph& g);
bool fuseSpillFillDirectives(HabanaGraph& g);
bool lowerDedx(HabanaGraph& g);
bool gcPerfChecks(HabanaGraph& g);
bool removeOppositeConcatSplitSequence(HabanaGraph& g);
bool staticTensorsFloatConversion(HabanaGraph&);
bool staticTensorsCastInsert(HabanaGraph& g);
bool checkMaxDimsPreCompilation(HabanaGraph& g);
bool checkMaxDimsPostCompilation(HabanaGraph& g);
bool addStaticShapeTensors(HabanaGraph& g);
bool validatePreSlicingSizes(HabanaGraph& g);
bool handleLogicalOpsPreProcess(HabanaGraph& g);
bool handleLogicalOpsPostProcess(HabanaGraph& g);
bool alignBPTFCDStrideToCacheLine(HabanaGraph& g);
bool runLayeredBrain(HabanaGraph& g);

bool registerMemoryCoherence(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(registerMemoryCoherence)

bool spillPersistentTensors(HabanaGraph& g);

bool handleLogicalOps(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(handleLogicalOps)

bool handleCtrlEdgesForLogicalNodes(HabanaGraph&);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(handleCtrlEdgesForLogicalNodes)

bool removeContiguousCastNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(removeContiguousCastNodes)

bool eliminateRedundantNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(eliminateRedundantNodes)

bool setTensorsSemanticType(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(setTensorsSemanticType)

bool removeRedundantLogicalNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(removeRedundantLogicalNodes)

bool optimizeTpcKernels(HabanaGraph&);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(optimizeTpcKernels)

bool removeContiguousReshapeNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(removeContiguousReshapeNodes)

bool eliminateNodesWithStaticInputs(HabanaGraph &g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(eliminateNodesWithStaticInputs)

bool validateDmaNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(validateDmaNodes);

bool markReductionInputs(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(markReductionInputs)

bool linkReductionMemsetShapes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(linkReductionMemsetShapes)

bool handleMemoryReuse(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(handleMemoryReuse)

bool removeContiguousTransposes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(removeContiguousTransposes)

bool verifyMemsetOutputShapes(HabanaGraph& g);

bool internalTensorsDynamicShape(HabanaGraph&);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(internalTensorsDynamicShape)

bool removeZeroSizedTensors(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(removeZeroSizedTensors)

bool fuseTransposeMme(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(fuseTransposeMme)

bool fuseConstantTranspose(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(fuseConstantTranspose)

bool fuseCastMme(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(fuseCastMme)

bool inPlaceInputReuseBinding(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(inPlaceInputReuseBinding)

bool optimizeMemcpyNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(optimizeMemcpyNodes)

bool castForTPCNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(castForTPCNodes)

bool updateMMENodePrecision(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(updateMMENodePrecision);

bool propagateCastNodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(propagateCastNodes);

bool injectScaleForMMENodes(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(injectScaleForMMENodes);

bool fuseConvertMme(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(fuseConvertMme);

bool removeContinguousConverts(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(removeContinguousConverts);

bool nodeCreatedWithoutOutputShape(HabanaGraph& g);
SET_HABANA_PASS_CAN_RUN_MULTIPLE_TIMES(nodeCreatedWithoutOutputShape);
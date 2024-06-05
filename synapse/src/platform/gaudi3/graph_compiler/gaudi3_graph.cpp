#include "gaudi3_graph.h"

#include "code_generation/code_generator_factory.h"
#include "gaudi3_types.h"
#include "gaudi_scheduler.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "graph_compiler/pass_dependencies/training/passes_dependencies.h"
#include "graph_traits.h"
#include "habana_pass.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "mme_dispatcher.h"
#include "passes.h"
#include "platform/gaudi3/graph_compiler/descriptor_generator.h"
#include "platform/gaudi3/graph_compiler/patch_point_generator.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "recipe_allocator.h"
#include "recipe_generator.h"
#include "rotator_dispatcher.h"
#include "tpc_dispatcher.h"
#include "tpc_fuser.h"
#include "types_exception.h"

#include <algorithm>

using namespace gaudi3;

HabanaGraphPtr instantiateGaudi3Graph()
{
    return std::make_unique<Gaudi3Graph>();
}

Gaudi3Graph::Gaudi3Graph()
: Gaudi3Graph(Gaudi3HalReader::instance()->getSRAMSizeInBytes(), Gaudi3HalReader::instance()->getDRAMSizeInBytes())
{
}

HabanaGraphPtr Gaudi3Graph::clone(bool cloneAllocators, bool keepMappings) const
{
    return HabanaGraphPtr(new Gaudi3Graph(*this, cloneAllocators, keepMappings));
}

Gaudi3Graph::Gaudi3Graph(uint64_t sramSize, uint64_t dramSize)
{
    GlobalConfManager::instance().setDeviceType(getDeviceType());

    m_graphTraits = std::make_shared<GraphTraits>(synDeviceGaudi3);

    m_codeGenerator = CodeGeneratorFactory::createCodeGenerator(getDeviceType(), this);
    m_codeGenerator->initSram(sramSize, getHALReader()->getSRAMBaseAddr());
    m_codeGenerator->setDramSize(dramSize);

    // set DRAM allocation mode
    LOG_DEBUG(GC, "Mark graph as DRAM allocated");
    getGraphAnnotation().memoryStrategyParams.allocatinMode            = ALL_IN_DRAM;
    getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc = true;
}

Gaudi3Graph::Gaudi3Graph(const Gaudi3Graph& other, bool cloneAllocators /*false*/, bool keepMappings /*false*/)
: HabanaGraph(other, cloneAllocators, keepMappings)
{
}

Gaudi3Graph& Gaudi3Graph::operator=(const Gaudi3Graph& other)
{
    if (this != &other)
    {
        HabanaGraph::operator=(other);
        Gaudi3Graph  tmp(other);
        std::swap(m_codeGenerator, tmp.m_codeGenerator);
    }

    return *this;
}

Gaudi3Graph::~Gaudi3Graph() = default;

void Gaudi3Graph::addAllPasses()
{
    LOG_INFO(GC, "Registering passes - Gaudi3");

    HabanaGraph::registerPassGroups();

    // clang-format off

    //                   Name                                ID                                              Dependency Set
    //                   ====                                ==                                              ==============
    REGISTER_HABANA_PASS(graphVisualizationPre,              PASS_ID_GRAPH_VISUALIZATION_PRE,                {GRAPH_VISUALIZATION_PRE_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(extractFunctionalComplexGuidNodes,  PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES,  {EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES_DEPENDENCY_SET} );
    REGISTER_HABANA_PASS(extractPerformanceComplexGuidNodes, PASS_ID_EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES, {EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES_DEPENDENCY_SET});
    REGISTER_HABANA_PASS(fuseBroadcast,                   PASS_ID_FUSE_BROADCAST_TPC,                     {FUSE_BROADCAST_TPC_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_BROADCAST_NODE_CREATED);
    REGISTER_HABANA_PASS(validateUserMemorySections,         PASS_ID_VALIDATE_USER_MEMORY_SECTIONS,          {VALIDATE_USER_MEMORY_SECTIONS_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(validateMemorySectionTensors,       PASS_ID_VALIDATE_MEMORY_SECTION_TENSORS,        {VALIDATE_MEMORY_SECTION_TENSORS_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(internalTensorsDynamicShape,        PASS_ID_INTERNAL_TENSORS_DYNAMIC_SHAPE,         {INTERNAL_TENSORS_DYNAMIC_SHAPE_DEPENDENCY_SET}        )->addPredicate(PREDICATE_ID_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseWaits,                          PASS_ID_FUSE_WAITS,                             {FUSE_WAITS_DEPENDENCY_SET}                            );
    REGISTER_HABANA_PASS(insertSerializeDeserialize,         PASS_ID_INSERT_SERIALIZE_DESERIALIZE,           {INSERT_SERIALIZE_DESERIALIZE_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(handleCtrlEdgesForLogicalNodes,     PASS_ID_HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES,    {HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES_DEPENDENCY_SET}   )->addPredicate(PREDICATE_ID_LOGICAL_NODE_RAN);
    REGISTER_HABANA_PASS(replaceOpsWithLogicalOps,           PASS_ID_REPLACE_OPS_WITH_LOGICAL_OPS,           {REPLACE_OPS_WITH_LOGICAL_OPS_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(convert1x1BatchGemmToGemm,          PASS_ID_CONVERT_1X1BATCH_GEMM_TO_GEMM,          {CONVERT_1X1BATCH_GEMM_TO_GEMM_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(splitTfBatchNorm,                   PASS_ID_SPLIT_TF_BATCH_NORM,                    {SPLIT_TF_BATCH_NORM_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(splitMoments,                       PASS_ID_SPLIT_MOMENTS,                          {SPLIT_MOMENTS_DEPENDENCY_SET}                         );
    REGISTER_HABANA_PASS(checkInputPersistence,              PASS_ID_CHECK_INPUT_PERSISTENCE,                {CHECK_INPUT_PERSISTENCE_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(transposeRemoveRedundantDimensions, PASS_ID_TRANSPOSE_REDUCE_DIMENSIONS,            {TRANSPOSE_REDUCE_DIMENSIONS_DEPENDENCY_SET}           );
    REGISTER_HABANA_PASS(removeContiguousTransposes,         PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES,           {REMOVE_CONTIGUOUS_TRANSPOSES_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED);
    REGISTER_HABANA_PASS(extractMultiNodes,                  PASS_ID_EXTRACT_MULTI_NODES,                    {EXTRACT_MULTI_NODES_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(extractDataMovementMultiNodes,      PASS_ID_EXTRACT_DATA_MOVEMENT_MULTI_NODES,      {EXTRACT_DATA_MOVEMENT_MULTI_NODES_DEPENDENCY_SET}     );
    REGISTER_HABANA_PASS(handleHugeTensors,                  PASS_ID_HANDLE_HUGE_TENSORS,                    {HANDLE_HUGE_TENSORS_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(addMmeBias,                         PASS_ID_ADD_MME_BIAS,                           {ADD_MME_BIAS_DEPENDENCY_SET}                          );
    REGISTER_HABANA_PASS(updateNodesWithAliasTensors,        PASS_ID_UPDATE_NODES_WITH_ALIAS_TENSORS,        {UPDATE_NODES_WITH_ALIAS_TENSORS_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(splitBatchNorm,                     PASS_ID_SPLIT_BATCH_NORM,                       {SPLIT_BATCH_NORM_DEPENDENCY_SET}                      );
    REGISTER_GAUDI3_PASS(loadTpcKernels,                     PASS_ID_LOAD_TPC_KERNELS,                       {LOAD_TPC_KERNELS_DEPENDENCY_SET}                      )->addPredicate(PREDICATE_ID_TPC_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseBatchNorm,                      PASS_ID_FUSE_BATCH_NORM,                        {FUSE_BATCH_NORM_DEPENDENCY_SET}                       );
    REGISTER_HABANA_PASS(tpcFuser,                           PASS_ID_TPC_FUSER,                              {TPC_FUSER_DEPENDENCY_SET}                             );
    REGISTER_HABANA_PASS(optimizeTpcKernels,                 PASS_ID_OPTIMIZE_TPC_KERNELS,                   {OPTIMIZE_TPC_KERNELS_DEPENDENCY_SET}                  )->addPredicate(PREDICATE_ID_TPC_NODE_INITIALIZED);
    REGISTER_HABANA_PASS(splitLayerNormBwd,                  PASS_ID_SPLIT_LAYER_NORM_BWD,                   {SPLIT_LAYER_NORM_BWD_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(inPlaceInputReuseBinding,           PASS_ID_IN_PLACE_INPUT_REUSE_BINDING,           {IN_PLACE_INPUT_REUSE_BINDING_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_TPC_NODE_INITIALIZED);
    REGISTER_HABANA_PASS(inPlaceInputReuseSuggestion,        PASS_ID_IN_PLACE_INPUT_REUSE_SUGGESTION,        {IN_PLACE_INPUT_REUSE_SUGGESTION_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(removeContiguousReshapeNodes,       PASS_ID_REMOVE_CONTIGUOUS_RESHAPES,             {REMOVE_CONTIGUOUS_RESHAPES_DEPENDENCY_SET}            )->addPredicate(PREDICATE_ID_RESHAPE_NODE_CREATED);
    REGISTER_HABANA_PASS(removeContiguousCastNodes,          PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES,           {REMOVE_CONTIGUOUS_CAST_NODES_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_CAST_NODE_CREATED);
    REGISTER_HABANA_PASS(eliminateNodesWithStaticInputs,     PASS_ID_ELIMINATE_NODES_WITH_STATIC_INPUTS,     {ELIMINATE_NODES_WITH_STATIC_INPUTS_DEPENDENCY_SET}    )->addPredicate(PREDICATE_ID_NODE_CREATED_CONST_INPUT);
    REGISTER_HABANA_PASS(calcDynamicRange,                   PASS_ID_CALC_DYNAMIC_RANGE,                     {CALC_DYNAMIC_RANGE_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(updateMMENodePrecision,             PASS_ID_UPDATE_MME_NODES_PRECISION,             {UPDATE_MME_NODES_PRECISION_DEPENDENCY_SET}            )->addPredicate(PREDICATE_ID_EXTERNAL_MME_NODE_CREATED);
    REGISTER_HABANA_PASS(propagateCastNodes,                 PASS_ID_PROPAGATE_CAST_NODES,                   {PROPAGATE_CAST_NODES_DEPENDENCY_SET}                  )->addPredicate(PREDICATE_ID_EXTERNAL_MME_NODE_CREATED);
    REGISTER_HABANA_PASS(removeContinguousConverts,          PASS_ID_REMOVE_CONTIGUOUS_CONVERTS,             {REMOVE_CONTIGUOUS_CONVERTS_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(calcQuantizationInfo,               PASS_ID_CALC_QUANTIZATION_INFO,                 {CALC_QUANTIZATION_INFO_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(adjustRestrictions,                 PASS_ID_ADJUST_RESTRICTIONS,                    {ADJUST_RESTRICTIONS_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(updatePadQuantizer,                 PASS_ID_UPDATE_PAD_QUANTIZER,                   {UPDATE_PAD_QUANTIZER_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(adjustScales,                       PASS_ID_ADJUST_SCALES,                          {ADJUST_SCALES_DEPENDENCY_SET}                         );
    REGISTER_HABANA_PASS(enforceNodePrecision,               PASS_ID_ENFORCE_NODE_PRECISION,                 {ENFORCE_NODE_PRECISION_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(validateQuantization,               PASS_ID_VALIDATE_QUANTIZATION,                  {VALIDATE_QUANTIZATION_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(removeZeroSizedPad,                 PASS_ID_REMOVE_ZERO_SIZED_PAD,                  {REMOVE_ZERO_SIZED_PAD_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(staticTensorsCastInsert,            PASS_ID_STATIC_TENSOR_CAST_INSERTION,           {STATIC_TENSOR_CAST_INSERTION_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(memsetNodeOutput,                   PASS_ID_MEMSET_NODE_OUTPUT,                     {MEMSET_NODE_OUTPUT_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(handleMemoryReuse,                  PASS_ID_HANDLE_MEMORY_REUSE,                    {HANDLE_MEMORY_REUSE_DEPENDENCY_SET}                   )->addPredicate(PREDICATE_ID_LOGICAL_NODE_RAN);
    REGISTER_GAUDI3_PASS(selectMemcpyEngine,                 PASS_ID_SELECT_MEMCPY_ENGINE,                   {SELECT_MEMCOPY_ENGINE_DEPENDENCY_SET}                 )->addPredicate(PREDICATE_ID_MEMCPY_NODE_CREATED);
    REGISTER_HABANA_PASS(identifyMmeConcurrency,             PASS_ID_MME_CONCURRENCY_IDENTIFIER,             {MME_CONCURRENCY_IDENTIFIER_DEPENDENCY_SET}             );
    REGISTER_HABANA_PASS(applyMmeConcurrencyMemset,          PASS_ID_MME_CONCURRENCY_MEMSET,                 {MME_CONCURRENCY_MEMSET_DEPENDENCY_SET}             );
    REGISTER_HABANA_PASS(handleLogicalOps,                   PASS_ID_HANDLE_LOGICAL_OPERATIONS,              {HANDLE_LOGICAL_OPERATIONS_DEPENDENCY_SET}             )->addPredicate(PREDICATE_ID_LOGICAL_NODE_CREATED);
    REGISTER_HABANA_PASS(handleLogicalOpsPreProcess,         PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS,  {HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS_DEPENDENCY_SET} );
    REGISTER_HABANA_PASS(handleLogicalOpsPostProcess,        PASS_ID_HANDLE_LOGICAL_OPERATIONS_POST_PROCESS, {HANDLE_LOGICAL_OPERATIONS_POST_PROCESS_DEPENDENCY_SET});
    REGISTER_HABANA_PASS(removeOppositeConcatSplitSequence,  PASS_ID_REMOVE_OPPOSITE_SPLIT_CONCAT,           {REMOVE_OPPOSITE_SPLIT_CONCAT_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(replaceGroupConvFilter2d,           PASS_ID_REPLACE_GROUP_CONV_FILTER2D,            {REPLACE_GROUP_CONV_FILTER2D_DEPENDENCY_SET}           );
    REGISTER_HABANA_PASS(handleGroupedConvolutions,          PASS_ID_HANDLE_GROUPED_CONVOLUTIONS,            {HANDLE_GROUPED_CONVOLUTIONS_DEPENDENCY_SET}           );
    REGISTER_HABANA_PASS(fuseBatchNormMemCpy,                PASS_ID_FUSE_BATCH_NORM_MEMCPY,                 {FUSE_BATCH_NORM_MEMCPY_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(fuseTransposeMme,                   PASS_ID_FUSE_TRANSPOSE_MME,                     {FUSE_TRANSPOSE_MME_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_FUSED_NODE_TO_MME)->addPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED)->addPredicate(PREDICATE_ID_PHYSICAL_TRANSPOSE_NODE_CREATED)->addPredicate(PREDICATE_ID_EXTERNAL_MME_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseConstantTranspose,              PASS_ID_FUSE_CONST_TRANSPOSE,                   {FUSE_CONST_TRANSPOSE_DEPENDENCY_SET}                  )->addPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED);
    REGISTER_HABANA_PASS(removeRedundantMemcpyNodes,         PASS_ID_REMOVE_REDUNDANT_MEMCPY_NODES,          {REMOVE_REDUNDANT_MEMCPY_NODES_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(removeRedundantLogicalNodes,        PASS_ID_REMOVE_REDUNDANT_LOGICAL_NODES,         {REMOVE_REDUNDANT_LOGICAL_NODES_DEPENDENCY_SET}        )->addPredicate(PREDICATE_ID_LOGICAL_NODE_CREATED);
    REGISTER_HABANA_PASS(packingMmeNodes,                    PASS_ID_PACKING_MME_NODES,                      {PACKING_MME_NODES_DEPENDENCY_SET}                     );
    REGISTER_HABANA_PASS(splitFrobeniusLayerNorm,            PASS_ID_SPLIT_FROBENIUS_LAYER_NORM,             {SPLIT_FROBENIUS_LAYER_NORM_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(removeZeroSizedTensors,             PASS_ID_REMOVE_ZERO_SIZED_TENSORS,              {REMOVE_ZERO_SIZED_TENSORS_DEPENDENCY_SET}             )->addPredicate(PREDICATE_ID_MEMCPY_NODE_CREATED);
    REGISTER_HABANA_PASS(registerMemoryCoherence,            PASS_ID_REGISTER_MEM_COHERENCE,                 {REGISTER_MEM_COHERENCE_DEPENDENCY_SET}                )->addPredicate(PREDICATE_ID_MEMORY_SECTION_TENSOR_CREATED);
    REGISTER_HABANA_PASS(sliceGraphForPipeline,              PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,           {SLICE_GRAPH_TO_SRAM_CAPACITY_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(runLayeredBrain,                    PASS_ID_RUN_LAYERED_BRAIN,                      {RUN_LAYERED_BRAIN_DEPENDENCY_SET});
    REGISTER_HABANA_PASS(bundleNodesSchedule,                PASS_ID_BUNDLE_NODES_SCHEDULE,                  {BUNDLE_NODES_SCHEDULE_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(bundleMemoryManagement,             PASS_ID_BUNDLE_MEMORY_MANAGEMENT,               {BUNDLE_MEMORY_MANAGEMENT_DEPENDENCY_SET}              );
    REGISTER_HABANA_PASS(handlePartialsWrites,               PASS_ID_HANDLE_PARTIALS_WRITES,                 {HANDLE_PARTIALS_WRITES_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(eliminateRedundantNodes,            PASS_ID_ELIMINATE_REDUNDANT_NODES,              {ELIMINATE_REDUNDANT_NODES_DEPENDENCY_SET}             )->addPredicate(PREDICATE_ID_ELIMINATE_REDUNDANT_NODES);
    REGISTER_HABANA_PASS(markReductionInputs,                PASS_ID_MARK_REDUCTION_INPUTS,                  {MARK_REDUCTION_INPUTS_DEPENDENCY_SET}                 )->addPredicate(PREDICATE_ID_REDUCTION_NODE_CREATED);
    REGISTER_HABANA_PASS(setReductionMemset,                 PASS_ID_SET_REDUCTION_MEMSET,                   {SET_REDUCTION_MEMSET_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(relaxCtrlDeps,                      PASS_ID_CONTROL_DEP_RELAXATION,                 {HANDLE_CONTROL_DEP_RELAXATION_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(spillPersistentTensors,             PASS_ID_SPILL_PERSISTENT_TENSORS,               {SPILL_PERSISTENT_TENSORS_DEPENDENCY_SET}              );
    REGISTER_HABANA_PASS(fuseCastMme,                        PASS_ID_FUSE_CAST_MME,                          {FUSE_CAST_MME_DEPENDENCY_SET}                         )->addPredicate(PREDICATE_ID_FUSED_NODE_TO_MME);
    REGISTER_HABANA_PASS(fuseConvertMme,                     PASS_ID_FUSE_CONVERT_MME,                       {FUSE_CONVERT_MME_DEPENDENCY_SET}                      )->addPredicate(PREDICATE_ID_FUSED_NODE_TO_MME);
    REGISTER_HABANA_PASS(alignAsymmetricBgemm,               PASS_ID_ALIGN_ASYMMETRIC_BGEMM,                 {ALIGN_ASYMMETRIC_BGEMM_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(lowerDedx,                          PASS_ID_LOWER_DEDX,                             {LOWER_DEDX_DEPENDENCY_SET}                            );

    REGISTER_HABANA_PASS(allocateTensors,                    PASS_ID_ALLOCATE_TENSORS,                       {ALLOCATE_TENSORS_DEPENDENCY_SET}                      );
    REGISTER_HABANA_PASS(validateMemoryAllocation,           PASS_ID_VALIDATE_MEMORY_ALLOCATION,             {VALIDATE_MEMORY_ALLOCATION_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(setSupportedLayouts,                PASS_ID_SET_SUPPORTED_LAYOUTS,                  {SET_SUPPORTED_LAYOUTS_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(setHabanaLayouts,                   PASS_ID_SET_HABANA_LAYOUTS,                     {SET_HABANA_LAYOUTS_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(optimizeStridedInsert,              PASS_ID_OPTIMIZE_STRIDED_INSERT,                {OPTIMIZE_STRIDED_INSERT_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(adjustDataLayout,                   PASS_ID_ADJUST_DATA_LAYOUT,                     {ADJUST_DATA_LAYOUT_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(transposeDontCareNodes,             PASS_ID_TRANSPOSE_DONT_CARE_NODES,              {TRANSPOSE_DONT_CARE_NODES_DEPENDENCY_SET}             );
    REGISTER_HABANA_PASS(handlePermutedTensors,              PASS_ID_HANDLE_PERMUTED_TENSORS,                {HANDLE_PERMUTED_TENSORS_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(linkReductionMemsetShapes,          PASS_ID_LINK_REDUCTION_MEMSET_SHAPES,           {LINK_REDUCTION_MEMSET_SHAPES_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_NODE_CREATED);
    REGISTER_HABANA_PASS(verifyMemsetOutputShapes,           PASS_ID_VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE,   {VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE_DEPENDENCY_SET}  );
    REGISTER_HABANA_PASS(validateDynamicShapes,              PASS_ID_VALIDATE_DYNAMIC_SHAPES,                {VALIDATE_DYNAMIC_SHAPES_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(generateROIs,                       PASS_ID_GENERATE_ROIS,                          {GENERATE_ROIS_DEPENDENCY_SET}                         );
    REGISTER_HABANA_PASS(splitToDcoreROIs,                   PASS_ID_SPLIT_TO_DCORE_ROIS,                    {SPLIT_TO_DCORE_ROIS_DEPENDENCY_SET}                   );
    REGISTER_GAUDI3_PASS(generateMmeDescriptors,             PASS_ID_GENERATE_MME_DESCRIPTORS,               {GENERATE_MME_DESCRIPTORS_DEPENDENCY_SET}              );
    REGISTER_HABANA_PASS(splitTPCDims,                       PASS_ID_SPLIT_TPC_DIMS,                         {SPLIT_TPC_DIMS_DEPENDENCY_SET}                        );
    REGISTER_HABANA_PASS(disableBundleRois,                  PASS_ID_DISABLE_BUNDLE_ROIS,                    {DISABLE_BUNDLE_ROIS_DEPENDENCY_SET}                   );
    REGISTER_GAUDI3_PASS(splitToLogicalROIs,                 PASS_ID_SPLIT_TO_LOGICAL_ROIS,                  {SPLIT_TO_LOGICAL_ROIS_DEPENDENCY_SET}                 );
    REGISTER_GAUDI3_PASS(patchMmeDescriptors,                PASS_ID_PATCH_MME_DESCRIPTORS,                  {PATCH_MME_DESCRIPTORS_DEPENDENCY_SET}                 );
    REGISTER_GAUDI3_PASS(patchMmeMcids,                      PASS_ID_PATCH_MME_MCIDS,                        {PATCH_MME_MCIDS_DEPENDENCY_SET}                       );
    REGISTER_HABANA_PASS(projectNodeROIs,                    PASS_ID_PROJECT_NODE_ROIS,                      {PROJECT_NODE_ROIS_DEPENDENCY_SET}                     );
    REGISTER_HABANA_PASS(validateNodesLayout,                PASS_ID_VALIDATE_NODES_LAYOUT,                  {VALIDATE_NODES_LAYOUT_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(generateProfilerDebugInfo,          PASS_ID_GENERATE_PROFILER_DEBUG_INFO,           {GENERATE_PROFILER_DEBUG_INFO_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(assignAddressesToTensorROIs,        PASS_ID_ASSIGN_ADDRESSES_TO_TENSOR_ROIS,        {ASSIGN_ADDRESSES_TO_TENSOR_ROIS_DEPENDENCY_SET}       );
    REGISTER_GAUDI3_PASS(allocateSyncs,                      PASS_ID_ALLOCATE_SYNCS,                         {ALLOCATE_SYNCS_DEPENDENCY_SET}                        );
    REGISTER_HABANA_PASS(splitToPhysicalROIs,                PASS_ID_SPLIT_TO_PHYSICAL_ROIS,                 {SPLIT_TO_PHYSICAL_ROIS_DEPENDENCY_SET}                );
    REGISTER_GAUDI3_PASS(calculateTensorROIsLinearRanges,    PASS_ID_CALCULATE_TENSOR_ROIS_LINEAR_RANGES,    {CALCULATE_TENSOR_ROIS_LINEAR_RANGES_DEPENDENCY_SET}   );
    REGISTER_GAUDI3_PASS(generateCacheMaitenanceTasks,       PASS_ID_GENERATE_CACHE_MAINTENANCE_TASKS,       {GENERATE_CACHE_MAINTENANCE_TASKS_DEPENDENCY_SET}      );
    REGISTER_GAUDI3_PASS(manageBaseRegsCache,                PASS_ID_MANAGE_BASE_REGS_CACHE,                 {MANAGE_BASE_REGS_CACHE_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(generateWorkDistribution,           PASS_ID_GENERATE_WORK_DISTRIBUTION,             {GENERATE_WORK_DISTRIBUTION_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(validateMMENodes,                   PASS_ID_VALIDATE_MME_NODES,                     {VALIDATE_MME_NODES_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(InitMmeBrainIfc,                    PASS_ID_INIT_MME_BRAIN_IFC,                     {INIT_MME_BRAIN_IFC_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_MME_NODE_CREATED);
    REGISTER_HABANA_PASS(validateAtomicNodes,                PASS_ID_VALIDATE_ATOMIC_NODES,                  {VALIDATE_ATOMIC_NODES_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(validateExecutionScheduleBundles,   PASS_ID_VALIDATE_EXECUTION_SCHEDULE_BUNDLES,    {VALIDATE_EXECUTION_SCHEDULE_BUNDLES_DEPENDENCY_SET}   );
    REGISTER_HABANA_PASS(setNonPersistentSectionInfo,        PASS_ID_SET_NON_PERSISTENT_SECTION_INFO,        {SET_NON_PERSISTENT_SECTION_INFO_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(handleTpcRmwKernels,                PASS_ID_HANDLE_RMW_TPC_KERNELS,                 {HANDLE_RMW_TPC_KERNELS_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(graphVisualizationPost,             PASS_ID_GRAPH_VISUALIZATION_POST,               {GRAPH_VISUALIZATION_POST_DEPENDENCY_SET}              );
    REGISTER_GAUDI3_PASS(allocateTpcKernels,                 PASS_ID_ALLOCATE_TPC_KERNELS,                   {ALLOCATE_TPC_KERNELS_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(handleBroadcastBatchGemm,           PASS_ID_HANDLE_PARTIAL_BROADCAST_BGEMM,         {HANDLE_PARTIAL_BROADCAST_BGEMM_DEPENDENCY_SET}        );
    REGISTER_GAUDI3_PASS(signalOutFromGraph,                 PASS_ID_SIGNAL_OUT_FROM_GRAPH,                  {SIGNAL_OUT_FROM_GRAPH_DEPENDENCY_SET}                 );
    REGISTER_GAUDI3_PASS(setTensorInHbm,                     PASS_ID_SET_TENSOR_IN_HBM,                      {SET_TENSOR_IN_HBM_DEPENDENCY_SET}                     );
    REGISTER_GAUDI3_PASS(alignTransposeViaGemmOutput,        PASS_ID_ALIGN_TRANSPOSE_VIA_GEMM_OUTPUT,        {ALIGN_TRANSPOSE_VIA_GEMM_OUTPUT_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(gcPerfChecks,                       PASS_ID_GC_PERF_CHECKS,                         {GC_PERF_CHECKS_DEPENDENCY_SET}                        );
    REGISTER_HABANA_PASS(injectScaleForMMENodes,             PASS_ID_INJECT_SCALE_FOR_MME_NODES,             {INJECT_SCALE_FOR_MME_NODES_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(addStaticShapeTensors,              PASS_ID_ADD_STATIC_SHAPE_TENSORS,               {ADD_STATIC_SHAPE_TENSORS_DEPENDENCY_SET}              );
    REGISTER_HABANA_PASS(validatePreSlicingSizes,            PASS_ID_VALIDATE_PRE_SLICING_SIZES,             {VALIDATE_PRE_SLICING_SIZES_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(alignBPTFCDStrideToCacheLine,       PASS_ID_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE,      {ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_DEPENDENCY_SET}     );
    REGISTER_HABANA_PASS(optimizeMemcpyNodes,                PASS_ID_OPTIMIZE_MEMCPY_NODES,                  {OPTIMIZE_MEMCPY_NODES_DEPENDENCY_SET}                 )->addPredicate(PREDICATE_ID_TPC_MEMCPY_NODE_CREATED);
    REGISTER_HABANA_PASS(scheduleFlashAttentionNodes,        PASS_ID_SET_FLASH_ATTN_SCHEDULE,                {SET_FLASH_ATTN_SCHEDULE_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(nodeCreatedWithoutOutputShape,      PASS_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE,      {CHECK_MAX_DIMS_PRE_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE);
    // clang-format on
}

bool Gaudi3Graph::compile()
{
    CompilationHalReaderSetter compHalReaderSetter(this);

    setTensorsAlignment();

    try
    {
        addAllPasses();
    }
    catch (PassFailedException& ex)
    {
        LOG_ERR(GC, "Registering all passes failed");
        return false;
    }

    if (isCompiled())
    {
        LOG_ERR(GC, "Graph re-compilation is not allowed.");
        return false;
    }

    setCompiled();

    if (!isValidConfig())
    {
        return false;
    }

    saveUsedGcfgFile();
    printGlobalConfigurations();

    if (!m_codeGenerator->init())
    {
        return false;
    }

    if (!generateExecutionSchedule())
    {
        return false;
    }

    LOG_DEBUG(GRAPH_DATA, "Initial graph data");
    PrintNodesAndOperands();

    collectConstSectionAndPersistentTensors();

    if (!runPassManager())
    {
        return false;
    }

    LOG_DEBUG(GRAPH_DATA, "Final graph data");
    PrintNodesAndOperands();

    LOG_DEBUG(GC, "Done optimizing graph");

    m_codeGenerator->generate(this);
    addAllDescriptors();

    m_codeGenerator->initQueues();
    m_codeGenerator->fillQueues();

    m_codeGenerator->generateRecipes(*this); //temporary

    return true;
}

bool Gaudi3Graph::graphSupports64BitDataTypes() const
{
    return true;
}

bool Gaudi3Graph::isValidConfig() const
{
    return true;
}

void Gaudi3Graph::addAllDescriptors()
{
    gaudi3::DescriptorGenerator generator(this);
    for (auto n : getExeSortedNodes())  // must be in execution order
    {
        n->accept(&generator);
    }
}

gaudi3::TpcDescriptorsWrappers& Gaudi3Graph::getTpcNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_tpcNodesDescriptorsWrappers[n];
}

gaudi3::MmeDescriptorsWrappers& Gaudi3Graph::getMmeNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_mmeNodesDescriptorsWrappers[n];
}

gaudi3::RotatorDescriptorsWrappers& Gaudi3Graph::getRotateNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_rotNodesDescriptorsWrappers[n];
}

recipe_t* Gaudi3Graph::serializeDataPlane(RecipeAllocator* recipeAlloc) const
{
    return m_codeGenerator->serializeDataPlane(recipeAlloc);
}

MmeDescriptorGenerator& Gaudi3Graph::getMmeNodeDescriptorGenerator(const NodePtr& n)
{
    auto descGenerator = m_mmeNodesDescriptorGenerator.find(n);
    HB_ASSERT(descGenerator != m_mmeNodesDescriptorGenerator.end(), "object not in map");
    return *descGenerator->second;
}

void Gaudi3Graph::updateMmeNodeDescriptorWrapper(const MmeNode& node, const gaudi3::MmeDesc& desc, const McidMmeUsage &mcidMmeUsage, NodeROI& roi, unsigned engineIdx)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT(nodeShared !=  nullptr, "Invalid node object");

    MmeDescWrapper descWrapper(desc);
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.getBasicFieldsContainerInfoForCtx().setRoi(&roi);

    Gaudi3MMEPatchPointGenerator ppGenerator;

    ppGenerator.generateMmePatchPoints(node, getMmeNodeDescriptorGenerator(nodeShared), descWrapper, mcidMmeUsage, engineIdx);
    // [CID: 42190] False positive - coverity ignores std::map and std::set default c'tor
    getMmeNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void Gaudi3Graph::updateTPCDescriptorWrapper(const TPCNode&                        node,
                                             const gaudi3::TpcDesc&                tpcDescriptor,
                                             const ValidityMask<gaudi3::TpcDesc>&  tpcMask,
                                             const TpcFwCtxVector&                 tpcFwCtxs,
                                             NodeROI&                              roi,
                                             TpcDescriptorGenerator::McidTpcUsage& mcidTpcUsage)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);
    TpcDescWrapper descWrapper(tpcDescriptor, tpcMask);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.getBasicFieldsContainerInfoForCtx().setRoi(&roi);
    for (const tpc_wd_ctxt_t& ctx : tpcFwCtxs)
    {
        descWrapper.addFwCtx(ctx);
    }

    Gaudi3TPCPatchPointGenerator ppGenerator;
    ppGenerator.setMcidTpcUsage(mcidTpcUsage);
    ppGenerator.generateTpcPatchPoints(node, descWrapper);
    // [CID: 42188] False positive - coverity ignores std::map and std::set default c'tor
    getTpcNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void Gaudi3Graph::setMmeNodeDescriptorGenerator(const NodePtr& n, MmeDescriptorGeneratorPtr descGenerator)
{
    HB_ASSERT(m_mmeNodesDescriptorGenerator.find(n) == m_mmeNodesDescriptorGenerator.end(), "object already in map");
    m_mmeNodesDescriptorGenerator.emplace(n, descGenerator);
}

void Gaudi3Graph::updateRotatorDescriptorWrapper(const RotateNode&                        node,
                                                 const gaudi3::RotatorDesc&               rotatorDescriptor,
                                                 const ValidityMask<gaudi3::RotatorDesc>& rotateMask,
                                                 const rot_wd_ctxt_t&                     rotFwCtx,
                                                 NodeROI&                                 roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);

    RotatorDescWrapper descWrapper(rotatorDescriptor, rotateMask);
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.setFwCtx(rotFwCtx);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;

    Gaudi3RotatorPatchPointGenerator ppGenerator;
    ppGenerator.generateRotatorPatchPoints(node, descWrapper);
    // [CID: 42186] False positive - coverity ignores std::map and std::set default c'tor
    getRotateNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

bool Gaudi3Graph::generateExecutionSchedule() const
{
    GaudiScheduler scheduler(this);
    return HabanaGraph::generateExecutionSchedule(&scheduler);
}

bool Gaudi3Graph::validateMemorySection(const InternalSectionHandle* section) const
{
    HB_ASSERT(section != nullptr, "Unexpected empty section handle");

    if (section->getPersistent() && section->getRMW())
    {
        return false;
    }

    return true;
}

void Gaudi3Graph::setLogicalQueueToMaxExecutionIndex()
{
    for (const NodePtr& node : getExeSortedNodes())
    {
        if (node == nullptr) return;

        HabanaDeviceType deviceType = node->getNodeDeviceType();
        if (deviceType != LAST_HABANA_DEVICE)
        {
            unsigned queueId = gaudi3::deviceTypeToLogicalQueue(deviceType, *node);
            m_logicalQueueToMaxExecutionIndex[queueId] = node->getExecutionOrderedIndex();
        }
    }
}

shape_plane_graph_t* Gaudi3Graph::serializeShapePlane(RecipeAllocator* recipeAlloc) const
{
    return m_codeGenerator->serializeShapePlane(recipeAlloc);
}

unsigned Gaudi3Graph::getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode) const
{
    if (getHALReader()->isRotateAngleSupported(rotateNode->getRotationAngle()))
    {
        return getHALReader()->getRotateStripeWidth();
    }

    unsigned stripeWidth = gaudi3::DescriptorGenerator::getRotateStripeWidth(rotateNode);

    LOG_DEBUG(GC,
              "Rotation angle: {} is unsupported  - calculated stripe width: {}",
              rotateNode->getRotationAngle(),
              stripeWidth);

    return stripeWidth;
}

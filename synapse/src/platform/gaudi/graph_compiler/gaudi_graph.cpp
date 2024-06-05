#include "gaudi_graph.h"

#include "code_generation/code_generator_factory.h"
#include "command_queue.h"
#include "defs.h"
#include "gaudi/asic_reg_structs/mme_regs.h"
#include "gaudi_scheduler.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "graph_compiler/pass_dependencies/training/passes_dependencies.h"
#include "graph_traits.h"
#include "habana_global_conf.h"
#include "habana_nodes/node_factory.h"
#include "habana_pass.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "node.h"
#include "passes.h"
#include "platform/gaudi/graph_compiler/descriptor_generator.h"
#include "platform/gaudi/graph_compiler/dma_dispatcher.h"
#include "platform/gaudi/graph_compiler/mme_dispatcher.h"
#include "platform/gaudi/graph_compiler/patch_point_generator.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "platform/gaudi/graph_compiler/tpc_dispatcher.h"
#include "section_handle.hpp"
#include "syn_logging.h"
#include "synapse_types.h"
#include "sync/sync_conventions.h"
#include "tpc_fuser.h"
#include "types_exception.h"
#include "utils.h"

#include <deque>

using namespace gaudi;

HabanaGraphPtr instantiateGaudiGraph()
{
    return std::make_unique<GaudiGraph>();
}

HabanaGraphPtr GaudiGraph::clone(bool cloneAllocators, bool keepMappings) const
{
    return HabanaGraphPtr(new GaudiGraph(*this, cloneAllocators, keepMappings));
}

GaudiGraph::GaudiGraph()
{
    GlobalConfManager::instance().setDeviceType(getDeviceType());

    m_graphTraits          = std::make_shared<GraphTraits>(synDeviceGaudi);
    m_codeGenerator        = CodeGeneratorFactory::createCodeGenerator(getDeviceType(), this);
    initGaudiHalDepMembers();

    // set DRAM allocation mode
    LOG_DEBUG(GC, "Mark graph as DRAM allocated");
    getGraphAnnotation().memoryStrategyParams.allocatinMode = ALL_IN_DRAM;
    getGraphAnnotation().memoryStrategyParams.dramInfo.unitMatricesInDram = true;
    getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc = true;
}

GaudiGraph::GaudiGraph(const GaudiGraph& other, bool cloneAllocators /*false*/, bool keepMappings /*false*/)
: HabanaGraph(other, cloneAllocators, keepMappings)
{
}

GaudiGraph& GaudiGraph::operator=(const GaudiGraph& other)
{
    if (this != &other)
    {
        HabanaGraph::operator=(other);
        GaudiGraph tmp(other);
        std::swap(m_codeGenerator, tmp.m_codeGenerator);
    }
    return *this;
}

GaudiGraph::~GaudiGraph() = default;

void GaudiGraph::initGaudiHalDepMembers()
{
    m_codeGenerator->initSram(getHALReader()->getSRAMSizeInBytes(), getHALReader()->getSRAMBaseAddr());
    m_codeGenerator->setDramSize(getHALReader()->getDRAMSizeInBytes());
}

void GaudiGraph::addAllPasses()
{
    LOG_INFO(GC, "Registering passes - Gaudi");

    // clang-format off

    HabanaGraph::registerPassGroups();
    //                   Name                                ID                                              Dependency Set
    //                   ====                                ==                                              ==============
    REGISTER_HABANA_PASS(graphVisualizationPre,              PASS_ID_GRAPH_VISUALIZATION_PRE,                {GRAPH_VISUALIZATION_PRE_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(fuseBroadcast,                   PASS_ID_FUSE_BROADCAST_TPC,                     {FUSE_BROADCAST_TPC_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_BROADCAST_NODE_CREATED);
    REGISTER_HABANA_PASS(extractFunctionalComplexGuidNodes,  PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES,  {EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES_DEPENDENCY_SET} );
    REGISTER_HABANA_PASS(extractPerformanceComplexGuidNodes, PASS_ID_EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES, {EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES_DEPENDENCY_SET});
    REGISTER_HABANA_PASS(validateUserMemorySections,         PASS_ID_VALIDATE_USER_MEMORY_SECTIONS,          {VALIDATE_USER_MEMORY_SECTIONS_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(validateMemorySectionTensors,       PASS_ID_VALIDATE_MEMORY_SECTION_TENSORS,        {VALIDATE_MEMORY_SECTION_TENSORS_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(internalTensorsDynamicShape,        PASS_ID_INTERNAL_TENSORS_DYNAMIC_SHAPE,         {INTERNAL_TENSORS_DYNAMIC_SHAPE_DEPENDENCY_SET}        )->addPredicate(PREDICATE_ID_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseWaits,                          PASS_ID_FUSE_WAITS,                             {FUSE_WAITS_DEPENDENCY_SET}                            );
    REGISTER_HABANA_PASS(insertSerializeDeserialize,         PASS_ID_INSERT_SERIALIZE_DESERIALIZE,           {INSERT_SERIALIZE_DESERIALIZE_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(relaxCtrlDeps,                      PASS_ID_CONTROL_DEP_RELAXATION,                 {HANDLE_CONTROL_DEP_RELAXATION_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(spillPersistentTensors,             PASS_ID_SPILL_PERSISTENT_TENSORS,               {SPILL_PERSISTENT_TENSORS_DEPENDENCY_SET}              );
    REGISTER_HABANA_PASS(handleCtrlEdgesForLogicalNodes,     PASS_ID_HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES,    {HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES_DEPENDENCY_SET}   )->addPredicate(PREDICATE_ID_LOGICAL_NODE_RAN);
    REGISTER_HABANA_PASS(replaceOpsWithLogicalOps,           PASS_ID_REPLACE_OPS_WITH_LOGICAL_OPS,           {REPLACE_OPS_WITH_LOGICAL_OPS_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(convert1x1BatchGemmToGemm,          PASS_ID_CONVERT_1X1BATCH_GEMM_TO_GEMM,          {CONVERT_1X1BATCH_GEMM_TO_GEMM_DEPENDENCY_SET}         );
    REGISTER_HABANA_PASS(splitTfBatchNorm,                   PASS_ID_SPLIT_TF_BATCH_NORM,                    {SPLIT_TF_BATCH_NORM_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(splitMoments,                       PASS_ID_SPLIT_MOMENTS,                          {SPLIT_MOMENTS_DEPENDENCY_SET}                         );
    REGISTER_HABANA_PASS(checkInputPersistence,              PASS_ID_CHECK_INPUT_PERSISTENCE,                {CHECK_INPUT_PERSISTENCE_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(transposeRemoveRedundantDimensions, PASS_ID_TRANSPOSE_REDUCE_DIMENSIONS,            {TRANSPOSE_REDUCE_DIMENSIONS_DEPENDENCY_SET}     );
    REGISTER_HABANA_PASS(removeContiguousTransposes,         PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES,           {REMOVE_CONTIGUOUS_TRANSPOSES_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED);
    REGISTER_HABANA_PASS(removeOppositeConcatSplitSequence,  PASS_ID_REMOVE_OPPOSITE_SPLIT_CONCAT,           {REMOVE_OPPOSITE_SPLIT_CONCAT_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(extractMultiNodes,                  PASS_ID_EXTRACT_MULTI_NODES,                    {EXTRACT_MULTI_NODES_DEPENDENCY_SET}             );
    REGISTER_HABANA_PASS(extractDataMovementMultiNodes,      PASS_ID_EXTRACT_DATA_MOVEMENT_MULTI_NODES,      {EXTRACT_DATA_MOVEMENT_MULTI_NODES_DEPENDENCY_SET}     );
    REGISTER_HABANA_PASS(addMmeBias,                         PASS_ID_ADD_MME_BIAS,                           {ADD_MME_BIAS_DEPENDENCY_SET}                          );
    REGISTER_HABANA_PASS(handleHugeTensors,                  PASS_ID_HANDLE_HUGE_TENSORS,                    {HANDLE_HUGE_TENSORS_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(updateNodesWithAliasTensors,        PASS_ID_UPDATE_NODES_WITH_ALIAS_TENSORS,        {UPDATE_NODES_WITH_ALIAS_TENSORS_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(splitBatchNorm,                     PASS_ID_SPLIT_BATCH_NORM,                       {SPLIT_BATCH_NORM_DEPENDENCY_SET}                      );
    REGISTER_HABANA_PASS(markReductionInputs,                PASS_ID_MARK_REDUCTION_INPUTS,                  {MARK_REDUCTION_INPUTS_DEPENDENCY_SET}                 )->addPredicate(PREDICATE_ID_REDUCTION_NODE_CREATED);
    REGISTER_GAUDI_PASS( loadTpcKernels,                     PASS_ID_LOAD_TPC_KERNELS,                       {LOAD_TPC_KERNELS_DEPENDENCY_SET}                      )->addPredicate(PREDICATE_ID_TPC_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseBatchNorm,                      PASS_ID_FUSE_BATCH_NORM,                        {FUSE_BATCH_NORM_DEPENDENCY_SET}                       );
    REGISTER_HABANA_PASS(tpcFuser,                           PASS_ID_TPC_FUSER,                              {TPC_FUSER_DEPENDENCY_SET}                             );
    REGISTER_HABANA_PASS(optimizeTpcKernels,                 PASS_ID_OPTIMIZE_TPC_KERNELS,                   {OPTIMIZE_TPC_KERNELS_DEPENDENCY_SET}                  )->addPredicate(PREDICATE_ID_TPC_NODE_INITIALIZED);
    REGISTER_HABANA_PASS(splitLayerNormBwd,                  PASS_ID_SPLIT_LAYER_NORM_BWD,                   {SPLIT_LAYER_NORM_BWD_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(inPlaceInputReuseBinding,           PASS_ID_IN_PLACE_INPUT_REUSE_BINDING,           {IN_PLACE_INPUT_REUSE_BINDING_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_TPC_NODE_INITIALIZED);
    REGISTER_HABANA_PASS(inPlaceInputReuseSuggestion,        PASS_ID_IN_PLACE_INPUT_REUSE_SUGGESTION,        {IN_PLACE_INPUT_REUSE_SUGGESTION_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(removeContiguousReshapeNodes,       PASS_ID_REMOVE_CONTIGUOUS_RESHAPES,             {REMOVE_CONTIGUOUS_RESHAPES_DEPENDENCY_SET}            )->addPredicate(PREDICATE_ID_RESHAPE_NODE_CREATED);
    REGISTER_HABANA_PASS(removeContiguousCastNodes,          PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES,           {REMOVE_CONTIGUOUS_CAST_NODES_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_CAST_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseCastMme,                        PASS_ID_FUSE_CAST_MME,                          {FUSE_CAST_MME_DEPENDENCY_SET}                         )->addPredicate(PREDICATE_ID_FUSED_NODE_TO_MME);
    REGISTER_HABANA_PASS(setDmaParallelLevel,                PASS_ID_SET_DMA_PARALLEL_LEVEL,                 {SET_DMA_PARALLEL_LEVEL_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(memsetNodeOutput,                   PASS_ID_MEMSET_NODE_OUTPUT,                     {MEMSET_NODE_OUTPUT_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(handleMemoryReuse,                  PASS_ID_HANDLE_MEMORY_REUSE,                    {HANDLE_MEMORY_REUSE_DEPENDENCY_SET}                   )->addPredicate(PREDICATE_ID_LOGICAL_NODE_RAN);
    REGISTER_GAUDI_PASS( selectMemcpyEngine,                 PASS_ID_SELECT_MEMCPY_ENGINE,                   {SELECT_MEMCOPY_ENGINE_DEPENDENCY_SET}                 )->addPredicate(PREDICATE_ID_MEMCPY_NODE_CREATED);
    REGISTER_HABANA_PASS(transposeFcdBroadcast,              PASS_ID_TRANSPOSE_FCD_BROADCAST,                {TRANSPOSE_FCD_BROADCAST_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(sliceGraphToSRAMCapacity,           PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,           {SLICE_GRAPH_TO_SRAM_CAPACITY_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(fuseSpillFillDirectives,            PASS_ID_FUSE_SPILL_FILL,                        {FUSE_SPILL_FILL_DEPENDENCY_SET}                       );
    REGISTER_HABANA_PASS(enableMmeNodeCommonDimConcurrency,  PASS_ID_MME_CONCURRENCY,                        {MME_CONCURRENCY_DEPENDENCY_SET}                       );
    REGISTER_HABANA_PASS(handleLogicalOps,                   PASS_ID_HANDLE_LOGICAL_OPERATIONS,              {HANDLE_LOGICAL_OPERATIONS_DEPENDENCY_SET}             )->addPredicate(PREDICATE_ID_LOGICAL_NODE_CREATED);
    REGISTER_HABANA_PASS(handleLogicalOpsPreProcess,         PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS,  {HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS_DEPENDENCY_SET} );
    REGISTER_HABANA_PASS(handleLogicalOpsPostProcess,        PASS_ID_HANDLE_LOGICAL_OPERATIONS_POST_PROCESS, {HANDLE_LOGICAL_OPERATIONS_POST_PROCESS_DEPENDENCY_SET});
    REGISTER_HABANA_PASS(optimizeMemcpyNodes,                PASS_ID_OPTIMIZE_MEMCPY_NODES,                  {OPTIMIZE_MEMCPY_NODES_DEPENDENCY_SET}                   )->addPredicate(PREDICATE_ID_DMA_NODE_CREATED);
    REGISTER_HABANA_PASS(replaceGroupConvFilter2d,           PASS_ID_REPLACE_GROUP_CONV_FILTER2D,            {REPLACE_GROUP_CONV_FILTER2D_DEPENDENCY_SET}           );
    REGISTER_HABANA_PASS(handleGroupedConvolutions,          PASS_ID_HANDLE_GROUPED_CONVOLUTIONS,            {HANDLE_GROUPED_CONVOLUTIONS_DEPENDENCY_SET}           );
    REGISTER_HABANA_PASS(fuseBatchNormMemCpy,                PASS_ID_FUSE_BATCH_NORM_MEMCPY,                 {FUSE_BATCH_NORM_MEMCPY_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(fuseTransposeMme,                   PASS_ID_FUSE_TRANSPOSE_MME,                     {FUSE_TRANSPOSE_MME_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_FUSED_NODE_TO_MME)->addPredicate(PREDICATE_ID_PHYSICAL_TRANSPOSE_NODE_CREATED)->addPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED);
    REGISTER_HABANA_PASS(fuseConstantTranspose,              PASS_ID_FUSE_CONST_TRANSPOSE,                   {FUSE_CONST_TRANSPOSE_DEPENDENCY_SET}                  )->addPredicate(PREDICATE_ID_TRANSPOSE_NODE_CREATED);
    REGISTER_HABANA_PASS(removeRedundantLogicalNodes,        PASS_ID_REMOVE_REDUNDANT_LOGICAL_NODES,         {REMOVE_REDUNDANT_LOGICAL_NODES_DEPENDENCY_SET}        )->addPredicate(PREDICATE_ID_LOGICAL_NODE_CREATED);
    REGISTER_HABANA_PASS(eliminateNodesWithStaticInputs,     PASS_ID_ELIMINATE_NODES_WITH_STATIC_INPUTS,     {ELIMINATE_NODES_WITH_STATIC_INPUTS_DEPENDENCY_SET}    )->addPredicate(PREDICATE_ID_NODE_CREATED_CONST_INPUT);
    REGISTER_HABANA_PASS(packingMmeNodes,                    PASS_ID_PACKING_MME_NODES,                      {PACKING_MME_NODES_DEPENDENCY_SET}                     );
    REGISTER_HABANA_PASS(registerMemoryCoherence,            PASS_ID_REGISTER_MEM_COHERENCE,                 {REGISTER_MEM_COHERENCE_DEPENDENCY_SET}                )->addPredicate(PREDICATE_ID_MEMORY_SECTION_TENSOR_CREATED);
    REGISTER_HABANA_PASS(commonSubExpressionElimination,     PASS_ID_CSE_OPTIMIZATION,                       {CSE_OPTIMIZATION_DEPENDENCY_SET}                      );
    REGISTER_HABANA_PASS(splitFrobeniusLayerNorm,            PASS_ID_SPLIT_FROBENIUS_LAYER_NORM,             {SPLIT_FROBENIUS_LAYER_NORM_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(removeZeroSizedTensors,             PASS_ID_REMOVE_ZERO_SIZED_TENSORS,              {REMOVE_ZERO_SIZED_TENSORS_DEPENDENCY_SET}             )->addPredicate(PREDICATE_ID_MEMCPY_NODE_CREATED)->addPredicate(PREDICATE_ID_DMA_NODE_CREATED);

    REGISTER_HABANA_PASS(allocateTensors,                    PASS_ID_ALLOCATE_TENSORS,                       {ALLOCATE_TENSORS_DEPENDENCY_SET}                      );
    REGISTER_HABANA_PASS(validateMemoryAllocation,           PASS_ID_VALIDATE_MEMORY_ALLOCATION,             {VALIDATE_MEMORY_ALLOCATION_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(validateDmaNodes,                   PASS_ID_VALIDATE_DMA_NODES,                     {VALIDATE_DMA_NODES_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_DMA_NODE_CREATED);
    // data layout passes
    REGISTER_HABANA_PASS(setSupportedLayouts,                PASS_ID_SET_SUPPORTED_LAYOUTS,                  {SET_SUPPORTED_LAYOUTS_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(setHabanaLayouts,                   PASS_ID_SET_HABANA_LAYOUTS,                     {SET_HABANA_LAYOUTS_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(optimizeStridedInsert,              PASS_ID_OPTIMIZE_STRIDED_INSERT,                {OPTIMIZE_STRIDED_INSERT_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(adjustDataLayout,                   PASS_ID_ADJUST_DATA_LAYOUT,                     {ADJUST_DATA_LAYOUT_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(transposeDontCareNodes,             PASS_ID_TRANSPOSE_DONT_CARE_NODES,              {TRANSPOSE_DONT_CARE_NODES_DEPENDENCY_SET}             );
    REGISTER_HABANA_PASS(handlePermutedTensors,              PASS_ID_HANDLE_PERMUTED_TENSORS,                {HANDLE_PERMUTED_TENSORS_DEPENDENCY_SET}               );
    //
    REGISTER_HABANA_PASS(linkReductionMemsetShapes,          PASS_ID_LINK_REDUCTION_MEMSET_SHAPES,           {LINK_REDUCTION_MEMSET_SHAPES_DEPENDENCY_SET}          )->addPredicate(PREDICATE_ID_NODE_CREATED);
    REGISTER_HABANA_PASS(verifyMemsetOutputShapes,           PASS_ID_VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE,   {VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE_DEPENDENCY_SET}  );
    REGISTER_HABANA_PASS(validateDynamicShapes,              PASS_ID_VALIDATE_DYNAMIC_SHAPES,                {VALIDATE_DYNAMIC_SHAPES_DEPENDENCY_SET}               );
    REGISTER_HABANA_PASS(generateROIs,                       PASS_ID_GENERATE_ROIS,                          {GENERATE_ROIS_DEPENDENCY_SET}                         );
    REGISTER_HABANA_PASS(splitTPCDims,                       PASS_ID_SPLIT_TPC_DIMS,                         {SPLIT_TPC_DIMS_DEPENDENCY_SET}                        );
    REGISTER_HABANA_PASS(disableBundleRois,                  PASS_ID_DISABLE_BUNDLE_ROIS,                    {DISABLE_BUNDLE_ROIS_DEPENDENCY_SET});
    REGISTER_GAUDI_PASS( splitToLogicalROIs,                 PASS_ID_SPLIT_TO_LOGICAL_ROIS,                  {SPLIT_TO_LOGICAL_ROIS_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(projectNodeROIs,                    PASS_ID_PROJECT_NODE_ROIS,                      {PROJECT_NODE_ROIS_DEPENDENCY_SET}                     );
    REGISTER_HABANA_PASS(validateNodesLayout,                PASS_ID_VALIDATE_NODES_LAYOUT,                  {VALIDATE_NODES_LAYOUT_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(generateProfilerDebugInfo,          PASS_ID_GENERATE_PROFILER_DEBUG_INFO,           {GENERATE_PROFILER_DEBUG_INFO_DEPENDENCY_SET}          );
    REGISTER_HABANA_PASS(assignAddressesToTensorROIs,        PASS_ID_ASSIGN_ADDRESSES_TO_TENSOR_ROIS,        {ASSIGN_ADDRESSES_TO_TENSOR_ROIS_DEPENDENCY_SET}       );
    REGISTER_GAUDI_PASS( calculateTensorROIsLinearRanges,    PASS_ID_CALCULATE_TENSOR_ROIS_LINEAR_RANGES,    {CALCULATE_TENSOR_ROIS_LINEAR_RANGES_DEPENDENCY_SET}   );
    REGISTER_GAUDI_PASS( createDMADispatchers,               PASS_ID_CREATE_DMA_DISPATCHERS,                 {CREATE_DMA_DISPATCHERS_DEPENDENCY_SET}                );
    REGISTER_GAUDI_PASS( allocateSyncs,                      PASS_ID_ALLOCATE_SYNCS,                         {ALLOCATE_SYNCS_DEPENDENCY_SET}                        );
    REGISTER_HABANA_PASS(splitToPhysicalROIs,                PASS_ID_SPLIT_TO_PHYSICAL_ROIS,                 {SPLIT_TO_PHYSICAL_ROIS_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(setRoiShapeType,                    PASS_ID_SET_ROI_SHAPE_TYPE,                     {SET_ROI_SHAPE_TYPE_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(eliminateRedundantNodes,            PASS_ID_ELIMINATE_REDUNDANT_NODES,              {ELIMINATE_REDUNDANT_NODES_DEPENDENCY_SET}             )->addPredicate(PREDICATE_ID_ELIMINATE_REDUNDANT_NODES);
    REGISTER_HABANA_PASS(setReductionMemset,                 PASS_ID_SET_REDUCTION_MEMSET,                   {SET_REDUCTION_MEMSET_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(validateExecutionScheduleBundles,   PASS_ID_VALIDATE_EXECUTION_SCHEDULE_BUNDLES,    {VALIDATE_EXECUTION_SCHEDULE_BUNDLES_DEPENDENCY_SET}   );
    REGISTER_HABANA_PASS(setNonPersistentSectionInfo,        PASS_ID_SET_NON_PERSISTENT_SECTION_INFO,        {SET_NON_PERSISTENT_SECTION_INFO_DEPENDENCY_SET}       );
    REGISTER_HABANA_PASS(handleTpcRmwKernels,                PASS_ID_HANDLE_RMW_TPC_KERNELS,                 {HANDLE_RMW_TPC_KERNELS_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(validateMMENodes,                   PASS_ID_VALIDATE_MME_NODES,                     {VALIDATE_MME_NODES_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(validateAtomicNodes,                PASS_ID_VALIDATE_ATOMIC_NODES,                  {VALIDATE_ATOMIC_NODES_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(graphVisualizationPost,             PASS_ID_GRAPH_VISUALIZATION_POST,               {GRAPH_VISUALIZATION_POST_DEPENDENCY_SET}              );
    REGISTER_GAUDI_PASS(allocateTpcKernels,                  PASS_ID_ALLOCATE_TPC_KERNELS,                   {ALLOCATE_TPC_KERNELS_DEPENDENCY_SET}                  );
    REGISTER_HABANA_PASS(handleBroadcastBatchGemm,           PASS_ID_HANDLE_PARTIAL_BROADCAST_BGEMM,         {HANDLE_PARTIAL_BROADCAST_BGEMM_DEPENDENCY_SET}        );
    REGISTER_HABANA_PASS(insertProbeNanNodes,                PASS_ID_INSERT_NAN_INF_PROBE,                   {INSERT_NAN_INF_PROBE_DEPENDENCY_SET}                  );
    REGISTER_GAUDI_PASS(addH2DOp,                            PASS_ID_ADD_H2D_OP,                             {ADD_H2D_OP_DEPENDENCY_SET}                            );
    REGISTER_GAUDI_PASS(signalOutFromGraph,                  PASS_ID_SIGNAL_OUT_FROM_GRAPH,                  {SIGNAL_OUT_FROM_GRAPH_DEPENDENCY_SET}                 );
    REGISTER_HABANA_PASS(forceExplicitPadding,               PASS_ID_FORCE_EXPLICIT_PADDING,                 {FORCE_EXPLICIT_PADDING_DEPENDENCY_SET}                );
    REGISTER_HABANA_PASS(gcPerfChecks,                       PASS_ID_GC_PERF_CHECKS,                         {GC_PERF_CHECKS_DEPENDENCY_SET}                        );
    REGISTER_HABANA_PASS(checkMaxDimsPreCompilation,         PASS_ID_CHECK_MAX_DIMS_PRE,                     {CHECK_MAX_DIMS_PRE_DEPENDENCY_SET}                    );
    REGISTER_HABANA_PASS(checkMaxDimsPostCompilation,        PASS_ID_CHECK_MAX_DIMS_POST,                    {CHECK_MAX_DIMS_POST_DEPENDENCY_SET}                   );
    REGISTER_HABANA_PASS(addStaticShapeTensors,              PASS_ID_ADD_STATIC_SHAPE_TENSORS,               {ADD_STATIC_SHAPE_TENSORS_DEPENDENCY_SET}              );
    REGISTER_HABANA_PASS(validatePreSlicingSizes,            PASS_ID_VALIDATE_PRE_SLICING_SIZES,             {VALIDATE_PRE_SLICING_SIZES_DEPENDENCY_SET}            );
    REGISTER_HABANA_PASS(nodeCreatedWithoutOutputShape,      PASS_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE,      {CHECK_MAX_DIMS_PRE_DEPENDENCY_SET}                    )->addPredicate(PREDICATE_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE);

    // clang-format on
}

bool GaudiGraph::compile()
{
    CompilationHalReaderSetter compHalReaderSetter(this);

    HB_ASSERT(GCFG_GAUDI_DEMO.value() == false, "GAUDI_DEMO flag is not supported.");

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
    PrintTopologyHBMBandwidth();
    LOG_DEBUG(GC, "Done optimizing graph");

    m_codeGenerator->generate(this);

    // Initialize and Fill MME, TPC and DMA queues.
    m_codeGenerator->initQueues();
    m_codeGenerator->fillQueues();
    m_codeGenerator->printQueues();

    m_codeGenerator->generateRecipes(*this);  // // temporary until Queues and addAllDescriptors will move to codeGen

    return true;
}

bool GaudiGraph::graphSupports64BitDataTypes() const
{
    return true;
}

bool GaudiGraph::generateExecutionSchedule() const
{
    GaudiScheduler scheduler(this);
    return HabanaGraph::generateExecutionSchedule(&scheduler);
}

recipe_t* GaudiGraph::serializeDataPlane(RecipeAllocator* recipeAlloc) const
{
    // temporary until serializeShapePlane will move to codeGen in all graphs
    return m_codeGenerator->serializeDataPlane(recipeAlloc);
}

shape_plane_graph_t* GaudiGraph::serializeShapePlane(RecipeAllocator* recipeAlloc) const
{
    // temporary until serializeShapePlane will move to codeGen in all graphs
    return m_codeGenerator->serializeShapePlane(recipeAlloc);
}

void GaudiGraph::PrintTopologyHBMBandwidth() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GRAPH_DATA)) return;

    LOG_DEBUG(GRAPH_DATA, "HBM Bandwidth");
    uint64_t totalTopologyRead  = 0;
    uint64_t totalTopologyWrite = 0;
    const NodeVector& exeSortedNodes     = getExeSortedNodes();
    for (const pNode& n : exeSortedNodes)
    {
        if (n == nullptr) continue;
        totalTopologyRead += n->getReadBytes(TENSOR_IN_DRAM, *GetNodeROIs(n), getHALReader()->getCacheLineSizeInBytes());
        totalTopologyWrite += n->getWriteBytes(TENSOR_IN_DRAM, *GetNodeROIs(n), getHALReader()->getCacheLineSizeInBytes());
    }
    LOG_DEBUG(GRAPH_DATA, " Total HBM read traffic: {} bytes ({} GB)", totalTopologyRead, (float)totalTopologyRead / (1024 * 1024 * 1024));
    LOG_DEBUG(GRAPH_DATA, " Total HBM write traffic: {} bytes ({} GB)", totalTopologyWrite, (float)totalTopologyWrite / (1024 * 1024 * 1024));
}

HabanaGraphPtr GaudiGraph::createEmptyGraph() const
{
    return std::make_unique<GaudiGraph>();
}

bool GaudiGraph::validateMemorySection(const InternalSectionHandle* section) const
{
    HB_ASSERT(section != nullptr, "Unexpected empty section handle");

    if (section->getPersistent() && section->getRMW())
    {
        return false;
    }
    return true;
}

std::vector<uint32_t> GaudiGraph::getRollupsArray(NodePtr mmeNode) const
{
    auto it = m_rollupsArray.find(mmeNode);
    if (it == m_rollupsArray.end())
    {
        return std::vector<uint32_t>();
    }
    return it->second;
}

void GaudiGraph::updateMmeRollupsArray(const MmeNode& node, unsigned numRollups)
{
    pNode nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);
    m_rollupsArray[nodeShared].push_back(numRollups);
}

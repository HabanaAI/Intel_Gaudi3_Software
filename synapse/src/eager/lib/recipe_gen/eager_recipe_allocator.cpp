#include "eager_recipe_allocator.h"

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "desc_gen/node2desc.h"
#include "program_data_blob_manager.h"
#include "recipe_gen/recipe_hal_base.h"
#include "recipe_gen/recipe_instantiation_dma.h"
#include "recipe_gen/recipe_instantiation_mme.h"
#include "recipe_gen/recipe_templates.h"
#include "utils/general_defs.h"
#include "utils/general_defs.h"
#include "utils/memory_utils.h"
#include "utils/numeric_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/data_type_utils.h"  // for getStringFromSynDataType

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace eager_mode
{
// Accumulation info of all recipe_t fields that are affected by multi-node
struct DataAccumulator
{
    explicit DataAccumulator(NodesNrType tpcNodeNr, const RecipeHalBase& recipeHal, bool isDebugInfoEnabled)
    : tpcNodeNr(tpcNodeNr), m_recipeHal(recipeHal), m_isDebugInfoEnabled(isDebugInfoEnabled)
    {
    }

    // totalHostSz + totalMappedSz is the total memory allocated for the recipe excluding the Host allocated graph and
    // tensor name strings.
    size_t totalHostSz = 0;    // Total HEAP memory size in bytes for a single alloc containing all unmapped regions
                               // excluding strings.
    size_t totalMappedSz = 0;  // Total memory in bytes for a single buffer alloc containing all mapped regions.

    // Required information to calculate size of static/dynamic ECB command lists
    std::array<EcbCmdsListInfo, Recipe::EngineType::ENGINES_NR> ecbCmdsListInfo = {};

    size_t debugInfoStringsLen = 0;

    // Existing recipe fields to be populated
#define RECIPE_COUNTER(COUNTER) decltype(recipe_t::COUNTER) COUNTER = 0
    RECIPE_COUNTER(blobs_nr);
    RECIPE_COUNTER(execution_blobs_buffer_size);
    RECIPE_COUNTER(patching_blobs_buffer_size);
    RECIPE_COUNTER(dynamic_blobs_buffer_size);
    RECIPE_COUNTER(arc_jobs_nr);
    RECIPE_COUNTER(node_nr);
    RECIPE_COUNTER(persist_tensors_nr);
    RECIPE_COUNTER(permute_tensors_views_nr);
    RECIPE_COUNTER(patch_points_nr);
    RECIPE_COUNTER(recipe_conf_nr);
#undef RECIPE_COUNTER

    NodesNrType tpcNodeNr;

public:
    void          proccessNode(const SingleNode2Desc& singleNode2Desc);
    ArcJobsNrType calcArcJobsNr();
    size_t        calcTotalEcbCmdListsSize();

private:
    using ExeBlobsBufSizeType = decltype(recipe_t::execution_blobs_buffer_size);
    ExeBlobsBufSizeType calcExeBlobsBufferSize(const TemplateOfEngine& templateOfEngine, size_t templateRepeats);
    BlobsNrType         calcBlobsNr(const TemplateOfEngine& templateOfEngine, size_t templateRepeats);

private:
    const RecipeHalBase& m_recipeHal;
    bool                 m_isDebugInfoEnabled;  // init debug info for the profiler

    // Flag to tell if const execution blob (if supported) was counted in blobs_nr.
    // For more details please have a look at ::calcExeBlobsBufferSize(...).
    bool m_isFirstNodeProcessingDone = false;
};

static std::size_t calcNodeDebugStringsLen(const DescGeneratorBase& descGenBase)
{
    const Node&  node        = descGenBase.getNode();
    const size_t nodeNameLen = node.getNodeName().size();
    const size_t nodeGuidLen = node.getGUID().size();
    // TODO: consider using a fixed table for datatypes in the DataAccumulator c'tor,
    //       in case of multiple nodes as it would
    //       1. require a single memcpy of 2-3 cachelines instead of tiny string copies,
    //          avoiding the calc cost here
    //       2. have single copy of each data types string
    //       3. have faster init since we just point without extra copies there based on the table entry
    const size_t nodeTypeLen =
        getStringFromSynDataType((node.getNumOutputs() ? node.getOutput(0) : node.getInput(0))->getElementType())
            .size();
    return (nodeNameLen + 1) + (nodeGuidLen + 1) + (nodeTypeLen + 1);  // including terminating '\0's
}

// Accumulate data given a new node
void DataAccumulator::proccessNode(const SingleNode2Desc& singleNode2Desc)
{
    const DescGeneratorBase& descGenBase = singleNode2Desc.getDescGen();
    const TemplateOfEngine&  templateOfEngine =
        RecipeTemplates::getInstance().getTemplate(descGenBase.getChipType(),
                                                   descGenBase.getEngineType(),
                                                   descGenBase.getPatchableTensorsNr());
    const recipe_t& templateRecipe = templateOfEngine.recipe;
    EAGER_ASSERT(templateOfEngine.descNr != 0, "Invalid descriptors number in template");
    const size_t templateRepeats = descGenBase.getDescNr() / templateOfEngine.descNr;

    // Data blob buffers
    execution_blobs_buffer_size += calcExeBlobsBufferSize(templateOfEngine, templateRepeats);
    patching_blobs_buffer_size += templateRecipe.patching_blobs_buffer_size;
    dynamic_blobs_buffer_size += templateRecipe.dynamic_blobs_buffer_size * descGenBase.getRequiredWdCtxNr();
    // Other structures
    blobs_nr += calcBlobsNr(templateOfEngine, templateRepeats);
    node_nr += templateRecipe.node_nr;
    patch_points_nr += templateRecipe.patch_points_nr;
    recipe_conf_nr = templateRecipe.recipe_conf_nr;  // TODO: consider change to actual config
    // Arc jobs
    EAGER_ASSERT(templateRecipe.arc_jobs_nr == 1, "Unsupported template");
    EAGER_ASSERT(templateRecipe.arc_jobs[0].logical_engine_id < ecbCmdsListInfo.size(), "Invalid engine id");
    EcbCmdsListInfo& ecbInfo = ecbCmdsListInfo[templateRecipe.arc_jobs[0].logical_engine_id];

    // Duplicate template but keep one patchable blob only
    const EcbCommandSizeType nonRepeatedSize =
        (descGenBase.getActivationNr() - 1) * m_recipeHal.getEcbCommandSize(EngineArcCommandId::STATIC_DESC_V2);
    ecbInfo.staticSz += templateOfEngine.ecbsNetSize.staticSz * descGenBase.getActivationNr() - nonRepeatedSize;
    ecbInfo.dynamicSz += templateOfEngine.ecbsNetSize.dynamicSz * descGenBase.getActivationNr() -
                         templateOfEngine.calcExtraAlignmentNopsSizeOfDynamicEcb(descGenBase.getActivationNr());
    ecbInfo.staticChunksNr = templateOfEngine.ecbsNetSize.staticChunksNr;

    // Must be last because calcExeBlobsBufferSize and calcBlobsNr use it
    m_isFirstNodeProcessingDone = true;

    if (unlikely(m_isDebugInfoEnabled))
    {
        debugInfoStringsLen += calcNodeDebugStringsLen(singleNode2Desc.getDescGen());
    }
}

// When const blob is supported, we allocate it for the first activation of the first node,
// The rest activations and nodes point to it at their static ECB commands.
DataAccumulator::ExeBlobsBufSizeType DataAccumulator::calcExeBlobsBufferSize(const TemplateOfEngine& templateOfEngine,
                                                                             size_t                  templateRepeats)
{
    EAGER_ASSERT(templateRepeats != 0, "Invalid template repeates");
    if (m_recipeHal.isConstExeBlobSupported())
    {
        EAGER_ASSERT(templateOfEngine.constExeBlobIndex < templateOfEngine.recipe.blobs_nr, "Invalid const blob index");
        const ExeBlobsBufSizeType sizeofConstBlob =
            templateOfEngine.recipe.blobs[templateOfEngine.constExeBlobIndex].size;
        const ExeBlobsBufSizeType sizeOf1stActivation =
            templateOfEngine.recipe.execution_blobs_buffer_size - (m_isFirstNodeProcessingDone ? sizeofConstBlob : 0);
        const ExeBlobsBufSizeType sizeOfRestActivations =
            templateOfEngine.recipe.execution_blobs_buffer_size - sizeofConstBlob;
        return sizeOf1stActivation + (templateRepeats - 1) * sizeOfRestActivations;
    }
    return templateRepeats * templateOfEngine.recipe.execution_blobs_buffer_size;
}

// Same reasoning mentioned at ::calcExeBlobsBufferSize(...)
BlobsNrType DataAccumulator::calcBlobsNr(const TemplateOfEngine& templateOfEngine, size_t templateRepeats)
{
    EAGER_ASSERT(templateRepeats != 0, "Invalid template repeates");
    const BlobsNrType nonExecBlobsNr           = templateOfEngine.getPatchableBlobsNr() + /*dynamic blob*/ 1;
    BlobsNrType       blobsNrOf1stActivation   = templateOfEngine.recipe.blobs_nr;
    BlobsNrType       blobsNrOfRestActivations = templateOfEngine.recipe.blobs_nr - nonExecBlobsNr;
    if (m_recipeHal.isConstExeBlobSupported())
    {
        if (m_isFirstNodeProcessingDone)
        {
            --blobsNrOf1stActivation;  // Drop const blob
        }
        --blobsNrOfRestActivations;  // Drop const blob
    }
    return blobsNrOf1stActivation + (templateRepeats - 1) * blobsNrOfRestActivations;
}

// Calculate arc_jobs_nr and return it
ArcJobsNrType DataAccumulator::calcArcJobsNr()
{
    arc_jobs_nr = std::count_if(ecbCmdsListInfo.begin(), ecbCmdsListInfo.end(), [](EcbCmdsListInfo& info) {
        return info.staticChunksNr != 0;
    });
    return arc_jobs_nr;
}

// Calculate EcbCmdsListInfo::staticSz and return total size of ECB command lists
size_t DataAccumulator::calcTotalEcbCmdListsSize()
{
    size_t     totalSize              = 0;
    const auto tailSize               = m_recipeHal.getTailSize();
    const auto dynamicNonRepeatedSize = m_recipeHal.getDynamicNonRepeatedSize();
    const auto staticEcbAlignment     = m_recipeHal.getEcbCommandAlignmentConstraint(EcbType::STATIC);
    const auto dynamicEcbAlignment    = m_recipeHal.getEcbCommandAlignmentConstraint(EcbType::DYNAMIC);

    for (EcbCmdsListInfo& info : ecbCmdsListInfo)
    {
        EAGER_ASSERT((info.staticChunksNr == 0) == (info.staticSz == 0), "Inconsistent ECB sizes");
        EAGER_ASSERT((info.staticChunksNr == 0) == (info.dynamicSz == 0), "Inconsistent ECB sizes");
        if (info.staticChunksNr == 0) continue;  // The engine doesn't participate

        // Calc size of static and dynamic ECB
        auto calcEcbSize = [&](EcbCommandSizeType repeatedSize,
                               EcbCommandSizeType chunksNr,
                               EcbCommandSizeType nonRepeatedSize,
                               std::size_t        alignment) {
            // Chunk size without padding
            EcbCommandSizeType unalignedChunkSize = repeatedSize + nonRepeatedSize;
            // Allocate space for padding NOP if necessary
            if ((unalignedChunkSize % alignment) != 0)
            {
                unalignedChunkSize += tailSize;
            }
            // Add padding at the end of the chunk
            const EcbCommandSizeType alignedChunkSize = alignUpTo(unalignedChunkSize, alignment);
            return alignedChunkSize * chunksNr;
        };
        info.staticSz =
            calcEcbSize(info.staticSz, info.staticChunksNr, m_recipeHal.getStaticNonRepeatedSize(), staticEcbAlignment);
        info.dynamicSz = calcEcbSize(info.dynamicSz, 1, dynamicNonRepeatedSize, dynamicEcbAlignment);

        totalSize += info.staticSz + info.dynamicSz;
    }
    return totalSize;
}

// Calculate the total size needed for the eager recipe allocation.
//
// Note: that the following aren't handled here but by the Instantiation,
// - recipe_t::nameSize
// - recipe_t::name
// - persist_tensor_info_t::name
// - persist_tensor_info_t::layout
//
// Note: Keep a single copy of the patching blobs while the rest are duplicated per instance
// Note: The placement order here MUST match doAllPlacements or there will be possible mismatch due to padding
//
void EagerRecipeAllocator::planAllAllocs(DataAccumulator& dataAccum) const
{
    planAlloc<recipe_t>(dataAccum.totalHostSz, 1);

    // SKIPPED:
    // - nameSize
    // - name

    planAlloc<decltype(*recipe_t::blobs)>(dataAccum.totalHostSz, dataAccum.blobs_nr);

    const size_t totalBlobsDataSz = dataAccum.execution_blobs_buffer_size + dataAccum.patching_blobs_buffer_size +
                                    dataAccum.dynamic_blobs_buffer_size;
    planAlloc<char>(dataAccum.totalMappedSz, totalBlobsDataSz);

    // SKIPPED:
    // - programs_nr
    // - programs
    // - activate_jobs_nr
    // - activate_jobs
    // - execute_jobs_nr
    // - execute_jobs

    // ARC jobs:
    planAlloc<decltype(*recipe_t::arc_jobs)>(dataAccum.totalHostSz, dataAccum.calcArcJobsNr());
    planAlloc<decltype(*ecb_t::cmds)>(dataAccum.totalMappedSz, dataAccum.calcTotalEcbCmdListsSize());

    planAlloc<decltype(*recipe_t::tensors)>(dataAccum.totalHostSz, m_tensorsNr);

    // These are only needed for TPC
    if (m_isProgramDataBlobsCopyRequired)
    {
        planAlloc<decltype(*recipe_t::program_data_blobs_buffer)>(dataAccum.totalMappedSz, m_dataBlobsSizeInBytes);
    }
    planAlloc<decltype(*recipe_t::program_data_blobs)>(dataAccum.totalHostSz, m_programDataBlobsNum);

    planAlloc<decltype(*recipe_t::patch_points)>(dataAccum.totalHostSz, dataAccum.patch_points_nr);

    // SKIPPED:
    // - sections_nr
    // - section_groups_nr
    // - section_groups_patch_points
    // - sobj_section_group_patch_points
    // - section_ids_nr
    // - section_blobs_indices

    planAlloc<decltype(*recipe_t::node_exe_list)>(dataAccum.totalHostSz, dataAccum.node_nr);
    planAlloc<decltype(*recipe_t::node_exe_list->program_blobs_nr)>(dataAccum.totalHostSz, dataAccum.node_nr);
    planAlloc<decltype(*recipe_t::workspace_sizes)>(dataAccum.totalHostSz, workspaceSizesNr);

    if (unlikely(dataAccum.debugInfoStringsLen != 0))
    {
        planAlloc<decltype(*debug_info_t::nodes)>(dataAccum.totalHostSz, dataAccum.node_nr);
        planAlloc<ProfilerNumEnginesType>(dataAccum.totalHostSz, dataAccum.tpcNodeNr);
    }

    planAlloc<decltype(*recipe_t::recipe_conf_params)>(dataAccum.totalHostSz, dataAccum.recipe_conf_nr);
}

// Generate a recipe and perform placement at the (already allocated) `base` addr.
//
// Note: that the following aren't handled here but by the Instantiation,
// - recipe_t::nameSize
// - recipe_t::name
// - persist_tensor_info_t::name
// - persist_tensor_info_t::layout
//
// Note: Keep a single copy of the patching blobs while the rest are duplicated per instance
// Note: The placement order here MUST match planAllAllocs or there will be possible mismatch due to padding
//
recipe_t* EagerRecipeAllocator::doAllPlacements(std::byte* const       heapBase,
                                                std::byte* const       mappedBase,
                                                const DataAccumulator& dataAccum) const
{
    std::byte* heapPtr   = heapBase;
    std::byte* mappedPtr = mappedBase;

    // Note that if this check fails, we'll have to pad our alignment and require a slightly larger overall size,
    // otherwise we'll run out of space for all of our allocations and fail on the exact alloc assert at the end.
    EAGER_ASSERT(alignFor<recipe_t>(heapPtr) == heapPtr, "default new alignment is insufficient for recipe_t");
    recipe_t& recipe = *doPlacement<recipe_t>(heapPtr, 1);

    // SKIPPED:
    recipe.nameSize = 0;
    recipe.name     = nullptr;

    recipe.blobs_nr = dataAccum.blobs_nr;
    recipe.blobs    = doPlacement<decltype(*recipe_t::blobs)>(heapPtr, recipe.blobs_nr);

    // Blob buffers:
    {
        using ExecBufType = decltype(*recipe_t::execution_blobs_buffer);
        EAGER_ASSERT(dataAccum.execution_blobs_buffer_size % sizeof(ExecBufType) == 0, "Invalid buffer size");
        recipe.execution_blobs_buffer_size = dataAccum.execution_blobs_buffer_size;
        recipe.execution_blobs_buffer =
            doPlacement<ExecBufType>(mappedPtr, recipe.execution_blobs_buffer_size / sizeof(ExecBufType));

        using PatchBufType = decltype(*recipe_t::patching_blobs_buffer);
        EAGER_ASSERT(dataAccum.patching_blobs_buffer_size % sizeof(PatchBufType) == 0, "Invalid buffer size");
        recipe.patching_blobs_buffer_size = dataAccum.patching_blobs_buffer_size;
        recipe.patching_blobs_buffer =
            doPlacement<PatchBufType>(mappedPtr, recipe.patching_blobs_buffer_size / sizeof(PatchBufType));

        using DynamicBufType = decltype(*recipe_t::dynamic_blobs_buffer);
        EAGER_ASSERT(dataAccum.dynamic_blobs_buffer_size % sizeof(DynamicBufType) == 0, "Invalid buffer size");
        recipe.dynamic_blobs_buffer_size = dataAccum.dynamic_blobs_buffer_size;
        recipe.dynamic_blobs_buffer =
            doPlacement<DynamicBufType>(mappedPtr, recipe.dynamic_blobs_buffer_size / sizeof(DynamicBufType));
    }

    // SKIPPED:
    recipe.programs_nr      = 0;
    recipe.programs         = nullptr;
    recipe.activate_jobs_nr = 0;
    recipe.activate_jobs    = nullptr;
    recipe.execute_jobs_nr  = 0;
    recipe.execute_jobs     = nullptr;

    // ARC jobs:
    {
        recipe.arc_jobs_nr = dataAccum.arc_jobs_nr;
        recipe.arc_jobs    = doPlacement<decltype(*recipe_t::arc_jobs)>(heapPtr, recipe.arc_jobs_nr);
        // Init sizes of ECB lists
        ArcJobsNrType curArcJobIdx = 0;
        EAGER_ASSERT(dataAccum.ecbCmdsListInfo.size() == Recipe::EngineType::ENGINES_NR, "Unsupported ARC!");
        for (unsigned i = 0; i < Recipe::EngineType::ENGINES_NR; ++i)
        {
            const Recipe::EngineType engine = static_cast<Recipe::EngineType>(i);
            const EcbCmdsListInfo&   info   = dataAccum.ecbCmdsListInfo[engine];
            if (info.staticChunksNr == 0) continue;  // The engine doesn't participate

            EAGER_ASSERT(curArcJobIdx < dataAccum.arc_jobs_nr, "Invalid ARC jobs number");
            arc_job_t& dstJob           = recipe.arc_jobs[curArcJobIdx++];
            dstJob.engines_filter       = 0;       // currently not used but better to set
            dstJob.logical_engine_id    = engine;  // We use it early at RecipeInstantiation::initEcbTrackers()
            dstJob.static_ecb.cmds_size = info.staticSz;
            dstJob.static_ecb.cmds      = doPlacement<decltype(*ecb_t::cmds)>(mappedPtr, dstJob.static_ecb.cmds_size);
            EAGER_ASSERT((info.staticSz % info.staticChunksNr) == 0, "Invalid ECB chunk size");
            // 1 means single chunk defacto cmds_eng_offset=0
            dstJob.static_ecb.cmds_eng_offset  = (info.staticChunksNr != 1) ? (info.staticSz / info.staticChunksNr) : 0;
            dstJob.dynamic_ecb.cmds_size       = info.dynamicSz;
            dstJob.dynamic_ecb.cmds            = doPlacement<decltype(*ecb_t::cmds)>(mappedPtr, info.dynamicSz);
            dstJob.dynamic_ecb.cmds_eng_offset = 0;
        }
    }

    recipe.persist_tensors_nr = m_tensorsNr;
    recipe.tensors            = doPlacement<decltype(*recipe_t::tensors)>(heapPtr, recipe.persist_tensors_nr);

    recipe.h2di_tensors_nr = 0;

    recipe.permute_tensors_views_nr = 0;
    recipe.permute_tensors_views    = nullptr;

    recipe.const_sections_nr = 0;
    recipe.const_sections    = nullptr;

    // Note: These are only needed for TPC; these set them to 0 and nulltpr
    {
        recipe.program_data_blobs_size = m_dataBlobsSizeInBytes;
        recipe.program_data_blobs_buffer =
            m_isProgramDataBlobsCopyRequired
                ? doPlacement<decltype(*recipe_t::program_data_blobs_buffer)>(mappedPtr, recipe.program_data_blobs_size)
                : nullptr;
        recipe.program_data_blobs_nr = m_programDataBlobsNum;
        recipe.program_data_blobs =
            doPlacement<decltype(*recipe_t::program_data_blobs)>(heapPtr, recipe.program_data_blobs_nr);
    }

    recipe.patch_points_nr = dataAccum.patch_points_nr;
    recipe.patch_points    = doPlacement<decltype(*recipe_t::patch_points)>(heapPtr, recipe.patch_points_nr);

    // SKIPPED:
    recipe.sections_nr                     = 0;
    recipe.section_groups_nr               = 0;
    recipe.section_groups_patch_points     = nullptr;
    recipe.sobj_section_group_patch_points = {};
    recipe.section_ids_nr                  = 0;
    recipe.section_blobs_indices           = nullptr;

    recipe.node_nr       = dataAccum.node_nr;
    recipe.node_exe_list = doPlacement<decltype(*recipe_t::node_exe_list)>(heapPtr, recipe.node_nr);
    for (NodesNrType i = 0; i < recipe.node_nr; ++i)
    {
        recipe.node_exe_list[i].program_blobs_nr =
            doPlacement<decltype(*recipe_t::node_exe_list->program_blobs_nr)>(heapPtr, 1);
        *recipe.node_exe_list[i].program_blobs_nr = 0;
    }

    recipe.workspace_nr    = workspaceSizesNr;
    recipe.workspace_sizes = doPlacement<decltype(*recipe_t::workspace_sizes)>(heapPtr, recipe.workspace_nr);

    recipe.debug_profiler_info = {};
    if (unlikely(dataAccum.debugInfoStringsLen != 0))
    {
        recipe.debug_profiler_info.num_nodes = recipe.node_nr;
        recipe.debug_profiler_info.nodes =
            doPlacement<decltype(*debug_info_t::nodes)>(heapPtr, recipe.debug_profiler_info.num_nodes);

        static constexpr auto TPC_ROI_COUNT    = 1;
        const auto&           singleNode2Descs = m_descriptors.getExecSequence();
        for (size_t i = 0; i < singleNode2Descs.size(); ++i)
        {
            auto& node    = recipe.debug_profiler_info.nodes[i];
            node.num_rois = singleNode2Descs[i].getDescGen().getEngineType() == EngineType::TPC ? TPC_ROI_COUNT : 0;
            node.num_working_engines = doPlacement<ProfilerNumEnginesType>(heapPtr, node.num_rois);
        }
    }

    // SKIPPED:
    recipe.debug_sync_scheme_info = {};

    recipe.recipe_conf_nr     = dataAccum.recipe_conf_nr;
    recipe.recipe_conf_params = doPlacement<decltype(*recipe_t::recipe_conf_params)>(heapPtr, recipe.recipe_conf_nr);

    // SKIPPED:
    recipe.nop_kernel_offset  = 0;
    recipe.nop_kernel_section = 0;
    recipe.valid_nop_kernel   = false;

    // SKIPPED:
    recipe.max_used_mcid_discard = 0;
    recipe.max_used_mcid_degrade = 0;

    EAGER_ASSERT(heapBase + dataAccum.totalHostSz == heapPtr, "Mismatched HEAP planned and actual alloc total size");
    EAGER_ASSERT(mappedBase + dataAccum.totalMappedSz == mappedPtr,
                 "Mismatched MAPPED planned and actual alloc total size");
    return &recipe;
}

EagerRecipeAllocator::EagerRecipeAllocator(const ProgramDataBlobManager& programDataBlobManager,
                                           uint64_t                      dataBlobsSizeInBytes,
                                           EagerRecipeMemoryAllocator&   recipeAllocator,
                                           const RecipeHalBase&          recipeHal,
                                           const Node2DescContainer&     descriptors,
                                           TensorsNrType                 tensorsNr,
                                           bool                          canUseCloneFastPath,
                                           bool                          isProgramDataBlobsCopyRequired,
                                           bool                          isDebugInfoEnabled)
: m_programDataBlobsNum(programDataBlobManager.getProgramDataBlobs().size()),
  m_dataBlobsSizeInBytes(dataBlobsSizeInBytes),
  m_recipeAllocator(recipeAllocator),
  m_recipeHal(recipeHal),
  m_descriptors(descriptors),
  m_tensorsNr(tensorsNr),
  m_isUsingCloneFastPath(canUseCloneFastPath),
  m_isProgramDataBlobsCopyRequired(isProgramDataBlobsCopyRequired),
  m_isDebugInfoEnabled(isDebugInfoEnabled)
{
    // While it will work with an existing allocator, due to allocator's recipe ownership, we want a dedicated one
    EAGER_ASSERT(m_recipeAllocator.isEmpty(), "Expected a fresh allocator");
    if (!m_isProgramDataBlobsCopyRequired && !programDataBlobManager.getProgramDataBlobsSharedPtrs().empty())
    {
        EAGER_ASSERT(programDataBlobManager.getProgramDataBlobsSharedPtrs().size() == 1,
                     "more program data blobs than expected");
        m_recipeAllocator.addKernelOwnership(programDataBlobManager.getProgramDataBlobsSharedPtrs().front());
    }
}

// Allocate recipe and all buffers other than the graph and tensor names and initialize the pointers.
recipe_t* EagerRecipeAllocator::allocateAndInit(size_t namesStrLen)
{
    // While it will work with an existing allocator, due to allocator's recipe ownership, we want a dedicated one
    EAGER_ASSERT(m_recipeAllocator.isEmpty(), "Expected a fresh allocator");

    if (m_isUsingCloneFastPath)
    {
        EAGER_ASSERT(m_descriptors.isSingleDesc(), "");
        return clone(m_descriptors.getFirstDescGen(), namesStrLen);
    }

    DataAccumulator dataAccum(m_descriptors.getStatistics(EngineType::TPC).nodeNum, m_recipeHal, m_isDebugInfoEnabled);
    // Calc total size of recipe
    for (const SingleNode2Desc& singleNode2Desc : m_descriptors.getExecSequence())
    {
        dataAccum.proccessNode(singleNode2Desc);
    }

    planAllAllocs(dataAccum);
    size_t totalAlloc = 0;
    planAlloc<char>(totalAlloc, dataAccum.totalMappedSz);
    planAlloc<char, 64>(totalAlloc, dataAccum.totalHostSz);
    planAlloc<char>(totalAlloc, namesStrLen + dataAccum.debugInfoStringsLen);

    // Note that allocator's using "new" which has a sufficiently large alignment
    auto* allocBase =
        reinterpret_cast<std::byte*>(m_recipeAllocator.allocate(totalAlloc, /*shouldBeMappedToDevice*/ false));

    auto*     mappedPtr = allocBase;
    auto*     heapPtr   = mappedPtr + alignUpTo(dataAccum.totalMappedSz, 64);
    recipe_t* res       = doAllPlacements(heapPtr, mappedPtr, dataAccum);

    auto* stringPoolPtr      = heapPtr + dataAccum.totalHostSz;
    auto* persTensorNamesBuf = reinterpret_cast<std::byte*>(doPlacement<char>(stringPoolPtr, namesStrLen));
    m_stringBufAlloc.init(persTensorNamesBuf, namesStrLen + dataAccum.debugInfoStringsLen);

    return res;
}

// Note: The placement order here MUST match planAllAllocs or there will be possible mismatch due to padding
recipe_t* EagerRecipeAllocator::clone(const DescGeneratorBase& descGenBase, size_t namesStrLen)
{
    static constexpr auto NODE_COUNT    = 1;
    static constexpr auto TPC_ROI_COUNT = 1;

    const TemplateOfEngine& templateOfEngine =
        RecipeTemplates::getInstance().getTemplate(descGenBase.getChipType(),
                                                   descGenBase.getEngineType(),
                                                   descGenBase.getPatchableTensorsNr());
    EAGER_ASSERT(templateOfEngine.recipe.node_nr == NODE_COUNT, "");

    const bool isTpc = descGenBase.getEngineType() == EngineType::TPC;
    EAGER_ASSERT(!isTpc || descGenBase.getLogicalRoiNr() == TPC_ROI_COUNT,
                 "Profiler alloc relies on TPC having a single ROI");

    // Plan allocation
    size_t totalAlloc = 0;
    planAlloc<recipe_t>(totalAlloc, 1);
    const size_t dataBufSize = templateOfEngine.allocSize - sizeof(recipe_t);
    planAlloc<std::byte>(totalAlloc, dataBufSize);
    using ProgDataBlobsBufType = decltype(*recipe_t::program_data_blobs_buffer);
    using ProgDataBlobsType    = decltype(*recipe_t::program_data_blobs);
    if (isTpc)
    {
        planAlloc<ProgDataBlobsBufType>(totalAlloc, m_isProgramDataBlobsCopyRequired ? m_dataBlobsSizeInBytes : 0);
        planAlloc<ProgDataBlobsType>(totalAlloc, m_programDataBlobsNum);
    }
    if (unlikely(m_isDebugInfoEnabled))
    {
        planAlloc<decltype(*debug_info_t::nodes)>(totalAlloc, NODE_COUNT);
        // Cheaper to always do it (it's += 1) but we want to be explicit
        planAlloc<ProfilerNumEnginesType>(totalAlloc, isTpc ? TPC_ROI_COUNT : 0);
    }
    const size_t debugInfoStringsLen = unlikely(m_isDebugInfoEnabled) ? calcNodeDebugStringsLen(descGenBase) : 0;
    planAlloc<char>(totalAlloc, namesStrLen + debugInfoStringsLen);

    // Actual allocation and pointers initialization
    auto* allocBase =
        reinterpret_cast<std::byte*>(m_recipeAllocator.allocate(totalAlloc, /*shouldBeMappedToDevice*/ false));
    // Recipe and data buffer
    auto* actualRecipe = doPlacement<recipe_t>(allocBase, 1);
    auto* dataBuf      = doPlacement<std::byte>(allocBase, dataBufSize);
    std::memcpy(actualRecipe, &templateOfEngine.recipe, templateOfEngine.allocSize);
    adjustPointers(*actualRecipe, templateOfEngine.recipe, dataBuf);

    if (isTpc)
    {
        actualRecipe->program_data_blobs_size = m_dataBlobsSizeInBytes;
        actualRecipe->program_data_blobs_buffer =
            m_isProgramDataBlobsCopyRequired
                ? doPlacement<ProgDataBlobsBufType>(allocBase, actualRecipe->program_data_blobs_size)
                : nullptr;

        actualRecipe->program_data_blobs_nr = m_programDataBlobsNum;
        actualRecipe->program_data_blobs =
            doPlacement<ProgDataBlobsType>(allocBase, actualRecipe->program_data_blobs_nr);
    }
    if (unlikely(m_isDebugInfoEnabled))
    {
        auto& prof = actualRecipe->debug_profiler_info;

        prof.num_nodes = NODE_COUNT;
        prof.nodes     = doPlacement<decltype(*debug_info_t::nodes)>(allocBase, prof.num_nodes);

        prof.nodes[0].num_rois            = isTpc ? TPC_ROI_COUNT : 0;
        prof.nodes[0].num_working_engines = doPlacement<ProfilerNumEnginesType>(allocBase, prof.nodes[0].num_rois);
    }
    // Strings buffer
    auto* stringPoolPtr      = doPlacement<char>(allocBase, namesStrLen + debugInfoStringsLen);
    auto* persTensorNamesBuf = reinterpret_cast<std::byte*>(stringPoolPtr);
    m_stringBufAlloc.init(persTensorNamesBuf, namesStrLen + debugInfoStringsLen);
    // In addition to program data, those should have their actual values
    actualRecipe->persist_tensors_nr = m_tensorsNr;

    return actualRecipe;
}

void EagerRecipeAllocator::adjustPointers(recipe_t& actualRecipe, const recipe_t& templateRecipe, std::byte* dataBuf)
{
    const auto* oldBase = reinterpret_cast<const std::byte*>(&templateRecipe);
    auto*       newBase = reinterpret_cast<std::byte*>(&actualRecipe);

#define ADJUST_PTR(FIELD)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        std::byte* newPtr = (reinterpret_cast<std::byte*>(actualRecipe.FIELD) - oldBase) + newBase;                    \
        EAGER_ASSERT(sizeof(newPtr) == sizeof(actualRecipe.FIELD), "Incompatible pointer sizes");                      \
        actualRecipe.FIELD = reinterpret_cast<decltype(actualRecipe.FIELD)>(newPtr);                                   \
    } while (0)

    // NOLINTBEGIN(bugprone-sizeof-expression)

    ADJUST_PTR(execution_blobs_buffer);
    ADJUST_PTR(patching_blobs_buffer);
    ADJUST_PTR(dynamic_blobs_buffer);
    ADJUST_PTR(blobs);
    for (BlobsNrType i = 0; i < actualRecipe.blobs_nr; ++i)
    {
        ADJUST_PTR(blobs[i].data);
    }
    ADJUST_PTR(arc_jobs);
    EAGER_ASSERT(actualRecipe.arc_jobs_nr == 1, "Invalid case for clone flow");
    ADJUST_PTR(arc_jobs[0].static_ecb.cmds);
    ADJUST_PTR(arc_jobs[0].dynamic_ecb.cmds);
    ADJUST_PTR(tensors);
    ADJUST_PTR(patch_points);
    ADJUST_PTR(node_exe_list);
    ADJUST_PTR(workspace_sizes);
    ADJUST_PTR(recipe_conf_params);

    // NOLINTEND(bugprone-sizeof-expression)

#undef ADJUST_PTR
}

}  // namespace eager_mode

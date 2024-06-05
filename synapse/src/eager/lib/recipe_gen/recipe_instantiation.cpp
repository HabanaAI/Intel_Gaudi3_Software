#include "recipe_instantiation.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "desc_gen/node2desc.h"
#include "node_info/tensor_info.h"
#include "recipe_gen/blob_to_desc_map_structs.h"
#include "recipe_gen/eager_recipe_allocator.h"
#include "recipe_gen/recipe_arc_job_writer.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_instantiation_dma.h"
#include "recipe_gen/recipe_instantiation_mme.h"
#include "recipe_gen/recipe_instantiation_tpc.h"
#include "recipe_gen/recipe_templates.h"
#include "utils/general_defs.h"
#include "utils/numeric_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"
#include "graph_compiler/habana_nodes/tpc_node.h"
#include "graph_compiler/layout.h"
#include "graph_compiler/utils.h"
#include "include/recipe_version.h"

// synapse api (relative to include/)
#include "internal/define_synapse_common.hpp"
#include "internal/recipe.h"

// std includes
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace eager_mode
{
// Global constants specific for this implementation
//
// Assume one type of patch points
constexpr patch_point_t::EPatchPointType defaultPatchPointType = patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT;
// Cache base reg value can be acquired by setting offset in section to zero

// TODO: Wrapped to avoid uncatchable exception, can drop the function if the init is constexpr
[[maybe_unused]] static CommandGranularityRawType GetCacheBaseRegMask()
{
    static const auto CACHE_BASE_REG_MASK = ~getVirtualAddressForMemoryID(getMaxMemorySectionID(), 0);
    return CACHE_BASE_REG_MASK;
}

#ifndef NDEBUG  // It's defined only in debug build because the checkers can only assert
// Static checker to be invoked once recipe template is created.
// It guarantees to satisfy all assumptions on how the template is created.
void RecipeInstantiation::checkTemplateAssumptions(const TemplateOfEngine& templateOfEngine,
                                                   const RecipeHalBase&    recipeHal)
{
    const recipe_t& recipe = templateOfEngine.recipe;

    // Version must be similar to current
    EAGER_ASSERT(recipe.version_major == RECIPE_VERSION_MAJOR, "Incompatible recipe major version");
    EAGER_ASSERT(recipe.version_minor == recipeHal.getVersionMinor(), "Incompatible recipe minor version");

    // Each template must represent one node
    EAGER_ASSERT(recipe.node_nr == 1, "Unsupported recipe template");

    // All patch point types equal to defaultPatchPointType
    for (auto i = 0; i < recipe.patch_points_nr; ++i)
    {
        EAGER_ASSERT(recipe.patch_points[i].type == defaultPatchPointType, "Unsupported recipe template");
    }

    EAGER_ASSERT(recipe.arc_jobs_nr == 1, "A template must have only one ARC job");

    // Calculate number of patchable blobs. For MME verify that they are located at the beginning
    BlobsNrType           patchableBlobsNr    = 0;
    [[maybe_unused]] bool expectPatchableBlob = true;
    for (auto i = 0; i < recipe.blobs_nr; ++i)
    {
        if (recipe.blobs[i].blob_type.requires_patching)
        {
            EAGER_ASSERT(expectPatchableBlob, "Unsupported recipe template");
            patchableBlobsNr++;
        }
        // MME is special case because the template consists of single activation. Once having actual case with multiple
        // activations, we need to locate patchable blobs (actually it's one blob) at the beginning - that is to
        // facilitate the implementation.
        else if (recipe.arc_jobs[0].logical_engine_id == Recipe::EngineType::MME)
        {
            expectPatchableBlob = false;
        }
    }

    // Only one patchable blob
    EAGER_ASSERT(patchableBlobsNr == 1, "Unsupported recipe template");

    // Valid index of const execution blob
    EAGER_ASSERT(recipeHal.isConstExeBlobSupported() == (templateOfEngine.constExeBlobIndex == (recipe.blobs_nr - 1)),
                 "Invalid index of const execution blob");
    // Const blob must be last
    EAGER_ASSERT((recipeHal.isConstExeBlobSupported() == false) ||
                     ((templateOfEngine.constExeBlobOffset +
                       templateOfEngine.recipe.blobs[templateOfEngine.constExeBlobIndex].size) ==
                      templateOfEngine.recipe.execution_blobs_buffer_size),
                 "Const execution blob must be last");

    // No other patchable blobs besides the ones to have a WrBulk command for cache base regMap
    EAGER_ASSERT(templateOfEngine.getPatchableBlobsNr() == patchableBlobsNr, "Unsupported recipe template");

    // No need more than 1 write_bulk command to write all values of the base-reg-cache.
    // One patchable blob iff number of patch points smaller that cache register file entries.
    [[maybe_unused]] const bool patchPointsCheck = (recipe.patch_points_nr <= recipeHal.getBaseRegistersCacheSize());
    [[maybe_unused]] const bool baseRegMapCheck  = (templateOfEngine.blob2DescMaps.baseRegOffsetsMap.size() == 1);
    EAGER_ASSERT(patchPointsCheck == baseRegMapCheck, "Unsupported recipe template");

    // One "WREG_BULK" command per patchable blob
    std::vector<bool> blobScoreboard(recipe.blobs_nr, true);
    for (const auto& mapElm : templateOfEngine.blob2DescMaps.baseRegOffsetsMap)
    {
        EAGER_ASSERT(blobScoreboard[mapElm.blobIdx], "Unsupported recipe template");
        blobScoreboard[mapElm.blobIdx] = false;
    }
}
#endif  // #ifndef NDEBUG

RecipeInstantiation::RecipeInstantiation(recipe_t&                 recipe,
                                         const RecipeHalBase&      recipeHal,
                                         const Node2DescContainer& descriptors,
                                         const DataBuf&            stringBuf,
                                         bool                      canUseCloneFastPath)
: m_recipe(recipe),
  m_recipeHal(recipeHal),
  m_descriptors(descriptors),
  m_templates(RecipeTemplates::getInstance()),
  m_isUsingCloneFastPath(canUseCloneFastPath),
  m_arcJobWriter(recipeHal, m_constExeBlobOffset),
  m_patchableBuf(m_recipe.patching_blobs_buffer, m_recipe.patching_blobs_buffer_size),
  m_executionBuf(m_recipe.execution_blobs_buffer, m_recipe.execution_blobs_buffer_size),
  m_dynamicBuf(m_recipe.dynamic_blobs_buffer, m_recipe.dynamic_blobs_buffer_size),
  m_stringBuf(stringBuf)
{
    EAGER_ASSERT(AsicRegValType(GetCacheBaseRegMask()) == -1, "Unsupported implementation");
    m_recipe.version_major          = RECIPE_VERSION_MAJOR;
    m_recipe.version_minor          = m_recipeHal.getVersionMinor();
    m_recipe.valid_nop_kernel       = false;
    unsigned baseRegistersCacheSize = m_recipeHal.getBaseRegistersCacheSize();
    m_offsetInSections.resize(baseRegistersCacheSize);
}

// This method should be invoked once per actual recipe to init non-node-specific fields
void RecipeInstantiation::instantiateGlobalInfo(std::string_view            recipeName,
                                                WorkspaceSizesType          workspaceSize,
                                                WorkspaceSizesType          programDataSize,
                                                const ProgramDataBlobsVec&  programDataBlobs,
                                                const EagerTensorsSet&      tensorSet,
                                                bool                        isProgramDataBlobsCopyRequired,
                                                std::optional<RecipeIdType> recipeDebugId,
                                                bool                        nopKernelAdded)
{
    m_recipe.nameSize = recipeName.size() + 1;
    m_recipe.name     = m_stringBuf.cloneAllocStr(recipeName);

    // Init workspace
    EAGER_ASSERT(m_recipe.workspace_nr == 3, "Invalid number workspaces");
    m_recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]    = workspaceSize;
    m_recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = programDataSize;
    m_recipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM]      = alignUpTo(
        m_recipe.execution_blobs_buffer_size + m_recipe.patching_blobs_buffer_size + m_recipe.dynamic_blobs_buffer_size,
        m_recipeHal.getCacheLineSizeInBytes());

    if (!programDataBlobs.empty())
    {
        initProgramDataBlobs(programDataBlobs, isProgramDataBlobsCopyRequired);
        if (nopKernelAdded)
        {
            m_recipe.valid_nop_kernel   = true;
            m_recipe.nop_kernel_offset  = maskOutMemoryID(programDataBlobs.back().deviceAddr);
            m_recipe.nop_kernel_section = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;
        }
    }
    createPersistentTensorsInfo(tensorSet);

    if (unlikely(recipeDebugId.has_value()))
    {
        initProfilerDebugInfo(*recipeDebugId);
    }

    if (!m_isUsingCloneFastPath)
    {
        // TODO: Init global config different way
        const DescGeneratorBase& desc = m_descriptors.getFirstDescGen();
        const recipe_t&          recipeTmpl =
            m_templates.getTemplate(desc.getChipType(), desc.getEngineType(), desc.getPatchableTensorsNr()).recipe;

        // Copy global config
        EAGER_ASSERT(m_recipe.recipe_conf_nr == recipeTmpl.recipe_conf_nr, "Invalid conf number");
        std::memcpy(m_recipe.recipe_conf_params,
                    recipeTmpl.recipe_conf_params,
                    m_recipe.recipe_conf_nr * sizeof(gc_conf_t));
    }
}

// Fill recipe with data from given descriptors
void RecipeInstantiation::instantiateNodeSpecificInfo()
{
    EAGER_ASSERT(!m_isInstantiated, "Recipe was instantiated");
    m_isInstantiated = true;

    m_arcJobWriter.init(m_recipe.arc_jobs, m_recipe.arc_jobs_nr, m_descriptors.getStatistics());
    if (m_isUsingCloneFastPath)
    {
        EAGER_ASSERT(m_descriptors.isSingleDesc(), "");
        processClone(m_descriptors.getExecSequence().front());
    }
    else
    {
        for (const SingleNode2Desc& singleNode2Desc : m_descriptors.getExecSequence())
        {
            processNewDesc(singleNode2Desc);
        }
    }

    // Verify complete allocation consumption
    EAGER_ASSERT(m_isUsingCloneFastPath || m_curNodesNr == m_recipe.node_nr, "Inconsistent node processing");
    EAGER_ASSERT(m_isUsingCloneFastPath || m_curBlobsNr == m_recipe.blobs_nr, "Inconsistent blobs processing");
    EAGER_ASSERT(m_curPatchPointsNr == m_recipe.patch_points_nr, "Inconsistent patch points processing");
    EAGER_ASSERT(m_isUsingCloneFastPath || m_arcJobWriter.isCompleted(), "Inconsistent ECB list processing");
    EAGER_ASSERT(m_isUsingCloneFastPath || m_executionBuf.isAllocationCompleted(),
                 "Invalid execution blobs data buffer allocation");
    EAGER_ASSERT(m_isUsingCloneFastPath || m_patchableBuf.isAllocationCompleted(),
                 "Invalid patching blobs data buffer allocation");
    EAGER_ASSERT(m_isUsingCloneFastPath || m_dynamicBuf.isAllocationCompleted(),
                 "Invalid dynamic blobs data buffer allocation");
    EAGER_ASSERT(m_isUsingCloneFastPath || m_arcJobWriter.isFullAllocationUtilization(m_patchableBuf.size(),
                                                                                      m_executionBuf.size(),
                                                                                      m_dynamicBuf.size()),
                 "Partial allocation utilization");
    EAGER_ASSERT(m_stringBuf.isAllocationCompleted(), "Invalid string buffer allocation");
    EAGER_ASSERT(verifyBlobOpCodes(), "Blob opcode checker failed");

#ifndef NDEBUG
    printAllJobs(m_recipe.arc_jobs, m_recipe.arc_jobs_nr, m_recipeHal);
#endif
}

bool RecipeInstantiation::verifyBlobOpCodes() const
{
    for (size_t i = 0; i < m_recipe.blobs_nr; ++i)
    {
        const blob_t& blob = m_recipe.blobs[i];
        if (blob.blob_type.dynamic_exe) continue;  // TODO: we probably don't care about these

        EAGER_ASSERT(blob.size % sizeof(uint64_t) == 0, "");
        const size_t count = blob.size / sizeof(uint64_t);

        for (size_t j = 0; j < count; ++j)
        {
            uint64_t cmd    = static_cast<uint64_t*>(blob.data)[j];
            auto     opCode = (cmd >> 56) & 0x1f;
            switch (opCode)
            {
                case 0x1 /*WREG32*/:
                    break;
                case 0x2 /*WREG_BULK*/:
                    j += cmd & 0xffff;  // Skip extra "size" qwords
                    break;
                case 0x8 /*FENCE*/:
                case 0x13 /*WREG_64_LONG*/:
                    j += 1;  // Skip extra 1 qword
                    break;
                default:
                    LOG_CRITICAL(GC, "Unexpected opcode 0x{:X} at dword #{} within blob index {}", opCode, j, i);
                    EAGER_ASSERT(false, "");
                    return false;
            }
        }
    }
    return true;
}

// Special flow to deal with single-activation graphs
void RecipeInstantiation::processClone(const SingleNode2Desc& singleNode2Desc)
{
    // Common preparations
    m_curDescGenBase = &singleNode2Desc.getDescGen();
    EAGER_ASSERT(m_curDescGenBase->getActivationNr() == 1, "Wrong flow");
    m_curNode      = &m_curDescGenBase->getNode();
    m_curTensorsNr = m_curDescGenBase->getPatchableTensorsNr();
    m_curTemplateOfEngine =
        &m_templates.getTemplate(m_curDescGenBase->getChipType(), m_curDescGenBase->getEngineType(), m_curTensorsNr);

    for (PatchPointNrType i = 0; i < m_recipe.patch_points_nr; ++i)
    {
        m_recipe.patch_points[i].memory_patch_point.section_idx = getSectionIdAndSetOffset(i);
    }
    m_curPatchPointsNr = m_recipe.patch_points_nr;

    instantiateCacheBaseRegBlob(m_recipe.blobs);
    instantiateTensorsAddresses(m_recipe.blobs);

    switch (m_curDescGenBase->getEngineType())
    {
        case EngineType::MME:
        {
            MmeInstantiation instantiation(curTemplate(),
                                           *m_curDescGenBase,
                                           m_recipeHal,
                                           m_constExeBlobOffset,
                                           /*isFirstNode*/ true);
            instantiation.instantiateExcBlobs(0, m_recipe.blobs);
        }
        break;

        case EngineType::TPC:
        {
            TpcInstantiation instantiation(curTemplate(), *m_curDescGenBase, m_recipeHal);
            instantiation.instantiateDynBlobs(m_recipe.blobs[m_curTemplateOfEngine->dynamicBlobIndex]);
            instantiation.instantiateExcBlobs(m_recipe.blobs);
        }
        break;

        default:
            EAGER_ASSERT(0, "Unsupported device");
    }
}

// Dispatcher to fill recipe data of single node
void RecipeInstantiation::processNewDesc(const SingleNode2Desc& singleNode2Desc)
{
    // Common preparations
    m_curDescGenBase = &singleNode2Desc.getDescGen();
    m_curNode        = &m_curDescGenBase->getNode();
    m_curTensorsNr   = m_curDescGenBase->getPatchableTensorsNr();
    m_curTemplateOfEngine =
        &m_templates.getTemplate(m_curDescGenBase->getChipType(), m_curDescGenBase->getEngineType(), m_curTensorsNr);

    if (m_curNode->getId() != m_lastNodeId)
    {
        m_lastNodeId = m_curNode->getId();
        ++m_curNodesNr;
    }
    createPatchPoints();

    // Reset ARC job writer
    EAGER_ASSERT(m_curTemplateOfEngine->recipe.arc_jobs_nr == 1, "Unsupported template");
    MultiChunkArcJobWriter& arcJobWriter =
        m_arcJobWriter.setActiveWriter(m_curTemplateOfEngine->recipe.arc_jobs[0].logical_engine_id,
                                       m_patchableBuf.getPos(),
                                       m_executionBuf.getPos(),
                                       m_dynamicBuf.getPos());

    const size_t descNr = m_curDescGenBase->getDescNr();
    // Device-specific handlers
    switch (m_curDescGenBase->getEngineType())
    {
        case EngineType::MME:
        {
            MmeInstantiation instantiation(curTemplate(),
                                           *m_curDescGenBase,
                                           m_recipeHal,
                                           m_constExeBlobOffset,
                                           m_curNodesNr == 1);
            {
                blob_t* blobs = cloneBlobs();
                instantiation.instantiateDynBlobs(blobs[m_curTemplateOfEngine->dynamicBlobIndex]);
                instantiation.instantiateExcBlobs(0, blobs);
            }
            // Handle the case of multi activation
            if (m_curDescGenBase->getActivationNr() >= 2)
            {
                const recipe_t&   recipeTemplate = m_curTemplateOfEngine->recipe;
                const BlobsNrType nonExecBlobsNr = m_curTemplateOfEngine->getPatchableBlobsNr() + /*dynamic blob*/ 1;
                EAGER_ASSERT(nonExecBlobsNr < recipeTemplate.blobs_nr, "Invalid blobs");
                const BlobsNrType execBlobsNr =
                    recipeTemplate.blobs_nr - nonExecBlobsNr - (m_constExeBlobIndex.has_value() ? 1 : 0);
                const BlobsNrType enginesNr = m_recipeHal.getMaxEngines(EngineType::MME);
                for (BlobsNrType i = enginesNr; i < descNr; i += enginesNr)
                {
                    blob_t* blobs = cloneExecBlobs(nonExecBlobsNr, execBlobsNr);
                    instantiation.instantiateExcBlobs(i, blobs);
                }
                instantiation.instantiateArcJobs(arcJobWriter);
            }
        }
        break;

        case EngineType::TPC:
        {
            EAGER_ASSERT(descNr == 1, "Unsupported multiple TPC activations");
            TpcInstantiation instantiation(curTemplate(), *m_curDescGenBase, m_recipeHal);
            blob_t*          blobs = cloneBlobs();
            instantiation.instantiateDynBlobs(blobs[m_curTemplateOfEngine->dynamicBlobIndex]);
            instantiation.instantiateExcBlobs(blobs);
        }
        break;

        case EngineType::DMA:
        {
            EAGER_ASSERT(GCFG_EDMA_NUM_BINNED.value() == 0, "Binning is not supported in eager mode");

            const recipe_t&   recipeTemplate = m_curTemplateOfEngine->recipe;
            const BlobsNrType nonExecBlobsNr = m_curTemplateOfEngine->getPatchableBlobsNr() + /*dynamic blob*/ 1;
            EAGER_ASSERT(nonExecBlobsNr < recipeTemplate.blobs_nr, "Invalid blobs");
            bool             isNopDescNeeded = m_curDescGenBase->isDmaNopDescNeeded();
            size_t           nodeDescNr      = descNr - (isNopDescNeeded ? 1 : 0);
            DmaInstantiation instantiation(curTemplate(),
                                           *m_curDescGenBase,
                                           m_recipeHal,
                                           arcJobWriter,
                                           nonExecBlobsNr,
                                           nodeDescNr,
                                           m_constExeBlobOffset,
                                           reinterpret_cast<const char*>(m_recipe.execution_blobs_buffer));
            blob_t*          firstBlobs = cloneBlobs();
            instantiation.initialize(firstBlobs,
                                     m_curTemplateOfEngine->dynamicBlobIndex,
                                     m_curTemplateOfEngine->patchingBlobIndex,
                                     reinterpret_cast<const char*>(m_recipe.patching_blobs_buffer),
                                     isNopDescNeeded);
            const BlobsNrType execBlobsNr =
                recipeTemplate.blobs_nr - nonExecBlobsNr - (m_constExeBlobIndex.has_value() ? 1 : 0);
            for (size_t i = (isNopDescNeeded ? 0 : 1); i < nodeDescNr; ++i)
            {
                blob_t* blobs = cloneExecBlobs(nonExecBlobsNr, execBlobsNr);
                instantiation.instantiateExcBlobs(i, blobs);
                instantiation.addExecBlobsToStaticEcbs(blobs + nonExecBlobsNr, i);
            }
            instantiation.finalize(firstBlobs + nonExecBlobsNr, isNopDescNeeded);
        }
        break;

        default:
            EAGER_ASSERT(0, "Unsupported device");
    };

    // Try to copy ECBs from template
    arcJobWriter.tryToCopyArcJob(*m_curTemplateOfEngine);
}

// Initialize new blobs to be instantiated from recipe template.
// The method returns a pointer to actual non-patchable blobs.
blob_t* RecipeInstantiation::cloneBlobs()
{
    const TemplateOfEngine& templateOfEngine = curTemplate();
    // Initialize and return a pointer to the new blobs
    blob_t*         curActualBlobs = m_recipe.blobs + m_curBlobsNr;
    const recipe_t& recipeTemplate = templateOfEngine.recipe;

    // Copy content all blobs from the template to the instantiated recipe
    const BlobsNrType blobsNr = recipeTemplate.blobs_nr - (m_constExeBlobIndex.has_value() ? 1 : 0);
    m_curBlobsNr += blobsNr;
    EAGER_ASSERT(m_curBlobsNr <= m_recipe.blobs_nr, "Invalid allocation of blobs");
    for (BlobsNrType i = 0; i < blobsNr; ++i)
    {
        const blob_t& templateBlob = recipeTemplate.blobs[i];
        blob_t&       actualBlob   = curActualBlobs[i];
        // Initialize data pointer of blob_t
        actualBlob.size          = templateBlob.size;
        actualBlob.blob_type_all = templateBlob.blob_type_all;
        switch (actualBlob.blob_type_all)
        {
            case blob_t::EXE:
                actualBlob.data = m_executionBuf.allocate(actualBlob.size);
                std::memcpy(actualBlob.data, templateBlob.data, actualBlob.size);
                if (templateOfEngine.constExeBlobIndex == i)
                {
                    EAGER_ASSERT(m_constExeBlobIndex.has_value() == false, "Invalid const blob tracking");
                    m_constExeBlobIndex  = i;
                    m_constExeBlobOffset = templateOfEngine.constExeBlobOffset;
                }
                break;

            case blob_t::PATCHING:
                actualBlob.data = m_patchableBuf.allocate(actualBlob.size);
                std::memcpy(actualBlob.data, templateBlob.data, actualBlob.size);
                break;

            case blob_t::DYNAMIC:
                actualBlob.size *= m_curDescGenBase->getRequiredWdCtxNr();
                actualBlob.data = m_dynamicBuf.allocate(actualBlob.size);
                break;

            default:
                EAGER_ASSERT(0, "Unsupported blob type");
        };
    }

    instantiateCacheBaseRegBlob(curActualBlobs);
    instantiateTensorsAddresses(curActualBlobs);
    return curActualBlobs;
}

// Initialize new execution blobs to be instantiated from recipe template.
// The method returns a pointer to actual non-patchable blobs.
blob_t* RecipeInstantiation::cloneExecBlobs(BlobsNrType nonExecBlobsNr, BlobsNrType execBlobsNr)
{
    const TemplateOfEngine& templateOfEngine = curTemplate();
    // Initialize and return a pointer to the new blobs
    blob_t*         curActualBlobs = m_recipe.blobs + m_curBlobsNr;
    const recipe_t& recipeTemplate = templateOfEngine.recipe;

    // Copy content all blobs from the template to the instantiated recipe
    m_curBlobsNr += execBlobsNr;
    EAGER_ASSERT(m_curBlobsNr <= m_recipe.blobs_nr, "Invalid allocation of blobs");

    // Copy execution blobs from template to actual recipe and adjust data pointers
    const blob_t* templateEexecBlobs = recipeTemplate.blobs + nonExecBlobsNr;
    for (BlobsNrType i = 0; i < execBlobsNr; ++i)
    {
        const blob_t& templateBlob = templateEexecBlobs[i];
        EAGER_ASSERT(templateBlob.blob_type_all == blob_t::EXE, "Expected execution blobs only");
        blob_t& actualBlob = curActualBlobs[i];
        // Initialize data pointer of blob_t
        actualBlob.size          = templateBlob.size;
        actualBlob.blob_type_all = blob_t::EXE;
        actualBlob.data          = m_executionBuf.allocate(actualBlob.size);
        std::memcpy(actualBlob.data, templateBlob.data, actualBlob.size);
    }

    // curActualBlobs will be returned to include junky blobs to keep calculations relative to
    // non-patchable blobs that suppose to be located at the beginning.
    curActualBlobs -= nonExecBlobsNr;
    instantiateTensorsAddresses(curActualBlobs);
    return curActualBlobs;
}

// Store tensor info of inputs and outputs. This method will be invoked once per node.
void RecipeInstantiation::createPersistentTensorsInfo(recipe_t&              recipe,
                                                      const EagerTensorsSet& tensorsSet,
                                                      StringBufAllocator&    stringBuf)
{
    TensorsNrType                curPersistentTensorIdx = 0;
    const VecTensors<TensorPtr>& tensors                = tensorsSet.getTensors();
    for (TensorsNrType i = 0; i < tensors.size(); ++i)
    {
        const Tensor& tensor = *tensors[i];
        if (!tensor.isPersistent()) continue;
        const auto sectionIdx = static_cast<SectionIdxType>(tensor.getMemorySectionID());
        EAGER_ASSERT(sectionIdx != MEMORY_ID_RESERVED_FOR_WORKSPACE, "Only persistent tensors are expected");

        // Process new tensor
        auto& recipeTensor                  = recipe.tensors[curPersistentTensorIdx++];
        recipeTensor.isInput                = tensorsSet.isGraphInput(i);
        recipeTensor.name                   = stringBuf.cloneAllocStr(tensor.getName());
        recipeTensor.section_idx            = sectionIdx;
        recipeTensor.layout                 = nullptr;
        recipeTensor.offset_in_section      = tensor.getMemorySectionOffset();
        recipeTensor.size                   = tensor.getTotalSizeInBytes();
        recipeTensor.elementType            = tensor.getElementType();
        recipeTensor.dimensions             = tensor.getDim();
        recipeTensor.tensorType             = tensor.getTensorType();
        recipeTensor.isExternal             = false;  // ignore SFG info in eager mode
        recipeTensor.extTensorExeOrder      = std::numeric_limits<uint32_t>::max();
        recipeTensor.multi_views_indices_nr = 0;
        recipeTensor.multi_views_indices    = nullptr;
        tensor.getAllNSizesInElements(recipeTensor.dimensionsSize);
#ifndef NDEBUG  // The following fields are not required, init them for debug readability
        recipeTensor.zp           = 0;
        recipeTensor.scale        = 1;
        recipeTensor.batchSize    = 0;
        recipeTensor.section_type = 0;
#endif

        // The permutation may be skipped for some executions but it's always needed for the query API
        if (const std::optional<gc::Permutation>& perm = tensor.getPermutation(); perm.has_value())
        {
            const auto& permVec = perm->getValues();
            const auto  count   = permVec.size();
            std::copy_n(permVec.begin(), count, std::begin(recipeTensor.permutation));
            std::iota(std::begin(recipeTensor.permutation) + count, std::end(recipeTensor.permutation), count);
        }
        else
        {
            std::iota(std::begin(recipeTensor.permutation), std::end(recipeTensor.permutation), 0);
        }
    }
    EAGER_ASSERT(curPersistentTensorIdx == recipe.persist_tensors_nr, "Invalid persistent tensor number");
}

void RecipeInstantiation::createPersistentTensorsInfo(const EagerTensorsSet& tensorsSet)
{
    createPersistentTensorsInfo(m_recipe, tensorsSet, m_stringBuf);
}

static void
addDebugInfo(node_symbol_info_t& result, StringBufAllocator& stringBuf, const DescGeneratorBase& descGenBase)
{
    // TODO: see comment about fixed dataType table in debugNodeStringLen
    const Node& node         = descGenBase.getNode();
    result.device_type       = node.getNodeDeviceType();
    result.context_id        = node.getContextId();
    result.full_context_id   = node.getFullContextId();
    result.num_descriptors   = descGenBase.getDescNr();
    result.kernel_blob_index = -1;  // DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT
    result.node_name         = stringBuf.cloneAllocStr(node.getNodeName());
    result.operation         = stringBuf.cloneAllocStr(node.getGUID());
    result.data_type         = stringBuf.cloneAllocStr(
        getStringFromSynDataType((node.getNumOutputs() ? node.getOutput(0) : node.getInput(0))->getElementType()));

    if (descGenBase.getEngineType() == EngineType::TPC)
    {
        static constexpr auto TPC_ROI_COUNT = 1;
        EAGER_ASSERT(result.num_rois == TPC_ROI_COUNT && result.num_working_engines != nullptr, "");

        const auto& utils = descGenBase.getNode().getNodeAnnotation().tpcMetaData.utilizationPerLogicalRoi;
        EAGER_ASSERT(utils.size() == TPC_ROI_COUNT, "Need to add support for multiple TPC ROIs");

        *result.num_working_engines = 0;
        for (const auto& dcore : utils[0])
        {
            EAGER_ASSERT(*result.num_working_engines + dcore.totalNumWorkingEngines <=
                             std::numeric_limits<std::remove_reference_t<decltype(*result.num_working_engines)>>::max(),
                         "Total engine number overflow");
            *result.num_working_engines += dcore.totalNumWorkingEngines;
        }
    }
    else
    {
        EAGER_ASSERT(result.num_rois == 0 && result.num_working_engines == nullptr, "");
    }
}

void RecipeInstantiation::initProfilerDebugInfo(RecipeIdType recipeDebugId)
{
    m_recipe.debug_profiler_info.version_major = RECIPE_DEBUG_INFO_VERSION_MAJOR;
    m_recipe.debug_profiler_info.version_minor = RECIPE_DEBUG_INFO_VERSION_MINOR;
    m_recipe.debug_profiler_info.recipe_id     = recipeDebugId;

    const auto& singleNode2Descs = m_descriptors.getExecSequence();
    for (size_t i = 0; i < singleNode2Descs.size(); ++i)
    {
        addDebugInfo(m_recipe.debug_profiler_info.nodes[i], m_stringBuf, singleNode2Descs[i].getDescGen());
    }
}

// Return section ID of tensor for the given index
SectionIdxType RecipeInstantiation::getSectionIdAndSetOffset(PatchPointNrType tensorIdx)
{
    static_assert(sizeof(deviceAddrOffset) == sizeof(uint64_t));
    EAGER_ASSERT_PTR(m_curNode);
    EAGER_ASSERT_PTR(m_curDescGenBase);
    // TPC kernel has a special handling
    if (tensorIdx == m_curTensorsNr && m_curDescGenBase->getEngineType() == EngineType::TPC)
    {
        return MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;
    }
    deviceAddrOffset virtualAddr = m_curDescGenBase->getTensorVirtualAddress(tensorIdx);
    EAGER_ASSERT(tensorIdx < m_offsetInSections.size(), "Invalid index for TPC node");
    EAGER_ASSERT(virtualAddr != -1, "Invalid virtual address");
    SectionIdxType section = -1;
    getSectionInfoFromVirtualAddress(virtualAddr, section, m_offsetInSections[tensorIdx]);
    return section;
}

// Create new patch points from scratch. It's not worth to memcpy them because of small number of fields
// to be taken from the template. But mainly because patch point number is equal to sections number,
// thus we cannot rely on template to reflect the actual number of patch points.
// This method should be invoked once per node.
void RecipeInstantiation::createPatchPoints()
{
    // Starting position of current actual patch points (to be initialized)
    patch_point_t* curPatchPoints = m_recipe.patch_points + m_curPatchPointsNr;

    // Update patch point tracker
    const PatchPointNrType templatePatchPointsNr = curTemplate().recipe.patch_points_nr;
    EAGER_ASSERT(templatePatchPointsNr >= 1, "Recipe template must have at least one patch point");
    m_curPatchPointsNr += templatePatchPointsNr;
    EAGER_ASSERT(m_curPatchPointsNr <= m_recipe.patch_points_nr, "Invalid allocation of patch points");
    // Initialize the new patch points to zero
    std::memset(curPatchPoints, 0, templatePatchPointsNr * sizeof(patch_point_t));

    // Complete initialization to match actual values of tensors
    EAGER_ASSERT(curTemplate().getPatchableBlobsNr() == 1, "This implementation assumes only one patchable blob");
    const BlobsNrType templateBaseRegBlobIdx = curTemplate().recipe.patch_points[0].blob_idx;
    const BlobsNrType actualBaseRegBlobIdx   = templateBaseRegBlobIdx + m_curBlobsNr;

    for (PatchPointNrType i = 0; i < templatePatchPointsNr; ++i)
    {
        curPatchPoints[i].type     = defaultPatchPointType;
        curPatchPoints[i].blob_idx = actualBaseRegBlobIdx;
        // Update section, it's affected by node - not by how we split it
        curPatchPoints[i].memory_patch_point.section_idx = getSectionIdAndSetOffset(i);
        curPatchPoints[i].dw_offset_in_blob              = asicRegsPerEntry * (i + 1);
        curPatchPoints[i].node_exe_index                 = m_curNodesNr;
    }

    EAGER_ASSERT(m_curNodesNr >= 1, "Invalid node index");
    m_recipe.node_exe_list[m_curNodesNr - 1].patch_points_nr = templatePatchPointsNr;
}

// This method assumes all patchable blobs are copied from template to actual recipe.
// It's assumed to be invoked once per node and after initializing patch points.
// It modifies commands of actual patchable blobs to match section ids of patch points.
// As cloneBlobs(...) stated, all blobs in curActualBlobs are valid (no junky blobs supposed to be).
void RecipeInstantiation::instantiateCacheBaseRegBlob(blob_t* curActualBlobs)
{
    const PatchPointNrType templatePatchPointsNr = curTemplate().recipe.patch_points_nr;
    // Iterate over base reg offsets map to find out patchable blobs
    const Blob2StructPosMap& baseRegsMap = curTemplate().blob2DescMaps.baseRegOffsetsMap;
    // The following loop is expected to iterate one time in most cases. 16 tensors cases per node are rare.
    for (const auto& mapElm : baseRegsMap)
    {
        Byte* rawDataBlob = static_cast<Byte*>(curActualBlobs[mapElm.blobIdx].data) + mapElm.blobPos;
        EAGER_ASSERT_PTR(rawDataBlob);
        auto* actualRegs = reinterpret_cast<deviceAddrOffset*>(rawDataBlob);
        for (PatchPointNrType i = 0; i < templatePatchPointsNr; ++i)
        {
            const PatchPointNrType patchPointIdx = m_curPatchPointsNr - templatePatchPointsNr + i;
            EAGER_ASSERT(m_curPatchPointsNr >= templatePatchPointsNr, "Invalid patch points");
            EAGER_ASSERT(patchPointIdx < m_recipe.patch_points_nr, "Invalid patch point index");

            const SectionIdxType sectionId = m_recipe.patch_points[patchPointIdx].memory_patch_point.section_idx;
            // Cache base reg value can be acquired by setting offset in section to zero
            actualRegs[i] = getVirtualAddressForMemoryID(sectionId, /*section offset*/ 0);
        }
    }
}

// Map a given tensor index to cache base reg id
CacheBaseRegIdType RecipeInstantiation::getCacheBaseRegId(PatchPointNrType tensorIdx) const
{
    EAGER_ASSERT(curTemplate().getPatchableBlobsNr() == 1, "This implementation assumes only one patchable blob");
    // Check for TPC kernel - it's equal to -1
    if (tensorIdx == -1)
    {
        return curTemplate().recipe.patch_points_nr - 1;  // Kernel reg follows tensors
    }
    EAGER_ASSERT(tensorIdx < curTemplate().recipe.patch_points_nr, "Invalid tensor index");
    return tensorIdx;
}

// Modify "WREG_64_LONG" commands to point to the right base reg.
// As cloneBlobs(...) stated, curActualBlobs can have some junky blobs at the beginning.
void RecipeInstantiation::instantiateTensorsAddresses(blob_t* curActualBlobs) const
{
    for (const auto& mapElm : curTemplate().blob2DescMaps.wrAddrMapPosMap)
    {
        EAGER_ASSERT(mapElm.blobIdx < curTemplate().recipe.blobs_nr, "Blob index is out of bound");
        auto* blobData = static_cast<Byte*>(curActualBlobs[mapElm.blobIdx].data);

        // Modify base reg id
        const CacheBaseRegIdType baseRegId = getCacheBaseRegId(mapElm.tensorIdx);
        auto& baseRegPosInBlob             = *reinterpret_cast<QmanCommandSizeType*>(blobData + mapElm.baseRegPos.base);
        baseRegPosInBlob = (baseRegPosInBlob & mapElm.baseRegPos.mask) | (baseRegId << mapElm.baseRegPos.offset);

        // Set offset in section of a kernel or data tensor
        TensorAddressType offset = 0;
        if (mapElm.tensorIdx != -1)
        {
            EAGER_ASSERT(baseRegId < m_offsetInSections.size(), "Invalid base register ID");
            offset = m_offsetInSections[baseRegId];
        }
        else
        {
            EAGER_ASSERT(m_curDescGenBase->getEngineType() == EngineType::TPC, "Unsupported device");
            offset = maskOutMemoryID(static_cast<const TPCNode*>(m_curNode)->getKernelOffsetInSection());
        }

        if (mapElm.isQWord)
        {
            auto& offsetField = *reinterpret_cast<TensorAddressType*>(blobData + mapElm.offsetPos);
            offsetField       = offset;
        }
        else
        {
            auto& offsetField = *reinterpret_cast<AsicRegValType*>(blobData + mapElm.offsetPos);
            offsetField       = static_cast<AsicRegValType>(offset);
        }
    }
}

// Init "recipe_t::program_data_blobs_buffer" with actual data.
// There will be program data blobs when at least one TPC node in the graph.
void RecipeInstantiation::initProgramDataBlobs(const ProgramDataBlobsVec& programDataBlobs,
                                               bool                       isProgramDataBlobsCopyRequired)
{
    EAGER_ASSERT(m_recipe.valid_nop_kernel || (m_descriptors.getStatistics(EngineType::TPC).nodeNum >= 1),
                 "Invalid program data");
    program_data_blob_t* pFiller    = m_recipe.program_data_blobs;

    // for a single program data blob we can avoid copy altogether and just point
    // to the retrieved tpc kernel.
    if (!isProgramDataBlobsCopyRequired && programDataBlobs.size() == 1)
    {
        const ProgramDataBlob& blob        = programDataBlobs.front();
        m_recipe.program_data_blobs_buffer = const_cast<char*>(blob.hostAddrPtr);
        pFiller->size                      = blob.binSize;
        pFiller->data                      = blob.hostAddrPtr;
        pFiller->offset_in_section         = maskOutMemoryID(blob.deviceAddr);
        pFiller->section_idx               = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;
        return;
    }

    unsigned             dataOffset = 0;
    for (const ProgramDataBlob& blob : programDataBlobs)
    {
        EAGER_ASSERT((dataOffset + blob.binSize) <= m_recipe.program_data_blobs_size,
                     "m_dataBlobsSizeInBytes overflow");
        EAGER_ASSERT(blob.hostAddrPtr, "host pointer was not set, this is impossible");
        pFiller->size = blob.binSize;
        pFiller->data = m_recipe.program_data_blobs_buffer + dataOffset;
        memcpy((char*)pFiller->data, blob.hostAddrPtr, blob.binSize);
        pFiller->offset_in_section = maskOutMemoryID(blob.deviceAddr);
        pFiller->section_idx       = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;

        // All GC allocations should have been done with cache-line alignment
        BlobSizeType alignedBlobSize = alignSizeUp(blob.binSize, m_recipeHal.getCacheLineSizeInBytes());
        dataOffset += alignedBlobSize;

        // set padding bytes to 0xFF (TPC NOP) similarly to graph mode as protection against rouge kernels
        // reading beyond the kernel boundary (SW-80804).
        // This protection is not bullet proof as the presence of the padding bytes and size is use case
        // dependent based on present kernels\const\auxilary tensors in the program data buffer and we also
        // have an optimization to avoid copy of the tpc kernel in case of a single program data blob entry
        // (isProgramDataBlobsCopyRequired).
        // Ideally we need to detect all such kernels and fix them, but adding an optimistic protection
        // is cheap and can help mitigate some cases which might be hard to catch.
        memset((char*)pFiller->data + blob.binSize, 0xFF, alignedBlobSize - blob.binSize);

        ++pFiller;
    }
}

}  // namespace eager_mode

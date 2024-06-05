
#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/tensor_info.h"
#include "program_data_blob_manager.h"
#include "recipe_gen/recipe_arc_job_writer.h"
#include "recipe_gen/recipe_defs.h"
#include "utils/general_defs.h"
#include "utils/memory_utils.h"

// synapse-internal includes
#include "graph_compiler/types.h"

// synapse api (relative to include/)
#include "internal/recipe.h"
#include "synapse_api_types.h"

// std includes
#include <optional>

class Node;

namespace eager_mode
{
class DescGeneratorBase;
class EagerRecipeAllocator;
class Node2DescContainer;
class RecipeTemplates;
class SingleNode2Desc;
struct TemplateOfEngine;

// Instantiate actual recipe by copying values from descriptors to recipe by following recipe templates.
// This process assumes actual recipe to be fully allocated by runtime memory allocator.
class RecipeInstantiation
{
public:
    RecipeInstantiation(recipe_t&                 recipe,
                        const RecipeHalBase&      recipeHal,
                        const Node2DescContainer& descriptors,
                        const DataBuf&            stringBuf,
                        bool                      canUseCloneFastPath);

#ifndef NDEBUG
    static void checkTemplateAssumptions(const TemplateOfEngine& templateOfEngine, const RecipeHalBase& recipeHal);
#endif  // #ifndef NDEBUG

    void instantiateGlobalInfo(std::string_view            recipeName,
                               WorkspaceSizesType          workspaceSize,
                               WorkspaceSizesType          programDataSize,
                               const ProgramDataBlobsVec&  programDataBlobs,
                               const EagerTensorsSet&      tensorSet,
                               bool                        isProgramDataBlobsCopyRequired,
                               std::optional<RecipeIdType> recipeDebugId,
                               bool                        nopKernelAdded);
    void instantiateNodeSpecificInfo();

    static void
    createPersistentTensorsInfo(recipe_t& recipe, const EagerTensorsSet& tensorsSet, StringBufAllocator& stringBuf);

private:
    const TemplateOfEngine& curTemplate() const
    {
        EAGER_ASSERT_PTR(m_curTemplateOfEngine);
        return *m_curTemplateOfEngine;
    }

    bool               verifyBlobOpCodes() const;
    void               processClone(const SingleNode2Desc& singleNode2Desc);
    void               processNewDesc(const SingleNode2Desc& singleNode2Desc);
    blob_t*            cloneBlobs();
    blob_t*            cloneExecBlobs(BlobsNrType nonExecBlobsNr, BlobsNrType execBlobsNr);
    void               createPersistentTensorsInfo(const EagerTensorsSet& tensorsSet);
    void               initProfilerDebugInfo(RecipeIdType recipeDebugId);
    SectionIdxType     getSectionIdAndSetOffset(PatchPointNrType tensorIdx);
    void               createPatchPoints();
    CacheBaseRegIdType getCacheBaseRegId(PatchPointNrType tensorIdx) const;
    void               instantiateCacheBaseRegBlob(blob_t* curActualBlobs);
    void               instantiateTensorsAddresses(blob_t* curActualBlobs) const;
    void initProgramDataBlobs(const ProgramDataBlobsVec& programDataBlobs, bool isProgramDataBlobsCopyRequired);

private:
    // Variables to prepare at constructor
    recipe_t&                 m_recipe;
    const RecipeHalBase&      m_recipeHal;
    const Node2DescContainer& m_descriptors;
    const RecipeTemplates&    m_templates;
    const bool                m_isUsingCloneFastPath;  // Use special allocation for single-activation graphs
    // Variables that tracks current status of recipe instantiation
    bool             m_isInstantiated   = false;
    NodesNrType      m_curNodesNr       = 0;
    PatchPointNrType m_curTensorsNr     = 0;
    BlobsNrType      m_curBlobsNr       = 0;
    PatchPointNrType m_curPatchPointsNr = 0;
    synNodeId        m_lastNodeId       = -1;
    ArcJobsWriter    m_arcJobWriter;  // Tracker to allow writing to ARC jobs any time
    llvm_vecsmall::SmallVector<TensorAddressType, RecipeHalBase::maxBaseRegistersCacheSize>
                                m_offsetInSections;    // Offset in section of data tensors
    std::optional<BlobsNrType>  m_constExeBlobIndex;   // Index of const execution blob
    std::optional<BlobSizeType> m_constExeBlobOffset;  // Offset relative to execution blob buffer

    // Blob data allocators and trackers
    DataBufAllocator<RECIPE_PINTER_TYPE(patching_blobs_buffer)>  m_patchableBuf;
    DataBufAllocator<RECIPE_PINTER_TYPE(execution_blobs_buffer)> m_executionBuf;
    DataBufAllocator<RECIPE_PINTER_TYPE(dynamic_blobs_buffer)>   m_dynamicBuf;
    StringBufAllocator                                           m_stringBuf;

    // Other variables
    const DescGeneratorBase* m_curDescGenBase      = nullptr;  // Pointer to current descriptor in process
    const Node*              m_curNode             = nullptr;  // Pointer to current node in process
    const TemplateOfEngine*  m_curTemplateOfEngine = nullptr;  // Pointer to template that is associated to current node
};

}  // namespace eager_mode

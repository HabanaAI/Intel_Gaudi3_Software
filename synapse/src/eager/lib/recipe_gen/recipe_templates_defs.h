#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/blob_to_desc_map.h"
#include "recipe_gen/recipe_arc_job_utils.h"
#include "recipe_gen/recipe_defs.h"
#include "tpc_node.h"
#include "utils/general_defs.h"

// std includes
#include <array>
#include <cstdalign>
#include <cstddef>
#include <memory>
#include <vector>

struct recipe_t;

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Recipe Templates Frontend API
///////////////////////////////////////////////////////////////////////////////////////////////////

// Per-chip template - a static information required to do the instantiation of actual recipe
struct TemplateOfEngine
{
    virtual ~TemplateOfEngine() = default;

    const recipe_t&     recipe;                    // Reference to recipe template
    Blob2DescMaps       blob2DescMaps       = {};  // Offsets in blobs to be patched
    EcbCmdsListInfo     ecbsNetSize         = {};  // Net size of ECBs w\o padding, head and tail
    BlobsNrType         descNr              = 0;   // Number of descriptors in the template
    StructSizeType      allocSize           = 0;   // Size of TemplateData to be used for clone operation
    SecondaryEngineType secondaryEngineType = SecondaryEngineType::NONE;  // Distinguish multiple rules of same engine
    // Special blob indices and offsets
    BlobsNrType  patchingBlobIndex  = -1;  // Index of patching blob
    BlobsNrType  dynamicBlobIndex   = -1;  // Index of dynamic blob
    BlobsNrType  constExeBlobIndex  = -1;  // Index of const execution blob
    BlobSizeType constExeBlobOffset = -1;  // Offset of const execution blob relative to execution blob buffer
    // MME and DMA can produce multiple descriptors. Recipe template contains a NOP at dynamic ECB to fix misalignment
    // of FENCE command. That NOP is optional. The following variable contribute in total size to be allocated:
    virtual EcbCommandSizeType calcExtraAlignmentNopsSizeOfDynamicEcb(size_t activationsNr) const = 0;

    TemplateOfEngine(const recipe_t& recipeTemplate) : recipe(recipeTemplate) {}
    BlobsNrType getPatchableBlobsNr() const { return blob2DescMaps.baseRegOffsetsMap.size(); }
};

// Alias for components of templates container - for compactness
using TemplateOfEnginePtr = std::unique_ptr<TemplateOfEngine>;
using TemplatesOfEngine   = std::vector<TemplateOfEnginePtr>;  // See note *
using TemplatesOfChip     = std::array<TemplatesOfEngine, static_cast<unsigned>(EngineType::ENGINES_NR)>;
using TemplatesOfAllChips = std::array<TemplatesOfChip, static_cast<unsigned>(ChipType::CHIPS_NR)>;
using NOPKernelsofAllChips = std::array<KernelInfo, static_cast<unsigned>(ChipType::CHIPS_NR)>;

// Note *: For TemplatesOfEngine we prefer to use a vector over SmallVector, because templates of
//         engines have variable numbers, TemplateOfEngine is not small, and these object live till
//         application exist. Small vector advantage is allocating a memory on the stack which is
//         faster than on heap. In order to utilize this privilege we need to fix vector's size to
//         the maximum possible number of templates (in Gaudi2 it's 15 for TPC), leading to engines
//         with less that that to have empty templates. Status que shows high standard deviation of
//         relative to that maximum, leaving high percentage of unutilized memory occupied by small
//         vector. By considering these facts the trade off is to use std's vector over llvm's SmallVector

///////////////////////////////////////////////////////////////////////////////////////////////////
// Recipe Templates Backend API
///////////////////////////////////////////////////////////////////////////////////////////////////

// Contiguous Storage representation for recipe_t followed by its data buffers
template<BlobSizeType dataBufSize>
struct TemplateData
{
    recipe_t recipe;                                  // recipe_t of the template
    alignas(uint64_t) Byte dataBuffers[dataBufSize];  // Data buffers used by the recipe
};
// This representation is good for cloning template to be the initial state of actual recipe
// of a single-node graph. This assumption prevents us to add unecessary variables in-between.
static_assert(offsetof(TemplateData<1>, recipe) + sizeof(recipe_t) == offsetof(TemplateData<1>, dataBuffers));

// Interface of engine creators that are chip-specific
template<TensorsNrType tensorsNr>
class TemplateOfEngineCreatorBase
{
public:
    virtual ~TemplateOfEngineCreatorBase() = default;
    // dataBufSize is provided to save runtime calculation of buffer's size and for API robustness
    virtual void create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const = 0;
    // Return size of NOP if it's relevant to dynamic Ecb misalignment
    virtual std::optional<EcbCommandSizeType> getNopSizeForDynamicEcbMisalignment() const = 0;
};

// Owner of TemplateData - the full information that is associated to TemplateOfEngine as reference to recipe_t
template<TensorsNrType tensorsNr, BlobSizeType dataBufSize>
class TemplateOfEngineBuilder : public TemplateOfEngine
{
public:
    TemplateOfEngineBuilder(const TemplateOfEngineCreatorBase<tensorsNr>& creator);
    // Default function to preserve 8 bytes alignment in dynamic ECB
    inline virtual EcbCommandSizeType calcExtraAlignmentNopsSizeOfDynamicEcb(size_t activationsNr) const override;

private:
    // Ownership of all recipe_t storage
    TemplateData<dataBufSize> m_templateData;
    // Size of NOP if it's relevant to dynamic Ecb misalignment
    const std::optional<EcbCommandSizeType> m_nopSizeForDynamicEcbMisalignment;
};

// Interface creator of all chip-specific templates
struct TemplatesCreatorBase
{
    virtual ~TemplatesCreatorBase()                       = default;
    virtual void create(TemplatesOfChip& templatesOfChip) = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr, BlobSizeType dataBufSize>
TemplateOfEngineBuilder<tensorsNr, dataBufSize>::TemplateOfEngineBuilder(
    const TemplateOfEngineCreatorBase<tensorsNr>& creator)
: TemplateOfEngine(m_templateData.recipe),
  m_nopSizeForDynamicEcbMisalignment(creator.getNopSizeForDynamicEcbMisalignment())
{
    allocSize = sizeof(m_templateData);
    std::memset(&m_templateData, 0, sizeof(m_templateData));
    creator.create(m_templateData.recipe, m_templateData.dataBuffers, dataBufSize);
}

template<TensorsNrType tensorsNr, BlobSizeType dataBufSize>
EcbCommandSizeType
TemplateOfEngineBuilder<tensorsNr, dataBufSize>::calcExtraAlignmentNopsSizeOfDynamicEcb(size_t activationsNr) const
{
    // Single activation must have one alignment NOP
    if (!m_nopSizeForDynamicEcbMisalignment.has_value() || activationsNr == 1) return 0;
    // In general is number of activations is odd then one NOP is required, otherwise none
    return ((activationsNr & 1) == 1 ? activationsNr - 1 : activationsNr) * *m_nopSizeForDynamicEcbMisalignment;
}

}  // namespace eager_mode
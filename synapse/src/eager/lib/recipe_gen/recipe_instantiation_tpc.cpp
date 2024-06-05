#include "recipe_instantiation_tpc.h"

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "recipe_gen/blob_to_desc_map.h"
#include "recipe_gen/recipe_hal_base.h"
#include "recipe_gen/recipe_templates_defs.h"
#include "utils/general_defs.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <cstring>

namespace eager_mode
{
TpcInstantiation::TpcInstantiation(const TemplateOfEngine&  tpcTemplate,
                                   const DescGeneratorBase& descGenerator,
                                   const RecipeHalBase&     recipeHal)
: m_template(tpcTemplate), m_descGenerator(descGenerator), m_recipeHal(recipeHal)
{
    EAGER_ASSERT(m_descGenerator.getDescNr() == 1, "Unsupported multiple TPC activations");
}

void TpcInstantiation::instantiateDynBlobs(blob_t& actualBlob)
{
    EAGER_ASSERT(actualBlob.blob_type.dynamic_exe, "Invalid dynamic blob");
    const StructSizeType wdCtxtSize = m_recipeHal.getWorkDistributionContextSize(EngineType::TPC);
    EAGER_ASSERT((m_descGenerator.getDescNr() * wdCtxtSize) == actualBlob.size, "Invalid allocation of dynamic blob");
    // This assert must be here although we have it at constructor:
    EAGER_ASSERT(m_descGenerator.getDescNr() == 1, "TODO: Support multiple descriptors");
    // Copy work distribution context
    std::memcpy(static_cast<Byte*>(actualBlob.data), m_descGenerator.getWorkDistributionContextRaw(0), wdCtxtSize);
}

// Main method to fill all field in recipe that are affected by adding single TPC descriptor
void TpcInstantiation::instantiateExcBlobs(blob_t* actualBlobs)
{
    EAGER_ASSERT_PTR(actualBlobs);
    const auto& descRaw = m_descGenerator.getDescRaw(0);  // Raw data
    // Modify the rest of descriptor (bypass first index)
    for (const auto& mapElm : m_template.blob2DescMaps.execDescMap)
    {
        EAGER_ASSERT(mapElm.blobIdx < m_template.recipe.blobs_nr, "Blob index is out of bound");
        auto&              blob     = actualBlobs[mapElm.blobIdx];
        const BlobSizeType dataSize = mapElm.regsNr * sizeOfAsicRegVal;
        EAGER_ASSERT((mapElm.blobPos + dataSize) <= blob.size, "Blob pos is out of bound");
        EAGER_ASSERT((mapElm.structPos + dataSize) <= m_recipeHal.getDescSize(EngineType::TPC),
                     "Descriptor pos is out of bound");
        // Pointers in raw data
        Byte*       blobPos = static_cast<Byte*>(blob.data) + mapElm.blobPos;
        const Byte* descPos = descRaw + mapElm.structPos;
        // Instantiate the new blobs based on descriptor
        if (mapElm.regsNr == 1)
        {
            auto&       blobVal = *reinterpret_cast<AsicRegValType*>(blobPos);
            const auto& descVal = *reinterpret_cast<const AsicRegValType*>(descPos);
            blobVal             = descVal;  // Simple case of one reg
        }
        else
        {
            std::memcpy(blobPos, descPos, dataSize);  // WREG_BULK case
        }
    }
}

}  // namespace eager_mode

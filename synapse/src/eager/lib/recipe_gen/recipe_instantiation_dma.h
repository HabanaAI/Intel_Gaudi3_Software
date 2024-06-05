#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"

// std includes
#include <optional>

namespace eager_mode
{
struct TemplateOfEngine;
class DescGeneratorBase;
class MultiChunkArcJobWriter;
class RecipeHalBase;

// Assign the right values to execution blobs of actual recipe based on descriptors and recipe template.
class DmaInstantiation
{
public:
    DmaInstantiation(const TemplateOfEngine&            dmaTemplate,
                     const DescGeneratorBase&           descGenerator,
                     const RecipeHalBase&               recipeHalBase,
                     MultiChunkArcJobWriter&            arcJobWriter,
                     BlobsNrType                        nonExecBlobsNr,
                     size_t                             nodeDescNr,
                     const std::optional<BlobSizeType>& constExeBlobOffset,
                     const char*                        executionBlobsBuffer);

    void initialize(blob_t*     firstBlobs,
                    BlobsNrType dynBlobsIdx,
                    BlobsNrType patchBlobIdx,
                    const char* patchingBlobsBuffer,
                    bool        isNopDescNeeded);
    void instantiateExcBlobs(unsigned descIdx, blob_t* actualBlobs);
    void addExecBlobsToStaticEcbs(const blob_t* blobs, size_t descIdx);
    void finalize(const blob_t* blobs, bool isNopDescNeeded);

private:
    void addExecBlobsToStaticEcbs(const blob_t* blobs);
    void instantiateDynBlobs(const blob_t& actualBlob);
    void instantiateDynamicEcbs();

private:
    const TemplateOfEngine&            m_template;
    const DescGeneratorBase&           m_descGenerator;
    const RecipeHalBase&               m_recipeHal;
    MultiChunkArcJobWriter&            m_dmaArcJobWriter;
    const BlobsNrType                  m_nonExecBlobsNr;
    const BlobsNrType                  m_execBlobsNr;
    const size_t                       m_requiredWdCtxNr;
    const size_t                       m_nodeDescNr;
    const std::optional<BlobSizeType>& m_constExeBlobOffset;
    const char*                        m_executionBlobsBuffer;
};

}  // namespace eager_mode

#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_arc_job_writer.h"
#include "recipe_gen/recipe_defs.h"

// std includes
#include <optional>
#include <vector>

struct blob_t;
struct ecb_t;

namespace eager_mode
{
struct TemplateOfEngine;
class DescGeneratorBase;
class MmeDescGeneratorBase;
class MmeArcJobInstantiationInfo;
class RecipeHalBase;

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeInstantiation
///////////////////////////////////////////////////////////////////////////////////////////////////

// Assign the right values to execution blobs of actual recipe based on descriptors and recipe template.
// User can locate tensors at any location at any section.
// However, recipe template must locate each tensor at separated section at offset 0.
class MmeInstantiation
{
public:
    MmeInstantiation(const TemplateOfEngine&            mmeTemplate,
                     const DescGeneratorBase&           descGenerator,
                     const RecipeHalBase&               recipeHal,
                     const std::optional<BlobSizeType>& constExeBlobOffset,
                     bool                               isFirstNode);

    void instantiateDynBlobs(const blob_t& actualBlob);
    void instantiateExcBlobs(unsigned firstEngineDescIdx, blob_t* actualBlobs);
    void instantiateArcJobs(MultiChunkArcJobWriter& arcJobWriter);

private:
    unsigned getDescIdx(unsigned firstEngineDescIdx, BlobsNrType blobIdx) const;
    void     instantiateStaticEcbs(MultiChunkArcJobWriter& arcJobWriter);
    void     instantiateDynamicEcbs(MultiChunkArcJobWriter& arcJobWriter);

private:
    // Variables initialized at constructor
    const TemplateOfEngine&            m_template;
    const MmeDescGeneratorBase&        m_descGenerator;
    const RecipeHalBase&               m_recipeHal;
    const std::optional<BlobSizeType>& m_constExeBlobOffset;
    const bool                         m_isFirstNode;
    const MmeArcJobInstantiationInfo&  m_arcJobInfo;
    const size_t                       m_activationsNr;
    const size_t                       m_requiredWdCtxNr;
    const BlobsNrType                  m_enginesNr;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeArcJobInstantiationInfo
///////////////////////////////////////////////////////////////////////////////////////////////////

class MmeArcJobInstantiationInfo
{
public:
    EcbCommandSizeType getStaticReadStartPos(unsigned chunkId) const;
    EcbCommandSizeType getStaticReadEndPos(unsigned chunkId) const;
    void               onetimeInit(const ecb_t& staticEcb, const RecipeHalBase& recipeHal);

private:
    bool                            m_isInitialized = false;  // Validity flag of the data to be initialized onetime
    const RecipeHalBase*            m_recipeHal     = nullptr;
    std::vector<EcbCommandSizeType> m_staticReadStartPos;  // Inclusive pos to be read from (in bytes) at static ECB
    std::vector<EcbCommandSizeType> m_staticReadEndPos;    // Exclusive pos to be read up to (in bytes) at static ECB
};

// Singleton to provide static info on ARC job to be initialized one time at recipe creation

class MmeArcJobInstantiationInfoManager
{
public:
    static MmeArcJobInstantiationInfo& getInstance(ChipType chipType)
    {
        static MmeArcJobInstantiationInfoManager instance;
        return instance.m_arcJobInfo[static_cast<unsigned>(chipType)];
    }

private:
    static constexpr unsigned                       chipsNr = static_cast<unsigned>(ChipType::CHIPS_NR);
    std::array<MmeArcJobInstantiationInfo, chipsNr> m_arcJobInfo;
};

}  // namespace eager_mode

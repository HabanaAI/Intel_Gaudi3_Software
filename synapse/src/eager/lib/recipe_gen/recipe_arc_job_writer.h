#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/blob_to_desc_map_structs.h"
#include "recipe_gen/recipe_arc_job_ecb_writer.h"
#include "recipe_gen/recipe_arc_job_utils.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_hal_base.h"
#include "recipe_gen/recipe_templates_defs.h"
#include "utils/general_defs.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// relative to <3rd-parties>/
#include "llvm/small_vector.h"

// std includes
#include <array>
#include <map>
#include <optional>

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// class ArcJobWriterBase
///////////////////////////////////////////////////////////////////////////////////////////////////

// Representation of baseic ARC job to be written with dynamic ECB list writer
class ArcJobWriterBase
{
public:
    ArcJobWriterBase(const RecipeHalBase& recipeHal, const std::optional<BlobSizeType>& constExeBlobOffset);
    virtual ~ArcJobWriterBase() = default;

    bool         isInitialized() const { return m_isInitialized; }
    virtual void init(const arc_job_t& arcJob, unsigned engineUsageNr, unsigned chunksNr);
    virtual bool isCompleted() const;

    void                  setActiveWriter(BitPosInBlobType patchPos, BitPosInBlobType execPos, BitPosInBlobType dynPos);
    void                  tryToCreateTails();
    void                  tryToCopyArcJob(const TemplateOfEngine& templateOfEngine);
    EcbWriter&            getDynamicEcbWriter() { return m_dynamicEcbWriter; }
    const EcbWriter&      getDynamicEcbWriter() const { return m_dynamicEcbWriter; }
    const PositionInBlob& getPatchingBlobsPos() const { return m_patchPos; }
    const PositionInBlob& getExecutionBlobsPos() const { return m_execPos; }
    const PositionInBlob& getDynamicBlobsPos() const { return m_dynPos; }

protected:
    virtual void createTails();
    virtual void copyArcJob(const TemplateOfEngine& templateOfEngine);
    bool         isLastUsage() const { return m_curEngineUsage == m_engineUsageNr; }

protected:
    const RecipeHalBase&               m_recipeHal;
    const std::optional<BlobSizeType>& m_constExeBlobOffset;  // Offset relative to execution blob buffer

private:
    bool      m_isInitialized = false;
    EcbWriter m_dynamicEcbWriter;    // Writer to dynamic ECB
    unsigned  m_engineUsageNr  = 0;  // Number times op use the engine (MME multi activation contributes to one usage)
    unsigned  m_curEngineUsage = 0;  // 1-based serial number to reflect engine usages so far
    bool      m_isTailCreationTried = false;  // Was tryToCreateTails() invoked for current node?

    // Current positions in data blobs of recipe_t
    PositionInBlob m_patchPos;
    PositionInBlob m_execPos;
    PositionInBlob m_dynPos;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MultiChunkArcJobWriter
///////////////////////////////////////////////////////////////////////////////////////////////////

// In addition to base interface, this class represents multi-chunk static ECB list.
class MultiChunkArcJobWriter final : public ArcJobWriterBase
{
public:
    MultiChunkArcJobWriter(const RecipeHalBase& recipeHal, const std::optional<BlobSizeType>& constExeBlobOffset)
    : ArcJobWriterBase(recipeHal, constExeBlobOffset)
    {
    }
    EcbCommandSizeType getChunksNr() const { return m_staticEcbWriters.size(); }
    virtual void       init(const arc_job_t& arcJob, unsigned engineUsageNr, unsigned chunksNr) override;
    virtual bool       isCompleted() const override;
    void               collectCommandsRanges(std::map<size_t, size_t>& patchingBlobBufRanges,
                                             std::map<size_t, size_t>& execBlobBufRanges,
                                             std::map<size_t, size_t>& dynamicBlobBufRanges) const;

    EcbWriter& getStaticEcbWriter(EcbCommandSizeType chunkIdx);
    void       addStaticBlob(const blob_t& blob, unsigned offsetInBlock);

private:
    virtual void createTails() override;
    virtual void copyArcJob(const TemplateOfEngine& templateOfEngine) override;

private:
    ecb_t                        m_staticEcb;  // Original static ECB
    static constexpr BlobsNrType maxChunksNr = std::max(RecipeHalBase::maxMmeEnginesNr, RecipeHalBase::maxDmaEnginesNr);
    llvm_vecsmall::SmallVector<EcbWriter, maxChunksNr> m_staticEcbWriters;  // One writer for each chunk at static ECB
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class ArcJobsWriter
///////////////////////////////////////////////////////////////////////////////////////////////////

// Manage all ARC jobs' writers and track incremental writes to all ECB lists
class ArcJobsWriter
{
public:
    explicit ArcJobsWriter(const RecipeHalBase& recipeHal, const std::optional<BlobSizeType>& constExeBlobOffset);
    void init(const arc_job_t* arcJobs, ArcJobsNrType arcJobsNr, const AllStatisticsType& stats);
    bool isCompleted() const;
    bool isFullAllocationUtilization(size_t patchingBlobBufSize, size_t execBlobBufSize, size_t dynamicBlobBufSize) const;

    MultiChunkArcJobWriter& setActiveWriter(Recipe::EngineType engine,
                                            BitPosInBlobType   patchPos,
                                            BitPosInBlobType   execPos,
                                            BitPosInBlobType   dynPos);
    MultiChunkArcJobWriter& getActiveWriter()
    {
        EAGER_ASSERT(m_activeWriter != nullptr, "Active ECB writer was not set");
        return *m_activeWriter;
    }

private:
    const RecipeHalBase&    m_recipeHal;
    MultiChunkArcJobWriter* m_activeWriter = nullptr;  // Pointer to ECB writer that is suitable to current node
    // Number of supported engines. Must comply with Recipe::EngineType
    static constexpr unsigned enginesNr = static_cast<unsigned>(EngineType::ENGINES_NR);
    // All supported ECB writers
    std::array<MultiChunkArcJobWriter, enginesNr> m_allWriters;  // Pointers to the writers above
};

}  // namespace eager_mode

#include "recipe_arc_job_writer.h"

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/mini_command_packets.h"

// synapse api (relative to include/)
#include "internal/define_synapse_common.hpp"

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// class ArcJobWriterBase
///////////////////////////////////////////////////////////////////////////////////////////////////

ArcJobWriterBase::ArcJobWriterBase(const RecipeHalBase&               recipeHal,
                                   const std::optional<BlobSizeType>& constExeBlobOffset)
: m_recipeHal(recipeHal), m_constExeBlobOffset(constExeBlobOffset), m_dynamicEcbWriter(recipeHal, EcbType::DYNAMIC)
{
}

// Attach actual dynamic ECB to the writer and determine total engine usage
void ArcJobWriterBase::init(const arc_job_t& arcJob, unsigned engineUsageNr, unsigned chunksNr)
{
    EAGER_ASSERT(!m_isInitialized, "ARC job writer is already initialized");
    m_isInitialized = true;

    EAGER_ASSERT(engineUsageNr >= 1, "Invalid usage for an engine");
    m_engineUsageNr = engineUsageNr;
    m_dynamicEcbWriter.init(arcJob.dynamic_ecb);
}

// Tail is the last thing to be created
bool ArcJobWriterBase::isCompleted() const
{
    return m_dynamicEcbWriter.isCompleted();
}

// Reset blob positions and other FSM flags. This should be done once starting new node.
void ArcJobWriterBase::setActiveWriter(BitPosInBlobType patchPos, BitPosInBlobType execPos, BitPosInBlobType dynPos)
{
    EAGER_ASSERT(m_curEngineUsage < m_engineUsageNr, "Invalid usage for an engine");
    ++m_curEngineUsage;
    m_isTailCreationTried = false;

    m_patchPos = patchPos;
    m_execPos  = execPos;
    m_dynPos   = dynPos;
}

// Either tail or copy will occur
void ArcJobWriterBase::tryToCreateTails()
{
    EAGER_ASSERT(!m_isTailCreationTried, "Wrong flow");
    m_isTailCreationTried = true;  // It's an indication we done handling current op
    if (isLastUsage())
    {
        createTails();
    }
}

// Either tail or copy will occur
void ArcJobWriterBase::tryToCopyArcJob(const TemplateOfEngine& templateOfEngine)
{
    if (!m_isTailCreationTried)
    {
        copyArcJob(templateOfEngine);
        if (isLastUsage())
        {
            m_isTailCreationTried = true;
            createTails();
        }
    }
}

void ArcJobWriterBase::createTails()
{
    EAGER_ASSERT(m_curEngineUsage == m_engineUsageNr, "Invalid engine usage number");
    m_dynamicEcbWriter.createTail();
}

// Copy ECB from template to actual. Used for single activation nodes
void ArcJobWriterBase::copyArcJob(const TemplateOfEngine& templateOfEngine)
{
    EAGER_ASSERT(templateOfEngine.recipe.arc_jobs_nr == 1, "Unsupported template");
    const arc_job_t& arcJob = templateOfEngine.recipe.arc_jobs[0];
    EAGER_ASSERT(arcJob.dynamic_ecb.cmds_eng_offset == 0, "Invalid dynamic ECB");
    if (m_dynamicEcbWriter.copy(arcJob.dynamic_ecb, templateOfEngine.ecbsNetSize.dynamicSz, m_dynPos.isPosChanged()))
    {
        m_dynamicEcbWriter.postMemcpyUpdate(/*patch*/ 0,
                                            /*exec*/ 0,
                                            m_dynPos,
                                            /*templateConstExeBlobOffset*/ -1,
                                            /*actualConstExeBlobOffset*/ {});
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MultiChunkArcJobWriter
///////////////////////////////////////////////////////////////////////////////////////////////////

void MultiChunkArcJobWriter::init(const arc_job_t& arcJob, unsigned engineUsageNr, unsigned chunksNr)
{
    ArcJobWriterBase::init(arcJob, engineUsageNr, chunksNr);
    m_staticEcb = arcJob.static_ecb;
    m_staticEcbWriters.reserve(chunksNr);
    for (EcbCommandSizeType chunkIdx = 0; chunkIdx < chunksNr; ++chunkIdx)
    {
        m_staticEcbWriters.push_back({m_recipeHal, EcbType::STATIC});
        m_staticEcbWriters[chunkIdx].init(m_staticEcb, chunkIdx);
    }
}

bool MultiChunkArcJobWriter::isCompleted() const
{
    const EcbCommandSizeType chunksNr = getChunksNr();
    for (EcbCommandSizeType chunkIdx = 0; chunkIdx < chunksNr; ++chunkIdx)
    {
        if (!m_staticEcbWriters[chunkIdx].isCompleted()) return false;
    }
    return ArcJobWriterBase::isCompleted();
}

// Fill blob consumption info for execution, patching and dynamic blobs buffers that are used by ECB commands
// This method is designed to be used in debug build.
void MultiChunkArcJobWriter::collectCommandsRanges(std::map<size_t, size_t>& patchingBlobBufRanges,
                                                   std::map<size_t, size_t>& execBlobBufRanges,
                                                   std::map<size_t, size_t>& dynamicBlobBufRanges) const
{
    const EcbCommandSizeType chunksNr = getChunksNr();
    for (EcbCommandSizeType chunkIdx = 0; chunkIdx < chunksNr; ++chunkIdx)
    {
        m_staticEcbWriters[chunkIdx].collectCommandsRanges(patchingBlobBufRanges, execBlobBufRanges);
    }
    getDynamicEcbWriter().collectCommandsRanges(dynamicBlobBufRanges);
}

void MultiChunkArcJobWriter::createTails()
{
    ArcJobWriterBase::createTails();
    for (auto& staticEcbWriter : m_staticEcbWriters)
    {
        staticEcbWriter.createTail();
    }
}

// Copy ECB from template to actual. Used for single activation nodes
void MultiChunkArcJobWriter::copyArcJob(const TemplateOfEngine& templateOfEngine)
{
    ArcJobWriterBase::copyArcJob(templateOfEngine);
    EAGER_ASSERT(templateOfEngine.recipe.arc_jobs_nr == 1, "Unsupported template");
    const arc_job_t&                  arcJob = templateOfEngine.recipe.arc_jobs[0];
    std::optional<EcbCommandSizeType> optChunkIdx;
    const PositionInBlob&             patchingBlobsPos  = getPatchingBlobsPos();
    const PositionInBlob&             executionBlobsPos = getExecutionBlobsPos();
    const bool               isPosChanged = patchingBlobsPos.isPosChanged() || executionBlobsPos.isPosChanged();
    const EcbCommandSizeType chunksNr     = getChunksNr();
    for (EcbCommandSizeType chunkIdx = 0; chunkIdx < chunksNr; ++chunkIdx)
    {
        if (chunksNr != 1)
        {
            optChunkIdx = chunkIdx;
        }
        if (m_staticEcbWriters[chunkIdx].copy(arcJob.static_ecb,
                                              templateOfEngine.ecbsNetSize.staticSz,
                                              isPosChanged,
                                              optChunkIdx))
        {
            m_staticEcbWriters[chunkIdx].postMemcpyUpdate(patchingBlobsPos,
                                                          executionBlobsPos,
                                                          /*dynamic*/ 0,
                                                          templateOfEngine.constExeBlobOffset,
                                                          m_constExeBlobOffset);
        }
    }
}

EcbWriter& MultiChunkArcJobWriter::getStaticEcbWriter(EcbCommandSizeType chunkIdx)
{
    EAGER_ASSERT(chunkIdx < getChunksNr(), "MME static chunk indices is out of bound");
    return m_staticEcbWriters[chunkIdx];
}

void MultiChunkArcJobWriter::addStaticBlob(const blob_t& blob, unsigned offsetInBlock)
{
    EngArcBufferAddrBase addrIndex = {};
    switch (blob.blob_type_all)
    {
        case blob_t::EBlobType::EXE:
            addrIndex = EngArcBufferAddrBase::EXECUTE_ADDR_BASE;
            break;
        case blob_t::EBlobType::PATCHING:
            addrIndex = EngArcBufferAddrBase::PATCHING_ADDR_BASE;
            break;
        default:
            EAGER_ASSERT(false, "Invalid static blob");
    }

    mini_ecb_packets::StaticDescV2 cmd = {};
    cmd.size                      = blob.size;
    cmd.addrIndex                 = addrIndex;
    cmd.addrOffset                = offsetInBlock;

    const EcbCommandSizeType chunksNr = getChunksNr();
    for (EcbCommandSizeType chunkIdx = 0; chunkIdx < chunksNr; ++chunkIdx)
    {
        cmd.cpuIndex = chunkIdx;
        m_staticEcbWriters[chunkIdx].writeStaticDescCommand(cmd);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// class ArcJobsWriter
///////////////////////////////////////////////////////////////////////////////////////////////////

ArcJobsWriter::ArcJobsWriter(const RecipeHalBase& recipeHal, const std::optional<BlobSizeType>& constExeBlobOffset)
: m_recipeHal(recipeHal),
  m_allWriters {MultiChunkArcJobWriter(recipeHal, constExeBlobOffset),
                MultiChunkArcJobWriter(recipeHal, constExeBlobOffset),
                MultiChunkArcJobWriter(recipeHal, constExeBlobOffset),
                MultiChunkArcJobWriter(recipeHal, constExeBlobOffset),
                MultiChunkArcJobWriter(recipeHal, constExeBlobOffset)}
{
}

// Init static and dynamic ECB lists trackers. Occurs one time
void ArcJobsWriter::init(const arc_job_t* arcJobs, ArcJobsNrType arcJobsNr, const AllStatisticsType& stats)
{
#define CHCEK_ENGINE_ID(ID) static_assert(static_cast<uint8_t>(EngineType::ID) == Recipe::EngineType::ID)
    CHCEK_ENGINE_ID(TPC);
    CHCEK_ENGINE_ID(MME);
    CHCEK_ENGINE_ID(DMA);
    CHCEK_ENGINE_ID(ROT);
    CHCEK_ENGINE_ID(CME);
#undef CHCEK_ENGINE_ID

    for (ArcJobsNrType i = 0; i < arcJobsNr; ++i)
    {
        const Recipe::EngineType engineId = arcJobs[i].logical_engine_id;
        unsigned                 chunksNr = 1;
        if (engineId == Recipe::EngineType::DMA || engineId == Recipe::EngineType::MME)
        {
            chunksNr = m_recipeHal.getMaxEngines(static_cast<EngineType>(engineId));
        }
        m_allWriters[engineId].init(arcJobs[i], stats[engineId].nodeNum, chunksNr);
    }
}

bool ArcJobsWriter::isCompleted() const
{
    bool res = false;  // When all writers are uninitialized, return false
    for (const MultiChunkArcJobWriter& writer : m_allWriters)
    {
        if (writer.isInitialized())
        {
            if (res = writer.isCompleted(); !res) break;
        }
    }
    return res;
}

// Check if execution, patching and dynamic blobs buffers were fully utilized by ECB commands
// This method is designed to be used in debug build.
bool ArcJobsWriter::isFullAllocationUtilization(size_t patchingBlobBufSize,
                                                size_t execBlobBufSize,
                                                size_t dynamicBlobBufSize) const
{
    struct BlobBufInfo
    {
        BlobBufInfo(size_t bufSize) : blobBufSize(bufSize) {}
        const size_t             blobBufSize;
        std::map<size_t, size_t> blobBufRanges;
        size_t                   sizeAcc = 0;  // Accumulation size of ranges so far
    };
    BlobBufInfo patching(patchingBlobBufSize);
    BlobBufInfo execution(execBlobBufSize);
    BlobBufInfo dynamic(dynamicBlobBufSize);

    // Collect ranges
    for (const MultiChunkArcJobWriter& writer : m_allWriters)
    {
        if (writer.isInitialized())
        {
            writer.collectCommandsRanges(patching.blobBufRanges, execution.blobBufRanges, dynamic.blobBufRanges);
        }
    }

    // Chek full utilization
    for (BlobBufInfo* bufInfo : {&patching, &execution, &dynamic})
    {
        EAGER_ASSERT(!bufInfo->blobBufRanges.empty(), "Invalid blob buffer");
        for (const auto& range : bufInfo->blobBufRanges)
        {
            // Ranges must be adjecent and zero-based
            if (range.first != bufInfo->sizeAcc) return false;
            bufInfo->sizeAcc += range.second;
        }
        // Check full coverage over blob buffer
        if (bufInfo->sizeAcc != bufInfo->blobBufSize) return false;
    }

    return true;
}

// 'patchPos', 'execPos' and 'dynPos' are current positions in data blobs of recipe_t
MultiChunkArcJobWriter& ArcJobsWriter::setActiveWriter(Recipe::EngineType engine,
                                                       BitPosInBlobType   patchPos,
                                                       BitPosInBlobType   execPos,
                                                       BitPosInBlobType   dynPos)
{
    MultiChunkArcJobWriter& writer = m_allWriters[engine];
    m_activeWriter                 = &writer;
    EAGER_ASSERT(writer.isInitialized(), "Invalid engine ID, it doesn't participate in current workload");
    writer.setActiveWriter(patchPos, execPos, dynPos);
    return writer;
}

}  // namespace eager_mode
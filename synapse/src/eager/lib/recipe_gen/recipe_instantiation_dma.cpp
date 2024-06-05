#include "recipe_instantiation_dma.h"

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "recipe_gen/blob_to_desc_map_structs.h"
#include "recipe_gen/blob_to_desc_map.h"
#include "recipe_gen/recipe_arc_job_writer.h"
#include "recipe_gen/recipe_hal_base.h"
#include "utils/general_defs.h"
#include "utils/numeric_utils.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <cstring>

namespace eager_mode
{
DmaInstantiation::DmaInstantiation(const TemplateOfEngine&            dmaTemplate,
                                   const DescGeneratorBase&           descGenerator,
                                   const RecipeHalBase&               recipeHalBase,
                                   MultiChunkArcJobWriter&            arcJobWriter,
                                   BlobsNrType                        nonExecBlobsNr,
                                   size_t                             nodeDescNr,
                                   const std::optional<BlobSizeType>& constExeBlobOffset,
                                   const char*                        executionBlobsBuffer)
: m_template(dmaTemplate),
  m_descGenerator(descGenerator),
  m_recipeHal(recipeHalBase),
  m_dmaArcJobWriter(arcJobWriter),
  m_nonExecBlobsNr(nonExecBlobsNr),
  m_execBlobsNr(dmaTemplate.recipe.blobs_nr - nonExecBlobsNr),
  m_requiredWdCtxNr(descGenerator.getRequiredWdCtxNr()),
  m_nodeDescNr(nodeDescNr),
  m_constExeBlobOffset(constExeBlobOffset),
  m_executionBlobsBuffer(executionBlobsBuffer)
{
    EAGER_ASSERT(m_nonExecBlobsNr != 0, "Invalid number of non-execution blobs in DMA template");
    EAGER_ASSERT((m_requiredWdCtxNr >= 1) && (m_requiredWdCtxNr <= 3), "Invalid required FW context number");
}

void DmaInstantiation::initialize(blob_t*     firstBlobs,
                                  BlobsNrType dynBlobsIdx,
                                  BlobsNrType patchBlobIdx,
                                  const char* patchingBlobsBuffer,
                                  bool        isNopDescNeeded)
{
    instantiateDynBlobs(firstBlobs[dynBlobsIdx]);

    // Add patching blob to static ECB chunks
    const blob_t& patchingBlob = firstBlobs[patchBlobIdx];
    EAGER_ASSERT(static_cast<const char*>(patchingBlob.data) >= patchingBlobsBuffer, "Invalid patching blob pointer");
    const size_t offsetInBlock = static_cast<const char*>(patchingBlob.data) - patchingBlobsBuffer;
    m_dmaArcJobWriter.addStaticBlob(patchingBlob, offsetInBlock);

    if (!isNopDescNeeded)
    {
        instantiateExcBlobs(0, firstBlobs);
        addExecBlobsToStaticEcbs(firstBlobs + m_nonExecBlobsNr, /*descIdx*/ 0);
    }
}

// Copy work distribution context to dynamic blob
void DmaInstantiation::instantiateDynBlobs(const blob_t& actualBlob)
{
    EAGER_ASSERT(actualBlob.blob_type.dynamic_exe, "Invalid dynamic blob");
    auto*                actualCtxt = reinterpret_cast<Byte*>(actualBlob.data);
    const StructSizeType wdCtxtSize = m_recipeHal.getWorkDistributionContextSize(EngineType::DMA);

    // Handle first activation
    std::memcpy(actualCtxt, m_descGenerator.getWorkDistributionContextRaw(0), wdCtxtSize);
    // Handle middle activations and NOPs
    if (m_requiredWdCtxNr >= 2)
    {
        EAGER_ASSERT(m_dmaArcJobWriter.getChunksNr() < m_nodeDescNr, "Wrong flow");
        actualCtxt += wdCtxtSize;
        std::memcpy(actualCtxt,
                    m_descGenerator.getWorkDistributionContextRaw(m_dmaArcJobWriter.getChunksNr()),
                    wdCtxtSize);
        // Handle last activation
        if (m_requiredWdCtxNr == 3)
        {
            EAGER_ASSERT(m_nodeDescNr >= 2, "Invalid DMA descriptors");
            actualCtxt += wdCtxtSize;
            std::memcpy(actualCtxt, m_descGenerator.getWorkDistributionContextRaw(m_nodeDescNr - 1), wdCtxtSize);
        }
    }
}

// Main method to fill all field in recipe that are affected by adding single DMA activation
void DmaInstantiation::instantiateExcBlobs(unsigned descIdx, blob_t* actualBlobs)
{
    EAGER_ASSERT_PTR(actualBlobs);
    // Initialize new blobs based on recipe template and instantiate them based on descriptor
    for (const auto& mapElm : m_template.blob2DescMaps.execDescMap)
    {
        EAGER_ASSERT(mapElm.blobIdx < m_template.recipe.blobs_nr, "Blob index is out of bound");
        auto& blob = actualBlobs[mapElm.blobIdx];
        EAGER_ASSERT(blob.blob_type_all == blob_t::EBlobType::EXE, "Expected execution blobs only");
        const BlobSizeType dataSize = mapElm.regsNr * sizeOfAsicRegVal;
        EAGER_ASSERT((mapElm.blobPos + dataSize) <= blob.size, "Blob pos is out of bound");
        EAGER_ASSERT((mapElm.structPos + dataSize) <= m_recipeHal.getDescSize(EngineType::DMA),
                     "Descriptor pos is out of bound");
        // Raw data
        const auto& descRaw = m_descGenerator.getDescRaw(descIdx);
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

void DmaInstantiation::addExecBlobsToStaticEcbs(const blob_t* blobs, size_t descIdx)
{
    EAGER_ASSERT_PTR(blobs);
    EAGER_ASSERT(descIdx < m_nodeDescNr, "Invalid descriptor index");

    const EcbCommandSizeType  chunkIdx = descIdx % m_dmaArcJobWriter.getChunksNr();
    mini_ecb_packets::StaticDescV2 cmd      = {};
    cmd.cpuIndex                       = chunkIdx;
    cmd.addrIndex                      = EngArcBufferAddrBase::EXECUTE_ADDR_BASE;

    EAGER_ASSERT((m_recipeHal.isConstExeBlobSupported() == false) || (m_template.constExeBlobIndex >= m_nonExecBlobsNr),
                 "Invalid const blob index");
    const BlobsNrType constExeBlobIndex =
        m_recipeHal.isConstExeBlobSupported() ? (m_template.constExeBlobIndex - m_nonExecBlobsNr) : -1;
    // We build the ECB commands from scratch, baseRegLatencyWaBlob was relocated last, thus we start from the end
    EAGER_ASSERT((constExeBlobIndex == -1) || (constExeBlobIndex == m_execBlobsNr - 1),
                 "Invalid assumbtion of baseRegLatencyWaBlob to be last in execution blobs list");

    EcbWriter& writer = m_dmaArcJobWriter.getStaticEcbWriter(chunkIdx);
    for (int i = m_execBlobsNr - 1; i >= 0; --i)
    {
        if (i == constExeBlobIndex)
        {
            cmd.size       = m_template.recipe.blobs[m_template.constExeBlobIndex].size;
            cmd.addrOffset = *m_constExeBlobOffset;
        }
        else
        {
            EAGER_ASSERT(blobs[i].blob_type_all == blob_t::EBlobType::EXE, "Expected execution blobs only");
            cmd.size       = blobs[i].size;
            cmd.addrOffset = static_cast<const char*>(blobs[i].data) - m_executionBlobsBuffer;
        }
        if (i == 0)
        {
            cmd.yield = true;
        }
        writer.writeStaticDescCommand(cmd);
    }
}

void DmaInstantiation::addExecBlobsToStaticEcbs(const blob_t* blobs)
{
    const size_t startIdx = m_nodeDescNr % m_dmaArcJobWriter.getChunksNr();
    EAGER_ASSERT(startIdx != 0, "Wrong flow");
    EAGER_ASSERT_PTR(blobs);

    mini_ecb_packets::StaticDescV2 cmd = {};
    cmd.addrIndex                 = EngArcBufferAddrBase::EXECUTE_ADDR_BASE;

    EAGER_ASSERT((m_recipeHal.isConstExeBlobSupported() == false) || (m_template.constExeBlobIndex >= m_nonExecBlobsNr),
                 "Invalid const blob index");
    const BlobsNrType constExeBlobIndex =
        m_recipeHal.isConstExeBlobSupported() ? (m_template.constExeBlobIndex - m_nonExecBlobsNr) : -1;
    // We build the ECB commands from scratch, baseRegLatencyWaBlob was relocated last, thus we start from the end
    EAGER_ASSERT((constExeBlobIndex == -1) || (constExeBlobIndex == m_execBlobsNr - 1),
                 "Invalid assumbtion of baseRegLatencyWaBlob to be last in execution blobs list");

    for (EcbCommandSizeType chunkIdx = startIdx; chunkIdx < m_dmaArcJobWriter.getChunksNr(); ++chunkIdx)
    {
        EcbWriter& writer = m_dmaArcJobWriter.getStaticEcbWriter(chunkIdx);
        for (int i = m_execBlobsNr - 1; i >= 0; --i)
        {
            if (i == constExeBlobIndex)
            {
                cmd.size       = m_template.recipe.blobs[m_template.constExeBlobIndex].size;
                cmd.addrOffset = *m_constExeBlobOffset;
            }
            else
            {
                EAGER_ASSERT(blobs[i].blob_type_all == blob_t::EBlobType::EXE, "Expected execution blobs only");
                cmd.size       = blobs[i].size;
                cmd.addrOffset = static_cast<const char*>(blobs[i].data) - m_executionBlobsBuffer;
            }
            if (i == 0)
            {
                cmd.yield = true;
            }
            cmd.cpuIndex = chunkIdx;
            writer.writeStaticDescCommand(cmd);
        }
    }
}

void DmaInstantiation::instantiateDynamicEcbs()
{
    // Determine initial values of the DMA commands
    // Why yield is 1 not 0? Here is an answer from Didi:
    //    "The rational is to activate the DMA to load a slot in the DCCM and then go and queue
    //    another static blob in the static fetcher queue, then yield back and load another
    //    DCCM slot and go back to static and load another static blob. After 7 cycles we can
    //    start fire rapidly all executions since all the queues are full."
    EAGER_ASSERT(m_dmaArcJobWriter.getDynamicBlobsPos() <= std::numeric_limits<uint32_t>::max(),
                 "Unsupported blob pos");

    const StructSizeType wdCtxtSize = m_recipeHal.getWorkDistributionContextSize(EngineType::DMA);

    mini_ecb_packets::SchedDma dmaCmd = {};
    dmaCmd.addrIndex                  = DYNAMIC_ADDR_BASE;
    dmaCmd.yield                      = true;
    dmaCmd.size                       = wdCtxtSize;
    dmaCmd.addrOffset                 = static_cast<uint32_t>(m_dmaArcJobWriter.getDynamicBlobsPos());

    // Determine initial values of the execution commands
    mini_ecb_packets::Fence wdCmd = {};
    wdCmd.yield                   = true;
    wdCmd.dmaCompletion           = 1;

    // alignment NOP
    mini_ecb_packets::Nop nopCmd = {};

    const unsigned activationsNr   = divRoundUp(m_nodeDescNr, m_dmaArcJobWriter.getChunksNr());
    const unsigned lastActivation  = activationsNr - 1;
    unsigned       curActivation   = 0;
    bool           is2ndCtxHandled = false;

    do
    {
        // Create DMA commands, take into account finate number of entries in the cyclic cache
        unsigned dmaCmdCnt               = 0;
        unsigned maxSupportedDmaCommands = m_recipeHal.getMaxSupportedWorkDistributionContextCount();
        while ((dmaCmdCnt < maxSupportedDmaCommands) && (curActivation < activationsNr))
        {
            dmaCmd.gcCtxtOffset = dmaCmdCnt * wdCtxtSize;  // dst
            if (curActivation != 0)
            {
                // Handle 2nd ctx
                if (!is2ndCtxHandled)
                {
                    EAGER_ASSERT(m_requiredWdCtxNr >= 2, "Invalid FW context requested number");
                    is2ndCtxHandled = true;
                    dmaCmd.addrOffset += wdCtxtSize;  // src
                }
                // Handle 3rd ctx
                else if (curActivation == lastActivation)
                {
                    EAGER_ASSERT(m_requiredWdCtxNr == 3, "Invalid FW context requested number");
                    dmaCmd.addrOffset += wdCtxtSize;  // src
                }
            }
            m_dmaArcJobWriter.getDynamicEcbWriter().writeSchedDmaCommand(dmaCmd);
            ++curActivation;
            ++dmaCmdCnt;
        }

        // Create exec commands
        for (unsigned i = 0; i < dmaCmdCnt; ++i)
        {
            wdCmd.wdCtxtId = i;  // slot id
            m_dmaArcJobWriter.getDynamicEcbWriter().writeFenceCommand(wdCmd);
        }
    } while (curActivation < activationsNr);

    // Add NOP to fix the misalignment the FENCE command above caused
    EAGER_ASSERT(m_recipeHal.isFenceCmdExistsInDynamicEcb(), "Unsupported case, turn the assert into 'if'");
    if ((activationsNr & 1) == 1)
    {
        m_dmaArcJobWriter.getDynamicEcbWriter().writeNopCommand(nopCmd);
    }

#ifndef NDEBUG
    const BitPosInBlobType actualNewCtxSize   = dmaCmd.addrOffset - m_dmaArcJobWriter.getDynamicBlobsPos();
    const BitPosInBlobType expectedNewCtxSize = (m_requiredWdCtxNr - 1) * wdCtxtSize;
    EAGER_ASSERT(actualNewCtxSize == expectedNewCtxSize, "Invalid FW context allocation");
#endif
}

void DmaInstantiation::finalize(const blob_t* blobs, bool isNopDescNeeded)
{
    if (isNopDescNeeded)
    {
        addExecBlobsToStaticEcbs(blobs);
    }

    // Put NOPs on idle engines
    instantiateDynamicEcbs();
    m_dmaArcJobWriter.tryToCreateTails();
}

}  // namespace eager_mode

#include "recipe_instantiation_mme.h"

// eager includes (relative to src/eager/lib/)
#include "desc_gen/mme_desc_base.h"
#include "recipe_gen/blob_to_desc_map_structs.h"
#include "recipe_gen/blob_to_desc_map.h"
#include "recipe_gen/mini_command_packets.h"
#include "recipe_gen/recipe_arc_job_utils.h"
#include "recipe_gen/recipe_hal_base.h"
#include "utils/general_defs.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <cstring>

namespace eager_mode
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeInstantiation
///////////////////////////////////////////////////////////////////////////////////////////////////

MmeInstantiation::MmeInstantiation(const TemplateOfEngine&            mmeTemplate,
                                   const DescGeneratorBase&           descGenerator,
                                   const RecipeHalBase&               recipeHalBase,
                                   const std::optional<BlobSizeType>& constExeBlobOffset,
                                   bool                               isFirstNode)
: m_template(mmeTemplate),
  m_descGenerator(static_cast<const MmeDescGeneratorBase&>(descGenerator)),
  m_recipeHal(recipeHalBase),
  m_constExeBlobOffset(constExeBlobOffset),
  m_isFirstNode(isFirstNode),
  m_arcJobInfo(MmeArcJobInstantiationInfoManager::getInstance(m_recipeHal.getChipType())),
  m_activationsNr(descGenerator.getActivationNr()),
  m_requiredWdCtxNr(descGenerator.getRequiredWdCtxNr()),
  m_enginesNr(m_recipeHal.getMaxEngines(EngineType::MME))
{
    EAGER_ASSERT(m_activationsNr != 0, "Invalid number of MME activations");
    EAGER_ASSERT((m_requiredWdCtxNr >= 1) && (m_requiredWdCtxNr <= 3), "Invalid required FW context number");
    EAGER_ASSERT(m_descGenerator.getActivationNr() != 0, "Invalid MME activations");
    EAGER_ASSERT(m_descGenerator.getDescNr() == (m_enginesNr * m_descGenerator.getActivationNr()),
                 "Invalid MME descriptors");
}

// Copy work distribution context to dynamic blob
void MmeInstantiation::instantiateDynBlobs(const blob_t& actualBlob)
{
    EAGER_ASSERT(actualBlob.blob_type.dynamic_exe, "Invalid dynamic blob");
    auto*                actualCtxt = reinterpret_cast<Byte*>(actualBlob.data);
    auto                 descNr     = m_descGenerator.getDescNr();
    const StructSizeType wdCtxtSize = m_recipeHal.getWorkDistributionContextSize(EngineType::MME);

    // Handle first activation
    std::memcpy(actualCtxt, m_descGenerator.getWorkDistributionContextRaw(0), wdCtxtSize);

    if (m_activationsNr >= 2)
    {
        EAGER_ASSERT(descNr > m_enginesNr, "Invalid MME descriptors");
        // Handle middle activations
        actualCtxt += wdCtxtSize;
        std::memcpy(actualCtxt, m_descGenerator.getWorkDistributionContextRaw(m_enginesNr), wdCtxtSize);

        // Handle last activation
        if (m_activationsNr >= 3)
        {
            EAGER_ASSERT(descNr == (m_activationsNr * m_enginesNr), "Invalid MME descriptors");
            actualCtxt += wdCtxtSize;
            std::memcpy(actualCtxt, m_descGenerator.getWorkDistributionContextRaw(descNr - 1), wdCtxtSize);
        }
    }
}

// Main method to fill all field in recipe that are affected by adding single MME activation
// descWrappers: Descriptors of single activation
// actualBlobs:  Blobs to be filled with actual data from descriptors
void MmeInstantiation::instantiateExcBlobs(unsigned firstEngineDescIdx, blob_t* actualBlobs)
{
    EAGER_ASSERT_PTR(actualBlobs);
    // Initialize new blobs based on recipe template and instantiate them based on descriptor
    for (const auto& mapElm : m_template.blob2DescMaps.execDescMap)
    {
        EAGER_ASSERT(mapElm.blobIdx < m_template.recipe.blobs_nr, "Blob index is out of bound");
        auto&              blob     = actualBlobs[mapElm.blobIdx];
        const BlobSizeType dataSize = mapElm.regsNr * sizeOfAsicRegVal;
        EAGER_ASSERT((mapElm.blobPos + dataSize) <= blob.size, "Blob pos is out of bound");
        EAGER_ASSERT((mapElm.structPos + dataSize) <= m_recipeHal.getDescSize(EngineType::MME),
                     "Descriptor pos is out of bound");
        // Copy descriptor data to blob
        const unsigned descIdx = getDescIdx(firstEngineDescIdx, mapElm.blobIdx);
        Byte*       blobPos = static_cast<Byte*>(blob.data) + mapElm.blobPos;
        m_descGenerator.copyDescToBlob(blobPos, descIdx, mapElm.structPos, dataSize);
    }
}

// Return MME descriptor that is associated with given blob id
unsigned MmeInstantiation::getDescIdx(unsigned firstEngineDescIdx, BlobsNrType blobIdx) const
{
    const Blob2EngineIdsMap& engineIdsMap = m_template.blob2DescMaps.blob2EngineIdsMap;
    EAGER_ASSERT(blobIdx < engineIdsMap.size(), "Invalid blob index");
    EAGER_ASSERT(engineIdsMap[blobIdx] < m_enginesNr, "Invalid engine id");
    return firstEngineDescIdx + engineIdsMap[blobIdx];
}

// Create actual static and dynamic ECBs
void MmeInstantiation::instantiateArcJobs(MultiChunkArcJobWriter& arcJobWriter)
{
    EAGER_ASSERT(m_activationsNr >= 2, "Wrong flow");
    instantiateStaticEcbs(arcJobWriter);
    instantiateDynamicEcbs(arcJobWriter);
    arcJobWriter.tryToCreateTails();
}

void MmeInstantiation::instantiateStaticEcbs(MultiChunkArcJobWriter& arcJobWriter)
{
    ecb_t&                   templateEcb    = m_template.recipe.arc_jobs[0].static_ecb;
    const EcbCommandSizeType patchActOffset = arcJobWriter.getPatchingBlobsPos();

    const BlobSizeType exeBlobsBufSizeOfFirstActivation =
        m_template.recipe.execution_blobs_buffer_size -
        ((m_constExeBlobOffset.has_value() && !m_isFirstNode)
             ? m_template.recipe.blobs[m_template.constExeBlobIndex].size
             : 0);
    EAGER_ASSERT((m_constExeBlobOffset.has_value() == false) ||
                     (m_template.constExeBlobIndex < m_template.recipe.blobs_nr),
                 "Invalid cobstant blobb index");
    BlobSizeType exeBlobsBufSizeOfRestActivations =
        m_template.recipe.execution_blobs_buffer_size -
        (m_constExeBlobOffset.has_value() ? m_template.recipe.blobs[m_template.constExeBlobIndex].size : 0);

    for (unsigned chunkIdx = 0; chunkIdx < arcJobWriter.getChunksNr(); ++chunkIdx)
    {
        bool                     countConstBlob  = true;
        EcbCommandSizeType       execBlobsOffset = arcJobWriter.getExecutionBlobsPos();
        const EcbCommandSizeType readStartPos    = m_arcJobInfo.getStaticReadStartPos(chunkIdx);
        const EcbCommandSizeType readEndPos      = m_arcJobInfo.getStaticReadEndPos(chunkIdx);
        for (unsigned i = 0; i < m_activationsNr; ++i)
        {
            // Copy execution packets from template while modifying their indices
            EcbCommandSizeType commandSize = m_recipeHal.getEcbCommandSize(EngineArcCommandId::STATIC_DESC_V2);
            for (EcbCommandSizeType j = readStartPos; j < readEndPos; j += commandSize)
            {
                auto packet = m_recipeHal.getStaticDescCommand(reinterpret_cast<Byte*>(&templateEcb.cmds[j]));
                EAGER_ASSERT(m_recipeHal.getEcbCommandOpcode(reinterpret_cast<Byte*>(&templateEcb.cmds[j])) ==
                                 EngineArcCommandId::STATIC_DESC_V2,
                             "Invalid ECB reader");
                if (packet.addrIndex == PATCHING_ADDR_BASE)
                {
                    // Offset of patch blob is updated for 1st activation as all the others uses same blob
                    if (i == m_template.patchingBlobIndex)
                    {
                        packet.addrOffset += patchActOffset;
                        arcJobWriter.getStaticEcbWriter(chunkIdx).writeStaticDescCommand(packet);
                    }
                }
                else
                {
                    EAGER_ASSERT(packet.addrIndex == EXECUTE_ADDR_BASE, "invalid ECB content");
                    if ((packet.addrOffset == m_template.constExeBlobOffset) && m_constExeBlobOffset.has_value())
                    {
                        packet.addrOffset = *m_constExeBlobOffset;  // Const execution blob reuse
                    }
                    else
                    {
                        packet.addrOffset += execBlobsOffset;
                    }
                    arcJobWriter.getStaticEcbWriter(chunkIdx).writeStaticDescCommand(packet);
                }
            }
            execBlobsOffset += countConstBlob ? exeBlobsBufSizeOfFirstActivation : exeBlobsBufSizeOfRestActivations;
            countConstBlob = false;
        }
    }
}

void MmeInstantiation::instantiateDynamicEcbs(MultiChunkArcJobWriter& arcJobWriter)
{
    // Determine initial values of the DMA commands
    // Why yield is 1 not 0? Here is an answer from Didi:
    //    "The rational is to activate the DMA to load a slot in the DCCM and then go and queue
    //    another static blob in the static fetcher queue, then yield back and load another
    //    DCCM slot and go back to static and load another static blob. After 7 cycles we can
    //    start fire rapidly all executions since all the queues are full."
    EAGER_ASSERT(arcJobWriter.getDynamicBlobsPos() <= std::numeric_limits<uint32_t>::max(), "Unsupported blob pos");
    const StructSizeType wdCtxtSize = m_recipeHal.getWorkDistributionContextSize(EngineType::MME);

    mini_ecb_packets::SchedDma dmaCmd = {};
    dmaCmd.addrIndex                  = DYNAMIC_ADDR_BASE;
    dmaCmd.yield                      = true;
    dmaCmd.size                       = wdCtxtSize;
    dmaCmd.addrOffset                 = static_cast<uint32_t>(arcJobWriter.getDynamicBlobsPos());

    // Determine initial values of the execution commands
    mini_ecb_packets::Fence wdCmd = {};
    wdCmd.yield                   = true;
    wdCmd.dmaCompletion           = 1;

    // alignment NOP
    mini_ecb_packets::Nop nopCmd = {};

    const unsigned lastActivation  = m_activationsNr - 1;
    unsigned       curActivation   = 0;
    bool           is2ndCtxHandled = false;

    do
    {
        // Create DMA commands, take into account finite number of entries in the cyclic cache
        unsigned       dmaCmdCnt               = 0;
        const unsigned maxSupportedDmaCommands = m_recipeHal.getMaxSupportedWorkDistributionContextCount();
        while ((dmaCmdCnt < maxSupportedDmaCommands) && (curActivation < m_activationsNr))
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
            arcJobWriter.getDynamicEcbWriter().writeSchedDmaCommand(dmaCmd);
            ++curActivation;
            ++dmaCmdCnt;
        }

        // Create exec commands
        for (unsigned i = 0; i < dmaCmdCnt; ++i)
        {
            wdCmd.wdCtxtId = i;  // slot id
            arcJobWriter.getDynamicEcbWriter().writeFenceCommand(wdCmd);  // FENCE command
        }
    } while (curActivation < m_activationsNr);

    // Add NOP to fix the misalignment the FENCE command above caused
    EAGER_ASSERT(m_recipeHal.isFenceCmdExistsInDynamicEcb(), "Unsupported case, turn the assert into 'if'");
    if ((m_activationsNr & 1) == 1)
    {
        arcJobWriter.getDynamicEcbWriter().writeNopCommand(nopCmd);
    }

#ifndef NDEBUG
    const BitPosInBlobType actualNewCtxSize   = dmaCmd.addrOffset - arcJobWriter.getDynamicBlobsPos();
    const BitPosInBlobType expectedNewCtxSize = (m_requiredWdCtxNr - 1) * wdCtxtSize;
    EAGER_ASSERT(actualNewCtxSize == expectedNewCtxSize, "Invalid FW context allocation");
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeArcJobInstantiationInfo
///////////////////////////////////////////////////////////////////////////////////////////////////

EcbCommandSizeType MmeArcJobInstantiationInfo::getStaticReadStartPos(unsigned chunkIdx) const
{
    EAGER_ASSERT(m_isInitialized, "Data is not initialized yet");
    EAGER_ASSERT(chunkIdx < m_staticReadStartPos.size(), "Chunk index out of bound");
    return m_staticReadStartPos[chunkIdx];
}

EcbCommandSizeType MmeArcJobInstantiationInfo::getStaticReadEndPos(unsigned chunkIdx) const
{
    EAGER_ASSERT(m_isInitialized, "Data is not initialized yet");
    EAGER_ASSERT(chunkIdx < m_staticReadStartPos.size(), "Chunk index out of bound");
    return m_staticReadEndPos[chunkIdx];
}

// Init static data (by recipe template)
void MmeArcJobInstantiationInfo::onetimeInit(const ecb_t& staticEcb, const RecipeHalBase& recipeHal)
{
    // This logic should be executed onetime only per device
    if (m_isInitialized) return;

    m_isInitialized = true;
    m_recipeHal     = &recipeHal;

    // Number of chunks at static ECB - must be equal to number of MME engines
    const EcbCommandSizeType staticChunksNr = EcbCmdsListInfo::calcChunksNr(staticEcb);
    EAGER_ASSERT(staticChunksNr == m_recipeHal->getMaxEngines(EngineType::MME),
                 "Invalid number of MME static ECB chunks");
    m_staticReadStartPos.resize(staticChunksNr, m_recipeHal->getTemplateHeadSize(EcbType::STATIC));
    m_staticReadEndPos.resize(staticChunksNr);

    // Calc start and end position for all chunks at static ECB
    const uint32_t*    cmds             = reinterpret_cast<const uint32_t*>(staticEcb.cmds);
    EcbCommandSizeType staticDescV2Size = m_recipeHal->getEcbCommandSize(EngineArcCommandId::STATIC_DESC_V2);
    EAGER_ASSERT((staticDescV2Size % sizeof(uint32_t)) == 0, "eng_arc_cmd_static_desc_v2_t wrong size");
    for (EcbCommandSizeType chunkIdx = 0; chunkIdx < staticChunksNr; ++chunkIdx)
    {
        const EcbCommandSizeType startOffset = chunkIdx * staticEcb.cmds_eng_offset;
        const EcbCommandSizeType endOffset   = startOffset + staticEcb.cmds_eng_offset;

        m_staticReadStartPos[chunkIdx] += startOffset;
        m_staticReadEndPos[chunkIdx] = m_staticReadStartPos[chunkIdx];
        for (; m_staticReadEndPos[chunkIdx] < endOffset; m_staticReadEndPos[chunkIdx] += staticDescV2Size)
        {
            const auto               opcodeVal = cmds[m_staticReadEndPos[chunkIdx] / sizeof(uint32_t)] & 0x7;
            const EngineArcCommandId opcode    = m_recipeHal->getEcbCommandOpcode(opcodeVal);
            if (opcode == EngineArcCommandId::NOP) break;
            EAGER_ASSERT(opcode == EngineArcCommandId::STATIC_DESC_V2, "invalid ECB content");
        }
    }
}

}  // namespace eager_mode

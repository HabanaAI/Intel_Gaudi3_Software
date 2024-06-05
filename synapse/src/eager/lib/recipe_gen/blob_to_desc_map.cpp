#include "blob_to_desc_map.h"

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/mini_command_packets.h"
#include "recipe_gen/recipe_hal_base.h"

// synapse api (relative to include/)
#include "internal/define_synapse_common.hpp"

// std includes
#include <cstddef>
#include <map>

namespace eager_mode
{
Blob2DescMapCreatorBase::Blob2DescMapCreatorBase(Blob2DescMaps&       blob2DescMaps,
                                                 EngineType           engineType,
                                                 SecondaryEngineType  secondaryEngineType,
                                                 const RecipeHalBase& recipeHal)
: m_recipeHal(recipeHal),
  m_blob2EngineIdsMap(blob2DescMaps.blob2EngineIdsMap),
  m_execDescMap(blob2DescMaps.execDescMap, engineType, secondaryEngineType, recipeHal),
  m_baseRegOffsetsMap(blob2DescMaps.baseRegOffsetsMap, recipeHal),
  m_wrAddrMapPosMap(blob2DescMaps.wrAddrMapPosMap, engineType, secondaryEngineType, recipeHal)
{
}

// Public method to fill struct Blob2DescMaps based on template recipe
void Blob2DescMapCreatorBase::fillMaps(const recipe_t& recipe)
{
    EAGER_ASSERT(!m_areMapsFilled, "Filling map is expected to be done once");
    m_areMapsFilled = true;

    m_blob2EngineIdsMap.init(recipe.blobs_nr);
    fillBlob2EngineIdsMap(recipe);
    m_blob2EngineIdsMap.lock();
    fillBlob2DescMaps(recipe.blobs, recipe.blobs_nr);
}

// Create mapping between blobs and engine ids
void Blob2DescMapCreatorBase::fillBlob2EngineIdsMap(const recipe_t& recipe)
{
    EAGER_ASSERT(recipe.arc_jobs_nr == 1, "Only templates with single ARC job are supported");
    const ecb_t&  staticEcb = recipe.arc_jobs[0].static_ecb;
    const blob_t* blobs     = recipe.blobs;
    EAGER_ASSERT_PTR(blobs);

    const unsigned enginesNr = (staticEcb.cmds_eng_offset == 0) ? 1 : (staticEcb.cmds_size / staticEcb.cmds_eng_offset);
    EAGER_ASSERT(enginesNr >= 1, "invalid static ECB");
    BlobsNrType                     patchingBlobIdx = -1;
    std::map<uint32_t, BlobsNrType> blobOffsetToIdx;

    // Map dynamic blob to engine 0, patching blob to all engines
    {
        [[maybe_unused]] bool isDynamicBlobMapped = false;
        for (BlobsNrType i = 0; i < recipe.blobs_nr; ++i)
        {
            if (blobs[i].blob_type_all == blob_t::EBlobType::PATCHING)
            {
                EAGER_ASSERT(patchingBlobIdx == -1, "Only one patching blob is expected to be in the recipe template");
                patchingBlobIdx = i;
            }
            else if (blobs[i].blob_type_all == blob_t::EBlobType::DYNAMIC)
            {
                EAGER_ASSERT(!isDynamicBlobMapped, "Only one dynamic blob is expected to be in the recipe template");
                isDynamicBlobMapped = true;
                m_blob2EngineIdsMap.addNewMapping(i, 0);
            }
            else
            {
                EAGER_ASSERT(blobs[i].blob_type_all == blob_t::EBlobType::EXE, "Unsupported blob type");
                const uint32_t blobOffset =
                    (static_cast<uint64_t*>(blobs[i].data) - recipe.execution_blobs_buffer) * sizeof(uint64_t);
                blobOffsetToIdx[blobOffset] = i;
            }
        }
        EAGER_ASSERT(isDynamicBlobMapped && (patchingBlobIdx != -1), "Incomplete recipe templates");
    }

    // Map execution blobs
    {
        for (unsigned engine = 0; engine < enginesNr; ++engine)
        {
            Byte* curCmd =
                reinterpret_cast<Byte*>(staticEcb.cmds) + static_cast<size_t>(engine) * staticEcb.cmds_eng_offset;
            bool  lastCmd = false;
            while (!lastCmd)
            {
                EngineArcCommandId opcode  = m_recipeHal.getEcbCommandOpcode(curCmd);
                EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(opcode);

                switch (opcode)
                {
                    case EngineArcCommandId::STATIC_DESC_V2:
                    {
                        mini_ecb_packets::StaticDescV2 cmd = m_recipeHal.getStaticDescCommand(curCmd);
                        EAGER_ASSERT(cmd.cpuIndex == engine ||
                                         cmd.cpuIndex == mini_ecb_packets::StaticDescV2::CPU_ID_ALL,
                                     "Invalid static ECB");
                        if (cmd.addrIndex == EngArcBufferAddrBase::EXECUTE_ADDR_BASE)
                        {
                            auto blobIdx = blobOffsetToIdx.find(cmd.addrOffset);
                            EAGER_ASSERT(blobIdx != blobOffsetToIdx.end(), "Invalid static ECB");
                            m_blob2EngineIdsMap.addNewMapping(blobIdx->second, engine);
                        }
                        else
                        {
                            EAGER_ASSERT(cmd.addrIndex == EngArcBufferAddrBase::PATCHING_ADDR_BASE,
                                         "Expecting patching blob");
                            m_blob2EngineIdsMap.addNewMapping(patchingBlobIdx, engine);
                        }
                    }
                    break;

                    case EngineArcCommandId::NOP:
                    {
                        lastCmd = (m_recipeHal.getNopCommand(curCmd).switchCq == 1);
                    }
                    break;
                    case EngineArcCommandId::LIST_SIZE:
                        break;

                    default:
                        EAGER_ASSERT(false, "Unsupported command in static ECB");
                        break;
                }
                curCmd += cmdSize;
            }
        }
    }
}

// Fill blob-to-desc maps. It takes blobs array as an input.
// For each blob it collects the positions that require modification with values from the actual descriptor.
// Blob positions are relative to that array. Descriptor positions are relative to the software representation of
// struct. It deals with executable and patchable blobs. Work distribution blobs are copied without any modification.
void Blob2DescMapCreatorBase::fillBlob2DescMaps(const blob_t* blobs, BlobsNrType blobsNr)
{
    EAGER_ASSERT_PTR(blobs);
    initBlackList();

    // Fill either patchable of non-patchable blob by scanning all commands and add mapping depends on command type
    for (auto blobIdx = 0; blobIdx < blobsNr; ++blobIdx)
    {
        const blob_t& blob = blobs[blobIdx];
        if (blob.blob_type.dynamic_exe)
        {
            continue;  // No handling for work distribution blobs
        }
        BlobSizeType cmdOffset = 0;  // Offset in in bytes of current command relative to blob.data
        while (cmdOffset < blob.size)
        {
            cmdOffset = fillMapsForCommand(blobIdx, blob, cmdOffset);
        }
        EAGER_ASSERT(cmdOffset == blob.size, "Invalid blob");
    }
}

// Collect all registers that should be transferred from recipe template to actual recipe without any change
void Blob2DescMapCreatorBase::initBlackList()
{
    // Each engine has its own black list
    initEngineBlackList();

    // Add all Qman regs to black list
    m_asicRegsBlackList.addBlackRange(AsicRange(m_recipeHal.getQmanBlockStart(), m_recipeHal.getQmanBlockEnd()));
    // Exclude cache base regs
    m_asicRegsBlackList.addWhiteRange(m_baseRegOffsetsMap.getRange());

    // Prevent further insertions
    m_asicRegsBlackList.lock();
}

// Add new position mapping from blob to descriptor or one of cache base regs
void Blob2DescMapCreatorBase::addNewMapping(blob_t::EBlobType blobTypte,
                                            BlobsNrType       blobIdx,
                                            BlobSizeType      posInBlob,
                                            AsicRegType       reg,
                                            DataNrType        regsNr)
{
    [[maybe_unused]] bool isAdded = false;
    switch (blobTypte)
    {
        case blob_t::EXE:
            isAdded = m_execDescMap.addNewMapping(blobIdx, posInBlob, reg, regsNr);
            break;
        case blob_t::PATCHING:
            isAdded = m_baseRegOffsetsMap.addNewMapping(blobIdx, posInBlob, reg, regsNr);
            break;
        default:
            EAGER_ASSERT(0, "Unsupported blob");
    };
    EAGER_ASSERT(isAdded, "Unsupported command");
}

QmanCommandSizeType
Blob2DescMapCreatorBase::fillMapsForCommand(BlobsNrType blobIdx, const blob_t& blob, BlobSizeType cmdOffset)
{
    const Byte*           blobData = static_cast<const Byte*>(blob.data) + cmdOffset;
    const PacketId        opcode   = m_recipeHal.getQmanCommandOpcode(blobData);
    QmanCommandSizeType   cmdSize  = m_recipeHal.getQmanCommandSize(opcode);
    EAGER_ASSERT((cmdOffset + cmdSize) <= blob.size, "Invalid command, or trying to use reserved opcode");

    switch (opcode)
    {
        case PacketId::WREG_32:
        {
            mini_qman_packets::Wreg32 packet = m_recipeHal.getWreg32Command(blobData);
            const AsicRegType    reg    = packet.regOffset;
            if (!m_asicRegsBlackList.isFound(reg))
            {
                const BlobSizeType posInBlob = m_recipeHal.getWreg32RegPos(cmdOffset);
                addNewMapping(blob.blob_type_all, blobIdx, posInBlob, reg);
            }
        }
        break;

        case PacketId::WREG_BULK:
        {
            mini_qman_packets::WregBulk packet     = m_recipeHal.getWregBulkCommand(blobData);
            const DataNrType       asicRegsNr = packet.size64 * asicRegsPerEntry;
            EAGER_ASSERT(asicRegsNr != 0, "Invalid WRBLK command");
            cmdSize += asicRegsNr * sizeOfAsicRegVal;
            AsicRegType firstReg = 0;
            DataNrType  regsNr   = 0;
            for (DataNrType i = 0; i < asicRegsNr; ++i)
            {
                const DataNrType nextReg    = i + 1;
                bool             addMapping = (nextReg == asicRegsNr);
                if (m_asicRegsBlackList.isFound(packet.regOffset + i * sizeOfAsicRegVal))
                {
                    addMapping = (regsNr != 0);  // "true": memcpy current regs. "false": keep scanning
                    if (!addMapping)
                    {
                        firstReg = nextReg;  // firstReg should always point to reg not in blacklist
                    }
                }
                else
                {
                    ++regsNr;  // Keep accumulating regs (to memcpy them later)
                }
                if (addMapping)  // Commit accumulated regs to new mapping element
                {
                    const BlobSizeType posInBlob = m_recipeHal.getWrBulkRegPos(cmdOffset, firstReg);
                    const AsicRegType  regOffset = packet.regOffset + firstReg;
                    addNewMapping(blob.blob_type_all, blobIdx, posInBlob, regOffset, regsNr);
                    regsNr   = 0;
                    firstReg = nextReg * sizeOfAsicRegVal;
                }
            }
        }
        break;

        case PacketId::WREG_64_LONG:
        {
            mini_qman_packets::Wreg64Long packet = m_recipeHal.getWreg64LongCommand(blobData);
            EAGER_ASSERT((packet.rel == 1) && (packet.dwEnable == 0x3), "Unsupported WREG_64_LONG command");
            // No need to check the black list, those commands are for addresses to be patched and are always valid
            const BlobSizeType     offsetPos  = m_recipeHal.getWreg64LongOffsetPos(cmdOffset);
            const UnalignedPosType baseRegPos = m_recipeHal.getWreg64LongBasePos(cmdOffset);
            const PatchPointNrType tensorId   = calcTensorId(m_wrAddrMapPosMap.calcPosInDesc(packet.dregOffset));
            m_wrAddrMapPosMap.addNewMapping(blobIdx, offsetPos, baseRegPos, tensorId, false);
        }
        break;

        default:
            break;
    }

    const BlobSizeType newCmdOffset = cmdOffset + cmdSize;
    EAGER_ASSERT(newCmdOffset <= blob.size, "Invalid command");
    return newCmdOffset;
}

}  // namespace eager_mode
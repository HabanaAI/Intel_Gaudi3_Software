#include "recipe_arc_job_ecb_writer.h"

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/mini_command_packets.h"

// synapse api (relative to include/)
#include "internal/define_synapse_common.hpp"

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// class EcbWriter
///////////////////////////////////////////////////////////////////////////////////////////////////

EcbWriter::EcbWriter(const RecipeHalBase& recipeHal, EcbType ecbType)
: m_recipeHal(recipeHal),
  m_cmdAlignmentConstraint(recipeHal.getEcbCommandAlignmentConstraint(ecbType)),
  m_ecbType(ecbType)
{
}

// 'ecb'  is the original ECB of the actual recipe
void EcbWriter::init(const ecb_t& ecb, EcbCommandSizeType chunkIdx)
{
    EAGER_ASSERT(!m_ecb.isValid, "ECB writer is already initialized");
    m_ecb.isValid = true;

    // Validate the given ECB
    EAGER_ASSERT((ecb.cmds_size % m_cmdAlignmentConstraint) == 0, "Invalid ECB list size");
    EAGER_ASSERT((ecb.cmds_eng_offset % m_cmdAlignmentConstraint) == 0, "Invalid ECB list chunk size");
    EAGER_ASSERT((ecb.cmds_size != 0) && (ecb.cmds_size >= ecb.cmds_eng_offset), "Invalid ECB list sizes");

    EAGER_ASSERT(m_writerPos == 0, "No write to ECB is expected to happen");
    const size_t offset = static_cast<size_t>(chunkIdx) * ecb.cmds_eng_offset;
    m_ecb.cmds          = ecb.cmds + offset;
    m_ecb.size          = (ecb.cmds_eng_offset == 0) ? ecb.cmds_size : ecb.cmds_eng_offset;
}

// Copy ECB from template to actual. It should not be used for multi-chunk static ECBs.
// Return decision to update blob offsets.
bool EcbWriter::copy(const ecb_t&                             srcEcb,
                     EcbCommandSizeType                       srcEcbNetSize,
                     bool                                     posChanged,
                     const std::optional<EcbCommandSizeType>& chunkIdx)
{
    EAGER_ASSERT(m_ecb.isValid, "ECB writer is not initialized yet");
    EAGER_ASSERT(!m_cmdsToCopy.isValid, "Writer pos was not updated");
    EAGER_ASSERT(!m_isTailCreated, "Wrong flow");
    EAGER_ASSERT(srcEcbNetSize != 0, "Invalid ECB net size");

    // Source: bypass head and switching
    uint8_t* srcCmds = srcEcb.cmds + m_recipeHal.getTemplateHeadSize(m_ecbType) +
                       ((m_ecbType == EcbType::DYNAMIC) ? m_recipeHal.getSwitchingSize() : 0);

    if (chunkIdx.has_value())
    {
        EAGER_ASSERT(srcEcb.cmds_eng_offset != 0, "Wrong flow");
        EAGER_ASSERT(m_ecb.size >= srcEcb.cmds_eng_offset, "Incompatible ECBs");
        EAGER_ASSERT(srcEcbNetSize < srcEcb.cmds_eng_offset, "Invalid source ECB size");
        srcCmds += static_cast<size_t>(*chunkIdx * srcEcb.cmds_eng_offset);
    }
    else
    {
        EAGER_ASSERT(srcEcb.cmds_eng_offset == 0, "Wrong flow");
        EAGER_ASSERT(m_ecb.size >= srcEcb.cmds_size, "Incompatible ECBs");
        EAGER_ASSERT(srcEcbNetSize < srcEcb.cmds_size, "Invalid source ECB size");
    }

    const bool wasFirstWrite = m_isFirstWrite;
    // Update write position and do the actual copy
    if (m_isFirstWrite)
    {
        createHead();
    }

    if (!posChanged && wasFirstWrite)
    {
        EAGER_ASSERT((m_ecb.size - m_writerPos) > srcEcbNetSize, "Invalid ECB allocation");
        // Actual copy
        uint8_t* dstCmds = m_ecb.cmds + m_writerPos;
        m_writerPos += srcEcbNetSize;
        std::memcpy(dstCmds, srcCmds, srcEcbNetSize);
        // If it's the first write, there is no need to update blob offsets
        return false;
    }

    // Otherwise after memcpy ECB from template to actual recipe, we need to translate
    // blob offsets relative to the existing blobs. In this case defer updating write pos.
    m_cmdsToCopy.cmds    = srcCmds;
    m_cmdsToCopy.size    = srcEcbNetSize;
    m_cmdsToCopy.isValid = true;

    return true;
}

// Use to update offsets of patching and execution for static blobs and work distributions for dynamic blobs
void EcbWriter::postMemcpyUpdate(BlobSizeType                       patchBlobBase,
                                 BlobSizeType                       execBlobBase,
                                 BlobSizeType                       dynBlobBase,
                                 BlobSizeType                       templateConstExeBlobOffset,
                                 const std::optional<BlobSizeType>& actualConstExeBlobOffset)
{
    EAGER_ASSERT(m_cmdsToCopy.isValid && (m_cmdsToCopy.size != 0), "Nothing to update writer pos");
    EAGER_ASSERT((m_ecbType == EcbType::STATIC) || ((patchBlobBase == 0) && (execBlobBase == 0)), "Wrong flow");
    EAGER_ASSERT((m_ecbType == EcbType::DYNAMIC) || (dynBlobBase == 0), "Wrong flow");

    uint8_t* cmdsDst = m_ecb.cmds + m_writerPos;
    m_writerPos += m_cmdsToCopy.size;

    for (EcbCommandSizeType pos = 0; pos < m_cmdsToCopy.size;)
    {
        Byte*              curCmdSrc = reinterpret_cast<Byte*>(m_cmdsToCopy.cmds) + pos;
        Byte*              curCmdDst = reinterpret_cast<Byte*>(cmdsDst) + pos;
        EngineArcCommandId cmdType   = m_recipeHal.getEcbCommandOpcode(curCmdSrc);
        EcbCommandSizeType cmdSize   = m_recipeHal.getEcbCommandSize(cmdType);
        pos += cmdSize;

        // Commands handling order is determined by their frequently occurring
        switch (cmdType)
        {
            case EngineArcCommandId::STATIC_DESC_V2:
            {
                mini_ecb_packets::StaticDescV2 cmd = m_recipeHal.getStaticDescCommand(curCmdSrc);
                EAGER_ASSERT(m_ecbType == EcbType::STATIC, "Invalid ECB");
                if (cmd.addrIndex == EngArcBufferAddrBase::EXECUTE_ADDR_BASE)
                {
                    if (templateConstExeBlobOffset != cmd.addrOffset)
                    {
                        cmd.addrOffset += execBlobBase;
                    }
                    else
                    {
                        EAGER_ASSERT(actualConstExeBlobOffset.has_value(), "Invalid const execution blob offset");
                        cmd.addrOffset = *actualConstExeBlobOffset;
                    }
                }
                else
                {
                    EAGER_ASSERT(cmd.addrIndex == EngArcBufferAddrBase::PATCHING_ADDR_BASE, "Invalid blob type");
                    cmd.addrOffset += patchBlobBase;
                }
                m_recipeHal.writeStaticDescCommand(curCmdDst, cmd);
            }
            break;

            case EngineArcCommandId::SCHED_DMA:
            {
                mini_ecb_packets::SchedDma cmd = m_recipeHal.getSchedDmaCommand(curCmdSrc);
                EAGER_ASSERT(m_ecbType == EcbType::DYNAMIC, "Invalid ECB");
                EAGER_ASSERT(cmd.addrIndex == EngArcBufferAddrBase::DYNAMIC_ADDR_BASE, "Invalid blob type");
                cmd.addrOffset += dynBlobBase;
                m_recipeHal.writeSchedDmaCommand(curCmdDst, cmd);
            }
            break;

            case EngineArcCommandId::NOP:
            {
                std::memcpy(curCmdDst, curCmdSrc, cmdSize);
                const EcbCommandSizeType padding = m_recipeHal.getNopCommand(curCmdSrc).padding * nopPaddingUnits;
                pos += padding;
            }
            break;

            case EngineArcCommandId::LIST_SIZE:
            case EngineArcCommandId::WD_FENCE_AND_EXE:
            {
                // no need for any manipulation of the source ecb command
                // so we simply use a direct copy.
                std::memcpy(curCmdDst, curCmdSrc, cmdSize);
            }
            break;

            default:
                EAGER_ASSERT(false, "Unsupported ECB command");
                break;
        }
    }

    // Source commands were consumed, invalidate it
    m_cmdsToCopy.isValid = false;
}

// Checks and actions to be done before writing to ECB
void EcbWriter::preWriteToEcb(EcbCommandSizeType cmdSize)
{
    EAGER_ASSERT(m_ecb.isValid, "ECB writer is not initialized yet");
    // Command size must be multiples of NOP size in order to allow padding on misalignment
    EAGER_ASSERT((m_cmdAlignmentConstraint % m_recipeHal.getEcbCommandSize(EngineArcCommandId::NOP)) == 0,
                 "Unsupported command size");
    EAGER_ASSERT((cmdSize % m_recipeHal.getEcbCommandSize(EngineArcCommandId::NOP)) == 0, "Unsupported command size");
    EAGER_ASSERT(cmdSize <= m_recipeHal.getSupportedMaxSizeCmd(), "Unsupported command size");

    if (m_isFirstWrite)
    {
        createHead();
    }
    else
    {
        // NOP is used to fix command alignment, so no command is expected to have smaller memory footprint
        EAGER_ASSERT(cmdSize >= m_recipeHal.getEcbCommandSize(EngineArcCommandId::NOP),
                     "Fixing commands misalignment is unsupported for the given command");
        // All the ECB commands should be within alignment boundaries (don't cross boundaries)
        EAGER_ASSERT(
            (cmdSize == m_recipeHal.getEcbCommandSize(EngineArcCommandId::NOP)) ||
                ((m_writerPos / m_cmdAlignmentConstraint) == ((m_writerPos + cmdSize - 1) / m_cmdAlignmentConstraint)),
            "Command crosses alignment boundaries");
    }
}

// Common commands that must be at the beginning of ECB
void EcbWriter::createHead()
{
    EAGER_ASSERT(m_ecb.isValid, "ECB writer is not initialized yet");
    EAGER_ASSERT(!m_isTailCreated, "Wrong flow");
    EAGER_ASSERT(m_isFirstWrite, "Head must be written once");
    m_isFirstWrite = false;

    // Create list-size command
    mini_ecb_packets::ListSize sizeCmd = {};
    sizeCmd.size                       = m_ecb.size;
    sizeCmd.topologyStart              = true;
    writeListSizeCommand(sizeCmd, false);

    // Set switch bit to move ARC head to the other queue
    mini_ecb_packets::Nop nopPacket = {};
    if (m_ecbType == EcbType::DYNAMIC)
    {
        nopPacket.switchCq = true;
    }
    writeNopCommand(nopPacket, false);
}

// Common commands that must be at the end of ECB
void EcbWriter::createTail()
{
    EAGER_ASSERT(m_ecb.isValid, "ECB writer is not initialized yet");
    EAGER_ASSERT(!m_isFirstWrite, "Head was not written");
    EAGER_ASSERT(!m_isTailCreated, "Wrong flow");
    m_isTailCreated = true;

    mini_ecb_packets::Nop nopPacket = {};

    // Set switch bit to move ARC head to the other queue
    if (m_ecbType == EcbType::STATIC)
    {
        nopPacket.switchCq = true;
        writeNopCommand(nopPacket, false);
        nopPacket.switchCq = false;
    }

    // Create padding NOP if necessary
    if ((m_writerPos % m_cmdAlignmentConstraint) != 0)
    {
        const EcbCommandSizeType actualPos = m_writerPos + m_recipeHal.getEcbCommandSize(EngineArcCommandId::NOP);
        EAGER_ASSERT(m_ecb.size >= actualPos, "Invalid ECB size");
        const EcbCommandSizeType padding = m_ecb.size - actualPos;
        EAGER_ASSERT(padding < m_cmdAlignmentConstraint, "Invalid allocation of ECB block");
        nopPacket.padding = padding / nopPaddingUnits;
        writeNopCommand(nopPacket, false);
    }
}

void EcbWriter::writeNopCommand(const mini_ecb_packets::Nop& cmd, bool validate)
{
    EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(EngineArcCommandId::NOP);
    if (validate)
    {
        preWriteToEcb(cmdSize);
    }
    EAGER_ASSERT((m_writerPos + cmdSize) <= m_ecb.size, "Invalid allocation for ECB");
    Byte* writer = reinterpret_cast<Byte*>(m_ecb.cmds + m_writerPos);
    m_recipeHal.writeNopCommand(writer, cmd);
    m_writerPos += cmdSize;
}

void EcbWriter::writeStaticDescCommand(const mini_ecb_packets::StaticDescV2& cmd, bool validate)
{
    EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(EngineArcCommandId::STATIC_DESC_V2);
    if (validate)
    {
        preWriteToEcb(cmdSize);
    }
    EAGER_ASSERT((m_writerPos + cmdSize) <= m_ecb.size, "Invalid allocation for ECB");
    Byte* writer = reinterpret_cast<Byte*>(m_ecb.cmds + m_writerPos);
    m_recipeHal.writeStaticDescCommand(writer, cmd);
    m_writerPos += cmdSize;
}

void EcbWriter::writeListSizeCommand(const mini_ecb_packets::ListSize& cmd, bool validate)
{
    EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(EngineArcCommandId::LIST_SIZE);
    if (validate)
    {
        preWriteToEcb(cmdSize);
    }
    EAGER_ASSERT((m_writerPos + cmdSize) <= m_ecb.size, "Invalid allocation for ECB");
    Byte* writer = reinterpret_cast<Byte*>(m_ecb.cmds + m_writerPos);
    m_recipeHal.writeListSizeCommand(writer, cmd);
    m_writerPos += cmdSize;
}

void EcbWriter::writeSchedDmaCommand(const mini_ecb_packets::SchedDma& cmd, bool validate)
{
    EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(EngineArcCommandId::SCHED_DMA);
    if (validate)
    {
        preWriteToEcb(cmdSize);
    }
    EAGER_ASSERT((m_writerPos + cmdSize) <= m_ecb.size, "Invalid allocation for ECB");
    Byte* writer = reinterpret_cast<Byte*>(m_ecb.cmds + m_writerPos);
    m_recipeHal.writeSchedDmaCommand(writer, cmd);
    m_writerPos += cmdSize;
}

void EcbWriter::writeFenceCommand(const mini_ecb_packets::Fence& cmd, bool validate)
{
    EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(EngineArcCommandId::WD_FENCE_AND_EXE);
    if (validate)
    {
        preWriteToEcb(cmdSize);
    }
    EAGER_ASSERT((m_writerPos + cmdSize) <= m_ecb.size, "Invalid allocation for ECB");
    Byte* writer = reinterpret_cast<Byte*>(m_ecb.cmds + m_writerPos);
    m_recipeHal.writeFenceCommand(writer, cmd);
    m_writerPos += cmdSize;
}

// Fill blob consumption info for execution and patching blobs buffers that are used by ECB commands
// This method is designed to be used in debug build.
void EcbWriter::collectCommandsRanges(std::map<size_t, size_t>& patchingBlobBufRanges,
                                      std::map<size_t, size_t>& execBlobBufRanges) const
{
    for (EcbCommandSizeType pos = 0; pos < m_ecb.size;)
    {
        Byte*              curCmd  = reinterpret_cast<Byte*>(m_ecb.cmds) + pos;
        EngineArcCommandId cmdType = m_recipeHal.getEcbCommandOpcode(curCmd);
        EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(cmdType);
        pos += cmdSize;
        switch (cmdType)
        {
            case EngineArcCommandId::STATIC_DESC_V2:
            {
                mini_ecb_packets::StaticDescV2 cmd = m_recipeHal.getStaticDescCommand(curCmd);
                EAGER_ASSERT(m_ecbType == EcbType::STATIC, "Invalid ECB");
                if (cmd.addrIndex == EngArcBufferAddrBase::EXECUTE_ADDR_BASE)
                {
                    if (execBlobBufRanges.find(cmd.addrOffset) == execBlobBufRanges.end())
                    {
                        execBlobBufRanges[cmd.addrOffset] = cmd.size;
                    }
                    else
                    {
                        // It's possible to have shared NOP in DMA
                        EAGER_ASSERT(execBlobBufRanges[cmd.addrOffset] == cmd.size, "Invalid offset");
                    }
                }
                else
                {
                    EAGER_ASSERT(cmd.addrIndex == EngArcBufferAddrBase::PATCHING_ADDR_BASE, "Invalid blob type");
                    if (patchingBlobBufRanges.find(cmd.addrOffset) == patchingBlobBufRanges.end())
                    {
                        patchingBlobBufRanges[cmd.addrOffset] = cmd.size;
                    }
                    else
                    {
                        // Patching blobs can be shared
                        EAGER_ASSERT(patchingBlobBufRanges[cmd.addrOffset] == cmd.size, "Invalid offset");
                    }
                }
            }
            break;

            case EngineArcCommandId::NOP:
            {
                const EcbCommandSizeType padding = m_recipeHal.getNopCommand(curCmd).padding * nopPaddingUnits;
                pos += padding;
            }
            break;

            case EngineArcCommandId::LIST_SIZE:
                break;

            default:
                EAGER_ASSERT(false, "Unsupported ECB command");
                break;
        }
    }
}

// Fill blob consumption info for dynamic blobs buffers that are used by ECB commands
// This method is designed to be used in debug build.
void EcbWriter::collectCommandsRanges(std::map<size_t, size_t>& dynamicBlobBufRanges) const
{
    for (EcbCommandSizeType pos = 0; pos < m_ecb.size;)
    {
        Byte*              curCmd  = reinterpret_cast<Byte*>(m_ecb.cmds) + pos;
        EngineArcCommandId cmdType = m_recipeHal.getEcbCommandOpcode(curCmd);
        EcbCommandSizeType cmdSize = m_recipeHal.getEcbCommandSize(cmdType);
        pos += cmdSize;
        switch (cmdType)
        {
            case EngineArcCommandId::SCHED_DMA:
            {
                mini_ecb_packets::SchedDma cmd = m_recipeHal.getSchedDmaCommand(curCmd);
                EAGER_ASSERT(m_ecbType == EcbType::DYNAMIC, "Invalid ECB");
                EAGER_ASSERT(cmd.addrIndex == EngArcBufferAddrBase::DYNAMIC_ADDR_BASE, "Invalid blob type");
                if (dynamicBlobBufRanges.find(cmd.addrOffset) == dynamicBlobBufRanges.end())
                {
                    dynamicBlobBufRanges[cmd.addrOffset] = cmd.size;
                }
                else
                {
                    // Work distribution context objects can be shared
                    EAGER_ASSERT(dynamicBlobBufRanges[cmd.addrOffset] == cmd.size, "Invalid offset");
                }
            }
            break;

            case EngineArcCommandId::NOP:
            {
                const EcbCommandSizeType padding = m_recipeHal.getNopCommand(curCmd).padding * nopPaddingUnits;
                pos += padding;
            }
            break;

            case EngineArcCommandId::LIST_SIZE:
            case EngineArcCommandId::WD_FENCE_AND_EXE:
                break;

            default:
                EAGER_ASSERT(false, "Unsupported ECB command");
                break;
        }
    }
}

}  // namespace eager_mode
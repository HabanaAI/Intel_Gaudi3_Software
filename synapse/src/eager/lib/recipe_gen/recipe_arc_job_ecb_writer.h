#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_hal_base.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <cstdint>
#include <map>
#include <optional>

namespace eager_mode
{
// Representation of actual ECB to be written
class EcbWriter
{
public:
    EcbWriter(const RecipeHalBase& recipeHal, EcbType ecbType);

    void init(const ecb_t& ecb, EcbCommandSizeType chunkIdx = 0);
    bool copy(const ecb_t&                             srcEcb,
              EcbCommandSizeType                       srcEcbNetSize,
              bool                                     posChanged,
              const std::optional<EcbCommandSizeType>& chunkIdx = {});
    void postMemcpyUpdate(BlobSizeType                       patchBlobBase,
                          BlobSizeType                       execBlobBase,
                          BlobSizeType                       dynBlobBase,
                          BlobSizeType                       templateConstExeBlobOffset,
                          const std::optional<BlobSizeType>& actualConstExeBlobOffset);
    void createTail();
    bool isCompleted() const { return m_isTailCreated; }

    void writeNopCommand(const mini_ecb_packets::Nop& cmd, bool validate = true);
    void writeStaticDescCommand(const mini_ecb_packets::StaticDescV2& cmd, bool validate = true);
    void writeListSizeCommand(const mini_ecb_packets::ListSize& cmd, bool validate = true);
    void writeSchedDmaCommand(const mini_ecb_packets::SchedDma& cmd, bool validate = true);
    void writeFenceCommand(const mini_ecb_packets::Fence& cmd, bool validate = true);

    void collectCommandsRanges(std::map<size_t, size_t>& patchingBlobBufRanges,
                               std::map<size_t, size_t>& execBlobBufRange) const;
    void collectCommandsRanges(std::map<size_t, size_t>& dynamicBlobBufRanges) const;

private:
    void preWriteToEcb(EcbCommandSizeType cmdSize);
    void createHead();

private:
    const RecipeHalBase& m_recipeHal;
    // values to be initialized at constructor
    const EcbCommandSizeType m_cmdAlignmentConstraint;  // All commands must be aligned to a certain constant
    const EcbType            m_ecbType;                 // Static or dynamic

    // FSM flags
    bool m_isFirstWrite  = true;
    bool m_isTailCreated = false;

    // Pointer to a chunk of Ecb command
    struct CmdsChunkType
    {
        uint8_t*           cmds    = nullptr;  // Pointer to the commands chunk
        EcbCommandSizeType size    = 0;        // Size of the chunks in bytes
        bool               isValid = false;    // Content validity
    };

    EcbCommandSizeType m_writerPos = 0;  // Next writing offset in actual ECB block
    CmdsChunkType      m_ecb;            // Representation of ECB raw data commands
    CmdsChunkType      m_cmdsToCopy;     // Chunk of commands to be copied
};

}  // namespace eager_mode

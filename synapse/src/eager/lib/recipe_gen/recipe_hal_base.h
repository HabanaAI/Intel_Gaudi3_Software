#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/mini_command_packets.h"
#include "recipe_gen/blob_to_desc_map_structs.h"
#include "synapse_common_types.h"

// std includes
#include <array>

namespace eager_mode
{
class RecipeHalBase
{
public:
    using EcbCommandSizesArray = std::array<EcbCommandSizeType, static_cast<unsigned>(EngineArcCommandId::COUNT)>;
    RecipeHalBase(const EcbCommandSizesArray& ecbCommandSizes) : m_ecbCommandSizes(ecbCommandSizes) {}
    virtual ~RecipeHalBase() {}

    // bounds across all chip types.
    // compile time asserts enforcing correctness in the relevant chip specific recipe hals.
    // used for small vectors local storage size definition.
    static constexpr BlobsNrType   maxMmeEnginesNr           = 8;
    static constexpr BlobsNrType   maxDmaEnginesNr           = 5;
    static constexpr unsigned      maxBaseRegistersCacheSize = 32;
    static constexpr TensorsNrType maxDmaTensorsNr           = 2;
    static constexpr TensorsNrType maxMmeTensorsNr           = 3;

    // specific tpc nodes memory footprint query (excluding ecb considerations
    // and ignoring cases where kernel is referenced by additional instnaces
    // so that we only pay the overhead for the kernel once).
    virtual unsigned getConstantNodeRecipeMemoryRequirement() const                             = 0;
    virtual unsigned getCastNodeRecipeMemoryRequirement(synDataType from, synDataType to) const = 0;

    // base register cache info
    virtual unsigned getBaseRegistersCacheSize() const             = 0;
    virtual unsigned getRegForBaseAddress(unsigned regIndex) const = 0;
    // general descriptor info
    // secondaryEngineType is only needed for Gaudi3 for MME
    virtual StructSizeType
                           getFirstRegPosInDesc(EngineType          engineType,
                                                SecondaryEngineType secondaryEngineType = SecondaryEngineType::NONE) const = 0;
    virtual StructSizeType getDescSize(EngineType engineType) const                    = 0;
    virtual StructSizeType getWorkDistributionContextSize(EngineType engineType) const = 0;
    // qman block position
    virtual AsicRegType getQmanBlockStart() const = 0;
    virtual AsicRegType getQmanBlockEnd() const   = 0;
    // qman commands offsets
    virtual BlobSizeType     getWreg32RegPos(BlobSizeType cmdOffset) const                    = 0;
    virtual BlobSizeType     getWrBulkRegPos(BlobSizeType cmdOffset, AsicRegType regId) const = 0;
    virtual BlobSizeType     getWreg64LongOffsetPos(BlobSizeType cmdOffset) const             = 0;
    virtual UnalignedPosType getWreg64LongBasePos(BlobSizeType cmdOffset) const               = 0;
    // MME descriptor specific information
    // secondaryEngineType is only needed for Gaudi3 for MME
    virtual AsicRange
    getMmeDescriptorBlackListRange(SecondaryEngineType secondaryEngineType = SecondaryEngineType::NONE) const = 0;
    virtual StructSizeType getMmeTensorAOffsetInDescriptor() const                                            = 0;
    virtual StructSizeType getMmeTensorBOffsetInDescriptor() const                                            = 0;
    virtual StructSizeType getMmeTensorCOffsetInDescriptor() const                                            = 0;
    // DMA descriptor specific information
    virtual StructSizeType getDmaTensorInputOffsetInDescriptor() const  = 0;
    virtual StructSizeType getDmaTensorOutputOffsetInDescriptor() const = 0;
    // TPC descriptor specific information
    virtual AsicRegType              getTpcDescriptorBlackListReg() const        = 0;
    virtual std::optional<AsicRange> getTpcDescriptorWhiteListRange() const      = 0;
    virtual StructSizeType           getTpcFirstTensorOffsetInDescriptor() const = 0;
    virtual StructSizeType           getTpcTensorsOffsetEndInDescriptor() const  = 0;
    virtual StructSizeType           getTpcTensorSizeInDescriptor() const        = 0;
    virtual TensorsNrType            getMaxTpcTensorsNr() const                  = 0;

    // Qman
    virtual PacketId            getQmanCommandOpcode(const Byte* data) const = 0;
    virtual QmanCommandSizeType getQmanCommandSize(PacketId opCode) const    = 0;
    // translate the chip specific commands into device agnostic information
    virtual void                          printEcbCommand(const Byte* data) const      = 0;
    virtual mini_qman_packets::Wreg32     getWreg32Command(const Byte* data) const     = 0;
    virtual mini_qman_packets::WregBulk   getWregBulkCommand(const Byte* data) const   = 0;
    virtual mini_qman_packets::Wreg64Long getWreg64LongCommand(const Byte* data) const = 0;

    // ECB
    virtual EngineArcCommandId getEcbCommandOpcode(const Byte* data) const   = 0;
    virtual EngineArcCommandId getEcbCommandOpcode(unsigned opcodeVal) const = 0;
    inline EcbCommandSizeType  getEcbCommandSize(EngineArcCommandId opCode) const
    {
        auto idx = static_cast<unsigned>(opCode);
        EAGER_ASSERT(idx < m_ecbCommandSizes.size(), " getEcbCommandSize unsupported ECB command type");
        return m_ecbCommandSizes[idx];
    }

    // translate the chip specific commands into device agnostic information
    virtual mini_ecb_packets::Nop          getNopCommand(const Byte* data) const        = 0;
    virtual mini_ecb_packets::StaticDescV2 getStaticDescCommand(const Byte* data) const = 0;
    virtual mini_ecb_packets::ListSize     getListSizeCommand(const Byte* data) const   = 0;
    virtual mini_ecb_packets::SchedDma     getSchedDmaCommand(const Byte* data) const   = 0;
    // translate and write chip agnostic commands into ECB buffer
    virtual void writeNopCommand(Byte* data, const mini_ecb_packets::Nop& cmd) const                 = 0;
    virtual void writeStaticDescCommand(Byte* data, const mini_ecb_packets::StaticDescV2& cmd) const = 0;
    virtual void writeListSizeCommand(Byte* data, const mini_ecb_packets::ListSize& cmd) const       = 0;
    virtual void writeSchedDmaCommand(Byte* data, const mini_ecb_packets::SchedDma& cmd) const       = 0;
    virtual void writeFenceCommand(Byte* data, const mini_ecb_packets::Fence& cmd) const             = 0;
    // ECB writer info
    virtual unsigned           getMaxSupportedWorkDistributionContextCount() const     = 0;
    virtual EcbCommandSizeType getEcbCommandAlignmentConstraint(EcbType ecbType) const = 0;
    virtual unsigned           getEcbNopCommandSize() const                            = 0;
    virtual bool               isFenceCmdExistsInDynamicEcbCmdBuf() const              = 0;
    inline bool                isFenceCmdExistsInDynamicEcb() const
    {
        return isFenceCmdExistsInDynamicEcbCmdBuf() &&
               ((getMaxSupportedWorkDistributionContextCount() % (getEcbNopCommandSize() * 2)) == 0);
    }
    // Return size of header in bytes
    inline EcbCommandSizeType getStaticHeadSize() const
    {
        return getEcbCommandSize(EngineArcCommandId::LIST_SIZE) + getEcbCommandSize(EngineArcCommandId::NOP);
    }
    inline EcbCommandSizeType getDynamicHeadSize() const { return getEcbCommandSize(EngineArcCommandId::LIST_SIZE); }
    inline EcbCommandSizeType getTemplateHeadSize(EcbType ecbType) const
    {
        return getEcbCommandSize(EngineArcCommandId::LIST_SIZE) +
               ((ecbType == EcbType::STATIC) ? getEcbCommandSize(EngineArcCommandId::NOP) : 0);
    }
    // Return size of tail in bytes
    inline EcbCommandSizeType getTailSize() const { return getEcbCommandSize(EngineArcCommandId::NOP); }
    // Return size of switching to other queue in bytes
    inline EcbCommandSizeType getSwitchingSize() const { return getEcbCommandSize(EngineArcCommandId::NOP); }
    // Sum of head and tail sizes for static ECBs
    inline EcbCommandSizeType getStaticNonRepeatedSize() const { return getStaticHeadSize() + getSwitchingSize(); }
    // Sum of head and tail sizes for dynamic ECBs
    inline EcbCommandSizeType getDynamicNonRepeatedSize() const { return getDynamicHeadSize() + getSwitchingSize(); }
    // Return the maximum size of ECB command we currently support
    inline EcbCommandSizeType getSupportedMaxSizeCmd() const { return getEcbCommandSize(EngineArcCommandId::NOP) * 2; }

    // general info
    virtual ChipType getChipType() const                        = 0;
    virtual unsigned getVersionMinor() const                    = 0;
    virtual unsigned getCacheLineSizeInBytes() const            = 0;
    virtual unsigned getMaxEngines(EngineType engineType) const = 0;
    virtual bool     isConstExeBlobSupported() const            = 0;

private:
    const EcbCommandSizesArray m_ecbCommandSizes;
};

}  // namespace eager_mode
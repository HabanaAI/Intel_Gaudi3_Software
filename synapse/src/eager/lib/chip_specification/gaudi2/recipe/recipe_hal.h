#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_hal_base.h"

namespace eager_mode::gaudi2_spec_info
{
class RecipeHal final : public RecipeHalBase
{
public:
    RecipeHal();

    // This is an optimization, this flag can be readjusted without affecting functionality
    static bool constexpr supportConstExeBlob = true;

    // base register cache info
    unsigned getBaseRegistersCacheSize() const override;
    unsigned getRegForBaseAddress(unsigned regIndex) const override;
    // specific tpc nodes memory footprint query
    unsigned getConstantNodeRecipeMemoryRequirement() const override;
    unsigned getCastNodeRecipeMemoryRequirement(synDataType from, synDataType to) const override;
    // general descriptor info
    StructSizeType
                   getFirstRegPosInDesc(EngineType          engineType,
                                        SecondaryEngineType secondaryEngineType = SecondaryEngineType::NONE) const override;
    StructSizeType getDescSize(EngineType engineType) const override;
    StructSizeType getWorkDistributionContextSize(EngineType engineType) const override;
    // qman block position
    AsicRegType getQmanBlockStart() const override;
    AsicRegType getQmanBlockEnd() const override;
    // qman commands offsets
    BlobSizeType     getWreg32RegPos(BlobSizeType cmdOffset) const override;
    BlobSizeType     getWrBulkRegPos(BlobSizeType cmdOffset, AsicRegType regId) const override;
    BlobSizeType     getWreg64LongOffsetPos(BlobSizeType cmdOffset) const override;
    UnalignedPosType getWreg64LongBasePos(BlobSizeType cmdOffset) const override;
    // MME descriptor specific information
    AsicRange
    getMmeDescriptorBlackListRange(SecondaryEngineType secondaryEngineType = SecondaryEngineType::NONE) const override;
    StructSizeType           getMmeTensorAOffsetInDescriptor() const override;
    StructSizeType           getMmeTensorBOffsetInDescriptor() const override;
    StructSizeType           getMmeTensorCOffsetInDescriptor() const override;
    // DMA descriptor specific information
    StructSizeType getDmaTensorInputOffsetInDescriptor() const override;
    StructSizeType getDmaTensorOutputOffsetInDescriptor() const override;
    // TPC descriptor specific information
    AsicRegType    getTpcDescriptorBlackListReg() const override;
    std::optional<AsicRange> getTpcDescriptorWhiteListRange() const override;
    StructSizeType getTpcFirstTensorOffsetInDescriptor() const override;
    StructSizeType getTpcTensorsOffsetEndInDescriptor() const override;
    StructSizeType getTpcTensorSizeInDescriptor() const override;
    TensorsNrType  getMaxTpcTensorsNr() const override;

    // Qman
    PacketId            getQmanCommandOpcode(const Byte* data) const override;
    QmanCommandSizeType getQmanCommandSize(PacketId opCode) const override;
    // translate the chip specific commands into device agnostic information
    void                                      printEcbCommand(const Byte* data) const override;
    eager_mode::mini_qman_packets::Wreg32     getWreg32Command(const Byte* data) const override;
    eager_mode::mini_qman_packets::WregBulk   getWregBulkCommand(const Byte* data) const override;
    eager_mode::mini_qman_packets::Wreg64Long getWreg64LongCommand(const Byte* data) const override;

    // ECB
    EngineArcCommandId getEcbCommandOpcode(const Byte* data) const override;
    EngineArcCommandId getEcbCommandOpcode(unsigned opcodeVal) const override;
    // translate the chip specific commands into device agnostic information
    eager_mode::mini_ecb_packets::Nop          getNopCommand(const Byte* data) const override;
    eager_mode::mini_ecb_packets::StaticDescV2 getStaticDescCommand(const Byte* data) const override;
    eager_mode::mini_ecb_packets::ListSize     getListSizeCommand(const Byte* data) const override;
    eager_mode::mini_ecb_packets::SchedDma     getSchedDmaCommand(const Byte* data) const override;
    // translate and write chip agnostic commands into ECB buffer
    void writeNopCommand(Byte* data, const eager_mode::mini_ecb_packets::Nop& cmd) const override;
    void writeStaticDescCommand(Byte* data, const eager_mode::mini_ecb_packets::StaticDescV2& cmd) const override;
    void writeListSizeCommand(Byte* data, const eager_mode::mini_ecb_packets::ListSize& cmd) const override;
    void writeSchedDmaCommand(Byte* data, const eager_mode::mini_ecb_packets::SchedDma& cmd) const override;
    void writeFenceCommand(Byte* data, const eager_mode::mini_ecb_packets::Fence& cmd) const override;
    // ECB writer info
    unsigned           getMaxSupportedWorkDistributionContextCount() const override;
    EcbCommandSizeType getEcbCommandAlignmentConstraint(EcbType ecbType) const override;
    unsigned           getEcbNopCommandSize() const override;
    bool               isFenceCmdExistsInDynamicEcbCmdBuf() const override;
    // general info
    ChipType getChipType() const override;
    unsigned getVersionMinor() const override;
    unsigned getCacheLineSizeInBytes() const override;
    unsigned getMaxEngines(EngineType engineType) const override;
    bool     isConstExeBlobSupported() const override { return supportConstExeBlob; }
};

}  // namespace eager_mode::gaudi2_spec_info
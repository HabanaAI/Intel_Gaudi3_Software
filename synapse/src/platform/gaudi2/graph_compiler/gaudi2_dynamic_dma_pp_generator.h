#pragma once

#include "gaudi2_types.h"
#include <dynamic_dma_pp_generator.h>

namespace gaudi2
{
class DynamicDMAPatchPointGenerator : public ::DynamicDMAPatchPointGenerator<gaudi2::DmaDesc>
{
public:
    using ::DynamicDMAPatchPointGenerator<gaudi2::DmaDesc>::DynamicDMAPatchPointGenerator;

    virtual void addDynamicExecutionPatchPoints(const DMANode& node) override;

    virtual DynamicShapeFieldInfoSharedPtr getDynamicAddressInfo(const DMAPhysicalMemoryOpNode& node,
                                                                 ShapeFuncID                 smf) override;

    virtual uint32_t fieldTypeAndDimToOffset(FieldType fieldType, uint32_t dim) override;
    virtual std::pair<uint32_t, BlockT> fieldTypeAndDimToOffsetAndBlock(FieldType fieldType, uint32_t dim) override;

    virtual uint64_t getAddressPtrForPhysicalMemOp(const DMAPhysicalMemoryOpNode& node) override;

    virtual uint32_t addressOffsetHi(bool isSrc) override;
    virtual uint32_t addressOffsetLo(bool isSrc) override;
    virtual uint32_t addressValueHi(bool isSrc) override;
    virtual uint32_t addressValueLo(bool isSrc) override;

    friend class LinearFieldInfo;
    friend class ZeroSizeFieldInfo;
    friend class ConstantFieldInfo;
};

class ConstantFieldInfo : public DynamicShapeFieldInfo
{
public:
    ConstantFieldInfo(DynamicDMAPatchPointGenerator& gen,
                      pNode                          origin,
                      NodeROI*                       roi,
                      FieldType                      fieldType,
                      uint32_t                       dim,
                      uint32_t                       constValue,
                      ShapeFuncID                    smf = ShapeFuncID::SMF_PATCH_ON_ZERO_SIZE)
    : DynamicShapeFieldInfo(gen.fieldTypeAndDimToOffsetAndBlock(fieldType, dim).first,
                            fieldType,
                            smf,
                            std::move(origin),
                            roi)
    {
        m_size = 1;

        std::vector<uint8_t> convertedMetadata(sizeof(constValue));
        memcpy(convertedMetadata.data(), &constValue, sizeof(constValue));

        setMetadata(convertedMetadata);
    }

    BasicFieldInfoSharedPtr clone() const override { return std::make_shared<ConstantFieldInfo>(*this); }
};

class ZeroSizeFieldInfo : public ConstantFieldInfo
{
public:
    ZeroSizeFieldInfo(DynamicDMAPatchPointGenerator& gen,
                      pNode                          origin,
                      NodeROI*                       roi,
                      FieldType                      fieldType,
                      int                            dim,
                      ShapeFuncID                    smf = ShapeFuncID::SMF_PATCH_ON_ZERO_SIZE)
    : ConstantFieldInfo(gen, origin, roi, fieldType, dim, 0, smf)
    {
        m_isUnskippable = true;
    }
    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<ZeroSizeFieldInfo>(*this); }
};

class LinearFieldInfo : public ConstantFieldInfo
{
public:
    LinearFieldInfo(DynamicDMAPatchPointGenerator& gen,
                    pNode                          origin,
                    NodeROI*                       roi,
                    ShapeFuncID                    smf = ShapeFuncID::SMF_PATCH_ON_ZERO_SIZE)
    : ConstantFieldInfo(gen, origin, roi, FieldType::FIELD_DYNAMIC_DMA_COMMIT, 0, getNewCommitRegisterValue(gen), smf)
    {
        m_isUnskippable = true;
    }
    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<LinearFieldInfo>(*this); }

private:
    static uint32_t getNewCommitRegisterValue(DynamicDMAPatchPointGenerator& gen)
    {
        dma_core_ctx::reg_commit reg;
        reg._raw = gen.wrapper().getFwCtx().dma_commit_reg;
        reg.lin  = 1;
        return reg._raw;
    }
};

class DynamicOffsetFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicOffsetFieldInfo(DynamicDMAPatchPointGenerator& gen,
                           pNode                          origin,
                           NodeROI*                       roi,
                           ShapeFuncID                    smf,
                           uint32_t                       fieldOffset)
    : DynamicShapeFieldInfo(fieldOffset, FieldType::FIELD_DYNAMIC_OFFSET, smf, origin, roi)
    {
        m_size = 2;
    }
    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicOffsetFieldInfo>(*this); }
};

}  // namespace gaudi2

#include "gaudi2_dynamic_dma_pp_generator.h"
#include "dynamic_dma_pp_generator.inl"
#include <memory>

namespace gaudi2
{
DynamicShapeFieldInfoSharedPtr DynamicDMAPatchPointGenerator::getDynamicAddressInfo(const DMAPhysicalMemoryOpNode& node,
                                                                                    ShapeFuncID                 smf)
{
    bool isSrc       = node.isSrcDynamicStrided();
    auto fieldOffset =
        (offsetof(block_axuser_dma_core_ctx, ctx) +
         (isSrc ? offsetof(block_dma_core_ctx, src_offset_lo) : offsetof(block_dma_core_ctx, dst_offset_lo))) /
        sizeof(uint32_t);
    auto& bfci = wrapper().getBasicFieldsContainerInfo();
    auto  fieldInfo =
        std::make_shared<DynamicOffsetFieldInfo>(*this,
                                                 const_cast<DMAPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                 bfci.getRoi(),
                                                 smf,
                                                 fieldOffset);
    return fieldInfo;
}

std::pair<uint32_t, DynamicDMAPatchPointGenerator::BlockT>
DynamicDMAPatchPointGenerator::fieldTypeAndDimToOffsetAndBlock(FieldType fieldType, uint32_t dim)
{
    switch (fieldType)
    {
        case FieldType::FIELD_DYNAMIC_DMA_COMMIT:
            return {offsetof(edma_wd_ctxt_t, dma_commit_reg) / sizeof(uint32_t), BlockT::WD_CTX};
        default:
            return {fieldTypeAndDimToOffset(fieldType, dim), BlockT::DESCRIPTOR};
    }
}

uint32_t DynamicDMAPatchPointGenerator::fieldTypeAndDimToOffset(FieldType fieldType, uint32_t dim)
{
    const std::size_t ctxOffset = offsetof(block_axuser_dma_core_ctx, ctx) / sizeof(uint32_t);

    switch (fieldType)
    {
        case FieldType::FIELD_DYNAMIC_DMA_SRC:
            switch (dim)
            {
                case 0:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_tsize_0) / sizeof(uint32_t);
                case 1:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_tsize_1) / sizeof(uint32_t);
                case 2:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_tsize_2) / sizeof(uint32_t);
                case 3:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_tsize_3) / sizeof(uint32_t);
                case 4:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_tsize_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_DMA_DST:
            switch (dim)
            {
                case 0:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_tsize_0) / sizeof(uint32_t);
                case 1:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_tsize_1) / sizeof(uint32_t);
                case 2:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_tsize_2) / sizeof(uint32_t);
                case 3:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_tsize_3) / sizeof(uint32_t);
                case 4:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_tsize_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE:
            switch (dim)
            {
                case 1:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_stride_1) / sizeof(uint32_t);
                case 2:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_stride_2) / sizeof(uint32_t);
                case 3:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_stride_3) / sizeof(uint32_t);
                case 4:
                    return ctxOffset + offsetof(block_dma_core_ctx, src_stride_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_SRC_LAST_STRIDE:
            return ctxOffset + offsetof(block_dma_core_ctx, src_stride_4) / sizeof(uint32_t);
        case FieldType::FIELD_DYNAMIC_DST_BULK_SIZE_STRIDE:
            switch (dim)
            {
                case 1:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_stride_1) / sizeof(uint32_t);
                case 2:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_stride_2) / sizeof(uint32_t);
                case 3:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_stride_3) / sizeof(uint32_t);
                case 4:
                    return ctxOffset + offsetof(block_dma_core_ctx, dst_stride_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_DST_LAST_STRIDE:
            return ctxOffset + offsetof(block_dma_core_ctx, dst_stride_4) / sizeof(uint32_t);
        default:
            break;
    }
    HB_ASSERT(false, "Invalid field type and/or dimension number for DMA dynamic shape patch");
    return 0;
}

uint32_t DynamicDMAPatchPointGenerator::addressOffsetHi(bool isSrc)
{
    return (isSrc ? offsetof(block_dma_core_ctx, src_offset_hi) : offsetof(block_dma_core_ctx, dst_offset_hi));
}

uint32_t DynamicDMAPatchPointGenerator::addressOffsetLo(bool isSrc)
{
    return (isSrc ? offsetof(block_dma_core_ctx, src_offset_lo) : offsetof(block_dma_core_ctx, dst_offset_lo));
}

uint32_t DynamicDMAPatchPointGenerator::addressValueHi(bool isSrc)
{
    const auto& descriptor = wrapper().getDescriptor();
    return (isSrc ? descriptor.ctx.src_offset_hi.val : descriptor.ctx.dst_offset_hi.val);
}

uint32_t DynamicDMAPatchPointGenerator::addressValueLo(bool isSrc)
{
    const auto& descriptor = wrapper().getDescriptor();
    return (isSrc ? descriptor.ctx.src_offset_lo.val : descriptor.ctx.dst_offset_lo.val);
}

void DynamicDMAPatchPointGenerator::addDynamicExecutionPatchPoints(const DMANode& node)
{
    auto& bfci = wrapper().getBasicFieldsContainerInfo();
    auto  roi  = bfci.getRoi();

    pNode nodePtr = (const_cast<DMANode&>(node)).shared_from_this();

    for (int i = 0; i < MAX_DIMENSIONS_NUM; ++i)
    {
        auto zeroSizeInfoSrc = std::make_shared<ZeroSizeFieldInfo>(*this,
                                                                   nodePtr,
                                                                   roi,
                                                                   FieldType::FIELD_DYNAMIC_DMA_SRC,
                                                                   i,
                                                                   ShapeFuncID::SMF_PATCH_ON_ZERO_SIZE_FIRST_INPUT);
        auto zeroSizeInfoDst = std::make_shared<ZeroSizeFieldInfo>(*this,
                                                                   nodePtr,
                                                                   roi,
                                                                   FieldType::FIELD_DYNAMIC_DMA_DST,
                                                                   i,
                                                                   ShapeFuncID::SMF_PATCH_ON_ZERO_SIZE_FIRST_INPUT);
        bfci.add(std::static_pointer_cast<BasicFieldInfo>(zeroSizeInfoSrc));
        bfci.add(std::static_pointer_cast<BasicFieldInfo>(zeroSizeInfoDst));
    }
}

uint64_t DynamicDMAPatchPointGenerator::getAddressPtrForPhysicalMemOp(const DMAPhysicalMemoryOpNode& node)
{
    bool     isSrc = node.isSrcDynamicStrided();
    ptrToInt baseAddress;
    baseAddress.u32[0] =
        isSrc ? wrapper().getDescriptor().ctx.src_offset_lo._raw : wrapper().getDescriptor().ctx.dst_offset_lo._raw;
    baseAddress.u32[1] =
        isSrc ? wrapper().getDescriptor().ctx.src_offset_hi._raw : wrapper().getDescriptor().ctx.dst_offset_hi._raw;
    return baseAddress.u64;
}

}  // namespace gaudi2

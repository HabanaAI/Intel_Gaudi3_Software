#include "gaudi_dynamic_dma_pp_generator.h"
#include "dynamic_dma_pp_generator.inl"

namespace gaudi
{
DynamicShapeFieldInfoSharedPtr DynamicDMAPatchPointGenerator::getDynamicAddressInfo(const DMAPhysicalMemoryOpNode& node,
                                                                                    ShapeFuncID                    smf)
{
    bool isSrc = node.isSrcDynamicStrided();
    auto fieldOffset =
        (isSrc ? offsetof(block_dma_core, src_base_lo) : offsetof(block_dma_core, dst_base_lo)) / sizeof(uint32_t);
    auto& bfci = wrapper().getBasicFieldsContainerInfo();
    auto  fieldInfo =
        std::make_shared<DynamicAddressFieldInfo>(fieldOffset,
                                                  FIELD_DYNAMIC_ADDRESS,
                                                  smf,
                                                  const_cast<DMAPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                  bfci.getRoi(),
                                                  bfci.findAddress(fieldOffset),
                                                  bfci.findAddress(fieldOffset + 1));
    return fieldInfo;
}

uint32_t DynamicDMAPatchPointGenerator::fieldTypeAndDimToOffset(FieldType fieldType, uint32_t dim)
{
    switch (fieldType)
    {
        case FieldType::FIELD_DYNAMIC_DMA_SRC:
            switch (dim)
            {
                case 0:
                    return offsetof(block_dma_core, src_tsize_0) / sizeof(uint32_t);
                case 1:
                    return offsetof(block_dma_core, src_tsize_1) / sizeof(uint32_t);
                case 2:
                    return offsetof(block_dma_core, src_tsize_2) / sizeof(uint32_t);
                case 3:
                    return offsetof(block_dma_core, src_tsize_3) / sizeof(uint32_t);
                case 4:
                    return offsetof(block_dma_core, src_tsize_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_DMA_DST:
            switch (dim)
            {
                case 0:
                    return offsetof(block_dma_core, dst_tsize_0) / sizeof(uint32_t);
                case 1:
                    return offsetof(block_dma_core, dst_tsize_1) / sizeof(uint32_t);
                case 2:
                    return offsetof(block_dma_core, dst_tsize_2) / sizeof(uint32_t);
                case 3:
                    return offsetof(block_dma_core, dst_tsize_3) / sizeof(uint32_t);
                case 4:
                    return offsetof(block_dma_core, dst_tsize_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE:
            switch (dim)
            {
                case 1:
                    return offsetof(block_dma_core, src_stride_1) / sizeof(uint32_t);
                case 2:
                    return offsetof(block_dma_core, src_stride_2) / sizeof(uint32_t);
                case 3:
                    return offsetof(block_dma_core, src_stride_3) / sizeof(uint32_t);
                case 4:
                    return offsetof(block_dma_core, src_stride_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_SRC_LAST_STRIDE:
            return offsetof(block_dma_core, src_stride_4) / sizeof(uint32_t);
        case FieldType::FIELD_DYNAMIC_DST_BULK_SIZE_STRIDE:
            switch (dim)
            {
                case 1:
                    return offsetof(block_dma_core, dst_stride_1) / sizeof(uint32_t);
                case 2:
                    return offsetof(block_dma_core, dst_stride_2) / sizeof(uint32_t);
                case 3:
                    return offsetof(block_dma_core, dst_stride_3) / sizeof(uint32_t);
                case 4:
                    return offsetof(block_dma_core, dst_stride_4) / sizeof(uint32_t);
            }
        case FieldType::FIELD_DYNAMIC_DST_LAST_STRIDE:
            return offsetof(block_dma_core, dst_stride_4) / sizeof(uint32_t);
        default:
            break;
    }
    HB_ASSERT(false, "Invalid field type and/or dimension number for DMA dynamic shape patch");
    return 0;
}

std::pair<uint32_t, DynamicDMAPatchPointGenerator::BlockT>
DynamicDMAPatchPointGenerator::fieldTypeAndDimToOffsetAndBlock(FieldType fieldType, uint32_t dim)
{
    // in Gaudi1, all fields are in the descriptor
    return {fieldTypeAndDimToOffset(fieldType, dim), BlockT::DESCRIPTOR};
}

uint32_t DynamicDMAPatchPointGenerator::addressOffsetHi(bool isSrc)
{
    return (isSrc ? offsetof(block_dma_core, src_base_hi) : offsetof(block_dma_core, dst_base_hi));
}

uint32_t DynamicDMAPatchPointGenerator::addressOffsetLo(bool isSrc)
{
    return (isSrc ? offsetof(block_dma_core, src_base_lo) : offsetof(block_dma_core, dst_base_lo));
}

uint32_t DynamicDMAPatchPointGenerator::addressValueHi(bool isSrc)
{
    const auto& descriptor = wrapper().getDescriptor();
    return (isSrc ? descriptor.src_base_hi.val : descriptor.dst_base_hi.val);
}

uint32_t DynamicDMAPatchPointGenerator::addressValueLo(bool isSrc)
{
    const auto& descriptor = wrapper().getDescriptor();
    return (isSrc ? descriptor.src_base_lo.val : descriptor.dst_base_lo.val);
}

uint64_t DynamicDMAPatchPointGenerator::getAddressPtrForPhysicalMemOp(const DMAPhysicalMemoryOpNode& node)
{
    bool isSrc = node.isSrcDynamicStrided();

    ptrToInt p;
    p.u32[0] = addressValueLo(isSrc);
    p.u32[1] = addressValueHi(isSrc);
    return maskOutMemoryID(p.u64);
}
}  // namespace gaudi

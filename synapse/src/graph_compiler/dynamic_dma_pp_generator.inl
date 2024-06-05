#pragma once

#include "dynamic_dma_pp_generator.h"

#include "habana_nodes.h"
#include "physical_memory_ops_nodes.h"
#include "h2d_tensors.h"

#include "recipe.h"

#include "defs.h"
#include "utils.h"

#include "physical_concat_split_subnode.h"

#include "recipe_metadata.h"
#include <optional>

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::generatePatchPoints(const DMANode& node)
{
    if (!node.isDynamicShape() || GCFG_DISABLE_DS_DMA_ROI_PATCHING.value())
    {
        return;
    }

    switch (node.getDynamicMemoryOpType())
    {
        case DMA_OP_SERIALIZE:
        case DMA_OP_DESERIALIZE:
            generateSerializationPatchPoints(node);
            if (node.isROIDynamic(m_wrapper.getBasicFieldsContainerInfo().getRoi()))
            {
                addDynamicExecutionPatchPoints(node);
            }
            return;  // do not patch dynamic shapes
        case DMA_OP_DYNAMIC_STRIDE:
            generateViewPatchPoints(node);
            break;
        case DMA_OP_DYNAMIC_BASE:
            generateDMADynamicAddressPatchPoints(node);
            break;
        case DMA_OP_DYNAMIC_SLICE:
            generateSlicePatchPoints(node);
            break;
        case DMA_OP_NONE:
            break;
    }

    if (node.isROIDynamic(m_wrapper.getBasicFieldsContainerInfo().getRoi()))
    {
        addDynamicExecutionPatchPoints(node);
        addDynamicShapePatchPoint(node, FieldType::FIELD_DYNAMIC_DMA_DST);
        addDynamicShapePatchPoint(node, FieldType::FIELD_DYNAMIC_DMA_SRC);
    }
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::addDynamicShapePatchPoint(const DMANode& node, FieldType fieldType)
{
    // We always need to patch src size even though memset node does not have any input tensors
    // We derive patch values from the output tensor instead
    //
    auto tensor =
        (fieldType == FieldType::FIELD_DYNAMIC_DMA_SRC && !node.isMemset()) ? node.getInput(0) : node.getOutput(0);
    auto origin = const_cast<DMANode&>(node).shared_from_this();

    for (uint32_t dim = 0; dim < MAX_DIMENSIONS_NUM; ++dim)
    {
        if (tensor->isDynamicDim(dim))
        {
            addSizePatchPoint(node, tensor, fieldType, dim);
        }
    }
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::addSizePatchPoint(const DMANode&   node,
                                                            const TensorPtr& tensor,
                                                            FieldType        fieldType,
                                                            unsigned         dim)
{
    auto& basicFieldsContainerInfo = m_wrapper.getBasicFieldsContainerInfo();

    uint32_t fieldIndexOffset = fieldTypeAndDimToOffset(fieldType, dim);
    auto     fieldInfo        = std::make_shared<DynamicDmaFieldInfo>(fieldIndexOffset,
                                                           fieldType,
                                                           const_cast<DMANode&>(node).shared_from_this(),
                                                           basicFieldsContainerInfo.getRoi());

    dma_sm_params_t metadata {0};

    metadata.this_dim       = dim;
    metadata.is_destination = fieldType == FIELD_DYNAMIC_DMA_DST;
    metadata.is_memset      = node.isMemset();
    metadata.element_size   = tensor->getElementSizeInBytes();
    metadata.is_total       = 0;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    BasicFieldInfoPair fieldInfoPair {fieldIndexOffset, fieldInfo};
    basicFieldsContainerInfo.add(fieldInfoPair);
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertBaseAddressPatchPoint(const DMAPhysicalMemoryOpNode& node, ShapeFuncID smf)
{
    address_sm_params_t metadata = {0};
    metadata.base_address        = getAddressPtrForPhysicalMemOp(node);
    metadata.element_size        = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src              = node.isSrcDynamicStrided();

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    auto& bfci      = m_wrapper.getBasicFieldsContainerInfo();
    auto  fieldInfo = getDynamicAddressInfo(node, smf);
    fieldInfo->setMetadata(convertedMetadata);
    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertViewBaseAddressPatchPoint(const DMAPhysicalMemoryOpNode& node)
{
    view_address_sm_params_t metadata = {0};
    metadata.base_address             = getAddressPtrForPhysicalMemOp(node);
    metadata.element_size             = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                   = node.isSrcDynamicStrided();
    if (node.getInput(1) && node.getInput(1)->isHost2DeviceTensor())
    {
        synDynamicStridedDmaH2dTensor* dynStridesMaxData =
            reinterpret_cast<synDynamicStridedDmaH2dTensor*>(node.getInput(1)->getHostMaxData());
        HB_ASSERT_PTR(dynStridesMaxData);
        metadata.max_offset = dynStridesMaxData->offset;
        HB_ASSERT(dynStridesMaxData->num_strides <= sizeof(metadata.max_strides) / sizeof(metadata.max_strides[0]),
                  "num_strides is greater than num of supported strides in metadata");
        std::copy(dynStridesMaxData->strides, dynStridesMaxData->strides + dynStridesMaxData->num_strides, metadata.max_strides);
    }

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    auto& bfci      = m_wrapper.getBasicFieldsContainerInfo();
    auto  fieldInfo = getDynamicAddressInfo(node, ShapeFuncID::SMF_DMA_VIEW_OFFSET);
    fieldInfo->setMetadata(convertedMetadata);
    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::generateSerializationPatchPoints(const DMANode& dmaNode)
{
    const auto& node = dynamic_cast<const DMAPhysicalMemoryOpNode&>(dmaNode);
    FieldType   sizeFieldType;
    FieldType   manyStridesFieldType;
    FieldType   lastStrideFieldType;
    TensorPtr   input       = node.getInput(0);
    bool        isSrcStride = node.isSrcDynamicStrided();
    std::optional<unsigned> firstDynamicDimIndex;

    // In case we patch the src stride - patch the dest sizes as regular and vice versa.
    if (isSrcStride)
    {
        addDynamicShapePatchPoint(node, FieldType::FIELD_DYNAMIC_DMA_DST);
        sizeFieldType        = FieldType::FIELD_DYNAMIC_DMA_SRC;
        manyStridesFieldType = FieldType::FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE;
        lastStrideFieldType  = FieldType::FIELD_DYNAMIC_SRC_LAST_STRIDE;
    }
    else
    {
        addDynamicShapePatchPoint(node, FieldType::FIELD_DYNAMIC_DMA_SRC);
        sizeFieldType        = FieldType::FIELD_DYNAMIC_DMA_DST;
        manyStridesFieldType = FieldType::FIELD_DYNAMIC_DST_BULK_SIZE_STRIDE;
        lastStrideFieldType  = FieldType::FIELD_DYNAMIC_DST_LAST_STRIDE;
    }

    firstDynamicDimIndex = input->getFirstDynamicDimIndex();
    HB_ASSERT(firstDynamicDimIndex, "Tensor is expected to by dynamic");

    unsigned bulkDimsToUpdate = SYN_MAX_TENSOR_DIM - (*firstDynamicDimIndex + 1);

    // Patch dim0 if necessary. It is the only one out of order (and also the last), and gets its own pp to simplify
    // things
    if (*firstDynamicDimIndex == 0)
    {
        addSizePatchPoint(node, input, sizeFieldType, 0);
        (*firstDynamicDimIndex)++;
        bulkDimsToUpdate--;
    }

    // We are patching many consecutive fields, and we know that when we set dynamic patch point the fields will
    // be written for sure (and can't be optimized out due to diff mechanism).
    // This means we can try and merge some patch points into a single one.

    // The first field (size field) is not aligned to 64 bit and can possibly be in its own cmd,
    // so it has to be in a single different pp. we add a regular size pp for that field.
    addSizePatchPoint(node, input, sizeFieldType, *firstDynamicDimIndex);

    // Now we have few consecutive fields that are aligned to 64bit, and are always written,
    // so they are going to be in the same WBULK for sure. Therefore we can have a single PP for all of them.
    // add the main patch point containing all but the last stride (All the fields that go into the WBULK).
    // This pp can not exist if the only dynamic dimension is 4 (We only have size_4, stride_4 - which is the next pp).
    insertBulkSizeStridePatchPoint(node, manyStridesFieldType, bulkDimsToUpdate, *firstDynamicDimIndex);

    // The last field will always be stride_4 (If we patch any size, we have to patch all teh strides bigger,
    // so if we got to this point, we need to patch stride_4 for sure). It is too unaligned to 64bit, and may go into
    // its own different command, so we have to add a different pp for it.
    // This pp is special for this case and isn't used in another place.
    insertLastStridePatchPoint(node, lastStrideFieldType);

    // If we have dynamic strides, we have dynmic address
    insertBaseAddressPatchPoint(node, SMF_DYNAMIC_OFFSET);
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::generateViewPatchPoints(const DMANode& dmaNode)
{
    const auto& node = dynamic_cast<const DMAPhysicalMemoryOpNode&>(dmaNode);

    synDynamicStridedDmaH2dTensor* dynStridesMaxData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(node.getInput(1)->getHostMaxData());
    synDynamicStridedDmaH2dTensor* dynStridesMinData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(node.getInput(1)->getHostMinData());

    HB_ASSERT_PTR(dynStridesMaxData);
    HB_ASSERT_PTR(dynStridesMinData);

    // patch every dimension with dynamic stride
    for (unsigned i = 1; i < node.getInput(0)->getDim(); i++)
    {
        if (dynStridesMaxData->strides[i] != dynStridesMinData->strides[i])
        {
            insertViewStridePatchPoint(node, i);
        }
    }

    // needed when any stride changes
    insertViewBaseAddressPatchPoint(node);
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertViewStridePatchPoint(const DMAPhysicalMemoryOpNode& node, unsigned dim)
{
    bool      isSrcView                = node.isSrcDynamicStrided();
    auto&     basicFieldsContainerInfo = m_wrapper.getBasicFieldsContainerInfo();
    FieldType fieldType =
        isSrcView ? FieldType::FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE : FieldType::FIELD_DYNAMIC_DST_BULK_SIZE_STRIDE;

    uint32_t fieldIndexOffset = fieldTypeAndDimToOffset(fieldType, dim);
    auto     fieldInfo =
        std::make_shared<PatchSingleStrideFieldInfo>(fieldIndexOffset,
                                                     fieldType,
                                                     ShapeFuncID::SMF_DMA_VIEW_STRIDE,
                                                     const_cast<DMAPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                     basicFieldsContainerInfo.getRoi());

    view_stride_sm_params_t metadata = {0};
    metadata.element_size            = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                  = isSrcView;
    metadata.this_dim                = dim;
    metadata.num_real_elements       = node.getRealParentSize() / metadata.element_size;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    basicFieldsContainerInfo.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertBulkSizeStridePatchPoint(const DMAPhysicalMemoryOpNode& node,
                                                                         FieldType                   fieldType,
                                                                         unsigned                    dimsToUpdate,
                                                                         unsigned                    firstDynamicDim)
{
    if (dimsToUpdate == 0) return;

    auto&    basicFieldsContainerInfo = m_wrapper.getBasicFieldsContainerInfo();
    uint32_t fieldIndexOffset         = fieldTypeAndDimToOffset(fieldType, firstDynamicDim);

    unsigned affectedFieldsCount = dimsToUpdate * 2;
    auto     fieldInfo =
        std::make_shared<PatchManyStridesFieldInfo>(fieldIndexOffset,
                                                    fieldType,
                                                    affectedFieldsCount,
                                                    const_cast<DMAPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                    basicFieldsContainerInfo.getRoi());

    bulk_size_stride_sm_params_t metadata = {0};
    metadata.first_dynamic_dim            = firstDynamicDim;
    metadata.affected_fields              = affectedFieldsCount;
    metadata.element_size                 = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                       = fieldType == FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    basicFieldsContainerInfo.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertLastStridePatchPoint(const DMAPhysicalMemoryOpNode& node,
                                                                     FieldType                   fieldType)
{
    auto& basicFieldsContainerInfo = m_wrapper.getBasicFieldsContainerInfo();

    uint32_t fieldIndexOffset = fieldTypeAndDimToOffset(fieldType, 4);
    auto     fieldInfo        = std::make_shared<PatchLastStride>(fieldIndexOffset,
                                                       fieldType,
                                                       const_cast<DMAPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                       basicFieldsContainerInfo.getRoi());

    last_stride_sm_params_t metadata = {0};
    metadata.element_size            = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                  = fieldType == FIELD_DYNAMIC_SRC_LAST_STRIDE;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    basicFieldsContainerInfo.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::generateDMADynamicAddressPatchPoints(const DMANode& node)
{
    const auto* physNodePtr = dynamic_cast<const PhysicalConcatSplitSubnodeDMA*>(&node);
    HB_ASSERT(physNodePtr != nullptr, "The node is not a physical concat subnode!");

    bool isSrc          = physNodePtr->isSrcDynamicStrided();
    auto concatSplitDim = physNodePtr->concatSplitDim();
    auto input0         = physNodePtr->getInput(0);
    auto output0        = physNodePtr->getOutput(0);

    ptrToInt p;

    p.u32[0] = addressValueLo(isSrc);
    p.u32[1] = addressValueHi(isSrc);

    auto baseAddress = p.u64;

    physical_concat_split_sm_params_t metadata = {0};

    metadata.roi_base_address       = maskOutMemoryID(baseAddress);
    metadata.element_size           = node.getInput(0)->getElementSizeInBytes();
    metadata.number_in_concat_split = physNodePtr->nodeNumberInConcatSplit();
    metadata.concat_split_dim       = concatSplitDim;
    if (output0->isAliasedTensor())
    {
        output0 = output0->getAliasTensor();
    }
    output0->getAllStridesInElements(metadata.output_strides);

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));
    auto& bfci = m_wrapper.getBasicFieldsContainerInfo();

    auto fieldInfo = getDynamicAddressInfo(*physNodePtr, ShapeFuncID::SMF_DMA_BASEADDR);

    fieldInfo->setMetadata(convertedMetadata);
    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::generateSlicePatchPoints(const DMANode& dmaNode)
{
    const auto& node = dynamic_cast<const DMAPhysicalMemoryOpNode&>(dmaNode);

    // patch every dimension with dynamic stride
    for (unsigned i = 1; i < node.getInput(0)->getDim(); i++)
    {
        if (node.getInput(1)->isDynamicDim(i))
        {
            insertSliceStridePatchPoint(node, i);
        }
    }

    // needed when any stride changes
    insertSliceBaseAddressPatchPoint(node);
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertSliceBaseAddressPatchPoint(const DMAPhysicalMemoryOpNode& node)
{
    slice_address_sm_params_t metadata = {0};
    metadata.base_address              = getAddressPtrForPhysicalMemOp(node);
    metadata.element_size              = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                    = node.isSrcDynamicStrided();
    metadata.num_real_elements         = node.getRealParentSize() / metadata.element_size;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    auto& bfci      = m_wrapper.getBasicFieldsContainerInfo();
    auto  fieldInfo = getDynamicAddressInfo(node, ShapeFuncID::SMF_DMA_SLICE_OFFSET);
    fieldInfo->setMetadata(convertedMetadata);
    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

template<typename Desc>
void DynamicDMAPatchPointGenerator<Desc>::insertSliceStridePatchPoint(const DMAPhysicalMemoryOpNode& node, unsigned dim)
{
    bool      isSrcView                = node.isSrcDynamicStrided();
    auto&     basicFieldsContainerInfo = m_wrapper.getBasicFieldsContainerInfo();
    FieldType fieldType =
        isSrcView ? FieldType::FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE : FieldType::FIELD_DYNAMIC_DST_BULK_SIZE_STRIDE;

    uint32_t fieldIndexOffset = fieldTypeAndDimToOffset(fieldType, dim);
    auto     fieldInfo =
        std::make_shared<PatchSingleStrideFieldInfo>(fieldIndexOffset,
                                                     fieldType,
                                                     ShapeFuncID::SMF_DMA_SLICE_STRIDE,
                                                     const_cast<DMAPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                     basicFieldsContainerInfo.getRoi());

    slice_stride_sm_params_t metadata = {0};
    metadata.element_size             = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                   = isSrcView;
    metadata.dim                      = dim;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    basicFieldsContainerInfo.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

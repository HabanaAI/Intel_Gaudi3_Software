#include "gaudi3_dynamic_tpc_pp_generator.h"
#include "dynamic_tpc_pp_generator.inl"
#include "synapse_api_types.h"
#include "physical_concat_split_subnode.h"

namespace gaudi3
{

// XXX Unify to work with pointers
void DynamicTPCPatchPointGenerator::generateDynamicStridePatchPointsForNode(const TPCNode& node)
{
    const TPCPhysicalMemoryOpNode* memcopyNode = dynamic_cast<const TPCPhysicalMemoryOpNode*>(&node);
    if (memcopyNode != nullptr)
    {
        uint32_t descTensorIdx = 0;

        const auto& inputs = node.getInputs();
        for (const auto& in : inputs)
        {
            if (in->isAuxTensor() || in->getTensorType() == OUTPUT_DESCRIBING_SHAPE_TENSOR) continue;
            if (in->isHost2DeviceTensor()) continue;
            if (memcopyNode->isSrcDynamicStrided())
                generateDynamicStridePatchPointsForTensor(*memcopyNode,
                                                          in,
                                                          descTensorIdx,
                                                          false,
                                                          &in - &inputs.front());
            ++descTensorIdx;
        }

        const auto& outputs = node.getOutputs();
        for (const auto& out : outputs)
        {
            if (out->isAuxTensor() || out->isShapeTensor()) continue;
            if (memcopyNode->isDstDynamicStrided())
                generateDynamicStridePatchPointsForTensor(*memcopyNode,
                                                          out,
                                                          descTensorIdx,
                                                          true,
                                                          &out - &outputs.front());
            ++descTensorIdx;
        }
    }
}

void DynamicTPCPatchPointGenerator::generateDynamicStridePatchPointsForTensor(const TPCPhysicalMemoryOpNode& node,
                                                                              const TensorPtr&               tensor,
                                                                              uint32_t descTensorIndex,
                                                                              bool     isOutput,
                                                                              uint32_t nodeTensorIndex)
{
    switch (node.getDynamicMemoryOpType())
    {
        case DMA_OP_SERIALIZE:
        case DMA_OP_DESERIALIZE:
            addDynamicStridesPatchPoints(node, tensor, descTensorIndex, isOutput, nodeTensorIndex);
            break;

        case DMA_OP_DYNAMIC_SLICE:
            addDynamicSlicePatchPoints(node, tensor, descTensorIndex, isOutput, nodeTensorIndex);
            break;

        case DMA_OP_DYNAMIC_STRIDE:
            addDynamicViewPatchPoints(node, tensor, descTensorIndex, isOutput, nodeTensorIndex);
            break;

        case DMA_OP_DYNAMIC_BASE:
            addDynamicBasePatchPoints(node, tensor, descTensorIndex, isOutput, nodeTensorIndex);
            break;

        default:
            break;
    }
}

void DynamicTPCPatchPointGenerator::addDynamicStridesPatchPoints(const TPCPhysicalMemoryOpNode& node,
                                                                 const TensorPtr&               tensor,
                                                                 uint32_t                       descTensorIndex,
                                                                 bool                           isOutput,
                                                                 uint32_t                       nodeTensorIndex)
{
    auto     origin      = const_cast<TPCPhysicalMemoryOpNode&>(node).shared_from_this();
    auto     tensorRank  = tensor->getDim();
    unsigned firstDynDim = tensor->getFirstDynamicDimIndex().value();

    // Last stride is never stored
    for (unsigned dim = firstDynDim; dim < tensorRank - 1; dim++)
    {
        // If any dimension is dynamic, all strides after it
        // should be modified even when corresponding dimensions
        // are static

        auto offset    = tensorIndexAndDimToStrideOffset(descTensorIndex, dim);
        auto fieldInfo = std::make_shared<DynamicTPCStrideFieldInfo>(offset,
                                                                     origin,
                                                                     m_wrapper.getBasicFieldsContainerInfo().getRoi(),
                                                                     ShapeFuncID::SMF_TPC_STRIDE);

        tpc_stride_sm_params_t metadata = {0};
        metadata.this_dim               = dim;
        metadata.first_dynamic_dim      = firstDynDim;
        metadata.is_output              = static_cast<uint32_t>(isOutput);
        metadata.element_size           = tensor->getElementSizeInBytes();
        metadata.tensor_index           = nodeTensorIndex;

        std::vector<uint8_t> convertedMetadata(sizeof(metadata));
        memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

        fieldInfo->setMetadata(convertedMetadata);

        BasicFieldInfoPair fieldInfoPair {offset, fieldInfo};
        m_wrapper.getBasicFieldsContainerInfo().add(fieldInfoPair);
    }
}

void DynamicTPCPatchPointGenerator::addDynamicBasePatchPoints(const TPCPhysicalMemoryOpNode& node,
                                                              const TensorPtr&               tensor,
                                                              uint32_t                       descTensorIndex,
                                                              bool                           isOutput,
                                                              uint32_t                       nodeTensorIndex)
{
      const auto* physNodePtr = dynamic_cast<const PhysicalConcatSplitSubnodeTPC*>(&node);
      HB_ASSERT(physNodePtr != nullptr, "The node is not a physical concat subnode!");

      bool isSrc          = physNodePtr->isSrcDynamicStrided();
      auto concatSplitDim = physNodePtr->concatSplitDim();
      auto input0         = physNodePtr->getInput(0);
      auto output0        = physNodePtr->getOutput(0);

      auto baseAddress = DynamicTPCPatchPointGenerator::getTensorBaseAddress(descTensorIndex);

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

      uint32_t fieldIndexOffset = tensorIndexToBaseAddressOffset(descTensorIndex);
      auto     fieldInfo =
          std::make_shared<DynamicAddressFieldInfo>(fieldIndexOffset,
                                                    FIELD_DYNAMIC_ADDRESS,
                                                    ShapeFuncID::SMF_DMA_BASEADDR,
                                                    const_cast<TPCPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                    bfci.getRoi(),
                                                    bfci.findAddress(fieldIndexOffset),
                                                    bfci.findAddress(fieldIndexOffset + 1));

      fieldInfo->setMetadata(convertedMetadata);
      bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

void DynamicTPCPatchPointGenerator::insertSliceStridePatchPoint(const TPCPhysicalMemoryOpNode& node, unsigned dim)
{
    bool     isSrc           = node.isSrcDynamicStrided();
    auto&    bfci            = m_wrapper.getBasicFieldsContainerInfo();
    uint32_t descTensorIndex = isSrc ? 0 : 1;

    uint32_t fieldIndexOffset = tensorIndexAndDimToStrideOffset(descTensorIndex, dim - 1);
    auto     fieldInfo =
        std::make_shared<DynamicTPCStrideFieldInfo>(fieldIndexOffset,
                                                    const_cast<TPCPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                    bfci.getRoi(),
                                                    ShapeFuncID::SMF_TPC_SLICE_STRIDE);

    slice_stride_sm_params_t metadata = {0};
    metadata.element_size             = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                   = isSrc;
    metadata.dim                      = dim;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

void DynamicTPCPatchPointGenerator::insertSliceBaseAddressPatchPoint(const TPCPhysicalMemoryOpNode& node)
{
    bool     isSrc           = node.isSrcDynamicStrided();
    auto&    bfci            = m_wrapper.getBasicFieldsContainerInfo();
    uint32_t descTensorIndex = isSrc ? 0 : 1;

    slice_address_sm_params_t metadata = {0};
    metadata.base_address              = getTensorBaseAddress(descTensorIndex);
    metadata.element_size              = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                    = isSrc;
    metadata.num_real_elements         = node.getRealParentSize() / metadata.element_size;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    uint32_t fieldIndexOffset = tensorIndexToBaseAddressOffset(descTensorIndex);
    auto     fieldInfo =
        std::make_shared<DynamicAddressFieldInfo>(fieldIndexOffset,
                                                  FIELD_DYNAMIC_ADDRESS,
                                                  SMF_TPC_SLICE_OFFSET,
                                                  const_cast<TPCPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                  bfci.getRoi(),
                                                  bfci.findAddress(fieldIndexOffset),
                                                  bfci.findAddress(fieldIndexOffset + 1));

    fieldInfo->setMetadata(convertedMetadata);
    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

void DynamicTPCPatchPointGenerator::addDynamicSlicePatchPoints(const TPCPhysicalMemoryOpNode& node,
                                                               const TensorPtr&               tensor,
                                                               uint32_t                       descTensorIndex,
                                                               bool                           isOutput,
                                                               uint32_t                       nodeTensorIndex)
{
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

void DynamicTPCPatchPointGenerator::insertViewStridePatchPoint(const TPCPhysicalMemoryOpNode& node, unsigned dim)
{
    bool     isSrc           = node.isSrcDynamicStrided();
    auto&    bfci            = m_wrapper.getBasicFieldsContainerInfo();
    uint32_t descTensorIndex = isSrc ? 0 : 1;

    uint32_t fieldIndexOffset = tensorIndexAndDimToStrideOffset(descTensorIndex, dim - 1);
    auto     fieldInfo =
        std::make_shared<DynamicTPCStrideFieldInfo>(fieldIndexOffset,
                                                    const_cast<TPCPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                    bfci.getRoi(),
                                                    ShapeFuncID::SMF_TPC_VIEW_STRIDE);

    view_stride_sm_params_t metadata = {0};
    metadata.element_size            = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                  = isSrc;
    metadata.this_dim                = dim;
    metadata.num_real_elements       = node.getRealParentSize() / metadata.element_size;

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

void DynamicTPCPatchPointGenerator::insertViewBaseAddressPatchPoint(const TPCPhysicalMemoryOpNode& node)
{
    bool     isSrc           = node.isSrcDynamicStrided();
    auto&    bfci            = m_wrapper.getBasicFieldsContainerInfo();
    uint32_t descTensorIndex = isSrc ? 0 : 1;

    view_address_sm_params_t metadata = {0};
    metadata.base_address             = getTensorBaseAddress(descTensorIndex);
    metadata.element_size             = node.getInput(0)->getElementSizeInBytes();
    metadata.is_src                   = isSrc;
    if (node.getInput(1) && node.getInput(1)->isHost2DeviceTensor())
    {
        synDynamicStridedDmaH2dTensor* dynStridesMaxData =
            reinterpret_cast<synDynamicStridedDmaH2dTensor*>(node.getInput(1)->getHostMaxData());
        HB_ASSERT_PTR(dynStridesMaxData);
        metadata.max_offset = dynStridesMaxData->offset;
        HB_ASSERT(dynStridesMaxData->num_strides <= sizeof(metadata.max_strides) / sizeof(metadata.max_strides[0]),
                  "num_strides is greater than num of supported strides in metadata");
        std::copy(dynStridesMaxData->strides,
                  dynStridesMaxData->strides + dynStridesMaxData->num_strides,
                  metadata.max_strides);
    }

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    uint32_t fieldIndexOffset = tensorIndexToBaseAddressOffset(descTensorIndex);
    auto     fieldInfo =
        std::make_shared<DynamicAddressFieldInfo>(fieldIndexOffset,
                                                  FIELD_DYNAMIC_ADDRESS,
                                                  SMF_TPC_VIEW_OFFSET,
                                                  const_cast<TPCPhysicalMemoryOpNode&>(node).shared_from_this(),
                                                  bfci.getRoi(),
                                                  bfci.findAddress(fieldIndexOffset),
                                                  bfci.findAddress(fieldIndexOffset + 1));

    fieldInfo->setMetadata(convertedMetadata);
    bfci.add(std::static_pointer_cast<BasicFieldInfo>(fieldInfo));
}

void DynamicTPCPatchPointGenerator::addDynamicViewPatchPoints(const TPCPhysicalMemoryOpNode& node,
                                                              const TensorPtr&               tensor,
                                                              uint32_t                       descTensorIndex,
                                                              bool                           isOutput,
                                                              uint32_t                       nodeTensorIndex)
{
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

size_t DynamicTPCPatchPointGenerator::tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim)
{
    size_t tensorOffset = offsetof(TpcDesc, m_tensors) + sizeof(TensorDescGaudi3) * tensorIndex;
#define SIZE_OFFSET(_D)                                                                                                \
    (tensorOffset + offsetof(TensorDescGaudi3, shared) + offsetof(block_tpc_tensor_shared, dim_##_D##_size)) /         \
        sizeof(uint32_t)
    switch (dim)
    {
        case 0:
            return SIZE_OFFSET(0);
        case 1:
            return SIZE_OFFSET(1);
        case 2:
            return SIZE_OFFSET(2);
        case 3:
            return SIZE_OFFSET(3);
        case 4:
            return SIZE_OFFSET(4);
        default:
            break;
    }
    HB_ASSERT(false, "Invalid dimension number for TPC dynamic shape patch");
    return 0;
#undef SIZE_OFFSET
}

size_t DynamicTPCPatchPointGenerator::tensorIndexAndDimToStrideOffset(uint32_t tensorIndex, uint32_t dim)
{
    size_t tensorOffset = offsetof(TpcDesc, m_tensors) + sizeof(TensorDescGaudi3) * tensorIndex;
#define STRIDE_OFFSET(_D)                                                                                              \
    (tensorOffset + offsetof(TensorDescGaudi3, shared) + offsetof(block_tpc_tensor_shared, dim_##_D##_stride)) /       \
        sizeof(uint32_t)
    switch (dim)
    {
        // stride0 is always 1, interesting strides start from stride1
        case 0:
            return STRIDE_OFFSET(1);
        case 1:
            return STRIDE_OFFSET(2);
        case 2:
            return STRIDE_OFFSET(3);
        case 3:
            return STRIDE_OFFSET(4);
        default:
            break;
    }
    HB_ASSERT(false, "Invalid dimension number for TPC dynamic shape patch");
    return 0;
#undef STRIDE_OFFSET
}

size_t DynamicTPCPatchPointGenerator::tensorIndexToBaseAddressOffset(uint32_t tensorIndex)
{
    size_t tensorOffset = offsetof(TpcDesc, m_tensors) + sizeof(TensorDescGaudi3) * tensorIndex;
    return (tensorOffset + offsetof(TensorDescGaudi3, base) + offsetof(block_tpc_tensor_base, base_addr_low)) /
           sizeof(uint32_t);
}

size_t DynamicTPCPatchPointGenerator::getTensorBaseAddress(uint32_t tensorIndex)
{
    ptrToInt addr;
    addr.u32[0] = m_wrapper.getDescriptor().m_tensors[tensorIndex].base.base_addr_low._raw;
    addr.u32[1] = m_wrapper.getDescriptor().m_tensors[tensorIndex].base.base_addr_high._raw;
    return maskOutMemoryID(addr.u64);
}

void DynamicTPCPatchPointGenerator::addPatchPointsForIndexSpace(
    const TPCNode&                                       node,
    const tpc_lib_api::HabanaKernelInstantiation&        instance,
    const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections)
{
    // One patch point to rule them all

    auto                      origin                   = const_cast<TPCNode&>(node).shared_from_this();
    BasicFieldsContainerInfo& basicFieldsContainerInfo = getIndexSpaceBFCI();
    const auto&               wrapper                  = getWrapper();
    const auto&               fwCtx                    = wrapper.getFwCtx();

    tpc_sm_params_gaudi3_t metadata;
    metadata.m_indexSpaceCopy  = fwCtx.ist;
    metadata.m_dimensions_mask = 0;

    const auto& nodeROI = basicFieldsContainerInfo.getRoi();

    if (nodeProjections.size() > 0)
    {
        for (const auto& prj : nodeProjections)
        {
            Settable<Node::NodeDynamicShapeProjection> nodeProjection;
            nodeProjection.set(prj);
            int projectionCount =
                fillDynamicShapePatchPointIndexSpaceProjection(node,
                                                               nodeROI,
                                                               instance,
                                                               prj.indexSpaceDim,
                                                               metadata.m_dimensions[prj.indexSpaceDim]);
            if (projectionCount > 0)
            {
                metadata.m_dimensions_mask |= (1 << prj.indexSpaceDim);
                metadata.m_dimensions[prj.indexSpaceDim].num_projections = projectionCount;
                metadata.m_dimensions[prj.indexSpaceDim].this_dim        = prj.indexSpaceDim;
            }
        }
    }
    else
    {
        for (uint32_t idx = 0; idx < instance.indexSpaceRank; ++idx)
        {
            Settable<Node::NodeDynamicShapeProjection> nodeProjection;
            int projectionCount = fillDynamicShapePatchPointIndexSpaceProjection(node,
                                                                                 nodeROI,
                                                                                 instance,
                                                                                 idx,
                                                                                 metadata.m_dimensions[idx]);
            if (projectionCount > 0)
            {
                metadata.m_dimensions_mask |= (1 << idx);
                metadata.m_dimensions[idx].num_projections = projectionCount;
                metadata.m_dimensions[idx].this_dim        = idx;
            }
        }
    }

    if (metadata.m_dimensions_mask == 0)
    {
        // nothing to patch!
        return;
    }

    auto fieldInfo = std::make_shared<DynamicTPCIndexSpaceFieldInfoGaudi3>(origin, nodeROI);

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    BasicFieldInfoPair fieldInfoPair {0, fieldInfo};
    basicFieldsContainerInfo.add(fieldInfoPair);
}

}  // namespace gaudi3

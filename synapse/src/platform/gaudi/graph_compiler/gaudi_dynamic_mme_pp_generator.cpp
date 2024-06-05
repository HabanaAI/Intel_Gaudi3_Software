#include "gaudi_dynamic_mme_pp_generator.h"
#include "recipe_metadata.h"
#include "platform/gaudi/graph_compiler/descriptor_generator.h"
#include "platform/gaudi/graph_compiler/smf/smf.h"
#include "mme_desc_gen_utils.h"

namespace gaudi
{
class DynamicMmeFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicMmeFieldInfo(uint32_t fieldIndexOffset, pNode origin, NodeROI* roi);

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicMmeFieldInfo>(*this); }
};

DynamicMmeFieldInfo::DynamicMmeFieldInfo(uint32_t fieldIndexOffset, pNode origin, NodeROI* roi)
: DynamicShapeFieldInfo(fieldIndexOffset, FIELD_DYNAMIC_MME_VALID_ELEMENTS, ShapeFuncID::SMF_MME, origin, roi)
{
    m_size = 1;
}

void DynamicMMEPatchPointGenerator::generateDynamicShapesPatchPoints(const MmeNode&                     node,
                                                                     DescriptorWrapper<gaudi::MmeDesc>& descWrapper)
{
    if (!node.isROIDynamic(descWrapper.getBasicFieldsContainerInfo().getRoi()) ||
        GCFG_DISABLE_DS_MME_ROI_PATCHING.value())
    {
        return;
    }

    pTensor xTensor, wTensor, yTensor, oTensor;
    auto opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);

    if (xTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, xTensor, MmeCommon::e_mme_op_x, descWrapper);
    }
    if (wTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, wTensor, MmeCommon::e_mme_op_w, descWrapper);
    }
    if (yTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, yTensor, MmeCommon::e_mme_op_y, descWrapper);
    }

    auto convNode = dynamic_cast<const ConvBaseNode*>(&node);
    if (convNode != nullptr && xTensor->isDynamicShape())
    {
        HB_ASSERT(convNode != nullptr, "Cannot convert a convolution node pointer!");
        const auto& params = convNode->getConvolutionParams();
        if (params.paddingType == PADDING_SAME)
        {
            generatePaddingPatchPoints(*convNode, descWrapper);
        }
    }
}

void DynamicMMEPatchPointGenerator::generateDynamicPatchPointsForOperand(const MmeNode&                     node,
                                                                         const pTensor&                     tensor,
                                                                         MmeCommon::EMmeOperand             op,
                                                                         DescriptorWrapper<gaudi::MmeDesc>& descWrapper)
{
    const Mme::MmeTensorDesc* tensorDesc = nullptr;
    MmeCommon::MmeLayerParams params     = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi);
    params.opType                        = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    gaudi::getOperandMapping(op, descWrapper.getDescriptor(), tensorDesc, !params.isConvOperation());
    if (tensorDesc == nullptr)
    {
        // No operand to patch
        return;
    }

    for (unsigned d = 0; d < tensor->getDim(); d++)
    {
        if (tensor->isDynamicDim(d))
        {
            addValidElementsPatchPoint(node, tensor, tensorDesc, d, descWrapper);
        }
    }
}

void DynamicMMEPatchPointGenerator::addValidElementsPatchPoint(const MmeNode&                     node,
                                                               const pTensor&                     tensor,
                                                               const Mme::MmeTensorDesc*          tensorDesc,
                                                               int                                dim,
                                                               DescriptorWrapper<gaudi::MmeDesc>& descWrapper)
{
    BasicFieldsContainerInfo& bfci = descWrapper.getBasicFieldsContainerInfo();
    MmeDesc&                  desc = descWrapper.getDescriptor();

    uint32_t fieldIndexOffset = ((uint64_t) & (tensorDesc->validElements[dim]) - (uint64_t)&desc) / sizeof(uint32_t);

    auto                           origin = const_cast<MmeNode&>(node).shared_from_this();
    DynamicShapeFieldInfoSharedPtr fieldInfo =
        std::make_shared<DynamicMmeFieldInfo>(fieldIndexOffset, origin, bfci.getRoi());

    mme_sm_params_t metadata {0};
    metadata.dim                 = dim;
    metadata.tensor_input_index  = getTensorIndex(node.getInputs(), tensor);
    metadata.tensor_output_index = getTensorIndex(node.getOutputs(), tensor);

    // In runtime we will get the actual size in elements, and we want to update the descriptor accordingly.
    // The valid elements of a dimension is multiplied by the stride (or other in case of layout optimizations)
    // If we save the multiply factor, when we get the actual size we can just multiply by it and update the descriptor.
    metadata.multiply_factor = tensorDesc->validElements[dim] / tensor->getSizeInElements(dim);

    std::vector<uint8_t> serializedMetadata(sizeof(metadata));
    memcpy(serializedMetadata.data(), &metadata, sizeof(metadata));
    fieldInfo->setMetadata(serializedMetadata);

    BasicFieldInfoPair fieldInfoPair {fieldIndexOffset, fieldInfo};
    bfci.add(fieldInfoPair);
}

int DynamicMMEPatchPointGenerator::getTensorIndex(const TensorVector& tensorVector, const pTensor& tensor)
{
    int i = 0;

    for (const pTensor& curr : tensorVector)
    {
        if (curr == tensor)
        {
            return i;
        }
        i++;
    }
    return INDEX_NOT_APPLICABLE;
}

class DynamicMmePaddingFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicMmePaddingFieldInfo(uint32_t fieldIndexOffset, pNode origin, NodeROI* roi);

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicMmePaddingFieldInfo>(*this); }
};

DynamicMmePaddingFieldInfo::DynamicMmePaddingFieldInfo(uint32_t fieldIndexOffset, pNode origin, NodeROI* roi)
: DynamicShapeFieldInfo(fieldIndexOffset, FIELD_DYNAMIC_MME_PADDING, ShapeFuncID::SMF_MME_PADDING, origin, roi)
{
    m_size = 1;
}

void DynamicMMEPatchPointGenerator::generatePaddingPatchPoints(const ConvBaseNode&                convNode,
                                                               DescriptorWrapper<gaudi::MmeDesc>& descWrapper)
{
    std::vector<int32_t*> roiBaseOffsetVec;
    auto                  opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, convNode);
    getRoiBaseOffsetFields(MmeCommon::e_mme_op_x, opType, descWrapper.getDescriptor(), roiBaseOffsetVec);

    for (unsigned i = 0; i < roiBaseOffsetVec.size(); ++i)
    {
        generateOneAguPaddingPatchPoint(convNode, descWrapper, roiBaseOffsetVec[i]);
    }
}

void DynamicMMEPatchPointGenerator::generateOneAguPaddingPatchPoint(const ConvBaseNode&                convNode,
                                                                    DescriptorWrapper<gaudi::MmeDesc>& descWrapper,
                                                                    const int32_t*                     roiOffsets)
{
    uint32_t lastDim = convNode.is3DConvolution() ? DIM_D_FOR_5D_TENSOR : DIM_H;

    for (uint32_t dim = DIM_W; dim <= lastDim; ++dim)
    {
        generateOneDimPaddingPatchPoint(convNode, descWrapper, dim, roiOffsets);
    }
}

void DynamicMMEPatchPointGenerator::generateOneDimPaddingPatchPoint(const ConvBaseNode&                convNode,
                                                                    DescriptorWrapper<gaudi::MmeDesc>& descWrapper,
                                                                    uint32_t                           dim,
                                                                    const int32_t*                     roiOffsets)
{
    BasicFieldsContainerInfo& bfci = descWrapper.getBasicFieldsContainerInfo();
    const auto&               desc = descWrapper.getDescriptor();

    const auto&             params = convNode.getConvolutionParams();
    mme_padding_sm_params_t metadata;

    std::copy(std::begin(params.padding), std::end(params.padding), metadata.old_padding);
    std::copy(std::begin(params.stride), std::end(params.stride), metadata.conv_stride);
    std::copy(std::begin(params.dilation), std::end(params.dilation), metadata.conv_dilation);
    std::copy(std::begin(params.kernel), std::end(params.kernel), metadata.conv_kernel);
    std::copy(roiOffsets, roiOffsets + MAX_DIMENSIONS_NUM, metadata.old_offsets);

    uint64_t tensorStrides[Tensor::c_numOfStrides];

    metadata.opType   = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, convNode);
    metadata.this_dim = dim;
    pTensor xTensor, wTensor, yTensor, oTensor;
    getTensorRolesCommon(convNode, metadata.opType, xTensor, wTensor, yTensor, oTensor);
    xTensor->getAllStridesInElements(tensorStrides);

    // tensorStrides is uint64_t and metadata.tensor_strides is uint32_t[]
    // MME can only have 32 bit strides
    // this std::copy does correct copying with narrowing 64->32
    std::copy(tensorStrides, tensorStrides + MAX_DIMENSIONS_NUM, metadata.tensor_strides);

    // XXX is this correct?
    uint32_t fieldIndexOffset =
        (reinterpret_cast<const char*>(roiOffsets) - reinterpret_cast<const char*>(&desc)) / sizeof(uint32_t) + dim;

    auto                           origin = const_cast<ConvBaseNode&>(convNode).shared_from_this();
    DynamicShapeFieldInfoSharedPtr fieldInfo =
        std::make_shared<DynamicMmePaddingFieldInfo>(fieldIndexOffset, origin, bfci.getRoi());
    std::vector<uint8_t> serializedMetadata(sizeof(metadata));
    memcpy(serializedMetadata.data(), &metadata, sizeof(metadata));
    fieldInfo->setMetadata(serializedMetadata);

    BasicFieldInfoPair fieldInfoPair {fieldIndexOffset, fieldInfo};
    bfci.add(fieldInfoPair);
}
}  // namespace gaudi

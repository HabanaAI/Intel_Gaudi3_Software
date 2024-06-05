#pragma once

#include "arc_dynamic_mme_pp_generator.h"
#include "recipe_metadata.h"
#include "platform/gaudi/graph_compiler/smf/smf.h"
#include "arc_mme_field_infos.h"
#include "mme_desc_gen_utils.h"
#include "mme_brain_ifc.h"

namespace arc_platforms
{

template <typename MmeTypes>
void DynamicMMEPatchPointGenerator<MmeTypes>::generateDynamicShapesPatchPoints(
        const MmeNode&                node,
        const MmeDescriptorGenerator& descGenerator,
        DescriptorWrapper<MmeDesc>&   descWrapper,
        unsigned                      engineIdx)
{
    auto* roi = descWrapper.getBasicFieldsContainerInfo().getRoi();

    if ((roi != nullptr && !node.isROIDynamic(roi)) ||
            GCFG_DISABLE_DS_MME_ROI_PATCHING.value())
    {
        return;
    }

    TensorPtr xTensor, wTensor, yTensor, oTensor;
    auto chipType = MmeBrainIfc::getMmeChipType(node.getGraphTraits()->getHalReader()->getDeviceType());
    auto opType = getOperationTypeCommon(chipType, node);
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);

    if (xTensor != nullptr && xTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, xTensor, MmeCommon::e_mme_op_x, descGenerator, descWrapper, engineIdx);
    }
    if (wTensor != nullptr && wTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, wTensor, MmeCommon::e_mme_op_w, descGenerator, descWrapper, engineIdx);
    }
    if (yTensor != nullptr && yTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, yTensor, MmeCommon::e_mme_op_y, descGenerator, descWrapper, engineIdx);
    }
    if (oTensor != nullptr && oTensor->isDynamicShape())
    {
        generateDynamicPatchPointsForOperand(node, oTensor, MmeCommon::e_mme_op_o, descGenerator, descWrapper, engineIdx);
    }

    generateDynamicExecutionPatchPoint(node, descWrapper);
}


template <typename MmeTypes>
void DynamicMMEPatchPointGenerator<MmeTypes>:: generateDynamicPatchPointsForOperand(
        const MmeNode&                node,
        const TensorPtr&              tensor,
        MmeCommon::EMmeOperand        op,
        const MmeDescriptorGenerator& descGenerator,
        DescriptorWrapper<MmeDesc>&   descWrapper,
        unsigned                      engineIdx)
{
    MmeTensorDesc* tensorDesc =
        descGenerator.template mmeGetTensorDescriptor<MmeTensorDesc>(op, descWrapper.getDescriptor());
    if (tensorDesc == nullptr)
    {
        // No operand to patch
        return;
    }

    bool haveTile = false;

    auto tensorTile = getTensorTileFromEngine(node, tensor, engineIdx, haveTile);

    for (unsigned d = 0; d < tensor->getDim(); d++)
    {
        if (tensor->isDynamicDim(d))
        {
            if (haveTile)
                addValidElementsPatchPoint(node, tensor, tensorDesc, d, descWrapper, tensorTile.offset[d], tensorTile.geometry[d]);
            else
                addValidElementsPatchPointNoTile(node, tensor, tensorDesc, d, descWrapper);
        }
    }
}

template <typename MmeTypes>
void DynamicMMEPatchPointGenerator<MmeTypes>::addValidElementsPatchPoint(
        const MmeNode&              node,
        const TensorPtr&            tensor,
        const MmeTensorDesc*        tensorDesc,
        int                         dim,
        DescriptorWrapper<MmeDesc>& descWrapper,
        uint64_t                    tileOffset,
        uint64_t                    tileSize)
{
    BasicFieldsContainerInfo& bfci = descWrapper.getBasicFieldsContainerInfo();
    MmeDesc&                  desc = descWrapper.getDescriptor();

    uint32_t fieldIndexOffset = ((uint64_t) & (tensorDesc->validElements[dim]) - (uint64_t)&desc) / sizeof(uint32_t);

    auto                           origin = const_cast<MmeNode&>(node).shared_from_this();
    DynamicShapeFieldInfoSharedPtr fieldInfo =
        std::make_shared<arc_platforms::DynamicMmeFieldInfo>(fieldIndexOffset, origin, bfci.getRoi(), ShapeFuncID::SMF_GAUDI3_MME_SIZE);

    mme_multi_dcore_sm_params_t metadata {0};
    metadata.dim                 = dim;
    metadata.tensor_input_index  = getTensorIndex(node.getInputs(), tensor);
    metadata.tensor_output_index = getTensorIndex(node.getOutputs(), tensor);
    metadata.dcore_roi_offset     = tileOffset;
    metadata.dcore_roi_size       = tileSize;

    // In runtime we will get the actual size in elements, and we want to update the descriptor accordingly.
    // The valid elements of a dimension is multiplied by the stride (or other in case of layout optimizations)
    // If we save the multiply factor, when we get the actual size we can just multiply by it and update the descriptor.
    metadata.multiply_factor = tensorDesc->validElements[dim] / tileSize;

    std::vector<uint8_t> serializedMetadata(sizeof(metadata));
    memcpy(serializedMetadata.data(), &metadata, sizeof(metadata));
    fieldInfo->setMetadata(serializedMetadata);

    BasicFieldInfoPair fieldInfoPair {fieldIndexOffset, fieldInfo};
    bfci.add(fieldInfoPair);
}

template <typename MmeTypes>
void DynamicMMEPatchPointGenerator<MmeTypes>::addValidElementsPatchPointNoTile(
        const MmeNode&              node,
        const TensorPtr&            tensor,
        const MmeTensorDesc*        tensorDesc,
        int                         dim,
        DescriptorWrapper<MmeDesc>& descWrapper)
{
    BasicFieldsContainerInfo& bfci = descWrapper.getBasicFieldsContainerInfo();
    MmeDesc&                  desc = descWrapper.getDescriptor();

    uint32_t fieldIndexOffset = ((uint64_t) & (tensorDesc->validElements[dim]) - (uint64_t)&desc) / sizeof(uint32_t);

    auto                           origin = const_cast<MmeNode&>(node).shared_from_this();
    DynamicShapeFieldInfoSharedPtr fieldInfo =
        std::make_shared<arc_platforms::DynamicMmeFieldInfo>(fieldIndexOffset, origin, bfci.getRoi(), ShapeFuncID::SMF_GAUDI2_MME_SIZE);

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

}  // namespace arc_platforms

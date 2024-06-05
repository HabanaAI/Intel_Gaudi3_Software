#pragma once

#include "habana_nodes.h"
#include "operation_slice.h"
#include "physical_memory_ops_nodes.h"
#include "recipe.h"

#include "defs.h"
#include "platform/gaudi/graph_compiler/smf/smf.h"
#include "utils.h"

#include "dynamic_tpc_pp_generator.h"

template <typename Desc>
void DynamicTPCPatchPointGenerator<Desc>::generatePatchPoints(const TPCNode& node)
{
    if (!node.isDynamicShape() || GCFG_DISABLE_DS_TPC_ROI_PATCHING.value())
    {
        return;
    }

    uint32_t descTensorIdx = 0;

    auto inputs = node.getInputs();

    for (const auto& in : inputs)
    {
        if (in->isAuxTensor() || in->getTensorType() == OUTPUT_DESCRIBING_SHAPE_TENSOR) continue;
        if (in->isDynamicShape())
        {
            addDynamicShapePatchPointsOneTensor(node, in, descTensorIdx, false, &in - &inputs.front());
        }
        ++descTensorIdx;
    }

    auto outputs = node.getOutputs();
    for (const auto& out : outputs)
    {
        if (out->isAuxTensor() || out->isShapeTensor()) continue;
        if (out->isDynamicShape())
        {
            addDynamicShapePatchPointsOneTensor(node, out, descTensorIdx, true, &out - &outputs.front());

        }
        ++descTensorIdx;
    }

    generateDynamicStridePatchPointsForNode(node);

    if (!node.isROIDynamic(m_wrapper.getBasicFieldsContainerInfo().getRoi()))
    {
        return;
    }

    auto nodeProjections = node.getDynamicShapeProjectionsTensors();

    // Add non-tensor (index space) patch points
    const auto& instance = node.getInstance();

    addPatchPointsForIndexSpace(node, instance, nodeProjections);

}

template <typename Desc>
void DynamicTPCPatchPointGenerator<Desc>::addDynamicShapePatchPointsOneTensor(const TPCNode& node,
                                                                              const pTensor& tensor,
                                                                              uint32_t descTensorIndex,
                                                                              bool isOutput,
                                                                              uint32_t nodeTensorIndex)
{
    BasicFieldsContainerInfo& basicFieldsContainerInfo = m_wrapper.getBasicFieldsContainerInfo();

    // When patching TPC sizes, the first pipeline level for each available engine must be patched,
    // but the next pipeline levels can be masked, as the patching will be the same.
    auto enginesInPipeline = node.getPhysicalRois()->size() / node.getLogicalRois()->size();
    auto roi               = basicFieldsContainerInfo.getRoi();
    auto deviceType        = node.getGraphTraits()->getDeviceId();
    bool maskable          = roi->pipelineLevel * enginesInPipeline + roi->engineIndex >= node.getMaxAvailableTpc(deviceType);
    auto origin            = const_cast<TPCNode&>(node).shared_from_this();

    for (uint32_t dim = 0; dim < tensor->getDim(); ++dim)
    {
        if (tensor->isDynamicDim(dim))
        {
            auto offset = tensorIndexAndDimToSizeOffset(descTensorIndex, dim);
            auto fieldInfo = std::make_shared<DynamicTPCSizeFieldInfo>(offset, origin, roi);

            if (GCFG_ENABLE_BIG_TENSOR_PP_PRUNE.value())
            {
                fieldInfo->setPatchedTensorInfo(tensor->getId(), descTensorIndex, dim);
            }

            fieldInfo->setIsMaskable(maskable);
            tpc_size_sm_params_t metadata {0};
            metadata.this_dim = dim;
            metadata.is_output = static_cast<uint32_t>(isOutput);
            metadata.tensor_index = nodeTensorIndex;
            metadata.offset = 0;
            auto tpcSlicePtr = std::dynamic_pointer_cast<OperationSlice>(origin);
            if (tpcSlicePtr)
            {
                // For TPC slices - save the offset inside the big tensor.
                // This will be used later to calc the size for patching (tpcSizeShapeManipulationFunction).
                metadata.offset = tpcSlicePtr->getTensorSliceOffsetInDim(tensor, dim);
                LOG_DEBUG(DYN_SHAPE,
                          "DynamicTPCPatchPointGenerator: TPC slice {} - "
                          "Update offset {} for tensor {} in dim {}",
                          origin->getNodeName(), metadata.offset,
                          tensor->getName(), dim);
            }

            std::vector<uint8_t> convertedMetadata(sizeof(metadata));
            memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

            fieldInfo->setMetadata(convertedMetadata);

            BasicFieldInfoPair fieldInfoPair{offset, fieldInfo};
            basicFieldsContainerInfo.add(fieldInfoPair);
        }
    }
}

template <typename Desc>
unsigned DynamicTPCPatchPointGenerator<Desc>::fillDynamicShapePatchPointIndexSpaceProjectionFromNodeProjection(const Node::NodeDynamicShapeProjection& nodeProjection,
                                                       const NodeROI *nodeROI,
                                                       const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                       uint32_t indexSpaceDim,
                                                       tpc_sm_params_t &metadata)
{
    if (nodeProjection.indexSpaceDim != indexSpaceDim)
    {
        return 0;
    }
    auto& projection      = metadata.projections[0];
    projection.is_output  = nodeProjection.isOutput;
    projection.tensor_idx = nodeProjection.tensorIdx;
    projection.tensor_dim = nodeProjection.tensorDim;
    auto& accessPattern =
        nodeProjection.isOutput ? instance.outputTensorAccessPattern : instance.inputTensorAccessPattern;
    projection.a    = accessPattern[nodeProjection.tensorIdx].mapping[nodeProjection.tensorDim].a;
    projection.size = nodeROI->size[indexSpaceDim];

    return 1;
}

template<typename Desc>
unsigned DynamicTPCPatchPointGenerator<Desc>::fillDynamicShapePatchPointIndexSpaceProjection(
    const TPCNode&                                node,
    const NodeROI*                                nodeROI,
    const tpc_lib_api::HabanaKernelInstantiation& instance,
    uint32_t                                      indexSpaceDim,
    tpc_sm_params_t&                              metadata)
{
    uint32_t projectionCount = 0;

    auto findDimTransform = [](uint32_t indexSpaceDim, const tpc_lib_api::DimIndexSpaceMapping transforms[], int rank) -> int
    {
        for (int i = 0; i < rank; ++i)
        {
            auto & trans = transforms[i];
            if (trans.indexSpaceDim != indexSpaceDim) continue;

            // check that the transformation is initialized.
            if (trans.indexSpaceDim == 0 && trans.a == 0 && trans.start_b == 0 && trans.end_b == 0) continue;

            return i;
        }
        return -1;
    };

    // Projections are mappings from a tensor dimension to an index space dimension.
    // An interval of an index space is projected to an interval of a tensor space.
    // The reverse mapping (from index space to tensor) is stored in DimIndexSpaceMapping
    // and consists of 3 numbers a, start_b, end_b.
    // When the actual tensor size becomes known (in the SMF), we calculate the new ROI size
    // that corresponds to that tensor size by the formula
    //
    //    actualROISize = maxROISize - (maxTensorSize - actualTensorSize)/a (rounding down).
    //
    // maxROIsize is stored in projection.size
    //
    // Each index space dimension may correspond to more than one tensor dimension from different tensors.
    // This is why we need projections for all tensors that reflect this index space dimension.
    // In the SMF we calculate actualROISize for every one of them and use the maximal result.

    auto fillProjection = [=](tpc_sm_params_t&                         metadata,
                              const tpc_lib_api::DimIndexSpaceMapping& transform,
                              uint32_t                                 projectionIdx,
                              uint32_t                                 tensorIdx,
                              uint32_t                                 tensorDim,
                              bool                                     isOutput) {
        auto& projection      = metadata.projections[projectionIdx];
        projection.is_output  = isOutput;
        projection.tensor_idx = tensorIdx;
        projection.tensor_dim = tensorDim;
        projection.a          = transform.a;
        projection.size       = nodeROI->size[indexSpaceDim];
    };

    auto inputs = node.getInputs();
    auto outputs = node.getOutputs();
    bool isDynamicDim = false;
    std::array<int, MAX_TENSOR_NR> inputDims;
    std::array<int, MAX_TENSOR_NR> outputDims;
    inputDims.fill(-1);
    outputDims.fill(-1);

    for (const auto& in : inputs)
    {
        if (in->isAuxTensor() || in->isShapeTensor() || in->isHost2DeviceTensor()) continue;

        auto tensorIdx = &in - &inputs.front();
        auto rank = in->getDim();
        inputDims[tensorIdx] = findDimTransform(indexSpaceDim, instance.inputTensorAccessPattern[tensorIdx].mapping, rank);
        if (inputDims[tensorIdx] >= 0 && in->isDynamicDim(inputDims[tensorIdx]))
        {
            isDynamicDim = true;
        }
    }
    for (const auto& out : outputs)
    {
        if (out->isAuxTensor() || out->isShapeTensor() || out->isHost2DeviceTensor()) continue;

        auto tensorIdx = &out - &outputs.front();
        auto rank = out->getDim();
        outputDims[tensorIdx] = findDimTransform(indexSpaceDim, instance.outputTensorAccessPattern[tensorIdx].mapping, rank);
        if (outputDims[tensorIdx] >= 0 && out->isDynamicDim(outputDims[tensorIdx]))
        {
            isDynamicDim = true;
        }
    }

    if (!isDynamicDim)  return 0;

    for (size_t tensorIdx = 0; tensorIdx < inputs.size(); tensorIdx++)
    {
        if (inputDims[tensorIdx] < 0) continue; // this tensor has no dimension that corresponds to this indexSpaceDim
        if (instance.inputTensorAccessPattern[tensorIdx].allRequired) continue;
        fillProjection(metadata, instance.inputTensorAccessPattern[tensorIdx].mapping[inputDims[tensorIdx]],
                        projectionCount, tensorIdx, inputDims[tensorIdx], false);
        LOG_TRACE(DYN_SHAPE, "Added input projection for node {}, input{}", node.getNodeName(), tensorIdx);
        ++projectionCount;
    }

    for (size_t tensorIdx = 0; tensorIdx < outputs.size(); tensorIdx++)
    {
        if (outputDims[tensorIdx] < 0 ) continue; // this tensor has no dimension that corresponds to this indexSpaceDim
        if (instance.outputTensorAccessPattern[tensorIdx].allRequired) continue;
        fillProjection(metadata, instance.outputTensorAccessPattern[tensorIdx].mapping[outputDims[tensorIdx]],
                        projectionCount, tensorIdx, outputDims[tensorIdx], true);
        LOG_TRACE(DYN_SHAPE, "Added output projection for node {}, output{}", node.getNodeName(), tensorIdx);
        ++projectionCount;
    }

    return projectionCount;
}

#include "gaudi_dynamic_tpc_pp_generator.h"
#include "dynamic_tpc_pp_generator.inl"

namespace gaudi
{

class DynamicTPCIndexSpaceFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicTPCIndexSpaceFieldInfo(uint32_t fieldIndexOffset, pNode origin, NodeROI* roi)
    : DynamicShapeFieldInfo(fieldIndexOffset,
                            FieldType::FIELD_DYNAMIC_TPC_TID,
                            ShapeFuncID::SMF_TPC_INDEX_SPACE,
                            std::move(origin),
                            roi)
    {
        m_size = 1;
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicTPCIndexSpaceFieldInfo>(*this); };
};

size_t DynamicTPCPatchPointGenerator::tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim)
{
    auto tensorOffset = offsetof(gaudi::TpcDesc, m_tensors) + sizeof(block_tpc_tensor) * tensorIndex;
#define SIZE_OFFSET(_D) (tensorOffset + offsetof(block_tpc_tensor, dim_##_D##_size)) / sizeof(uint32_t)
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
    }
    HB_ASSERT(false, "Invalid dimension number for TPC dynamic shape patch");
    return 0;
#undef SIZE_OFFSET
}

size_t DynamicTPCPatchPointGenerator::tensorIndexAndDimToStrideOffset(uint32_t tensorIndex, uint32_t dim)
{
    HB_ASSERT(false, "TPC strides should not be patched in Gaudi");
    return 0;
}

size_t DynamicTPCPatchPointGenerator::tidToOffset(uint32_t tid)
{
    auto nonTensorOffset = offsetof(gaudi::TpcDesc, m_desc);
#define TID_OFFSET(_D) (nonTensorOffset + offsetof(block_tpc_non_tensor_descriptor, tid_size_dim_##_D)) / sizeof(uint32_t)
    switch (tid)
    {
        case 0:
            return TID_OFFSET(0);
        case 1:
            return TID_OFFSET(1);
        case 2:
            return TID_OFFSET(2);
        case 3:
            return TID_OFFSET(3);
        case 4:
            return TID_OFFSET(4);
    }
    HB_ASSERT(false, "Invalid dimension number for TPC dynamic shape patch");
    return 0;
#undef TID_OFFSET
}

void DynamicTPCPatchPointGenerator::addPatchPointsForIndexSpace(const TPCNode& node,
        const tpc_lib_api::HabanaKernelInstantiation& instance,
        const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections)
{
    if (nodeProjections.size() > 0)
    {
        for (const auto& prj: nodeProjections)
        {
            Settable<Node::NodeDynamicShapeProjection> nodeProjection;
            nodeProjection.set(prj);
            addDynamicShapePatchPointIndexSpace(nodeProjection, node, instance, prj.indexSpaceDim);
        }
    }
    else
    {
        for (uint32_t idx = 0; idx < instance.indexSpaceRank; ++idx)
        {
            Settable<Node::NodeDynamicShapeProjection> nodeProjection;
            addDynamicShapePatchPointIndexSpace(nodeProjection, node, instance, idx);
        }
    }
}

void DynamicTPCPatchPointGenerator::addDynamicShapePatchPointIndexSpace(Settable<Node::NodeDynamicShapeProjection> &nodeProjection,
                                                                        const TPCNode& node,
                                                                        const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                                        uint32_t indexSpaceDim)
{
    if (GCFG_DISABLE_DS_TPC_INDEX_SPACE_PATCHING.value())
    {
        LOG_TRACE(DYN_SHAPE, "TPC index space patching was disabeld by config. Skipping node: {}.", node.getNodeName());
        return;
    }

    tpc_sm_params_t metadata {0};
    metadata.this_dim = indexSpaceDim;

    BasicFieldsContainerInfo& basicFieldsContainerInfo = getIndexSpaceBFCI();
    auto origin = const_cast<TPCNode&>(node).shared_from_this();

    const auto& nodeROI = basicFieldsContainerInfo.getRoi();

    uint32_t projectionCount = 0;
    if (nodeProjection.is_set())
    {
        projectionCount = fillDynamicShapePatchPointIndexSpaceProjectionFromNodeProjection(
            nodeProjection.value(),
            nodeROI,
            instance,
            indexSpaceDim,
            metadata);
    }
    else
    {
        projectionCount = fillDynamicShapePatchPointIndexSpaceProjection(
            node,
            nodeROI,
            instance,
            indexSpaceDim,
            metadata);
    }

    if (projectionCount == 0)
    {
        // there's nothing to patch
        return;
    }

    metadata.num_projections = projectionCount;

    auto offset = tidToOffset(indexSpaceDim);
    auto fieldInfo = std::make_shared<DynamicTPCIndexSpaceFieldInfo>(
            offset,
            origin,
            nodeROI);
    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    BasicFieldInfoPair fieldInfoPair{offset, fieldInfo};
    basicFieldsContainerInfo.add(fieldInfoPair);
};

}  // namespace gaudi

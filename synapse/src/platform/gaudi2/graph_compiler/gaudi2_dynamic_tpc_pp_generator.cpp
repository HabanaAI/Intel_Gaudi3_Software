#include "gaudi2_dynamic_tpc_pp_generator.h"
#include "dynamic_tpc_pp_generator.inl"


namespace gaudi2
{
size_t DynamicTPCPatchPointGenerator::tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim)
{
    auto tensorOffset = offsetof(TpcDesc, m_tensors) + sizeof(block_tpc_tensor) * tensorIndex;
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
    HB_ASSERT(false, "TPC strides should not be patched in Gaudi2");
    return 0;
}


void DynamicTPCPatchPointGenerator::addPatchPointsForIndexSpace(const TPCNode& node,
        const tpc_lib_api::HabanaKernelInstantiation& instance,
        const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections)
{
    // One patch point to rule them all

    LOG_TRACE_DYNAMIC_PATCHING("GENERATING: Num node projections {}", nodeProjections.size());

    auto origin = const_cast<TPCNode&>(node).shared_from_this();
    BasicFieldsContainerInfo& basicFieldsContainerInfo = getIndexSpaceBFCI();
    const auto& wrapper = getWrapper();
    const auto& fwCtx = wrapper.getFwCtx();

    tpc_sm_params_gaudi2_t metadata;
    metadata.m_indexSpaceCopy = fwCtx.ist;
    metadata.m_dimensions_mask = 0;

    const auto& nodeROI = basicFieldsContainerInfo.getRoi();

    if (nodeProjections.size() > 0)
    {
        for (const auto& prj: nodeProjections)
        {
            int projectionCount = fillDynamicShapePatchPointIndexSpaceProjectionFromNodeProjection(
                    prj,
                    nodeROI,
                    instance,
                    prj.indexSpaceDim,
                    metadata.m_dimensions[prj.indexSpaceDim]);
            if (projectionCount > 0)
            {
                metadata.m_dimensions_mask |= (1 << prj.indexSpaceDim);
                metadata.m_dimensions[prj.indexSpaceDim].num_projections = projectionCount;
                metadata.m_dimensions[prj.indexSpaceDim].this_dim = prj.indexSpaceDim;
            }
        }
    }
    else
    {
        for (uint32_t idx = 0; idx < instance.indexSpaceRank; ++idx)
        {
            Settable<Node::NodeDynamicShapeProjection> nodeProjection;
            int projectionCount = fillDynamicShapePatchPointIndexSpaceProjection(
                    node,
                    nodeROI,
                    instance,
                    idx,
                    metadata.m_dimensions[idx]);
            if (projectionCount > 0)
            {
                metadata.m_dimensions_mask |= (1 << idx);
                metadata.m_dimensions[idx].num_projections = projectionCount;
                metadata.m_dimensions[idx].this_dim = idx;
            }
        }
    }

    if (metadata.m_dimensions_mask == 0)
    {
        // nothing to patch!
        return;
    }

    auto fieldInfo = std::make_shared<DynamicTPCIndexSpaceFieldInfoGaudi2>(
            origin,
            nodeROI);

    std::vector<uint8_t> convertedMetadata(sizeof(metadata));
    memcpy(convertedMetadata.data(), &metadata, sizeof(metadata));

    fieldInfo->setMetadata(convertedMetadata);

    BasicFieldInfoPair fieldInfoPair{0, fieldInfo};
    basicFieldsContainerInfo.add(fieldInfoPair);
}

}  // namespace gaudi2

#pragma once

#include "gaudi2_types.h"
#include <dynamic_tpc_pp_generator.h>
#include "gaudi2_tpc_tid_metadata.h"

namespace gaudi2
{
class DynamicTPCPatchPointGenerator : public ::DynamicTPCPatchPointGenerator<gaudi2::TpcDesc>
{
public:
    using ::DynamicTPCPatchPointGenerator<gaudi2::TpcDesc>::DynamicTPCPatchPointGenerator;

private:
    virtual size_t tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim) override;
    virtual size_t tensorIndexAndDimToStrideOffset(uint32_t tensorIndex, uint32_t dim) override;

    virtual BasicFieldsContainerInfo& getIndexSpaceBFCI() override
    {
        return getWrapper().getBasicFieldsContainerInfoForCtx();
    }

    virtual void addPatchPointsForIndexSpace(const TPCNode&                                       node,
                                             const tpc_lib_api::HabanaKernelInstantiation&        instance,
                                             const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections) override;
};

class DynamicTPCIndexSpaceFieldInfoGaudi2 : public DynamicShapeFieldInfo
{
public:
    DynamicTPCIndexSpaceFieldInfoGaudi2(pNode origin, NodeROI* roi)
    : DynamicShapeFieldInfo(0,
                            FieldType::FIELD_DYNAMIC_TPC_TID_GAUDI2,
                            ShapeFuncID::SMF_TPC_INDEX_SPACE_GAUDI2,
                            std::move(origin), roi)
    {
        m_size = sizeof(tpc_wd_ctxt_t) / sizeof(uint32_t);
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicTPCIndexSpaceFieldInfoGaudi2>(*this); };
};

}  // namespace gaudi2

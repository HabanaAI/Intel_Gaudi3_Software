#pragma once

#include "gaudi_types.h"
#include <dynamic_tpc_pp_generator.h>

namespace gaudi
{
class DynamicTPCPatchPointGenerator : public ::DynamicTPCPatchPointGenerator<gaudi::TpcDesc>
{
public:
    using ::DynamicTPCPatchPointGenerator<gaudi::TpcDesc>::DynamicTPCPatchPointGenerator;

private:
    virtual size_t tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim) override;
    virtual size_t tensorIndexAndDimToStrideOffset(uint32_t tensorIndex, uint32_t dim) override;

    size_t tidToOffset(uint32_t tid);

    virtual BasicFieldsContainerInfo& getIndexSpaceBFCI() override
    {
        return getWrapper().getBasicFieldsContainerInfo();
    }

    void addPatchPointsForIndexSpace(const TPCNode&                                       node,
                                     const tpc_lib_api::HabanaKernelInstantiation&        instance,
                                     const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections) override;

    void addDynamicShapePatchPointIndexSpace(Settable<Node::NodeDynamicShapeProjection>&    nodeProjection,
                                             const TPCNode&                                 node,
                                             const tpc_lib_api::HabanaKernelInstantiation&  instance,
                                             uint32_t                                       indexSpaceDim);
};
}  // namespace gaudi

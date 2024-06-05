#pragma once

#include "address_fields_container_info.h"
#include "gaudi_types.h"
#include "patch_point_generator.h"
#include "patch_point_generator.inl"

#include "physical_memory_ops_nodes.h"

#include "include/gaudi/mme_descriptor_generator.h"

#include "recipe_metadata.h"

namespace gaudi
{
class GaudiDMAPatchPointGenerator : public DMAPatchPointGenerator<gaudi::DmaDesc>
{
public:

    using DescWrapper = DescriptorWrapper<gaudi::DmaDesc>;
    GaudiDMAPatchPointGenerator() : DMAPatchPointGenerator<gaudi::DmaDesc>() {}

    void generateDmaPatchPoints (const DMANode& node, DescWrapper& descWrapper) override;
};

class GaudiTPCPatchPointGenerator : public TPCPatchPointGenerator<gaudi::TpcDesc>
{
public:
    GaudiTPCPatchPointGenerator() : TPCPatchPointGenerator<gaudi::TpcDesc>()     {}

    void generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<gaudi::TpcDesc>& descWrapper) override;

protected:
    uint32_t* getBaseAddrHigh(gaudi::TpcDesc& desc) override;
    uint32_t* getBaseAddrLow(gaudi::TpcDesc& desc) override;

private:
    static unsigned FillDynamicShapePatchPointIndexSpaceProjectionFromNodeProjection(
        Node::NodeDynamicShapeProjection&             nodeProjection,
        const NodeROI*                                nodeROI,
        const tpc_lib_api::HabanaKernelInstantiation& instance,
        uint32_t                                      indexSpaceDim,
        tpc_sm_params_t&                              metadata);
    static unsigned
    FillDynamicShapePatchPointIndexSpaceProjection(const TPCNode&                                node,
                                                   const NodeROI*                                nodeROI,
                                                   const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                   uint32_t                                      indexSpaceDim,
                                                   tpc_sm_params_t&                              metadata);
};

class GaudiMMEPatchPointGenerator : public MMEPatchPointGenerator<gaudi::MmeDesc>
{
public:
    GaudiMMEPatchPointGenerator() : MMEPatchPointGenerator<gaudi::MmeDesc>()     {}

    void generateMmePatchPoints(const MmeNode& node, DescriptorWrapper<gaudi::MmeDesc>& descWrapper);

private:
    void generatePatchPoint(const pTensor&                     tensor,
                            MmeCommon::EMmeOperand             op,
                            DescriptorWrapper<gaudi::MmeDesc>& descWrapper,
                            const MmeNode&                     node);
};
}

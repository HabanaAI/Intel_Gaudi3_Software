#pragma once

#include "gaudi3_types.h"
#include <dynamic_tpc_pp_generator.h>
#include "gaudi3_tpc_tid_metadata.h"

namespace gaudi3
{
class DynamicTPCPatchPointGenerator : public ::DynamicTPCPatchPointGenerator<gaudi3::TpcDesc>
{
public:
    using ::DynamicTPCPatchPointGenerator<gaudi3::TpcDesc>::DynamicTPCPatchPointGenerator;

private:
    void addDynamicStridesPatchPoints(const TPCPhysicalMemoryOpNode& node,
                                      const pTensor&       tensor,
                                      uint32_t             descTensorIndex,
                                      bool                 isOutput,
                                      uint32_t             nodeTensorIndex);

    void addDynamicSlicePatchPoints(const TPCPhysicalMemoryOpNode& node,
                                    const pTensor&       tensor,
                                    uint32_t             descTensorIndex,
                                    bool                 isOutput,
                                    uint32_t             nodeTensorIndex);

    void addDynamicViewPatchPoints(const TPCPhysicalMemoryOpNode& node,
                                   const pTensor&       tensor,
                                   uint32_t             descTensorIndex,
                                   bool                 isOutput,
                                   uint32_t             nodeTensorIndex);

    void addDynamicBasePatchPoints(const TPCPhysicalMemoryOpNode& node,
                                   const pTensor&       tensor,
                                   uint32_t             descTensorIndex,
                                   bool                 isOutput,
                                   uint32_t             nodeTensorIndex);

    void insertSliceStridePatchPoint(const TPCPhysicalMemoryOpNode& node, unsigned dim);
    void insertSliceBaseAddressPatchPoint(const TPCPhysicalMemoryOpNode& node);

    void insertViewStridePatchPoint(const TPCPhysicalMemoryOpNode& node, unsigned dim);
    void insertViewBaseAddressPatchPoint(const TPCPhysicalMemoryOpNode& node);

    virtual size_t tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim) override;
    virtual size_t tensorIndexAndDimToStrideOffset(uint32_t tensorIndex, uint32_t dim) override;
    size_t tensorIndexToBaseAddressOffset(uint32_t tensorIndex);
    size_t getTensorBaseAddress(uint32_t tensorIndex);

    virtual BasicFieldsContainerInfo& getIndexSpaceBFCI() override
    {
        return getWrapper().getBasicFieldsContainerInfoForCtx();
    }

    virtual void generateDynamicStridePatchPointsForNode(const TPCNode&) override;
    void generateDynamicStridePatchPointsForTensor(const TPCPhysicalMemoryOpNode& node,
                                                   const pTensor& tensor,
                                                   uint32_t       descTensorIndex,
                                                   bool           isOutput,
                                                   uint32_t       nodeTensorIndex);
    void addPatchPointsForIndexSpace(const TPCNode&                                       node,
                                     const tpc_lib_api::HabanaKernelInstantiation&        instance,
                                     const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections) override;
};

class DynamicTPCIndexSpaceFieldInfoGaudi3 : public DynamicShapeFieldInfo
{
public:
    DynamicTPCIndexSpaceFieldInfoGaudi3(pNode origin, NodeROI* roi)
    : DynamicShapeFieldInfo(0,
                            FieldType::FIELD_DYNAMIC_TPC_TID_GAUDI3,
                            ShapeFuncID::SMF_TPC_INDEX_SPACE_GAUDI3,
                            std::move(origin), roi)
    {
        // Cannot
        m_size = sizeof(tpc_wd_ctxt_t)/sizeof(uint32_t);
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicTPCIndexSpaceFieldInfoGaudi3>(*this); };
};
}  // namespace gaudi3

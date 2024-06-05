
#pragma once
#include "descriptor_wrapper.h"
#include "graph_compiler/address_fields_container_info.h"
#include "physical_memory_ops_nodes.h"
#include "recipe_metadata.h"

template<typename Desc>
class DynamicTPCPatchPointGenerator
{
public:
    using BlockT = typename DescriptorWrapper<Desc>::BlockT;
    DynamicTPCPatchPointGenerator(DescriptorWrapper<Desc>& wrapper) : m_wrapper(wrapper) {}

    virtual void generatePatchPoints(const TPCNode& node);

    DescriptorWrapper<Desc>& getWrapper() { return m_wrapper; }

protected:
    virtual void addDynamicShapePatchPointsOneTensor(const TPCNode& node,
                                                     const pTensor& tensor,
                                                     uint32_t       descTensorIndex,
                                                     bool           isOutput,
                                                     uint32_t       nodeTensorIndex);

    DescriptorWrapper<Desc>& m_wrapper;

    virtual unsigned fillDynamicShapePatchPointIndexSpaceProjectionFromNodeProjection(
        const Node::NodeDynamicShapeProjection&       nodeProjection,
        const NodeROI*                                nodeROI,
        const tpc_lib_api::HabanaKernelInstantiation& instance,
        uint32_t                                      indexSpaceDim,
        tpc_sm_params_t&                              metadata);

    virtual unsigned
    fillDynamicShapePatchPointIndexSpaceProjection(const TPCNode&                                node,
                                                   const NodeROI*                                nodeROI,
                                                   const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                   uint32_t                                      indexSpaceDim,
                                                   tpc_sm_params_t&                              metadata);

    virtual void addPatchPointsForIndexSpace(const TPCNode&                                       node,
                                             const tpc_lib_api::HabanaKernelInstantiation&        instance,
                                             const std::vector<Node::NodeDynamicShapeProjection>& nodeProjections) = 0;

    virtual size_t tensorIndexAndDimToSizeOffset(uint32_t tensorIndex, uint32_t dim)   = 0;
    virtual size_t tensorIndexAndDimToStrideOffset(uint32_t tensorIndex, uint32_t dim) = 0;

    virtual BasicFieldsContainerInfo& getIndexSpaceBFCI() = 0;

    virtual void generateDynamicStridePatchPointsForNode(const TPCNode&) {}  // no op for most archs
};

class DynamicTPCSizeFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicTPCSizeFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi)
    : DynamicShapeFieldInfo(fieldIndexOffset,
                            FieldType::FIELD_DYNAMIC_TPC_SIZE,
                            ShapeFuncID::SMF_TPC_SIZE,
                            std::move(origin),
                            roi)
    {
        m_size = 1;
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicTPCSizeFieldInfo>(*this); }

    void setIsMaskable(bool isMaskable) { m_isMaskable = isMaskable; }

    void setPatchedTensorInfo(unsigned tensorId, unsigned operandIndex, unsigned dim)
    {
        m_hasPatchedTensorInfo             = true;
        m_patchedTensorInfo.m_tensorId     = tensorId;
        m_patchedTensorInfo.m_operandIndex = operandIndex;
        m_patchedTensorInfo.m_dim          = dim;
    };

    const PatchedTensorInfo* getPatchedTensorInfo() const override
    {
        return m_hasPatchedTensorInfo ? &m_patchedTensorInfo : nullptr;
    }

private:
    bool              m_hasPatchedTensorInfo = false;
    PatchedTensorInfo m_patchedTensorInfo;
};

class DynamicTPCStrideFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicTPCStrideFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi, ShapeFuncID smf)
    : DynamicShapeFieldInfo(fieldIndexOffset, FieldType::FIELD_DYNAMIC_TPC_STRIDE, smf, std::move(origin), roi)
    {
        m_size = 1;
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicTPCStrideFieldInfo>(*this); }
};

class DynamicTPCIndexSpaceFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicTPCIndexSpaceFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi, ShapeFuncID smf)
    : DynamicShapeFieldInfo(fieldIndexOffset, FieldType::FIELD_DYNAMIC_TPC_TID, smf, std::move(origin), roi)
    {
        m_size = 1;
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicTPCIndexSpaceFieldInfo>(*this); };
};

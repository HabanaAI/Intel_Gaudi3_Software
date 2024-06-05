#pragma once

#include "descriptor_wrapper.h"
#include "graph_compiler/address_fields_container_info.h"
#include "physical_memory_ops_nodes.h"

#include "recipe_metadata.h"

template<typename Desc>
class DynamicDMAPatchPointGenerator
{
public:
    using BlockT = typename DescriptorWrapper<Desc>::BlockT;
    DynamicDMAPatchPointGenerator(DescriptorWrapper<Desc>& wrapper) : m_wrapper(wrapper) {}

    virtual void generatePatchPoints(const DMANode& node);

    virtual void addDynamicShapePatchPoint(const DMANode& node, FieldType fieldType);

    virtual void addSizePatchPoint(const DMANode& node, const TensorPtr& tensor, FieldType fieldType, unsigned dim);

protected:
    DescriptorWrapper<Desc>& wrapper() { return m_wrapper; };

    virtual void generateSerializationPatchPoints(const DMANode& node);

    virtual void generateViewPatchPoints(const DMANode& dmaNode);
    virtual void generateSlicePatchPoints(const DMANode& dmaNode);

    virtual void insertBulkSizeStridePatchPoint(const DMAPhysicalMemoryOpNode& node,
                                                FieldType                   fieldType,
                                                unsigned                    dimsToUpdate,
                                                unsigned                    firstDynamicDim);
    virtual void generateDMADynamicAddressPatchPoints(const DMANode& node);

    virtual uint64_t getAddressPtrForPhysicalMemOp(const DMAPhysicalMemoryOpNode& node) = 0;

    virtual void insertLastStridePatchPoint(const DMAPhysicalMemoryOpNode& node, FieldType fieldType);
    virtual void insertBaseAddressPatchPoint(const DMAPhysicalMemoryOpNode& node, ShapeFuncID smf);
    virtual void insertViewBaseAddressPatchPoint(const DMAPhysicalMemoryOpNode& node);
    virtual void insertViewStridePatchPoint(const DMAPhysicalMemoryOpNode& node, unsigned dim);
    virtual void insertSliceBaseAddressPatchPoint(const DMAPhysicalMemoryOpNode& node);
    virtual void insertSliceStridePatchPoint(const DMAPhysicalMemoryOpNode& node, unsigned dim);

    // functions to override in platform specific descendants
    virtual void addDynamicExecutionPatchPoints(const DMANode& node) = 0;

    virtual DynamicShapeFieldInfoSharedPtr getDynamicAddressInfo(const DMAPhysicalMemoryOpNode& node, ShapeFuncID smf) = 0;

    virtual uint32_t fieldTypeAndDimToOffset(FieldType fieldType, uint32_t dim) = 0;
    virtual std::pair<uint32_t, BlockT> fieldTypeAndDimToOffsetAndBlock(FieldType fieldType, uint32_t dim) = 0;

    virtual uint32_t addressOffsetHi(bool isSrc) = 0;
    virtual uint32_t addressOffsetLo(bool isSrc) = 0;
    virtual uint32_t addressValueHi(bool isSrc)  = 0;
    virtual uint32_t addressValueLo(bool isSrc)  = 0;

private:
    DescriptorWrapper<Desc>& m_wrapper;
};

class DynamicDmaFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicDmaFieldInfo(uint32_t fieldIndexOffset, FieldType fieldType, pNode origin, NodeROI* roi)
    : DynamicShapeFieldInfo(fieldIndexOffset, fieldType, ShapeFuncID::SMF_DMA_SIZE, std::move(origin), roi)
    {
        m_size = 1;
    }
    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicDmaFieldInfo>(*this); }
};

class PatchManyStridesFieldInfo : public DynamicShapeFieldInfo
{
public:
    PatchManyStridesFieldInfo(uint32_t  fieldIndexOffset,
                              FieldType fieldType,
                              size_t    affectedFieldCount,
                              pNode     origin,
                              NodeROI*  roi)
    : DynamicShapeFieldInfo(fieldIndexOffset, fieldType, ShapeFuncID::SMF_MANY_STRIDES, std::move(origin), roi)
    {
        m_size = affectedFieldCount;
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<PatchManyStridesFieldInfo>(*this); }
};

class PatchLastStride : public DynamicShapeFieldInfo
{
public:
    PatchLastStride(uint32_t fieldIndexOffset, FieldType fieldType, pNode origin, NodeROI* roi)
    : DynamicShapeFieldInfo(fieldIndexOffset, fieldType, ShapeFuncID::SMF_LAST_STRIDE, std::move(origin), roi)
    {
        m_size = 1;
    }
    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<PatchLastStride>(*this); }
};

class PatchSingleStrideFieldInfo : public DynamicShapeFieldInfo
{
public:
    PatchSingleStrideFieldInfo(uint32_t    fieldIndexOffset,
                               FieldType   fieldType,
                               ShapeFuncID smf,
                               pNode       origin,
                               NodeROI*    roi)
    : DynamicShapeFieldInfo(fieldIndexOffset, fieldType, smf, std::move(origin), roi)
    {
        m_size = 1;
    }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<PatchSingleStrideFieldInfo>(*this); }
};

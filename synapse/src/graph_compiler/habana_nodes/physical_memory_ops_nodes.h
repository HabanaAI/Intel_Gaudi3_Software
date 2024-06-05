#pragma once

#include "dma_memcopy_node.h"
#include "habana_nodes.h"
#include "node_visitor.h"
#include "physical_memory_ops_nodes.h"

class NodeWithParentInfo
{
public:
    virtual void setParentInfo(uint64_t size) = 0;
};


template <class BASE>
class PhysicalMemoryOpNode : public BASE, public NodeWithParentInfo
{
public:
    virtual bool isDynamicMemoryOp() const override { return true; }

    virtual bool isSrcDynamicStrided() const = 0;
    virtual bool isDstDynamicStrided() const = 0;

    // use for calculating linear ranges - since we do not know the strides until runtime,
    // we must assume that every activation reads/writes from/to the entire parent tensor
    virtual void calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const;

    void setParentInfo(uint64_t size) override { m_realParentSize = size; }

    uint64_t getRealParentSize() const { return m_realParentSize; }

protected:
    PhysicalMemoryOpNode(const TensorVector& in,
                         const TensorVector& out,
                         std::string_view    name,
                         ShapeFuncID         sifId = SIF_DMA_MEMCPY)
    : BASE(in, out, name), m_realParentSize(in[0]->getDenseSizeInBytes()) {};

    void fixLinearRangesToRealParentStart(TensorROI& tRoi) const;

    void applyFullViewLinearRange(TensorROI& tRoi) const;

    uint64_t m_realParentSize;  // for out-of-bound validations during patching
};

using DMAPhysicalMemoryOpNode = PhysicalMemoryOpNode<DMAMemcpyNode>;
using TPCPhysicalMemoryOpNode = PhysicalMemoryOpNode<TPCMemcpyNode>;

template <class BASE>
class SerializeNode : public PhysicalMemoryOpNode<BASE>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr             clone() const override;
    virtual bool                validateNode() const override;
    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const override { return DMA_OP_SERIALIZE; }
    std::string                 getNodeTypeStr() const override;

    virtual bool canHandleStridedInput(synDeviceType device = synDeviceTypeInvalid) const override { return false; }

    virtual bool isSrcDynamicStrided() const override { return false; }
    virtual bool isDstDynamicStrided() const override { return true; }

    virtual void calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    SerializeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);
};

// Template implementation and explicit instantiation in serialize_node.cpp

template <class BASE>
class DeserializeNode : public PhysicalMemoryOpNode<BASE>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr             clone() const override;
    virtual bool                validateNode() const override;
    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const override { return DMA_OP_DESERIALIZE; }
    std::string                 getNodeTypeStr() const override;
    virtual bool canHandleStridedInput(synDeviceType device = synDeviceTypeInvalid) const override { return false; }

    virtual bool isSrcDynamicStrided() const override { return true; }
    virtual bool isDstDynamicStrided() const override { return false; }

    virtual void calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    DeserializeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);
};

// Template implementation and explicit instantiations in deserialize_node.cpp

template <class BASE>
class DynamicStridedMemcpyNode : public PhysicalMemoryOpNode<BASE>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr             clone() const override;
    virtual bool                validateNode() const override;
    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const override { return DMA_OP_DYNAMIC_STRIDE; }
    std::string                 getNodeTypeStr() const override;

    virtual bool isSrcDynamicStrided() const override { return m_isSrc; }
    virtual bool isDstDynamicStrided() const override { return !m_isSrc; }
    virtual bool isDynamicShape() const override;

    static bool isDynamicStridedDmaNode(const NodePtr& n, bool isSrc);

    virtual void calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;
    virtual bool canHaveAdditionalInputs() const override { return true; }

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    DynamicStridedMemcpyNode(const TensorVector& inputs,
                          const TensorVector& outputs,
                          std::string_view    name,
                          UserParams          params);

    // if true, the dynamic strides/offset refer to the Source operand (read). otherwise they refer to the Destination
    // operand (write)
    bool m_isSrc;
};

using DynamicStridedDMAMemcpyNode = DynamicStridedMemcpyNode<DMAMemcpyNode>;
using DynamicStridedTPCMemcpyNode = DynamicStridedMemcpyNode<TPCMemcpyNode>;

// Template implementation and explicit instantiations in dynamic_stride_node.cpp
//
class PhysicalFlattenNode : public MultiNode
{
public:
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
    using BaseClass = MultiNode;

    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;
    virtual NodeList extract() override;
    virtual NodeList extract(const HabanaGraph&, MultiNode::MultiNodeDependencies& deps) override;
    virtual NodePtr  clone() const override;
    virtual void     setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    SifNodeParams getShapeInferenceFunctionUserParams() override;
    size_t        getShapeInferenceFunctionUserParamsSize() const override;

private:
    synFlattenParams m_flattenParams;

    PhysicalFlattenNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        UserParams          flattenParams,
                        std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

struct DynamicSliceMemcpyNodeBase
{
    enum SliceDmaInputs
    {
        INPUT_TENSOR = 0,
        STEPS_TENSOR,
        STARTS_TENSOR,
        MAX_NUM_INPUTS
    };

};

template <class BASE>
class DynamicSliceMemcpyNode : public PhysicalMemoryOpNode<BASE>,
      DynamicSliceMemcpyNodeBase
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr             clone() const override;
    virtual bool                validateNode() const override;
    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const override { return DMA_OP_DYNAMIC_SLICE; }
    std::string                 getNodeTypeStr() const override;

    virtual bool isSrcDynamicStrided() const override { return m_isSrc; }
    virtual bool isDstDynamicStrided() const override { return !m_isSrc; }

    virtual void calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    DynamicSliceMemcpyNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        std::string_view    name,
                        UserParams          params);

    static bool isOutOfOriginalLinearRanges(const TensorPtr& starts, const TensorPtr& steps);

    // if true, the dynamic strides/offset refer to the Source operand (read). otherwise they refer to the Destination
    // operand (write)
    bool m_isSrc;
};

// Template implementation and explicit instantiations in dynamic_slice_node.cpp

using DynamicSliceDMAMemcpyNode = DynamicSliceMemcpyNode<DMAMemcpyNode>;
using DynamicSliceTPCMemcpyNode = DynamicSliceMemcpyNode<TPCMemcpyNode>;

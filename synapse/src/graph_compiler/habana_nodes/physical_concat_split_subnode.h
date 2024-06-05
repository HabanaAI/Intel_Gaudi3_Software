#pragma once

#include "habana_nodes.h"
#include "physical_memory_ops_nodes.h"
#include "node_visitor.h"
#include "types.h"

template <class BASE>
class PhysicalConcatSplitSubnode : public PhysicalMemoryOpNode<BASE>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr  clone()        const override;
    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool     RunOnCpu ()  override;

    virtual bool isSrcDynamicStrided() const override { return m_isSplit; }
    virtual bool isDstDynamicStrided() const override { return !m_isSplit; }
    virtual void calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const override;

    virtual bool     canHaveAdditionalInputs () const override { return true; }

    unsigned concatSplitDim() const { return m_concatSplitDim; }
    unsigned nodeNumberInConcatSplit() const { return m_nodeNumberInConcatSplit; }

    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const override { return DMA_OP_DYNAMIC_BASE; }
    virtual void                setParams(UserParams userParams, unsigned userParamsSize) override;

    virtual std::vector<Node::NodeDynamicShapeProjection> getDynamicShapeProjectionsTensors() const override;
protected:
    unsigned m_concatSplitDim;
    unsigned m_nodeNumberInConcatSplit;
    bool     m_isSplit;

    PhysicalConcatSplitSubnode(const TensorVector& in,
                               const TensorVector& out,
                               std::string_view    name,
                               UserParams          params);

};

// DMA nodes have `isLinearDma` virtual function but TPC nodes do not.
// Hence we cannot have isLinearDma () const override in the base template.

class PhysicalConcatSplitSubnodeDMA : public PhysicalConcatSplitSubnode<DMAMemcpyNode>
{
private:
    using PhysicalConcatSplitSubnode<DMAMemcpyNode>::PhysicalConcatSplitSubnode;
public:
    virtual bool isLinearDma () const override {  return false; }
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
    virtual NodePtr  clone()        const override;
};

class PhysicalConcatSplitSubnodeTPC : public PhysicalConcatSplitSubnode<TPCMemcpyNode>
{
private:
    using PhysicalConcatSplitSubnode<TPCMemcpyNode>::PhysicalConcatSplitSubnode;
public:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
    virtual NodePtr  clone()        const override;

};

#pragma once

#include "habana_nodes.h"
#include "node_visitor.h"
#include "dma_node.h"

class DMAMemcpyNode : public DMANode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    static const unsigned MAX_SUPPORTED_DIM = 5;
    virtual NodePtr       clone() const override;
    virtual bool          validateNode() const override;
    virtual bool          validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool          isMemset() const override;
    virtual bool          isLinearDma() const override;
    virtual bool          isNode64BitCompatible() const override;
    bool isCreatedFromSemanticMemcpy() const { return m_isCreatedFromSemanticMemcpy; }
    void setIsCreatedFromSemanticMemcpy() { m_isCreatedFromSemanticMemcpy = true; }

    virtual DMA_OP_TYPE getOpType() const override;
    std::string         getNodeTypeStr() const override { return "DmaMemcpy"; }

    virtual bool RunOnCpu() override;

protected:
    DMAMemcpyNode(const TensorVector& in,
                  const TensorVector& out,
                  std::string_view    name,
                  ShapeFuncID         sifId = SIF_DMA_MEMCPY);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    bool m_isCreatedFromSemanticMemcpy = false;
};

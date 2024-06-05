#pragma once

#include "habana_nodes.h"
#include "node_visitor.h"
#include "dma_node.h"

class DMAMemsetNode : public DMANode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr  clone()        const override;
    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool     isMemset() const override;
    virtual bool     isNode64BitCompatible() const override { return true; }

    virtual DMA_OP_TYPE getOpType() const override;
    std::string         getNodeTypeStr() const override { return "DmaMemset"; }

private:
    DMAMemsetNode(const TensorVector& in, const TensorVector& out, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

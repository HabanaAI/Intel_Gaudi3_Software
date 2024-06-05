#pragma once
#include "node_visitor.h"
#include "multi_node.h"

class PhysicalReshapeNode : public MultiNode
{
public:
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
    using BaseClass = MultiNode;

    static bool requiresPhysicalReshapeToHandleDynamicity(const Tensor& in, const Tensor& out);

    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;
    virtual NodeList extract() override;
    virtual NodePtr  clone() const override;

private:
    PhysicalReshapeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

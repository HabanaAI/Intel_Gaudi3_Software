#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "habana_nodes.h"

class SplitShapeNode : public ShapeOperationNode<SplitNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<SplitNode>;

    ~SplitShapeNode() override = default;

    bool validateNode() const override;
    NodePtr clone() const override;

private:
    SplitShapeNode(const TensorVector& inputs,
                   const TensorVector& outputs,
                   UserParams          userParams,
                   std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
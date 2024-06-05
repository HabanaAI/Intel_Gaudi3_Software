#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "habana_nodes.h"

class ExpandDimsShapeNode : public ShapeOperationNode<ExpandDimsNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<ExpandDimsNode>;

    ~ExpandDimsShapeNode() override = default;

    bool validateNode() const override;
    NodePtr clone() const override;

private:
    ExpandDimsShapeNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        UserParams          userParams,
                        std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
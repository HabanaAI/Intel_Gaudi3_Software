#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "habana_nodes.h"

class StaticReshapeShapeNode : public ShapeOperationNode<StaticReshapeNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<StaticReshapeNode>;

    ~StaticReshapeShapeNode() override = default;

    StaticReshapeShapeNode(const TensorVector&       inputs,
                           const TensorVector&       outputs,
                           synStaticReshapeSifParams params,
                           std::string_view          name);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

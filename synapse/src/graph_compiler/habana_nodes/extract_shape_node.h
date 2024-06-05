#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "identity_node.h"

// Extract Shape Node is responsible to create a shape tensor out of another tensor.
// It can be used to save the shape of a tensor which data is no longer needed for the purpose of shape inference.

class ExtractShapeNode : public ShapeOperationNode<IdentityNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<IdentityNode>;

    ~ExtractShapeNode() override = default;

    bool validateNode() const override;
    NodePtr clone() const override;

private:
    ExtractShapeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "tensor_view_node.h"

// Extract Shape Node is responsible to create a shape tensor out of another tensor.
// It can be used to save the shape of a tensor which data is no longer needed for the purpose of shape inference.

class TensorViewShapeNode : public ShapeOperationNode<TensorViewNode>
{
    DEFINE_VISITOR_METHOD

public:
    using BaseClass = ShapeOperationNode<TensorViewNode>;

    /**
 * @param realTensor   IN  The shape tensor we wish to access
 * @param accessInput  IN  Whether the shape tensor is input or output,
 *                         true indicates the shape tensor is an input
 */
    explicit TensorViewShapeNode(const TensorPtr&   realTensor  = std::make_shared<Tensor>(),
                                 bool               accessInput = true,
                                 const std::string& name        = "");

    ~TensorViewShapeNode() override = default;

    NodePtr clone() const override;

};
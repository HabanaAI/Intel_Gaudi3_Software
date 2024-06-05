#include "tensor_view_shape_node.h"

TensorViewShapeNode::TensorViewShapeNode(const TensorPtr&   realTensor,
                                         bool               accessInput,
                                         const std::string& name)
: BaseClass(realTensor, accessInput, name, TYPE_TENSOR_VIEW_SHAPE_NODE)
{
    HB_ASSERT(realTensor->isShapeTensor(), "TensorViewShapeNode expects a shape tensor");
}

NodePtr TensorViewShapeNode::clone() const
{
    return NodePtr(new TensorViewShapeNode(*this));
}

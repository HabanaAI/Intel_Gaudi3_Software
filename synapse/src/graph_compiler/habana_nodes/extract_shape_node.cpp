#include "extract_shape_node.h"

ExtractShapeNode::ExtractShapeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: BaseClass(inputs, outputs, name, TYPE_EXTRACT_SHAPE)
{}

bool ExtractShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1,
                    "ExtractShapeNode Expects 1 input");

    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
                    "ExtractShapeNode Expects 1 output that is a shape tensor");

    HB_ASSERT(m_outputs.front()->compareGeometry(*m_inputs.front()),
              "ExtractShapeNode expects an input and an output tensor with the same geometry");

    // Inheritance order is ShapeOperationNode : IdentityNode : LogicalOpNode, use LogicalOpNode's validation
    return BaseClass::BaseClass::BaseClass::validateNode();
}

NodePtr ExtractShapeNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    return NodePtr(new ExtractShapeNode(inputs, outputs, name));
}

NodePtr ExtractShapeNode::clone() const
{
    return NodePtr(new ExtractShapeNode(*this));
}

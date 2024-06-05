#include "expand_dims_shape_node.h"

ExpandDimsShapeNode::ExpandDimsShapeNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          userParams,
                                         std::string_view    name)
: BaseClass(inputs, outputs, userParams, name, TYPE_EXPAND_DIMS_SHAPE)
{}

bool ExpandDimsShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1, "ExpandDimsShapeNode Expects 1 input");

    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
            "ExpandDimsShapeNode Expects 1 output that is a shape tensor");

    return BaseClass::validateNode();
}

NodePtr ExpandDimsShapeNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    LOG_TRACE(HABANA_NODE, "ExpandDimNode name - {}", name);
    return NodePtr(new ExpandDimsShapeNode(inputs, outputs, userParams, name));
}

NodePtr ExpandDimsShapeNode::clone() const
{
    return NodePtr(new ExpandDimsShapeNode(*this));
}
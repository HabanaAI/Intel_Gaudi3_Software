#include "split_shape_node.h"

SplitShapeNode::SplitShapeNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    name)
: BaseClass(inputs, outputs, userParams, name, TYPE_SPLIT_SHAPE)
{}

bool SplitShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1 && m_inputs.front()->isShapeTensor(),
            "SplitShapeNode Expects 1 input that is a shape tensor");

    CHECK_RET_FALSE(m_outputs.size() > 0, "SplitShapeNode Expects at least 1 output");

    return BaseClass::validateNode();
}

NodePtr SplitShapeNode::createNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name)
{
    LOG_TRACE(HABANA_NODE, "SplitShapeNode name - {}", name);
    return NodePtr(new SplitShapeNode(inputs, outputs, userParams, name));
}

NodePtr SplitShapeNode::clone() const
{
    return NodePtr(new SplitShapeNode(*this));
}
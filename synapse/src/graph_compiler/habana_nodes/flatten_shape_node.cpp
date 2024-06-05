#include "flatten_shape_node.h"

#include "types_exception.h"

#include <string>

FlattenShapeNode::FlattenShapeNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    name)
: BaseClass(inputs, outputs, userParams, name, TYPE_FLATTEN_SHAPE)
{}

bool FlattenShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1 && m_inputs.front()->isShapeTensor(),
            "FlattenShapeNode Expects 1 input that is a shape tensor");

    CHECK_RET_FALSE(m_outputs.size() > 0, "FlattenShapeNode Expects at least 1 output");

    return BaseClass::validateNode();
}

NodePtr FlattenShapeNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "FlattenShapeNode userParams is null");
        throw InvalidNodeParamsException(std::string {name}, "userParams");
    }
    LOG_TRACE(HABANA_NODE, "FlattenShapeNode name - {}", name);
    return NodePtr(new FlattenShapeNode(inputs, outputs, userParams, name));
}

NodePtr FlattenShapeNode::clone() const
{
    return NodePtr(new FlattenShapeNode(*this));
}
#include "shape_to_h2d_conversion_nodes.h"

StridedOpsConversionNode::StridedOpsConversionNode(const TensorVector& inputs,
                                                   const TensorVector& outputs,
                                                   std::string_view    name)
: LogicalOpNode(inputs, outputs, name, INPUT_TO_OUTPUT, TYPE_H2D_OP, SIF_SHAPE_TO_H2D_STRIDED)
{
}

NodePtr StridedOpsConversionNode::clone() const
{
    return NodePtr(new StridedOpsConversionNode(*this));
}

bool StridedOpsConversionNode::validateNode() const
{
    if (m_inputs.size() != 2)
    {
        LOG_ERR(HABANA_NODE, "StridedOpsConversionNode {} expects exactly 2 inputs", getNodeName());
        return false;
    }
    if (m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "StridedOpsConversionNode {} expects exactly 1 output", getNodeName());
        return false;
    }
    if (!m_inputs[0]->isShapeTensor() || !m_inputs[1]->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "StridedOpsConversionNode {}: all inputs must be shape tensors", getNodeName());
        return false;
    }
    if (!m_outputs[0]->isHost2DeviceTensor())
    {
        LOG_ERR(HABANA_NODE, "StridedOpsConversionNode {}: the output must be host to device tensor", getNodeName());
        return false;
    }

    return BaseClass::validateNode();
}

NodePtr StridedOpsConversionNode::createNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             std::string_view    guid,
                                             std::string_view    name)
{
    return NodePtr(new StridedOpsConversionNode(inputs, outputs, name));
}

SliceConversionNode::SliceConversionNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, TYPE_H2D_OP, SIF_SHAPE_TO_H2D_SLICE)
{
}

NodePtr SliceConversionNode::clone() const
{
    return NodePtr(new SliceConversionNode(*this));
}

bool SliceConversionNode::validateNode() const
{
    if (m_inputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "SliceConversionNode {} expects exactly 1 input", getNodeName());
        return false;
    }
    if (m_outputs.size() != 2)
    {
        LOG_ERR(HABANA_NODE, "SliceConversionNode {} expects exactly 2 outputs", getNodeName());
        return false;
    }
    if (!m_inputs[0]->isHost2DeviceTensor())
    {
        LOG_ERR(HABANA_NODE, "SliceConversionNode {}: the input must be host to device tensor", getNodeName());
        return false;
    }
    if (!m_outputs[0]->isShapeTensor() || !m_outputs[1]->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "SliceConversionNode {}: all outputs must be shape tensors", getNodeName());
        return false;
    }

    return BaseClass::validateNode();
}

NodePtr SliceConversionNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new SliceConversionNode(inputs, outputs, name));
}

#include "dynamic_range_node.hpp"

#include "types_exception.h"

DynamicRangeNode::DynamicRangeNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   std::string_view    name,
                                   UserParams          userParams)
: Node(inputs, outputs, name, Node::TYPE_DYNAMIC_RANGE)
{
    setParams(userParams, sizeof(synQuantDynamicRange));
}

NodePtr DynamicRangeNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    return NodePtr(new DynamicRangeNode(inputs, outputs, name, userParams));
}

void DynamicRangeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(synQuantDynamicRange))
    {
        LOG_ERR(HABANA_NODE, "DynamicRangeNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synQuantDynamicRange));
    }
    synQuantDynamicRange params = *(synQuantDynamicRange*)userParams;
    if (!setRange(params.min, params.max))
    {
        LOG_TRACE(HABANA_NODE,
                  "DynamicRange: name={}, range was NOT set min={}, max={}",
                  m_name,
                  params.min,
                  params.max);
    }
    else
    {
        LOG_TRACE(HABANA_NODE,
                  "DynamicRangeNode name - {}, params - min={}, max={}, isSet={}",
                  getNodeName(),
                  m_dynamicRange.min,
                  m_dynamicRange.max,
                  m_dynamicRange.isSet);
    }
}

bool DynamicRangeNode::setRange(double min, double max)
{
    // Handling ranges the way tensors handle them.
    if (max >= min)
    {
        LOG_TRACE(HABANA_NODE, "DynamicRangeNode: name={}, Set range min={}, max={}", m_name, min, max);
        m_dynamicRange.min   = min;
        m_dynamicRange.max   = max;
        m_dynamicRange.isSet = true;
        return true;
    }
    return false;
}

DynamicRange DynamicRangeNode::getRange() const
{
    return m_dynamicRange;
}

NodePtr DynamicRangeNode::clone() const
{
    return NodePtr(new DynamicRangeNode(*this));
}

bool DynamicRangeNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // Valid for inference only
    if (g.getTraits().trainingGraph())
    {
        LOG_ERR(HABANA_NODE, "DynamicRangeNode isn't supported in training");
        return false;
    }
    return true;
}

bool DynamicRangeNode::validateNode() const
{
    // Single input/output validation
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "DynamicRangeNode, Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    TensorPtr inputTensor  = *m_inputs.begin();
    TensorPtr outputTensor = *m_outputs.begin();
    if (!inputTensor || !outputTensor)
    {
        LOG_ERR(HABANA_NODE, "DynamicRangeNode input/output tensor cannot be null");
        return false;
    }
    if (inputTensor->isPersistent() && outputTensor->isPersistent())
    {
        LOG_ERR(HABANA_NODE, "DynamicRangeNode cannot have both persistent input and output");
        return false;
    }
    // Input and output should have the same type and shape (operator == for tensors)
    if (*inputTensor != *outputTensor)
    {
        LOG_ERR(HABANA_NODE, "DynamicRangeNode input and output tensors have different shape or datatype");
        return false;
    }
    return Node::validateNode();
}

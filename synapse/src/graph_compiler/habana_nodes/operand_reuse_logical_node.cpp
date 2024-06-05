#include "operand_reuse_logical_node.h"
#include "identity_node.h"
#include "defs.h"
#include "synapse_common_types.h"

OperandReuseInternalLogicalNode::OperandReuseInternalLogicalNode(const NodePtr& node,
                                                                 const unsigned inputIndex,
                                                                 const unsigned outputIndex)
: LogicalOpNode(node->getInputs(),
                node->getOutputs(),
                node->getNodeName(),
                INPUT_TO_OUTPUT,
                TYPE_OPERAND_REUSE_INTERNAL,
                (ShapeFuncID)node->getShapeInferenceFunctionId().sm_func_index),
  m_inputIndex(inputIndex),
  m_outputIndex(outputIndex),
  m_originalNode(node)
{
    HB_ASSERT(inputIndex < m_inputs.size(),
              "input index is {}, but there are only {} inputs",
              inputIndex,
              m_inputs.size());
    HB_ASSERT(outputIndex < m_outputs.size(),
              "output index is {}, but there are only {} outputs",
              outputIndex,
              m_outputs.size());
}

NodePtr OperandReuseInternalLogicalNode::clone() const
{
    return NodePtr(new OperandReuseInternalLogicalNode(*this));
}

NStrideArray OperandReuseInternalLogicalNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getRealTensor();
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void OperandReuseInternalLogicalNode::runLogicalOperation() const
{
    IdentityNode::runLogicalOperation(getRealTensor(), getAliasTensor());
}

TensorPtr OperandReuseInternalLogicalNode::getRealTensor() const
{
    return m_outputs[m_outputIndex];
}

TensorVector OperandReuseInternalLogicalNode::getAliasTensors() const
{
    return {getAliasTensor()};
}

const TensorPtr& OperandReuseInternalLogicalNode::getAliasTensor() const
{
    return m_inputs[m_inputIndex];
}

uint64_t OperandReuseInternalLogicalNode::getShapeInferenceFunctionVersion() const
{
    return m_originalNode->getShapeInferenceFunctionVersion();
}

SifNodeParams OperandReuseInternalLogicalNode::getShapeInferenceFunctionUserParams()
{
    return m_originalNode->getShapeInferenceFunctionUserParams();
}

size_t OperandReuseInternalLogicalNode::getShapeInferenceFunctionUserParamsSize() const
{
    return m_originalNode->getShapeInferenceFunctionUserParamsSize();
}
#include "frobenius_norm_node.h"

#include "utils.h"

FrobeniusNormNode::FrobeniusNormNode(const TensorVector& in, const TensorVector& out, std::string_view name)
: Node(in, out, name, TYPE_FROBENIUS_NORM_NODE, SIF_FROBENIUS_NORM)
{
}

bool FrobeniusNormNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

NodePtr FrobeniusNormNode::createNode(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams /*userParams*/,
                                      std::string_view /*guid*/,
                                      std::string_view name)
{
    return NodePtr(new FrobeniusNormNode(inputs, outputs, name));
}

NodePtr FrobeniusNormNode::clone() const
{
    return NodePtr(new FrobeniusNormNode(*this));
}

bool FrobeniusNormNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE,
                "Frobenius norm node- invalid number of operands; Expects 1 input and 1 output while recieved: {} "
                "inputs, and  {} outputs",
                m_inputs.size(),
                m_outputs.size());

        return false;
    }

    const TensorPtr& inputTensor  = getInput(TENSOR_IFM);
    const TensorPtr& outputTensor = getOutput(TENSOR_OFM);

    if (inputTensor->getDim() < 2 || inputTensor->getDim() > 4)
    {
        LOG_ERR(HABANA_NODE,
                "Input dimensionality for frobenius norm node should be 2D-4D. Received value of {}",
                inputTensor->getDim());
        return false;
    }

    if (outputTensor->getDim() != 1)
    {  // add number of  values.
        LOG_ERR(HABANA_NODE,
                "Output dimensionality for frobenius norm node should be equal to one. Received value of {}",
                outputTensor->getDim());
        return false;
    }

    if (inputTensor->getElementType() != syn_type_bf16 && inputTensor->getElementType() != syn_type_float)
    {
        LOG_ERR(HABANA_NODE,
                "Frobenius norm node input data type must ne BF16 or FP. Received value of {}",
                inputTensor->getElementType());
        return false;
    }

    return true;
}

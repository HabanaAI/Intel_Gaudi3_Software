#include "defs.h"
#include "synapse_common_types.h"
#include "utils.h"
#include "data_type_utils.h"

#include "reinterpret_cast_node.h"
#include "identity_node.h"


ReinterpretCastNode::ReinterpretCastNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         std::string_view    name,
                                         Node::eNodeType     type)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_REINTERPRET_CAST)
{
    HB_ASSERT(inputs.size() == 1 && outputs.size() == 1, "except 1 input and 1 output");
    HB_ASSERT_PTR(inputs[0]);
    HB_ASSERT_PTR(outputs[0]);

    m_sifMetadata.inputElementSizeInBytes  = inputs[0]->getElementSizeInBytes();
    m_sifMetadata.outputElementSizeInBytes = outputs[0]->getElementSizeInBytes();
}

NodePtr ReinterpretCastNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new ReinterpretCastNode(inputs, outputs, name));
}

NodePtr ReinterpretCastNode::clone() const
{
    return NodePtr(new ReinterpretCastNode(*this));
}

NStrideArray ReinterpretCastNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void ReinterpretCastNode::runLogicalOperation() const
{
    IdentityNode::runLogicalOperation(getRealTensor(), getAliasTensors().front());
}


// As opposed to most logical ops, in reinterpret cast the point is for the input and output to have different data
// types - so need to override the LogicalOpNode's getRequiredInput/OutputType APIs to just return the tensor's dtype
synDataType ReinterpretCastNode::getRequiredInputType(uint32_t tensorIdx) const
{
    return getInput(tensorIdx)->getElementType();
}

synDataType ReinterpretCastNode::getRequiredOutputType(uint32_t tensorIdx) const
{
    return getOutput(tensorIdx)->getElementType();
}


bool ReinterpretCastNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "ReinterpretCast Node {}: invalid number of operands", getNodeName());
        return false;
    }

    const TensorPtr inputTensor  = m_inputs.front();
    const TensorPtr outputTensor = m_outputs.front();

    if (inputTensor->getDim() != outputTensor->getDim())
    {
        LOG_ERR(HABANA_NODE, "ReinterpretCast Node {}: input and output dimension is different", getNodeName());
        return false;
    }

    auto inputElementSize  = inputTensor->getElementSizeInBytes();
    auto outputElementSize = outputTensor->getElementSizeInBytes();

    if (inputTensor->getSizeInElements(0) * inputElementSize != outputTensor->getSizeInElements(0) * outputElementSize)
    {
        LOG_ERR(HABANA_NODE, "ReinterpretCast Node {}: FCD Max sizes mismatch", getNodeName());
        return false;
    }
    if (inputTensor->getMinimalSizeInElements(0) * inputElementSize !=
        outputTensor->getMinimalSizeInElements(0) * outputElementSize)
    {
        LOG_ERR(HABANA_NODE, "ReinterpretCast Node {}: FCD Min sizes mismatch", getNodeName());
        return false;
    }

    for (unsigned dim = 1; dim < inputTensor->getDim(); ++dim)
    {
        if (inputTensor->getSizeInElements(dim) != outputTensor->getSizeInElements(dim) ||
            inputTensor->getMinimalSizeInElements(dim) != outputTensor->getMinimalSizeInElements(dim))
        {
            LOG_ERR(HABANA_NODE, "ReinterpretCast Node {}: sizes mismatch", getNodeName());
            return false;
        }
    }
    return true;
}

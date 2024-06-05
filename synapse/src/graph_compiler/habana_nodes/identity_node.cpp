#include "identity_node.h"

#include "data_type_utils.h"
#include "defs.h"
#include "synapse_common_types.h"
#include "utils.h"

IdentityNode::IdentityNode(const TensorVector& inputs,
                           const TensorVector& outputs,
                           std::string_view    name,
                           Node::eNodeType     type)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_IDENTITY)
{
    auto input  = inputs.front();
    auto output = outputs.front();

    HB_ASSERT(input != nullptr && output != nullptr, "Trying to construct Identity node without input/output");

    // Identity node with different types shouldn't be removed from graph, because they are used as
    // casts for signed-unsigned conversions.
    // Also, identity node with shape tensor needed to re-infer the shape after InferMaxShape node.
    if (inputs.size() == 2 || input->getElementType() != output->getElementType())
    {
        m_persistent = true;
    }
}

NodePtr IdentityNode::createNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 UserParams          userParams,
                                 std::string_view    guid,
                                 std::string_view    name)
{
    auto input  = inputs.front();
    auto output = outputs.front();

    HB_ASSERT(input != nullptr && output != nullptr, "Trying to create Identity node without input/output");

    return NodePtr(new IdentityNode(inputs, outputs, name));
}

NodePtr IdentityNode::clone() const
{
    return NodePtr(new IdentityNode(*this));
}

NStrideArray IdentityNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void IdentityNode::runLogicalOperation() const
{
    runLogicalOperation(getRealTensor(), getAliasTensors().front());
}

void IdentityNode::runLogicalOperation(const TensorPtr& real, const TensorPtr& alias)
{
    NStrideArray strides = {1};
    real->getNStridesInBytes(strides.data());
    strides[0] = alias->getElementSizeInBytes();  // needed for reinterpret cast
    alias->setAsAliasSubTensor(real);
    alias->reshape(alias->getDim(),
                   alias->getNSizesInElements().data(),
                   strides.data(),
                   alias->getNMinimalSizesInElements().data());
}

bool IdentityNode::validateNode() const
{
    if ((m_inputs.size() != 1 && m_inputs.size() != 2) || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Identity Node invalid number of operands");
        return false;
    }

    const TensorPtr input  = m_inputs[0];
    const TensorPtr output = m_outputs[0];
    const TensorPtr shape  = m_inputs.size() == 2 ? m_inputs[1] : nullptr;

    if (!input->compareGeometry(*output) || (shape && !input->compareGeometry(*shape)))
    {
        LOG_ERR(HABANA_NODE, "Identity Node input output geometry mismatch");
        return false;
    }

    auto inputType  = input->getElementType();
    auto outputType = output->getElementType();

    if (!isSameBitRepresentation(inputType, outputType))
    {
        LOG_ERR(HABANA_NODE, "Identity Node input output type mismatch");
        return false;
    }
    return true;
}

bool IdentityNode::validateDynamicShapes() const
{
    const TensorPtr input  = m_inputs[0];
    const TensorPtr output = m_outputs[0];

    bool existsUserManagedDramTensor = input->isUserManagedDram() || output->isUserManagedDram();
    if (m_inputs.size() == 2 && !input->isDynamicShape() && output->isDynamicShape() && existsUserManagedDramTensor)
    {
        LOG_ERR(HABANA_NODE,
                "Identity Node {} input is static and output is dynamic, so user managed dram tensors are disallowed",
                getNodeName());
        return false;
    }
    return true;
}

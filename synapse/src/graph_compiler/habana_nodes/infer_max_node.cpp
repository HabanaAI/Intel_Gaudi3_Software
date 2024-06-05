#include "defs.h"
#include "synapse_common_types.h"
#include "utils.h"
#include "data_type_utils.h"

#include "infer_max_node.h"
#include "identity_node.h"


InferMaxShapeNode::InferMaxShapeNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    name,
                                     Node::eNodeType     type)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_INFER_MAX_SHAPE)
{
    setParams(userParams, sizeof(synInferMaxParams));
}

NodePtr InferMaxShapeNode::createNode(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name)
{
    return NodePtr(new InferMaxShapeNode(inputs, outputs, userParams, name));
}

void InferMaxShapeNode::setParams(UserParams userParams, unsigned userParamsSize)
{
    if (userParams == nullptr)
    {
        auto& arr = m_sifMetadata.params.shouldInferMax;
        std::fill(arr, arr + ARRAY_SIZE(arr), true);
    }
    else
    {
        HB_ASSERT(userParamsSize == sizeof(m_sifMetadata.params), "params size mismatch");
        m_sifMetadata.params = std::move(*reinterpret_cast<synInferMaxParams*>(userParams));
    }
    Node::setParams(&m_sifMetadata.params, userParamsSize);
}

NodePtr InferMaxShapeNode::clone() const
{
    return NodePtr(new InferMaxShapeNode(*this));
}

NStrideArray InferMaxShapeNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void InferMaxShapeNode::runLogicalOperation() const
{
    IdentityNode::runLogicalOperation(getRealTensor(), getAliasTensors().front());
}

SifNodeParams InferMaxShapeNode::getShapeInferenceFunctionUserParams()
{
    m_inputs[0]->getAllSizesInElements(m_sifMetadata.inputMaxSizes, SYN_MAX_TENSOR_DIM);
    return (SifNodeParams*)&m_sifMetadata;
}

bool InferMaxShapeNode::validateNode() const
{
    if (m_inputs.size() != 1 || (m_outputs.size() != 1 && m_outputs.size() != 2))
    {
        LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: invalid number of operands", getNodeName());
        return false;
    }

    if (m_outputs.size() == 2 && !m_outputs[1]->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: invalid number of data outputs", getNodeName());
        return false;
    }

    const TensorPtr& inputTensor  = m_inputs[0];
    const TensorPtr& outputTensor = m_outputs[0];
    const TensorPtr& shapeTensor  = getOutput(1);

    if (inputTensor->getDim() != outputTensor->getDim())
    {
        LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: input and output dimension is different", getNodeName());
        return false;
    }

    if (inputTensor->getElementType() != outputTensor->getElementType())
    {
        LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: element type mismatch", getNodeName());
        return false;
    }

    for (unsigned dim = 0; dim < inputTensor->getDim(); ++dim)
    {
        if (inputTensor->getSizeInElements(dim) != outputTensor->getSizeInElements(dim))
        {
            LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: data tensor sizes mismatch", getNodeName());
            return false;
        }
        if (shapeTensor && (inputTensor->getSizeInElements(dim) != shapeTensor->getSizeInElements(dim) ||
                            inputTensor->getMinimalSizeInElements(dim) != shapeTensor->getMinimalSizeInElements(dim)))
        {
            LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: shape tensor sizes mismatch", getNodeName());
            return false;
        }

        const auto* shouldInferMax = m_sifMetadata.params.shouldInferMax;
        if (shouldInferMax[dim] && outputTensor->isDynamicDim(dim))
        {
            LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: dim {} in data output must be static", getNodeName(), dim);
            return false;
        }
    }
    return BaseClass::validateNode();
}

bool InferMaxShapeNode::validateDynamicShapes() const
{
    const TensorPtr& inputTensor  = m_inputs[0];
    const TensorPtr& outputTensor = m_outputs[0];
    if (inputTensor->isUserManagedDram() || outputTensor->isUserManagedDram())
    {
        LOG_ERR(HABANA_NODE, "InferMaxShapeNode Node {}: user managed dram tensors are not supported", getNodeName());
        return false;
    }
    return BaseClass::validateDynamicShapes();
}

#include "defs.h"

#include "tile_shape_node.h"

TileShapeNode::TileShapeNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          userParams,
                             unsigned            userParamsSize,
                             std::string_view    name)
: BaseClass(inputs, outputs, name, OUTPUT_TO_INPUT, Node::TYPE_TILE_SHAPE, SIF_TILE_SHAPE)
{
    setParams(userParams, userParamsSize);
}

void TileShapeNode::setParams(UserParams userParams, unsigned userParamsSize)
{
    HB_ASSERT(userParamsSize == sizeof(m_params), "params size mismatch");
    m_params = *reinterpret_cast<ns_TileKernel::ParamsV2*>(userParams);
}

NodePtr TileShapeNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  unsigned            userParamsSize,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new TileShapeNode(inputs, outputs, userParams, userParamsSize, name));
}

NodePtr TileShapeNode::clone() const
{
    return NodePtr(new TileShapeNode(*this));
}

SifNodeParams TileShapeNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams*)&m_params;
}

bool TileShapeNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "TileShapeNode Node {}: invalid number of operands", getNodeName());
        return false;
    }

    const TensorPtr& in  = m_inputs[0];
    const TensorPtr& out = m_outputs[0];

    if (!in->isShapeTensor() || !out->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "TileShapeNode Node {}: operands must be shape tensors", getNodeName());
        return false;
    }

    if (in->getDim() != out->getDim())
    {
        LOG_ERR(HABANA_NODE, "TileShapeNode Node {}: input and output dimension is different", getNodeName());
        return false;
    }

    if (in->getDim() > ARRAY_SIZE(m_params.repeat))
    {
        LOG_ERR(HABANA_NODE,
                "TileShapeNode Node {}: dim is bigger than max size, dim: {} max: {}",
                in->getDim(),
                ARRAY_SIZE(m_params.repeat),
                getNodeName());
        return false;
    }

    if (in->getElementType() != out->getElementType())
    {
        LOG_ERR(HABANA_NODE, "TileShapeNode Node {}: element type mismatch", getNodeName());
        return false;
    }

    for (unsigned dim = 0; dim < in->getDim(); ++dim)
    {
        if (in->getSizeInElements(dim) * m_params.repeat[dim] != out->getSizeInElements(dim) ||
            in->getMinimalSizeInElements(dim) * m_params.repeat[dim] != out->getMinimalSizeInElements(dim))
        {
            LOG_ERR(HABANA_NODE,
                    "TileShapeNode Node {}: sizes mismatch at dim {}, input [max: {}, min: {}], output [max: {}, min: "
                    "{}], repeat: {}",
                    getNodeName(),
                    dim,
                    in->getSizeInElements(dim),
                    in->getMinimalSizeInElements(dim),
                    out->getSizeInElements(dim),
                    out->getMinimalSizeInElements(dim),
                    m_params.repeat[dim]);
            return false;
        }
    }

    return BaseClass::validateNode();
}

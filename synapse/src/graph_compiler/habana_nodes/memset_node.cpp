
#include "habana_graph.h"
#include "memset_node.h"

MemsetNode::MemsetNode(const TensorVector& in, const TensorVector& out, std::string_view name)
: Node(in, out, name, TYPE_MEMSET, SIF_DMA_MEMSET /* To allow shape inference before concrete node is selected */)
{
    //This node is just for the semantics of memset. It is to be replaced in a pass by dma memset node.
}

NodePtr MemsetNode::createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    guid,
                               std::string_view    name)
{

    return NodePtr(new MemsetNode(inputs, outputs, name));
}

bool MemsetNode::validateNode() const
{
    if (!m_inputs.empty() && (m_inputs.size() != 1 || !m_inputs.front()->isShapeTensor()))
    {
        LOG_ERR(HABANA_NODE, "MemsetNode Invalid number of operands (expecting no inputs or a single shape tensor as input)");
        return false;
    }
    if (m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "MemsetNode Invalid number of operands (expecting 1 output)");
        return false;
    }

    return Node::validateNode();
}

NodePtr MemsetNode::clone() const
{
    return NodePtr(new MemsetNode(*this));
}

bool MemsetNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // Currently only gaudi, gaudi2 and goya2 has support for memset nodes
    return true;
}

bool MemsetNode::isLinear() const
{
    TensorPtr dst = !m_outputs.empty() ? getOutput(TENSOR_OFM) : nullptr;

    if (dst && !dst->isDenseLayout())
    {
        return false;
    }
    return !isDynamicShape();
}

#include "memcopy_node.h"

MemcpyNode::MemcpyNode(const TensorVector& in, const TensorVector& out, std::string_view name)
: Node(in, out, name, TYPE_MEMCOPY, SIF_DMA_MEMCPY /* To allow shape inference before concrete node is selected */)
{
    //This node is just for the semantics of memcpy. It is to be replaced in a pass by a tpc or dma memcpy node.
}

NodePtr MemcpyNode::createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    guid,
                               std::string_view    name)
{

    return NodePtr(new MemcpyNode(inputs, outputs, name));
}

bool MemcpyNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "MemCpyNode {} : Invalid number of operands (expecting 1 input and 1 output). "
                             "Actual num of inputs: {} outputs: {}",
                m_name, m_inputs.size(), m_outputs.size());
        return false;
    }
    if (!m_inputs[0]->compareGeometry(*m_outputs[0]))
    {
        LOG_ERR(HABANA_NODE, "Memcopy Node {} input output geometry mismatch: "
                             "input tensor name: {} sizes={} minSizes={} strides={}, "
                             "output tensor name: {} sizes={} minSizes={} strides={}",
                m_name,
                m_inputs[0]->getName(),
                m_inputs[0]->getDimSizesStr(),
                m_inputs[0]->getDimSizesStr(false, true),
                m_inputs[0]->getStridesStr(),
                m_outputs[0]->getName(),
                m_outputs[0]->getDimSizesStr(),
                m_outputs[0]->getDimSizesStr(false, true),
                m_outputs[0]->getStridesStr() );
        return false;
    }

    return Node::validateNode();
}

NodePtr MemcpyNode::clone() const
{
    return NodePtr(new MemcpyNode(*this));
}

bool MemcpyNode::validateNodeForGraph(const HabanaGraph& g) const
{
    //Memcpy is later replaced by the correct engine, it is fine for both graphs. Always.
    return true;
}

bool MemcpyNode::isNode64BitCompatible() const
{
    return true;
}

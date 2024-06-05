#include "dma_memset_node.h"

#include "habana_graph.h"

DMAMemsetNode::DMAMemsetNode(const TensorVector& in, const TensorVector& out, std::string_view name)
: DMANode(!in.empty() && in[0]->isShapeTensor() ? TensorVector({in[0]}) : TensorVector(),
          TensorVector({out[0]}),
          name,
          DMA_TYPE_INTERNAL,
          SIF_DMA_MEMSET)
{
}

NodePtr DMAMemsetNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    HB_ASSERT(inputs.empty() || (inputs.size() == 1 && inputs.front()->isShapeTensor()),
            "DMAMemset node should not have input tensors (except optional shape tensor)");
    return NodePtr(new DMAMemsetNode(inputs, outputs, name));
}

bool DMAMemsetNode::validateNode() const
{
    if ((!m_inputs.empty() && m_inputs.size() != 1) || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "DMAMemsetNode Invalid number of operands (expecting 0 or 1 inputs and 1 output)");
        return false;
    }
    if (m_inputs.size() == 1 && !m_inputs.back()->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Invalid inputs, expecting shape tensor at index 0");
        return false;
    }

    return DMANode::validateNode();
}

NodePtr DMAMemsetNode::clone() const
{
    return NodePtr(new DMAMemsetNode(*this));
}

bool DMAMemsetNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return DMANode::validateNodeForGraph(g);
}

bool DMAMemsetNode::isMemset() const
{
    return true;
}

DMA_OP_TYPE DMAMemsetNode::getOpType() const
{
    return DMA_OP_TYPE::DMA_OP_MEMSET;
}

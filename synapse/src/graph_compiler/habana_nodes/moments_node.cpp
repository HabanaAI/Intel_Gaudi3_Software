
#include "habana_graph.h"
#include "moments_node.h"
#include "graph_traits.h"
#include <string_view>

MomentsNode::MomentsNode(const TensorVector& in, const TensorVector& out, std::string_view name)
: Node(in, out, name, TYPE_MOMENTS, SIF_MOMENTS)
{
    //This node is just for the semantics
}

NodePtr MomentsNode::createNode(const TensorVector& inputs,
                                const TensorVector& outputs,
                                UserParams          userParams,
                                std::string_view    guid,
                                std::string_view    name)
{

    return NodePtr(new MomentsNode(inputs, outputs, name));
}

bool MomentsNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 2)
    {
        LOG_ERR(HABANA_NODE, "MomentsNode invalid number of operands: expecting 1 inputs (got {}) and 2 outputs (got {})",
                m_inputs.size(), m_outputs.size());
        return false;
    }

    return Node::validateNode();
}

NodePtr MomentsNode::clone() const
{
    return NodePtr(new MomentsNode(*this));
}

bool MomentsNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // Currently only gaudi, gaudiM and gaudi2 has support for memset nodes
    return (g.getTraits().trainingGraph());
}

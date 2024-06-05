#include "multi_node.h"

MultiNode::MultiNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     std::string_view    name,
                     eNodeType           type,
                     ShapeFuncID         sifId)
: Node(inputs, outputs, name, type, sifId)
{
}

bool MultiNode::RunOnCpu()
{
    const NodeList& extractedNodes = extract();
    for (auto node : extractedNodes)
    {
        if (!node->RunOnCpu())
        {
            return false;
        }
    }
    return true;
}

bool MultiNode::RunOnCpu(const HabanaGraph& g)
{
    const NodeList& extractedNodes = extract(g);
    for (auto node : extractedNodes)
    {
        if (!node->RunOnCpu())
        {
            return false;
        }
    }
    return true;
}

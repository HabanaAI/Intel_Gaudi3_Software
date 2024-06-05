#include "physical_memory_ops_nodes.h"

template<class BASE>
NodePtr DeserializeNode<BASE>::clone() const
{
    return NodePtr(new DeserializeNode(*this));
}

template<class BASE>
bool DeserializeNode<BASE>::validateNode() const
{
    if (this->m_inputs.size() != 1)
    {
        return false;
    }

    if (this->m_outputs.size() != 1)
    {
        return false;
    }

    const TensorPtr input  = this->getInput(0);
    const TensorPtr output = this->getOutput(0);

    // Deserialize layout assume same sizes but adding strides to the input
    for (int d = 0; d < input->getDim(); d++)
    {
        if (input->getStrideInElements(d + 1) > output->getStrideInElements(d + 1))
        {
            LOG_ERR(HABANA_NODE,
                    "DeserializeNode invalid strides dim {} input {} output {}",
                    d,
                    input->getStrideInElements(d + 1),
                    output->getStrideInElements(d + 1));
            return false;
        }

        if (input->getSizeInElements(d) != output->getSizeInElements(d))
        {
            LOG_ERR(HABANA_NODE,
                    "DeserializeNode invalid Sizes dim {} input {} output {}",
                    d,
                    input->getSizeInElements(d),
                    output->getSizeInElements(d));
            return false;
        }
    }

    return BASE::validateNode();
}

template<class BASE>
void DeserializeNode<BASE>::calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const
{
    PhysicalMemoryOpNode<BASE>::calculateLinearRanges(tRoi, n, isInput);
    if (isSrcDynamicStrided() == isInput)
    {
        this->fixLinearRangesToRealParentStart(tRoi);
    }
}

template<class BASE>
NodePtr DeserializeNode<BASE>::createNode(const TensorVector& inputs,
                                          const TensorVector& outputs,
                                          UserParams          userParams,
                                          std::string_view    guid,
                                          std::string_view    name)
{
    return NodePtr(new DeserializeNode(inputs, outputs, name));
}

template<class BASE>
DeserializeNode<BASE>::DeserializeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: PhysicalMemoryOpNode<BASE>(inputs, outputs, name)
{
    // Don't actually do anything for now.
}

template<class BASE>
std::string DeserializeNode<BASE>::getNodeTypeStr() const
{
    return "Deserialize";
}

template class DeserializeNode<DMAMemcpyNode>;
template class DeserializeNode<TPCMemcpyNode>;


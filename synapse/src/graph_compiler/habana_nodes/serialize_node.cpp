#include "physical_memory_ops_nodes.h"

template<class BASE>
NodePtr SerializeNode<BASE>::clone() const
{
    return NodePtr(new SerializeNode(*this));
}

template<class BASE>
bool SerializeNode<BASE>::validateNode() const
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

    // Serialize layout assume same sizes but removing strides from the input to the output
    for (int d = 0; d < input->getDim(); d++)
    {
        if (input->getStrideInElements(d + 1) < output->getStrideInElements(d + 1))
        {
            LOG_ERR(HABANA_NODE,
                    "SerializeNode invalid strides dim {} input {} output {}",
                    d,
                    input->getStrideInElements(d + 1),
                    output->getStrideInElements(d + 1));
            return false;
        }

        if (input->getSizeInElements(d) != output->getSizeInElements(d))
        {
            LOG_ERR(HABANA_NODE,
                    "SerializeNode invalid Sizes dim {} input {} output {}",
                    d,
                    input->getSizeInElements(d),
                    output->getSizeInElements(d));
            return false;
        }
    }

    return BASE::validateNode();
}

template<class BASE>
NodePtr SerializeNode<BASE>::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new SerializeNode(inputs, outputs, name));
}

template<class BASE>
void SerializeNode<BASE>::calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const
{
    PhysicalMemoryOpNode<BASE>::calculateLinearRanges(tRoi, n, isInput);
    if (isSrcDynamicStrided() == isInput)
    {
        this->fixLinearRangesToRealParentStart(tRoi);
    }
}

template<class BASE>
SerializeNode<BASE>::SerializeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: PhysicalMemoryOpNode<BASE>(inputs, outputs, name)
{
    // Don't actually do anything for now.
}

template<class BASE>
std::string SerializeNode<BASE>::getNodeTypeStr() const
{
    return "Serialize";
}

template class SerializeNode<DMAMemcpyNode>;
template class SerializeNode<TPCMemcpyNode>;

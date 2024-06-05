#include "defs.h"
#include "habana_graph.h"
#include "physical_memory_ops_nodes.h"
#include "smf/shape_func_registry.h"
#include "slice_logical_node.h"
#include "types_exception.h"
#include "types.h"


template <class BASE>
NodePtr DynamicSliceMemcpyNode<BASE>::clone() const
{
    return NodePtr(new DynamicSliceMemcpyNode<BASE>(*this));
}

template <class BASE>
bool DynamicSliceMemcpyNode<BASE>::validateNode() const
{
    if (this->m_inputs.size() != MAX_NUM_INPUTS)
    {
        return false;
    }

    if (this->m_outputs.size() != 1)
    {
        return false;
    }

    const TensorPtr& in     = this->m_inputs[INPUT_TENSOR];
    const TensorPtr& steps  = this->m_inputs[STEPS_TENSOR];
    const TensorPtr& starts = this->m_inputs[STARTS_TENSOR];
    const TensorPtr& out    = this->m_outputs[0];

    if (in->isShapeTensor() || !starts->isShapeTensor() || !steps->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "expected [DATA,SHAPE,SHAPE] as tensor types for node input {}", this->m_name);
        return false;
    }

    if (!in->compareGeometry(*out))
    {
        LOG_ERR(HABANA_NODE, "expected same shape for input and output of node {}", this->m_name);
        return false;
    }

    if ((in->getDim() != starts->getDim()) || (in->getDim() != steps->getDim()))
    {
        LOG_ERR(HABANA_NODE, "wrong dimensions for inputs of node {}", this->m_name);
        return false;
    }

    return BASE::validateNode();
}

/*
    Check if the accessed addresses during runtime are a subset of the static compilation linear ranges.
    They are not a subset if either one of these conditions apply:
    a. min start < static compile-time start
    b. different patched step
    c. step larger than 1 and different start
*/

template <class BASE>
bool DynamicSliceMemcpyNode<BASE>::isOutOfOriginalLinearRanges(const TensorPtr& starts, const TensorPtr& steps)
{
    HB_ASSERT_PTR(starts);
    HB_ASSERT_PTR(steps);
    bool     ret = false;
    unsigned dim = starts->getDim();
    for (unsigned d = 0; d < dim; d++)
    {
        ret |= starts->getMinimalSizeInElements(d) < starts->getSizeInElements(d);
        ret |= steps->getMinimalSizeInElements(d) != steps->getSizeInElements(d);
        ret |=
            (steps->getSizeInElements(d) > 1) && (starts->getMinimalSizeInElements(d) != starts->getSizeInElements(d));
    }
    return ret;
}

template <class BASE>
void DynamicSliceMemcpyNode<BASE>::calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const
{
    if (isSrcDynamicStrided() == isInput &&
        isOutOfOriginalLinearRanges(n->getInput(STARTS_TENSOR), n->getInput(STEPS_TENSOR)))
    {
        this->applyFullViewLinearRange(tRoi);  // take the full address ranges of the parent tensor
    }
    else
    {
        PhysicalMemoryOpNode<BASE>::calculateLinearRanges(tRoi, n, isInput);
    }
}

template <class BASE>
NodePtr DynamicSliceMemcpyNode<BASE>::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new DynamicSliceMemcpyNode<BASE>(inputs, outputs, name, userParams));
}

template <class BASE>
void DynamicSliceMemcpyNode<BASE>::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(bool))
    {
        LOG_ERR(HABANA_NODE, "DynamicSliceMemcpyNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(this->m_name, userParamsSize, sizeof(bool));
    }
    m_isSrc = *reinterpret_cast<bool*>(userParams);
    LOG_TRACE(HABANA_NODE, "DynamicSliceMemcpyNode name - {}, params - is source operand={}", this->getNodeName(), m_isSrc);
}

template <class BASE>
DynamicSliceMemcpyNode<BASE>::DynamicSliceMemcpyNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         std::string_view    name,
                                         UserParams          params)
: PhysicalMemoryOpNode<BASE>(inputs, outputs, name)
{
    this->m_inputs = inputs;
    setParams(params, sizeof(bool));
}

template <class BASE>
std::string DynamicSliceMemcpyNode<BASE>::getNodeTypeStr() const
{
    return "DynamicSliceMemcpyNode";
}

template class DynamicSliceMemcpyNode<DMAMemcpyNode>;
template class DynamicSliceMemcpyNode<TPCMemcpyNode>;

#include "defs.h"
#include "h2d_tensors.h"
#include "habana_graph.h"
#include "physical_memory_ops_nodes.h"
#include "smf/shape_func_registry.h"
#include "strided_insert_logical_node.h"
#include "strided_view_logical_node.h"
#include "types_exception.h"
#include "types.h"

template <class BASE>
NodePtr DynamicStridedMemcpyNode<BASE>::clone() const
{
    return NodePtr(new DynamicStridedMemcpyNode<BASE>(*this));
}

template <class BASE>
bool DynamicStridedMemcpyNode<BASE>::validateNode() const
{
    if (this->m_inputs.size() != 2)
    {
        LOG_ERR(HABANA_NODE, "Node {} expects 2 inputs", this->m_name);
        return false;
    }

    if (this->m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Node {} expects 1 output", this->m_name);
        return false;
    }

    const TensorPtr& in  = this->m_inputs[0];
    const TensorPtr& h2d = this->m_inputs[1];
    const TensorPtr& out = this->m_outputs[0];

    if (!in->compareGeometry(*out))
    {
        LOG_ERR(HABANA_NODE, "Node {} expects same shape for data input and data output", this->m_name);
        return false;
    }

    if (in->isShapeTensor() || !h2d->isHost2DeviceTensor())
    {
        LOG_ERR(HABANA_NODE, "Node {} expects [DATA,H2D] as tensor types for node input", this->m_name);
        return false;
    }

    return BASE::validateNode();
}

template <class BASE>
NodePtr DynamicStridedMemcpyNode<BASE>::createNode(const TensorVector& inputs,
                                          const TensorVector& outputs,
                                          UserParams          userParams,
                                          std::string_view    guid,
                                          std::string_view    name)
{
    return NodePtr(new DynamicStridedMemcpyNode<BASE>(inputs, outputs, name, userParams));
}

template <class BASE>
void DynamicStridedMemcpyNode<BASE>::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(bool))
    {
        LOG_ERR(HABANA_NODE, "DynamicStridedMemcpyNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(this->m_name, userParamsSize, sizeof(bool));
    }
    m_isSrc = *reinterpret_cast<bool*>(userParams);
    LOG_TRACE(HABANA_NODE, "DynamicStridedMemcpyNode name - {}, params - is source operand={}", this->getNodeName(), m_isSrc);
}

template <class BASE>
DynamicStridedMemcpyNode<BASE>::DynamicStridedMemcpyNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             std::string_view    name,
                                             UserParams          params)
: PhysicalMemoryOpNode<BASE>(inputs, outputs, name)
{
    this->m_inputs = inputs;
    setParams(params, sizeof(bool));
}

/*
    in the dma read/write view, we are using the wrong strides during the compilation.
    we patch the correct strides (and offset) only during runtime.
    That is why we cannot rely on the linear ranges for the tensorROI and need to assume we use the entire real tensor.

    [real_input] -> (StridedView) -> [fake_input] -> (DmaViewRead) -> [output]
    or
    [in] -> (DmaViewWrite) -> [fake_output] -> (StridedInsert) -> [real_output]

    Use the real_tensor's full size for linear address calculation.
*/
template <class BASE>
void DynamicStridedMemcpyNode<BASE>::calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const
{
    if (isSrcDynamicStrided() == isInput)
    {
        this->applyFullViewLinearRange(tRoi);  // take the full address ranges of the parent tensor
    }
    else
    {
        PhysicalMemoryOpNode<BASE>::calculateLinearRanges(tRoi, n, isInput);
    }
}

template <class BASE>
bool DynamicStridedMemcpyNode<BASE>::isDynamicStridedDmaNode(const NodePtr& n, bool isSrc)
{
    if (!n->isDma()) return false;
    auto dynamicStrideNode = dynamic_cast<DynamicStridedMemcpyNode*>(n.get());
    if (!dynamicStrideNode) return false;
    return dynamicStrideNode->isSrcDynamicStrided() == isSrc;
}

template <class BASE>
bool DynamicStridedMemcpyNode<BASE>::isDynamicShape() const
{
    if (isDynamicStridedDmaH2DTensorDynamic(this->m_inputs[1])) return true;
    return BASE::isDynamicShape();
}

template <class BASE>
std::string DynamicStridedMemcpyNode<BASE>::getNodeTypeStr() const
{
    return "DynamicStridedMemcpyNode";
}

template class DynamicStridedMemcpyNode<DMAMemcpyNode>;
template class DynamicStridedMemcpyNode<TPCMemcpyNode>;

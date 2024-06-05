#include "dma_memcopy_node.h"

#include "defs.h"
#include "habana_graph.h"
#include "synapse_common_types.h"
#include "types.h"
#include "utils.h"

DMAMemcpyNode::DMAMemcpyNode(const TensorVector& in, const TensorVector& out, std::string_view name, ShapeFuncID sifId)
: DMANode(in.empty() ? TensorVector() : TensorVector({in[0]}), TensorVector({out[0]}), name, DMA_TYPE_INTERNAL, sifId),
  m_isCreatedFromSemanticMemcpy(false)
{
    if (m_inputs.empty() || m_inputs[0]->isShapeTensor())
    {
        m_shapeInferenceFunctionID = SIF_DMA_MEMSET;
    }
}

NodePtr DMAMemcpyNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new DMAMemcpyNode(inputs, outputs, name));
}

bool DMAMemcpyNode::validateNode() const
{
    if (!isMemset() && !isDynamicMemoryOp() && m_inputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Memcpy operation: Invalid number of operands (expecting 1 input)");
        return false;
    }

    if (m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "{} Invalid number of operands (expecting 1 output)", isMemset()? "Memset operation: " : "Memcpy operation: ");
        return false;
    }

    return DMANode::validateNode();
}

NodePtr DMAMemcpyNode::clone() const
{
    return NodePtr(new DMAMemcpyNode(*this));
}

bool DMAMemcpyNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return DMANode::validateNodeForGraph(g);
}

bool DMAMemcpyNode::isMemset() const
{
    return getNumInputsDataTensors() == 0;
}

bool DMAMemcpyNode::isLinearDma() const
{
    return !isMemset() && DMANode::isLinearDma();
}

DMA_OP_TYPE DMAMemcpyNode::getOpType() const
{
    return DMA_OP_TYPE::DMA_OP_COPY;
}

bool DMAMemcpyNode::isNode64BitCompatible() const
{
    return true;
}

bool DMAMemcpyNode::RunOnCpu()
{
    std::shared_ptr<Tensor> in  = getInput(TENSOR_IFM);
    std::shared_ptr<Tensor> out = getOutput(TENSOR_OFM);
    HB_ASSERT(in->getTotalSizeInBytes() == out->getTotalSizeInBytes(),
              "DMAMemcpyNode::RunOnCpu: mismatch between input and output size");
    HB_ASSERT(in->getElementSizeInBytes() >= 1,
              "DMAMemcpyNode::RunOnCpu: tensors with less than one byte element size are currently not supported");
    NSizeArray                               sa;
    std::array<uint64_t, HABANA_DIM_MAX + 1> stridesIn;
    in->getNStridesInBytes(stridesIn.data());
    HB_ASSERT(stridesIn[0] == in->getElementSizeInBytes(), "fcd stride");
    std::array<uint64_t, HABANA_DIM_MAX + 1> stridesOut;
    out->getNStridesInBytes(stridesOut.data());
    HB_ASSERT(stridesOut[0] == out->getElementSizeInBytes(), "fcd stride");
    NSizeArray total         = in->getNSizesInElements();
    auto       totalElements = multiplyElements(total.begin(), total.end());

    char* pOut = static_cast<char*>(out->map());
    char* pIn  = static_cast<char*>(in->map());

    HB_ASSERT(pOut, "Output tensor for node {} is not mapped", getNodeName());
    HB_ASSERT(pIn, "Input tensor for node {} is not mapped", getNodeName());

    for (int i = 0; i < totalElements; i++)
    {
        auto     itemIndex = i;
        uint32_t inOffset  = 0;
        uint32_t outOffset = 0;
        for (int dimIndex = 0; dimIndex < in->getDim(); dimIndex++)
        {
            sa[dimIndex] = itemIndex % in->getSizeInElements(dimIndex);
            itemIndex /= in->getSizeInElements(dimIndex);
            inOffset += sa[dimIndex] * stridesIn[dimIndex];
            outOffset += sa[dimIndex] * stridesOut[dimIndex];
        }

        std::memcpy(pOut + outOffset, pIn + inOffset, in->getElementSizeInBytes());
    }

    return true;
}

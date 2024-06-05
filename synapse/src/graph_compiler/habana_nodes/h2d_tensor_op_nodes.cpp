#include "h2d_tensor_op_nodes.h"
#include "h2d_tensors.h"

H2DTensorOpNode::H2DTensorOpNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 std::string_view    name,
                                 ShapeFuncID         sifId)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, TYPE_H2D_OP, sifId)
{
}

bool H2DTensorOpNode::validateNode() const
{
    // all inputs and outputs must be H2D tensors
    for (const TensorPtr& t : m_inputs)
    {
        if (!t->isHost2DeviceTensor())
        {
            LOG_ERR(HABANA_NODE, "H2D tensor op node {}: all inputs must be H2D tensors", getNodeName());
            return false;
        }
    }
    for (const TensorPtr& t : m_outputs)
    {
        if (!t->isHost2DeviceTensor())
        {
            LOG_ERR(HABANA_NODE, "H2D tensor op node {}: all outputs must be H2D tensors", getNodeName());
            return false;
        }
    }
    return BaseClass::validateNode();
}

DynamicStridedDmaExpandH2DNode::DynamicStridedDmaExpandH2DNode(const TensorVector& inputs,
                                                               const TensorVector& outputs,
                                                               unsigned            dim,
                                                               std::string_view    name)
: H2DTensorOpNode(inputs, outputs, name, SIF_H2D_DYN_STRIDE_DMA_EXPAND), m_expandDim(dim)
{
}

NodePtr DynamicStridedDmaExpandH2DNode::clone() const
{
    return NodePtr(new DynamicStridedDmaExpandH2DNode(*this));
}

bool DynamicStridedDmaExpandH2DNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1,
                    "DynamicStridedDmaExpandH2DNode {} expects exactly one input",
                    getNodeName());

    CHECK_RET_FALSE(m_outputs.size() == 1,
                    "DynamicStridedDmaExpandH2DNode {} expects exactly one output",
                    getNodeName());

    return BaseClass::validateNode();
}

NodePtr DynamicStridedDmaExpandH2DNode::createNode(const TensorVector& inputs,
                                                   const TensorVector& outputs,
                                                   UserParams          userParams,
                                                   std::string_view    guid,
                                                   std::string_view    name)
{
    HB_ASSERT_PTR(userParams);
    unsigned dim = *reinterpret_cast<unsigned*>(userParams);
    return NodePtr(new DynamicStridedDmaExpandH2DNode(inputs, outputs, dim, name));
}

DynamicStridedDmaReinterpretH2DNode::DynamicStridedDmaReinterpretH2DNode(const TensorVector& inputs,
                                                                         const TensorVector& outputs,
                                                                         unsigned            factor,
                                                                         std::string_view    name)
: H2DTensorOpNode(inputs, outputs, name, SIF_H2D_DYN_STRIDE_DMA_REINTERPRET), m_factor(factor)
{
}

NodePtr DynamicStridedDmaReinterpretH2DNode::clone() const
{
    return NodePtr(new DynamicStridedDmaReinterpretH2DNode(*this));
}

bool DynamicStridedDmaReinterpretH2DNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1,
                    "DynamicStridedDmaReinterpretH2DNode {} expects exactly one input",
                    getNodeName());

    CHECK_RET_FALSE(m_outputs.size() == 1,
                    "DynamicStridedDmaReinterpretH2DNode {} expects exactly one output",
                    getNodeName());

    return BaseClass::validateNode();
}

NodePtr DynamicStridedDmaReinterpretH2DNode::createNode(const TensorVector& inputs,
                                                        const TensorVector& outputs,
                                                        UserParams          userParams,
                                                        std::string_view    guid,
                                                        std::string_view    name)
{
    HB_ASSERT_PTR(userParams);
    unsigned factor = *reinterpret_cast<unsigned*>(userParams);
    return NodePtr(new DynamicStridedDmaReinterpretH2DNode(inputs, outputs, factor, name));
}

DynamicSliceDmaExpandH2DNode::DynamicSliceDmaExpandH2DNode(const TensorVector& inputs,
                                                           const TensorVector& outputs,
                                                           unsigned            dim,
                                                           std::string_view    name)
: H2DTensorOpNode(inputs, outputs, name, SIF_H2D_DYN_SLICE_DMA), m_expandDim(dim)
{
}

NodePtr DynamicSliceDmaExpandH2DNode::clone() const
{
    return NodePtr(new DynamicSliceDmaExpandH2DNode(*this));
}

bool DynamicSliceDmaExpandH2DNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1,
                    "DynamicSliceDmaExpandH2DNode {} expects exactly one input",
                    getNodeName());

    CHECK_RET_FALSE(m_outputs.size() == 1,
                    "DynamicSliceDmaExpandH2DNode {} expects exactly one output",
                    getNodeName());

    return BaseClass::validateNode();
}

NodePtr DynamicSliceDmaExpandH2DNode::createNode(const TensorVector& inputs,
                                                 const TensorVector& outputs,
                                                 UserParams          userParams,
                                                 std::string_view    guid,
                                                 std::string_view    name)
{
    HB_ASSERT_PTR(userParams);
    unsigned dim = *reinterpret_cast<unsigned*>(userParams);
    return NodePtr(new DynamicSliceDmaExpandH2DNode(inputs, outputs, dim, name));
}

TransposeSliceH2DNode::TransposeSliceH2DNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             unsigned            userParamsSize,
                                             std::string_view    name)
: H2DTensorOpNode(inputs, outputs, name, SIF_H2D_TRANSPOSE_SLICE)
{
    setParams(userParams, userParamsSize);
}

NodePtr TransposeSliceH2DNode::clone() const
{
    return NodePtr(new TransposeSliceH2DNode(*this));
}

bool TransposeSliceH2DNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1,
                    "TransposeSliceH2DNode {} expects exactly one input",
                    getNodeName());

    CHECK_RET_FALSE(m_outputs.size() == 1,
                    "TransposeSliceH2DNode {} expects exactly one output",
                    getNodeName());

    return BaseClass::validateNode();
}

void TransposeSliceH2DNode::setParams(UserParams userParams, unsigned userParamsSize)
{
    HB_ASSERT_PTR(userParams);
    TransposePermutationArray permutation;
    unsigned                  tensorDim = reinterpret_cast<synDynamicSliceDmaH2dTensor*>(m_inputs[0]->getHostMaxData())->dims;
    if (userParamsSize != sizeof(synTransposeParamsNDims))
    {
        HB_ASSERT(userParamsSize == sizeof(synTransposeParams), "TransposeNode userParams size is incorrect");

        synTransposeParams transposeParams = *(synTransposeParams*)userParams;
        permutation = {transposeParams.permutation, transposeParams.permutation + tensorDim};
    }
    else
    {
        synTransposeParamsNDims transposeParams = *(synTransposeParamsNDims*)userParams;

        for (int i = 0; i < tensorDim; i++)
        {
            permutation.push_back((TransposePermutationDim)transposeParams.permutation[i]);
        }
    }

    LOG_TRACE(HABANA_NODE,
              "TransposeNode name - {}, params - tensorDim={}, permutation={}, in sizes={}",
              m_name,
              tensorDim,
              toString(permutation, ','),
              toString(m_inputs[0]->getNSizesInElements(), ','));
    m_permutation = permutation;
    memcpy(m_sifMetadata.permutation, permutation.data(), permutation.size() * sizeof(permutation[0]));
}

NodePtr TransposeSliceH2DNode::createNode(const TensorVector& inputs,
                                          const TensorVector& outputs,
                                          UserParams          userParams,
                                          unsigned            userParamsSize,
                                          std::string_view    guid,
                                          std::string_view    name)
{
    return NodePtr(new TransposeSliceH2DNode(inputs, outputs, userParams, userParamsSize, name));
}
#include "transposed_shape_node.h"

#include "synapse_common_types.hpp"
#include "types_exception.h"

TransposedShapeNode::TransposedShapeNode(const TensorPtr& input,
                                         const TensorPtr& output,
                                         UserParams       params,
                                         unsigned         paramsSize,
                                         std::string_view name)
: BaseClass(input, output, name, Node::TYPE_TRANSPOSED_SHAPE_NODE)
{
    setParams(params, paramsSize);
}

bool TransposedShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1,
                    "TransposedShapeNode Expects 1 input");

    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
                    "TransposedShapeNode Expects 1 output that is a shape tensor");

    return LogicalOpNode::validateNode();
}

NodePtr TransposedShapeNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        unsigned            userParamsSize,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new TransposedShapeNode(inputs[TENSOR_IFM], outputs[TENSOR_OFM], userParams, userParamsSize, name));
}

void TransposedShapeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    TransposePermutationArray permutation;

    if (userParamsSize == sizeof(synTransposeParams))
    {
        synTransposeParams transposeParams = *(synTransposeParams*)userParams;
        permutation                        = TransposePermutationArray(transposeParams.permutation,
                                                transposeParams.permutation + transposeParams.tensorDim);
    }
    else
    {
        if (userParamsSize != sizeof(synTransposeParamsNDims))
        {
            LOG_ERR(HABANA_NODE, "TransposedShapeNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name);
        }
        auto transposeParams = *(synTransposeParamsNDims*)userParams;
        permutation.reserve(transposeParams.tensorDim);
        for (int i = 0; i < transposeParams.tensorDim; i++)
        {
            permutation.push_back((TransposePermutationDim)transposeParams.permutation[i]);
        }
    }
    m_permutation = permutation;
    memcpy(m_sifMetadata.permutation, m_permutation.data(), m_permutation.size() * sizeof(m_permutation[0]));
    LOG_TRACE(HABANA_NODE,
              "TransposedShapeNode name - {}, params - permutation={}, in sizes={}",
              getNodeName(),
              toString(m_permutation, ','),
              toString(m_inputs[0]->getNSizesInElements(), ','));
}

NodePtr TransposedShapeNode::clone() const
{
    return NodePtr(new TransposedShapeNode(*this));
}

#include "strided_insert_logical_node.h"

#include "strided_insert_node.h"
#include "strided_op_node_utils.h"
#include "types_exception.h"

LogicalStridedInsertNode::LogicalStridedInsertNode(const TensorVector& in,
                                                   const TensorVector& out,
                                                   UserParams          params,
                                                   std::string_view    name)
: LogicalOpNode(in, out, name, INPUT_TO_OUTPUT, TYPE_STRIDED_INSERT, SIF_STRIDED_INSERT)
{
    setParams(params, sizeof(synStridedOpParams));
}

NodePtr LogicalStridedInsertNode::createNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             std::string_view    guid,
                                             std::string_view    name)
{
    HB_ASSERT(!outputs.empty(), "no output for node {}", name);
    HB_ASSERT(inputs.size() == 2, "no insert input for node {}", name);
    return NodePtr(new LogicalStridedInsertNode(inputs, outputs, userParams, name));
}

void LogicalStridedInsertNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    HB_ASSERT_PTR(userParams);
    if (userParamsSize != sizeof(synStridedOpParams))
    {
        LOG_ERR(HABANA_NODE, "LogicalStridedInsertNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    const auto& params = *(synStridedOpParams*)userParams;
    LOG_TRACE(HABANA_NODE,
              "StridedInsertNode name - {}, params - {}",
              m_name,
              StridedOpUtils::stridedOpParamsString(params, m_inputs[INSERT_TENSOR]->getDim()));
    m_params = params;
}

bool LogicalStridedInsertNode::validateNode() const
{
    if (!LogicalOpNode::validateNode())
    {
        return false;
    }

    // do not validate out of bound memory access, because it was validated in StridedInsert node.
    // this is for the case where we "trick" the logical SI with fake params (dynamic access)
    if (!StridedInsertNode::validateInsertNode(this, m_params, false /* validateAccess */))
    {
        return false;
    }

    if (m_params.strides[0] != 1)
    {
        LOG_ERR(HABANA_NODE, "Logical Strided Insert Node {}, does not support FCD strides", m_name);
        return false;
    }

    return true;
}

NodePtr LogicalStridedInsertNode::clone() const
{
    return NodePtr(new LogicalStridedInsertNode(*this));
}

NStrideArray LogicalStridedInsertNode::calculateAliasStrides(unsigned idx) const
{
    if (idx == INSERT_TENSOR)
    {
        const TensorPtr& inInsert = m_inputs[INSERT_TENSOR];
        NStrideArray     strides  = {1};
        inInsert->getNStridesInBytes(strides.data());
        for (unsigned d = 0; d < inInsert->getDim(); d++)
        {
            strides[d] = m_params.strides[d] * inInsert->getElementSizeInBytes();
        }
        return strides;
    }
    else  // ORIGINAL_TENSOR
    {
        const TensorPtr& real = m_outputs[TENSOR_OFM];
        NStrideArray     ret  = {1};
        real->getNStridesInBytes(ret.data());
        return ret;
    }
}

void LogicalStridedInsertNode::runLogicalOperation() const
{
    const TensorPtr& out        = m_outputs[TENSOR_OFM];
    const TensorPtr& inOriginal = m_inputs[ORIGINAL_TENSOR];
    const TensorPtr& inInsert   = m_inputs[INSERT_TENSOR];

    NStrideArray insertStrides = calculateAliasStrides(INSERT_TENSOR);

    // have the insert input as an alias to the output according to the node params
    // have the original input as an alias to the output
    inInsert->reshape(inInsert->getDim(),
                      inInsert->getNSizesInElements().data(),
                      insertStrides.data(),
                      inInsert->getNMinimalSizesInElements().data());
    inInsert->setAsAliasSubTensor(out, m_params.baseOffset * out->getElementSizeInBytes());
    inOriginal->setAsAliasSubTensor(out);

    HB_ASSERT(!inInsert->isStridedOnFCD(), "strided insert does not support fcd strides");
}

void LogicalStridedInsertNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

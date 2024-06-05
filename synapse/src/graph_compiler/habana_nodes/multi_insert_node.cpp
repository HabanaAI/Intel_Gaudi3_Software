#include "multi_insert_node.h"

#include "types_exception.h"

NodePtr MultiInsertNode::clone() const
{
    return NodePtr(new MultiInsertNode(*this));
}

bool MultiInsertNode::validateNode() const
{
    if (!BaseClass::validateNode())
    {
        return false;
    }

    if (m_inputs.size() < 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "{}: unexpected number of input/outputs for node {}", HLLOG_FUNC, m_name);
        return false;
    }

    const TensorPtr& in  = m_inputs[ORIGINAL_TENSOR];
    const TensorPtr& out = m_outputs[0];

    if (!in->compareGeometry(*out))
    {
        LOG_ERR(HABANA_NODE, "{}: input and output have different shapes. node {}", HLLOG_FUNC, m_name);
        return false;
    }

    if (m_inputOffsets.size() != m_inputs.size())
    {
        LOG_ERR(HABANA_NODE,
                "{}: expected {} input offsets but got {}. node: {}",
                HLLOG_FUNC,
                m_inputs.size(),
                m_inputOffsets.size(),
                m_name);
        return false;
    }

    for (unsigned i = 0; i < m_inputs.size(); i++)
    {
        if (m_inputs[i]->getDenseSizeInElements() + m_inputOffsets[i] > out->getDenseSizeInElements())
        {
            LOG_ERR(HABANA_NODE, "{}: input {} is out of bounds for original tensor. node {}", HLLOG_FUNC, i, m_name);
            return false;
        }
    }

    return true;
}

void MultiInsertNode::runLogicalOperation() const
{
    const TensorPtr& out = m_outputs[0];
    for (unsigned i = 0; i < m_inputs.size(); i++)
    {
        m_inputs[i]->setAsAliasSubTensor(out, m_inputOffsets[i] * out->getElementSizeInBytes());
    }
}

void MultiInsertNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)m_inputOffsets.data(), sizeof(m_inputOffsets.data()));
}

MultiInsertNode::MultiInsertNode(const TensorVector& in,
                                 const TensorVector& out,
                                 UserParams          params,
                                 std::string_view    name)
: LogicalOpNode(in, out, name, INPUT_TO_OUTPUT, TYPE_MULTI_INSERT, SIF_DMA_MEMCPY)
{
    setParams(params, sizeof(MultiInsertParams));
}

NodePtr MultiInsertNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    HB_ASSERT(!outputs.empty(), "no output for node {}", name);
    HB_ASSERT(!inputs.empty(), "no input for node {}", name);
    return NodePtr(new MultiInsertNode(inputs, outputs, userParams, name));
}

void MultiInsertNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    HB_ASSERT_PTR(userParams);
    if (userParamsSize != sizeof(MultiInsertParams))
    {
        LOG_ERR(HABANA_NODE, "MultiInsertNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    const auto& params = *(MultiInsertParams*)userParams;
    LOG_TRACE(HABANA_NODE, "MultiInsertNode name - {}, params - {}", m_name, toString(params, ','));
    m_inputOffsets = params;
}

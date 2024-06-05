#include "strided_view_logical_node.h"

#include "strided_op_node_utils.h"
#include "strided_view_node.h"
#include "types_exception.h"

LogicalStridedViewNode::LogicalStridedViewNode(const TensorVector& in,
                                               const TensorVector& out,
                                               UserParams          params,
                                               std::string_view    name)
: LogicalOpNode(in, out, name, OUTPUT_TO_INPUT, TYPE_STRIDED_VIEW, SIF_STRIDED_VIEW)
{
    setParams(params, sizeof(synStridedOpParams));
}

NodePtr LogicalStridedViewNode::createNode(const TensorVector& inputs,
                                           const TensorVector& outputs,
                                           UserParams          userParams,
                                           std::string_view    guid,
                                           std::string_view    name)
{
    HB_ASSERT(!outputs.empty(), "no output for node {}", name);
    HB_ASSERT(!inputs.empty(), "no input for node {}", name);
    return NodePtr(new LogicalStridedViewNode(inputs, outputs, userParams, name));
}

void LogicalStridedViewNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    HB_ASSERT_PTR(userParams);
    if (userParamsSize != sizeof(synStridedOpParams))
    {
        LOG_ERR(HABANA_NODE, "LogicalStridedViewNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    const auto& params = *(synStridedOpParams*)userParams;
    LOG_TRACE(HABANA_NODE,
              "LogicalStridedView name - {}, params - {}",
              m_name,
              StridedOpUtils::stridedOpParamsString(params, m_outputs[0]->getDim()));
    m_params = params;
}

bool LogicalStridedViewNode::validateNode() const
{
    if (!LogicalOpNode::validateNode())
    {
        return false;
    }

    // do not validate out of bound memory access, because it was validated in StridedView node.
    // this is for the case where we "trick" the logical SV with fake params (dynamic access)
    bool valid = StridedViewNode::validateViewNode(this, m_params, false /* validateAccess */);
    if (!valid)
    {
        return false;
    }

    if (m_params.strides[0] != 1)
    {
        LOG_ERR(HABANA_NODE, "StridedView Node {}, currently does not support FCD strides", m_name);
        return false;
    }
    return true;
}

NodePtr LogicalStridedViewNode::clone() const
{
    return NodePtr(new LogicalStridedViewNode(*this));
}

NStrideArray LogicalStridedViewNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& out = m_outputs.front();
    NStrideArray     strides = {1};
    out->getNStridesInBytes(strides.data());
    for (unsigned d = 0; d < out->getDim(); d++)
    {
        strides[d] = m_params.strides[d] * out->getElementSizeInBytes();
    }
    return strides;
}

void LogicalStridedViewNode::runLogicalOperation() const
{
    const TensorPtr& out = m_outputs.front();
    const TensorPtr& in  = m_inputs.front();

    NStrideArray strides = calculateAliasStrides(0);

    // have the output as an alias to the input according to the node params
    out->reshape(out->getDim(),
                 out->getAllNSizesInElements().data(),
                 strides.data(),
                 out->getNMinimalSizesInElements().data());
    out->setAsAliasSubTensor(in, m_params.baseOffset * out->getElementSizeInBytes());

    HB_ASSERT(!out->isStridedOnFCD(), "strided view node does not support fcd strides");
}

void LogicalStridedViewNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

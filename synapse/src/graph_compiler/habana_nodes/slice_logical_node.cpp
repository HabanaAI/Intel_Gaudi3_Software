#include "slice_logical_node.h"

#include "types_exception.h"

#include <string_view>

using SliceNodeStaticParams = SliceNode::SliceNodeStaticParams;

LogicalSliceNode::LogicalSliceNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          params,
                                   AliasDirection      direction,
                                   std::string_view    name,
                                   eNodeType           nodeType,
                                   ShapeFuncID         sifId)
: BaseClass(inputs, outputs, name, direction, nodeType, sifId)
{
    setParams(params, sizeof(m_params));
}

void LogicalSliceNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "SliceNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize != sizeof(SliceNodeStaticParams))
    {
        LOG_ERR(HABANA_NODE, "LogicalSliceNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    m_params = *(SliceNodeStaticParams*)userParams;
    LOG_TRACE(HABANA_NODE,
              "LogicalSliceNode name - {}, params - starts={}, ends={}, steps={}",
              m_name,
              toString(m_params.starts.data(), m_params.starts.data() + m_inputs[0]->getDim(), ','),
              toString(m_params.ends.data(), m_params.ends.data() + m_inputs[0]->getDim(), ','),
              toString(m_params.steps.data(), m_params.steps.data() + m_inputs[0]->getDim(), ','));
}

std::pair<NStrideArray, uint64_t> LogicalSliceNode::calcStridesAndOffset(const TensorPtr& real) const
{
    NStrideArray strides = {1};
    real->getNStridesInBytes(strides.data());
    uint64_t offset;
    TStride stride = real->getElementSizeInBytes();
    offset          = stride * m_params.starts[0];
    for (unsigned i = 1; i < real->getDim(); i++)
    {
        stride = real->getStrideInBytes(i);
        if (m_params.steps[i] != 0)
        {
            strides[i] *= m_params.steps[i];
        }
        offset += stride * m_params.starts[i];
    }
    return std::make_pair(strides, offset);
}

void LogicalSliceNode::runSlice(TensorPtr real, TensorPtr aliased) const
{
    // Apply strides and offset to the aliased tensor and set it as alias to the real tensor
    auto [strides, offset] = calcStridesAndOffset(real);
    aliased->setAsSliceSubTensor(real, offset, strides.data());
}

bool LogicalSliceNode::RunOnCpu()
{
    return false;
}

bool LogicalSliceNode::isRedundantNode() const
{
    if (isDynamicShape()) return false;
    const TensorPtr real    = getRealTensor();
    const TensorPtr aliased = getAliasTensors().front();
    return SliceNode::isRedundantSlice(real, aliased, m_params);
}

void LogicalSliceNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

SifNodeParams LogicalSliceNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams)&m_params;
}

size_t LogicalSliceNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_params);
}

bool LogicalSliceNode::validateNode() const
{
    if (m_inputs.size() != 1 && m_inputs.size() != 2)
    {
        LOG_ERR(HABANA_NODE, "Slice ({}): expecting 1 or 2 inputs", m_name);
        return false;
    }

    if (!m_inputs[0]->isDataTensor() || !m_outputs[0]->isDataTensor())
    {
        LOG_ERR(HABANA_NODE, "Slice ({}): expecting data tensors for in0 and out0", m_name);
        return false;
    }

    if (m_params.steps[0] != 1 && m_params.steps[0] < m_params.ends[0])
    {
        LOG_ERR(HABANA_NODE, "Slice ({}): step in the FCD must be equal to 1, got {})", m_name, m_params.steps[0]);
        return false;
    }

    return LogicalOpNode::validateNode();
}
#include "slice_bwd_logical_node.h"
#include "utils.h"

LogicalSliceBwdNode::LogicalSliceBwdNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          params,
                                         std::string_view    name)
: BaseClass(inputs, outputs, params, INPUT_TO_OUTPUT, name, TYPE_SLICE_BWD, SIF_SLICE_BACKWARD)
{
}

NodePtr LogicalSliceBwdNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    LOG_TRACE(HABANA_NODE, "LogicalSliceBwdNode guid - {} name - {}", guid, name);
    return NodePtr(new LogicalSliceBwdNode(inputs, outputs, userParams, name));
}

NStrideArray LogicalSliceBwdNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real             = getOutput(TENSOR_OFM);
    auto             stridesAndOffset = calcStridesAndOffset(real);
    return stridesAndOffset.first;
}
void LogicalSliceBwdNode::runLogicalOperation() const
{
    runSlice(getOutput(TENSOR_OFM), getInput(TENSOR_IFM));
}

NodePtr LogicalSliceBwdNode::clone() const
{
    return NodePtr(new LogicalSliceBwdNode(*this));
}

bool LogicalSliceBwdNode::validateNode() const
{
    if (m_inputs.size() == 2 && !m_inputs[1]->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Logical SliceBwd ({}): expecting 2nd input to be a shape tensor!", m_name);
        return false;
    }
    return BaseClass::validateNode();
}

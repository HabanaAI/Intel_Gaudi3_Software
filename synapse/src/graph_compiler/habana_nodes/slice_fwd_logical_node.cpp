#include "slice_fwd_logical_node.h"
#include "slice_fwd_node.h"
#include "utils.h"

LogicalSliceFwdNode::LogicalSliceFwdNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          params,
                                         std::string_view    name)
: BaseClass(inputs, outputs, params, OUTPUT_TO_INPUT, name, TYPE_SLICE, SIF_SLICE)
{
}

NodePtr LogicalSliceFwdNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new LogicalSliceFwdNode(inputs, outputs, userParams, name));
}

NodePtr LogicalSliceFwdNode::clone() const
{
    return NodePtr(new LogicalSliceFwdNode(*this));
}

bool LogicalSliceFwdNode::validateNode() const
{
    if (m_inputs.size() == 2 && !m_inputs[1]->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Logical SliceFwd ({}): expecting 2nd input to be a shape tensor!", m_name);
        return false;
    }
    return BaseClass::validateNode();
}

NStrideArray LogicalSliceFwdNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real             = getInput(TENSOR_IFM);
    auto             stridesAndOffset = calcStridesAndOffset(real);
    return stridesAndOffset.first;
}

void LogicalSliceFwdNode::runLogicalOperation() const
{
    runSlice(getInput(TENSOR_IFM), getOutput(TENSOR_OFM));
}

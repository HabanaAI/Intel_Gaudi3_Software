#include "slice_insert_logical_node.h"
#include "slice_insert_node.h"
#include "utils.h"

LogicalSliceInsertNode::LogicalSliceInsertNode(const TensorVector& inputs,
                                               const TensorVector& outputs,
                                               UserParams          params,
                                               std::string_view    name)
: BaseClass(inputs, outputs, params, INPUT_TO_OUTPUT, name, TYPE_SLICE_INSERT, SIF_SLICE_INSERT)
{
}

NodePtr LogicalSliceInsertNode::createNode(const TensorVector& inputs,
                                           const TensorVector& outputs,
                                           UserParams          userParams,
                                           std::string_view    guid,
                                           std::string_view    name)
{
    LOG_TRACE(HABANA_NODE, "LogicalSliceInsertNode guid - {} name - {}", guid, name);
    return NodePtr(new LogicalSliceInsertNode(inputs, outputs, userParams, name));
}

NStrideArray LogicalSliceInsertNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getOutput(TENSOR_OFM);
    if (idx == SliceInsertNode::INSERT_TENSOR)
    {
        auto stridesAndOffset = calcStridesAndOffset(real);
        return stridesAndOffset.first;
    }
    else  // idx == SliceInsertNode::ORIGINAL_TENSOR
    {
        NStrideArray ret = {1};
        real->getNStridesInBytes(ret.data());
        return ret;
    }
}

void LogicalSliceInsertNode::runLogicalOperation() const
{
    TensorPtr real     = getOutput(TENSOR_OFM);
    TensorPtr original = getInput(SliceInsertNode::ORIGINAL_TENSOR);
    TensorPtr insert   = getInput(SliceInsertNode::INSERT_TENSOR);

    runSlice(real, insert);
    original->setAsSliceSubTensor(real, 0, real->getNStridesInBytes());
}

bool LogicalSliceInsertNode::isAliasStrided(unsigned idx) const
{
    return idx == SliceInsertNode::INSERT_TENSOR;
}

NodePtr LogicalSliceInsertNode::clone() const
{
    return NodePtr(new LogicalSliceInsertNode(*this));
}

bool LogicalSliceInsertNode::validateNode() const
{
    if (m_inputs.size() != 2 || !m_inputs[0]->isDataTensor() || !m_inputs[1]->isDataTensor())
    {
        LOG_ERR(HABANA_NODE, "Logical SliceInsert ({}): expecting 2 data tensor inputs!", m_name);
        return false;
    }
    return BaseClass::validateNode();
}

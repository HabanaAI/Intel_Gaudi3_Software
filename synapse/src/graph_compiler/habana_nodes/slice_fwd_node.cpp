#include "slice_fwd_node.h"
#include "physical_memory_ops_nodes.h"
#include "slice_fwd_logical_node.h"
#include "node_factory.h"
#include "utils.h"

using SliceNodeStaticParams = SliceNode::SliceNodeStaticParams;

SliceFwdNode::SliceFwdNode(const TensorVector& inputs,
                           const TensorVector& outputs,
                           UserParams          params,
                           unsigned            paramsSize,
                           std::string_view    name)
: BaseClass(inputs, outputs, name, Node::TYPE_SLICE, SIF_SLICE)
{
    setParams(params, paramsSize);
}

NodePtr SliceFwdNode::clone() const
{
    return NodePtr(new SliceFwdNode(*this));
}

NodePtr
SliceFwdNode::getSliceNode(const TensorVector& inputs, const TensorPtr& output, const SliceNodeStaticParams& params)
{
    auto sliceNode = NodeFactory::createNode(inputs, {output}, &params, NodeFactory::sliceNodeTypeName, m_name);
    sliceNode->getNodeAnnotation().originatedFromCguid = getNodeAnnotation().originatedFromCguid;
    return sliceNode;
}

NodeList SliceFwdNode::extract()
{
    NodeList ret = SliceNode::extractNodes();
    if (!ret.empty()) return ret;

    if (isDynamicSlice())
    {
        return extractDynamicSlice();
    }

    return {getLogicalNode(getUnslicedTensor(), getSlicedTensor(), m_params)};
}

TensorPtr SliceFwdNode::getUnslicedTensor() const
{
    return m_inputs[0];
}

TensorPtr SliceFwdNode::getSlicedTensor() const
{
    return m_outputs[0];
}

bool SliceFwdNode::canTranspose() const
{
    return true;
}

NodePtr SliceFwdNode::getLogicalNode(const TensorPtr&             unsliced,
                                     const TensorPtr&             sliced,
                                     const SliceNodeStaticParams& params) const
{
    TensorVector inputs = {unsliced};
    if (m_inputs.size() > SliceNode::SHAPE_TENSOR)
    {
        inputs.push_back(m_inputs[SliceNode::SHAPE_TENSOR]);
    }
    NodePtr logicalSlice =
        NodeFactory::createNode(inputs, {sliced}, &m_params, NodeFactory::logicalSliceFwdNodeTypeName, m_name);
    logicalSlice->getNodeAnnotation().originatedFromCguid = getNodeAnnotation().originatedFromCguid;
    return logicalSlice;
}

NodePtr SliceFwdNode::createNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 UserParams          userParams,
                                 unsigned            userParamsSize,
                                 std::string_view    guid,
                                 std::string_view    name)
{
    return NodePtr(new SliceFwdNode(inputs, outputs, userParams, userParamsSize, name));
}

bool SliceFwdNode::validateNode() const
{
    return validateSlice(getInput(TENSOR_IFM), getOutput(TENSOR_OFM));
}

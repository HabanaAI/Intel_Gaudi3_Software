#include "slice_insert_node.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "utils.h"

using SliceNodeStaticParams = SliceNode::SliceNodeStaticParams;

SliceInsertNode::SliceInsertNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 UserParams          params,
                                 unsigned            paramsSize,
                                 std::string_view    name)
: BaseClass(inputs, outputs, name, Node::TYPE_SLICE_INSERT, SIF_SLICE_INSERT)
{
    setParams(params, paramsSize);
}

NodePtr SliceInsertNode::clone() const
{
    return NodePtr(new SliceInsertNode(*this));
}

NodePtr
SliceInsertNode::getSliceNode(const TensorVector& inputs, const TensorPtr& output, const SliceNodeStaticParams& params)
{
    return NodeFactory::createNode(inputs, {output}, &params, NodeFactory::sliceInsertNodeTypeName, m_name);
}

NodeList SliceInsertNode::extract()
{
    NodeList ret = SliceNode::extractNodes();
    if (!ret.empty()) return ret;

    if (isDynamicSlice())
    {
        return extractDynamicSlice();
    }

    return {getLogicalNode(getUnslicedTensor(), getSlicedTensor(), m_params)};
}

TensorPtr SliceInsertNode::getUnslicedTensor() const
{
    return m_outputs[0];
}

TensorPtr SliceInsertNode::getSlicedTensor() const
{
    return m_inputs[INSERT_TENSOR];
}

bool SliceInsertNode::canTranspose() const
{
    return true;
}

NodePtr SliceInsertNode::getLogicalNode(const TensorPtr&             unsliced,
                                        const TensorPtr&             sliced,
                                        const SliceNodeStaticParams& params) const
{
    TensorVector inputs = {m_inputs[ORIGINAL_TENSOR], sliced};
    return NodeFactory::createNode(inputs, {unsliced}, &m_params, NodeFactory::logicalSliceInsertNodeTypeName, m_name);
}

NodePtr SliceInsertNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    unsigned            userParamsSize,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    return NodePtr(new SliceInsertNode(inputs, outputs, userParams, userParamsSize, name));
}

bool SliceInsertNode::validateNode() const
{
    if (m_inputs.size() < 2 || m_outputs.size() > 1)
    {
        LOG_ERR(HABANA_NODE, "SliceInsert: invalid number of operands");
        return false;
    }
    if (!m_inputs[ORIGINAL_TENSOR]->compareGeometry(*m_outputs[0]))
    {
        LOG_ERR(HABANA_NODE, "SliceInsert: original tensor has a different geometry than output tensor");
        return false;
    }
    return validateSlice(getOutput(TENSOR_OFM), getInput(INSERT_TENSOR));
}

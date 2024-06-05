#include "slice_grad_node.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "utils.h"

using SliceNodeStaticParams = SliceNode::SliceNodeStaticParams;

SliceGradNode::SliceGradNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          params,
                             unsigned            paramsSize,
                             std::string_view    name)
: BaseClass(inputs, outputs, name, Node::TYPE_SLICE_GRAD, SIF_SLICE_BACKWARD)
{
    setParams(params, paramsSize);
}

NodePtr SliceGradNode::clone() const
{
    return NodePtr(new SliceGradNode(*this));
}

NodePtr
SliceGradNode::getSliceNode(const TensorVector& inputs, const TensorPtr& output, const SliceNodeStaticParams& params)
{
    return NodeFactory::createNode(inputs, {output}, &params, NodeFactory::stridedSliceGradNodeTypeName, m_name);
}

NodeList SliceGradNode::extract()
{
    NodeList ret = SliceNode::extractNodes();
    if (!ret.empty()) return ret;

    return extractGradSlice();
}

TensorPtr SliceGradNode::getUnslicedTensor() const
{
    return m_outputs[0];
}

TensorPtr SliceGradNode::getSlicedTensor() const
{
    return m_inputs[0];
}

bool SliceGradNode::canTranspose() const
{
    return true;
}

NodePtr SliceGradNode::getLogicalNode(const TensorPtr&             unsliced,
                                      const TensorPtr&             sliced,
                                      const SliceNodeStaticParams& params) const
{
    HB_ASSERT(0, "slice grad does not have a logical node!");
    return nullptr;
}

NodePtr SliceGradNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  unsigned            userParamsSize,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new SliceGradNode(inputs, outputs, userParams, userParamsSize, name));
}

bool SliceGradNode::validateNode() const
{
    return validateSlice(getOutput(TENSOR_OFM), getInput(TENSOR_IFM));
}

// extracting slice grad into slice insert + memset.
NodeList SliceGradNode::extractGradSlice() const
{
    std::string nodeNamePrefix = getNodeName();
    NodeList    retNodes;

    TensorPtr gradOut     = getOutput(0);
    TensorPtr shapeTensor = getInput(1);
    TensorPtr zeros = gradOut->clone();
    zeros->setName(nodeNamePrefix + "_zeros");

    TensorVector memsetInputs({});
    if (shapeTensor != nullptr && shapeTensor->isShapeTensor())
    {
        memsetInputs.push_back(shapeTensor);
    }
    TensorVector newInputs = {zeros, getInput(0)};
    for (unsigned i = 2; i < m_inputs.size(); i++)  // dynamic parameters
    {
        newInputs.push_back(getInput(i));
    }

    NodePtr memset      = NodeFactory::createNode(memsetInputs,
                                                  {zeros},
                                             nullptr,
                                             NodeFactory::memsetNodeTypeName,
                                             nodeNamePrefix + "_memset");
    NodePtr sliceInsert = NodeFactory::createNode(newInputs,
                                                  {gradOut},
                                                  &m_params,
                                                  NodeFactory::sliceInsertNodeTypeName,
                                                  nodeNamePrefix + "_insert");
    if (!m_enableFcdExpansion)
    {
        disableFcdExpansion(*dynamic_cast<SliceNode*>(sliceInsert.get()));
    }
    retNodes.push_back(sliceInsert);
    retNodes.push_back(memset);
    return retNodes;
}

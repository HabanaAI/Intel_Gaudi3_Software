#include "defs.h"
#include "fcd_ops_utils.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "physical_split_node.h"
#include "types_exception.h"

#include <cstring>

PhysicalSplitNode::PhysicalSplitNode(const TensorVector& in,
                                     const TensorVector& out,
                                     std::string_view    name,
                                     UserParams          params)
: MultiNode(in, out, name, TYPE_PHYSICAL_SPLIT, SIF_PHYSICAL_SPLIT)
{
    setParams(params, sizeof(unsigned));
}

NodePtr PhysicalSplitNode::createNode(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name)
{
    HB_ASSERT(guid == NodeFactory::physicalSplitNodeTypeName, "Invalid node GUID!");
    return NodePtr(new PhysicalSplitNode(inputs, outputs, name, userParams));
}

void PhysicalSplitNode::setParams(UserParams userParams, unsigned userParamsSize)
{
    if (userParamsSize != sizeof(unsigned) && userParamsSize != 0)
    {
        LOG_ERR(HABANA_NODE, "PhysicalSplitNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    m_splitDim = *reinterpret_cast<unsigned*>(userParams);
    LOG_TRACE(HABANA_NODE, "PhysicalSplitNode name - {}, params - dim={}", getNodeName(), m_splitDim);
}

bool PhysicalSplitNode::validateNode() const
{
    if (m_outputs.size() <= 1 || m_inputs.size() != 2)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting >1 outputs and 2 inputs)");
        return false;
    }

    // TODO check for host_shape_tensor
    if (!m_inputs[1]->isHostShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Second input must be a host shape tensor!");
        return false;
    }

    auto nDims = m_outputs[0]->getDim();

    if (m_splitDim >= nDims)
    {
        LOG_ERR(HABANA_NODE, "Split dimension is outside the shape");
        return false;
    }

    auto operands = getOperands();

    for (const auto& op : operands)
    {
        if (!op->isShapeTensor() && op->getDim() != nDims)
        {
            LOG_ERR(HABANA_NODE, "Output shapes do not match");
            return false;
        }
    }

    // Do not check dynamic sizes of inputs vs the output.
    // They are not necessarily up-to-date at node creation time.

    return Node::validateNode();
}

bool PhysicalSplitNode::validateDynamicShapes() const
{
    if (isDynamicShape())
    {
        // Dynamic split node must have 2 inputs
        if (m_inputs.size() != 2 || m_inputs[1] == nullptr || !m_inputs[1]->isHostShapeTensor())
        {
            LOG_ERR(HABANA_NODE,
                    "Node {} has invalid number of inputs: dynamic shapes require a host shape tensor at index 1",
                    getNodeName());
            return false;
        }
    }
    return true;
}

bool PhysicalSplitNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

NodePtr PhysicalSplitNode::clone() const
{
    return NodePtr(new PhysicalSplitNode(*this));
}

NodeList PhysicalSplitNode::extract()
{
    NodeList     result;
    TensorVector allClones;
    NodeVector   extractShapeNodes;

    // Create a DMA node for each of the inputs
    for (unsigned i = 0; i < m_outputs.size(); ++i)
    {
        TensorVector outputs    = {m_outputs[i]};
        auto         inputClone = m_outputs[i]->clone(false, false);
        inputClone->setConnectAtomicNodes();
        TensorVector inputs = {inputClone};
        allClones.push_back(inputClone);

        synPhysicalConcatSplitSubnodeParams params {m_splitDim, i, true};
        NodePtr extractShapeNode = FcdOpsUtils::createExtractShapeNode(m_outputs[i]);
        extractShapeNodes.push_back(extractShapeNode);
        result.push_back(extractShapeNode);
        NodePtr dmaNode = NodeFactory::createNode(inputs,
                                                  outputs,
                                                  &params,
                                                  NodeFactory::getPhysicalSplitConcatSubNodeGUID(),
                                                  fmt::format("{}_sub_{}", getNodeName(), i));

        // save original size to dynamic dma node for later use (validation during patching)
        dynamic_cast<NodeWithParentInfo*>(dmaNode.get())->setParentInfo(inputs[INPUT_TENSOR]->getTotalSizeInBytes());

        for (auto k = 0; k < i; ++k)
        {
            dmaNode->addInput(k + 1, extractShapeNodes[k]->getOutput(0));
        }
        result.push_back(dmaNode);
    }

    synSplitParams logicalNodeParams = {m_splitDim};

    NodePtr logicalSplitNode = NodeFactory::createNode(m_inputs,
                                                       allClones,
                                                       &logicalNodeParams,
                                                       NodeFactory::dynamicSplitNodeTypeName,
                                                       fmt::format("{}_final", getNodeName()));

    result.push_back(logicalSplitNode);

    return result;
}

SifNodeParams PhysicalSplitNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(&m_splitDim);
}

size_t PhysicalSplitNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_splitDim);
}
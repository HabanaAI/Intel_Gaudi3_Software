#include "physical_concat_node.h"

#include "defs.h"
#include "fcd_ops_utils.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "types_exception.h"

#include <cstring>

PhysicalConcatNode::PhysicalConcatNode(const TensorVector& in,
                                       const TensorVector& out,
                                       std::string_view    name,
                                       UserParams          userParams)
: MultiNode(in, out, name, TYPE_PHYSICAL_CONCAT, SIF_CONCATENATE)
{
    setParams(userParams, sizeof(unsigned));
}

NodePtr PhysicalConcatNode::createNode(const TensorVector& inputs,
                                       const TensorVector& outputs,
                                       UserParams          userParams,
                                       std::string_view    guid,
                                       std::string_view    name)
{
    HB_ASSERT(guid == NodeFactory::physicalConcatNodeTypeName, "Invalid node GUID!");
    return NodePtr(new PhysicalConcatNode(inputs, outputs, name, userParams));
}

void PhysicalConcatNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    HB_ASSERT(userParams != nullptr, "Null user params!");
    if (userParamsSize != sizeof(unsigned))
    {
        LOG_ERR(HABANA_NODE, "PhysicalConcatNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(unsigned));
    }
    m_concatDim = *reinterpret_cast<unsigned*>(userParams);
    LOG_TRACE(HABANA_NODE, "PhysicalConcatNode name - {}, params - dim={}", getNodeName(), m_concatDim);
}

bool PhysicalConcatNode::validateNode() const
{

    if (m_inputs.size() <= 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting >1 inputs and 1 output)");
        return false;
    }


    auto nDims = m_inputs[0]->getDim();

    if (m_concatDim >= nDims)
    {
        LOG_ERR(HABANA_NODE, "Concat dimension is outside the shape");
        return false;
    }

    auto operands = getOperands();

    for (const auto& op: operands)
    {
        if (op->getDim() != nDims)
        {
            LOG_ERR(HABANA_NODE, "Input shapes do not match");
            return false;
        }
    }

    // Do not check dynamic sizes of inputs vs the output.
    // They are not necessarily up-to-date at node creation time.

    return Node::validateNode();
}

bool PhysicalConcatNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

NodePtr PhysicalConcatNode::clone() const
{
    return NodePtr(new PhysicalConcatNode(*this));
}

NodeList PhysicalConcatNode::extract()
{
    NodeList result;
    TensorVector allClones;
    NodeVector   extractShapeNodes;

    // Create a DMA node for each of the inputs except possibly the shape

    unsigned numInputs      = m_inputs.size();
    bool     hasShapeTensor = getInputs().back()->isShapeTensor();

    if (hasShapeTensor)
    {
        --numInputs;
    }
    for (unsigned i = 0; i < numInputs; ++i)
    {
        TensorVector inputs = { m_inputs[i] };
        auto outputClone = m_inputs[i]->clone(false, false);
        outputClone->setConnectAtomicNodes();
        TensorVector outputs = { outputClone };
        allClones.push_back(outputClone);

        synPhysicalConcatSplitSubnodeParams params {m_concatDim, i, false};

        // no need shapeNode for the last tensor
        if (i < m_inputs.size() - 1)
        {
            NodePtr extractShapeNode = FcdOpsUtils::createExtractShapeNode(m_inputs[i]);
            extractShapeNodes.push_back(extractShapeNode);
            result.push_back(extractShapeNode);
        }

        NodePtr dmaNode = NodeFactory::createNode(inputs,
                                                  outputs,
                                                  &params,
                                                  NodeFactory::getPhysicalSplitConcatSubNodeGUID(),
                                                  fmt::format("{}_sub_{}", getNodeName(), i));

        // save original size to dynamic dma node for later use (validation at patching)
        dynamic_cast<NodeWithParentInfo*>(dmaNode.get())->setParentInfo(m_outputs[0]->getTotalSizeInBytes());

        for (auto k = 0; k < i; ++k)
        {
            dmaNode->addInput(k + 1, extractShapeNodes[k]->getOutput(0));
        }
        result.push_back(dmaNode);
    }

    synConcatenateParams logicalNodeParams = {m_concatDim};

    if (hasShapeTensor)
    {
        allClones.push_back(m_inputs.back());
    }

    NodePtr logicalConcatNode = NodeFactory::createNode(allClones,
                                                        m_outputs,
                                                        &logicalNodeParams,
                                                        NodeFactory::concatenateNodeLogicalInternalTypeName,
                                                        fmt::format("{}_final", getNodeName()));

    result.push_back(logicalConcatNode);

    return result;
}

SifNodeParams PhysicalConcatNode::getShapeInferenceFunctionUserParams()
{
    if (!m_metadata.is_set())
    {
        m_metadata.set({m_concatDim, getInputs().back()->isShapeTensor()});
    }
    return (SifNodeParams)&m_metadata.value();
}

size_t PhysicalConcatNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifConcatenateMetadata);
}

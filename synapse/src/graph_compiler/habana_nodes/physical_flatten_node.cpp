#include "data_type_utils.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "tpc_node.h"
#include "types_exception.h"
#include "utils.h"

PhysicalFlattenNode::PhysicalFlattenNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          flattenParams,
                                         std::string_view    name)
: MultiNode(inputs, outputs, name, TYPE_PHYSICAL_FLATTEN)
{
    setParams(flattenParams, sizeof(synFlattenParams));
    m_shapeInferenceFunctionID = ShapeFuncID::SIF_FLATTEN;
}

NodePtr PhysicalFlattenNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new PhysicalFlattenNode(inputs, outputs, userParams, name));
}

void PhysicalFlattenNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(synFlattenParams) && userParamsSize != 0)
    {
        LOG_ERR(HABANA_NODE, "PhysicalFlattenNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    m_flattenParams = *(synFlattenParams*)userParams;
    LOG_TRACE(HABANA_NODE, "PhysicalFlattenNode name - {}, params - dim={}", getNodeName(), m_flattenParams.axis);
}

NodeList PhysicalFlattenNode::extract()
{
    HB_ASSERT(false, "Habana graph and multinode dependencies required");
}

NodeList PhysicalFlattenNode::extract(const HabanaGraph& g, MultiNode::MultiNodeDependencies& deps)
{
    NodeList result;

    TensorPtr serializedTensor = m_inputs[0]->clone(false, false);

    NodePtr serializeNode = NodeFactory::createNode({m_inputs[0]},
                                                    {serializedTensor},
                                                    nullptr,
                                                    NodeFactory::getSerializeNodeGUID(),
                                                    getNodeName() + "_serialize");

    TensorPtr flattenedTensor = m_outputs[0]->clone(false, false);

    FlattenNode::setForceLogicalFlag(m_flattenParams.axis);
    NodePtr flattenNode = NodeFactory::createNode({serializedTensor},
                                                  {flattenedTensor},
                                                  &m_flattenParams,
                                                  NodeFactory::flattenNodeTypeName,
                                                  getNodeName() + "_logical_flatten");

    NodePtr deserializeNode = NodeFactory::createNode({flattenedTensor},
                                                      {m_outputs[0]},
                                                      nullptr,
                                                      NodeFactory::getDeserializeNodeGUID(),
                                                      getNodeName() + "_deserialize");

    result.push_back(serializeNode);
    result.push_back(flattenNode);
    result.push_back(deserializeNode);

    deps.push_back({serializeNode, deserializeNode, Tensor::ControlEdgeType::SYNC});
    return result;
}

NodePtr PhysicalFlattenNode::clone() const
{
    return NodePtr(new PhysicalFlattenNode(*this));
}

bool PhysicalFlattenNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

bool PhysicalFlattenNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }

    unsigned outTotalElem(getOutput(TENSOR_OFM)->getDenseSizeInElements());
    unsigned inTotalElem(getInput(TENSOR_IFM)->getDenseSizeInElements());

    if (outTotalElem != inTotalElem)
    {
        LOG_ERR(HABANA_NODE,
                "Output tensor and input tensor of {} doesn't match in elements' count ( {} , {} )",
                getNodeName(),
                outTotalElem,
                inTotalElem);
        return false;
    }

    return true;
}

SifNodeParams PhysicalFlattenNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams)&m_flattenParams;
}

size_t PhysicalFlattenNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_flattenParams);
}

#include "tf_batch_norm_node.h"

#include "graph_traits.h"
#include "habana_graph.h"
#include "moments_node.h"

#include "types_exception.h"
#include "types_exception.h"

TfBatchNormNode::TfBatchNormNode(const TensorVector& in,
                                 const TensorVector& out,
                                 std::string_view    name,
                                 UserParams          params)
: Node(in, out, name, TYPE_TF_BATCH_NORM, SIF_TF_BATCH_NORM)
{
    //This node is just for the semantics
    setParams(params, sizeof(synTfBatchNormalizationParams));
}

NodePtr TfBatchNormNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    return NodePtr(new TfBatchNormNode(inputs, outputs, name, userParams));
}

void TfBatchNormNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "TfBatchNormNode userParams is null");
        throw InvalidNodeParamsException(getNodeName(), "userParams");
    }
    if (userParamsSize != sizeof(synTfBatchNormalizationParams))
    {
        LOG_ERR(HABANA_NODE, "TfBatchNormNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synTfBatchNormalizationParams));
    }
    m_params = *(synTfBatchNormalizationParams*)userParams;
    LOG_TRACE(HABANA_NODE,
              "TfBatchNormNode name - {}, Node params - variance_epsilon={}",
              getNodeName(),
              m_params.variance_epsilon);
}

const synTfBatchNormalizationParams& TfBatchNormNode::getParams()
{
    return m_params;
}

void TfBatchNormNode::printParamsRawData() const
{
    Node::printParamsRawData((void*)&m_params, sizeof(m_params));
}

bool TfBatchNormNode::validateNode() const
{
    /* Inputs:
     *      0: X (IFM)
     *      1: mean
     *      2: variance
     *      3: Offset (beta)
     *      4: Scale (gamma)
     * Outputs:
     *      0: Y - normalized X (OFM)
     * Params:
     *      epsilon
     */
    if (m_inputs.size() != 5 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "TfBatchNormNode Invalid number of operands (expecting 5 inputs and 1 output)");
        return false;
    }

    return Node::validateNode();
}

NodePtr TfBatchNormNode::clone() const
{
    return NodePtr(new TfBatchNormNode(*this));
}

bool TfBatchNormNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

TfFusedBatchNormGradNode::TfFusedBatchNormGradNode(const TensorVector& in,
                                                   const TensorVector& out,
                                                   std::string_view    name,
                                                   UserParams          userParams)
: Node(in, out, name, TYPE_TF_FUSED_BATCH_NORM_GRAD, SIF_NO_SUPPORT)
{
    //This node is just for the semantics
    setParams(userParams, sizeof(synTfBatchNormalizationParams));
}

NodePtr TfFusedBatchNormGradNode::createNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             std::string_view    guid,
                                             std::string_view    name)
{
    return NodePtr(new TfFusedBatchNormGradNode(inputs, outputs, name, userParams));
}

void TfFusedBatchNormGradNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "TfBatchNormNode userParams is null");
        throw InvalidNodeParamsException(getNodeName(), "userParams");
    }
    if (userParamsSize != sizeof(synTfBatchNormalizationParams))
    {
        LOG_ERR(HABANA_NODE, "TfFusedBatchNormGradNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synTfBatchNormalizationParams));
    }
    m_params = *(synTfBatchNormalizationParams*)userParams;
    LOG_TRACE(HABANA_NODE,
              "TfFusedBatchNormGradNode name - {}, Node params - variance_epsilon={}",
              getNodeName(),
              m_params.variance_epsilon);
}

const synTfBatchNormalizationParams& TfFusedBatchNormGradNode::getParams()
{
    return m_params;
}

void TfFusedBatchNormGradNode::printParamsRawData() const
{
    Node::printParamsRawData((void*)&m_params, sizeof(m_params));
}

bool TfFusedBatchNormGradNode::validateNode() const
{
    /* Inputs:
     *      0: Grad Input (dY)
     *      1: X (IFM)
     *      2: Scale (gamma)
     *      3: saved mean in
     *      4: saved variance in
     * Outputs:
     *      0: dX - Gradient of X
     *      1: dScale - Gradient of gamma
     *      2: dBias - Gradient of gamma
     * Params:
     *      epsilon
     */
    if (m_inputs.size() != 5 || m_outputs.size() != 3)
    {
        LOG_ERR(HABANA_NODE, "TfFusedBatchNormGradNode Invalid number of operands (expecting 5 inputs and 3 output)");
        return false;
    }

    return Node::validateNode();
}

NodePtr TfFusedBatchNormGradNode::clone() const
{
    return NodePtr(new TfFusedBatchNormGradNode(*this));
}

bool TfFusedBatchNormGradNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

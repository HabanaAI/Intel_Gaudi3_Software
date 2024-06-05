#include "aggregation_node.h"

#include "node_io_manager.h"
#include "types_exception.h"

AggregationNode::AggregationNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 std::string_view    name,
                                 AliasDirection      direction,
                                 eNodeType           type,
                                 ShapeFuncID         sifId,
                                 UserParams          userParams)
: BaseClass(inputs, outputs, name, direction, type, sifId)
{
    setParams(userParams, sizeof(unsigned));
}

void AggregationNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(unsigned))
    {
        LOG_ERR(HABANA_NODE, "AggregationNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(unsigned));
    }
    m_aggDim = *reinterpret_cast<unsigned*>(userParams);
    LOG_TRACE(HABANA_NODE, "AggregationNode name - {}, params - dim={}", getNodeName(), m_aggDim);
}

bool AggregationNode::validateAggregation(const Node*     node,
                                          const TensorVector& aggTensorsVec,
                                          const TensorVector& aggregations,
                                          unsigned            aggDim)
{
    if (aggTensorsVec.size() != 1)
    {
        LOG_ERR(HABANA_NODE,
                "{} Node {}: Invalid number of operands (expecting 1 aggregated tensor)",
                node->getNodeTypeStr(),
                node->getNodeName());
        return false;
    }

    const TensorPtr& aggTensor = aggTensorsVec[0];

    if (aggDim >= aggTensor->getDim())
    {
        LOG_ERR(HABANA_NODE,
                "{} Node {}: Trying to aggregate non-existent dimension {} of aggregated tensor {}",
                node->getNodeTypeStr(),
                node->getNodeName(),
                aggDim,
                aggTensor->getName());
        return false;
    }

    if (aggregations.size() == 0)
    {
        LOG_ERR(HABANA_NODE,
                "{} Node {}: Invalid number of operands (expecting at least 1 aggregations)",
                node->getNodeTypeStr(),
                node->getNodeName());
        return false;
    }

    TensorPtr temp = aggregations[0];
    bool squeezed = false;
    unsigned expectedDimNum = aggTensor->getDim();
    TSize aggregateDimSize = 0;
    if (temp->getDim() == expectedDimNum - 1)
    {
        --expectedDimNum;
        squeezed = true;
        aggregateDimSize = aggregations.size();
    }

    bool hasShapeTensor = false;

    //Validate match in geometry in other dimensions and calculate aggregate dimension size
    for (TensorPtr t : aggregations)
    {
        if (t->isShapeTensor())
        {
            hasShapeTensor   = true;
            aggregateDimSize = t->getSizeInElements(aggDim);
            break;
        }
        if (t->getDim() != expectedDimNum)
        {
            LOG_ERR(HABANA_NODE,
                    "{} Node {}: aggregation tensors have a different number of dimensions. "
                    "{}, tensor {} has {} dims. expected: {} dims",
                    node->getNodeName(),
                    node->getNodeTypeStr(),
                    node->getNodeName(),
                    t->getName(),
                    t->getDim(),
                    expectedDimNum);
            return false;
        }

        for (unsigned d = 0; d < aggTensor->getDim(); ++d)
        {
            if (d == aggDim)
            {
                if (!squeezed)
                {
                    aggregateDimSize += t->getSizeInElements(aggDim);
                }
            }
            else
            {
                unsigned aliasDim = ((d > aggDim) && (squeezed)) ? (d - 1) : d;
                if (t->getSizeInElements(aliasDim) != aggTensor->getSizeInElements(d))
                {
                    LOG_ERR(HABANA_NODE,
                            "{} Node {}: aggregation tensors have a different size along dimension {}",
                            node->getNodeTypeStr(),
                            node->getNodeName(),
                            d);
                    return false;
                }
            }
        }
    }

    if (aggTensor->getSizeInElements(aggDim) != aggregateDimSize)
    {
        LOG_ERR(HABANA_NODE,
                "{} Node {}: Aggregate size ({}) of tensors along aggregate dimension ({}) doesn't match aggregated "
                "tensor size ({}) in that dimension (hasShapeTensor={})",
                node->getNodeTypeStr(),
                node->getNodeName(),
                aggregateDimSize,
                aggDim,
                aggTensor->getStrideInElements(aggDim + 1),
                hasShapeTensor);
        return false;
    }

    return true;
}

void AggregationNode::permuteParams(const PermutationVector& inputPermutations)
{
    for (const auto& p : inputPermutations)
    {
        HB_ASSERT(p == inputPermutations[0], "Cannot convert params. All input permutations should be identical");
    }
    m_aggDim = inputPermutations[0].permuteDim(m_aggDim);
}

bool AggregationNode::validateNode() const
{
    if (!BaseClass::validateNode())
    {
        return false;
    }

    return validateAggregation(this,
                               getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs : m_outputs,
                               getAliasDirection() == OUTPUT_TO_INPUT ? m_outputs : m_inputs,
                               m_aggDim);
}

NStrideArray AggregationNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void AggregationNode::runLogicalOperation() const
{
    const TensorPtr& aggTensor = getAliasDirection() == OUTPUT_TO_INPUT? m_inputs[0] : m_outputs[0];
    const TensorVector& aggregations = getAliasDirection() == OUTPUT_TO_INPUT? m_outputs : m_inputs;

    TStride stride = aggTensor->getElementSizeInBytes();
    unsigned dim    = m_aggDim;
    if (dim > 0)
    {
        stride = aggTensor->getStrideInBytes(dim);
    }
    TSize outSize = 0;
    for (TensorPtr t : aggregations)
    {
        if (unlikely(!t)) continue;
        if (t->isShapeTensor()) continue;

        t->setAsConcatSubTensor(aggTensor, stride * outSize, dim);
        outSize += t->getSizeInElements(dim);
    }
}

bool AggregationNode::isRedundantNode() const
{
    return (getNumOutputs() == 1 && getNumInputs() == 1);
}

void AggregationNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_aggDim, sizeof(m_aggDim));
}

bool AggregationNode::isAliasStrided() const
{
    return (m_aggDim != getRealTensor()->getDim()) || !getRealTensor()->isDenseLayout();
}

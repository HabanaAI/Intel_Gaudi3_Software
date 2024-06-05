#include "aggregate_fcd_node.h"

#include <cstddef>
#include <cstdint>
#include "fcd_ops_utils.h"
#include "graph_traits.h"
#include "synapse_common_types.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "types.h"
#include "habana_graph.h"

static constexpr unsigned AGGREGATION_DIM = 0;

AggregateFcdNode::AggregateFcdNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   std::string_view    name,
                                   eNodeType           type,
                                   ShapeFuncID         sifId)
: MultiNode(inputs, outputs, name, type, sifId) {};

void AggregateFcdNode::printParamsRawData() const
{
    const unsigned aggregateDim = 0;
    BaseClass::printParamsRawData((void*)&aggregateDim, sizeof(aggregateDim));
}

// newFcdDim contain the dim that became dim 0 (FCD),
// since the permutation is cyclic and we know the newFcdDim which fully define the permutation,
// so permutation[dim] is (dim + newFcdDim) % numDims
// to know the new axis we need to know the index 0 of the inverse permutation which is:
// invPermutation[dim] is (dim - newFcdDim) % numDims
// the new axis should be the index 0 of the inverse permutation which is numDims - newFcdDim

// Example: [8,4,3,3] -> [4,3,3,8]
// newFcdDim: 1, numDims = 4, permutation [1,2,3,0], permutation formula: perm[dim] = (dim + 1) % 4
// inverse [3,0,1,2], inverse permutation formula:[dim] = (dim - 1) % 4
// for dim = 0: (-1) % 4 = (4 - 1) % 4 = 4 - 1 = 3
unsigned AggregateFcdNode::getNewDimForAggregate(unsigned newFcdDim) const
{
    return m_inputs[0]->getDim() - newFcdDim;
}

// Returns the total DMA effort (in bytes) needed for executing this AggregateFcdNode
uint64_t AggregateFcdNode::getExpectedCost() const
{
    const TensorVector& aliasedTensors       = (m_inputs.size() > 1) ? m_inputs : m_outputs;
    const unsigned      cacheLineSizeInBytes = m_graphTraits->getHalReader()->getCacheLineSizeInBytes();
    const unsigned      elementSize          = aliasedTensors[0]->getElementSizeInBytes();

    uint64_t cost = 0;
    for (const TensorPtr& tensor : aliasedTensors)
    {
        cost += FcdOpsUtils::calculateExpectedCost(*tensor,
                                                   cacheLineSizeInBytes,
                                                   static_cast<uint64_t>(tensor->getSizeInElements(0)) * elementSize);
    }

    return cost;
}

bool AggregateFcdNode::isDataMovementMultiNode() const
{
    // if this is actually just a logical operation, don't treat it as a multi node - extract as soon as possible
    return !shouldUseLogicalAggregate();
}

bool AggregateFcdNode::shouldUseLogicalAggregate() const
{
    unsigned numDims      = m_inputs[0]->getDim();
    uint64_t expectedCost = getExpectedCost();

    FcdOpsUtils::ShiftTransposesForFcdOpsResults transposesResults;
    findBestDimForTranspose(*(m_graphTraits->getHalReader()), 0, m_inputs, m_outputs, transposesResults);

    if (!GCFG_OPTIMIZE_SPLIT_CONCAT_ON_FCD.value() || (transposesResults.expectedCost >= expectedCost) ||
        (numDims == 1))
    {
        return true;
    }
    return false;
}

NodeList AggregateFcdNode::extract()
{
    HB_ASSERT(m_type == Node::TYPE_INTERNAL_CONCAT || m_type == Node::TYPE_INTERNAL_SPLIT, "unexpected node type");
    bool isConcat = m_type == Node::TYPE_INTERNAL_CONCAT;

    const char* guid = isConcat ? NodeFactory::concatenateNodeInternalTypeName : NodeFactory::splitNodeInternalTypeName;

    if (shouldUseLogicalAggregate())
    {
        // Don't optimize, create regular split node
        synSplitParams splitParams;
        splitParams.axis = 0;
        return {NodeFactory::createNode(getInputs(), getOutputs(), &splitParams, guid, getNodeName())};
    }

    // Add high performance transpose sequence before and after the split operation to avoid low utilization DMAs
    FcdOpsUtils::ShiftTransposesForFcdOpsResults transposesResults =
        FcdOpsUtils::createOppositeShiftTransposes(*(m_graphTraits->getHalReader()), m_name, m_inputs, m_outputs);

    NodeList ret = transposesResults.newNodes;

    synSplitParams params = {.axis = getNewDimForAggregate(transposesResults.newFcdDim)};

    ret.push_back(NodeFactory::createNode(transposesResults.newInputs,
                                          transposesResults.newOutputs,
                                          &params,
                                          guid,
                                          getNodeName() + "/transposed"));

    return ret;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////

SplitFcdNode::SplitFcdNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: AggregateFcdNode(inputs, outputs, name, Node::TYPE_INTERNAL_SPLIT, SIF_SPLIT)
{
}

NodePtr SplitFcdNode::clone() const
{
    return NodePtr(new SplitFcdNode(*this));
}

bool SplitFcdNode::validateNode() const
{
    if (!MultiNode::validateNode())
    {
        return false;
    }

    return AggregationNode::validateAggregation(this, m_inputs, m_outputs, 0);
}

bool SplitFcdNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

SifNodeParams SplitFcdNode::getShapeInferenceFunctionUserParams()
{
    return SplitNode::getShapeInferenceFunctionUserParams(m_sifMetadataBuffer,
                                                          getShapeInferenceFunctionUserParamsSize(),
                                                          AGGREGATION_DIM,
                                                          m_outputs);
}

size_t SplitFcdNode::getShapeInferenceFunctionUserParamsSize() const
{
    return SplitNode::getShapeInferenceFunctionUserParamsSize(m_outputs.size());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

ConcatFcdNode::ConcatFcdNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: AggregateFcdNode(inputs, outputs, name, Node::TYPE_INTERNAL_CONCAT, SIF_CONCATENATE)
{
}

NodePtr ConcatFcdNode::clone() const
{
    return NodePtr(new ConcatFcdNode(*this));
}

bool ConcatFcdNode::validateNode() const
{
    if (!MultiNode::validateNode())
    {
        return false;
    }

    return AggregationNode::validateAggregation(this, m_outputs, m_inputs, 0);
}

bool ConcatFcdNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

SifNodeParams ConcatFcdNode::getShapeInferenceFunctionUserParams()
{
    return ConcatenateNode::getShapeInferenceFunctionUserParams(m_metadata, AGGREGATION_DIM, m_inputs);
}

size_t ConcatFcdNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifConcatenateMetadata);
}

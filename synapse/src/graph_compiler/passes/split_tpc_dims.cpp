#include <memory>
#include <algorithm>

#include <vector>
#include "defs.h"
#include "range.h"
#include "tpc_node.h"
#include "habana_graph.h"
#include "synapse_common_types.h"

static const int c_max_tensor_dims = 5;

static std::pair<unsigned, unsigned> gcTensorIndexToUserTensor(const TensorVector& nodeTensors, unsigned tensorIndex)
{
    unsigned startIndex = 0;
    for (int i = 0; i < tensorIndex; i++)
    {
        unsigned used = div_round_up(nodeTensors[i]->getDim(), SYN_MAX_TENSOR_DIM);
        // At least 1
        startIndex += std::max<unsigned>(used, 1);
    }

    std::pair<unsigned, unsigned> ret;

    // Put inside the vector the range startIndex..startIndex+usedTensors-1
    unsigned usedTensors = div_round_up(nodeTensors[tensorIndex]->getDim(), SYN_MAX_TENSOR_DIM);
    // At least 1
    usedTensors = std::max<unsigned>(usedTensors, 1);
    ret.first   = startIndex;
    ret.second  = startIndex + usedTensors - 1;
    return ret;
}

static unsigned getLastProducedInputIdx(HabanaGraph&                                  g,
                                        const TensorVector&                           nodeInputTensors,
                                        const tpc_lib_api::HabanaKernelInstantiation& instance)
{
    int      latestSchedNodeIdx = -1;
    unsigned latestInputIdx     = 0;
    for (unsigned tensorIndex = 0; tensorIndex < nodeInputTensors.size(); ++tensorIndex)
    {
        auto range = gcTensorIndexToUserTensor(nodeInputTensors, tensorIndex);

        bool allRequired = true;
        for (int i = range.first; i < range.second; i++)
        {
            HB_ASSERT(i < MAX_TENSOR_NR, "Invalid index {}", i);
            if (!instance.inputTensorAccessPattern[i].allRequired) allRequired = false;
        }
        if (allRequired) continue;

        pNode nodeProducer = g.getTensorProducer(nodeInputTensors[tensorIndex]);
        if (!nodeProducer) continue;
        if (nodeProducer->getExecutionOrderedIndex() > latestSchedNodeIdx)
        {
            latestSchedNodeIdx = nodeProducer->getExecutionOrderedIndex();
            latestInputIdx     = tensorIndex;
        }
    }
    return latestInputIdx;
}

void splitTpcDims(HabanaGraph& g, TPCNode& tpcNode)
{
    NodeAnnotation& ann = tpcNode.getNodeAnnotation();

    const auto& instance = tpcNode.getInstance();
    unsigned    nDims    = instance.indexSpaceRank;

    // Get the inputs of the node
    const TensorVector& nodeInputTensors = tpcNode.getInputs();

    // In case there are no inputs - default to nDims->0
    if (nodeInputTensors.empty())
    {
        // Update the annotation
        ann.tpcSplitDims.resize(nDims);

        for (unsigned i = 0; i < nDims; ++i)
        {
            ann.tpcSplitDims[i] = (nDims - 1) - i;
        }
        return;
    }

    bool canSplitAllDims = true;
    // Iterate over inputs and get the index of the input that was produced last (according to execution order)
    unsigned lastProducedInputIdx = getLastProducedInputIdx(g, nodeInputTensors, instance);
    auto     usedIndicies         = gcTensorIndexToUserTensor(nodeInputTensors, lastProducedInputIdx);
    unsigned inputTensorNumDims =
        std::min<unsigned>(nodeInputTensors[lastProducedInputIdx]->getDim(), SYN_MAX_TENSOR_DIM);
    unsigned userTensorIndex      = usedIndicies.first;
    unsigned relativeIndex        = 0;
    if (usedIndicies.first != usedIndicies.second)
    {
        canSplitAllDims = false;  // Can split in nD
        // Check only one has not allRequired
        std::vector<unsigned> notAllRequired;
        for (int i = usedIndicies.first; i <= usedIndicies.second; i++)
        {
            if (!instance.inputTensorAccessPattern[i].allRequired)
            {
                notAllRequired.push_back(i);
            }
        }
        HB_ASSERT(notAllRequired.size() <= 1, "We have more than one user tensor which is not all required");
        if (notAllRequired.empty())
        {
            userTensorIndex = usedIndicies.first;
        }
        else
        {
            userTensorIndex = notAllRequired.front();
        }
        relativeIndex = userTensorIndex - usedIndicies.first;
        inputTensorNumDims =
            std::min<unsigned>(nodeInputTensors[lastProducedInputIdx]->getDim() - relativeIndex * SYN_MAX_TENSOR_DIM,
                               SYN_MAX_TENSOR_DIM);
    }
    auto& inputTensorDims = instance.inputTensorAccessPattern[userTensorIndex].mapping;

    for (unsigned i = 0; i < inputTensorNumDims; ++i)
    {
        unsigned newDim = inputTensorDims[inputTensorNumDims - 1 - i].indexSpaceDim;
        if (std::find(ann.tpcSplitDims.begin(), ann.tpcSplitDims.end(), newDim) == ann.tpcSplitDims.end())
        {
            ann.tpcSplitDims.push_back(newDim);
        }
        else
        {
            LOG_WARN(GC,
                     "Dimension {} in tensor {} had duplicated dims in glue code's access pattern (num dims: {})",
                     newDim,
                     nodeInputTensors[lastProducedInputIdx]->getName(),
                     inputTensorNumDims);
        }
    }

    // In case the index space dims and the tensor dims does not match
    if (nDims < inputTensorNumDims)
    {
        ann.tpcSplitDims.clear();
    }

    if (canSplitAllDims)
    {
        for (unsigned i = 0; i < nDims; ++i)
        {
            unsigned newDim = (nDims - 1) - i;
            if (std::find(ann.tpcSplitDims.begin(), ann.tpcSplitDims.end(), newDim) == ann.tpcSplitDims.end())
            {
                ann.tpcSplitDims.push_back(newDim);
            }
        }
    }
}


// Decide on the order of index-space partitioning if possible. Mandatory first split dim, like "FCD First" or
// "Mandatory Split Dim" which is specified by the TPC lib, are enforced by the work distribution pass.
bool splitTPCDims(HabanaGraph& g)
{
    for (const pNode& n : g.getExeSortedNodes())
    {
        if (!HabanaGraph::runsOnTPC(n)) continue;
        std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(n);
        HB_ASSERT(tpcNode != nullptr, "invalid node type");
        splitTpcDims(g, *tpcNode);
    }

    return true;
}

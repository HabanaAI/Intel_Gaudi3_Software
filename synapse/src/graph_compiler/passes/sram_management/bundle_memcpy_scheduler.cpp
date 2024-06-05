#include "habana_graph.h"
#include "node.h"
#include "tensor.h"
#include "bundle_memcpy_scheduler.h"
#include "tensor_view_node.h"
#include "max_path_scheduler.h"
#include <stack>

BundleMemcpyScheduler::BundleMemcpyScheduler()
        : m_opIdx(0)
{
}

// used to create maps with BufferPlacement as key
bool BundleMemcpyScheduler::BufferPlacement::operator<(const BufferPlacement& o) const
{
    if (operationIndex < o.operationIndex) return true;
    if (operationIndex > o.operationIndex) return false;
    if (tensorIndex < o.tensorIndex) return true;
    if (tensorIndex > o.tensorIndex) return false;
    return isInput < o.isInput;
}

bool BundleMemcpyScheduler::TensorBuffer::operator<(const TensorBuffer& o) const
{
    return bufferId < o.bufferId;
}

void BundleMemcpyScheduler::addOperationBuffer(const pSliceReference& sliceRef,
                                               uint64_t               opIdx,
                                               bool                   isInput,
                                               uint64_t               sectionIdx,
                                               uint32_t               tensorIdx)
{
    if (!sliceRef->operand->resideInSRAM && !sliceRef->operand->originalTensor->inSram())
    {
        return;
    }
    BufferId        buffId      = sectionIdx;
    uint32_t        bufferLevel = sliceRef->operand->numOfBuffers;
    BufferPlacement placement   = {opIdx, tensorIdx, isInput};
    auto            it          = m_bufferMapping.find(placement);
    if (it != m_bufferMapping.end())
    {
        HB_ASSERT(it->second.bufferLevel == bufferLevel, "Access to same buffer with different level");
        HB_ASSERT(it->second.bufferId == buffId, "Access to same buffer with different ids");
    }
    else
    {
        m_bufferMapping[placement] = TensorBuffer({buffId, bufferLevel});
    }
}

// check if all the nodes in the bundle are of the same engine - in that case no need to schedule anything
bool BundleMemcpyScheduler::areAllSameEngine(const NodeVector& nodes)
{
    bool allMME = true;
    bool allTPC = true;
    for (const NodePtr& n : nodes)
    {
        if (n->isLogicalOperation() && !n->isDebug()) continue;
        allTPC &= HabanaGraph::runsOnTPC(n);
        allMME &= HabanaGraph::runsOnMME(n);
    }
    return allTPC || allMME;
}

void BundleMemcpyScheduler::scheduleBundleGraph(HabanaGraph& bundleGraph)
{
    /*
     * assume that the preOrder, decided by the sram slicer, is correct - each slice is scheduled until completion
     * before continuing to the next slice.
     *
     * Inner scheduling in the bundle:
     * 1 - create node dependencies according to place in buffers
     * 2 - schedule the bundle according to the new dependencies
     */
    const NodeVector& preOrder = bundleGraph.getExeSortedNodes();  // preOrder - decided by sram slicer.
    if (areAllSameEngine(preOrder) || GCFG_ENABLE_PIPELINE_MANAGEMENT_SRAM_OVERRIDE.value())
    {
        // in that case there can be no memory dependency, it is best to use the pre order
        setOperationIndices(preOrder);
    }
    else
    {
        createNodeDependencies(bundleGraph);
        NodeList   scheduleNodeList = scheduleNodes(bundleGraph);
        NodeVector scheduleNodeVec(scheduleNodeList.begin(), scheduleNodeList.end());
        setOperationIndices(scheduleNodeVec);  // schedule nodes according to heuristic (specified in comparator).
    }
    printBundleExecutionOrder(bundleGraph);
}

void BundleMemcpyScheduler::setOperationIndices(const NodeVector& nodes)
{
    for (const NodePtr& n : nodes)
    {
        n->getNodeAnnotation().bundleInfo->operationIndex = m_opIdx++;
    }
}

void BundleMemcpyScheduler::printBundleExecutionOrder(HabanaGraph& graph) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(BE_SLICER)) return;

    graph.invalidateExecutionSchedule();
    const NodeVector& nodes = graph.getExeSortedNodes();
    HB_ASSERT(!nodes.empty(), "graph is empty");
    HB_ASSERT(nodes.front()->getNodeAnnotation().bundleInfo.is_set(), "Non bundle nodes in bundle memcpy scheduler");
    LOG_DEBUG(BE_SLICER, "Execution order for bundle {}", nodes.front()->getNodeAnnotation().bundleInfo->bundleIndex);

    for (const NodePtr& node : nodes)
    {
        LOG_DEBUG(BE_SLICER, "    Node {}, id {}, operation index {}",
                  node->getNodeName(),
                  node->getId(),
                  node->getNodeAnnotation().bundleInfo->operationIndex);
    }
}

bool BundleMemcpyScheduler::isComputeNode(const NodePtr& node)
{
    if (node->isLogicalOperation() && !node->isDebug()) return false;
    switch (node->getNodeType())
    {
        case Node::TYPE_MEMCOPY:
            return false;
        default:
            return true;
    }
}

// we use a local alias mapping since logical operation are haven't ran yet.
// it doesn't have be "correct", just persistant across all logical nodes
TensorMap BundleMemcpyScheduler::createRealTensorMapping(const HabanaGraph& graph)
{
    TensorMap ret;
    for (const NodePtr& node : graph.getNodes())
    {
        if (!node->isLogicalOperation() || node->isDebug() || node->isShapeOperation()) continue;
        auto* logicalNode = dynamic_cast<LogicalOpNode*>(node.get());
        if (logicalNode->canSwapAliasDirection() ||
            logicalNode->getAliasDirection() == AliasDirection::OUTPUT_TO_INPUT)
        {
            HB_ASSERT(logicalNode->getNumInputs() - logicalNode->getNumInputsShapeTensors() == 1,
                      "alias:OUT->IN with more than 1 input. Node {}",
                      logicalNode->getNodeName());
            for (const TensorPtr& t : logicalNode->getOutputs())
            {
                ret[t] = logicalNode->getInput(DATA_TENSOR);
            }
        }
        else
        {
            HB_ASSERT(logicalNode->getNumOutputs() - logicalNode->getNumOutputsShapeTensors() == 1,
                      "alias:IN->OUT with more than 1 output. Node {}",
                      logicalNode->getNodeName());
            for (const TensorPtr& t : logicalNode->getInputs())
            {
                ret[t] = logicalNode->getOutput(DATA_TENSOR);
            }
        }
    }
    return ret;
}

// this is for debug purposes - easy to check if the buffers contain the right tensors
void BundleMemcpyScheduler::printNodeDependencies(const HabanaGraph& graph,
                                                  const std::map<TensorBuffer, TensorList>& bufferToTensors) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(BE_SLICER)) return;

    // print buffers
    for (const auto& bufferToTensor : bufferToTensors)
    {
        LOG_DEBUG(BE_SLICER, "bufferID: {}, bufferLevel: {}", bufferToTensor.first.bufferId,
                  bufferToTensor.first.bufferLevel);
        for (const TensorPtr& t : bufferToTensor.second)
        {
            LOG_DEBUG(BE_SLICER, "\t {}", t->getName());
        }
    }

    if (!LOG_LEVEL_AT_LEAST_TRACE(BE_SLICER)) return;

    // print dependencies
    for (const NodePtr& node : graph.getExeSortedNodes())
    {
        LOG_TRACE(BE_SLICER, "node {} producers:", node->getNodeName());
        for (const NodePtr& prod : m_nodeProducers.at(node))
        LOG_TRACE(BE_SLICER, "\t{}", prod->getNodeName());
        LOG_TRACE(BE_SLICER, "node {} consumers:", node->getNodeName());
        for (const NodePtr& cons : m_nodeConsumers.at(node))
        LOG_TRACE(BE_SLICER, "\t{}", cons->getNodeName());
    }
}

void BundleMemcpyScheduler::createBufferDepedencies(const HabanaGraph& graph, uint32_t opIndex,
                                                    const TensorVector& tensors, bool isInput,
                                                    const TensorMap& realTensorMapping,
                                                    std::map<TensorBuffer, TensorList>& bufferToTensors) const
{
    for (uint32_t index = 0; index < tensors.size(); index++)
    {
        TensorPtr t = tensors[index];
        if (!t || !t->inSram()) continue;
        // get real tensor
        auto realIt = realTensorMapping.find(t);
        while (realIt != realTensorMapping.end())
        {
            t = realIt->second;
            realIt = realTensorMapping.find(t);
        }
        // create buffer placement and push into buffer
        BufferPlacement placement = {opIndex, index, isInput};
        auto it = m_bufferMapping.find(placement);
        HB_ASSERT(it != m_bufferMapping.end(), "placement not found!");
        const TensorBuffer& buffer = it->second;
        TensorList& bufferTensors = bufferToTensors[buffer];
        if (std::find(bufferTensors.begin(), bufferTensors.end(), t) == bufferTensors.end())
        {
            bufferTensors.push_back(t);
        }
    }
}

// create additional dependencies (memory dependencies in buffers)
void BundleMemcpyScheduler::createNodeDependencies(const HabanaGraph& graph)
{
    std::map<TensorBuffer, TensorList> bufferToTensors;
    TensorMap realTensorMapping = createRealTensorMapping(graph);

    for (const NodePtr& node : graph.getExeSortedNodes())
    {
        LOG_TRACE(BE_SLICER, "creating dependencies for node {}",
                  node->getNodeName()); // this log will also show the pre order
        // add initial dependencies from graph
        m_nodeProducers[node] = graph.getNodeProducers(node, Node::eTensorType::TENSOR_TYPE_ALL);
        m_nodeConsumers[node] = graph.getNodeConsumers(node, Node::eTensorType::TENSOR_TYPE_ALL);
        if (!isComputeNode(node)) continue;

        // create a (ordered) list of tensors per buffer
        uint32_t opIndex = node->getNodeAnnotation().bundleInfo->operationIndex;
        createBufferDepedencies(graph, opIndex, node->getOutputs(), /* is input */ false, realTensorMapping, bufferToTensors);
        createBufferDepedencies(graph, opIndex, node->getInputs(), /* is input */ true, realTensorMapping, bufferToTensors);
    }

    // create memory dependencies according to tensors in each buffer (and buffer level)
    for (const auto& bufferToTensor : bufferToTensors)
    {
        uint32_t bufferLevel = bufferToTensor.first.bufferLevel;
        const TensorList& slices = bufferToTensor.second;
        if (slices.size() <= bufferLevel) continue; // all tensors fit in SRAM trivially

        auto blockingIt = slices.begin(); // tensor that is blocking
        auto blockedIt = slices.begin(); // tensor that is blocked
        std::advance(blockedIt, bufferLevel);
        while (blockedIt != slices.end())
        {
            NodeSet blockingNodes = graph.getRealConsumers(*blockingIt);
            NodeSet blockedNodes = graph.getRealProducers(*blockedIt);

            for (const NodePtr& node : blockingNodes)
            {
                m_nodeConsumers[node].insert(blockedNodes.begin(), blockedNodes.end());
            }
            for (const NodePtr& node : blockedNodes)
            {
                m_nodeProducers[node].insert(blockingNodes.begin(), blockingNodes.end());
            }
            blockingIt++;
            blockedIt++;
        }
    }

    if (LOG_LEVEL_AT_LEAST_DEBUG(BE_SLICER))
    {
        printNodeDependencies(graph, bufferToTensors);
    }
}

namespace
{
/*
 * This is the schedule heuristic. there may be a better solution, but it is easy to try new things.
 * also, every schedule will at least be "legal" (considering memory dependencies).
 */
struct InnerBundleComparator
{
    bool operator()(const NodePtr& n1, const NodePtr& n2) const
    {
        // prefer fill nodes if possible
        bool isFill1 = (n1->getNodeType() == Node::TYPE_MEMCOPY) && !n1->getInput(0)->inSram();
        bool isFill2 = (n2->getNodeType() == Node::TYPE_MEMCOPY) && !n2->getInput(0)->inSram();
        if (isFill1 && !isFill2) return true;
        if (!isFill1 && isFill2) return false;

        bool isEvict1 = (n1->getNodeType() == Node::TYPE_MEMCOPY) && !n1->getOutput(0)->inSram();
        bool isEvict2 = (n2->getNodeType() == Node::TYPE_MEMCOPY) && !n2->getOutput(0)->inSram();
        if (!isEvict1 && isEvict2) return true;
        if (isEvict1 && !isEvict2) return false;

        // If both nodes are fill with the same initial operation index (might happen at the snake walking pattern
        // turning points), prefer the one with bigger output, to avoid cases where we run out of SRAM memory due to
        // fragmentation.
        if (isFill1 && isFill2 &&
            (n1->getNodeAnnotation().bundleInfo->operationIndex ==
             n2->getNodeAnnotation().bundleInfo->operationIndex) &&
            (n1->getOutput(0)->getTotalSizeInBytes() != n2->getOutput(0)->getTotalSizeInBytes()))
        {
            return n1->getOutput(0)->getTotalSizeInBytes() > n2->getOutput(0)->getTotalSizeInBytes();
        }

        // fallback - use pre order
        return n1->getExecutionOrderedIndex() < n2->getExecutionOrderedIndex();
    }
};
} // anonymous namespace

class BundleMaxPathScheduler : public MaxPathScheduler
{
public:
    using ConnectivityMap = std::map<NodePtr, NodeSet>;

    BundleMaxPathScheduler(const HabanaGraph*           graph,
                           const MaxPathTieBreakerFunc& cmp,
                           const ConnectivityMap&       getBlockingNodes,
                           const ConnectivityMap&       getBlockedNodes)
    : MaxPathScheduler(graph, cmp), m_getBlockingNodes(getBlockingNodes), m_getBlockedNodes(getBlockedNodes)
    {
    }

protected:
    NodeSet getBlockingNodes(const NodePtr& n) const override { return m_getBlockingNodes.at(n); }
    NodeSet getBlockedNodes(const NodePtr& n) const override { return m_getBlockedNodes.at(n); }

    const ConnectivityMap& m_getBlockingNodes;
    const ConnectivityMap& m_getBlockedNodes;
};

NodeList BundleMemcpyScheduler::scheduleNodes(const HabanaGraph& graph)
{
    NodeList ret;
    graph.getExeSortedNodes();
    return BundleMaxPathScheduler(&graph, InnerBundleComparator(), m_nodeProducers, m_nodeConsumers).scheduleNodes();
}

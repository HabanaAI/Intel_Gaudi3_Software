#include "eager_node.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_graph.h"
#include "graph_compiler/habana_nodes/node.h"

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerNode
///////////////////////////////////////////////////////////////////////////////////////////////////

EngineType EagerNode::calcEngineType(const NodePtr& node)
{
    if (!node) return EngineType::INVALID;
    if (HabanaGraph::runsOnTPC(node)) return EngineType::TPC;
    if (HabanaGraph::runsOnMME(node)) return EngineType::MME;
    if (HabanaGraph::isActivationDMA(node)) return EngineType::DMA;
    if (HabanaGraph::runsOnRotator(node)) return EngineType::ROT;
    return EngineType::INVALID;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerNodes
///////////////////////////////////////////////////////////////////////////////////////////////////

EagerNodes::EagerNodes(bool isOriginalGraph, uint64_t tensorSizeThresholdForParallelExecution)
: m_isOriginalGraph(isOriginalGraph), m_tensors(tensorSizeThresholdForParallelExecution)
{
}

EagerNodes::EagerNodes(const EagerNodes& other)
: m_isOriginalGraph(other.m_isOriginalGraph),
  m_isAddingNewNodesEnabled(other.m_isAddingNewNodesEnabled),
  m_tensors(other.m_tensors)
{
    m_nodes.reserve(other.m_nodes.size());
    for (const auto& otherNode : other.m_nodes)
    {
        TensorVector inputs;
        fillTensorVectorForDuplicatedNode(other.m_tensors, otherNode->getInputs(), inputs);
        TensorVector outputs;
        fillTensorVectorForDuplicatedNode(other.m_tensors, otherNode->getOutputs(), outputs);
        auto newNode = otherNode->clone();
        newNode->replaceAllTensors(std::move(inputs), std::move(outputs));
        m_nodes.push_back(std::move(newNode));
#ifndef NDEBUG
        m_nodes.back()->print();
#endif
    }
}

void EagerNodes::fillTensorVectorForDuplicatedNode(const EagerTensorsSetBuilder& srcGraphTensors,
                                                   const TensorVector&           src,
                                                   TensorVector&                 dst)
{
    const auto& dstGraphTensorsVec = m_tensors.getTensors();
    const auto& srcGraphTensorsVec = srcGraphTensors.getTensors();
    dst.reserve(src.size());
    for (const auto& inTensor : src)
    {
        if (!inTensor)
        {
            dst.emplace_back();
            continue;
        }
        auto origIter = std::find(srcGraphTensorsVec.begin(), srcGraphTensorsVec.end(), inTensor);
        EAGER_ASSERT(origIter != srcGraphTensorsVec.end(),
                     "node tensor not found in graph tensors for Eager original graph");
        dst.emplace_back(dstGraphTensorsVec[distance(srcGraphTensorsVec.begin(), origIter)]);
    }
}

void EagerNodes::push_back(const EagerNode& node)
{
    handleNewNode(node);
    m_nodes.push_back(node);
}

void EagerNodes::push_back(EagerNode&& node)
{
    handleNewNode(node);
    m_nodes.push_back(std::move(node));
}

const EagerNode* EagerNodes::findNodeByID(synNodeId nodeID) const
{
    for (const EagerNode& node : m_nodes)
    {
        if (node->getId() == nodeID) return &node;
    }
    return nullptr;
}

void EagerNodes::handleNewNode(const EagerNode& node)
{
    EAGER_ASSERT(m_isAddingNewNodesEnabled, "Adding new nodes to eager nodes list is not allowed");

    // Handle logical operation
    if (node->isLogicalOperation())
    {
        m_tensors.addTensors(node);
        return;
    }

    // Only original graph can have complex GUID nodes that cannot be mapped to a physical engine yet
    EAGER_ASSERT(m_isOriginalGraph || node.getEngineType() != EngineType::INVALID, "Engine type is not supported");
    ++m_physicalNodesNr;

    // Process tensors of the new node and decide if it's relevant to do some other
    // preprocessing for the parallel/serial execution decision.
    // Most of the preprocessing is done on original nodes, however, they can have complex GUID nodes, i.e.
    // no specific engine type. Because of that we do a last decision (parallel or serial) at early stage after
    // downloading nodes to Eager graph.
    // I tried another direction, which is to do most of the preprocessing on that download, but unfortunately
    // The performance was not better.
    if (!m_isOriginalGraph || !m_tensors.allowParallelExecHandling() || areMultipleEnginesUsed())
    {
        // Here tensors are processed without any further logic related to parallel execution decision.
        // !m_isOriginalGraph check: reflects what stated above.
        // !m_tensors.allowParallelExecHandling() check: deals with pure serial execution as dictated by the enviornment
        // variable. areMultipleEnginesUsed() check: when multiple engines are used there is no more need for further
        // decision processing.
        m_tensors.addTensors(node);
    }
    // In addition to process new tensors, the following branch detects if there is any tensor with desired size for the
    // parallel execution decision, if so it returns true to update the score board. Could be a scenario where there are
    // indeed two (or more) different engines, however, the desired tensor size is detected when scoreboard is clear. In
    // this case the decision will be revised ExecScheduler::reorderLast. Keep in mind this is a common case.
    else if (m_tensors.addTensorsWithParallelExecCheck(node))
    {
        // Update used engine for the new node
        const EngineType engine = node.getEngineType();
        if (engine != EngineType::INVALID)  // Can be invalid for complex GUID
        {
            m_usedEnginesScoreBoard.set(static_cast<unsigned>(engine));
        }
    }
}

// Search bwd from a node at nodeIdx for the producer of one of its input tensors
std::optional<size_t> EagerNodes::getInputProducerIdx(const Tensor* tensor, size_t nodeIdx) const
{
    while (nodeIdx-- > 0)
    {
        for (const auto& t : m_nodes[nodeIdx]->getOutputs())
        {
            if (t.get() == tensor) return std::make_optional(nodeIdx);
        }
    }
    return {};
}

std::optional<size_t> EagerNodes::getNextConsumerIdx(const Tensor* tensor, size_t startFrom) const
{
    for (size_t i = startFrom; i < m_nodes.size(); ++i)
    {
        for (const auto& t : m_nodes[i]->getInputs())
        {
            if (t.get() == tensor) return std::make_optional(i);
        }
    }
    return {};
}

bool EagerNodes::hasSingleConsumer(const Tensor* tensor, size_t startFrom) const
{
    auto consumerIdx = getNextConsumerIdx(tensor, startFrom);
    return consumerIdx && !getNextConsumerIdx(tensor, *consumerIdx + 1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerNodesBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////

EagerNodesBuilder::EagerNodesBuilder(bool isOriginalGraph, uint64_t tensorSizeThresholdForParallelExecution)
: EagerNodes(isOriginalGraph, tensorSizeThresholdForParallelExecution)
{
}

void EagerNodesBuilder::disableAddingNewNodes()
{
    EAGER_ASSERT(m_isAddingNewNodesEnabled,
                 "Trying to prevent adding new nodes to eager nodes list, while it's already prevented");
    m_isAddingNewNodesEnabled = false;

    // Stats on tensor names (For recipe allocation) and checks for SRAM and dynamic shape are done on the original.
    m_tensors.disableAddingNewTensors(m_isOriginalGraph);
    m_tensors.setGraphInputs(*this);
}

// Give approximated decision if parallel execution is now possible.
// Used for last chance decision.
bool EagerNodesBuilder::isParallelExecPossible()
{
    EAGER_ASSERT(!m_isOriginalGraph, "Wrong flow");
    EAGER_ASSERT(m_physicalNodesNr >= 2 && !areMultipleEnginesUsed(), "Wrong flow");
    if (!m_tensors.shouldUtilizeParallelExecution()) return false;
    for (const EagerNode& node : m_nodes)
    {
        if (node->isLogicalOperation()) continue;
        const EngineType engine = node.getEngineType();
        EAGER_ASSERT(engine != EngineType::INVALID, "Unsupported engine");

        // Check outputs only, in most cases they are less and give sufficient info to populate the decision
        if (!m_usedEnginesScoreBoard.test(static_cast<unsigned>(engine)) &&
            m_tensors.checkForTensorExceedParallelExecutionThreshold(node->getOutputs()))
        {
            EAGER_ASSERT(m_usedEnginesScoreBoard.count() == 1, "Expected single physical engine");
            m_usedEnginesScoreBoard.set(static_cast<unsigned>(engine));
            return true;  // There must be at least one, now there are two
        }
    }
    return false;
}

}  // namespace eager_mode
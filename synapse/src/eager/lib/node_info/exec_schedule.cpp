#include "exec_schedule.h"

// eager includes (relative to src/eager/lib/)
#include "node_info/node_info_defs.h"
#include "utils/algorithm_utils.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"
#include "include/tensor.h"

// std includes
#include <algorithm>
#include <numeric>

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// ExecScheduler
///////////////////////////////////////////////////////////////////////////////////////////////////

// Apply topological reorder on all given nodes
bool ExecScheduler::reorderAll(EagerNodesBuilder& nodes)
{
    EAGER_ASSERT(m_processNewNodesEn, "Processing new nodes was disabled");
    EAGER_ASSERT(m_isReorderAllInvoked == false, "Invalid flow");
    m_isReorderAllInvoked = true;

    // Approximate decision if multiple HW engines are not utilized so no need to enable parallel execution
    if (!nodes.areMultipleEnginesUsed())
    {
        m_globalDependencies.disableParallelExecution();
    }

    if (nodes.empty()) return true;

    // Special case 1: don't apply reordering on single-node graphs
    if (nodes.size() == 1) return true;

    // Special case 2: with two nodes do nothing or swap
    if (nodes.size() == 2)
    {
        return reorderTwoNodes(nodes.front(), nodes.back(), nodes.getTensors());
    }

    // General case: apply topological reordering on 3 nodes and above
    return m_rangeReorder.reorder(&nodes.front(), nodes.size());
}

// Apply topological reorder on recently added nodes.
// The method maintains information about previous nodes that had been processed.
bool ExecScheduler::reorderLast(EagerNodesBuilder& nodes)
{
    EAGER_ASSERT(m_processNewNodesEn, "Processing new nodes was disabled");
    if (nodes.empty()) return true;  // Note: Could have no nodes added in case of ZST

    if (m_nodesArrPtr.arr == nullptr)
    {
        m_nodesArrPtr.arr = &nodes;
    }
    else if (m_nodesArrPtr.size == nodes.size())
    {
        return true;  // Note: Could have no nodes added in case of ZST
    }
    EAGER_ASSERT(m_nodesArrPtr.arr == &nodes, "Nodes list were replaced");

    // Maintain node list size of current and previous
    const NodesNrType prevNodesArrSz = m_nodesArrPtr.size;
    m_nodesArrPtr.size               = nodes.size();
    EAGER_ASSERT(m_nodesArrPtr.size > prevNodesArrSz, "Node list is expected to extend");
    const NodesNrType newAddedNodesNr = m_nodesArrPtr.size - prevNodesArrSz;

    // This section of code represents the final opportunity to reconsider the execution strategy from serial to
    // parallel. Initially, nodes are single entities, but later the first node might be divided into multiple physical
    // nodes. The condition (prevNodesArrSz == 0) limits this decision to the first node. However, subsequent nodes
    // could also undergo extraction. If this occurs, we might be compelled to stick with a serial execution policy,
    // even if there is potential for parallelism. It's worth noting that the likelihood of real parallelism on the
    // device is unknown. Additionally, the process of extracting nodes is resource-intensive during compilation.
    // Considering these factors, the emphasis on optimizing against compilation favour diminishes.
    if (unlikely((prevNodesArrSz == 0) && (nodes.getPhysicalNodesNr() >= 2) &&
                 m_globalDependencies.isReEnablingParallelExecPossible() && nodes.isParallelExecPossible()))
    {
        m_globalDependencies.enableParallelExecution();
    }

    // Special case 1: node was not extracted
    if (newAddedNodesNr == 1)
    {
        m_globalDependencies.processSingleNode(nodes.back());
        return true;
    }

    EagerNode& firstNode = *(nodes.begin() + prevNodesArrSz);
    // Special case 2: node was extracted to two nodes
    if (newAddedNodesNr == 2)
    {
        EagerNode&               secondNode = nodes.back();
        const ReorderTwoNodesRes res        = reorderTwoNodes(firstNode, secondNode);
        if (unlikely(res == ReorderTwoNodesRes::FAIL)) return false;
        m_globalDependencies.processTwoNodes(firstNode, secondNode, res == ReorderTwoNodesRes::SWAP);
        return true;
    }

    // General case: apply topological reordering on 3 nodes and above
    if (unlikely(!m_rangeReorder.reorder(&firstNode, newAddedNodesNr))) return false;
    m_globalDependencies.processThreeNodesAndMore(&firstNode, newAddedNodesNr, m_rangeReorder);
    return true;
}

void ExecScheduler::disableProcessingNewNodes(const EagerNodesBuilder& nodes)
{
    EAGER_ASSERT(m_processNewNodesEn, "Wrong flow");
    m_processNewNodesEn = false;
    m_globalDependencies.finalize(nodes);
}

void ExecScheduler::injectNodes(EagerNodes& nodes, VecNodes<std::pair<std::size_t, EagerNode>>& nodesToInject)
{
    EAGER_ASSERT(!m_globalDependencies.isReEnablingParallelExecPossible(),
                 "re-enabling parallel execution not supported!");

    EAGER_ASSERT(!nodesToInject.empty(), "");
    moveIntoContainerAtIndices(nodes, nodesToInject);

    // If we only have a reduction + one memcpy (since nodesToInject isn't empty), we can skip reorder after injection.
    if (nodes.size() > 2 &&
        std::any_of(nodes.begin(), nodes.end(), [](const EagerNode& n) { return isReductionOp(n->getNodeType()); }))
    {
        // General case: apply topological reordering on 3 nodes and above
        m_rangeReorder.reorder(&nodes.front(), nodes.size());
    }
}

// This version is invoked by reorderAll(...)
bool ExecScheduler::reorderTwoNodes(EagerNode& node1, EagerNode& node2, const EagerTensorsSetBuilder& tensorsSet)
{
    if (tensorsSet.isRoot(node1) == false)
    {
        std::swap(node1, node2);
        if (unlikely(tensorsSet.isRoot(node1) == false))
        {
            EAGER_REPORT_ERROR("{}: Cycle was detected in the graph", HLLOG_FUNC);
            return false;
        }
    }
    return true;
}

// This version is invoked by reorderLast(...) or logical ops,
// such that EagerTensorsSetBuilder::isRoot(...) becomes irrelevant.
ExecScheduler::ReorderTwoNodesRes ExecScheduler::reorderTwoNodes(EagerNode& node1, EagerNode& node2)
{
    if (isOrdered(node1, node2) == false)
    {
        std::swap(node1, node2);
        if (unlikely(isOrdered(node1, node2) == false))
        {
            EAGER_REPORT_ERROR("{}: Cycle was detected in the graph", HLLOG_FUNC);
            return ReorderTwoNodesRes::FAIL;
        }
        return ReorderTwoNodesRes::SWAP;
    }
    return ReorderTwoNodesRes::SUCCESS;
}

// Check if node1 does not consume any of node2's outputs
bool ExecScheduler::isOrdered(const Node& node1, const Node& node2)
{
    for (const TensorPtr& node1Input : node1.getInputs())
    {
        if (unlikely(node1Input == nullptr)) continue;
        for (const TensorPtr& node2Output : node2.getOutputs())
        {
            if (unlikely(node2Output == nullptr)) continue;
            if (node1Input.get() == node2Output.get()) return false;
        }
    }
    return true;  // node1 can be placed before node2 in the execution sequence
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TopologyReorder
///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace reorder_defs;

// Apply topology reordering on a given range as array, represented by first node and number of nodes.
// Implementation is based on Kahn’s algorithm.
bool TopologyReorder::reorder(EagerNode* nodeArr, NodesNrType nodesNr)
{
    EAGER_ASSERT(nodesNr >= 2, "Wrong flow");
    m_nodes.arr  = nodeArr;
    m_nodes.size = nodesNr;
    // clear data from previous iterations
    m_inDegree.clear();
    m_sequence.clear();
    // re-initialize to default values
    m_inDegree.resize(m_nodes.size, 0);
    m_sequence.resize(m_nodes.size, -1);
    m_rootsQueue.head = m_rootsQueue.tail = -1;  // Reset the queue

    // Step 1: Init
    if (!initConsumptionInfo()) return true;
    // Step 2: Build relations map
    if (!fillConsumptionInfo()) return false;
    // Step 3: Determine initial roots
    if (!initRootsQueue()) return false;
    // Early exit in case all nodes are disjoint
    if (m_rootsQueue.tail == m_nodes.size) return true;
    // Step 4: Finalize order sequence
    if (!completeReordering()) return false;
    // Step 5: Rearrange the nodes in final execution sequence
    permuteResult();
    return true;
}

// Step 1: Init node and consumption info
bool TopologyReorder::initConsumptionInfo()
{
    m_consumptionInfo.clear();  // Could be reuse
    for (NodesNrType i = 0; i < m_nodes.size; ++i)
    {
        for (const TensorPtr& output : m_nodes.arr[i]->getOutputs())
        {
            if (output == nullptr) continue;
            m_consumptionInfo.emplace_back(i, output.get());
        }
    }

    // Theoretical case: no output of any node. nodes can be executed in any order
    if (m_consumptionInfo.size() == 0) return false;
    return true;  // Proceed to reorder nodes
}

// Step 2: Fill consumer nodes and reorder nodes with only graph-inputs.
bool TopologyReorder::fillConsumptionInfo()
{
    auto addConsumer = [this](NodesNrType consumerNodeIdx, reorder_defs::TensorConsumptionInfo* producer) {
        ++m_inDegree[consumerNodeIdx];
        producer->consumers.push_back(consumerNodeIdx);
    };

    for (NodesNrType i = 0; i < m_nodes.size; ++i)
    {
        const EagerNode&    node   = m_nodes.arr[i];
        const TensorVector& inputs = node->getInputs();
        // "Reduction" node has special handling because its inputs are not similar.
        // Producer of first input must run before producers of the rest inputs.
        if (!isReductionOp(node->getNodeType()))
        {
            for (const TensorPtr& input : inputs)
            {
                if (unlikely(input == nullptr)) continue;
                // Looking for tensors that are not inputs of the graph
                auto result = getTensorConsumptionInfo(input.get());
                if (result == nullptr) continue;
                addConsumer(i, result);
            }
        }
        else
        {
            EAGER_ASSERT(inputs.size() >= 2 && node->getOutputs().size() == 1, "Invalid reduction operator");
            auto firstInputProducer = getTensorConsumptionInfo(inputs[0].get());
            // view node is an alias of the output tensor (strided\slice insert) - handling as for regular node
            // and for user graph we might also get here with input graph tensors pre memcpy addition as part
            // of node extraction.
            if (firstInputProducer == nullptr)
            {
                for (TensorsNrType j = 1; j < inputs.size(); ++j)
                {
                    auto currentInputProducer = getTensorConsumptionInfo(inputs[j].get());
                    if (currentInputProducer == nullptr) continue;
                    addConsumer(i, currentInputProducer);
                }
            }
            // Handling the reduction op by adding artificial dependencies between the first input producer and
            // the rest of the node's inputs producers.
            else
            {
                // Current node - the "reduction", is a consumer of the firstInputProducer
                addConsumer(i, firstInputProducer);
                // Artificially route remaining inputs of "reduction" node to be consumers of the firstInputProducer
                for (TensorsNrType j = 1; j < inputs.size(); ++j)
                {
                    auto currentInputProducer = getTensorConsumptionInfo(inputs[j].get());
                    if (currentInputProducer == nullptr) continue;
                    // Artificially make currentInputProducer a consumer of firstInputProducer
                    addConsumer(currentInputProducer->producerNodeId, firstInputProducer);
                    // "Reduction" node is consumer of currentInputProducer, this is an actual attribute of the graph
                    addConsumer(i, currentInputProducer);
                }
            }
        }
    }
    return true;
}

// Step 3: initialize the information to track which node is next to be treated as root
bool TopologyReorder::initRootsQueue()
{
    for (NodesNrType i = 0; i < m_nodes.size; ++i)
    {
        if (m_inDegree[i] == 0)  // Handle a root
        {
            EAGER_ASSERT(m_sequence[i] == -1, "Invalid execution sequence");
            if (m_rootsQueue.head != -1)
            {
                m_sequence[m_rootsQueue.tail++] = i;
            }
            else
            {
                m_sequence[0] = i;  // First node in the execution sequence
                // Initialize the queue
                m_rootsQueue.head = 0;
                m_rootsQueue.tail = 1;
            }
        }
    }

    if (unlikely(m_rootsQueue.head == -1))  // A graph with no roots
    {
        EAGER_REPORT_ERROR("{}: Cycle was detected in the graph", HLLOG_FUNC);
        return false;
    }
    return true;
}

// Step 4: Complete reordering
bool TopologyReorder::completeReordering()
{
    EAGER_ASSERT(m_rootsQueue.head != -1, "Invalid execution sequence");

    for (NodesNrType i = 0; i < m_nodes.size; ++i)
    {
        const NodesNrType curRoot = m_sequence[m_rootsQueue.head++];
        EAGER_ASSERT(curRoot < m_nodes.size, "cycle in the graph was detected");
        for (const TensorPtr& output : m_nodes.arr[curRoot]->getOutputs())
        {
            if (unlikely(output == nullptr)) continue;
            auto result = getTensorConsumptionInfo(output.get());
            EAGER_ASSERT(result != nullptr, "Missing consumption info for an output");

            for (NodesNrType consumerId : result->consumers)
            {
                EAGER_ASSERT(consumerId < m_nodes.size, "Consumer node index is out of bound");
                if (unlikely(m_inDegree[consumerId] == 0))
                {
                    EAGER_REPORT_ERROR("{}: Cycle was detected in the graph", HLLOG_FUNC);
                    return false;
                }
                if (--m_inDegree[consumerId] == 0)
                {
                    EAGER_ASSERT(m_rootsQueue.tail != -1,
                                 "Position to place a node at execution sequence is out of bound");
                    m_sequence[m_rootsQueue.tail++] = consumerId;
                }
            }
        }
    }

    EAGER_ASSERT(m_rootsQueue.head == m_rootsQueue.tail, "Invalid node in-degree calculation");
    return true;
}

// Step 5: Do the actual reordering on nodes
void TopologyReorder::permuteResult()
{
    for (NodesNrType i = 0; i < m_nodes.size; ++i)
    {
        const NodesNrType perm = m_sequence[i];
        EAGER_ASSERT(perm != -1, "Invalid execution sequence");
        EAGER_ASSERT(m_inDegree[perm] == 0, "Invalid execution sequence");

        NodesNrType idxToSwap = perm;
        while (idxToSwap < i)  // Original node to be swapped had ben replaced, restore it
        {
            idxToSwap = m_sequence[idxToSwap];
        }

        EagerNode& node = m_nodes.arr[i];
        if (i != idxToSwap)
        {
            EAGER_ASSERT(idxToSwap != -1, "Invalid execution sequence");
            std::swap(node, m_nodes.arr[idxToSwap]);
        }
#ifndef NDEBUG
#if 0  // Enable only when debugging topology reordering issues
        printNode(node, perm, i);
#endif
#endif  // NDEBUG
    }
}

reorder_defs::TensorConsumptionInfo* TopologyReorder::getTensorConsumptionInfo(const Tensor* t)
{
    return const_cast<reorder_defs::TensorConsumptionInfo*>(
        const_cast<const TopologyReorder*>(this)->getTensorConsumptionInfo(t));
}

const reorder_defs::TensorConsumptionInfo* TopologyReorder::getTensorConsumptionInfo(const Tensor* t) const
{
    auto ptr = std::find_if(m_consumptionInfo.begin(), m_consumptionInfo.end(), [=](const auto& ci) {
        return ci.producedTensor == t;
    });
    return ptr != m_consumptionInfo.end() ? ptr : nullptr;
}

void TopologyReorder::printNode(const EagerNode& node, NodesNrType oldIdx, NodesNrType newIdx)
{
    if (newIdx == 0)
    {
        LOG_INFO(EAGER, "Node execution sequence:");
    }
    const char* opType = node->isLogicalOperation() ? "logical" : "physical";
    LOG_INFO(EAGER, "  {}->{}) \"{}\" [{}, {}]", oldIdx, newIdx, node->getNodeName(), node->getGUID(), opType);

    auto printTensors = [](const std::string& label, const TensorVector& tensors) {
        for (const TensorPtr& tensor : tensors)
        {
            if (tensor == nullptr) return;
            if (tensor->isAliasedTensor())
            {
                const TensorPtr& realTensor = Tensor::getRealTensor(tensor);
                EAGER_ASSERT_PTR(realTensor);
                LOG_INFO(EAGER, "       {}: {}  --->  {}", label, tensor->getName(), realTensor->getName());
            }
            else
            {
                LOG_INFO(EAGER, "       {}: {}", label, tensor->getName());
            }
        }
    };

    printTensors("i", node->getInputs());
    printTensors("o", node->getOutputs());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GlobalDependencies
///////////////////////////////////////////////////////////////////////////////////////////////////

// Associate single tensor to its producer
inline void ProducedTensorsMap::add(const TensorPtr& tensor, NodesNrType producer)
{
    EAGER_ASSERT(!m_isSorted, "Wrong flow");
    m_producedTensorsMap.emplace_back(producer, tensor.get());
}

// Add mapping of multiple tensors to their producers
void ProducedTensorsMap::add(const reorder_defs::ConsumptionInfo& consumptionInfo,
                             const VecNodes<NodesNrType>&         invSequence,
                             NodesNrType                          existingNodesNr)
{
    EAGER_ASSERT(!m_isSorted, "Wrong flow");
    for (const TensorConsumptionInfo& info : consumptionInfo)
    {
        EAGER_ASSERT(info.producerNodeId < invSequence.size(), "Invalid producer node id");
        // Producers were made prior reordering, so need to map them to the new order
        const NodesNrType producerNodeId = invSequence[info.producerNodeId] + existingNodesNr;
        m_producedTensorsMap.emplace_back(producerNodeId, info.producedTensor);
    }
}

// Sort the tensor/producer according to tensors, in order to make search more efficient
void ProducedTensorsMap::sort()
{
    EAGER_ASSERT(!m_isSorted, "Wrong flow");
    // Sort the map according to pointers of the tensors
    std::sort(m_producedTensorsMap.begin(), m_producedTensorsMap.end());
    m_isSorted = true;
}

// Return the producer of a given tensor. If producer is unavailable return -1 as an indication to a graph root.
NodesNrType ProducedTensorsMap::findProducer(const TensorPtr& tensor) const
{
    auto       result = std::lower_bound(m_producedTensorsMap.begin(), m_producedTensorsMap.end(), tensor.get());
    const bool isProducedTensor = ((result != m_producedTensorsMap.end()) && (result->producedTensor == tensor.get()));
    return isProducedTensor ? result->producerNodeId : -1;
}

GlobalDependenciesBuilder::GlobalDependenciesBuilder(bool isParallelExecutionPossible)
: m_isParallelExecutionPossible(isParallelExecutionPossible), m_enableParallelExecution(m_isParallelExecutionPossible)
{
}

// Collect dependency information for single node
void GlobalDependenciesBuilder::processSingleNode(const EagerNode& node)
{
    if (m_enableParallelExecution)
    {
        processSingleNodeForParallelExecution(node);
    }
    else  // Serial execution case
    {
        processSingleNodeForSerialExecution(node);
    }
}

// Collect dependency information for two nodes.
// 'isNode2DependOnNode1' is passed to optimize the calculations.
void GlobalDependenciesBuilder::processTwoNodes(const EagerNode& node1,
                                                const EagerNode& node2,
                                                bool             isNode2DependOnNode1)
{
    if (m_enableParallelExecution)
    {
        processSingleNodeForParallelExecution(node1);
        if (isNode2DependOnNode1)
        {
            processSingleNodeForSerialExecution(node2);  // Similar to serial execution case
        }
        else
        {
            processSingleNodeForParallelExecution(node2);
        }
    }
    else  // Serial execution case
    {
        processSingleNodeForSerialExecution(node1);
        processSingleNodeForSerialExecution(node2);
    }
}

// Collect dependency information for three nodes and above
void GlobalDependenciesBuilder::processThreeNodesAndMore(EagerNode*             nodeArr,
                                                         NodesNrType            nodesNr,
                                                         const TopologyReorder& rangeReorder)
{
    const VecNodes<NodesNrType>& sequence = rangeReorder.getSequence();
    EAGER_ASSERT(nodesNr >= 3, "Wrong flow");
    EAGER_ASSERT(nodesNr == sequence.size(), "Inconsistent sequencing info");
    if (m_enableParallelExecution)
    {
        // Invert mapping of sequences to fix producer ids in the consumption info, as it was populated prior reordering
        VecNodes<NodesNrType> invSequence(nodesNr);
        for (NodesNrType i = 0; i < nodesNr; ++i)
        {
            EAGER_ASSERT(sequence[i] < nodesNr, "Invalid sequence content");
            invSequence[sequence[i]] = i;
        }

        const NodesNrType existingNodesNr = m_latestPhysicalProducers.size();
        addNewNode(nodeArr[0]);  // First node is always a local root
        for (NodesNrType i = 1; i < nodesNr; ++i)
        {
            const EagerNode&           node = nodeArr[i];
            std::optional<NodesNrType> closestProducer;
            // Locate the producer to the node, it's the one that produced one of its inputs and has the highest
            // sequence id
            for (const TensorPtr& input : node->getInputs())
            {
                auto result = rangeReorder.getTensorConsumptionInfo(input.get());
                if (result != nullptr)
                {
                    EAGER_ASSERT(result->producerNodeId < invSequence.size(), "Producer node order id is out of bound");
                    // Producers were made prior reordering, so need to map them to the new order
                    const NodesNrType curProducerId = invSequence[result->producerNodeId] + existingNodesNr;
                    closestProducer =
                        closestProducer.has_value() ? std::max(curProducerId, *closestProducer) : curProducerId;
                }
            }
            if (closestProducer.has_value())
            {
                addNewNode(node, *closestProducer);  // This is final mapping
            }
            else
            {
                addNewNode(node);  // It could be  a root or has a producer at previous 'add_latest' iteration
            }
        }
        // Add new tensors and map them to their producers
        m_producedTensorsMap.add(rangeReorder.getConsumptionInfo(), invSequence, existingNodesNr);
    }
    else  // Serial execution case
    {
        for (NodesNrType i = 0; i < nodesNr; ++i)
        {
            processSingleNodeForSerialExecution(nodeArr[i]);
        }
    }
}

// Serial execution of a new single node is adding a dependency on last physical one
// that have an output that is consumed by the given node.
void GlobalDependenciesBuilder::processSingleNodeForSerialExecution(const EagerNode& node)
{
    // By default that node is the previous one. If node is root then id its producer is -1
    NodesNrType previousNode = m_latestPhysicalProducers.size() - 1;
    // Update all node-related vectors
    m_latestPhysicalProducers.push_back(previousNode);
}

void GlobalDependenciesBuilder::processSingleNodeForParallelExecution(const EagerNode& node)
{
    EAGER_ASSERT(m_enableParallelExecution, "Wrong flow");
    for (const TensorPtr& output : node->getOutputs())
    {
        if (unlikely(output == nullptr)) continue;
        const NodesNrType curNodeId = m_latestPhysicalProducers.size();
        m_producedTensorsMap.add(output, curNodeId);
    }
    addNewNode(node);
}

// Add new node and id to a producer. "producerId" can point to a logical node while building
// m_latestPhysicalProducers, however we fix that at fixDependenciesOnLogicalOps(...).
void GlobalDependenciesBuilder::addNewNode(const EagerNode& node, NodesNrType producerId)
{
    if (producerId == -1 && !m_latestPhysicalProducers.empty())
    {
        const NodesNrType curNodeId = m_latestPhysicalProducers.size();
        m_missingProducers.emplace_back(node, curNodeId);
    }
    m_latestPhysicalProducers.push_back(producerId);
}

// Scan all mappings that have -1.
// Since the whole information is available it's possible to find the producer or determine a root of the graph.
// To clarify why such cases can happen, need to remember that reorderLast(...) can work on a multi-node-sub-graph.
// That sub-graph can have one or more roots (almost one), one of the roots may be connected to previous sub-graph
// (this is the common case). So that root we are talking about is a local-root relative to the sub-graph but it's
// not relative to the global-graph.
void GlobalDependenciesBuilder::fillMissingProducers()
{
    EAGER_ASSERT(!m_missingProducers.empty(), "Wrong flow");
    m_producedTensorsMap.sort();
    for (const MissingProducerInfo& info : m_missingProducers)
    {
        EAGER_ASSERT(info.sequenceId < m_latestPhysicalProducers.size(), "Invalid sequential id of a node");
        EAGER_ASSERT(m_latestPhysicalProducers[info.sequenceId] == -1, "Invalid physical producing info");

        std::optional<NodesNrType> closestProducer;
        for (const TensorPtr& input : info.node->getInputs())
        {
            const NodesNrType curProducerId = m_producedTensorsMap.findProducer(input);
            if (curProducerId != -1)
            {
                closestProducer =
                    closestProducer.has_value() ? std::max(curProducerId, *closestProducer) : curProducerId;
            }
        }
        if (closestProducer.has_value())
        {
            m_latestPhysicalProducers[info.sequenceId] = *closestProducer;
        }
    }
    m_missingProducers.clear();
}

// Typically physical nodes must not rely on logical node for signaling
void GlobalDependenciesBuilder::fixDependenciesOnLogicalOps(const EagerNodesBuilder& nodes)
{
    EAGER_ASSERT(m_latestPhysicalProducers.size() == nodes.size(), "Invalid physical producers info");
    for (NodesNrType& prod : m_latestPhysicalProducers)
    {
        if (prod != -1)
        {
            EAGER_ASSERT(prod < nodes.size(), "Invalid producer id");
            if (nodes[prod].getEngineType() == EngineType::INVALID)
            {
                prod = m_latestPhysicalProducers[prod];
            }
        }
    }
}

void GlobalDependenciesBuilder::redoSerialDependencies(NodesNrType nodeNr)
{
    EAGER_ASSERT(!m_enableParallelExecution, "injectLogicalNodes path doesn't support parallel execution");
    EAGER_ASSERT(!m_isParallelExecutionPossible, "re-enabling parallel execution not supported!");

    auto& vec = m_latestPhysicalProducers;
    if (vec.size() == nodeNr) return;

    EAGER_ASSERT(vec.size() < nodeNr, "redoSerialDependencies is expected to be used after node addition");

    vec.resize(nodeNr);
    std::iota(vec.begin(), vec.end(), -1);
}

// Produce the mapping that has final dependencies
void GlobalDependenciesBuilder::finalize(const EagerNodesBuilder& nodes)
{
    if (m_latestPhysicalProducers.size() == 1) return;  // Single node has no dependencies
    if (!m_missingProducers.empty())
    {
        fillMissingProducers();
    }
    fixDependenciesOnLogicalOps(nodes);
}

}  // namespace eager_mode

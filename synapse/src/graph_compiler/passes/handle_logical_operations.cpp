#include "handle_logical_operations.h"

#include "code_generation/tensor_size_validator.h"
#include "defs.h"
#include "graph_editor.h"
#include "habana_global_conf.h"
#include "habana_nodes.h"
#include "handle_memory_reuse.h"
#include "node_factory.h"
#include "reduction_node.h"
#include "strided_insert_node.h"

#include "tensor_annotation.h"
#include "transpose_node.h"
#include "graph_traversal_generator.h"
#include "types.h"
#include "utils.h"
#include <algorithm>
#include <stack>

using LogicalOpPtr = LogicalOpsHandler::LogicalOpPtr;

bool handleLogicalOps(HabanaGraph& g)
{
    LogicalOpsHandler(g).handleLogicalOps();
    g.invalidateExecutionSchedule();  // Force re-generation of the execution schedule
    return true;
}

//  Logical ops general algorithm
//  Two iterations, backward and forward
//    a. Backward:
//       For each node in (sorted backward - execution order):
//        i. If the node is a don't care logical operation (where the real tensor can be either way) with persistent
// output
//            1. Convert to backward, so real tensor is the output
//            2. Will be handled in (iii)
//        ii. If the node is a don't care logical operation and the producer is "real" operation and the logical node
// is the only consumer
//            1. Convert to backward, so real tensor is the output
//            2. Will be handled in (iii)
//        iii. If the node is backward logical operation
//            1. Handle node ()
//            2. If one of the inputs is produced by a don't care logical operation
//                a. Convert the producer from don't care to backward, so now its real tensor is our node's input
// (which is an alias) b. Forward: For each node in execution order: i. If the node is an unhandled logical operation
//           1. Handle node () (here all forward nodes (where the "real" tensor is the input) will be handle)
bool LogicalOpsHandler::handleLogicalOps()
{
    // We make a copy of the sortedNodes list since during the functions below we are going to
    // invalidate the graph's sortedNodes cache as we insert memory copy nodes.

    LogicalOpVector sortedLogicalNodes;
    for (const NodePtr& node : m_graph.getExeSortedNodes())
    {
        if (node->isLogicalOperation())
        {
            sortedLogicalNodes.emplace_back(std::static_pointer_cast<LogicalOpNode>(node));
        }
    }

    if (sortedLogicalNodes.empty())
    {
        return false;
    }

    m_graph.getGraphAnnotation().logicalOperationsHandled = true;
    handleAndRunLogicalOps(/*isFwd*/ false);
    handleAndRunLogicalOps(/*isFwd*/ true);
    m_graph.turnOnPredicate(PREDICATE_ID_LOGICAL_NODE_RAN);
    handleSparseLayoutTensors(sortedLogicalNodes);
    handleRealInLogical(sortedLogicalNodes);

    return true;
}

bool LogicalOpsHandler::areSameBundle(const NodePtr& n1, const NodePtr& n2)
{
    if (!n1 || !n2) return false;
    const auto& info1 = n1->getNodeAnnotation().bundleInfo;
    const auto& info2 = n2->getNodeAnnotation().bundleInfo;
    return (info1.is_set() == info2.is_set()) && (!info1.is_set() || (info2->bundleIndex == info1->bundleIndex));
}

void LogicalOpsHandler::cacheMemcopy(const NodePtr& copy, bool insertMemcpyForInput)
{
    const TensorPtr& originalTensor = insertMemcpyForInput ? copy->getInput(0) : copy->getOutput(0);
    m_memcopiesCache[std::make_pair(originalTensor, insertMemcpyForInput)] = copy;
}

TensorPtr LogicalOpsHandler::getCachedCopiedTensor(const TensorPtr& originalTensor,
                                                   const NodePtr&   originalNode,
                                                   bool             insertMemcpyForInput) const
{
    const NodePtr& copy = m_memcopiesCache.at(std::make_pair(originalTensor, insertMemcpyForInput));
    // if the New node is created inside a bundle, it must have the bundle info of that bundle (SW-125084)
    const NodePtr& copyProducer = m_graph.getTensorProducer(copy->getInput(0));
    if (copyProducer && originalNode->getNodeAnnotation().bundleInfo.is_set() &&
        areSameBundle(originalNode, copyProducer))
    {
        copy->getNodeAnnotation().bundleInfo = copyProducer->getNodeAnnotation().bundleInfo;
    }
    return insertMemcpyForInput ? copy->getOutput(0) : copy->getInput(0);
}

bool LogicalOpsHandler::canReuseMemcpy(const NodePtr&   originalNode,
                                       const TensorPtr& originalTensor,
                                       bool             insertMemcpyForInput) const
{
    auto replaceTensorIter = m_memcopiesCache.find(std::make_pair(originalTensor, insertMemcpyForInput));
    if (replaceTensorIter == m_memcopiesCache.end()) return false;
    const NodePtr&   copy         = replaceTensorIter->second;
    const TensorPtr& copiedTensor = insertMemcpyForInput ? copy->getOutput(0) : copy->getInput(0);
    return !copiedTensor->isAliasedTensor();
}

bool LogicalOpsHandler::isRealInLogical(const TensorPtr& t)
{
    return t->isRealInLogical() || t->isRealInAliasing();
}

bool LogicalOpsHandler::isReduction(const NodePtr& node)
{
    switch (node->getNodeType())
    {
        case Node::TYPE_INTERNAL_REDUCTION:
        case Node::TYPE_STRIDED_INSERT:
        case Node::TYPE_MULTI_INSERT:
        case Node::TYPE_SLICE_INSERT:
            return true;
        default:
            break;
    }
    return false;
}

NodePtr LogicalOpsHandler::getReductionInAliasChain(const LogicalOpPtr& node, const TensorPtr& t)
{
    // reduction nodes are allways backward. so if current node is forward, reduction cannot be in alias chain.
    if (!isBackwardNode(*node)) return nullptr;
    if (isReduction(node)) return node;

    TensorPtr real = node->getRealTensor();
    HB_ASSERT_PTR(real);
    while (real->isAliasedTensor())  // check nodes in alias chain
    {
        real              = real->getAliasTensor();
        HB_ASSERT_PTR(real);
        NodePtr reduction = m_graph.getTensorProducer(real);
        if (!reduction) break;
        if (isReduction(reduction)) return reduction;
    }
    return nullptr;
}

bool LogicalOpsHandler::hasMemoryCoherenceRisk(const LogicalOpPtr& node, const TensorPtr& aliasTensor)
{
    // prevent memory coherency risks due to aliasing (see [SW-74559])
    // if any other producer of the reduction can run before the consumers of the aliasTensor, then we are at risk.
    if (aliasTensor->isUserManagedDram()) return false;  // handled separetly using control edges

    // if there is no reduction in the alias chain, there is no coherency risk
    NodePtr reduction = getReductionInAliasChain(node, aliasTensor);
    if (!reduction) return false;

    // if there is only 1 consumer there is no risk
    if (m_graph.getNumberOfTensorConsumers(aliasTensor) == 1) return false;

    // get all the real consumers of aliasTensor except 'node'
    NodeSet allReaders = m_graph.getRealConsumersExcept(aliasTensor, node, true /* includeGraphEdges */);

    // get all other writers for other inputs of the reduction
    NodeSet allWriters;
    for (const TensorPtr& input : reduction->getInputs())
    {
        if (input == aliasTensor) continue;  // we want only producers of the other reduction inputs
        const auto& writers = m_graph.getRealProducersExcept(
            input,
            [&](const NodePtr& n) { return n == node; },
            true /* includeGraphEdges */);
        allWriters.insert(writers.begin(), writers.end());
    }

    // check for any potential 'write after read' risks
    bool atRisk = std::any_of(allWriters.begin(), allWriters.end(), [&](const NodePtr& writer) {
        return std::any_of(allReaders.begin(), allReaders.end(), [&](const NodePtr& reader) {
            return m_graph.getNumberOfPaths(reader, writer) == 0;
        });
    });
    return atRisk;
}

void LogicalOpsHandler::handleRealTensor(const LogicalOpPtr& node)
{
    /* iii. If the real tensor is strided */
    const TensorPtr& realTensor = node->getRealTensor();
    bool             isAlias    = realTensor->isAliasedTensor();
    bool             isStrided  = !realTensor->isTrivialStrided();

    if (realTensor->isZeroSizedDataTensor())
    {
        LOG_TRACE(OPT_LOGICAL_OPS, "Tensor: {} size is zero, thus no memcopy will be planted.", realTensor->getName());
        return;
    }

    LOG_TRACE(OPT_LOGICAL_OPS,
              "{}: tensor: {}, isAlias: {}, isStrided: {}.",
              HLLOG_FUNC,
              realTensor->getName(),
              isAlias,
              isStrided);

    if (isStrided && !node->canHandleStridedRealTensor())
    {
        HB_ASSERT(!node->isPureLogical(),
                  "{} is pure logical, but memcpy is required in function: {}",
                  node->getNodeName(),
                  __func__);
        bool inputMemcpy = isForwardNode(*node);
        if (canReuseMemcpy(node, realTensor, inputMemcpy))
        {
            LOG_TRACE(OPT_LOGICAL_OPS,
                      "{}: Use created memcpy for {} {} (real tensor)",
                      HLLOG_FUNC,
                      inputMemcpy ? "input" : "output",
                      node->getNodeName());
            GraphEditor::replaceTensor(m_graph, node, realTensor, getCachedCopiedTensor(realTensor, node, inputMemcpy));
        }
        else
        {
            LOG_TRACE(OPT_LOGICAL_OPS,
                      "{}: inserting memory copy for {} {}, (real tensor)",
                      HLLOG_FUNC,
                      inputMemcpy ? "input" : "output",
                      node->getNodeName());
            const NodePtr& copy = GraphEditor::insertMemcpy(m_graph, node, inputMemcpy, realTensor);
            cacheMemcopy(copy, inputMemcpy);
        }
    }
}

bool LogicalOpsHandler::canHandleHugeTensor(const NodePtr& n)
{
    if (isMemcpy(*n)) return true;
    if (n->isDma() && static_cast<const DMANode*>(n.get())->isDynamicMemoryOp()) return true;
    return false;
}

bool LogicalOpsHandler::isHugeAliasTensor(const LogicalOpPtr& node, unsigned index) const
{
    bool             hugeTensorsDetected = false;
    const TensorPtr& aliasTensor         = isForwardNode(*node) ? node->getOutput(index) : node->getInput(index);
    if (!aliasTensor->isDataTensor()) return false;

    // logical op didn't change strides so other nodes shouldn't be affected
    if (!node->isAliasStrided(index)) return false;

    NStrideArray        strides = node->calculateAliasStrides(index);
    TensorSizeValidator validator(m_graph, /* print only on trace */ SPDLOG_LEVEL_TRACE);

    auto surroundingNodes = m_graph.getTensorConsumers(aliasTensor);
    surroundingNodes.push_back(m_graph.getTensorProducer(aliasTensor));
    for (const NodePtr& node : surroundingNodes)
    {
        if (!node) continue;
        if (canHandleHugeTensor(node)) continue;  // memcopy support huge tensor strides by roi splitting
        if (!validator.validateTensor(node, aliasTensor, aliasTensor->getAllNSizesInElements(), strides))
        {
            return true;
        }
    }
    return false;
}

void LogicalOpsHandler::handleAliasedTensors(const LogicalOpPtr& node)
{
    LOG_TRACE(OPT_LOGICAL_OPS, "{}", HLLOG_FUNC);

    std::set<TensorPtr>   setOfUniqueInTensors;
    std::vector<unsigned> tensorIndicesForMemcpy;
    std::vector<unsigned> tensorIndicesForReuse;
    bool                inputMemcpy = !isForwardNode(*node);  // Forward means that the aliased tensors are the outputs
    bool                isPureLogical              = node->isPureLogical();
    const TensorVector& aliasTensors               = node->getAliasTensors();
    const TensorVector& allTensorsInAliasDirection = isForwardNode(*node) ? node->getOutputs() : node->getInputs();

    for (unsigned tensorIndex = 0; tensorIndex < allTensorsInAliasDirection.size(); ++tensorIndex)
    {
        const TensorPtr& aliasTensor = allTensorsInAliasDirection.at(tensorIndex);
        if (std::find(aliasTensors.begin(), aliasTensors.end(), aliasTensor) == aliasTensors.end()) continue;
        if (aliasTensor->isShapeTensor()) continue;
        if (aliasTensor->isZeroSizedDataTensor())
        {
            LOG_DEBUG(OPT_LOGICAL_OPS, "Skipping zero sized tensor {}", aliasTensor->getName());
            continue;
        }

        /* User tensor for Goya  - input or output of the graph
         * User tensor for Gaudi - a user managed in DRAM tensor  */
        bool isUserManagedTensor = m_graph.isUserManagedDram(aliasTensor);
        bool isAlias    = aliasTensor->isAliasedTensor() && Tensor::getRealTensor(aliasTensor) != node->getRealTensor();
        bool firstOccur = setOfUniqueInTensors.insert(aliasTensor).second;
        bool isConstTensor = aliasTensor->isStaticParam();
        bool realInLogical = isRealInLogical(aliasTensor);
        bool isHugeTensor  = isHugeAliasTensor(node, tensorIndex);

        // prevent memory coherency risks due to aliasing (see [SW-74559])
        bool nonCoherentReduce = hasMemoryCoherenceRisk(node, aliasTensor);

        if (isUserManagedTensor && !node->isAliasStrided(tensorIndex) &&
            MemoryReuseHandler::isExactOverlap(aliasTensor, Tensor::getRealTensor(node->getRealTensor())))
        {
            // special case where a persistent tensor is an alias to another persistent tensor,
            // and the both have the exact same address, size, and strides
            isUserManagedTensor = false;
        }

        LOG_TRACE(OPT_LOGICAL_OPS,
                  "Tensor: {}, isUserManagedTensor: {}, firstOccur: {}, isAlias: {}, isConstTensor: {} "
                  "realInLogical: {}, nonCoherentReduce: {}, isHugeTensor: {}",
                  aliasTensor->getName(),
                  isUserManagedTensor,
                  firstOccur,
                  isAlias,
                  isConstTensor,
                  realInLogical,
                  nonCoherentReduce,
                  isHugeTensor);

        if (isUserManagedTensor || /* i.   Any of the aliases is user Managed(Gaudi) or input/output of the graph(Goya)
                                    */
            isAlias ||             /* ii.   Any of the aliases is already an alias                                    */
            !firstOccur ||         /* iv.   Duplicated aliased tensors                                                */
            isConstTensor ||       /* v.    Any of the aliases is const                                               */
            realInLogical ||       /* vi.   Any of the aliases is real in a logical node from a previous pass         */
            nonCoherentReduce ||   /* vii.  Alias has other consumers that are at risk for non coherent memory        */
            isHugeTensor)          /* viii. Alias tensor has a consumer that cannot handle the huge tensor strides    */
        {
            HB_ASSERT(!isPureLogical,
                      "{} is pure logical, but memcpy is required in function: {}",
                      node->getNodeName(),
                      __func__);
            // If its not first occur, we would like to force use a different memcpy node
            if (firstOccur && canReuseMemcpy(node, aliasTensor, inputMemcpy))
            {
                tensorIndicesForReuse.push_back(tensorIndex);
            }
            else
            {
                tensorIndicesForMemcpy.push_back(tensorIndex);
            }
        }
    }
    if (!tensorIndicesForMemcpy.empty())
    {
        LOG_TRACE(OPT_LOGICAL_OPS,
                  "{}: inserting memory copy for {} {} {}",
                  HLLOG_FUNC,
                  tensorIndicesForMemcpy.size(),
                  inputMemcpy ? "inputs" : "outputs",
                  node->getNodeName());
        NodeList newMemcopies = GraphEditor::insertMemcpies(m_graph, node, inputMemcpy, tensorIndicesForMemcpy);
        HB_ASSERT(tensorIndicesForMemcpy.size() == newMemcopies.size(),
                  "created {} memcopies, but should create {}",
                  newMemcopies.size(),
                  tensorIndicesForMemcpy.size());
        for (const NodePtr& copyNode : newMemcopies)
        {
            cacheMemcopy(copyNode, inputMemcpy);
        }
    }
    if (!tensorIndicesForReuse.empty())
    {
        LOG_TRACE(OPT_LOGICAL_OPS,
                  "{}: Use created memcpy for {} {} {}",
                  HLLOG_FUNC,
                  tensorIndicesForReuse.size(),
                  inputMemcpy ? "inputs" : "outputs",
                  node->getNodeName());
        GraphEditor::editNode(m_graph, node, [&]() {
            for (unsigned tensorIndex : tensorIndicesForReuse)
            {
                const TensorPtr& originalTensor = allTensorsInAliasDirection.at(tensorIndex);
                const TensorPtr& copiedTensor   = getCachedCopiedTensor(originalTensor, node, inputMemcpy);
                if (inputMemcpy)
                {
                    node->replaceInput(tensorIndex, copiedTensor);
                }
                else
                {
                    node->replaceOutput(tensorIndex, copiedTensor);
                }
            }
        });
    }
}

void LogicalOpsHandler::addMemcpyIfRequired(const LogicalOpPtr& node)
{
    /*   Insert the memcpy nodes for:
    *    i.   If any of the aliases is persistent   ///TODO [SW-5139] - Add support for persistent tensors
         ii.  If any of the aliases is already an alias
         iii. If the real tensor is already an alias:
               If cannot run as logical (strided)
         iv.  Duplicated input tensors in case they should become aliases
         v.    If any of the aliases is const (i.e. gaudi's static param)
         vi. Any of the aliases is real in a logical node from a previous pass
    */

    if (node->isShapeOperation())
    {
        return;
    }

    handleRealTensor(node);
    handleAliasedTensors(node);
}

void LogicalOpsHandler::handleNode(const LogicalOpPtr& logicalNode)
{
    addMemcpyIfRequired(logicalNode);

    logicalNode->runAndSetLogicalOp();
    HB_ASSERT(logicalNode->validateAlias(), "problem with alias tensor");
}

bool LogicalOpsHandler::shouldResetLogicalOp(const LogicalOpPtr& node)
{
    if (isForwardNode(*node)) return false;
    for (const TensorPtr& alias : node->getAliasTensors())
    {
        if (alias->isPersistent() && !Tensor::getRealTensor(alias)->isPersistent()) return true;
    }
    return false;
}

bool LogicalOpsHandler::swapLogicalAfterReal(const HabanaGraph& g, const TensorPtr& realTensor)
{
    bool     inputIsUserManagedTensor = g.isUserManagedDram(realTensor);
    NodePtr  producer                 = g.getTensorProducer(realTensor);
    unsigned numOfConsumers           = g.getTensorConsumers(realTensor).size();

    return !inputIsUserManagedTensor && producer && !producer->isLogicalOperation() && (numOfConsumers == 1) &&
           producer->canHandleStridedOutput(g.getDeviceType());
}

bool LogicalOpsHandler::wantBackwardDirectionShouldCallSwapDirection(const LogicalOpPtr& logicalNode)
{
    if (!logicalNode->canSwapAliasDirection()) return false;

    if (!isForwardNode(*logicalNode)) return false;  // already backwards

    if (isRealInLogical(logicalNode->getRealTensor())) return false;  // Backward will cause a memcpy

    if (logicalNode->getRunLogicalOperationDone()) return false;  // Already done
    return true;
}

bool LogicalOpsHandler::isSwapAliasDirectionProfitable(std::shared_ptr<LogicalOpNode> logicNode, const HabanaGraph& g)
{
    const auto& aliasTensors = logicNode->getAliasTensors();
    HB_ASSERT(aliasTensors.size() == 1, "Try to swap node with more than 1 alias tensors");
    bool swapAliasDirectionIsProfitable = g.isUserManagedDram(aliasTensors.front()) ||
                                          // If to be alias tensor is already aliased
                                          aliasTensors.front()->isAliasedTensor() ||
                                          // If in previous pass, a forward operation was solved and
                                          // the aliased tensor is already considered as real
                                          isRealInLogical(aliasTensors.front()) ||
                                          // If producer is not logical operation, and the logical node is
                                          // the only consumer, then swap direction is preferred
                                          // to avoid unnecessary internal Memcpy nodes
                                          swapLogicalAfterReal(g, logicNode->getRealTensor());
    return swapAliasDirectionIsProfitable;
}

bool LogicalOpsHandler::compareLogicalOps(const NodePtr& a, const NodePtr& b, bool isFwd)
{
    if (!a) return false;
    if (!b) return true;

    if (!b->isLogicalOperation()) return false;
    if (!a->isLogicalOperation()) return true;

    // now their both logical nodes
    LogicalOpPtr logicalA = std::dynamic_pointer_cast<LogicalOpNode>(a);
    LogicalOpPtr logicalB = std::dynamic_pointer_cast<LogicalOpNode>(b);

    // nodes that are in the opposite direction won't be handled now, so prefer iterating over them
    bool isDirectionA = isFwd ? isForwardNode(*logicalA) : isBackwardNode(*logicalA);
    bool isDirectionB = isFwd ? isForwardNode(*logicalB) : isBackwardNode(*logicalB);
    if (!isDirectionB) return false;
    if (!isDirectionA) return true;

    // now their both backward logical nodes. prefer handling node that doesn't add strides to the graph.
    if (logicalA->isAliasStrided()) return false;
    if (logicalB->isAliasStrided()) return true;

    return logicalA->getExecutionOrderedIndex() < logicalB->getExecutionOrderedIndex();
}

void LogicalOpsHandler::handleBackwardLogicalOp(const LogicalOpPtr& logicalNode)
{
    HB_ASSERT_PTR(logicalNode);
    bool isUserManagedTensor = m_graph.isUserManagedDram(logicalNode->getRealTensor());

    if (shouldResetLogicalOp(logicalNode))
    {
        resetLogicalOp(m_graph, logicalNode);
    }

    bool logicalOpDone = logicalNode->getRunLogicalOperationDone();

    LOG_TRACE(OPT_LOGICAL_OPS,
              "{}, isUserManagedTensor: {}, isAlias: {}, isRealInLogical: {}, canSwap: {}, isForward: {}, "
              "logicalOpDone: {}",
              logicalNode->getNodeName(),
              isUserManagedTensor,
              logicalNode->getRealTensor()->isAliasedTensor(),
              isRealInLogical(logicalNode->getRealTensor()),
              logicalNode->canSwapAliasDirection(),
              isForwardNode(*logicalNode),
              logicalOpDone);

    if (logicalOpDone) return;

    if (wantBackwardDirectionShouldCallSwapDirection(logicalNode) &&
        isSwapAliasDirectionProfitable(logicalNode, m_graph))
    {
        // TODO [SW-5139] - support all don't care logical operations
        // The node is a forward don't care logical operation with persistent output
        // Convert to backward, so real tensor is the output and handle node
        LOG_DEBUG(OPT_LOGICAL_OPS,
                  "{} is a forward logical don't care. Converting to backward",
                  logicalNode->getNodeName());
        logicalNode->swapAliasDirection();
    }

    if (isBackwardNode(*logicalNode))
    {
        LogicalOpVector nodesToHandle;
        nodesToHandle.push_back(logicalNode);
        auto consumers = m_graph.getTensorConsumers(logicalNode->getRealTensor());
        while (consumers.size() == 1)
        {
            if (nodesToHandle.back()->getRealTensor()->isUserManagedDram() ||
                nodesToHandle.back()->getRealTensor()->isRealInLogical())
                break;
            auto con = std::dynamic_pointer_cast<LogicalOpNode>(*consumers.begin());
            consumers.clear();

            if (con != nullptr && !con->getRunLogicalOperationDone() && con->canSwapAliasDirection() &&
                nodesToHandle.back()->canHandleStridedRealTensor())
            {
                HB_ASSERT(isForwardNode(*con),
                          "Backward nodes which are consumer to the current node should be resolved at this point");
                con->swapAliasDirection();
                LOG_DEBUG(OPT_LOGICAL_OPS,
                          "{} is a forward logical don't care. Converting to backward",
                          con->getNodeName());
                // Run from the real tensor backward
                nodesToHandle.push_back(con);
                consumers = m_graph.getTensorConsumers(con->getRealTensor());
            }
        }
        for (auto it = nodesToHandle.rbegin(); it != nodesToHandle.rend(); ++it)
        {
            handleNode(*it);
        }

        for (const TensorPtr& inputTensor : logicalNode->getAliasTensors())
        {
            NodePtr producer = m_graph.getTensorProducer(inputTensor);
            if (producer && producer->isLogicalOperation() && !producer->isDebug())
            {
                auto producerNode = std::dynamic_pointer_cast<LogicalOpNode>(producer);
                HB_ASSERT_PTR(producerNode);
                if (wantBackwardDirectionShouldCallSwapDirection(producerNode))
                {
                    // If one of the inputs is produced by a don't care logical operation
                    // Convert the producer from don't care to backward and handle it
                    LOG_DEBUG(OPT_LOGICAL_OPS,
                              "Node: {},  input producer: {},  produced by a don't care logical operation. "
                              "Converting to backward",
                              logicalNode->getNodeName(),
                              producerNode->getNodeName());
                    producerNode->swapAliasDirection();
                }
            }
        }
    }
}

// Handle all backward logical ops
// and some don't care nodes that fit the conditions described at the algorithm
void LogicalOpsHandler::handleAndRunLogicalOps(bool isForward)
{
    GraphTraversalGenerator::GraphTraversalComparator comp = [isForward](const NodePtr& a, const NodePtr& b) {
        return compareLogicalOps(a, b, /* isFwd */ isForward);
    };
    GraphTraversalGenerator graphTraversalGenerator(m_graph, /* reverse */ !isForward, comp);

    LOG_DEBUG(OPT_LOGICAL_OPS, "Handling {} logical operation.", isForward ? "Forward" : "Backward");
    while (!graphTraversalGenerator.empty())
    {
        auto nextNode = graphTraversalGenerator.getNext();
        if (!nextNode->isLogicalOperation() || nextNode->isShapeOperation() || nextNode->isDebug()) continue;
        if (isForward)
        {
            handleForwardLogicalOp(std::dynamic_pointer_cast<LogicalOpNode>(nextNode));
        }
        else
        {
            handleBackwardLogicalOp(std::dynamic_pointer_cast<LogicalOpNode>(nextNode));
        }
    }
}

void LogicalOpsHandler::handleForwardLogicalOp(const LogicalOpPtr& logicalNode)
{
    HB_ASSERT_PTR(logicalNode);
    bool logicalOpDone = logicalNode->getRunLogicalOperationDone();

    LOG_TRACE(OPT_LOGICAL_OPS,
              "{}: logicalNode: {}, logicalOpDone: {}.",
              HLLOG_FUNC,
              logicalNode->getNodeName(),
              logicalOpDone);

    // handle rest of the nodes
    if (!logicalOpDone)
    {
        if (isForwardNode(*logicalNode) == false)
        {
            LOG_ERR(OPT_LOGICAL_OPS, "Error! Backward node not handled {}", logicalNode->getNodeName());
            HB_ASSERT(false, "Error! Backward node not handled {}", logicalNode->getNodeName());
        }
        handleNode(logicalNode);
    }
}

bool LogicalOpsHandler::validMemoryRelocation(const LogicalOpPtr& logicalNode, const TensorPtr& t) const
{
    const TensorPtr& real          = Tensor::getRealTensor(t);
    const auto&      realProducers = m_graph.getRealProducers(real);
    const auto&      realConsumers = m_graph.getRealConsumers(real);

    auto sameBundle = [&](const NodePtr& n) { return areSameBundle(logicalNode, n); };
    bool valid = (real->getDenseSizeInElements() == t->getDenseSizeInElements() &&        // same num of elements
                  std::all_of(realProducers.begin(), realProducers.end(), sameBundle) &&  // same bundle
                  std::all_of(realConsumers.begin(), realConsumers.end(), sameBundle));

    if (!valid)
    {
        logAdjacentBundles(logicalNode, t);
    }

    return valid;
}

void LogicalOpsHandler::logAdjacentBundles(const LogicalOpPtr& logicalNode, const TensorPtr& t) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(OPT_LOGICAL_OPS)) return;

    const auto&      lnbi = logicalNode->getNodeAnnotation().bundleInfo;
    const TensorPtr& real = Tensor::getRealTensor(t);

    LOG_DEBUG(OPT_LOGICAL_OPS, "Logical Node {}:", logicalNode->getNodeName());
    LOG_DEBUG(OPT_LOGICAL_OPS, " bundle idx: {}", lnbi.is_set() ? std::to_string(lnbi->bundleIndex) : "N/A");

    const auto& realProducers = m_graph.getRealProducers(real);
    for (const auto& prod : realProducers)
    {
        const auto& bi = prod->getNodeAnnotation().bundleInfo;
        LOG_DEBUG(OPT_LOGICAL_OPS,
                  " Producer {}: bundle idx: {}",
                  prod->getNodeName(),
                  bi.is_set() ? std::to_string(bi->bundleIndex) : "N/A");
    }

    const auto& realConsumers = m_graph.getRealConsumers(real);
    for (const auto& cons : realConsumers)
    {
        const auto& bi = cons->getNodeAnnotation().bundleInfo;
        LOG_DEBUG(OPT_LOGICAL_OPS,
                  " Consumer {}: bundle idx: {}",
                  cons->getNodeName(),
                  bi.is_set() ? std::to_string(bi->bundleIndex) : "N/A");
    }
}

void LogicalOpsHandler::handleRealInLogical(const LogicalOpVector& sortedNodes) const
{
    LOG_DEBUG(OPT_LOGICAL_OPS, "{}: Marking real tensors in logical operations.", HLLOG_FUNC);
    for (const LogicalOpPtr& logicalNode : sortedNodes)
    {
        if (logicalNode->isDebug() || logicalNode->isShapeOperation()) continue;

        HB_ASSERT_PTR(logicalNode);

        TensorPtr realTensor = logicalNode->getRealTensor();
        realTensor->setIsRealInAliasing(true);
        LOG_TRACE(OPT_LOGICAL_OPS,
                  "{}: logicalNode: {}, realTensor: {}",
                  HLLOG_FUNC,
                  logicalNode->getNodeName(),
                  realTensor->getName());

        TensorVector aliasTensors = logicalNode->getAliasTensors();
        for (const TensorPtr& t : aliasTensors)
        {
            // generally, logical operations should not change the memory location of tensors
            if (t->getTensorAnnotation().memory.location == TENSOR_IN_SRAM && !t->inSram())
            {
                // with the exception of:
                HB_ASSERT(validMemoryRelocation(logicalNode, t),
                          "logical node {} changes location of tensor",
                          logicalNode->getNodeName());

                Tensor::getRealTensor(t)->setTensorInSram();  // in that case, mark the real tensor in sram.
            }
        }
    }
}

void LogicalOpsHandler::resetLogicalOp(const HabanaGraph& g, const LogicalOpPtr& logicNode)
{
    if (!logicNode) return;
    LOG_DEBUG(OPT_LOGICAL_OPS, "resetting node {}", logicNode->getNodeName());
    std::stack<LogicalOpPtr> logicalOpsToBeReset;
    logicalOpsToBeReset.push(logicNode);

    while (!logicalOpsToBeReset.empty())
    {
        LogicalOpPtr node = logicalOpsToBeReset.top();
        logicalOpsToBeReset.pop();

        // find all dependant logical nodes
        for (const TensorPtr& aliasTensor : node->getAliasTensors())
        {
            if (!aliasTensor) continue;
            // all nodes that use this tensor as their real tensor will be reset
            aliasTensor->setIsRealInAliasing(false);
            // find nodes that potentially use this tensor as their 'real tensor'
            auto dependantNodes = g.getTensorConsumers(aliasTensor);
            dependantNodes.push_back(g.getTensorProducer(aliasTensor));
            for (const NodePtr& dependantNode : dependantNodes)
            {
                // nodes that are node LogicalOps should be skipped since they use the "realInLogical" notation, and
                // node the "realInAlias" notation set by this pass.
                if (!dependantNode || (dependantNode == node) || !dependantNode->isLogicalOperation()) continue;
                LogicalOpPtr dependantLogicalNode = std::dynamic_pointer_cast<LogicalOpNode>(dependantNode);
                // skip logical ops that aren't done
                if (!dependantLogicalNode->getRunLogicalOperationDone()) continue;
                logicalOpsToBeReset.push(std::move(dependantLogicalNode));
            }
        }

        // reset logical op
        node->resetLogicalOp();
    }
}

void LogicalOpsHandler::handleSparseLayoutTensors(const LogicalOpVector& sortedNodes)
{
    for (const LogicalOpPtr& logicalNode : sortedNodes)
    {
        if (logicalNode->isDebug() || logicalNode->isShapeOperation()) continue;

        for (const TensorPtr& tensor : logicalNode->getAliasTensors())
        {
            if (tensor->isDenseLayout()) continue;  // tensor didn't change its strides

            if (tensor->isZeroSizedDataTensor()) continue;

            if (isForwardNode(*logicalNode))
            {
                const auto& consumers = m_graph.getTensorConsumers(tensor);
                for (const NodePtr& consumer : consumers)
                {
                    if (!consumer->canHandleStridedInput(m_graph.getDeviceType()))
                    {
                        LOG_TRACE(OPT_LOGICAL_OPS,
                                  "{}: strided input not supported, memcpy added",
                                  consumer->getNodeName());
                        NodePtr copy = GraphEditor::insertMemcpyForInput(m_graph, consumer, tensor);
                        // handle corner case where consumer has the same input more than once
                        for (unsigned i = 0; i < consumer->getNumInputs(); i++)
                        {
                            if (consumer->getInput(i) == tensor)
                            {
                                GraphEditor::replaceInput(m_graph, consumer, i, copy->getOutput(0));
                            }
                        }
                    }
                }
            }
            else
            {
                const NodePtr& producer = m_graph.getTensorProducer(tensor);
                HB_ASSERT_PTR(producer);  // aliased tensor must have producer
                if (!tensor->isTrivialStrided() && !producer->canHandleStridedOutput(m_graph.getDeviceType()))
                {
                    LOG_TRACE(OPT_LOGICAL_OPS,
                              "{}: strided output not supported, memcpy added",
                              producer->getNodeName());
                    GraphEditor::insertMemcpyForOutput(m_graph, producer, tensor);
                }
            }
        }
    }
}

TensorSet LogicalOpsHandler::getPotentialRealTensors(const HabanaGraph& g, const TensorPtr& tensor, AliasDirection direction)
{
    TensorSet ret;
    HB_ASSERT_PTR(tensor);
    std::stack<TensorPtr> toCheck;
    toCheck.push(tensor);
    while (!toCheck.empty())
    {
        TensorPtr t = toCheck.top();
        toCheck.pop();
        ret.insert(t);
        // For backward direction, go over all the tensor consumers, and collect the outputs of all the logical
        // consumers. Ignore nodes that can't be backward (symmetrical for fwd direction).
        auto nextNodes = direction == INPUT_TO_OUTPUT ? g.getTensorConsumers(t) : NodeList({g.getTensorProducer(t)});
        for (const NodePtr& n : nextNodes)
        {
            if (!n->isLogicalOperation() || n->isShapeOperation()) continue;

            const auto& logicalNode = std::static_pointer_cast<LogicalOpNode>(n);
            HB_ASSERT_PTR(logicalNode);
            if (logicalNode->getAliasDirection() != direction && !logicalNode->canSwapAliasDirection()) continue;

            HB_ASSERT(logicalNode->getAliasDirection() == direction || logicalNode->getAliasTensors().size() == 1,
                      "Found {} logical node with {} alias tensors that can swap direction",
                      (direction == OUTPUT_TO_INPUT) ? "forward" : "backward",
                      logicalNode->getAliasTensors().size());

            toCheck.push(logicalNode->getAliasDirection() == direction ? logicalNode->getRealTensor()
                                                                       : logicalNode->getAliasTensors().front());
        }
    }
    return ret;
}
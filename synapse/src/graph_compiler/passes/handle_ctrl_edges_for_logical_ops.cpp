#include "habana_graph.h"
#include "strided_insert_node.h"
#include "slice_insert_node.h"
#include "bundle_plane_graph.h"
#include "handle_memory_reuse.h"
#include "register_memory_coherence.h"
#include "node_factory.h"
#include "handle_logical_operations.h"
#include "gaudi_scheduler.h"
#include "operand_reuse_logical_node.h"
#include <stack>
#include "handle_ctrl_edges_for_logical_ops.h"

CtrlEdgesHandler::CtrlEdgesHandler(HabanaGraph& graph)
: m_graph(graph), m_firstNodeInBundle(GaudiScheduler::findFirstNodePerBundle(graph)), m_bPGraphCreated(false)
{
}

CtrlEdgesHandler::~CtrlEdgesHandler()
{
    if (m_bPGraphCreated)
    {
        m_graph.discardBPGraph();
    }
}

bool CtrlEdgesHandler::isDynamicPersistent(const TensorPtr& t)
{
    std::optional<unsigned> firstDynamicDimIndex = t->getFirstDynamicDimIndex();

    // When only outer dim is dynamic, it can be treated as if it was static and copied out
    return firstDynamicDimIndex && t->isPersistent() && (*firstDynamicDimIndex + 1) < t->getDim();
}

// checks if "real" is somewhere along the aliasing chain of "alias"
bool CtrlEdgesHandler::isInAliasChain(const TensorPtr& alias, const TensorPtr& real)
{
    TensorPtr realTensor = alias;
    if (realTensor == real) return true;
    while (realTensor->isAliasedTensor())  // move up the alias chain
    {
        realTensor = realTensor->getAliasTensor();
        if (realTensor == real) return true;
    }
    return false;
}

NodeSet CtrlEdgesHandler::getTensorProducerAndConsumers(const TensorPtr& t) const
{
    NodeSet ret       = {m_graph.getTensorProducer(t)};
    auto    consumers = m_graph.getTensorConsumers(t);
    ret.insert(consumers.begin(), consumers.end());
    return ret;
}

NodeSet CtrlEdgesHandler::getSurroundingRealNodes(const TensorPtr& t) const
{
    std::stack<NodePtr> toCheck;
    NodeSet             visited;
    NodeSet             ret;
    for (const NodePtr& n : getTensorProducerAndConsumers(t))
    {
        toCheck.push(n);
        visited.insert(n);
    }
    while (!toCheck.empty())
    {
        NodePtr node = toCheck.top();
        toCheck.pop();
        if (!node || node->isShapeOperation()) continue;
        if (node->isLogicalOperation())
        {
            for (const TensorPtr& operand : node->getOperands())
            {
                if (!operand) continue;
                if (operand->isShapeTensor()) continue;
                if (!isInAliasChain(operand, t)) continue;
                for (const NodePtr& connectedNode : getTensorProducerAndConsumers(operand))
                {
                    if (visited.find(connectedNode) != visited.end()) continue;
                    visited.insert(connectedNode);
                    toCheck.push(connectedNode);
                }
            }
        }
        else  // real node
        {
            ret.insert(node);
        }
    }
    return ret;
}

NodeSet CtrlEdgesHandler::getRealConsumers(const TensorPtr& t) const
{
    NodeSet ret = getSurroundingRealNodes(t);
    eraseIf(ret, [&t](const NodePtr& n) {
        for (const TensorPtr& in : n->getInputs())
        {
            if (in && isInAliasChain(in, t)) return false;
        }
        return true;
    });
    return ret;
}

NodeSet CtrlEdgesHandler::getRealProducers(const TensorPtr& t) const
{
    NodeSet ret = m_graph.getRealProducers(t);
    eraseIf(ret, [&t](const NodePtr& n) {
        for (const TensorPtr& out : n->getOutputs())
        {
            if (out && isInAliasChain(out, t)) return false;
        }
        return true;
    });
    return ret;
}

const BundlePlane* CtrlEdgesHandler::getBPGraph() const
{
    if (!m_bPGraphCreated)
    {
        m_graph.constructBPGraph(true /* use anotations */);
        m_bPGraphCreated = true;
    }
    return m_graph.getBPGraph();
}

bool CtrlEdgesHandler::areSameNode(const NodePtr& n1, const NodePtr& n2) const
{
    const BundlePlane* bp = getBPGraph();
    return bp->getBundlePlaneRepresentation(n1) == bp->getBundlePlaneRepresentation(n2);
}

// checks if there is a graph-path between source and target (including bundles).
bool CtrlEdgesHandler::areConnected(const NodePtr& source, const NodePtr& target) const
{
    const BundlePlane* bp       = getBPGraph();
    const NodePtr&     bpSource = bp->getBundlePlaneRepresentation(source);
    const NodePtr&     bpTarget = bp->getBundlePlaneRepresentation(target);
    if (bpSource == bpTarget && source->getNodeAnnotation().bundleInfo.is_set())
    {
        HB_ASSERT(target->getNodeAnnotation().bundleInfo.is_set(), "{} not part of bundle", target->getNodeName());
        // same bundle, connectivity is driven by bundle order
        return source->getNodeAnnotation().bundleInfo->operationIndex <=
               target->getNodeAnnotation().bundleInfo->operationIndex;
    }
    return bp->getBundlePlaneGraph()->getNumberOfPaths(bpSource, bpTarget, Node::TENSOR_TYPE_ALL) > 0;
}

NodeVector CtrlEdgesHandler::getReductionNodes(const HabanaGraph& g)
{
    NodeVector ret;
    for (const NodePtr& node : g.getTopoSortedNodes())
    {
        if (!isReductionOp(node->getNodeType())) continue;
        if (node->isLogicalOperation() && !node->isDebug())
        {
            LOG_TRACE(CTRL_LOGICAL_OP, "{}, adding node {} to handling list", HLLOG_FUNC, node->getNodeName());
            ret.push_back(std::dynamic_pointer_cast<LogicalOpNode>(node));
        }
    }
    return ret;
}

NodePtr CtrlEdgesHandler::plantReductionMemcopy(const NodePtr& node, const TensorPtr& insertTensor)
{
    LOG_DEBUG(CTRL_LOGICAL_OP, "{}, node {}. planting memcopy to avoid cycle", HLLOG_FUNC, node->getNodeName());
    NodePtr memcpy = GraphEditor::insertMemcpyForInput(m_graph, node, insertTensor, insertTensor->location());
    HB_ASSERT_PTR(memcpy);
    memcpy->getOutput(0)->cloneAliasInfo(insertTensor);
    insertTensor->resetAliasing();
    // since the memcpy is planted to avoid a cycle, we don't want it to accidentally belong to the producer bundle
    memcpy->getNodeAnnotation().bundleInfo = node->getNodeAnnotation().bundleInfo;
    return memcpy;
}

NodeSet CtrlEdgesHandler::replaceBlockedNodesWithFirstInBundle(const NodeSet& blockedNodes, const NodePtr& node)
{
    NodeSet     ret        = {};
    const auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
    for (const NodePtr& n : blockedNodes)
    {
        if (n->getNodeAnnotation().bundleInfo.is_set() &&
            (!bundleInfo.is_set() || bundleInfo->bundleIndex != n->getNodeAnnotation().bundleInfo->bundleIndex))
        {
            auto it = m_firstNodeInBundle.find(n->getNodeAnnotation().bundleInfo->bundleIndex);
            HB_ASSERT(it != m_firstNodeInBundle.end(), "{} bundle not found!", __func__);
            ret.insert(it->second);
        }
        else
        {
            ret.insert(n);
        }
    }
    return ret;
}

// If there is a chain of reductions, we want to find the direct memset producer of this reduction
NodePtr CtrlEdgesHandler::findDirectReductionMemset(const NodePtr& reduction) const
{
    HB_ASSERT(reduction->getNodeType() == Node::TYPE_INTERNAL_REDUCTION,
              "expected reduction for node {}",
              reduction->getNodeName());
    NodeSet memsetProducer =
        m_graph.getNodeRealProducersExcept(reduction, Node::TENSOR_TYPE_DATA, [](const NodePtr& node) {
            return node && node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION;
        });
    auto memsetNodesIter = find_if(memsetProducer.begin(), memsetProducer.end(), [](const NodePtr& node) {
        return node && node->isMemset();
    });
    return (memsetNodesIter != memsetProducer.end()) ? *memsetNodesIter : nullptr;
}

NodeSet CtrlEdgesHandler::getReductionProducersExceptMemset(const NodePtr& reductiontNode,
                                                            const NodePtr& memsetNode) const
{
    NodeSet reductionProducersWithoutMemset;

    const auto& reductionProducers = m_graph.getNodeRealProducers(reductiontNode, Node::TENSOR_TYPE_DATA);
    std::copy_if(reductionProducers.begin(),
                 reductionProducers.end(),
                 std::inserter(reductionProducersWithoutMemset, reductionProducersWithoutMemset.end()),
                 [&memsetNode](const NodePtr& n) { return n != memsetNode; });

    return reductionProducersWithoutMemset;
}

/*
    handling implicit WAW control dependencies of reductions.
    currently only for strided insert - operand 0, must be written before operand 1.
    in the future we might move here handling of ReductionNode as well (memset producer before the rest)
*/
void CtrlEdgesHandler::handleReductionControl()
{
    NodeVector reductionNodes = getReductionNodes(m_graph);
    for (const auto& reductionNode : reductionNodes)
    {
        // the producer for the original tensor of StridedInsert must run before the producers of the view tensor.
        if (isInsertReduction(reductionNode->getNodeType()))
        {
            LOG_TRACE(CTRL_LOGICAL_OP, "{}, handling node {}", HLLOG_FUNC, reductionNode->getNodeName());
            const TensorPtr& original         = reductionNode->getInput(0);
            const NodePtr&   blockingProducer = m_graph.getTensorProducer(original);  // producer that should run first
            if (blockingProducer == nullptr)
            {
                continue;  // nothing to block. if needed to block something, user has placed control edges
            }
            for (unsigned i = 1; i < reductionNode->getNumInputsDataTensors(); i++)
            {
                TensorPtr insert           = reductionNode->getInput(i);
                auto      blockedProducers = getRealProducers(insert);  // producers for the view tensor

                bool plantCopy = std::any_of(blockedProducers.begin(), blockedProducers.end(), [&](const NodePtr& n) {
                    return n != blockingProducer && areConnected(n, blockingProducer);  // cycle in graph
                });
                if (plantCopy)  // if the view producers cannot be blocked by the original tensor producer
                {
                    // this is now the real producer for the view tensor
                    blockedProducers = {plantReductionMemcopy(reductionNode, insert)};
                }
                // plant explicit control dependencies.
                // use real blocking node, a cycle cannot be formed (see relaxControlDeps)
                NodeSet actualBlockingNodes = getRealProducers(original);
                if (actualBlockingNodes.empty() || blockedProducers.empty()) continue;
                // in order to make sure we don't place unbundled nodes between nodes in the bundle, we must
                // set dependencies from the first node of each bundle.
                blockedProducers = replaceBlockedNodesWithFirstInBundle(blockedProducers, reductionNode);
                LOG_TRACE(CTRL_LOGICAL_OP,
                          "{}, node {}. adding reduction control",
                          HLLOG_FUNC,
                          reductionNode->getNodeName());
                m_graph.addControlDependency(actualBlockingNodes, blockedProducers, Tensor::ControlEdgeType::SYNC);
            }
        }
        // Handles memset producers for reduction
        if (reductionNode->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            const NodePtr& blockingMemset = findDirectReductionMemset(reductionNode);
            if (blockingMemset == nullptr)  // if there is no memset - no need to handle
            {
                continue;
            }

            NodeSet blockedProducers = getReductionProducersExceptMemset(reductionNode, blockingMemset);
            HB_ASSERT(!blockedProducers.empty(), "reduction node {} has no producers", reductionNode->getNodeName());

            LOG_TRACE(CTRL_LOGICAL_OP,
                      "{}, node {}. adding reduction control",
                      HLLOG_FUNC,
                      reductionNode->getNodeName());
            m_graph.addControlDependency({blockingMemset}, blockedProducers, Tensor::ControlEdgeType::SCHEDULE);
        }
    }
}

NodePtr CtrlEdgesHandler::findPermutationLogicalTransposeProducer(const TensorPtr& t) const
{
    NodePtr transposeProducer = nullptr;

    if (t->isPersistent() && t->getPermutation().has_value())
    {
        // not supported. should get to this situation that is handled in a previous pass
        HB_ASSERT(m_graph.getNumberOfTensorConsumers(t) == 0,
                  "permuted tensor {} has a consumer and a producer!",
                  t->getName());

        transposeProducer = m_graph.getTensorProducer(t);
        HB_ASSERT(transposeProducer && transposeProducer->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE,
                  "permuted tensor {} with non-logical-transpose producer",
                  t->getName());
        LOG_DEBUG(CTRL_LOGICAL_OP,
                  "{}, operand: {}, planting memcopy before logical transpose {}",
                  HLLOG_FUNC,
                  t->getName(),
                  transposeProducer->getNodeName());
    }
    return transposeProducer;
}

/*
    producer -> [t] -> consumers
    will turn into:
    producer -> [copy_t] -> consumer
                      `-> (copy) -> [t]
    need to reset logical ops surrounding [t] so that aliases will be corrected in next handleLogicalOperations pass.
*/
NodePtr CtrlEdgesHandler::addCopyOutForTensor(TensorPtr t)
{
    // in case there is a "logical permutation transpose" node, we need to plant the copy before that node.
    NodePtr transposeProducer = findPermutationLogicalTransposeProducer(t);
    if (transposeProducer)
    {
        t = transposeProducer->getInput(0);
    }

    // only logical transpose nodes are allowed to "touch" such tensor
    HB_ASSERT(!t->getPermutation().has_value(), "{}: cannot copy out a permuted tensor! {}", __func__, t->getName());
    // only serialize node is allowed to produce such tensor
    HB_ASSERT(!isDynamicPersistent(t), "{}: cannot copy out a dynamic user tensor! {}", __func__, t->getName());

    LOG_INFO(CTRL_LOGICAL_OP, "{}, operand: {}, planting memcopy", HLLOG_FUNC, t->getName());

    // create new tensor copy
    TensorPtr copy = t->clone(false, false, false);
    copy->setName(t->getName() + "_internal_memcopy");

    // add memcopy
    NodePtr memcpyNode =
        NodeFactory::createNode({copy}, {t}, nullptr, NodeFactory::memcpyNodeTypeName, t->getName() + "_internal");
    // connect producer with copy output
    NodePtr producer = m_graph.getTensorProducer(t);
    HB_ASSERT(producer != nullptr, "producer of blocked tensor {} doesn't exist, cycle cannot be solved", t->getName());
    GraphEditor::replaceTensor(m_graph, producer, t, copy);
    if (producer->isLogicalOperation())
    {
        LogicalOpsHandler::resetLogicalOp(m_graph, std::dynamic_pointer_cast<LogicalOpNode>(producer));
    }

    for (NodePtr consumer : m_graph.getTensorConsumers(t))
    {
        if (consumer == transposeProducer) continue;  // don't change the following permutation transpose

        GraphEditor::replaceTensor(m_graph, consumer, t, copy);
        if (consumer->isLogicalOperation())
        {
            LogicalOpsHandler::resetLogicalOp(m_graph, std::dynamic_pointer_cast<LogicalOpNode>(consumer));
        }
    }

    GraphEditor::addNode(m_graph, memcpyNode);
    return memcpyNode;
}

bool CtrlEdgesHandler::isOverlap(const NodePtr& blocking, const NodePtr& blocked) const
{
    for (const TensorPtr& out : blocked->getOutputs())
    {
        if (!out) continue;
        for (const TensorPtr& t : blocking->getOperands())
        {
            if (!t) continue;
            if (MemoryReuseHandler::isStridedOverlap(out, t)) return true;
        }
    }
    return false;
}

bool CtrlEdgesHandler::doesControlExist(const NodeSet& blocking, const NodeSet& blocked) const
{
    if (blocked.size() * blocking.size() > m_maxBlockingBlockedProduct)
    {
        // checking graph paths here is an optimization, to avoid unnecessary control edges.
        // if the check will takes too much compile time - don't do it.
        return false;
    }
    bool ctrlExists = std::all_of(blocking.begin(), blocking.end(), [&](const NodePtr& src) {
        return std::all_of(blocked.begin(), blocked.end(), [&](const NodePtr& dst) {
            return m_graph.getNumberOfPaths(src, dst, Node::TENSOR_TYPE_ALL) > 0;
        });
    });
    return ctrlExists;
}

void CtrlEdgesHandler::handleOperandReuseInternalLogicalNodesControl()
{
    const auto& nodes = m_graph.getNodes();
    NodeVector  operandReuseNodes;
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(operandReuseNodes), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_OPERAND_REUSE_INTERNAL;
    });

    for (const auto& node : operandReuseNodes)
    {
        const auto& operandReuseNode = std::dynamic_pointer_cast<OperandReuseInternalLogicalNode>(node);
        HB_ASSERT_PTR(operandReuseNode);

        HB_ASSERT(LogicalOpsHandler::isBackwardNode(*operandReuseNode),
                  "Operand reuse logical node should be backward");
        const auto& aliasedInput  = operandReuseNode->getAliasTensor();
        const auto& realConsumers = getRealConsumers(aliasedInput);

        if (realConsumers.empty()) continue;  // nothing to do

        bool insertMemcpy = std::any_of(realConsumers.begin(),
                                        realConsumers.end(),
                                        [this, &operandReuseNode](const NodePtr& realConsumer) {
                                            return areConnected(operandReuseNode, realConsumer);
                                        });

        if (insertMemcpy)
        {
            LOG_DEBUG(CTRL_LOGICAL_OP,
                      "exists control between {} and the aliased input real consumers, insert memcpy",
                      operandReuseNode->getNodeName());
            LogicalOpsHandler::resetLogicalOp(m_graph, operandReuseNode);
            GraphEditor::insertMemcpyForInput(m_graph, operandReuseNode, aliasedInput);
        }
        else
        {
            LOG_DEBUG(CTRL_LOGICAL_OP,
                      "set control dependency he aliased input real consumers and {}",
                      operandReuseNode->getNodeName());

            m_graph.addControlDependency(realConsumers, {operandReuseNode});
        }
    }
}

void CtrlEdgesHandler::handleMemoryCoherence()
{
    HB_ASSERT_PTR(m_graph.getGraphAnnotation().memoryCoherence);  // created at the beginning of the compilation
    const auto& memoryCoherence = m_graph.getGraphAnnotation().memoryCoherence;
    for (const auto& coherenceMap : memoryCoherence->getAllSectionsTensorCoherence())
    {
        handleMemoryCoherence(coherenceMap);
    }
}

/*
    enforce the memory coherence of all tensors.
    check for all WriteAfterRead and WriteAfterWrite dependencies that concern section tensors.
    add control dependencies if necassery.
*/
void CtrlEdgesHandler::handleMemoryCoherence(const TensorCoherenceMapping::TensorCoherence& allSectionTensors)
{
    for (auto it : allSectionTensors)
    {
        const TensorVector& sectionTensors = it.second;
        if (sectionTensors.size() < 2) continue;  // no overlapping tensors in this section
        for (auto firstIt = sectionTensors.begin(); std::next(firstIt) != sectionTensors.end(); firstIt++)
        {
            TensorPtr t1 = *firstIt;
            // get blocking nodes - readers/writers of t1
            NodeSet blocking = getRealConsumers(t1);
            NodeSet blockingProducers = getRealProducers(t1);
            blocking.insert(blockingProducers.begin(), blockingProducers.end());

            for (auto secondIt = std::next(firstIt); secondIt != sectionTensors.end(); secondIt++)
            {
                TensorPtr t2 = *secondIt;
                if (!MemoryReuseHandler::isStridedOverlap(t1, t2)) continue;         // no overlap
                NodeSet blocked = getRealProducers(t2);                              // WAR dependency

                // a path from every blocking node to all blocked nodes  already exists
                if (doesControlExist(blocking, blocked)) continue;

                // check is a cycle will be formed by adding addition control dependencies
                bool plantCopy = std::any_of(blocked.begin(), blocked.end(), [&](const NodePtr& blocked) {
                    return std::any_of(blocking.begin(), blocking.end(), [&](const NodePtr& blocking) {
                        return isOverlap(blocking, blocked) && blocked != blocking && areConnected(blocked, blocking);
                    });
                });
                if (plantCopy)
                {
                    // in this case, adding a control edge would create a cycle. so add a copy to current operand
                    blocked = {addCopyOutForTensor(t2)};
                }

                // add additional explicit control edges
                for (const NodePtr& blockingNode : blocking)
                {
                    for (const NodePtr& blockedNode : blocked)
                    {
                        if (areSameNode(blockingNode, blockedNode)) continue;
                        if (!isOverlap(blockingNode, blockedNode)) continue;
                        LOG_INFO(CTRL_LOGICAL_OP,
                                 "{}, adding ctrl {}->{} because of tensors {}->{}",
                                 HLLOG_FUNC,
                                 blockingNode->getNodeName(),
                                 blockedNode->getNodeName(),
                                 t1->getName(),
                                 t2->getName());
                        m_graph.addControlDependency(blockingNode, blockedNode, Tensor::ControlEdgeType::SCHEDULE);
                    }
                }
            }
        }
    }
}

void CtrlEdgesHandler::handleCtrlEdges()
{
    LOG_DEBUG(CTRL_LOGICAL_OP, "{}, handling reduction control...", HLLOG_FUNC);
    handleReductionControl();
    handleOperandReuseInternalLogicalNodesControl();

    if (GCFG_HANDLE_MEMORY_COHERENCE.value())
    {
        LOG_DEBUG(CTRL_LOGICAL_OP, "{}, handling memory coherence...", HLLOG_FUNC);
        handleMemoryCoherence();
    }
}

bool handleCtrlEdgesForLogicalNodes(HabanaGraph& g)
{
    CtrlEdgesHandler handler(g);
    handler.handleCtrlEdges();
    return true;
}

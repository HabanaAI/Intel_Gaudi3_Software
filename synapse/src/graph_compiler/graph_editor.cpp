#include "graph_editor.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "gaudi_scheduler.h"
#include "bundle_plane_graph.h"
#include "register_memory_coherence.h"
#include "types.h"
#include <algorithm>
#include "node_tensor_accessor.h"

template<class CONTAINER>
static std::optional<synNodeId> getMinParentId(const CONTAINER& nodes)
{
    if (nodes.empty()) return std::nullopt;
    const NodePtr& minNode = *std::min_element(nodes.begin(), nodes.end(), [](const NodePtr& n1, const NodePtr& n2) {
        return n1->getParentId() < n2->getParentId();
    });

    return minNode->getParentId();
}

std::string_view GraphEditor::replaceNodeReturnStatusToString(ReplaceNodeReturnStatus status)
{
#if MAGIC_ENUM_SUPPORTED
    return magic_enum::enum_name(status);
#else
    switch (status)
    {
        TRANSLATE_ENUM_TO_STRING(REPLACE_NODE_SUCCESS)
        TRANSLATE_ENUM_TO_STRING(REPLACE_FAILED_INVALID_NEW_NODES)
        TRANSLATE_ENUM_TO_STRING(REPLACE_FAILED_INGROUP_DEPS)
    };

    return "ERROR - UNKNOWN RETURN TYPE";
#endif
}

bool GraphEditor::canEliminateTensor(const HabanaGraph& g, const TensorPtr t,
                                     unsigned maxOutConsumers, bool eliminateStaticParams/*=false*/)
{
    // cannot eliminate tensor if:
    if (g.getNumberOfTensorConsumers(t) > maxOutConsumers ||  // there are multiple consumers
        g.isUserManagedDram(t))                               // tensor is in user managed DRAM
    {
        // some passes (such as mme_activation_pass) allow to eliminate graph inputs if they are static params.
        return g.isInputTensor(t) && t->isStaticParam() && eliminateStaticParams;
    }
    return true;
}

NodeList GraphEditor::insertMemcpies(HabanaGraph& g,
                                     NodePtr currNode,
                                     bool insertForInputs,
                                     const std::vector<unsigned>& tensorIndices,
                                     TensorLocation location)
{
    NodeList ret;
    const TensorVector& tensors = (insertForInputs) ? currNode->getInputs() : currNode->getOutputs();

    synNodeId parentId = currNode->getParentId();

    GraphEditor::editNode(g,
                          currNode,
                          [&](const NodePtr& node)
    {
        for (unsigned tensorIndex : tensorIndices)
        {
            HB_ASSERT(tensorIndex < tensors.size(), "{}: {} is invalid {} index, num of {}: {}",
                      currNode->getNodeName(),
                      tensorIndex,
                      insertForInputs ? "input" : "output",
                      insertForInputs ? "inputs" : "outputs",
                      tensors.size());
            const TensorPtr& originalTensor = tensors.at(tensorIndex);
            HB_ASSERT_PTR(originalTensor);
            TensorPtr copyTensor = originalTensor->clone(false, false, false);
            if (location != UNDEFINED_LOCATION)
            {
                HB_ASSERT((location == TENSOR_IN_SRAM) || (location == TENSOR_IN_DRAM),
                          "Unsupported tensor memory location");
                if (location == TENSOR_IN_SRAM)
                {
                    copyTensor->setTensorInSram();
                }
                else
                {
                    copyTensor->setTensorInWorkspace();
                }
            }

            copyTensor->setName(originalTensor->getName() + "_memcpy");

            NodePtr memcpyNode = NodeFactory::createNode({insertForInputs ? originalTensor : copyTensor},
                                                         {insertForInputs ? copyTensor : originalTensor},
                                                         nullptr,
                                                         NodeFactory::memcpyNodeTypeName,
                                                         copyTensor->getName() + "_internal");

            memcpyNode->setParentId(parentId);

            // New node should be created with the original's bundle id and op index for execution schedule to work
            memcpyNode->getNodeAnnotation().bundleInfo = currNode->getNodeAnnotation().bundleInfo;
            if (originalTensor->isEnforcedOutput())
            {
                copyTensor->enforceOutput(false);
            }

            /* Need to reset the strides of the cloned tensor, the will be set by the logical op */
            copyTensor->setDenseStrides();

            // The cloned tensor has the same tensor annotations as the realTensor, specifically sectionId.
            // This can cause an issue since the cloned tensor will be added to the same
            // multibuffer as realTensor, although it might not fit in terms of liveness. To prevent this from
            // happening, assign a new sectionId for each copyTensor (same sectionId for all slices).
            g.resetMultibufferInfo(copyTensor);

            HB_ASSERT(currNode != nullptr, "trying to insert a memcpy node after a null pointer");

            GraphEditor::addNode(g, memcpyNode, false);

            // Maintain tracking of origin nodes for debug purposes
            memcpyNode->setOriginNodes(currNode->getOriginNodes());

            LOG_DEBUG(GC,
                      "{}: inserted {}, to copy tensor {}",
                      HLLOG_FUNC,
                      memcpyNode->getNodeName(),
                      originalTensor->getName());

            if (insertForInputs)
            {
                node->replaceInput(tensorIndex, copyTensor);
            }
            else
            {
                node->replaceOutput(tensorIndex, copyTensor);
            }
            ret.push_back(memcpyNode);
        }
    });
    return ret;
}

NodePtr GraphEditor::insertMemcpy(HabanaGraph& g,
                                  NodePtr currNode,
                                  bool insertForInput,
                                  const unsigned tensorIndex,
                                  TensorLocation location)
{
    return insertMemcpies(g, currNode, insertForInput, std::vector<unsigned>({tensorIndex}), location).front();
}

NodePtr GraphEditor::insertMemcpy(HabanaGraph& g,
                                  NodePtr currNode,
                                  bool insertForInput,
                                  const TensorPtr& oldTensor,
                                  TensorLocation location)
{
    const TensorVector& tensors = (insertForInput) ? currNode->getInputs() : currNode->getOutputs();
    auto tensorIndexIt = std::find(tensors.begin(), tensors.end(), oldTensor); // return first occur
    HB_ASSERT(tensorIndexIt != tensors.end(), "{}: {} is not node {}",
              currNode->getNodeName(),
              oldTensor->getName(),
              insertForInput ? "input" : "output");
    unsigned tensorIndex = std::distance(tensors.begin(), tensorIndexIt);
    return insertMemcpies(g, currNode, insertForInput, std::vector<unsigned>({tensorIndex}), location).front();
}

NodePtr GraphEditor::insertMemcpyForInput(HabanaGraph& g, NodePtr currNode,
                                          const unsigned tensorIndex, TensorLocation location)
{
    /* Insert the the memcpy node between the input tensor and the given currNode
     *
     *     original graph: --[inTensor]--------------------------------->(currNode)--
     *     modified graph: --[inTensor]-->(memcpyNode)--[clonedTensor]-->(currNode)--
     * */
    return insertMemcpy(g, currNode, true /* insert for input */, tensorIndex, location);
}

NodePtr GraphEditor::insertMemcpyForOutput(HabanaGraph& g, NodePtr prevNode,
                                           const unsigned tensorIndex, TensorLocation location)
{
    /* Insert the the memcpy node between the previous node and the given output tensor
     * to ensures that the user may use the outTensor's data (persistent tensors)
     *
     *    original graph: (prevNode)----------------------------------[outTensor]-->
     *    modified graph: (prevNode)--[cloned-Tensor]-->(memcpyNode)--[outTensor]-->
     *    */
    return insertMemcpy(g, prevNode, false /* insert for input */, tensorIndex, location);
}

NodePtr GraphEditor::insertMemcpyForInput(HabanaGraph& g, NodePtr currNode,
                                          const TensorPtr& oldTensor, TensorLocation location)
{
    /* Insert the the memcpy node between the input tensor and the given currNode
     *
     *     original graph: --[inTensor]--------------------------------->(currNode)--
     *     modified graph: --[inTensor]-->(memcpyNode)--[clonedTensor]-->(currNode)--
     * */
    return insertMemcpy(g, currNode, true /* insert for input */, oldTensor, location);
}

NodePtr GraphEditor::insertMemcpyForOutput(HabanaGraph& g, NodePtr prevNode,
                                           const TensorPtr& oldTensor, TensorLocation location)
{
    /* Insert the the memcpy node between the previous node and the given output tensor
     * to ensures that the user may use the outTensor's data (persistent tensors)
     *
     *    original graph: (prevNode)----------------------------------[outTensor]-->
     *    modified graph: (prevNode)--[cloned-Tensor]-->(memcpyNode)--[outTensor]-->
     *    */
    return insertMemcpy(g, prevNode, false /* insert for input */, oldTensor, location);
}

bool GraphEditor::isPossibleLoopDueToExtEdges(HabanaGraph& g, const NodeSet& nodesSet)
{
    // check for possible loop due to edges external to this node set
    NodeComparator comp;

    for (NodePtr node : nodesSet)
    {
        NodeSet consumers;
        runOnTensorsForType<Node::USAGE_OUTPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& out) {
            NodeList currConsumers = g.getTensorConsumers(out);
            consumers.insert(currConsumers.begin(), currConsumers.end());
        });

        if (consumers.size() <= 1)
        {
            // only one consumer - this node does not create risk
            continue;
        }

        // looking for differences between all of this node consumers and the node set
        NodeVector diff;
        std::set_difference(consumers.begin(),
                            consumers.end(),
                            nodesSet.begin(),
                            nodesSet.end(),
                            std::back_inserter(diff), comp);

        if (diff.size() == 0) // all of the consumers are part of the set
        {
            // this node has more than one consumer, all of them are part of the set.
            // this set is branching and each branch may block each other from external offsprings.
            LOG_INFO(GC, "node {} has {} consumers in the set, may create loop",
                     node->getNodeName(), consumers.size());
            return true;

        }
        else if (diff.size() < consumers.size()) // we have both internal and external consumers
        {
            // one of this node consumers (or its child) may produce an output (control or data) that is consumed by
            // other node in the set and can cause a loop
            LOG_INFO(GC,
                     "node {} has consumers both in the fused set and outside, may create loop",
                     node->getNodeName());
            return true;

        }
        else if (diff.size() == consumers.size()) // all of the consumers are external
        {
            // so no possible loop -  would have been caught earlier as an acylic graph
            continue;
        }
        else
        {
            HB_ASSERT(false, "Unexpected diff size");
        }
    }

    return false;
}

/*
 * this function checks if fusion of the given group into one node will create a loop in the graph
 * considering both data and control tensors
 * */
bool GraphEditor::isLoopDueFusionInNodeSet(HabanaGraph& g, const NodeSet& nodesSet)
{
    NodePtr fusedNode;
    return isLoopDueFusionInUpdateGraph(g, nodesSet, fusedNode, false);
}

/*
 * this function checks if fusion of the given group into one node will create a loop in the graph
 * considering both data and control tensors
 * if there is no loop then the graph is updated replacing the nodeSet with the fused node
 * */
bool GraphEditor::isLoopDueFusionInUpdateGraph(HabanaGraph&   g,
                                               const NodeSet& nodesSet,
                                               NodePtr&       testNode,
                                               bool           updateGraph)
{
    // mapping all group input and output tensors. those tensors will be the test node inputs and outputs.
    // all intermediates will not be referred.
    TensorSet outTensors;
    TensorSet inTensors;
    for (const NodePtr& node : nodesSet)
    {
        // nodes output will be added to the outputs list only if it has consumers fron outside the group
        runOnTensorsForType<Node::USAGE_OUTPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& t) {
            NodeList consumers = g.getTensorConsumers(t);
            if (consumers.size() == 0)
            {
                outTensors.insert(t);
                return;
            }
            for (const NodePtr& consumer : consumers)
            {
                if (nodesSet.find(consumer) == nodesSet.end())
                {
                    outTensors.insert(t);
                    break;
                }
            }
        });

        // add node input to the inputs list only if its producer is from outside the group
        runOnTensorsForType<Node::USAGE_INPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& t) {
            NodePtr producer = g.getTensorProducer(t);
            if (producer == nullptr || nodesSet.find(producer) == nodesSet.end())
            {
                inTensors.insert(t);
            }
        });
    }

    // storing the executing schedule to avoid re-sorting due to temporary nodes addition and removal
    g.storeExecutionSchedule();
    auto* bp = g.getBPGraph();
    if (bp != nullptr)
    {
        bp->freezeGraph();
    }

    // removing the group of nodes and adding the test node
    // Maintain tracking of origin nodes for debug purposes
    NodesIDs superSetOfOrigins;
    for (const NodePtr& node : nodesSet)
    {
        superSetOfOrigins.insert(node->getOriginNodes().begin(), node->getOriginNodes().end());
        GraphEditor::removeNode(g, node);
    }

    TensorVector inVec(inTensors.begin(), inTensors.end());
    TensorVector outVec(outTensors.begin(), outTensors.end());
    testNode = NodeFactory::createNode(inVec, outVec, nullptr, NodeFactory::DebugNodeTypeName, "test_fuse_node");

    // Maintain tracking of origin nodes for debug purposes
    testNode->setOriginNodes(superSetOfOrigins);

    GraphEditor::addNode(g, testNode);

    // checking for graph cycles
    bool isAcyclicGraph = g.isAcyclicGraph();

    if (updateGraph && isAcyclicGraph)
    {
        // Return no loop and do not restore old graph
        g.clearStoredExecutionSchedule();
        return false;
    }

    // removing the test node and re-adding the nodes back to the graph
    GraphEditor::removeNode(g, testNode);
    for (const NodePtr& node : nodesSet)
    {
        GraphEditor::addNode(g, node, false);
    }

    // restoring the previous execution schedule
    g.restoreExecutionSchedule();
    if (bp != nullptr)
    {
        bp->unfreezeGraph();
    }

    return !isAcyclicGraph;
}

template<typename NodeContainer>
bool GraphEditor::isInGroupDependencies(HabanaGraph& g, const NodeContainer& nodes)
{
    if (!g.isControlDependencyConfigured()) return false;

    /* need to check if there is a control dependency between the nodes in the list */
    NodeSet nodesSet(nodes.begin(), nodes.end());

    NodeSet blockedSet;
    for (NodePtr node : nodesSet)
    {
        NodeSet nodeBlockedSet(g.getBlockedNodes(node));
        blockedSet.insert(nodeBlockedSet.begin(), nodeBlockedSet.end());
    }

    NodeVector     intersection;
    NodeComparator comp;

    std::set_intersection(nodesSet.begin(),
                          nodesSet.end(),
                          blockedSet.begin(),
                          blockedSet.end(),
                          std::back_inserter(intersection), comp);

    if (intersection.size() != 0)
    {
        LOG_WARN(GC, "Can't fuse nodes, there is dependency within the fused nodes elements");
        return true;
    }

    // checking if there is a risk for loop in case of fusion, if there is then checking for loop
    // isLoopDueFusionInNodeSet has high complexity.
    // therefore it is executed only if isPossibleLoopDueToExtEdges return true
    if (isPossibleLoopDueToExtEdges(g, nodesSet) && isLoopDueFusionInNodeSet(g, nodesSet))
    {
        LOG_WARN(GC, "Can't fuse nodes, fusion will create loop in graph");
        return true;
    }

    return false;
}

template<typename NodeContainer>
void GraphEditor::getFirstNodesPerEngineForBundle(HabanaGraph&         g,
                                                  const NodeContainer& nodes,
                                                  NodeSet&             firstNodesPerEngine)
{
    unsigned mmeCount = 0, tpcCount = 0, dmaCount = 0;
    const unsigned NUMBER_OF_FIRST_MME_NODES = 1;
    const unsigned NUMBER_OF_FIRST_TPC_NODES = 1;
    const unsigned NUMBER_OF_FIRST_DMA_NODES = 6;

    for (NodePtr node : nodes)
    {
        if (node->isLogicalOperation())
            continue;

        if (g.runsOnMME(node))
        {
            if (mmeCount < NUMBER_OF_FIRST_MME_NODES)
            {
                mmeCount++;
                firstNodesPerEngine.insert(node);
            }
        }
        else if (g.runsOnTPC(node))
        {
            if (tpcCount < NUMBER_OF_FIRST_TPC_NODES)
            {
                tpcCount++;
                firstNodesPerEngine.insert(node);
            }
        }
        else if (node->isDma() || node->getNodeType() == Node::TYPE_MEMCOPY)
        {
            if (dmaCount < NUMBER_OF_FIRST_DMA_NODES)
            {
                dmaCount++;
                firstNodesPerEngine.insert(node);
            }
        }

        if ((dmaCount + tpcCount + mmeCount) ==
            (NUMBER_OF_FIRST_MME_NODES + NUMBER_OF_FIRST_TPC_NODES + NUMBER_OF_FIRST_DMA_NODES))
        {
            break;
        }
    }
}

template<typename NodeContainer1, typename NodeContainer2>
void GraphEditor::updateWaitCycles(const NodeContainer1& oldNodes, const NodeContainer2& newNodes)
{
    unsigned maxWaitCycles = 0;
    for (NodePtr n : oldNodes)
    {
        maxWaitCycles = std::max(maxWaitCycles, n->getNodeAnnotation().waitCycles);
    }
    if (maxWaitCycles)
    {
        for (NodePtr n : newNodes)
        {
            n->getNodeAnnotation().waitCycles = maxWaitCycles;
        }
    }
}

void GraphEditor::replaceTensor(HabanaGraph& g, const NodePtr& node, TensorPtr oldTensor, TensorPtr newTensor)
{
    GraphEditor::removeNode(g, node);
    node->replaceTensor(oldTensor, newTensor);
    bool status = GraphEditor::addNode(g, node, false);
    HB_ASSERT(status, "{}: failed to add node {}", __FUNCTION__, node->getNodeName());
}

void GraphEditor::replaceTensor(HabanaGraph& g, TensorPtr oldTensor, TensorPtr newTensor)
{
    const auto& producer = g.getTensorProducer(oldTensor);
    if (producer != nullptr)
    {
        GraphEditor::replaceTensor(g, producer, oldTensor, newTensor);
    }

    const auto& consumers = g.getTensorConsumers(oldTensor);
    for (const auto& consumer : consumers)
    {
        if (consumer != nullptr)
        {
            GraphEditor::replaceTensor(g, consumer, oldTensor, newTensor);
        }
    }
}

void GraphEditor::replaceInput(HabanaGraph& g, const NodePtr& node, unsigned int index, TensorPtr t, Node::eTensorType tensorType)
{
    GraphEditor::removeNode(g, node);
    node->replaceInput(index, t, tensorType);
    bool status = GraphEditor::addNode(g, node, false);
    HB_ASSERT(status, "{}: failed to add node {}", __FUNCTION__, node->getNodeName());
}

void GraphEditor::replaceOutput(HabanaGraph& g, const NodePtr& node, unsigned int index, TensorPtr t)
{
    GraphEditor::removeNode(g, node);
    node->replaceOutput(index, t);
    bool status = GraphEditor::addNode(g, node, false);
    HB_ASSERT(status, "{}: failed to add node {}", __FUNCTION__, node->getNodeName());
}

void GraphEditor::editNode(HabanaGraph& g, const NodePtr& node, std::function<void(const NodePtr&)> editFunc)
{
    GraphEditor::removeNode(g, node);
    editFunc(node);
    bool status = GraphEditor::addNode(g, node, false);
    HB_ASSERT(status, "{}: failed to add node {}", __FUNCTION__, node->getNodeName());
}

void GraphEditor::editNode(HabanaGraph& g, const NodePtr& node, std::function<void()> editFunc)
{
    GraphEditor::removeNode(g, node);
    editFunc();
    bool status = GraphEditor::addNode(g, node, false);
    HB_ASSERT(status, "{}: failed to add node {}", __FUNCTION__, node->getNodeName());
}

template<typename NodeContainer1, typename NodeContainer2>
void GraphEditor::removeNodes(HabanaGraph& g, const NodeContainer1& nodes, const NodeContainer2& newProducers)
{
    HB_ASSERT(nodes.size() == newProducers.size(), "number of new producers doesn't match nodes size");

    auto oldNode = nodes.begin();
    auto newProd = newProducers.begin();

    for(; oldNode != nodes.end(); ++oldNode, ++newProd)
    {
        GraphEditor::removeNode(g, *oldNode, *newProd);
    }
}

template<typename NodeContainer>
void GraphEditor::removeNodes(HabanaGraph& g, const NodeContainer& nodes)
{
    if (nodes.size() == 0)
    {
        return;
    }

    /* loop through the nodes, remove them from the graph, and handle their control edges accordingly */
    for (const NodePtr& oldNode : nodes)
    {
        GraphEditor::removeNode(g, oldNode);
    }
}

bool GraphEditor::canRemoveNodeControl(const HabanaGraph& g, const NodePtr& node)
{
    const auto& memoryCoherence = g.getGraphAnnotation().memoryCoherence;
    // check if there are any paths between producers to blocking nodes (might cause a cycle when moving ctrl edge)
    const NodeSet& blockingNodes =
        memoryCoherence ? memoryCoherence->calculateBlockingNodes(g, node) : g.getBlockingNodes(node);
    if (!blockingNodes.empty())
    {
        const NodeSet& producers = g.getNodeProducers(node);
        for (const NodePtr& dst : blockingNodes)
        {
            for (const NodePtr& src : producers)
            {
                if (src == dst) continue;
                if (g.getNumberOfPaths(src, dst, Node::TENSOR_TYPE_ALL) > 0) return false;
            }
        }
    }
    // check if there are any paths between blocked nodes to consumers and inputs' consumers (might cause a cycle when
    // moving ctrl edge)
    const NodeSet& blockedNodes =
        memoryCoherence ? memoryCoherence->calculateBlockedNodes(g, node) : g.getBlockedNodes(node);
    if (!blockedNodes.empty())
    {
        // All consumers of node and other inputs of the same node.
        NodeSet consumers = g.getNodeConsumers(node);
        for (const auto& tensor : node->getInputs())
        {
            auto inputConsumerList = g.getTensorConsumers(tensor);
            for (const auto& inputConsumer : inputConsumerList)
            {
                if (inputConsumer != node)
                {
                    consumers.insert(inputConsumer);
                }
            }
        }
        for (const NodePtr& dst : consumers)
        {
            for (const NodePtr& src : blockedNodes)
            {
                if (src == dst) continue;
                if (g.getNumberOfPaths(src, dst, Node::TENSOR_TYPE_ALL) > 0) return false;
            }
        }
    }
    return true;
}

void GraphEditor::removeOneToOneNode(HabanaGraph& g, const NodePtr& node)
{
    if (!canRemoveNodeControl(g, node))
    {
        LOG_DEBUG(GC, "Can't remove node {} due to control edge risk", node->getNodeName());
        return;
    }

    TensorPtr output = node->getOutput(0);
    TensorPtr input  = node->getInput(0);

    /* if the output of the removed node is output of the graph, we need to keep this tensor,
     * and consider removing the input of the node */
    if (g.isOutputTensor(output) || output->isUserManagedDram())
    {
        /* if the input of the removed node is output tensor or input tensor of the graph - we can't remove the node */
        if (g.isOutputTensor(input) || g.isInputTensor(input) || input->isUserManagedDram())
        {
            LOG_DEBUG(GC, "Can't remove node {} due to persistent memory", node->getNodeName());
            return;
        }

        pNode producer = g.getTensorProducer(input);

        /* First remove the node to disable the relationship with the graph output */
        LOG_DEBUG(GC, "Removing node '{}'", node->getNodeName());
        GraphEditor::removeNode(g, node);
        /* Switch the producer node's output */
        GraphEditor::replaceTensor(g, producer, input, output);

        // Maintain origin nodes tracking for debugging purposes
        producer->addOriginNodes(node->getOriginNodes());

        // Replace on all consumers which using this input
        const NodeList& consumers = g.getTensorConsumers(input);
        for (auto c : consumers)
        {
            // Maintain origin nodes tracking for debugging purposes
            c->addOriginNodes(node->getOriginNodes());

            GraphEditor::replaceTensor(g, c, input, output);
        }

        return;
    }

    /* In this case, the output is removed and the input of the removed node will
     * be the new input of the removed node output consumers */
    NodeList consumers = g.getTensorConsumers(output);

    for (pNode consumer : consumers)
    {
        // Maintain origin nodes tracking for debugging purposes
        consumer->addOriginNodes(node->getOriginNodes());

        /* the input of the removed node, will be the input of the next node */
        GraphEditor::replaceTensor(g, consumer, output, input);
    }

    LOG_DEBUG(GC, "Removing node '{}'", node->getNodeName());
    GraphEditor::removeNode(g, node);
}

template<typename NodeContainer1, typename NodeContainer2>
bool GraphEditor::isOneForOneReplacement(const NodeContainer1& oldNodes, const NodeContainer2& newNodes)
{
    return (oldNodes.size() == 1) && (newNodes.size() == 1);
}

template<typename NodeContainer1, typename NodeContainer2>
bool GraphEditor::isTrivialReplacement(const NodeContainer1& oldNodes, const NodeContainer2& newNodes)
{
    // trivial replacement - one for one, with same tensors (like in selecting memcopy engine).
    if (!isOneForOneReplacement(oldNodes, newNodes)) return false;
    const NodePtr& oldNode = oldNodes.front();
    const NodePtr& newNode = newNodes.front();

    return Graph::haveSameConnectivity(oldNode, newNode);
}

void GraphEditor::replaceTrivialNodes(HabanaGraph& g, const NodePtr& oldNode, const NodePtr& newNode)
{
    // add control dependencies
    for (uint32_t i = 0; i < oldNode->getNumInputs(Node::TENSOR_TYPE_CONTROL); i++)
    {
        newNode->addInput(i, oldNode->getControlInput(i), Node::TENSOR_TYPE_CONTROL);
    }
    for (const TensorPtr& t : oldNode->getControlOutputs())
    {
        newNode->addOutput(t, Node::TENSOR_TYPE_CONTROL);
    }
    // propagate wait cycles
    updateWaitCycles({oldNode}, {newNode});

    // Maintain tracking of origin nodes for debug purposes
    newNode->setOriginNodes(oldNode->getOriginNodes());

    g.replaceSemanticNodes(oldNode, newNode);
}

template<typename NodeContainer>
std::unordered_set<synNodeId> GraphEditor::getUniqueFlashAttentionParentIds(const HabanaGraph&   g,
                                                                            const NodeContainer& nodes)
{
    const auto&                   flashAttentionDb = g.getGraphAnnotation().flashAttentionDb;
    std::unordered_set<synNodeId> uniqueFlashAttentionIds;
    std::for_each(nodes.begin(),
                  nodes.end(),
                  [&flashAttentionDb, &uniqueFlashAttentionIds](const NodePtr& n) {
                      auto parentId = n->getParentId();
                      if (flashAttentionDb.isRegistered(parentId))
                      {
                          uniqueFlashAttentionIds.insert(parentId);
                      }
                  });

    return uniqueFlashAttentionIds;
}

template<typename NodeContainer1, typename NodeContainer2>
ReplaceNodeReturnStatus
GraphEditor::replaceNodes(HabanaGraph& g, const NodeContainer1& oldNodes, const NodeContainer2& newNodes, bool isSlicer)
{
    HB_ASSERT(!oldNodes.empty(), "oldNodes size is 0");
    HB_ASSERT(!newNodes.empty(), "newNodes size is 0");

    if (GraphEditor::isInGroupDependencies(g, oldNodes))
    {
        LOG_WARN(GC,
                 "{}: Fusion is not allowed - {}",
                 HLLOG_FUNC,
                 replaceNodeReturnStatusToString(REPLACE_FAILED_INGROUP_DEPS));
        return REPLACE_FAILED_INGROUP_DEPS;  // The graph remains unchanged, replaceNodes failed
    }

    /* fuses a list of nodes, to one node  - makes sure dependency is kept, and that the new nodes are valid */
    bool invalidNodeExists =
        std::any_of(newNodes.begin(), newNodes.end(), [&](const NodePtr& n) { return !g.validateNode(n); });
    if (invalidNodeExists)
    {
        LOG_WARN(GC,
                 "{}: Fusion is not allowed - {}",
                 HLLOG_FUNC,
                 replaceNodeReturnStatusToString(REPLACE_FAILED_INVALID_NEW_NODES));
        return REPLACE_FAILED_INVALID_NEW_NODES;  // The graph remains unchanged, replaceNodes failed
    }

    const auto& uniqueFlashAttentionIds = getUniqueFlashAttentionParentIds(g, oldNodes);
    HB_ASSERT(uniqueFlashAttentionIds.size() <= 1,
              "Expecting that there are not two old nodes that belong to different flash attention subgraphs");
    // If none of the old nodes belong to flash attention subgraph, set the minimal parent ID of the old nodes to all the
    // new nodes, to make sure they are scheduled as early as the first original node. Otherwise, set the flash
    // attention id.
    synNodeId repParentId =
        uniqueFlashAttentionIds.empty() ? getMinParentId(oldNodes).value() : *uniqueFlashAttentionIds.begin();
    for (const NodePtr& n : newNodes)
    {
        n->setParentId(repParentId);
    }

    // When atomic node is replaced one for one - the atomic attribute passed on to the replacing node
    if (isOneForOneReplacement(oldNodes, newNodes))
    {
        g.getGraphAnnotation().replaceAtomicNode(oldNodes.front().get(), newNodes.front());
    }

    if (isTrivialReplacement(oldNodes, newNodes))
    {
        replaceTrivialNodes(g, oldNodes.front(), newNodes.front());
        return REPLACE_NODE_SUCCESS;
    }

    NodesIDs superSetOfOriginNodes;
    /* loop through the nodes that is about to be fused, remove their ctrl edges and remove the node from the graph. */
    for (const NodePtr& oldNode : oldNodes)
    {
        const auto& oldOriginNodes = oldNode->getOriginNodes();
        superSetOfOriginNodes.insert(oldOriginNodes.begin(), oldOriginNodes.end());
        GraphEditor::removeNode(g, oldNode);
    }

    // make sure fused nodes delay is propagated
    updateWaitCycles(oldNodes, newNodes);

    for (const NodePtr& newNode : newNodes)
    {
        newNode->setOriginNodes(superSetOfOriginNodes);
        bool status = GraphEditor::addNode(g, newNode, false);
        HB_ASSERT(status, "{}: failed to add node {}", __FUNCTION__, newNode->getNodeName());
    }

    return REPLACE_NODE_SUCCESS;
}

// If the node is DMA memset which output tensor is input to a ReductionNode, return the reduction node (assume
// only one in this case), Else return nullptr.
NodePtr GraphEditor::checkNodeMemsetForReduction(const HabanaGraph& graph, const NodePtr& node)
{
    if (!node->isMemset()) return nullptr;

    HB_ASSERT(node->getOutputs().size() == 1, "Expecting memset node to have exactly 1 output.");

    TensorPtr memsetOut = node->getOutputs().front();
    for (NodePtr consumer : graph.getTensorConsumers(memsetOut))
    {
        if (consumer->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            return consumer;
        }
    }
    return nullptr;
}

template<typename NodeContainer>
void GraphEditor::removeMemsetReductionFromFusion(HabanaGraph&         g,
                                                  const NodeContainer& newNodes,
                                                  NodeSet&             fusedBlocked,
                                                  NodeSet&             fusedBlocking)
{
    for (const NodePtr& node : newNodes)
    {
        NodePtr reductionNode = checkNodeMemsetForReduction(g, node);
        if (reductionNode)
        {
            fusedBlocked.erase(node);
            fusedBlocked.erase(reductionNode);
            fusedBlocking.erase(node);
            fusedBlocking.erase(reductionNode);
        }
    }

}

bool GraphEditor::addNode(HabanaGraph& g, const NodePtr& node, bool setParentId)
{
    bool nodeAdded = g.addNode(node);
    if (!nodeAdded) return nodeAdded;

    if (g.getGraphAnnotation().memoryCoherence != nullptr)
    {
        addNodeControlDependencies(g, node);
    }

    if (setParentId && (!g.getGraphAnnotation().flashAttentionDb.isRegistered(node->getParentId())))
    {
        // Inherit the parent ID from producers or consumers, to make sure this node's consumers with lower ID
        // will be scheduled in order as much as close to the pre graph. Without this parent ID, the new node has higher
        // node ID than all pre graph nodes, and will "lose" in scheduling to any of them if both are free. Thus, its
        // consumers are scheduled later, although their node ID is lower than the nodes scheduled before.
        std::optional<synNodeId> minId = getMinParentId(g.getNodeProducers(node));
        if (!minId.has_value())  // no producers
        {
            minId = getMinParentId(g.getNodeConsumers(node));
        }
        if (minId.has_value())
        {
            node->setParentId(minId.value());
        }
    }
    return nodeAdded;
}

// Add nodes which should be shceduled together, and get the same minimal parent ID
template<typename NodeContainer>
bool GraphEditor::addNodes(HabanaGraph& g, const NodeContainer& nodes, bool setParentId)
{
    bool allNodesAdded = true;
    for (const NodePtr& n : nodes)
    {
        bool nodeAdded = GraphEditor::addNode(g, n, setParentId);
        if (!nodeAdded)
        {
            LOG_ERR(GC, "{}: adding node {} failed", HLLOG_FUNC, n->getNodeName());
        }
        allNodesAdded &= nodeAdded;
    }

    if (allNodesAdded && setParentId)
    {
        std::optional<synNodeId> minId = getMinParentId(nodes);
        if (minId.has_value())
        {
            for (const NodePtr& n : nodes)
            {
                n->setParentId(minId.value());
            }
        }
    }
    return allNodesAdded;
}

void GraphEditor::removeNode(HabanaGraph& g, const NodePtr& node, const NodePtr& newProducer)
{
    bool    useMemoryCoherence = g.getGraphAnnotation().memoryCoherence != nullptr;
    NodeSet nodeConsumers;
    if (useMemoryCoherence)
    {
        if (newProducer != nullptr)
        {
            nodeConsumers = g.getNodeConsumers(node);
        }
        // if the node will be added to the graph, its control dependencies will be recalculated
        g.removeNodeControlDependencies(node);
    }

    g.removeNode(node, newProducer);

    if (useMemoryCoherence && newProducer != nullptr)
    {
        // all the original consumers of "node" now consume the outputs of "newProducer" instead.
        // so we need to adjust their control dependencies
        for (const NodePtr& consumer : nodeConsumers)
        {
            recalculateNodeControlDependencies(g, consumer);
        }
    }
}

void GraphEditor::addNodeControlDependencies(HabanaGraph& g, const NodePtr& node)
{
    const auto& memoryCoherence = g.getGraphAnnotation().memoryCoherence;
    HB_ASSERT_PTR(memoryCoherence);

    for (const NodePtr& blocking : memoryCoherence->calculateBlockingNodes(g, node))
    {
        g.addControlDependency(blocking, node);
    }
    for (const NodePtr& blocked : memoryCoherence->calculateBlockedNodes(g, node))
    {
        g.addControlDependency(node, blocked);
    }
}

void GraphEditor::recalculateNodeControlDependencies(HabanaGraph& g, const NodePtr& node)
{
    g.removeNodeControlDependencies(node);
    addNodeControlDependencies(g, node);
}

// instantiate tempalte functions for used containers
template void GraphEditor::removeNodes<NodeList>(HabanaGraph& g, const NodeList& nodes);
template void GraphEditor::removeNodes<NodeVector>(HabanaGraph& g, const NodeVector& nodes);
template void GraphEditor::removeNodes<NodeSet>(HabanaGraph& g, const NodeSet& nodes);

template void
GraphEditor::removeNodes<NodeList, NodeList>(HabanaGraph& g, const NodeList& nodes, const NodeList& newProducers);
template void GraphEditor::removeNodes<NodeVector, NodeVector>(HabanaGraph&      g,
                                                               const NodeVector& nodes,
                                                               const NodeVector& newProducers);
template void
GraphEditor::removeNodes<NodeList, NodeVector>(HabanaGraph& g, const NodeList& nodes, const NodeVector& newProducers);
template void
GraphEditor::removeNodes<NodeVector, NodeList>(HabanaGraph& g, const NodeVector& nodes, const NodeList& newProducers);

template bool GraphEditor::addNodes<NodeList>(HabanaGraph& g, const NodeList& nodes, bool setParentId = true);
template bool GraphEditor::addNodes<NodeVector>(HabanaGraph& g, const NodeVector& nodes, bool setParentId = true);

template ReplaceNodeReturnStatus GraphEditor::replaceNodes<NodeList, NodeList>(HabanaGraph&    g,
                                                                               const NodeList& oldNodes,
                                                                               const NodeList& newNodes,
                                                                               bool            isSlicer);
template ReplaceNodeReturnStatus GraphEditor::replaceNodes<NodeList, NodeVector>(HabanaGraph&      g,
                                                                                 const NodeList&   oldNodes,
                                                                                 const NodeVector& newNodes,
                                                                                 bool              isSlicer);
template ReplaceNodeReturnStatus GraphEditor::replaceNodes<NodeVector, NodeList>(HabanaGraph&      g,
                                                                                 const NodeVector& oldNodes,
                                                                                 const NodeList&   newNodes,
                                                                                 bool              isSlicer);
template ReplaceNodeReturnStatus GraphEditor::replaceNodes<NodeVector, NodeVector>(HabanaGraph&      g,
                                                                                   const NodeVector& oldNodes,
                                                                                   const NodeVector& newNodes,
                                                                                   bool              isSlicer);

template bool GraphEditor::isTrivialReplacement<NodeList, NodeList>(const NodeList& oldNodes, const NodeList& newNodes);
template bool GraphEditor::isTrivialReplacement<NodeList, NodeVector>(const NodeList&   oldNodes,
                                                                      const NodeVector& newNodes);
template bool GraphEditor::isTrivialReplacement<NodeVector, NodeList>(const NodeVector& oldNodes,
                                                                      const NodeList&   newNodes);
template bool GraphEditor::isTrivialReplacement<NodeVector, NodeVector>(const NodeVector& oldNodes,
                                                                        const NodeVector& newNodes);

template void GraphEditor::updateWaitCycles<NodeList, NodeList>(const NodeList& oldNodes, const NodeList& newNodes);
template void GraphEditor::updateWaitCycles<NodeList, NodeVector>(const NodeList& oldNodes, const NodeVector& newNodes);
template void GraphEditor::updateWaitCycles<NodeVector, NodeList>(const NodeVector& oldNodes, const NodeList& newNodes);
template void GraphEditor::updateWaitCycles<NodeVector, NodeVector>(const NodeVector& oldNodes,
                                                                    const NodeVector& newNodes);

template bool GraphEditor::isInGroupDependencies<NodeList>(HabanaGraph& g, const NodeList& nodes);
template bool GraphEditor::isInGroupDependencies<NodeVector>(HabanaGraph& g, const NodeVector& nodes);

template void GraphEditor::removeMemsetReductionFromFusion<NodeList>(HabanaGraph&    g,
                                                                     const NodeList& newNodes,
                                                                     NodeSet&        fusedBlocked,
                                                                     NodeSet&        fusedBlocking);
template void GraphEditor::removeMemsetReductionFromFusion<NodeVector>(HabanaGraph&      g,
                                                                       const NodeVector& newNodes,
                                                                       NodeSet&          fusedBlocked,
                                                                       NodeSet&          fusedBlocking);

template void GraphEditor::getFirstNodesPerEngineForBundle<NodeList>(HabanaGraph&    g,
                                                                     const NodeList& nodes,
                                                                     NodeSet&        firstNodesPerEngine);
template void GraphEditor::getFirstNodesPerEngineForBundle<NodeVector>(HabanaGraph&      g,
                                                                       const NodeVector& nodes,
                                                                       NodeSet&          firstNodesPerEngine);

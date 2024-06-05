#include "habana_graph.h"
#include "habana_pass.h"
#include "graph_editor.h"
#include "node_factory.h"
#include "cast_nodes_handler.h"
#include "graph_traits.h"
#include "replace_ops_with_logical_ops.h"
#include "tensor_annotation.h"
#include "data_type_utils.h"
#include "gc_ops_db.h"

namespace
{
// CastWrapper wrap NodeList with the cast Node and  opposite reshape Nodes if they exists
class CastWrapper
{
public:
    CastWrapper() : m_nodes(NodeList({})), m_castNode(NodePtr(nullptr)), m_removed(false), m_canRemove(false) {}
    CastWrapper(const HabanaGraph& g, const NodePtr& castNode) :
                m_nodes({castNode}),
                m_removed(false)
    {
        m_castNode = m_nodes.front();
        HB_ASSERT(castNode->isCast(), "{} is not cast node", castNode->getNodeName());
        if (isWrappedByOppositeReshapes(g, castNode))
        {
            m_nodes.push_front(g.getTensorProducer(castNode->getInput(0)));
            m_nodes.push_back(g.getTensorConsumers(castNode->getOutput(0)).front());
        }
        m_canRemove = !existsHandledLogicalNode(g, m_nodes.front()->getInput(0), m_nodes.back()->getOutput(0)) &&
                      !existsControlDep(g);
    }

    TensorPtr getInput() const
    {
        return m_nodes.front()->getInput(0);
    }

    TensorPtr getOutput() const
    {
        return m_nodes.back()->getOutput(0);
    }

    NodeList getNodesList() const
    {
        return m_nodes;
    }

    // remove nodes of castWrppaer from the graph and move the control dependency to another node
    void removeNodesAndMoveCtlDep(HabanaGraph& g, const NodePtr& newCtlDepNode)
    {
        if (m_removed) return; // nodes already removed
        GraphEditor::removeNodes(g, m_nodes);
        m_removed = true;
    }

    NodePtr getCastNode() const
    {
        return m_castNode;
    }

    unsigned size() const
    {
        return m_nodes.size();
    }

    bool canRemove() const
    {
        return m_canRemove;
    }

    bool existsUserManagedTensor(const HabanaGraph& g, bool isBackwardPass = true) const
    {
        for (const NodePtr& node : m_nodes)
        {
            if (isBackwardPass && g.isUserManagedDram(node->getOutput(0))) return true;
            if (!isBackwardPass && g.isUserManagedDram(node->getInput(0))) return true;
        }
        return false;
    }

    bool existsControlDep(const HabanaGraph& g) const
    {
        for (const NodePtr& node : m_nodes)
        {
            if (!g.getBlockedNodes(node).empty()) return true;
            if (!g.getBlockingNodes(node).empty()) return true;
        }
        return false;
    }

    bool operator==(const CastWrapper& c) const
    {
        return (m_nodes == c.m_nodes);
    }

    bool operator!=(const CastWrapper& c) const
    {
        return !(*this == c);
    }

    void update(const HabanaGraph& g, const NodePtr& castNode)
    {
        CastWrapper c(g, castNode);
        m_nodes = c.m_nodes;
        m_castNode = c.m_castNode;
        m_removed = c.m_removed;
        m_canRemove = c.m_canRemove;
    }

    // remove cast node (wrapped) from graph and replace the output tensor (input for consumers) with new input tensor
    void removeFromGraph(HabanaGraph& g, TensorPtr newInputTensor, bool oppositeCastNode)
    {
        NodePtr  newCtlDepNode = g.getTensorProducer(getInput());
        NodeList consumers = g.getTensorConsumers(getOutput());
        // should not remove persistent tensor
        bool insertMemcpy = (g.isUserManagedDram(getOutput()) && oppositeCastNode);
        // if output tensor location not equal to new input tensor location should insert memcpy (and there are consumers)
        insertMemcpy |= (!consumers.empty() && (newInputTensor->inSram() != getOutput()->inSram()));
        // if input tensor is input to graph shold add memcpy to avoid control dependency issues
        insertMemcpy |= !newCtlDepNode && g.isInputTensor(getInput());
        if (insertMemcpy)
        {
            if (oppositeCastNode)
            {
                // TODO [SW-37565]: Preserve undefined instead of forcing to DRAM
                auto loc      = getInput()->inSram() ? TENSOR_IN_SRAM : TENSOR_IN_DRAM;
                newCtlDepNode = GraphEditor::insertMemcpyForOutput(g, getNodesList().back(), getOutput(), loc);
                consumers = NodeList({newCtlDepNode}); // update consumers to be inserted memcpy node
            }
            else
            {
                // TODO [SW-37565]: Preserve undefined instead of forcing to DRAM
                auto loc      = getOutput()->inSram() ? TENSOR_IN_SRAM : TENSOR_IN_DRAM;
                newCtlDepNode = GraphEditor::insertMemcpyForInput(g, getNodesList().front(), getInput(), loc);
                GraphEditor::replaceTensor(g, newCtlDepNode, newCtlDepNode->getInput(0), newInputTensor);
                newInputTensor = getInput();
            }
        }
        for (const NodePtr& consumer : NodeSet(consumers.begin(), consumers.end()))
        {
            GraphEditor::replaceTensor(g, consumer, getOutput(), newInputTensor);
        }
        HB_ASSERT_PTR(newCtlDepNode);

        if (insertMemcpy && newCtlDepNode->getInput(0)->inSram() && !newCtlDepNode->getOutput(0)->inSram())
        {
            // In case the casts nodes are replaced with memcopy from SRAM to HBM and the producer of the input
            // is inside a bundle - add the memcopy node to the same bundle, to ensure the
            // tensor will be evicted from SRAM at the end of the bundle.
            // Otherwise this memcopy might be scheduled later and reduce the free memory in SRAM for the next bundles.
            // A clean solution will be done in SW-68630.
            const auto& memcopyInputProducer = g.getTensorProducer(newCtlDepNode->getInput(0));
            if (memcopyInputProducer)
            {
                const auto& producerBundleInfo = memcopyInputProducer->getNodeAnnotation().bundleInfo;
                if (producerBundleInfo.is_set() && !newCtlDepNode->getNodeAnnotation().bundleInfo.is_set())
                {
                    newCtlDepNode->getNodeAnnotation().bundleInfo = producerBundleInfo;
                }
            }
        }

        if (insertMemcpy)  // TODO - remove in [SW-68630]
        {
            replace_ops_with_logical_ops::tryReplaceMemcopyWithIdentity(g, newCtlDepNode);
        }

        // remove node and move control dependency to cast producer or inserted memcpy node
        removeNodesAndMoveCtlDep(g, newCtlDepNode);
    }

private:
    // cast node and opposite reshape if they exists
    NodeList m_nodes;
    // the cast node
    NodePtr m_castNode;
    // nodes already removed from graph
    bool m_removed;
    // can remove nodes from graph (removing nodes is forbidden if logical consumers/producers that were handled exist)
    bool m_canRemove;

    static bool isValidReshapeNodeForPass(const NodePtr& node)
    {
        // producer exists, input tensor exists and producer is reshape node
        if (!node || !node->getInput(0) || !std::dynamic_pointer_cast<const ReshapeNode>(node)) return false;
        // currentlly this pass does not optimize dynamic shapes
        if (node->isDynamicShape()) return false;
        HB_ASSERT(node->getNumInputs() == 1 || (node->getNumInputs() == 2 && node->getInput(1)->isShapeTensor()),
                  "reshape node has impropper inputs");
        HB_ASSERT(node->getNumOutputs() == 1, "reshape node has {} outputs", node->getNumOutputs());
        return true;
    }

    static bool isWrappedByOppositeReshapes(const HabanaGraph& g, const NodePtr& node)
    {
        const NodePtr& producer = g.getTensorProducer(node->getInput(0));
        if (!isValidReshapeNodeForPass(producer)) return false;

        NodeSet consumers = g.getNodeConsumers(node);
        if (consumers.size() != 1) return false; // must be 1 consumer to cast node
        const NodePtr& consumer = *consumers.begin();
        if (!isValidReshapeNodeForPass(consumer)) return false;

        // reshape nodes are opposite
        if (producer->getInput(0)->getDim() != consumer->getOutput(0)->getDim()) return false; // dimension mismatch
        if (!producer->getInput(0)->compareGeometry(*consumer->getOutput(0))) return false; // shape mismatch
        return true;
    }

    bool existsHandledLogicalNode(const HabanaGraph& g, const TensorPtr& input, const TensorPtr& output)
    {
        //check if input producer is logical node that was already handled
        const NodePtr& producer = g.getTensorProducer(input);
        std::shared_ptr<LogicalOpNode> logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(producer);
        if (logicalNode && logicalNode->getRunLogicalOperationDone()) return true;

        //check if input consumers are logical node that already handled
        NodeList consumers = g.getTensorConsumers(input);
        for (const NodePtr& consumer : consumers)
        {
            if (consumer == m_nodes.front()) continue; // ignore the reshape node if exists
            logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(consumer);
            if (logicalNode && logicalNode->getRunLogicalOperationDone()) return true;
        }
        //check if output consumers are logical node that already handled
        consumers = g.getTensorConsumers(output);
        for (const NodePtr& consumer : consumers)
        {
            logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(consumer);
            if (logicalNode && logicalNode->getRunLogicalOperationDone()) return true;
        }
        return false;
    }
};

} // anonymous namespace

static bool canResetCastGuid(const CastWrapper& castWrapper)
{
    auto tpcCastNode = std::dynamic_pointer_cast<TPCNode>(castWrapper.getCastNode());
    HB_ASSERT(tpcCastNode, "Failed to convert cast node {} to TPC node", castWrapper.getCastNode()->getNodeName());
    if (tpcCastNode->isInstantiated())
    {
        LOG_TRACE(GC,
                  "{}: Can't optimize TPC cast node {} - was already instanciated",
                  HLLOG_FUNC,
                  castWrapper.getCastNode()->getNodeName());
        return false;
    }
    return true;
}

static void updateTensorsType(HabanaGraph& g, const TensorVector& tensors, const synDataType dataType)
{
    for (const TensorPtr& tensor : tensors)
    {
        // create new tensor in new type with data converted to new type
        TensorPtr newTensor = createCastTensor(tensor, dataType, tensor->getName());
        const NodePtr& producer = g.getTensorProducer(tensor);
        if (producer)
        {
            GraphEditor::replaceTensor(g, producer, tensor, newTensor);
        }
        for (const NodePtr& consumer : g.getTensorConsumers(tensor))
        {
            if (consumer)
            {
                GraphEditor::replaceTensor(g, consumer, tensor, newTensor);
            }
        }
    }
}

// remove cast nodes from graph and update tensors type between casts
static void removeCastNodes(HabanaGraph& g,
                            CastWrapper& root,
                            std::vector<CastWrapper>& leafs,
                            const TensorVector& tensorsBetweenCasts,
                            bool isBackwardPass)
{
    HB_ASSERT(!leafs.empty(), "should be cast in both side of the tree");
    synDataType dataType = isBackwardPass ?
                           root.getInput()->getElementType() :
                           root.getOutput()->getElementType();
    // change the type of all tensors between cast nodes
    updateTensorsType(g, tensorsBetweenCasts, dataType);
    for (CastWrapper& castNode : leafs)
    {
        castNode.removeFromGraph(g, castNode.getInput(), isBackwardPass);
    }
    root.removeFromGraph(g, root.getInput(), !isBackwardPass);
}

// replace cast nodes with new cast node and update tensors type between casts
static void replaceCastNodes(HabanaGraph& g,
                             CastWrapper& root,
                             std::vector<CastWrapper>& leafs,
                             const TensorVector& tensorsBetweenCasts,
                             const std::string& newCastGuid,
                             bool isBackwardPass)
{
    HB_ASSERT(!leafs.empty(), "should be cast in both side of the tree");
    /* part A.1.2.1, A.1.2.2, B.1.2.1 and B.1.2.2 in the algorithm:
       should remove leafs if there is more than 1 leaf, if not set the tensors between cast to be in
       lower type to avoid larger copies that can be added between logical nodes */
    bool removeLeafs = ((leafs.size() > 1) ||
                        (leafs[0].getOutput()->getElementSizeInBytes() < root.getInput()->getElementSizeInBytes()));
    synDataType dataType    = (isBackwardPass) ? root.getInput()->getElementType() : root.getOutput()->getElementType();
    if (removeLeafs)
    {
        if (isBackwardPass)
        {
            dataType = leafs[0].getOutput()->getElementType();
        }
        else
        {
            dataType = leafs[0].getInput()->getElementType();
        }
    }
    // change the type of all tensors between cast nodes
    updateTensorsType(g, tensorsBetweenCasts, dataType);
    if (removeLeafs)
    {
        for (CastWrapper& castNode : leafs)
        {
            castNode.removeFromGraph(g, castNode.getInput(), isBackwardPass);
        }
        root.getCastNode()->setGUID(newCastGuid);
    }
    else
    {
        leafs[0].getCastNode()->setGUID(newCastGuid);
        root.removeFromGraph(g, root.getInput(), !isBackwardPass);
    }
}

// get empty castWrapper and update him if the node is part of cast wrapper
static bool isPartOfCastWrapper(const HabanaGraph& g, const NodePtr& node, CastWrapper& cast, bool isBackwardPass)
{
    if (node->isCast())
    {
        cast.update(g, node);
        return true;
    }
    NodeSet nodes = isBackwardPass ? g.getNodeConsumers(node) : g.getNodeProducers(node);
    // there is 1 consumer/producer that is cast node that wrapped by opposite reshapes
    if (nodes.size() == 1 && (*nodes.begin())->isCast() && CastWrapper(g, *nodes.begin()).size() == 3)
    {
        cast.update(g, *nodes.begin());
        return true;
    }
    return false;
}

static bool canRemoveOrMergeOppositeCast(const HabanaGraph& g,
                                         const CastWrapper& oppositeCastNode,
                                         const QuantizationData& outputCastNodeParams,
                                         const QuantizationData& oppositeCastNodeParams,
                                         bool isBackwardPass)
{
    // exists opposite casts with different output params
    if (outputCastNodeParams != oppositeCastNodeParams) return false;
    // in forward pass exists opposite casts with more than one consumer
    if (!isBackwardPass && g.getNumberOfTensorConsumers(oppositeCastNode.getOutput()) != 1) return false;
    if (!oppositeCastNode.canRemove()) return false;
    return true;
}

/* get cast node and his consumers (after the logical node sequence if exists)
   and check if it's possible remove or merge the nodes and return the number of casts that removed */
static unsigned removeOrMergeCasts(HabanaGraph& g,
                                   CastWrapper& castNode,
                                   const NodeSet& nodes,
                                   const TensorVector& tensorsBetweenCasts,
                                   bool isBackwardPass)
{
    // TODO: Remove once [SW-136615] is done
    gc::ops::ScopedSkipValidation skipOpValidation;
    unsigned removedNodes = 0;
    LOG_DEBUG(GC,
              "{}: checking cast node {}, isBackwardPass={}",
              HLLOG_FUNC,
              castNode.getCastNode()->getNodeName(),
              isBackwardPass);

    QuantizationData castNodeParams = isBackwardPass ?
                                      castNode.getInput()->getQuantizationParams() :
                                      castNode.getOutput()->getQuantizationParams();
    LOG_TRACE(GC,
              "{}: cast node params: dtype={}, numChannels={}, scale[0]={}, zp[0]={}",
              HLLOG_FUNC,
              castNodeParams.getDataTypeString(castNodeParams.m_qDataType),
              castNodeParams.m_numChannels,
              castNodeParams.scale(0),
              castNodeParams.zp(0));
    // to check if quantization data of all opposite casts are same
    QuantizationData outputCastNodeParams;
    std::vector<CastWrapper> oppositeCastNodes;
    for (const NodePtr& node : nodes)
    {
        CastWrapper oppositeCastNode; // will update in isPartOfCastWrapper function
        if (!isPartOfCastWrapper(g, node, oppositeCastNode, isBackwardPass)) continue; // is not part of cast wrapper
        QuantizationData oppositeCastNodeParams = isBackwardPass ?
                                                  oppositeCastNode.getOutput()->getQuantizationParams() :
                                                  oppositeCastNode.getInput()->getQuantizationParams();
        LOG_TRACE(GC,
                  "{}: checking vs. opposite cast node {}",
                  HLLOG_FUNC,
                  oppositeCastNode.getCastNode()->getNodeName());
        LOG_TRACE(GC,
                  "{}: opposite cast node params: dtype={}, numChannels={}, scale[0]={}, zp[0]={}",
                  HLLOG_FUNC,
                  oppositeCastNodeParams.getDataTypeString(oppositeCastNodeParams.m_qDataType),
                  oppositeCastNodeParams.m_numChannels,
                  oppositeCastNodeParams.scale(0),
                  oppositeCastNodeParams.zp(0));
        // removing casts of Float->Integer->Float can change the values
        if (isBackwardPass &&
            castNode.getInput()->isTypeFloat() &&
            !castNode.getOutput()->isTypeFloat() &&
            oppositeCastNode.getOutput()->isTypeFloat()) continue;

        // removing casts of Float->Integer->Float can change the values
        if (!isBackwardPass &&
            castNode.getOutput()->isTypeFloat() &&
            !castNode.getInput()->isTypeFloat() &&
            oppositeCastNode.getInput()->isTypeFloat()) continue;

        if (oppositeCastNode.getInput() == castNode.getOutput())  // there are no logical nodes between cast wrappers
        {
            LOG_TRACE(GC, "{}: no logical nodes between the casts", HLLOG_FUNC);
            if (!isBackwardPass) continue;
            // part A.3 in the algorithm:
            if (castNodeParams == oppositeCastNodeParams)
            {
                LOG_TRACE(GC, "{}: casts have same dtype and quantization params", HLLOG_FUNC);
                if (oppositeCastNode.canRemove())
                {
                    LOG_DEBUG(GC,
                              "{}: removing opposite cast node {}",
                              HLLOG_FUNC,
                              oppositeCastNode.getCastNode()->getNodeName());
                    removedNodes += 1;
                    // remove the opposite cast node and replace his output with first cast node input
                    oppositeCastNode.removeFromGraph(g, castNode.getInput(), true);
                }
                else
                {
                    LOG_TRACE(GC,
                              "{}: can't remove opposite cast node {}",
                              HLLOG_FUNC,
                              oppositeCastNode.getCastNode()->getNodeName());
                }
            }
            // part A.2 in the algorithm:
            else if ((nodes.size() == 1) &&
                     canResetCastGuid(oppositeCastNode))  // try to replace both casts with one cast
            {
                LOG_TRACE(GC, "{}: casts have different dtype or quantization params", HLLOG_FUNC);
                StringWithHash newCastGUIDHash(
                    getCastGUID(castNodeParams.getSynDataType(), oppositeCastNodeParams.getSynDataType()));
                if (!KernelDB::instance().isPerfLibKernel(newCastGUIDHash, g.getDeviceId()))
                    continue;  // cast node of new type not exists
                LOG_DEBUG(GC,
                          "{}: replace the second cast with the new cast guid {} and leave the first cast without"
                          " consumers",
                          HLLOG_FUNC,
                          newCastGUIDHash.getKey());
                removedNodes += 1;
                // replace the second cast with the new cast and leave the first cast without consumers
                oppositeCastNode.getCastNode()->setGUID(newCastGUIDHash.getKey());
                // In case the cast is wrapped by reshapes, need to replace the input of the first reshape
                GraphEditor::replaceTensor(g,
                                           oppositeCastNode.getNodesList().front(),
                                           oppositeCastNode.getInput(),
                                           castNode.getInput());
                // Update the data type of cast node input if needed
                if (oppositeCastNode.getCastNode()->getInput(0)->getElementType() !=
                    castNode.getInput()->getElementType())
                {
                    ::updateTensorsType(g,
                                        {oppositeCastNode.getCastNode()->getInput(0)},
                                        castNode.getInput()->getElementType());
                }
            }
        }
        else // there are logical nodes between casts
        {
            LOG_TRACE(GC, "{}: logical nodes between the casts", HLLOG_FUNC);
            if(oppositeCastNodes.empty())
            {
                LOG_TRACE(GC, "{}: first opposite cast to handle - set it as outputCastNodeParams", HLLOG_FUNC);
                outputCastNodeParams = oppositeCastNodeParams;
            }
            oppositeCastNodes.push_back(oppositeCastNode);
            if (!canRemoveOrMergeOppositeCast(g,
                                              oppositeCastNode,
                                              outputCastNodeParams,
                                              oppositeCastNodeParams,
                                              isBackwardPass))
            {
                LOG_TRACE(GC,
                          "{}: can't remove or merge opposite cast (not all opposite casts are the same) - "
                          "stop handling",
                          HLLOG_FUNC);
                oppositeCastNodes.clear();
                break;
            }
        }
    }
    if (oppositeCastNodes.size() == nodes.size()) // can remove or merge all opposite cast nodes (around logical nodes)
    {
        LOG_TRACE(GC, "{}: can remove or merge all opposite cast nodes (around logical nodes)", HLLOG_FUNC);
        // part A.1.1 and B.1.1 in the algorithm:
        // input and output have same quantization data
        LOG_TRACE(GC,
                  "{}: output cast node params: dtype={}, numChannels={}, scale[0]={}, zp[0]={}",
                  HLLOG_FUNC,
                  outputCastNodeParams.getDataTypeString(outputCastNodeParams.m_qDataType),
                  outputCastNodeParams.m_numChannels,
                  outputCastNodeParams.scale(0),
                  outputCastNodeParams.zp(0));
        if (outputCastNodeParams == castNodeParams)
        {
            LOG_DEBUG(GC, "{}: casts have same dtype and quantization params, removing all casts", HLLOG_FUNC);
            removedNodes += oppositeCastNodes.size() + 1;
            removeCastNodes(g, castNode, oppositeCastNodes, tensorsBetweenCasts, isBackwardPass);
        }
        // part A.1.2 and B.1.2 in the algorithm:
        // input and output haven't same quantization data
        else
        {
            LOG_TRACE(GC,
                      "{}: casts have different dtype or quantization params,"
                      " trying to merge all casts",
                      HLLOG_FUNC);
            const QuantizationData& paramsFrom = (isBackwardPass) ? castNodeParams : outputCastNodeParams;
            const QuantizationData& paramsTo = (isBackwardPass) ? outputCastNodeParams : castNodeParams;
            StringWithHash newCastGUIDHash(getCastGUID(paramsFrom.getSynDataType(), paramsTo.getSynDataType()));

            if (KernelDB::instance().isPerfLibKernel(newCastGUIDHash,
                                                     g.getDeviceId()) &&  // cast node of new type exists
                canResetCastGuid(castNode) &&
                canResetCastGuid(oppositeCastNodes[0]))  // It's enough to check only oppositeCastNodes[0] since we
                                                         // modify one node at most
            {
                LOG_DEBUG(GC, "{}: merging casts with guid {}", HLLOG_FUNC, newCastGUIDHash.getKey());
                removedNodes += oppositeCastNodes.size();
                replaceCastNodes(g,
                                 castNode,
                                 oppositeCastNodes,
                                 tensorsBetweenCasts,
                                 newCastGUIDHash.getKey(),
                                 isBackwardPass);
            }
        }

    }
    // part A.4 in the algorithm:
    // all of the consumers of the cast node deleted (not around logical nodes)
    // and his output tensor is not output of the graph
    else if (isBackwardPass &&
             g.getTensorConsumers(castNode.getOutput()).empty() &&
             castNode.canRemove() &&
             !castNode.getOutput()->isEnforcedOutput())
    {
        LOG_DEBUG(GC,
                  "{}: removing first cast node {} since it has no consumers",
                  HLLOG_FUNC,
                  castNode.getCastNode()->getNodeName());
        removedNodes += 1;
        castNode.removeFromGraph(g, castNode.getInput(), false);
    }
    return removedNodes;
}

static bool isNodeWithSingleDataInput(const NodePtr& node)
{
    auto     inputsNum      = node->getNumInputs();
    unsigned shapeInputsNum = node->getNumInputsShapeTensors();
    return ((inputsNum - shapeInputsNum) == 1);
}

// can move backward if there is only one producer that is logical node with one consumer
static bool canMoveBackward(const HabanaGraph& g, const NodeSet& producers)
{
    if (producers.empty()) return false; // there are no producer (tensor is input to the graph)
    if (producers.size() != 1) return false;
    HB_ASSERT_PTR(*producers.begin()); // producer is nullptr
    const NodePtr& producer = *producers.begin();
    if (!producer->isLogicalOperation()) return false; // not logical operation

    if (g.getNodeConsumers(producer).size() != 1) return false; // there are more than one consumers
    return true;
}

// part B in the algorithm:
static unsigned forwardPass(HabanaGraph& g)
{
    unsigned removedNodes = 0;

    auto       fn          = [](const NodePtr& node) { return node->isCast(); };
    NodeVector sortedNodes = g.getTopoSortedNodesCond(fn);

    // forward pass on the graph, in this way we remove always nodes that were already handled
    for (const NodePtr& node : sortedNodes)
    {
        CastWrapper castNode(g, node);
        if (!castNode.canRemove()) continue;
        const NodePtr& producer = g.getTensorProducer(castNode.getInput());

        if (!producer) continue; // First graph node

        // check if between casts there is chain of logical nodes
        bool         existsPersistentTensor = castNode.existsUserManagedTensor(g, false);
        TensorVector tensorsBetweenCasts({castNode.getInput()});
        NodeSet producers({producer});

        // part B.1 in the algorithm:
        while (canMoveBackward(g, producers))
        {
            if (existsPersistentTensor) break; // can't remove or change type of persistent Tensor
            const NodePtr logicalNode = *producers.begin();
            for (const TensorPtr& tensor : logicalNode->getInputs())
            {
                // can't remove or change type of persistent Tensor
                if (g.isUserManagedDram(tensor))
                {
                    existsPersistentTensor = true;
                    break;
                }
                tensorsBetweenCasts.push_back(tensor);
            }
            // move backward to producers of logical node
            producers = g.getNodeProducers(logicalNode, Node::eTensorType::TENSOR_TYPE_DATA, false);
            if (!isNodeWithSingleDataInput(logicalNode)) break;
        }
        if (existsPersistentTensor) continue; // can't remove or change type of persistent Tensor
        removedNodes += removeOrMergeCasts(g, castNode, producers, tensorsBetweenCasts, false);
    }
    return removedNodes;
}

// can move forward if there is only one consumer that is logical node with one input
static bool canMoveForward(const NodeSet& consumers)
{
    if (consumers.size() != 1) return false; // number of consumers not equal to one
    HB_ASSERT_PTR(*consumers.begin()); // consumer is nullptr
    const NodePtr& consumer = *consumers.begin();
    if (!consumer->isLogicalOperation()) return false; // not logical operation

    return isNodeWithSingleDataInput(consumer);
}

// part A in the algorithm:
static unsigned backwardPass(HabanaGraph& g)
{
    unsigned removedNodes = 0;

    auto       fn          = [](const NodePtr& node) { return node->isCast(); };
    NodeVector sortedNodes = g.getTopoSortedNodesCond(fn);
    // backward pass on the graph, in this way we remove always nodes that were already handled
    for (auto nodeIt = sortedNodes.rbegin(); nodeIt != sortedNodes.rend(); ++nodeIt)
    {
        HB_ASSERT((*nodeIt)->getInputs().size() == 1 && (*nodeIt)->getOutputs().size() == 1,
                  "Expecting cast node to have single input and output.");
        CastWrapper castNode(g, *nodeIt);
        if (!castNode.canRemove()) continue;
        NodeList consumersList = g.getTensorConsumers(castNode.getOutput());
        NodeSet consumers(consumersList.begin(), consumersList.end());

        if (consumers.empty()) continue; // Last graph node

        // check if between casts there is chain of logical nodes
        bool         existsPersistentTensor = castNode.existsUserManagedTensor(g, true);
        TensorVector tensorsBetweenCasts({castNode.getOutput()});
        // part A.1 in the algorithm:
        while (canMoveForward(consumers))
        {
            if (existsPersistentTensor) break; // can't remove or change type of persistent Tensor
            const NodePtr logicalNode = *consumers.begin();
            for (const TensorPtr& tensor : logicalNode->getOutputs())
            {
                // can't remove or change type of persistent Tensor
                if (g.isUserManagedDram(tensor))
                {
                    existsPersistentTensor = true;
                    break;
                }
                tensorsBetweenCasts.push_back(tensor);
            }
            consumers = g.getNodeConsumers(logicalNode); // move forward to consumers of logical node
            // stop moving forward if the logical node have more than one output (like split)
            if (logicalNode->getNumOutputs() != 1) break;
        }
        if (existsPersistentTensor) continue; // can't remove or change type of persistent Tensor
        removedNodes += removeOrMergeCasts(g, castNode, consumers, tensorsBetweenCasts, true);
    }
    return removedNodes;
}

bool removeContiguousCastNodes(HabanaGraph& g)
{
    if (!GCFG_ENABLE_CONTIGUOUS_CAST_REMOVAL.value())
    {
        return true;
    }
/*
    General algorithm Two iterations, backward and forward (in this way we allways remove nodes were that already handled)
    a. Backward:
        For each node in (sorted backward - execution order):
            If the node is a cast node (or packing->cast->packing) the candidates are consumers of the cast node
                1. since there is only 1 consumer that is a logical node with 1 input
                    candidates are logical node consumers (while loop)
            *** Exists logical node(s) ***
                    1.1 if all candidates are opposite casts, remove all casts
                    1.2 if all candidates are of the same cast type, and can be replaced with a new cast
                        1.2.1 if there is 1 candidate remove it if it is a down cast, otherwise remove the first cast
                        1.2.2 if there are more than 1 candidates remove the candidates and modify the root cast
            *** Not exists logical nodes ***
                2. if there is only 1 candidate that is not an opposite cast, try to replace both casts with a new single cast
                3. for all candidates: if candidate is an opposite cast remove it
                4. if all of the candidates removed, remove the first cast
    b. Forward:
        For each node in (sorted forward - execution order):
            If the node is a cast node (or packing->cast->packing) the candidate is producer of the cast node
                1. since there is only 1 producer that is logical node with 1 consumer
                    candidates ares logical node producers (while loop)
            *** Exists logical node(s) ***
                    1.1 if all candidates are opposite casts, remove all casts
                    1.2 if all candidates are of the same cast type, and can be replaced with a new cast
                        1.2.1 if there is 1 candidate it's a bug because it should be handled in Backward pass
                        1.2.2 if there are more than 1 candidates remove them
*/

    unsigned int removedNodes = 0;
    removedNodes += backwardPass(g);
    removedNodes += forwardPass(g);
    LOG_DEBUG(GC, "Remove {} cast nodes", removedNodes);
    return true;
}

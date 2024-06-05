#pragma once

#include "habana_graph.h"
#include "graph_compiler/habana_nodes/logical_op_node.h"

class LogicalOpsHandler
{
public:
    using LogicalOpPtr    = std::shared_ptr<LogicalOpNode>;
    using LogicalOpVector = std::vector<LogicalOpPtr>;

    LogicalOpsHandler(HabanaGraph& graph) : m_graph(graph) {}

    bool        handleLogicalOps();
    static bool wantBackwardDirectionShouldCallSwapDirection(const LogicalOpPtr& logicalNode);
    static bool isBackwardNode(const LogicalOpNode& logicalNode)
    {
        return logicalNode.getAliasDirection() == INPUT_TO_OUTPUT;
    }

    static bool isForwardNode(const LogicalOpNode& logicalNode)
    {
        return logicalNode.getAliasDirection() == OUTPUT_TO_INPUT;
    }
    static bool isSwapAliasDirectionProfitable(LogicalOpPtr logicNode, const HabanaGraph& g);
    // safely reset a logical operation, including its dependant nodes
    static void resetLogicalOp(const HabanaGraph& g, const LogicalOpPtr& logicNode);
    static bool isRealInLogical(const TensorPtr& t);

    // Given a tensor and aliasing direction, return all the potential real tensors. The actual real tensor
    // will be determined after the execution of HandleLogicalOps pass, this is just a collection of all the possible
    // real tensors.
    static TensorSet getPotentialRealTensors(const HabanaGraph& g,
                                             const TensorPtr&   tensor,
                                             AliasDirection     direction);

private:
    void handleAndRunLogicalOps(bool isForward);
    void handleBackwardLogicalOp(const LogicalOpPtr& logicalNode);
    void handleForwardLogicalOp(const LogicalOpPtr& logicalNode);
    void handleRealInLogical(const LogicalOpVector& sortedNodes) const;
    void handleSparseLayoutTensors(const LogicalOpVector& sortedNodes);

    void        handleRealTensor(const LogicalOpPtr& node);
    void        handleAliasedTensors(const LogicalOpPtr& node);
    void        addMemcpyIfRequired(const LogicalOpPtr& node);
    void        handleNode(const LogicalOpPtr& logicalNode);
    bool        isHugeAliasTensor(const LogicalOpPtr& logicalNode, unsigned index) const;

    static bool canHandleHugeTensor(const NodePtr& n);

    // memcopy caching
    bool canReuseMemcpy(const NodePtr& originalNode, const TensorPtr& originalTensor, bool insertMemcpyForInput) const;
    void cacheMemcopy(const NodePtr& copy, bool insertMemcpyForInput);
    TensorPtr getCachedCopiedTensor(const TensorPtr& originalTensor,
                                    const NodePtr&   originalNode,
                                    bool             insertMemcpyForInput) const;

    bool        isReduction(const NodePtr& node);
    NodePtr     getReductionInAliasChain(const LogicalOpPtr& node, const TensorPtr& t);
    bool        hasMemoryCoherenceRisk(const LogicalOpPtr& node, const TensorPtr& t);
    bool        shouldResetLogicalOp(const LogicalOpPtr& node);
    static bool swapLogicalAfterReal(const HabanaGraph& g, const TensorPtr& aliasTensor);

    bool validMemoryRelocation(const LogicalOpPtr& logicalNode, const TensorPtr& t) const;
    void logAdjacentBundles(const LogicalOpPtr& logicalNode, const TensorPtr& t) const;
    static bool areSameBundle(const NodePtr& n1, const NodePtr& n2);

    // prioritization function to determine handling order
    static bool compareLogicalOps(const NodePtr& a, const NodePtr& b, bool isFwd);

    HabanaGraph& m_graph;
    // map from (tensor, is-memcpy-input) to new-memcpy
    std::map<std::pair<TensorPtr, bool>, NodePtr> m_memcopiesCache;

    std::vector<std::pair<NodePtr /* original node */, NodePtr /* semantic Node */>> m_nodesToRestore;
};

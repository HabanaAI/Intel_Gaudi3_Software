#pragma once

#include "habana_graph.h"
#include "register_memory_coherence.h"

class CtrlEdgesHandler
{
public:
    CtrlEdgesHandler(HabanaGraph& graph);
    ~CtrlEdgesHandler();
    void handleCtrlEdges();

    NodeSet getRealProducers(const TensorPtr& t) const;
    NodeSet getRealConsumers(const TensorPtr& t) const;

private:
    static NodeVector getReductionNodes(const HabanaGraph& g);

    void handleReductionControl();
    void handleOperandReuseInternalLogicalNodesControl();
    void handleMemoryCoherence();
    void handleMemoryCoherence(const TensorCoherenceMapping::TensorCoherence& allSectionTensors);

    bool areConnected(const NodePtr& source, const NodePtr& target) const;
    bool areSameNode(const NodePtr& n1, const NodePtr& n2) const;
    bool isOverlap(const NodePtr& blocking, const NodePtr& blocked) const;
    bool doesControlExist(const NodeSet& blocking, const NodeSet& blocked) const;

    NodePtr            addCopyOutForTensor(TensorPtr t);
    NodeSet            getTensorProducerAndConsumers(const TensorPtr& t) const;
    NodeSet            getSurroundingRealNodes(const TensorPtr& t) const;
    static bool        isInAliasChain(const TensorPtr& alias, const TensorPtr& real);
    static bool        isDynamicPersistent(const TensorPtr& t);
    NodePtr            findPermutationLogicalTransposeProducer(const TensorPtr& t) const;
    NodePtr            plantReductionMemcopy(const NodePtr& node, const TensorPtr& insertTensor);
    NodeSet            replaceBlockedNodesWithFirstInBundle(const NodeSet& blockedNodes, const NodePtr& node);
    const BundlePlane* getBPGraph() const;
    NodePtr            findDirectReductionMemset(const NodePtr& reduction) const;
    NodeSet getReductionProducersExceptMemset(const NodePtr& reductiontNode, const NodePtr& memsetNode) const;

    HabanaGraph&                m_graph;
    std::map<unsigned, NodePtr> m_firstNodeInBundle;
    mutable bool                m_bPGraphCreated;
    // just a magic number, don't use a optimization with more nodes
    // otherwise compilation time might increase
    static constexpr unsigned m_maxBlockingBlockedProduct = 200;
};
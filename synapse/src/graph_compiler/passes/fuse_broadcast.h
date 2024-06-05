#pragma once
#include "habana_graph.h"
#include "log_manager.h"
#include "types.h"
#include "node_utils.h"

// Currently only producer chain is supported
struct BroadcastChain
{
    NodeVector producerChain;
    NodePtr    sourceNode;
    unsigned   sourceNodeInputIndex;

    bool containsReshape() const
    {
        return std::any_of(producerChain.begin(), producerChain.end(), [](const NodePtr& prod){
            return isLogicalReshape(prod);
        });
    }
    void printChain() const
    {
        LOG_DEBUG(FUSE_BROADCAST, "Broadcast chain starting from: {}", sourceNode->getNodeName());
        for (const auto& prod : getProducerChain())
        {
            LOG_DEBUG(FUSE_BROADCAST, "\t{}", prod->getNodeName());
        }
    }
    unsigned       size() const { return producerChain.size() + 1; }
    const NodePtr& getBroadcastNode() const
    {
        const auto& broadcast = producerChain.back();
        HB_ASSERT(broadcast && broadcast->getNodeType() == Node::TYPE_BROADCAST,
                  "Expecting that the last node in producer chain is broadcast");
        return broadcast;
    }

    NodePtr& getBroadcastNode()
    {
        return const_cast<NodePtr&>(static_cast<const BroadcastChain&>(*this).getBroadcastNode());
    }

    const NodeVector& getProducerChain() const { return producerChain; }

    NodeVector& getProducerChain()
    {
        return const_cast<NodeVector&>(static_cast<const BroadcastChain&>(*this).getProducerChain());
    }

    const NodePtr& getSourceNode() const { return sourceNode; }

    NodePtr& getSourceNode() { return const_cast<NodePtr&>(static_cast<const BroadcastChain&>(*this).getSourceNode()); }
};

using BroadcastChains = std::vector<BroadcastChain>;

class BroadcastChainSourceNodeOperator
{
public:
    virtual bool handleSourceNode(HabanaGraph& g, NodePtr& broadcastNode, NodePtr& sourceNode)    = 0;
    virtual void printOperationInfoToLog(const NodePtr& sourceNode, const NodePtr& broadcastNode) = 0;
};

using BroadcastChainSourceNodeOperatorPtr = std::shared_ptr<BroadcastChainSourceNodeOperator>;

class FuseBroadcastToSourceNode : public BroadcastChainSourceNodeOperator
{
public:
    void printOperationInfoToLog(const NodePtr& sourceNode, const NodePtr& broadcastNode) override;
    bool handleSourceNode(HabanaGraph& g, NodePtr& broadcastNode, NodePtr& sourceNode) override;
};

class PropagateBroadcastAfterSourceNode : public BroadcastChainSourceNodeOperator
{
public:
    void printOperationInfoToLog(const NodePtr& sourceNode, const NodePtr& broadcastNode) override;
    bool handleSourceNode(HabanaGraph& g, NodePtr& broadcastNode, NodePtr& sourceNode) override;

private:
    void resizeNodeOutput(HabanaGraph& g, const NodePtr& node);
};

// Operates on producer chain that start with broadcast in order to remove/propagate broadcast.
class BroadcastChainOperator
{
public:
    BroadcastChainOperator(const BroadcastChainSourceNodeOperatorPtr& sourceNodeOperator)
    : m_sourceNodeOperator(sourceNodeOperator) {};
    virtual ~BroadcastChainOperator() = default;

    virtual bool handleOperation(HabanaGraph& g, BroadcastChain& chain);

private:
    void printOperationInfoToLog(const BroadcastChain& chain)
    {
        HB_ASSERT(m_sourceNodeOperator, "Expecting that source node operator is set");
        m_sourceNodeOperator->printOperationInfoToLog(chain.getSourceNode(), chain.getBroadcastNode());
    };
    void propagateBroadcast(NodePtr& sourceNode, const std::vector<unsigned>& broadcastedDims);
    void propagateBroadcastRemovalUntilSourceNode(HabanaGraph&   g,
                                                  NodeVector&    prodChainWithoutBroadcast,
                                                  const NodePtr& broadcastNode,
                                                  NodePtr&       sourceNode,
                                                  unsigned       sourceNodeInputIndex);
    bool validateChainForOperation(const HabanaGraph& g, const BroadcastChain& chain);
    BroadcastChainSourceNodeOperatorPtr m_sourceNodeOperator;
};

using BroadcastChainOperatorPtr = std::shared_ptr<BroadcastChainOperator>;

class BgemmBroadcastChainFusionHandler;

// Extended broadcast chain operator with move reshape out of broadcast chain..
class BroadcastChainExtendedOperator : public BroadcastChainOperator
{
public:
    BroadcastChainExtendedOperator(const BroadcastChainSourceNodeOperatorPtr& sourceNodeOperator)
    : BroadcastChainOperator(sourceNodeOperator) {};
    virtual ~BroadcastChainExtendedOperator() = default;

    bool handleOperation(HabanaGraph& g, BroadcastChain& chain);

    friend BgemmBroadcastChainFusionHandler;

private:
    void printOperationInfoToLog(const BroadcastChain& chain)
    {
        LOG_DEBUG(FUSE_BROADCAST, "Replace reshape with two other reshapes (for other op and output)");
    }
    static bool    isReshapeSupportsReplacement(const HabanaGraph&                    g,
                                                const std::shared_ptr<BatchGemmNode>& sourceNode,
                                                const NodePtr&                        reshape);
    static NodePtr createLogicalReshape(const TensorPtr& sourceTensor, const TensorPtr& targetTensor);
    NodePtr        createInverseReshapeForOtherOperand(const NodePtr& origReshape,
                                                       const NodePtr& sourceNode,
                                                       unsigned       otherOperandIndex);
    NodePtr        createReshapeForOutputOperand(const NodePtr& origReshape, const NodePtr& sourceNode);
    bool           validateChainForOperation(const HabanaGraph& g, const BroadcastChain& chain, const NodePtr& reshape);
};

// Handlers:
class BroadcastChainHandler
{
public:
    BroadcastChainHandler(HabanaGraph& g, BroadcastChain& chain, const BroadcastChainOperatorPtr& chainOperator)
    : m_graph(g), m_chain(chain), m_chainOperator(chainOperator) {};
    virtual ~BroadcastChainHandler() = default;

    static bool  canHandle(const HabanaGraph& g, const BroadcastChain& chain);
    virtual bool handleChain();
    virtual void turnOnPredicates(){};

protected:
    static bool validSizesForFusion(const BroadcastChain& chain);
    static bool canProducerPropagateBroadcastRemoval(const NodePtr& producer);

    HabanaGraph&              m_graph;
    BroadcastChain&           m_chain;
    BroadcastChainOperatorPtr m_chainOperator;
};

using BroadcastChainHandlerPtr = std::shared_ptr<BroadcastChainHandler>;

class TpcBroadcastChainHandler : public BroadcastChainHandler
{
public:
    TpcBroadcastChainHandler(HabanaGraph& g, BroadcastChain& chain, const BroadcastChainOperatorPtr& chainOperator)
    : BroadcastChainHandler(g, chain, chainOperator) {};
    virtual ~TpcBroadcastChainHandler() = default;

    static bool canHandle(const HabanaGraph& g, const BroadcastChain& chain);

protected:
    static bool doesTensorExistsMoreThanOnce(const BroadcastChain& chain);
    static bool validUtilizationForFusion(const BroadcastChain& chain);
    static bool producerCanBeHandled(const NodePtr& producer);
};

// TpcBroadcastChainFusionHandler example:
//
//            +---+
//            |[1]|
//            +-+-+
//              |
//         +----v----+
//         |broadcast|
//         +----+----+
//              |
//              v
//            +-+-+                                +---+
//            |[x]|                                |[1]|
//            +-+-+                                +-+-+
//              |                                    |
//              v          +--------->               v
//            +-+--+                               +-+--+
//            |cast|                               |cast|
//            +-+--+                               +-+--+
//              |                                    |
//              v                                    v
// +---+      +-+-+                     +---+      +-+-+
// |[x]|      |[x]|                     |[x]|      |[1]|
// +-+-+      +-+-+                     +-+-+      +-+-+
//   |          |                         |          |
//   |          v                         |          v
//   |        +-+-+                       |        +-+-+
//   +------->|tpc|                       +------->|tpc|
//            +-+-+                                +-+-+
//              |                                    |
//              v                                    v
//            +-+-+                                +-+-+
//            |[x]|                                |[x]|
//            +---+                                +---+
class TpcBroadcastChainFusionHandler : public TpcBroadcastChainHandler
{
public:
    TpcBroadcastChainFusionHandler(HabanaGraph& g, BroadcastChain& chain)
    : TpcBroadcastChainHandler(g,
                               chain,
                               std::make_shared<BroadcastChainOperator>(std::make_shared<FuseBroadcastToSourceNode>()))
    {
    }
    virtual ~TpcBroadcastChainFusionHandler() = default;

    static bool canHandle(const HabanaGraph& g, const BroadcastChain& chain);
};

// TpcBroadcastChainPropagationHandler example:
//
//            +---+
//            |[1]|
//            +-+-+
//              |
//         +----v----+
//         |broadcast|
//         +----+----+
//              |
//              v
//            +-+-+                                +---+
//            |[x]|                                |[1]|
//            +-+-+                                +-+-+
//              |                                    |
//              v          +--------->               v
//            +-+--+                               +-+--+
//            |cast|                               |cast|
//            +-+--+                               +-+--+
//              |                                    |
//              v                                    v
// +---+      +-+-+                     +---+      +-+-+
// |[1]|      |[x]|                     |[1]|      |[1]|
// +-+-+      +-+-+                     +-+-+      +-+-+
//   |          |                         |          |
//   |          v                         |          v
//   |        +-+-+                       |        +-+-+
//   +------->|tpc|                       +------->|tpc|
//            +-+-+                                +-+-+
//              |                                    |
//              v                                    v
//            +-+-+                                 -+-+
//            |[x]|                                |[1]|
//            +---+                                +-+-+
//                                                   |
//                                              +----v----+
//                                              |broadcast|
//                                              +----+----+
//                                                   |
//                                                   v
//                                                 +-+-+
//                                                 |[x]|
//                                                 +---+
class TpcBroadcastChainPropagationHandler : public TpcBroadcastChainHandler
{
public:
    TpcBroadcastChainPropagationHandler(HabanaGraph& g, BroadcastChain& chain)
    : TpcBroadcastChainHandler(
          g,
          chain,
          std::make_shared<BroadcastChainOperator>(std::make_shared<PropagateBroadcastAfterSourceNode>()))
    {
    }
    virtual ~TpcBroadcastChainPropagationHandler() = default;

    static bool canHandle(const HabanaGraph& g, const BroadcastChain& chain);
};

// BgemmBroadcastChainFusionHandler example:
//
//                +---------+                                   +---------+
//                |[x,n,1,y]|                                   |[x,n,1,y]|
//                +----+----+                                   +----+----+
//                     |                                             |
//                     v                                             |
//                 +---+-----+                                       |
//                 |broadcast|                                       |
//                 +---+-----+                                       |
//                     |                                             |
//                     v                                             |
//                +----+----+                                        |
//                |[x,n,z,y]|                                        |
//                +----+----+                                        |
//                     |                                             |
//                     v                                             v
//                  +--+-+                                        +--+-+
//                  |cast|                                        |cast|
//                  +--+-+                                        +--+-+
//                     |                                             |
//                     v           +------------>                    v
//                +----+----+                     +---------+   +----+----+
//                |[x,n,z,y]|                     |[n,x,a,b]|   |[x,n,1,y]|
//                +----+----+                     +----+----+   +----+----+
//                     |                               |             |
//                     v                               v             |
//                  +--+----+                       +--+----+        |
//                  |reshape|                       |inverse|        |
//                  |       |                       |reshape|        |
//                  +--+----+                       +--+----+        |
//                     |                               |             |
//                     v                               v             |
// +---------+     +---+-----+                     +---+-----+       |
// |[n,x,a,b]|     |[x,n,a,b]|                     |[n,x,z,y]|       |
// +----+----+     +---+-----+                     +--+------+       |
//      |              |                              |              |
//      |              v                              |              v
//      |         +----+-----+                        |         +----+-----+
//      +-------->|batch_gemm|                        +-------->|batch_gemm|
//                +----+-----+                                  +----+-----+
//                     |                                             |
//                     v                                             v
//                +----+----+                                   +----+----+
//                |[x,x,a,b]|                                   |[x,x,z,y]|
//                +---------+                                   +----+----+
//                                                                   |
//                                                                   v
//                                                                +--+----+
//                                                                |reshape|
//                                                                +--+----+
//                                                                   |
//                                                                   v
//                                                               +---+-----+
//                                                               |[x,x,a,b]|
//                                                               +---------+
class BgemmBroadcastChainFusionHandler : public BroadcastChainHandler
{
public:
    BgemmBroadcastChainFusionHandler(HabanaGraph& g, BroadcastChain& chain)
    : BroadcastChainHandler(
          g,
          chain,
          std::make_shared<BroadcastChainExtendedOperator>(std::make_shared<FuseBroadcastToSourceNode>())) {};
    virtual ~BgemmBroadcastChainFusionHandler() = default;

    void        turnOnPredicates() override { m_graph.turnOnPredicate(PREDICATE_ID_FUSED_NODE_TO_MME); };
    static bool canHandle(const HabanaGraph& g, const BroadcastChain& chain);

private:
    static bool producerCanBeHandled(const HabanaGraph&                    g,
                                     const std::shared_ptr<BatchGemmNode>& sourceNode,
                                     const NodePtr&                        producer);
    static bool canProducerBeRemovedFromBroadcastChain(const HabanaGraph&                    g,
                                                       const std::shared_ptr<BatchGemmNode>& sourceNode,
                                                       const NodePtr&                        producer);
};

// Handlers factories
class BroadcastChainHandlerFactory
{
public:
    // broadcastChain starts with the source node and ends with broadcast
    virtual BroadcastChainHandlerPtr createForChain(HabanaGraph& g, BroadcastChain& chain) = 0;
};

class TpcBroadcastChainFusionHandlerFactory : public BroadcastChainHandlerFactory
{
public:
    BroadcastChainHandlerPtr createForChain(HabanaGraph& g, BroadcastChain& chain) override;
};

class TpcBroadcastChainPropagationHandlerFactory : public BroadcastChainHandlerFactory
{
public:
    BroadcastChainHandlerPtr createForChain(HabanaGraph& g, BroadcastChain& chain) override;
};

using BroadcastChainHandlerFactoryPtr = std::shared_ptr<BroadcastChainHandlerFactory>;

class BgemmBroadcastChainFusionHandlerFactory : public BroadcastChainHandlerFactory
{
public:
    BroadcastChainHandlerPtr createForChain(HabanaGraph& g, BroadcastChain& chain) override;
};

using BroadcastChainHandlersVector = std::vector<BroadcastChainHandlerPtr>;

class BroadcastChainHandlerSelector
{
public:
    static BroadcastChainHandlersVector
    selectHandlers(HabanaGraph&                                        g,
                   BroadcastChain&                                     chain,
                   const std::vector<BroadcastChainHandlerFactoryPtr>& handlersFactory)
    {
        BroadcastChainHandlersVector handlers;
        for (const auto& factory : handlersFactory)
        {
            if (auto handler = factory->createForChain(g, chain))
            {
                handlers.push_back(handler);
            }
        }
        return handlers;
    }
};

// Optimize broadcasts that are part of specific producer chains.
class BroadcastChainManager
{
public:
    using BroadcastChainOperatorPtr = std::shared_ptr<BroadcastChainOperator>;

    BroadcastChainManager(HabanaGraph& g, NodePtr& node, const std::vector<BroadcastChainHandlerFactoryPtr>& factory)
    : m_graph(g), m_sourceNode(node), m_broadcastChainHandlerFactory(factory) {};
    virtual void optimizeChains();

protected:
    BroadcastChainHandlersVector getHandlers(BroadcastChain& chain);

    std::vector<BroadcastChainHandlerFactoryPtr> getFusionHandlersFactories() { return m_broadcastChainHandlerFactory; }

    HabanaGraph&                                 m_graph;
    NodePtr                                      m_sourceNode;
    std::vector<BroadcastChainHandlerFactoryPtr> m_broadcastChainHandlerFactory;

    // Optimize chain, returns true if it was optimized.
    bool optimizeChain(BroadcastChain& chain);

private:
    bool validateChainNodes(const BroadcastChain& chain);
};

class TpcBroadcastChainManager : public BroadcastChainManager
{
public:
    TpcBroadcastChainManager(HabanaGraph& g, NodePtr& node)
    : BroadcastChainManager(g,
                            node,
                            {std::make_shared<TpcBroadcastChainFusionHandlerFactory>(),
                             std::make_shared<TpcBroadcastChainPropagationHandlerFactory>()}) {};
};

class BgemmBroadcastChainManager : public BroadcastChainManager
{
public:
    BgemmBroadcastChainManager(HabanaGraph& g, NodePtr& node)
    : BroadcastChainManager(g, node, {std::make_shared<BgemmBroadcastChainFusionHandlerFactory>()}) {};

    void optimizeChains();
};

// Finds Broadcast chain with given graph, source node, input index.
class BroadcastChainFinder
{
    using ProducerChecker = std::function<bool(const NodePtr& producer)>;

public:
    BroadcastChainFinder(const HabanaGraph& g, const NodePtr& node) : m_graph(g), m_sourceNode(node) {};

    // Returns broadcast chains,
    // broadcast chain is a chain of nodes ordered from the given source node to broadcast node.
    // The returned nodes can be handled with broadcast removal.
    NodeVector getProducerChain(unsigned sourceNodeInputIndex);

    BroadcastChains getChains();

private:
    const HabanaGraph& m_graph;
    const NodePtr&     m_sourceNode;
};
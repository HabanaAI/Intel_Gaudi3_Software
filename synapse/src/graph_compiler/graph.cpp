#include "graph.h"

#include "graph_compiler/print_cycles.h"
#include "habana_pass.h"
#include "infra/defs.h"
#include "log_manager.h"
#include "node_tensor_accessor.h"
#include "tensor.h"
#include "types.h"
#include "utils/bit_array_2d.h"

#include <lemon/connectivity.h>
#include <lemon/core.h>
#include <lemon/list_graph.h>
#include <limits>

#define GTOKEN(object)  (object)->m_graphToken->tokens[m_id]

typedef lemon::ListDigraph          Digraph;
typedef Digraph::NodeMap<NodePtr>   NodeMap;
// map from (tensor, consumer) to list of arcs
typedef std::map<std::pair<TensorPtr, Digraph::Node>, Digraph::Arc> ArcMap;

class Graph::Reachability : public lemon::Dfs<Digraph>
{
public:
    Reachability(lemon::ListDigraph& g) : lemon::Dfs<Digraph>(g) {}
};

// The declaration of GraphContainer should remain in graph.cpp
// so only the class Graph is familiar with the Lemon library.
class GraphContainer
{
public:
    GraphContainer() : m_nm(m_g), m_am({}) {}
    ~GraphContainer()
    {
        m_g.clear();
        m_am.clear();
    }

    GraphContainer(const GraphContainer& other) : m_nm(m_g), m_am(other.arcMap())
    {
        digraphCopy(other.g(), m_g).
            nodeMap(other.nodeMap(), m_nm).
            run();
    }

    GraphContainer& operator=(const GraphContainer& other)
    {
        if (this != &other)
        {
            digraphCopy(other.g(), m_g).
                nodeMap(other.nodeMap(), m_nm).
                run();
            m_am = other.arcMap();
        }
        return *this;
    }

    // non-const accessors
    Digraph& g()        {return m_g;}
    NodeMap& nodeMap()  {return m_nm;}
    ArcMap&  arcMap()   {return m_am;}
    // const accessors
    const Digraph& g() const        {return m_g;}
    const NodeMap& nodeMap() const  {return m_nm;}
    const ArcMap&  arcMap() const   {return m_am;}

private:

    Digraph  m_g;
    NodeMap  m_nm;
    ArcMap   m_am;
};

// The declarations of TensorGraphToken and NodeGraphToken should remain
// in graph.cpp so only the class Graph can interpret the tokens content.
struct TensorGraphToken
{
    struct TokenData
    {
        TokenData() : lmProducer(lemon::INVALID) {}
        Digraph::Node            lmProducer;
        std::set<Digraph::Node>  lmConsumers;
    };
    // Tensor may belong to more than one graph
    std::map<uint64_t, TensorGraphToken::TokenData> tokens;
};
struct NodeGraphToken
{
    struct TokenData
    {
        TokenData() : lmNode(lemon::INVALID) {}
        Digraph::Node  lmNode;
    };
    // Node may belong to more than one graph
    std::map<uint64_t, NodeGraphToken::TokenData> tokens;
};

struct Graph::ConnectivityMap
{
    BitArray2D                 bitArray;
    Digraph::NodeMap<unsigned> sortedIndices;
    Node::eTensorType          tensorType;
};

std::atomic<uint64_t> Graph::s_graphId{0};

Graph::Graph() : m_debugMode(false), m_graphBreakpointMode(false), m_graph(new GraphContainer), m_id(s_graphId++), m_graphChangedInLastPass(false) {}

Graph::Graph(const Graph& other)
: m_debugMode(other.m_debugMode),
  m_graphBreakpointMode(other.m_graphBreakpointMode),
  m_graph(new GraphContainer),
  m_id(s_graphId++),
  m_graphChangedInLastPass(other.m_graphChangedInLastPass)
{
    copyNodesAndTensors(other);
}

Graph::~Graph()
{
    delete m_graph;
}

Graph& Graph::operator=(const Graph& other)
{
    if (this != &other)
    {
        // We cannot use the copy-and-swap idiom because it would invalidate the tokens
        // inside the nodes and tensors. Therefore, we must operate on the current object.
        clear();
        copyNodesAndTensors(other);
        m_debugMode              = other.m_debugMode;
        m_graphBreakpointMode    = other.m_graphBreakpointMode;
        m_graphChangedInLastPass = other.m_graphChangedInLastPass;
    }
    return *this;
}

bool Graph::containsNode(const NodePtr& node) const
{
    if (node == nullptr ||
        node->m_graphToken == nullptr ||
        GTOKEN(node).lmNode == lemon::INVALID)
    {
        return false; // Not a graph node(s)
    }
    return true;
}

bool Graph::hasConsumer(const Node& node) const
{
    for (const TensorPtr& tensor : node.getOutputs())
    {
        if (tensor == nullptr || tensor->m_graphToken == nullptr)
        {
            continue;
        }
        if (!GTOKEN(tensor).lmConsumers.empty())
        {
            return true;
        }
    }

    return false;
}

void Graph::clear()
{
    m_graph->g().clear();
    m_cacheAllNodes.clear();
    m_cacheAllTensors.clear();
}

template <typename CacheT, typename ObjT>
void Graph::updateCache(CacheT& cache, const ObjT& obj, bool add)
{
    if (obj == nullptr)
    {
        return; // no object was given for update
    }

    if (cache.empty())
    {
        // We don't want to maintain the cache during the graph build-up
        // phase, so if cache was not built yet, do not update it.
        return;
    }

    if (add)
    {
        cache.insert(obj);
    }
    else //remove
    {
        cache.erase(obj);
    }
}

void Graph::copyNodesAndTensors(const Graph& other)
{
    TensorMap clonedTensors = cloneTensors(other);

    // Clone nodes
    for (NodePtr n : other.getNodes())
    {
        NodePtr newNode = n->clone();
        clonedTensorsReplacer(n, newNode, clonedTensors); //replace tensors with clones
        addNode(newNode); //insert node to graph
    }
}

void Graph::clonedTensorsReplacer(const NodePtr&                      node,
                                  NodePtr                             newNode,
                                  const TensorMap&  clonedTensors)
{
    HB_ASSERT(node != nullptr && newNode != nullptr, "invalid input null pointers");

    TensorVector replacedTensors;

    //replace tensors with clones
    for (TensorPtr t : node->getOperands())
    {
        if (t != nullptr)
        {
            //avoid replacing a Tensor that was already been replaced
            if (std::find(replacedTensors.begin(), replacedTensors.end(), t) != replacedTensors.end())
            {
                LOG_DEBUG(GC,
                          "{}: Avoid Replace tensor {} ID 0x{:x} of Node {} Type {} BEFORE(1)",
                          HLLOG_FUNC,
                          t->getName(),
                          t->getId(),
                          node->getNodeName(),
                          node->getNodeType());
                continue;
            }

            replacedTensors.push_back(t);

            TensorMap::const_iterator it = clonedTensors.find(t);

            if (it != clonedTensors.end())
            {
                newNode->replaceTensor(t, it->second);
            }
            else
            {
                LOG_ERR(GC,
                        "{}: clonedTensors map doesn't contain value for tensor {} ID 0x{:x} of Node {}",
                        HLLOG_FUNC,
                        t->getName(),
                        t->getId(),
                        node->getNodeName());
                HB_ASSERT(false, "clonedTensors map is missing values");
            }
        }
    }
}

TensorMap Graph::cloneTensors(const Graph& other,
                              bool         copyAddresses /*false*/,
                              bool         keepPersistent /*false*/,
                              bool         keepNames /*false*/) const
{
    TensorMap  clonedTensors;

    // Clone other's tensors
    for (TensorPtr oldTensor : other.getTensors())
    {
        // As tensor may be associated with multiple arcs, clone only those we weren't processed already
        if (clonedTensors.find(oldTensor) == clonedTensors.end())
        {
            TensorPtr newTensor =
                oldTensor->clone(copyAddresses,
                                 true,
                                 keepPersistent,
                                 keepNames ? TensorNameClonePolicy::COPY_NAME : TensorNameClonePolicy::DEFUALT_NAME);
            clonedTensors[oldTensor] = newTensor;
        }
    }
    // Find all tensors that hold pointers to original (non-cloned) tensors and update those pointers to the clones
    for (TensorPtr t : other.getTensors())
    {
        if (t->isAliasedTensor())
        {
            TensorPtr cT  = clonedTensors[t];
            auto cAlias = clonedTensors.find(t->getAliasTensor());
            if (cAlias == clonedTensors.end())
            {
                LOG_ERR(GC, "Error in clone graph, original graph had dangling pointers");
            }
            else
            {
                cT->resetAliasing();
                cT->cloneAliasInfo(t, cAlias->second);
            }
        }
        if (t->isHostAliasedTensor())
        {
            TensorPtr cT  = clonedTensors[t];
            auto cAlias = clonedTensors.find(t->getHostAliasTensor());
            if (cAlias == clonedTensors.end())
            {
                //clone dangling tensor
                TensorPtr clonedHostAlias = t->getHostAliasTensor()->clone(copyAddresses);
                clonedTensors[t->getHostAliasTensor()] = clonedHostAlias;
                cAlias = clonedTensors.find(t->getHostAliasTensor());
            }
            cT->resetHostAliasing();
            cT->cloneHostAliasInfo(t, cAlias->second);
        }
        if (t->hasDramAllocatedTensor())
        {
            cloneDramTensors(t, clonedTensors, copyAddresses);
        }
    }
    //Return the tensor map
    return clonedTensors;
}

void Graph::cloneDramTensors(TensorPtr origTensor, TensorMap &clonedTensorsMap, bool copyAddresses) const
{
    TensorPtr curTensor = origTensor;
    TensorPtr clonedTensor  = clonedTensorsMap[origTensor];
    TensorPtr parentTensor = curTensor->getDramAllocatedTensor();
    TensorPtr clonedParent = nullptr;

    while (parentTensor != nullptr)
    {
        //dram parent tensor is part of the graph tensors
        if (clonedTensorsMap.find(parentTensor) != clonedTensorsMap.end())
        {
            clonedParent = clonedTensorsMap[parentTensor];
            clonedTensor->setDramAllocatedTensor(clonedParent, curTensor->getDramAllocatedTensorOffset());
            break;
        }
        //parent is floating
        else
        {
            //create cloned parent tensor
            clonedParent = parentTensor->clone(copyAddresses);
            clonedTensorsMap[parentTensor] = clonedParent;

            //bind cloned dram to cloned parent
            clonedTensor->setDramAllocatedTensor(clonedParent, curTensor->getDramAllocatedTensorOffset());

            //move up the graph
            curTensor = parentTensor;
            clonedTensor  = clonedParent;
            parentTensor = parentTensor->getDramAllocatedTensor();
            clonedParent = nullptr;
        }
    }
}

bool Graph::operator==(const Graph& other) const
{
    return isomorphicTo(other);
}

NodeSet Graph::getNodeProducers(const NodePtr& node, Node::eTensorType tensorType, bool includeShapeTensors) const
{
    // returning NodeList for backwards compatibility
    NodeSet results;
    if (!containsNode(node))
    {
        LOG_WARN(GC, "Producers query was done on a node that is not part of the graph");
        return results;
    }

    runOnTensorsForType<Node::USAGE_INPUT>(node, tensorType, [&](const TensorPtr& tensor) {
        if (tensor == nullptr) return;
        if (includeShapeTensors == false && tensor->isShapeTensor()) return;
        const NodePtr& prod = getTensorProducer(tensor);
        if(prod != nullptr)
        {
            results.insert(prod);
        }
    });

    return results;
}

NodeSet Graph::getNodeConsumers(const NodePtr& node, Node::eTensorType tensorType, bool includeShapeTensors) const
{
    NodeSet results;
    if (!containsNode(node))
    {
        LOG_WARN(GC, "Consumers query was done on a node that is not part of the graph");
        return results;
    }

    runOnTensorsForType<Node::USAGE_OUTPUT>(node, tensorType, [&](const TensorPtr& t) {
        if (t == nullptr || t->m_graphToken == nullptr || (includeShapeTensors == false && t->isShapeTensor()))
        {
            return;
        }
        for (const Digraph::Node& lmConsumer : GTOKEN(t).lmConsumers)
        {
            results.insert(m_graph->nodeMap()[lmConsumer]);
        }
    });

    return results;
}

bool Graph::addNode(NodePtr node)
{
    if (node == nullptr)
    {
        LOG_ERR(GC, "Graph::addNode Failed to add node- node is null");
        return false;
    }
    if (node->m_graphToken != nullptr && GTOKEN(node).lmNode != lemon::INVALID)
    {
        LOG_ERR(GC, "Graph::addNode Failed to add node- node already exists");
        return false;
    }

    // Add the node to this graph and set node token
    Digraph::Node lmNode = m_graph->g().addNode();
    m_graph->nodeMap()[lmNode] = node;
    if (node->m_graphToken == nullptr)
    {
        node->m_graphToken = std::shared_ptr<NodeGraphToken>(new NodeGraphToken);
    }
    GTOKEN(node).lmNode = lmNode;

    runOnTensorsForType<Node::USAGE_INPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& t) {
        if (t != nullptr)
        {
            addRelationship(t, node, Node::USAGE_INPUT); // Create lemon arcs and set tensor token
        }
    });
    runOnTensorsForType<Node::USAGE_OUTPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& t) {
        if (t != nullptr)
        {
            addRelationship(t, node, Node::USAGE_OUTPUT); // Create lemon arcs and set tensor token
        }
    });

    updateCache(m_cacheAllNodes, node, true);
    m_nodesByID[node->getId()] = node;
    LOG_TRACE(GC,
              "{}: Node {} with guid {} and ID {} was added to the graph",
              HLLOG_FUNC,
              node->getNodeName(),
              node->getGUID(),
              node->getId());
    if (GCFG_CYCLE_PRINTING_LEVEL.value() >= CyclePrintLevel::PRINT_IN_GRAPH_EDIT)
    {
        assert(!printGraphCycles());
    }
    return true;
}

void Graph::removeNode(NodePtr node, NodePtr newProducer /*=nullptr*/)
{
    HB_ASSERT(node != nullptr, "invalid input - null node");
    HB_ASSERT(node->m_graphToken != nullptr && GTOKEN(node).lmNode != lemon::INVALID,
              "Can't remove node which doesn't belong to the graph");

    runOnTensorsForType<Node::USAGE_INPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& t) {
        if (t != nullptr)
        {
            removeRelationship(t, node, Node::USAGE_INPUT); // Remove lemon arc(s) and unset tensor token
        }
    });
    runOnTensorsForType<Node::USAGE_OUTPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& t) {
        if (t != nullptr)
        {
            removeRelationship(t, node, Node::USAGE_OUTPUT); // Remove lemon arc(s) and unset tensor token
        }
    });

    // Move the consumers of the removed node to the new producer if such producer was provided
    if (newProducer != nullptr)
    {
        HB_ASSERT(node->getNumOutputs(Node::TENSOR_TYPE_ALL) == newProducer->getNumOutputs(Node::TENSOR_TYPE_ALL),
              "Removed node and new producer do not have the same output structure");

        for (unsigned i = 0; i < node->getNumOutputs(Node::TENSOR_TYPE_ALL); ++i)
        {
            TensorPtr removedNodeOutput = node->getOutput(i);
            TensorPtr newProducerOutput = newProducer->getOutput(i);

            if (removedNodeOutput == nullptr && newProducerOutput == nullptr) continue;
            HB_ASSERT(removedNodeOutput != nullptr && newProducerOutput != nullptr, "unexpected null");
            HB_ASSERT(removedNodeOutput->m_graphToken != nullptr, "unexpected null");

            // Find all consumers of this output tensor and perform the switch
            // We make a copy of the list since removeRelationship alters the list
            std::set<Digraph::Node> lmConsumers = GTOKEN(removedNodeOutput).lmConsumers;
            for (Digraph::Node lmConsumer : lmConsumers)
            {
                NodePtr consumerNode = m_graph->nodeMap()[lmConsumer];
                removeRelationship(removedNodeOutput, consumerNode);
                consumerNode->replaceTensor(removedNodeOutput, newProducerOutput);
                addRelationship(newProducerOutput, consumerNode);
            }
            HB_ASSERT(GTOKEN(removedNodeOutput).lmProducer == lemon::INVALID &&
                  GTOKEN(removedNodeOutput).lmConsumers.empty(),
                  "Removed node's output was not removed from the graph, even though a new producer was given");
        }
    }

    m_nodeReachability.erase(node->getId());
    // Remove node from graph
    m_graph->g().erase(GTOKEN(node).lmNode);
    GTOKEN(node).lmNode = lemon::INVALID;
    node->m_graphToken  = nullptr;

    updateCache(m_cacheAllNodes, node, false);
    m_nodesByID.erase(node->getId());
    LOG_TRACE(GC, "{}: Node {} with ID {} was removed from the graph", HLLOG_FUNC, node->getNodeName(), node->getId());
}

void Graph::attachNodes(NodePtr from, NodePtr to, unsigned outputIndex, unsigned inputIndex)
{
    HB_ASSERT(from != nullptr && to != nullptr, "invalid input - null node(s)");

    // Connect 'from' node at outputIndex with 'to' node at inputIndex, and disconnect the old inputIndex of 'to'
    TensorPtr outputTensor = from->getOutput(outputIndex);
    removeRelationship(to->getInput(inputIndex), to, Node::USAGE_INPUT);
    to->replaceInput(inputIndex, outputTensor);
    addRelationship(outputTensor, to, Node::USAGE_INPUT);
}

bool Graph::haveSameConnectivity(const NodePtr& n1, const NodePtr& n2, Node::eTensorType tensorType)
{
    bool checkData    = tensorType == Node::TENSOR_TYPE_DATA || tensorType == Node::TENSOR_TYPE_ALL;
    bool checkControl = tensorType == Node::TENSOR_TYPE_CONTROL || tensorType == Node::TENSOR_TYPE_ALL;

    return (n1->getNumInputs(tensorType) == n2->getNumInputs(tensorType)) &&
           (n1->getNumOutputs(tensorType) == n2->getNumOutputs(tensorType)) &&
           (!checkData || (n1->getInputs() == n2->getInputs() && n1->getOutputs() == n2->getOutputs())) &&
           (!checkControl ||
            (n1->getControlInputs() == n2->getControlInputs() && n1->getControlOutputs() == n2->getControlOutputs()));
}

void Graph::replaceSemanticNodes(NodePtr oldNode, NodePtr newNode)
{
    HB_ASSERT(haveSameConnectivity(oldNode, newNode, Node::TENSOR_TYPE_ALL),
              "unable to replace node {} with node {} since they have different connectivity",
              oldNode->getNodeName(),
              newNode->getNodeName());
    HB_ASSERT(oldNode->m_graphToken != nullptr && GTOKEN(oldNode).lmNode != lemon::INVALID,
              "Can't remove node which doesn't belong to the graph");
    HB_ASSERT(newNode->m_graphToken == nullptr || GTOKEN(newNode).lmNode == lemon::INVALID,
              "Can't add node which is already in graph");

    if (newNode->m_graphToken == nullptr)
    {
        newNode->m_graphToken = std::shared_ptr<NodeGraphToken>(new NodeGraphToken);
    }

    Digraph::Node lmNode = GTOKEN(oldNode).lmNode;

    // Remove node from graph
    GTOKEN(oldNode).lmNode = lemon::INVALID;
    m_nodesByID.erase(oldNode->getId());
    updateCache(m_cacheAllNodes, oldNode, false);

    // add node to graph
    GTOKEN(newNode).lmNode        = lmNode;
    m_nodesByID[newNode->getId()] = newNode;
    updateCache(m_cacheAllNodes, newNode, true);
    m_graph->nodeMap()[lmNode] = newNode;

    // erase node reachability, numPaths and topoSort
    invalidateCachedData();
}

void Graph::addRelationship(const TensorPtr& t, const NodePtr& n, Node::eParamUsage usage)
{
    if (t == nullptr || n == nullptr) return;

    Digraph&  g         = m_graph->g();
    ArcMap&   arcMap    = m_graph->arcMap();

    // We expect the node to belong to this graph
    HB_ASSERT(n->m_graphToken != nullptr && GTOKEN(n).lmNode != lemon::INVALID,
              "Cannot add relationship to node which doesn't belong to the graph");

    if (t->m_graphToken == nullptr)
    {
        t->m_graphToken = std::shared_ptr<TensorGraphToken>(new TensorGraphToken);
    }
    if (usage == Node::UNUSED) usage = n->getParamUsage(t);
    if (usage == Node::USAGE_OUTPUT) // n is producer of t
    {
        // Tensor can have only one producer; thus, we expect to find an invalid producer token
        HB_ASSERT(GTOKEN(t).lmProducer == lemon::INVALID, "producer is already registered");

        // Register the producer
        GTOKEN(t).lmProducer = GTOKEN(n).lmNode;

        // Add arc for each registered consumer.
        // Consumers that will come afterwards will add the arc for themselves.
        for (Digraph::Node lmConsumer : GTOKEN(t).lmConsumers)
        {
            if (arcMap.count({t, lmConsumer}) == 0)
            {
                Digraph::Arc a          = g.addArc(GTOKEN(n).lmNode, lmConsumer);
                arcMap[{t, lmConsumer}] = a;
            }
        }
    }
    else if (usage == Node::USAGE_INPUT) // n is consumer of t
    {
        std::set<Digraph::Node>&  lmConsumers  = GTOKEN(t).lmConsumers;
        const Digraph::Node&      lmConsumer   = GTOKEN(n).lmNode;
        const Digraph::Node&      lmProducer   = GTOKEN(t).lmProducer;

        // We allow double-consumption, this is okay and actually needed in some cases

        // Register this node as a consumer
        lmConsumers.insert(lmConsumer);

        // Create arc if the producer has been registered already
        if (lmProducer != lemon::INVALID && arcMap.count({t, lmConsumer}) == 0)
        {
            Digraph::Arc a = g.addArc(lmProducer, lmConsumer);
            arcMap[{t, lmConsumer}] = a;
        }
    }
    else
    {
        HB_ASSERT(false, "Unknown node to tensor relationship");
    }

    invalidateCachedData();
    updateCache(m_cacheAllTensors, t, true);
}

void Graph::removeRelationship(const TensorPtr& t, const NodePtr& n, Node::eParamUsage usage)
{
    if (t == nullptr || n == nullptr) return;

    Digraph&  g         = m_graph->g();
    NodeMap&  nodeMap   = m_graph->nodeMap();
    ArcMap&   arcMap    = m_graph->arcMap();

    // We expect the node and the tensor to belong to this graph
    HB_ASSERT(n->m_graphToken != nullptr && GTOKEN(n).lmNode != lemon::INVALID,
          "Cannot remove relationship to node {} which doesn't belong to the graph", n->getNodeName());
    HB_ASSERT(t->m_graphToken != nullptr,
          "Cannot remove relationship to tensor {} which doesn't belong to the graph", t->getName());

    if (usage == Node::UNUSED) usage = n->getParamUsage(t);
    if (usage == Node::USAGE_OUTPUT) // n is producer of t
    {
        Digraph::Node lmProducer = GTOKEN(t).lmProducer;
        HB_ASSERT(lmProducer != lemon::INVALID && nodeMap[lmProducer] == n, "mismatch in graph structure");
        GTOKEN(t).lmProducer = lemon::INVALID;
        for (const Digraph::Node& lmConsumer : GTOKEN(t).lmConsumers)
        {
            auto arcIt = arcMap.find({t, lmConsumer});
            if (arcIt != arcMap.end())
            {
                g.erase(arcIt->second);
                arcMap.erase(arcIt);
            }
        }
    }
    else if (usage == Node::USAGE_INPUT) // n is consumer of t
    {
        Digraph::Node lmProducer = GTOKEN(t).lmProducer;
        Digraph::Node lmConsumer = GTOKEN(n).lmNode;
        GTOKEN(t).lmConsumers.erase(lmConsumer);
        if (lmProducer != lemon::INVALID)
        {
            auto arcIt = arcMap.find({t, lmConsumer});
            if (arcIt != arcMap.end())
            {
                g.erase(arcIt->second);
                arcMap.erase(arcIt);
            }
        }
    }
    else
    {
        HB_ASSERT(false, "Unknown node to tensor relationship");
    }

    invalidateCachedData();
    if (GTOKEN(t).lmProducer == lemon::INVALID && GTOKEN(t).lmConsumers.empty())
    {
        updateCache(m_cacheAllTensors, t, false);
    }
}

bool Graph::validateConnections() const
{
    Digraph&  g         = m_graph->g();
    NodeMap&  nodeMap   = m_graph->nodeMap();
    ArcMap&   arcMap    = m_graph->arcMap();

    for (Digraph::ArcIt a(g); a != lemon::INVALID; ++a)
    {
        NodePtr producer = nodeMap[g.source(a)];
        NodePtr consumer = nodeMap[g.target(a)];
        if (consumer != nullptr)
        {
            auto checkFunc = [&](const TensorPtr& input) {
                auto arcIt = arcMap.find({input, GTOKEN(consumer).lmNode});
                return (arcIt != arcMap.end() && arcIt->second == a);
            };

            if (std::none_of(consumer->getInputs().begin(), consumer->getInputs().end(), checkFunc) &&
                std::none_of(consumer->getControlInputs().begin(), consumer->getControlInputs().end(), checkFunc))
            {
                LOG_ERR(GC,
                        "{}: Node {} is the target of arc ({} -> {}), but not exists in arcs map",
                        HLLOG_FUNC,
                        consumer->getNodeName(),
                        (producer != nullptr) ? producer->getNodeName() : "Null",
                        consumer->getNodeName());
                return false;
            }
        }
    }
    return true;
}

bool Graph::isAcyclicGraph() const
{
    return lemon::dag(m_graph->g());
}

bool Graph::isConnectedGraph() const
{
    // check graph connectivity by undirecting the graph, since connectivity is a property of undirected graph
    return lemon::connected(lemon::Undirector(m_graph->g()));
}

NodeSet Graph::getIntersectingNodes(const NodeVector& nodes) const
{
    NodeSet ret;

    // container dedicated to hold potential intersection nodes.
    // if a node exists in the container - there is a path from one of the input nodes to it.
    // if the value is "true" - there is also a path from that node to one of the input nodes.
    std::unordered_map<NodePtr, bool> intersectingNodes;

    // mark all inputs nodes as having a path from input nodes and having a path to input nodes.
    for (const NodePtr& n : nodes)
    {
        intersectingNodes[n] = true;
    }

    // find the smallest section in topological sort containing all input nodes (for performance optimization)
    // nodes that are out of this range cannot be intersecting the input nodes.
    const NodeVector& topoSort = getTopoSortedNodes();
    auto              startIt  = std::find_if(topoSort.begin(), topoSort.end(), [&](const NodePtr& n) {
        return intersectingNodes.find(n) != intersectingNodes.end();
    });
    auto              endIt    = std::find_if(topoSort.rbegin(), topoSort.rend(), [&](const NodePtr& n) {
        return intersectingNodes.find(n) != intersectingNodes.end();
    });

    int startIdx = startIt - topoSort.begin();
    int endIdx   = topoSort.size() - (endIt - topoSort.rbegin()) - 1;
    HB_ASSERT((startIdx >= 0) && (endIdx < topoSort.size()) && (startIdx <= endIdx), "intersection nodes not found!");

    // find all nodes that have a path -from- one of the input nodes
    for (int i = startIdx; i <= endIdx; i++)
    {
        const NodePtr& node = topoSort[i];
        auto           it   = intersectingNodes.find(node);
        if (it != intersectingNodes.end())  // means there is a path from one of the input nodes to "node"
        {
            // all its consumers also have a path to them.
            for (const NodePtr& consumer : getNodeConsumers(node, Node::eTensorType::TENSOR_TYPE_ALL))
            {
                // if the consumer does exist in map it will keep it's original value. otherwise, it will be inserted.
                intersectingNodes.emplace(consumer, false);
            }
        }
    }

    // find all nodes that also have path -to- them from one of the input nodes
    for (int i = endIdx; i >= startIdx; i--)
    {
        const NodePtr& node = topoSort[i];
        auto           it   = intersectingNodes.find(node);
        if ((it != intersectingNodes.end()) &&
            it->second)  // means there is a path from and to one of the input nodes from "node"
        {
            ret.insert(node);  // node is intersecting
            for (const NodePtr& producer : getNodeProducers(node, Node::eTensorType::TENSOR_TYPE_ALL))
            {
                // its producers also have a path to one of the input nodes.
                // mark them only if they exist in container (have a path -to- them as well)
                auto producerIt = intersectingNodes.find(producer);
                if (producerIt != intersectingNodes.end())
                {
                    producerIt->second = true;
                }
            }
        }
    }
    return ret;
}

bool Graph::isAncestor(const Node& source, const Node& target) const
{
    if (source.m_graphToken == nullptr ||
        target.m_graphToken == nullptr ||
        GTOKEN(&source).lmNode == lemon::INVALID ||
        GTOKEN(&target).lmNode == lemon::INVALID)
    {
        return false; // Not a graph node(s)
    }
    NodePtr s = m_graph->nodeMap()[GTOKEN(&source).lmNode];
    NodePtr t = m_graph->nodeMap()[GTOKEN(&target).lmNode];
    return isAncestor(s, t);
}

bool Graph::isAncestor(const NodePtr& source, const NodePtr& target) const
{
    if (!containsNode(source) || !containsNode(target))
    {
        return false; // Not a graph node(s)
    }

    /*
     * From lemon docs:
     * "Warning:
     *     The source nodes are inditated as unreachable." [sic]
     * So adding the edge case below.
     */
    if (source == target) return true;

    auto nodeReachabilityIter = m_nodeReachability.find(source->getId());
    //Find dfs search algorithm for source node in cache
    if (nodeReachabilityIter == m_nodeReachability.end())
    {
        nodeReachabilityIter =
            m_nodeReachability.emplace(source->getId(), std::make_shared<Reachability>(m_graph->g())).first;
        nodeReachabilityIter->second->init();
        nodeReachabilityIter->second->addSource(GTOKEN(source).lmNode);
    }

    //If target node not found in dfs yet, continue search until it's done or the node is found
    Reachability& reachability = *nodeReachabilityIter->second;
    reachability.start(GTOKEN(target).lmNode);

    return reachability.reached(GTOKEN(target).lmNode);
}

NodeList Graph::getRootNodes() const
{
    NodeList roots;

    for (NodePtr node : getNodes())
    {
        bool hasNoProducer = true;
        for (TensorPtr t : node->getInputs())
        {
            if (t == nullptr) continue;
            if (getTensorProducer(t) != nullptr)
            {
                hasNoProducer = false;
                break;
            }
        }
        if (hasNoProducer)
        {
            roots.push_back(node);
        }
    }
    return roots;
}

NodeList Graph::getFinalNodes() const
{
    NodeList finalNodes;
    for (const NodePtr& node : getNodes())
    {
        if (!hasConsumer(*node))
        {
            finalNodes.push_back(node);
        }
    }
    return finalNodes;
}

void Graph::setScheduleFlashAttention()
{
    HB_ASSERT(!m_scheduleFlashAttention, "FlashAttention schedule was already determined, it shouldn't be changed");
    m_scheduleFlashAttention = true;
}

std::list<TensorPtr> Graph::getGraphOutputs() const
{
    std::list<TensorPtr> ret;

    for (TensorPtr t : getTensors())
    {
        if (isOutputTensor(t))
        {
            ret.push_back(t);
        }
    }
    return ret;
}

static bool isMatchingConnectivity(const NodePtr&                        n1,
                                   const NodePtr&                        n2,
                                   const TensorPtr&                      t1,
                                   const TensorPtr&                      t2,
                                   std::function<bool(NodePtr, NodePtr)> matchFunc)
{
    if (!n1 || !n2 || !t1 || !t2) return false;
    if (n1->getOutputIndexOfTensor(t1) != n2->getOutputIndexOfTensor(t2)) return false;
    if (!matchFunc(n1, n2)) return false;
    return true;
}

// Try to find a pattern within the graph.
// The function traverses the pattern starting from patternLeaf (3rd parameter)
// and going backwards, trying to match it to the graph starting at graphLeaf
// (1st parameter) and going backwards.
Graph::PatternMatch Graph::patternSearch(const NodePtr&                        graphLeaf,
                                         const Graph&                          patternGraph,
                                         const NodePtr&                        patternLeaf,
                                         std::function<bool(NodePtr, NodePtr)> matchFunc) const
{
    std::stack<std::pair<NodePtr, NodePtr>> toFind;
    PatternMatch                            match;
    match.insert(std::make_pair(patternLeaf, graphLeaf));
    toFind.push(std::make_pair(patternLeaf, graphLeaf));

    while (!toFind.empty())
    {
        auto [patternNode, graphNode] = toFind.top();
        toFind.pop();
        for (unsigned tensorIdx = 0; tensorIdx < patternNode->getNumInputs(); tensorIdx++)
        {
            const TensorPtr& patternT        = patternNode->getInput(tensorIdx);
            const NodePtr&   patternProducer = patternGraph.getTensorProducer(patternT);
            if (!patternProducer) continue;
            const TensorPtr& graphT        = graphNode->getInput(tensorIdx);
            const NodePtr&   graphProducer = getTensorProducer(graphT);

            if (!isMatchingConnectivity(patternProducer, graphProducer, patternT, graphT, matchFunc))
                return PatternMatch();

            match.insert(std::make_pair(patternProducer, graphProducer));
            toFind.push(std::make_pair(patternProducer, graphProducer));
        }
    }
    return match;
}

bool Graph::isSingleProducerForAllOutputs(const Graph* graph)
{
    NodePtr producer;

    for (const auto& output : graph->getGraphOutputs())
    {
        NodePtr curProducer = graph->getTensorProducer(output);
        if (producer == nullptr)
        {
            producer = curProducer;
        }
        else if (producer != curProducer)
        {
            LOG_DEBUG(GC,
                      "{}: producer: {} curProducer: {}",
                      HLLOG_FUNC,
                      producer->getNodeName(),
                      curProducer->getNodeName());
            return false;
        }
    }

    return true;
}

std::vector<Graph::PatternMatch> Graph::findMatches(const Graph&                          patternGraph,
                                                    std::function<bool(NodePtr, NodePtr)> matchFunc) const
{
    HB_ASSERT(isSingleProducerForAllOutputs(&patternGraph), "Pattern does not have a single output node");
    std::vector<PatternMatch> ret;
    const TensorPtr           t           = patternGraph.getGraphOutputs().front();
    const NodePtr             patternLeaf = patternGraph.getTensorProducer(t);
    HB_ASSERT(patternLeaf != nullptr, "No producer for pattern graph output");

    for (const NodePtr& n : getNodes())
    {
        if (!n || !matchFunc(patternLeaf, n)) continue;
        auto match = patternSearch(n, patternGraph, patternLeaf, matchFunc);
        if (!match.empty())
        {
            ret.emplace_back(std::move(match));
        }
    }
    return ret;
}

// Find all occurrences of pattern within the graph.
// The pattern should have only one output (leaf) node.
// Each node returned by this function is a leaf of pattern-occurrence in the graph.
NodeSet Graph::matchPatternWithSingleOutputNode(Graph* pattern, std::function<bool(NodePtr, NodePtr)> matchFunc) const
{
    HB_ASSERT_PTR(pattern);
    HB_ASSERT(isSingleProducerForAllOutputs(pattern), "Pattern does not have a single output node");
    NodeSet matches;

    TensorPtr t          = pattern->getGraphOutputs().front();
    NodePtr   targetNode = pattern->getTensorProducer(t);
    HB_ASSERT(targetNode != nullptr, "No producer for pattern graph output");

    for (const NodePtr& n : getNodes())
    {
        if (!n || !matchFunc(targetNode, n)) continue;
        auto match = patternSearch(n, *pattern, targetNode, matchFunc);
        if (!match.empty())
        {
            matches.insert(n);
        }
    }

    return matches;
}

unsigned Graph::getNumNodes() const
{
    return getNodes().size();
}

bool Graph::isEmpty() const
{
    return getNumNodes() == 0;
}

void Graph::storeTopologicalSort()
{
    m_storedTopoSortedNodes = m_cacheTopoSortedNodes;
}

void Graph::restoreTopologicalSort()
{
    m_cacheTopoSortedNodes = m_storedTopoSortedNodes;
    m_storedTopoSortedNodes.clear();
}

const NodeVector& Graph::getTopoSortedNodes() const
{
    if (m_cacheTopoSortedNodes.empty())
    {
        generateTopoSortedNodes();
    }
    return m_cacheTopoSortedNodes;
}

void Graph::generateTopoSortedNodes() const
{
    if (!isAcyclicGraph())
    {
        if (GCFG_CYCLE_PRINTING_LEVEL.value() >= CyclePrintLevel::PRINT_IF_CYCLE_FOUND)
        {
            this->printGraphCycles();
        }
        LOG_ERR(GC, "found cycle in graph!");
        return; // return empty vector
    }

    Digraph&                    g = m_graph->g();
    NodeMap&                    nodeMap = m_graph->nodeMap();
    Digraph::NodeMap<unsigned>  sortedIndices(g);
    m_cacheTopoSortedNodes = NodeVector(lemon::countNodes(g), nullptr);

    lemon::topologicalSort(g, sortedIndices);

    // Copy according to topologically sorted indices
    for (Digraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        m_cacheTopoSortedNodes[sortedIndices[n]] = nodeMap[n];
    }
}

const NodeSet& Graph::getNodes() const
{
    if (m_cacheAllNodes.empty())
    {
        Digraph&  g        = m_graph->g();
        NodeMap&  nodeMap  = m_graph->nodeMap();

        for (Digraph::NodeIt n(g); n != lemon::INVALID; ++n)
        {
            m_cacheAllNodes.insert(nodeMap[n]);
        }
    }
    return m_cacheAllNodes;
}

const TensorSet& Graph::getTensors() const
{
    if (m_cacheAllTensors.empty())
    {
        Digraph&  g        = m_graph->g();
        NodeMap&  nodeMap  = m_graph->nodeMap();

        for (Digraph::NodeIt n(g); n != lemon::INVALID; ++n)
        {
            for (TensorPtr t : nodeMap[n]->getOperands())
            {
                if (t != nullptr && m_cacheAllTensors.find(t) == m_cacheAllTensors.end())
                {
                    m_cacheAllTensors.insert(t);
                }
            }
        }
    }
    return m_cacheAllTensors;
}

NodeList Graph::getTensorConsumers(const TensorPtr& t) const
{
    NodeList results;
    if (t == nullptr || t->m_graphToken == nullptr)
    {
        return NodeList(); // return empty list
    }
    for (Digraph::Node lmConsumer : GTOKEN(t).lmConsumers)
    {
        results.push_back(m_graph->nodeMap()[lmConsumer]);
    }
    return results;
}

NodePtr Graph::getTensorSingleConsumer(const TensorPtr& t) const
{
    if (t == nullptr || t->m_graphToken == nullptr || GTOKEN(t).lmConsumers.size() != 1)
    {
        return nullptr;
    }
    return m_graph->nodeMap()[*GTOKEN(t).lmConsumers.begin()];
}

NodePtr Graph::getTensorProducer(const TensorPtr& t) const
{
    if (t == nullptr || t->m_graphToken == nullptr || GTOKEN(t).lmProducer == lemon::INVALID)
    {
        return nullptr;
    }
    return m_graph->nodeMap()[GTOKEN(t).lmProducer];
}

uint32_t Graph::getNumberOfTensorProducers(const TensorPtr& t) const
{
    if (t == nullptr || t->m_graphToken == nullptr || GTOKEN(t).lmProducer == lemon::INVALID)
    {
        return 0;
    }
    return 1;
}

uint32_t Graph::getNumberOfTensorConsumers(const TensorPtr& t) const
{
    if (t == nullptr || t->m_graphToken == nullptr)
    {
        return 0;
    }
    return GTOKEN(t).lmConsumers.size();
}

bool Graph::isomorphicTo(const Graph& g) const
{
    //Todo: this doesn't work for graphs with multiple root nodes
    NodeList my_roots = getRootNodes();
    NodeList o_roots  = g.getRootNodes();

    if (my_roots.size() != o_roots.size())
    {
        return false;
    }

    for (auto r : my_roots)
    {
        bool match_found = false;
        for (auto it = o_roots.begin(); it != o_roots.end(); ++it)
        {
            if (compareSubgraphsFromNode(r, g, *it))
            {
                o_roots.erase(it);
                match_found = true;
                break;
            }
        }
        if (!match_found) return false;
    }

    return true;
}

bool Graph::compareSubgraphsFromNode(const NodePtr& mine, const Graph& o, const NodePtr& theirs) const
{
    HB_ASSERT(mine != nullptr && theirs != nullptr, "unexpected null pointer(s)");
    if (*mine != *theirs) return false;
    TensorVector myTensors = mine->getOutputs();
    TensorVector oTensors = theirs->getOutputs();

    for (unsigned i = 0; i < myTensors.size(); ++i)
    {
        NodeList myConsumers = getTensorConsumers(myTensors[i]);
        NodeList oConsumers  = o.getTensorConsumers(oTensors[i]);

        for (auto n : myConsumers)
        {
            bool matchFound = false;
            //Look for a matching node
            for (auto o_it = oConsumers.begin(); o_it != oConsumers.end(); ++o_it)
            {
                NodePtr m = *o_it;
                if (*n == *m)
                {
                    if (compareSubgraphsFromNode(n, o, m))
                    {
                        matchFound = true;
                        oConsumers.erase(o_it);
                        break;
                    }
                }
            }
            if (!matchFound) return false; //Found a node that has no equivalent
        }
    }
    return true;
}

void Graph::topoPrint() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;
    if (isAcyclicGraph())
    {
        LOG_DEBUG(GC, "Graph Tensors:");
        for (const TensorPtr& t : getTensors())
        {
            t->debugPrint();
            printTensorProducerConsumers(t);
        }
        LOG_DEBUG(GC, "Graph Nodes:");
        for (const NodePtr& n : getNodes())
        {
            n->print();
        }
    }
}

void Graph::printTensorProducerConsumers(const TensorPtr& t) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;
    LOG_DEBUG(GC, "    Tensor Producer Consumer Info:");
    if (getTensorProducer(t) == nullptr)
    {
        LOG_DEBUG(GC, "      No producer.");
    }
    else
    {
        LOG_DEBUG(GC, "      Producer: {}", getTensorProducer(t)->getNodeName());
    }
    LOG_DEBUG(GC, "      {} consumers:", getNumberOfTensorConsumers(t));
    for (auto c : getTensorConsumers(t))
    {
        LOG_DEBUG(GC, "            {}", c->getNodeName());
    }
}

void Graph::setDebugMode()
{
    m_debugMode = true;
}

bool Graph::isDebugMode() const
{
    return m_debugMode;
}

void Graph::invalidateCachedData()
{
    m_nodeReachability.clear();
    m_numOfGraphPaths.clear();
    m_cacheTopoSortedNodes.clear();
    m_connectivityMap.reset();
    setGraphChangedInLastPass();
}

void Graph::countAndCache(const NodePtr& source, const NodePtr& target, Node::eTensorType tensorType) const
{
    // calculate number of paths from source to target using a topological sort and a dynamic programming method
    // complexity of O(V+E*log(V)) at worst case, and faster if used a lot (like in sram slicer pass)
    // using the transition rule: numPaths[U] = sum(numPaths[V] : V:consumer(U))
    const NodeVector& topoSort = getTopoSortedNodes();
    // iterate in reverse from target node to source node
    auto it = std::find(topoSort.rbegin(), topoSort.rend(), target);
    auto sourceIt = std::find(it, topoSort.rend(), source);
    HB_ASSERT(it != topoSort.rend(), "target node not found!");
    // if source node comes after target node in the topological sort - there are 0 paths.
    if(sourceIt == topoSort.rend())
    {
        m_numOfGraphPaths.insert({GraphPathsKey(source->getId(), target->getId(), tensorType), 0});
        return;
    }
    // initialize target numPaths - 1 path to itself.
    m_numOfGraphPaths.insert({GraphPathsKey(target->getId(), target->getId(), tensorType), 1});
    // iterate over all nodes in range [Source, Target-1].
    it++; sourceIt++;
    for (; it != sourceIt; it++)
    {
        auto currentKey = GraphPathsKey((*it)->getId(), target->getId(), tensorType);
        auto res = m_numOfGraphPaths.insert({currentKey, 0});
        // meaning numPaths(*it, target) was already calculated and cached.
        if (!res.second) continue;
        for (const NodePtr& consumer : getNodeConsumers(*it, tensorType))
        {
            // if the key doesn't exist, then 'consumer' comes after 'target' in topological sort - no paths.
            auto consumerKey = GraphPathsKey(consumer->getId(), target->getId(), tensorType);
            auto consumerNumPaths = m_numOfGraphPaths.find(consumerKey);
            if (consumerNumPaths == m_numOfGraphPaths.end()) continue;
            // otherwise, apply transition rule.
            if (res.first->second > std::numeric_limits<uint64_t>::max() - consumerNumPaths->second)
            {
                res.first->second = std::numeric_limits<uint64_t>::max();
            }
            else
            {
                res.first->second += consumerNumPaths->second;
            }
        }
    }
}

uint64_t Graph::getNumberOfPaths(const NodePtr& source, const NodePtr& target, Node::eTensorType tensorType) const
{
    HB_ASSERT(containsNode(source), "source node {} doesn't exist in graph!", source->getNodeName());
    HB_ASSERT(containsNode(target), "target node {} doesn't exist in graph!", target->getNodeName());
    auto key = GraphPathsKey(source->getId(), target->getId(), tensorType);
    if(m_numOfGraphPaths.find(key) == m_numOfGraphPaths.end())
    {
        countAndCache(source, target, tensorType);
    }
    return m_numOfGraphPaths.at(key);
}

bool Graph::areConnected(const NodePtr& source, const NodePtr& target, Node::eTensorType tensorType) const
{
    HB_ASSERT(containsNode(source), "source node {} doesn't exist in graph!", source->getNodeName());
    HB_ASSERT(containsNode(target), "target node {} doesn't exist in graph!", target->getNodeName());
    if (m_connectivityMap && m_connectivityMap->tensorType == tensorType)
    {
        unsigned sourceIdx = m_connectivityMap->sortedIndices[GTOKEN(source).lmNode];
        unsigned targetIdx = m_connectivityMap->sortedIndices[GTOKEN(target).lmNode];
        return m_connectivityMap->bitArray.getBit(targetIdx, sourceIdx);
    }
    else
    {
        return getNumberOfPaths(source, target, tensorType) > 0;
    }
}

void Graph::buildConnectivityMap(Node::eTensorType tensorType) const
{
    // already built
    if (m_connectivityMap && m_connectivityMap->tensorType == tensorType) return;

    const Digraph& g        = m_graph->g();
    const unsigned numNodes = getNumNodes();
    getTopoSortedNodes();  // generate m_cacheTopoSortedNodes (index -> node)
    m_connectivityMap = std::unique_ptr<ConnectivityMap>(
        new ConnectivityMap {BitArray2D(numNodes), Digraph::NodeMap<unsigned>(g), tensorType});
    auto& sortedIndices = m_connectivityMap->sortedIndices;
    auto& bitArray      = m_connectivityMap->bitArray;

    lemon::topologicalSort(g, sortedIndices);  //  (node -> index)
    bitArray.setDiagonal();
    for (unsigned i = 0; i < m_cacheTopoSortedNodes.size(); ++i)
    {
        for (const NodePtr& producer : getNodeProducers(m_cacheTopoSortedNodes[i], tensorType))
        {
            unsigned producerIdx = sortedIndices[GTOKEN(producer).lmNode];
            bitArray.bitwiseOr(producerIdx, i, i);
        }
    }
}

NodePtr Graph::getNodeByID(synNodeId nodeID) const
{
    NodesByIDMap::iterator it = m_nodesByID.find(nodeID);
    if(it == m_nodesByID.end())
    {
        return nullptr;
    }

    return it->second;
}

// Use HabanaGraph::isInputTensor by default to avoid missing nodes whose producer is DMA from the host
bool Graph::isInputTensor(const TensorPtr& t) const
{
    return t != nullptr && getTensorProducer(t) == nullptr;
}

bool Graph::isOutputTensor(const TensorPtr& t) const
{
    if (t != nullptr)
    {
        // Output tensors are tensors that have no consumers or their sole consumer is DMA node writing to the host
        const bool isEnforcedOutput = t->isEnforcedOutput();
        const bool isMasked         = t->isMaskedOutput();
        HB_ASSERT(!(isMasked && isEnforcedOutput), "Mutually exclusive flags defined for tensor");

        if (isMasked) return false;
        if (isEnforcedOutput) return true;

        if (getNumberOfTensorConsumers(t) == 0) return true;
    }
    return false;
}

bool Graph::isInputNode(const NodePtr& n) const
{
    const TensorVector& inputs = n->getInputs();
    return std::any_of(inputs.begin(), inputs.end(), [&](const TensorPtr& t) { return isInputTensor(t); });
}
bool Graph::isOutputNode(const NodePtr& n) const
{
    const TensorVector& outputs = n->getOutputs();
    return std::any_of(outputs.begin(), outputs.end(), [&](const TensorPtr& t) { return isOutputTensor(t); });
}

NodeSet Graph::getRealProducers(const TensorPtr& input) const
{
    return getRealProducersExcept(input, [](const NodePtr& n) { return n == nullptr; });
}

NodeSet Graph::getRealProducersExcept(const TensorPtr&                           input,
                                      const std::function<bool(const NodePtr&)>& discardIfFn,
                                      bool                                       includeGraphEdges) const
{
    std::stack<NodePtr> logicalLayers;
    NodeSet             visitedLogicalLayers;  // needed to avoid recheck logical layers that already checked
    NodeSet             realLayers;
    NodePtr             producer = getTensorProducer(input);
    if (producer == nullptr || discardIfFn(producer))
    {
        return realLayers;
    }
    if (producer->isLogicalOperation())
    {
        visitedLogicalLayers.insert(producer);
        logicalLayers.push(producer);
    }
    else
    {
        realLayers.insert(producer);
    }

    while (!logicalLayers.empty())
    {
        producer = logicalLayers.top();
        logicalLayers.pop();
        for (const TensorPtr& producerInput : producer->getInputs())
        {
            if (producerInput->isShapeTensor()) continue;

            const NodePtr& producersProducer = getTensorProducer(producerInput);
            if (producersProducer == nullptr || discardIfFn(producersProducer)) continue;
            else if (producersProducer->isLogicalOperation())
            {
                const auto& res = visitedLogicalLayers.insert(producersProducer);
                if (res.second)  // add logical layer only if it the first occurrence
                {
                    logicalLayers.push(producersProducer);
                }
            }
            else
            {
                realLayers.insert(producersProducer);
            }
        }
        if (includeGraphEdges && isInputNode(producer))
        {
            realLayers.insert(producer);
        }
    }
    return realLayers;
}

NodeSet Graph::getRealConsumers(const TensorPtr& t) const
{
    return getRealConsumersExcept(t, nullptr);
}

NodeSet Graph::getRealConsumersExcept(const TensorPtr& t, const NodePtr& exception, bool includeGraphEdges) const
{
    std::stack<NodePtr> logicalLayers;
    NodeSet             visitedLogicalLayers;  // needed to avoid recheck logical layers that already checked
    NodeSet             realLayers;

    for (const NodePtr& consumer : getTensorConsumers(t))
    {
        if (consumer == nullptr || consumer == exception || consumer->isShapeOperation()) continue;
        if (consumer->isLogicalOperation())
        {
            visitedLogicalLayers.insert(consumer);
            logicalLayers.push(consumer);
        }
        else
        {
            realLayers.insert(consumer);
        }
    }

    while (!logicalLayers.empty())
    {
        const NodePtr consumer = logicalLayers.top();
        logicalLayers.pop();
        for (const NodePtr& consumersConsumer : getNodeConsumers(consumer))
        {
            if (consumersConsumer == nullptr || consumersConsumer == exception || consumersConsumer->isShapeOperation())
            {
                continue;
            }
            if (consumersConsumer->isLogicalOperation())
            {
                const auto& res = visitedLogicalLayers.insert(consumersConsumer);
                if (res.second)  // add logical layer only if it the first occurrence
                {
                    logicalLayers.push(consumersConsumer);
                }
            }
            else
            {
                realLayers.insert(consumersConsumer);
            }
        }
        if (includeGraphEdges && isOutputNode(consumer))
        {
            realLayers.insert(consumer);
        }
    }
    return realLayers;
}

NodeSet Graph::getNodeRealProducers(const NodePtr& node, Node::eTensorType tensorType) const
{
    NodeSet ret;
    runOnTensorsForType<Node::USAGE_INPUT>(node, tensorType, [&](const TensorPtr& input) {
        const NodeSet& producers = getRealProducers(input);
        ret.insert(producers.begin(), producers.end());
    });

    return ret;
}

NodeSet Graph::getNodeRealProducersExcept(const NodePtr&                             node,
                                          Node::eTensorType                          tensorType,
                                          const std::function<bool(const NodePtr&)>& discardIfFn) const
{
    NodeSet ret;
    runOnTensorsForType<Node::USAGE_INPUT>(node, tensorType, [&](const TensorPtr& input) {
        const NodeSet& producers = getRealProducersExcept(input, discardIfFn);
        ret.insert(producers.begin(), producers.end());
    });

    return ret;
}

NodeSet Graph::getNodeRealConsumers(const NodePtr& node, Node::eTensorType tensorType) const
{
    NodeSet ret;
    runOnTensorsForType<Node::USAGE_OUTPUT>(node, tensorType, [&](const TensorPtr& output) {
        const NodeSet& consumers = getRealConsumers(output);
        ret.insert(consumers.begin(), consumers.end());
    });
    return ret;
}

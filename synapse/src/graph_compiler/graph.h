#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "synapse_types.h"
#include "habana_nodes/node.h"
#include "params_file_manager.h"
#include "types.h"

#include <atomic>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

struct recipe_t;
struct shape_plane_graph_t;
class RecipeAllocator;

typedef class GraphContainer* GraphContainer_t;
template<typename T>
using SortableNodeMap = std::map<pNode, T, NodeComparator>;
using GraphPathsKey =
    std::tuple<unsigned, unsigned, Node::eTensorType>;  //<source node id, target node id, tensor type>
class Graph
{
public:
    typedef enum
    {
        PRODUCE,
        CONSUME,
        UNDEFINED
    } NodeTensorRelationship;

    /**
     * Input sizes that should be part of the recipe
     */
    struct TensorData
    {
        std::string name;
        uint32_t    sampleSize;
        TSize       batchSize;
        uint32_t    elementType;
        double      zp;
        double      scale;
        uint32_t    dimensions;
        TSize       dimensionsSize[Tensor::c_tensorMaxNDim];
        uint64_t    sliceOffset;
        TSize       sliceSize;
        TSize       roiSize;
        uint32_t    firstDmaIndex;
        TSize       totalSize;

        // todo - Currently we don't know, for TPC nodes, which input Tensor is ModelParameter.
        //        Due to that we ignore Batch-Size verification for TPC Nodes => Multi-Input where some of them are TPC
        //        nodes is not fully supported.
        bool ignoreBatchVerification;
    };

    typedef std::vector<TensorData> GraphTensorsData;

    Graph();
    Graph(const Graph& other);
    Graph& operator=(const Graph& other);
    virtual ~Graph();
    virtual void clear();

    virtual bool operator==(const Graph& other) const;
    virtual bool operator!=(const Graph& other) const { return !(*this == other); }

    virtual bool     addNode(pNode node);
    virtual void     removeNode(pNode node, pNode newProducer = nullptr);
    void             addRelationship(const TensorPtr& t, const NodePtr& n, Node::eParamUsage usage = Node::UNUSED);
    void             removeRelationship(const TensorPtr& t, const NodePtr& n, Node::eParamUsage usage = Node::UNUSED);
    virtual void     attachNodes(pNode from, pNode to, unsigned outputIndex, unsigned inputIndex);
    virtual NodeList getRootNodes() const;
    virtual NodeList getFinalNodes() const;
    bool             isAncestor(const Node& source, const Node& target) const;
    bool             isAncestor(const pNode& source, const pNode& target) const;
    // Return true if one of the src nodes is ancestor of one of the dst nodes
    template<class NODE_CONT>
    bool                         isAncestor(const NODE_CONT& srcNodes, const NODE_CONT& dstNodes) const;
    NodeList                     getTensorConsumers(const pTensor& t) const;
    // If exists only 1 consumer to the tensor return it, else return nullptr
    NodePtr                      getTensorSingleConsumer(const pTensor& t) const;
    NodePtr                      getTensorProducer(const pTensor& t) const;
    uint32_t                     getNumberOfTensorProducers(const pTensor& t) const;
    uint32_t                     getNumberOfTensorConsumers(const pTensor& t) const;
    NodeSet                      getNodeProducers(const pNode&      node,
                                                  Node::eTensorType tensorType          = Node::TENSOR_TYPE_DATA,
                                                  bool              includeShapeTensors = true) const;
    NodeSet                      getNodeConsumers(const pNode&      node,
                                                  Node::eTensorType tensorType          = Node::TENSOR_TYPE_DATA,
                                                  bool              includeShapeTensors = true) const;
    virtual bool                 serialize(std::stringstream& input, ParamsManager* params) { return false; }
    virtual recipe_t*            serializeDataPlane(RecipeAllocator* recipeAlloc) const { return nullptr; }
    virtual shape_plane_graph_t* serializeShapePlane(RecipeAllocator* recipeAlloc) const { return nullptr; }
    bool                         isEmpty() const;
    bool                         containsNode(const pNode& node) const;
    bool                         hasConsumer(const Node& node) const;
    virtual unsigned             getNumNodes() const;
    virtual const NodeSet&       getNodes() const;

    template<typename Predicate>
    const NodeSet getNodesCond(Predicate predicate) const;

    virtual const TensorSet&     getTensors() const;
    bool                         validateConnections() const;
    bool                         isAcyclicGraph() const;
    bool                         isConnectedGraph() const;
    bool                         isomorphicTo(const Graph& g) const;
    void                         topoPrint() const;  // for debug
    void                         setDebugMode();
    void                         setGraphBreakpointMode() { m_graphBreakpointMode = true; }
    bool                         getGraphBreakpointMode() const { return m_graphBreakpointMode; }
    bool                         isDebugMode() const;
    void                         printTensorProducerConsumers(const pTensor& t) const;
    virtual NodePtr              getNodeByID(synNodeId nodeID) const;
    const NodeVector&            getTopoSortedNodes() const;

    template<typename Predicate>
    const NodeVector getTopoSortedNodesCond(Predicate predicate) const;

    // return all the nodes intersecting on paths between any 2 input nodes
    NodeSet      getIntersectingNodes(const NodeVector& nodes) const;
    void         storeTopologicalSort();
    void         restoreTopologicalSort();
    NodeSet      getRealProducers(const TensorPtr& t) const;
    NodeSet      getRealConsumers(const TensorPtr& t) const;

    // get all real producers of 'input', skip nodes according to predicate 'discardIfFn',
    // if 'includeGraphEdges' is true, return also logical nodes that don't have real producers
    NodeSet getRealProducersExcept(const TensorPtr&                           t,
                                   const std::function<bool(const NodePtr&)>& discardIfFn,
                                   bool                                       includeGraphEdges = false) const;

    // get all real consumers of 't', skip nodes that match 'exception'
    // if 'includeGraphEdges' is true, return also logical nodes that don't have real consumers
    NodeSet getRealConsumersExcept(const TensorPtr& t, const NodePtr& exception, bool includeGraphEdges = false) const;

    NodeSet      getNodeRealProducers(const NodePtr& node, Node::eTensorType tensorType) const;
    NodeSet      getNodeRealProducersExcept(const NodePtr&                             node,
                                            Node::eTensorType                          tensorType,
                                            const std::function<bool(const NodePtr&)>& discardIfFn) const;
    NodeSet      getNodeRealConsumers(const NodePtr& node, Node::eTensorType tensorType) const;
    virtual bool isInputTensor(const pTensor& t) const;
    virtual bool isOutputTensor(const pTensor& t) const;
    bool         isInputNode(const NodePtr& n) const;
    bool         isOutputNode(const NodePtr& n) const;
    bool         printGraphCycles() const;
    static bool
    haveSameConnectivity(const NodePtr& n1, const NodePtr& n2, Node::eTensorType tensorType = Node::TENSOR_TYPE_DATA);

    // return number of paths between source and target nodes
    // if there's no path, the function returns zero
    // if number of paths exceed uint64_t numeric limit, the function returns std::numeric_limits<uint64_t>::max()
    uint64_t getNumberOfPaths(const pNode&      source,
                              const pNode&      target,
                              Node::eTensorType tensorType = Node::TENSOR_TYPE_DATA) const;

    // checks if source and target are connected.
    bool areConnected(const NodePtr& source, const NodePtr& target, Node::eTensorType tensorType) const;

    // builds efficient connectivity map to be used for multiple 'areConnected' calls,
    // as long as the connectivity of the graph hasn't changed
    void buildConnectivityMap(Node::eTensorType tensorType) const;

    // Find all occurrences of pattern within the graph.
    // The pattern should have only one output (leaf) node.
    // Each node returned by this function is a leaf of pattern-occurrence in the graph.
    NodeSet matchPatternWithSingleOutputNode(Graph* pattern, std::function<bool(pNode, pNode)> matchFunc) const;
    std::list<pTensor> getGraphOutputs() const;
    bool isGraphChangedInLastPass() const { return m_graphChangedInLastPass;};
    void setGraphChangedInLastPass() { m_graphChangedInLastPass = true;};
    void clearGraphChangedInLastPass() { m_graphChangedInLastPass = false;};
    bool shouldScheduleFlashAttention() const { return m_scheduleFlashAttention; }
    void setScheduleFlashAttention();

    // a mapping between node in pattern graph -> node in real graph
    using PatternMatch = std::unordered_map<NodePtr, NodePtr>;
    // Try to find a pattern within the graph.
    // The function traverses the pattern starting from patternLeaf (3rd parameter)
    // and going backwards, trying to match it to the graph starting at graphNode
    // (1st parameter) and going backwards.
    // Parameters:
    //   graphLeaf   - start the search from this node in the graph (and going backwards)
    //   patternGraph     - the pattern (which is basically a graph) we are looking for
    //   patternLeaf - start the search from this node in the pattern (and going backwards)
    //   matchFunc   - function to compare nodes, return true upon a match
    PatternMatch              patternSearch(const NodePtr&                        graphLeaf,
                                            const Graph&                          patternGraph,
                                            const NodePtr&                        patternLeaf,
                                            std::function<bool(NodePtr, NodePtr)> matchFunc) const;
    std::vector<PatternMatch> findMatches(const Graph&                          patternGraph,
                                          std::function<bool(NodePtr, NodePtr)> matchFunc) const;

protected:
    static bool isSingleProducerForAllOutputs(const Graph* graph);

    bool         compareSubgraphsFromNode(const pNode& mine, const Graph& o, const pNode& theirs) const;
    virtual void replaceSemanticNodes(NodePtr oldNode, NodePtr newNode);
    TensorMap    cloneTensors(const Graph& other,
                              bool         copyAddresses  = false,
                              bool         keepPersistent = false,
                              bool         keepNames      = false) const;
    void         clonedTensorsReplacer(const pNode&                                        node,
                                       pNode                                               newNode,
                                       const std::map<pTensor, pTensor, TensorComparator>& clonedTensors);
    bool         m_debugMode;
    bool         m_graphBreakpointMode;

private:
    void copyNodesAndTensors(const Graph& other);
    void cloneDramTensors(pTensor t, TensorMap& clonedTensorsMap, bool copyAddresses) const;
    template<typename CacheT, typename ObjT>
    void updateCache(CacheT& cache, const ObjT& obj, bool add);
    void invalidateCachedData();
    void countAndCache(const pNode& source, const pNode& target, Node::eTensorType tensorType) const;
    void generateTopoSortedNodes() const;

    GraphContainer_t     m_graph;                 // underline graph implementation
protected:
    mutable NodeSet      m_cacheAllNodes;         // cache for unsorted nodes
    mutable TensorSet    m_cacheAllTensors;       // cache for all tensors
private:
    mutable NodesByIDMap m_nodesByID;             // map nodes by node ID
    mutable NodeVector   m_cacheTopoSortedNodes;  // cache for topological sort
    NodeVector           m_storedTopoSortedNodes;
    const uint64_t       m_id;

    // Fwd declaration of class to hold whether there is a path to any node from a specific node (root).
    class Reachability;
    struct ConnectivityMap;
    mutable std::unordered_map<unsigned, std::shared_ptr<Reachability>> m_nodeReachability;
    mutable std::map<GraphPathsKey, uint64_t>                           m_numOfGraphPaths;
    mutable std::unique_ptr<ConnectivityMap>                            m_connectivityMap;
    bool                                                                m_graphChangedInLastPass;
    bool                                                                m_scheduleFlashAttention = false;

    static std::atomic<uint64_t> s_graphId;
};

using GraphPtr = std::shared_ptr<Graph>;

template<class NODE_CONT>
bool Graph::isAncestor(const NODE_CONT& srcNodes, const NODE_CONT& dstNodes) const
{
    for (const NodePtr& src : srcNodes)
    {
        for (const NodePtr& dst : dstNodes)
        {
            if (isAncestor(src, dst)) return true;
        }
    }
    return false;
}

// Return nodes set that match predicate
template<typename PredicateFnType>
    const NodeSet Graph::getNodesCond(PredicateFnType predicate) const
{
    const auto& allNodes = getNodes();
    NodeSet applicableNodes;

    std::copy_if(allNodes.begin(),
                 allNodes.end(),
                 std::inserter(applicableNodes, applicableNodes.end()),
                 predicate);

    return applicableNodes;
}

// Returns topological sort of all nodes that match predicate
template<typename PredicateFnType>
const NodeVector Graph::getTopoSortedNodesCond(PredicateFnType predicate) const
{
    const auto& allNodes = getTopoSortedNodes();
    NodeVector applicableNodes;

    std::copy_if(allNodes.begin(),
                 allNodes.end(),
                 std::inserter(applicableNodes, applicableNodes.end()),
                 predicate);

    return applicableNodes;
}

#endif

#pragma once

#include "types.h"
#include "habana_nodes/node_factory.h"
// TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include <gc_interface.h>
#include <gc_protocol.hpp>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

typedef std::shared_ptr<gcapi::FuserTensorAttributes> pFuserTensorAttributes;
typedef std::shared_ptr<gcapi::FuserSection>          pFuserSection;
typedef std::shared_ptr<gcapi::CommonTensorV4_t>      FuserTensorPtrV4;
typedef std::shared_ptr<gcapi::CommonNodeV4_t>        FuserNodePtrV4;
typedef gcapi::FuserTensorAttributes                  FuserTensorAttributesType;
typedef gcapi::FuserSection                           FuserSectionType;
typedef gcapi::CommonEdgeV4_t                         FuserEdgeTypeV4;
typedef gcapi::CommonGraphV4_t                        FuserGraphTypeV4;
typedef gcapi::CommonNodeV4_t                         FuserNodeTypeV4;
typedef gcapi::CommonTensorV4_t                       FuserTensorTypeV4;
typedef gc_protocol::ProtocolGraph                    ProtocolGraph;
typedef gc_protocol::ProtocolNode                     ProtocolNode;
typedef gc_protocol::ProtocolTensor                   ProtocolTensor;

typedef std::pair<uint64_t, DataRange<uint64_t>>      SecIdAddrRangePair;  // first= sectionId, second=addrRange
typedef std::set<SecIdAddrRangePair>                  SecIdAddrRangeSet;   // first= sectionId, second=addrRange

struct PersistentTensorIdAddrRangeStruct
{
    SecIdAddrRangePair secIdAddrRangePair;
    bool isInput;
};

struct ClusterPersistentAddrRangeStruct
{
    // DB for storing all persistent nodes address (sectionId and offset) in the cluster
    SecIdAddrRangeSet                                              persistAddrRangeInTensorsSet;
    SecIdAddrRangeSet                                              persistAddrRangeOutTensorsSet;
    std::unordered_map<int, PersistentTensorIdAddrRangeStruct>     tensorsInPersistMap; //Map Key=TensorId, value=PersistentTensorIdAddrRangeStruct
    std::unordered_set<unsigned int>                               overlapNodes;
};

typedef enum optimizedGraphStatus
{
    optimizedGraphSuccess           = 0,
    optimizedGraphInvalidFusedGraph = 1,
    optimizedGraphFailNoOutputs     = 2,
    optimizedGraphFail              = 3
} optimizedGraphStatus;

class HabanaGraph;
class UnionFindContainer;

class TPCClusterConstructor
{
public:
    explicit TPCClusterConstructor(HabanaGraph& g);
    void computeClusters();
    int getNodeCluster(unsigned nodeId) const;
    std::unordered_set<NodePtr>                                 popNextCluster();
    const std::unordered_map<int, std::unordered_set<NodePtr>>& getClusters();
    int getNumOfClusters() const;

private:
    void createCluster(NodePtr node, bool supportMultiConsmers = false);

    bool updateClusterPersistentAddrRangeStruct(NodePtr                           node,
                                                ClusterPersistentAddrRangeStruct& clusterStruct,
                                                int&                              clusterId);

    void initClusterableNode(NodePtr node);

    bool canBeClustered(HabanaGraph&                      graph,
                        NodePtr                           currNode,
                        NodePtr                           nextNode,
                        ClusterPersistentAddrRangeStruct& currNodeClusterStruct);

    bool addPersistTensorToStructs(TensorPtr tensor, ClusterPersistentAddrRangeStruct& clusterStruct, bool isInput);

    void printOverlapToCluster(ClusterPersistentAddrRangeStruct& currNodeClusterStruct) const;

    bool isOverlapClusters(ClusterPersistentAddrRangeStruct& currNodeClusterStruct,
                           ClusterPersistentAddrRangeStruct& nextNodeClusterStruct);

    void updateclustersOverlapMap(int clusterId, ClusterPersistentAddrRangeStruct& currNodeClusterStruct);

    bool findPartialAndPerfectOverlap(SecIdAddrRangeSet& currClusterInOrOutDB,
                                      SecIdAddrRangeSet& nextClusterInOrOutDB,
                                      bool&              isPerfectOverlap,
                                      bool               deleteInsertedElement = false);

    NodeVector handlePotentialCyclesInClusters();

    bool isOverlapToCluster(NodePtr                           currNode,
                            NodePtr                           nextNode,
                            ClusterPersistentAddrRangeStruct& currNodeClusterStruct,
                            int&                              clusterIdCurrNode,
                            int&                              clusterIdNextNode);

    void updateCycleDetectionDB(bool isUpdate, unsigned clusterId, unsigned oldClusterId, NodePtr node);

    bool handleMultiConsumers(NodePtr   currentNode,
                              NodePtr&  nextNode,
                              bool&     isMultiConsumerNode,
                              bool&     isMultiConsumerNextNode,
                              bool&     isFirstTime,
                              unsigned& clusterId);

    unsigned getTotalNumOfMultiConsCurrNextNodes(NodePtr currentNode,
                                                 NodePtr nextNode,
                                                 bool    isMultiConsumerNode,
                                                 bool&   isMultiConsumerNextNode);

    bool isNodeMultiConsumerOrOutputs(NodePtr node);

    bool isThereCTRLDepCurrNextNode(NodePtr  currNode,
                                    NodePtr  nextNode,
                                    unsigned clusterIDCurrNode,
                                    unsigned clusterIDNextNode);

    void updateClusterCtrlDepDB(unsigned clusterId, unsigned oldClusterId, NodePtr nextNode);

    void updateNodeCtrlDepDB(unsigned clusterId, NodePtr node);

    void updateClustersDBs(bool&                             isMultiConsumerNode,
                           bool                              isMultiConsumerNextNode,
                           unsigned                          clusterId,
                           unsigned                          oldClusterId,
                           unsigned                          oldClusterIdNextNode,
                           NodePtr                           currentNode,
                           NodePtr                           nextNode,
                           ClusterPersistentAddrRangeStruct& currNodeClusterStruct);

    void addComplexGuidExtractedClusters();

    // Maps cluster-id -> ClusterPersistentAddrRangeStruct
    std::unordered_map<int, ClusterPersistentAddrRangeStruct> m_clustersPersistentAddrRangeMap;
    std::unordered_map<int, bool>       m_nodeClustered;       // Map[node-id] is true if the node clustered
    NodeVector                          m_nodes;               // Nodes pushed in topo order
    HabanaGraph&                        m_g;                   // The graph
    std::shared_ptr<UnionFindContainer> m_unionFindContainer;  // Contains all lemon library structures
    // Maps cluster-id -> set of nodes in the cluster
    std::unordered_map<int, std::unordered_set<NodePtr>> m_clustersMap;

    // Maps cluster-id -> Multi Consumers nodes (nodes that it output is consumed by more than 1 node) IDs in the
    // cluster
    std::unordered_map<unsigned, std::unordered_set<unsigned int>> m_numberOfMultiConsumersInClusterMap;

    // Stores the ids of clusters originate from specific complex guids.
    // Those clusters can't be expanded (can't join to or with other nodes).
    std::unordered_set<int> m_complexGuidClusterId;

    bool     m_supportMultiConsumer;           // Represent the current state of id multi consumer supported
    unsigned m_numMaxMultiConsumersInCluster;  // Max allowed multi consumers in cluster
    bool     m_isToUpdateOverlapStruct;        // if needed to update the overlapping struct
    NodeList m_consumersList;                  // the consumers list of the current node that need to try
                                               // union with current node cluster
    unsigned m_currMultiConsumerNodeId;        // the nodeId of the current multi consumer node

    // CTRL DEP info
    // Maps cluster-id -> nodes in the cluster that have ctrl dependencies
    std::unordered_map<unsigned, NodeSet> m_ctrlDepNodes;
    // Maps cluster-id -> nodes that are blocking or blocked by nodes in the cluster
    std::unordered_map<unsigned, NodeSet> m_blockingAndBlockedNodes;
    bool                                  m_isToUpdateCtrlDepDB;
};

typedef std::unordered_map<synNodeId, NodePtr> ClusterNodeMap;  // map of original cluster nodes according to their id

namespace
{
using FuserSectionId   = uint32_t;
using GCSectionId      = uint32_t;
using FuserTensorId    = uint32_t;
using GCTensorId       = uint32_t;
using FuserNodeId      = uint32_t;
using ProtocolNodeId   = uint64_t;
using ProtocolTensorId = uint64_t;
}  // namespace

// This is a wrapper for a single cluster
class GCTPCFuserWrapper
{
public:
    GCTPCFuserWrapper(const std::unordered_set<NodePtr>& clusterNodes,
                      HabanaGraph&                       g,
                      gcapi::pfnFuseGraphV4              graphFuncPtr,
                      gcapi::pfnGetFusedNodePreGraphV4   getPreGraphFuncPtr);
    ~GCTPCFuserWrapper();
    bool optimizeCluster(const HabanaGraph& g);

    const FuserGraphTypeV4& getFuserGraph() { return m_fuserGraph; };

    FuserGraphTypeV4& getOptimizedFuserGraph() { return m_optimizedFuserGraph; };

    const std::unordered_map<unsigned, TensorPtr>& getExternalTensors() { return m_externalTensors; };
    const std::unordered_map<unsigned, TensorPtr>& getInternalTensors() { return m_internalTensors; };
    std::unordered_map<unsigned, TensorPtr>&       getNewGCTensors() { return m_newGCTensors; };
    ClusterNodeMap&                                getClusterNodes() { return m_clusterNodesMap; };

    bool mapNewTensorSection(gcapi::CommonSection& fuserSection);  // map section of new tensor to internal data structs
    // assign new GC Tensor with GC section info, according to Fuser section mapping
    void setNewGCTensorSectionInfo(TensorPtr& gcTensor, gcapi::CommonSection& fuserSection);

    void                    printFuserGraph(const FuserGraphTypeV4& fuserGraph) const;
    gcapi::GlueCodeReturn_t getNodePreGraph(const FuserNodePtrV4& node, FuserGraphTypeV4& preGraph) const;

private:
    void             constructFuserGraph(const HabanaGraph& g);
    FuserTensorPtrV4 gcTensor2FuserTensor(const HabanaGraph& g, TensorPtr gcTensor);
    void             reduceAndClassifyInputs(const HabanaGraph& g, const NodePtr& node, FuserNodePtrV4& fuserNode);
    void             reduceAndClassifyOutputs(const HabanaGraph& g, const NodePtr& node, FuserNodePtrV4& fuserNode);
    bool             isInCluster(const ClusterNodeMap& clusterMap, const NodePtr& node);
    bool             hasTPCNode();

    // Input of the tpc fuser. Constructed from the cluster
    FuserGraphTypeV4 m_fuserGraph;
    // Output of the tpc fuser
    FuserGraphTypeV4 m_optimizedFuserGraph;
    // Node id --> NodePtr
    ClusterNodeMap m_clusterNodesMap;
    // Node id --> fuser node
    std::unordered_map<int, FuserNodePtrV4> m_fuserGraphNodes;
    // Tensor id --> fuser tensor
    std::unordered_map<int, FuserTensorPtrV4> m_fuserGraphTensors;
    // Internal tensor is a tensor that:
    //     1) It's producer is a node from the cluster
    //     2) It's consumer is in the cluster
    std::unordered_map<unsigned, TensorPtr> m_internalTensors;

    // External tensor is a tensor that fulfill one of the following:
    //     1) Tensor that has a producer in the cluster and consumer out of the cluster  (input)
    //     2) Tensor that has a consumer in the cluster and producer out of the cluster (output)
    //     3) Tensor without consumer (graph output)
    //     4) Tensor without producer (graph input)
    std::unordered_map<unsigned, TensorPtr> m_externalTensors;

    // New GC tensor is a tensor that :
    //     1) Wasn't in the original cluster
    //     2) Created by external lib
    std::unordered_map<unsigned, TensorPtr> m_newGCTensors;

    // below mappings store section info.
    // since there can be several tensors with same section id, we need to store section info for reuse.
    std::unordered_map<FuserSectionId, gcapi::FuserSectionType> m_fuserSectionIdToSectionType;
    std::unordered_map<FuserSectionId, GCSectionId>             m_fuserSectionIdToGCSectionId;
    // mapping will be used for handling internal control dependencies
    HabanaGraph&                     m_g;  // Processed graph
    gcapi::pfnFuseGraphV4            m_FuserGraphFunc;
    gcapi::pfnGetFusedNodePreGraphV4 m_getPreGraphFunc;
};

class TPCFuserSharedObject
{
public:
    static TPCFuserSharedObject& instance();

private:
    friend class KernelDB;
    void init(std::string_view fuserLibName);
    void destroy();

public:
    bool isInitialized() const { return m_initialized; };

    gcapi::pfnFuseGraphV4            getFuseGraphFuncPtr() const;
    gcapi::pfnGetFusedNodePreGraphV4 getPreGraphFuncPtr() const;
    void                             releaseFusedGraph(const std::shared_ptr<GCTPCFuserWrapper>& fuser) const;
    const std::string&               getTPCFuserSharedObjectName() const { return m_tpcFuserLibName; }

private:
    // clang-format demands some totally atrocious formatting here
    void                             dynamicLoadSharedObject();
    void                             dynamicUnloadSharedObject();
    bool                             m_initialized = false;          // is the instance initialized
    gcapi::pfnFuseGraphV4            m_fuseGraphFuncPtr;             // a pointer to the entry graph function of TPC Fuser lib
    gcapi::pfnReleaseFuseGraphV4     m_fuserGraphReleaseFuncPtr;     // a pointer to the entry release graph function of TPC Fuser lib
    gcapi::pfnGetFusedNodePreGraphV4 m_fuserGraphGetPreGraphFuncPtr; // a pointer to the entry get pregraph function of TPC Fuser lib
    std::string                      m_tpcFuserLibName;              // tpc-fuser library name.
};

namespace ClusteringUtils
{
bool canBeClusteredBasic(const HabanaGraph& g, const NodePtr node);
bool isClusterConnected(const NodeList& cluster);
}  // namespace ClusteringUtils

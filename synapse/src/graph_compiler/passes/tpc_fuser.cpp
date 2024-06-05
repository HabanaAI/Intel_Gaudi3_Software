#include "tpc_fuser.h"

#include "data_type_utils.h"
#include "gc_interface_utils.hpp"
#include "graph_editor.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "multi_sif.h"
#include "node_factory.h"
#include "node_utils.h"
#include "tpc_kernel_loader.h"
#include "tpc_kernel_names.h"
#include "types_exception.h"
#include "types.h"
#include "utils.h"

// TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include <gc_interface_private.hpp>
#include <gc_interface.h>
#include "tpc_kernel_lib_interface_private.h"
#include "tpc_kernel_lib_interface.h"

#include <lemon/list_graph.h>
#include <lemon/unionfind.h>

#include <memory>
#include <mutex>
#include <string.h>
#include <string>
#include <unordered_set>
#include <vector>

// multi consumer limitation
const unsigned MAX_MULTI_CONSUMERS_IN_CLUSTER = 10;

// Lemon types
typedef lemon::ListDigraph              DirectedGraph;  // Graph for lemon union find
typedef DirectedGraph::Node             UntypedNode;    // Each tpc node represented by untyped node
typedef DirectedGraph::NodeMap<NodePtr> NodePtrMap;     // Untyped-Node -> NodePtr
typedef DirectedGraph::NodeMap<int>
    ItemIntMap;  // Untyped-Node -> Int. NodeMap<T> Read write map of the nodes to type T
typedef lemon::UnionFind<ItemIntMap> UnionFind;  // Union find object
// Internal types
typedef std::unordered_map<int, UntypedNode> UntypedNodeMap;

/********************************************************************************************************************/
/********************************************UnionFindContainer******************************************************/
/********************************************************************************************************************/

/* *
 * UnionFindContainer defined and presented here to insure that
 * TpcFuser only is familiar with the Lemon library
 * */
class UnionFindContainer
{
    /* *
     * Lemon directed graph used only as an input of union find structure
     * this graph does not contain the relations between the nodes (arcs).
     * The relations will be deduced from the dali-graph
     * */
public:
    UnionFindContainer()
    : m_tpcGraph(), m_NodePtrMap(m_tpcGraph), m_untypedNode2cluster(m_tpcGraph), m_unionFind(m_untypedNode2cluster)
    {
    }

    DirectedGraph  m_tpcGraph;             // Graph for internal computations
    NodePtrMap     m_NodePtrMap;           // Maps untyped-node -> NodePtr
    UntypedNodeMap m_nodeId2untypedNode;   // Maps node.id -> untyped-node
    ItemIntMap     m_untypedNode2cluster;  // Maps each node to a cluster. Should not be modified.
                                           // That may cause segmentation fault, invalid data structure,
                                           // or infinite loop.
    UnionFind m_unionFind;                 // Union find  data structure
};

TPCFuserSharedObject& TPCFuserSharedObject::instance()
{
    static TPCFuserSharedObject tpcFuserSharedObject;

    return tpcFuserSharedObject;
}

void TPCFuserSharedObject::init(std::string_view fuserLibName)
{
    if (fuserLibName.empty())
    {
        LOG_ERR(GC_TPC_FUSER, "fuserLibName is wrongly initialized (empty).");
        return;
    }

    if (!m_initialized || (m_tpcFuserLibName != fuserLibName))
    {
        m_tpcFuserLibName = fuserLibName;
        dynamicLoadSharedObject();
        // check if the shared object was loaded successfully
        if (m_fuseGraphFuncPtr)
        {
            m_initialized = true;
            LOG_INFO(GC_TPC_FUSER, "TPC Fuser shared object loaded successfully.");
        }
        else
        {
            LOG_ERR(GC_TPC_FUSER, "TPC Fuser shared object could not be loaded.");
        }
    }
}

void TPCFuserSharedObject::destroy()
{
    dynamicUnloadSharedObject();
    m_fuseGraphFuncPtr             = nullptr;
    m_fuserGraphGetPreGraphFuncPtr = nullptr;
    m_fuserGraphReleaseFuncPtr     = nullptr;

    m_initialized = false;
}

void TPCFuserSharedObject::dynamicLoadSharedObject()
{
    LOG_DEBUG(GC_TPC_FUSER, "TPCFuserSharedObject: Loading shared object {}.", m_tpcFuserLibName);

    // handle is acquired in the destructor using dlopen with appropriate
    // flags, so there is no resource leak of handle.
    void* handle = LoadSharedObject(m_tpcFuserLibName.c_str());
    if (!handle)
    {
        LOG_ERR(GC_TPC_FUSER, "TPCFuserSharedObject: Cannot load library: {} {}", m_tpcFuserLibName, dlerror());
        return;
    }

    // Load the symbols
    *(void**)(&m_fuseGraphFuncPtr) = GetFunction(handle, FUSER_SUPPORTING_DYNAMIC_SHAPES_V4_ENTRY_POINT_NAME);
    if (!m_fuseGraphFuncPtr)
    {
        // No such symbol
        LOG_WARN(GC_TPC_FUSER, "TPCFuserSharedObject: Failed loading FUSER_ENTRY_POINT_NAME, error {}.", dlerror());
        UnloadSharedObject(handle);
        return;
    }

    *(void**)(&m_fuserGraphReleaseFuncPtr) =
        GetFunction(handle, RELEASE_GRAPH_SUPPORTING_DYNAMIC_SHAPES_V4_ENTRY_POINT_NAME);
    if (!m_fuserGraphReleaseFuncPtr)
    {
        // No such symbol
        LOG_WARN(GC_TPC_FUSER,
                 "TPCFuserSharedObject: Failed loading RELEASE_GRAPH_ENTRY_POINT_NAME, error {}.",
                 dlerror());
    }

    *(void**)(&m_fuserGraphGetPreGraphFuncPtr) = GetFunction(handle, FUSER_GET_FUSED_NODE_PREGRAPH_V4_ENTRY_POINT_NAME);
    if (!m_fuserGraphGetPreGraphFuncPtr)
    {
        // No such symbol
        LOG_WARN(GC_TPC_FUSER,
                 "TPCFuserSharedObject: Failed loading FUSER_GET_FUSED_NODE_PREGRAPH_V3_ENTRY_POINT_NAME, error {}.",
                 dlerror());
    }
}

void TPCFuserSharedObject::dynamicUnloadSharedObject()
{
    LOG_DEBUG(GC_TPC_FUSER, "TPCFuserSharedObject: Unloading shared object {}.", m_tpcFuserLibName);
    // test if the object is loaded.
    void* handle = dlopen(m_tpcFuserLibName.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle)
    {
        // unload shared object if loaded.
        UnloadSharedObject(handle);
    }
}

void TPCFuserSharedObject::releaseFusedGraph(const std::shared_ptr<GCTPCFuserWrapper>& fuser) const
{
    // Release TPC Fuser graph
    LOG_TRACE(GC_TPC_FUSER, "{}: Releasing Fuser Graph", HLLOG_FUNC);
    if (m_fuserGraphReleaseFuncPtr)
    {
        FuserGraphTypeV4* graphOut = &fuser->getOptimizedFuserGraph();
        auto              retVal   = m_fuserGraphReleaseFuncPtr(graphOut);
        if (retVal != gcapi::FUSER_SUCCESS)
        {
            LOG_WARN(GC_TPC_FUSER, "{}: TPC fuser release operation failed. return value: {}.", HLLOG_FUNC, retVal);
        }
        // Loop through all the optimized fuser graph nodes and check that all data released
        for (auto node : fuser->getOptimizedFuserGraph().nodes)
        {
            HB_ASSERT((node->paramsSize == 0),
                      "{}: TPC fuser release operation failed, paramsSize is not 0",
                      __FUNCTION__);
            HB_ASSERT((node->nodeParams == nullptr),
                      "{}: TPC fuser release operation failed, nodeParams is not NULL",
                      __FUNCTION__);
        }
    }
    else
    {
        LOG_WARN(GC_TPC_FUSER, "{}: pointer to release fuser graph function is NULL", HLLOG_FUNC);
    }
}

gcapi::pfnFuseGraphV4 TPCFuserSharedObject::getFuseGraphFuncPtr() const
{
    HB_ASSERT(m_initialized, "TPC Fuser not initialized");
    return m_fuseGraphFuncPtr;
}

gcapi::pfnGetFusedNodePreGraphV4 TPCFuserSharedObject::getPreGraphFuncPtr() const
{
    HB_ASSERT(m_initialized, "TPC Fuser not initialized");
    return m_fuserGraphGetPreGraphFuncPtr;
}

bool GCTPCFuserWrapper::hasTPCNode()
{
    return std::any_of(std::begin(m_clusterNodesMap), std::end(m_clusterNodesMap), [](const auto& pair) {
        return HabanaGraph::runsOnTPC(pair.second);
    });
}

/********************************************************************************************************************/
/******************************************TPCClusterConstructor*****************************************************/
/********************************************************************************************************************/

TPCClusterConstructor::TPCClusterConstructor(HabanaGraph& graph)
: m_g(graph), m_unionFindContainer(new UnionFindContainer())
{
    unsigned maxMultiConsGCFG = GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.value();
    if (maxMultiConsGCFG > MAX_MULTI_CONSUMERS_IN_CLUSTER)
    {
        LOG_WARN(GC_TPC_FUSER,
                 "{} Unsupported numer of max allowed multi consumers in cluster {} "
                 "Set to 1",
                 HLLOG_FUNC,
                 maxMultiConsGCFG);
        m_numMaxMultiConsumersInCluster = 1;
    }
    else
    {
        LOG_INFO(GC_TPC_FUSER,
                 "{} Set numer of max allowed multi consumers in cluster"
                 " to {} ",
                 HLLOG_FUNC,
                 maxMultiConsGCFG);
        m_numMaxMultiConsumersInCluster = maxMultiConsGCFG;
    }

    m_supportMultiConsumer = (m_numMaxMultiConsumersInCluster > 0);

    m_isToUpdateCtrlDepDB = false;

    // 1.)Reduction: input nodes to internal representation
    for (const NodePtr& node : graph.getTopoSortedNodes())
    {
        if (!ClusteringUtils::canBeClusteredBasic(graph, node))
        {
            continue;
        }

        initClusterableNode(node);
        // Will be used in a fifo(topological) order
        m_nodes.push_back(node);

        LOG_DEBUG(GC_TPC_FUSER, "TPCClusterConstructor: node name {}", node->getNodeName());
    }
}

void TPCClusterConstructor::initClusterableNode(NodePtr node)
{
    // 1.1 Add new node to the graph
    UntypedNode untypedNode = m_unionFindContainer->m_tpcGraph.addNode();

    // 1.2 Create a binding between the tpc NodePtr to the untyped node
    //     Mapping: untyped-node -> NodePtr
    m_unionFindContainer->m_NodePtrMap[untypedNode] = node;

    // 1.3 Mapping: TPC node id -> untyped node
    m_unionFindContainer->m_nodeId2untypedNode[node->getId()] = untypedNode;

    // 1.4 Set the state of the node to - un-clustered
    m_nodeClustered[node->getId()] = false;

    // 1.5 Add the node to union find. Each node is a part of
    //     set/cluster at this step
    m_unionFindContainer->m_unionFind.insert(untypedNode);
}

bool TPCClusterConstructor::addPersistTensorToStructs(TensorPtr                         tensor,
                                                      ClusterPersistentAddrRangeStruct& clusterStruct,
                                                      bool                              isInput)
{
    // Add persistent tensor to overlapping cluster structures
    // Return true if added and false if the tensor is already there

    uint64_t sectionId        = tensor->getMemorySectionID();
    uint64_t sectionOffset    = tensor->getMemorySectionOffset();
    uint64_t sectionOffsetEnd = sectionOffset + tensor->getTotalSizeInBytes();

    DataRange<uint64_t> rangePair(sectionOffset, sectionOffsetEnd);

    PersistentTensorIdAddrRangeStruct element;
    element.secIdAddrRangePair = std::make_pair(sectionId, rangePair);
    element.isInput            = isInput;

    std::pair<int, PersistentTensorIdAddrRangeStruct> tensorIdPair = std::make_pair(tensor->getId(), element);

    auto const insertionResult   = clusterStruct.tensorsInPersistMap.insert(tensorIdPair);
    auto       wasAlreadyInTheDB = !insertionResult.second;
    if (wasAlreadyInTheDB)
    {
        LOG_TRACE(GC_TPC_FUSER, "{}: tensor {} is already in DB", HLLOG_FUNC, tensor->getName());
        return false;
    }

    LOG_TRACE(GC_TPC_FUSER, "{}: tensor {} added to set tensorsIds", HLLOG_FUNC, tensor->getName());

    bool wasAlreadyInTheInDB  = false;
    bool wasAlreadyInTheOutDB = false;
    if (isInput)
    {
        auto const insertionResultIn =
            clusterStruct.persistAddrRangeInTensorsSet.insert(std::make_pair(sectionId, rangePair));
        wasAlreadyInTheInDB = !insertionResultIn.second;
    }
    else
    {
        auto const insertionResultOut =
            clusterStruct.persistAddrRangeOutTensorsSet.insert(std::make_pair(sectionId, rangePair));
        wasAlreadyInTheOutDB = !insertionResultOut.second;
    }

    if ((wasAlreadyInTheInDB) || (wasAlreadyInTheOutDB))
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "{}: tensor {} sectionId and offset were"
                  " already in DB, isInput ={}",
                  HLLOG_FUNC,
                  tensor->getName(),
                  isInput);
        return false;
    }
    else
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "{}: insert tensor {} to clusterStruct set with"
                  "SectionID {} offset {}-{} to DB, isInput ={}",
                  HLLOG_FUNC,
                  tensor->getName(),
                  sectionId,
                  sectionOffset,
                  sectionOffsetEnd,
                  isInput);
    }

    return true;
}

bool TPCClusterConstructor::updateClusterPersistentAddrRangeStruct(NodePtr                           node,
                                                                   ClusterPersistentAddrRangeStruct& clusterStruct,
                                                                   int&                              clusterId)
{
    // If there are persistent tensors with the same address already in cluster
    // Insert the address of the tensor if it's persistent to the it set
    // There are 2 sets - 1 for all inputs tensors and the other for outputs
    // The function return true if insertion sucessed and false if the
    // address was already existed

    LOG_TRACE(GC_TPC_FUSER, "{}: node {} ", HLLOG_FUNC, node->getNodeName());

    clusterId = getNodeCluster(node->getId());

    if ((m_nodeClustered[node->getId()]) && (clusterId))
    {
        std::unordered_map<int, PersistentTensorIdAddrRangeStruct> tensorsIdSet =
            m_clustersPersistentAddrRangeMap[clusterId].tensorsInPersistMap;
        if (tensorsIdSet.size() != 0)
        {
            // node is clustered, get all details from the cluster
            clusterStruct.tensorsInPersistMap = m_clustersPersistentAddrRangeMap[clusterId].tensorsInPersistMap;
            clusterStruct.persistAddrRangeInTensorsSet =
                m_clustersPersistentAddrRangeMap[clusterId].persistAddrRangeInTensorsSet;
            clusterStruct.persistAddrRangeOutTensorsSet =
                m_clustersPersistentAddrRangeMap[clusterId].persistAddrRangeOutTensorsSet;
            clusterStruct.overlapNodes = m_clustersPersistentAddrRangeMap[clusterId].overlapNodes;

            LOG_INFO(GC_TPC_FUSER,
                     "{}: node {} is clustered, clusterId {}",
                     HLLOG_FUNC,
                     node->getNodeName(),
                     clusterId);
            printOverlapToCluster(clusterStruct);
            return true;
        }
    }

    auto inputsTensors  = node->getInputs();
    auto outputsTensors = node->getOutputs();

    for (const auto& inTensor : inputsTensors)
    {
        if (inTensor == nullptr) continue;
        if (inTensor->isPersistent())
        {
            LOG_TRACE(GC_TPC_FUSER,
                      "{}: node {} has persistent input tensor ID {}",
                      HLLOG_FUNC,
                      node->getNodeName(),
                      inTensor->getId());
            addPersistTensorToStructs(inTensor, clusterStruct, true);
        }
    }

    for (const auto& outTensor : outputsTensors)
    {
        if (outTensor == nullptr) continue;
        if (outTensor->isPersistent())
        {
            LOG_TRACE(GC_TPC_FUSER,
                      "{}: node {} has persistent output tensor ID {} ",
                      HLLOG_FUNC,
                      node->getNodeName(),
                      outTensor->getId());
            addPersistTensorToStructs(outTensor, clusterStruct, false);
        }
    }

    if (clusterStruct.tensorsInPersistMap.empty())
    {
        LOG_TRACE(GC_TPC_FUSER, "{}: node {} has no persistent tesnsors", HLLOG_FUNC, node->getNodeName());
        return false;
    }

    return true;
}

bool TPCClusterConstructor::findPartialAndPerfectOverlap(SecIdAddrRangeSet& currClusterInOrOutDB,
                                                         SecIdAddrRangeSet& nextClusterInOrOutDB,
                                                         bool&              isPerfectOverlap,
                                                         bool               deleteInsertedElement)
{
    for (auto it = nextClusterInOrOutDB.begin(); it != nextClusterInOrOutDB.end(); it++)
    {
        // count duplicate tensors that exist in both clusters
        std::pair<std::set<SecIdAddrRangePair>::iterator, bool> insertionResult    = currClusterInOrOutDB.insert(*it);
        auto                                                    wasAlreadyInTheSet = !insertionResult.second;

        // Count perfect/direct overlap
        if (wasAlreadyInTheSet)
        {
            LOG_INFO(GC_TPC_FUSER,
                     "{}: Find perfect overlap - tensor with sectionId {} range {}-{} already in set",
                     HLLOG_FUNC,
                     (*it).first,
                     (*it).second.start(),
                     (*it).second.end());
            isPerfectOverlap = true;
            return true;
        }
        else
        {
            LOG_INFO(GC_TPC_FUSER,
                     "{}: tensor with sectionId {} range {}-{} added to set",
                     HLLOG_FUNC,
                     (*it).first,
                     (*it).second.start(),
                     (*it).second.end());

            std::set<SecIdAddrRangePair>::iterator iterator1;
            iterator1 = insertionResult.first;

            if (currClusterInOrOutDB.size() == 1)
            {
                LOG_TRACE(GC_TPC_FUSER,
                          "{}: tensor with sectionId {} range {}-{} is the only one in set",
                          HLLOG_FUNC,
                          (*it).first,
                          (*it).second.start(),
                          (*it).second.end());
            }
            else
            {
                // Find partial overlap

                // new inserted element to set
                uint64_t            newElementSectionId = it->first;
                DataRange<uint64_t> newElementRange     = it->second;

                if (iterator1 != currClusterInOrOutDB.begin())
                {
                    // pre element in sorted set
                    iterator1--;
                    uint64_t            preElementSectionId = iterator1->first;
                    DataRange<uint64_t> preElementRange     = iterator1->second;

                    LOG_INFO(GC_TPC_FUSER,
                             "{}: Search overlap with pre tensor sectionId {} range {}-{} ",
                             HLLOG_FUNC,
                             preElementSectionId,
                             preElementRange.start(),
                             preElementRange.end());

                    if ((newElementSectionId == preElementSectionId) && (newElementRange.isOverlap(preElementRange)))
                    {
                        // Found partial overlap
                        LOG_INFO(GC_TPC_FUSER,
                                 "{}: Found overlap tensorwith sectionId {} range {}-{} already in set",
                                 HLLOG_FUNC,
                                 (*it).first,
                                 (*it).second.start(),
                                 (*it).second.end());
                        return true;
                    }
                }
                else
                {
                    LOG_INFO(GC_TPC_FUSER,
                             "{}: The element tensor sectionId {} range {}-{} was inserted to the beginning of the set",
                             HLLOG_FUNC,
                             newElementSectionId,
                             newElementRange.start(),
                             newElementRange.end());
                }

                std::set<SecIdAddrRangePair>::iterator iterator2 = insertionResult.first;

                // Checking if the element is not last in the DB
                iterator2++;
                if (iterator2 != currClusterInOrOutDB.end())
                {
                    // next element in the sorted set
                    uint64_t            nextElementSectionId = iterator2->first;
                    DataRange<uint64_t> nextElementRange     = iterator2->second;

                    LOG_INFO(GC_TPC_FUSER,
                             "{}: Search overlap with next tensor sectionId {} range {}-{} ",
                             HLLOG_FUNC,
                             nextElementSectionId,
                             nextElementRange.start(),
                             newElementRange.end());

                    if ((newElementSectionId == nextElementSectionId) && (newElementRange.isOverlap(nextElementRange)))
                    {
                        // Found partial overlap
                        LOG_INFO(GC_TPC_FUSER,
                                 "{}: Found overlap tensorwith sectionId {} range {}-{} already in set",
                                 HLLOG_FUNC,
                                 (*it).first,
                                 (*it).second.start(),
                                 (*it).second.end());
                        return true;
                    }
                }
                else
                {
                    LOG_INFO(GC_TPC_FUSER,
                             "{}: The element tensor sectionId {} range {}-{} was inserted to the end of the set",
                             HLLOG_FUNC,
                             newElementSectionId,
                             newElementRange.start(),
                             newElementRange.end());
                }
            }

            if (deleteInsertedElement)
            {
                LOG_TRACE(GC_TPC_FUSER,
                          "{}: deleteInsertedElement tensor with sectionId {} range {}-{} ",
                          HLLOG_FUNC,
                          (*it).first,
                          (*it).second.start(),
                          (*it).second.end());

                // delete element duo to only checking for overlap
                currClusterInOrOutDB.erase(insertionResult.first);
            }
        }
    }
    m_isToUpdateOverlapStruct = true;
    return false;
}

bool TPCClusterConstructor::isOverlapClusters(ClusterPersistentAddrRangeStruct& currNodeClusterStruct,
                                              ClusterPersistentAddrRangeStruct& nextNodeClusterStruct)
{
    // check if 2 clusters are overlap
    auto currClusterTensorsID = &currNodeClusterStruct.tensorsInPersistMap;
    auto nextClusterTensorsID = &nextNodeClusterStruct.tensorsInPersistMap;

    std::map<int, PersistentTensorIdAddrRangeStruct> tensorsIdAlreadyInCluster;

    auto currClusterInDB = &currNodeClusterStruct.persistAddrRangeInTensorsSet;
    auto nextClusterInDB = &nextNodeClusterStruct.persistAddrRangeInTensorsSet;

    auto currClusterOutDB = &currNodeClusterStruct.persistAddrRangeOutTensorsSet;
    auto nextClusterOutDB = &nextNodeClusterStruct.persistAddrRangeOutTensorsSet;

    for (auto it = nextClusterTensorsID->begin(); it != nextClusterTensorsID->end(); it++)
    {
        // search for duplicate tensors that exist in both clusters
        // and delete all from next node DB
        auto const insertionResult    = currClusterTensorsID->insert(*it);
        auto       wasAlreadyInTheSet = !insertionResult.second;
        if (wasAlreadyInTheSet)
        {
            // tensorsIdAlreadyInCluster.insert(std::make_pair(it->first, it->second));
            LOG_INFO(GC_TPC_FUSER,
                     "{}: tensor {}, is already in map"
                     "SecId {} dataRange {}-{} isInput {} ",
                     HLLOG_FUNC,
                     it->first,                                     // tensorId
                     it->second.secIdAddrRangePair.first,           // sectionId
                     it->second.secIdAddrRangePair.second.start(),  // start offset
                     it->second.secIdAddrRangePair.second.end(),    // end offset
                     it->second.isInput);                           // isInput

            // Remove all tensors that already in the cluster from nextNodeClusterStruct
            bool isInputInCurrNodeDB = insertionResult.first->second.isInput;
            if ((it->second.isInput) || isInputInCurrNodeDB)
            {
                // Remove from input in DB
                if (isInputInCurrNodeDB)
                {
                    nextClusterInDB->erase(it->second.secIdAddrRangePair);
                }
                else
                {
                    currClusterInDB->erase(it->second.secIdAddrRangePair);
                }
            }
        }
    }

    bool isPerfectOverlap = false;

    if ((findPartialAndPerfectOverlap(*currClusterInDB, *nextClusterOutDB, isPerfectOverlap, true)) ||
        (findPartialAndPerfectOverlap(*currClusterOutDB, *nextClusterInDB, isPerfectOverlap, true)))
    {
        if (isPerfectOverlap == true)
        {
            // Found perfect overlap
            LOG_INFO(GC_TPC_FUSER, "{}: Found perfect overlap between IN and Out tensors that allowed", HLLOG_FUNC);
        }
        else
        {
            // Found partial overlap
            LOG_INFO(GC_TPC_FUSER, "{}: Found partial overlap between IN and Out tensors", HLLOG_FUNC);
            return true;
        }
    }

    if ((findPartialAndPerfectOverlap(*currClusterInDB, *nextClusterInDB, isPerfectOverlap)) ||
        (findPartialAndPerfectOverlap(*currClusterOutDB, *nextClusterOutDB, isPerfectOverlap)))
    {
        // Found partial/perfect overlap
        LOG_INFO(GC_TPC_FUSER, "{}: Found overlap", HLLOG_FUNC);
        return true;
    }

    // There is no not allowed overlap between the clusters
    return false;
}

bool TPCClusterConstructor::isThereCTRLDepCurrNextNode(NodePtr  currNode,
                                                       NodePtr  nextNode,
                                                       unsigned clusterIDCurrNode,
                                                       unsigned clusterIDNextNode)
{
    // check if there are ctrl dep between currNode to nextNode (or their clusters), if so return true
    // Each cluster has 2 maps-
    // m_ctrlDepNodes - Maps cluster-id -> nodes in the cluster that have ctrl dependencies
    // m_blockingAndBlockedNodes - Maps cluster-id -> nodes that are blocking or blocked by nodes in the cluster
    // For each nextNode (or its cluster) check if it has CTRL DEP with currNode (or its cluster) by checking
    // if nextNode cluster's nodes in m_ctrlDepNodes belongs to currNode cluster's nodes in m_blockingAndBlockedNodes

    NodeSet currNodeBockingAndBlockedSet;

    // Get currNode ctrl dep info

    // Insert currNode to new clusterId ctrl DEP DB
    updateNodeCtrlDepDB(clusterIDCurrNode, currNode);

    currNodeBockingAndBlockedSet.insert(m_blockingAndBlockedNodes[clusterIDCurrNode].begin(),
                                        m_blockingAndBlockedNodes[clusterIDCurrNode].end());

    // Get nextNode ctrl dep info

    NodeSet tempNextNodesSet;

    updateNodeCtrlDepDB(clusterIDNextNode, nextNode);
    tempNextNodesSet.insert(m_ctrlDepNodes[clusterIDNextNode].begin(), m_ctrlDepNodes[clusterIDNextNode].end());

    // Check if nextNode (its cluster) is blocked/blocking by currNode (its cluster)

    LOG_TRACE(GC_TPC_FUSER,
              "currNode {} tempNextNodesSet.size() {} and nextNode {} currNodeBockingAndBlockedSet.size() {}",
              currNode->getNodeName(),
              tempNextNodesSet.size(),
              nextNode->getNodeName(),
              currNodeBockingAndBlockedSet.size());

    if (!tempNextNodesSet.empty() && !currNodeBockingAndBlockedSet.empty())
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "currNode {} and nextNode {} (or its clusters), have control dep, check if it dependence",
                  currNode->getNodeName(),
                  nextNode->getNodeName());

        m_isToUpdateCtrlDepDB = true;
        NodeVector     intersection;
        NodeComparator comp;

        std::set_intersection(tempNextNodesSet.begin(),
                              tempNextNodesSet.end(),
                              currNodeBockingAndBlockedSet.begin(),
                              currNodeBockingAndBlockedSet.end(),
                              std::back_inserter(intersection),
                              comp);

        if (intersection.size() != 0)
        {
            LOG_TRACE(
                GC_TPC_FUSER,
                "Can't clustered currNode {} and nextNode {}, there is dependency within the fused nodes elements",
                currNode->getNodeName(),
                nextNode->getNodeName());
            return true;
        }
    }
    return false;
}

bool TPCClusterConstructor::canBeClustered(HabanaGraph&                      graph,
                                           NodePtr                           currNode,
                                           NodePtr                           nextNode,
                                           ClusterPersistentAddrRangeStruct& currNodeClusterStruct)
{
    // Check if nextNode can be clustered together with currNode
    // TODO SW-77041 - stop the loop when reaching node in the same cluster and m_consumersList is empty

    if (!ClusteringUtils::canBeClusteredBasic(m_g, nextNode))
    {
        LOG_TRACE(GC_TPC_FUSER, "{}: node {} is not clusterable.", HLLOG_FUNC, nextNode->getNodeName());
        return false;
    }

    if (m_complexGuidClusterId.find(getNodeCluster(nextNode->getId())) != m_complexGuidClusterId.end())
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "{}: node {} is part of non-expandable complex guid cluster.",
                  HLLOG_FUNC,
                  nextNode->getNodeName());
        return false;
    }

    int clusterIDCurrNode = -1;
    int clusterIDNextNode = -1;

    if (isOverlapToCluster(currNode, nextNode, currNodeClusterStruct, clusterIDCurrNode, clusterIDNextNode))
    {
        LOG_TRACE(GC_TPC_FUSER, "{}: node {} is overlap with current cluster.", HLLOG_FUNC, nextNode->getNodeName());
        return false;
    }

    if ((clusterIDCurrNode == clusterIDNextNode) && (clusterIDCurrNode != -1))
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "{}: currNode {} and nextNode {} have the same clusterId {}.",
                  HLLOG_FUNC,
                  currNode->getNodeName(),
                  nextNode->getNodeName(),
                  clusterIDCurrNode);
        return true;
    }

    if (m_g.isControlDependencyConfigured())
    {
        // check if there are ctrl dep between currNode to nextNode, if so return true
        if (isThereCTRLDepCurrNextNode(currNode, nextNode, clusterIDCurrNode, clusterIDNextNode))
        {
            // There is internal control dependencies between currNode cluster to NextNode
            return false;
        }
    }

    if (m_supportMultiConsumer)
    {
        // check if both nodes will be clustered together the number of multi consumers nodes
        // would not reach more than max allowed
        bool isMultiConsumerNextNode;
        bool isMultiConsumerNode = isNodeMultiConsumerOrOutputs(currNode);

        auto totalNumOfMultiConsumers =
            getTotalNumOfMultiConsCurrNextNodes(currNode, nextNode, isMultiConsumerNode, isMultiConsumerNextNode);
        if (totalNumOfMultiConsumers > m_numMaxMultiConsumersInCluster)
        {
            // Reach to max number of multi consumers in this cluster, Done clustering
            LOG_DEBUG(GC_TPC_FUSER,
                      "reach to max number of multi consumers in this cluster for currNode {} and nexNode {}"
                      " Done clustering",
                      currNode->getNodeName(),
                      nextNode->getNodeName());
            return false;
        }
    }

    return true;
}

bool TPCClusterConstructor::isOverlapToCluster(NodePtr                           currNode,
                                               NodePtr                           nextNode,
                                               ClusterPersistentAddrRangeStruct& currNodeClusterStruct,
                                               int&                              clusterIDCurrNode,
                                               int&                              clusterIDNextNode)
{
    // Check if there is not allowed overlap between 2 nodes (that might be part of clusters)

    LOG_TRACE(GC_TPC_FUSER,
              "{}: curr node {} next node {}",
              HLLOG_FUNC,
              currNode->getNodeName(),
              nextNode->getNodeName());

    ClusterPersistentAddrRangeStruct nextNodeClusterStruct;
    bool                             isNodeHasPersistOps = false;

    // Check for persistent tensors in curr node update it cluster struct
    isNodeHasPersistOps = updateClusterPersistentAddrRangeStruct(currNode, currNodeClusterStruct, clusterIDCurrNode);
    if (isNodeHasPersistOps == false)
    {
        LOG_INFO(GC_TPC_FUSER,
                 "{}: current node {} "
                 "is not overlap due to not having persistent tensors",
                 HLLOG_FUNC,
                 currNode->getNodeName());
    }

    // Check for persistent tensors in new node
    bool isNextNodeHasPersistOps =
        updateClusterPersistentAddrRangeStruct(nextNode, nextNodeClusterStruct, clusterIDNextNode);

    if ((isNodeHasPersistOps == false) || (isNextNodeHasPersistOps == false))
    {
        LOG_INFO(GC_TPC_FUSER,
                 "{}: next node {} that we want to union with existing cluster/node "
                 "is not overlap due to not having persistent tensors",
                 HLLOG_FUNC,
                 nextNode->getNodeName());
        return false;
    }

    // Check if nextNode is already in currNode overlap node set
    const auto& currNodeOverlapNodesSet = currNodeClusterStruct.overlapNodes;
    const auto& nextNodeOverlapNodesSet = nextNodeClusterStruct.overlapNodes;
    if (currNodeOverlapNodesSet.count(nextNode->getId()) || nextNodeOverlapNodesSet.count(currNode->getId()))
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "{}: node {} is overlap with cluster of node {}",
                  HLLOG_FUNC,
                  nextNode->getNodeName(),
                  currNode->getNodeName());
        return true;
    }

    LOG_INFO(GC_TPC_FUSER,
             "{}: curr node clusterId {} next node  clusterID {}",
             HLLOG_FUNC,
             clusterIDCurrNode,
             clusterIDNextNode);

    if ((clusterIDCurrNode > -1) && (clusterIDCurrNode == clusterIDNextNode))
    {
        LOG_INFO(GC_TPC_FUSER,
                 "{}: curr node and next node are in the same clusterID {} ",
                 HLLOG_FUNC,
                 clusterIDCurrNode);
        return false;
    }

    // Check for overlap

    if (isOverlapClusters(currNodeClusterStruct, nextNodeClusterStruct))
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "{}: node {} is overlap with cluster {}",
                  HLLOG_FUNC,
                  nextNode->getNodeName(),
                  clusterIDCurrNode);
        m_clustersPersistentAddrRangeMap[clusterIDCurrNode].overlapNodes.insert(nextNode->getId());
        return true;
    }

    return false;
}

void TPCClusterConstructor::printOverlapToCluster(ClusterPersistentAddrRangeStruct& currNodeClusterStruct) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC_TPC_FUSER)) return;

    LOG_TRACE(GC_TPC_FUSER, "printOverlapToCluster");

    // print lists
    for (const auto& p : currNodeClusterStruct.tensorsInPersistMap)
    {
        LOG_TRACE(GC_TPC_FUSER, "printOverlapToCluster - tensorId {}, isInput {}", p.first, p.second.isInput);
    }

    for (const auto& pIn : currNodeClusterStruct.persistAddrRangeInTensorsSet)
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "printOverlapToCluster inputs - secId {}, start {}, end {}",
                  pIn.first,
                  pIn.second.start(),
                  pIn.second.end());
    }
    for (const auto& pOut : currNodeClusterStruct.persistAddrRangeOutTensorsSet)
    {
        LOG_TRACE(GC_TPC_FUSER,
                  "printOverlapToCluster outputs - secId {}, start {}, end {}",
                  pOut.first,
                  pOut.second.start(),
                  pOut.second.end());
    }
}

void TPCClusterConstructor::updateclustersOverlapMap(int                               clusterId,
                                                     ClusterPersistentAddrRangeStruct& currNodeClusterStruct)
{
    LOG_TRACE(GC_TPC_FUSER, "update overlap struct of clusterId {}", clusterId);
    m_clustersPersistentAddrRangeMap[clusterId].tensorsInPersistMap = currNodeClusterStruct.tensorsInPersistMap;
    m_clustersPersistentAddrRangeMap[clusterId].persistAddrRangeInTensorsSet =
        currNodeClusterStruct.persistAddrRangeInTensorsSet;
    m_clustersPersistentAddrRangeMap[clusterId].persistAddrRangeOutTensorsSet =
        currNodeClusterStruct.persistAddrRangeOutTensorsSet;
    m_clustersPersistentAddrRangeMap[clusterId].overlapNodes = currNodeClusterStruct.overlapNodes;
}

NodeVector TPCClusterConstructor::handlePotentialCyclesInClusters()
{
    // Check for cycles in clusters that contain multi consumers
    // Re-cluster all nodes for each cluster that contains cycles
    // this time without allowing multi consumers
    std::set<unsigned> clusterIdsToDelete;
    auto&              clusterIdMultiConsNumberDB = m_numberOfMultiConsumersInClusterMap;

    std::list<std::pair<NodePtr, NodeList>> listFuseNodeNodesList;
    for (auto const& clusterToMultiConsumersItr : clusterIdMultiConsNumberDB)
    {
        // TODO SW-49687 - Support more than 1 multi consumer in cluster
        int  clusterId = clusterToMultiConsumersItr.first;
        auto pCluster  = m_clustersMap.find(clusterId);

        LOG_TRACE(GC_TPC_FUSER, "Search for cycles in cluster ID {} ", clusterId);

        HB_ASSERT(pCluster != m_clustersMap.end(), "Cluster ID not found {}!", clusterId);

        NodeList nodesToFuse;
        for (auto node : pCluster->second)
        {
            LOG_TRACE(GC_TPC_FUSER, "node {} ", node->getNodeName());
            nodesToFuse.push_back(node);
        }

        // Do not search for cycles if there is only 1 node in cluster
        if (nodesToFuse.size() < 2) continue;

        NodeSet nodesSet(nodesToFuse.begin(), nodesToFuse.end());
        NodePtr fusedNode;
        if (GraphEditor::isLoopDueFusionInUpdateGraph(m_g, nodesSet, fusedNode, true))
        {
            LOG_TRACE(GC_TPC_FUSER, "In cluster ID {} has cycles", clusterId);
            clusterIdsToDelete.insert(clusterId);
        }
        else
        {
            // Add to list for restore the graph
            listFuseNodeNodesList.push_back(std::make_pair(fusedNode, nodesToFuse));
        }
    }

    // restore old graph
    for (auto currentPair : listFuseNodeNodesList)
    {
        auto status = GraphEditor::replaceNodes(m_g, {currentPair.first}, currentPair.second);
        HB_ASSERT(status == REPLACE_NODE_SUCCESS,
                  "failed to replace fused test node with the original nodes in the graph");
    }

    NodeVector nodesToDel;
    // Reset all clusters that have cycles
    for (auto clusterId : clusterIdsToDelete)
    {
        auto pCluster = m_clustersMap.find(clusterId);
        HB_ASSERT(pCluster != m_clustersMap.end(), "Cluster ID not found {}!", clusterId);

        for (const NodePtr& node : pCluster->second)
        {
            LOG_TRACE(GC_TPC_FUSER, "Delete from cluster {} TPC node: {}", clusterId, node->getNodeName());

            initClusterableNode(node);
            // Create a list of nodes to delete from current clusters
            // Will be used later also for re-cluster all nodes
            nodesToDel.push_back(node);
        }

        m_clustersMap[clusterId].clear();

        // clear the cluster persistent map
        m_clustersPersistentAddrRangeMap[clusterId].tensorsInPersistMap.clear();
        m_clustersPersistentAddrRangeMap[clusterId].persistAddrRangeOutTensorsSet.clear();
        m_clustersPersistentAddrRangeMap[clusterId].persistAddrRangeInTensorsSet.clear();

        m_clustersPersistentAddrRangeMap.erase(clusterId);
        m_clustersMap.erase(clusterId);

        clusterIdMultiConsNumberDB.erase(clusterId);
    }

    // TODO SW-49687 - Support more than 1 multi consumer in cluster

    // Re-cluster all nodes that deleted from the clusters
    for (const auto& tpcNode : nodesToDel)
    {
        if (m_nodeClustered[tpcNode->getId()]) continue;
        m_nodeClustered[tpcNode->getId()] = true;
        LOG_INFO(GC_TPC_FUSER, "Re-cluster TPC node: {}", tpcNode->getNodeName());
        createCluster(tpcNode, false);
    }
    return nodesToDel;
}

void TPCClusterConstructor::updateNodeCtrlDepDB(unsigned clusterId, NodePtr node)
{
    // Update both m_blockingAndBlockedNodes and m_ctrlDepNodes DBs for {node} in its cluster {clusterId}

    bool isCtrlDepSetInNode = (m_g.getBlockedNodes(node).size()) || (m_g.getBlockingNodes(node).size());

    LOG_TRACE(GC, " node {}  cluster {} isCtrlDepSetInNode {}", node->getNodeName(), clusterId, isCtrlDepSetInNode);

    if (isCtrlDepSetInNode && m_ctrlDepNodes[clusterId].insert(node).second)
    {
        // Update cluster blocking and blocked nodes

        NodeSet nodeBlockedSet(m_g.getBlockedNodes(node));
        NodeSet nodeBlockingSet(m_g.getBlockingNodes(node));

        m_blockingAndBlockedNodes[clusterId].insert(nodeBlockedSet.begin(), nodeBlockedSet.end());
        m_blockingAndBlockedNodes[clusterId].insert(nodeBlockingSet.begin(), nodeBlockingSet.end());

        LOG_TRACE(GC,
                  " node {} has ctrl dependency and belong to cluster {} updating block list to size {}",
                  node->getNodeName(),
                  clusterId,
                  m_blockingAndBlockedNodes[clusterId].size());
    }
}

void TPCClusterConstructor::updateClusterCtrlDepDB(unsigned clusterId, unsigned oldClusterId, NodePtr nextNode)
{
    // nextNode (or its cluster {oldClusterId}) will be merged with {clusterId}
    // Move all {oldClusterId} CTRL DEP DB to the new cluster {clusterId}

    // First check if need to update structs
    if (m_isToUpdateCtrlDepDB || (m_g.isControlDependencyConfigured() && oldClusterId != clusterId &&
                                  m_ctrlDepNodes.find(oldClusterId) != m_ctrlDepNodes.end()))
    {
        // Insert nextNode to new clusterId ctrl DEP DB
        updateNodeCtrlDepDB(clusterId, nextNode);

        if (oldClusterId != clusterId && m_ctrlDepNodes.find(oldClusterId) != m_ctrlDepNodes.end())
        {
            // Move all CTRL DEP info from old cluster to new
            m_ctrlDepNodes[clusterId].insert(m_ctrlDepNodes[oldClusterId].begin(), m_ctrlDepNodes[oldClusterId].end());
            m_blockingAndBlockedNodes[clusterId].insert(m_blockingAndBlockedNodes[oldClusterId].begin(),
                                                        m_blockingAndBlockedNodes[oldClusterId].end());

            NodeVector     intersection;
            NodeComparator comp;

            std::set_intersection(m_ctrlDepNodes[clusterId].begin(),
                                  m_ctrlDepNodes[clusterId].end(),
                                  m_blockingAndBlockedNodes[clusterId].begin(),
                                  m_blockingAndBlockedNodes[clusterId].end(),
                                  std::back_inserter(intersection),
                                  comp);

            HB_ASSERT(intersection.size() == 0,
                      "ERROR while clustering nextNode {}, there is dependency within the fused nodes elements",
                      nextNode->getNodeName());

            // remove old clusterId from structs
            m_ctrlDepNodes.erase(oldClusterId);
            m_blockingAndBlockedNodes.erase(oldClusterId);

            LOG_TRACE(GC_TPC_FUSER, "{}: Remove old clusterID {} from CycleDetectionDB", HLLOG_FUNC, oldClusterId);
        }
    }

    m_isToUpdateCtrlDepDB = false;
}

void TPCClusterConstructor::updateClustersDBs(bool&                             isMultiConsumerNode,
                                              bool                              isMultiConsumerNextNode,
                                              unsigned                          clusterId,
                                              unsigned                          oldClusterId,
                                              unsigned                          oldClusterIdNextNode,
                                              NodePtr                           currentNode,
                                              NodePtr                           nextNode,
                                              ClusterPersistentAddrRangeStruct& currNodeClusterStruct)
{
    // update new cluster multi consumer table
    // action will take place if the current node or next node is multi consumer
    // and if needed to delete the old clusterId of currNode or nextNode
    updateCycleDetectionDB(isMultiConsumerNode, clusterId, oldClusterIdNextNode, currentNode);
    updateCycleDetectionDB(isMultiConsumerNextNode, clusterId, oldClusterId, nextNode);

    isMultiConsumerNode = false;

    updateClusterCtrlDepDB(clusterId, oldClusterIdNextNode, nextNode);
    updateClusterCtrlDepDB(clusterId, oldClusterId, nextNode);

    if (m_isToUpdateOverlapStruct)
    {
        updateclustersOverlapMap(clusterId, currNodeClusterStruct);
    }

    printOverlapToCluster(currNodeClusterStruct);
}

void TPCClusterConstructor::updateCycleDetectionDB(bool     isUpdate,
                                                   unsigned clusterId,
                                                   unsigned oldClusterId,
                                                   NodePtr  node)
{
    auto& clusterIdMultiConsNumberDB = m_numberOfMultiConsumersInClusterMap;

    LOG_TRACE(GC_TPC_FUSER,
              "{}:  node {} ID {} oldclusterId {} new clusterId {} multi consumers map, isUpdate {}",
              HLLOG_FUNC,
              node->getNodeName(),
              node->getId(),
              oldClusterId,
              clusterId,
              isUpdate);

    if ((oldClusterId != clusterId) &&
        (clusterIdMultiConsNumberDB.find(oldClusterId) != clusterIdMultiConsNumberDB.end()))
    {
        // There are multi consumers that belongs to old clusterId
        // first add them to the new clusterId and at end delete this oldClusterId
        // multi consumer DB
        auto oldClusterIdMultiConSet = clusterIdMultiConsNumberDB[oldClusterId];
        // node belongs to a cluster that has multiple consumers nodes
        // add all to the new cluster multi consumer set
        for (auto const& nodeId : oldClusterIdMultiConSet)
        {
            clusterIdMultiConsNumberDB[clusterId].insert(nodeId);
            LOG_TRACE(GC_TPC_FUSER,
                      "{}: Add multi consumer node {} ID {} from clusterId {} to clusterId {} multi consumers map",
                      HLLOG_FUNC,
                      node->getNodeName(),
                      node->getId(),
                      oldClusterId,
                      clusterId);
        }
    }

    if (isUpdate)
    {
        // add nodeId to it new cluster DB that has multiple consumers nodes
        if (clusterIdMultiConsNumberDB[clusterId].insert(node->getId()).second)  // insertion took place
        {
            LOG_TRACE(GC_TPC_FUSER,
                      "{}: Add multi consumer node {} ID {} to clusterId {} multi consumers map",
                      HLLOG_FUNC,
                      node->getNodeName(),
                      node->getId(),
                      clusterId);
        }

        // Check if current cluster already reach to max number of multi consumers
        if (clusterIdMultiConsNumberDB.find(clusterId) != clusterIdMultiConsNumberDB.end())
        {
            // Check if current cluster already reach to max number of multi consumers
            unsigned numOfMultiConsumersInCurrNode = 0;
            numOfMultiConsumersInCurrNode          = clusterIdMultiConsNumberDB[clusterId].size();
            HB_ASSERT(numOfMultiConsumersInCurrNode <= m_numMaxMultiConsumersInCluster,
                      "{}- clusterId {} has multi consumers more than allowed {}",
                      HLLOG_FUNC,
                      clusterId,
                      numOfMultiConsumersInCurrNode);
        }
    }

    if (oldClusterId != clusterId)
    {
        // remove pre clusterId from structs
        clusterIdMultiConsNumberDB.erase(oldClusterId);

        LOG_TRACE(GC_TPC_FUSER,
                  "{}: Remove old clusterID from CycleDetectionDB clusterId {}",
                  HLLOG_FUNC,
                  oldClusterId);
    }
}

bool TPCClusterConstructor::isNodeMultiConsumerOrOutputs(NodePtr node)
{
    // return true if node has multiple consumers or node has multiple outputs

    TensorPtr outputTensor   = node->getOutputs().front();
    bool      isMultiOutputs = (node->getOutputs().size() != 1);
    auto      consumers      = m_g.getTensorConsumers(outputTensor);
    bool      isMultiConsumer =
        (consumers.size() > 1 &&
         (std::adjacent_find(consumers.begin(), consumers.end(), std::not_equal_to<NodePtr>()) != consumers.end()));

    LOG_INFO(GC_TPC_FUSER,
             "{}: output tensor {} has more than 1 consumer ({}) or node {} has multi output ({}) ",
             HLLOG_FUNC,
             outputTensor->getName(),
             isMultiConsumer,
             node->getNodeName(),
             isMultiOutputs);

    return (isMultiConsumer || isMultiOutputs);
}

bool TPCClusterConstructor::handleMultiConsumers(NodePtr   currentNode,
                                                 NodePtr&  nextNode,
                                                 bool&     isMultiConsumerNode,
                                                 bool&     isMultiConsumerNextNode,
                                                 bool&     isFirstTime,
                                                 unsigned& clusterId)
{
    // Support multi consumers nodes in cluster
    // If the node has multi consumers tensor and multi consumer is not allowed now
    // return false
    // In addition check if it possible to union currentNode with it producer
    // and check if it possible to union currentNode with all its consumers

    auto clusterIdMultiConsNumberDB = m_numberOfMultiConsumersInClusterMap;
    clusterId                       = getNodeCluster(currentNode->getId());
    TensorPtr outputTensor          = currentNode->getOutputs().front();
    unsigned  numOfConsumers        = m_g.getNumberOfTensorConsumers(outputTensor);

    // Check if current node has multiple consumers
    if (isNodeMultiConsumerOrOutputs(currentNode))
    {
        if (m_supportMultiConsumer == false)
        {
            LOG_INFO(GC_TPC_FUSER, "{}: Multi consumers / outsput is not supported, done clustering", HLLOG_FUNC);
            return false;
        }
        else
        {
            // multi consumers / outputs is supported
            LOG_INFO(GC_TPC_FUSER,
                     "{}: node {} is multi consumer, Multi consumers / outputs is supported",
                     HLLOG_FUNC,
                     currentNode->getNodeName());
            isMultiConsumerNode = true;
            updateCycleDetectionDB(isMultiConsumerNode, clusterId, clusterId, currentNode);
        }
    }
    else
    {
        HB_ASSERT(m_consumersList.empty(), "consumer list is not empty");
    }

    nextNode = nullptr;
    // Check if it possible to cluster currNode with its producer
    if (isFirstTime && m_supportMultiConsumer)
    {
        isFirstTime = false;
        // Check if it possible to union curr node with its one of it producers
        for (TensorPtr input : currentNode->getInputs())
        {
            if (input == nullptr) continue;
            NodePtr producer = m_g.getTensorProducer(input);

            if (producer)
            {
                ClusterPersistentAddrRangeStruct currNodeClusterStruct;
                if (!canBeClustered(m_g, currentNode, producer, currNodeClusterStruct))
                {
                    LOG_DEBUG(GC_TPC_FUSER, "{}: can't cluster with producer {}.", HLLOG_FUNC, producer->getNodeName());
                    continue;
                }

                // Can union current node with its producer
                nextNode                = producer;
                isMultiConsumerNextNode = isNodeMultiConsumerOrOutputs(nextNode);
                LOG_TRACE(GC_TPC_FUSER,
                          "{}: try union node {} with its multiple producer {}",
                          HLLOG_FUNC,
                          currentNode->getNodeName(),
                          producer->getNodeName());
                return true;
            }
        }
    }

    // check if it possible to union currentNode with it's consumers
    if (isMultiConsumerNode && (numOfConsumers > 1) && (m_consumersList.empty()) &&
        (m_currMultiConsumerNodeId != currentNode->getId()))
    {
        m_currMultiConsumerNodeId = currentNode->getId();
        m_consumersList           = m_g.getTensorConsumers(outputTensor);
        LOG_DEBUG(GC_TPC_FUSER,
                  "{}: init m_consumersList with multi consumer node consumers size {}",
                  HLLOG_FUNC,
                  m_consumersList.size());
        // Remove the first consumer due to trying first to union with all of the consumers
        // beside the first one. after union with the rest the algorithm will return partial DFS
        // from it.
        m_consumersList.pop_front();
    }

    if (!m_consumersList.empty())
    {
        // set nextNode to be the one of it consumers
        while ((!m_consumersList.empty()) || (nextNode != nullptr))
        {
            NodePtr consumer = m_consumersList.front();
            m_consumersList.pop_front();
            if (consumer == nullptr)
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "{}: try to cluster with null consumer list size {}",
                          HLLOG_FUNC,
                          m_consumersList.size());
                continue;
            }
            LOG_DEBUG(GC_TPC_FUSER,
                      "{}: try to cluster with consumer of {} list size {}",
                      HLLOG_FUNC,
                      consumer->getNodeName(),
                      m_consumersList.size());

            ClusterPersistentAddrRangeStruct currNodeClusterStruct;
            if (!canBeClustered(m_g, currentNode, consumer, currNodeClusterStruct))
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "{}: can't cluster with first consumer of second output {}.",
                          HLLOG_FUNC,
                          consumer->getNodeName());
                continue;
            }

            // Can union current node with one of its consumers
            nextNode                = consumer;
            isMultiConsumerNextNode = isNodeMultiConsumerOrOutputs(nextNode);
            LOG_DEBUG(GC_TPC_FUSER,
                      "{}: try to cluster with one of it consumers- {}",
                      HLLOG_FUNC,
                      consumer->getNodeName());

            return true;
        }
    }

    return true;
}

unsigned TPCClusterConstructor::getTotalNumOfMultiConsCurrNextNodes(NodePtr currentNode,
                                                                    NodePtr nextNode,
                                                                    bool    isMultiConsumerNode,
                                                                    bool&   isMultiConsumerNextNode)
{
    // Count the total number of multi consumers if both currNode and NextNode cluster will union
    unsigned totalNumOfMultiConsumers   = 0;
    auto     clusterIdMultiConsNumberDB = m_numberOfMultiConsumersInClusterMap;

    unsigned clusterId         = getNodeCluster(currentNode->getId());
    unsigned clusterIdNextNode = getNodeCluster(nextNode->getId());

    if (clusterId == clusterIdNextNode)
    {
        totalNumOfMultiConsumers = clusterIdMultiConsNumberDB[clusterId].size();
        LOG_TRACE(GC_TPC_FUSER,
                  "currNode {} and NextNode {} are in the same cluster cluster - {}, totalNumOfMultiConsumers {}",
                  currentNode->getNodeName(),
                  nextNode->getNodeName(),
                  clusterId,
                  totalNumOfMultiConsumers);

        return totalNumOfMultiConsumers;
    }

    if (clusterIdMultiConsNumberDB.find(clusterId) != clusterIdMultiConsNumberDB.end())
    {
        // currentNode have multiple consumers
        totalNumOfMultiConsumers += clusterIdMultiConsNumberDB[clusterId].size();
        LOG_TRACE(GC_TPC_FUSER,
                  "currNode {} is multi consumer in clusterId {}, has {} multi consumers nodes",
                  currentNode->getNodeName(),
                  clusterId,
                  totalNumOfMultiConsumers);
    }
    else if (isMultiConsumerNode)
    {
        totalNumOfMultiConsumers++;
        LOG_TRACE(GC_TPC_FUSER, "currNode {} is multi consumer", currentNode->getNodeName());
    }

    if (clusterIdMultiConsNumberDB.find(clusterIdNextNode) != clusterIdMultiConsNumberDB.end())
    {
        // nextNode has multiple consumers
        totalNumOfMultiConsumers += clusterIdMultiConsNumberDB[clusterIdNextNode].size();
        LOG_TRACE(GC_TPC_FUSER,
                  "Total number of multi consumers of NextNode name {} in cluster {} is- {}",
                  nextNode->getNodeName(),
                  clusterIdNextNode,
                  clusterIdMultiConsNumberDB[clusterIdNextNode].size());
    }
    else
    {
        // check if nextNode has multi consumers
        isMultiConsumerNextNode = isNodeMultiConsumerOrOutputs(nextNode);
        if (isMultiConsumerNextNode)
        {
            // nextNode has multi consumers
            totalNumOfMultiConsumers++;
            LOG_TRACE(GC_TPC_FUSER, "nextNode {} is multi consumer", nextNode->getNodeName());
        }
    }

    LOG_TRACE(GC_TPC_FUSER,
              "Total number of multi consumers if both currNode {} and NextNode {} cluster - {}",
              currentNode->getNodeName(),
              nextNode->getNodeName(),
              totalNumOfMultiConsumers);

    return totalNumOfMultiConsumers;
}

void TPCClusterConstructor::createCluster(NodePtr node, bool supportMultiConsmers)
{
    bool    doneClustering      = true;
    NodePtr currentNode         = node;
    bool    isFirstTime         = true;
    bool    isMultiConsumerNode = false;

    auto clusterIdMultiConsNumberDB = m_numberOfMultiConsumersInClusterMap;
    m_currMultiConsumerNodeId       = 0;
    m_consumersList.clear();

    m_supportMultiConsumer &= supportMultiConsmers;

    do
    {
        doneClustering            = true;
        m_isToUpdateOverlapStruct = false;
        // Loop Invariant: currentNode clustered
        // 1.) A join operation performed if the following conditions fulfilled:
        //           a.) Current node is not multi consumers node
        //           b.) Its output tensor has got a single consumer (unless multi-consumers supported)
        //           c.) The consumer is a tpc node, reshape or broadcast
        //           d.) There are no memory overlaps between nodes in the clusters (share same or partial memory for
        //           persistent tensors) e.) None of the consumers are in workspace sections,
        //               since they may have multiple consumers
        //           f.) There is no inner ctrl-dep between the nodes in the clusters

        LOG_TRACE(GC_TPC_FUSER,
                  "{}: curr node is {} id {}",
                  HLLOG_FUNC,
                  currentNode->getNodeName(),
                  currentNode->getId());

        if (currentNode->getNumOutputs() == 0)
        {
            LOG_TRACE(GC_TPC_FUSER, "node {} has no outputs. Done clustering.", currentNode->getNodeName());
            return;
        }

        // TODO [SW-53417] consider fusing workspace section tensors like persistent tensors.
        if (currentNode->getOutput(TENSOR_OFM)->isPartOfWorkspaceSection())
        {
            LOG_TRACE(GC_TPC_FUSER,
                      "TPCClusterConstructor: current node has workspace tensor {}. Done clustering.",
                      currentNode->getOutput(TENSOR_OFM)->getName());
            return;
        }

        unsigned clusterId = 0;

        TensorPtr outputTensor = currentNode->getOutputs().front();
        // checking if node's output tensor has consumers
        // if it has more than one consumer need to check that they are all the same node
        // otherwise need to stop clustering
        auto numConsumers = m_g.getNumberOfTensorConsumers(outputTensor);

        NodePtr nextNode                = nullptr;
        NodePtr nextNextNode            = nullptr;
        bool    isMultiConsumerNextNode = false;

        // Check if multi consumers is supported
        // If so - try union currentNode with it producers first
        if (!handleMultiConsumers(currentNode,
                                  nextNode,
                                  isMultiConsumerNode,
                                  isMultiConsumerNextNode,
                                  isFirstTime,
                                  clusterId))
        {
            // Can't union with nextNode, Done clustering
            return;
        }

        bool                             isNextNodeTheProducer = false;
        ClusterPersistentAddrRangeStruct currNodeClusterStruct;
        if (nextNode != nullptr)
        {
            // nextNode is the producer
            isNextNodeTheProducer = true;
            // set nextNextNode to currentNode so in the second iteration nexNode will be the first consumer
            nextNextNode = currentNode;
        }
        if ((numConsumers >= 1) && (nextNode == nullptr))
        {
            // set nextNode to be the first consumer of currentNode
            nextNode = m_g.getTensorConsumers(outputTensor).front();

            if (!canBeClustered(m_g, currentNode, nextNode, currNodeClusterStruct))
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "{}: can't cluster node {}. Done clustering.",
                          HLLOG_FUNC,
                          nextNode->getNodeName());
                return;
            }
        }
        else if (nextNode == nullptr)
        {
            LOG_TRACE(GC_TPC_FUSER,
                      "TPCClusterConstructor: consumers.size of node {} is 0. Done clustering.",
                      currentNode->getNodeName());
            return;
        }

        unsigned oldClusterId         = clusterId;
        unsigned oldClusterIdNextNode = getNodeCluster(nextNode->getId());
        // 2.) Performs the actual clusters union operation.
        //     Cluster(current node) = Cluster(current node) union Cluster(next node)
        m_unionFindContainer->m_unionFind.join(m_unionFindContainer->m_nodeId2untypedNode[currentNode->getId()],
                                               m_unionFindContainer->m_nodeId2untypedNode[nextNode->getId()]);

        clusterId                          = getNodeCluster(currentNode->getId());
        m_nodeClustered[nextNode->getId()] = true;

        LOG_DEBUG(GC_TPC_FUSER,
                  "TPCClusterConstructor: ClusterOf({}) union ClusterOf({}), clusterId {}",
                  currentNode->getNodeName(),
                  nextNode->getNodeName(),
                  clusterId);

        // update clusters DBs
        updateClustersDBs(isMultiConsumerNode,
                          isMultiConsumerNextNode,
                          clusterId,
                          oldClusterId,
                          oldClusterIdNextNode,
                          currentNode,
                          nextNode,
                          currNodeClusterStruct);

        // 3.) Move to the next node
        if (isNextNodeTheProducer)
        {
            if (nextNextNode == nullptr)
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "TPCClusterConstructor: consumers.size of node {} is 0. Done clustering.",
                          currentNode->getNodeName());
                return;
            }
            // Next node will be the first consumer of currentNode
            currentNode = nextNextNode;
            LOG_TRACE(GC_TPC_FUSER,
                      "TPCClusterConstructor: Next node will be the first consumer of currentNode is {} id {}",
                      currentNode->getNodeName(),
                      currentNode->getId());
        }
        else
        {
            currentNode = nextNode;
            LOG_TRACE(GC_TPC_FUSER,
                      "TPCClusterConstructor: Next node is {} id {}",
                      currentNode->getNodeName(),
                      currentNode->getId());
        }

        doneClustering = false;

        // We stop clustering due to one of the following reasons:
        // 1.) We are visiting a non-tpc node -> the cluster is fully constructed
        // 2.) There is more that single consumer to the tensor and multi consumers is not allowed -> the cluster is
        // fully constructed
        // 3.) If the node has more than one output -> the cluster is fully constructed
    } while (!doneClustering);
}

/*
 * Create clusters from the lists of nodes extracted from specific complex guid nodes.
 * Mark all the nodes as clustered.
 */
void TPCClusterConstructor::addComplexGuidExtractedClusters()
{
    auto& complexGuidClusters = m_g.getGraphAnnotation().complexGuidExtractedClusters;
    LOG_DEBUG(GC_TPC_FUSER,
              "Adding {} clusters of nodes extracted from complex guid nodes",
              complexGuidClusters.size());
    for (auto& cluster : complexGuidClusters)
    {
        // join all nodes of same cluster into single cluster in the underlying unionFind container.
        // we know all of these nodes are in same cluster in the graph, so we can join them to the same node (lastNode).
        LOG_DEBUG(GC_TPC_FUSER, "Adding cluster extracted from complex guid node Id {}", cluster.first);

        NodePtr lastNode   = cluster.second.back();
        int     lastNodeId = lastNode->getId();
        while (m_unionFindContainer->m_nodeId2untypedNode.find(lastNodeId) ==
               m_unionFindContainer->m_nodeId2untypedNode.end())
        {
            cluster.second.pop_back();
            lastNode   = cluster.second.back();
            lastNodeId = lastNode->getId();
        }

        UntypedNode                      lastUntypedNode  = m_unionFindContainer->m_nodeId2untypedNode.at(lastNodeId);
        int                              currentClusterId = getNodeCluster(lastNode->getId());
        ClusterPersistentAddrRangeStruct currentClusterPersistentStruct;
        updateClusterPersistentAddrRangeStruct(lastNode, currentClusterPersistentStruct, currentClusterId);

        // add each node to the cluster of lastNode
        for (auto it = cluster.second.begin(); it != cluster.second.end();)
        {
            NodePtr currentNode = *it;
            if (currentNode == lastNode) break;
            uint64_t currentNodeId      = currentNode->getId();
            auto     currentUntypedNode = m_unionFindContainer->m_nodeId2untypedNode.find(currentNodeId);
            if (currentUntypedNode != m_unionFindContainer->m_nodeId2untypedNode.end())
            {
                int currentNodeClusterId = getNodeCluster(currentNodeId);
                if (isOverlapToCluster(lastNode,
                                       currentNode,
                                       currentClusterPersistentStruct,
                                       currentClusterId,
                                       currentNodeClusterId))
                {
                    const std::string errStr =
                        fmt::format("Cluster from complex guid Id {} has overlapping persistent tensors",
                                    cluster.first);
                    LOG_ERR(GC_TPC_FUSER, "{}", errStr);
                    throw SynapseStatusException(errStr);
                }

                LOG_DEBUG(GC_TPC_FUSER, "Joining  node id {} with node id {}", currentNodeId, lastNodeId);
                m_unionFindContainer->m_unionFind.join(lastUntypedNode, currentUntypedNode->second);
                // mark current node as clustered so it won't be joined with other clusters
                m_nodeClustered[currentNodeId] = true;
                // add current node to cycle detection db
                m_numberOfMultiConsumersInClusterMap[currentClusterId].insert(currentNodeId);
                it++;
            }
            else
            {
                it = cluster.second.erase(it);
            }
        }
        if (!ClusteringUtils::isClusterConnected(cluster.second))
        {
            const std::string errStr = fmt::format("Cluster from complex guid Id {} is not connected", cluster.first);
            LOG_ERR(GC_TPC_FUSER, "{}", errStr);
            throw SynapseStatusException(errStr);
        }
        m_nodeClustered[lastNodeId] = true;
        m_numberOfMultiConsumersInClusterMap[currentClusterId].insert(lastNodeId);
        m_complexGuidClusterId.insert(currentClusterId);
        LOG_DEBUG(GC_TPC_FUSER,
                  "Finished clustering {} nodes that were extracted from complex guid node Id {}."
                  "Cluster id {}.",
                  cluster.second.size(),
                  cluster.first,
                  currentClusterId);
    }
}

void TPCClusterConstructor::computeClusters()
{
    if (GCFG_COMPLEX_GUID_CLUSTERING.value())
    {
        addComplexGuidExtractedClusters();
    }
    // Init clusterCycleDetectionProp struct
    auto& clusterIdMultiConsNumberDB = m_numberOfMultiConsumersInClusterMap;
    clusterIdMultiConsNumberDB.clear();

    // 1.) Run over TPC nodes in execution order - this order takes us closer to the optimal cluster
    for (const auto& tpcNode : m_nodes)
    {
        if (m_nodeClustered[tpcNode->getId()])
        {
            continue;
        }

        // The below line has no affect on the functionality since there is no way we will visit it again
        // this may be deduced from the iteration order.
        m_nodeClustered[tpcNode->getId()] = true;

        LOG_DEBUG(GC_TPC_FUSER, "TPCClusterConstructor: Processing TPC node: {}", tpcNode->getNodeName());

        // Construct a cluster - this cluster is constructed in internal representation
        createCluster(tpcNode, true);
        m_isToUpdateCtrlDepDB = false;
    }

    // 2.) Construct the output clusters
    for (const auto& tpcNode : m_nodes)
    {
        // Map cluster ID to a set of TpcNode in this cluster
        int clusterId = getNodeCluster(tpcNode->getId());
        m_clustersMap[clusterId].insert(tpcNode);
    }

    if (LOG_LEVEL_AT_LEAST_TRACE(GC_TPC_FUSER))
    {
        // print the amount of multi consumers in this cluster
        for (const auto& p : clusterIdMultiConsNumberDB)
        {
            int clusterId = p.first;
            LOG_TRACE(GC_TPC_FUSER,
                      "clusterId {}: numberOfMultiConsumers {}",
                      clusterId,
                      clusterIdMultiConsNumberDB[clusterId].size());
        }

        LOG_TRACE(GC_TPC_FUSER, "Clusters after first clustering:");
        for (const auto& p : m_clustersMap)
        {
            for (const NodePtr& n : p.second)
            {
                LOG_TRACE(GC_TPC_FUSER, "ClusterId {}: nodeName {}", p.first, n->getNodeName());
            }
        }
    }

    // Re-cluster each cluster that contain cycle while not allowing multiple consumers
    NodeVector reClusteredNodes = handlePotentialCyclesInClusters();

    // Clear overlap map
    for (auto it = m_clustersPersistentAddrRangeMap.begin(); it != m_clustersPersistentAddrRangeMap.end(); it++)
    {
        it->second.persistAddrRangeOutTensorsSet.clear();
        it->second.persistAddrRangeInTensorsSet.clear();
        it->second.tensorsInPersistMap.clear();
    }
    m_clustersPersistentAddrRangeMap.clear();

    for (const auto& tpcNode : reClusteredNodes)
    {
        // Map cluster ID to a set of TpcNode in this cluster
        int clusterId = getNodeCluster(tpcNode->getId());
        m_clustersMap[clusterId].insert(tpcNode);
    }
    LOG_INFO(GC_TPC_FUSER, "Created {} clusters", m_clustersMap.size());
    LOG_INFO(GC_TPC_FUSER, "Final clusters after last clustering:");
    for (const auto& p : m_clustersMap)
    {
        for (const NodePtr& n : p.second)
        {
            LOG_INFO(GC_TPC_FUSER, "ClusterId {}: nodeName {}", p.first, n->getNodeName());
        }
    }

    // Post-condition: all nodes clustered
    for (const auto& tpcNode : m_nodes)
    {
        HB_ASSERT(m_nodeClustered[tpcNode->getId()],
                  "TPCClusterConstructor: un-clustered node {} {}!",
                  tpcNode->getNodeName(),
                  tpcNode->getId());
    }

    for (auto it = clusterIdMultiConsNumberDB.begin(); it != clusterIdMultiConsNumberDB.end(); it++)
    {
        it->second.clear();
    }
    clusterIdMultiConsNumberDB.clear();
}

std::unordered_set<NodePtr> TPCClusterConstructor::popNextCluster()
{
    std::unordered_set<NodePtr> tpcCluster;

    if (!m_clustersMap.empty())
    {
        // Pop a cluster
        tpcCluster = m_clustersMap.begin()->second;
        m_clustersMap.erase(m_clustersMap.begin());
    }

    return tpcCluster;
}

const std::unordered_map<int, std::unordered_set<NodePtr>>& TPCClusterConstructor::getClusters()
{
    return m_clustersMap;
}

int TPCClusterConstructor::getNumOfClusters() const
{
    int numOfClusters = 0;
    for (auto it = m_clustersMap.begin(); it != m_clustersMap.end(); it++)
    {
        if (it->second.size() > 0)
        {
            // count the number of clusters that have nodes
            numOfClusters++;
        }
    }
    return numOfClusters;
}

int TPCClusterConstructor::getNodeCluster(unsigned nodeId) const
{
    return m_unionFindContainer->m_unionFind.find(m_unionFindContainer->m_nodeId2untypedNode[nodeId]);
}

/********************************************************************************************************************/
/************************************************GCTPCFuserWrapper***************************************************/
/********************************************************************************************************************/

GCTPCFuserWrapper::GCTPCFuserWrapper(const std::unordered_set<pNode>& clusterNodes,
                                     HabanaGraph&                     g,
                                     gcapi::pfnFuseGraphV4            graphFuncPtr,
                                     gcapi::pfnGetFusedNodePreGraphV4 getPreGraphFuncPtr)
: m_g(g), m_FuserGraphFunc(graphFuncPtr), m_getPreGraphFunc(getPreGraphFuncPtr)
{
    // For each node in cluster update FuserNode params struct before sending to TPC fuser
    for (const auto& node : clusterNodes)
    {
        FuserNodePtrV4 fn = std::make_shared<FuserNodeTypeV4>();
        createFuserNode(g, fn, node);

        // Update the control edges info
        fn->controlEdgesToNode.clear();
        for (auto& b : g.getBlockingNodes(node))
        {
            fn->controlEdgesToNode.insert(b->getId());
        }

        m_clusterNodesMap[node->getId()] = node;
        LOG_TRACE(GC_TPC_FUSER, "GCTPCFuserWrapper: tpc node guid: {}", fn->guid);
        m_fuserGraphNodes[node->getId()] = fn;
    }
}

GCTPCFuserWrapper::~GCTPCFuserWrapper() {}

FuserTensorPtrV4 GCTPCFuserWrapper::gcTensor2FuserTensor(const HabanaGraph& g, TensorPtr gcTensor)
{
    auto search = m_fuserGraphTensors.find(gcTensor->getId());

    if (search != m_fuserGraphTensors.end())
    {
        // Tensor exists
        return search->second;
    }
    else
    {
        // Need to translate the gc-tensor to fuser
        FuserTensorPtrV4 fuserTensor = std::make_shared<FuserTensorTypeV4>();
        createFuserTensor(fuserTensor, gcTensor);
        m_fuserGraphTensors[gcTensor->getId()] = fuserTensor;
        return fuserTensor;
    }
}

// Continue updating the fusing structure and send it tpc-fuser:
// Iterates over all nodes and tensors and translates each one of them to tpc-fuser language
// Classify each tensor according to the following:
//     1.) Internal tensor - definition at .h file
//     2.) External tensor - definition at .h file
void GCTPCFuserWrapper::constructFuserGraph(const HabanaGraph& g)
{
    LOG_DEBUG(GC_TPC_FUSER, "GCTPCFuserWrapper: Constructing fuser Graph.");

    // Get deviceId according to device type (Goya/Guadi)
    m_fuserGraph.deviceId = newGlueCodeToOldDeviceId(deviceTypeToDeviceID(g.getDeviceType()));

    // We have no differentiation between training fwd/bwd, but do want to give inference indication if relevant
    m_fuserGraph.kernelType = g.getInferenceMode() ? gcapi::KERNEL_TYPE_INFERENCE : gcapi::KERNEL_TYPE_TRAINING_FWD;

    // 1.) Run over the nodes of the cluster
    for (const NodePtr& node : g.getTopoSortedNodes())
    {
        auto it = m_fuserGraphNodes.find(node->getId());
        if (it == m_fuserGraphNodes.end() || !it->second) continue;
        FuserNodePtrV4 fuserNode = it->second;

        // 2.) Reduce the input edges and classify the input tensor
        reduceAndClassifyInputs(g, node, fuserNode);

        // 3.) Reduce the output edges and classify the output tensor
        reduceAndClassifyOutputs(g, node, fuserNode);

        // 4.) Add the fuser node to the fuser graph
        m_fuserGraph.nodes.push_back(fuserNode);
    }
}

// Input: node - is the node from Habana-graph that the caller wishes to reduce
//        fuserNode - input node representation in tpcFuser language
// Functionality: Reduce output tensors and classify them
void GCTPCFuserWrapper::reduceAndClassifyOutputs(const HabanaGraph& g, const NodePtr& node, FuserNodePtrV4& fuserNode)
{
    // 1. Iterate over the outputs
    for (const TensorPtr& outTensor : node->getOutputs())
    {
        FuserEdgeTypeV4 fuserOutEdge;
        fuserOutEdge.tensor     = std::shared_ptr<FuserTensorTypeV4>(nullptr);
        fuserOutEdge.targetNode = std::weak_ptr<FuserNodeTypeV4>();
        if (outTensor == nullptr)
        {
            // handle also null (optional) tensors
            fuserNode->outputEdges.push_back(fuserOutEdge);
            continue;
        }

        bool externalTensor = false;

        fuserOutEdge.tensor = gcTensor2FuserTensor(g, outTensor);

        // 1.1) Graph output tensor is an external tensor
        if (m_g.isOutputTensor(outTensor))
        {
            LOG_DEBUG(GC_TPC_FUSER,
                      "{}: node:{} --> output-tensor:[{}]",
                      HLLOG_FUNC,
                      node->getNodeName(),
                      outTensor->getId());

            // Classification: This is an external edge since it is graph output
            externalTensor = true;
        }
        else
        {
            // Iterate over the consumers and reduce: (outTensor--->consumer) edge
            const NodeList& consumers = m_g.getTensorConsumers(outTensor);

            for (auto const& consumer : consumers)
            {
                if (!isInCluster(m_clusterNodesMap, consumer))
                {
                    externalTensor = true;
                }
                else
                {
                    fuserOutEdge.targetNode = m_fuserGraphNodes[consumer->getId()];
                    fuserNode->outputEdges.push_back(fuserOutEdge);
                }

                LOG_DEBUG(GC_TPC_FUSER,
                          "{}: node:{} --> tensor:[{}]",
                          HLLOG_FUNC,
                          node->getNodeName(),
                          outTensor->getId());
            }

            if (consumers.size() == 0)
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "{}: node:{} --> output-tensor:[{}] has no consumers",
                          HLLOG_FUNC,
                          node->getNodeName(),
                          outTensor->getId());

                // Classification: This is an external edge since it has no consumers
                externalTensor = true;
            }
        }

        if (externalTensor)
        {
            fuserOutEdge.targetNode.reset();
            fuserNode->outputEdges.push_back(fuserOutEdge);
        }

        // 1.2) Classification
        if (externalTensor || m_g.isPersistentTensor(outTensor))
        {
            m_externalTensors[outTensor->getId()] = outTensor;
        }
        else
        {
            m_internalTensors[outTensor->getId()] = outTensor;
        }
    }
}

// Input: node - is the node from Habana-graph that the caller wishes to reduce
//        fuserNode - input node representation in tpcFuser language
// Functionality: Reduce input tensors and classify them
void GCTPCFuserWrapper::reduceAndClassifyInputs(const HabanaGraph& g, const NodePtr& node, FuserNodePtrV4& fuserNode)
{
    // 1.) Run over input-tensors of the node
    unsigned tensorIndex       = 0;
    auto     inputPermutations = node->getNodeAnnotation().inputPermutations;
    bool     gotPermutations   = !inputPermutations.empty();
    for (TensorPtr inTensor : node->getInputs())
    {
        FuserEdgeTypeV4 fuserInEdge;
        fuserInEdge.tensor     = std::shared_ptr<FuserTensorTypeV4>(nullptr);
        fuserInEdge.targetNode = std::weak_ptr<FuserNodeTypeV4>();
        if (inTensor == nullptr)
        {
            // handle also null (optional) tensors
            fuserNode->inputEdges.push_back(fuserInEdge);
            tensorIndex++;
            continue;
        }
        fuserInEdge.tensor = gcTensor2FuserTensor(g, inTensor);
        /*
         * set permutations.
         * don't set permutations for aux and output shape tensors which are inputs.
         * This is same as glue code Tensor creation logic in tpc node.
         */
        if (gotPermutations && !inTensor->isTensorAuxOrShapeOutput())
        {
            setCommonTensorPermutations(fuserInEdge.tensor, inputPermutations[tensorIndex++]);
        }
        NodePtr producer = m_g.getTensorProducer(inTensor);

        // 2.) Classification: There is no producer or producer is not in the cluster
        if (!producer || !isInCluster(m_clusterNodesMap, producer) || m_g.isInputTensor(inTensor))
        {
            // Leave the edge's targetNode empty and add it to the fuserNode inputs
            fuserNode->inputEdges.push_back(fuserInEdge);

            m_externalTensors[inTensor->getId()] = inTensor;
        }
        else
        {
            // Set the edge's targetNode to the producer and add it to the fuserNode inputs
            fuserInEdge.targetNode = m_fuserGraphNodes[producer->getId()];
            fuserNode->inputEdges.push_back(fuserInEdge);

            // 3.) Classification: Tensor is persistent
            if (m_g.isPersistentTensor(inTensor))
            {
                m_externalTensors[inTensor->getId()] = inTensor;
            }
            // 4.) Classification: Producer is in the cluster
            else
            {
                m_internalTensors[inTensor->getId()] = inTensor;
            }
        }

        LOG_TRACE(GC_TPC_FUSER, "GCTPCFuserWrapper: tensor:[{}] --> node:({})", inTensor->getId(), node->getNodeName());
    }
}

bool GCTPCFuserWrapper::optimizeCluster(const HabanaGraph& g)
{
    CHECK_RET_FALSE(m_FuserGraphFunc != nullptr, "TPC Fuser object is not loaded correctly");

    if (m_clusterNodesMap.size() <= 1)
    {
        LOG_DEBUG(GC_TPC_FUSER, "GCTPCFuserWrapper: Cluster size isn't larger than 1, not optimizing");
        return false;
    }
    if (!hasTPCNode())
    {
        LOG_WARN(GC_TPC_FUSER, "GCTPCFuserWrapper: Cluster has no TPC nodes, not optimizing");
        return false;
    }

    // 1.) Continue updating the data to send TPC fuser (tensors part)
    //     and construct Fuser Graph
    constructFuserGraph(g);

    // 2.) Call tpc fuser
    //  Post-condition: The inputs and the outputs of the cluster remains unmodified.
    LOG_DEBUG(GC_TPC_FUSER, "GCTPCFuserWrapper: Calling to the tpc fuser library.");

    // Send current cluster (fused Graph) to TPC fuser
    gcapi::FuserRetVal_t retVal = m_FuserGraphFunc(&m_fuserGraph, &m_optimizedFuserGraph, false);

    if (retVal == gcapi::FUSER_FAILED)
    {
        LOG_WARN(GC_TPC_FUSER, "GCTPCFuserWrapper: TPC fuser operation failed. return value: {}.", retVal);
        return false;
    }

    LOG_DEBUG(GC_TPC_FUSER,
              "GCTPCFuserWrapper: Successfully reduced tpc nodes cluster to fuserGraph. Fuser return value {}.",
              retVal);

    printFuserGraph(m_optimizedFuserGraph);

    return true;
}

bool GCTPCFuserWrapper::mapNewTensorSection(gcapi::CommonSection& section)
{
    // get fuser section info
    FuserSectionId fuserSectionId = section.id;
    if (m_fuserSectionIdToSectionType.count(fuserSectionId) == 0)  // if fuser section id not mapped
    {
        LOG_TRACE(GC_TPC_FUSER, "Mapping info of fuser section with id {}", fuserSectionId);
        CHECK_RET_FALSE(section.type != gcapi::SECTION_PERSISTENT,
                        "New persistent sections created by external lib are not allowed");
        // map fuser section id to section type and new gc section id
        // we need this mapping since there can be several tensors with same section id,
        // so we need to store this section id somewhere.
        gcapi::FuserSectionType sectionType           = section.type;
        GCSectionId             gcSectionId           = 0;
        m_fuserSectionIdToSectionType[fuserSectionId] = sectionType;
        if (sectionType == gcapi::SECTION_RMW)  // RMW
        {
            gcSectionId = m_g.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);
        }
        else  // workspace
        {
            gcSectionId = MEMORY_ID_RESERVED_FOR_WORKSPACE;
        }
        m_fuserSectionIdToGCSectionId[fuserSectionId] = gcSectionId;
        LOG_TRACE(GC_TPC_FUSER,
                  "fuser section with id {} is {} and mapped to GC section id {}",
                  fuserSectionId,
                  sectionType == gcapi::SECTION_PERSISTENT ? "persistent" : "workspace or RMW",
                  gcSectionId);
    }
    return true;
}

void GCTPCFuserWrapper::setNewGCTensorSectionInfo(TensorPtr& gcTensor, gcapi::CommonSection& fuserSection)
{
    // assign GC tensor with section info according to section id mapping
    // it's possible to have a mapped persistent section - if one of original tensors was assigned to it
    FuserSectionId fuserSectionId = fuserSection.id;
    if (m_fuserSectionIdToSectionType[fuserSectionId] == gcapi::SECTION_PERSISTENT)
    {
        gcTensor->setMemoryDescriptor(synMemoryDescriptor(true));
        gcTensor->setMemorySectionID(m_fuserSectionIdToGCSectionId[fuserSectionId]);
        gcTensor->setMemorySectionOffset(fuserSection.offset);
        gcTensor->setProp(synTensorPropSection);
    }
    else if (m_fuserSectionIdToSectionType[fuserSectionId] == gcapi::SECTION_RMW)
    {
        gcTensor->setTensorInSram();
        auto& nonPersistentSectionInfo = gcTensor->getTensorAnnotation().nonPersistentSectionInfo;
        nonPersistentSectionInfo.sectionId.set(m_fuserSectionIdToGCSectionId[fuserSectionId]);
        nonPersistentSectionInfo.offsetFromBase.set(fuserSection.offset);
        gcTensor->setProp(synTensorPropSection);
    }
    // TODO SW-43106 consider handling multiple workspace sections when it's added
}

void GCTPCFuserWrapper::printFuserGraph(const FuserGraphTypeV4& fuserGraph) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC_TPC_FUSER)) return;

    for (const FuserNodePtrV4& node : fuserGraph.nodes)
    {
        for (const FuserEdgeTypeV4& edge : node->inputEdges)
        {
            auto targetNode = edge.targetNode.lock();
            LOG_DEBUG(GC_TPC_FUSER,
                      "---{}---->{}",
                      edge.tensor ? std::to_string(edge.tensor->uniqueIdentifier) : "null",
                      targetNode.get() ? targetNode.get()->guid : "null");
        }

        for (const FuserEdgeTypeV4& edge : node->outputEdges)
        {
            auto targetNode = edge.targetNode.lock();
            LOG_DEBUG(GC_TPC_FUSER,
                      "---{}---->{}",
                      edge.tensor ? std::to_string(edge.tensor->uniqueIdentifier) : "null",
                      targetNode.get() ? targetNode.get()->guid : "null");
        }
    }
}

bool GCTPCFuserWrapper::isInCluster(const ClusterNodeMap& clusterMap, const NodePtr& node)
{
    return clusterMap.find(node->getId()) != clusterMap.end();
}

gcapi::GlueCodeReturn_t GCTPCFuserWrapper::getNodePreGraph(const FuserNodePtrV4& node, FuserGraphTypeV4& preGraph) const
{
    // the API requires a pointer to a pointer, this is not necessary
    // TODO change to a one-level pointer at both sides
    FuserGraphTypeV4* pPreGraph = &preGraph;
    return m_getPreGraphFunc(node.get(), &pPreGraph);
}

/********************************************************************************************************************/
/**************************************************TpcFuser**********************************************************/
/********************************************************************************************************************/

TensorPtr findGcTensorByFuserTensorID(const std::shared_ptr<GCTPCFuserWrapper>& fuser, unsigned uniqueID)
{
    const std::unordered_map<unsigned, TensorPtr>& externalTensors = fuser->getExternalTensors();
    const std::unordered_map<unsigned, TensorPtr>& internalTensors = fuser->getInternalTensors();
    const std::unordered_map<unsigned, TensorPtr>& newGCTensors    = fuser->getNewGCTensors();

    auto search = externalTensors.find(uniqueID);
    if (search != externalTensors.end())
    {
        return search->second;
    }

    search = internalTensors.find(uniqueID);
    if (search != internalTensors.end())
    {
        return search->second;
    }

    auto searchNew = newGCTensors.find(uniqueID);
    if (searchNew != newGCTensors.end())
    {
        return searchNew->second;
    }

    LOG_DEBUG(GC_TPC_FUSER, "tpcFuser: GC tensor not found! uniqueId: {}", uniqueID);
    return nullptr;
}

/*
 * Validate correctness of  optimized graph
 * 1. Nodes validations :
 *      1.1 Validate RMW tensors in node
 * 2. Tensors (Optimized graph inputs and outputs) validation
 *      2.1 Validate all original inputs and outputs exist
 *      2.2 Validate new inputs and outputs are valid
 */
bool validateOptimizedGraph(const std::shared_ptr<GCTPCFuserWrapper>& fuser)
{
    const FuserGraphTypeV4 optGraph = fuser->getOptimizedFuserGraph();
    CommonIRGraphValidator<FuserGraphTypeV4, FuserTensorTypeV4, FuserEdgeTypeV4> validator(&optGraph,
                                                                                           fuser->getExternalTensors());
    validator.prepareForValidation();
    return validator.validateNodes() && validator.validateGraphInputsAndOutputs();
}

bool getFusedNodeEdgeAsGcTensor(const FuserEdgeTypeV4&                    edge,
                                const std::shared_ptr<GCTPCFuserWrapper>& fuser,
                                TensorVector&                             tensorVector)
{
    if (edge.tensor == nullptr)
    {
        // get also null (optional) tensors
        tensorVector.push_back(nullptr);
        return true;
    }
    TensorPtr gcTensor = findGcTensorByFuserTensorID(fuser, edge.tensor->uniqueIdentifier);
    if (gcTensor == nullptr)
    {
        // Case of new tensor
        LOG_TRACE(GC_TPC_FUSER, "Creating gc tensor");
        fuser->mapNewTensorSection(edge.tensor->section);
        gcTensor = std::make_shared<Tensor>();
        createGCTensor(gcTensor, edge.tensor);
        fuser->setNewGCTensorSectionInfo(gcTensor, edge.tensor->section);

        if (!gcTensor->isPropsValid())  //  validate tensors properties
        {
            LOG_ERR(GC_TPC_FUSER, "Tensor {} has non valid props", gcTensor->getName());
            return false;
        }
        fuser->getNewGCTensors().insert({edge.tensor->uniqueIdentifier, gcTensor});
        LOG_TRACE(GC_TPC_FUSER, "Finished creating gc tensor from edge");
    }
    tensorVector.push_back(gcTensor);
    return true;
}

TensorVector getFusedNodeInputEdgesAsGcTensorList(const FuserNodeTypeV4&                    node,
                                                  const std::shared_ptr<GCTPCFuserWrapper>& fuser)
{
    TensorVector tensorVector;
    for (const FuserEdgeTypeV4& edge : node.inputEdges)
    {
        if (!getFusedNodeEdgeAsGcTensor(edge, fuser, tensorVector))
        {
            tensorVector.clear();
            return tensorVector;
        }
    }
    return tensorVector;
}

TensorVector getFusedNodeOutputEdgesAsGcTensorList(const FuserNodeTypeV4&                    node,
                                                   const std::shared_ptr<GCTPCFuserWrapper>& fuser)
{
    TensorVector tensorVector;
    if (node.outputEdges.empty())
    {
        return tensorVector;
    }
    std::set<unsigned> handledTensorIds;

    for (const FuserEdgeTypeV4& edge : node.outputEdges)
    {
        if (edge.tensor == nullptr) continue;
        unsigned tensorId = edge.tensor->uniqueIdentifier;
        if (handledTensorIds.count(tensorId) == 0)
        {
            handledTensorIds.insert(tensorId);
            if (!getFusedNodeEdgeAsGcTensor(edge, fuser, tensorVector))
            {
                tensorVector.clear();
                return tensorVector;
            }
        }
    }
    return tensorVector;
}

NodePtr createFusedTpcNode(int                 fusedClustersCounter,
                           int                 fusedNodesCounter,
                           const char*         guid,
                           const TensorVector& inputs,
                           const TensorVector& outputs,
                           UserParams          params,
                           unsigned            paramsSize)
{
    std::stringstream nodeNameStream;
    nodeNameStream << "fusedTPCNode_" << std::to_string(fusedClustersCounter) << "_"
                   << std::to_string(fusedNodesCounter);

    std::string nodeName = nodeNameStream.str().substr(0, tpc_lib_api::MAX_NODE_NAME - 1);

    LOG_DEBUG(GC_TPC_FUSER,
              "tpcFuser: Creating new fused node, guid: {}, name: {}, "
              "params size: {}",
              guid,
              nodeName,
              paramsSize);

    return NodeFactory::createNode(inputs, outputs, params, paramsSize, guid, nodeName);
}

static optimizedGraphStatus FusedNodeToMultiSifData(HabanaGraph& hg, const FuserNodePtrV4& fusedNode, NodePtr& newNode);

static optimizedGraphStatus FuserGraphToMultiSifData(HabanaGraph&            hg,
                                                     const FuserNodePtrV4&   fusedNode,
                                                     NodePtr&                newNode,
                                                     const FuserGraphTypeV4& preGraph);

void markAllowedForStitching(HabanaGraph& graph, const NodeList& nodesToFuse, const NodePtr& fusedNode)
{
    bool canStitch = true;
    for (auto n : nodesToFuse)
    {
        TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(n);
        canStitch &= tpcNode ? tpcNode->isAllowedForStitching(graph) : false;
    }

    if (canStitch)
    {
        TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(fusedNode);
        if (tpcNode)
        {
            LOG_DEBUG(GC_TPC_FUSER, "Setting fused kernel {} as allowed for stitching", fusedNode->getGUID());
            tpcNode->setAllowedForStitching(true);
        }
    }
}

bool findOriginalNodes(FuserNodePtrV4        fusedNode,
                       const ClusterNodeMap& originalClusterNodes,
                       NodeList&             originalNodesForNewNode)
{
    LOG_DEBUG(GC_TPC_FUSER, "finding original nodes for fused node {}", fusedNode->nodeName);

    // return the actual original node pointers for a fused node according to their unique ids returned from the fuser
    if (fusedNode->fusedIdentifiers.empty())
    {
        LOG_ERR(GC_TPC_FUSER,
                "fused node {} doesn't have any fused identifiers (that mark its original nodes)",
                fusedNode->nodeName);
        return false;
    }

    for (auto origNodeID : fusedNode->fusedIdentifiers)
    {
        auto origNodeIter = originalClusterNodes.find(origNodeID);
        if (origNodeIter == originalClusterNodes.end())
        {
            LOG_ERR(GC_TPC_FUSER,
                    "Node ID {} in the fusedIdentifiers of fused node {} doesn't exist in the original cluster",
                    origNodeID,
                    fusedNode->nodeName);
            return false;
        }
        originalNodesForNewNode.push_back(origNodeIter->second);
        LOG_TRACE(GC_TPC_FUSER,
                  "   marking node {} (id {}) as original node for fused node {}",
                  origNodeIter->second->getNodeName(),
                  origNodeID,
                  fusedNode->nodeName);
    }

    return true;
}

optimizedGraphStatus tryNodesReplacementInGraph(HabanaGraph&                              graph,
                                                const std::pair<NodePtr, NodeList>&       fusedNodeOrigNodePair,
                                                std::map<NodePtr, std::vector<unsigned>>& newNodesPerFusedNode,
                                                std::map<unsigned, NodePtr>&              newNodeIdMap,
                                                bool                                      isNeedToReplaceNodes = true,
                                                bool                                      isPartOfSubgraph     = false)
{
    // Replace received nodes from fuser with their original nodes in the graph
    // If the node belongs to subgraph send all subgraph nodes and then originals to the same call of replaceNodes

    static NodeList nodesToAdd;
    const NodePtr&  fusedNode     = fusedNodeOrigNodePair.first;
    NodeList        originalNodes = fusedNodeOrigNodePair.second;

    HB_ASSERT(!originalNodes.empty(), "No original nodes for node {}", fusedNode->getNodeName());

    markAllowedForStitching(graph, originalNodes, fusedNode);
    LOG_DEBUG(GC_TPC_FUSER,
              "replacing original nodes with fused node {} and new nodes connected to it (if any), isPartOfSubgraph {}",
              fusedNode->getNodeName(),
              isPartOfSubgraph);

    // if there are new nodes connected to the fused node, send them all together to the replaceNodes functions,
    // so that in case any of them fail to be added, the replacement for all of them will be aborted

    nodesToAdd.emplace_back(fusedNode);
    for (unsigned newNodeId : newNodesPerFusedNode[fusedNode])
    {
        if (newNodeIdMap.count(newNodeId) == 0)
        {
            LOG_ERR(GC_TPC_FUSER,
                    "fused node {} marked as having new node id {} attached but this new node doesn't exist or was"
                    "already added by another fused node",
                    fusedNode->getNodeName(),
                    newNodeId);
            return optimizedGraphFail;
        }
        NodePtr newNode = newNodeIdMap[newNodeId];
        LOG_TRACE(GC_TPC_FUSER,
                  "fused node {} marked as having new node id {} attached - node name {}",
                  fusedNode->getNodeName(),
                  newNodeId,
                  newNode->getNodeName());
        nodesToAdd.emplace_back(newNode);
        // remove this new node from the map so that it couldn't be added twice (if the tpc gave it by mistake in 2
        // different fused nodes)
        newNodeIdMap.erase(newNodeId);
    }

    if (isPartOfSubgraph)
    {
        // The node is part of subgraph, add it all original nodes to the total original nodes in this subgraph
        static std::unordered_set<NodePtr> totalOriginalNodesSet;
        totalOriginalNodesSet.insert(originalNodes.begin(), originalNodes.end());

        if (isNeedToReplaceNodes)
        {
            // this node is the last one on subgraph, create the originalNodes list for the replacement
            originalNodes.assign(totalOriginalNodesSet.begin(), totalOriginalNodesSet.end());
            totalOriginalNodesSet.clear();
        }
    }

    if (isNeedToReplaceNodes)
    {
        // this node is the last one on subgraph or the node dosn't belong to subgraph - replace it
        if (GraphEditor::replaceNodes(graph, originalNodes, nodesToAdd) != REPLACE_NODE_SUCCESS)
        {
            LOG_WARN(GC_TPC_FUSER,
                     "tpcFuser: fusion could not be completed for node {}, skipping it{}",
                     fusedNode->getNodeName(),
                     !newNodesPerFusedNode[fusedNode].empty() ? " and its connected new nodes" : "");
        }
        nodesToAdd.clear();
        LOG_TRACE(GC_TPC_FUSER, "replaceNodes finished success (isSubgraph {})", isPartOfSubgraph);
        fusedNode->getNodeAnnotation().fusedNodes = originalNodes;
    }
    return optimizedGraphSuccess;
}

// Performs the actual fusing.
// Precondition:
//     TPC-Optimized-graph computed
// Post-condition:
//     If false value returned to the caller - GC graph remains unmodified
optimizedGraphStatus replaceOptimizedCluster(HabanaGraph& graph, const std::shared_ptr<GCTPCFuserWrapper>& fuser)
{
    static int fusedClustersCounter = -1;
    int        fusedNodesCounter    = 0;
    int        newNodesCounter      = 0;

    fusedClustersCounter++;

    LOG_DEBUG(GC_TPC_FUSER, "tpcFuser: fusing nodes of subgraph (cluster) {}.", fusedClustersCounter);

    if (!validateOptimizedGraph(fuser))
    {
        LOG_ERR(GC_TPC_FUSER,
                "{}: not all cluster external tensors exists in optimized graph. not fusing.",
                HLLOG_FUNC);
        return optimizedGraphInvalidFusedGraph;
    }

    NodeToItemOrderedMap<NodeList>
        fusedNodesMap;  // map from fused node to nodes that it replaces in the original cluster

    std::map<NodePtr, std::vector<unsigned>> newNodesPerFusedNode;  // map from fused node to the ids of the new nodes
                                                                    // inserted by the fuser that are connected to it

    std::map<unsigned, NodePtr> newNodeIdMap;  // map from new node unique id given by the fuser to its node ptr in GC

    ClusterNodeMap& nodeMap = fuser->getClusterNodes();

    // SubGraph info
    bool                                   foundSubgraphs = false;
    std::unordered_map<unsigned, NodeList> subGraphMap;  // key - originalComplexGuidId that represent subgraphId
                                                         // value - list of synapse nodes that belong in same subGraph

    std::unordered_map<unsigned, std::pair<bool, unsigned>>
        subgraphNodesHandleStatusMap;  // key - nodeId
                                       // values:
                                       // bool - true if it already handled (replace nodes)
                                       // unsigned - the subgraphId from subGraphMap

    // A) Loop through all the optimized fuser graph nodes
    for (auto node : fuser->getOptimizedFuserGraph().nodes)
    {
        // Reduce tpc-fuser-inputs-vector - vector of edges - to gc-tensors.
        TensorVector inputs = getFusedNodeInputEdgesAsGcTensorList(*node, fuser);
        // Reduce tpc-fuser-outputs-vector - vector of edges - to gc-tensors.
        TensorVector outputs = getFusedNodeOutputEdgesAsGcTensorList(*node, fuser);
        if (!outputs.size())
        {
            LOG_WARN(GC_TPC_FUSER,
                     "no outputs for optimized graph node: {}, guid: {}, can't replace cluster",
                     node->nodeName,
                     node->guid);
            return optimizedGraphFailNoOutputs;
        }

        // Create a node that represents a node in the optimized cluster.
        // The external lib can return the following nodes:
        // 1. A fused kernel
        // 2. An existing original node.
        //    In that case we should find the original node in the nodes cluster map and re-add it to the graph.
        // 3. A new node
        NodePtr newNode;
        if (!KernelDB::isFusedGUID(node->guid))
        {
            // either new non-fused node or existing node
            unsigned uniqueId = ((FuserNodeTypeV4*)node.get())->uniqueIdentifier;

            ClusterNodeMap::iterator origNodeIt = nodeMap.find(uniqueId);
            if (origNodeIt == nodeMap.end())  // new non-fused node
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "optimized graph contains a new node. name: {}, guid: {}, Id: {}",
                          node->nodeName,
                          node->guid,
                          uniqueId);
                newNode = NodeFactory::createNode(inputs,
                                                  outputs,
                                                  node->nodeParams,
                                                  node->paramsSize,
                                                  node->guid,
                                                  fmt::format("{}_{}", node->nodeName, uniqueId));
                newNodesCounter++;
                // Validate Ndims Correctness
                synDeviceType deviceType = graph.getDeviceType();
                for (const auto& tensor : newNode->getOperands())
                {
                    if (tensor == nullptr) continue;
                    if (!isTensorDimsValidForNode(tensor, newNode, true))
                    {
                        LOG_ERR(GC_TPC_FUSER,
                                "Tensor {} of node {} dimensions validation failed",
                                tensor->getName(),
                                newNode->getNodeName());
                        return optimizedGraphInvalidFusedGraph;
                    }
                }

                newNodeIdMap[uniqueId] = newNode;
            }
            else  // existing node
            {
                LOG_DEBUG(GC_TPC_FUSER,
                          "optimized graph contains an existing node. name: {}, guid: {}, Id: {}",
                          node->nodeName,
                          node->guid,
                          uniqueId);
                newNode = origNodeIt->second;
            }
        }
        else
        {
            LOG_DEBUG(GC_TPC_FUSER,
                      "optimized graph contains a fused kernel node. name: {}, guid: {}, Id: {}",
                      node->nodeName,
                      node->guid,
                      node->uniqueIdentifier);
            // fused kernel
            if (node->nodeParams == nullptr)
            {
                node->paramsSize = 0;
            }
            newNode = createFusedTpcNode(fusedClustersCounter,
                                         fusedNodesCounter,
                                         node->guid,
                                         inputs,
                                         outputs,
                                         node->nodeParams,
                                         node->paramsSize);

            // search for the original nodes that were fused in the cluster nodes that were sent to the fuser
            NodeList originalNodesForNode;
            if (!findOriginalNodes(node, nodeMap, originalNodesForNode))
            {
                LOG_ERR(GC_TPC_FUSER, "Cannot find original nodes of fused node {}", newNode->getNodeName());
                return optimizedGraphInvalidFusedGraph;
            }

            fusedNodesMap[newNode] = originalNodesForNode;

            // save the ids of the new nodes inserted by the fuser connected to this fused node (since they should be
            // inserted to the graph together)
            newNodesPerFusedNode[newNode] = node->newIdentifiers;

            if (GCFG_ENABLE_MULTI_SIF.value() && newNode->isDynamicShape())
            {
                optimizedGraphStatus ret;
                // If we get nodeParams with the fused node, this means that new
                // split fused SIF should be used instead of MultiSIF.
                if (node->nodeParams)
                {
                    ret = FusedNodeToMultiSifData(graph, node, newNode);
                }
                else
                {
                    FuserGraphTypeV4 preGraph;
                    // query the tpc fuser for the original pregraph that was fused to create this node
                    if (fuser->getNodePreGraph(node, preGraph) != gcapi::GLUE_SUCCESS)
                    {
                        LOG_ERR(GC_TPC_FUSER, "Cannot get pre-graph of node {}", newNode->getNodeName());
                        return optimizedGraphFailNoOutputs;
                    }

                    ret = FuserGraphToMultiSifData(graph, node, newNode, preGraph);
                }

                if (ret != optimizedGraphSuccess)
                {
                    LOG_ERR(GC_TPC_FUSER,
                            "Cannot convert fuser pre-graph for node {} to multi-sif data",
                            newNode->getNodeName());
                    return ret;
                }
            }

            // save subGraphs info
            // search if the current node belong to current subgraph map
            if (node->originalComplexGuidId != 0 && node->originalComplexGuidId != ~0u)
            {
                foundSubgraphs      = true;
                unsigned subgraphId = node->originalComplexGuidId;

                // add the node to it's subgraph maps
                subgraphNodesHandleStatusMap.insert({newNode->getId(), {false, subgraphId}});
                subGraphMap[subgraphId].emplace_back(newNode);

                LOG_DEBUG(GC_TPC_FUSER,
                          "Found node (uniqueIdentifier {}) that belongs to subgraphId {}",
                          node->uniqueIdentifier,
                          subgraphId);
            }

            // should be incremented, only in case of fused node.
            fusedNodesCounter++;
        }
    }

    LOG_DEBUG(GC_TPC_FUSER,
              "tpcFuser: Handling of fuser graph number {} finished with {} fusions and {} new nodes",
              fusedClustersCounter,
              fusedNodesCounter,
              newNodesCounter);

    if (fusedNodesCounter == 0 && newNodesCounter == 0)
    {
        LOG_DEBUG(GC_TPC_FUSER,
                  "tpcFuser: Handling of fuser graph number (cluster) {} finished with no fusions or new nodes!",
                  fusedClustersCounter);
        return optimizedGraphSuccess;
    }

    optimizedGraphStatus status = optimizedGraphSuccess;
    // insert all fused nodes to the graph instead of their original nodes
    for (auto pair : fusedNodesMap)
    {
        const NodePtr& fusedNode = pair.first;

        if (foundSubgraphs)
        {
            // there are subgraphs in the received graph from tpc fuser
            auto itr = subgraphNodesHandleStatusMap.find(fusedNode->getId());
            if (itr != subgraphNodesHandleStatusMap.end())
            {
                // this node is part of subgraph
                bool     isCurrNodeHandled    = (*itr).second.first;
                unsigned nodeKeyInSubGraphMap = (*itr).second.second;
                LOG_TRACE(GC_TPC_FUSER,
                          "curr node is part of subraph {} Node {} nodeId {} isCurrNodeHandled {}",
                          nodeKeyInSubGraphMap,
                          fusedNode->getNodeName(),
                          fusedNode->getId(),
                          isCurrNodeHandled);
                if (isCurrNodeHandled == false)
                {
                    // handle this node in the subgraph
                    HB_ASSERT(subGraphMap.find(nodeKeyInSubGraphMap) != subGraphMap.end(),
                              "subGraphMap must contain nodesin subgraphId {}",
                              nodeKeyInSubGraphMap);
                    NodeList nodesInSubgraph = subGraphMap.at(nodeKeyInSubGraphMap);
                    unsigned nodeCount       = 0;
                    for (auto& node : nodesInSubgraph)
                    {
                        std::pair<NodePtr, NodeList> fusedNodeOrigNodesPair;
                        fusedNodeOrigNodesPair.first  = node;
                        fusedNodeOrigNodesPair.second = fusedNodesMap[node];

                        bool isLastNodeInSubgraph = false;
                        nodeCount++;
                        if (nodeCount == nodesInSubgraph.size())
                        {
                            isLastNodeInSubgraph = true;
                        }

                        LOG_TRACE(GC_TPC_FUSER,
                                  "call tryNodesReplacementInGraph with fusedNodeId {} isLastNodeInSubgraph "
                                  "{} subgraph size {}",
                                  node->getNodeName(),
                                  isLastNodeInSubgraph,
                                  nodesInSubgraph.size());
                        status = tryNodesReplacementInGraph(graph,
                                                            fusedNodeOrigNodesPair,
                                                            newNodesPerFusedNode,
                                                            newNodeIdMap,
                                                            isLastNodeInSubgraph,
                                                            true);
                        if (status != optimizedGraphSuccess) return status;
                        // mark handled in order to not to try replace it again
                        subgraphNodesHandleStatusMap[node->getId()].first = true;
                    }
                }
            }
            else
            {
                // There are subgraphs in this graph but this node doesn't belong to subgraph
                LOG_DEBUG(GC_TPC_FUSER,
                          "call tryNodesReplacementInGraph for node that doesn't belongs to subgraph with "
                          "fusedNode {} nodeId {}",
                          fusedNode->getNodeName(),
                          fusedNode->getId());

                // this node is not part of subgraph
                status = tryNodesReplacementInGraph(graph, pair, newNodesPerFusedNode, newNodeIdMap, true);
                if (status != optimizedGraphSuccess) return status;
            }
        }

        else
        {
            LOG_DEBUG(GC_TPC_FUSER,
                      "call tryNodesReplacementInGraph for node that doesn't belongs to subgraph with "
                      "fusedNode {} nodeId {}",
                      fusedNode->getNodeName(),
                      fusedNode->getId());

            // this node is not part of subgraph
            status = tryNodesReplacementInGraph(graph, pair, newNodesPerFusedNode, newNodeIdMap, true);
            if (status != optimizedGraphSuccess) return status;
        }
    }

    LOG_DEBUG(GC_TPC_FUSER, "tpcFuser: Fusion of subgraph {} is done!", fusedClustersCounter);

    return optimizedGraphSuccess;
}

bool canRunTpcFuser()
{
    if (!TPCFuserSharedObject::instance().isInitialized())
    {
        LOG_DEBUG(GC_TPC_FUSER, "tpcFuser: TPC fuser is not initialized or enabled");
        return false;
    }
    return GCFG_RUN_TPC_FUSER.value();
}

bool getTPCFuserEntryPoints(gcapi::pfnFuseGraphV4& fuserGraphFunc, gcapi::pfnGetFusedNodePreGraphV4& getPreGraphFunc)
{
    fuserGraphFunc  = TPCFuserSharedObject::instance().getFuseGraphFuncPtr();
    getPreGraphFunc = TPCFuserSharedObject::instance().getPreGraphFuncPtr();
    if (fuserGraphFunc == nullptr || getPreGraphFunc == nullptr)
    {
        LOG_ERR(GC_TPC_FUSER, "Failed getting TPC Fuser shared object entry points");
        return false;
    }
    return true;
}

// a redundant control edge is a control edge that is parallel to a data path in the graph.
// remove these control edges before passing the graph to fuser, so that we fuse clusters
// that would have control edges otherwise (and not be fused)
void removeRedundantControlEdges(HabanaGraph& g)
{
    // build efficient connectivity map, to be used in 'areConnected' calls.
    g.buildConnectivityMap(Node::TENSOR_TYPE_DATA);

    // find all redundant control edges in graph
    std::vector<std::pair<NodePtr, TensorVector>> redundantControlEdges;
    for (const NodePtr& blocked : g.getNodes())
    {
        if (!blocked) continue;
        TensorVector redundantCtrlInputs;
        for (const TensorPtr& ctrl : blocked->getControlInputs())
        {
            // remove only control edges that protect coherent memory
            if (ctrl->getControlEdgeType() != Tensor::ControlEdgeType::MEM) continue;
            const NodePtr& blocking = g.getTensorProducer(ctrl);
            HB_ASSERT_PTR(blocking);  // not expecting a control edge without a producer
            if (g.areConnected(blocking, blocked, Node::TENSOR_TYPE_DATA))
            {
                // find all edges before modifing graph connectivity so that we utilize the "number of paths" caching
                redundantCtrlInputs.push_back(ctrl);
            }
        }
        redundantControlEdges.push_back(std::make_pair(blocked, std::move(redundantCtrlInputs)));
    }
    // remove all redundant control edges
    for (const auto& [node, ctrlInputs] : redundantControlEdges)
    {
        for (const TensorPtr& ctrl : ctrlInputs)
        {
            g.removeNodeControlDependency(node, ctrl, Node::eParamUsage::USAGE_INPUT);
        }
    }
}

bool tpcFuser(HabanaGraph& g)
{
    if (!canRunTpcFuser())
    {
        LOG_DEBUG(GC_TPC_FUSER, "TPC Fuser is disabled, pass will not run.");
        return true;
    }
    removeRedundantControlEdges(g);

    gcapi::pfnFuseGraphV4            fuserGraphFunc;
    gcapi::pfnGetFusedNodePreGraphV4 getPreGraphFunc;
    // Check for basic conditions to use TPC fuser
    if (!getTPCFuserEntryPoints(fuserGraphFunc, getPreGraphFunc))
    {
        return false;
    }

    const NodeVector&           topologicallySortedNodes = g.getTopoSortedNodes();
    std::unordered_set<NodePtr> cluster;
    // Save all CanCluster nodes in data structure and add each of them to different cluster
    TPCClusterConstructor              clusterConstructor(g);
    std::shared_ptr<GCTPCFuserWrapper> optimizedGraph;

    LOG_DEBUG(GC_TPC_FUSER, "TpcFuser: computing connected clusters.");

    // The cluster now contains the full graph
    std::copy(std::begin(topologicallySortedNodes),
              std::end(topologicallySortedNodes),
              std::inserter(cluster, std::begin(cluster)));

    // 2.1) Update nodes params for each cluster
    std::shared_ptr<GCTPCFuserWrapper> clusterTPCFuser =
        std::make_shared<GCTPCFuserWrapper>(cluster, g, fuserGraphFunc, getPreGraphFunc);

    // 2.2) Continue updating the data to send TPC (tensors part)
    //      Fuse tpc nodes cluster in our graph. This step creates new tpc kernels
    if (clusterTPCFuser->optimizeCluster(g))
    {
        // Fusing was done, add the new cluster to the optimizedGraph data structure
        optimizedGraph = clusterTPCFuser;
    }

    if (optimizedGraph == nullptr)
    {
        LOG_INFO(GC_TPC_FUSER, "tpcFuser: There are no clusters to fuse.");
        return true;
    }

    LOG_INFO(GC_TPC_FUSER, "tpcFuser: Performing fusion to cluster");

    // 3.) Perform the actual fusion - replace cluster sub-graph with optimized sub-graphs
    optimizedGraphStatus status = replaceOptimizedCluster(g, optimizedGraph);
    TPCFuserSharedObject::instance().releaseFusedGraph(optimizedGraph);
    if ((status == optimizedGraphInvalidFusedGraph) || (status == optimizedGraphFail))
    {
        LOG_ERR(GC_TPC_FUSER, "tpcFuser: Replacing the optimized cluster failed");
        return false;
    }

    LOG_DEBUG(GC_TPC_FUSER, "Finished tpc fuser pass");
    return true;
}

static optimizedGraphStatus FusedNodeToMultiSifData(HabanaGraph& hg, const FuserNodePtrV4& fusedNode, NodePtr& newNode)
{
    auto multiSifInfo                 = std::make_shared<MultiSifNodeInfo>();
    multiSifInfo->m_internalTensorsNr = 0;
    multiSifInfo->m_nodes.resize(1);

    auto& singleNodeInfo = multiSifInfo->m_nodes.back();

    singleNodeInfo.m_sifID.sm_func_index = SIF_SPLIT_FUSED;
    singleNodeInfo.m_sifVersion          = 1;

    singleNodeInfo.m_nodeName = newNode->getNodeName() + "_fused_" + std::to_string(fusedNode->uniqueIdentifier);
    singleNodeInfo.m_nodeGUID = fusedNode->guid;

    if (fusedNode->paramsSize > 0)
    {
        singleNodeInfo.m_nodeParams.resize(fusedNode->paramsSize);
        memcpy(singleNodeInfo.m_nodeParams.data(), fusedNode->nodeParams, fusedNode->paramsSize);
    }

    unsigned inputCount = 0;
    singleNodeInfo.m_inputs.resize(fusedNode->inputEdges.size());

    for (const auto& input : fusedNode->inputEdges)
    {
        auto& currentInputInfo = singleNodeInfo.m_inputs[inputCount];

        currentInputInfo.m_shape.setDim(input.tensor->geometry.dims);
        currentInputInfo.m_shape.setMaxSize(input.tensor->geometry.maxSizes);
        currentInputInfo.m_shape.setMinSize(input.tensor->geometry.minSizes);

        tpc_lib_api::NodeTensorPermutation thisPerm;
        std::copy(std::begin(input.tensor->geometry.permutation),
                  std::begin(input.tensor->geometry.permutation) + tpc_lib_api::MAX_TENSOR_DIM,
                  thisPerm.permutation);
        singleNodeInfo.m_inputPermutations.emplace_back(thisPerm);

        currentInputInfo.m_isInternal = false;
        currentInputInfo.m_index      = inputCount++;
    }

    // eliminate duplicated output edges
    std::vector<unsigned> uniqueIndices;
    unsigned              prevUniqueIdentifier = -1;
    for (unsigned index = 0; index < fusedNode->outputEdges.size(); index++)
    {
        if (fusedNode->outputEdges[index].tensor->uniqueIdentifier == prevUniqueIdentifier)
        {
            continue;
        }
        else
        {
            prevUniqueIdentifier = fusedNode->outputEdges[index].tensor->uniqueIdentifier;
            uniqueIndices.push_back(index);
        }
    }

    unsigned outputCount = 0;
    singleNodeInfo.m_outputs.resize(uniqueIndices.size());

    for (unsigned index : uniqueIndices)
    {
        const auto& output            = fusedNode->outputEdges[index];
        auto&       currentOutputInfo = singleNodeInfo.m_outputs[outputCount];

        currentOutputInfo.m_shape.setDim(output.tensor->geometry.dims);
        currentOutputInfo.m_shape.setMaxSize(output.tensor->geometry.maxSizes);
        currentOutputInfo.m_shape.setMinSize(output.tensor->geometry.minSizes);

        tpc_lib_api::NodeTensorPermutation thisPerm;
        std::copy(std::begin(output.tensor->geometry.permutation),
                  std::begin(output.tensor->geometry.permutation) + tpc_lib_api::MAX_TENSOR_DIM,
                  thisPerm.permutation);
        singleNodeInfo.m_outputPermutations.emplace_back(thisPerm);

        currentOutputInfo.m_isInternal = false;
        currentOutputInfo.m_index      = outputCount++;
    }

    auto cleanupPermutations = [](std::vector<tpc_lib_api::NodeTensorPermutation>& perms) {
        tpc_lib_api::NodeTensorPermutation identity;
        std::iota(std::begin(identity.permutation), std::end(identity.permutation), 0);

        if (std::all_of(
                std::begin(perms),  // compare all permutations
                std::end(perms),    // to identity
                [&identity](const tpc_lib_api::NodeTensorPermutation& perm) {
                    return std::equal(std::begin(perm.permutation), std::end(perm.permutation), identity.permutation);
                }))
        {
            perms.clear();
        }
    };

    cleanupPermutations(singleNodeInfo.m_inputPermutations);
    cleanupPermutations(singleNodeInfo.m_outputPermutations);

    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(newNode);
    HB_ASSERT_PTR(tpcNode);
    tpcNode->setMultiSifInfo(multiSifInfo);

    return optimizedGraphSuccess;
}

static optimizedGraphStatus FuserGraphToMultiSifData(HabanaGraph&            hg,
                                                     const FuserNodePtrV4&   fusedNode,
                                                     NodePtr&                newNode,
                                                     const FuserGraphTypeV4& preGraph)
{
    std::unordered_map<unsigned, unsigned> internalEdgeMap;

    auto     multiSifInfo      = std::make_shared<MultiSifNodeInfo>();
    unsigned nodeCount         = 0;
    unsigned nextInternalIndex = 0;

    const auto& kernelDB = KernelDB::instance();

    auto getSifId = [&hg](char* name, const std::string& guid, sm_function_id_t& sifID, uint64_t& sifVersion) {
        auto deviceId = deviceTypeToDeviceID(hg.getDeviceType());

        tpc_lib_api::UniqueShapeInferenceHash sifHash = {};

        if (!KernelDB::instance().GetKernelShapeInferenceFunctionID(deviceId, guid, &sifHash))
        {
            LOG_ERR(GC, "Can't get shape inference function for dynamic shape tpc node {}", guid);
            return gcapi::GLUE_FAILED;
        }
        sifVersion = KernelDB::instance().GetLibraryVersion(deviceId, guid);
        LOG_TRACE(GC,
                  "TPC node name {} (guid {}) got shape inference function id {} version {}",
                  name,
                  guid,
                  sifHash.Value,
                  sifVersion);

        sifID.sm_func_index = sifHash.Value;
        return gcapi::GLUE_SUCCESS;
    };

    for (const auto& node : preGraph.nodes)
    {
        multiSifInfo->m_nodes.resize(++nodeCount);
        auto& singleNodeInfo = multiSifInfo->m_nodes.back();

        // get sif ID and version from GUID
        StringWithHash guidAndHash(node->guid);
        if (kernelDB.isDynamicShapeKernel(guidAndHash, hg.getDeviceId()))
        {
            auto ret = getSifId(node->nodeName, guidAndHash.getKey(), singleNodeInfo.m_sifID, singleNodeInfo.m_sifVersion);
            if (ret != gcapi::GLUE_SUCCESS)
            {
                LOG_ERR(GC, "Can't get shape inference function for dynamic shape tpc node {}", guidAndHash.getKey());
                return optimizedGraphFail;
            }
        }
        else
        {
            singleNodeInfo.m_sifID.sm_func_index = INVALID_SHAPE_FUNC_ID;
            singleNodeInfo.m_sifVersion          = (uint64_t)(-1);
        }

        // construct made up node name
        singleNodeInfo.m_nodeName = newNode->getNodeName() + "_fused_" + std::to_string(node->uniqueIdentifier);
        singleNodeInfo.m_nodeGUID = guidAndHash.getKey();

        if (node->paramsSize > 0)
        {
            singleNodeInfo.m_nodeParams.resize(node->paramsSize);
            memcpy(singleNodeInfo.m_nodeParams.data(), node->nodeParams, node->paramsSize);
        }

        const auto& nodeInputs = node->inputEdges;
        unsigned    inputCount = 0;

        for (const auto& input : nodeInputs)
        {
            singleNodeInfo.m_inputs.resize(++inputCount);
            bool isInternal =
                input.targetNode.lock() != nullptr && input.tensor->section.type != gcapi::SECTION_PERSISTENT;
            // special case of the diagram below:
            // in case the output t4 is not persistent, there's actually two edges representing the same tensor
            if (isInternal)
            {
                for (const auto& output : input.targetNode.lock()->outputEdges)
                {
                    if ((output.tensor->uniqueIdentifier == input.tensor->uniqueIdentifier) &&
                        (output.targetNode.lock() == nullptr))
                    {
                        isInternal = false;
                        break;
                    }
                }
            }
            auto& currentInputInfo = singleNodeInfo.m_inputs.back();

            currentInputInfo.m_shape.setDim(input.tensor->geometry.dims);
            currentInputInfo.m_shape.setMaxSize(input.tensor->geometry.maxSizes);
            currentInputInfo.m_shape.setMinSize(input.tensor->geometry.minSizes);

            tpc_lib_api::NodeTensorPermutation thisPerm;
            std::copy(std::begin(input.tensor->geometry.permutation),
                      std::begin(input.tensor->geometry.permutation) + tpc_lib_api::MAX_TENSOR_DIM,
                      thisPerm.permutation);
            singleNodeInfo.m_inputPermutations.emplace_back(thisPerm);

            if (isInternal)
            {
                currentInputInfo.m_isInternal = true;
                auto inserted = internalEdgeMap.insert({input.tensor->uniqueIdentifier, nextInternalIndex});
                if (inserted.second)  // insertion took place
                {
                    nextInternalIndex++;
                }
                currentInputInfo.m_index = inserted.first->second;
            }
            else
            {
                currentInputInfo.m_isInternal = false;

                auto it = std::find_if(std::begin(fusedNode->inputEdges),
                                       std::end(fusedNode->inputEdges),
                                       [&input](const FuserEdgeTypeV4& e) {
                                           return e.tensor->uniqueIdentifier == input.tensor->uniqueIdentifier;
                                       });

                if (it == std::end(fusedNode->inputEdges))
                {
                    // Maybe it is the output of the fused node? (like t4 below)
                    //
                    //        |  |                 | fused
                    //        |  |                 | node
                    //  +=====+==+=================+======+
                    //  |     |  |                 |      |
                    //  |    t1  t2     +------+   t3     |
                    //  |     |  |      |      |   |      |
                    //  |     v  v      |      v   v      |
                    //  |    +--------+ |     +--------+  |
                    //  |    |internal| |     |internal|  |
                    //  |    |node1   | |     |node2   |  |
                    //  |    +--------+ |     +--------+  |
                    //  |       | t4    |        |        |
                    //  |       +-------+       t5        |
                    //  |       |                |        |
                    //  +=======+================+========*
                    //          |                |
                    //          v                v
                    //
                    auto it2 = std::find_if(std::begin(fusedNode->outputEdges),
                                            std::end(fusedNode->outputEdges),
                                            [&input](const FuserEdgeTypeV4& e) {
                                                return e.tensor->uniqueIdentifier == input.tensor->uniqueIdentifier;
                                            });

                    HB_ASSERT(it2 != std::end(fusedNode->outputEdges),
                              "Could not find tensor {} in fused pregraph",
                              input.tensor->uniqueIdentifier);

                    currentInputInfo.m_index          = it2 - std::begin(fusedNode->outputEdges);
                    currentInputInfo.m_takeFromOutput = true;
                }
                else
                {
                    currentInputInfo.m_index = it - std::begin(fusedNode->inputEdges);
                }
            }
        }

        const auto& nodeOutputs = node->outputEdges;
        unsigned    outputCount = 0;
        for (auto it = nodeOutputs.begin(), end = nodeOutputs.end(); it != end;)
        {
            const auto& output = *it;

            // Run through all the edges with the same tensor, to determine if it is used externally and increment the
            // iterator beyond them for the next iteration.
            bool isInternal = output.tensor->section.type != gcapi::SECTION_PERSISTENT;
            do
            {
                if (it->targetNode.expired()) isInternal = false;
            } while (++it != end && it->tensor->uniqueIdentifier == output.tensor->uniqueIdentifier);

            singleNodeInfo.m_outputs.resize(++outputCount);

            auto& currentOutputInfo = singleNodeInfo.m_outputs.back();

            currentOutputInfo.m_shape.setDim(output.tensor->geometry.dims);
            currentOutputInfo.m_shape.setMaxSize(output.tensor->geometry.maxSizes);
            currentOutputInfo.m_shape.setMinSize(output.tensor->geometry.minSizes);

            tpc_lib_api::NodeTensorPermutation thisPerm;
            std::copy(std::begin(output.tensor->geometry.permutation),
                      std::begin(output.tensor->geometry.permutation) + tpc_lib_api::MAX_TENSOR_DIM,
                      thisPerm.permutation);
            singleNodeInfo.m_outputPermutations.emplace_back(thisPerm);

            if (isInternal)
            {
                currentOutputInfo.m_isInternal = true;
                auto inserted = internalEdgeMap.insert({output.tensor->uniqueIdentifier, nextInternalIndex});
                if (inserted.second)  // insertion took place
                {
                    nextInternalIndex++;
                }
                currentOutputInfo.m_index = inserted.first->second;
            }
            else
            {
                currentOutputInfo.m_isInternal = false;

                // Output edges can be duplicated. The duplicates do not
                // participate in the GC node.
                // We need to skip the duplicates.
                //
                int  prevUniqueIdentifier = -1;
                int  index                = 0;
                bool found                = false;
                for (auto it = std::begin(fusedNode->outputEdges); it != std::end(fusedNode->outputEdges); ++it)
                {
                    if (it->tensor->uniqueIdentifier == output.tensor->uniqueIdentifier)
                    {
                        found = true;
                        break;
                    }
                    if (it->tensor->uniqueIdentifier != prevUniqueIdentifier)
                    {
                        ++index;
                        prevUniqueIdentifier = it->tensor->uniqueIdentifier;
                    }
                }

                HB_ASSERT(found, "Could not find tensor {} in fused pregraph outputs", output.tensor->uniqueIdentifier);

                currentOutputInfo.m_index = index;
            }
        }

        // if all permutations are identities, we can save a bit of space
        // and performance on SIF calling
        // by passing null pointers instead

        auto cleanupPermutations = [](std::vector<tpc_lib_api::NodeTensorPermutation>& perms) {
            tpc_lib_api::NodeTensorPermutation identity;
            std::iota(std::begin(identity.permutation), std::end(identity.permutation), 0);

            if (std::all_of(std::begin(perms),  // compare all permutations
                            std::end(perms),    // to identity
                            [&identity](const tpc_lib_api::NodeTensorPermutation& perm) {
                                return std::equal(std::begin(perm.permutation),
                                                  std::end(perm.permutation),
                                                  identity.permutation);
                            }))
            {
                perms.clear();
            }
        };

        cleanupPermutations(singleNodeInfo.m_inputPermutations);
        cleanupPermutations(singleNodeInfo.m_outputPermutations);
    }

    multiSifInfo->m_internalTensorsNr = nextInternalIndex;

    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(newNode);
    HB_ASSERT_PTR(tpcNode);
    tpcNode->setMultiSifInfo(multiSifInfo);

    return optimizedGraphSuccess;
}

bool ClusteringUtils::canBeClusteredBasic(const HabanaGraph& graph, const NodePtr node)
{
    bool isBroadcast = node->getNodeType() == Node::TYPE_BROADCAST;
    return ((graph.runsOnTPC(node) || isBroadcast || isReshapeNode(node)) && !node->is64BitOperands());
}

bool ClusteringUtils::isClusterConnected(const NodeList& cluster)
{
    LOG_TRACE(GC_TPC_FUSER, "Verifying if cluster is connected");
    Graph graph;
    std::for_each(cluster.begin(), cluster.end(), [&graph](const NodePtr& node) { graph.addNode(node); });
    return graph.isConnectedGraph();
}

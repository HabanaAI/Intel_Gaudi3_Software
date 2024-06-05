//
// Created by yocohen on 04/02/2021.
//

#ifndef GRAPH_OPTIMIZER_GRAPH_ENTRIES_CONTAINER_HPP
#define GRAPH_OPTIMIZER_GRAPH_ENTRIES_CONTAINER_HPP
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include "habana_graph.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "graph_factory.h"
#include "llvm/small_vector.h"
#include "chromium/small_map.h"
#include "chromium/small_set.h"

typedef uint32_t                                 GraphId;
typedef std::unordered_set<synNodeId>            nodeIdSet;
typedef std::unordered_map<synNodeId, nodeIdSet> nodeDescendantsMap;
typedef std::unordered_map<synNodeId, synNodeId> nodeParentsMaps;
struct InternalGraphHandle
{
    uint32_t graphId;
};

class GraphEntriesContainer
{
public:
    GraphEntriesContainer(ConcurrentSlotMapAlloc<InternalSectionHandle>& sectionHandleSlotMap)
    : m_sectionHandleSlotMap(sectionHandleSlotMap)
    {
    }

    ~GraphEntriesContainer();

    void registerGraph(synGraphHandle* pGraphHandle, HabanaGraphPtr&& graph);
    HabanaGraphPtr replaceGraph(const synGraphHandle graphHandle, CompilationMode compilationMode);

    HabanaGraph* operator[](const synGraphHandle graphHandle);

    HabanaGraph* operator[](GraphId graphId);

    synStatus getGraphId(const synGraphHandle graphHandle, GraphId& graphId);

    synDeviceType getDeviceTypeByGraphId(GraphId graphId);

    synDeviceType getDeviceType(const synGraphHandle graphHandle);

    synStatus getCurrentGraphId(GraphId& graphId);

    HabanaGraphPtr destroyGraph(synGraphHandle graphHandle);

    void destroyAllGraphs();

    bool validateGraphSection(InternalSectionHandle* sectionHandle);

    synStatus sectionLockAndSetID(InternalSectionHandle* sectionHandle);

    HabanaGraph* begin();

    // tensor methods
    bool doTensorExistByName(GraphId graphId, const std::string& name) const;
    bool doTensorExistByPtr(GraphId graphId, void* addr) const;

    void      setTensorByName(GraphId graphId, const std::string& name, const TensorPtr& t);
    void      setTensorByPtr(GraphId graphId, void* addr, const TensorPtr& t);
    TensorPtr getTensorByName(GraphId graphId, const std::string& name);
    TensorPtr getTensorByPtr(GraphId graphId, void* addr);
    TensorPtr findTensor(synTensor tensor);
    void      setTensorSharedPtr(GraphId currGraph, const TensorPtr& t);
    void      setTensorSharedPtr(GraphId currGraph, const HabanaGraph::TensorPtrMappingVec& tensorsPtrMapVec);
    size_t    getNumTensors(GraphId currGraph) const;
    bool      destroyTensor(synTensor tensor);
    void      clearTensorAddr(GraphId graphId);
    synStatus removeTensorByPtr(GraphId graphId, void* tensorHostAddress, Tensor* tensor);
    TensorPtr createTensor(const synTensorDescriptor* pDescriptor,
                           bool                       isOutput,
                           bool                       isInput,
                           synStatus*                 status,
                           void*                      ptr,
                           uint32_t                   graphId);

    using SectionHandleVec = llvm_vecsmall::SmallVector<synSectionHandle, MAX_TENSOR_NR>;
    void setSections(GraphId currGraph, const SectionHandleVec& sectionHandles);
    void setSection(GraphId currGraph, synSectionHandle sectionHandle);
    void removeSection(GraphId currGraph, synSectionHandle sectionHandle);
    // Duplicate Synapse API requires cloning all the user graph's tensors including ones not attached to a node.
    // Persistent tensors not attached to a node are only available in the graph's entry in GraphEntriesContainer.
    // So we fill the original graph's tensors in the function bellow, later to be cloned by the appropriate Graph API.
    void fillWithOriginalTensors(GraphId currGraph, HabanaGraph::TensorPtrMappingVec& tensorsPtrMapVec);

    synStatus setNodeDescendant(GraphId graphId, synNodeId node, synNodeId descendantNode);

    bool isComplexGuidWithDescendants(synNodeId nodeId, GraphId graphId) const;

    const nodeIdSet& getNodeDescendantsIds(synNodeId nodeId, GraphId graphId) const;  // may throw std::out_of_range

    HabanaGraph* getGraphForCompilation(const synGraphHandle graphHandle);

private:
    struct GraphEntry;

    HabanaGraph* _getGraph(GraphId graphId);
    void         _destroyTensors(GraphEntry& graph);
    void         _destroySections(GraphEntry& graph);
    bool         _registerTensorName(GraphId graphId, const char* tensorName);

    typedef uint32_t TensorHandle;

    struct GraphEntry
    {
        static constexpr size_t                                            SMALL_MAP_TENSOR_COUNT = 8;
        HabanaGraphPtr                                                     graph;
        SmallMap<std::map<std::string, TensorPtr>, SMALL_MAP_TENSOR_COUNT> nameTensorMap;
        SmallMap<std::map<void*, TensorPtr>, SMALL_MAP_TENSOR_COUNT>       addrTensorMap;
        SmallMap<std::map<synTensor, TensorPtr>, SMALL_MAP_TENSOR_COUNT>   tensorSharedPtrMap;
        // sectionHandles is needed to track sections created for the graph but not yet assigned to a tensor
        // so that we are still able to clean those up as part of graph destruction.
        // And also to handle cases where a tensor attached to a section is destroyed, as in that case as well
        // the graph loses track of it's associated sections.
        // otherwise, we could have just retrieved it directly from the associated tensors.
        SmallSet<std::set<synSectionHandle>, SMALL_MAP_TENSOR_COUNT> sectionHandles;
        std::unordered_set<std::string>                              registeredTensorNames;
        nodeDescendantsMap                                           complexNodeDescendantsMap;
        nodeParentsMaps                                              complexNodeParentsMap;
        std::atomic<uint32_t>                                        compileTime = 0;
        GraphEntry()                                                             = default;
        GraphEntry(HabanaGraphPtr&& g) : graph(std::move(g)) {}
    };
    GraphId                                        m_graphsNextId = {};
    std::unordered_map<GraphId, GraphEntry> m_graphs;
    ConcurrentSlotMapAlloc<InternalSectionHandle>& m_sectionHandleSlotMap;
};
#endif  // GRAPH_OPTIMIZER_GRAPH_ENTRIES_CONTAINER_HPP

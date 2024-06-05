//
// Created by yocohen on 04/02/2021.
//
#include "graph_entries_container.hpp"
#include "defs.h"
#include "section_handle.hpp"
#include "utils.h"

GraphEntriesContainer::~GraphEntriesContainer()
{
    destroyAllGraphs();
}

void GraphEntriesContainer::registerGraph(synGraphHandle* pGraphHandle, HabanaGraphPtr&& graph)
{
    // make sure key is not found in m_graphs
    bool inserted = m_graphs.try_emplace(m_graphsNextId, std::move(graph)).second;
    HB_ASSERT(inserted, "graph ID already present in m_graphs");

    // Update user's handle
    *pGraphHandle            = new InternalGraphHandle();
    (*pGraphHandle)->graphId = m_graphsNextId++;
}

HabanaGraphPtr GraphEntriesContainer::replaceGraph(const synGraphHandle graphHandle, CompilationMode compilationMode)
{
    auto itGraph = m_graphs.find(graphHandle->graphId);
    if (itGraph == m_graphs.end())
    {
        LOG_ERR(SYN_GRAPH, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return HabanaGraphPtr();
    }

    const auto& curGraph = itGraph->second.graph;
    if (compilationMode == curGraph->getCompilationMode())
    {
        LOG_ERR(SYN_GRAPH, "{}: Trying to fallback from same type of compiler.", HLLOG_FUNC);
        return HabanaGraphPtr();
    }
    auto deviceType       = curGraph->getDeviceType();
    auto oldGraph         = std::move(itGraph->second.graph);
    itGraph->second.graph = GraphFactory::createGraph(deviceType, compilationMode);
    if (itGraph->second.graph == nullptr)
    {
        LOG_ERR(SYN_GRAPH, "{}: Failed to create a new graph.", HLLOG_FUNC);
        return HabanaGraphPtr();
    }
    return oldGraph;
}

HabanaGraph* GraphEntriesContainer::operator[](const synGraphHandle graphHandle)
{
    return _getGraph(graphHandle->graphId);
}
HabanaGraph* GraphEntriesContainer::operator[](GraphId graphId)
{
    return _getGraph(graphId);
}

bool GraphEntriesContainer::validateGraphSection(InternalSectionHandle* sectionHandle)
{
    const auto* graph = _getGraph(sectionHandle->getGraphID());
    if (graph == nullptr)
    {
        LOG_ERR(SYN_GRAPH,
                "{}: Section 0x{:x} invalid graph ID: {}",
                HLLOG_FUNC,
                TO64(sectionHandle),
                sectionHandle->getGraphID());
        return false;
    }
    return graph->validateMemorySection(sectionHandle);
}

synStatus GraphEntriesContainer::sectionLockAndSetID(InternalSectionHandle* sectionHandle)
{
    if (!sectionHandle->isLocked())
    {
        auto* graph = _getGraph(sectionHandle->getGraphID());
        if (nullptr == graph)
        {
            LOG_ERR(SYN_GRAPH,
                    "{}: Section 0x{:x} invalid graph ID: {}",
                    HLLOG_FUNC,
                    TO64(sectionHandle),
                    sectionHandle->getGraphID());
            return synInvalidArgument;
        }
        LOG_TRACE(SYN_GRAPH, "{}: Locking section 0x{:x}", HLLOG_FUNC, TO64(sectionHandle));
        InternalSectionHandle::SectionIDType sectionId;
        if (sectionHandle->getPersistent())
        {
            sectionId = graph->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
            graph->setSectionIdSectionType(sectionId, sectionHandle->getGroup());
        }
        else
        {
            sectionId = graph->getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);
        }
        sectionHandle->setIDAndLock(sectionId);
    }
    return synSuccess;
}

HabanaGraph* GraphEntriesContainer::_getGraph(GraphId graphId)
{
    auto itGraph = m_graphs.find(graphId);
    if (itGraph == m_graphs.end())
    {
        LOG_ERR(SYN_GRAPH, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return nullptr;
    }

    return itGraph->second.graph.get();
}

HabanaGraphPtr GraphEntriesContainer::destroyGraph(synGraphHandle graphHandle)
{
    GraphId graphId = graphHandle->graphId;
    auto    itGraph = m_graphs.find(graphId);
    if (itGraph == m_graphs.end())
    {
        LOG_ERR(SYN_GRAPH, "{}: ֳGraph Handle is incorrect. No graph to destroy", HLLOG_FUNC);
        return HabanaGraphPtr();
    }
    _destroyTensors(itGraph->second);
    _destroySections(itGraph->second);
    HabanaGraphPtr graphPtr(std::move(itGraph->second.graph));
    m_graphs.erase(graphId);
    return graphPtr;
}

synStatus GraphEntriesContainer::getGraphId(const synGraphHandle graphHandle, GraphId& graphId)
{
    if (m_graphs.size() == 0)
    {
        LOG_ERR(SYN_GRAPH, "There are no existing graphs!");
        return synFail;
    }

    if (!graphHandle)
    {
        graphId = m_graphsNextId - 1;
    }
    else
    {
        auto itGraph = m_graphs.find(graphHandle->graphId);

        if (itGraph == m_graphs.end())
        {
            LOG_ERR(SYN_GRAPH, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
            return synFail;
        }

        graphId = graphHandle->graphId;
    }
    return synSuccess;
}

// This method will return the last graph Id this class provided and is currently used only for Gaudi
// Demo mode, which is an internal synapse mode where we assume the demo don't destroy graph
// during execution, and is setting the graph address at the beginning.
synStatus GraphEntriesContainer::getCurrentGraphId(GraphId& graphId)
{
    if (m_graphs.size() == 0)
    {
        LOG_ERR(SYN_GRAPH, "There are no existing graphs!");
        return synFail;
    }
    graphId = (m_graphsNextId - 1);

    return synSuccess;
}

void GraphEntriesContainer::destroyAllGraphs()
{
    for (auto& graphPair : m_graphs)
    {
        _destroyTensors(graphPair.second);
        _destroySections(graphPair.second);
    }
    m_graphs.clear();
}

HabanaGraph* GraphEntriesContainer::begin()
{
    return m_graphs.begin()->second.graph.get();
}

// tensor methods
bool GraphEntriesContainer::doTensorExistByName(GraphId graphId, const std::string& name) const
{
    const auto itr = m_graphs.find(graphId);
    return itr != m_graphs.end() && (itr->second.nameTensorMap.count(name) > 0);
}

bool GraphEntriesContainer::doTensorExistByPtr(GraphId graphId, void* addr) const
{
    const auto itr = m_graphs.find(graphId);
    return itr != m_graphs.end() && (itr->second.addrTensorMap.count(addr) > 0);
}

void GraphEntriesContainer::setTensorByName(GraphId graphId, const std::string& name, const TensorPtr& t)
{
    m_graphs[graphId].nameTensorMap[name] = t;
}

void GraphEntriesContainer::setTensorByPtr(GraphId graphId, void* addr, const TensorPtr& t)
{
    m_graphs[graphId].addrTensorMap[addr] = t;
}

TensorPtr GraphEntriesContainer::getTensorByName(GraphId graphId, const std::string& name)
{
    if (doTensorExistByName(graphId, name))
    {
        return m_graphs[graphId].nameTensorMap[name];
    }
    return nullptr;
}

TensorPtr GraphEntriesContainer::getTensorByPtr(GraphId graphId, void* addr)
{
    if (doTensorExistByPtr(graphId, addr))
    {
        return m_graphs[graphId].addrTensorMap[addr];
    }
    return nullptr;
}
TensorPtr GraphEntriesContainer::findTensor(synTensor tensor)
{
    if (tensor)
    {
        Tensor*    t               = reinterpret_cast<Tensor*>(tensor);
        const auto currentGraphItr = m_graphs.find(t->getGraphID());
        if (currentGraphItr != m_graphs.end())
        {
            const auto& tensorSharedPtrMap = currentGraphItr->second.tensorSharedPtrMap;
            const auto  currentTensorItr   = tensorSharedPtrMap.find(tensor);
            if (currentTensorItr != tensorSharedPtrMap.end())
            {
                return currentTensorItr->second;
            }
        }
    }
    return nullptr;
}

size_t GraphEntriesContainer::getNumTensors(GraphId currGraph) const
{
    const auto currentGraphItr = m_graphs.find(currGraph);
    if (currentGraphItr != m_graphs.end())
    {
        return currentGraphItr->second.tensorSharedPtrMap.size();
    }
    HB_ASSERT(false, "graph entry not found graphId: {}", currGraph);
    return 0;
}

void GraphEntriesContainer::fillWithOriginalTensors(GraphId                           currGraph,
                                                    HabanaGraph::TensorPtrMappingVec& tensorsPtrMapVec)
{
    const auto currentGraphItr = m_graphs.find(currGraph);
    if (currentGraphItr != m_graphs.end())
    {
        auto& tensorSharedPtrMap = currentGraphItr->second.tensorSharedPtrMap;
        tensorsPtrMapVec.reserve(tensorSharedPtrMap.size());
        for (auto& origTensorIter : tensorSharedPtrMap)
        {
            tensorsPtrMapVec.emplace_back(origTensorIter.second.get(), nullptr);
        }
        return;
    }
    HB_ASSERT(false, "graph entry not found graphId: {}", currGraph);
}

void GraphEntriesContainer::setTensorSharedPtr(GraphId                                 currGraph,
                                               const HabanaGraph::TensorPtrMappingVec& tensorsPtrMapVec)
{
    auto& graph = m_graphs[currGraph];
    for (const auto& tensorPtrMapping : tensorsPtrMapVec)
    {
        synTensor synTen                               = reinterpret_cast<synTensor>(tensorPtrMapping.newTensor.get());
        graph.tensorSharedPtrMap[synTen]               = tensorPtrMapping.newTensor;
    }
}

void GraphEntriesContainer::setTensorSharedPtr(GraphId currGraph, const TensorPtr& t)
{
    synTensor synTen                               = reinterpret_cast<synTensor>(t.get());
    m_graphs[currGraph].tensorSharedPtrMap[synTen] = t;
}

void GraphEntriesContainer::setSections(GraphId currGraph, const SectionHandleVec& sectionHandles)
{
    m_graphs[currGraph].sectionHandles.insert(sectionHandles.begin(), sectionHandles.end());
}

void GraphEntriesContainer::setSection(GraphId currGraph, synSectionHandle sectionHandle)
{
    m_graphs[currGraph].sectionHandles.insert(sectionHandle);
}

void GraphEntriesContainer::removeSection(GraphId currGraph, synSectionHandle sectionHandle)
{
    // If graph destruction takes place concurrently with sectionDestroy call
    // on two different threads, then removeSection can be called after
    // the graph entry was already removed. So we need to protect against it.
    auto currGraphIter = m_graphs.find(currGraph);
    if (currGraphIter != m_graphs.end())
    {
        currGraphIter->second.sectionHandles.erase(sectionHandle);
    }
}

bool GraphEntriesContainer::destroyTensor(synTensor tensor)
{
    Tensor* t   = reinterpret_cast<Tensor*>(tensor);
    auto    itr = m_graphs.find(t->getGraphID());
    if (itr != m_graphs.end())
    {
        if (itr->second.tensorSharedPtrMap.count(tensor) > 0)
        {
            auto& currentGraph = itr->second;
            // Remove tensor from name map
            auto    registeredItr = currentGraph.registeredTensorNames.find(t->getName());
            if (registeredItr != currentGraph.registeredTensorNames.end())
            {
                currentGraph.registeredTensorNames.erase(registeredItr);
            }

            auto nameTensor = currentGraph.nameTensorMap.find(t->getName());
            if (nameTensor != currentGraph.nameTensorMap.end())
            {
                currentGraph.nameTensorMap.erase(nameTensor);
            }
            currentGraph.tensorSharedPtrMap.erase(tensor);
            return true;
        }
    }
    return false;
}

void GraphEntriesContainer::_destroySections(GraphEntry& graph)
{
    for (synSectionHandle sectionHandle : graph.sectionHandles)
    {
        if (!m_sectionHandleSlotMap.erase((SMHandle)sectionHandle))
        {
            LOG_ERR(SYN_GRAPH, "failed to delete section 0x{:x} for graph", (SMHandle)sectionHandle);
        }
    }
    graph.sectionHandles.clear();
}

void GraphEntriesContainer::_destroyTensors(GraphEntry& graph)
{
    graph.addrTensorMap.clear();
    graph.nameTensorMap.clear();
    graph.tensorSharedPtrMap.clear();
    graph.registeredTensorNames.clear();
}

void GraphEntriesContainer::clearTensorAddr(GraphId graphId)
{
    auto itr = m_graphs.find(graphId);
    if (itr != m_graphs.end())
    {
        itr->second.addrTensorMap.clear();
        itr->second.registeredTensorNames.clear();
        itr->second.addrTensorMap.clear();
    }
}

synStatus GraphEntriesContainer::removeTensorByPtr(GraphId graphId, void* tensorHostAddress, Tensor* tensor)
{
    if (m_graphs.count(graphId) == 0)
    {
        return synFail;
    }
    auto& currentGraph = m_graphs[graphId];
    auto  tensorItr    = currentGraph.addrTensorMap.find(tensorHostAddress);
    if (tensorItr != currentGraph.addrTensorMap.end())
    {
        currentGraph.addrTensorMap.erase(tensorItr);
        return synSuccess;
    }

    bool found = false;
    // look for it by the tensor handle
    for (auto it : currentGraph.addrTensorMap)
    {
        if (it.second.get() == tensor)
        {
            currentGraph.addrTensorMap.erase(it.first);
            found = true;
            break;
        }
    }
    if (!found)
    {
        // It means that it is on the sharedPtr-map, but not here...
        // We will NOT return, but clean per request, but will warn the user about this misbehavior - Shame on him!
        LOG_DEBUG(SYN_GRAPH,
                  "Attempt to free tensor {} (key {}, not found (only) in tensor map",
                  (void*)tensor,
                  tensor->getAddress());
        return synFail;
    }
    return synSuccess;
}

bool GraphEntriesContainer::_registerTensorName(GraphId graphId, const char* tensorName)
{
    if (!WORKSPACE_MEMORY_SECTION_NAME.compare(tensorName) || !PROGRAM_MEMORY_SECTION_NAME.compare(tensorName))
    {
        return false;
    }
    return m_graphs[graphId].registeredTensorNames.insert(std::string(tensorName)).second;
}

TensorPtr GraphEntriesContainer::createTensor(const synTensorDescriptor* pDescriptor,
                                              bool                       isOutput,
                                              bool                       isInput,
                                              synStatus*                 pStatus,
                                              void*                      ptr,
                                              uint32_t                   graphId)
{
    if (pDescriptor->m_dims == 0)
    {
        LOG_ERR(SYN_GRAPH, "GraphEntriesContainer::createTensor Cannot create tensor with 0 dimensions");
        *pStatus = synInvalidArgument;
        return nullptr;
    }

    for (int i = 0; i < pDescriptor->m_dims; i++)
    {
        if (pDescriptor->m_minSizes[i] > pDescriptor->m_sizes[i])
        {
            LOG_ERR(SYN_GRAPH, "{}: Minimal tensor size is larger than maximal ", HLLOG_FUNC);
            *pStatus = synInvalidArgument;
            return nullptr;
        }
        if (pDescriptor->m_minSizes[i] != 0 && pDescriptor->m_minSizes[i] != pDescriptor->m_sizes[i] &&
            pDescriptor->m_tensorType == DATA_TENSOR)
        {
            LOG_WARN(SYN_GRAPH,
                     "{}: tensor {} has dynamic sizes but is specified as static - Ignoring dynamicity",
                     HLLOG_FUNC,
                     pDescriptor->m_name);
        }
    }

    TSize sizes[SYN_MAX_TENSOR_DIM];
    TSize minSizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, pDescriptor->m_sizes, SYN_MAX_TENSOR_DIM);
    castNcopy(minSizes, pDescriptor->m_minSizes, SYN_MAX_TENSOR_DIM);
    bool valid = verifyDeviceShapeTensor(pDescriptor->m_dims,
                                         sizes,
                                         pDescriptor->m_dataType,
                                         pDescriptor->m_name,
                                         pDescriptor->m_tensorType == DATA_TENSOR ? nullptr : minSizes);

    if (pDescriptor->m_tensorType == DEVICE_SHAPE_TENSOR && !valid)
    {
        *pStatus = synInvalidArgument;
        return nullptr;
    }
    TensorPtr tensorPtr = std::make_shared<Tensor>(pDescriptor->m_dims,
                                                   sizes,
                                                   pDescriptor->m_dataType,
                                                   static_cast<char*>(ptr),
                                                   nullptr,
                                                   isOutput,
                                                   isInput,
                                                   pDescriptor->m_batchPos,
                                                   pDescriptor->m_tensorType == DATA_TENSOR ? nullptr : minSizes,
                                                   pDescriptor->m_tensorType);

    // If the user provided a name, enforce its uniqueness; otherwise, the name is the ID
    if (pDescriptor->m_name)
    {
        if (_registerTensorName(graphId, pDescriptor->m_name) == false)  // must be unique PER Graph
        {
            /*If the user didn't provide a name, we assign a name internally (Tensor_<id>). The user doesn't know the
            name of such tensor and won't be able to provide an address for it later at RT.*/
            LOG_DEBUG(SYN_GRAPH,
                      "GraphEntriesContainer::createTensor tensor with the same name {} already exists",
                      pDescriptor->m_name);
            *pStatus = synNameIsAlreadyUsed;
            return nullptr;
        }
        tensorPtr->setName(pDescriptor->m_name, true);
    }
    tensorPtr->setGraphID(graphId);
    return tensorPtr;
}

synDeviceType GraphEntriesContainer::getDeviceType(const synGraphHandle graphHandle)
{
    return getDeviceTypeByGraphId(graphHandle->graphId);
}

synDeviceType GraphEntriesContainer::getDeviceTypeByGraphId(GraphId graphId)
{
    const auto itr = m_graphs.find(graphId);
    return (itr != m_graphs.end() && itr->second.graph) ? itr->second.graph->getDeviceType() : synDeviceTypeInvalid;
}

synStatus GraphEntriesContainer::setNodeDescendant(GraphId graphId, synNodeId node, synNodeId descendantNode)
{
    const auto itr = m_graphs.find(graphId);
    if (itr == m_graphs.end())
    {
        LOG_WARN(SYN_GRAPH, "{}: Wrong graph Id {} ", HLLOG_FUNC, graphId);
        return synInvalidArgument;
    }
    nodeDescendantsMap& descendantsMap = itr->second.complexNodeDescendantsMap;
    if (descendantsMap.find(node) == descendantsMap.end())
    {
        LOG_TRACE(SYN_GRAPH, "{}: Mapping complex node {} in graph entry {} ", HLLOG_FUNC, node, graphId);
        descendantsMap.insert({node, std::unordered_set<synNodeId> {}});  // insert node with empty set
    }
    // verify not already descendant of another complex node
    nodeParentsMaps& parentsMap     = itr->second.complexNodeParentsMap;
    auto             nodeParentIter = parentsMap.find(descendantNode);
    if (nodeParentIter != parentsMap.end() && nodeParentIter->second != node)
    {
        LOG_ERR(SYN_GRAPH,
                "{}: Setting node {} as descendant of complex node {} in graph entry {} failed - already descendant of "
                "node {}",
                HLLOG_FUNC,
                descendantNode,
                node,
                graphId,
                nodeParentIter->second);
        return synFail;
    }
    // update descendants set
    LOG_TRACE(SYN_GRAPH,
              "{}: Setting node {} as descendant of complex node {} in graph entry {} ",
              HLLOG_FUNC,
              descendantNode,
              node,
              graphId);
    auto res = descendantsMap[node].insert(descendantNode);
    if (!res.second)
    {
        LOG_WARN(SYN_GRAPH,
                 "{}: Node {} already a descendant of complex node {} in graph entry {}",
                 HLLOG_FUNC,
                 descendantNode,
                 node,
                 graphId);
    }
    else
    {
        // set inverse mapping for later use
        parentsMap.insert({descendantNode, node});
    }

    return synSuccess;
}

bool GraphEntriesContainer::isComplexGuidWithDescendants(synNodeId nodeId, GraphId graphId) const
{
    const auto itr = m_graphs.find(graphId);
    HB_ASSERT(itr != m_graphs.end(), "wrong graph id");
    const nodeDescendantsMap& descendantsMap = itr->second.complexNodeDescendantsMap;
    return descendantsMap.find(nodeId) != descendantsMap.end();
}

const nodeIdSet& GraphEntriesContainer::getNodeDescendantsIds(synNodeId nodeId, GraphId graphId) const
{
    const auto itr = m_graphs.find(graphId);
    HB_ASSERT(itr != m_graphs.end(), "wrong graph id");
    const nodeDescendantsMap& descendantsMap = itr->second.complexNodeDescendantsMap;
    HB_ASSERT(descendantsMap.find(nodeId) != descendantsMap.end(), "node has no descendants");
    return descendantsMap.at(nodeId);
}

HabanaGraph* GraphEntriesContainer::getGraphForCompilation(const synGraphHandle graphHandle)
{
    auto itGraph = m_graphs.find(graphHandle->graphId);
    if (itGraph == m_graphs.end())
    {
        LOG_ERR(SYN_GRAPH, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return nullptr;
    }
    if (itGraph->second.compileTime++ > 0)
    {
        LOG_ERR(SYN_GRAPH, "Graph with handle {} was already compiled", graphHandle->graphId);
        return nullptr;
    }
    return itGraph->second.graph.get();
}

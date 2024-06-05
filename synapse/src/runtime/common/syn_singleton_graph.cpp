#include "syn_singleton.hpp"

#include "defenders.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "graph_compiler/layout.h"
#include "habana_graph.h"
#include "log_manager.h"
#include "runtime/common/device/device_interface.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types_exception.h"
#include "types.h"

#include <memory>
#include <vector>
#include <malloc.h>

using namespace std;
using namespace gc;

HabanaGraph* synSingleton::_getGraphForCompilation(const synGraphHandle graphHandle)
{
    HabanaGraph* graph = m_graphEntries.getGraphForCompilation(graphHandle);
    if (graph == nullptr) return nullptr;

    const CompilationMode compilationMode = graph->calcTypeForCompilation();
    if (compilationMode != graph->getCompilationMode())
    {
        HB_ASSERT((GCFG_FORCE_EAGER.value() & 1) == 0,
                  "Can't fallback to graph mode while FORCE_EAGER={}, compilation aborted.",
                  GCFG_FORCE_EAGER.value());

        // Create new graph and replace it with existing one at mapping table
        HabanaGraphPtr oldGraph = m_graphEntries.replaceGraph(graphHandle, compilationMode);
        if (!oldGraph)
        {
            LOG_ERR(SYN_GRAPH, "{}: Failed to replace graph.", HLLOG_FUNC);
            return nullptr;
        }
        graph         = m_graphEntries[graphHandle];
        HB_ASSERT_PTR(graph);

        // Transfer nodes to new graph then dispose the old one
        if (!oldGraph->moveNodesToGraph(*graph))
        {
            return nullptr;
        }

        LOG_DEBUG(SYN_GRAPH, "{}: Graph replaced, fallback to default compiler is done.", HLLOG_FUNC);
    }

    return graph;
}

synStatus
synSingleton::createGraph(synGraphHandle* pGraphHandle, synDeviceType deviceType, CompilationMode compilationMode)
{
    VERIFY_IS_NULL_POINTER(SYN_API, pGraphHandle, "pGraphHandle");

    if ((compilationMode == CompilationMode::Eager) && (deviceType != synDeviceGaudi2) &&
        (deviceType != synDeviceGaudi3))
    {
        LOG_ERR(SYN_API, "{}: Eager Mode Compiler is not supported for device({})", HLLOG_FUNC, deviceType);
        return synInvalidArgument;
    }

    // Create new graph and add it to the mapping table
    HabanaGraphPtr graph = GraphFactory::createGraph(deviceType, compilationMode);

    synDeviceLimitationInfo deviceLimitationInfo {};
    uint16_t numOfDevices = m_deviceManager.getNumDevices();

    for (unsigned idx = 0; idx < numOfDevices; idx++)
    {
        std::shared_ptr<DeviceInterface> pDevice = m_deviceManager.getDeviceInterface(__FUNCTION__);
        if (pDevice != nullptr)
        {
            pDevice->getDeviceLimitationInfo(deviceLimitationInfo);
            if (deviceLimitationInfo.fp32Limited)
            {
                graph.get()->setFP32LimitedDevice();
                break;
            }
        }
    }

    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        m_graphEntries.registerGraph(pGraphHandle, std::move(graph));
    }

    LOG_DEBUG(SYN_GRAPH, "{}: Graph created, fp32Limited: {}", HLLOG_FUNC, deviceLimitationInfo.fp32Limited);
    LOG_DEBUG(SYN_API, "{}: deviceType {} synGraphHandle {}", HLLOG_FUNC, deviceType, TO64(*pGraphHandle));
    return synSuccess;
}

/*DEPRECATED*/
synStatus synSingleton::graphSetAttribute(synGraphHandle           graphHandle,
                                          const synGraphAttribute* attributes,
                                          const uint64_t*          values,
                                          uint32_t                 size)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    HabanaGraph*                 graph = m_graphEntries[graphHandle];

    if (!graph)
    {
        LOG_ERR(SYN_API, "{}: Failed to find  graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (graph->isCompiled())
    {
        LOG_ERR(SYN_API, "{}: Can't set attribute to an already compiled graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (attributes == nullptr || values == nullptr || !size || size > GRAPH_ATTRIBUTE_MAX)
    {
        LOG_ERR(SYN_API, "{}: Can't set attribute with bad params {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }
    else
    {
        for (int i = 0 ; i < size ; ++i)
        {
            switch (attributes[i])
            {
                case GRAPH_ATTRIBUTE_INFERENCE:
                {
                    if ((values[i] == true) || (values[i] == false))
                    {
                        if (!(graph->isEmpty()))
                        {
                            LOG_ERR(SYN_API, "{}: Can't set inference mode ({}) to a graph with nodes", HLLOG_FUNC, values[i]);
                            return synFail;
                        }
                        graph->setInferenceMode(values[i]);
                        LOG_DEBUG(SYN_GRAPH,
                                  "{}: setting graph inference mode {} for graph {}",
                                  HLLOG_FUNC,
                                  values[i],
                                  graphHandle->graphId);
                    }
                    else
                    {
                        LOG_ERR(SYN_API, "{}: Can't set inference with bad value {}", HLLOG_FUNC, values[i]);

                        return synFail;
                    }
                    break;
                }
                case GRAPH_ATTRIBUTE_QUANTIZATION:
                {
                    if ((values[i] == true) || (values[i] == false))
                    {
                        graph->setQuantizationEnabled(values[i]);
                        LOG_DEBUG(SYN_GRAPH,
                                  "{}: setting quantization enabled {} for graph {}",
                                  HLLOG_FUNC,
                                  values[i],
                                  graphHandle->graphId);
                    }
                    else
                    {
                        LOG_ERR(SYN_API, "{}: Can't set quantization enabled with bad value {}", HLLOG_FUNC, values[i]);

                        return synFail;
                    }
                    break;
                }
                default:
                {
                    LOG_DEBUG(SYN_GRAPH, "{}: unknown attributes found for graph {}", HLLOG_FUNC, graphHandle->graphId);
                }
            }

        }
    }
    LOG_DEBUG(SYN_GRAPH, "{}: managed to set {} attributes for graph {} ",
              HLLOG_FUNC,
              size,
              graphHandle->graphId);
    return synSuccess;
}

synStatus synSingleton::graphSetAttributes(synGraphHandle              graphHandle,
                                           const synGraphAttribute*    attributes,
                                           const synGraphAttributeVal* values,
                                           uint32_t                    size)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "pGraphHandle");
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    HabanaGraph*                 graph = m_graphEntries[graphHandle];

    if (!graph)
    {
        LOG_ERR(SYN_API, "{}: Failed to find  graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (graph->isCompiled())
    {
        LOG_ERR(SYN_API, "{}: Can't set attribute to an already compiled graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    int attributesSet = size;
    if (attributes == nullptr || values == nullptr || !size || size > GRAPH_ATTRIBUTE_MAX)
    {
        LOG_ERR(SYN_API, "{}: Can't set attribute with bad params {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }
    else
    {
        for (int i = 0 ; i < size ; ++i)
        {
            switch (attributes[i])
            {
                case GRAPH_ATTRIBUTE_INFERENCE:
                {
                    if (!(graph->isEmpty()))
                    {
                        LOG_ERR(SYN_API, "{}: Can't set inference mode ({}) to a graph with nodes", HLLOG_FUNC, values[i].iAttrVal);
                        return synFail;
                    }
                    graph->setInferenceMode(values[i].iAttrVal);
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: setting graph inference mode {} for graph {}",
                              HLLOG_FUNC,
                              values[i].iAttrVal,
                              graphHandle->graphId);
                    break;
                }
                case GRAPH_ATTRIBUTE_QUANTIZATION:
                {
                    graph->setQuantizationEnabled(values[i].iAttrVal);
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: setting quantization enabled {} for graph {}",
                              HLLOG_FUNC,
                              values[i].iAttrVal,
                              graphHandle->graphId);
                    break;
                }
                case GRAPH_ATTRIBUTE_BACKOFF_FACTOR:
                {
                    if (values[i].dAttrVal > 0)
                    {
                        graph->setBackoffFactor(values[i].dAttrVal);
                        LOG_DEBUG(SYN_GRAPH,
                                "{}: setting graph backoff factor {} for graph {}",
                                HLLOG_FUNC,
                                values[i].dAttrVal,
                                graphHandle->graphId);
                    }
                    else
                    {
                        LOG_ERR(SYN_API, "{}: Can't set backoff factor with bad (<=0) value {}", HLLOG_FUNC, values[i].dAttrVal);
                        return synFail;
                    }
                    break;
                }
                default:
                {
                    attributesSet--;
                    LOG_DEBUG(SYN_GRAPH, "{}: unknown attributes found for graph {}", HLLOG_FUNC, graphHandle->graphId);
                }
            }

        }
    }
    LOG_DEBUG(SYN_GRAPH, "{}: managed to set {} attributes for graph {} ", HLLOG_FUNC, attributesSet, graphHandle->graphId);
    return synSuccess;
}

/*DEPRECATED*/
synStatus synSingleton::graphGetAttribute(synGraphHandle            graphHandle,
                                          const synGraphAttribute*  attributes,
                                          uint64_t*                 values,
                                          const uint32_t            size)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    HabanaGraph*                 graph = m_graphEntries[graphHandle];

    if (!graph)
    {
        LOG_ERR(SYN_API, "{}: Failed to find  graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (graph->isCompiled())
    {
        LOG_ERR(SYN_API, "{}: Can't get attribute to an already compiled graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    int attributesGot = size;
    if (attributes == nullptr || values == nullptr || !size || size > GRAPH_ATTRIBUTE_MAX)
    {
        LOG_ERR(SYN_API, "{}: Can't get attribute with bad params {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }
    else
    {
        for (int i = 0 ; i < size ; ++i)
        {
            switch (attributes[i])
            {
                case GRAPH_ATTRIBUTE_INFERENCE:
                {
                    values[i] = graph->getInferenceMode();
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: got graph inference mode {} for graph {}",
                              HLLOG_FUNC,
                              values[i],
                              graphHandle->graphId);
                    break;
                }
                case GRAPH_ATTRIBUTE_QUANTIZATION:
                {
                    values[i] = graph->getQuantizationEnabled();
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: got graph quantization enabled {} for graph {}",
                              HLLOG_FUNC,
                              values[i],
                              graphHandle->graphId);
                    break;
                }
                default:
                {
                    attributesGot--;
                    LOG_DEBUG(SYN_GRAPH, "{}: no known attributes found for graph {}", HLLOG_FUNC, graphHandle->graphId);
                }
            }
        }
    }
    LOG_DEBUG(SYN_GRAPH, "{}: managed to get {} attributes for graph {} ", HLLOG_FUNC, attributesGot, graphHandle->graphId);
    return synSuccess;
}

synStatus synSingleton::graphGetAttributes(synGraphHandle           graphHandle,
                                           const synGraphAttribute* attributes,
                                           synGraphAttributeVal*    values,
                                           const uint32_t           size)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "pGraphHandle");
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    HabanaGraph*                 graph = m_graphEntries[graphHandle];

    if (!graph)
    {
        LOG_ERR(SYN_API, "{}: Failed to find  graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (graph->isCompiled())
    {
        LOG_ERR(SYN_API, "{}: Can't get attribute to an already compiled graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (attributes == nullptr || values == nullptr || !size || size > GRAPH_ATTRIBUTE_MAX)
    {
        LOG_ERR(SYN_API, "{}: Can't get attribute with bad params {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }
    else
    {
        for (int i = 0 ; i < size ; ++i)
        {
            switch (attributes[i])
            {
                case GRAPH_ATTRIBUTE_INFERENCE:
                {
                    values[i].iAttrVal = graph->getInferenceMode();
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: got graph inference mode {} for graph {}",
                              HLLOG_FUNC,
                              values[i].iAttrVal,
                              graphHandle->graphId);
                    break;
                }
                case GRAPH_ATTRIBUTE_QUANTIZATION:
                {
                    values[i].iAttrVal = graph->getQuantizationEnabled();
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: got graph quantization enabled {} for graph {}",
                              HLLOG_FUNC,
                              values[i].iAttrVal,
                              graphHandle->graphId);
                    break;
                }
                case GRAPH_ATTRIBUTE_BACKOFF_FACTOR:
                {
                    values[i].dAttrVal = graph->getBackoffFactor();
                    LOG_DEBUG(SYN_GRAPH,
                              "{}: got graph backoff factor {} for graph {}",
                              HLLOG_FUNC,
                              values[i].dAttrVal,
                              graphHandle->graphId);
                    break;
                }
                default:
                {
                    LOG_DEBUG(SYN_GRAPH, "{}: no known attributes found for graph {}", HLLOG_FUNC, graphHandle->graphId);
                }
            }
        }
    }
    LOG_DEBUG(SYN_GRAPH, "{}: managed to get attributes for graph {} ", HLLOG_FUNC, graphHandle->graphId);
    return synSuccess;
}

synStatus synSingleton::duplicateGraph(synGraphHandle      graphHandle,
                                       synGraphHandle*     newGraphHandle,
                                       synTensorHandleMap* tensorsMap,
                                       uint32_t*           numTensors,
                                       synNodeHandleMap*   nodesMap,
                                       uint32_t*           numNodes)
{
    VERIFY_IS_NULL_POINTER(SYN_API, newGraphHandle, "newGraphHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, numTensors, "numTensors");
    VERIFY_IS_NULL_POINTER(SYN_API, numNodes, "numNodes");

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    HabanaGraph*                 origGraph = m_graphEntries[graphHandle];

    if (!origGraph)
    {
        LOG_ERR(SYN_API, "{}: Failed to find original graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (origGraph->isCompiled())
    {
        LOG_ERR(SYN_API, "{}: Can't duplicate an already compiled graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (tensorsMap == nullptr || nodesMap == nullptr)
    {
        *numTensors = m_graphEntries.getNumTensors(graphHandle->graphId);
        *numNodes   = origGraph->getNumNodesPreCompilation();
        return synSuccess;
    }
    else
    {
        auto origGraphNumTensors = m_graphEntries.getNumTensors(graphHandle->graphId);
        auto origGraphNumNodes   = origGraph->getNumNodesPreCompilation();
        if (*numTensors != origGraphNumTensors)
        {
            LOG_ERR(SYN_API,
                    "{}: Failed to create duplicate graph for graph {} wrong numTensors passed. expected {} passed {}",
                    HLLOG_FUNC,
                    graphHandle->graphId,
                    origGraphNumTensors,
                    *numTensors);
            return synFail;
        }
        if (*numNodes != origGraphNumNodes)
        {
            LOG_ERR(SYN_API,
                    "{}: Failed to create duplicate graph for graph {} wrong numNodes passed. expected {} passed {}",
                    HLLOG_FUNC,
                    graphHandle->graphId,
                    origGraphNumNodes,
                    *numNodes);
            return synFail;
        }
    }

    HabanaGraph::NodeIdMappingVec nodesMapVec;
    nodesMapVec.reserve(*numNodes);

    HabanaGraph::TensorPtrMappingVec tensorsPtrMapVec;
    m_graphEntries.fillWithOriginalTensors(graphHandle->graphId, tensorsPtrMapVec);

    HabanaGraphPtr newGraph = origGraph->duplicate(tensorsPtrMapVec, nodesMapVec);
    if (!newGraph)
    {
        LOG_ERR(SYN_API, "{}: Failed to create duplicate graph for graph {}", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    m_graphEntries.registerGraph(newGraphHandle, std::move(newGraph));
    GraphId newGraphId = (*newGraphHandle)->graphId;
    SmallMap<std::map<synSectionHandle, synSectionHandle>, MAX_TENSOR_NR> sectionMapping;
    GraphEntriesContainer::SectionHandleVec                               newSections;
    newSections.reserve(*numTensors);
    for (int i = 0; i < *numTensors; i++)
    {
        const SlotMapItemSptr<InternalSectionHandle> origSectionPtr = tensorsPtrMapVec[i].origTensor->getSectionHandle();

        if (origSectionPtr != nullptr)
        {
            synSectionHandle                       newSection = nullptr;
            SlotMapItemSptr<InternalSectionHandle> newSectionPtr;
            const synSectionHandle                 origSection = origSectionPtr->getSectionHandle();
            auto                                   sectionIter = sectionMapping.find(origSection);

            if (sectionIter == sectionMapping.end())
            {
                auto [handle, ptr] = m_sectionHndlSlopMap.insert(0, newGraphId, *origSectionPtr);
                if (ptr == nullptr)
                {
                    LOG_ERR(SYN_API,
                            "{}: Failed to create duplicate graph for graph {}, section duplication failed",
                            HLLOG_FUNC,
                            graphHandle->graphId);
                    // cleanup on failure
                    for (synSectionHandle newlyCreatedSection : newSections)
                    {
                        m_sectionHndlSlopMap.erase((SMHandle)newlyCreatedSection);
                    }
                    m_graphEntries.destroyGraph(*newGraphHandle);
                    return synFail;
                }
                ptr->setSectionHandle((synSectionHandle)handle);
                newSection    = (synSectionHandle)handle;
                newSectionPtr = ptr;
                sectionMapping.emplace(origSection, newSection);
                newSections.push_back(newSection);
            }
            else
            {
                newSection    = sectionIter->second;
                newSectionPtr = m_sectionHndlSlopMap[(SMHandle)newSection];
            }
            tensorUpdateSectionInfoInternal(reinterpret_cast<synTensor>(tensorsPtrMapVec[i].newTensor.get()),
                                            newSection,
                                            newSectionPtr,
                                            tensorsPtrMapVec[i].origTensor->getMemorySectionOffset());
        }
        tensorsPtrMapVec[i].newTensor->setDuplicatedTensor();  // mark the tensor as duplicated
        tensorsPtrMapVec[i].newTensor->setGraphID(newGraphId);  // associate the new tensor with the duplicated graph
        tensorsMap[i] = {reinterpret_cast<synTensor>(tensorsPtrMapVec[i].origTensor),
                         reinterpret_cast<synTensor>(tensorsPtrMapVec[i].newTensor.get())};
    }
    m_graphEntries.setSections(newGraphId, newSections);
    m_graphEntries.setTensorSharedPtr(newGraphId, tensorsPtrMapVec);

    for (int i = 0; i < *numNodes; i++)
    {
        nodesMap[i] = {nodesMapVec[i].origHandle, nodesMapVec[i].newHandle};
    }

    LOG_DEBUG(SYN_GRAPH, "{}: created duplicate graph {} for graph {}", HLLOG_FUNC, newGraphId, graphHandle->graphId);
    return synSuccess;
}

synStatus synSingleton::inferGraphShapes(const synGraphHandle graphHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");

    HabanaGraph* graph = nullptr;
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        graph = m_graphEntries[graphHandle];
    }

    if (unlikely(!graph))
    {
        LOG_ERR(SYN_API, "{}: Failed to find graph {} for shape inference", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    if (unlikely(!graph->wasCreatedUsingDuplicateAPI()))
    {
        LOG_ERR(SYN_API, "{}: Graph {} was not created using duplicate API", HLLOG_FUNC, graphHandle->graphId);
        return synFail;
    }

    auto res = graph->performMaxShapeInference() ? synSuccess : synFail;

    LOG_DEBUG(SYN_API, "{}: Shape Inference for graph {} result: {}", HLLOG_FUNC, graphHandle->graphId, res);
    return res;
}

synStatus synSingleton::getGraphDeviceType(const synGraphHandle graphHandle, synDeviceType* deviceType)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        *deviceType = m_graphEntries.getDeviceType(graphHandle);
    }
    if (synDeviceTypeInvalid == *deviceType)
    {
        LOG_ERR(SYN_API, "{}: Unable to get graph type", HLLOG_FUNC);
        return synFail;
    }
    LOG_DEBUG(SYN_GRAPH, "{}: Device type of Graph handle is {} ", HLLOG_FUNC, _deviceTypeToStrings(*deviceType)[0]);

    return synSuccess;
}

// Temporary code for internal use
synStatus synSingleton::setGraphDeviceAddress(int32_t devIdx, uint64_t size, uint64_t buffer)
{
    LOG_TRACE(SYN_API, "{} setting graph recipe addr to {}", HLLOG_FUNC, buffer);
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    uint32_t                     graphId;

    if (synSuccess != m_graphEntries.getCurrentGraphId(graphId))
    {
        LOG_ERR(SYN_API, "{} Can not set graph ID {}", HLLOG_FUNC, graphId);
        return synFail;
    }

    HabanaGraph* graph = m_graphEntries[graphId];
    if (graph)
    {
        graph->getCodeGenerator()->initDram(size, buffer);
    }
    else
    {
        LOG_ERR(SYN_API, "{} Can not set graph address, graph not found", HLLOG_FUNC);
        return synFail;
    }
    return synSuccess;
}

HabanaGraph* synSingleton::getGraph(const synGraphHandle graphHandle)
{
    CHECK_POINTER(SYN_API, graphHandle, "graphHandle", nullptr);
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    auto                         returnValue = m_graphEntries[graphHandle];
    if (nullptr == returnValue)
    {
        LOG_ERR(SYN_API, "{} failed", HLLOG_FUNC);
    }
    return returnValue;
}

/*
 ***************************************************************************************************
 *   @brief Compile the current graph
 *          After the compilation is done, save the compiled graph and clear
 *          the current graph handler.
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus synSingleton::compileGraph(synRecipeHandle*     pRecipeHandle,
                                     const synGraphHandle graphHandle,
                                     const char*          fileName,
                                     const char*          buildLog)
{
    VERIFY_IS_NULL_POINTER(SYN_GRAPH, pRecipeHandle, "pRecipeHandle");

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    HabanaGraph*                 graph = _getGraphForCompilation(graphHandle);
    if (GCFG_ENABLE_PARALLEL_COMPILATION.value())
    {
        guard.unlock();
    }

    if (graph == nullptr)
    {
        return synFail;
    }

    InternalRecipeHandle* pInternalRecipeHandle = nullptr;
    synStatus             status =
        m_recipeManager
            .addRecipeHandleAndCompileGraph(graph, false, nullptr, 0, fileName, buildLog, pInternalRecipeHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Failed to add recipe handle into Recipe-Singleton status {}", HLLOG_FUNC, status);
        return status;
    }

    // We expect that the GC-compiled program will contain content for each of the allocated execution-streams,
    // We will not check that
    // Otherwise the ARB feature will collapse...
    VERIFY_IS_NULL_POINTER(SYN_API, pInternalRecipeHandle->basicRecipeHandle.recipe, "recipe");

    *pRecipeHandle = pInternalRecipeHandle;

    LOG_DEBUG_T(SYN_API,
                "Recipe 0x{:x} Name {}",
                TO64(pInternalRecipeHandle->basicRecipeHandle.recipe),
                graph->getRecipeName());

    return synSuccess;
}

synStatus synSingleton::_createGenericNode(const synGraphHandle graphHandle,
                                           const synTensor*     inputs,
                                           const synTensor*     outputs,
                                           const uint32_t       sizeInputs,
                                           const uint32_t       sizeOutputs,
                                           const void*          userParams,
                                           const unsigned       paramsSize,
                                           const char*          guid,
                                           const char**         inputLayouts,
                                           const char**         outputLayouts,
                                           const std::string&   name,
                                           synNodeId*           nodeUniqueId)
{
    void* graphParams;

    // validate arguments
    if (inputs == nullptr && sizeInputs != 0)
    {
        LOG_ERR(SYN_API, "{}: Invalid input arguments, will not create generic node {}", HLLOG_FUNC, name);
        return synInvalidArgument;
    }
    if (outputs == nullptr && sizeOutputs != 0)
    {
        LOG_ERR(SYN_API, "{}: Invalid output arguments, will not create generic node {}", HLLOG_FUNC, name);
        return synInvalidArgument;
    }
    if (guid == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Invalid guid, will not create generic node {}", HLLOG_FUNC, name);
        return synInvalidArgument;
    }

    auto fillTensorVector = [this, &name](const synTensor* src, uint32_t srcSize, TensorVector& dst) {
        dst.reserve(srcSize);
        for (uint32_t i = 0; i < srcSize; i++)
        {
            dst.emplace_back(m_graphEntries.findTensor(src[i]));
            if (!dst.back() && src[i])
            {
                LOG_ERR(SYN_API, "{}: Failed creating node {}. input tensor index {} not found", HLLOG_FUNC, name, i);
                return synInvalidArgument;
            }
        }
        return synSuccess;
    };

    HabanaGraph* graph;
    TensorVector inputTensorPtrs;
    TensorVector outputTensorPtrs;
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        graph = m_graphEntries[graphHandle];
        if (graph == nullptr)
        {
            LOG_ERR(SYN_API, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
            return synFail;
        }
        synStatus result = fillTensorVector(inputs, sizeInputs, inputTensorPtrs);
        if (result != synSuccess) return result;
        result = fillTensorVector(outputs, sizeOutputs, outputTensorPtrs);
        if (result != synSuccess) return result;
    }

    LOG_DEBUG(SYN_GRAPH,
              "{}: creating generic node: guid - {}, name - {}, number of input tensors {}, "
              "number of output tensors - {}",
              HLLOG_FUNC,
              guid,
              name,
              inputTensorPtrs.size(),
              outputTensorPtrs.size());

    const uint32_t graphId = graphHandle->graphId;

    for (const auto& tensorVec : {inputTensorPtrs, outputTensorPtrs})
    {
        for (const auto& tensor : tensorVec)
        {
            if (tensor)
            {
                if (!isTensorValidForGraph(tensor, graphId))
                {
                    return synInvalidArgument;
                }
                if (!tensor->isPropsValid())
                {
                    LOG_ERR(SYN_API,
                            "{}: Failed creating node {}. tensor {} props not valid",
                            HLLOG_FUNC,
                            name,
                            tensor->getName());
                    return synInvalidArgument;
                }
                tensor->lockPropsAndFinalizeTensor();
            }
        }
    }

    graphParams = const_cast<void*>(userParams);

    Node::NodeProperties properties;
    auto                 fillNodeProperties =
        [&name](const TensorVector& tensors, const char** srcLayouts, LayoutVector& dstLayouts) {
            for (int i = 0; i < tensors.size(); i++)
            {
                if (srcLayouts != nullptr && srcLayouts[i] != nullptr)
                {
                    if (dstLayouts.empty())
                    {
                        dstLayouts.resize(tensors.size());
                    }
                    dstLayouts[i] = Layout(srcLayouts[i]);
                    LOG_TRACE(SYN_API,
                              "{}: Node name {}, layout of tensor {} is {}",
                              HLLOG_FUNC,
                              name,
                              i,
                              dstLayouts[i].toString());
                }
            }
        };

    fillNodeProperties(inputTensorPtrs, inputLayouts, properties.inputLayouts);
    fillNodeProperties(outputTensorPtrs, outputLayouts, properties.outputLayouts);

    StringWithHash guidAndHash(guid);
    if (NodeFactory::isInternalNode(guidAndHash) && GCFG_GAUDI_DEMO.value() == false &&
        !GCFG_ENABLE_INTERNAL_NODES.value())
    {
        LOG_ERR(SYN_API,
                "{}: Failed creating node {} with GUID {}: creating internal nodes not enabled",
                HLLOG_FUNC,
                name,
                guidAndHash.getKey());
        return synFail;
    }
    NodePtr node;
    try
    {
        auto deviceType = graph->getDeviceType();
        node            = NodeFactory::createNode(inputTensorPtrs,
                                       outputTensorPtrs,
                                       (const UserParams)graphParams,
                                       paramsSize,
                                       guid,
                                       name,
                                       properties,
                                       &deviceType);
    }
    catch (InvalidNodeParamsException& e)
    {
        LOG_ERR(SYN_API, "{}: Can not create {} generic node ({})", HLLOG_FUNC, guid, name);
        return synInvalidArgument;
    }

    if (node == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Can not create {} generic node ({})", HLLOG_FUNC, guid, name);
        return synFail;
    }

    // Check N-Dims only if it can't be handled by CGUID.
    bool checkNDims = !KernelDB::instance().isSupportedComplexGuid(guidAndHash, graph->getDeviceId());
    for (const auto& tensorVec : {inputTensorPtrs, outputTensorPtrs})
    {
        for (auto& t : tensorVec)
        {
            if (t && t->getDim() > SYN_MAX_TENSOR_DIM && !isTensorDimsValidForNode(t, node, checkNDims))
            {
                return synInvalidTensorDimensions;
            }
        }
    }

    // setParentId should be false to allow pre graph order be applied to parent ID of the original nodes.
    // If set to true all nodes get the same parent ID.
    if (!GraphEditor::addNode(*graph, node, false /*setParentId*/))
    {
        LOG_ERR(SYN_API, "{}: Can not add {} generic node ({}) to graph", HLLOG_FUNC, guid, name);
        return synFail;
    }

    if (nodeUniqueId != nullptr)
    {
        *nodeUniqueId = node->getId();
    }

    return synSuccess;
}

synStatus synSingleton::createGenericNode(const synGraphHandle graphHandle,
                                          const synTensor*     inputs,
                                          const synTensor*     outputs,
                                          const uint32_t       sizeInputs,
                                          const uint32_t       sizeOutputs,
                                          const void*          userParams,
                                          const unsigned       paramsSize,
                                          const char*          guid,
                                          const char**         inputLayouts,
                                          const char**         outputLayouts,
                                          const std::string&   name)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    return _createGenericNode(graphHandle,
                              inputs,
                              outputs,
                              sizeInputs,
                              sizeOutputs,
                              userParams,
                              paramsSize,
                              guid,
                              inputLayouts,
                              outputLayouts,
                              name);
}

synStatus synSingleton::createGenericNodeWithId(const synGraphHandle graphHandle,
                                                const synTensor*     inputs,
                                                const synTensor*     outputs,
                                                const uint32_t       sizeInputs,
                                                const uint32_t       sizeOutputs,
                                                const void*          userParams,
                                                const unsigned       paramsSize,
                                                const char*          guid,
                                                const char**         inputLayouts,
                                                const char**         outputLayouts,
                                                const std::string&   name,
                                                synNodeId*           nodeUniqueId)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    return _createGenericNode(graphHandle,
                              inputs,
                              outputs,
                              sizeInputs,
                              sizeOutputs,
                              userParams,
                              paramsSize,
                              guid,
                              inputLayouts,
                              outputLayouts,
                              name,
                              nodeUniqueId);
}

synStatus
synSingleton::nodeTypeSetUserPrecision(const synGraphHandle graphHandle, const char* guid, synDataType nodePrecision)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    HabanaGraph* graph;
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        graph = m_graphEntries[graphHandle];
    }

    if (graph == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return synFail;
    }

    graph->setUserNodeTypePrecision(guid, nodePrecision);
    return synSuccess;
}

NodePtr synSingleton::_findNode(synNodeId nodeUniqueId, uint32_t graphId)
{
    HabanaGraph* graph = m_graphEntries[graphId];
    HB_ASSERT(graph != nullptr, "The graph does not exist");
    return graph->getNodeByID(nodeUniqueId);
}

NodePtr synSingleton::_findNodeLockRequired(synNodeId nodeUniqueId, uint32_t graphId, const synNodeId nodeId)
{
    HabanaGraph* graph;
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        graph = m_graphEntries[graphId];
    }
    if (graph == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return NodePtr();
    }
    NodePtr node = graph->getNodeByID(nodeUniqueId);
    if (node == nullptr)
    {
        LOG_ERR(SYN_API, "{}: nodeId {} is incorrect.", HLLOG_FUNC, nodeId);
        return NodePtr();
    }
    return node;
}

void synSingleton::_destroyAllGraphs()
{
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    m_graphEntries.destroyAllGraphs();
}

synStatus synSingleton::destroyGraph(const synGraphHandle graphHandle)
{
    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");

    HabanaGraphPtr graphPtr;
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        graphPtr = m_graphEntries.destroyGraph(graphHandle);
        if (!graphPtr)
        {
            return synFail;
        }
    }
    // actual graph destructor is called outside of the lock scope
    const auto nbNodes = graphPtr->getNumNodes();
    graphPtr.reset();
    delete graphHandle;
    // large graphs create glibc heap fragmentation (objects have different liftime)
    // glibc does not handle fragmentation automatically (jemalloc does)
    // malloc_trim returns unused memory to OS
    // don't call malloc_trim for small graphs because of a possible performance impact
    const unsigned mallocTrimNbNodesThreshold = 1000;
    if (nbNodes > mallocTrimNbNodesThreshold)
    {
        int r = malloc_trim(10);
        LOG_TRACE(SYN_API, "heap defragmentation. malloc_trim return {}", r);
    }
    return synSuccess;
}

synStatus synSingleton::createControlDependency(const synGraphHandle graphHandle,
                                                const synNodeId*     pBlockingNodesIdList,
                                                const synNodeId*     pBlockedNodesIdList,
                                                const uint32_t       numberblocking,
                                                const uint32_t       numberblocked)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    auto                         graph = m_graphEntries[graphHandle];
    if (graph == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return synFail;
    }
    return _createControlDependency(graph,
                                    graphHandle->graphId,
                                    synDeviceGaudi,
                                    pBlockingNodesIdList,
                                    pBlockedNodesIdList,
                                    numberblocking,
                                    numberblocked);
}

synStatus synSingleton::nodeSetDeterministic(const synGraphHandle graphHandle,
                                             const synNodeId      nodeId,
                                             const bool           useDeterministic)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");

    NodePtr node = _findNodeLockRequired(nodeId, graphHandle->graphId, nodeId);
    if (!node) return synFail;
    node->setDeterministic(useDeterministic);
    return synSuccess;
}

synStatus
synSingleton::nodeGetDeterministic(const synGraphHandle graphHandle, const synNodeId nodeId, bool* pUseDeterministic)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, pUseDeterministic, "pUseDeterministic");

    NodePtr node = _findNodeLockRequired(nodeId, graphHandle->graphId, nodeId);
    if (!node) return synFail;
    *pUseDeterministic = node->getDeterministic();
    return synSuccess;
}

synStatus synSingleton::nodeSetRoundingMode(const synGraphHandle  graphHandle,
                                            const synNodeId       nodeId,
                                            const synRoundingMode roundingMode)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");

    NodePtr node = _findNodeLockRequired(nodeId, graphHandle->graphId, nodeId);
    if (!node) return synFail;

    if (!HabanaGraph::runsOnMME(node))
    {
        LOG_ERR(SYN_API, "{}: Changing rounding mode isn't supported in non-MME node. nodeId: {}", HLLOG_FUNC, nodeId);
        return synFail;
    }

    node->setRoundingMode(roundingMode);
    return synSuccess;
}

synStatus synSingleton::nodeGetRoundingMode(const synGraphHandle graphHandle,
                                            const synNodeId      nodeId,
                                            synRoundingMode*     pRoundingMode)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, pRoundingMode, "pRoundingMode");

    NodePtr node = _findNodeLockRequired(nodeId, graphHandle->graphId, nodeId);
    if (!node) return synFail;

    if (!HabanaGraph::runsOnMME(node))
    {
        LOG_ERR(SYN_API,
                "{}: Retrieving rounding mode isn't supported in non-MME node. nodeId: {}",
                HLLOG_FUNC,
                nodeId);
        return synFail;
    }

    *pRoundingMode = node->getRoundingMode();
    return synSuccess;
}

synStatus synSingleton::insertDescendantsIds(NodeSet& nodeSet, synNodeId complexGuidNodeId, uint32_t graphId)
{
    const nodeIdSet& nodeDescendants = m_graphEntries.getNodeDescendantsIds(complexGuidNodeId, graphId);
    for (const synNodeId& nodeId : nodeDescendants)
    {
        LOG_DEBUG(SYN_GRAPH,
                  "Adding node with id {} to control dependency set as "
                  "descendant of complex guid node with id {}",
                  nodeId,
                  complexGuidNodeId);
        NodePtr descendantNode = _findNode(nodeId, graphId);
        if (descendantNode != nullptr)
        {
            nodeSet.insert(descendantNode);
            continue;
        }
        // if descendant not in graph - it's another complex guid node, set the descendants recursively
        HB_ASSERT(m_graphEntries.isComplexGuidWithDescendants(nodeId, graphId),
                  "Descendant node with id {} not found in graph with id {} and isn't a complex guid node",
                  nodeId,
                  graphId);
        LOG_DEBUG(SYN_GRAPH, "Descendant node with id {} is a complex guid node. graphId: {}", nodeId, graphId);
        insertDescendantsIds(nodeSet, nodeId, graphId);
    }
    return synSuccess;
}

synStatus synSingleton::getUniqueNodeId(synNodeId& nodeId)
{
    nodeId = Node::getUniqueId();
    return synSuccess;
}

synStatus
synSingleton::setOriginalComplexNode(synGraphHandle graphHandle, synNodeId nodeId, synNodeId originalComplexNodeId)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);
    synStatus status = synSuccess;

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    status = m_graphEntries.setNodeDescendant(graphHandle->graphId, originalComplexNodeId, nodeId);
    return status;
}

synStatus synSingleton::validateControlDependencyDevice(HabanaGraph* graph, NodeSet& blockingNodesSet)
{
    return synSuccess;
}

synStatus synSingleton::addNodeToControlDependencySet(NodeSet&         cdSet,
                                                      const synNodeId* pNodeIdList,
                                                      unsigned         numberOfNodes,
                                                      uint32_t         graphId)
{
    for (unsigned i = 0; i < numberOfNodes; i++)
    {
        NodePtr node = _findNode(pNodeIdList[i], graphId);
        if (node == nullptr)
        {
            if (!m_graphEntries.isComplexGuidWithDescendants(pNodeIdList[i], graphId))
            {
                LOG_ERR(SYN_API,
                        "{}: Can not find node with uniqueId {} on graphId {}",
                        HLLOG_FUNC,
                        pNodeIdList[i],
                        graphId);
                cdSet.clear();
                return synInvalidArgument;
            }
            // replace complex guid node with its descendants
            LOG_DEBUG(SYN_GRAPH, "Getting descendants of complex guid node with id {}", pNodeIdList[i]);
            insertDescendantsIds(cdSet, pNodeIdList[i], graphId);
            continue;
        }
        cdSet.insert(node);
    }

    return synSuccess;
}

synStatus synSingleton::_createControlDependency(HabanaGraph*        graph,
                                                 uint32_t            graphId,
                                                 const synDeviceType deviceType,
                                                 const synNodeId*    pBlockingNodesIdList,
                                                 const synNodeId*    pBlockedNodesIdList,
                                                 const uint32_t      numberblocking,
                                                 const uint32_t      numberblocked)
{
    if (pBlockingNodesIdList == nullptr || numberblocking == 0)
    {
        LOG_ERR(SYN_API, "{}: Invalid blocking params", HLLOG_FUNC);
        return synInvalidArgument;
    }

    if (pBlockedNodesIdList == nullptr || numberblocked == 0)
    {
        LOG_ERR(SYN_API, "{}: Invalid blocked params", HLLOG_FUNC);
        return synInvalidArgument;
    }

    NodeSet blockingNodesSet;
    NodeSet blockedNodesSet;

    synStatus status;

    status = addNodeToControlDependencySet(blockingNodesSet, pBlockingNodesIdList, numberblocking, graphId);
    if (status != synSuccess) return status;

    status = addNodeToControlDependencySet(blockedNodesSet, pBlockedNodesIdList, numberblocked, graphId);
    if (status != synSuccess) return status;

    status = validateControlDependencyDevice(graph, blockingNodesSet);
    if (status != synSuccess) return status;

    NodeVector     intersection;
    NodeComparator comp;

    std::set_intersection(blockingNodesSet.begin(),
                          blockingNodesSet.end(),
                          blockedNodesSet.begin(),
                          blockedNodesSet.end(),
                          std::back_inserter(intersection),
                          comp);

    if (intersection.size() != 0)
    {
        LOG_ERR(SYN_API, "{}: Some nodes appears both in blocking and in blocked lists", HLLOG_FUNC);
        return synInvalidArgument;
    }

    graph->addControlDependency(blockingNodesSet, blockedNodesSet);

    return synSuccess;
}

synStatus synSingleton::nodeSetParams(const synGraphHandle graphHandle,
                                      const synNodeId      nodeId,
                                      const void*          userParams,
                                      const unsigned       paramsSize)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    if ((userParams == nullptr) != (paramsSize == 0))
    {
        LOG_ERR(SYN_API, "{}: userParams and paramSize are incompatible.", HLLOG_FUNC);
        return synFail;
    }

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    auto                         graph = m_graphEntries[graphHandle];
    if (graph == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return synFail;
    }

    NodePtr node = _findNode(nodeId, graphHandle->graphId);

    if (node == nullptr)
    {
        LOG_ERR(SYN_API, "{}: nodeId {} is incorrect.", HLLOG_FUNC, nodeId);
        return synFail;
    }

    node->setParamsRawData(const_cast<void*>(userParams), paramsSize);
    node->setParams(const_cast<void*>(userParams), paramsSize);
    return synSuccess;
}

synStatus synSingleton::nodeGetParams(const synGraphHandle graphHandle,
                                      const synNodeId      nodeId,
                                      void*                userParams,
                                      unsigned*            paramsSize)
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_API, graphHandle, "graphHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, paramsSize, "paramsSize");

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    auto                         graph = m_graphEntries[graphHandle];
    if (graph == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Graph Handle is incorrect.", HLLOG_FUNC);
        return synFail;
    }

    NodePtr node = _findNode(nodeId, graphHandle->graphId);

    if (node == nullptr)
    {
        LOG_ERR(SYN_API, "{}: nodeId {} is incorrect.", HLLOG_FUNC, nodeId);
        return synFail;
    }

    const auto&          rawData = node->getParamsRawData();
    size_t               size    = rawData.size();

    if (userParams == nullptr)
    {
        *paramsSize = size;
        return synSuccess;
    }
    else if (*paramsSize != size)
    {
        LOG_ERR(SYN_API,
                "{}: params size 0x{:x} for nodeId {} is incorrect and should be {}.",
                HLLOG_FUNC,
                TO64(paramsSize),
                nodeId,
                size);
        return synFail;
    }

    std::memcpy(const_cast<void*>(userParams), rawData.data(), size);
    return synSuccess;
}

#pragma once

#include "data_serializer/data_serializer.h"
#include "graph_serializer/graph_serializer.h"
#include "shim_json_types.hpp"
#include "shim_fcn.hpp"
#include "syn_singleton_interface.hpp"
#include <atomic>
#include <memory>
#include <unordered_map>

extern "C" {
uint32_t GraphCompiler_Load();
void     GraphCompiler_Unload();

ShimFunctions SYNAPSE_GraphCompiler_Init(ShimFunctions pDefaultFunctions);
void          SYNAPSE_GraphCompiler_Fini();
const char*   SYNAPSE_GraphCompiler_GetVersion();
bool          GraphCompiler_GetScheme(const char* schemeName, json* scheme);
const char**  GraphCompiler_GetSchemeNames();
void          GraphCompiler_SetSchemeValues(const std::string& values);
const char**  GetPluginsNames();
};

namespace gc_shim
{
class GraphCompilerSingleton : public synSingletonInterface
{
    using DataSerializerPtr = std::shared_ptr<data_serialize::DataSerializer>;

public:
    explicit GraphCompilerSingleton(synSingletonInterface* interface);
    GraphCompilerSingleton(const GraphCompilerSingleton&) = delete;
    virtual ~GraphCompilerSingleton()                     = default;

    GraphCompilerSingleton& operator=(const GraphCompilerSingleton&) = delete;

    synStatus initialize() override;
    synStatus destroy() override;

    virtual synStatus enqueue(const synStreamHandle      streamHandle,
                              const synLaunchTensorInfo* enqueueInputTensorsInfo,
                              const uint32_t             inputSize,
                              const synLaunchTensorInfo* enqueueOutputTensorsInfo,
                              const uint32_t             outputSize,
                              uint64_t                   pWorkspace,
                              const synRecipeHandle      pRecipeHandle,
                              uint32_t                   flags) override;

    virtual synStatus enqueue(const synStreamHandle      streamHandle,
                              const synLaunchTensorInfo* enqueueTensorsInfo,
                              const uint32_t             enqueueTensorsAmount,
                              uint64_t                   pWorkspace,
                              const synRecipeHandle      pRecipeHandle,
                              uint32_t                   flags) override;

    virtual synStatus enqueueWithExternalEvents(const synStreamHandle      streamHandle,
                                                const synLaunchTensorInfo* enqueueTensorsInfo,
                                                const uint32_t             enqueueTensorsAmount,
                                                uint64_t                   pWorkspace,
                                                const synRecipeHandle      pRecipeHandle,
                                                synEventHandle*            eventHandleList,
                                                uint32_t                   numberOfEvents,
                                                uint32_t                   flags) override;
    // deprecated
    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueInputTensorsInfo,
                              const uint32_t                inputSize,
                              const synLaunchTensorInfoExt* enqueueOutputTensorsInfo,
                              const uint32_t                outputSize,
                              uint64_t                      pWorkspace,
                              const synRecipeHandle         pRecipeHandle,
                              uint32_t                      flags) override;
    // deprecated
    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                              const uint32_t                enqueueTensorsAmount,
                              uint64_t                      pWorkspace,
                              const synRecipeHandle         pRecipeHandle,
                              uint32_t                      flags) override;

    // deprecated
    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                              const uint32_t                enqueueTensorsAmount,
                              uint64_t                      pWorkspace,
                              const synRecipeHandle         pRecipeHandle,
                              synEventHandle*               eventHandleList,
                              uint32_t                      numberOfEvents,
                              uint32_t                      flags) override;

    synStatus enqueueWithExternalEventsExt(const synStreamHandle         streamHandle,
                                           const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                           const uint32_t                enqueueTensorsAmount,
                                           uint64_t                      workspaceAddress,
                                           const synRecipeHandle         pRecipeHandle,
                                           synEventHandle*               eventHandleList,
                                           uint32_t                      numberOfEvents,
                                           uint32_t                      flags) override;

    virtual synStatus
    createGraph(synGraphHandle* pGraphHandle, synDeviceType deviceType, CompilationMode compilationMode) override;

    virtual synStatus destroyGraph(const synGraphHandle graphHandle) override;

    virtual synStatus duplicateGraph(synGraphHandle      graphHandle,
                                     synGraphHandle*     newGraphHandle,
                                     synTensorHandleMap* tensorsMap,
                                     uint32_t*           numTensors,
                                     synNodeHandleMap*   nodesMap,
                                     uint32_t*           numNodes) override;

    virtual synStatus createGenericNode(const synGraphHandle graphHandle,
                                        const synTensor*     pInputsTensorList,
                                        const synTensor*     outputs,
                                        const uint32_t       sizeInputs,
                                        const uint32_t       sizeOutputs,
                                        const void*          userParams,
                                        const unsigned       paramsSize,
                                        const char*          guid,
                                        const char**         inputLayouts,
                                        const char**         outputLayouts,
                                        const std::string&   name = "") override;

    virtual synStatus createGenericNodeWithId(const synGraphHandle graphHandle,
                                              const synTensor*     pInputsTensorList,
                                              const synTensor*     outputs,
                                              const uint32_t       sizeInputs,
                                              const uint32_t       sizeOutputs,
                                              const void*          userParams,
                                              const unsigned       paramsSize,
                                              const char*          guid,
                                              const char**         inputLayouts,
                                              const char**         outputLayouts,
                                              const std::string&   name         = "",
                                              synNodeId*           nodeUniqueId = nullptr) override;

    virtual synStatus createControlDependency(const synGraphHandle graphHandle,
                                              const synNodeId*     pBlockingNodesIdList,
                                              const synNodeId*     pBlockedNodesIdList,
                                              const uint32_t       numberblocking,
                                              const uint32_t       numberblocked) override;

    virtual synStatus nodeSetDeterministic(const synGraphHandle graphHandle,
                                           const synNodeId      nodeId,
                                           const bool           useDeterministic) override;

    virtual synStatus
    nodeGetDeterministic(const synGraphHandle graphHandle, const synNodeId nodeId, bool* pUseDeterministic) override;

    virtual synStatus nodeSetRoundingMode(const synGraphHandle  graphHandle,
                                          const synNodeId       nodeId,
                                          const synRoundingMode roundingMode) override;

    virtual synStatus nodeGetRoundingMode(const synGraphHandle graphHandle,
                                          const synNodeId      nodeId,
                                          synRoundingMode*     pRoundingMode) override;

    virtual synStatus nodeSetParams(const synGraphHandle graphHandle,
                                    const synNodeId      nodeId,
                                    const void*          userParams,
                                    const unsigned       paramsSize) override;

    /*DEPRECATED*/
    virtual synStatus graphSetAttribute(const synGraphHandle     graphHandle,
                                        const synGraphAttribute* attributes,
                                        const uint64_t*          values,
                                        const uint32_t           size) override;

    virtual synStatus graphSetAttributes(const synGraphHandle        graphHandle,
                                         const synGraphAttribute*    attributes,
                                         const synGraphAttributeVal* values,
                                         const uint32_t              size) override;

    virtual synStatus compileGraph(synRecipeHandle*     pRecipeHandle,
                                   const synGraphHandle graphHandle,
                                   const char*          fileName,
                                   const char*          buildLog) override;

private:
    std::optional<uint16_t> getRecipeId(const synRecipeHandle recipeHandle);

    template<class T>
    synStatus enqueueAndCapture(const synStreamHandle         streamHandle,
                                const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                const uint32_t                enqueueTensorsAmount,
                                const synRecipeHandle         pRecipeHandle,
                                T                             callback);

    void serializeLaunchTensors(const synStreamHandle                        streamHandle,
                                std::vector<data_serialize::TensorMetadata>& tensorsMetadata,
                                const DataSerializerPtr                      dataSerializer,
                                bool                                         input);

    bool validateTensors(const synStreamHandle                        streamHandle,
                         std::vector<data_serialize::TensorMetadata>& tensorsMetadata,
                         bool                                         input);

    template<class T>
    synStatus dumpToFile(const std::string&                           recipeName,
                         const synStreamHandle                        streamHandle,
                         std::vector<data_serialize::TensorMetadata>& tensorsMetadata,
                         const DataSerializerPtr                      dataSerializer,
                         T                                            callback);

    std::vector<data_serialize::TensorMetadata> getTensorsMetadata(const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                                   const uint32_t                enqueueTensorsAmount,
                                                                   const synRecipeHandle         pRecipeHandle,
                                                                   uint64_t                      launchIndex);

    std::shared_ptr<uint64_t> getTensorData(const uint32_t        deviceId,
                                            const synTensorType&  tensorType,
                                            const uint64_t        tensorAddress,
                                            const uint64_t        tensorDataSize,
                                            const synStreamHandle stream);

    bool skipGraphRecording(const std::string& recipeName);
    bool skipTensorRecording(const std::string& recipeName);

    const uint64_t                                    m_rankId;
    std::unique_ptr<graph_serialize::GraphSerializer> m_graphSerializer;
    std::atomic<std::uint64_t>                        m_launchCounter;
};

}  // namespace gc_shim
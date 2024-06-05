#pragma once

#include "syn_singleton_interface.hpp"

#include <memory>
#include <mutex>
#include <vector>

#include "define_synapse_common.hpp"
#include "transpose_permutation.h"
#include "movable_atomic.hpp"
#include "device/device_manager.hpp"
#include "graph_entries_container.hpp"
#include "syn_event_dispatcher.hpp"
#include "syn_api_stat.hpp"
#include "statistics.hpp"
#include "infra/containers/slot_map_alloc.hpp"
#include "hcl_public_streams.h"
#include "host_to_virtual_address_mapper.hpp"
#include "global_statistics.hpp"
#include "runtime/common/common_types.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"
#include "runtime/qman/common/qman_types.hpp"

class CommandSubmissionDataChunks;
class HabanaGraph;
class DeviceInterface;
struct StagedInfo;
class EventInterface;

typedef void* libHandle;

#define _SYN_SINGLETON_                     synSingleton::getInstance()
#define _SYN_SINGLETON_INTERNAL             synSingleton::getInstanceInternal()

class synSingleton : public synSingletonInterface
{
    friend class CoeffTableConfManager;
    friend class SynTest;

public:
    static synSingletonInterface* getInstance();

    static synSingleton*          getInstanceInternal()
    {
        if (m_pInstanceInternal == nullptr)
        {
            LOG_CRITICAL(SYN_API, "getInstanceInternal call without previous synInitialize");
            throw SynapseStatusException("getInstanceInternal call without previous synInitialize", synUninitialized);
        }
        return m_pInstanceInternal;
    }

    static bool                   isSynapseInitialized() { return (m_pInstanceInternal != nullptr); }
    static synStatus              initializeInstance();
    static synStatus              destroyInstance();

    ~synSingleton() override;

    synStatus   initialize() override;
    synStatus   destroy() override;

    synStatus preApiCallExecution(const char * apiFuncName);
    synStatus postApiCallExecution();

    synStatus   acquireDevice(          uint32_t*               pDeviceId,
                                        const char*             pciBus,
                                        synDeviceType           deviceType = synDeviceTypeInvalid,
                                        synModuleId             moduleId   = INVALID_MODULE_ID) override;

    synStatus   releaseDevice(          uint32_t                devIdx                        ) override;
    uint16_t    getNumOfAcquiredDevices() override;

    synStatus
    createGraph(synGraphHandle* pGraphHandle, synDeviceType deviceType, CompilationMode compilationMode) override;

    /*DEPRECATED*/
    synStatus graphSetAttribute(synGraphHandle           graphHandle,
                                const synGraphAttribute* attributes,
                                const uint64_t*          values,
                                uint32_t                 size) override;

    synStatus graphSetAttributes(synGraphHandle              graphHandle,
                                 const synGraphAttribute*    attributes,
                                 const synGraphAttributeVal* values,
                                 uint32_t                    size) override;

    /*DEPRECATED*/
    synStatus graphGetAttribute(synGraphHandle           graphHandle,
                                const synGraphAttribute* attributes,
                                uint64_t*                values,
                                uint32_t                 size) override;

    synStatus graphGetAttributes(synGraphHandle           graphHandle,
                                 const synGraphAttribute* attributes,
                                 synGraphAttributeVal*    values,
                                 uint32_t                 size) override;

    synStatus duplicateGraph(synGraphHandle      graphHandle,
                             synGraphHandle*     newGraphHandle,
                             synTensorHandleMap* tensorsMap,
                             uint32_t*           numTensors,
                             synNodeHandleMap*   nodesMap,
                             uint32_t*           numNodes) override;

    synStatus inferGraphShapes(const synGraphHandle graphHandle) override;

    synStatus   destroyGraph(           const synGraphHandle     graphHandle ) override;

    synStatus getGraphDeviceType(const synGraphHandle graphHandle, synDeviceType* deviceType) override;

    synStatus   getDeviceDramMemoryInfo(uint32_t  devIdx,
                                        uint64_t& free,
                                        uint64_t& total) const override;

    HabanaGraph* getGraph(const synGraphHandle          graphHandle);

    synStatus   compileGraph( synRecipeHandle*              pRecipeHandle,
                              const synGraphHandle          graphHandle,
                              const char*                   fileName,
                              const char*                   buildLog) override;


    synStatus   allocateDeviceMemory(   unsigned                devIdx,
                                        uint64_t                size,
                                        uint32_t                flags,
                                        void**                  buffer,
                                        uint64_t                reqVAAddress        = 0,
                                        uint64_t*               deviceVA            = nullptr   ) override;

    synStatus   deallocateDeviceMemory( unsigned                devIdx,
                                        void*                   pBuffer,
                                        uint32_t                flags               = synMemFlags::synMemHost ) override;

    synStatus   mapBufferToDevice(      unsigned                devIdx,
                                        uint64_t                size,
                                        void*                   buffer,
                                        uint64_t                reqVAAddress        = 0         ) override;

    synStatus   unmapBufferFromDevice(  unsigned                devIdx,
                                        void*                   buffer                          ) override;

    synTensor   createTensor(           synTensorDescriptor*    pDescriptor,
                                        synStatus*              pStatus,
                                        bool                    isOutput           = false,
                                        bool                    isInput            = false,
                                        bool                    isStaticParam      = false) override;

    synStatus   createTensor(           const synTensorDescriptor*    pDescriptor,
                                        synTensor*                    tensor,
                                        synSectionHandle              sectionHandle = nullptr,
                                        uint64_t                      sectionOffset = 0,
                                        bool                          isConst       = false) override;

    synStatus   createTensor(           synTensor*               tensor,
                                        synGraphHandle           graph,
                                        synTensorType            type,
                                        const char*              tensorName) override;

    synStatus   tensorAssignToSection(  synTensor               tensor,
                                        synSectionHandle        section,
                                        uint64_t                byteOffset) override;

    synStatus tensorSetSectionOffset(synTensor tensor, uint64_t byteOffset) override;

    synStatus tensorGetSection(synTensor tensor, synSectionHandle* section, uint64_t* byteOffset) override;

    synStatus tensorSetAllowPermutation(synTensor tensor, int8_t allow) override;

    synStatus tensorGetAllowPermutation(synTensor tensor, int8_t* allow) override;

    synStatus   tensorSetHostPtr(       synTensor               tensor,
                                        void*                   hostPtr,
                                        uint64_t                size,
                                        synDataType             dataType,
                                        bool                    copyBuffer) override;

    synStatus tensorGetHostPtr(synTensor tensor, void** hostPtr, uint64_t* size, synDataType* dataType) override;

    synStatus   tensorSetQuantizationData(      synTensor               tensor,
                                                synQuantizationProperty prop,
                                                void*                   propVal,
                                                uint64_t                propSize) override;

    synStatus tensorGetQuantizationData(synTensor               tensor,
                                        synQuantizationProperty prop,
                                        void*                   propVal,
                                        uint64_t                propSize) override;

    synStatus   tensorSetGeometry(      synTensor                   tensor,
                                        const synTensorGeometry*    geometry,
                                        synGeometryType             geometryType) override;

    synStatus tensorGetGeometry(        const synTensor         tensor,
                                        synTensorGeometry*      geometry,
                                        synGeometryType         geometryType) override;

    synStatus tensorSetIsExternal(synTensor tensor, bool isExternal) override;

    synStatus tensorGetIsExternal(const synTensor tensor, bool* isExternal) override;

    synStatus tensorSetDeviceLayout(    synTensor                    tensor,
                                        const synTensorDeviceLayout* layout) override;

    synStatus tensorGetDeviceLayout(    const synTensor         tensor,
                                        synTensorDeviceLayout*  layout) override;

    synStatus tensorGetName(const synTensor tensor, const uint64_t size, char* name) override;

    synStatus tensorGetType(const synTensor tensor, synTensorType* type) override;

    synStatus   setTensorDeviceAddr(    synTensor                     tensor,
                                        uint64_t                      addr                            ) override;

    synStatus   setGraphDeviceAddress(  int32_t                  devIdx,
                                        uint64_t                 size,
                                        uint64_t                 buffer                          ) override;

    synStatus   destroyTensor(synTensor tensor) override;

    synStatus   createGenericNode(      const synGraphHandle    graphHandle,
                                        const synTensor*        pInputsTensorList,
                                        const synTensor*        outputs,
                                        const uint32_t          sizeInputs,
                                        const uint32_t          sizeOutputs,
                                        const void*             userParams,
                                        const unsigned          paramsSize,
                                        const char*             guid,
                                        const char**            inputLayouts,
                                        const char**            outputLayouts,
                                        const std::string&      name = "") override;

    synStatus createGenericNodeWithId(const synGraphHandle graphHandle,
                                      const synTensor* pInputsTensorList,
                                      const synTensor* outputs,
                                      const uint32_t sizeInputs,
                                      const uint32_t sizeOutputs,
                                      const void* userParams,
                                      const unsigned paramsSize,
                                      const char* guid,
                                      const char** inputLayouts,
                                      const char** outputLayouts,
                                      const std::string& name = "",
                                      uint64_t* nodeUniqueId = nullptr) override;

    synStatus  nodeTypeSetUserPrecision( const synGraphHandle    graphHandle,
                                         const char*             guid,
                                         synDataType             nodePrecision) override;

    synStatus   submitCommandBuffers(   CommandSubmission*      cs,
                                        uint64_t*               handle);

    synStatus recipeDestroy(synRecipeHandle hRecipe) override;

    synStatus tensorRetrieveMetadatasInfosByNameExt(const synRecipeHandle  pRecipeHandle,
                                                    const uint32_t         numOfTensors,
                                                    TensorMetadataInfoExt* tensorsMetadataInfo) const override;

    synStatus tensorRetrievePersistentAmount(    const synRecipeHandle   pRecipeHandle,
                                                 uint32_t&               numOfTensors) const override;

    synStatus tensorRetrieveNames(               const synRecipeHandle   pRecipeHandle,
                                                 char                    tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                                                 const uint32_t          numOfTensors) const override;

    synStatus tensorRetrieveLaunchAmount(const synRecipeHandle pRecipeHandle, uint32_t& numOfTensors) const override;

    synStatus tensorRetrieveLaunchIds(const synRecipeHandle pRecipeHandle,
                                      uint64_t*             tensorsIds,
                                      const uint32_t        numOfTensors) const override;

    synStatus tensorRetrieveLaunchInfoByIdExt(const synRecipeHandle            pRecipeHandle,
                                              const uint32_t                   numOfTensors,
                                              synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo) const override;

    synStatus tensorRetrieveIds(const synRecipeHandle pRecipeHandle,
                                const char**          tensorNames,
                                uint64_t*             tensorIds,
                                const uint32_t        numOfTensors) override;

    synStatus sectionCreate(synSectionHandle* phSection, uint64_t sectionDescriptor, const synGraphHandle graph) override;

    synStatus sectionDestroy(synSectionHandle hSection) override;

    synStatus sectionGroupSet(synSectionHandle hSection, uint64_t group)          const override;
    synStatus sectionGroupGet(synSectionHandle hSection, uint64_t* group)         const override;
    synStatus sectionPersistentSet(synSectionHandle hSection, bool isPersistent)  const override;
    synStatus sectionPersistentGet(synSectionHandle hSection, bool* isPersistent) const override;
    synStatus   sectionConstSet(synSectionHandle hSection, bool isConst) const override;
    synStatus   sectionConstGet(synSectionHandle hSection, bool* isConst) const override;
    synStatus sectionRMWSet(synSectionHandle hSection, bool isRMW)                const override;
    synStatus sectionRMWGet(synSectionHandle hSection, bool* isRMW)               const override;
    synStatus sectionSetDeviceAddress(synSectionHandle hSection, uint64_t deviceAddress) const override;
    synStatus sectionGetDeviceAddress(synSectionHandle hSection, uint64_t* deviceAddress) const override;
    synStatus   getTopologyWorkspaceSize(uint64_t* pWorkspaceSize, const synRecipeHandle recipeHandle) override;
    synStatus   sectionGetProp(const synRecipeHandle  pRecipeHandle,
                               const synSectionId     sectionId,
                               const synSectionProp   prop,
                               uint64_t*              propertyPtr) const override;

    // Deprecated API, can be reused as another API. keeping function signature for ABI compatibility.
    virtual synStatus createStream(synStreamHandle*   pStreamHandle,
                                   const uint32_t     devIdx,
                                   uint32_t           streamType,
                                   const unsigned int flags) override;
    virtual synStatus
    createStream(synStreamHandle* pStreamHandle, const uint32_t devIdx, const unsigned int flags) override;

    virtual synStatus destroyStream(synStreamHandle streamHandle) override;

    virtual synStatus createEvent(synEventHandle*      pEventHandle,
                                  const uint32_t       devIdx,
                                  const unsigned int   flags) override;

    virtual synStatus destroyEvent(synEventHandle eventHandle) override;

    virtual synStatus synchronizeStream(const synStreamHandle streamHandle) override;

    virtual synStatus synchronizeAllStreams(const uint32_t devIdx) override;

    virtual synStatus synchronizeEvent(const synEventHandle eventHandle) override;

    virtual synStatus eventElapsedTime(uint64_t*               pNanoseconds,
                                       const synEventHandle    eventHandleStart,
                                       const synEventHandle    eventHandleEnd) override;

    virtual synStatus eventRecord(synEventHandle        eventHandle,
                                  const synStreamHandle streamHandle) override;

    virtual synStatus eventMapTensorBaseExt(synEventHandle*               eventHandle,
                                            size_t                        numOfEvents,
                                            const synLaunchTensorInfoExt* launchTensorsInfo,
                                            const synRecipeHandle         recipeHandle) const override;

    // this function is for testing only
    EventInterface* getEventInterface(synEventHandle eventHandle);

    virtual synStatus externalTensorsExtractExecutionOrder(const synRecipeHandle recipeHandle,
                                                           uint32_t              numOfEvents,
                                                           uint64_t*             tensorIds) const override;
    virtual synStatus streamWaitEvent(const synStreamHandle streamHandle,
                                      synEventHandle        eventHandle,
                                      const unsigned int    flags) override;

    virtual synStatus eventQuery(const synEventHandle eventHandle) override;

    virtual synStatus streamQuery(const synStreamHandle streamHandle) override;

    virtual synStatus recipeSerialize(  const synRecipeHandle recipeHandle,
                                        const char*           recipeFileName) override;

    virtual synStatus recipeDeSerialize(synRecipeHandle*      pRecipeHandle,
                                        const char*           recipeFileName) override;


    virtual synStatus recipeGetAttribute( uint64_t*                 retVal,
                                          const synRecipeAttribute* recipeAttr,
                                          const unsigned            querySize,
                                          const synRecipeHandle     recipeHandle) override;

    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueInputTensorsInfo,
                              const uint32_t                inputSize,
                              const synLaunchTensorInfoExt* enqueueOutputTensorsInfo,
                              const uint32_t                outputSize,
                              uint64_t                      workspaceAddress,
                              const synRecipeHandle         pRecipeHandle,
                              uint32_t                      flags) override;

    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                              const uint32_t                enqueueTensorsAmount,
                              uint64_t                      workspaceAddress,
                              const synRecipeHandle         pRecipeHandle,
                              synEventHandle*               eventHandleList,
                              uint32_t                      numberOfEvents,
                              uint32_t                      flags) override;

    virtual synStatus enqueueWithExternalEventsExt(const synStreamHandle         streamHandle,
                                                   const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                   const uint32_t                enqueueTensorsAmount,
                                                   uint64_t                      workspaceAddress,
                                                   const synRecipeHandle         pRecipeHandle,
                                                   synEventHandle*               eventHandleList,
                                                   uint32_t                      numberOfEvents,
                                                   uint32_t                      flags) override;

    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                              const uint32_t                enqueueTensorsAmount,
                              uint64_t                      workspaceAddress,
                              const synRecipeHandle         pRecipeHandle,
                              uint32_t                      flags) override;

    virtual synStatus enqueue(const synStreamHandle      streamHandle,
                              const synLaunchTensorInfo* enqueueInputTensorsInfo,
                              const uint32_t             inputInfoSize,
                              const synLaunchTensorInfo* enqueueOutputTensorsInfo,
                              const uint32_t             outputInfoSize,
                              uint64_t                   workspaceAddress,
                              const synRecipeHandle      pRecipeHandle,
                              uint32_t                   flags) override;

    virtual synStatus deviceGetCount(uint32_t* pCount) override;

    virtual synStatus deviceGetModuleIds(uint32_t *pDeviceModuleIds, uint32_t*  size) override;

    virtual synStatus deviceGetCountByDeviceType(uint32_t* pCount, const synDeviceType deviceType) override;

    virtual synStatus deviceCount(uint32_t count[synDeviceTypeSize]) override;

    virtual synStatus deviceGetPCIBusId(char* pPciBusId, const int len, const synDeviceId devIdx) override;

    virtual synStatus setCfg(const char* cfgName, const char* cfgValue) override;

    virtual synStatus getCfg(const char* cfgName, char* cfgValue, uint64_t size) override;

    synStatus deviceGetFd(int *pFd, const synDeviceId devIdx) override;

    uint64_t profilerGetCurrentTimeNs() override { return 0; }

    synStatus profileInternalFunction(const char* funcName, uint64_t startTime) override { return synUnsupported; }

    synStatus profileInternalFunctionWithArgsAndThread(const char* funcName,
                                                       uint64_t startTime,
                                                       const char** args,
                                                       size_t argsSize,
                                                       const char* threadName) override { return synUnsupported; }

    synStatus profileInternalFunctionWithArgs(const char* funcName,
                                              uint64_t startTime,
                                              const char** args,
                                              size_t argsSize) override { return synUnsupported; }

    synStatus profilerAddCustomEvent(const char* funcName,
                                     uint64_t startTime,
                                     uint64_t endTime,
                                     const char** args,
                                     size_t argsSize) override { return synUnsupported; }

    synStatus profilerQueryRequiredMemory(const synDeviceId deviceId, uint32_t* bytesRequired) override { return synUnsupported; }

    synStatus profilerSetUserBuffer(const synDeviceId deviceId, void* userBuffer) override { return synUnsupported; }

    synStatus dumpProfilerJson() override { return synUnsupported; }

    synStatus getRecipeDebugInfo(synRecipeHandle recipe, const debug_info_t** recipeDebugInfo) override;

    synStatus getRecipeProgramDataBlobs(synRecipeHandle recipe,
                                        const program_data_blob_t** program_data_blobs,
                                        size_t *program_data_blobs_nr) override;

    synStatus kernelsPrintf(synRecipeHandle recipeHandle, uint64_t workspaceAddr, void* hostBuff);

    virtual synStatus   getDeviceId(const synStreamHandle   streamHandle,
                                    synDeviceId&            devIdx)      const override;

    synStatus   getDeviceInfo(unsigned                devIdx,
                              synDeviceInfo*          pDeviceInfo) const override;

    std::shared_ptr<DeviceInterface> getDevice() const;

    virtual synStatus getDeviceAttribute(const synDeviceId         devIdx,
                                         const synDeviceAttribute* deviceAttr,
                                         const unsigned            querySize,
                                         uint64_t*                 retVal) const override;

    virtual synStatus getDeviceTypeAttribute(const synDeviceType       deviceType,
                                             const synDeviceAttribute* deviceAttr,
                                             const unsigned            querySize,
                                             uint64_t*                 retVal) const override;

    synStatus writeI2cReg(uint32_t devIdx,
                          uint32_t i2cBus,
                          uint32_t i2cAddress,
                          uint32_t regAddress,
                          uint32_t value) override;
    synStatus readI2cReg(uint32_t devIdx,
                         uint32_t i2cBus,
                         uint32_t i2cAddress,
                         uint32_t regAddress,
                         uint32_t* pValue) override;
    synStatus setLedState(uint32_t devIdx, uint32_t ledId, bool state) override;
    synStatus setFrequency(uint32_t devIdx, uint32_t pllId, uint32_t frequency) override;
    synStatus getFrequency(uint32_t devIdx, uint32_t pllId, uint32_t* pFrequency) override;

    virtual synStatus memcpyAsync(const synStreamHandle   streamHandle,
                                  const uint64_t*         pSrc,
                                  const uint64_t*         pSize,
                                  const uint64_t*         pDst,
                                  const synDmaDir         direction,
                                  const uint64_t          numCopies = 1) override;

    virtual synStatus memsetAsync(const synStreamHandle    streamHandle,
                                  uint64_t                 pDeviceMem,
                                  const uint32_t           value,
                                  const size_t             numOfElements,
                                  const size_t             elementSize) override;


    synStatus getTPCLibraryVersionSize(uint32_t* size) override;
    synStatus getTPCLibraryVersions(const char** libs, uint32_t* versions) override;

    synStatus getDeviceName(char *pName,
                            const int len,
                            const synDeviceId devIdx) override;

    synStatus profile( unsigned                       devIdx,
                       hl_debug_args*                 debugParams ) override;

    synStatus profilerStart(synTraceType type, const synDeviceId deviceId) override { return synUnsupported; }

    synStatus profilerStop(synTraceType type, const synDeviceId deviceId) override { return synUnsupported; }

    synStatus profilerGetTrace(synTraceType      type,
                               const synDeviceId deviceId,
                               synTraceFormat    format,
                               void*             buffer,
                               size_t*           size,
                               size_t*           numEntries) override
    {
        return synUnsupported;
    }

    synStatus profilerGetTrace2(synTraceType      type,
                                const synDeviceId deviceId,
                                synTraceFormat    format,
                                void*             buffer,
                                size_t*           size,
                                size_t*           numEntries) override
    {
        return synUnsupported;
    }

    virtual synStatus getClockSyncInfo( unsigned                devIdx,
                                        hlthunk_time_sync_info* infoOut ) override;

    virtual synStatus
    getPllFrequency(unsigned devIdx, uint32_t index, struct hlthunk_pll_frequency_info* freqOut) override;

    virtual synStatus getModuleId(uint32_t& idOut) override;


    synStatus createControlDependency(const synGraphHandle graphHandle,
                                      const synNodeId* pBlockingNodesIdList,
                                      const synNodeId* pBlockedNodesIdList,
                                      const uint32_t numberblocking,
                                      const uint32_t numberblocked) override;

    synStatus nodeSetDeterministic(const synGraphHandle graphHandle,
                                   const synNodeId      nodeId,
                                   const bool           useDeterministic) override;

    synStatus
    nodeGetDeterministic(const synGraphHandle graphHandle, const synNodeId nodeId, bool* pUseDeterministic) override;

    synStatus nodeSetRoundingMode(const synGraphHandle  graphHandle,
                                  const synNodeId       nodeId,
                                  const synRoundingMode roundingMode) override;

    synStatus nodeGetRoundingMode(const synGraphHandle graphHandle,
                                  const synNodeId      nodeId,
                                  synRoundingMode*     pRoundingMod) override;

    synStatus getUniqueNodeId(        synNodeId& nodeId) override;

    synStatus setOriginalComplexNode( synGraphHandle graphHandle,
                                      synNodeId      nodeId,
                                      synNodeId      originalComplexNodeId) override;

    synStatus supportsComplexGuid() override;

    virtual synStatus getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                      synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;

    virtual synStatus getDynamicShapesTensorInfoArrayV2(synStreamHandle             streamHandle,
                                                        synRecipeHandle             recipeHandle,
                                                        std::vector<tensor_info_t>& tensorInfoArray) const override;

    synStatus getRecipeSyncScheme(const synRecipeHandle recipe, const debug_sync_scheme_t** recipeSyncScheme) override;

    virtual synStatus setStreamAffinity(const synDeviceId       deviceId,
                                        const synStreamHandle   pStreamHandle,
                                        uint64_t                streamAffinityMask) override;

    virtual synStatus getStreamAffinity(const synDeviceId       deviceId,
                                        const synStreamHandle   pStreamHandle,
                                        uint64_t*               streamAffinityMask) override;

    virtual synStatus getDeviceAffinityMaskRange(const synDeviceId  deviceId,
                                                 uint64_t*          deviceAffinityMaskRange) override;

    virtual synStatus getDeviceNextStreamAffinity(const synDeviceId deviceId, uint64_t* nextDeviceAffinity) override;

    eHostAddrToVirtualAddrMappingStatus _getDeviceVirtualAddress(bool           isUserRequest,
                                                                 void*          hostAddress,
                                                                 uint64_t       bufferSize,
                                                                 uint64_t*      pDeviceVA,
                                                                 bool*          pIsExactKeyFound = nullptr);

    synStatus submitCommandBuffers(CommandSubmission&   commandSubmission,
                                   uint64_t*            csHandle,
                                   uint64_t*            mappedBuff,
                                   const uint32_t       physicalQueueOffset,
                                   const StagedInfo*    pStagedInfo,
                                   globalStatPointsEnum point = globalStatPointsEnum::colLast);

    synStatus getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress);

    synStatus addDeviceAddressToReleaseOnStreamDestroy(uint64_t address);

    static std::string getConfigFilename();

    synStatus waitAndReleaseCS(uint64_t  handle,
                               uint64_t  timeout,
                               bool      returnUponTimeout = false,
                               bool      collectStats      = false,
                               uint64_t* usrEventTime      = nullptr);

    static void printVersionToLog(synapse::LogManager::LogType logType, const std::string& description);

    synStatus submitLinDmaCommand(const internalMemcopyParams& rMemcpyParams,
                                  internalDmaDir               direction,
                                  bool                         isArbitrationRequired,
                                  PhysicalQueuesId             physicalQueueId,
                                  InternalWaitHandle*          waitHandle,
                                  DataChunksDB&                rDataChunks,
                                  CommandSubmissionDataChunks* pCsDataChunks,
                                  bool                         isUserRequest,
                                  bool                         isMemset,
                                  bool                         isInspectCopiedContent,
                                  uint64_t                     maxLinDmaBufferSize,
                                  uint64_t                     arbCommandSize,
                                  uint64_t                     sizeOfLinDmaCommand,
                                  uint64_t                     sizeOfWrappedLinDmaCommand,
                                  uint64_t                     sizeOfSingleCommandBuffer);

    synStatus generateApiId(uint8_t& rApiId);

    synStatus syncHCLStreamHandle(synStreamHandle streamHandle);

    synStatus isStreamInitialized(synStreamHandle streamHandle, bool& rIsInitialized);

    synStatus flushWaitsOnCollectiveStream(synStreamHandle streamHandle);

    uint32_t getNetworkStreamPhysicalQueueOffset(synStreamHandle streamHandle);

    hcl::hclStreamHandle getNetworkStreamHclStreamHandle(synStreamHandle streamHandle);

    void notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo = {});

    synStatus submitTrainingConfigurationCS(synDeviceType      deviceType,
                                            char*&             pPackets,
                                            uint64_t           packetsSize,
                                            const std::string& operationDescription,
                                            uint32_t           queueId,
                                            bool               isConfigOnInternal         = false,
                                            bool               isSyncWithExternalRequired = false,
                                            uint32_t           waitQmanId                 = 0);

    synStatus waitAndReleaseStreamHandles(const InternalWaitHandlesVector& streamWaitHandles,
                                          uint64_t                         timeout,
                                          bool                             returnUponTimeout = false);

    synStatus tensorSetDeviceFullLayout(synTensor tensor, const synTensorDeviceFullLayout* layout) override;

    synStatus tensorGetDeviceFullLayout(const synTensor tensor, synTensorDeviceFullLayout* layout) override;

    synStatus tensorSetPermutation(synTensor tensor, const synTensorPermutation* permutation) override;

    synStatus tensorGetPermutation(const synTensor tensor, synTensorPermutation* permutation) override;

    synStatus tensorSetGeometryExt(synTensor                   tensor,
                                   const synTensorGeometryExt* geometry,
                                   synGeometryType             geometryType) override;

    synStatus tensorGetGeometryExt(const synTensor         tensor,
                                   synTensorGeometryExt*   geometry,
                                   synGeometryType         geometryType) override;

    synStatus tensorSetDeviceDataType(synTensor   tensor,
                                      synDataType deviceDataType) override;

    synStatus tensorGetDeviceDataType(synTensor   tensor,
                                      synDataType* deviceDataType) override;

    static void elevateSynLaunchTensorInfo(synLaunchTensorInfoExt*    launchTensorsInfoExt,
                                           const synLaunchTensorInfo* launchTensorsInfo,
                                           uint32_t                   numberTensors);

    static void lowerSynRetrievedLaunchTensorInfoExt(const synRetrievedLaunchTensorInfoExt* launchTensorsInfoExt,
                                                     synRetrievedLaunchTensorInfo*          launchTensorsInfo,
                                                     uint32_t                               numberTensors);

    static void lowerTensorMetadataInfoExt(const TensorMetadataInfoExt* tensorsMetadataInfoExt,
                                           TensorMetadataInfo*          tensorsMetadataInfo,
                                           uint32_t                     numberTensors);

    static void setSynRetrievedLaunchTensorInfoExtIDs(synRetrievedLaunchTensorInfoExt*    launchTensorsInfoExt,
                                                      const synRetrievedLaunchTensorInfo* launchTensorsInfo,
                                                      uint32_t                            numberTensors);

    static void setTensorsMetadataInfoNamesExt(TensorMetadataInfoExt*    tensorsMetadataInfoExt,
                                               const TensorMetadataInfo* tensorsMetadataInfo,
                                               uint32_t                  numberTensors);

    synStatus nodeSetParams(const synGraphHandle graphHandle,
                            const synNodeId      nodeId,
                            const void*          userParams,
                            const unsigned       paramsSize) override;

    synStatus nodeGetParams(const synGraphHandle graphHandle,
                            const synNodeId      nodeId,
                            void*                userParams,
                            unsigned*            paramsSize) override;

    void collectStat(StatApiPoints point, uint64_t val) { m_statApi.collect(point, val); }
    bool apiStatsIsEnabled() { return m_statApi.isEnabled(); }

    SlotMapItemSptr<InternalSectionHandle> sectionHandleToPtr(synSectionHandle handle)
    {
        auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)handle];

        if (!sectionPtr)
        {
            if ((SMHandle)handle != 0) // if user gave a "null" handle, no need to log an error
            {
                LOG_CRITICAL(SYN_API, "section doesn't exist (deleted?) {:x}", TO64((SMHandle)handle));
            }
        }
        return sectionPtr;
    }

    virtual synStatus setSmfCallbacks(smf_callbacks_t* callbacks) override;

    static synStatus convertStatusToString(synStatus status, char* statusDescription, size_t len);

    virtual synStatus getDeviceAttributesByModuleId(const synModuleId         moduleId,
                                                    const synDeviceAttribute* deviceAttr,
                                                    const unsigned            querySize,
                                                    uint64_t*                 retVal) const override;

    virtual synStatus setHostProfilerArg(const std::vector<synTraceEventArg>& keyValArgs) override;

    virtual synStatus sectionsClearHostBuffer( synRecipeHandle     recipeHandle,
                                               const synSectionId* sectionIds,
                                               size_t              numOfSections ) override;

    virtual synStatus getClockSyncPerDieInfo( unsigned          devIdx,
                                              uint32_t          dieIndex,
                                              hlthunk_time_sync_info* infoOut) override;

    synStatus getDeviceInfo(unsigned         devIdx,
                            synDeviceInfoV2* pDeviceInfo) const override;

private:
    static void                   initSingleton();
    static synSingletonInterface* getInstance(bool createInstance);
    synStatus                     _createGenericNode(const synGraphHandle graphHandle,
                                                     const synTensor*     inputs,
                                                     const synTensor*     outputs,
                                                     const uint32_t       sizeInputs,
                                                     const uint32_t       sizeOutputs,
                                                     const void*          userParams,
                                                     const unsigned       paramsSize,
                                                     const char*          guid,
                                                     const char**         inputLayouts,
                                                     const char**         outputLayouts,
                                                     const std::string&   name         = "",
                                                     synNodeId*           nodeUniqueId = nullptr);

    bool isTensorValidForGraph(const TensorPtr tensor, uint32_t graphId);

    synStatus _createControlDependency(HabanaGraph*        graph,
                                       uint32_t            graphId,
                                       const synDeviceType deviceType,
                                       const synNodeId*    pBlockingNodesIdList,
                                       const synNodeId*    pBlockedNodesIdList,
                                       const uint32_t      numberBlocking,
                                       const uint32_t      numberBlocked);

    synStatus insertDescendantsIds(NodeSet& nodeSet, synNodeId complexGuidNodeId, uint32_t graphId);

    synStatus validateControlDependencyDevice(HabanaGraph* graph, NodeSet& blockingNodesSet);

    synStatus addNodeToControlDependencySet(NodeSet&         cdSet,
                                            const synNodeId* pNodeIdList,
                                            unsigned         numberOfNodes,
                                            uint32_t         graphId);

    HabanaGraph* _getGraphForCompilation(const synGraphHandle graphHandle);

    NodePtr _findNodeLockRequired(synNodeId nodeUniqueId, uint32_t graphId, const synNodeId nodeId);
    NodePtr _findNode(synNodeId nodeUniqueId, uint32_t graphId);

    synSingleton();
    static void beforeFork();
    synSingleton(synSingleton const&) = delete;
    void operator=(synSingleton const&) = delete;

    void _destroyAllGraphs();

    synStatus _releaseAllRecipes();

    static std::vector<std::string> _deviceTypeToStrings(synDeviceType deviceType);

    static void _getTensorScaleZp(const synTensorDescriptor& pDescriptor, double& scale, double& zp);

    bool _validateSectionForGraph(InternalSectionHandle* sectionHandle);

    synStatus _sectionLockAndSetID(InternalSectionHandle* sectionHandle);

    synStatus _destroy();

    synStatus tensorAssignToSectionInternal(synTensor              tensor,
                                            synSectionHandle       sectionHandle,
                                            SlotMapItemSptr<InternalSectionHandle> sectionPtr,
                                            uint64_t               byteOffset);

    synStatus tensorUpdateSectionInfoInternal(synTensor              tensor,
                                              synSectionHandle       sectionHandle,
                                              SlotMapItemSptr<InternalSectionHandle> sectionPtr,
                                              uint64_t               byteOffset);

    static synSingletonInterface* m_pInstance;
    static synSingleton*          m_pInstanceInternal;
    static std::mutex             m_singletonCreationMutex;
    static bool                   m_isChildProcessWithAcquiredDevice;
    static libHandle              m_profLibHandle;

    mutable ConcurrentSlotMapAlloc<InternalSectionHandle> m_sectionHndlSlopMap;
    GraphEntriesContainer m_graphEntries;

    struct DfaGlblStatus
    {
        DfaPhase        dfaPhase = DfaPhase::NONE;
        EventConnection dfaPhaseConnection;
        std::mutex      mutex;
    };

    DfaGlblStatus m_dfaGlblStatus;
    DeviceManager m_deviceManager;

    // Mutexes
    std::mutex m_graphsMutex;

    RecipeManager              m_recipeManager;
    Statistics<enumNameSynApi> m_statApi;
};

SlotMapItemSptr<InternalSectionHandle> getSectionPtrFromHandle(synSectionHandle handle);
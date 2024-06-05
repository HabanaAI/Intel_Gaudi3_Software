/*
.
1) This class is used for Compute operations
.
.
2) Usage of DCs (DC = Data Chunks) (execution operation) -
    a) Compute operation:
    We need to acquire MMU DCs per launch and CB DCs statically (done during static processing, and using the CB DC
allocator of the Recipe-Singleton
.
.
3) Release of DCs due to a CS DC release -
    a) Compute operation:
    We need to clear the MMU DCs
    [The CB DCs are static, as defined above]
*/
#pragma once

#include "queue_base_qman_wcm.hpp"
#include "runtime/qman/common/cs_dc_allocator.hpp"
#include "memory_manager.hpp"
#include "runtime/qman/common/arb_master_helper.hpp"
#include "runtime/qman/common/stream_master_helper.hpp"
#include "runtime/qman/common/stream_dc_downloader.hpp"

#include <memory>

class PhysicalQueuesManager;
struct reserve_sig_handle;
struct recipe_t;
class DataChunksAllocatorCommandBuffer;
class SubmitCommandBuffersInterface;
class DevMemoryAllocInterface;
class DeviceRecipeAddressesGeneratorInterface;
class DataChunksAllocator;
struct DeviceAgnosticRecipeInfo;
struct RecipeTensorsInfo;

class SynGaudiInflightParserTests;

class RecipeStaticInfo;
class DeviceDownloaderInterface;
class DeviceRecipeDownloaderContainerInterface;
class StreamMasterHelperInterface;
class InferenceSramCacheMgr;

class RecipeProgramBuffer;
typedef std::shared_ptr<RecipeProgramBuffer> SpRecipeProgramBuffer;

enum eCommandSumissionDataChunkType : uint8_t;

class QueueComputeQman : public QueueBaseQmanWcm
{
    friend class SynGaudiInflightParserTests;
    friend class UTStreamComputeTest;

public:
    QueueComputeQman(const BasicQueueInfo&                        rBasicQueueInfo,
                     uint32_t                                     physicalQueueOffset,
                     uint32_t                                     amountOfEnginesInArbGroup,
                     bool                                         isReduced,
                     synDeviceType                                deviceType,
                     PhysicalQueuesManagerInterface*              pPhysicalStreamsManager,
                     WorkCompletionManagerInterface&              rWorkCompletionManager,
                     SubmitCommandBuffersInterface&               rSubmitter,
                     DevMemoryAllocInterface&                     rDevMemAlloc,
                     DeviceRecipeAddressesGeneratorInterface&     rDevRecipeAddress,
                     DeviceRecipeDownloaderContainerInterface&    rDeviceRecipeDownloaderContainer,
                     QueueInterface&                              rStreamCopy,
                     std::unique_ptr<StreamMasterHelperInterface> pStreamMasterHelper);

    virtual ~QueueComputeQman();

    synStatus initAllocators();

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override;

    virtual synStatus memcopy(internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest,
                              QueueInterface*        pPreviousStream,
                              const uint64_t         overrideMemsetVal,
                              bool                   inspectCopiedContent,
                              SpRecipeProgramBuffer* pRecipeProgramBuffer,
                              uint8_t                apiId) override
    {
        return synFail;
    }

    virtual synStatus launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             uint64_t                      assertAsyncMappedAddress,
                             uint32_t                      flags,
                             EventWithMappedTensorDB&      events,
                             uint8_t                       apiId) override;

    // Destroys any DB related to the recipe-handle on this stream
    // Will return false in case recipe is being used
    virtual void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle);

    virtual void notifyAllRecipeRemoval();

    virtual void notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed) override;

    virtual bool isRecipeHasInflightCsdc(InternalRecipeHandle* pRecipeHandle) override;

    static void setDcMprotectSignalHandler();

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const;

private:
    typedef std::map<InternalRecipeHandle*, IDynamicInfoProcessor*> RecipeHandleToRecipeProcessorMap;

    static uint64_t getRecipeCacheReferenceCount(synDeviceType deviceType);

    // Creates a dedicated RecipeProcessor to this recipe_t and initializes it
    synStatus retrieveDynamicInfoProcessor(const basicRecipeInfo&          rBasicRecipeInfo,
                                           const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                           const RecipeStaticInfo&         rRecipeStaticInfo,
                                           InternalRecipeHandle*           pRecipeHandle,
                                           uint64_t                        recipeId,
                                           IDynamicInfoProcessor*&         pRecipeProcessor);

    synStatus _enqueue(const synLaunchTensorInfoExt*   launchTensorsInfo,
                       uint32_t                        launchTensorsAmount,
                       const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                       basicRecipeInfo&                basicRecipeHandle,
                       const RecipeStaticInfo&         rRecipeStaticInfo,
                       uint64_t                        recipeId,
                       InternalRecipeHandle*           pRecipeHandle,
                       IDynamicInfoProcessor*&         pRecipeProcessor,
                       uint64_t                        programDataHandle,
                       uint64_t                        programCodeHandle,
                       uint64_t                        workspaceAddress,
                       uint64_t                        prgDataSubTypeAllocId,
                       uint64_t                        assertAsyncMappedAddress,
                       uint32_t                        flags,
                       bool                            programCodeInCache,
                       bool                            programDataInCache,
                       uint64_t                        programDataDeviceAddress,
                       uint64_t                        programCodeDeviceAddr,
                       uint64_t&                       csHandle,
                       eAnalyzeValidateStatus          analyzeValidateStatus,
                       EventWithMappedTensorDB&        events,
                       reserve_sig_handle&             sigHandle,
                       eCsDcProcessorStatus&           csDcProcessingStatus);

    // Performs activate operation followed by TSM update

    // Performs launch operation followed by TSM update

    synStatus _enqueueAndSync(const synLaunchTensorInfoExt*   launchTensorsInfo,
                              uint32_t                        launchTensorsAmount,
                              basicRecipeInfo&                basicRecipeHandle,
                              const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                              const RecipeStaticInfo&         rRecipeStaticInfo,
                              uint64_t                        recipeId,
                              InternalRecipeHandle*           pRecipeHandle,
                              IDynamicInfoProcessor*&         pRecipeProcessor,
                              uint64_t                        programDataHandle,
                              uint64_t                        programCodeHandle,
                              uint64_t                        workspaceAddress,
                              uint64_t                        prgDataSubTypeAllocId,
                              uint64_t                        assertAsyncMappedAddress,
                              uint32_t                        flags,
                              bool                            programCodeInCache,
                              bool                            programDataInCache,
                              uint64_t                        programDataDeviceAddress,
                              uint64_t                        programCodeDeviceAddr,
                              eAnalyzeValidateStatus          analyzeValidateStatus,
                              EventWithMappedTensorDB&        events,
                              eCsDcProcessorStatus&           csDcProcessingStatus);

    // Performs stream-synchronization
    synStatus _syncPreOperation(QueueInterface& rPrecedingStream);

    synStatus _updateStreamPostExecution(const BasicQueueInfo& rBasicQueueInfo,
                                         uint64_t              operationHandle,
                                         const std::string&    desc);

    void _tryToReleaseProcessorsCsDc();

    static bool isRecipeStageCsDcCpDmaReady(uint64_t                           recipeId,
                                            eExecutionStage                    stage,
                                            uint64_t                           programCodeHandle,
                                            const CommandSubmissionDataChunks& rCsDataChunks);

    void _clearCsdcDB();

    void _notifyCsCompleted(uint64_t waitForEventHandle, bool csFailed);

    // Static mappings are those which are alocated by the StreamDcDownloader,
    // are common to all CS-DCs of that stream, and are for the content of Gaudi-1 Compute-Stream's exteranl HW-Stream
    virtual bool _addStaticMapping(AddressRangeMapper& addressRangeMap) const override;

    virtual void _dfaLogCsDcInfo(CommandSubmissionDataChunks* csPtr, int logLevel, bool errorCsOnly) override;

    void _addRecipeProcessor(IDynamicInfoProcessor* pRecipeProcessor, InternalRecipeHandle* pRecipeHandle);

    void _deleteAllRecipeProcessors();

    void _waitForRecipeCsdcs(InternalRecipeHandle* pRecipeHandle, bool eraseFromDb);

    void _waitForRecipeCopyCsdcs(uint64_t recipeId);

    eCsDataChunkStatus retrieveCsDc(const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                    IDynamicInfoProcessor&          rRecipeProcessor,
                                    uint64_t                        recipeId,
                                    const recipe_t&                 rRecipe,
                                    InternalRecipeHandle*           pRecipeHandle,
                                    uint64_t                        cpDmaChunksAmount,
                                    uint64_t                        commandsDataChunksAmount,
                                    bool                            isOldCsDcReuseAllowed,
                                    eExecutionStage                 stage,
                                    CommandSubmissionDataChunks*&   pCsDataChunks);

    void _popCsdc();

    void _clearProcessorsDb();

    synStatus _runSifPreDownload(const basicRecipeInfo&          rBasicRecipeInfo,
                                 uint64_t                        recipeId,
                                 const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                 const RecipeStaticInfo&         rRecipeStaticInfo,
                                 InternalRecipeHandle*           pRecipeHandle,
                                 const synLaunchTensorInfoExt*   pLaunchTensorsInfo,
                                 uint32_t                        launchTensorsAmount,
                                 uint64_t                        programDataHostAddress);

    void releaseSifOwnership(const basicRecipeInfo&          rBasicRecipeInfo,
                             const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                             const RecipeStaticInfo&         rRecipeStaticInfo,
                             InternalRecipeHandle*           pRecipeHandle,
                             uint64_t                        recipeId);

    synStatus _processRecipe(DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader,
                             std::vector<uint64_t>&           executionBlocksDeviceAddresses,
                             uint64_t                         programCodeHandle,
                             uint64_t                         programCodeDeviceAddress,
                             uint64_t                         workspaceAddress,
                             bool                             programCodeInCache);

    synStatus _downloadBuffersToCache(DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader,
                                      std::vector<uint64_t>&           executionBlocksDeviceAddresses,
                                      uint64_t                         programCodeHandle,
                                      uint64_t                         programDataHandle,
                                      uint64_t                         programDataDeviceAddress,
                                      uint64_t                         workspaceAddress,
                                      SpRecipeProgramBuffer            programDataRecipeBuffer,
                                      bool                             programCodeInCache,
                                      bool                             programDataInCache);

    synStatus _downloadBuffersToWorkspace(DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader,
                                          bool&                            isSyncWithDmaSynapseRequired,
                                          uint64_t                         recipeId,
                                          SpRecipeProgramBuffer            programDataRecipeBuffer,
                                          uint64_t                         programCodeHandle,
                                          uint64_t                         programDataHandle,
                                          uint64_t                         programCodeDeviceAddress,
                                          uint64_t                         programDataDeviceAddress,
                                          bool                             programCodeInCache,
                                          bool                             programDataInCache);

    synStatus _syncPreOperation(bool&                         rIsSyncWithDmaSynapseRequired,
                                const IDynamicInfoProcessor*& pRecipeProcessor,
                                uint64_t                      programCodeHandle,
                                uint64_t                      programDataHandle);

    synStatus _activateAndEnqueue(const synLaunchTensorInfoExt*&  launchTensorsInfo,
                                  uint32_t                        launchTensorsAmount,
                                  basicRecipeInfo&                basicRecipeHandle,
                                  const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                  const RecipeStaticInfo&         rRecipeStaticInfo,
                                  uint64_t                        recipeId,
                                  InternalRecipeHandle*           pRecipeHandle,
                                  IDynamicInfoProcessor*&         pRecipeProcessor,
                                  uint64_t                        workspaceAddress,
                                  uint64_t                        programCodeDeviceAddress,
                                  uint64_t                        programDataDeviceAddress,
                                  uint64_t                        programCodeHandle,
                                  uint64_t                        programDataHandle,
                                  uint64_t                        prgDataSubTypeAllocId,
                                  uint64_t                        assertAsyncMappedAddress,
                                  uint64_t&                       refCount,
                                  uint32_t                        flags,
                                  bool                            programCodeInCache,
                                  bool                            programDataInCache,
                                  EventWithMappedTensorDB&        events);

    void tryToReleaseCommandSubmissionDataChunk(CommandSubmissionDataChunks*& pCsDataChunks);

    void tryToReleaseProcessorCsDc(IDynamicInfoProcessor& rRecipeProcessor);

    eCsDataChunkStatus tryToRetrieveCsDc(synDeviceType                             deviceType,
                                         const RecipeHandleToRecipeProcessorMap&   rRecipeProcessorsDB,
                                         const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                         IDynamicInfoProcessor&                    rRecipeProcessor,
                                         uint64_t                                  recipeId,
                                         const recipe_t&                           rRecipe,
                                         InternalRecipeHandle*                     pRecipeHandle,
                                         uint64_t                                  cpDmaChunksAmount,
                                         uint64_t                                  commandsDataChunksAmount,
                                         bool                                      isOldCsDcReuseAllowed,
                                         eExecutionStage                           stage,
                                         bool                                      isFirst,
                                         bool                                      isLast,
                                         CommandSubmissionDataChunks*&             pCsDataChunks);

    bool releaseCsDataChunksFromProcessors(const RecipeHandleToRecipeProcessorMap& rRecipeProcessorsDB,
                                           DataChunksAmounts&                      dcAmountsAvailable,
                                           DataChunksDBs&                          dcDbs,
                                           const DataChunksAmounts&                dcAmountsRequired);

    static bool createCsDc(synDeviceType                 deviceType,
                           CommandSubmissionDataChunks*& pCsDataChunks,
                           const RecipeTensorsInfo&      rRecipeTensorInfo,
                           uint64_t                      recipeId,
                           const recipe_t&               rRecipe,
                           InternalRecipeHandle*         pRecipeHandle,
                           eExecutionStage               stage);

    const uint32_t m_amountOfEnginesInArbGroup;

    static IDynamicInfoProcessor* s_pTestDynamicInfoProcessor;

    // Number of elements to (try to) release from each SRP, when trying to acquire DCs
    static const uint32_t numOfElementsToRelease;

    SubmitCommandBuffersInterface& m_rSubmitter;

    DevMemoryAllocInterface& m_rDevMemAlloc;

    DeviceRecipeAddressesGeneratorInterface& m_rDevRecipeAddress;

    DeviceRecipeDownloaderContainerInterface& m_rDeviceRecipeDownloaderContainer;

    QueueInterface& m_rStreamCopy;

    MemoryManager m_memoryManager;

    CsDcAllocator m_csDcAllocator;

    ArbMasterHelper m_arbMasterHelper;

    std::unique_ptr<StreamMasterHelperInterface> m_pStreamMasterHelper;

    StreamDcDownloader m_streamDcDownloader;

    RecipeHandleToRecipeProcessorMap m_recipeProcessorsDB;

    std::vector<uint64_t> m_lastLaunchTensorsAddresses;

    bool m_isSyncWithDmaSynapseRequired;

    // Inference usage only
    // We will ignore a use-case where a topology had been loaded, but not activated,
    // prior another topology execution (overrunning its loaded content)
    uint64_t m_lastRecipeIdActivated;

    uint64_t m_lastWsDwldPrgCodeAddress;
    uint64_t m_lastWsDwldPrgCodeRecipeId;
    uint64_t m_lastWsDwldPrgDataAddress;
    uint64_t m_lastWsDwldPrgDataRecipeId;
};

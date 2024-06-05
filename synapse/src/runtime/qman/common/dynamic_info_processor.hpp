#pragma once

#include "recipe_package_types.hpp"

#include "dynamic_info_processor_interface.hpp"

#include "synapse_api_types.h"
#include "synapse_common_types.h"

#include "runtime/common/recipe/patching/define.hpp"
#include "runtime/common/recipe/patching/host_address_patcher.hpp"
#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/common/recipe/recipe_patch_processor.hpp"

#include <deque>
#include <map>
#include <mutex>
#include <cstdint>

class CommandSubmission;
class CommandSubmissionDataChunks;
class DataChunk;
class DataChunksAllocatorCommandBuffer;
class RecipeStaticInfo;
class ArbMasterHelper;
class StreamMasterHelperInterface;
class StreamDcDownloader;
class SubmitCommandBuffersInterface;
class DevMemoryAllocInterface;
class QmanDefinitionInterface;
struct BasicQueueInfo;

namespace patching
{
class HostAddressPatchingInformation;
}

struct blob_t;
struct job_t;
struct recipe_t;
struct basicRecipeInfo;
struct DeviceAgnosticRecipeInfo;
struct StagedInfo;
class SubmitCommandBuffersInterface;

class DynamicInfoProcessor : public IDynamicInfoProcessor
{
    using PatchPointPointerPerSection = std::unordered_map<uint64_t, const data_chunk_patch_point_t*>;

public:
    DynamicInfoProcessor(synDeviceType                      deviceType,
                         const BasicQueueInfo&              rBasicQueueInfo,
                         uint32_t                           physicalQueueOffset,
                         uint32_t                           amountOfEnginesInArbGroup,
                         uint64_t                           recipeId,
                         const basicRecipeInfo&             rBasicRecipeInfo,
                         const DeviceAgnosticRecipeInfo&    rDeviceAgnosticRecipeInfo,
                         const RecipeStaticInfo&            rRecipeInfo,
                         const ArbMasterHelper&             rArbMasterHelper,
                         const StreamMasterHelperInterface& rStreamMasterHelper,
                         const StreamDcDownloader&          rStreamDcDownloader,
                         SubmitCommandBuffersInterface&     rSubmitter,
                         DevMemoryAllocInterface&           rDevMemAlloc);

    virtual ~DynamicInfoProcessor();

    virtual synStatus enqueue(const synLaunchTensorInfoExt* launchTensorsInfo,
                              uint32_t                      launchTensorsAmount,
                              CommandSubmissionDataChunks*& pCsDataChunks,
                              uint64_t                      scratchPadAddress,
                              uint64_t                      programDataDeviceAddress,
                              uint64_t                      programCodeHandle,
                              uint64_t                      programDataHandle,
                              bool                          programCodeInCache,
                              bool                          programDataInCache,
                              uint64_t                      assertAsyncMappedAddress,
                              uint32_t                      flags,
                              uint64_t&                     csHandle,
                              eAnalyzeValidateStatus        analyzeValidateStatus,
                              uint32_t                      sigHandleId,
                              uint32_t                      sigHandleSobjBaseAddressOffset,
                              eCsDcProcessorStatus&         csDcProcessingStatus) override;

    virtual bool
    notifyCsCompleted(CommandSubmissionDataChunks* pCsDataChunks, uint64_t waitForEventHandle, bool csFailed) override;

    // Releasing and retrieving CS-DC elements of any handle
    // The "current" execution-handle items should be last option -> TBD
    virtual uint32_t releaseCommandSubmissionDataChunks(uint32_t                        numOfElementsToRelease,
                                                        CommandSubmissionDataChunksVec& releasedElements,
                                                        bool                            keepOne) override;

    // Release and retrieve a CS-DC element of current execution-handle
    virtual CommandSubmissionDataChunks* getAvailableCommandSubmissionDataChunks(eExecutionStage executionStage,
                                                                                 bool            isCurrent) override;

    virtual void incrementExecutionHandle() override;

    virtual uint64_t getExecutionHandle() override;

    virtual void getProgramCodeHandle(uint64_t& programCodeHandle) const override;

    virtual void getProgramDataHandle(uint64_t& programDataHandle) const override;

    virtual void setProgramCodeHandle(uint64_t staticCodeHandle) override;

    virtual void setProgramCodeAddrInWS(uint64_t programCodeAddrInWS) override;

    virtual void setProgramDataHandle(uint64_t programDataHandle) override;

    virtual bool resolveTensorsIndices(std::vector<uint32_t>*&       pTensorIdx2userIdx,
                                       const uint32_t                launchTensorsAmount,
                                       const synLaunchTensorInfoExt* launchTensorsInfo) override;

    bool isAnyInflightCsdc() override;

    const char* getRecipeName() override { return m_rRecipe.name; }
    virtual uint64_t getRecipeId() const override { return m_recipeId; }

    virtual const basicRecipeInfo&          getRecipeBasicInfo() override { return m_rBasicRecipeInfo; }
    virtual const DeviceAgnosticRecipeInfo& getDevAgnosticInfo() override { return m_rDeviceAgnosticRecipeInfo; }

    virtual std::vector<tensor_info_t> getDynamicShapesTensorInfoArray() const override
    {
        return m_dynamicRecipe->getDynamicShapesTensorInfoArray();
    };

    virtual DynamicRecipe* getDsdPatcher() override
    {
        return m_dynamicRecipe.get();
    };

private:
    typedef std::deque<eExecutionStage> UsedCsDcStageTypeDB;

    struct staticBlobsAddressInWS
    {
        staticBlobsDeviceAddresses staticBlobsAddresses;
        uint64_t                   programCodeAddrInWS;
    };

    virtual synStatus _enqueue(const synLaunchTensorInfoExt* launchTensorsInfo,
                               uint32_t                      launchTensorsAmount,
                               CommandSubmissionDataChunks*& pCsDataChunks,
                               uint64_t                      scratchPadAddress,
                               uint64_t                      programDataDeviceAddress,
                               uint64_t                      programCodeHandle,
                               uint64_t                      programDataHandle,
                               bool                          programCodeInCache,
                               bool                          programDataInCache,
                               uint64_t                      assertAsyncMappedAddress,
                               uint32_t                      flags,
                               uint32_t                      sigHandleId,
                               uint32_t                      sigHandleSobjBaseAddressOffset,
                               uint64_t&                     csHandle,
                               eCsDcProcessorStatus&         csDcProcessingStatus,
                               eAnalyzeValidateStatus&       analyzeValidateStatus);

    // Execution handle to CS-DC DB
    // We assume that it will not get to a wraparound, and if it will,
    // then it will have a neglectful effect on overall performance (otherwise... deque of deques)
    //
    // TODO: No real sense of having CS-DCs DB, as there is no real-chance of having a wraparound
    //       of uint64_t of handles
    typedef std::map<uint64_t, CommandSubmissionDataChunksDB> freeCommandSubmissionDataChunksDB;

    bool _createCommandSubmission(CommandSubmission&           commandSubmission,
                                  CommandSubmissionDataChunks& commandSubmissionDataChunks,
                                  uint64_t                     numOfPqsForInternalQueues,
                                  eExecutionStage              stage);

    bool createExternalQueueCB(CommandSubmission& commandSubmission);

    static bool _createInternalQueuePqCommand(CommandSubmission&           commandSubmission,
                                              CommandSubmissionDataChunks& commandSubmissionDataChunks,
                                              uint64_t&                    pqIndex,
                                              const job_t*                 pCurrentJob);

    bool _sendCommandSubmission(CommandSubmission*& pCommandSubmission,
                                uint64_t&           csHandle,
                                const StagedInfo*   pStagedInfo,
                                unsigned int        stageIdx);

    void _insertCsDataChunkToCacheUponCompletion(CommandSubmissionDataChunks* pCsDataChunks,
                                                 eCsDcExecutionType           csDcExecutionType,
                                                 eExecutionStage              executionStage,
                                                 bool                         isSucceeded,
                                                 eCsDcProcessorStatus&        csDcProcessingStatus);

    void _releaseCommandSubmissionDataChunks(uint32_t&                       leftElementsToRelease,
                                             CommandSubmissionDataChunksVec& releasedElements,
                                             CommandSubmissionDataChunksDB&  currentHandleFreeCsDataChunks);

    void _storeStaticBlobsDeviceAddresses(uint64_t programCodeHandle);

    synStatus _validateNewTensors(const synLaunchTensorInfoExt* launchTensorsInfo,
                                  uint32_t                      launchTensorsAmount,
                                  uint32_t                      flags,
                                  uint64_t                      scratchPadAddress,
                                  uint64_t                      programCodeHandle,
                                  uint64_t                      sectionAddressForData,
                                  bool                          programCodeInCache,
                                  bool                          programDataInCache);

    synStatus _analyzeNewTensors(CommandSubmissionDataChunks*& pCsDataChunks,
                                 const synLaunchTensorInfoExt* launchTensorsInfo,
                                 uint32_t                      launchTensorsAmount,
                                 uint64_t                      scratchPadAddress,
                                 uint64_t                      programCodeHandle,
                                 uint64_t                      sectionAddressForData,
                                 eCsDcExecutionType            csDcExecutionType,
                                 eExecutionStage               executionStage,
                                 bool&                         hasNewAddress,
                                 bool                          programCodeInCache,
                                 bool                          programDataInCache,
                                 eCsDcProcessorStatus&         csDcProcessingStatus,
                                 uint32_t                      sobBaseAddrOffset,
                                 uint64_t                      assertAsyncMappedAddress,
                                 uint32_t                      flags);

    synStatus _patchOnDc(CommandSubmissionDataChunks*& pCsDataChunks,
                         uint64_t                      scratchPadAddress,
                         eCsDcExecutionType            csDcExecutionType,
                         eExecutionStage               executionStage,
                         PatchPointPointerPerSection&  currPatchPoints,
                         SectionTypesToPatch&          sectionTypesToPatch,
                         uint32_t                      lastStageNode,
                         uint32_t                      sobBaseAddr,
                         eCsDcProcessorStatus&         csDcProcessingStatus);

    bool _isPatchingRequired(const CommandSubmissionDataChunks& rCsDataChunks,
                             uint32_t                           launchTensorsAmount,
                             uint64_t                           scratchPadAddress,
                             uint64_t                           sectionAddressForData,
                             uint64_t                           sobjAddress,
                             eExecutionStage                    stage) const;

    bool getHostAddressPatchingInfo(CommandSubmissionDataChunks*&              pCsDataChunks,
                                    patching::HostAddressPatchingInformation*& pPatchingInformation);

    void _conditionalAddProgramDataSectionForPatching(CommandSubmissionDataChunks* pCsDataChunks,
                                                      uint64_t                     programDataHandle);

    SectionTypesToPatch& _setupStartingPatchPointsPerSection(CommandSubmissionDataChunks* pCsDataChunks,
                                                             PatchPointPointerPerSection& currPatchPoint,
                                                             eExecutionStage              stage);

    void storeDataChunksHostAddresses(CommandSubmissionDataChunks& rCsDataChunks);

    static int const                NUM_TENSOR_TYPES = 2;
    const synDeviceType             m_deviceType;
    const BasicQueueInfo&           m_rBasicQueueInfo;
    const uint32_t                  m_physicalQueueOffset;
    const uint32_t                  m_amountOfEnginesInArbGroup;
    const recipe_t&                 m_rRecipe;
    const uint64_t                  m_recipeId;
    const basicRecipeInfo&          m_rBasicRecipeInfo;
    const DeviceAgnosticRecipeInfo& m_rDeviceAgnosticRecipeInfo;
    const RecipeStaticInfo&         m_rRecipeInfo;
    // Currently used by program-data which is the only one  being downloaded (non static Gaudi-Demo)
    //
    // It will be updated after first successful signaling to the compute stream
    // This ensures anything on that HW-stream done after download execution)
    uint64_t m_lastPatchedWorkspaceAddress;   // WS-address of last patching
    uint64_t m_lastDownloadWorkspaceAddress;  // WS-address of last download
    uint64_t m_lastPatchedProgramDataHandle;
    uint64_t m_lastSobjAddress;

    const ArbMasterHelper& m_rArbMasterHelper;

    const StreamMasterHelperInterface& m_rStreamMasterHelper;

    const StreamDcDownloader& m_rStreamDcDownloader;

    // A pool of used and free CS-DCs ordered by submission [per execution stage]
    CommandSubmissionDataChunksDB m_usedCsDataChunksDb[EXECUTION_STAGE_LAST];
    UsedCsDcStageTypeDB           m_usedCsDcStageTypeDb;

    // std::map<uint64_t, CommandSubmissionDataChunksDB> execHandle-> deq(csdc) [per execution stage]
    freeCommandSubmissionDataChunksDB m_freeCsDataChunksDb[EXECUTION_STAGE_LAST];

    std::vector<uint32_t> m_tensorIdx2userIdx[NUM_TENSOR_TYPES];

    uint64_t m_currentExecutionHandle;

    uint64_t m_lastProgramDataHandle;
    uint64_t m_lastProgramCodeHandle;

    std::vector<uint64_t> m_dataChunksHostAddresses;

    std::unique_ptr<DynamicRecipe> m_dynamicRecipe;

    QmanDefinitionInterface* m_qmansDef;

    staticBlobsAddressInWS m_staticBlobsAddressesInWs;

    SubmitCommandBuffersInterface& m_rSubmitter;

    DevMemoryAllocInterface& m_rDevMemAlloc;

    mutable std::mutex m_csdcMutex;
};

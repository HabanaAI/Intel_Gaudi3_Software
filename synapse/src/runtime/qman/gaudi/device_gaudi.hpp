#pragma once

#include "runtime/common/device/device_common.hpp"
#include "runtime/qman/common/device_queues.hpp"
#include "synapse_common_types.h"
#include "runtime/qman/common/device_mapper.hpp"
#include "recipe_package_types.hpp"
#include "global_statistics.hpp"
#include "recipe.h"
#include "runtime/qman/common/wcm/wcm_cs_querier.hpp"
#include "runtime/qman/common/wcm/work_completion_manager.hpp"
#include "runtime/qman/common/device_recipe_addresses_generator.hpp"
#include "runtime/qman/common/command_buffer_packet_generator.hpp"
#include "runtime/qman/common/device_downloader.hpp"
#include "runtime/qman/common/device_recipe_downloader_container.hpp"
#include <queue>

struct basicRecipeInfo;
struct DeviceAgnosticRecipeInfo;
class RecipeStaticInfo;
class DeviceDownloaderInterface;
struct StagedInfo;
class DeviceRecipeDownloader;
class QmanDefinitionInterface;

class DeviceGaudi : public DeviceCommon
{
public:
    DeviceGaudi(const DeviceConstructInfo& deviceConstructInfo);

    virtual ~DeviceGaudi() {}

    virtual synStatus acquire(const uint16_t numSyncObj) override;

    static uint32_t getAmountOfEnginesInComputeArbGroupAquire();

    virtual const std::vector<uint64_t> getTpcAddrVector() override
    {
        return s_tpcAddrVector;
    }

    virtual synStatus release(std::atomic<bool>& rDeviceBeingReleased) override;

    virtual synStatus getDramMemInfo(uint64_t& free, uint64_t& total) const override;

    virtual synStatus getDeviceInfo(synDeviceInfo& rDeviceInfo) const override;

    // Stream operations
    virtual synStatus
    createStreamQueue(QueueType queueType, uint32_t flags, bool isReduced, QueueInterface*& rpQueueInterface) override;

    virtual synStatus destroyStreamQueue(QueueInterface* pQueueInterface) override;

    synStatus synchronizeEvent(const EventInterface* pEventInterface) override;

    synStatus eventQuery(const EventInterface* pEventHandle) override;

    synStatus createEvent(synEventHandle* pEventHandle, const unsigned int flags) override;

    synStatus destroyEvent(synEventHandle eventHandle) override;

    EventSptr getEventSptr(synEventHandle eventHandle) override;

    synStatus getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const override;

    virtual void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle) override;

    virtual void
    checkDevFailure(uint64_t csSeqTimeout, DfaStatus dfaStatus, ChkDevFailOpt option, bool isSimulator) override;

    virtual void bgWork() override {}

    virtual void debugCheckWorkStatus() override {}

    void addAddrToReleaseOnStreamDestroy(uint64_t addr);

    QueueInterface* getDmaDownStream() { return m_deviceStreams.getDmaDownStream(); }

    synStatus getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress);

    synStatus allocateRecipeMemory(const synRecipeHandle      recipeHandle,
                                   const synLaunchTensorInfo* launchTensorsInfo,
                                   const uint32_t             numberTensors,
                                   DeviceRecipeMemory&        recipeMem,
                                   bool&                      isMemoryAddedToReleaseList);

    synStatus releaseRecipeMemory(const DeviceRecipeMemory& recipeMem, const synRecipeHandle recipeHandle);

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

    virtual synStatus
    kernelsPrintf(const InternalRecipeHandle& rInternalRecipeHandle, uint64_t wsAddr, void* hostBuff) override;

    synStatus submitCommandBuffers(CommandSubmission&   commandSubmission,
                                   uint64_t*            csHandle,
                                   uint64_t*            mappedBuff,
                                   const uint32_t       physicalQueueOffset,
                                   const StagedInfo*    pStagedInfo,
                                   globalStatPointsEnum point = globalStatPointsEnum::colLast);

    static synStatus
    generateArbitratorsDefaultConfigPacketsCommon(char*&                                 pPackets,
                                                  uint64_t&                              packetsSize,
                                                  generic::CommandBufferPktGenerator*    pCmdBuffPktGenerator,
                                                  generic::masterSlaveArbitrationInfoDB& masterSlaveArbInfoDb,
                                                  const std::deque<uint32_t>*            pEnginesWithArbitrator);

    void dfaInfo(DfaReq dfaReq, uint64_t csSeq);

    virtual synStatus getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                      synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;

    synStatus submitTrainingConfigurationCS(synDeviceType      deviceType,
                                            char*&             pPackets,
                                            uint64_t           packetsSize,
                                            const std::string& operationDescription,
                                            uint32_t           queueId,
                                            bool               isConfigOnInternal         = false,
                                            bool               isSyncWithExternalRequired = false,
                                            uint32_t           waitQmanId                 = 0);

    virtual void getDeviceHbmVirtualAddresses(uint64_t& hbmBaseAddr, uint64_t& hbmEndAddr) override;

protected:
    virtual synStatus eventRecord(EventInterface* pEventInterface, synStreamHandle streamHandle) override;
    virtual synStatus streamGenericWaitEvent(synStreamHandle       streamHandle,
                                             const EventInterface& rEventInterface,
                                             const unsigned int    flags) override;

    // By design - we expect the user to ensure that there is no destroy of handles during execution
    // To emphasize - Not only during the call, but until execution ends
    synStatus launch(Stream*                       pStream,
                     const synLaunchTensorInfoExt* launchTensorsInfo,
                     uint32_t                      launchTensorsInfoAmount,
                     uint64_t                      workspaceAddress,
                     InternalRecipeHandle*         pRecipeHandle,
                     EventWithMappedTensorDB&      events,
                     uint32_t                      flags) override;

    TrainingRetCode validateEventHandle(const EventInterface* pEventHandle);

    synStatus allocateResources(const uint16_t numSyncObj);

    synStatus allocateDevMem();

    synStatus allocateSharedStreams();

    void stopWorkCompletionManager();

    synStatus releaseAllStreams();

    synStatus releaseDevMem();

    synStatus releaseResources();

    synStatus deallocateRecipesAddresses();

    static synStatus _getPhysicalQueueId(uint32_t& queueId, synDeviceType deviceType, bool isCopyFromHost);

    static const uint32_t INVALID_PHYSICAL_QUEUE_ID;

    static synDmaDir getDir(internalDmaDir direction);

    void kernelsPrintfAfterLaunch(InternalRecipeHandle& rInternalRecipeHandle, uint64_t workspaceAddress);

    void dumpCsStatistics() const;

    void notifyAllRecipeRemoval();

    synStatus getTdrIrqMonitorArmRegAddr(volatile uint32_t*& tdrIrqMonitorArmRegAddr) override
    {
        return synUnsupported;
    }

    // for tracing we use context_id to identify packets coming from the same api call
    static std::atomic<uint32_t> s_linDmaContextId;

    DeviceRecipeAddressesGenerator                                m_deviceRecipeAddressesGenerator;
    std::queue<uint64_t>                                          m_deviceAddressesToReleaseOnStreamDestroy;
    std::unordered_map<InternalRecipeHandle*, DeviceRecipeMemory> m_recipeToMemoryMap;

    WcmCsQuerier                                     m_multiCsQuerier;
    WorkCompletionManager                            m_workCompletionManager;
    DeviceMapper                                     m_deviceMapper;
    QmanDefinitionInterface*                         m_pQmansDefinition;
    generic::CommandBufferPktGenerator*              m_pCmdBuffPktGenerator;
    const AffinityCountersArray                      m_allocationAffinities;
    DeviceQueues                                     m_deviceStreams;
    std::unique_ptr<DeviceDownloader>                m_pDeviceDownloader;
    std::unique_ptr<DeviceRecipeDownloaderContainer> m_pDeviceRecipeDownloaderContainer;

private:
    synStatus
    _syncWithInnerQueueOperation(synDeviceType deviceType, synCommandBuffer** ppCommandBuffers, uint32_t waitQmanId);

    void _mapCB(CommandSubmission* cb);

    synStatus submitPredicateDefaultConfiguration();

    synStatus deviceAcquireConfig();

    synStatus submitArbitratorsDefaultConfigurationForGaudi();

    synStatus generateArbitratorsDefaultConfigPackets(char*& pPackets, uint64_t& packetsSize, synDeviceType deviceType);

    synStatus generatePredicateDataOnHostWithMapping(unsigned numPreds, std::shared_ptr<uint8_t>& table);

    synStatus generatePredicateConfigurationPackets(char*&    pPackets,
                                                    uint64_t& packetsSize,
                                                    uint64_t  predAddr,
                                                    bool      isSyncWithExternalRequired);

    static const AffinityCountersArray s_maxAffinitiesDefault;
    static const AffinityCountersArray s_maxAffinitiesHCLDisable;
    static const AffinityCountersArray s_allocationAffinitiesDefault;
    static const AffinityCountersArray s_allocationAffinitiesHCLDisable;
    static const std::vector<uint64_t> s_tpcAddrVector;
};

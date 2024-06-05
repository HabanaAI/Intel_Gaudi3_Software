#pragma once

#include "runtime/common/device/device_interface.hpp"
#include "runtime/common/device/device_mem_alloc.hpp"
#include "runtime/common/device/dfa_base.hpp"
#include "runtime/common/queues/queue_compute_utils.hpp"
#include <shared_mutex>
#include <memory>
#include "efd_controller.hpp"
#include "synapse_common_types.h"
#include "runtime/common/streams/streams_container.hpp"

struct hl_debug_args;
struct hlthunk_time_sync_info;
class QueueInterface;
struct BasicQueueInfo;
class DfaBase;
class Stream;

namespace generic
{
class CommandBufferPktGenerator;
}

struct DeviceConstructInfo
{
    const synDeviceInfo deviceInfo;
    const int           fdControl;
    const uint32_t      hlIdx;
    const uint32_t      devModuleIdx;
    const uint32_t      devIdType;
    const std::string   pciAddr;
};

class DeviceCommon : public DeviceInterface
{
public:
    DeviceCommon(synDeviceType                devType,
                 DevMemoryAlloc*              devMemoryAlloc,
                 const DeviceConstructInfo&   deviceConstructInfo,
                 bool                         setStreamAffinityByJobType,
                 const AffinityCountersArray& rMaxAffinities);

    virtual ~DeviceCommon();

    virtual synStatus mapBufferToDevice(uint64_t           size,
                                        void*              buffer,
                                        bool               isUserRequest,
                                        uint64_t           reqVAAddress,
                                        const std::string& mappingDesc) override;

    virtual synStatus unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize) override;

    // need to refactor allocateMemory()
    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) override;

    virtual synStatus deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest) override;

    // Note, do not use the one in m_osalInfo,
    // they are not always the same (see GaudiB)
    virtual synDeviceType getDevType() const override { return m_devType; }

    uint32_t getDevHlIdx() const override final { return m_hlIdx; }

    virtual eMappingStatus getDeviceVirtualAddress(bool         isUserRequest,
                                                   void*        hostAddress,
                                                   uint64_t     bufferSize,
                                                   uint64_t*    pDeviceVA,
                                                   bool*        pIsExactKeyFound = nullptr) override
    {
        return m_devMemoryAlloc.get()->getDeviceVirtualAddress(isUserRequest,
                                                               hostAddress,
                                                               bufferSize,
                                                               pDeviceVA,
                                                               pIsExactKeyFound);
    };

    virtual synStatus getDeviceLimitationInfo(synDeviceLimitationInfo& rDeviceLimitationInfo) override;

    virtual const std::vector<uint64_t> getTpcAddrVector() = 0;

    synStatus createSynCommandBuffer(synCommandBuffer* pSynCB, uint32_t queueId, uint64_t commandBufferSize) override;

    synStatus waitAndReleaseCS(uint64_t  handle,
                               uint64_t  timeout,
                               bool      returnUponTimeout = false,
                               bool      collectStats      = false,
                               uint64_t* usrEventTime      = nullptr) override;

    synStatus profile(hl_debug_args* args) override;

    synStatus getClockSyncInfo(hlthunk_time_sync_info* infoOut) override;

    synStatus getPllFrequency(uint32_t index, struct hlthunk_pll_frequency_info* freqOut) override;

    virtual synStatus streamQuery(synStreamHandle streamHandle) override;
    virtual synStatus
    streamWaitEvent(synStreamHandle streamHandle, synEventHandle eventHandle, unsigned int flags) override;

    virtual synStatus synchronizeAllStreams() override;

    synStatus eventRecord(synEventHandle eventHandle, synStreamHandle streamHandle) override;
    using DeviceInterface::eventRecord;
    synStatus synchronizeEvent(synEventHandle eventHandle) override;
    using DeviceInterface::synchronizeEvent;
    synStatus eventQuery(synEventHandle eventHandle) override;
    using DeviceInterface::eventQuery;
    synStatus synchronizeStream(synStreamHandle streamHandle) override;

    virtual synStatus launchWithExternalEvents(const synStreamHandle         streamHandle,
                                               const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                               const uint32_t                enqueueTensorsAmount,
                                               uint64_t                      workspaceAddress,
                                               const synRecipeHandle         pRecipeHandle,
                                               synEventHandle*               eventHandleList,
                                               uint32_t                      numberOfEvents,
                                               uint32_t                      flags) override;

    virtual synStatus memcopy(const synStreamHandle  streamHandle,
                              internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest) override;

    virtual synStatus memSet(const synStreamHandle streamHandle,
                             uint64_t              pDeviceMem,
                             const uint32_t        value,
                             const size_t          numOfElements,
                             const size_t          elementSize) override;

    synStatus waitAndReleaseStreamHandles(const InternalWaitHandlesVector& streamWaitHandles,
                                          uint64_t                         timeout,
                                          bool                             returnUponTimeout) override;

    virtual synStatus createStream(QueueType queueType, unsigned int flags, synStreamHandle& rStreamHandle) override;

    virtual synStatus destroyStream(synStreamHandle streamHandle) override;

    static generic::CommandBufferPktGenerator* _getCommandBufferGeneratorInstance(synDeviceType deviceType);

    synStatus getDeviceAttribute(const synDeviceAttribute* deviceAttr,
                                 const unsigned            querySize,
                                 uint64_t*                 retVal) override;

    virtual synStatus getPCIBusId(char* pPciBusId, const int len) override;

    virtual void notifyHlthunkFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo = {}) override;
    virtual void notifyDeviceFailure(DfaErrorCode errCode, const DfaExtraInfo& dfaExtraInfo = {}) override;
    virtual void notifyEventFd(uint64_t events) override;

    virtual void addDfaObserver(DfaObserver* pObserver) override { m_dfa.addObserver(pObserver); }

    virtual DfaStatus getDfaStatus() override;

    static synStatus createAndAddHangCommandBuffer(synCommandBuffer*& pSynCB,
                                                   uint32_t           cbIndex,
                                                   const void*        pBuffer,
                                                   uint64_t           commandBufferSize,
                                                   uint32_t           queueId,
                                                   bool               isForceMmuMapped = false);

    virtual synStatus setStreamAffinity(synStreamHandle streamHandle, uint64_t streamAffinityMask) override;

    virtual synStatus getStreamAffinity(synStreamHandle streamHandle, uint64_t* streamAffinityMask) const override;

    virtual synStatus getDeviceAffinityMaskRange(uint64_t& rDeviceAffinityMaskRange) const override;

    virtual synStatus getDeviceNextStreamAffinity(uint64_t& nextDeviceAffinity) override;

    virtual synStatus syncHCLStreamHandle(synStreamHandle streamHandle) override;

    virtual synStatus flushWaitsOnCollectiveStream(synStreamHandle streamHandle) override;

    virtual uint32_t getNetworkStreamPhysicalQueueOffset(synStreamHandle streamHandle) override;

    virtual hcl::hclStreamHandle getNetworkStreamHclStreamHandle(synStreamHandle streamHandle) override;

    enum ChkDevFailOpt {MAIN, CCB};
    virtual void
    checkDevFailure(uint64_t csSeqTimeout, DfaStatus dfaStatus, ChkDevFailOpt option, bool isSimulator) = 0;

    void dfaLogMappedMem() const { m_devMemoryAlloc->dfaLogMappedMem(); };

    virtual bool isDirectModeUserDownloadStream() const override { return false; };

    std::chrono::system_clock::time_point getAcquireTime() { return m_acquireTime; }
    std::string                           getPciAddr()     { return m_pciAddr; }

    void testingOnlySetBgFreq(std::chrono::milliseconds period);

    virtual synStatus getDeviceInfo(synDeviceInfoV2& rDeviceInfo) const override;

    virtual synStatus getTdrIrqMonitorArmRegAddr(volatile uint32_t*& tdrIrqMonitorArmRegAddr) = 0;

    const synDeviceInfo& getDeviceOsalInfo() const { return m_osalInfo; };

    virtual void getDeviceHbmVirtualAddresses(uint64_t& hbmBaseAddr, uint64_t& hbmEndAddr) = 0;

    synStatus getClockSyncPerDieInfo(uint32_t dieIndex, hlthunk_time_sync_info* infoOut) override;


protected:
    friend class DfaDevCrashTests;
    friend class SynGaudiInflightParserTests;
    friend class MultiStreamsTest;
    friend class SynAPITest;
    friend class ScalStreamTest;
    friend class ScalStreamTest;
    friend class SynScalLaunchDummyRecipe;

    virtual synStatus
    createStreamQueue(QueueType queueType, uint32_t flags, bool isReduced, QueueInterface*& rpQueueInterface) = 0;

    virtual synStatus destroyStreamQueue(QueueInterface* pQueueInterface) = 0;

    virtual synStatus createStreamGeneric(uint32_t flags, synStreamHandle& rStreamHandle) override;

    synStatus destroyStreamGeneric(synStreamHandle streamHandle);

    synStatus synchronizeAllStreamsGeneric();

    SlotMapItemSptr<Stream> loadAndValidateStream(synStreamHandle streamHandle, const char* functionName) const;

    virtual synStatus eventRecord(EventInterface* pEventInterface, synStreamHandle streamHandle) = 0;

    synStatus
    streamWaitEventInterface(synStreamHandle streamHandle, EventInterface* pEventInterface, unsigned int flags);

    virtual synStatus streamGenericWaitEvent(synStreamHandle       streamHandle,
                                             const EventInterface& rEventInterface,
                                             const unsigned int    flags)                           = 0;
    EventSptr loadAndValidateEvent(synEventHandle eventHandle, const char* functionName);

    static bool _checkRecipeCacheOverlap(uint64_t  srcAddress,
                                         uint64_t  dstAddress,
                                         uint64_t  recipeCacheBaseAddress,
                                         uint64_t  recipeCacheLastAddress,
                                         uint64_t  size,
                                         synDmaDir dir,
                                         bool      isMemset);

    synStatus startEventFdThread();
    synStatus stopEventFdThread();

    const void* getAssertAsyncHostAddress() const { return m_assertAsyncBufferHostAddr; }
    const uint64_t getAssertAsyncMappedAddress() const { return m_assertAsyncBufferMappedAddr; }

    bool isAssertAsyncNoticed() const override { return m_dfa.isAssertAsyncNoticed(); }

    static synStatus getInternalStreamTypes(QueueType queueType, internalStreamType* internalType);

    static void dfaLogFunc(int logLevel, const char* msg);
    static void dfaLogFuncErr(int logLevel, const char* msg);

    static const std::unordered_map<unsigned, internalStreamType> s_streamToInternalStreamMap;

    // By design - we expect the user to ensure that there is no destroy of handles during execution
    // To emphasize - Not only during the call, but until execution ends
    virtual synStatus launch(Stream*                       pStream,
                             const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsInfoAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             EventWithMappedTensorDB&      events,
                             uint32_t                      flags) = 0;

    synStatus addStreamAffinities(const AffinityCountersArray& rAffinityArr, bool isReduced);

    synStatus removeAllStreamAffinities();

    synStatus
    createStreamQueues(AffinityCountersArray affinityArr, bool isReduced, QueueInterfacesArrayVector& rQueueInterfaces);

    synStatus destroyStreamQueues(AffinityCountersArray affinityArr, QueueInterfacesArrayVector& rQueueInterfaces);

    uint8_t generateApiId() override { return ((m_apiId++) & s_apiIdMask); }

    const synDeviceType                         m_devType;
    const synDeviceInfo                         m_osalInfo;  // Note: the device type in here should not be used
    const std::string                           m_pciAddr;
    const std::chrono::system_clock::time_point m_acquireTime;
    const uint32_t                              m_hlIdx;
    const AffinityCountersArray                 m_maxAffinities;

    // The device mutex logic is as follows:
    // WR lock: acquire, release, create/destroy stream
    // RD lock: otherwise
    mutable std::shared_mutex                m_mutex;
    std::unique_ptr<DevMemoryAllocInterface> m_devMemoryAlloc;
    DfaBase                                  m_dfa;
    EventFdController                        m_eventFdController;
    void*                                    m_assertAsyncBufferHostAddr;
    uint64_t                                 m_assertAsyncBufferMappedAddr;
    StreamsContainer                         m_streamsContainer;
    std::atomic_uchar                        m_apiId;
    static const uint8_t                     s_apiIdMask;
};

#include <gtest/gtest.h>
#include "runtime/scal/common/recipe_launcher/recipe_launcher.hpp"
#include "common/dev_memory_alloc_mock.hpp"
#include "runtime/scal/common/recipe_launcher/mem_mgrs.hpp"
#include "scal_stream_compute_interface.hpp"
#include "recipe_handle_impl.hpp"

class LaunchTrackerMock : public LaunchTrackerInterface
{
public:
    synStatus waitForCompletionCopy() override { return synSuccess; };
};

class ScalStreamComputeMock : public ScalStreamComputeInterface
{
public:
    bool isDirectMode() const override { return false; };

    const std::string getName() const override { return ""; };

    synStatus addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo) override { return synSuccess; };

    void longSoRecord(bool isUserReq, ScalLongSyncObject& rLongSo) const override {};

    synStatus eventRecord(bool isUserReq, ScalEvent& scalEvent) const override { return synSuccess; };

    synStatus longSoQuery(const ScalLongSyncObject& rLongSo, bool alwaysWaitForInterrupt = false) const override
    {
        return synSuccess;
    };

    // Wait on host for last longSo (e.g. see isUserReq + longSo @ addDispatchBarrier) to complete on the device
    synStatus longSoWaitForLast(bool isUserReq, uint64_t timeoutMicroSec, const char* caller) const override
    {
        return synSuccess;
    };

    synStatus longSoWaitOnDevice(const ScalLongSyncObject& rLongSo, bool isUserReq) override { return synSuccess; };

    ScalLongSyncObject getIncrementedLongSo(bool isUserReq, uint64_t targetOffset = 1) override { return {}; };

    ScalLongSyncObject getTargetLongSo(uint64_t targetOffset) const override { return {}; };

    // Wait on host for longSo (e.g. see longSo @ addDispatchBarrier) to complete on the device
    synStatus longSoWait(const ScalLongSyncObject& rLongSo, uint64_t timeoutMicroSec, const char* caller) const override
    {
        return synSuccess;
    };

    synStatus addStreamFenceWait(uint32_t target, bool isUserReq, bool isInternalComputeSync) override
    {
        return synSuccess;
    };

    ScalMonitorBase* testGetScalMonitor() override { return nullptr; };

    // Adds a packet, which writes data to an LBW-address
    synStatus addLbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool send, bool isInSyncMgr) override
    {
        return synSuccess;
    };

    synStatus getStreamInfo(std::string& info, uint64_t& devLongSo) override { return synSuccess; };

    bool prevCmdIsWait() override { return synSuccess; };

    void dfaDumpScalStream() override {};

    TdrRtn tdr(TdrType tdrType) override { return TdrRtn(); };

    void printCgTdrInfo(bool tdr) const override {};

    bool getStaticMonitorPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData) const override { return true; };

    ResourceStreamType getResourceType() const override { return ResourceStreamType::USER_DMA_DOWN; };

    bool isComputeStream() override { return true; };

    synStatus memcopy(ResourceStreamType           resourceType,
                      const internalMemcopyParams& memcpyParams,
                      bool                         isUserRequest,
                      bool                         send,
                      uint8_t                      apiId,
                      ScalLongSyncObject&          longSo,
                      uint64_t                     overrideMemsetVal,
                      MemcopySyncInfo&             memcopySyncInfo) override
    {
        return synSuccess;
    };

    synStatus addGlobalFenceWait(uint32_t target, bool send) override { return synSuccess; };

    synStatus addGlobalFenceInc(bool send) override { return synSuccess; };

    synStatus addAllocBarrierForLaunch(bool allocSoSet, bool relSoSet, McidInfo mcidInfo, bool send) override { return synSuccess; };

    synStatus addDispatchBarrier(ResourceStreamType  resourceType,
                                 bool                isUserReq,
                                 bool                send,
                                 ScalLongSyncObject& longSo,
                                 uint16_t            additionalTdrIncrement = 0) override
    {
        return synSuccess;
    };

    synStatus addDispatchBarrier(const EngineGrpArr& engineGrpArr,
                                 bool                isUserReq,
                                 bool                send,
                                 ScalLongSyncObject& longSo,
                                 uint16_t            additionalTdrIncrement = 0) override
    {
        return synSuccess;
    };

    synStatus addUpdateRecipeBaseAddresses(const EngineGrpArr& engineGrpArr,
                                           uint32_t            numOfRecipeBaseElements,
                                           const uint64_t*     recipeBaseAddresses,
                                           const uint16_t*     recipeBaseIndices,
                                           bool                send) override
    {
        return synSuccess;
    };

    synStatus addDispatchComputeEcbList(uint8_t  logicalEngineId,
                                        uint32_t addrStatic,
                                        uint32_t sizeStatic,
                                        uint32_t singlePhysicalEngineStaticOffset,
                                        uint32_t addrDynamic,
                                        uint32_t sizeDynamic,
                                        bool     shouldUseGcNopKernel,
                                        bool     send) override
    {
        return synSuccess;
    };
};

TEST(UTRecipeLauncherTests, basic)
{
    ScalStreamComputeMock           computeScalStream;
    ScalStreamComputeInterface*     pComputeScalStream = &computeScalStream;
    const ComputeCompoundResources* pComputeResources  = nullptr;
    DevMemoryAllocMock              devMemoryAlloc;
    MemMgrs                         memMgrs("mock", devMemoryAlloc);
    LaunchTrackerMock               launchTracker;
    recipe_t                        recipe {};
    InternalRecipeHandle            internalRecipeHandle {};
    internalRecipeHandle.basicRecipeHandle.recipe       = &recipe;
    const InternalRecipeHandle* pRecipeHandle           = &internalRecipeHandle;
    DynamicRecipe*              pDynamicRecipeProcessor = nullptr;
    const uint64_t              runningId               = 0;
    const uint8_t               apiId                   = 0;

    RecipeLauncher launcher(pComputeScalStream,
                            pComputeResources,
                            devMemoryAlloc,
                            memMgrs,
                            launchTracker,
                            pRecipeHandle,
                            pDynamicRecipeProcessor,
                            runningId,
                            apiId);

    const InternalRecipeHandle& actualInternalRecipeHandle = launcher.getInternalRecipeHandle();
    ASSERT_EQ(&actualInternalRecipeHandle, &internalRecipeHandle);
}

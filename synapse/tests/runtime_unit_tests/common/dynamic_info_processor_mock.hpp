#pragma once

#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/qman/common/wcm/wcm_types.hpp"
#include "runtime/qman/common/dynamic_info_processor_interface.hpp"

class DynamicInfoProcessorMock : public IDynamicInfoProcessor
{
    friend class UTStreamComputeTest;

public:
    DynamicInfoProcessorMock(const InternalRecipeHandle& internalRecipeHandle);

    virtual ~DynamicInfoProcessorMock() override = default;

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
                                                        bool                            keepOne) override
    {
        return 0;
    }

    // Release and retrieve a CS-DC element of current execution-handle
    virtual CommandSubmissionDataChunks* getAvailableCommandSubmissionDataChunks(eExecutionStage executionStage,
                                                                                 bool            isCurrent) override
    {
        return nullptr;
    }

    virtual void incrementExecutionHandle() override { m_executionHandle++; }

    virtual uint64_t getExecutionHandle() override { return m_executionHandle; }

    virtual void getProgramCodeHandle(uint64_t& programCodeHandle) const override
    {
        programCodeHandle = s_programCodeHandle;
    }

    virtual void getProgramDataHandle(uint64_t& programDataHandle) const override
    {
        programDataHandle = s_programDataHandle;
    }

    virtual void setProgramCodeHandle(uint64_t staticCodeHandle) override {}

    virtual void setProgramCodeAddrInWS(uint64_t programCodeAddrInWS) override {}

    virtual void setProgramDataHandle(uint64_t programDataHandle) override {}

    bool isAnyInflightCsdc() override { return false; }

    virtual const char* getRecipeName() override { return ""; }
    virtual uint64_t    getRecipeId() const override { return 0; }

    virtual const basicRecipeInfo& getRecipeBasicInfo() override { return m_rBasicRecipeInfo; }
    virtual const DeviceAgnosticRecipeInfo& getDevAgnosticInfo() override { return m_rDeviceAgnosticRecipeInfo; }

    virtual std::vector<tensor_info_t> getDynamicShapesTensorInfoArray() const override
    {
        return std::vector<tensor_info_t>();
    }

    virtual DynamicRecipe* getDsdPatcher()
    {
        return m_dsdPatcher.get();
    }

    virtual bool resolveTensorsIndices(std::vector<uint32_t>*&       tensorIdx2userIdx,
                                       const uint32_t                launchTensorsAmount,
                                       const synLaunchTensorInfoExt* launchTensorsInfo) override
    {
        return true;
    }

private:
    static const uint64_t s_programCodeHandle;
    static const uint64_t s_programDataHandle;

    const basicRecipeInfo&          m_rBasicRecipeInfo;
    const DeviceAgnosticRecipeInfo& m_rDeviceAgnosticRecipeInfo;

    uint64_t m_executionHandle;
    uint64_t m_csHandle;

    std::deque<std::pair<WcmCsHandle, bool>> m_csHandles;
    std::unique_ptr<DynamicRecipe>           m_dsdPatcher;
};

#include "dynamic_info_processor_mock.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include <gtest/gtest.h>
#include "defenders.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

const uint64_t DynamicInfoProcessorMock::s_programCodeHandle = 0x0;
const uint64_t DynamicInfoProcessorMock::s_programDataHandle = 0x0;

DynamicInfoProcessorMock::DynamicInfoProcessorMock(const InternalRecipeHandle& internalRecipeHandle)
: m_rBasicRecipeInfo(internalRecipeHandle.basicRecipeHandle),
  m_rDeviceAgnosticRecipeInfo(internalRecipeHandle.deviceAgnosticRecipeHandle),
  m_executionHandle(0), m_csHandle(0)
{
}

synStatus DynamicInfoProcessorMock::enqueue(const synLaunchTensorInfoExt* launchTensorsInfo,
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
                                            eCsDcProcessorStatus&         csDcProcessingStatus)
{
    csHandle = ++m_csHandle;
    pCsDataChunks->addWaitForEventHandle(csHandle);
    csDcProcessingStatus = CS_DC_PROCESSOR_STATUS_STORED_AND_SUBMITTED;
    return synSuccess;
}

bool DynamicInfoProcessorMock::notifyCsCompleted(CommandSubmissionDataChunks* pCsDataChunks,
                                                 uint64_t                     waitForEventHandle,
                                                 bool                         csFailed)
{
    bool status = pCsDataChunks->containsHandle(waitForEventHandle);
    HB_ASSERT(status,
              "{}: CSDC {:#x} does not contain waitForEventHandle {:#x}",
              __FUNCTION__,
              TO64(pCsDataChunks),
              waitForEventHandle);
    status = pCsDataChunks->popWaitForEventHandle(waitForEventHandle);
    HB_ASSERT(status,
              "{}: CSDC {:#x} cannot pop waitForEventHandle {:#x}",
              __FUNCTION__,
              TO64(pCsDataChunks),
              waitForEventHandle);
    m_csHandles.push_back({waitForEventHandle, csFailed});
    return true;
}

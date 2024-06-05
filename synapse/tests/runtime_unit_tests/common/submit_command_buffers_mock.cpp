#include "submit_command_buffers_mock.hpp"

SubmitCommandBuffersMock::SubmitCommandBuffersMock() : m_csHandle(0x1234), mSubmitCommandBuffersCounter(0) {}

synStatus SubmitCommandBuffersMock::submitCommandBuffers(const BasicQueueInfo& rBasicQueueInfo,
                                                         uint32_t              physicalQueueOffset,
                                                         CommandSubmission&    commandSubmission,
                                                         uint64_t*             csHandle,
                                                         uint32_t              queueOffset,
                                                         const StagedInfo*     pStagedInfo,
                                                         unsigned int          stageIdx)
{
    *csHandle = m_csHandle;
    mSubmitCommandBuffersCounter++;
    return synSuccess;
}
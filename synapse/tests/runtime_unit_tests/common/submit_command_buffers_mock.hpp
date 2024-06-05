#pragma once

#include "runtime/qman/common/submit_command_buffers_interface.hpp"

class SubmitCommandBuffersMock : public SubmitCommandBuffersInterface
{
public:
    SubmitCommandBuffersMock();

    virtual ~SubmitCommandBuffersMock() = default;

    virtual synStatus submitCommandBuffers(const BasicQueueInfo& rBasicQueueInfo,
                                           uint32_t              physicalQueueOffset,
                                           CommandSubmission&    commandSubmission,
                                           uint64_t*             csHandle,
                                           uint32_t              queueOffset,
                                           const StagedInfo*     pStagedInfo,
                                           unsigned int          stageIdx) override;

    const uint64_t m_csHandle;
    uint64_t       mSubmitCommandBuffersCounter;
};

#pragma once

#include "submit_command_buffers_interface.hpp"

// Todo change to work without SynSingleton (object will be supplied from Device)
class SubmitCommandBuffers : public SubmitCommandBuffersInterface
{
public:
    SubmitCommandBuffers()          = default;
    virtual ~SubmitCommandBuffers() = default;
    virtual synStatus submitCommandBuffers(const BasicQueueInfo& rBasicQueueInfo,
                                           uint32_t              physicalQueueOffset,
                                           CommandSubmission&    commandSubmission,
                                           uint64_t*             csHandle,
                                           uint32_t              queueOffset,
                                           const StagedInfo*     pStagedInfo,
                                           unsigned int          stageIdx) override;
};

#pragma once

#include <cstdint>
#include "synapse_common_types.h"

struct BasicQueueInfo;
class CommandSubmission;
struct StagedInfo;

class SubmitCommandBuffersInterface
{
public:
    virtual ~SubmitCommandBuffersInterface()                      = default;
    virtual synStatus submitCommandBuffers(const BasicQueueInfo& rBasicQueueInfo,
                                           uint32_t              physicalQueueOffset,
                                           CommandSubmission&    commandSubmission,
                                           uint64_t*             csHandle,
                                           uint32_t              queueOffset,
                                           const StagedInfo*     pStagedInfo,
                                           unsigned int          stageIdx) = 0;
};

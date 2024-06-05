#pragma once

#include <memory>
#include "synapse_common_types.h"
#include "define_synapse_common.hpp"
#include "runtime/qman/common/qman_types.hpp"

class QueueInfo;

typedef std::shared_ptr<QueueInfo> spQueueInfo;

struct BasicQueueInfo
{
    union
    {
        struct
        {
            uint32_t reserved;
            uint32_t queueIndex;
        };

        uint64_t handle;
    };

    internalStreamType queueType;
    TrainingQueue      logicalType;
    uint32_t           userQueueIndex;
    spQueueInfo        pQueueInfo;

    std::string getDescription() const;
};

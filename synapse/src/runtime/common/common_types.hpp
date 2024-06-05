// This file is intended for synapse runtime internal define declarations

#pragma once

#include "synapse_api_types.h"
#include <vector>
#include <array>

struct InternalWaitHandle
{
    uint64_t handle;
};

enum QueueType
{
    QUEUE_TYPE_COPY_DEVICE_TO_HOST,
    QUEUE_TYPE_COPY_HOST_TO_DEVICE,
    QUEUE_TYPE_COPY_DEVICE_TO_DEVICE,
    QUEUE_TYPE_COMPUTE,
    QUEUE_TYPE_NETWORK_COLLECTIVE,
    // Todo [SW-136740] remove QUEUE_TYPE_MAX_USER_TYPES
    QUEUE_TYPE_MAX_USER_TYPES,
    // Todo [SW-136740] remove QUEUE_TYPE_RESERVED_1
    QUEUE_TYPE_RESERVED_1 = QUEUE_TYPE_MAX_USER_TYPES,
    QUEUE_TYPE_MAX
};

using InternalWaitHandlesVector = std::vector<InternalWaitHandle>;

using AffinityCountersArray = std::array<unsigned, QUEUE_TYPE_MAX_USER_TYPES>;

class QueueInterface;

using QueueInterfacesArray = std::array<QueueInterface*, QUEUE_TYPE_MAX_USER_TYPES>;

using QueueInterfacesArrayVector = std::vector<QueueInterfacesArray>;

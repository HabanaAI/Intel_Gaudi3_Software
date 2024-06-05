#pragma once

#include "habana_device_types.h"

#include "../../graph_compiler/sync/sync_types.h"

#include "drm/habanalabs_accel.h"

#include <cstdint>

namespace gaudi2
{
    uint64_t getQMANBase(     HabanaDeviceType   type,
                              unsigned           deviceID);

    uint64_t getCPFenceOffset(HabanaDeviceType   type,
                              unsigned           deviceID,
                              WaitID             waitID,
                              unsigned           streamID,
                              bool               isForceStreamId = false);

    uint64_t getCPFenceOffset(gaudi2_queue_id    queueId,
                              WaitID             waitID);

    bool getQueueIdInfo(      HabanaDeviceType&   type,
                              unsigned&           deviceID,
                              unsigned&           streamID,
                              gaudi2_queue_id     queueId);
}

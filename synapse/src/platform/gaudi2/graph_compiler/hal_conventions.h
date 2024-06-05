#pragma once

#include "drm/habanalabs_accel.h"
#include "habana_device_types.h"
#include "hal_reader/gaudi2/hal_reader.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace gaudi2
{
enum LogicalQueue
{
    DEVICE_MME_LOGICAL_QUEUE        = 0,
    DEVICE_TPC_LOGICAL_QUEUE        = 1,
    DEVICE_DMA_LOGICAL_QUEUE        = 2,
    DEVICE_ROT_LOGICAL_QUEUE        = 3,
    DEVICE_COMPLETION_LOGICAL_QUEUE = 4,  // For legacy sync scheme, doesn't use groups

    LOGICAL_QUEUE_MAX_ID
};

gaudi2_queue_id  getQueueID(HabanaDeviceType type, unsigned id);
std::string      getEngineName(gaudi2_queue_id id);
std::string_view getQmanIdName(uint32_t id);
unsigned         deviceTypeToLogicalQueue(HabanaDeviceType deviceType);

}  // namespace gaudi2

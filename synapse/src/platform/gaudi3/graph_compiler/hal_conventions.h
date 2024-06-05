#pragma once

#include "hal_reader/gaudi3/hal_reader.h"
#include "habana_device_types.h"
#include "types.h"
#include "node.h"

namespace gaudi3
{
enum LogicalQueue
{
    DEVICE_MME_LOGICAL_QUEUE = 0,
    DEVICE_TPC_LOGICAL_QUEUE = 1,
    DEVICE_ROT_LOGICAL_QUEUE = 2,
    DEVICE_XPS_LOGICAL_QUEUE = 3,  // transpose engine inside the MME block

    LOGICAL_QUEUE_MAX_ID
};

constexpr HabanaDeviceType LogicalQueue2DeviceType[LOGICAL_QUEUE_MAX_ID] = {DEVICE_MME,      // DEVICE_MME_LOGICAL_QUEUE
                                                                            DEVICE_TPC,      // DEVICE_TPC_LOGICAL_QUEUE
                                                                            DEVICE_ROTATOR,  // DEVICE_ROT_LOGICAL_QUEUE
                                                                            DEVICE_MME};     // DEVICE_XPS_LOGICAL_QUEUE

gaudi3_engine_id getQueueID(HabanaDeviceType type, unsigned id);
std::string      getEngineName(gaudi3_engine_id id);
unsigned         deviceTypeToLogicalQueue(HabanaDeviceType deviceType, const Node& node);
HabanaDeviceType logicalQueueToDeviceType(unsigned queue);

}  // namespace gaudi3

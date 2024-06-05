#pragma once

#include "graph_compiler/habana_nodes/habana_nodes.h"
#include "habana_device_types.h"
#include "hal_reader/gaudi1/hal_reader.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace gaudi
{
    enum LogicalQueue {
        DEVICE_MME_LOGICAL_QUEUE                  = 0,
        DEVICE_TPC_LOGICAL_QUEUE                  = 1,
        // multiple logical queues per parallel level
        DEVICE_DMA_1_1_DRAM_SRAM_LOGICAL_QUEUE    = 2,
        DEVICE_DMA_1_2_DRAM_SRAM_LOGICAL_QUEUE    = 3,
        DEVICE_DMA_1_3_DRAM_SRAM_LOGICAL_QUEUE    = 4,
        DEVICE_DMA_1_4_DRAM_SRAM_LOGICAL_QUEUE    = 5,
        DEVICE_DMA_1_5_DRAM_SRAM_LOGICAL_QUEUE    = 6,
        DEVICE_DMA_1_6_DRAM_SRAM_LOGICAL_QUEUE    = 7,
        DEVICE_DMA_2_1_DRAM_SRAM_LOGICAL_QUEUE    = 8,
        DEVICE_DMA_2_2_DRAM_SRAM_LOGICAL_QUEUE    = 9,
        DEVICE_DMA_2_3_DRAM_SRAM_LOGICAL_QUEUE    = 10,
        DEVICE_DMA_3_1_DRAM_SRAM_LOGICAL_QUEUE    = 11,
        DEVICE_DMA_3_2_DRAM_SRAM_LOGICAL_QUEUE    = 12,
        DEVICE_DMA_4_1_DRAM_SRAM_LOGICAL_QUEUE    = 13,
        DEVICE_DMA_5_1_DRAM_SRAM_LOGICAL_QUEUE    = 14,
        DEVICE_DMA_6_1_DRAM_SRAM_LOGICAL_QUEUE    = 15,
        DEVICE_DMA_HOST_DEVICE_LOGICAL_QUEUE      = 16, // Doesn't use groups
        DEVICE_DMA_DEVICE_HOST_LOGICAL_QUEUE      = 17, // Doesn't use groups
        DEVICE_COMPLETION_LOGICAL_QUEUE           = 18,


        LOGICAL_QUEUE_MAX_ID
    };
    gaudi_queue_id      getQueueID(HabanaDeviceType type, unsigned id);
    std::string         getEngineName(gaudi_queue_id id, HalReader* pHalReader = nullptr);
    std::string_view    getQmanIdName(uint32_t id);
    unsigned            deviceTypeToLogicalQueue(const pNode& node, HabanaDeviceType deviceType);
    unsigned            baseDmaLogicalQueue(unsigned numOfEngines);

}

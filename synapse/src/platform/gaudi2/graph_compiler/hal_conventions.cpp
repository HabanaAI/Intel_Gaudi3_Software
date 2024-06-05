#include "hal_conventions.h"

#include "command_queue.h"
#include "hal_reader/gaudi2/hal_reader.h"

#include <sstream>

using namespace gaudi2;

static gaudi2_queue_id getEngQueueId(unsigned                     engine,
                                     unsigned                     maxEngines,
                                     unsigned                     numEnginesOnDcore,
                                     const std::vector<unsigned>& baseIds)
{
    unsigned maxStreams = Gaudi2HalReader::instance()->getNumEngineStreams();
    HB_ASSERT(engine < maxEngines, "{}: engine number {} is above the limit", __func__, engine);
    unsigned baseId = baseIds[engine / numEnginesOnDcore];
    return (gaudi2_queue_id)(baseId + (engine % numEnginesOnDcore) * maxStreams);
}

gaudi2_queue_id gaudi2::getQueueID(HabanaDeviceType type, unsigned id)
{
    switch (type)
    {
    case DEVICE_TPC:
        return getEngQueueId(
            id,
            Gaudi2HalReader::instance()->getNumTpcEngines(),
            Gaudi2HalReader::instance()->getNumTpcEnginesOnDcore(),//6
            {GAUDI2_QUEUE_ID_DCORE0_TPC_0_0, GAUDI2_QUEUE_ID_DCORE1_TPC_0_0, GAUDI2_QUEUE_ID_DCORE2_TPC_0_0, GAUDI2_QUEUE_ID_DCORE3_TPC_0_0});

    case DEVICE_MME:
        return getEngQueueId(
            id,
            Gaudi2HalReader::instance()->getNumMmeEngines(),
            Gaudi2HalReader::instance()->getNumMmeEnginesOnDcore(),//1
            {GAUDI2_QUEUE_ID_DCORE0_MME_0_0, GAUDI2_QUEUE_ID_DCORE2_MME_0_0}); //GC uses DCORE0 and DCORE2 MMEs in pairs with DCORE1 and DCORE3 respectively

    case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
        return getEngQueueId(
            id,
            Gaudi2HalReader::instance()->getNumInternalDmaEngines(),
            Gaudi2HalReader::instance()->getNumInternalDmaEnginesOnDcore(),//2
            {GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0, GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0, GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0, GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0});

    case DEVICE_ROTATOR:
        return getEngQueueId(
            id,
            Gaudi2HalReader::instance()->getNumRotatorEngines(),
            Gaudi2HalReader::instance()->getNumRotatorEnginesOnDcore(),//2
            {GAUDI2_QUEUE_ID_ROT_0_0});

    default:
        break;
    }
    LOG_ERR(GC, "Invalid queue requested. Type = {} ID = {}", type, id);
    return GAUDI2_QUEUE_ID_SIZE;
}

unsigned gaudi2::deviceTypeToLogicalQueue(HabanaDeviceType deviceType)
{
    switch (deviceType)
    {
        case DEVICE_MME:
            return DEVICE_MME_LOGICAL_QUEUE;
        case DEVICE_TPC:
            return DEVICE_TPC_LOGICAL_QUEUE;
        case DEVICE_ROTATOR:
            return DEVICE_ROT_LOGICAL_QUEUE;
        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            return DEVICE_DMA_LOGICAL_QUEUE;
        case DEVICE_COMPLETION_QUEUE:
            return DEVICE_COMPLETION_LOGICAL_QUEUE;
        default:
            HB_ASSERT(0, "Device type is not supported");
    }
    return LOGICAL_QUEUE_MAX_ID;
}

std::string gaudi2::getEngineName(gaudi2_queue_id id)
{
    if (id == GAUDI2_QUEUE_ID_SIZE)
    {
        HB_ASSERT_DEBUG_ONLY(false, "Unknown engine name");
        return "UNKNOWN";
    }

    static const char* sQueueNames[] =
    {
        "GAUDI2_QUEUE_ID_PDMA_0_0",
        "GAUDI2_QUEUE_ID_PDMA_0_1",
        "GAUDI2_QUEUE_ID_PDMA_0_2",
        "GAUDI2_QUEUE_ID_PDMA_0_3",
        "GAUDI2_QUEUE_ID_PDMA_1_0",
        "GAUDI2_QUEUE_ID_PDMA_1_1",
        "GAUDI2_QUEUE_ID_PDMA_1_2",
        "GAUDI2_QUEUE_ID_PDMA_1_3",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_0_1",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_0_2",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_0_3",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_1_0",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_1_1",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_1_2",
        "GAUDI2_QUEUE_ID_DCORE0_EDMA_1_3",
        "GAUDI2_QUEUE_ID_DCORE0_MME_0_0",
        "GAUDI2_QUEUE_ID_DCORE0_MME_0_1",
        "GAUDI2_QUEUE_ID_DCORE0_MME_0_2",
        "GAUDI2_QUEUE_ID_DCORE0_MME_0_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_0_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_0_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_0_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_0_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_1_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_1_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_1_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_1_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_2_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_2_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_2_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_2_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_3_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_3_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_3_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_3_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_4_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_4_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_4_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_4_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_5_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_5_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_5_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_5_3",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_6_0",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_6_1",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_6_2",
        "GAUDI2_QUEUE_ID_DCORE0_TPC_6_3",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_0_1",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_0_2",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_0_3",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_1_0",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_1_1",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_1_2",
        "GAUDI2_QUEUE_ID_DCORE1_EDMA_1_3",
        "GAUDI2_QUEUE_ID_DCORE1_MME_0_0",
        "GAUDI2_QUEUE_ID_DCORE1_MME_0_1",
        "GAUDI2_QUEUE_ID_DCORE1_MME_0_2",
        "GAUDI2_QUEUE_ID_DCORE1_MME_0_3",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_0_0",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_0_1",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_0_2",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_0_3",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_1_0",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_1_1",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_1_2",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_1_3",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_2_0",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_2_1",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_2_2",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_2_3",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_3_0",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_3_1",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_3_2",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_3_3",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_4_0",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_4_1",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_4_2",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_4_3",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_5_0",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_5_1",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_5_2",
        "GAUDI2_QUEUE_ID_DCORE1_TPC_5_3",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_0_1",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_0_2",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_0_3",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_1_0",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_1_1",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_1_2",
        "GAUDI2_QUEUE_ID_DCORE2_EDMA_1_3",
        "GAUDI2_QUEUE_ID_DCORE2_MME_0_0",
        "GAUDI2_QUEUE_ID_DCORE2_MME_0_1",
        "GAUDI2_QUEUE_ID_DCORE2_MME_0_2",
        "GAUDI2_QUEUE_ID_DCORE2_MME_0_3",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_0_0",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_0_1",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_0_2",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_0_3",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_1_0",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_1_1",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_1_2",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_1_3",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_2_0",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_2_1",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_2_2",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_2_3",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_3_0",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_3_1",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_3_2",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_3_3",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_4_0",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_4_1",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_4_2",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_4_3",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_5_0",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_5_1",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_5_2",
        "GAUDI2_QUEUE_ID_DCORE2_TPC_5_3",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_0_1",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_0_2",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_0_3",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_1_0",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_1_1",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_1_2",
        "GAUDI2_QUEUE_ID_DCORE3_EDMA_1_3",
        "GAUDI2_QUEUE_ID_DCORE3_MME_0_0",
        "GAUDI2_QUEUE_ID_DCORE3_MME_0_1",
        "GAUDI2_QUEUE_ID_DCORE3_MME_0_2",
        "GAUDI2_QUEUE_ID_DCORE3_MME_0_3",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_0_0",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_0_1",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_0_2",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_0_3",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_1_0",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_1_1",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_1_2",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_1_3",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_2_0",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_2_1",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_2_2",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_2_3",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_3_0",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_3_1",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_3_2",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_3_3",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_4_0",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_4_1",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_4_2",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_4_3",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_5_0",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_5_1",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_5_2",
        "GAUDI2_QUEUE_ID_DCORE3_TPC_5_3",
        "GAUDI2_QUEUE_ID_NIC_0_0",
        "GAUDI2_QUEUE_ID_NIC_0_1",
        "GAUDI2_QUEUE_ID_NIC_0_2",
        "GAUDI2_QUEUE_ID_NIC_0_3",
        "GAUDI2_QUEUE_ID_NIC_1_0",
        "GAUDI2_QUEUE_ID_NIC_1_1",
        "GAUDI2_QUEUE_ID_NIC_1_2",
        "GAUDI2_QUEUE_ID_NIC_1_3",
        "GAUDI2_QUEUE_ID_NIC_2_0",
        "GAUDI2_QUEUE_ID_NIC_2_1",
        "GAUDI2_QUEUE_ID_NIC_2_2",
        "GAUDI2_QUEUE_ID_NIC_2_3",
        "GAUDI2_QUEUE_ID_NIC_3_0",
        "GAUDI2_QUEUE_ID_NIC_3_1",
        "GAUDI2_QUEUE_ID_NIC_3_2",
        "GAUDI2_QUEUE_ID_NIC_3_3",
        "GAUDI2_QUEUE_ID_NIC_4_0",
        "GAUDI2_QUEUE_ID_NIC_4_1",
        "GAUDI2_QUEUE_ID_NIC_4_2",
        "GAUDI2_QUEUE_ID_NIC_4_3",
        "GAUDI2_QUEUE_ID_NIC_5_0",
        "GAUDI2_QUEUE_ID_NIC_5_1",
        "GAUDI2_QUEUE_ID_NIC_5_2",
        "GAUDI2_QUEUE_ID_NIC_5_3",
        "GAUDI2_QUEUE_ID_NIC_6_0",
        "GAUDI2_QUEUE_ID_NIC_6_1",
        "GAUDI2_QUEUE_ID_NIC_6_2",
        "GAUDI2_QUEUE_ID_NIC_6_3",
        "GAUDI2_QUEUE_ID_NIC_7_0",
        "GAUDI2_QUEUE_ID_NIC_7_1",
        "GAUDI2_QUEUE_ID_NIC_7_2",
        "GAUDI2_QUEUE_ID_NIC_7_3",
        "GAUDI2_QUEUE_ID_NIC_8_0",
        "GAUDI2_QUEUE_ID_NIC_8_1",
        "GAUDI2_QUEUE_ID_NIC_8_2",
        "GAUDI2_QUEUE_ID_NIC_8_3",
        "GAUDI2_QUEUE_ID_NIC_9_0",
        "GAUDI2_QUEUE_ID_NIC_9_1",
        "GAUDI2_QUEUE_ID_NIC_9_2",
        "GAUDI2_QUEUE_ID_NIC_9_3",
        "GAUDI2_QUEUE_ID_NIC_10_0",
        "GAUDI2_QUEUE_ID_NIC_10_1",
        "GAUDI2_QUEUE_ID_NIC_10_2",
        "GAUDI2_QUEUE_ID_NIC_10_3",
        "GAUDI2_QUEUE_ID_NIC_11_0",
        "GAUDI2_QUEUE_ID_NIC_11_1",
        "GAUDI2_QUEUE_ID_NIC_11_2",
        "GAUDI2_QUEUE_ID_NIC_11_3",
        "GAUDI2_QUEUE_ID_NIC_12_0",
        "GAUDI2_QUEUE_ID_NIC_12_1",
        "GAUDI2_QUEUE_ID_NIC_12_2",
        "GAUDI2_QUEUE_ID_NIC_12_3",
        "GAUDI2_QUEUE_ID_NIC_13_0",
        "GAUDI2_QUEUE_ID_NIC_13_1",
        "GAUDI2_QUEUE_ID_NIC_13_2",
        "GAUDI2_QUEUE_ID_NIC_13_3",
        "GAUDI2_QUEUE_ID_NIC_14_0",
        "GAUDI2_QUEUE_ID_NIC_14_1",
        "GAUDI2_QUEUE_ID_NIC_14_2",
        "GAUDI2_QUEUE_ID_NIC_14_3",
        "GAUDI2_QUEUE_ID_NIC_15_0",
        "GAUDI2_QUEUE_ID_NIC_15_1",
        "GAUDI2_QUEUE_ID_NIC_15_2",
        "GAUDI2_QUEUE_ID_NIC_15_3",
        "GAUDI2_QUEUE_ID_NIC_16_0",
        "GAUDI2_QUEUE_ID_NIC_16_1",
        "GAUDI2_QUEUE_ID_NIC_16_2",
        "GAUDI2_QUEUE_ID_NIC_16_3",
        "GAUDI2_QUEUE_ID_NIC_17_0",
        "GAUDI2_QUEUE_ID_NIC_17_1",
        "GAUDI2_QUEUE_ID_NIC_17_2",
        "GAUDI2_QUEUE_ID_NIC_17_3",
        "GAUDI2_QUEUE_ID_NIC_18_0",
        "GAUDI2_QUEUE_ID_NIC_18_1",
        "GAUDI2_QUEUE_ID_NIC_18_2",
        "GAUDI2_QUEUE_ID_NIC_18_3",
        "GAUDI2_QUEUE_ID_NIC_19_0",
        "GAUDI2_QUEUE_ID_NIC_19_1",
        "GAUDI2_QUEUE_ID_NIC_19_2",
        "GAUDI2_QUEUE_ID_NIC_19_3",
        "GAUDI2_QUEUE_ID_NIC_20_0",
        "GAUDI2_QUEUE_ID_NIC_20_1",
        "GAUDI2_QUEUE_ID_NIC_20_2",
        "GAUDI2_QUEUE_ID_NIC_20_3",
        "GAUDI2_QUEUE_ID_NIC_21_0",
        "GAUDI2_QUEUE_ID_NIC_21_1",
        "GAUDI2_QUEUE_ID_NIC_21_2",
        "GAUDI2_QUEUE_ID_NIC_21_3",
        "GAUDI2_QUEUE_ID_NIC_22_0",
        "GAUDI2_QUEUE_ID_NIC_22_1",
        "GAUDI2_QUEUE_ID_NIC_22_2",
        "GAUDI2_QUEUE_ID_NIC_22_3",
        "GAUDI2_QUEUE_ID_NIC_23_0",
        "GAUDI2_QUEUE_ID_NIC_23_1",
        "GAUDI2_QUEUE_ID_NIC_23_2",
        "GAUDI2_QUEUE_ID_NIC_23_3",
        "GAUDI2_QUEUE_ID_ROT_0_0",
        "GAUDI2_QUEUE_ID_ROT_0_1",
        "GAUDI2_QUEUE_ID_ROT_0_2",
        "GAUDI2_QUEUE_ID_ROT_0_3",
        "GAUDI2_QUEUE_ID_ROT_1_0",
        "GAUDI2_QUEUE_ID_ROT_1_1",
        "GAUDI2_QUEUE_ID_ROT_1_2",
        "GAUDI2_QUEUE_ID_ROT_1_3"
    };
    HB_ASSERT(id < ARRAY_SIZE(sQueueNames), "invalid queue id");
    return std::string(sQueueNames[id]);
}

std::string_view gaudi2::getQmanIdName(uint32_t id)
{
    // Aligned with habanalabs gaudi2_engine_id
    static constexpr std::string_view sQmanNames[] = {
        "GAUDI2_DCORE0_ENGINE_ID_EDMA_0", "GAUDI2_DCORE0_ENGINE_ID_EDMA_1",
        "GAUDI2_DCORE0_ENGINE_ID_MME",    "GAUDI2_DCORE0_ENGINE_ID_TPC_0",
        "GAUDI2_DCORE0_ENGINE_ID_TPC_1",  "GAUDI2_DCORE0_ENGINE_ID_TPC_2",
        "GAUDI2_DCORE0_ENGINE_ID_TPC_3",  "GAUDI2_DCORE0_ENGINE_ID_TPC_4",
        "GAUDI2_DCORE0_ENGINE_ID_TPC_5",  "GAUDI2_DCORE0_ENGINE_ID_DEC_0",
        "GAUDI2_DCORE0_ENGINE_ID_DEC_1",  "GAUDI2_DCORE1_ENGINE_ID_EDMA_0",
        "GAUDI2_DCORE1_ENGINE_ID_EDMA_1", "GAUDI2_DCORE1_ENGINE_ID_MME",
        "GAUDI2_DCORE1_ENGINE_ID_TPC_0",  "GAUDI2_DCORE1_ENGINE_ID_TPC_1",
        "GAUDI2_DCORE1_ENGINE_ID_TPC_2",  "GAUDI2_DCORE1_ENGINE_ID_TPC_3",
        "GAUDI2_DCORE1_ENGINE_ID_TPC_4",  "GAUDI2_DCORE1_ENGINE_ID_TPC_5",
        "GAUDI2_DCORE1_ENGINE_ID_DEC_0",  "GAUDI2_DCORE1_ENGINE_ID_DEC_1",
        "GAUDI2_DCORE2_ENGINE_ID_EDMA_0", "GAUDI2_DCORE2_ENGINE_ID_EDMA_1",
        "GAUDI2_DCORE2_ENGINE_ID_MME",    "GAUDI2_DCORE2_ENGINE_ID_TPC_0",
        "GAUDI2_DCORE2_ENGINE_ID_TPC_1",  "GAUDI2_DCORE2_ENGINE_ID_TPC_2",
        "GAUDI2_DCORE2_ENGINE_ID_TPC_3",  "GAUDI2_DCORE2_ENGINE_ID_TPC_4",
        "GAUDI2_DCORE2_ENGINE_ID_TPC_5",  "GAUDI2_DCORE2_ENGINE_ID_DEC_0",
        "GAUDI2_DCORE2_ENGINE_ID_DEC_1",  "GAUDI2_DCORE3_ENGINE_ID_EDMA_0",
        "GAUDI2_INVALID_ENGINE_ID",       "GAUDI2_DCORE3_ENGINE_ID_MME",
        "GAUDI2_DCORE3_ENGINE_ID_TPC_0",  "GAUDI2_DCORE3_ENGINE_ID_TPC_1",
        "GAUDI2_DCORE3_ENGINE_ID_TPC_2",  "GAUDI2_DCORE3_ENGINE_ID_TPC_3",
        "GAUDI2_DCORE3_ENGINE_ID_TPC_4",  "GAUDI2_DCORE3_ENGINE_ID_TPC_5",
        "GAUDI2_DCORE3_ENGINE_ID_DEC_0",  "GAUDI2_DCORE3_ENGINE_ID_DEC_1",
        "GAUDI2_DCORE0_ENGINE_ID_TPC_6",  "GAUDI2_ENGINE_ID_PDMA_0",
        "GAUDI2_ENGINE_ID_PDMA_1",        "GAUDI2_ENGINE_ID_ROT_0",
        "GAUDI2_ENGINE_ID_ROT_1",         "UNKNOWN"};

    HB_ASSERT(id < sizeof(sQmanNames)/sizeof(sQmanNames[0]), "invalid queue id");
    return sQmanNames[id];
}

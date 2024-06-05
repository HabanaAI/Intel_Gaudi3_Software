#include "hal_conventions.h"

#include "command_queue.h"
#include "hal_reader/gaudi3/hal_reader.h"

using namespace gaudi3;

gaudi3_engine_id gaudi3::getQueueID(HabanaDeviceType type, unsigned id)
{
    unsigned realEngineIdx = 0;
    switch (type)
    {
        case DEVICE_TPC:
            return GAUDI3_HDCORE0_ENGINE_ID_TPC_0;

        case DEVICE_MME:
            realEngineIdx = id + GAUDI3_HDCORE0_ENGINE_ID_MME_0;
            HB_ASSERT(realEngineIdx >= GAUDI3_HDCORE0_ENGINE_ID_MME_0 &&
                          realEngineIdx <= GAUDI3_HDCORE7_ENGINE_ID_MME_0,
                      "MME engine id is out of range: {}",
                      realEngineIdx);
            return (gaudi3_engine_id)realEngineIdx;

        case DEVICE_ROTATOR:
            realEngineIdx = id + GAUDI3_HDCORE1_ENGINE_ID_ROT_0;
            HB_ASSERT(realEngineIdx >= GAUDI3_HDCORE1_ENGINE_ID_ROT_0 &&
                          realEngineIdx <= GAUDI3_HDCORE6_ENGINE_ID_ROT_1,
                      "Rotator engine id is out of range: {}",
                      realEngineIdx);
            return (gaudi3_engine_id)realEngineIdx;

        default:
            break;
    }
    LOG_ERR(GC, "Invalid queue requested. Type = {} ID = {}", type, realEngineIdx);
    return GAUDI3_ENGINE_ID_SIZE;
}

unsigned gaudi3::deviceTypeToLogicalQueue(HabanaDeviceType deviceType, const Node& node)
{
    switch (deviceType)
    {
        case DEVICE_MME:
            if (node.getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE)
            {
                try
                {
                    if (!(dynamic_cast<const MmeNode&>(node).isTransposeViaGemm())) return DEVICE_XPS_LOGICAL_QUEUE;
                }
                catch(const std::bad_cast& exp)
                {
                    HB_ASSERT(0, "Node with device type DEVICE_MME must also be MmeNode");
                }
            }
            return DEVICE_MME_LOGICAL_QUEUE;
        case DEVICE_TPC:
            return DEVICE_TPC_LOGICAL_QUEUE;
        case DEVICE_ROTATOR:
            return DEVICE_ROT_LOGICAL_QUEUE;
        default:
            HB_ASSERT(0, "Device type is not supported");
    }
    return LOGICAL_QUEUE_MAX_ID;
}

HabanaDeviceType gaudi3::logicalQueueToDeviceType(unsigned queue)
{
    HB_ASSERT(queue < LOGICAL_QUEUE_MAX_ID, "Invalid logical queue requested: {}", queue);

    return gaudi3::LogicalQueue2DeviceType[queue];
}

std::string gaudi3::getEngineName(gaudi3_engine_id id)
{
    if (id == GAUDI3_ENGINE_ID_SIZE)
    {
        HB_ASSERT_DEBUG_ONLY(false, "Unknown engine name");
        return "UNKNOWN";
    }

    static const char* sEngineNames[] = {"GAUDI3_HDCORE0_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE0_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE1_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE1_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE2_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE2_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE3_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE3_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE4_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE4_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE5_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE5_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE6_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE6_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE7_ENGINE_ID_DEC_0",
                                         "GAUDI3_HDCORE7_ENGINE_ID_DEC_1",
                                         "GAUDI3_HDCORE1_ENGINE_ID_EDMA_0",
                                         "GAUDI3_HDCORE1_ENGINE_ID_EDMA_1",
                                         "GAUDI3_HDCORE3_ENGINE_ID_EDMA_0",
                                         "GAUDI3_HDCORE3_ENGINE_ID_EDMA_1",
                                         "GAUDI3_HDCORE4_ENGINE_ID_EDMA_0",
                                         "GAUDI3_HDCORE4_ENGINE_ID_EDMA_1",
                                         "GAUDI3_HDCORE6_ENGINE_ID_EDMA_0",
                                         "GAUDI3_HDCORE6_ENGINE_ID_EDMA_1",
                                         "GAUDI3_HDCORE0_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE1_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE2_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE3_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE4_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE5_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE6_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE7_ENGINE_ID_MME_0",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE1_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE3_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE4_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE6_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_0",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_1",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_2",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_3",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_4",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_5",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_6",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_7",
                                         "GAUDI3_HDCORE0_ENGINE_ID_TPC_8",
                                         "GAUDI3_HDCORE2_ENGINE_ID_TPC_8",
                                         "GAUDI3_HDCORE5_ENGINE_ID_TPC_8",
                                         "GAUDI3_HDCORE7_ENGINE_ID_TPC_8",
                                         "GAUDI3_DIE0_ENGINE_ID_NIC_0",
                                         "GAUDI3_DIE0_ENGINE_ID_NIC_1",
                                         "GAUDI3_DIE0_ENGINE_ID_NIC_2",
                                         "GAUDI3_DIE0_ENGINE_ID_NIC_3",
                                         "GAUDI3_DIE0_ENGINE_ID_NIC_4",
                                         "GAUDI3_DIE0_ENGINE_ID_NIC_5",
                                         "GAUDI3_DIE1_ENGINE_ID_NIC_0",
                                         "GAUDI3_DIE1_ENGINE_ID_NIC_1",
                                         "GAUDI3_DIE1_ENGINE_ID_NIC_2",
                                         "GAUDI3_DIE1_ENGINE_ID_NIC_3",
                                         "GAUDI3_DIE1_ENGINE_ID_NIC_4",
                                         "GAUDI3_DIE1_ENGINE_ID_NIC_5",
                                         "GAUDI3_HDCORE1_ENGINE_ID_ROT_0",
                                         "GAUDI3_HDCORE1_ENGINE_ID_ROT_1",
                                         "GAUDI3_HDCORE3_ENGINE_ID_ROT_0",
                                         "GAUDI3_HDCORE3_ENGINE_ID_ROT_1",
                                         "GAUDI3_HDCORE4_ENGINE_ID_ROT_0",
                                         "GAUDI3_HDCORE4_ENGINE_ID_ROT_1",
                                         "GAUDI3_HDCORE6_ENGINE_ID_ROT_0",
                                         "GAUDI3_HDCORE6_ENGINE_ID_ROT_1",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_0",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_1",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_2",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_3",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_4",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_0_CH_5",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_0",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_1",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_2",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_3",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_4",
                                         "GAUDI3_DIE0_ENGINE_ID_PDMA_1_CH_5",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_0",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_1",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_2",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_3",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_4",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_0_CH_5",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_0",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_1",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_2",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_3",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_4",
                                         "GAUDI3_DIE1_ENGINE_ID_PDMA_1_CH_5",
                                         "GAUDI3_ENGINE_ID_SIZE"};

    HB_ASSERT(id < ARRAY_SIZE(sEngineNames), "invalid queue id");
    return std::string(sEngineNames[id]);
}

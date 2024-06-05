#include <limits>
#include <cstddef>

#include "utils.hpp"

#include "../../graph_compiler/utils.h"

#include "hal_reader/gaudi2/hal_reader.h"

#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"

using namespace gaudi2;

uint64_t gaudi2::getQMANBase(HabanaDeviceType   type,
                             unsigned           deviceID)
{
    switch (type)
    {
        case DEVICE_DMA_HOST_DEVICE:
            return mmPDMA0_QM_BASE;

        case DEVICE_DMA_DEVICE_HOST:
            return mmPDMA1_QM_BASE;

        case DEVICE_COMPLETION_QUEUE:
            return mmDCORE0_TPC0_QM_BASE;

        case DEVICE_MME:
            switch (deviceID)
            {
                case 0: return mmDCORE0_MME_QM_BASE;
                case 1: return mmDCORE2_MME_QM_BASE;
                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        case DEVICE_TPC:
            switch (deviceID)
            {
                case 0:  return mmDCORE0_TPC0_QM_BASE;
                case 1:  return mmDCORE0_TPC1_QM_BASE;
                case 2:  return mmDCORE0_TPC2_QM_BASE;
                case 3:  return mmDCORE0_TPC3_QM_BASE;
                case 4:  return mmDCORE0_TPC4_QM_BASE;
                case 5:  return mmDCORE0_TPC5_QM_BASE;
                case 6:  return mmDCORE1_TPC0_QM_BASE;
                case 7:  return mmDCORE1_TPC1_QM_BASE;
                case 8:  return mmDCORE1_TPC2_QM_BASE;
                case 9:  return mmDCORE1_TPC3_QM_BASE;
                case 10: return mmDCORE1_TPC4_QM_BASE;
                case 11: return mmDCORE1_TPC5_QM_BASE;
                case 12: return mmDCORE2_TPC0_QM_BASE;
                case 13: return mmDCORE2_TPC1_QM_BASE;
                case 14: return mmDCORE2_TPC2_QM_BASE;
                case 15: return mmDCORE2_TPC3_QM_BASE;
                case 16: return mmDCORE2_TPC4_QM_BASE;
                case 17: return mmDCORE2_TPC5_QM_BASE;
                case 18: return mmDCORE3_TPC0_QM_BASE;
                case 19: return mmDCORE3_TPC1_QM_BASE;
                case 20: return mmDCORE3_TPC2_QM_BASE;
                case 21: return mmDCORE3_TPC3_QM_BASE;
                case 22: return mmDCORE3_TPC4_QM_BASE;
                case 23: return mmDCORE3_TPC5_QM_BASE;
                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            switch (deviceID)
            {
                case 0: return mmDCORE0_EDMA0_QM_BASE;
                case 1: return mmDCORE0_EDMA1_QM_BASE;
                case 2: return mmDCORE1_EDMA0_QM_BASE;
                case 3: return mmDCORE1_EDMA1_QM_BASE;
                case 4: return mmDCORE2_EDMA0_QM_BASE;
                case 5: return mmDCORE2_EDMA1_QM_BASE;
                case 6: return mmDCORE3_EDMA0_QM_BASE;
                case 7: return mmDCORE3_EDMA1_QM_BASE;
                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        case DEVICE_ROTATOR:
            switch (deviceID)
            {
                case 0: return mmROT0_QM_BASE;
                case 1: return mmROT1_QM_BASE;
                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        default:
            HB_ASSERT(0, "Invalid device");
            return (uint64_t)(-1);
    }
}

uint64_t gaudi2::getCPFenceOffset(HabanaDeviceType   type,
                                  unsigned           deviceID,
                                  WaitID             waitID,
                                  unsigned           streamID,
                                  bool               isForceStreamId /* = false */)
{
    // For host dma QMAN, use fence of the stream, otherwise use the lower cp fence
    static const unsigned lowerCpIndex = 4;
    unsigned cpFenceIdx = (isHostDma(type) || isCompletionQueue(type)) ? streamID : lowerCpIndex;

    if (isForceStreamId)
    {
        cpFenceIdx = streamID;
    }

    // Gets the fence address of the cp
    uint64_t engineQmanBase  = 0;
    uint64_t cpFenceOffset   = 0;
    switch (waitID)
    {
        case ID_0:
            cpFenceOffset = offsetof(block_qman, cp_fence0_rdata) + sizeof(struct qman::reg_cp_fence0_rdata) * cpFenceIdx;
            break;

        case ID_1:
            cpFenceOffset = offsetof(block_qman, cp_fence1_rdata) + sizeof(struct qman::reg_cp_fence1_rdata) * cpFenceIdx;
            break;

        case ID_2:
            cpFenceOffset = offsetof(block_qman, cp_fence2_rdata) + sizeof(struct qman::reg_cp_fence2_rdata) * cpFenceIdx;
            break;

        case ID_3:
            cpFenceOffset = offsetof(block_qman, cp_fence3_rdata) + sizeof(struct qman::reg_cp_fence3_rdata) * cpFenceIdx;
            break;
    }

    engineQmanBase = getQMANBase(type, deviceID);

    return engineQmanBase + cpFenceOffset;
}

uint64_t gaudi2::getCPFenceOffset(gaudi2_queue_id     queueId,
                                  WaitID              waitID)
{
    HabanaDeviceType    type;
    unsigned            deviceID;
    unsigned            streamID;

    if (!getQueueIdInfo(type, deviceID, streamID, queueId))
    {
        return (uint64_t)(-1);
    }

    return getCPFenceOffset(type, deviceID, waitID, streamID, true);
}

bool gaudi2::getQueueIdInfo(HabanaDeviceType& type, unsigned& deviceID, unsigned& streamID, gaudi2_queue_id queueId)
{
    bool retStatus = true;
    streamID = queueId % 4;

    // Downstream DMA
    if (queueId >= GAUDI2_QUEUE_ID_PDMA_0_0 && queueId <= GAUDI2_QUEUE_ID_PDMA_0_3)
    {
        type     = DEVICE_DMA_HOST_DEVICE;
        deviceID = 0;
    }
    // Upstream DMA
    else if (queueId >= GAUDI2_QUEUE_ID_PDMA_1_0 && queueId <= GAUDI2_QUEUE_ID_PDMA_1_3)
    {
        type     = DEVICE_DMA_DEVICE_HOST;
        deviceID = 0;
    }
    // TPC, DCORE 0
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE0_TPC_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE0_TPC_5_3)
    {
        type     = DEVICE_TPC;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE0_TPC_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (0 * Gaudi2HalReader::instance()->getNumTpcEnginesOnDcore());  // 0..5
    }
    // TPC, DCORE 1
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE1_TPC_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE1_TPC_5_3)
    {
        type     = DEVICE_TPC;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE1_TPC_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (1 * Gaudi2HalReader::instance()->getNumTpcEnginesOnDcore());  // 6..11
    }
    // TPC, DCORE 2
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE2_TPC_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE2_TPC_5_3)
    {
        type     = DEVICE_TPC;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE2_TPC_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (2 * Gaudi2HalReader::instance()->getNumTpcEnginesOnDcore());  // 12..17
    }
    // TPC, DCORE 3
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE3_TPC_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE3_TPC_5_3)
    {
        type     = DEVICE_TPC;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE3_TPC_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (3 * Gaudi2HalReader::instance()->getNumTpcEnginesOnDcore());  // 18..23
    }
    // DMA, DCORE 0
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE0_EDMA_1_3)
    {
        type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (0 * Gaudi2HalReader::instance()->getNumInternalDmaEnginesOnDcore());  // 0..1
    }
    // DMA, DCORE 1
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE1_EDMA_1_3)
    {
        type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (1 * Gaudi2HalReader::instance()->getNumInternalDmaEnginesOnDcore());  // 2..3
    }
    // DMA, DCORE 2
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE2_EDMA_1_3)
    {
        type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (2 * Gaudi2HalReader::instance()->getNumInternalDmaEnginesOnDcore());  // 4..5
    }
    // DMA, DCORE 3
    else if (queueId >= GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0 && queueId <= GAUDI2_QUEUE_ID_DCORE3_EDMA_1_3)
    {
        type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
        deviceID = (queueId - GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();
        deviceID += (3 * Gaudi2HalReader::instance()->getNumInternalDmaEnginesOnDcore());  // 6..7
    }
    // MME, DCORE 0 (first master)
    else if (queueId == GAUDI2_QUEUE_ID_DCORE0_MME_0_0)
    {
        type     = DEVICE_MME;
        deviceID = 0;
    }
    // MME, DCORE 2 (second master)
    else if (queueId == GAUDI2_QUEUE_ID_DCORE2_MME_0_0)
    {
        type     = DEVICE_MME;
        deviceID = 2;
    }
    // ROT
    else if (queueId >= GAUDI2_QUEUE_ID_ROT_0_0 && queueId <= GAUDI2_QUEUE_ID_ROT_1_3)
    {
        type     = DEVICE_ROTATOR;
        deviceID = (queueId - GAUDI2_QUEUE_ID_ROT_0_0) / Gaudi2HalReader::instance()->getNumEngineStreams();  // 0..1
    }
    else
    {
        HB_ASSERT(0, "Got unexpected queue ID");
        retStatus = false;
    }

    return retStatus;
}

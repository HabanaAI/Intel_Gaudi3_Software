#include <cstddef>
#include "utils.hpp"
#include "../../graph_compiler/utils.h"
#include "gaudi/asic_reg/gaudi_blocks.h"
#include "gaudi/asic_reg_structs/qman_regs.h"
#include "gaudi/gaudi_packets.h"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"

namespace gaudi
{
static const std::vector<int32_t> packetTypesSize = {MAX_INVALID_PKT_SIZE,  // Invalid
                                                     sizeof(packet_wreg32),
                                                     sizeof(packet_wreg_bulk),
                                                     sizeof(packet_msg_long),
                                                     sizeof(packet_msg_short),
                                                     sizeof(packet_cp_dma),
                                                     sizeof(packet_repeat),
                                                     sizeof(packet_msg_prot),
                                                     sizeof(packet_fence),
                                                     sizeof(packet_lin_dma),
                                                     sizeof(packet_nop),
                                                     sizeof(packet_stop),
                                                     sizeof(packet_arb_point),
                                                     sizeof(packet_wait),
                                                     MAX_INVALID_PKT_SIZE,  // Invalid
                                                     sizeof(packet_load_and_exe)};

static const std::vector<std::string> packetsName = {"Invalid_1",
                                                     "WREG32",
                                                     "WREG_BULK",
                                                     "MSG_LONG",
                                                     "MSG_SHORT",
                                                     "CP_DMA",
                                                     "REPEAT",
                                                     "MSG_PROT",
                                                     "FENCE",
                                                     "LIN_DMA",
                                                     "NOP",
                                                     "STOP",
                                                     "ARB_POINT",
                                                     "WAIT",
                                                     "Invalid_2",
                                                     "LOAD_AND_EXECUTE"};
}  // namespace gaudi

using namespace gaudi;

bool gaudi::getQueueIdInfo(HabanaDeviceType&   type,
                           unsigned&           deviceID,
                           unsigned&           streamID,
                           gaudi_queue_id      queueId)
{
    bool retStatus = true;

    switch (queueId)
    {
        case GAUDI_QUEUE_ID_DMA_0_0:
        case GAUDI_QUEUE_ID_DMA_0_1:
        case GAUDI_QUEUE_ID_DMA_0_2:
        case GAUDI_QUEUE_ID_DMA_0_3:
            type     = DEVICE_DMA_HOST_DEVICE;
            deviceID = 0;  // external downstream
            streamID = queueId - GAUDI_QUEUE_ID_DMA_0_0;
            break;

        case GAUDI_QUEUE_ID_DMA_1_0:
        case GAUDI_QUEUE_ID_DMA_1_1:
        case GAUDI_QUEUE_ID_DMA_1_2:
        case GAUDI_QUEUE_ID_DMA_1_3:
            type     = DEVICE_DMA_DEVICE_HOST;
            deviceID = 0;  // external upstream
            streamID = queueId - GAUDI_QUEUE_ID_DMA_1_0;
            break;

        case GAUDI_QUEUE_ID_DMA_2_0:
        case GAUDI_QUEUE_ID_DMA_2_1:
        case GAUDI_QUEUE_ID_DMA_2_2:
        case GAUDI_QUEUE_ID_DMA_2_3:
            type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
            deviceID = 0;  // internal, first
            streamID = queueId - GAUDI_QUEUE_ID_DMA_2_0;
            break;
        case GAUDI_QUEUE_ID_DMA_3_0:
        case GAUDI_QUEUE_ID_DMA_3_1:
        case GAUDI_QUEUE_ID_DMA_3_2:
        case GAUDI_QUEUE_ID_DMA_3_3:
            type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
            deviceID = 1;  // internal, second ...
            streamID = queueId - GAUDI_QUEUE_ID_DMA_3_0;
            break;
        case GAUDI_QUEUE_ID_DMA_4_0:
        case GAUDI_QUEUE_ID_DMA_4_1:
        case GAUDI_QUEUE_ID_DMA_4_2:
        case GAUDI_QUEUE_ID_DMA_4_3:
            type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
            deviceID = 2;
            streamID = queueId - GAUDI_QUEUE_ID_DMA_4_0;
            break;
        case GAUDI_QUEUE_ID_DMA_5_0:
        case GAUDI_QUEUE_ID_DMA_5_1:
        case GAUDI_QUEUE_ID_DMA_5_2:
        case GAUDI_QUEUE_ID_DMA_5_3:
            type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
            deviceID = 3;
            streamID = queueId - GAUDI_QUEUE_ID_DMA_5_0;
            break;
        case GAUDI_QUEUE_ID_DMA_6_0:
        case GAUDI_QUEUE_ID_DMA_6_1:
        case GAUDI_QUEUE_ID_DMA_6_2:
        case GAUDI_QUEUE_ID_DMA_6_3:
            type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
            deviceID = 4;
            streamID = queueId - GAUDI_QUEUE_ID_DMA_6_0;
            break;
        case GAUDI_QUEUE_ID_DMA_7_0:
        case GAUDI_QUEUE_ID_DMA_7_1:
        case GAUDI_QUEUE_ID_DMA_7_2:
        case GAUDI_QUEUE_ID_DMA_7_3:
            type     = DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;
            deviceID = 5;
            streamID = queueId - GAUDI_QUEUE_ID_DMA_7_0;
            break;

        case GAUDI_QUEUE_ID_MME_0_0:
        case GAUDI_QUEUE_ID_MME_0_1:
        case GAUDI_QUEUE_ID_MME_0_2:
        case GAUDI_QUEUE_ID_MME_0_3:
            type     = DEVICE_MME;
            deviceID = 0;
            streamID = queueId - GAUDI_QUEUE_ID_MME_0_0;
            break;
        case GAUDI_QUEUE_ID_MME_1_0:
        case GAUDI_QUEUE_ID_MME_1_1:
        case GAUDI_QUEUE_ID_MME_1_2:
        case GAUDI_QUEUE_ID_MME_1_3:
            type     = DEVICE_MME;
            deviceID = 1;
            streamID = queueId - GAUDI_QUEUE_ID_MME_1_0;
            break;

        case GAUDI_QUEUE_ID_TPC_0_0:
        case GAUDI_QUEUE_ID_TPC_0_1:
        case GAUDI_QUEUE_ID_TPC_0_2:
        case GAUDI_QUEUE_ID_TPC_0_3:
            type     = DEVICE_TPC;
            deviceID = 0;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_0_0;
            break;
        case GAUDI_QUEUE_ID_TPC_1_0:
        case GAUDI_QUEUE_ID_TPC_1_1:
        case GAUDI_QUEUE_ID_TPC_1_2:
        case GAUDI_QUEUE_ID_TPC_1_3:
            type     = DEVICE_TPC;
            deviceID = 1;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_1_0;
            break;
        case GAUDI_QUEUE_ID_TPC_2_0:
        case GAUDI_QUEUE_ID_TPC_2_1:
        case GAUDI_QUEUE_ID_TPC_2_2:
        case GAUDI_QUEUE_ID_TPC_2_3:
            type     = DEVICE_TPC;
            deviceID = 2;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_2_0;
            break;
        case GAUDI_QUEUE_ID_TPC_3_0:
        case GAUDI_QUEUE_ID_TPC_3_1:
        case GAUDI_QUEUE_ID_TPC_3_2:
        case GAUDI_QUEUE_ID_TPC_3_3:
            type     = DEVICE_TPC;
            deviceID = 3;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_3_0;
            break;
        case GAUDI_QUEUE_ID_TPC_4_0:
        case GAUDI_QUEUE_ID_TPC_4_1:
        case GAUDI_QUEUE_ID_TPC_4_2:
        case GAUDI_QUEUE_ID_TPC_4_3:
            type     = DEVICE_TPC;
            deviceID = 4;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_4_0;
            break;
        case GAUDI_QUEUE_ID_TPC_5_0:
        case GAUDI_QUEUE_ID_TPC_5_1:
        case GAUDI_QUEUE_ID_TPC_5_2:
        case GAUDI_QUEUE_ID_TPC_5_3:
            type     = DEVICE_TPC;
            deviceID = 5;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_5_0;
            break;
        case GAUDI_QUEUE_ID_TPC_6_0:
        case GAUDI_QUEUE_ID_TPC_6_1:
        case GAUDI_QUEUE_ID_TPC_6_2:
        case GAUDI_QUEUE_ID_TPC_6_3:
            type     = DEVICE_TPC;
            deviceID = 6;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_6_0;
            break;
        case GAUDI_QUEUE_ID_TPC_7_0:
        case GAUDI_QUEUE_ID_TPC_7_1:
        case GAUDI_QUEUE_ID_TPC_7_2:
        case GAUDI_QUEUE_ID_TPC_7_3:
            type     = DEVICE_TPC;
            deviceID = 7;
            streamID = queueId - GAUDI_QUEUE_ID_TPC_7_0;
            break;

        default:
            HB_ASSERT(0, "Invalid queue-ID");
            retStatus = false;
    }

    return retStatus;
}

uint64_t gaudi::getQMANBase(HabanaDeviceType   type,
                            unsigned           deviceID)
{
    switch (type)
    {
        case DEVICE_DMA_HOST_DEVICE:
            return mmDMA0_QM_BASE;

        case DEVICE_COMPLETION_QUEUE:
            return QmansDefinition::getInstance()->getComputeStreamsMasterQmanBaseAddr();

        case DEVICE_MME:
            switch (deviceID)
            {
                // Intentionally swapped - Aligned to driver qman mapping
                case 0: return mmMME2_QM_BASE;
                case 1: return mmMME0_QM_BASE;
                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        case DEVICE_TPC:
            switch (deviceID)
            {
                case 0: return mmTPC0_QM_BASE;
                case 1: return mmTPC1_QM_BASE;
                case 2: return mmTPC2_QM_BASE;
                case 3: return mmTPC3_QM_BASE;
                case 4: return mmTPC4_QM_BASE;
                case 5: return mmTPC5_QM_BASE;
                case 6: return mmTPC6_QM_BASE;
                case 7: return mmTPC7_QM_BASE;

                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            switch (deviceID)
            {
                // mmDMA0_QM_BASE and mmDMA1_QM_BASE are used for Host->Device and Device->Host operations
                case 0: return mmDMA2_QM_BASE;
                case 1: return mmDMA3_QM_BASE;
                case 2: return mmDMA4_QM_BASE;
                case 3: return mmDMA5_QM_BASE;
                case 4: return mmDMA6_QM_BASE;
                case 5: return mmDMA7_QM_BASE;

                default:
                    HB_ASSERT(0, "Not supported");
                    return 0;
            };
            break;

        case DEVICE_DMA_DRAM_HOST:
            //Intentional fall-through

        default:
            HB_ASSERT(0, "Invalid device");
            return (uint64_t)(-1);
    }
}

uint64_t gaudi::getQMANBase(gaudi_queue_id queueId)
{
    HabanaDeviceType    type;
    unsigned            deviceID;
    unsigned            streamID;

    if (!getQueueIdInfo(type, deviceID, streamID, queueId))
    {
        return (uint64_t)(-1);
    }

    return getQMANBase(type, deviceID);
}

uint64_t gaudi::getCPFenceOffset(HabanaDeviceType   type,
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

uint64_t gaudi::getCPFenceOffset(gaudi_queue_id     queueId,
                                 WaitID             waitID)
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

/* Checks if there is an overlap of currBuffer with existing buffers in the DB */
bool gaudi::noOverlap(std::map<uint64_t, uint64_t>& bufferDB, uint64_t currTensorAddress, uint64_t currTensorSize)
{
    auto nextEntry = bufferDB.upper_bound(currTensorAddress);

    if (nextEntry != bufferDB.end())
    {
        if (currTensorAddress + currTensorSize > nextEntry->first)
        {
            return false;
        }
    }

    if (nextEntry != bufferDB.begin())
    {
        auto prevEntry = --nextEntry;

        if (prevEntry->first + prevEntry->second > currTensorAddress)
        {
            return false;
        }
    }
    return true;

}

bool gaudi::isValidPacket(const uint32_t*&             pCurrentPacket,
                          int64_t&                     leftBufferSize,
                          utilPacketType&              packetType,
                          uint64_t&                    cpDmaBufferAddress,
                          uint64_t&                    cpDmaBufferSize,
                          ePacketValidationLoggingMode loggingMode)
{
    return ::isValidPacket<packet_cp_dma,
                           packet_lin_dma,
                           packet_wreg_bulk,
                           packet_arb_point,
                           PACKET_CP_DMA,
                           PACKET_LIN_DMA,
                           PACKET_WREG_BULK,
                           PACKET_ARB_POINT>(pCurrentPacket,
                                             leftBufferSize,
                                             packetType,
                                             cpDmaBufferAddress,
                                             cpDmaBufferSize,
                                             packetTypesSize,
                                             packetsName,
                                             loggingMode);
}

bool gaudi::checkForUndefinedOpcode(const void*&                 pCommandsBuffer,
                                    uint64_t                     bufferSize,
                                    ePacketValidationLoggingMode loggingMode)
{
    return ::checkForUndefinedOpcode<packet_cp_dma,
                                     packet_lin_dma,
                                     packet_wreg_bulk,
                                     packet_arb_point,
                                     PACKET_CP_DMA,
                                     PACKET_LIN_DMA,
                                     PACKET_WREG_BULK,
                                     PACKET_ARB_POINT>(pCommandsBuffer,
                                                       bufferSize,
                                                       packetTypesSize,
                                                       packetsName,
                                                       loggingMode);
}

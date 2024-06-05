#pragma once

#include "habana_device_types.h"

#include "../../graph_compiler/sync/sync_types.h"
#include "../../graph_compiler/types.h"

#include "drm/habanalabs_accel.h"

#include <cstdint>
#include <map>

namespace gaudi
{
    bool getQueueIdInfo(HabanaDeviceType&   type,
                        unsigned&           deviceID,
                        unsigned&           streamID,
                        gaudi_queue_id      queueId);

    uint64_t getQMANBase(HabanaDeviceType   type,
                         unsigned           deviceID);

    uint64_t getQMANBase(gaudi_queue_id     queueId);

    uint64_t getCPFenceOffset(HabanaDeviceType   type,
                              unsigned           deviceID,
                              WaitID             waitID,
                              unsigned           streamID,
                              bool               isForceStreamId = false);

    uint64_t getCPFenceOffset(gaudi_queue_id     queueId,
                              WaitID             waitID);

    bool noOverlap(std::map<uint64_t, uint64_t>& bufferDB,
                   uint64_t                       currTensorAddress,
                   uint64_t                       currTensorSize);

    bool isValidPacket(const uint32_t*&             pCurrentPacket,
                       int64_t&                     leftBufferSize,
                       utilPacketType&              packetType,
                       uint64_t&                    cpDmaBufferAddress,
                       uint64_t&                    cpDmaBufferSize,
                       ePacketValidationLoggingMode loggingMode = PKT_VAIDATION_LOGGING_MODE_DISABLED);

    bool checkForUndefinedOpcode(const void*&                 pCommandsBuffer,
                                 uint64_t                     bufferSize,
                                 ePacketValidationLoggingMode loggingMode = PKT_VAIDATION_LOGGING_MODE_DISABLED);
}

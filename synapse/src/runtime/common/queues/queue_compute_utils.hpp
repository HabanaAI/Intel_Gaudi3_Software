#pragma once
#include <vector>
#include "synapse_api_types.h"
#include "infra/containers/slot_map.hpp"
#include "event_with_mapped_tensor.hpp"

struct BasicQueueInfo;
class EventInterface;

using EventWithMappedTensorSptr = SlotMapItemSptr<EventWithMappedTensor>;
using EventWithMappedTensorDB   = std::vector<EventWithMappedTensorSptr>;

class QueueComputeUtils
{
public:
    static synStatus prepareLaunch(const BasicQueueInfo&         rBasicStreamHandle,
                                   const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                   const uint32_t                enqueueTensorsAmount,
                                   const synRecipeHandle         pRecipeHandle,
                                   EventWithMappedTensorDB&      events,
                                   uint32_t                      flags);

    static void logLaunchTensors(const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                 const uint32_t                enqueueTensorsAmount,
                                 int                           logLevel,
                                 uint32_t                      flags);

    static uint64_t getAlignedWorkspaceAddress(uint64_t workspaceAddress);

    static bool isRecipeEmpty(InternalRecipeHandle& rRecipeHandle);

    static uint64_t getLaunchSeqId()  { return ++launchSeqId; };
    static uint64_t getCurrnetSeqId() { return launchSeqId.load(); }

private:
    static const uint64_t        REQUIRED_DATA_BITS_ALIGNMENT;
    static const uint64_t        REQUIRED_DATA_ALIGNMENT;
    static std::atomic<uint64_t> launchSeqId;
};

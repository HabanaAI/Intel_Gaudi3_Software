#pragma once

#include "define_synapse_common.hpp"
#include "runtime/qman/common/qman_types.hpp"

namespace common
{
class StreamCreator
{
public:
    StreamCreator() {};

    virtual ~StreamCreator() {};

    virtual bool getPhysicalQueueId(PhysicalQueuesId&   physicalQueuesId,
                                    const TrainingQueue logicalQueue,
                                    uint32_t            physicalQueueOffset) const = 0;

    // Each TQ represents a standalone HW stream
    virtual TrainingRetCode
    selectTrainingQueue(const internalStreamType& queueType, TrainingQueue& type, uint32_t elementIndex) const;

    virtual uint32_t getStreamsElementsRestriction(internalStreamType queueType) const = 0;
};
}  // namespace common
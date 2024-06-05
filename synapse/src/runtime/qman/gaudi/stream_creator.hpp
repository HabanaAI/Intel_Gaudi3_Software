#pragma once

#include "runtime/qman/common/queue_creator.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "define_synapse_common.hpp"

namespace gaudi
{
class StreamCreator : public common::StreamCreator
{
public:
    static StreamCreator* getInstance();

    virtual ~StreamCreator() {};

    virtual bool getPhysicalQueueId(PhysicalQueuesId&   physicalQueuesId,
                                    const TrainingQueue logicalQueue,
                                    uint32_t            physicalQueueOffset) const override;

    virtual uint32_t getStreamsElementsRestriction(internalStreamType queueType) const override;

private:
    StreamCreator() {};
    static StreamCreator* s_streamCreator;
};
}  // namespace gaudi